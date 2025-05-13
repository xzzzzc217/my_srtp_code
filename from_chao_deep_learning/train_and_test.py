import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from numpy import array
from sklearn.neighbors import LocalOutlierFactor
from scipy import signal
import csv

# 设备总数
device_cnt = 9

def plot_spectrogram(filepath):
    """
    绘制信号的时频图
    参数:
        filepath: 信号数据文件路径
    功能:
        1. 读取复数信号数据
        2. 计算STFT
        3. 绘制时频图
    """
    # 读取信号数据
    var = np.fromfile(filepath, dtype=np.complex64)
    
    # 计算STFT
    f_sample = 5000000  # 采样频率5MHz
    f, t, Zxx = signal.stft(var, fs=f_sample, nperseg=256, noverlap=128, scaling='spectrum')
    
    # 绘制时频图
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f/1e6, np.abs(Zxx), shading='gouraud')
    plt.colorbar(label='Magnitude')
    plt.title('Signal Spectrogram')
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [s]')
    plt.show()
    
    # 绘制信号的实部和虚部
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(np.real(var))
    plt.title('Real Part of Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(212)
    plt.plot(np.imag(var))
    plt.title('Imaginary Part of Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def train_phase(device,path, device_data_csv, n):
    """
    训练设备指纹模型
    参数:
        device: 设备编号
        path: 数据路径
        path2: 结果保存路径
        n: LocalOutlierFactor的邻居数
    功能:
        1. 读取训练数据
        2. 使用LOF去除异常值
        3. 计算平均值作为模型
        4. 保存模型参数
    """
    # 初始化数据存储数组
    x = []
    y = []
    y1 = [[] for _ in range(8)]  # 存储16个符号的相位差
    y2 = [[] for _ in range(8)]  # 用于异常值检测
    res_y = []

    # 读取训练数据
    file = open(path + device_data_csv)
    reader = csv.reader(file)
    data = list(reader)

    # 数据预处理
    for i in range(len(data)):
        for j in range(8):  # 处理16个符号
            y2[j].append([float(data[i][j])])

    # 使用LOF进行异常值检测
    clf = LocalOutlierFactor(n_neighbors=n)
    for i in range(8):
        clf.fit_predict(y2[i])
        # 保留非异常值（阈值设为-1.5）
        for j in range(clf.n_samples_fit_):
            if clf.negative_outlier_factor_[j] >= -1.5:
                x.append(i)
                y.append(y2[i][j][0])
                y1[i].append(y2[i][j][0])

    # 计算每个符号的平均值作为模型参数
    for i in range(8):
        res_y.append(sum(y1[i])/len(y1[i]))

    #return res_y
    # 保存模型参数
    with open(path + "phase_model_"+ device +".csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        w.writerow(res_y)

def train_cfo(device,path, device_data_csv,):
    """
    训练CFO模型
    参数:
        path: 数据路径
    功能:
        1. 读取CFO数据
        2. 计算CFO范围
        3. 保存CFO模型参数
    """
    # 读取CFO数据
    file = open(path + device_data_csv)
    reader = csv.reader(file)
    data = list(reader)

    # 计算CFO范围
    max_cfo1 = min_cfo1 = float(data[0][8])
    max_cfo2 = min_cfo2 = float(data[0][9])
    
    for i in range(1, len(data)):
        temp1 = float(data[i][8])
        temp2 = float(data[i][9])
        max_cfo1 = max(max_cfo1, temp1)
        min_cfo1 = min(min_cfo1, temp1)
        max_cfo2 = max(max_cfo2, temp2)
        min_cfo2 = min(min_cfo2, temp2)

    #return [min_cfo1, max_cfo1, (max_cfo1+min_cfo1)/2, (max_cfo1-min_cfo1)/2], [min_cfo2, max_cfo2, (max_cfo2+min_cfo2)/2, (max_cfo2-min_cfo2)/2]

    # 保存CFO模型参数（最小值、最大值、中心值、范围）
    with open(path + "cfo_model_"+ device +".csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        w.writerow([min_cfo1, max_cfo1, (max_cfo1+min_cfo1)/2, (max_cfo1-min_cfo1)/2])
        w.writerow([min_cfo2, max_cfo2, (max_cfo2+min_cfo2)/2, (max_cfo2-min_cfo2)/2])

def check_cfo(real, model, d):
    """
    检查CFO是否在允许范围内
    参数:
        real: 实际CFO值
        model: 模型CFO中心值
        d: 允许的偏差范围
    返回:
        布尔值，表示是否在范围内
    """
    return abs(real-model) - d <= 1e-9

def mse(data, model):
    """
    计算均方误差
    参数:
        data: 测试数据
        model: 模型数据
    返回:
        均方误差值
    """
    sum = 0
    for i in range(8):  # 计算8个符号的MSE
        sum += (float(data[i])-float(model[i]))**2
    return sum

def test(path):
    """
    测试设备识别性能
    参数:
        path: 数据路径
    功能:
        1. 加载所有设备的模型
        2. 对测试数据进行识别
        3. 计算识别准确率
        4. 绘制混淆矩阵
    """
    # 加载模型
    model = []
    cfo_model = []
    for i in range(1, device_cnt+1):
        # 加载相位模型
        file = open(path+"phase_model_device_"+str(i)+".csv")
        reader = csv.reader(file)
        model.append(list(reader)[0])

        # 加载CFO模型
        file = open(path+"cfo_model_device_"+str(i)+".csv")
        reader = csv.reader(file)
        cfo_model.append(list(reader))

    # 测试识别性能
    correct_sum = 0
    test_sum = 0
    predict_res = []
    true_res = []

    # 对每个设备进行测试
    for i in range(1, device_cnt+1):
        file = open(path+"test_datadevice_"+str(i)+".csv")
        reader = csv.reader(file)
        test_data = list(reader)
        correct_i = 0

        # 对每个测试样本进行识别
        for j in range(len(test_data)):
            min_value = 10000
            label = 1
            
            # 遍历所有设备模型进行匹配
            for k in range(len(model)):
                # 检查CFO是否匹配
                if (check_cfo(float(test_data[j][8]), float(cfo_model[k][0][2]), float(cfo_model[k][0][3])) and 
                    check_cfo(float(test_data[j][9]), float(cfo_model[k][1][2]), float(cfo_model[k][1][3]))):
                    # 计算相位特征的MSE
                    temp = mse(test_data[j], model[k])
                    if temp < min_value:
                        min_value = temp
                        label = k+1

            predict_res.append(str(label))
            true_res.append(str(i))
            if label == i:
                correct_i += 1

        # 输出每个设备的识别准确率
        print(f"device {i}: {correct_i/len(test_data):.4f}, error: {len(test_data)-correct_i}, sum: {len(test_data)}")
        correct_sum += correct_i
        test_sum += len(test_data)

    # 输出总体准确率
    print(f"总体准确率: {correct_sum/test_sum:.4f}")

    # 绘制混淆矩阵
    ll = [str(i+1) for i in range(device_cnt)]
    C = confusion_matrix(true_res, predict_res, labels=ll)
    plt.matshow(C, cmap=plt.cm.Blues)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(range(0,device_cnt), labels=ll)
    plt.yticks(range(0,device_cnt), labels=ll)
    plt.show()

if __name__ == '__main__':
    train_mode = False
    path = "C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\"
    if train_mode:
        for ii in range(1, 10):
            device = "device_"+str(ii)
            device_data_csv = "train_data"+device+".csv"
            
            train_phase(device,path, device_data_csv, 50)
            train_cfo(device,path, device_data_csv)
    else:
        test(path)
        #print(diff_phase_feature)
    #device = "device_1"
    #device_data_csv = "test_data"+device+".csv"
    #path = "C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\"
    #C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\test_datadevice_1.csv
    # 绘制设备8的第一帧信号时频图
    #signal_path = path + device + "/train_data/frame1"
    #plot_spectrogram(signal_path)
    
    # 训练设备8的模型
    #train(device,path, device_data_csv, 50)
    # 测试所有设备的识别性能
    #test(path)