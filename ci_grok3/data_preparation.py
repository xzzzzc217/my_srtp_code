import numpy as np
import os
import random
import torch
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# 全局参数
B = 125000  # 带宽
T = 128 / B  # 时间长度
f_sample = 1000000  # 采样率
Ts = 1 / f_sample  # 采样间隔
L = int(T * f_sample)  # 单帧样本数
n = np.arange(0, L, 1)
ideal_sample = np.exp(1j * (-np.pi * B * n * Ts + np.pi * B / T * (n * Ts)**2))  # 理想信号

# 数据路径
base_path_0309 = r"C:\Users\21398\Desktop\sophomore\SRTP\data\processed_dataset_0309"
base_path_0316 = r"C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0316"
output_base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data"
ci_fig_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\ci_figures"  # 新增图片保存路径

# 归一化函数
def normalized(x):
    """将信号归一化到RMS"""
    rms = np.sqrt(np.mean(np.abs(x)**2))
    return x / rms if rms != 0 else x

# CFO补偿
def cfo_compensation(sample):
    """对信号进行CFO（载波频率偏移）补偿"""
    # 粗估
    angle = np.unwrap(np.angle(sample[0:L]))
    freq = np.diff(angle) / (2 * np.pi) * f_sample
    cfo_coarse = sum(freq) / L
    # 粗补偿
    n = np.arange(0, 8*L, 1)
    cfo_corr_coarse = sample[0:8*L] * np.exp(-1j * 2 * np.pi * cfo_coarse * n * Ts)
    # 细估
    temp = np.angle(sum(cfo_corr_coarse[0:L] * np.conj(cfo_corr_coarse[L:2*L])))
    cfo_fine = temp / (-2 * np.pi * T)
    # 细补偿
    cfo_corr = cfo_corr_coarse[0:8*L] * np.exp(-1j * 2 * np.pi * cfo_fine * n * Ts)
    return cfo_corr

# 同步函数
def sync(sample):
    """对信号进行粗同步和精同步，返回起始索引"""
    # 粗同步
    for i in range(len(sample) - 2*L):
        a = np.abs(sum(sample[i:i+L] * np.conj(sample[i+L:i+2*L])))
        b = np.abs(sum(sample[i+L:i+2*L] * np.conj(sample[i+L:i+2*L])))
        M = a / b if b != 0 else 0
        if M > 0.95:
            start_coarse = i
            break
    else:
        start_coarse = 0

    # 精同步
    angle = np.unwrap(np.angle(sample))
    freq = np.diff(angle) / (2 * np.pi) * f_sample
    ideal_freq = -B / 2 + (B / T) * n * Ts
    ideal = np.abs(np.correlate(ideal_freq, ideal_freq))[0]
    th, index = 0, start_coarse
    for i in range(start_coarse, start_coarse + L):
        t = np.abs(np.correlate(ideal_freq, freq[i:i+L]))[0] if i + L < len(freq) else 0
        if t / ideal > th:
            th = t / ideal
            index = i
    return index

# 读取信号文件
def read_signal_file(file_path):
    """读取无后缀的信号文件"""
    return np.fromfile(file_path, dtype=np.complex64)

# 划分训练和测试集
def split_train_test(device_dir, train_ratio=0.8):
    """将设备信号文件划分为训练集和测试集"""
    data_dir = os.path.join(device_dir, 'data')
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random.shuffle(files)
    train_size = int(len(files) * train_ratio)
    return files[:train_size], files[train_size:]

# 提取信道无关时频图
def extract_ci_time_frequency(_signal):
    """提取信道无关的时频图"""
    f, t, spec = signal.stft(_signal, window='boxcar',nperseg=256, noverlap=128, nfft=256, 
                             return_onesided=False, padded=False, boundary=None)
    spec = np.fft.fftshift(spec, axes=0)
    chan_ind_spec = spec[:, 1:] / spec[:, :-1]  # 信道无关特征
    chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2)
    return chan_ind_spec_amp[round(256 * 0.3):round(256 * 0.7)]

# 数据预处理主函数
def prepare_data(base_path, output_path, is_train_test_split=True):
    """处理数据并保存特征"""
    for i in range(1, 19):  # 18个设备
        device_path = os.path.join(base_path, f"device_{i}")
        if is_train_test_split:
            train_files, test_files = split_train_test(device_path)
            splits = {'train': train_files, 'test': test_files}
        else:
            data_dir = os.path.join(device_path, 'data')
            all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            splits = {'test': all_files}

        for split, files in splits.items():
            # 创建张量保存目录
            output_dir = os.path.join(output_path, f"ci_tensor_{'0309' if '0309' in base_path else '0316'}", split, f"device_{i}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建图片保存目录
            fig_dir = os.path.join(ci_fig_path, f"ci_fig_{'0309' if '0309' in base_path else '0316'}", split, f"device_{i}")
            os.makedirs(fig_dir, exist_ok=True)
            
            for j in range(0, len(files), 2):  # 每两个文件为一组
                if j + 1 >= len(files):
                    break
                try:
                    # 读取连续双帧
                    frame1 = read_signal_file(os.path.join(device_path, 'data', files[j]))
                    frame2 = read_signal_file(os.path.join(device_path, 'data', files[j+1]))
                    # 同步和CFO补偿
                    start1 = sync(frame1)
                    start2 = sync(frame2)
                    signal1 = cfo_compensation(frame1[start1:start1+8*L])
                    signal2 = cfo_compensation(frame2[start2:start2+8*L])
                    signal1 = normalized(signal1)
                    signal2 = normalized(signal2)
                    # 提取特征
                    spec1 = extract_ci_time_frequency(signal1)
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(signal1[:, :], xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
                        plt.gca().invert_yaxis()
                        
                        plt.show()
                        
                        
                    except Exception as e:
                        print(f"保存图片时出错 (设备 {i}, 配对 {j//2}): {str(e)}")
                        
                    
                        
                    spec2 = extract_ci_time_frequency(signal2)
                    # 拼接双帧特征
                    data_pair = np.vstack([spec1, spec2])
                    
                    # 保存为张量
                    # torch.save(torch.from_numpy(data_pair), os.path.join(output_dir, f"pair_{j//2}.pth"))
                    
                    # 保存为图片
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(data_pair[:, :], xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
                        plt.gca().invert_yaxis()
                        plt.savefig(os.path.join(fig_dir, f"pair_{j//2}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close()
                    except Exception as e:
                        print(f"保存图片时出错 (设备 {i}, 配对 {j//2}): {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"处理数据时出错 (设备 {i}, 配对 {j//2}): {str(e)}")
                    continue
                
            print(f"设备 {i} 的 {split} 数据处理完成")

if __name__ == "__main__":
    # 创建主图片保存目录
    os.makedirs(ci_fig_path, exist_ok=True)
    # 处理3月9日数据（划分训练和测试集）
    prepare_data(base_path_0309, output_base_path, True)
    # 处理3月16日数据（仅测试集）
    prepare_data(base_path_0316, output_base_path, False)