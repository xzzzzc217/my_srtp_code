import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from scipy.fft import fftshift
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import csv
import os

# 基本参数设置
B = 125000        # 带宽 (Hz)
T = 128 / B       # 符号周期
f_sample = 1000000  # 采样频率 (1MHz)
Ts = 1 / f_sample   # 采样间隔
L = int(T*f_sample) # 每个符号的采样点数
n = np.arange(0, L, 1)
ideal_sample = np.exp(complex(0, 1) * (-1 * np.pi * B * n * Ts + np.pi * B / T * pow(n * Ts, 2)))

def normalized1(x):
    """
    最大最小值归一化
    参数:
        x: 输入信号
    返回:
        归一化后的信号 (范围: 0-1)
    """
    x = (x - x.min()) / (x.max() - x.min())
    return x

def cfo_compensation(sample):
    """
    载波频偏(CFO)补偿
    参数:
        sample: 输入信号
    返回:
        cfo_corr: CFO补偿后的信号
        cfo_coarse + cfo_fine: 总的频偏估计值
    """
    #粗估
    angle = np.unwrap(np.angle(sample[0:L]))
    freq = np.diff(angle) / (2 * np.pi) * f_sample
    cfo_coarse=sum(freq) / L
    #粗补偿
    n = np.arange(0, 8*L, 1)
    cfo_corr_coarse=sample[0:8*L]*np.exp(complex(0, -1) * 2 * np.pi * cfo_coarse*n *Ts)

    #细估
    temp=np.angle(sum(cfo_corr_coarse[0:L]*np.conj( cfo_corr_coarse[L:2*L])))
    cfo_fine=temp/(-2*np.pi*T)
    #细补偿
    cfo_corr=cfo_corr_coarse[0:8*L]*np.exp(complex(0, -1) * 2 * np.pi * cfo_fine*n *Ts)

    return cfo_corr,cfo_coarse+cfo_fine

def sync(sample):
    """
    信号同步
    参数:
        sample: 输入信号
        test: 测试计数器
    返回:
        index: 同步后的起始位置
    """
    #粗同步
    for i in range(len(sample)):
        a=array(sample[i:i+L]*np.conj(sample[i+L:i+L+L])).sum()
        b=array(sample[i+L:i+L+L]*np.conj(sample[i+L:i+L+L])).sum()
        M=abs(a)/abs(b)
        if M>0.95:
            start_coarse=i
            break

    #精同步
    n = np.arange(0, L, 1)
    ideal_freq = -B / 2 + (B / T) * n * Ts
    ideal = np.abs(np.correlate(ideal_freq, ideal_freq))[0]

    angle=np.unwrap(np.angle(sample))
    freq=np.diff(angle)/(2*np.pi)*f_sample
    th=0
    index=0

    for i in range(start_coarse,start_coarse+L):
        if freq[i]<-80000 or freq[i]>80000:
            freq[i]=0
    for j in range(start_coarse,start_coarse+L):
        t = np.abs(np.correlate(ideal_freq, freq[j:j + L]))[0]
        if t/ideal>th:
            th=t/ideal
            index=j
    return index

def cal_phase(sample):
    """
    计算相位误差
    参数:
        sample: 输入信号
    返回:
        相位误差数组
    """
    p=[]
    for i in range(8):
        ph=np.angle(sum(sample[i*L:(i+1)*L]*np.conj(ideal_sample))/L)
        p.append(ph)
    return array(p)

def find_center(x,y):
    """
    使用K-means聚类找到数据中心点
    参数:
        x: x坐标
        y: y坐标数组
    返回:
        聚类中心的y坐标
    """
    data=[]
    for i in range(len(y)):
        data.append([x,y[i]])
    data=array(data)
    kmeans=KMeans(n_clusters=1,random_state=0)
    y=kmeans.fit_predict(data)
    return kmeans.cluster_centers_[0][1]

if __name__ == '__main__':
    for ii in range(1, 10):
        device = "device_"+str(ii)
        index=1
        path="C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\"+device+"\\data"
        res_cfo=[]
        res_phase=[]
        x=[]
        y=[]
        y1=[[],[],[],[],[],[],[],[]]
        res_y=[]
        data_type="train_data"

        while index<500:
            filepath1=path+"\\"+str(index)
            var1 = np.fromfile(filepath1, dtype=np.complex64)

            index+=1
            filepath2=path+"\\"+str(index)
            var2 = np.fromfile(filepath2, dtype=np.complex64)

            start_index1=sync(var1)
            var1,cfo1=cfo_compensation(var1[start_index1:start_index1+8*L])
            rssi1=10 * np.log(np.mean(np.abs(var1) ** 2))
            print(index-1,cfo1,rssi1)

            ph1=cal_phase(var1)

            start_index2=sync(var2)
            var2,cfo2=cfo_compensation(var2[start_index2:start_index2+8*L])
            rssi2=10*np.log(np.mean(np.abs(var2)**2))
            print(index,cfo2,rssi2,cfo2-cfo1)
            
            index += 1
            if rssi1>=rssi2:
                print("data error,next one")
            else:
                # 采集cfo
                res_cfo.append([cfo1,cfo2])

                # 采集相位差
                ph2=cal_phase(var2)
                temp=normalized1(ph2)-normalized1(ph1)
                for j in range(8):
                    x.append(j)
                    y.append(temp[j])
                    y1[j].append(temp[j])

                temp1=np.append(temp,[cfo1,cfo2])
                res_phase.append(temp1)

        # # 保存相位数据
        # for j in range(8):
        #     res_y.append(find_center(j,y1[j]))
        # data = np.vstack([x, y])
        # kde = gaussian_kde(data)(data)
        # plt.scatter(x, y, c=kde, s=100)
        # plt.colorbar()
        # plt.plot(np.arange(0,8,1),res_y,'r*')
        # plt.show()

        cnt=0
        with open("C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\" + data_type + device +".csv",'w',newline='') as cfile:
            w=csv.writer(cfile)
            #print("ph pre", cnt)
            for j in range(len(res_phase)):
                w.writerow(res_phase[j])
                cnt += 1
        print("ph sum",cnt)