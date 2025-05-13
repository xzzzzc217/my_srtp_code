import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from scipy.fftpack import fft
import csv
import os
from numpy import sum,sqrt
from numpy.random import standard_normal
from math import log10

# 基本参数设置
B = 125000        # 带宽 (Hz)
T = 128 / B       # 符号周期
f_sample = 1000000  # 采样频率 (Hz)
Ts = 1 / f_sample   # 采样间隔
L = int(T*f_sample) # 每个符号的采样点数
n = np.arange(0, L, 1)

# 生成理想信号模板
ideal_sample = np.exp(complex(0, 1) * (-1 * np.pi * B * n * Ts + np.pi * B / T * pow(n * Ts, 2)))
down_ideal = np.exp(complex(0, 1) * (1 * np.pi * B * n * Ts - np.pi * B / T * pow(n * Ts, 2)))

def normalized1(x):
    """
    对信号进行最大最小值归一化
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
    # 粗估计
    angle = np.unwrap(np.angle(sample[0:L]))
    freq = np.diff(angle) / (2 * np.pi) * f_sample
    cfo_coarse = sum(freq) / L
    
    # 粗补偿
    n = np.arange(0, 8*L, 1)
    cfo_corr_coarse = sample[0:8*L] * np.exp(complex(0, -1) * 2 * np.pi * cfo_coarse * n * Ts)
    
    # 细估计
    temp = np.angle(sum(cfo_corr_coarse[0:L] * np.conj(cfo_corr_coarse[L:2*L])))
    cfo_fine = temp / (-2 * np.pi * T)
    
    # 细补偿
    cfo_corr = cfo_corr_coarse[0:8*L] * np.exp(complex(0, -1) * 2 * np.pi * cfo_fine * n * Ts)
    
    return cfo_corr, cfo_coarse + cfo_fine

def sync(sample):
    """
    信号同步
    参数:
        sample: 输入信号
    返回:
        index: 同步后的起始位置
    """
    # 粗同步：通过相邻符号相关
    for i in range(len(sample)):
        a = array(sample[i:i+L] * np.conj(sample[i+L:i+L+L])).sum()
        b = array(sample[i+L:i+L+L] * np.conj(sample[i+L:i+L+L])).sum()
        M = abs(a) / abs(b)
        if M > 0.95:
            start_coarse = i
            break
    
    # 精同步：通过频率特征
    n = np.arange(0, L, 1)
    ideal_freq = -B / 2 + (B / T) * n * Ts
    ideal = np.abs(np.correlate(ideal_freq, ideal_freq))[0]
    
    angle = np.unwrap(np.angle(sample))
    freq = np.diff(angle) / (2 * np.pi) * f_sample
    th = 0
    index = 0
    
    for i in range(start_coarse, start_coarse+L):
        if freq[i] < -90000 or freq[i] > 90000:
            freq[i] = 0
    for j in range(start_coarse, start_coarse+L):
        t = np.abs(np.correlate(ideal_freq, freq[j:j + L]))[0]
        if t/ideal > th:
            th = t/ideal
            index = j
    
    return index

def cal_phase(sample):
    """
    计算信号相位误差
    参数:
        sample: 输入信号
    返回:
        相位误差数组
    """
    p = []
    for i in range(8):
        ph = np.angle(sum(sample[i*L:(i+1)*L] * np.conj(ideal_sample)) / L)
        p.append(ph)
    return array(p)

def snr(data):
    """
    计算信号信噪比
    参数:
        data: 输入信号
    返回:
        平均信噪比(dB)
    """
    s = 0
    for i in range(8):
        temp = data[i * L: i * L + L] * down_ideal
        temp = abs(fft(temp)) ** 2
        maxid = np.argmax(temp)
        tmax = 0
        for i in range(-9, 10):
            if maxid+i >= L:
                tmax += temp[(maxid+i) % L]
            else:
                tmax += temp[maxid + i]
        s += 10 * log10(tmax / (sum(temp) - tmax))
    return s / 8

def awgn(data, SNRdB):
    """
    添加高斯白噪声
    参数:
        data: 输入信号
        SNRdB: 目标信噪比(dB)
    返回:
        添加噪声后的信号
    """
    print(snr(data))
    s = data
    SNR_linear = 10 ** (SNRdB / 10)
    P = sum(abs(s)**2) / len(s)
    N0 = P / SNR_linear
    n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
    data = s + n
    print(snr(data))
    return data
from scipy import signal
def time_frequency_domain_plt(iq_sample,wid_len=256,title="时频信号图",img_save_path=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}
    
    plt.figure(figsize=(12,3))
    Pxx, freqs, bins, im=plt.specgram(iq_sample,NFFT=wid_len,noverlap=wid_len//2)
    
    plt.xlabel('t(ns)')
    plt.ylabel('Amp')
    plt.title(title)
    if img_save_path:
    	plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
    plt.show()
    
if __name__ == '__main__':
    """
    主函数：处理多个设备的信号数据
    功能：
    1. 遍历9个设备的数据
    2. 对每个设备处理500帧数据
    3. 进行信号处理：同步、噪声添加、CFO补偿、相位计算
    4. 保存处理结果
    """
    # 遍历30个设备
    for ii in range(1, 10):
        device = "device_"+str(ii)
        index = 1
        path = "C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\"+device+"\\data"
        
        # 初始化结果存储数组
        res_cfo = []      # 存储CFO结果
        res_phase = []    # 存储相位结果
        x = []
        y = []
        y1 = [[] for _ in range(16)]  # 存储相位数据
        res_y = []
        
        # 设置处理参数
        data_type = "test_data"
        #snrdb = 11
        
        # 处理500帧数据
        while index < 500:
            # 读取并处理第一帧
            filepath1 = path + "\\" + str(index)
            index += 1
            var1 = np.fromfile(filepath1, dtype=np.complex64)
            #time_frequency_domain_plt(iq_sample=var1)
            start_index1 = sync(var1)
            var1=var1[start_index1:start_index1 + 8 * L]
            #time_frequency_domain_plt(iq_sample=var1)
            #var1 = awgn(var1[start_index1:start_index1 + 8 * L], snrdb)
            var1, cfo1 = cfo_compensation(var1)
            #time_frequency_domain_plt(iq_sample=var1)
            #time_frequency_domain_plt(iq_sample=var1)
            ph1 = cal_phase(var1)
            
            # 读取并处理第二帧
            filepath2 = path + "\\" + str(index)
            var2 = np.fromfile(filepath2, dtype=np.complex64)
            start_index2 = sync(var2)
            var2=var2[start_index2:start_index2 + 8 * L]
            #var2 = awgn(var2[start_index2:start_index2 + 8 * L], snrdb)
            var2, cfo2 = cfo_compensation(var2)
            ph2 = cal_phase(var2)
            
            # 存储CFO结果
            res_cfo.append([cfo1, cfo2])
            
            # 计算并存储相位差
            temp = normalized1(ph2) - normalized1(ph1)
            for j in range(8):
                x.append(j)
                y.append(temp[j])
                y1[j].append(temp[j])
            
            # 存储结果
            temp1 = np.append(temp, [cfo1, cfo2])
            res_phase.append(temp1)
            index += 1
        
        # 保存处理结果到CSV文件
        cnt = 0
        with open("C:\\Users\\czc_seu\\Desktop\\diff_adf_code\\raw_dataset\\" + data_type + device +".csv", 'w', newline='') as cfile:
            w = csv.writer(cfile)
            for j in range(len(res_phase)):
                w.writerow(res_phase[j])
                cnt += 1