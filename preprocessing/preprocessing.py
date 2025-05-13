# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:46:15 2023

@author: Chao
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import os
from scipy import signal
import time

from sklearn import manifold

#%%
def time_domain_plt(iq_sample,fs=1,title="IQ时域信号图",img_save_path=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}
    
    plt.figure(figsize=(8,3))
    length = len(iq_sample)
    t = np.arange(0, length/fs, 1/fs)
    plt.plot(t, np.real(iq_sample))
    plt.xlabel('t(ns)')
    plt.ylabel('Amp')
    plt.title(title)
    #===保存图片====#
    if img_save_path:
    	plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
    plt.show()
def time_frequency_domain_plt(iq_sample,wid_len=256,title="时频信号图",img_save_path=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}
    
    plt.figure(figsize=(12,3))
    Pxx, freqs, bins, im=plt.specgram(iq_sample,NFFT=wid_len,noverlap=wid_len/2)
    
    plt.xlabel('t(ns)')
    plt.ylabel('Amp')
    plt.title(title)
    if img_save_path:
    	plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
    plt.show()
def plot_spectrogram(w, fs=500e3):
     ff, tt, Sxx = signal.spectrogram(w, fs=fs, nperseg=256, nfft=576)
     plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r', shading='gouraud')
     
     plt.xlabel('t (sec)')
     plt.ylabel('Frequency (Hz)')
     plt.grid()
def generate_ideal_upchirp(fs=500e3,sf=8,bw=125e3):
    T=2**sf/bw #一个upchirp的周期
    samples = int(fs*T)#一个upchirp的离散采样的点数
    t = np.arange(samples) / fs #一个upchirp的时间坐标列表
    upchirp = -bw/2+bw/T*t #一个upchirp的瞬时频率
    #upchirp=signal.chirp(t,f0=-bw/2,f1=bw/2,t1=t[-1])
    #time_frequency_domain_plt(upchirp)
    #plot_spectrogram(upchirp)
    #plt.plot(t,upchirp)
    return upchirp,samples

def inst_freq(frame, fs):
    # Compute the analytic signal of x
    #analytic_signal = signal.hilbert(frame)
    # Compute the instantaneous phase of x_a
    instantaneous_phase = np.unwrap(np.angle(frame))
    # Compute the instantaneous frequency of x_a
    f = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    
    return f

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range




 
 
def exact_preamble(ideal_upchirp,raw_frame,prea_len=8):
    
    raw_frame_inst_freq=inst_freq(np.real(raw_frame),fs)
    ideal_upchirp_inst_freq=inst_freq(ideal_upchirp,fs)
    #ideal_upchirp_inst_freq=ideal_upchirp
    #互相关计算
    correlate_list=signal.correlate(raw_frame_inst_freq,ideal_upchirp_inst_freq,mode="valid")
    #归一化
    norm_raw_frame_inst_freq=normalization(raw_frame_inst_freq)
    norm_correlate_list=normalization(correlate_list)
    #寻找局部峰值
    preamble_start,_=signal.find_peaks(norm_correlate_list[:12000],height=0.6,distance=800)
    """
    plt.plot(norm_raw_frame_inst_freq[:12000])
    #plt.plot(norm_correlate_list[:12000])
    #plt.plot(ideal_upchirp_inst_freq)
    plt.legend(['inst_freq', 'correlate'], loc='best')
    
    
    
    plt.plot(preamble_start, norm_correlate_list[preamble_start],"x")
    plt.show()"""
    raw_preamble=raw_frame[preamble_start[0]:preamble_start[prea_len]]
    return raw_preamble
    #plt.ylim(0, 1)
def coarse_sync(rx_signal,L=1024): 
    for n in range(len(rx_signal)):
        P=np.dot(rx_signal[n:n+L], np.conj(rx_signal)[n+L:n+L+L])
        R=np.dot(rx_signal[n+L:n+L+L], np.conj(rx_signal)[n+L:n+L+L])   
        P=abs(P)
        R=abs(R)
        M=P/R
        """
        P,R,M=0.0,0.0,0.0
        for k in range(L):
            P+=rx_signal[n+k]*np.conj(rx_signal)[n+k+L]
            R+=rx_signal[n+k+L]*np.conj(rx_signal)[n+k+L]
        P=abs(P)
        R=abs(R)
        M=P/R
        """
        if M>0.95:
            return rx_signal[n:]
def fine_sync(ideal_upchirp,rx_signal,prea_len=8):
    raw_frame_inst_freq=inst_freq(rx_signal,fs)
    #ideal_upchirp_inst_freq=inst_freq(ideal_upchirp,fs)
    ideal_upchirp_inst_freq=ideal_upchirp
    
    #互相关计算
    correlate_list=signal.correlate(raw_frame_inst_freq,ideal_upchirp_inst_freq,mode="valid")

    norm_raw_frame_inst_freq=normalization(raw_frame_inst_freq)
    norm_correlate_list=normalization(correlate_list)
    #寻找局部峰值
    upchirp_start,_=signal.find_peaks(norm_correlate_list[:12000],height=0.8,distance=800)
    """
    plt.plot(norm_raw_frame_inst_freq[:12000])
    plt.plot(norm_correlate_list[:12000])
    #plt.plot(ideal_upchirp_inst_freq)
    plt.legend(['inst_freq', 'correlate'], loc='best')
    
    
    
    plt.plot(upchirp_start, norm_correlate_list[upchirp_start],"x")
    plt.show()"""
    raw_preamble=rx_signal[upchirp_start[0]:upchirp_start[prea_len]]
    return raw_preamble,upchirp_start[0],upchirp_start[1]

def coarse_CFO(rx_signal,Lstart,Lend):
    rx_signal_inst_freq=inst_freq(rx_signal,fs)
    #max_fre=np.max(rx_signal_inst_freq[:Lend-Lstart])
    #if max_fre>70e3:
    #    print("max_fre:",max_fre)
    CFO=np.mean(rx_signal_inst_freq[:Lend-Lstart])
    coarse_cfo_signal=CFO_compensation(rx_signal,CFO)
    return coarse_cfo_signal

def fine_CFO(rx_signal,Lstart,Lend):
    Ts=1/fs
    L=Lend-Lstart
    # 将相邻信号的时域样点相乘
    signal_mul = np.dot(rx_signal[:L],np.conj(rx_signal)[L:2*L])
    # 计算相位角
    phase_diff = np.angle(signal_mul)
    # 计算CFO
    CFO = - phase_diff / (2 * np.pi * L * Ts)
    print(f"CFO:{CFO}")
    fine_cfo_signal=CFO_compensation(rx_signal,CFO)
    return fine_cfo_signal
    

def CFO_compensation(rx_signal,CFO):
    Ts=1/fs
    #print("coarse:",c_CFO)
    #print("fine:",f_CFO)
    t = np.arange(0, len(rx_signal)) * Ts
    cfo_compensation_signal = (np.complex64)(np.exp(-1j * 2 * np.pi * CFO * t))
    signal_compensated = rx_signal * cfo_compensation_signal
    return signal_compensated

def datanum_count(main_path):
    datanum=0
    for i in range(len(os.listdir(main_path))):
        device_path=os.path.join(main_path,os.listdir(main_path)[i])
        list_sample_path=os.listdir(device_path)
        datanum+=len(list_sample_path)
        #for k in range(len(list_sample_path)):
    return datanum   

def single_device_datanum_count(main_path,device_index=0):
    datanum=0
    day_label_np=np.array([])
    device_class_count=0
    for i in os.listdir(main_path):
        device_class_count+=1
        day_path=os.path.join(main_path,i)
        for j in range(len(os.listdir(day_path))):
            if j==device_index:
                device_path=os.path.join(day_path,os.listdir(day_path)[j])
                list_sample_path=os.listdir(device_path)
                datanum+=len(list_sample_path)
                day_label_np=np.concatenate((day_label_np,np.full((len(list_sample_path)), device_class_count)))
    
    return datanum,day_label_np

def t_sne_show(X,y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
     
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
     
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1],  str(int(y[i])),color=plt.cm.Set1(int(y[i])), 
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
#%%

raw_frame_dataset_path=r'C:\Users\21398\Desktop\sophomore\SRTP\data\original_frame_data\processed_dataset\processed_dataset\device_3'
# = os.listdir(raw_frame_dataset_path)
ideal_upchirp,_=generate_ideal_upchirp()
fs=1e6
#pre_length_list=[]
#pre_list=[]

#with h5py.File("ideal_los_dataset.h5", "a") as f:
#os.remove("ideal_los_dataset.h5")
""""""
Mydataset = h5py.File('train_dataset_no_aug_no_cfo.h5', 'a')
Mydataset.flush()
Mydataset.clear()

#%%
time_start = time.time()
device_index=3
#tsne_num,day_label_np=single_device_datanum_count(raw_frame_dataset_path,device_index)
#tsne_data=np.zeros((tsne_num,8192),dtype=float)
#tsne_data_count=0

for i in os.listdir(raw_frame_dataset_path):
    day_path=os.path.join(raw_frame_dataset_path,i)
    #创建数据集
    dataset_num=datanum_count(day_path)

    preamble_np=np.zeros((dataset_num,8192),dtype=np.complex64)
    device_label_np=np.array([])
    device_class_count=0
    current_sample_count=0
    if i=="Day_3":
        for j in range(len(os.listdir(day_path))):
            device_class_count+=1
            device_path=os.path.join(day_path,os.listdir(day_path)[j])
            #if j==device_index:
            list_sample_path=os.listdir(device_path)
            device_label_np=np.concatenate((device_label_np,np.full((len(list_sample_path)), device_class_count)))
            for k in range(len(list_sample_path)):
                #if k==482:
                #    print(list_sample_path[k])
                sample_path=os.path.join(device_path,list_sample_path[k])
                raw_frame_IQ=np.fromfile(file=sample_path, dtype=np.complex64, count=-1, sep='', offset=0)
                #time_frequency_domain_plt(raw_frame_IQ[:12000],title="原始信号")
                #1-粗同步
                raw_frame_IQ=coarse_sync(raw_frame_IQ)
                #time_frequency_domain_plt(raw_frame_IQ[:12000],title="帧同步后的信号")
                
                #2-精同步
                raw_preamble,L_start,L_end=fine_sync(ideal_upchirp,raw_frame_IQ)
                
                #time_frequency_domain_plt(raw_preamble,title="前导码")
                
                #3-粗cfo补偿
                cCFO_preamble=coarse_CFO(raw_preamble,L_start,L_end)
                #4-精cfo补偿
                CFO_preamble=fine_CFO(cCFO_preamble,L_start,L_end)
                
                #time_frequency_domain_plt(CFO_preamble,title="CFO补偿后的前导码")
                #sys.exit()
                #5-重采样固定样本长度
                #cfo_preamble=signal.resample(CFO_preamble,8192)
                no_cfo_preamble=signal.resample(raw_preamble,8192)
                
                preamble_np[current_sample_count,:]=no_cfo_preamble.reshape(1, -1)
                #tsne_data[tsne_data_count,:]=np.real(signal.resample(raw_preamble,8192)).reshape(1, -1)
                #tsne_data[tsne_data_count,:]=np.real(cfo_preamble).reshape(1, -1)
                #tsne_data_count+=1
                current_sample_count+=1
                #sys.exit()
    
                #time_frequency_domain_plt(cfo_preamble)
    #day_group = Mydataset.create_group(i)
    #day_group.create_dataset(name='data',data=preamble_np)
    #day_group.create_dataset(name='label',data=device_label_np)
        Mydataset.create_dataset(name='data',data=preamble_np)
        Mydataset.create_dataset(name='label',data=device_label_np)
#%%

#t_sne_show(tsne_data,day_label_np)
time_end = time.time()
time_c= time_end - time_start   #运行所花时间
print('time cost', time_c, 's')


   
print(Mydataset.keys())
Mydataset.close() 
""""""
