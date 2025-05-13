import numpy as np
import matplotlib.pyplot as plt
from numpy import array, sign, zeros
from scipy import signal
from scipy.fft import fftshift
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import csv
import os
from numpy import sum,sqrt
from numpy.random import standard_normal,uniform
from math import log10
from sklearn import metrics

B = 125000
T = 128 / B
f_sample = 5000000
Ts = 1 / f_sample
L=int(T*f_sample)
n = np.arange(0, L, 1)
ideal_sample = np.exp(complex(0, 1) * (-1 * np.pi * B * n * Ts + np.pi * B / T * pow(n * Ts, 2)))
down_ideal=np.exp(complex(0, 1) * (1 * np.pi * B * n * Ts - np.pi * B / T * pow(n * Ts, 2)))

#归一化
def normalized(x):
    am = 0.0
    for i in range(len(x)):
        am = am + abs(x[i])
    am = am / len(x)
    if am != 0:
        x = x / am
    return x

def normalized1(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def normalized2(x):
    x = x - x.min()
    return x


#cfo补偿
def cfo_compensation(sample):
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

#同步
def sync(sample,test):
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
        if freq[i]<-90000 or freq[i]>90000:
            freq[i]=0
    for j in range(start_coarse,start_coarse+L):
        t = np.abs(np.correlate(ideal_freq, freq[j:j + L]))[0]
        if t/ideal>th:
            th=t/ideal
            index=j


    # n=np.arange(0,index+1,1)
    # plt.plot(n,freq[0:index+1])
    # plt.show()
    # plt.cla()
    # n=np.arange(0,8*L,1)
    # plt.plot(n,freq[index:index+8*L])
    # plt.show()
    # plt.cla()
    return index

#计算相位误差
def cal_phase(sample):
    p=[]
    for i in range(8):
        ph=np.angle(sum(sample[i*L:(i+1)*L]*np.conj(ideal_sample))/L)
        p.append(ph)
    return array(p)

def my_fft(var):
    f, t, Zxx = signal.stft(var,fs=f_sample,scaling='psd',boundary=None)
    # y=fft(var)
    # y=normalized(y)
    # plt.plot(np.real(y),np.imag(y),'.')
    # plt.show()
    plt.pcolormesh(t, fftshift(f), fftshift(abs(Zxx) ** 2, axes=0))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    Zxx1=[]
    for i in range(256):
        temp=[]
        for j in range(1,len(Zxx[0])):
            temp.append(Zxx[i][j]/Zxx[i][j-1])
        Zxx1.append(temp)
    Zxx1=array(Zxx1)
    plt.pcolormesh(t[1:], fftshift(f), fftshift(abs(Zxx1)**2,axes=0))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def find_center(x,y):
    data=[]
    for i in range(len(y)):
        data.append([x,y[i]])
    data=array(data)
    kmeans=KMeans(n_clusters=1,random_state=0)
    y=kmeans.fit_predict(data)
    # plt.scatter(data[:, 0], data[:, 1], c=y)
    # plt.plot(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],'r.')
    # plt.show()
    return kmeans.cluster_centers_[0][1]

def channel_est(sample):
    f_ideal=fft(ideal_sample)
    s_f=fft(sample)
    h=f_ideal/s_f
    temp=ifft(h)
    temp[1:L-1]=0
    h1=fft(temp)
    return h1

def channel_compensation(h,sample):
    res=ifft(fft(sample[0:L]) * h)
    for i in range(1,8):
        res=np.concatenate((res,ifft(fft(sample[i*L:(i+1)*L])*h)))
    return res

# 添加信噪比
def snr(data):
    s = 0
    for i in range(8):
        temp = data[i * L: i * L + L] * down_ideal
        temp = abs(fft(temp)) ** 2
        maxid=np.argmax(temp)
        # plt.plot(np.arange(128),temp)
        # plt.show()
        tmax=0
        for i in range(-9,10):#-9,10
            if maxid+i>=L:
                tmax+=temp[(maxid+i)%L]
            else:
                tmax += temp[maxid + i]
        # for j in range(L):
        #     if temp[j] > 1000:
        #         tmax += temp[j]
        s += 10 * log10(tmax / (sum(temp) - tmax))
    return s / 8

def awgn(data,SNRdB):
    print(snr(data))

    s = data
    SNR_linear = 10 ** (SNRdB / 10)
    P = sum(abs(s)**2)/len(s)
    N0 = P / SNR_linear
    n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
    data = s + n

    print(snr(data))
    return data

if __name__ == '__main__':
    for ii in range(1,31):
        device=str(ii)
        index=1
        path="/mnt/data/hewei/data-final/device"+device
        # path2 = "C:\\Users\\FAD18\\Desktop\\res-star\\device"+device
        res_cfo=[]
        res_phase=[]
        x=[]
        y=[]
        y1=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]#2*8
        res_y=[]
        cc=0
        data_type="test_data_10db"#train_data#test_data#test_data_nlos#test_data_od
        snrdb=11#14,6
        #600，-60，40
        #缺设备1的14db

        while cc<500:
            filepath1 = path + "/test_data/frame" + str(index)
            #
            index += 1
            filepath2 = path + "/test_data/frame" + str(index)


            var1 = np.fromfile(filepath1, dtype=np.complex64)
            start_index1 = sync(var1, cc)
            var1 = awgn(var1[start_index1:start_index1 + 8 * L], snrdb)
            var1, cfo1 = cfo_compensation(var1)
            ph1 = cal_phase(var1)
            print(index - 1, cfo1)

            var2 = np.fromfile(filepath2, dtype=np.complex64)
            start_index2 = sync(var2, cc)
            var2 = awgn(var2[start_index2:start_index2 + 8 * L], snrdb)
            var2, cfo2 = cfo_compensation(var2)
            ph2 = cal_phase(var2)
            print(index, cfo2, cfo2 - cfo1)

            # start_index1 = sync(var1, cc)
            # var1 = awgn(var1[start_index1:start_index1 + 8 * L], snrdb)
            # var1, cfo1 = cfo_compensation(var1)
            # rssi1 = 10 * np.log(np.mean(np.abs(var1) ** 2))
            # print(index - 1, cfo1, rssi1)

            # start_index2 = sync(var2, cc)
            # var2=awgn(var2[start_index2:start_index2 + 8 * L],snrdb)
            # var2, cfo2 = cfo_compensation(var2)
            # rssi2 = 10 * np.log(np.mean(np.abs(var2) ** 2))
            # print(index, cfo2, rssi2, cfo2 - cfo1)

            # 保存原始数据
            # tp=os.listdir(path+"\\"+data_type)
            # if len(tp)==0:
            #     os.rename(filepath1,path+"\\"+data_type+"\\frame1")
            #     os.rename(filepath2, path + "\\"+data_type+"\\frame2")
            # else:
            #     tp.sort(key=lambda x:int(x[5:]))
            #     os.rename(filepath1, path + "\\"+data_type+"\\frame"+str(int(tp[-1][5:])+1))
            #     os.rename(filepath2, path + "\\"+data_type+"\\frame" + str(int(tp[-1][5:]) + 2))

            # 采集cfo
            res_cfo.append([cfo1,cfo2])

            # 采集相位差
            temp=normalized1(ph2)-normalized1(ph1)
            for j in range(1*8):
                x.append(j)
                y.append(temp[j])
                y1[j].append(temp[j])

            # plt.plot(np.arange(0,8,1),temp)
            # # plt.show()
            # filepath_fig = path2 + "\\ares" + str(cc) + ".jpg"
            # plt.savefig(filepath_fig)
            # plt.cla()

            temp1=np.append(temp,[cfo1,cfo2])
            res_phase.append(temp1)
            index += 1

            cc+=1

        # # 保存cfo数据
        # file = open(path+"\\cfo_database_20.csv")
        # reader = csv.reader(file)
        # original = list(reader)
        # cnt_cfo=0
        # with open(path + "\\cfo_database_"+data_type+".csv", 'w', newline='') as cfile:
        #     w = csv.writer(cfile)
        #     # for row in original:
        #     #     w.writerow(row)
        #     #     cnt_cfo+=1
        #     print("cfo pre:",cnt_cfo)
        #     for i in range(len(res_cfo)):
        #         w.writerow(res_cfo[i])
        #         cnt_cfo += 1
        #     print("cfo sum:",cnt_cfo)

        # for j in range(8):
        #     res_y.append(find_center(j,y1[j]))
        # data = np.vstack([x, y])
        # kde = gaussian_kde(data)(data)
        # plt.scatter(x, y, c=kde, s=100)
        # plt.colorbar()
        # plt.plot(np.arange(0,8,1),res_y,'r*')
        # plt.savefig(path+"\\"+data_type+".jpg")
        # plt.show()

        # 保存相位数据
        # file = open(path+"\\"+data_type+".csv")
        # reader = csv.reader(file)
        # original = list(reader)
        cnt=0
        with open(path+"/"+data_type+".csv",'w',newline='') as cfile:
            w=csv.writer(cfile)
            # for row in original:
            #     w.writerow(row)
            #     cnt+=1
            print("ph pre", cnt)
            for j in range(len(res_phase)):
                w.writerow(res_phase[j])
                cnt += 1
        print("ph sum",cnt)

    # while cc < 100:
    #     filepath1 = path + "\\frame\\frame" + str(index)
    #     var1 = np.fromfile(filepath1, dtype=np.complex64)
    #     start_index1 = sync(var1, cc)
    #     var1, cfo1 = cfo_compensation(var1[start_index1:start_index1 + 8 * L])
    #     rssi1 = 10 * np.log(np.mean(np.abs(var1) ** 2))
    #     print(index, cfo1, rssi1)
    #
    #     # start_index1 = sync(var1, cc)
    #     # var1 = awgn(var1[start_index1:start_index1 + 8 * L], snrdb)
    #     # var1, cfo1 = cfo_compensation(var1)
    #     # rssi1 = 10 * np.log(np.mean(np.abs(var1) ** 2))
    #     # print(index - 1, cfo1, rssi1)
    #
    #     ph1 = cal_phase(var1)
    #     # 保存原始数据
    #     tp = os.listdir(path + "\\" + data_type)
    #     if len(tp) == 0:
    #         os.rename(filepath1, path + "\\" + data_type + "\\frame1")
    #     else:
    #         tp.sort(key=lambda x: int(x[5:]))
    #         os.rename(filepath1, path + "\\" + data_type + "\\frame" + str(int(tp[-1][5:]) + 1))
    #
    #
    #     # 采集相位差
    #     temp = normalized1(ph1)
    #     for j in range(8):
    #         x.append(j)
    #         y.append(temp[j])
    #         y1[j].append(temp[j])
    #
    #     # plt.plot(np.arange(0,8,1),temp)
    #     # # plt.show()
    #     # filepath_fig = path2 + "\\ares" + str(cc) + ".jpg"
    #     # plt.savefig(filepath_fig)
    #     # plt.cla()
    #     res_phase.append(temp)
    #     index += 1
    #
    #     cc += 1
    #
    # #保存相位数据
    # for j in range(8):
    #     res_y.append(find_center(j,y1[j]))
    # data = np.vstack([x, y])
    # kde = gaussian_kde(data)(data)
    # # plt.scatter(x, y, c=kde, s=100)
    # # plt.colorbar()
    # # plt.plot(np.arange(0,8,1),res_y,'r*')
    # # plt.show()
    #
    # # file = open(path+"\\"+data_type+".csv")
    # # reader = csv.reader(file)
    # # original = list(reader)
    # cnt=0
    # with open(path+"\\"+data_type+".csv",'w',newline='') as cfile:
    #     w=csv.writer(cfile)
    #     # for row in original:
    #     #     w.writerow(row)
    #     #     cnt+=1
    #     print("ph pre", cnt)
    #     for j in range(len(res_phase)):
    #         w.writerow(res_phase[j])
    #         cnt += 1
    # print("ph sum",cnt)

            # j=1
            # # test=(var1[j*L:L*(j+1)]*np.exp(complex(0,-1)*ph1[j])) *\
            # #          np.conj(var2[(j)*L:(j+1)*L]*np.exp(complex(0,-1)*ph2[j]))
            #
            # test = (var1[j * L:L * (j + 7)]) * \
            #        np.conj(var2[(j) * L:(j + 7) * L])
            # center_point=find_center(test)
            # #
            # plt.xlim(-1.5, 1.5)
            # plt.ylim(-1.5, 1.5)
            # plt.plot(np.real(test), np.imag(test),'r.')
            # filepath_fig = path2+"\\res" + str(i) + ".jpg"
            # plt.savefig(filepath_fig)
            # plt.cla()

            # if sum_flag:
            #     res_center.append(center_point)
            #     if i==116:
            #         # plt.plot(np.real(var1), np.imag(var1))
            #         plt.xlim(-1.5,1.5)
            #         plt.ylim(-1.5,1.5)
            #
            #         data=np.vstack([np.real(res_center),np.imag(res_center)])
            #         kde = gaussian_kde(data)(data)
            #         plt.scatter(np.real(res_center), np.imag(res_center), c=kde, s=100)
            #         plt.colorbar()
            #         # plt.show()
            #         # plt.plot(np.real(res), np.imag(res), 'r.')
            #         # plt.plot(np.real(test2), np.imag(test2), 'g.')
            #         filepath_fig=path2+"\\test0220_sum.jpg"
            #         plt.savefig(filepath_fig)
            #         plt.cla()