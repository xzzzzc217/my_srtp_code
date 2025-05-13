import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt


def generate_ideal_upchirp(fs, sf, bw):
    T = 2 ** sf / bw
    samples = int(fs * T)  # samples就是L
    t = np.arange(samples) / fs
    upchirp = -bw / 2 + bw / T * t
    return upchirp, samples


def inst_freq(frame, fs):
    instantaneous_phase = np.unwrap(np.angle(frame))
    f = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    return f


def zip_inst_freq(inst_freq_list):
    zip_inst_freq_list = inst_freq_list / (4 * bw)
    middle = (np.max(zip_inst_freq_list) + np.min(zip_inst_freq_list)) / 2  # 定义中轴线
    zip_inst_freq_list = middle + zip_inst_freq_list  # 将压缩之后的函数向上平移
    return zip_inst_freq_list + 0.5


def time_freq_plt(file_content, L, Fs, ylim=200e3):
    '''
    定义一个绘制时频图的函数
    :return:
    '''

    plt.figure(figsize=(10, 8))
    # 原始信号时频图（含噪声
    plt.subplot(6, 1, 1)
    print('原始信号内容：')
    print(file_content)
    plt.specgram(file_content[:12000], Fs=Fs, cmap='viridis')  # Fs 为采样频率，可根据需要调
    plt.ylim(-ylim, ylim)  # 对应带宽 125 kHz  # 设置纵轴范围，适配信号频率范围
    plt.title('raw_data_time_freq_plt')

    # 粗同步之后的时频图
    coarse_sync_info = coarse_sync(file_content, L)

    plt.subplot(6, 1, 2)
    print('粗同步之后信号内容：')
    print(coarse_sync_info)
    plt.specgram(coarse_sync_info[:8 * L], Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    plt.title('coarse_sync_data_time_freq_plt')

    # 精同步之后的时频图
    plt.subplot(6, 1, 3)
    fine_sync_info, first_upchirp_start, secend_upchirp_start = fine_sync(ideal_upchirp, coarse_sync_info, Fs, 8)
    print('精同步之后信号内容：')
    print(fine_sync_info)
    raw_preamble = fine_sync_info
    plt.specgram(fine_sync_info, Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    plt.title('fine_sync_data_time_freq_plt_aka_raw_preamble')

    # 粗 CFO 补偿
    delta_f_coarse = coarse_cfo_estimation(raw_preamble, Fs)
    print('粗cfo补偿的delta')
    print(delta_f_coarse)
    coarse_compensated_preamble = cfo_compensation(raw_preamble, delta_f_coarse, Fs)

    # 精 CFO 补偿
    delta_f_fine = fine_cfo_estimation(coarse_compensated_preamble, L, Fs)
    print('精cfo补偿的delta')
    print(delta_f_fine)
    fine_compensated_preamble = cfo_compensation(coarse_compensated_preamble, delta_f_fine, Fs)

    # 粗CFO补偿之后的时频图
    plt.subplot(6, 1, 4)
    print('粗CFO补偿之后信号内容：')
    print(coarse_compensated_preamble)
    plt.specgram(coarse_compensated_preamble[:8 * L], Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    plt.title('coarse CFO Compensated Preamble Time-Frequency Plot')

    # 精CFO补偿后时频图
    plt.subplot(6, 1, 5)
    print('精CFO补偿之后信号内容：')
    print(fine_compensated_preamble)
    plt.specgram(fine_compensated_preamble[:8 * L], Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    plt.title('Fine CFO Compensated Preamble Time-Frequency Plot')

    # 归一化后的时频图
    normalized_preamble = rms_normalization(fine_compensated_preamble)
    plt.subplot(6, 1, 6)
    print('归一化之后信号内容：')
    print(normalized_preamble)
    plt.specgram(normalized_preamble[:8 * L], Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    plt.title('Normalized Preamble Time-Frequency Plot')

    plt.tight_layout()  # 调整子图间距
    plt.show()

    # 精CFO补偿后时频图单独再放一遍
    plt.figure()
    plt.specgram(fine_compensated_preamble[:8 * L], Fs=Fs, cmap='viridis')
    plt.ylim(-ylim, ylim)
    # 添加辅助横线
    plt.axhline(y=bw / 2, color='red', linestyle='--', linewidth=1)  # 在 y=50000 添加红色虚线
    plt.axhline(y=-bw / 2, color='red', linestyle='--', linewidth=1)  # 在 y=-50000 添加红色虚线
    plt.title('Fine CFO Compensated Preamble Time-Frequency Plot')
    plt.show()
    # 功能图
    raw_frame = coarse_sync_info
    raw_frame_inst_freq = inst_freq(raw_frame, fs)  # 原始帧的实时频率
    ideal_upchirp_inst_freq = ideal_upchirp
    # 互相关计算
    correlate_list = signal.correlate(raw_frame_inst_freq, ideal_upchirp_inst_freq, mode="valid")
    # 压缩
    zip_inst_freq_list = zip_inst_freq(raw_frame_inst_freq)
    # 归一化
    norm_raw_frame_inst_freq = normalization(raw_frame_inst_freq)
    norm_correlate_list = normalization(correlate_list)
    # 寻找局部峰值
    preamble_start, _ = signal.find_peaks(norm_correlate_list[:12000], height=0.6, distance=800)

    plt.plot(zip_inst_freq_list[:12000])
    plt.plot(norm_correlate_list[:12000])
    # plt.plot(ideal_upchirp_inst_freq)
    plt.legend(['inst_freq', 'correlate'], loc='best')

    plt.plot(preamble_start, norm_correlate_list[preamble_start], "x")
    plt.show()


def coarse_sync(rx_signal, L):
    for n in range(len(rx_signal) - 2 * L):
        P = np.dot(rx_signal[n:n + L], np.conj(rx_signal)[n + L:n + 2 * L])
        R = np.dot(rx_signal[n + L:n + 2 * L], np.conj(rx_signal)[n + L:n + 2 * L])
        P = abs(P)
        R = abs(R)
        M = P / R
        if M > 0.95:
            return rx_signal[n:]
    return np.array([])  # 返回空数组，而不是 None 或一个整数


def normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def fine_sync(ideal_upchirp, rx_signal, fs, prea_len=8):
    raw_frame_inst_freq = inst_freq(rx_signal, fs)
    ideal_upchirp_inst_freq = ideal_upchirp

    correlate_list = signal.correlate(raw_frame_inst_freq, ideal_upchirp_inst_freq, mode="valid")
    norm_correlate_list = normalization(correlate_list)

    upchirp_start, _ = signal.find_peaks(norm_correlate_list[:12000], height=0.8, distance=800)

    if len(upchirp_start) < prea_len:
        print("未找到足够的upchirp峰值")
        return None, None, None

    raw_preamble = rx_signal[upchirp_start[0]:upchirp_start[prea_len - 1] + len(ideal_upchirp)]
    return raw_preamble, upchirp_start[0], upchirp_start[1]


def coarse_cfo_estimation(preamble, fs):
    """估计粗 CFO"""
    inst_freq_preamble = inst_freq(preamble, fs)  # 计算瞬时频率
    delta_f_coarse = np.mean(inst_freq_preamble)  # 取平均值
    return delta_f_coarse


def cfo_compensation(signal, delta_f, fs):
    """CFO 补偿（粗和精共用）"""
    t = np.arange(len(signal)) / fs  # 时间向量
    compensation_factor = np.exp(-1j * 2 * np.pi * delta_f * t)  # 补偿因子
    compensated_signal = signal * compensation_factor  # 相乘补偿
    return compensated_signal


def fine_cfo_estimation(signal, L, fs):
    """估计精 CFO"""
    sum_term = 0
    for n in range(L):
        sum_term += signal[n] * np.conj(signal[n + L])  # 自相关求和
    angle = np.angle(sum_term)  # 计算相位角
    delta_f_fine = -angle / (2 * np.pi * (1 / fs) * L)  # 精 CFO
    return delta_f_fine


def rms_normalization(signal):
    """RMS 归一化"""
    rms = np.sqrt(np.mean(np.abs(signal) ** 2))  # 计算 RMS
    return signal / rms  # 归一化


#  rx_signal 是接收到的信号

def read_file(file_path):
    """
    读取文件内容。
    """
    try:
        signal = np.fromfile(file_path, dtype=np.complex64)
        return signal
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
        return None


if __name__ == "__main__":
    # 示例调用
    fs = 1e6  # 采样率
    sf = 7  # 扩展因子
    bw = 125e3  # 带宽
    ideal_upchirp, L = generate_ideal_upchirp(fs, sf, bw)
    folder_path_data = r"C:\Users\21398\Desktop\sophomore\SRTP\data\original_frame_data\dataset_1229\dataset_1229\processed_dataset\de1\data"
    save_folder_path = r"C:\Users\21398\Desktop\sophomore\SRTP\test"  # 将处理后的文件放在这里

    for file in sorted(os.listdir(folder_path_data)):
        data_file_path = os.path.join(folder_path_data, file)
        save_path = os.path.join(save_folder_path, file)

        # 读取数据文件
        raw_data = read_file(data_file_path)
        if raw_data is None:
            print(f"读取文件失败，跳过文件：{data_file_path}")
            continue

        coarse_sync_data = coarse_sync(raw_data, L)  # 粗同步
        if coarse_sync_data.size > 0:  # 检查 coarse_sync_data 是否为空
            fine_sync_data, _, _ = fine_sync(ideal_upchirp, coarse_sync_data, fs, 8)  # 精同步
            if fine_sync_data is not None:  # 检查 fine_sync_data 是否为 None
                delta_f_coarse = coarse_cfo_estimation(fine_sync_data, fs)  # 算粗CFO
                coarse_compensated_preamble = cfo_compensation(fine_sync_data, delta_f_coarse, fs)[:8 * L]  # 粗CFO补偿
                delta_f_fine = fine_cfo_estimation(coarse_compensated_preamble, L, fs)  # 算精CFO
                fine_compensated_preamble = cfo_compensation(coarse_compensated_preamble, delta_f_fine, fs)[
                                            :8 * L]  # 精CFO补偿
                rms_preamble = rms_normalization(fine_compensated_preamble)  # rms归一化
                print('rms归一化后的信号内容')
                print(rms_preamble)
                try:
                    # 将复数数组转换为字节流并保存
                    arr = np.array(rms_preamble, dtype=np.complex64)
                    arr.tofile(save_path)
                    print(f"复数数组已成功保存到：{save_path}")
                except Exception as e:
                    print(f"保存复数数组时出错：{e}")
            else:
                print(f"精同步失败，未找到足够的 upchirp 峰值，文件：{data_file_path}")
        else:
            print(f"粗同步失败，文件：{data_file_path}")



