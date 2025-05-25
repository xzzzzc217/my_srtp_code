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
    rms_signal = signal / rms
    return rms_signal  # 归一化


#  rx_signal 是接收到的信号

def extract_phase_noise(rms_signal, upchirp, L):
    '''
    用于提取相位噪声
    :return:
    '''

    z = np.tile(upchirp, 8)  # 理想的信号z[n] 也许不需要扩展到8个
    sum_phase = [0] * 8  # 初始化数组
    angle_phase = [0] * 8
    for i in range(8):
        for n in range(L):
            sum_phase[i] += rms_signal[i * L:(i + 1) * L] * np.conj(upchirp[n])
            angle_phase[i] = np.angle(sum_phase[i] / L)  # 相位噪声。每个啁啾有一个phi。一个前导码有8个phi

    angle_phase_min = np.min(angle_phase)
    angle_phase_max = np.max(angle_phase)
    norm_phase_noise = [0] * 8
    for i in range(8):
        norm_phase_noise[i] = (angle_phase[i] - angle_phase_min) / (angle_phase_max - angle_phase_min)
    return norm_phase_noise


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
    fs = 1e6  #
    sf = 7  #
    bw = 125e3  #
    ideal_upchirp, L = generate_ideal_upchirp(fs, sf, bw)

    # 原始信号文件夹路径（注意原始路径中的反斜杠需使用原始字符串）
    base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0511_cubecell"
    #
    dest_base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\temp_phase"

    # 要处理的文件夹列表
    folders = [
        "27",
        "29",
        "32",
        "35"]

    # 遍历每个文件夹
    for folder in folders:
        data_folder = os.path.join(base_path, folder, "data")  # C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0511_cubecell
        if os.path.exists(data_folder):
            print(f"开始处理 {data_folder} 中的文件...")
            dest_path = os.path.join(dest_base_path, folder)  # 指定新文件夹路径
            try:
                # 创建文件夹
                os.mkdir(dest_path)
                print(f"文件夹 '{dest_path}' 创建成功")
            except FileExistsError:
                print(f"文件夹 '{dest_path}' 已经存在")
            except FileNotFoundError:
                print(f"父目录不存在，无法创建文件夹 '{dest_path}'")
            # 遍历 data 文件夹内的所有文件
            for file_name in os.listdir(data_folder):
                file_path = os.path.join(data_folder,
                                         file_name)  #
                if os.path.isfile(file_path):
                    save_path = os.path.join(dest_path, file_name)  # 存储地址

                    raw_data = read_file(file_path)
                    coarse_sync_data = coarse_sync(raw_data, L)  # 粗同步
                    if coarse_sync_data.size > 0:  # 检查 coarse_sync_data 是否为空
                        fine_sync_data, _, _ = fine_sync(ideal_upchirp, coarse_sync_data, fs, 8)  # 精同步
                        if fine_sync_data is not None:  # 检查 fine_sync_data 是否为 None
                            delta_f_coarse = coarse_cfo_estimation(fine_sync_data, fs)  # 算粗CFO
                            coarse_compensated_preamble = cfo_compensation(fine_sync_data, delta_f_coarse, fs)[
                                                          :8 * L]  # 粗CFO补偿
                            delta_f_fine = fine_cfo_estimation(coarse_compensated_preamble, L, fs)  # 算精CFO
                            fine_compensated_preamble = cfo_compensation(coarse_compensated_preamble, delta_f_fine, fs)[
                                                        :8 * L]  # 精CFO补偿
                            rms_preamble = rms_normalization(fine_compensated_preamble)  # rms归一化
                            phase_noise = extract_phase_noise(rms_signal=rms_preamble,upchirp = ideal_upchirp,L = L)#得到每一帧的相位噪声数组


                        else:
                            print(f"精同步失败，未找到足够的 upchirp 峰值，文件：{file_path}")
                    else:
                        print(f"粗同步失败，文件：{file_path}")


