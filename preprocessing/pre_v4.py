import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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

    # 动态调整搜索范围，避免超出信号长度
    search_len = min(12000, len(norm_correlate_list))
    upchirp_start, _ = signal.find_peaks(norm_correlate_list[:search_len], height=0.8, distance=800)

    # 如果找不到足够的峰值，降低要求
    if len(upchirp_start) < prea_len:
        # 尝试降低阈值
        upchirp_start, _ = signal.find_peaks(norm_correlate_list[:search_len], height=0.6, distance=600)
        if len(upchirp_start) < prea_len:
            # 再次降低要求
            upchirp_start, _ = signal.find_peaks(norm_correlate_list[:search_len], height=0.4, distance=400)
            if len(upchirp_start) < prea_len:
                print(f"仅找到 {len(upchirp_start)} 个upchirp峰值，需要 {prea_len} 个")
                # 如果至少找到2个峰值，可以估算间距
                if len(upchirp_start) >= 2:
                    # 使用前两个峰值估算chirp间距
                    chirp_spacing = upchirp_start[1] - upchirp_start[0]
                    print(f"使用估算的chirp间距: {chirp_spacing}")
                    # 基于估算间距生成完整的upchirp_start数组
                    estimated_starts = []
                    for i in range(prea_len):
                        estimated_pos = upchirp_start[0] + i * chirp_spacing
                        if estimated_pos < len(rx_signal) - len(ideal_upchirp):
                            estimated_starts.append(estimated_pos)
                    upchirp_start = np.array(estimated_starts)
                else:
                    return None, None, None

    # 确保至少有足够的峰值
    if len(upchirp_start) < prea_len:
        prea_len = len(upchirp_start)
        print(f"调整前导码长度为: {prea_len}")

    if prea_len < 2:
        return None, None, None

    # 计算所需的信号长度，确保不超出边界
    start_pos = upchirp_start[0]
    end_pos = upchirp_start[prea_len - 1] + len(ideal_upchirp)

    # 边界检查
    if end_pos > len(rx_signal):
        end_pos = len(rx_signal)
        print(f"调整结束位置到信号末尾: {end_pos}")

    if start_pos >= end_pos:
        print("起始位置超过结束位置")
        return None, None, None

    raw_preamble = rx_signal[start_pos:end_pos]

    # 确保preamble长度是chirp长度的整数倍
    expected_len = prea_len * len(ideal_upchirp)
    if len(raw_preamble) < expected_len:
        # 如果长度不足，调整prea_len
        actual_prea_len = len(raw_preamble) // len(ideal_upchirp)
        if actual_prea_len < 2:
            return None, None, None
        raw_preamble = raw_preamble[:actual_prea_len * len(ideal_upchirp)]
        print(f"实际前导码长度调整为: {actual_prea_len} 个chirp")

    return raw_preamble, upchirp_start[0], upchirp_start[1] if len(upchirp_start) > 1 else upchirp_start[0]


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


def extract_phase_noise(rms_signal, upchirp, L):
    '''
    用于提取相位噪声 - 改进版本，能够处理任意长度的信号
    :return:
    '''
    signal_len = len(rms_signal)
    available_chirps = signal_len // L  # 计算实际可用的完整chirp数量

    if available_chirps == 0:
        print(f"警告: 信号长度 {signal_len} 小于单个chirp长度 {L}")
        return None

    # 如果少于8个chirp，我们仍然可以处理
    actual_chirps = min(available_chirps, 8)
    print(f"处理 {actual_chirps} 个chirp (信号长度: {signal_len}, chirp长度: {L})")

    sum_phase = []
    angle_phase = []

    for i in range(actual_chirps):
        # 提取第i个chirp的数据
        chirp_start = i * L
        chirp_end = min((i + 1) * L, signal_len)  # 确保不超出边界
        chirp_data = rms_signal[chirp_start:chirp_end]

        # 如果chirp数据长度不足，用零填充或截断upchirp
        if len(chirp_data) < L:
            # 截断upchirp以匹配数据长度
            upchirp_segment = upchirp[:len(chirp_data)]
        else:
            upchirp_segment = upchirp

        # 计算相关性
        correlation = np.sum(chirp_data[:len(upchirp_segment)] * np.conj(upchirp_segment))
        angle_phase.append(np.angle(correlation))

    # 如果不足8个chirp，我们可以：
    # 1. 用现有数据的平均值填充
    # 2. 用0填充
    # 3. 重复最后一个值
    while len(angle_phase) < 8:
        if len(angle_phase) > 0:
            # 用最后一个值填充
            angle_phase.append(angle_phase[-1])
        else:
            angle_phase.append(0)

    # 归一化处理
    angle_phase = np.array(angle_phase)
    angle_phase_min = np.min(angle_phase)
    angle_phase_max = np.max(angle_phase)

    # 避免除零错误
    if angle_phase_max == angle_phase_min:
        norm_phase_noise = [0.5] * 8  # 如果所有值相同，设为0.5
    else:
        norm_phase_noise = []
        for i in range(8):
            normalized_val = (angle_phase[i] - angle_phase_min) / (angle_phase_max - angle_phase_min)
            norm_phase_noise.append(normalized_val)

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


def process_single_file(file_path, ideal_upchirp, L, fs):
    """处理单个文件，返回相位噪声 - 改进版本"""
    raw_data = read_file(file_path)
    if raw_data is None:
        return None

    coarse_sync_data = coarse_sync(raw_data, L)  # 粗同步
    if coarse_sync_data.size > 0:  # 检查 coarse_sync_data 是否为空
        fine_sync_data, _, _ = fine_sync(ideal_upchirp, coarse_sync_data, fs, 8)  # 精同步
        if fine_sync_data is not None:  # 检查 fine_sync_data 是否为 None
            try:
                delta_f_coarse = coarse_cfo_estimation(fine_sync_data, fs)  # 算粗CFO
                coarse_compensated_preamble = cfo_compensation(fine_sync_data, delta_f_coarse, fs)

                # 检查补偿后的信号长度
                if len(coarse_compensated_preamble) == 0:
                    print(f"CFO补偿后信号为空，文件：{file_path}")
                    return None

                # 精CFO估计需要至少2*L长度的信号
                if len(coarse_compensated_preamble) >= 2 * L:
                    delta_f_fine = fine_cfo_estimation(coarse_compensated_preamble[:2 * L], L, fs)  # 算精CFO
                    fine_compensated_preamble = cfo_compensation(coarse_compensated_preamble, delta_f_fine, fs)
                else:
                    print(f"信号长度不足以进行精CFO估计，跳过精CFO补偿")
                    fine_compensated_preamble = coarse_compensated_preamble

                # RMS归一化
                rms_preamble = rms_normalization(fine_compensated_preamble)

                # 提取相位噪声 - 现在可以处理任意长度的信号
                phase_noise = extract_phase_noise(rms_signal=rms_preamble, upchirp=ideal_upchirp, L=L)

                if phase_noise is None:
                    print(f"相位噪声提取失败，文件：{file_path}")
                    return None

                return phase_noise

            except Exception as e:
                print(f"处理文件时发生错误 {file_path}: {e}")
                return None
        else:
            print(f"精同步失败，文件：{file_path}")
            return None
    else:
        print(f"粗同步失败，文件：{file_path}")
        return None

if __name__ == "__main__":
    # 参数设置
    fs = 1e6
    sf = 7
    bw = 125e3
    ideal_upchirp, L = generate_ideal_upchirp(fs, sf, bw)

    # 路径设置
    base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0511_cubecell"
    dest_base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\temp_phase"

    # 要处理的文件夹列表
    folders = ["27", "29", "32", "35"]

    # 为每个设备处理数据
    for folder in folders:
        print(f"\n开始处理设备 {folder}...")
        data_folder = os.path.join(base_path, folder, "data")

        if not os.path.exists(data_folder):
            print(f"文件夹 {data_folder} 不存在，跳过...")
            continue

        # 创建目标文件夹
        dest_path = os.path.join(dest_base_path, folder)
        os.makedirs(dest_path, exist_ok=True)

        # 获取所有文件并排序
        all_files = sorted([f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))])

        # 存储相位噪声数据
        phase_noise_data = {
            'first_frames': [],  # 奇数帧（第一帧）
            'second_frames': [],  # 偶数帧（第二帧）
            'differences': []  # 差分
        }

        successful_pairs = 0

        # 按对处理文件
        for i in range(0, len(all_files) - 1, 2):
            first_file = all_files[i]  # 奇数文件（第一帧）
            second_file = all_files[i + 1]  # 偶数文件（第二帧）

            first_file_path = os.path.join(data_folder, first_file)
            second_file_path = os.path.join(data_folder, second_file)

            print(f"处理文件对: {first_file} & {second_file}")

            # 处理第一帧
            phase_noise_1 = process_single_file(first_file_path, ideal_upchirp, L, fs)
            # 处理第二帧
            phase_noise_2 = process_single_file(second_file_path, ideal_upchirp, L, fs)

            if phase_noise_1 is not None and phase_noise_2 is not None:
                # 计算差分
                phase_diff = np.array(phase_noise_2) - np.array(phase_noise_1)

                # 保存数据
                phase_noise_data['first_frames'].append(phase_noise_1)
                phase_noise_data['second_frames'].append(phase_noise_2)
                phase_noise_data['differences'].append(phase_diff.tolist())

                successful_pairs += 1
            else:
                print(f"文件对处理失败: {first_file} & {second_file}")

        print(f"设备 {folder} 处理完成，成功处理 {successful_pairs} 对文件")

        # 保存相位噪声数据到文件
        #save_file = os.path.join(dest_path, f"phase_noise_data_{folder}.pkl")
        #with open(save_file, 'wb') as f:
            #pickle.dump(phase_noise_data, f)
        #print(f"相位噪声数据已保存到: {save_file}")

        # 可视化
        if successful_pairs > 0:
            visualize_phase_noise(phase_noise_data, folder)
        else:
            print(f"设备 {folder} 没有成功处理的数据，跳过可视化")

    print("\n所有设备处理完成!")

'''
已知C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0511_cubecell\27\data下保存了连续双帧发射的数据，如1和2是一对连续双帧，3和4是另一对连续双帧。
将程序进行修改，使得我可以将每一帧的8个相位噪声保存下来；然后计算前后两帧相位噪声的差分。
现在对于一个设备（如27），其有1000多个信号文件，每个文件能计算8个相位噪声，奇数文件是第一帧，偶数文件是第二帧，
给出plt可视化（可以参考附件图片的样子但不是一定要长这样）：第一帧（奇数帧）按chirp索引（07）分布的相位噪声的分布情况统计；
第2帧（偶数帧）按chirp索引（07）分布的相位噪声的分布情况统计；同一对前后两帧的差分相位噪声按chirp索引（0~7）分布的情况统计
'''