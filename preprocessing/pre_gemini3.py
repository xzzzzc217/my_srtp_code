import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import numpy as np
from scipy import stats  # 用于偏度、峰度和IQR
import pandas as pd  # 用于数据处理和CSV输出
import os  # 已在你的代码中
import pickle  # 已在你的代码中
import matplotlib.pyplot as plt  # 已在你的代码中
import seaborn as sns  # 用于更美观的可视化

# 设置中文字体 - 这部分已在你的代码中，确保它在绘图前执行
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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


def extract_advanced_features(phase_noise_1_list, phase_noise_2_list):
    """
    从一对相位噪声序列中提取高级统计特征。

    参数:
        phase_noise_1_list (list or np.array): 第一个帧的8个相位噪声值。
        phase_noise_2_list (list or np.array): 第二个帧的8个相位噪声值。

    返回:
        dict: 包含提取特征的字典。
    """
    features = {}
    pn1_arr = np.array(phase_noise_1_list)
    pn2_arr = np.array(phase_noise_2_list)
    phase_diff_arr = pn2_arr - pn1_arr

    # --- 特征来自 phase_diff_arr ---
    features['diff_mean'] = np.mean(phase_diff_arr)
    features['diff_var'] = np.var(phase_diff_arr)
    features['diff_std'] = np.std(phase_diff_arr)
    # 检查标准差是否为零，以避免 stats.skew 和 stats.kurtosis 的警告或错误
    if np.std(phase_diff_arr) == 0:
        features['diff_skew'] = 0.0
        features['diff_kurtosis'] = -3.0  # 正态分布的峰度（Fisher定义下）为0，这里是当所有值相同时的峰度
    else:
        features['diff_skew'] = stats.skew(phase_diff_arr)
        features['diff_kurtosis'] = stats.kurtosis(phase_diff_arr)  # Fisher 定义 (正态分布峰度为0)
    features['diff_mean_abs'] = np.mean(np.abs(phase_diff_arr))
    features['diff_median'] = np.median(phase_diff_arr)
    features['diff_range'] = np.ptp(phase_diff_arr)  # Peak-to-Peak (max - min)
    features['diff_iqr'] = stats.iqr(phase_diff_arr)
    # 差分序列的线性趋势 (斜率)
    x_indices = np.arange(len(phase_diff_arr))
    if len(x_indices) > 1 and np.any(phase_diff_arr != phase_diff_arr[0]):  # 确保至少有两个点且值不完全相同
        features['diff_slope'] = np.polyfit(x_indices, phase_diff_arr, 1)[0]
    else:
        features['diff_slope'] = 0.0

    # --- 可选: 特征来自 phase_noise_1_arr ---
    features['pn1_mean'] = np.mean(pn1_arr)
    features['pn1_var'] = np.var(pn1_arr)
    features['pn1_std'] = np.std(pn1_arr)
    if np.std(pn1_arr) == 0:
        features['pn1_skew'] = 0.0
        features['pn1_kurtosis'] = -3.0
    else:
        features['pn1_skew'] = stats.skew(pn1_arr)
        features['pn1_kurtosis'] = stats.kurtosis(pn1_arr)

    # --- 可选: 特征来自 phase_noise_2_arr ---
    features['pn2_mean'] = np.mean(pn2_arr)
    features['pn2_var'] = np.var(pn2_arr)
    features['pn2_std'] = np.std(pn2_arr)
    if np.std(pn2_arr) == 0:
        features['pn2_skew'] = 0.0
        features['pn2_kurtosis'] = -3.0
    else:
        features['pn2_skew'] = stats.skew(pn2_arr)
        features['pn2_kurtosis'] = stats.kurtosis(pn2_arr)

    # --- 帧间关系特征 ---
    # 避免在标准差为0时计算相关系数（会导致NaN或错误）
    if np.std(pn1_arr) > 0 and np.std(pn2_arr) > 0:
        features['pn1_pn2_corr'] = np.corrcoef(pn1_arr, pn2_arr)[0, 1]
    else:
        features['pn1_pn2_corr'] = 0.0  # 或者设为 np.nan，取决于后续处理

    # RMSE 实际上是 np.sqrt(np.mean(phase_diff_arr**2))
    features['pn1_pn2_rmse'] = np.sqrt(np.mean(phase_diff_arr ** 2))

    return features

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

def extract_single_frame_features(pn):
    features = {}
    pn_arr = np.array(pn)

    features['mean'] = np.mean(pn_arr)
    features['var'] = np.var(pn_arr)
    features['std'] = np.std(pn_arr)
    features['median'] = np.median(pn_arr)
    features['range'] = np.ptp(pn_arr)
    features['iqr'] = stats.iqr(pn_arr)
    if np.std(pn_arr) == 0:
        features['skew'] = 0.0
        features['kurtosis'] = -3.0
    else:
        features['skew'] = stats.skew(pn_arr)
        features['kurtosis'] = stats.kurtosis(pn_arr)

    return features

def visualize_selected_advanced_features(all_devices_data_dict, features_to_plot, devices_to_compare):
    """
    可视化选定的高级特征，以比较它们在不同设备间的分布。

    参数:
        all_devices_data_dict (dict): 字典，键是设备名称，值是包含 'advanced_features_list' 的 device_specific_data 字典。
        features_to_plot (list): 要绘制的特征名称列表 (字符串)。
        devices_to_compare (list): 要在图表中比较的设备名称列表 (字符串)。
    """
    if not features_to_plot:
        print("未指定要绘制的特征。")
        return

    plot_data_frames = []
    for device_name in devices_to_compare:
        if device_name in all_devices_data_dict and all_devices_data_dict[device_name]['advanced_features_list']:
            df_device = pd.DataFrame(all_devices_data_dict[device_name]['advanced_features_list'])
            df_device['device'] = device_name  # 添加一列用于区分设备
            plot_data_frames.append(df_device)
        else:
            print(f"警告: 设备 {device_name} 的数据不足或未在 all_devices_data_dict 中找到。")

    if not plot_data_frames:
        print("没有足够的数据进行可视化。")
        return

    combined_df = pd.concat(plot_data_frames, ignore_index=True)

    for feature_name in features_to_plot:
        if feature_name not in combined_df.columns:
            print(f"警告: 特征 '{feature_name}' 未在数据中找到，跳过绘图。")
            continue

        plt.figure(figsize=(10, 6))
        # 使用箱形图
        sns.boxplot(x='device', y=feature_name, data=combined_df, order=devices_to_compare)
        # 或者使用小提琴图，可以显示更详细的分布形状
        sns.violinplot(x='device', y=feature_name, data=combined_df, order=devices_to_compare)
        # 可以叠加散点图显示原始数据点
        sns.stripplot(x='device', y=feature_name, data=combined_df, order=devices_to_compare, color='black', alpha=0.3,
                      jitter=True, dodge=False)

        plt.title(f'高级特征 "{feature_name}" 在不同设备间的分布')
        plt.xlabel('设备编号')
        plt.ylabel(f'特征值: {feature_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


# --- 如何调用新的可视化函数 ---
# 假设 all_devices_feature_data 已经填充完毕 (如上面的 main 流程所示)
# selected_features_for_vis = ['diff_mean', 'diff_var', 'diff_skew', 'diff_kurtosis', 'diff_slope', 'pn1_pn2_corr', 'pn1_mean', 'pn1_var']
# devices_to_show = ["27", "29", "32", "35"] # 你关心的设备

# print("\n开始可视化高级特征...")
# visualize_selected_advanced_features(all_devices_feature_data,
#                                      selected_features_for_vis,
#                                      devices_to_show)
if __name__ == "__main__":

    # 参数设置
    fs = 1e6
    sf = 7
    bw = 125e3
    ideal_upchirp, L = generate_ideal_upchirp(fs, sf, bw)
    base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\dataset_0511_cubecell"  #
    dest_base_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\temp_phase_advanced"  # 用新目录保存高级特征
    os.makedirs(dest_base_path, exist_ok=True)

    folders = ["27", "29", "32", "35"]  #

    all_devices_feature_data = {}  # 用于存储所有设备的高级特征以供后续可视化或分类

    for folder in folders:
        print(f"\n开始处理设备 {folder} (高级特征)...")
        data_folder = os.path.join(base_path, folder, "data")  #

        if not os.path.exists(data_folder):
            print(f"文件夹 {data_folder} 不存在，跳过...")  #
            continue

        dest_path = os.path.join(dest_base_path, folder)  #
        os.makedirs(dest_path, exist_ok=True)  #

        all_files = sorted([f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))])  #

        # 存储每个设备的原始相位噪声和提取的高级特征
        device_specific_data = {
            'raw_first_frames': [],
            'raw_second_frames': [],
            'raw_differences': [],
            'advanced_features_list': []  # 存储每个成功帧对的特征字典
        }
        successful_pairs_count = 0

        for i in range(0, len(all_files) - 1, 2):  #
            first_file = all_files[i]  #
            second_file = all_files[i + 1]  #
            first_file_path = os.path.join(data_folder, first_file)  #
            second_file_path = os.path.join(data_folder, second_file)  #

            print(f"处理文件对: {first_file} & {second_file}")  #

            phase_noise_1 = process_single_file(first_file_path, ideal_upchirp, L, fs)  #
            phase_noise_2 = process_single_file(second_file_path, ideal_upchirp, L, fs)  #

            if phase_noise_1 is not None and phase_noise_2 is not None:
                # 原始8点相位噪声数据
                pn1_arr = np.array(phase_noise_1)
                pn2_arr = np.array(phase_noise_2)
                diff_arr = pn2_arr - pn1_arr

                device_specific_data['raw_first_frames'].append(pn1_arr.tolist())
                device_specific_data['raw_second_frames'].append(pn2_arr.tolist())
                device_specific_data['raw_differences'].append(diff_arr.tolist())

                # 提取高级特征
                #current_adv_features = extract_advanced_features(pn1_arr,pn2_arr)
                #device_specific_data['advanced_features_list'].append(current_adv_features)
                '''
                替换为第一针或第二帧
                '''
                current_adv_features = extract_single_frame_features(pn1_arr)  # 或 pn2_arr
                device_specific_data['advanced_features_list'].append(current_adv_features)

                successful_pairs_count += 1
            else:
                print(f"文件对处理失败: {first_file} & {second_file}")  #

        all_devices_feature_data[folder] = device_specific_data  # 存储当前设备的数据

        print(f"设备 {folder} 处理完成，成功处理 {successful_pairs_count} 对文件的高级特征")  #

        # 输出和保存特征
        #if device_specific_data['advanced_features_list']:
            # 1. 保存为 Pickle 文件 (包含原始数据和高级特征)
            #adv_save_file_pkl = os.path.join(dest_path, f"device_{folder}_data_with_adv_features.pkl")
            #with open(adv_save_file_pkl, 'wb') as f:
                #pickle.dump(device_specific_data, f)
            #print(f"设备 {folder} 的完整数据（含高级特征）已保存到: {adv_save_file_pkl}")

            # 2. 将高级特征保存为 CSV 文件，便于查看或用于其他工具
            #df_adv_features = pd.DataFrame(device_specific_data['advanced_features_list'])
            #adv_csv_file = os.path.join(dest_path, f"device_{folder}_advanced_features.csv")
            #df_adv_features.to_csv(adv_csv_file, index=False)
            #print(f"设备 {folder} 的高级特征已保存到 CSV: {adv_csv_file}")

        # 注意: 原始的 visualize_phase_noise 仍然可以用于可视化原始8点差分分布
        # 如果需要，可以加载 device_specific_data['raw_...'] 进行可视化
        # 例如，构建一个临时的 phase_noise_data 结构给旧的 visualize_phase_noise 函数


    print("\n所有设备高级特征处理完成!")
    #selected_features_for_vis = ['diff_mean', 'diff_var', 'diff_skew', 'diff_kurtosis', 'diff_slope', 'pn1_pn2_corr', 'pn1_mean', 'pn1_var'] # 是用于双帧
    '''
    替换为第一针或第二帧
    '''
    selected_features_for_vis = ['mean', 'std', 'skew', 'kurtosis', 'iqr']#适用于单帧

    devices_to_show = ["27", "29", "32", "35"] # 你关心的设备

    print("\n开始可视化高级特征...")
    visualize_selected_advanced_features(all_devices_feature_data,
                                          selected_features_for_vis,
                                          devices_to_show)

'''
提取新特征（方差/）
'''
