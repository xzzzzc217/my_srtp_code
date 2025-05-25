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


def visualize_phase_noise(phase_noise_data, device_name):
    """可视化相位噪声分布"""
    # 提取数据
    first_frame_data = []
    second_frame_data = []
    diff_data = []

    for pair_idx in range(len(phase_noise_data['first_frames'])):
        if pair_idx < len(phase_noise_data['first_frames']) and pair_idx < len(phase_noise_data['second_frames']):
            first_frame = phase_noise_data['first_frames'][pair_idx]
            second_frame = phase_noise_data['second_frames'][pair_idx]
            diff = phase_noise_data['differences'][pair_idx]

            first_frame_data.append(first_frame)
            second_frame_data.append(second_frame)
            diff_data.append(diff)

    # 转换为numpy数组
    first_frame_data = np.array(first_frame_data)
    second_frame_data = np.array(second_frame_data)
    diff_data = np.array(diff_data)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制第一帧相位噪声分布
    for chirp_idx in range(8):
        if first_frame_data.size > 0:
            y_values = first_frame_data[:, chirp_idx]
            x_values = [chirp_idx] * len(y_values)
            axes[0].scatter(x_values, y_values, alpha=0.6, s=10, c=y_values, cmap='viridis')

    axes[0].set_xlabel('前导符号索引')
    axes[0].set_ylabel('归一化相位噪声')
    axes[0].set_title('(a) 第一帧')
    axes[0].set_xticks(range(8))
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # 绘制第二帧相位噪声分布
    for chirp_idx in range(8):
        if second_frame_data.size > 0:
            y_values = second_frame_data[:, chirp_idx]
            x_values = [chirp_idx] * len(y_values)
            axes[1].scatter(x_values, y_values, alpha=0.6, s=10, c=y_values, cmap='viridis')

    axes[1].set_xlabel('前导符号索引')
    axes[1].set_ylabel('归一化相位噪声')
    axes[1].set_title('(b) 第二帧')
    axes[1].set_xticks(range(8))
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # 绘制差分相位噪声分布
    for chirp_idx in range(8):
        if diff_data.size > 0:
            y_values = diff_data[:, chirp_idx]
            x_values = [chirp_idx] * len(y_values)
            axes[2].scatter(x_values, y_values, alpha=0.6, s=10, c=y_values, cmap='viridis')

    axes[2].set_xlabel('前导符号索引')
    axes[2].set_ylabel('相位噪声差分')
    axes[2].set_title('(c) 差分')
    axes[2].set_xticks(range(8))
    axes[2].grid(True, alpha=0.3)

    # 整体标题
    fig.suptitle(f'连续双帧的相位噪声及其差分特征 (设备 {device_name}, {len(first_frame_data)} 对连续双帧)',
                 fontsize=14, y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    # 打印统计信息
    print(f"\n设备 {device_name} 统计信息:")
    print(f"成功处理的帧对数: {len(first_frame_data)}")
    if len(first_frame_data) > 0:
        print(f"第一帧相位噪声范围: [{np.min(first_frame_data):.3f}, {np.max(first_frame_data):.3f}]")
        print(f"第二帧相位噪声范围: [{np.min(second_frame_data):.3f}, {np.max(second_frame_data):.3f}]")
        print(f"差分相位噪声范围: [{np.min(diff_data):.3f}, {np.max(diff_data):.3f}]")


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
        save_file = os.path.join(dest_path, f"phase_noise_data_{folder}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(phase_noise_data, f)
        print(f"相位噪声数据已保存到: {save_file}")

        # 可视化
        if successful_pairs > 0:
            visualize_phase_noise(phase_noise_data, folder)
        else:
            print(f"设备 {folder} 没有成功处理的数据，跳过可视化")

    print("\n所有设备处理完成!")
    '''


## 问题根源分析

长度不足的问题主要出现在几个地方：

1. **精同步阶段**：`fine_sync`函数期望找到8个upchirp峰值，但可能由于信号质量问题找不到足够的峰值
2. **信号截取**：按照`upchirp_start[prea_len - 1] + len(ideal_upchirp)`计算的长度可能超出实际信号长度
3. **CFO补偿**：补偿过程中可能改变信号长度


## 改进后的算法解决了以下问题：

### 1. **精同步算法改进** (`fine_sync`函数)：
- **自适应阈值**：如果找不到足够的峰值，会自动降低检测阈值（0.8→0.6→0.4）
- **智能估算**：当找到的峰值不足时，基于前两个峰值估算chirp间距，生成完整的峰值位置
- **动态调整**：根据实际找到的峰值数量调整前导码长度
- **边界保护**：确保所有计算的位置都在信号范围内

### 2. **相位噪声提取算法改进** (`extract_phase_noise`函数)：
- **灵活处理**：能够处理任意数量的chirp（不强制要求8个）
- **智能填充**：当chirp不足8个时，用合理的策略填充（重复最后一个值）
- **数据对齐**：当chirp数据长度不匹配时，智能截断或填充
- **更好的相关计算**：使用numpy的向量化操作提高效率

### 3. **信号处理流程改进** (`process_single_file`函数)：
- **渐进式处理**：不再强制要求8*L长度，能处理较短信号
- **异常处理**：添加try-catch块，避免程序崩溃
- **条件CFO补偿**：当信号不足时跳过精CFO补偿
- **智能长度管理**：根据实际信号长度调整处理策略

### 4. **算法优势**：
- **鲁棒性更强**：能够处理各种信号质量和长度的情况
- **容错能力**：即使部分步骤失败，仍能提取有用信息
- **自适应性**：根据实际数据情况调整处理参数
- **信息保留**：尽可能多地保留和利用可用的信号信息

这些改进让算法能够处理实际环境中的各种信号条件，而不是严格要求完美的8个chirp信号。现在程序应该能够处理更多的文件，提高成功率。
    '''