import struct
import os
# 文件路径
frame_info_filepath = r"C:\Users\21398\Desktop\sophomore\SRTP\data\original_frame_data\processed_dataset\processed_dataset\ebyte_device_1\frame_info"  # 替换为你的文件路径
for file in os.listdir(frame_info_filepath):
    # 读取二进制文件
    path = os.path.join(frame_info_filepath,file)
    with open(path, "rb") as f:
        data = f.read(3 * 8)  # 读取 3 个 double，每个 double 8 字节

    # 解析二进制数据
    quarter_index, temp_index, m_cfo = struct.unpack("ddd", data)

    # 输出解析后的数据
    print(f"file:{file}")
    print(f"quarter_index: {quarter_index}")
    print(f"temp_index: {temp_index}")
    print(f"cfo: {m_cfo}")

