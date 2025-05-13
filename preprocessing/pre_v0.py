import numpy as np

# 原始复数数组（Python列表格式）
complex_list = [
    0.97949018+0.27053108j,
    1.01305469+0.10127635j,
    1.01814934-0.09668441j,
    0.52428447-0.83867444j,
    0.64483333-0.74841282j,
    0.70583810-0.69223496j
]

# 转换为numpy数组并指定数据类型为complex64
complex_array = np.array(complex_list, dtype=np.complex64)

# 保存到文件（二进制格式）
complex_array.tofile("C:/Users/21398/Desktop/sophomore/SRTP/test/1")

def read_file(file_path):
    try:
        signal = np.fromfile(file_path, dtype=np.complex64)
        return signal
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
        return None

# 读取文件
file_path = "signal.bin"
signal = read_file(file_path)

# 验证数据
if signal is not None:
    print("读取成功，复数为:")
    print(signal)