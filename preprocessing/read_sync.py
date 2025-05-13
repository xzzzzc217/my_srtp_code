import matplotlib.pyplot as plt
import numpy as np
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
if __name__ =="__main__":
    file_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\preamble_data\de1\1"
    signal = read_file(file_path)
    print(signal)
    plt.specgram(signal,Fs = 1e6)
    plt.show()