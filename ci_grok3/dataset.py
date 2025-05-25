import torch
from torch.utils.data import Dataset
import os

class TensorDataset(Dataset):
    """自定义张量数据集类"""
    def __init__(self, tensor_dir, label):
        """初始化数据集
        Args:
            tensor_dir (str): 张量文件目录
            label (int): 设备标签（0到17）
        """
        self.tensor_paths = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith('.pth')]
        self.label = label

    def __len__(self):
        """返回数据集大小"""
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        """获取数据项"""
        tensor = torch.load(self.tensor_paths[idx])
        return tensor.unsqueeze(0), self.label  # 添加通道维度