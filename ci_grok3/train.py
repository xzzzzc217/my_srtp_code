import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TensorDataset as CustomTensorDataset  # 重命名避免冲突
from model import ConvNet
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 训练参数
batch_size = 32
lr = 3e-4
epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = r"C:\Users\21398\Desktop\sophomore\SRTP\data\ci_tensor_0309\train"

def train_model():
    """训练模型"""
    # 加载训练数据
    train_datasets = [CustomTensorDataset(os.path.join(data_path, f"device_{i}"), i-1) for i in range(1, 19)]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in train_loader:
            print("inputs.shape: ", inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, prob = model(inputs.float())
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            pred = torch.argmax(prob, dim=1)
            correct += (pred == labels).sum().item()
        scheduler.step()
        avg_loss = total_loss / len(train_dataset)
        accuracy = correct / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), r"C:\Users\21398\Desktop\sophomore\SRTP\data\rff_cnn.pth")
    print("模型训练完成并保存")

if __name__ == "__main__":
    train_model()