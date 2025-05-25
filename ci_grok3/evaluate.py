import torch
from torch.utils.data import DataLoader
from dataset import TensorDataset
from model import ConvNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(data_path, model_path, dataset_name):
    """评估模型性能"""
    # 加载测试数据
    test_datasets = [TensorDataset(os.path.join(data_path, f"device_{i}"), i-1) for i in range(1, 19)]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 测试
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, prob = model(inputs.float())
            preds = torch.argmax(prob, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算成功率
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"{dataset_name} overall accuracy: {accuracy:.4f}")

    # 绘制混淆矩阵
    labels = [str(i) for i in range(1, 19)]
    cm = confusion_matrix(all_labels, all_preds)
    plt.matshow(cm, cmap=plt.cm.Blues)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[i, j], xy=(j, i), ha='center', va='center')
    plt.xlabel("predicted label",fontname='Times New Roman')
    plt.ylabel("true label",fontname='Times New Roman')
    plt.xticks(range(18), labels,fontname='Times New Roman')
    plt.yticks(range(18), labels,fontname='Times New Roman')
    plt.title(f"{dataset_name} confusion matrix acc: {accuracy:.4f}",fontname='Times New Roman')
    plt.savefig(os.path.join(os.path.dirname(model_path), f"cm_{dataset_name}.png"))
    plt.show()

if __name__ == "__main__":
    # 评估3月9日测试集
    evaluate_model(r"C:\Users\21398\Desktop\sophomore\SRTP\data\ci_tensor_0309\test",
                   r"C:\Users\21398\Desktop\sophomore\SRTP\data\rff_cnn.pth", "0309_test")
    # 评估3月16日数据
    evaluate_model(r"C:\Users\21398\Desktop\sophomore\SRTP\data\ci_tensor_0316\test",
                   r"C:\Users\21398\Desktop\sophomore\SRTP\data\rff_cnn.pth", "0316_test")