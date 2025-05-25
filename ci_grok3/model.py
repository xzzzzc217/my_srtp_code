import torch.nn as nn

class ConvNet(nn.Module):
    """卷积神经网络模型"""
    def __init__(self, num_classes=18):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 输入通道为1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 512),  # 调整输入维度为32 * 6 * 7
            nn.LeakyReLU()
        )
        self.layer5 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """前向传播"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print("x.shape: ", x.shape)
        x = self.layer4(x)
        logits = self.layer5(x)
        prob = self.softmax(logits)
        return logits, prob