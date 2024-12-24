import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 합성곱 층 (입력: 1x32x32 → 출력: 6x28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # S2: 평균 풀링 층 (출력: 6x14x14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 합성곱 층 (출력: 16x10x10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # S4: 평균 풀링 층 (출력: 16x5x5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 합성곱 층 (출력: 120x1x1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        # F6: 완전 연결층 (출력: 84)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        # Output: 완전 연결층 (출력: 10)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 입력 크기: (batch_size, 1, 28, 28)
        # LeNet-5는 32x32 입력을 기대하지만, MNIST는 28x28이므로 패딩을 추가
        x = F.pad(x, (2, 2, 2, 2))  # 패딩 추가하여 32x32로 변환
        x = F.relu(self.conv1(x))    # C1
        x = self.pool1(x)            # S2
        x = F.relu(self.conv2(x))    # C3
        x = self.pool2(x)            # S4
        x = F.relu(self.conv3(x))    # C5
        x = x.view(x.size(0), -1)    # Flatten
        x = F.relu(self.fc1(x))      # F6
        x = self.fc2(x)              # Output
        return x
