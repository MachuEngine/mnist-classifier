import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================
# 2. 모델 정의
# ========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Fully Connected Layers
        self.fc1 = None  # Input size will be dynamically calculated
        self.fc2 = nn.Linear(128, 10)  # Output layer: 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # Apply Conv2D and MaxPooling
        x = x.view(x.size(0), -1)  # Flatten tensor to 2D

        # Dynamically initialize fc1 based on input size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x