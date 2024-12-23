import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ========================
# 1. 데이터 로드 및 전처리
# ========================
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 평균 0.5, 표준편차 0.5로 정규화
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

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

# ========================
# 3. 학습 함수
# ========================
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ========================
# 4. 평가 함수
# ========================
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

# ========================
# 5. 예측 시각화
# ========================
def visualize_predictions(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:5].to(device), labels[:5].to(device)

    # 모델 예측
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # 시각화
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[idx].item()}\nPred: {preds[idx].item()}')
        ax.axis('off')
    plt.show()

# ========================
# 6. 메인 실행
# ========================
if __name__ == "__main__":
    # 하이퍼파라미터
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001

    # 데이터 로드
    train_loader, test_loader = load_data(batch_size)

    # 모델 초기화
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 모델 평가
    evaluate_model(model, test_loader)

    # 예측 시각화
    visualize_predictions(model, test_loader)
