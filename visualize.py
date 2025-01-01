import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 학습 없이 모델 불러와서 평가 및 예측 시각화 진행 코드

# ========================
# 1. LeNet5 모델 정의
# ========================
class LeNet5(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=6
        )
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=16
        )
        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.fc1 = nn.Linear(
            in_features=120,
            out_features=84
        )
        self.dropout = nn.Dropout(
            p=dropout_rate
        )
        self.fc2 = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))  # 입력 이미지를 32x32로 패딩
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 플래튼
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========================
# 2. 데이터 로드 함수
# ========================
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    return train_dataset, test_dataset

# ========================
# 3. 모델 불러오기 함수
# ========================
def load_model(model_path, dropout_rate=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 전환
    return model

# ========================
# 4. 모델 평가 함수
# ========================
def evaluate_model_accuracy(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# ========================
# 5. 예측 시각화 함수
# ========================
def visualize_predictions(model, test_loader, num_images=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    for idx in range(num_images):
        # 원본 이미지
        ax = axes[0, idx]
        img = images[idx].cpu().squeeze()
        img = img * 0.5 + 0.5  # 정규화 해제
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels[idx].item()}')
        ax.axis('off')
        
        # 예측된 이미지
        ax = axes[1, idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {preds[idx].item()}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/predictions.png')  # 파일로 저장
    plt.close(fig)  # 창을 닫아 메모리 해제
    print("예측 시각화가 'outputs/predictions.png' 파일로 저장되었습니다.")

# ========================
# 6. 메인 함수
# ========================
def main():
    # 디렉토리 생성
    os.makedirs('outputs', exist_ok=True)

    # 모델 경로
    model_path = 'outputs/best_lenet5_mnist.pth'

    # 모델 불러오기
    print("모델 불러오는 중...")
    if not os.path.exists(model_path):
        print(f"모델 파일 '{model_path}'이 존재하지 않습니다.")
        return
    model = load_model(model_path)
    print("모델 불러오기 완료.")

    # 데이터 로드
    print("데이터 로드 중...")
    _, test_dataset = load_data(batch_size=256)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )
    print("데이터 로드 완료.")

    # 모델 평가
    print("모델 평가 중...")
    final_accuracy = evaluate_model_accuracy(model, test_loader)
    print(f"최종 테스트 정확도: {final_accuracy:.2f}%")

    # 예측 시각화
    print("예측 시각화 시작...")
    visualize_predictions(model, test_loader, num_images=5)

    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
