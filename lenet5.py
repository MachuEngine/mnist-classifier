import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import optuna

# ========================
# 1. 데이터 로드 및 전처리 (수정됨)
# ========================
def load_data(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ========================
# 2. 모델 정의 (수정됨)
# ========================
class LeNet5(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========================
# 3. 학습 함수 (수정됨)
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
# 4. 평가 함수 (수정됨)
# ========================
def evaluate_model_accuracy(model, test_loader):
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
    return accuracy

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

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[idx].item()}\nPred: {preds[idx].item()}')
        ax.axis('off')
    plt.show()

# ========================
# 6. Optuna를 사용한 하이퍼파라미터 튜닝
# ========================
def objective(trial):
    # 하이퍼파라미터 제안
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    num_epochs = trial.suggest_int('num_epochs', 5, 20)
    
    # 데이터 로드
    train_loader, test_loader = load_data(batch_size)
    
    # 모델 초기화
    model = LeNet5(dropout_rate=dropout_rate)
    
    # 손실 함수 및 최적화 알고리즘
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # 모델 평가
    accuracy = evaluate_model_accuracy(model, test_loader)
    
    return accuracy

# ========================
# 7. Optuna 스터디 생성 및 최적화 실행
# ========================
if __name__ == "__main__":
    # Optuna 스터디 생성 (목표는 최대화, 즉 정확도 최대화)
    study = optuna.create_study(direction='maximize')
    
    # 최적화 실행 (예: 20번의 시도)
    study.optimize(objective, n_trials=20)
    # objective 함수에서 각 trial마다 모델은 초기화된다. 
    # 하이퍼파라미터 튜닝이 완료된 후, 최적의 하이퍼파라미터를 사용하여 최종 모델을 다시 학습할 때도 모델은 새로 초기화된다.
    
    # 최적의 하이퍼파라미터 출력
    print("Best trial:")
    trial = study.best_trial

    print(f"  Accuracy: {trial.value:.2f}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 최적의 하이퍼파라미터로 다시 모델 학습 및 평가
    best_params = trial.params
    train_loader, test_loader = load_data(best_params['batch_size'])
    best_model = LeNet5(dropout_rate=best_params['dropout_rate'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    
    # 최적의 에포크 수로 학습
    train_model(best_model, train_loader, criterion, optimizer, best_params['num_epochs'])
    
    # 최종 평가
    final_accuracy = evaluate_model_accuracy(best_model, test_loader)
    print(f"Final Accuracy with Best Params: {final_accuracy:.2f}%")
    
    # 출력 예시
    # Best trial:
    # Accuracy: 98.57%
    # Params: 
    # batch_size: 64
    # dropout_rate: 0.2
    # learning_rate: 0.001
    # num_epochs: 20
    # Final Accuracy with Best Params: 98.43%
    
    # 예측 시각화
    visualize_predictions(best_model, test_loader)
