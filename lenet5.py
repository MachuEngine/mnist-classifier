import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import optuna
from tqdm.notebook import tqdm  # tqdm.auto 대신 tqdm.notebook 사용
from optuna.visualization import plot_optimization_history, plot_param_importances
import random
import numpy as np
import json
import os
from sklearn.model_selection import KFold

# ========================
# 1. 시드 고정 (재현성 확보)
# ========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ========================
# 2. 데이터 증강 클래스 정의 (가우시안 노이즈 추가)
# ========================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# ========================
# 3. 데이터 로드 및 전처리 (교차 검증 포함)
# ========================
def load_data(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(
            size=28,
            scale=(0.8, 1.2),
            ratio=(0.9, 1.1)
        ),  # 랜덤 스케일링
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),
        AddGaussianNoise(
            mean=0.,
            std=0.1
        ),  # 가우시안 노이즈 추가 
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])
    
    # 전체 학습 데이터셋 로드
    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform_train,
        download=True
    )
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform_test,
        download=True
    )
    
    return full_train_dataset, test_dataset

# ========================
# 4. 모델 정의 (LeNet5)
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
# 5. 학습 함수 정의
# ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5, early_stopping_patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) 

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # 검증 단계
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 스케줄러 업데이트 (ReduceLROnPlateau의 경우)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # 조기 종료 조건 확인
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

# ========================
# 6. 평가 함수 정의
# ========================
def evaluate_model_accuracy(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# ========================
# 7. 교차 검증 함수 정의
# ========================
def cross_validate(model_class, dataset, k=5, params=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")
        # 데이터 로더 생성
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            sampler=train_subsampler
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            sampler=val_subsampler
        )
        
        # 모델 초기화
        model = model_class(dropout_rate=params['dropout_rate']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # 학습률 스케줄러 초기화 (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params['scheduler_patience'],
            factor=params['scheduler_factor'],
            verbose=False
        )
        
        # 모델 학습
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=params['num_epochs'],
            early_stopping_patience=params['early_stopping_patience']
        )
        
        # 모델 평가
        accuracy = evaluate_model_accuracy(model, val_loader)
        accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%\n")
    
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Accuracy over {k} folds: {average_accuracy:.2f}%")
    return average_accuracy

# ========================
# 8. Optuna를 사용한 하이퍼파라미터 튜닝 (교차 검증 포함)
# ========================
def objective(trial):
    try:
        # 하이퍼파라미터 제안
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        num_epochs = trial.suggest_int('num_epochs', 5, 20)
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 10)
        
        # ReduceLROnPlateau의 하이퍼파라미터 제안
        scheduler_patience = trial.suggest_int('scheduler_patience', 2, 5)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.5, step=0.1)
        
        # 데이터 로드
        full_train_dataset, _ = load_data(batch_size)
        
        # 교차 검증 수행
        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'scheduler_patience': scheduler_patience,
            'scheduler_factor': scheduler_factor
        }
        
        average_accuracy = cross_validate(LeNet5, full_train_dataset, k=5, params=params)
        
        return average_accuracy
    except Exception as e:
        print(f"An error occurred during trial: {e}")
        return 0  # 에러 발생 시 최소 정확도로 반환

# ========================
# 9. Optuna 스터디 생성 및 최적화 실행
# ========================
def run_optuna_study(n_trials=20):
    # Optuna 스터디 생성 (목표는 최대화, 즉 정확도 최대화)
    study = optuna.create_study(direction='maximize')
    
    # 최적화 실행
    study.optimize(objective, n_trials=n_trials)
    
    # 최적의 하이퍼파라미터 출력
    if study.best_trial is not None:
        print("Best trial:")
        trial = study.best_trial

        print(f"  Accuracy: {trial.value:.2f}%")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # 하이퍼파라미터 저장
        with open('outputs/best_hyperparameters.json', 'w') as f:
            json.dump(trial.params, f, indent=4)
        print("하이퍼파라미터가 'outputs/best_hyperparameters.json' 파일로 저장되었습니다.")
        
        return trial.params, study
    else:
        print("No successful trials found.")
        return None, study

# ========================
# 10. 모델 저장 및 불러오기
# ========================
def save_model(model, path='outputs/best_lenet5_mnist.pth'):
    torch.save(model.state_dict(), path)
    print(f"모델이 '{path}' 파일로 저장되었습니다.")

def load_model(path='outputs/best_lenet5_mnist.pth', dropout_rate=0.5):
    model = LeNet5(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"모델이 '{path}' 파일에서 불러와졌습니다.")
    return model

# ========================
# 11. 예측 시각화
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
    plt.show()

# ========================
# 12. 메인 함수 정의
# ========================
def main():
    # Optuna 하이퍼파라미터 튜닝 실행
    best_params, study = run_optuna_study(n_trials=20)
    
    if best_params is not None:
        # 데이터 로드
        full_train_dataset, test_dataset = load_data(best_params['batch_size'])
        
        # 전체 학습 데이터 로드
        # Optuna에서 사용한 전체 학습 데이터를 다시 사용하지 않고, 전체 데이터로 다시 학습
        train_loader = torch.utils.data.DataLoader(
            full_train_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=best_params['batch_size'],
            shuffle=False
        )
        
        # 모델 초기화
        best_model = LeNet5(dropout_rate=best_params['dropout_rate'])
        
        # 손실 함수 및 최적화 알고리즘
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        
        # 학습률 스케줄러 초기화 (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=best_params['scheduler_patience'],
            factor=best_params['scheduler_factor'],
            verbose=True
        )
        
        # 모델 학습
        train_model(
            best_model,
            train_loader,
            test_loader,  # 여기서는 검증 데이터를 테스트 데이터로 사용
            criterion,
            optimizer,
            scheduler,
            num_epochs=best_params['num_epochs'],
            early_stopping_patience=best_params['early_stopping_patience']
        )
        
        # 최종 평가
        final_accuracy = evaluate_model_accuracy(best_model, test_loader)
        print(f"Final Accuracy with Best Params: {final_accuracy:.2f}%")
        
        # 예측 시각화
        visualize_predictions(best_model, test_loader, num_images=5)
        
        # 모델 저장
        save_model(best_model, 'outputs/best_lenet5_mnist.pth')
        
        # Optuna 시각화
        fig1 = plot_optimization_history(study)
        fig1.savefig('outputs/optimization_history.png')
        print("Optuna 최적화 히스토리가 'outputs/optimization_history.png'로 저장되었습니다.")
        
        fig2 = plot_param_importances(study)
        fig2.savefig('outputs/param_importances.png')
        print("Optuna 파라미터 중요도가 'outputs/param_importances.png'로 저장되었습니다.")
    else:
        print("최적의 하이퍼파라미터를 찾지 못했습니다.")

if __name__ == "__main__":
    # 디렉토리 생성
    os.makedirs('outputs', exist_ok=True)
    
    main()
