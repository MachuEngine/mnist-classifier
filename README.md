# MNIST 분류 프로젝트
### MNIST 데이터셋을 이용해 분류기 모델을 구현, 학습 내용 정리

- MNIST 데이터셋을 사용하여 손글씨 숫자(0~9)를 분류하는 합성곱 신경망(CNN) 모델 및 LeNet5 아키텍처를 구현
- 교차 검증 및 하이퍼파라미터 최적화를 통해 결과를 비교 분석

### 프레임워크
Pytorch

### 개요
MNIST 데이터셋은 28x28 크기의 흑백 손글씨 이미지. 총 60,000개의 학습 데이터와 10,000개의 테스트 데이터를 포함.

진행 내용: 
- CNN 모델을 설계/ 학습
- LeNet5 모델을 설계/ 학습
- 교차 검증(K-Fold)을 통해 모델의 일반화 성능을 평가
- Optuna를 사용하여 하이퍼파라미터 최적화를 수행
- 학습률 스케줄러를 도입하여 학습의 안정성과 효율성을 향상
- 학습된 모델을 테스트 데이터로 평가
- 예측 결과 및 평가 지표를 시각화
- CNN과 LeNet5 모델 간 결과를 비교

---

## 모델 비교 분석
### CNN 모델
- 세 개의 합성곱 층과 풀링 층
- 드롭아웃(Dropout)과 배치 정규화(Batch Normalization) 적용
### LeNet5 모델
- 전통적인 LeNet-5 아키텍처 구현
- 평균 풀링 사용 및 드롭아웃 레이어 추가

---

## 프로젝트에서 사용된 특징
- 데이터 증강(Data Augmentation): 다양한 데이터 증강 기법(랜덤 회전, 크기 조정, 가우시안 노이즈 추가 등)을 통해 데이터의 다양성을 확보하고 모델의 일반화 성능을 향상시킵니다.
  
- 명시적인 매개변수 지정: 모델 레이어 정의 시 매개변수를 명시적으로 지정하여 코드의 가독성과 유지보수성을 향상시켰습니다.
  
- 교차 검증(K-Fold Cross-Validation): K-Fold 교차 검증을 통해 모델의 일반화 성능을 보다 정확하게 평가합니다.
  
- 학습률 스케줄러(Learning Rate Scheduler): ReduceLROnPlateau 스케줄러를 도입하여 검증 손실이 개선되지 않을 때 학습률을 감소시킵니다.
  
- 하이퍼파라미터 최적화(Optuna): Optuna를 사용하여 학습률, 배치 크기, 드롭아웃 비율 등 주요 하이퍼파라미터를 최적화합니다.
  
- 모델 저장 및 시각화: 학습된 모델을 저장하고, 학습 과정과 예측 결과를 시각화하여 분석할 수 있습니다.

---
튜토리얼 형식으로 진행한 내용을 정리. 

### 사전 요구 사항
1. Python 3.8 이상
2. 아래의 라이브러리 설치 필요:
   - `torch`
   - `torchvision`
   - `matplotlib`
   - `optuna`
   - `scikit-learn`
   - `tqdm`
   - `plotly`

### 설치 방법
1. **레포지토리 복제**:
  ```bash
  git clone https://github.com/MachuEngine/mnist-classifier.git
  ```
  ```bash
  cd mnist-classification
  ```

2. **가상 환경 생성**:
  ```bash
  python -m venv mnist_env # 원하는 이름으로 변경 가능
  ```
  ```bash
  source mnist_env/bin/activate  # Windows: mnist_env\Scripts\activate
  ```
3. **필요한 라이브러리 설치**:
  ```bash
  pip install -r requirements.txt
  ```

---

## 프로젝트 구조
```bash
mnist-classification/
│
├── data/                  # 데이터셋 (자동 다운로드)
├── outputs/               # 학습 로그, 모델 체크포인트, 시각화 결과
├── mnist.py               # 학습 및 평가를 위한 메인 스크립트
├── requirements.txt       # Python 의존성 목록
└── README.md              # 프로젝트 설명 파일
```

---

### 사용방법
### 모델 학습 및 하이퍼파라미터 튜닝
Optuna를 사용하여 하이퍼파라미터 튜닝과 교차 검증을 수행하며 모델을 학습하려면 다음 명령어를 실행하세요:
```bash
python mnist.py # Custom CNN 모델 사용
```
```bash
python lenet5.py # LeNet5 모델 사용
```
### 결과 시각화
스크립트 실행 후 outputs 폴더에 다음과 같은 파일들이 저장됩니다:

- [학습 손실 및 정확도 그래프](loss_accuracy_curve.png)
- [예측 결과 이미지](predictions.png)
- [혼동 행렬(Confusion Matrix)](confusion_matrix.png)
- [Optuna 최적화 히스토리](optimization_history.png)
- [Optuna 파라미터 중요도](param_importances.png)
- [학습된 모델 가중치](best_lenet5_mnist.pth)

### 모델 체크포인트
학습된 모델은 outputs/best_lenet5_mnist.pth 파일로 저장됩니다. 이를 로드하여 평가나 추가 학습에 사용할 수 있습니다.

---

## 추가 개발 방향
- 더 깊은 신경망 구조 실험
-> 더 많은 합성곱 층과 완전 연결 층을 추가하여 모델의 성능을 향상시킬 수 있습니다.
- 데이터 증강 기법 확장
-> 추가적인 데이터 증강 기법을 도입하여 모델의 일반화 성능을 더욱 향상시킬 수 있습니다.
- 다른 데이터셋 적용
-> Fashion-MNIST, CIFAR-10 등 다른 데이터셋에 모델을 적용하여 성능을 비교 분석할 수 있습니다.
- 학습률 스케줄러 다양화
-> 다양한 학습률 스케줄러를 시도하여 최적의 학습률 조정 전략을 찾을 수 있습니다.
- 다양한 옵티마이저 실험
-> SGD, RMSprop 등 다른 옵티마이저를 사용하여 모델의 성능을 비교할 수 있습니다.
- 교차 검증 방법 개선
-> Stratified K-Fold 등 다른 교차 검증 방법을 도입하여 클래스 불균형을 고려할 수 있습니다.

### 하이퍼파라미터 튜닝
- 학습률, 배치 크기, 에포크 수 등을 조정하여 모델의 성능을 최적화할 수 있습니다.
**주요 하이퍼파라미터**
*학습률 (Learning Rate)*: 모델의 가중치를 얼마나 크게 업데이트할지를 결정합니다.
*배치 크기 (Batch Size)*: 한 번에 처리하는 데이터 샘플의 수를 의미합니다.
*에포크 수 (Number of Epochs)*: 전체 데이터셋을 몇 번 반복해서 학습할지를 결정합니다.
*최적화 알고리즘 (Optimizer)*: 모델을 학습시키기 위해 사용하는 알고리즘 (예: SGD, Adam).
*드롭아웃 비율 (Dropout Rate)*: 과적합을 방지하기 위해 일부 뉴런을 무작위로 비활성화하는 비율입니다.
*스케줄러 패티언스 (Scheduler Patience)*: 검증 손실이 개선되지 않을 때 학습률을 감소시키기까지 기다리는 에포크 수입니다.
*스케줄러 팩터 (Scheduler Factor)*: 학습률을 감소시킬 때 곱하는 비율입니다.

### 데이터 증강
- MNIST와 같은 간단한 데이터셋에서는 크게 필요하지 않을 수 있지만, 데이터 증강 기법을 적용하여 모델의 일반화 성능을 향상시킬 수 있습니다.

### 모델 구조 개선
- 더 깊은 네트워크, 드롭아웃(Dropout), 배치 정규화(Batch Normalization) 등의 기법을 도입하여 모델의 성능을 개선할 수 있습니다.

### 학습률 스케줄러 도입
- 학습 과정에서 학습률을 조정하는 스케줄러를 사용하여 학습의 안정성과 속도를 향상시킬 수 있습니다.

---

## 결과
- 최종 테스트 정확도: 약 99.28%
- Optuna 하이퍼파라미터 튜닝 결과:
  - Accuracy: 99.28%
  - Params:
    - learning_rate: 0.0029
    - batch_size: 256
    - dropout_rate: 0.235
    - num_epochs: 18
    - scheduler_patience: 3
    - scheduler_factor: 0.1
- 혼동 행렬 분석: 3과 5처럼 모양이 비슷한 숫자 간에 일부 오분류가 발생하지만, 전반적으로 높은 정확도를 보임.

### 결과 요약
##### CNN 모델:
  - 최종 정확도: 98.76%
##### LeNet5 모델:
  - 최종 정확도: 99.00%
##### LeNet5 모델 + 데이터 증강, 하이퍼파라미터 튜닝, 학습률 스케줄러 적용 등 모델 개선 작업 적용
  - 최종 정확도: 99.28%
  
LeNet5 모델은 더 높은 테스트 정확도를 보였으나, 학습 손실은 더 큰 경향을 보였습니다. 
-> LeNet5 모델의 일반화 성능이 더 높음.

---

## 참고 자료
- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [합성곱 신경망(CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Optuna 공식 문서](https://optuna.readthedocs.io/en/stable/)
- [K-Fold 교차 검증]
  
---

## License
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참고하세요.

---

## Author
Machu
