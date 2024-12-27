# MNIST 분류 프로젝트

이 프로젝트는 [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)을 사용하여 손글씨 숫자(0~9)를 분류하는 **합성곱 신경망(CNN)** 모델 및 LeNet5 아키텍처를 구현하여 결과를 비교합니다.

이 프로젝트는 PyTorch를 기반으로 작성되었습니다.


## 프로젝트 개요
MNIST 데이터셋은 28x28 크기의 흑백 손글씨 이미지로 이루어져 있으며, 총 60,000개의 학습 데이터와 10,000개의 테스트 데이터를 포함합니다. 이 프로젝트에서는 다음을 수행합니다:
- CNN 모델을 설계하고 학습합니다.
- 학습된 모델을 테스트 데이터로 평가합니다.
- 예측 결과 및 평가 지표를 시각화합니다.
- LeNet5 모델을 설계하고 학습합니다.
- 학습된 모델을 테스트 데이터로 평가합니다.
- 예측 결과 및 평가 지표를 시각화합니다.
- CNN <> LeNet5 모델 간 결과를 비교합니다.


## 주요 특징
- 이미지 데이터 전처리 (정규화).
- 간단한 CNN 모델 및 LeNet5를 사용한 손글씨 숫자 분류.
- 학습 및 평가 파이프라인 구현.
- 시각화 및 결과 분석 지원.

---

## 시작하기
### 사전 요구 사항
1. Python 3.8 이상
2. 아래의 라이브러리 설치 필요:
   - `torch`
   - `torchvision`
   - `matplotlib`

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
  python -m venv mnist_env # 이름은 원하는대로
  ```
  ```bash
  source mnist_env/bin/activate  # Windows: mnist_env\Scripts\activate
  ```
3. **필요한 라이브러리 설치**:
  ```bash
  pip install -r requirements.txt
  ```


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


## 사용방법
### 모델 학습
CNN 모델을 학습하려면 다음 명령어를 실행하세요:
```bash
python mnist.py
```
### 결과 시각화
스크립트 실행 후 아래와 같은 파일들이 `outputs` 폴더에 저장됩니다:
- [손실 및 정확도 그래프](outputs/loss_accuracy_curve.png)
- [예측 결과 이미지](outputs/predictions.png)
- [혼동 행렬(Confusion Matrix)](outputs/confusion_matrix.png)

### 모델 체크포인트
학습된 모델은 outputs/final_model.pth로 저장됩니다. 이를 로드하여 평가나 추가 학습에 사용할 수 있습니다.


## 결과
- 최종 테스트 정확도: 약 97%
- fc1 레이어를 정적 초기화하도록 수정 한 뒤 정확도 : 98.71%
- 혼동 행렬 분석: 3과 5처럼 모양이 비슷한 숫자 간에 일부 오분류가 발생.

## 추가 개발 방향
- 더 깊은 신경망 구조를 실험.
- 데이터 증강(Data Augmentation)을 사용하여 일반화 성능 개선.
- Fashion-MNIST 또는 CIFAR-10과 같은 다른 데이터셋에 적용.

## 추가 개발 세부 사항
### 하이퍼파라미터 튜닝: 학습률, 배치 크기, 에포크 수 등을 조정하여 모델의 성능을 최적화할 수 있습니다.
**주요 하이퍼파라미터**
*학습률 (Learning Rate)*: 모델의 가중치를 얼마나 크게 업데이트할지를 결정합니다.
*배치 크기 (Batch Size)*: 한 번에 처리하는 데이터 샘플의 수를 의미합니다.
*에포크 수 (Number of Epochs)*: 전체 데이터셋을 몇 번 반복해서 학습할지를 결정합니다.
*최적화 알고리즘 (Optimizer)*: 모델을 학습시키기 위해 사용하는 알고리즘 (예: SGD, Adam).
*드롭아웃 비율 (Dropout Rate)*: 과적합을 방지하기 위해 일부 뉴런을 무작위로 비활성화하는 비율입니다.
### 데이터 증강: MNIST와 같은 간단한 데이터셋에서는 크게 필요하지 않을 수 있지만, 데이터 증강 기법을 적용하여 모델의 일반화 성능을 향상시킬 수 있습니다.
### 모델 구조 개선: 더 깊은 네트워크, 드롭아웃(Dropout), 배치 정규화(Batch Normalization) 등의 기법을 도입하여 모델의 성능을 개선할 수 있습니다.
### 학습률 스케줄러 도입: 학습 과정에서 학습률을 조정하는 스케줄러를 사용하여 학습의 안정성과 속도를 향상시킬 수 있습니다.

#### 하이퍼파라미터 튜닝, 데이터 증강, 모델 구조 개선(드롭아웃 적용) 시 결과
- 튜닝 및 학습, 테스트 시간 : Execution Time 109m 23.2s
- best hyperparameters : 
  Accuracy: 99.28%
  Params: 
    learning_rate: 0.0029079897364446237
    batch_size: 256
    dropout_rate: 0.2354694644424466
    num_epochs: 18
- 최적의 하이퍼파라미터로 학습 후 테스트 결과 
Final Accuracy with Best Params: 99.13%

## MNIST 문서 인식을 위한 CNN 및 LeNet5 모델 비교

### 내용
본 프로젝트는 MNIST 데이터셋을 사용하여 두 가지 CNN 아키텍처(CNN 모델과 LeNet5)를 구현하고 비교 분석하는 것을 목표로 합니다.

### 모델 아키텍처
#### CNN 모델
- 두 개의 합성곱 층과 풀링 층을 포함
- 정적 초기화된 fc1 레이어 사용

#### LeNet5 모델
- 전통적인 LeNet-5 아키텍처 구현
- 평균 풀링 사용 및 드롭아웃 레이어 추가

### 결과 요약
- **CNN 모델:**
  - 최종 정확도: 98.76%
- **LeNet5 모델:**
  - 최종 정확도: 99.00%
  
LeNet5 모델은 더 높은 테스트 정확도를 보였으나, 학습 손실은 더 큰 경향을 보였습니다. 이는 모델의 일반화 성능이 더 뛰어났음을 시사.
---

## 참고 자료
- [MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [합성곱 신경망(CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)


## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Author
Machu
