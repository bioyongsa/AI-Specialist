# VSN 02 — DNN & CNN 정리

> AI Specialist 양성과정 | Basic Architecture

---

## 📋 목차 (Contents Hierarchy)

```
VSN 02 — DNN & CNN
│
├── Part 1. DNN (Deep Neural Network)
│   ├── 1.1 딥러닝의 발전 역사
│   ├── 1.2 Perceptron
│   ├── 1.3 MLP (Multi-Layer Perceptron)
│   ├── 1.4 DNN (Deep Neural Network)
│   │   └── [비교표] Perceptron vs MLP vs DNN
│   ├── 1.5 Back-propagation (역전파)
│   ├── 1.6 활성화 함수 (Activation Functions)
│   │   ├── Sigmoid
│   │   └── Softmax
│   ├── 1.7 손실 함수 (Loss Functions)
│   └── 1.8 Regression vs Classification
│
├── Part 2. CNN (Convolutional Neural Network)
│   ├── 2.1 DNN의 한계: 통계적 패턴 매칭
│   ├── 2.2 Convolution 연산
│   ├── 2.3 왜 Convolution인가? (FCN → CNN)
│   ├── 2.4 핵심 연산 요소 (Padding / Stride / Pooling)
│   ├── 2.5 CNN 아키텍처 개요
│   ├── 2.6 Receptive Field
│   ├── 2.7 Convolution 변형 (1×1 / Group / Depthwise / Transposed / Deformable)
│   ├── 2.8 CNN이 학습하는 것
│   └── 2.9 FCN vs CNN 비교
│
└── Part 3. CNN 모델 발전사
    ├── 3.1 CNN 모델 비교 (LeNet / AlexNet / VGGNet / ResNet)
    ├── 3.2 VGGNet
    ├── 3.3 ResNet
    │   ├── Degradation Problem
    │   ├── Skip Connection 원리
    │   └── Architecture
    └── 3.4 Human Visual System vs CNN
```

---

## Part 1. DNN (Deep Neural Network)

---

### 1.1 딥러닝의 발전 역사

```
  성능
  ▲
  │                                                    ●  Deep Neural Network
  │                                              ●  SVM    (Pretraining)
  │                              ●  MLP+Backprop
  │         ●  XOR Problem          (1986)
  │   ●  ADALINE  (한계 발견)
  │  Perceptron
  ●  Electronic Brain
  └───────────────────────────────────────────────────────────► 시간
   1943  1957  1960  1969       1986        1995        2006  2010+

  ┌─────────────────────────────────────────────────────────┐
  │ 1943  McCulloch-Pitts │ 최초 뉴런 모델, 논리회로 흉내    │
  │ 1957  Rosenblatt      │ Perceptron: 학습 가능한 가중치   │
  │ 1960  Widrow-Hoff     │ ADALINE: Golden Age 시작         │
  │ 1969  Minsky-Papert   │ XOR 불가 증명 → AI Winter 시작  │
  │ 1986  Rumelhart-Hinton│ MLP + Backprop: 비선형 해결      │
  │ 1995  Vapnik-Cortes   │ SVM: 커널 기반 학습              │
  │ 2006  Hinton-Ruslan   │ DNN: 계층적 특징 학습 시대 개막  │
  └─────────────────────────────────────────────────────────┘
```

---

### 1.2 Perceptron

**개념:** 인공 뉴런의 가장 단순한 형태. 가중합(weighted sum)과 임계값(threshold)으로 **이진 분류**.

```
  생물학적 뉴런                   인공 퍼셉트론

  수상돌기(Dendrite)              입력 신호
  ┌────────────┐                 Constant(1)──(w₀=bias)──┐
  │  세포체    │                   x₁ ────────(w₁)────────┤
  │  (Soma)   │──►축삭──► 출력    x₂ ────────(w₂)────────┼──► Σ ──► Step fn ──► 출력
  └────────────┘                   x₃ ────────(w₃)────────┤
  시냅스(Synapse)                  xₙ ────────(wₙ)────────┘

  Step Function:
  출력
   1 ──────────────────────
   │
   0 ────────────                      f(x) = 1  if Σwᵢxᵢ ≥ θ
              θ (임계값)                        0  otherwise
   └────────────────────── 입력
```

**XOR 한계 (선형 분리 불가):**
```
  AND (해결 가능)   OR (해결 가능)    XOR (해결 불가 ✗)

  x₂↑              x₂↑               x₂↑
  1│  ○  ●         1│  ●  ●          1│  ●  ○
   │/               │/                │ ← 직선 하나로
  0│  ○  ○         0│  ○  ●          0│  ○  ●    분리 불가!
   └────────x₁      └────────x₁       └────────x₁
       0    1            0    1             0    1
   ●=1, ○=0
```

**Python pseudo code:**
```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w  = np.zeros(n_inputs + 1)   # +1 for bias
        self.lr = lr

    def predict(self, x):
        z = np.dot(np.insert(x, 0, 1), self.w)  # bias 추가 후 내적
        return 1 if z >= 0 else 0                 # Step function

    def fit(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                error = yi - self.predict(xi)
                self.w += self.lr * error * np.insert(xi, 0, 1)
```

---

### 1.3 MLP (Multi-Layer Perceptron)

**개념:** 하나 이상의 **은닉층(Hidden Layer)** 을 가진 순방향 신경망. 비선형 활성화 함수로 XOR 등 비선형 문제 해결.

```
  입력층         은닉층 1          은닉층 2         출력층
  (Input)        (Hidden 1)       (Hidden 2)       (Output)

   x₁ ─────────► ● ──────────────► ● ───────────►
                 ↗↘               ↗↘              ↘
   x₂ ─────────► ● ─────────────► ● ────────────► ● ──► Ŷ
                 ↘↗               ↘↗              ↗
   x₃ ─────────► ● ──────────────► ● ───────────►
                 ↗
   x₄ ─────────► ●

  각 연결의 계산:
  ┌──────────────────────────────────────────────┐
  │  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b (선형합)│
  │  a = activation(z)                 (비선형화)│
  └──────────────────────────────────────────────┘

  활성화 함수 비교 (0 근방):
  f(x)
   2 │                    ReLU /
   1 │           GELU  ·─/
   0 │─────────────────── → x
  -0 │       SiLU (부드러운 ReLU)
  -1 │  Sigmoid (S자, 0~1 압축)
     └────────────────────────
       -3  -2  -1   0   1   2
```

**Python pseudo code:**
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),               # 비선형성 부여
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)           # 입력 → 출력
```

---

### 1.4 DNN (Deep Neural Network)

**개념:** 은닉층을 **많이 쌓은** MLP. "Deep MLP + 확장 가능 아키텍처"

```
  DNN의 주요 문제점과 해결책:

  문제 1: 과적합 (Overfitting)
  ┌─────────────────────────────────────────────────────┐
  │  Underfitting     Good Fit      Overfitting         │
  │                                                     │
  │  ·   ·            · ·           · ·                │
  │    ·   ·     vs   ·/ ·   vs   ·╰─·╮·              │
  │  ────────         /─────        ──╯──╰──           │
  │  너무 단순        적절           너무 복잡           │
  └─────────────────────────────────────────────────────┘
  → 해결: Dropout / L2 정규화 / BatchNorm / Data Augmentation

  문제 2: 기울기 소실 (Vanishing Gradient)
  ┌──────────────────────────────────────────────────────┐
  │  출력층    Layer N    Layer 3    Layer 2    Layer 1  │
  │  ∂L/∂W :  0.5   →    0.1   →   0.01   →   ≈ 0 !!  │
  │           역전파 과정에서 gradient가 점점 작아짐     │
  └──────────────────────────────────────────────────────┘
  → 해결: ReLU 활성화 / BatchNorm / Residual Connection
```

#### [비교표] Perceptron vs MLP vs DNN

| 항목 | Perceptron | MLP | DNN |
|------|:----------:|:---:|:---:|
| 층 깊이 | 1 (입력→출력) | 2~3 | **3+** |
| 활성화 | Step | Sigmoid/ReLU | ReLU/GELU 등 |
| 표현력 | 낮음 | 중간 | **높음** |
| 정규화 | 불필요 | 종종 사용 | **필수** |
| 계산 비용 | 매우 낮음 | 낮음 | **높음(GPU)** |
| 역사 의의 | 신경망 기원 (1958) | 신경망 부활 (1980s) | 현대 딥러닝 근간 |

---

### 1.5 Back-propagation (역전파)

```
  ┌──────────────────── 학습 사이클 ────────────────────────┐
  │                                                         │
  │   ① Forward pass          ② Loss 계산                  │
  │   X ──► [모델 W] ──► Ŷ ──► L = f(Y, Ŷ)               │
  │         ▲                            │                  │
  │         │                            ▼                  │
  │   ④ Weight 업데이트        ③ Backward (Chain Rule)     │
  │   W_{t+1} = W_t - α·∂L/∂W ◄── gradient ∂L/∂W         │
  └─────────────────────────────────────────────────────────┘

  Chain Rule 전개 (2층 네트워크 예시):

   w₁──►[a₁]──►o₁──►w₂──►[a₂]──►ŷ──►L

  ∂L    ∂L    ∂ŷ    ∂a₂    ∂o₁    ∂a₁
  ── = ──── · ─── · ──── · ──── · ────
  ∂w₁   ∂ŷ   ∂a₂   ∂o₁   ∂a₁    ∂w₁
      = 2(ŷ-y) · a'(a₂) · w₂ · a'(a₁) · o₀

  Weight 업데이트 공식:
  ┌─────────────────────────────────────────────┐
  │  W_{t+1} = W_t  -  α  ·  ∂L/∂W_t          │
  │              ↑        ↑       ↑             │
  │           이전 가중치  학습률  gradient      │
  └─────────────────────────────────────────────┘

  메모리에 유지해야 하는 것:
  ┌─────────────────────────────────┐
  │ ✔ Weights (가중치)              │
  │ ✔ 활성화 함수의 미분값          │
  │ ✔ 활성화 값 (forward 이후)     │
  │ ✔ Gradient history              │
  └─────────────────────────────────┘
```

**Python pseudo code (PyTorch 학습 루프):**
```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()                            # Dropout/BN 활성화
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                # ① gradient 초기화
        logits = model(images)               # ② Forward pass
        loss   = criterion(logits, labels)   # ③ Loss 계산
        loss.backward()                      # ④ Backward (gradient 계산)
        optimizer.step()                     # ⑤ Weight 업데이트

# 실전: AMP(Automatic Mixed Precision)로 속도/메모리 최적화
scaler = torch.amp.GradScaler()

for images, labels in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type='cuda'):   # FP16 자동 전환
        logits = model(images)
        loss   = criterion(logits, labels)
    scaler.scale(loss).backward()                  # scaled backward
    scaler.step(optimizer)
    scaler.update()
```

---

### 1.6 활성화 함수 (Activation Functions)

#### Sigmoid (이진 분류)

```
  수식 유도:
  ┌─────────────────────────────────────────────────┐
  │  Odds = p/(1-p)          ← 사건 발생 확률 비율  │
  │  Logit  l = ln(Odds)     ← 로그 스케일 (-∞~+∞) │
  │  e^l = p/(1-p)                                  │
  │  ┌────────────────────────────┐                 │
  │  │  p = 1 / (1 + e^{-l})     │  ← Sigmoid 함수 │
  │  └────────────────────────────┘                 │
  │  dp/dl = p(1-p)           ← 미분 (backprop 사용)│
  └─────────────────────────────────────────────────┘

  Sigmoid 그래프:
  p(l)
  1.0 │                   ·─────────────
      │                ·  (3, 0.953)
      │             ·
  0.5 ├──── · ─ (0, 0.500)
      │  ·
  0.0 ├──────────────────────────────── l (logit)
      -10   -5    0    5    10

  미분 그래프 (dp/dl = p(1-p)):
  0.25│          ╭──╮
      │       ╭──╯  ╰──╮
  0.0 ├───────────────────── l
      (최댓값: l=0 에서 0.25)

  ⚠ 기울기 소실 문제: 양 끝(l≪0, l≫0)에서 미분값 ≈ 0
```

**Python pseudo code:**
```python
import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))        # 출력 범위: (0, 1)

def sigmoid_derivative(x):
    p = sigmoid(x)
    return p * (1 - p)                     # 최댓값 0.25 (x=0에서)

# 이진분류: nn.BCEWithLogitsLoss() 사용 권장 (내부에서 sigmoid 포함)
criterion = torch.nn.BCEWithLogitsLoss()
```

#### Softmax (다중 클래스 분류)

```
  로짓(Logit) → Softmax → 확률 분포

  출력층          Softmax 변환            확률
  ┌─────┐       e^{zᵢ}               ┌──────┐
  │ 1.3 │  ─── ─────────────────►   │ 0.02 │
  │ 5.1 │       Σⱼ e^{zⱼ}           │ 0.90 │ ← 최고 확률
  │ 2.2 │                            │ 0.05 │   합계 = 1.0
  │ 0.7 │                            │ 0.01 │
  │ 1.1 │                            │ 0.02 │
  └─────┘                            └──────┘

  지수 함수가 점수 차이를 증폭:
  e^5.1 / e^1.3 = e^3.8 ≈ 44.7배 차이!
  → 높은 점수: 더욱 강조 / 낮은 점수: 억제
```

**Python pseudo code:**
```python
def softmax(logits):
    # 수치 안정성: max를 빼서 overflow 방지
    e = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)

# 실전: CrossEntropyLoss = LogSoftmax + NLLLoss (내부 처리)
criterion = torch.nn.CrossEntropyLoss()
# logits를 그대로 입력 (별도 Softmax 불필요)
```

---

### 1.7 손실 함수 (Loss Functions)

```
  출력의 확률 분포 가정 → MLE 최대화 = NLL 최소화 = Loss 최소화

  ┌──────────────────────────────────────────────────────────────────┐
  │   분포       Likelihood           NLL (= Loss)        용도      │
  ├──────────────────────────────────────────────────────────────────┤
  │ Bernoulli  ŷ^y · (1-ŷ)^(1-y)   BCE                  이진분류  │
  │                                  -[y·logŷ+(1-y)log(1-ŷ)]       │
  ├──────────────────────────────────────────────────────────────────┤
  │ Categorical  ∏ ŷₖ^{yₖ}         CE                   다중분류  │
  │                                  -∑ yₖ · log(ŷₖ)               │
  ├──────────────────────────────────────────────────────────────────┤
  │  Normal    가우시안 분포         MSE = (ŷ-y)²         회귀      │
  ├──────────────────────────────────────────────────────────────────┤
  │  Laplace   이중지수 분포         MAE = |ŷ-y|          회귀(↑이상치)│
  └──────────────────────────────────────────────────────────────────┘

  핵심 연결:
  MLE 최대화  ←→  NLL 최소화  ←→  Loss 최소화
```

---

### 1.8 Regression vs Classification

```
  ┌──────────────────────────────────────────────────────────────┐
  │               Regression          Classification             │
  ├──────────────────────────────────────────────────────────────┤
  │  출력      연속값 (실수)           이산 클래스 레이블        │
  │  활성화    Linear (일반)           Sigmoid / Softmax         │
  │  손실      MSE, MAE               Cross Entropy             │
  │  지표      RMSE                   Accuracy, F1-score, mAP   │
  ├──────────────────────────────────────────────────────────────┤
  │  응용      Estimation             Object Category            │
  │            Pose Estimation        Image Segmentation         │
  │            Bounding Box Coord.    Scene Classification       │
  └──────────────────────────────────────────────────────────────┘

  Object Detection: Regression + Classification 동시 사용
  ┌──────────────────────────────────────────────────────┐
  │  이미지 ──► CNN ──► FC ──┬──► [cx, cy, w, h]  (MSE) │
  │                          └──► [클래스 확률]    (CE)  │
  └──────────────────────────────────────────────────────┘
```

---

## Part 2. CNN (Convolutional Neural Network)

---

### 2.1 DNN의 한계: 통계적 패턴 매칭

```
  FCN으로 이미지 처리 시 문제:

  원본 이미지 (4×4)          FCN 입력 (16차원 벡터)
  ┌────────────────┐          [p1, p2, p3, p4, p5, ..., p16]
  │ 1   2   3   4 │   →       ↑
  │ 5   6   7   8 │           공간 정보(위치 관계) 완전 소실!
  │ 9  10  11  12 │
  │13  14  15  16 │          100×100 grayscale + 1,000 뉴런
  └────────────────┘          = 10,000,000 파라미터 → 과적합!

  핵심:
  ┌─────────────────────────────────────────────────────────┐
  │  "딥러닝은 시각적 의미(semantics)가 아닌               │
  │   통계적 상관관계(statistical correlation)를 학습한다"  │
  └─────────────────────────────────────────────────────────┘
```

---

### 2.2 Convolution 연산

#### Convolution vs Cross-correlation vs Autocorrelation

```
  ┌─────────────────────────────────────────────────────────┐
  │  Convolution    │  Cross-correlation  │  Autocorrelation │
  │  f * g          │  g ★ f             │  f ★ f           │
  ├─────────────────┼─────────────────────┼──────────────────┤
  │ ① 필터 반전(Mirror)│ 반전 없음         │  자기 자신과     │
  │ ② 슬라이딩 이동   │ 슬라이딩           │  슬라이딩        │
  │ ③ 원소별 곱셈     │ 원소별 곱셈        │  원소별 곱셈     │
  │ ④ 합산           │ 합산               │  합산            │
  └─────────────────┴─────────────────────┴──────────────────┘
  ⚠ CNN에서는 실제로 Cross-correlation을 사용!
    (관례적으로 "Convolution"이라 부름)
```

#### 기존 수작업 필터 (Edge Detection)

```
  Sobel 수직 엣지           Sobel 수평 엣지
  ┌───────────────┐         ┌───────────────┐
  │  1    0   -1  │         │  1    2    1  │
  │  2    0   -2  │         │  0    0    0  │
  │  1    0   -1  │         │ -1   -2   -1  │
  └───────────────┘         └───────────────┘
  좌→우 밝기 변화 감지        위→아래 밝기 변화 감지

  원본           수직 Sobel        수평 Sobel
  ┌──────────┐  ┌──────────┐      ┌──────────┐
  │  ████    │  │  █  █   │      │          │
  │  █  █    │  │  █  █   │      │ ████████ │
  │  ████    │  │  █  █   │      │          │
  │  █  █    │  │  █  █   │      │ ████████ │
  └──────────┘  └──────────┘      └──────────┘
  (문자 이미지)  (세로 선 강조)    (가로 선 강조)

  CNN은 이런 필터를 데이터로부터 자동으로 학습!
```

**Python pseudo code:**
```python
import torch.nn.functional as F

sobel_v = torch.tensor([[ 1.,  0., -1.],
                         [ 2.,  0., -2.],
                         [ 1.,  0., -1.]]).view(1, 1, 3, 3)

output = F.conv2d(image, sobel_v, padding=1)  # 수작업 필터 적용
# CNN: 이런 필터를 nn.Conv2d가 데이터로부터 자동 학습
```

#### CNN에서의 Convolution (Multi-channel)

```
  입력 (6×6×3)     필터들 (학습 대상)      출력 (4×4×2)

  ┌──────────┐     필터1 (3×3×3)          ┌────────┐
  │ R채널 6×6│     ┌───────────┐           │ 채널 1 │
  │ G채널 6×6│  *  │3×3 kernel │ × 3채널 = │  4×4  │
  │ B채널 6×6│     │ (학습가능) │           ├────────┤
  └──────────┘     └───────────┘           │ 채널 2 │
                   필터2 (3×3×3)           │  4×4  │
                   ┌───────────┐           └────────┘
                   │3×3 kernel │ × 3채널 =
                   │ (학습가능) │
                   └───────────┘

  ┌──────────────────────────────────────────────────────┐
  │  필터 수     = 출력 채널 수 (설계자가 결정)          │
  │  필터 크기   = 수용 영역(receptive field) 크기 결정  │
  │  필터 가중치 = 학습으로 결정 ✔ (핵심!)              │
  │  Activation Map = 필터를 통과한 결과물 (시각화 대상) │
  └──────────────────────────────────────────────────────┘

  출력 크기 공식:
  ┌─────────────────────────────────────────────┐
  │  H_out = (H_in + 2×P - K) / S + 1         │
  │   P: Padding  K: Kernel size  S: Stride    │
  └─────────────────────────────────────────────┘
```

**Python pseudo code:**
```python
import torch.nn as nn

# 입력: (B, 3, H, W) → 출력: (B, 64, H, W)
conv = nn.Conv2d(
    in_channels=3,    # 입력 채널 (RGB)
    out_channels=64,  # 필터 수 = 출력 채널 수
    kernel_size=3,    # 필터 크기 3×3
    stride=1,
    padding=1         # same padding → 공간 크기 유지
)
```

---

### 2.3 왜 Convolution인가? (FCN → CNN)

```
  [1] Parameter Sharing (파라미터 공유)

  FCN (100×100 gray + 1,000 뉴런):   CNN (3×3 필터 64개):
  ┌──────────────────────────────┐   ┌──────────────────────────────┐
  │  파라미터: 10,000,000        │   │  파라미터: 576               │
  │  연결마다 고유한 가중치      │   │  같은 가중치를 전체에 재사용  │
  │  → 과적합 위험 높음 ✗       │   │  → 과적합 방지 ✔            │
  └──────────────────────────────┘   └──────────────────────────────┘

  [2] Local Connectivity (국소 연결)

  인간 시각 피질 V1:             CNN:
  ┌────────────────────────┐     ┌────────────────────────┐
  │ 각 뉴런은 시야의        │     │ 각 뉴런은 입력의       │
  │ 작은 영역(수용영역)만   │◄───►│ 국소 패치(patch)에만   │
  │ 처리 (Receptive Field) │     │ 연결                   │
  └────────────────────────┘     └────────────────────────┘
  → 엣지, 텍스처, 형태 같은 국소 패턴 학습

  [3] Translation Invariance (평이 불변성)

  ┌──────┐    ┌──────┐    ┌──────┐
  │🐱   │    │ 🐱  │    │   🐱│   → 모두 "고양이"로 인식
  └──────┘    └──────┘    └──────┘
  왼쪽 위      중앙          오른쪽    같은 필터 가중치 사용
```

---

### 2.4 핵심 연산 요소

#### Padding

```
  목적: ① 출력 크기 감소 방지   ② 경계(border) 픽셀 정보 활용

  padding=0 (적용 전):           padding=1 (적용 후):
  ┌───────────┐                  ┌─────────────────┐
  │ ■  ■  ■  │  3×3 이미지에    │ 0  0  0  0  0  │
  │ ■  ■  ■  │  3×3 kernel →   │ 0 ┌──────────┐ 0│
  │ ■  ■  ■  │  출력: 1×1 !!   │ 0 │ ■  ■  ■ │ 0│
  └───────────┘                  │ 0 │ ■  ■  ■ │ 0│
  크기 급격히 줄어듦              │ 0 │ ■  ■  ■ │ 0│
                                  │ 0 └──────────┘ 0│
                                  │ 0  0  0  0  0  │
                                  └─────────────────┘
                                  출력: 3×3 (크기 유지)
```

#### Stride

```
  stride=1 (한 칸씩 이동):

  ┌──────────────────────┐
  │[■ ■ ■]. . . . . .  │  1번 위치
  │[■ ■ ■]. . . . . .  │
  │[■ ■ ■]. . . . . .  │
  │ . .[■ ■ ■]. . . .  │  2번 위치 (1칸 이동)
  └──────────────────────┘
  → 전체 해상도 유지

  stride=2 (두 칸씩 이동):

  ┌──────────────────────┐
  │[■ ■ ■]. [■ ■ ■]. .  │  1번, 2번 위치
  │[■ ■ ■]. [■ ■ ■]. .  │
  │[■ ■ ■]. [■ ■ ■]. .  │
  └──────────────────────┘
  → 공간 크기 절반으로 감소
```

#### Pooling

```
  Max Pooling (2×2, stride=2):       Average Pooling (2×2, stride=2):

  입력 4×4            출력 2×2       입력 4×4             출력 2×2
  ┌──────────────┐    ┌──────┐      ┌──────────────┐      ┌──────────┐
  │[1  1] [2  4] │    │ 6  8 │      │[1  1] [2  4] │      │3.25  5.25│
  │[5  6] [7  8] │    │      │      │[5  6] [7  8] │      │          │
  │[3  2] [1  0] │    │ 5  4 │      │[3  2] [1  0] │      │2.00  2.50│
  │[1  2] [3  4] │    └──────┘      │[1  2] [3  4] │      └──────────┘
  └──────────────┘                  └──────────────┘
  [ ] 영역의 최댓값 선택              [ ] 영역의 평균 계산

  목적: 공간 크기 감소 + 위치 변화에 강건(robustness)
```

**Python pseudo code:**
```python
conv_block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # same padding
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)   # 공간 크기 절반
)
# 입력: (B, 3, 32, 32) → 출력: (B, 64, 16, 16)
```

---

### 2.5 CNN 아키텍처 개요

```
  입력            Convolutional Block (반복)         분류기
  ┌──────┐       ┌──────────────────────────┐       ┌──────────────┐
  │      │       │  Conv(3×3)               │       │  Flatten     │
  │Image │──────►│  BatchNorm               │──────►│     ↓        │──► 예측
  │3×H×W│       │  ReLU                    │       │  FC Layer    │
  │      │       │  MaxPool(2×2)            │       │     ↓        │
  └──────┘       └──────────────────────────┘       │  Softmax     │
                 (특징 추출: 해상도↓ 채널↑)          └──────────────┘

  핵심 포인트:
  ┌──────────────────────────────────────────────────────────────┐
  │  CNN이 실제로 학습하는 것: 필터(커널) 가중치 (3×3 행렬들)   │
  │  시각화로 보이는 것:       Activation Map (필터 통과 결과)   │
  │     → Activation Map ≠ 필터 그 자체                         │
  └──────────────────────────────────────────────────────────────┘
```

---

### 2.6 Receptive Field (수용 영역)

```
  개념: 출력 픽셀 하나가 참조하는 입력 이미지의 영역 크기

  3×3 Conv × 2번 vs 5×5 Conv × 1번:

  [Layer 1, 3×3]     [Layer 2, 3×3]       [Layer 3 의 한 픽셀이
  ┌─────────┐         ┌─────────┐          참조하는 원본 영역]
  │■ ■ ■   │         │         │          ┌─────────────────┐
  │■ ■ ■   │ ───────►│ · · ·   │ ───────► │■ ■ ■ ■ ■      │
  │■ ■ ■   │         │ · ● ·   │          │■ ■ ■ ■ ■      │
  └─────────┘         │ · · ·   │          │■ ■ ■ ■ ■      │
  3×3 패치 참조        └─────────┘          │■ ■ ■ ■ ■      │
                       ●의 출력은            │■ ■ ■ ■ ■      │
                       5×5 원본 참조         └─────────────────┘
                                            5×5 수용 영역 (동일!)

  ┌───────────────────────────────────────────────────────────────┐
  │                Two 3×3 Conv    vs    One 5×5 Conv             │
  ├──────────────────┬─────────────────┬───────────────────────────┤
  │  수용 영역       │     5×5         │           5×5             │
  │  파라미터 수     │  2×9 = 18  ✔   │          5×5 = 25         │
  │  비선형 활성화   │     2회    ✔   │           1회             │
  │  계산 속도       │    빠름    ✔   │          느림             │
  │  특징 학습 깊이  │    깊음    ✔   │          얕음             │
  └──────────────────┴─────────────────┴───────────────────────────┘
  → VGGNet, ResNet 모두 3×3 Conv 스택 방식 채택!

  Dilated Convolution (팽창 합성곱):
  dilation=1 (일반):    dilation=2 (팽창):     효과:
  ┌───────────┐         ┌─────────────────┐   파라미터는 그대로,
  │ ■  ■  ■  │         │ ■  .  ■  .  ■  │   수용 영역은 확장!
  │ ■  ■  ■  │         │ .  .  .  .  .  │
  │ ■  ■  ■  │         │ ■  .  ■  .  ■  │   3×3 → 5×5 영역 커버
  └───────────┘         │ .  .  .  .  .  │   (■만 계산에 사용)
  수용 영역 3×3          │ ■  .  ■  .  ■  │
                         └─────────────────┘
                         수용 영역 5×5 (·는 건너뜀)
```

**Python pseudo code (Receptive Field 근사 계산):**
```python
def rf_update(rf, jump, kernel_size, stride):
    rf_new   = rf + (kernel_size - 1) * jump   # receptive field 누적
    jump_new = jump * stride                    # 누적 stride
    return rf_new, jump_new

# ResNet18 CIFAR용 RF 계산
rf, jump = 1, 1
rf, jump = rf_update(rf, jump, 3, 1)          # conv1 (3×3, s=1)
for _ in range(2 * 2):                         # layer1: 2블록×2conv
    rf, jump = rf_update(rf, jump, 3, 1)
rf, jump = rf_update(rf, jump, 3, 2)           # layer2 (stride=2)
# ... layer3, layer4 동일
print(f"Estimated Receptive Field: {rf}")
# → 32×32 이미지 전체를 초과 → 전체 이미지 참조 가능
```

---

### 2.7 Convolution 변형

#### 1×1 Convolution (Pointwise)

```
  역할: 채널 수 변경 (공간 크기는 유지)

  입력 (H × W × C)  +  K개 1×1 필터  →  출력 (H × W × K)

  각 위치에서의 동작:
  위치(i,j)의 채널벡터 [a₁, a₂, a₃]
       × 가중치 행렬 C (3×4)
  = [a₁c₁₁+a₂c₂₁+a₃c₃₁,  a₁c₁₂+...,  ...]  → 4차원 출력

  활용:
  ┌─────────────────────────────────────────────────────┐
  │  채널 축소 (Bottleneck): 256ch → 64ch  (연산량 감소)│
  │  채널 확장:               64ch → 256ch              │
  │  FC Layer 대체:      (H×W×C) → (H×W×K)            │
  └─────────────────────────────────────────────────────┘
```

**Python pseudo code:**
```python
pointwise = nn.Conv2d(256, 64, kernel_size=1)  # 채널 256→64, 공간 유지
```

#### Group Convolution

```
  일반 Conv:                    Group Conv (G=3):

  입력 채널: 1  2  3  4  5  6   그룹1    │ 그룹2    │ 그룹3
             │  │  │  │  │  │   [1, 2]  │ [3, 4]  │ [5, 6]
             └──┴──┴──┴──┴──┘     ↓     │   ↓     │   ↓
           (전체 채널 혼합)       [필터] │ [필터]  │ [필터]
  출력: 전체 파라미터 사용          ↓     │   ↓     │   ↓
                                 [1, 2] │ [3, 4]  │ [5, 6]  ← 출력 합산

  장점: 파라미터 1/G, 메모리 절약, 병렬화
  단점: 그룹 간 채널 상호작용 없음 → 1×1 conv로 보완
```

**Python pseudo code:**
```python
group_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=4)
```

#### Depthwise Separable Convolution

```
  파라미터 비교 (입력 3ch, 출력 64ch, 3×3 kernel):
  ┌───────────────────────────────────────────────────────┐
  │  일반 Conv:           3×3×3×64 = 1,728 파라미터      │
  │  Depthwise Separable: 3×3×3 + 1×1×3×64 = 219 파라미터│
  │                       ↑ 87% 감소!                     │
  └───────────────────────────────────────────────────────┘

  Step 1: Depthwise Conv (공간 특징 추출, 채널 독립)
  입력 (H×W×C)           C개 3×3 필터            출력 (H×W×C)
  ┌──────────┐           ┌──────────┐             ┌──────────┐
  │ 채널 1   │ ────────► │  kernel1 │ ──────────► │ 채널 1   │
  │ 채널 2   │ ────────► │  kernel2 │ ──────────► │ 채널 2   │
  │ 채널 3   │ ────────► │  kernel3 │ ──────────► │ 채널 3   │
  └──────────┘           └──────────┘             └──────────┘
  채널 간 상호작용 없음!

  Step 2: Pointwise Conv (채널 정보 혼합)
  입력 (H×W×C)    K개 1×1 필터    출력 (H×W×K)
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 채널 1   │    │          │    │ 채널 1   │
  │ 채널 2   │─── │  1×1×C×K│───►│ 채널 2   │
  │ 채널 3   │    │          │    │  ...     │
  └──────────┘    └──────────┘    └──────────┘
  채널 간 혼합 ✔

  → MobileNet, EfficientNet 등 경량(Edge) 모델에서 활용
```

**Python pseudo code:**
```python
depthwise_separable = nn.Sequential(
    # Step 1: Depthwise (채널별 독립)
    nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # groups=in_ch
    nn.BatchNorm2d(64),
    nn.ReLU(),
    # Step 2: Pointwise (채널 간 혼합)
    nn.Conv2d(64, 128, kernel_size=1),
    nn.BatchNorm2d(128),
    nn.ReLU()
)
```

#### Transposed Convolution (업샘플링)

```
  목적: feature map 공간 해상도 증가 (upsampling)
  용도: U-Net, FCN, GAN Generator

  1D 예시 (입력 [a, b], 필터 [x, y, z]):

  입력   필터      출력 (크기: in_size × stride 이상)
   a  ──────────► ax
                  ay
   b  ──────────► az + bx  ← 겹치는 부분은 합산
                  by
                  bz
  출력: [ax, ay, az+bx, by, bz]

  2D 예시 (stride=2):
  입력 (2×2)       출력 (4×4 영역 분산)
  ┌──────┐         ┌─────────────────┐
  │ 1  2 │  Conv   │ 1  2  2  4     │
  │ 3  0 │ ──────► │ 3  4  0  0     │
  └──────┘  Transp │ 3  6  0  0     │
                   │ 9  0  0  0     │
                   └─────────────────┘

  ⚠ Checkerboard Artifact:
  ┌────────────────────────────────────────────────────┐
  │ 겹치는 영역의 불균등한 기여 → 격자 무늬 노이즈     │
  │ 권장 대안: Bilinear Upsample + Conv 조합          │
  └────────────────────────────────────────────────────┘
```

**Python pseudo code:**
```python
upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 크기 ×2

# 체커보드 방지 대안 (권장)
upsample_safe = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(64, 32, kernel_size=3, padding=1)
)
```

#### Deformable Convolution (2017)

```
  일반 CNN (고정 격자):         Deformable Conv (학습된 오프셋):

  샘플링 위치 (항상 고정):       샘플링 위치 (객체에 따라 적응):
  ● ● ●                         ●     ●
  ● ● ●  ← 3×3 격자             ● ●             ← 유연하게 이동
  ● ● ●                              ●  ●
                                 ●        ●

  수식:
  ┌──────────────────────────────────────────────────────────┐
  │ 표준:      y(p₀) = Σ w(pₙ) · x(p₀ + pₙ)               │
  │ Deformable: y(p₀) = Σ w(pₙ) · x(p₀ + pₙ + Δpₙ)       │
  │                                              ↑           │
  │                                  학습 가능한 오프셋       │
  └──────────────────────────────────────────────────────────┘

  실험 결과 (Scaled MNIST):
  ┌─────────────────────────────────────────┐
  │              Regular   Deformable       │
  │  일반 MNIST   98.74%    97.27%          │
  │  크기변환 MNIST 57.01%   92.55%  ✔    │
  └─────────────────────────────────────────┘
  크기·형태 변형에 강건!
```

---

### 2.8 CNN이 학습하는 것

```
  계층별 학습 특징 (얕은 층 → 깊은 층):

  Layer 1          Layer 2           Layer 3+          깊은 층
  ┌───────────┐    ┌─────────────┐   ┌───────────┐    ┌──────────────┐
  │ /  |  \  │    │ ╔╗ ╭─╮ ╱╲ │   │ ╭──────╮  │    │ 🐱 얼굴    │
  │ ─  +  ×  │    │ 텍스처,     │   │ 형태,     │    │ 🚗 바퀴    │
  │ 엣지, 방향│    │ 코너, 곡선  │   │ 패턴      │    │ 🏠 지붕    │
  └───────────┘    └─────────────┘   └───────────┘    └──────────────┘
  저수준 (Low)       중간 (Mid)                          고수준 (High)
  ← 공간 해상도 크고, 채널 수 적음          공간 작고, 채널 수 많음 →
```

**Activation Map + Grad-CAM 시각화 코드 (노트북):**
```python
# ① Activation Map: Hook으로 중간 레이어 출력 캡처
def capture_activation(model, layer_name, x_single):
    activ = {}

    def hook(m, inp, out):
        activ['feat'] = out.detach().cpu()   # (B, C, H, W)

    layer = get_module_by_name(model, layer_name)
    h = layer.register_forward_hook(hook)    # hook 등록
    model.eval()
    with torch.no_grad():
        _ = model(x_single)
    h.remove()
    return activ['feat'][0]                  # (C, H, W)

# ② 얕은 층 vs 깊은 층 비교
for title, layer_name in {
    "얕은 층 (layer1)": "layer1.1.conv2",   # 엣지 감지
    "깊은 층 (layer4)": "layer4.1.conv2",   # 고수준 특징
}.items():
    feat = capture_activation(model, layer_name, x_in)
    # Top-16 채널 시각화 (절댓값 평균 기준)
    ch_scores = feat.abs().mean(dim=(1, 2))
    top_idx   = torch.topk(ch_scores, k=16).indices
```

---

### 2.9 FCN vs CNN 비교

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                   FCN                    CNN                     │
  ├──────────────────────────────────────────────────────────────────┤
  │  연결 방식  모든 노드 연결           국소 영역 (receptive field) │
  │  가중치     연결마다 고유            가중치 공유 ✔               │
  │  공간 구조  무시 ✗                  보존 ✔                       │
  │  과적합     높음 ✗                  낮음 ✔                       │
  │  해석성     어려움                  낮은 층은 직관적 ✔           │
  │  주 용도    분류 헤드               비전, 특징 추출              │
  ├──────────────────────────────────────────────────────────────────┤
  │  FCN 구조:                          CNN 구조:                    │
  │                                                                  │
  │  Input ●─────────●                  Input ─► Conv ─► Pool ─► FC │
  │        ●─────────●  Output                    ↓               ↑  │
  │        ●─────────●               국소패턴학습        전역판단   │
  │  (전체 연결, 공간 무시)            (weight sharing, 공간 보존)  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Part 3. CNN 모델 발전사

---

### 3.1 CNN 모델 연대기

```
  1998         2012           2014           2015
   │            │              │              │
   ▼            ▼              ▼              ▼
 LeNet-5     AlexNet        VGGNet         ResNet
 ┌──────┐    ┌──────┐      ┌──────┐       ┌──────┐
 │ 7    │    │ 8    │      │16/19 │       │18~   │
 │layers│    │layers│      │layers│       │152+  │
 │32×32 │    │224×224      │224×224       │layers│
 │Tanh  │    │ReLU  │      │ReLU  │       │ReLU  │
 │수만  │    │~60M  │      │~138M │       │~25M  │
 └──────┘    └──────┘      └──────┘       └──────┘
    ↑             ↑              ↑              ↑
 최초 실용     GPU+Dropout    균일 3×3       Skip
  CNN 모델    ImageNet 1위   conv 스택     Connection

  ┌────────────────────────────────────────────────────────────────┐
  │         LeNet-5   │   AlexNet   │  VGGNet    │    ResNet      │
  ├──────────────────────────────────────────────────────────────  │
  │동기  MLP 파라미터  │  GPU 대규모 │  깊은 네트 │ Degradation   │
  │      과다         │  학습       │  워크      │ 문제           │
  │혁신  가중치 공유   │  GPU,Dropout│  균일 3×3  │ Residual      │
  │                   │             │  Conv      │ Connection     │
  │성능  낮은 정확도   │ ImageNet 1위│ 높음(무거움)│ SOTA ✔       │
  └────────────────────────────────────────────────────────────────┘
```

---

### 3.2 VGGNet

```
  핵심: 3×3 Conv만 균일하게 쌓아 단순하고 깊은 네트워크 구성

  입력 (3×224×224)
  │
  ├─► [Conv3-64 × 2] + MaxPool(/2)          → (64 × 112 × 112)
  │
  ├─► [Conv3-128 × 2] + MaxPool(/2)         → (128 × 56 × 56)
  │
  ├─► [Conv3-256 × 3] + MaxPool(/2)         → (256 × 28 × 28)
  │
  ├─► [Conv3-512 × 3] + MaxPool(/2)         → (512 × 14 × 14)
  │
  ├─► [Conv3-512 × 3] + MaxPool(/2)         → (512 × 7 × 7)
  │
  ├─► Flatten                               → (1 × 25,088)
  ├─► FC-4096 + ReLU + Dropout
  ├─► FC-4096 + ReLU + Dropout
  └─► FC-1000 + Softmax                     → (1,000 클래스)

  Conv 표기: conv{kernel}-{out_ch}
  예) conv3-64 = 3×3 필터 64개

  필터 파라미터 이해:
  ┌─────────────────────────────────────────────────────────┐
  │  첫 번째 레이어: Conv3-64 → 가중치 64 × 3 × 3 × 3     │
  │                              ↑출력ch  ↑입력ch ↑커널크기│
  │  = 1,728 파라미터                                       │
  │                                                         │
  │  출력 Activation Map: (64, 224, 224)                   │
  │  = 64장의 224×224 특징 지도                             │
  └─────────────────────────────────────────────────────────┘
```

---

### 3.3 ResNet

#### Degradation Problem

```
  실험: CIFAR-10에서 56-layer vs 20-layer plain network

  학습 오차(%)                    테스트 오차(%)
  15│─ ─ ─ 56-layer ─ ─ ─       15│─ ─ ─ 56-layer ─ ─ ─
    │                              │
  10│          20-layer            │          20-layer
    │         (낮은 오차)          │         (낮은 오차)
   5│                              │
    └──────────────► 학습 횟수     └──────────────► 학습 횟수

  56-layer가 오히려 더 높은 오차! → Degradation Problem

  원인 분석:
  ┌──────────────────────────────────────────────────────────────┐
  │  과적합?       → ✗ 학습 오차도 높음                        │
  │  기울기 소실?  → ✗ BatchNorm으로 상당 부분 해결됨          │
  │                                                              │
  │  진짜 원인: Optimization Difficulty                         │
  │  고차원 비선형 공간에서 'Identity Mapping' 학습이 어려움    │
  │    → f(x) = x 를 학습하는 것 자체가 복잡한 최적화 문제     │
  └──────────────────────────────────────────────────────────────┘
```

#### Skip Connection 원리

```
  일반 연결 (Plain):             Skip Connection (Residual):

       x                               x
       │                             ┌─┼─────────────────────┐
       ▼                             │ ▼                     │  identity
  ┌─────────┐                    ┌─────────┐                 │
  │ Conv    │                    │ Conv    │                  │
  │ (3×3)  │                    │ (3×3)  │                  │
  └────┬────┘                   └────┬────┘                  │
       │                             │                        │
     BatchNorm                    BatchNorm                   │
       │                             │                        │
     ReLU                          ReLU                       │
       │                             │                        │
  ┌─────────┐                    ┌─────────┐                 │
  │ Conv    │                    │ Conv    │                  │
  │ (3×3)  │                    │ (3×3)  │                  │
  └────┬────┘                   └────┬────┘                  │
       │                             │                        │
     BatchNorm                    BatchNorm                   │
       │                             │         ┌─────────┐   │
       ▼                             └────────►│    ⊕    │◄──┘
     ReLU                                      └────┬────┘
       │                                            │
       ▼                                          ReLU
    H(x) = σ(σ(xW₁)W₂)                           │
                                                   ▼
                                          H(x) = F(x) + x
                                          F(x) = σ(σ(xW₁)W₂)

  ─────────────────────────────────────────────────────────────────
  왜 효과적인가?

  초기화: Xavier/He → W ≈ 0 근방에서 시작

  일반 연결:  H(x) = σ(σ(xW₁)W₂)
    → W₁ = W₂ = I (항등 행렬) 학습이 어려움

  Skip 연결:  H(x) = F(x) + x
    → W₁ = W₂ = 0 이면 F(x) = 0,  H(x) = x 자동 성립! ✔
    → 초기화 편리 + 활성화 함수 비선형 영역(≈0 근방) 활용
  ─────────────────────────────────────────────────────────────────

  Loss Landscape 비교:
  plain 네트워크:            ResNet:
  ┌─────────────────┐        ┌─────────────────┐
  │  /\  /\/\ /\   │        │                 │
  │ /  \/     \/   │  ───►  │    ╭──────╮    │
  │ 복잡, 울퉁불퉁  │        │    매끄러움     │
  └─────────────────┘        └─────────────────┘
  local minimum 위험           안정적 수렴
```

**Python pseudo code (BasicBlock):**
```python
class BasicBlock(nn.Module):
    """ResNet-18/34의 기본 블록"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1,      1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # 채널/크기 불일치 시 identity를 맞추는 1×1 Conv
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x                               # skip 저장
        out = F.relu(self.bn1(self.conv1(x)))      # F(x) 첫 번째 conv
        out = self.bn2(self.conv2(out))            # F(x) 두 번째 conv

        if self.downsample:
            identity = self.downsample(x)         # 차원 맞추기

        return F.relu(out + identity)              # F(x) + x → ReLU
```

#### Architecture

```
  ResNet-18 구조 (CIFAR-10 개조 버전):

  입력 (3×32×32)
  │
  ├─► conv1: Conv(3×3, 64, s=1, p=1) → BN → ReLU   (64×32×32)
  │          ※ 원본 ImageNet: 7×7, s=2 / MaxPool 있음
  │          ※ CIFAR: 3×3, s=1 / MaxPool 제거 (이미지가 작아서)
  │
  ├─► layer1: [BasicBlock × 2]  64ch              (64×32×32)
  │     블록 구조: Conv(3×3)→BN→ReLU→Conv(3×3)→BN ⊕ identity → ReLU
  │
  ├─► layer2: [BasicBlock × 2] 128ch, stride=2    (128×16×16)
  │     첫 블록: stride=2로 downsample + 1×1 conv identity 조정
  │
  ├─► layer3: [BasicBlock × 2] 256ch, stride=2    (256×8×8)
  │
  ├─► layer4: [BasicBlock × 2] 512ch, stride=2    (512×4×4)
  │
  ├─► AdaptiveAvgPool2d(1×1)                       (512×1×1)
  ├─► Flatten                                      (512,)
  └─► FC(512 → 10)                                 (10,) → CrossEntropy

  총 파라미터: ~11.17M   |   추론 연산량: ~555 MFLOPs

  Bottleneck Block (ResNet-50/101/152):
  ┌──────────────────────────────────────────────────────┐
  │  입력 (256ch)                                        │
  │       │                  identity                    │
  │       ├──────────────────────────────────────────────┤
  │       ▼                                              │
  │  1×1 Conv, 64ch   ← 채널 압축 (연산량↓)             │
  │       ▼                                              │
  │  3×3 Conv, 64ch   ← 공간 특징 추출                  │
  │       ▼                                              │
  │  1×1 Conv, 256ch  ← 채널 복원                       │
  │       │                                              │
  │       ⊕ ◄────────────────────────────────────────── │
  │       ▼                                              │
  │  출력 (256ch) + ReLU                                │
  └──────────────────────────────────────────────────────┘
```

**Python pseudo code (CIFAR용 ResNet18 빌드):**
```python
import torchvision.models as models

def build_resnet18_for_cifar10(num_classes=10):
    model = models.resnet18(weights=None)

    # CIFAR-10은 32×32 → ImageNet용 stem을 소형화
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3,     # 7×7 → 3×3
                             stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()                       # MaxPool 제거
    model.fc      = nn.Linear(model.fc.in_features, num_classes)
    return model

model = build_resnet18_for_cifar10().to(device)
```

---

### 3.4 Human Visual System vs CNN

```
  인간 시각 시스템                    CNN

  망막(Retina)                        Input Image (픽셀)
       │                                   │
       ▼                                   ▼
  V1 피질                            Conv Layer 1
  (엣지, 방향 감지)          ◄─────► (엣지 검출 필터)
  [ / | \ ─ ]                        [ / | \ ─ ]
       │                                   │
       ▼                                   ▼
  V2 피질                            Conv Layer 2~3
  (코너, 텍스처)             ◄─────► (텍스처, 코너)
  [ ╔╗ ╭─╮ ]                         [ ╔╗ ╭─╮ ]
       │                                   │
       ▼                                   ▼
  V4 피질                            Deep Conv Layers
  (형태, 패턴)               ◄─────► (형태, 물체 부위)
  [ ○ △ ◇ ]                          [ ○ △ ◇ ]
       │                                   │
       ▼                                   ▼
  IT 피질                            FC Layers
  (얼굴, 자동차 등)          ◄─────► (객체 인식)
  🐱 🚗 🏠                            🐱 🚗 🏠

  ┌─────────────────────────────────────────────────────────────────┐
  │                │  Human Visual System    │  CNN                 │
  ├────────────────┼─────────────────────────┼──────────────────────┤
  │  처리 단위     │  뉴런 (수용 영역)        │  Conv 필터 (커널)    │
  │  학습 방식     │  가소성, Hebbian 학습    │  역전파 + GD         │
  │  강점          │  소수 예시 일반화        │  대규모 데이터 성능  │
  │  약점          │  착시에 취약            │  Adversarial 공격    │
  └─────────────────────────────────────────────────────────────────┘

  Adversarial Attack 예시:
  ┌──────────────────────────────────────────────────────────────┐
  │  🐼 판다 이미지  +  ε×노이즈(육안 감지 불가)  =  긴팔원숭이?! │
  │  "panda" 99.3%       +0.007×noise         →  "gibbon" 99.3%  │
  │  → CNN은 통계적 상관관계 학습, 의미 이해 아님!               │
  └──────────────────────────────────────────────────────────────┘
```

---

## 📎 부록: 노트북 핵심 코드 요약

### Grad-CAM 동작 원리 및 구현

```
  Grad-CAM 처리 흐름:

  입력 이미지
       │
       ▼  ① Forward pass
  ┌──────────────────────────────────────────────────────┐
  │  CNN Feature Extractor                              │
  │  ...                                                │
  │  [target layer] → Activation Map A (B×C×H×W) 저장 │← forward hook
  │  ...                                                │
  │  classifier → logits (B×K)                          │
  └────────────────────────────┬─────────────────────────┘
                               │
  ② target class score 역전파  │
                               ▼
  ┌──────────────────────────────────────────────────────┐
  │  [target layer] ← Gradient ∂score/∂A (B×C×H×W) 저장│← backward hook
  └────────────────────────────┬─────────────────────────┘
                               │
  ③ CAM 계산                  ▼
  ┌──────────────────────────────────────────────────────┐
  │  w_c = GAP(∂score/∂A)      ← 채널별 중요도 가중치   │
  │  CAM = ReLU(Σ w_c · A_c)   ← 양의 기여만 선택       │
  │  CAM → 원본 크기로 upsample + 원본에 overlay         │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  "모델이 어디를 보고 이 클래스로 예측했는가?" 시각화
```

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activ = None   # forward activation
        self.grad  = None   # backward gradient
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activ', o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grad', go[0]))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        logits[:, class_idx].sum().backward()  # target class backward

        w   = self.grad.mean(dim=(2, 3), keepdim=True)  # 채널별 중요도
        cam = F.relu((w * self.activ).sum(dim=1, keepdim=True))
        cam = (cam - cam.min()) / (cam.max() + 1e-6)     # 0~1 정규화
        return cam

# 사용 예시
target_layer = model.layer4[1].conv2
grad_cam     = GradCAM(model, target_layer)
cam          = grad_cam(x.unsqueeze(0))      # (1, 1, H, W) heatmap
```

### t-SNE Embedding 시각화

```
  고차원 특징(512차원) → 2D 시각화

  잘 학습된 경우:                잘못 학습된 경우:
  ● ● ●  (airplane)             ● ▲ ■ ● ▲ (뒤섞임)
      ▲ ▲  (bird)               ■ ● ▲ ■ ●
  ■ ■ ■   (car)                → 클래스 분리 안됨
  → 클래스가 명확히 분리됨
```

```python
from sklearn.manifold import TSNE

# avgpool feature 추출 (Hook)
feats, labels = [], []
def hook(m, i, o):
    feats.append(torch.flatten(o, 1).detach().cpu())

model.avgpool.register_forward_hook(hook)
with torch.no_grad():
    for x, y in test_loader:
        _ = model(x.to(device))
        labels.append(y)

X      = torch.cat(feats).numpy()             # (N, 512)
y_all  = torch.cat(labels).numpy()

X_2d   = TSNE(n_components=2, perplexity=30).fit_transform(X)

plt.scatter(X_2d[:,0], X_2d[:,1], c=y_all, s=6, alpha=0.6)
plt.title('t-SNE of penultimate features')
plt.show()
```

### 전체 학습 파이프라인

```
  ┌────────────────────────────────────────────────────────────────┐
  │  CIFAR-10 데이터셋                                            │
  │  10 classes × 6,000장 = 60,000장 (32×32×3 RGB)               │
  └──────────────────────────┬─────────────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────────────┐
  │  Data Augmentation (train only)                               │
  │  RandomCrop(32, pad=4) → RandomHorizontalFlip → Normalize     │
  └──────────────────────────┬─────────────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────────────┐
  │  ResNet18 (CIFAR 개조)                                        │
  │  3×32×32 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4        │
  │          → AvgPool(1×1) → FC(512→10)                          │
  └──────────────────────────┬─────────────────────────────────────┘
                             │ logits
  ┌──────────────────────────▼─────────────────────────────────────┐
  │  CrossEntropyLoss (= Softmax + NLL)                           │
  └──────────────────────────┬─────────────────────────────────────┘
                             │ loss
  ┌──────────────────────────▼─────────────────────────────────────┐
  │  Backward + SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)      │
  │  CosineAnnealingLR 스케줄러 + AMP(FP16 혼합 정밀도)           │
  └──────────────────────────┬─────────────────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────────────────┐
  │  Best Checkpoint 저장 (val_acc 기준) → 추론/시각화            │
  └────────────────────────────────────────────────────────────────┘
```

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()   # Dropout 비활성화, BN Running Stats 고정
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        loss    = criterion(logits, labels)
        preds   = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_acc  += (preds == labels).float().sum().item()
        n += images.size(0)
    return total_loss / n, total_acc / n
```

---

*정리: SANGGU | AI Specialist 양성과정 Ch.02 DNN & CNN*
