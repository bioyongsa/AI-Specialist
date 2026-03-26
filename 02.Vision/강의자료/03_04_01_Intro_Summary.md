# AI Specialist 양성과정 — Introduction 요약

---

## 1. 과정 개요 (Course Overview)

| 일차 | 구분 | 세부 내용 | 실습 |
|------|------|-----------|------|
| 1일차 오전 | Basic Architecture | DNN, CNN and variants | ResNet |
| 1일차 오후 | Attention | Transformer, Vision Transformer and variants | ViT |
| 2일차 오전 | Applications | Detection, Tracking | DETR |
| 2일차 오후 | Generative Models | GAN, Diffusion, Latent Diffusion | DDPM, Stable Diffusion |

- **강의 자료**: 영어, 핵심 논문 심층 분석, 수식은 개념 중심
- **실습**: Jupyter Notebook (PyTorch), 6개 논문 중 4개 구현
- **평가**: 실습 코드 기반

---

## 2. Computer Vision의 발전사

### 2-1. 초기 시대 (1960 ~ 2010) — Feature Engineering Era

| 단계 | 주요 기술 |
|------|-----------|
| Early Attempts | Sobel Filter, Canny Edge Detector, Optical Flow |
| Pattern Recognition & Feature Engineering | SIFT, HOG |
| ML-based Vision | SVM, Adaboost |

> 핵심: 사람이 직접 특징(feature)을 설계(handcrafted)

---

### 2-2. 딥러닝 혁명 (2011 ~ 2020) — DNN / CNN Paradigm Shift

**ImageNet Breakthrough**
- **AlexNet**: CNN 부활, ReLU / Dropout / Max Pooling / Multi-GPU
- **VGGNet**: 3×3 필터 중첩
- **ResNet**: Skip Connection (잔차 학습)

**Vision Applications**
- **Detection**: R-CNN → Fast R-CNN → Faster R-CNN → YOLO, SSD
- **Segmentation**: FCN, U-Net, Mask R-CNN, DeepLab
- **Generative**: Auto Encoder, VAE, GAN

---

### 2-3. Modern AI 통합 (2021 ~ ) — Multi-modal Vision

| 분류 | 모델 |
|------|------|
| Transformer-based Vision | ViT (Google), DETR (Meta), SAM (Meta) |
| Vision-Language Integration | CLIP (OpenAI), ALIGN (Google), BLIP (Salesforce), Flamingo (DeepMind) |
| Generative & Interactive | DALL-E, Stable Diffusion, GPT, Gemini |

---

## 3. Vision 모델의 학습 방식

| 구분 | Supervised | Semi-Supervised | Self-Supervised |
|------|------------|-----------------|-----------------|
| 정의 | 레이블 데이터로 학습 | 소량 레이블 + 대량 비레이블 | 레이블 없이 데이터 자체에서 학습 |
| 목표 | Ground truth 예측 | 제한된 레이블로 성능 향상 | 유용한 표현 학습 |
| 레이블 비용 | 매우 높음 | 중간 | 없음 |
| 대표 방법 | Classification, Detection, Segmentation | Pseudo-labeling, Consistency Regularization | Pretraining → Fine-tuning (BERT, GPT) |

---

## 4. Human Vision vs. Computer Vision

| 기능 단계 | 인간 시각 시스템 | 컴퓨터 비전 시스템 |
|-----------|-----------------|-------------------|
| 데이터 수집 | 망막의 간상/추상세포 | 카메라 센서 (RGB 픽셀 배열) |
| 전처리 | LGN 필터링 | Normalization, Augmentation |
| 저수준 특징 | V1 — 엣지, 방향, 움직임 | CNN Conv Layer (엣지, 텍스처) |
| 중간 수준 | V4, IT — 형태, 색상 | 깊은 CNN / Transformer Layer |
| 고수준 이해 | 전두엽 피질 — 기억·감정 연결 | FC Layer / ViT (의미 분류) |
| 결정·행동 | 운동 피질 | 자율주행, 로보틱스, VLM |
| 학습·적응 | 경험과 피드백 | 대규모 레이블 데이터 / 강화 신호 |

---

## 5. 배경 지식 (Background Knowledge)

### 5-1. 카메라 이미지 센서

```
빛 → (1) 렌즈 → (2) 마이크로렌즈 → (3) Bayer 컬러 필터 → (4) 포토다이오드 → 전기 신호
```

- **Bayer Filter**: `Y = 0.299R + 0.587G + 0.114B`
- **ISP Pipeline**: Raw → 블랙 보정 → 렌즈 보정 → Demosaicing → 노이즈 제거 → White Balance → 감마 보정

---

### 5-2. 벡터와 행렬

```python
import numpy as np

# 벡터
a = np.array([1, 2, 3])                   # shape: (3,)

# 행렬
C = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])  # shape: (4,3)

# 단위행렬
I = np.eye(3)

# 행렬 곱셈  R^(m×n) = R^(m×k) @ R^(k×n)
result = C @ a                             # (4,3) @ (3,) = (4,)

# 전치
print((C @ a.reshape(-1,1)).T)            # (Ca)^T = a^T C^T

# 내적 (Inner Product)
dot = np.dot(a, a)

# 코사인 유사도 (Cosine Similarity)
cos_sim = np.dot(a, a) / (np.linalg.norm(a) * np.linalg.norm(a))

# 벡터 노름
l1 = np.linalg.norm(a, ord=1)   # L1
l2 = np.linalg.norm(a, ord=2)   # L2 (유클리드)
linf = np.linalg.norm(a, ord=np.inf)  # L∞
```

> **Inner Product vs Cosine Similarity**
> - Inner Product: 크기 + 방향, 범위 무한
> - Cosine Similarity: 방향만, 범위 [-1, 1]

---

### 5-3. 미분 (Math)

```python
import sympy as sp

x, y = sp.symbols('x y')

# 일변수 함수 도함수
f = x**2 + 3*x + 1
df = sp.diff(f, x)
print(df)   # 2x + 3

# 다변수 함수 편미분
g = x**2 + x*y + y**2
dg_dx = sp.diff(g, x)   # ∂g/∂x
dg_dy = sp.diff(g, y)   # ∂g/∂y

# 그래디언트 (∇f) — 가장 가파른 증가 방향
gradient = [dg_dx, dg_dy]
print(gradient)
```

---

### 5-4. 확률과 통계

```python
import numpy as np
from scipy import stats

# --- PMF: 이산 분포 ---
p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
x = np.arange(1, 6)
E_X = np.sum(x * p)   # E[X] = Σ x·p(x)

# --- PDF: 연속 분포 (정규분포 예시) ---
rv = stats.norm(loc=0, scale=1)
E_X_cont = rv.mean()

# --- 엔트로피 (Entropy) ---
from scipy.stats import entropy
H = entropy(p, base=2)   # -Σ p(x) log2 p(x)

# --- 크로스 엔트로피 (Cross Entropy) ---
q = np.array([0.15, 0.25, 0.25, 0.2, 0.15])
H_pq = -np.sum(p * np.log2(q))   # H(p,q)

# --- KL Divergence ---
KL = entropy(p, q, base=2)   # D_KL(p||q) = H(p,q) - H(p)

# --- 베이즈 정리 ---
# p(θ|x) = p(x|θ)·p(θ) / p(x)
# Posterior ∝ Likelihood × Prior

# --- MLE ---
# θ_MLE = argmax_θ p(x|θ)

# --- 코사인 유사도 (벡터 연산) ---
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**ELBO (Evidence Lower Bound)**
$$\log p(x) \geq \mathcal{L}(q) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$
- reconstruction term + KL divergence로 구성
- VAE 학습의 핵심 목적함수

---

### 5-5. 학습 사이클 (Training Cycles)

```python
# 예시: 총 1000개 샘플, batch_size=32
total_samples = 1000
batch_size = 32
epochs = 10

iterations_per_epoch = total_samples // batch_size  # ≈ 31
total_iterations = iterations_per_epoch * epochs

print(f"Epoch당 Iteration 수: {iterations_per_epoch}")
print(f"전체 Iteration 수: {total_iterations}")
```

| 개념 | 설명 |
|------|------|
| **Epoch** | 전체 훈련 데이터를 한 번 완전히 통과 |
| **Batch Size** | 1회 forward/backward에 사용하는 샘플 수 |
| **Iteration (Step)** | 배치 1개 처리 후 가중치 업데이트 1회 |

- 큰 배치: 안정적 gradient, VRAM 많이 사용
- 작은 배치: 노이즈 → local minima 탈출에 유리

---

## 6. 주요 이미지 데이터셋

### Classification

| Dataset | Classes | Images | 크기 | 용도 |
|---------|---------|--------|------|------|
| MNIST | 10 | 70K | 28×28 | 기본 벤치마크 |
| CIFAR-10 | 10 | 60K | 32×32 | CNN/ViT baseline |
| ImageNet-1K | 1,000 | 1.28M | 224×224 | 표준 벤치마크 |
| ImageNet-21K | 21K | 14M | 224×224 | 대규모 사전학습 |

### Vision-Language (대규모 사전학습)

| Dataset | 규모 | 특징 |
|---------|------|------|
| LAION-400M | 400M 쌍 | CLIP-style 학습 |
| LAION-5B | 5B 쌍 | Foundation Model 수준 |
| YFCC100M | 100M | Flickr 기반, 약한 레이블 |

### Detection / Segmentation

| Dataset | Task | Classes | Images |
|---------|------|---------|--------|
| MS COCO | Detection/Seg/Keypoints | 80 | 330K |
| Cityscapes | Segmentation | 19 | 25K |
| ADE20K | Segmentation | 150 | 25K |
| PASCAL VOC | Detection/Seg | 20 | ~11K |

---

*요약 작성일: 2026-03-25*
