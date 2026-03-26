# Vision Transformer (ViT) + CIFAR-10 코드 설명

> **참고 자료:** `03_04_03_Attention.pdf` (AI Specialist 양성과정 — Attention 강의자료)  
> **대상 독자:** Python 및 Vision LLM 입문자  
> **다루는 범위:** 노트북 처음 ~ `8) 오분류(실패) 샘플만 시각화하기` 까지

---

## 목차

1. [ViT란 무엇인가?](#1-vit란-무엇인가)
2. [0) Setup — 환경 준비](#2-0-setup--환경-준비)
3. [1) 데이터 로딩 & 전처리](#3-1-데이터-로딩--전처리)
4. [2) ViT 모델 구성 개요 — 핵심 블록 구현](#4-2-vit-모델-구성-개요--핵심-블록-구현)
5. [2-1) ViTConfig + ViT 모델 본체](#5-2-1-vitconfig--vit-모델-본체)
6. [2-2) 하이퍼파라미터 설정](#6-2-2-하이퍼파라미터-설정)
7. [3) 모델 구조 요약 & 파라미터 확인](#7-3-모델-구조-요약--파라미터-확인)
8. [4) 체크포인트 저장/이어학습](#8-4-체크포인트-저장이어학습)
9. [5) Training — 학습 루프](#9-5-training--학습-루프)
10. [6) Attention 시각화](#10-6-attention-시각화)
11. [7) Confusion Matrix로 오분류 패턴 확인](#11-7-confusion-matrix로-오분류-패턴-확인)
12. [8) 오분류 샘플 시각화](#12-8-오분류-샘플-시각화)

---

## 1. ViT란 무엇인가?

> **강의자료 참고: PDF p.9 — "Vision Transformer"**

CNN(합성곱 신경망)은 이미지의 **국소적(local) 패턴**만 학습합니다.  
반면 ViT는 이미지를 **패치(patch, 작은 조각)** 단위로 잘라 NLP(자연어 처리)의 Transformer에 그대로 넣습니다.

핵심 질문: **"Self-Attention이 Convolution을 대체할 수 있는가?"** (PDF p.9)

```
[ 이미지 ] → [ 패치로 자르기 ] → [ 토큰 시퀀스 ] → [ Transformer Encoder ] → [ 분류 ]
```

CNN과 ViT의 차이를 한 줄로 정리하면:

| 항목 | CNN | ViT |
|------|-----|-----|
| 입력 단위 | 픽셀 단위 합성곱 | 패치(예: 4×4) 토큰 |
| 수용 영역 | 국소(local) | 전역(global) Self-Attention |
| 장점 | 작은 데이터에서도 강함 | 대규모 데이터에서 뛰어남 |

---

## 2. 0) Setup — 환경 준비

> **강의자료 참고: PDF p.9 (ViT 도입 배경)**

### 어떤 코드인가?

```python
import importlib.util
import subprocess
import sys

def ensure_package(pkg_name: str, import_name=None):
    name = import_name or pkg_name
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])
    else:
        print(f"[ok] {pkg_name} 이미 설치되어 있습니다.")

packages = [
    ("einops", "einops"),
    ("torchinfo", "torchinfo")
]
for pkg, imp in packages:
    ensure_package(pkg, imp)
```

### 초보자를 위한 설명

- **`importlib.util.find_spec(name)`**: 이 패키지가 이미 설치됐는지 검사합니다.
- **`subprocess.check_call([sys.executable, "-m", "pip", "install", ...])`**: 설치 안 된 경우 자동으로 `pip install`을 실행합니다.

### 왜 이 패키지들이 필요한가?

| 패키지 | 역할 |
|--------|------|
| `einops` | 텐서 형태 변환을 직관적으로 작성 (예: `"b c h w -> b (h w) c"`) |
| `torchinfo` | 모델 구조와 파라미터 수를 테이블 형식으로 출력 |

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

**GPU(`cuda`)가 있으면 GPU를, 없으면 CPU를 씁니다.**  
ViT는 행렬 연산이 많아 GPU가 있으면 훨씬 빠릅니다.

---

## 3. 1) 데이터 로딩 & 전처리

> **강의자료 참고: PDF p.12 — "Patch embedding"**

### CIFAR-10이란?

10가지 클래스(비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)의 32×32 컬러 이미지 데이터셋입니다.

### 핵심 코드 설명

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # R, G, B 채널별 평균
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)  # R, G, B 채널별 표준편차

train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),   # 데이터 증강: 랜덤 크롭
    T.RandomHorizontalFlip(),      # 데이터 증강: 좌우 반전
    T.ToTensor(),                  # PIL 이미지 → (C, H, W) 텐서
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 정규화
])

test_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 테스트는 증강 없이 정규화만
])
```

**왜 정규화(Normalize)를 할까?**  
픽셀 값은 0~255인데 이를 평균 0, 표준편차 1 근처로 만들어야 학습이 안정됩니다.

**데이터 증강(Augmentation)이란?**  
학습 데이터를 크롭, 반전 등으로 인위적으로 늘려 모델의 과적합을 방지합니다.

```python
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, ...)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, ...)
```

**DataLoader**: 데이터를 `batch_size`(256장)씩 묶어 모델에 한 번에 전달합니다.

### 텐서 shape 이해하기

```
이미지 한 장: (C, H, W) = (3, 32, 32)  ← 채널 3개(RGB), 높이 32, 너비 32
배치 단위:    (B, C, H, W) = (256, 3, 32, 32)  ← 256장 묶음
```

**PDF p.12 연결:** 이 32×32 이미지가 패치로 분할되어 `x ∈ ℝ^{H×W×C}` 형태로 처리됩니다.

---

## 4. 2) ViT 모델 구성 개요 — 핵심 블록 구현

> **강의자료 참고:**
> - **PDF p.11 — "Architecture (2/2)"** (수식: MSA, SA, LN 등)
> - **PDF p.15 — "Query / Key / Value"**
> - **PDF p.16 — "Attention among patches"**
> - **PDF p.17 — "Self-attention"**
> - **PDF p.18 — "Multihead self-attention"**

### 4-1. PreNorm (Pre-LayerNorm)

```python
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 토큰 차원 기준 정규화
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 정규화 → 함수 실행
```

**PDF p.11 연결:** 수식 `z'ℓ = MSA(LN(zℓ₋₁)) + zℓ₋₁` 에서 `LN()`이 바로 이 PreNorm입니다.

**왜 먼저 정규화할까?**  
학습 초기 값이 너무 크거나 작아지는 것을 방지해 학습을 안정시킵니다.

```
입력 x → LayerNorm → Attention 또는 FFN → 출력
```

### 4-2. FeedForward (FFN, MLP 블록)

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 차원 확장
            nn.ReLU(),                   # 비선형 활성화
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # 원래 차원으로 복원
            nn.Dropout(dropout)
        )
```

**PDF p.11 연결:** 수식 `zℓ = MLP(LN(z'ℓ)) + z'ℓ` 의 MLP 부분입니다.

**직관적 설명:** Attention이 "어느 패치에 집중할지"를 결정한다면, FFN은 "그 정보로 무엇을 표현할지"를 학습합니다.

```
dim(64) → hidden_dim(128) → dim(64)  ← 확장했다 복원하는 구조
```

### 4-3. Attention (Multi-Head Self-Attention)

> **PDF p.5 — "Encoder (2/2)"**: `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 전체 head 차원

        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/√d_k (스케일링)

        self.attend = nn.Softmax(dim=-1)  # attention score → 확률
        self.last_attn = None             # 시각화용 저장

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # Q, K, V 한 번에 생성
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
```

**Q, K, V란?**

| 역할 | 비유 |
|------|------|
| Query (Q) | "나는 어떤 정보를 찾고 있나?" |
| Key (K) | "나는 어떤 정보를 갖고 있나?" |
| Value (V) | "실제로 전달할 정보" |

**PDF p.4 연결:** Word Embedding → WQ, WK, WV 행렬로 Q, K, V를 생성하는 구조와 동일합니다.

```python
def forward(self, x):
    b, n, _, h = *x.shape, self.heads
    qkv = self.to_qkv(x).chunk(3, dim=-1)        # Q, K, V로 분리
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

    dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # QKᵀ / √d
    attn = self.attend(dots)  # softmax → attention 가중치
    self.last_attn = attn.detach()  # 시각화용 저장

    out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 가중합
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)
```

**차원 흐름 (dimension trace):**

```
입력  x:      (B, N+1, dim)       = (1, 65, 64)
Q, K, V:      (B, heads, N+1, d_h) = (1, 4, 65, 64)
dots (QKᵀ):   (B, heads, N+1, N+1) = (1, 4, 65, 65)  ← 패치 간 유사도 행렬
attn:         (1, 4, 65, 65)        ← softmax 후 attention 가중치
out:          (1, 65, 256) → (1, 65, 64)  ← head 합치고 투영
```

**PDF p.16 연결:** `A = softmax(qkᵀ / √D_h)` 이 코드에서 `dots * self.scale` → `self.attend(dots)`로 구현됩니다.

### 4-4. Transformer (Encoder 블록 쌓기)

```python
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):   # depth(6)만큼 블록 반복 생성
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # Residual(Skip) Connection
            x = ff(x) + x     # Residual(Skip) Connection
        return x
```

**Residual Connection(잔차 연결)이란?**  
`x = attn(x) + x` → Attention의 출력에 원래 입력을 더합니다.  
학습 도중 기울기 소실(Gradient Vanishing) 문제를 방지하는 핵심 기법입니다.

**PDF p.11 연결:**
```
z'ℓ = MSA(LN(zℓ₋₁)) + zℓ₋₁   ← attn(x) + x
zℓ  = MLP(LN(z'ℓ))  + z'ℓ   ← ff(x) + x
```

---

## 5. 2-1) ViTConfig + ViT 모델 본체

> **강의자료 참고:**
> - **PDF p.10 — "Architecture (1/2)"** (ViT 전체 구조도)
> - **PDF p.12 — "Patch embedding"**
> - **PDF p.13 — "Linear projection"**
> - **PDF p.14 — "Position embedding"**
> - **PDF p.19 — "Classification"**

### ViTConfig — 설정값 한 곳에 모으기

```python
@dataclass
class ViTConfig:
    image_size: Union[int, Tuple[int, int]]   # 입력 이미지 크기
    patch_size: Union[int, Tuple[int, int]]   # 패치 크기
    channels: int = 3          # RGB = 3채널
    dim: int = 64              # 토큰 임베딩 차원 (= Transformer hidden size)
    depth: int = 6             # Encoder 블록 반복 횟수
    heads: int = 4             # Multi-Head의 head 수
    dim_head: int = 64         # 각 head의 Q/K/V 차원
    mlp_dim: int = 128         # FFN 중간 차원
    num_classes: int = 10      # 분류 클래스 수
    pool: str = "cls"          # "cls": CLS 토큰 사용
    dropout: float = 0.0
    emb_dropout: float = 0.0
```

**PDF p.20 연결:** ViT-Base(B/16) 기준으로 dim=768, heads=12, depth=12인 반면 이 실습에서는 CIFAR-10(32×32)에 맞게 소형 모델을 씁니다.

### ViT.`__init__` — 모델 조립

```python
class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        # 1) 패치 개수 계산
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # CIFAR-10: (32/4) × (32/4) = 8 × 8 = 64 패치
        patch_dim = cfg.channels * patch_height * patch_width
        # 4×4×3 = 48 (한 패치를 펼친 크기)
```

**PDF p.12 연결:** `N = HW/P²` 공식 그대로입니다.

```python
        # 2) Patch Embedding: (B,C,H,W) → (B, N, dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, cfg.dim),  # 48 → 64
        )
```

**PDF p.13 연결:** 이것이 "Linear Projection of Flattened Patches"입니다.  
패치를 1D로 펼치고(`p1 p2 c`), Linear로 `dim` 차원으로 투영합니다.

```
이미지: (1, 3, 32, 32)
  → 패치로 분할 및 펼침: (1, 64, 48)  [64개 패치, 각 48차원]
  → Linear 투영: (1, 64, 64)           [각 패치 → 64차원 토큰]
```

```python
        # 3) CLS 토큰 + 위치 임베딩
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.dim))
```

**PDF p.14 연결:** `z₀ = [x_class; x¹_p E; x²_p E; ... ; xᴺ_p E] + E_pos`

- `cls_token`: 분류 결과를 모으는 특별 토큰 (index 0)
- `pos_embedding`: 각 패치의 위치 정보 (1D 학습 가능 파라미터)

### ViT.`forward` — 데이터 흐름

```python
    def forward(self, img):
        # A) 패치 임베딩
        x = self.to_patch_embedding(img)    # (B, 64, 64)

        # B) CLS 토큰 붙이기
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 65, 64)  ← 64+1=65

        # C) 위치 임베딩 더하기
        x = x + self.pos_embedding          # (B, 65, 64)

        # D) Transformer Encoder 통과
        x = self.transformer(x)             # (B, 65, 64)

        # E) CLS 토큰만 추출
        x = x[:, 0]                         # (B, 64)

        # F) 분류 헤드
        return self.mlp_head(x)             # (B, 10)
```

**전체 dimension trace:**

```
입력:            (B, 3, 32, 32)
패치 임베딩:     (B, 64, 64)
CLS 토큰 추가:   (B, 65, 64)     ← N+1 = 64+1
위치 임베딩:     (B, 65, 64)
Transformer:     (B, 65, 64)
CLS 추출:        (B, 64)
MLP Head:        (B, 10)         ← 10개 클래스 logit
```

**PDF p.19 연결:** `y = LN(z⁰_L)` — L번째 레이어 출력의 CLS 토큰(0번째)을 LayerNorm 후 MLP Head로 분류합니다.

---

## 6. 2-2) 하이퍼파라미터 설정

```python
vit_cfg = ViTConfig(
    image_size=32,    # CIFAR-10: 32×32
    patch_size=4,     # 4×4 패치 → (32/4)² = 64개 패치 토큰
    channels=3,
    dim=64,           # 토큰 임베딩 차원
    depth=6,          # Encoder 블록 6개 반복
    heads=4,          # MHSA head 4개
    dim_head=64,
    mlp_dim=128,
    num_classes=10,
    pool="cls",
)

model = ViT(vit_cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
```

**PDF p.20 연결:** ViT-Tiny(Ti/16)는 heads=3, depth=12로 훨씬 크지만, 이 실습은 소형(dim=64, depth=6)으로 빠른 실험에 적합합니다.

**patch_size가 작아지면?**

```
patch_size=4  → 패치 64개 → attention map: 65×65
patch_size=2  → 패치 256개 → attention map: 257×257 (약 15배 연산량 증가!)
```

**PDF p.34 연결:** ViT의 연산 복잡도는 O((HW)²)로, 패치 수의 제곱에 비례합니다.

---

## 7. 3) 모델 구조 요약 & 파라미터 확인

```python
from torchinfo import summary

_summary = summary(model, input_size=(1, 3, 32, 32), device=str(device))
print(_summary)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print("Trainable parameters:", count_parameters(model))
```

**출력 예시 (실제 숫자는 환경마다 다를 수 있음):**

```
================================================================
Layer (type)            Output Shape          Param #
================================================================
...
Transformer Encoder ×6
  PreNorm + Attention
  PreNorm + FeedForward
...
================================================================
Total trainable params: ~XXX,XXX
```

**파라미터 수 공식 이해:**

- CLS 토큰: `1 × 64 = 64`
- 위치 임베딩: `65 × 64 = 4,160`
- Attention (레이어당): `to_qkv: 64 × (64×4×3) = 49,152` 등
- MLP Head: `64 × 10 = 640`

**PDF p.37 연결:** ViT-Tiny는 약 5.7M 파라미터인데, 이 실습 모델은 훨씬 소형입니다.

---

## 8. 4) 체크포인트 저장/이어학습

```python
ckpt_path = "vit_cifar10.pth"
start_epoch = 0
last_acc = 0.0

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])         # 모델 가중치 복원
    optimizer.load_state_dict(ckpt["optimizer"]) # 옵티마이저 상태 복원
    start_epoch = int(ckpt.get("epoch", 0))
    last_acc = float(ckpt.get("test_acc", 0.0))
    extra_epochs = 10  # 이어서 추가 10 epoch
else:
    extra_epochs = 20  # 처음부터 20 epoch

best_acc = last_acc
target_epoch = start_epoch + extra_epochs
```

**왜 체크포인트를 쓸까?**  
학습이 중간에 끊겨도 처음부터 다시 할 필요 없이 이어서 학습할 수 있습니다.

**저장되는 내용:**
- `model`: 학습된 가중치 (weights, biases)
- `optimizer`: Adam 옵티마이저의 내부 상태 (momentum 등)
- `epoch`: 마지막으로 학습한 에포크 번호
- `test_acc`: 당시 테스트 정확도

---

## 9. 5) Training — 학습 루프

> 이 섹션은 셀 17~22에 걸쳐 train_epoch, evaluate 함수 정의와 실제 학습 루프로 구성됩니다.

### train_epoch 함수

```python
def train_epoch(model, optimizer, data_loader, loss_history):
    model.train()  # Dropout 활성화 모드

    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()                            # 기울기 초기화
        output = F.log_softmax(model(data), dim=1)       # 예측 (log 확률)
        loss = F.nll_loss(output, target)                # NLL Loss 계산
        loss.backward()                                  # 역전파
        optimizer.step()                                 # 가중치 업데이트
```

**학습 5단계:**
```
① zero_grad()     ← 이전 기울기 초기화 (안 하면 기울기가 누적됨!)
② forward()       ← 모델 통과 → 예측값 계산
③ loss 계산       ← 정답과 예측의 차이
④ backward()      ← 역전파로 기울기 계산
⑤ optimizer.step() ← 기울기로 가중치 업데이트
```

**`log_softmax` + `nll_loss` = `CrossEntropyLoss`**  
두 함수를 합친 것이 PyTorch의 `nn.CrossEntropyLoss()`와 동일합니다.

### evaluate 함수

```python
def evaluate(model, data_loader, loss_history):
    model.eval()   # Dropout 비활성화 모드

    with torch.no_grad():   # 기울기 계산 OFF (메모리/속도 절약)
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            _, pred = torch.max(output, dim=1)  # 가장 높은 확률의 클래스
            correct_samples += int(pred.eq(target).sum().item())

    acc = correct_samples / total_samples
    return avg_loss, acc
```

**`model.train()` vs `model.eval()` 차이:**

| | train() | eval() |
|--|---------|--------|
| Dropout | 활성화 (일부 노드 끔) | 비활성화 (전부 사용) |
| BatchNorm | 배치 통계 사용 | 저장된 통계 사용 |
| 용도 | 학습 시 | 평가/추론 시 |

### 학습 루프 (메인)

```python
for epoch in range(start_epoch + 1, target_epoch + 1):
    train_epoch(model, optimizer, train_loader, train_loss_history)
    test_loss, test_acc = evaluate(model, test_loader, test_loss_history)

    # 정확도 개선 시에만 체크포인트 저장 (best model 유지)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "test_acc": test_acc,
        }, ckpt_path)
```

**정확도가 개선된 경우에만 저장**하기 때문에 항상 가장 좋은 모델이 파일에 남습니다.

---

## 10. 6) Attention 시각화

> **강의자료 참고:**
> - **PDF p.16 — "Attention among patches"**
> - **PDF p.17 — "Self-attention"**
> - **PDF p.51 — "(Recap.) Self-attention map"** (DINO 강의 내 복습)

### 핵심 아이디어

ViT의 최종 분류는 **CLS 토큰**에서 나옵니다.  
CLS 토큰이 "어떤 패치를 얼마나 참고했는지"가 바로 attention map입니다.

```
CLS 토큰의 attention row: [a¹, a², ..., aᴺ]  ← 각 패치에 대한 가중치
```

**PDF p.16 연결:** 그림에서 CLS 토큰(index=0)의 첫 번째 행이 각 패치로의 attention입니다.

### 코드 흐름

```python
def _extract_last_attn(model, layer_idx):
    # Attention 모듈이 forward 시 저장해둔 last_attn을 꺼냄
    attn_module = model.transformer.layers[layer_idx][0].fn
    return attn_module.last_attn  # [B, heads, N+1, N+1]

def _cls_to_patch_map(attn_bhNN, head_idx=None):
    # CLS 토큰(index=0)이 각 패치(index=1..N)에 주는 attention 추출
    if head_idx is None:
        cls_attn = attn_bhNN.mean(dim=1)[0, 0, 1:]  # head 평균
    else:
        cls_attn = attn_bhNN[0, head_idx, 0, 1:]    # 특정 head
    return cls_attn  # shape: [N] = [64]
```

```python
def _upsample_to_image(attn_2d, out_h, out_w):
    # 8×8 attention grid를 32×32로 bilinear upsample
    attn_up = F.interpolate(attn_t, size=(out_h, out_w), mode="bilinear")
    return attn_up.cpu().numpy()  # (32, 32)
```

**attention map 흐름:**
```
[B, heads, N+1, N+1]          ← Attention 행렬 전체
  → CLS 행 추출 [N] = [64]   ← CLS→패치 attention
  → reshape (8, 8)            ← 8×8 패치 그리드
  → upsample (32, 32)         ← 원본 이미지 크기로 확장
  → 이미지에 overlay           ← 반투명하게 겹쳐 표시
```

### 시각화 함수 호출

```python
show_attention_overlays(
    model=model,
    test_set=test_set,
    vit_cfg=vit_cfg,
    device=device,
    n_samples=6,
    layer_idx=None,        # None → 마지막 레이어 자동 선택
    head_idx=None,         # None → head 평균 (MEAN)
    show_all_heads=True,   # 모든 head를 개별로 함께 표시
    sample_mode="correct", # 정답 샘플만 확인
)
```

**왜 마지막 레이어를 보는 것이 좋을까?**  
초반 레이어는 색상·엣지 같은 저수준 특징에 반응하지만,  
마지막 레이어는 "고양이", "자동차" 같은 고수준 의미에 반응합니다.

**PDF p.21 연결:** 오른쪽 그래프에서 레이어(depth)가 깊어질수록 attention 거리가 멀어지는 현상(= 전역 패턴 학습)과 일치합니다.

---

## 11. 7) Confusion Matrix로 오분류 패턴 확인

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.show()

plot_confusion_matrix(model, test_loader)
```

**Confusion Matrix란?**

```
행(row) = 실제 정답 클래스
열(col) = 모델의 예측 클래스

대각선(diagonal) = 올바른 예측
대각선 외 = 오분류 (클수록 많이 헷갈림)
```

**PDF p.9 연결:** CNN은 Locality(국소성) 때문에 비슷한 외형의 클래스를 헷갈립니다.  
ViT도 소형·소량 데이터에서는 비슷한 경향이 나타납니다.

**자주 헷갈리는 CIFAR-10 클래스 쌍:**

| 혼동 쌍 | 이유 |
|---------|------|
| cat ↔ dog | 비슷한 동물 형태 |
| automobile ↔ truck | 차량 유형 유사 |
| deer ↔ horse | 4족 보행 동물 |

---

## 12. 8) 오분류 샘플 시각화

```python
def plot_misclassified_samples(model, data_loader, max_samples=10):
    model.eval()
    mis_images, mis_labels, mis_preds = [], [], []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            logits = model(data)
            preds = logits.argmax(dim=1)         # 예측 클래스

            wrong_mask = preds.ne(target)        # 오답인 위치만 True
            if wrong_mask.any():
                wrong_data   = data[wrong_mask].detach().cpu()
                wrong_target = target[wrong_mask].detach().cpu()
                wrong_preds  = preds[wrong_mask].detach().cpu()

                for x, y, p in zip(wrong_data, wrong_target, wrong_preds):
                    mis_images.append(x)
                    mis_labels.append(int(y))
                    mis_preds.append(int(p))
                    if len(mis_images) >= max_samples:
                        break
            if len(mis_images) >= max_samples:
                break

    # 시각화: 2×5 그리드
    for i in range(n):
        img_vis = denormalize_cifar10(mis_images[i]).numpy().transpose(1, 2, 0)
        ax.imshow(img_vis)
        ax.set_title(f"GT: {class_names[mis_labels[i]]} / Pred: {class_names[mis_preds[i]]}")

plot_misclassified_samples(model, test_loader, max_samples=10)
```

### 핵심 코드 포인트

**`preds.ne(target)`**: `ne` = "Not Equal". 예측이 정답과 다른 위치에 `True`를 반환합니다.

**`logits.argmax(dim=1)`**: 10개 클래스 점수 중 가장 높은 인덱스 = 예측 클래스.

**`denormalize_cifar10()`**: 정규화 역변환. 모델에 넣기 위해 변환했던 픽셀 값을 원래 0~1 범위로 되돌립니다.

```python
# 정규화 역변환
x = img_chw * std + mean   # y = (y - μ) / σ 의 역산
x = x.clamp(0, 1)           # 0~1 범위 클리핑
```

### 오분류 샘플로 무엇을 알 수 있나?

**PDF p.22 연결:** ViT의 한계:
- 소규모 데이터셋에서 과적합 발생 가능
- Self-Attention은 O(N²) 연산이므로 패치 수가 늘면 부담 증가

오분류 샘플을 보면:
- 배경과 객체가 비슷한 색인 경우 (예: 하얀 고양이 → 흰 개)
- 객체가 너무 작거나 잘린 경우
- 학습 데이터에서 보기 드문 자세/각도

이를 바탕으로:
- 데이터 증강 강화 (RandomRotate, ColorJitter 등 추가)
- 모델 크기 조절 (depth, heads 증가)
- 학습률/에포크 조정

---

## 전체 흐름 요약

```
[CIFAR-10 이미지 32×32]
        ↓ (패치 분할: 4×4 → 64개 패치)
[Patch Embedding]  → (B, 64, 48) → Linear → (B, 64, 64)
        ↓ (CLS 토큰 추가 + 위치 임베딩)
[Token Sequence]   → (B, 65, 64)
        ↓ (Transformer Encoder × 6회)
   [PreNorm] → [Multi-Head Attention] → [Residual]
   [PreNorm] → [FeedForward(MLP)]     → [Residual]
        ↓ (CLS 토큰 추출)
[CLS Token]        → (B, 64)
        ↓ (LayerNorm + Linear)
[MLP Head]         → (B, 10)  ← 10개 클래스 확률
        ↓
[CrossEntropyLoss] → 역전파 → Adam 업데이트
```

---

## 강의자료(PDF) 연결 요약

| 노트북 섹션 | PDF 페이지 | 핵심 개념 |
|------------|-----------|----------|
| Setup / 데이터 | p.9 | ViT 등장 배경 (CNN 한계) |
| Patch Embedding | p.12 | `N = HW/P²`, `xp ∈ ℝ^{N×(P²C)}` |
| Linear Projection | p.13 | `xp E` (패치 → dim 차원 투영) |
| Position Embedding | p.14 | `z₀ = [x_class; x¹_pE; ...] + E_pos` |
| QKV / Attention | p.5, p.15, p.16 | `A = softmax(qkᵀ/√D_h)` |
| Self-Attention | p.17 | `SA(z) = Av` |
| Multi-Head | p.18 | `MSA(z) = [SA₁(z); ...; SAₖ(z)] U_msa` |
| Classification | p.19 | `y = LN(z⁰_L)` → MLP Head |
| 오분류 분석 | p.22 | ViT 한계 (소규모 데이터, O(N²)) |
