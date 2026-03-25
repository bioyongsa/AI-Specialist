# 04일차 (03/26) — Attention & Vision Transformer 정리

> AI Specialist 양성과정 | Chapter 03 Attention  
> 원본 논문: _Attention Is All You Need_ (NeurIPS 2017), _An Image Is Worth 16×16 Words_ (ICLR 2021) 외

---

## 목차

1. [Transformer 개요](#1-transformer-개요)
2. [Encoder — Word Embedding & Linear Projection](#2-encoder--word-embedding--linear-projection)
3. [Scaled Dot-Product Attention](#3-scaled-dot-product-attention)
4. [Attention의 의미 (직관적 이해)](#4-attention의-의미-직관적-이해)
5. [Vision Transformer (ViT)](#5-vision-transformer-vit)
6. [ViT Variants](#6-vit-variants)
   - 6.1 [DeiT](#61-deit--data-efficient-image-transformers)
   - 6.2 [Swin Transformer](#62-swin-transformer)
   - 6.3 [CLIP](#63-clip)
   - 6.4 [DINO](#64-dino)

---

## 1. Transformer 개요

```
Input Embedding
      │
Positional Encoding
      │
┌─────────────┐  ×N
│ Multi-Head  │
│  Attention  │
│  Add & Norm │
│ Feed Forward│
│  Add & Norm │
└─────────────┘ (Encoder)
      │
      ▼
  Decoder (유사 구조, Masked Attention 추가)
      │
  Linear → Softmax → Output Probabilities
```

| 구성 요소 | 역할 |
|---|---|
| Input Embedding | 토큰 → 연속 벡터 공간 매핑 |
| Positional Encoding | 위치 정보 주입 (순서 정보) |
| Multi-Head Attention | 다양한 관점의 병렬 Attention |
| Feed Forward | 위치별 독립적인 비선형 변환 |
| Add & Norm | Residual connection + Layer Normalization |

- **핵심 아이디어**: CNN의 지역적 수용 영역(local receptive field)과 달리, **모든 토큰이 모든 토큰에 Attention** (Global receptive field)
- 병렬 연산 가능 → RNN 대비 학습 속도 대폭 향상

---

## 2. Encoder — Word Embedding & Linear Projection

### 2-1. Word Embedding (단어 임베딩)

```
Word Space         Embedding Space
(n_vocabulary)  →  (d_embedding)

one-hot vector → dense vector (W_E 행렬 곱)
```

- 이산 토큰(정수 인덱스) → **밀집 연속 벡터**로 매핑
- `W_E ∈ R^(n_vocabulary × d_embedding)` — 학습 가능 파라미터

### 2-2. Q / K / V Linear Projection

```
Embedding vector (d_embedding, 1)
      │
      │  × W_Q, W_K, W_V  ← 학습 가능 가중치 (d_embedding × d_qkv)
      ▼
  Q, K, V  각각 (d_qkv, 1)
```

```python
import torch
import torch.nn as nn

d_embedding = 512
d_qkv = 64

# 학습 가능한 선형 투영 레이어
W_Q = nn.Linear(d_embedding, d_qkv, bias=False)
W_K = nn.Linear(d_embedding, d_qkv, bias=False)
W_V = nn.Linear(d_embedding, d_qkv, bias=False)

x = torch.randn(10, d_embedding)  # (seq_len=10, d_embedding)
Q = W_Q(x)  # (10, d_qkv)
K = W_K(x)  # (10, d_qkv)
V = W_V(x)  # (10, d_qkv)
```

| 벡터 | 역할 | 비유 |
|---|---|---|
| Q (Query) | 현재 토큰이 "무엇을 찾는가" | 검색어 |
| K (Key) | 각 토큰이 "무엇을 제공하는가" | 색인 키워드 |
| V (Value) | 실제로 가져올 정보 | 검색 결과 내용 |

---

## 3. Scaled Dot-Product Attention

### 핵심 수식

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 단계별 설명

```
1. Q × K^T  →  (N×N) Attention Score 행렬  [내적: 유사도 계산]
2. ÷ √d_k   →  스케일링  [내적값이 너무 커지는 것 방지]
3. Softmax  →  (N×N) Attention Weight 행렬  [확률 분포로 정규화]
4. × V      →  (N×d_v) Attention Output     [가중합]
```

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)

    # Step 1 & 2: 스케일된 내적
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, seq_len_q, seq_len_k)

    # Step 3: 마스킹 (옵션 — Decoder에서 미래 정보 차단)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax → Attention Weight
    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights: (batch, seq_len_q, seq_len_k)

    # Step 5: Value 가중합
    output = torch.matmul(attn_weights, V)
    # output: (batch, seq_len_q, d_v)

    return output, attn_weights

# 사용 예
batch, seq_len, d_k, d_v = 2, 5, 64, 64
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_v)
out, weights = scaled_dot_product_attention(Q, K, V)
print(out.shape)     # (2, 5, 64)
print(weights.shape) # (2, 5, 5) ← Attention Map
```

### ⚠️ 중요 포인트: 왜 √d_k 로 나누는가?

- d_k가 크면 Q·K^T 의 내적값이 매우 커져서 **Softmax가 극단적 확률**로 수렴
- 기울기 소실(Vanishing Gradient) 발생 가능
- √d_k 로 나눔으로써 **안정적인 확률 분포** 유지

### ⚠️ 중요 포인트: Attention Map 복잡도

- Attention Map 크기 = N×N (시퀀스 길이의 제곱)
- **O(N²)** 의 메모리 및 계산 복잡도 → 긴 시퀀스에서 병목

---

## 4. Attention의 의미 (직관적 이해)

### 4-1. Weighted Sum of Values

```
Value Space에 존재하는 여러 벡터들을 Attention Weight에 따라 가중합

V₁ (Young)    ×  a¹_N
V₂ (children) ×  a²_N   →  합산  →  V'_N (문맥이 반영된 새 벡터)
V₃ (often)    ×  a³_N
...
V_N (themselves) × a^N_N
```

### 4-2. Global Attention — 장거리 의존성 포착

예: `"Young children often solve problems by themselves."`

- `themselves` 토큰이 **Young, children 과 강한 Attention** → 주어-대명사 관계 포착
- CNN: 커널 크기 내 지역 패턴만 학습 (장거리 의존성 취약)
- Transformer: **모든 토큰 쌍의 관계**를 한 번에 계산

---

## 5. Vision Transformer (ViT)

> 논문: _An Image Is Worth 16×16 Words_, ICLR 2021

### 5-1. 등장 배경 (CNN의 한계)

| CNN의 Inductive Bias | 문제점 |
|---|---|
| Locality (지역성) | 장거리 의존성 학습 어려움 |
| Translation Equivariance | 데이터 규모 확장 시 성능 포화 |

→ **"Self-Attention이 Convolution을 대체할 수 있는가?"** 라는 질문에서 출발

### 5-2. ViT 전체 파이프라인

```
입력 이미지 (H×W×C)
      │
① Patch 분할: P×P 크기 패치로 분할, N = HW/P² 개
      │
② Flatten: 각 패치를 1D 벡터로 펼침 → xₚ ∈ R^(N × P²C)
      │
③ Linear Projection (E 행렬): P²C → D 차원
      │
④ [CLS] 토큰 앞에 concat + Position Embedding 더하기
      │  z₀ = [x_class ; x¹ₚE ; x²ₚE ; ··· ; xᴺₚE] + E_pos
      │
⑤ Transformer Encoder × L번
      │  z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁
      │  zₗ  = MLP(LN(z'ₗ))  + z'ₗ
      │
⑥ [CLS] 토큰 추출 → y = LN(z⁰_L)
      │
⑦ MLP Head → n_class 차원 logit → Softmax → 분류
```

### 5-3. Patch Embedding 상세

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # N = (224/16)² = 196
        # Conv2d로 Patch Flatten + Linear Projection을 동시에 수행
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, N)
        x = x.transpose(1, 2)     # (B, N, embed_dim)
        return x

# 예: ViT-Base 기준
patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
img = torch.randn(4, 3, 224, 224)
out = patch_embed(img)
print(out.shape)  # (4, 196, 768) ← (batch, num_patches, embed_dim)
```

- **패치 수**: N = HW/P² = (224×224)/16² = **196개**
- **패치당 픽셀 수**: P²C = 16×16×3 = 768

### 5-4. Position Embedding & CLS Token

```python
class ViTEmbedding(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768):
        super().__init__()
        # 학습 가능한 [CLS] 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 학습 가능한 위치 임베딩 (N+1 개 — CLS 포함)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        # x: (B, N, D)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)              # (B, N+1, D)
        x = x + self.pos_embed                      # 위치 정보 더하기
        return x
```

> **왜 1D Position Embedding?**  
> ViT는 1D 위치 임베딩만 사용하지만, 학습 과정에서 2D 공간 정보를 **암묵적으로 학습**한다 (논문 실험으로 확인)

### 5-5. Multi-Head Self-Attention (MSA) 전체 구현

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_h = embed_dim // num_heads       # 헤드당 차원: D_h = 768/12 = 64

        # Q, K, V 동시 투영 (U_qkv: D → 3D_h)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # MSA 출력 투영 (U_msa: k·D_h → D)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        # Step 1: Q, K, V 계산
        qkv = self.qkv(x)                                 # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, heads, N, D_h)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Step 2: Scaled Dot-Product Attention (각 헤드 병렬)
        scale = math.sqrt(self.d_h)
        A = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        # A: (B, heads, N, N)  ← Attention Map

        # Step 3: Value 가중합
        SA = A @ V                                        # (B, heads, N, D_h)

        # Step 4: 헤드 Concat + 선형 투영
        SA = SA.transpose(1, 2).reshape(B, N, D)         # (B, N, D)
        out = self.proj(SA)                               # (B, N, D)
        return out
```

### 5-6. Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp  = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN + Residual (ViT 논문 구조)
        x = x + self.attn(self.norm1(x))   # z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁
        x = x + self.mlp(self.norm2(x))    # zₗ  = MLP(LN(z'ₗ))  + z'ₗ
        return x
```

### 5-7. 분류 (CLS Token → MLP Head)

- [CLS] 토큰: 모든 패치 토큰에 Attention → **전역 정보 집약**
- `y = LN(z⁰_L)` → MLP Head → n_class 차원 logit

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.vit_embed   = ViTEmbedding(num_patches, embed_dim)
        self.blocks      = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)   # (B, N, D)
        x = self.vit_embed(x)     # (B, N+1, D)
        x = self.blocks(x)        # (B, N+1, D)
        x = self.norm(x)
        cls_token = x[:, 0]       # [CLS] 토큰만 추출
        return self.head(cls_token)

# ViT-Base/16 생성
model = ViT(img_size=224, patch_size=16, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12)
img = torch.randn(2, 3, 224, 224)
print(model(img).shape)  # (2, 1000)
```

### 5-8. 주요 ViT 모델별 하이퍼파라미터

| 모델 | Hidden Dim (D) | Heads (k) | Depth (L) | Params |
|---|---|---|---|---|
| ViT-Tiny (Ti/16) | 192 | 3 | 12 | 5.7M |
| ViT-Small (S/16) | 384 | 6 | 12 | 22M |
| ViT-Base (B/16) | 768 | 12 | 12 | **86M** |
| ViT-Large (L/16) | 1024 | 16 | 24 | 307M |
| ViT-Huge (H/14) | 1280 | 16 | 32 | 632M |

### 5-9. ViT 실험 결과 요약

| 특성 | 결과 |
|---|---|
| 소규모 데이터 | CNN(BiT)에 비해 성능 열위 (Inductive bias 부족) |
| 대규모 데이터 (JFT-300M) | CNN을 압도하는 성능 |
| 스케일링 | 데이터/모델 크기 증가에 따라 지속 향상 |
| Positional Encoding | 1D지만 2D 공간 정보 암묵적으로 학습 |

### 5-10. ViT 강점 & 한계

| 강점 | 한계 |
|---|---|
| Global pattern 효과적 학습 | 대용량 데이터 필요 (소규모 Overfitting) |
| 행렬 연산 → 하드웨어 친화적 | Self-Attention 복잡도 **O(N²)** |
| 대규모 태스크에서 뛰어난 성능 | Dense prediction 태스크에 적용 어려움 |

---

## 6. ViT Variants

```
ViT (2020)
 ├─ DeiT  (2020) — 데이터 효율적 학습 (Knowledge Distillation)
 ├─ Swin  (2021) — 계층적 구조 + Window Attention → O(N)
 ├─ CLIP  (2021) — 텍스트-이미지 대조 학습
 └─ DINO  (2021) — 레이블 없는 자기지도 학습
```

---

### 6.1 DeiT — Data-Efficient Image Transformers

> 논문: _Training data-efficient image transformers & distillation through attention_, 2020

#### 동기

- ViT 원 논문은 JFT-300M(3억 장) 필요 → 실제 환경에서 비현실적
- **ImageNet-1K(128만 장)만으로 ViT 수준의 성능** 달성 목표

#### 핵심: Knowledge Distillation

```
Teacher Model (강력한 CNN: RegNetY 등, Freeze)
      │
      │  소프트 레이블 / 하드 레이블
      ▼
Student Model (DeiT, Trainable)
  ├── [CLS] Token  →  Classification Head  →  Cross-Entropy(GT label)
  └── [Distill] Token → Distillation Head  →  KL-div or CE(Teacher output)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeiTLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5, hard_distill=True):
        """
        temperature: Soft distillation 온도
        alpha:       CE loss와 distillation loss 비율
        hard_distill: True → Hard distillation, False → Soft distillation
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.hard = hard_distill

    def forward(self, student_cls, student_dist, teacher_logits, labels):
        # Classification loss (GT label 기준)
        loss_cls = F.cross_entropy(student_cls, labels)

        if self.hard:
            # Hard distillation: Teacher의 argmax를 레이블로 사용
            teacher_labels = teacher_logits.argmax(dim=1)
            loss_dist = F.cross_entropy(student_dist, teacher_labels)
        else:
            # Soft distillation: KL divergence
            s = F.log_softmax(student_dist / self.T, dim=1)
            t = F.softmax(teacher_logits / self.T, dim=1)
            loss_dist = F.kl_div(s, t, reduction='batchmean') * (self.T ** 2)

        # 전체 손실: L = (1-α)·L_CE + α·L_dist
        return (1 - self.alpha) * loss_cls + self.alpha * loss_dist
```

#### Soft vs Hard Distillation

| 방식 | Teacher 레이블 | 손실 함수 | 수식 |
|---|---|---|---|
| **Soft** | Teacher의 소프트 확률 분포 | KL-divergence | L = (1-λ)·CE(Z_s, y) + λτ²·KL(softmax(Z_s/τ), softmax(Z_t/τ)) |
| **Hard** | Teacher의 argmax 예측 | Cross-Entropy | L = ½·CE(Z_s, y) + ½·CE(Z_s, y_t) |

#### 실험 결과 핵심

- CNN 교사 모델(RegNetY-16GF)이 Transformer 교사보다 높은 성능 → **Inductive bias 전달 효과**
- Hard Distillation + [CLS]+[Distill] 토큰 병합 사용 시 최고 성능

---

### 6.2 Swin Transformer

> 논문: _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_, 2021

#### 동기 — ViT의 두 가지 한계 극복

| ViT 한계 | Swin의 해결책 |
|---|---|
| Global Attention → O(N²) | Window 내 Local Attention → **O(N)** |
| Flat token 구조 (계층 없음) | 계층적(Hierarchical) 표현 → Detection/Segmentation 가능 |

#### 핵심 메커니즘: Shifted Window Attention

```
Layer l:   [Window-MSA]  — 고정 윈도우 내 Attention
Layer l+1: [SW-MSA]      — 윈도우를 M/2 만큼 Shift 후 Attention

효과: 인접 윈도우 간 정보 교환 (전역 컨텍스트 확보)
```

```
W-MSA (Layer l)         SW-MSA (Layer l+1)
┌───┬───┐               ┌─┬───┬─┐
│ 0 │ 1 │               │ │   │ │
├───┼───┤   →  shift →  ├─┼───┼─┤
│ 2 │ 3 │               │ │   │ │
└───┴───┘               └─┴───┴─┘
윈도우 내부만 Attention    이전 경계를 넘어 Attention
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """Feature map을 (window_size × window_size) 윈도우로 분할"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    """분할된 윈도우를 Feature map으로 복원"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.d_h = dim // num_heads
        self.scale = self.d_h ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape  # B_: 배치 × 윈도우 수
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:  # SW-MSA: 다른 윈도우 마스킹
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        return (self.proj((attn @ V).transpose(1, 2).reshape(B_, N, C)),
                attn)
```

#### Patch Merging (계층적 구조)

```
Stage 1: H/4  × W/4  × C    (기본 패치: 4×4 픽셀)
   ↓ Patch Merging
Stage 2: H/8  × W/8  × 2C
   ↓ Patch Merging
Stage 3: H/16 × W/16 × 4C
   ↓ Patch Merging
Stage 4: H/32 × W/32 × 8C
```

#### 복잡도 비교

| Attention | 수식 | 복잡도 | 의미 |
|---|---|---|---|
| **Global MSA (ViT)** | 4HWC² + 2(HW)²C | **O((HW)²)** | 패치 수 제곱에 비례 |
| **Window MSA (Swin)** | 4HWC² + 2M²HWC | **O(HW)** | 패치 수에 선형 비례 |

> M²: 윈도우 당 패치 수 (고정 상수)이므로 O(HW)로 단순화 가능

#### ViT vs Swin 비교

| 항목 | ViT | Swin Transformer |
|---|---|---|
| 구조 | Flat tokens | 계층적 Stage |
| Attention 범위 | Global | Window 기반 + Shift |
| 복잡도 | O((HW)²) | O(HW) |
| 주 적용 분야 | 분류(Classification) | **분류 + 검출 + 분할** |

---

### 6.3 CLIP

> 논문: _Learning Transferable Visual Models From Natural Language Supervision_, ICML 2021

#### 핵심 아이디어

- 인터넷의 **4억 쌍 (이미지, 텍스트) 데이터**로 학습
- 이미지 인코더 + 텍스트 인코더를 **동일한 임베딩 공간**에서 정렬
- **Zero-shot 분류** 가능 → 새로운 클래스에 재학습 불필요

#### 학습 방법: Contrastive Learning (InfoNCE Loss)

```
N개 (Image, Text) 쌍이 주어졌을 때:
- 대각선 (매칭 쌍): 유사도 최대화 ↑
- 비대각선 (불일치 쌍): 유사도 최소화 ↓

         T₁    T₂   ...   Tₙ
I₁  [  HIGH  low   ...   low  ]
I₂  [  low   HIGH  ...   low  ]
...
Iₙ  [  low   low   ...   HIGH ]
```

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z^i_{\text{img}}, z^i_{\text{text}})/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z^i_{\text{img}}, z^j_{\text{text}})/\tau)}$$

```python
import torch
import torch.nn.functional as F

def clip_contrastive_loss(image_features, text_features, temperature=0.07):
    """
    image_features: (N, d) — L2 정규화된 이미지 임베딩
    text_features:  (N, d) — L2 정규화된 텍스트 임베딩
    """
    # 코사인 유사도 행렬 (N×N)
    logits = (image_features @ text_features.T) / temperature

    # 정답 레이블: 대각선 (i번째 이미지 ↔ i번째 텍스트)
    labels = torch.arange(len(image_features), device=image_features.device)

    # 대칭 Loss: 이미지→텍스트 방향 + 텍스트→이미지 방향
    loss_i2t = F.cross_entropy(logits,   labels)  # 행 방향
    loss_t2i = F.cross_entropy(logits.T, labels)  # 열 방향
    return (loss_i2t + loss_t2i) / 2

# 실제 코드 (numpy pseudo-code 기준, 논문 Figure 3)
# I_e = l2_normalize(I_f @ W_i)    # 이미지 임베딩
# T_e = l2_normalize(T_f @ W_t)    # 텍스트 임베딩
# logits = I_e @ T_e.T * exp(t)    # 스케일된 코사인 유사도
```

#### Zero-shot 추론 방법

```python
# 1. 클래스 이름을 프롬프트로 감싸기 (Prompt Engineering)
classes = ["cat", "dog", "bird"]
prompts = [f"A photo of a {c}." for c in classes]
# → 단순 클래스 이름보다 ImageNet 정확도 1.3% 향상

# 2. 텍스트 임베딩 → 분류기 가중치로 사용
# 3. 입력 이미지 임베딩과 코사인 유사도 계산 → 가장 높은 클래스 선택
```

#### 실험 결과 하이라이트

| 비교 대상 | 결과 |
|---|---|
| Zero-shot vs 지도학습 ResNet-50 | 27개 중 **16개** 데이터셋에서 우위 |
| Zero-shot vs 4-shot Linear Probe | **동등한 성능** |
| Distribution Shift 강건성 | ImageNet-A에서 ResNet-101 대비 **+74.4%** |

> **강점**: 레이블 없이 인터넷 텍스트로 학습 → 강력한 Zero-shot 일반화  
> **약점**: 전문 도메인(위성, 의료 영상)에서 성능 저조

---

### 6.4 DINO

> 논문: _Emerging Properties in Self-Supervised Vision Transformers_, 2021

#### 핵심 아이디어

- **레이블 없이** ViT를 학습 → 객체 중심 표현 자동 학습
- [CLS] 토큰의 Self-Attention Map = **자동 객체 분할(Segmentation)**

#### Student-Teacher Framework (Self-Distillation)

```
입력 이미지 x
    │
    ├──→ 증강 x₁ (Global view ≥50%) ──→ Student g_θs ──→ p₁ (Softmax, 높은 τ=0.1)
    │                                                           │
    └──→ 증강 x₂ (Global view ≥50%) ──→ Teacher g_θt ──→ p₂ (Softmax, 낮은 τ=0.04)
              + Local views (≤50%)                        Centering 적용
                    ↓
          Loss = -p₂ log p₁  (Cross-Entropy)
                    ↓
          Student만 역전파 (Teacher: stop gradient)
                    ↓
          θ_teacher ← m·θ_teacher + (1-m)·θ_student  (EMA 업데이트)
```

#### 주요 컴포넌트 상세

| 컴포넌트 | 역할 | 상세 |
|---|---|---|
| **EMA** (m≈0.996) | Teacher 안정적 업데이트 | 느린 이동 평균 → 일관된 감독 신호 |
| **Centering** | Collapse 방지 | Teacher logit에서 배치 평균 차감 |
| **Temperature** | 분포 제어 | Teacher 낮음(Sharp) → 고신뢰 pseudo-label |
| **Multi-crop** | 로컬-글로벌 대응 학습 | 로컬 패치에서 전역 의미 예측 |
| **Stop Gradient** | Teacher 보호 | Teacher에 역전파 차단 |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        """
        student_out: list of (B, out_dim) — 모든 crop의 student 출력
        teacher_out: list of (B, out_dim) — global crop의 teacher 출력
        """
        s = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        t = [F.softmax((t - self.center) / self.teacher_temp, dim=-1)
             for t in teacher_out]

        total_loss = 0
        n_loss_terms = 0
        for t_i in t:
            for s_j in s:
                # Teacher-Student 쌍이 동일한 crop이면 건너뜀
                loss = -(t_i * s_j).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Centering 업데이트 (EMA)
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center = (self.center_momentum * self.center
                       + (1 - self.center_momentum) * batch_center)
        return total_loss


def update_teacher_ema(student, teacher, momentum=0.996):
    """Teacher 파라미터를 Student의 EMA로 업데이트"""
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
```

#### 왜 레이블 없이도 작동하는가?

```
Multi-crop: 같은 이미지의 다른 영역 → Student(로컬 패치)가 Teacher(전역 이미지)를 예측
→ "로컬 패치에서 전역 의미를 예측하는 방법" 학습

EMA Teacher: 일관된 감독 신호 제공 → Self-Distillation 안정화

결과: ViT의 [CLS] Attention Map이 객체 경계를 정확히 구분
```

---

## 전체 정리 비교표

| 모델 | 연도 | 핵심 기여 | 데이터 요구 | 복잡도 |
|---|---|---|---|---|
| **ViT** | 2020 | 이미지를 패치 시퀀스로 처리 | 매우 많음 (JFT-300M) | O(N²) |
| **DeiT** | 2020 | CNN 교사 모델 Distillation으로 데이터 효율화 | ImageNet-1K 충분 | O(N²) |
| **Swin** | 2021 | 계층적 구조 + Window Attention | ImageNet 수준 | **O(N)** |
| **CLIP** | 2021 | 텍스트-이미지 대조 학습, Zero-shot | 4억 쌍 (웹 크롤링) | — |
| **DINO** | 2021 | 레이블 없는 자기지도 학습 | 레이블 불필요 | O(N²) |

---

## 핵심 개념 요약

```
Attention의 본질
  Q·K^T → 유사도(관련성) 측정
  Softmax → 확률 가중치로 정규화
  × V → 관련 정보의 가중합

ViT의 본질
  이미지 = 단어 시퀀스로 취급
  패치 = 단어, 패치 임베딩 = 단어 임베딩

Swin의 핵심
  O(N²) → O(N): Window 내부만 Attention
  전역 정보: 매 레이어마다 Window Shift

CLIP의 핵심
  텍스트가 시각 개념의 보편적 감독 신호
  → Zero-shot 전이 학습 가능

DINO의 핵심
  Teacher = Student의 EMA (느린 자기 자신)
  로컬→전역 예측으로 객체 표현 자동 학습
```

---

*참고 논문*

- Vaswani et al., _Attention Is All You Need_, NeurIPS 2017  
- Dosovitskiy et al., _An Image Is Worth 16×16 Words_, ICLR 2021  
- Touvron et al., _Training data-efficient image transformers_, 2020  
- Liu et al., _Swin Transformer_, ICCV 2021  
- Radford et al., _Learning Transferable Visual Models From Natural Language Supervision_, ICML 2021  
- Caron et al., _Emerging Properties in Self-Supervised Vision Transformers_, ICCV 2021
