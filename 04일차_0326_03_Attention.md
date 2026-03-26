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

#### 슬라이드 그림으로 보는 Shape 흐름 (ViT-Base/16 기준)

> ViT-Base 하이퍼파라미터: `H=W=224, C=3, P=16, D=768, k=12 heads, D_h=64, L=12 layers`

```
이미지 (224×224×3)
      │
      ▼  [패치 분할 + Flatten]
xp  ∈ R^(196 × 768)          ← N=196개 패치, 패치당 P²C=768 픽셀
      │
      ▼  [Linear Projection: × E]
      R^(196 × 768)           ← D=768 임베딩 차원 (ViT-Base에서 P²C=D)
      │
      ▼  [[CLS] concat + Position Embedding 더하기]
z₀  ∈ R^(197 × 768)          ← N+1=197 (CLS 토큰 1개 추가)
      │
      ▼  [Transformer Encoder × 12]
z_L ∈ R^(197 × 768)          ← shape 유지 (Residual 구조)
      │
      ▼  [CLS 토큰 추출 + LayerNorm]
y   ∈ R^(1 × 768)             ← 0번째 토큰만
      │
      ▼  [MLP Head: 768 → 1000]
logit ∈ R^(1 × 1000)          ← 클래스 수만큼
```

Transformer Encoder **1개 블록** 내부 shape 변화:

```
입력 zₗ₋₁  ∈ R^(197 × 768)
      │
      ├─ LayerNorm
      │       ↓
      │   MSA(·)   ∈ R^(197 × 768)   ← shape 유지
      │       ↓
      └─ + (Residual)  →  z'ₗ ∈ R^(197 × 768)
              │
              ├─ LayerNorm
              │       ↓
              │   MLP(·)  ∈ R^(197 × 768)   ← shape 유지
              │       ↓
              └─ + (Residual)  →  zₗ ∈ R^(197 × 768)
```

Architecture (2/2) 슬라이드에 등장하는 핵심 수식별 shape:

| 수식 | 출력 shape |
|---|---|
| `z₀ = [x_class; x¹ₚE; ··· ; xᴺₚE] + E_pos` | `R^(197 × 768)` |
| `z'ₗ = MSA(LN(zₗ₋₁)) + zₗ₋₁` | `R^(197 × 768)` |
| `zₗ = MLP(LN(z'ₗ)) + z'ₗ` | `R^(197 × 768)` |
| `MSA(z) = [SA₁(z);···;SAₖ(z)] · U_msa` | `R^(197 × 768)` |
| `[q,k,v] = z · U_qkv` | `R^(197 × 192)` → q/k/v 각 64씩 분리 |
| `A = softmax(qk^T / √D_h)` | `R^(197 × 197)` |
| `SA(z) = A·v` | `R^(197 × 64)` |
| `y = LN(z⁰_L)` | `R^(1 × 768)` |

---

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

#### 슬라이드 그림 해설 — Patch Embedding

```
원본 이미지 (슬라이드 오른쪽)       Flatten된 패치 행렬 (슬라이드 왼쪽)
┌─────────────────────────┐        ┌──────────────────────┐
│  H=224                  │        │     P²C = 768         │
│       ┌───┬ ··· ┬───┐   │        │  ┌──────────────────┐ │
│  W=224│P×P│     │P×P│   │  →     │N │  xp (patch 1)    │ │
│       └───┴ ··· ┴───┘   │        │= │  xp (patch 2)    │ │
│  C=3 (RGB 채널)           │        │196│  ...            │ │
└─────────────────────────┘        │  └──────────────────┘ │
  x ∈ R^(224 × 224 × 3)            └──────────────────────┘
                                     xp ∈ R^(196 × 768)
```

**핵심 수치 계산:**

```
패치 수:      N  = HW / P²  = (224×224) / (16×16) = 196
패치당 픽셀:  P²C = 16×16×3 = 768
결과 행렬:    xp ∈ R^(N × P²C) = R^(196 × 768)
```

#### 슬라이드 그림 해설 — Linear Projection

패치 행렬 `xp`에 학습 가능한 행렬 `E`를 곱해 임베딩 공간으로 투영한다.

```
오른쪽(파란 박스): E (학습 가능 Projection 행렬)
┌──────────────────────────────┐
│         D = 768              │
│  ┌────────────────────────┐  │
│P²│                        │  │
│C │       E                │D_h│
│= │   (P²C × D 행렬)       │=64│
│768└────────────────────────┘  │
└──────────────────────────────┘
          E ∈ R^(768 × 768)

가운데(빨간 박스): xp (입력)         왼쪽(빨간 박스): xpE (출력)
┌────────────┐                      ┌────────────┐
│  P²C = 768 │                      │   D = 768  │
│ ┌────────┐ │                      │ ┌────────┐ │
│N│ patch1 │ │   × E (768×768)  →   │N│ x¹ₚE  │ │
│=│ patch2 │ │                      │=│ x²ₚE  │ │
│196  ...  │ │                      │196  ...  │ │
│ └────────┘ │                      │ └────────┘ │
└────────────┘                      └────────────┘
 xp ∈ R^(196 × 768)                 xpE ∈ R^(196 × 768)
```

**shape 변환:**

```
xp  ∈ R^(196 × 768)  ×  E ∈ R^(768 × 768)  =  xpE ∈ R^(196 × 768)
     ↑ N × P²C              ↑ 학습 가능              ↑ N × D
```

> ViT-Base에서는 `P²C = 768 = D` 이므로 행/열 수가 같지만,  
> 일반적으로는 `P²C ≠ D` 일 수 있다 (예: ViT-Ti는 D=192).

```python
import torch.nn as nn

D = 768
pixels_per_patch = 768   # P²C

E = nn.Linear(pixels_per_patch, D, bias=False)  # R^(768 → 768)
# xp: (batch, N, P²C) = (1, 196, 768)
xpE = E(xp)   # (1, 196, 768) = R^(N × D)
print(f"xpE shape: {xpE.shape}")
```

> **구현 tip**: `nn.Conv2d(3, D, kernel_size=P, stride=P)` 를 쓰면  
> Patch Flatten + Linear Projection 을 **한 번의 연산**으로 처리할 수 있다.

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

#### 슬라이드 그림 해설 — Position Embedding

```
오른쪽(파란 박스): E_pos — 학습 가능한 위치 임베딩
┌──────────────┐
│   D = 768    │
│ ┌──────────┐ │
│ │pos_emb 0 │ │  ← [CLS] 토큰 위치 임베딩
│N│pos_emb 1 │ │  ← patch 1 위치 임베딩
│+│pos_emb 2 │ │
│1│   ...    │ │
│=│pos_emb196│ │
│197└────────┘ │
└──────────────┘
    E_pos ∈ R^(197 × 768)   ← 학습 가능

가운데: [x_class; x¹ₚE; ··· ; xᴺₚE]  — CLS concat 후
┌────────────────────────┐
│        D = 768          │
│ ┌──────────────────┐   │
│1│  x_class (파란)  │   │  ← 학습 가능한 CLS 벡터 (별도 Parameter)
│ ├──────────────────┤   │
│N│  x¹ₚE  (patch 1)│   │  ← Linear Projection 결과 (빨간)
│=│  x²ₚE  (patch 2)│   │
│196   ...            │   │
│ └──────────────────┘   │
└────────────────────────┘
    ∈ R^(197 × 768)

왼쪽(빨간 박스): z₀ — element-wise 합산 결과
┌──────────────┐
│   D = 768    │
│ ┌──────────┐ │
│N│  z₀[0]   │ │  ← x_class + pos_emb[0]
│+│  z₀[1]   │ │  ← x¹ₚE   + pos_emb[1]
│1│   ...    │ │
│=│  z₀[196] │ │  ← xᴺₚE   + pos_emb[196]
│197└────────┘ │
└──────────────┘
    z₀ ∈ R^(197 × 768)
```

**shape 변환 단계별 요약:**

```
단계          연산                            shape
───────────────────────────────────────────────────────
① xpE       Linear Projection 결과          (196, 768)
② x_class   학습 가능한 CLS 벡터            (  1, 768)
③ concat    torch.cat([cls, xpE], dim=1)    (197, 768)
④ E_pos     학습 가능한 위치 임베딩          (197, 768)
⑤ z₀ = ③+④  element-wise 덧셈 (브로드캐스트 없음)  (197, 768)
```

### 5-5. Multi-Head Self-Attention (MSA) 전체 구현

#### 슬라이드 그림 해설 — Query / Key / Value

```
오른쪽(파란 박스): U_qkv — 학습 가능 가중치
┌─────────────────────────────────┐
│        3D_h = 192               │
│  ┌────────┬────────┬────────┐   │
│D │        │        │        │   │
│= │  W_q   │  W_k   │  W_v   │D_h│
│768│       │        │        │=64│
│  └────────┴────────┴────────┘   │
└─────────────────────────────────┘
    U_qkv ∈ R^(D × 3D_h) = R^(768 × 192)
    D_h = D/k = 768/12 = 64,  3D_h = 192

가운데(빨간 박스): LN(z) — LayerNorm 적용 입력
┌──────────────┐
│   D = 768    │
│ ┌──────────┐ │
│N│ LN(z[0]) │ │   ← CLS 토큰 (0번)
│+│ LN(z[1]) │ │   ← patch 1
│1│   ...    │ │
│=└──────────┘ │
│197           │
└──────────────┘
    LN(z) ∈ R^(197 × 768)

왼쪽(빨간 박스): [q, k, v] 분리 결과 (헤드 1개 기준)
┌──────────────────────────────┐
│        3D_h = 192             │
│ ┌──────┬──────┬──────┐        │
│N│  q   │  k   │  v   │ D_h   │
│+│      │      │      │ = 64  │
│1└──────┴──────┴──────┘        │
│=197                           │
└──────────────────────────────┘
    q, k, v 각각 ∈ R^(197 × 64)
```

**shape 변환 (헤드 1개 기준):**

```
LN(z) ∈ R^(197 × 768)  ×  U_qkv ∈ R^(768 × 192)  =  R^(197 × 192)
                                                           ↓ 3등분
                                         q ∈ R^(197 × 64)
                                         k ∈ R^(197 × 64)
                                         v ∈ R^(197 × 64)
```

**실제 12-head 처리 시 shape:**

```
LN(z)              (B, 197, 768)
× U_qkv            (768 → 2304 = 12 heads × 3 × 64)
qkv                (B, 197, 2304)
reshape            (B, 197, 3, 12, 64)
permute(2,0,3,1,4) (3, B, 12, 197, 64)
q, k, v 각각       (B, 12, 197, 64)   ← 헤드별로 분리 완료
```

---

#### 슬라이드 그림 해설 — Attention Among Patches (Self-Attention Map)

```
가운데(빨간): q ∈ R^(197×64)           오른쪽(빨간): k^T ∈ R^(64×197)
┌──────────┐                            ┌──────────────────────────┐
│  D_h=64  │                            │        N+1 = 197          │
│ ┌──────┐ │                            │  D_h ┌──────────────────┐ │
│N│q[0]  │ │   ← CLS 토큰의 query       │  =64 │       k^T        │ │
│+│q[1]  │ │                            │      └──────────────────┘ │
│1│  ... │ │                            └──────────────────────────┘
│=└──────┘ │
│197       │

왼쪽(빨간): A = softmax(q·k^T / √D_h) ∈ R^(197×197)
┌──────────────────────────────────┐
│           N+1 = 197              │
│  ┌──────────────────────────┐    │
│N │  A[i][j] = 토큰 i 가     │    │
│+ │           토큰 j 에       │    │
│1 │           주목하는 가중치  │    │
│= └──────────────────────────┘    │
│197    (각 행의 합 = 1.0)          │
└──────────────────────────────────┘
```

**shape 변환:**

```
q      ∈ R^(197 × 64)
k^T    ∈ R^(64 × 197)
─────────────────────────────────────────────────────
q·k^T  ∈ R^(197 × 197)   ← 모든 토큰 쌍의 내적 (유사도 Score)
÷ √64                     ← 스케일링 (√D_h = √64 = 8.0)
softmax(dim=-1)           ← 행 방향으로 확률 정규화
─────────────────────────────────────────────────────
A      ∈ R^(197 × 197)   ← Attention Map
```

---

#### 슬라이드 그림 해설 — Self-Attention (SA)

```
가운데(빨간): A ∈ R^(197×197)     오른쪽(빨간): v ∈ R^(197×64)
┌─────────────────┐                ┌──────────┐
│   N+1 = 197     │                │  D_h=64  │
│ ┌─────────────┐ │                │ ┌──────┐ │
│N│             │ │   @   →        │N│v[0]  │ │
│+│      A      │ │                │+│v[1]  │ │
│1└─────────────┘ │                │1│  ... │ │
│=197             │                │=└──────┘ │
└─────────────────┘                │197       │
                                   └──────────┘
왼쪽: SA(z) = A·v ∈ R^(197×64)
┌──────────┐
│  D_h=64  │
│ ┌──────┐ │
│N│SA[0] │ │  ← 각 토큰의 "문맥 반영 표현"
│+│SA[1] │ │
│1│  ... │ │
│=└──────┘ │
│197       │
└──────────┘
```

**shape 변환:**

```
A ∈ R^(197×197)  @  v ∈ R^(197×64)  =  SA(z) ∈ R^(197×64)

의미: SA(z)[i] = Σⱼ A[i][j] · v[j]
    → 토큰 i의 새 표현 = 모든 토큰 j의 Value를 Attention Weight로 가중합
```

---

#### 슬라이드 그림 해설 — Multihead Self-Attention (MSA)

```
가운데(빨간 박스): 12개 헤드의 SA 결과를 concat
┌──────────────────────────────────────────────┐
│                k·D_h = 768                   │
│ ┌──────────┬──────────┬─ ··· ─┬──────────┐  │
│N│  SA₁(z)  │  SA₂(z)  │       │  SA₁₂(z) │D_h│
│+│  (64dim) │  (64dim) │       │  (64dim) │=64│
│1└──────────┴──────────┴───────┴──────────┘  │
│=197                                          │
└──────────────────────────────────────────────┘
    concat ∈ R^(197 × 768)   (12 × 64 = 768 = D)

오른쪽(파란 박스): U_msa ∈ R^(768 × 768)   왼쪽: MSA(z) ∈ R^(197 × 768)
```

**shape 변환:**

```
헤드별 SA:   각 R^(197×64)  × 12개 헤드
                  ↓  concat(dim=-1)
             R^(197 × 768)
                  ↓  × U_msa (768×768)
MSA(z):      R^(197 × 768)   ← 입력 z 와 동일한 shape → Residual 가능 ✓
```

#### MSA 전체 구현 코드
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

#### 슬라이드 그림 해설 — Classification

```
Transformer Encoder 최종 출력 z_L:

                   N+1 = 197
┌──────────────────────────────────┐
│D=768 ┌────────────────────────┐  │
│      │ ◀── z⁰_L  (CLS 토큰)  │  │  ← 슬라이드에서 검은 굵은 선으로 강조
│      │  z¹_L (patch 1)       │  │
│      │  z²_L (patch 2)       │  │
│      │     ...               │  │
│      └────────────────────────┘  │
└──────────────────────────────────┘
    z_L ∈ R^(197 × 768)

① CLS 토큰만 추출:  z⁰_L = z_L[:, 0, :]  ∈ R^(1 × 768)
② LayerNorm 적용:   y = LN(z⁰_L)          ∈ R^(1 × 768)
③ MLP Head 통과:    R^(768) → R^(1000)
④ Softmax:          클래스 확률 분포

슬라이드 왼쪽 그림:
┌──────────┐
│   1      │   ← CLS 토큰 1개만
│ ┌──────┐ │
│n│logit │ │   logit ∈ R^(n_class=1000)
│c│      │ │
│l└──────┘ │
│a         │
│s         │
└──────────┘
```

**shape 변환 요약:**

```
z_L           ∈ R^(197 × 768)   : Encoder 최종 출력 (전체 토큰)
z⁰_L          ∈ R^(1   × 768)   : CLS 토큰만 추출 (0번째 행)
y = LN(z⁰_L) ∈ R^(1   × 768)   : LayerNorm 적용
MLP Head:     R^(768) → R^(1000): 클래스 수만큼 선형 투영
logits        ∈ R^(1000)        : 클래스별 점수
```

> **왜 CLS 토큰을 쓰는가?**  
> [CLS] 토큰은 학습 가능한 초기 벡터로 시작하지만,  
> Transformer Encoder를 통과하면서 **모든 패치 토큰에 Self-Attention** 을 수행한다.  
> 결국 이미지 전체의 정보를 집약한 대표 벡터가 되어 분류에 활용한다.

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
        cls_token = x[:, 0]       # [CLS] 토큰만 추출 → (B, D)
        return self.head(cls_token)  # (B, n_class)

# ViT-Base/16 생성
model = ViT(img_size=224, patch_size=16, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12)
img = torch.randn(2, 3, 224, 224)
print(model(img).shape)  # (2, 1000)
```

#### 전체 Shape 변환 한눈에 보기 (ViT-Base/16)

```
단계                    연산                        출력 shape
──────────────────────────────────────────────────────────────────────
입력 이미지              —                           (B, 3, 224, 224)
Patch Flatten           xp = unfold + reshape       (B, 196, 768)
Linear Projection       xpE = xp · E                (B, 196, 768)
CLS concat              [cls; xpE]                  (B, 197, 768)
+ Position Embedding    z₀ = above + E_pos          (B, 197, 768)
──────────────────────────────────────────────────── ×12 반복 ────────
LayerNorm               LN(zₗ₋₁)                    (B, 197, 768)
QKV Projection          z · U_qkv  (→ 2304)         (B, 197, 2304)
헤드 분리               reshape + permute            (3, B, 12, 197, 64)
QK^T Score              q @ k^T                     (B, 12, 197, 197)
Scale + Softmax         Attention Map A             (B, 12, 197, 197)
Value 가중합            SA = A @ v                  (B, 12, 197, 64)
헤드 Concat             reshape                     (B, 197, 768)
MSA 출력 투영           × U_msa                     (B, 197, 768)
Residual                z + MSA_out  →  z'ₗ         (B, 197, 768)
LayerNorm + MLP         FeedForward                 (B, 197, 768)
Residual                z'ₗ + MLP  →  zₗ            (B, 197, 768)
─────────────────────────────────────────────────────────────────────
CLS 토큰 추출           z_L[:, 0, :]                (B, 768)
LayerNorm               y = LN(z⁰_L)                (B, 768)
MLP Head                Linear(768 → 1000)          (B, 1000)
─────────────────────────────────────────────────────────────────────
```

> **핵심**: Encoder 내부에서 `(B, 197, 768)` shape은 **절대 변하지 않는다**.  
> Residual 연산이 항상 같은 shape을 유지하기 때문에 임의 깊이(L)로 쌓을 수 있다.

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

#### 슬라이드 그림 해설 — Architecture & Loss (Distillation 구조)

```
슬라이드 오른쪽: Teacher Model (Pretrained, Freeze)
  ┌────────────────────────────────────────┐
  │  CNN (RegNet, EfficientNet 계열 등)    │
  │  입력: (B, 3, H, W)                   │
  │  출력: Teacher logits  ∈ R^(B × C)    │  C = class 수
  └────────────────────────────────────────┘
        │   Hard Label: argmax → y_t ∈ {0,..,C-1}
        │   Soft Label: softmax(Z_t/τ) ∈ R^(B × C)
        ▼
슬라이드 왼쪽: Student Model (DeiT, Trainable)
  입력 이미지 + 동일 증강
  ┌──────────────────────────────────────────────────────┐
  │                 Transformer Encoder                   │
  │                                                      │
  │  [class token] [patch1] [patch2] ··· [dist token]   │
  │      ↑                                    ↑          │
  │   학습 가능한 CLS                       학습 가능한   │
  │   임베딩 (ViT와 동일)                  Distill 토큰  │
  └──────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
  Classification Head                 Distillation Head
  (Linear: D → C)                     (Linear: D → C)
         │                                    │
         ▼                                    ▼
  Hard Prediction                      Hard Prediction
  Z_s ∈ R^(B × C)                      Z_s ∈ R^(B × C)
         │                                    │
         ▼                                    ▼
  True Label (GT)                  Teacher Output
  Cross-Entropy Loss               Hard: CE(Z_s, argmax(Z_t))
  L_CE                             Soft: KL(softmax(Z_s/τ), softmax(Z_t/τ))
  ↑ Classification Loss            ↑ Distillation Loss
```

**DeiT 입출력 Shape 변환:**

```
입력 이미지:              (B, 3, H, W)
Patch Embedding:          (B, N, D)           ← ViT와 동일
[CLS] + [Dist] concat:    (B, N+2, D)         ← 토큰 2개 추가 (ViT는 N+1)
Transformer Encoder:      (B, N+2, D)         ← shape 유지
                                ↓ 분리
CLS 토큰 (index 0):       (B, D)
Dist 토큰 (index N+1):    (B, D)
                                ↓ 각각 Linear
Classification Head:      (B, C)   ← Z_cls
Distillation Head:        (B, C)   ← Z_dist
```

> ViT vs DeiT 토큰 수 비교:  
> ViT: `[CLS] + N patches` = **N+1** 토큰  
> DeiT: `[CLS] + N patches + [Distill]` = **N+2** 토큰

**Loss 계산 Shape:**

```
Hard Distillation:
  student_cls:    (B, C)   ← Classification Head 출력
  student_dist:   (B, C)   ← Distillation Head 출력
  teacher_logits: (B, C)   ← Teacher 출력
  labels:         (B,)     ← GT 레이블

  teacher_hard = teacher_logits.argmax(dim=1)  → (B,)
  L_CE   = CE(student_cls,  labels)            → scalar
  L_dist = CE(student_dist, teacher_hard)      → scalar
  L_total = (1-α)·L_CE + α·L_dist             → scalar

Soft Distillation:
  teacher_soft = softmax(teacher_logits / τ)   → (B, C)
  student_soft = log_softmax(student_dist / τ) → (B, C)
  L_dist = KL(student_soft, teacher_soft) × τ² → scalar
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

#### 슬라이드 그림 해설 — Experiments

```
슬라이드 Table 2: Teacher 모델에 따른 Student 성능 (ImageNet Top-1 acc, %)

Teacher Models  | Teacher acc | DeiT-B pretrain | DeiT-B ↑384
─────────────────────────────────────────────────────────────
DeiT-B          |    81.8     |      81.9       |    83.1   ← Transformer 교사
RegNetY-4GF     |    80.0     |      82.7       |    83.6   ┐
RegNetY-8GF     |    81.7     |      82.7       |    83.8   │ CNN 교사
RegNetY-12GF    |    82.4     |      83.1       |    84.1   │ → Inductive bias
RegNetY-16GF    |    82.9     |      83.1       |    84.2   ┘ 전달 효과!

주목할 점: CNN 교사(RegNetY-16GF)가 Transformer 교사(DeiT-B)보다
           Student 성능을 더 높임
→ CNN의 locality/translation-equivariance 편향이 유용한 지식으로 전달됨

슬라이드 Table 3: Distillation 방식 비교

method              | label | teacher |  Ti224  |  S224  | B224  | B↑384
────────────────────────────────────────────────────────────────────────
DeiT - no distill   |   ✓   |    ✗    |  72.2  |  79.8  | 81.8  | 83.1
DeiT - usual dist   |   ✗   |  soft   |  72.2  |  79.8  | 81.8  | 83.2
DeiT - hard dist    |   ✗   |  hard   |  74.3  |  80.9  | 83.0  | 84.0   ← hard 효과
DeiT☆: class embed  |   ✓   |  hard   |  73.9  |  80.9  | 83.0  | 84.2
DeiT☆: dist embed   |   ✓   |  hard   |  74.6  |  81.1  | 83.1  | 84.4
DeiT☆: class+dist   |   ✓   |  hard   |  74.5  |  81.2  | 83.4  | 84.5  ← 최고
```

```python
# 추론 시 두 헤드 출력을 합산하여 최종 예측
# (Late fusion: class + distillation embeddings)
logits_cls  = cls_head(cls_token)    # (B, C)
logits_dist = dist_head(dist_token)  # (B, C)

# 두 logit을 평균 → 단일 예측
logits = (logits_cls + logits_dist) / 2  # (B, C)
pred   = logits.argmax(dim=-1)           # (B,)
```

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

#### 슬라이드 그림 해설 — Window Mechanism

```
슬라이드 왼쪽: Window-based self attention 전체 그림
(예: 4×4 패치 격자, M=2 윈도우)

Feature map (H/P × W/P 패치 격자):
  ┌──0──┬──1──┬──2──┬──3──┐
  │     │     │     │     │
  ├──4──┼──5──┼──6──┼──7──┤    ← 슬라이드에서 숫자는 패치 index
  │     │     │     │     │
  ├──8──┼──9──┼─10──┼─11──┤
  │     │     │     │     │
  ├─12──┼─13──┼─14──┼─15──┤
  └─────┴─────┴─────┴─────┘
  (×표시: 윈도우 경계를 넘는 long-range attention — W-MSA에서 불가)

슬라이드 오른쪽: Layer l (W-MSA) → Layer l+1 (SW-MSA) 전환

Layer l: W-MSA                    Layer l+1: SW-MSA
┌─────────┬─────────┐              ┌──┬────────┬──┐
│         │         │              │  │        │  │
│  Win 0  │  Win 1  │   →shift→   │  │        │  │
│         │         │              ├──┼────────┼──┤
├─────────┼─────────┤              │  │        │  │
│         │         │              │  │        │  │
│  Win 2  │  Win 3  │              ├──┼────────┼──┤
│         │         │              │  │        │  │
└─────────┴─────────┘              └──┴────────┴──┘
각 윈도우 내부에서만 Attention     이전 경계를 가로지르는 Attention 발생
```

**W-MSA Shape 변환 (Swin-T 기준: C=96, M=7):**

```
입력 feature map:    (B, H/4, W/4, C)      ← Stage 1 기준 (패치: 4×4 픽셀)
                예:  (B, 56, 56, 96)       ← 224/4=56, C=96

window_partition:    (B×num_win, M, M, C)
                   = (B×64, 7, 7, 96)     ← num_win = (56/7)² = 64

flatten (window):    (B×64, M², C)
                   = (B×64, 49, 96)       ← M²=49 토큰이 Attention

QKV Projection:      (B×64, 49, 3C)
Attention Map:       (B×64, heads, 49, 49) ← M²×M² = 49×49 (작은 크기!)
SA Output:           (B×64, 49, C)
                   = (B×64, 49, 96)

window_reverse:      (B, H/4, W/4, C)
                   = (B, 56, 56, 96)     ← 원래 shape 복원
```

**SW-MSA: Cyclic Shift + Masking:**

```
① Cyclic shift: torch.roll(x, shifts=(-M//2, -M//2), dims=(1,2))
  → feature map을 (-3, -3) 픽셀만큼 순환 이동

② 이동 후 동일하게 window partition → Attention

③ 다른 윈도우에서 온 패치끼리는 Masked Attention (−∞ 마스크)
  → softmax 후 가중치 = 0 → 실질적으로 같은 윈도우끼리만 Attention

④ Reverse cyclic shift: torch.roll(x, shifts=(M//2, M//2), dims=(1,2))
  → 원래 위치로 복원
```

```python
# SW-MSA에서 사용하는 Attention Mask 생성 예시
import torch

def compute_mask(H, W, window_size, shift_size, device):
    """Cyclic shift 후 다른 윈도우 패치끼리 마스킹하기 위한 mask"""
    img_mask = torch.zeros(1, H, W, 1, device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # 윈도우로 분할 후 같은 구역이면 0, 다른 구역이면 음수 마스크
    mask_windows = window_partition(img_mask, window_size)  # (num_win, M, M, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask  # (num_win, M², M²)
```

---

#### 슬라이드 그림 해설 — Architecture (1/2): Two Successive Swin Transformer Blocks

```
슬라이드 왼쪽: Layer l feature map (7×7 윈도우 2개 예시)
┌──────────────┬──────────────┐
│              │              │
│   Window 0   │   Window 1   │  ← 각 윈도우: M×M = 7×7 패치
│  (7×7 패치) │  (7×7 패치) │
├──────────────┼──────────────┤
│              │              │
│   Window 2   │   Window 3   │
│              │              │
└──────────────┴──────────────┘

슬라이드 가운데: Swin Block 쌍 (반드시 2개가 한 쌍)
           zˡ⁻¹
            │
       ┌────┴────┐
       │   LN    │
       └────┬────┘
            │
       ┌────┴────┐
       │  W-MSA  │  ← Layer l: 고정 윈도우 내 Attention
       └────┬────┘
            │
       ┌────┴────┐  + zˡ⁻¹ (Residual)
       │  z̃ˡ     │
       └────┬────┘
            │
       ┌────┴────┐
       │   LN    │
       └────┬────┘
            │
       ┌────┴────┐
       │   MLP   │
       └────┬────┘
            │
           zˡ (+ Residual)
            │
       ┌────┴────┐
       │   LN    │
       └────┬────┘
            │
       ┌────┴────┐
       │  SW-MSA │  ← Layer l+1: Shifted Window Attention
       └────┬────┘
            │
       ┌────┴────┐  + zˡ (Residual)
       │  z̃ˡ⁺¹   │
       └────┬────┘
            │
       ┌────┴────┐
       │   LN    │
       └────┴────┘
            │
           zˡ⁺¹ (+ Residual)

슬라이드 오른쪽: Layer l+1 (SW-MSA 후)
  윈도우가 M/2 shift → 이전 경계를 가로지르는 새 윈도우 생성
```

---

#### 슬라이드 그림 해설 — Architecture (2/2): 전체 파이프라인 & Patch Merging

```
입력: H×W×3 이미지

Patch Partition (4×4 픽셀 → 1 토큰):
  (B, H, W, 3) → (B, H/4, W/4, 48)    ← 4×4×3=48

Linear Embedding:
  (B, H/4, W/4, 48) → (B, H/4, W/4, C)  ← C=96 (Swin-T)

─────────────────────────────────────────────────────────
Stage 1:  (B, H/4, W/4, C) = (B, 56, 56, 96)
  Swin Transformer Block × 2
  → (B, 56, 56, 96)

Patch Merging 1:
  2×2 이웃 패치를 concat → (B, H/8, W/8, 4C)
  Linear 투영으로 채널 축소 → (B, H/8, W/8, 2C) = (B, 28, 28, 192)

─────────────────────────────────────────────────────────
Stage 2:  (B, 28, 28, 192)
  Swin Transformer Block × 2
  → (B, 28, 28, 192)

Patch Merging 2:
  → (B, H/16, W/16, 4C) → Linear → (B, H/16, W/16, 4C)
  = (B, 14, 14, 384)

─────────────────────────────────────────────────────────
Stage 3:  (B, 14, 14, 384)
  Swin Transformer Block × 6
  → (B, 14, 14, 384)

Patch Merging 3:
  → (B, 7, 7, 768)

─────────────────────────────────────────────────────────
Stage 4:  (B, 7, 7, 768)
  Swin Transformer Block × 2
  → (B, 7, 7, 768)

Global Average Pooling → (B, 768)
Classification Head    → (B, n_class)
```

**Patch Merging 상세 (슬라이드 오른쪽 위 그림):**

```
H/4 × W/4 × C 특징 맵에서 2×2 이웃 패치를 선택:

  ┌───┬───┐
  │ A │ C │   A, B, C, D 각각 (B, H/8, W/8, C)
  ├───┼───┤
  │ B │ D │
  └───┴───┘

concat(A, B, C, D, dim=-1):  (B, H/8, W/8, 4C)
                                  ↓
Linear Projection (4C → 2C):  (B, H/8, W/8, 2C)   ← 채널은 2배, 해상도는 절반
```

```python
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        # 2×2 이웃 패치 선택
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)                      # (B, H/2, W/2, 2C)
        return x
```

**Swin-T 전체 Shape 요약:**

```
단계                  출력 shape                비고
─────────────────────────────────────────────────────────────
입력 이미지           (B, 3, 224, 224)
Patch Partition       (B, 56, 56, 48)           4×4 픽셀 = 1패치
Linear Embedding      (B, 56, 56, 96)           C=96
Stage 1 (×2 block)   (B, 56, 56, 96)           W-MSA→SW-MSA
Patch Merging 1       (B, 28, 28, 192)          해상도↓2, 채널×2
Stage 2 (×2 block)   (B, 28, 28, 192)
Patch Merging 2       (B, 14, 14, 384)
Stage 3 (×6 block)   (B, 14, 14, 384)
Patch Merging 3       (B, 7,  7,  768)
Stage 4 (×2 block)   (B, 7,  7,  768)
Global Avg Pool       (B, 768)
MLP Head              (B, n_class)
```

---

#### 슬라이드 그림 해설 — Complexity: ViT vs Swin 복잡도 비교

```
ViT Global MSA 복잡도 유도 (슬라이드 하단 그림):

  Q ∈ R^(N × C)      K^T ∈ R^(C × N)     N = HW (전체 패치 수)
  ┌──────────┐         ┌──────────┐
  │          │    @    │          │  →  Q·K^T ∈ R^(N × N)
  │  N × C   │         │  C × N   │     연산량: N × 2C × N = 2CN²
  └──────────┘         └──────────┘

전체 MSA 연산량: 4HWC² + 2(HW)²C → O((HW)²)
                 ↑QKV+Proj        ↑Attention

Window MSA 복잡도 유도:
  윈도우 1개 내 Attention: 2M²·M²·C = 2(M²)²C
  전체 윈도우 수:           HW / M²
  합산:     (HW/M²) × 2(M²)²C = 2M²HWC

전체 W-MSA 연산량: 4HWC² + 2M²HWC → O(HW) (M은 상수이므로)
  ↑QKV+Proj        ↑Attention

결론:
  ViT:  O((HW)²) — 해상도 올리면 연산 폭발적 증가
  Swin: O(HW)    — 해상도가 늘어도 선형적으로만 증가
```

```python
# 복잡도 비교 수치 예시 (H=W=56, C=96, M=7)
H, W, C, M = 56, 56, 96, 7
N = H * W   # 3136 패치

vit_attn_ops  = 2 * (N**2) * C         # 1,887,436,800 ≈ 1.9B ops
swin_attn_ops = 2 * (M**2) * N * C     # 2 * 49 * 3136 * 96 = 29,491,200 ≈ 29M ops
print(f"ViT Attention: {vit_attn_ops/1e9:.1f}B")  # 1.9B
print(f"Swin Attention: {swin_attn_ops/1e6:.0f}M") # 29M
print(f"Swin is {vit_attn_ops/swin_attn_ops:.0f}x faster")  # ~64x
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

### 6.3 CLIP (Constrative Language-Image Pre-training)

> 논문: _Learning Transferable Visual Models From Natural Language Supervision_, ICML 2021

#### 핵심 아이디어

- 인터넷의 **4억 쌍 (이미지, 텍스트) 데이터**로 학습 (e.g. Instagram에 올린 이미지와 hashing text에 대한 data 조합 사용)
- 이미지 인코더 + 텍스트 인코더를 **동일한 임베딩 공간**에서 정렬
- **Zero-shot 분류** 가능 → 새로운 클래스에 재학습 불필요 (zero-shot capability: 처음 보는 data를 가지고 처리하는 능력)
- No manual labeling required!!

#### 슬라이드 그림 해설 — Architecture (이미지-텍스트 인코더 구조)

```
슬라이드 왼쪽: Text Encoder (Transformer)
  입력 텍스트: "Pepper the aussie pup"
         ↓ 토크나이저
  토큰 시퀀스 T ∈ R^(N, l)           ← N = 배치 수, l = 텍스트 길이
         ↓ Text Encoder (Transformer)
  T_f ∈ R^(N, d_t)                   ← d_t: 텍스트 인코더 출력 차원
         ↓ Linear Projection (W_t)
  T_e = l2_normalize(T_f @ W_t)      → R^(N, d)  ← 공통 임베딩 공간

슬라이드 아래쪽: Image Encoder (ResNet 또는 ViT)
  입력 이미지: (N, 3, H, W)
         ↓ Image Encoder (ResNet/ViT)
  I_f ∈ R^(N, d_i)                   ← d_i: 이미지 인코더 출력 차원
         ↓ Linear Projection (W_i)
  I_e = l2_normalize(I_f @ W_i)      → R^(N, d)  ← 공통 임베딩 공간

슬라이드 가운데: 유사도 행렬 (N×N) (Similarity: Inner Product로 계산)
  ┌──────────────────────────────────────┐
  │           T₁   T₂   T₃  ···  T_N   │
  │  I₁  │ I₁·T₁ I₁·T₂ I₁·T₃ ··· │   │
  │  I₂  │ I₂·T₁ I₂·T₂ I₂·T₃ ··· │   │
  │  I₃  │ I₃·T₁ I₃·T₂ I₃·T₃ ··· │   │
  │  ⋮   │  ⋮     ⋮     ⋮         │   │
  │  I_N │             ···  I_N·T_N│   │
  └──────────────────────────────────────┘
  logits = I_e @ T_e.T × exp(t)   ∈ R^(N × N)
  ↑ 대각선: 매칭 쌍 (High similarity 목표)
  ↑ 비대각: 불일치 쌍 (Low similarity 목표)
```

**Shape 변환 전체 흐름:**

```
입력
  이미지:  (N, 3, H, W)
  텍스트:  (N, l)               ← l: 최대 토큰 길이 (예: 77)

인코딩
  I_f = image_encoder(I)        → (N, d_i)
  T_f = text_encoder(T)         → (N, d_t)

투영 (공통 d-차원 임베딩 공간으로)
  I_e = l2_norm(I_f @ W_i)      → (N, d)   W_i: (d_i × d)
  T_e = l2_norm(T_f @ W_t)      → (N, d)   W_t: (d_t × d)

유사도 계산
  logits = I_e @ T_e.T * exp(t) → (N, N)   t: 학습 가능한 temperature
  labels = [0, 1, 2, ..., N-1]  → (N,)    대각선 = 정답

손실 (대칭)
  loss_i2t = CE(logits,   labels)  ← 이미지 기준 (행 방향)
  loss_t2i = CE(logits.T, labels)  ← 텍스트 기준 (열 방향)
  loss = (loss_i2t + loss_t2i) / 2 → scalar
```

---

#### 슬라이드 그림 해설 — Training (2/2): InfoNCE Loss 수식

```
유사도 행렬 (N×N):
  ┌──────────────────────────────────────┐
  │   (파란 대각선): 매칭 쌍 High sim    │
  │   ┌──┬──┬──┬──┐                     │
  │   │██│  │  │  │  ← I₁·T₁ 최대화    │
  │   ├──┼──┼──┼──┤                     │
  │   │  │██│  │  │  ← I₂·T₂ 최대화    │
  │   ├──┼──┼──┼──┤                     │
  │   │  │  │██│  │  ← I₃·T₃ 최대화    │
  │   ├──┼──┼──┼──┤                     │
  │   │  │  │  │██│  ← I₄·T₄ 최대화    │
  │   └──┴──┴──┴──┘                     │
  └──────────────────────────────────────┘

InfoNCE Loss (각 행에 대해 Softmax Cross-Entropy):

  L_i = -log [ exp(sim(zᵢ_img, zⁱ_text) / τ) ]
              ─────────────────────────────────────
              Σⱼ exp(sim(zⁱ_img, zʲ_text) / τ)

Temperature τ의 효과:
  τ 작음 → exp 값 차이 크게 → 대각선 집중 강화 → 빠른 학습
  τ 큼   → exp 값 유사 → 분포 평탄 → 수렴 느림
  → τ는 학습 가능한 파라미터 (exp(t)로 표현, 초기값 ≈ 0.07)
```

---

#### 슬라이드 그림 해설 — Inference (1/2): Zero-shot 분류

```
(1) Zero-shot 분류기 구성 (슬라이드 위쪽 그림)

  클래스 이름 목록: ["plane", "car", "dog", ..., "bird"]
         ↓ Prompt Engineering
  프롬프트 생성: ["A photo of a plane.",
                 "A photo of a car.",
                 "A photo of a dog.",
                  ...,
                 "A photo of a bird."]
         ↓ Text Encoder
  텍스트 임베딩 [T₁, T₂, ..., T_N]  각 ∈ R^(1 × d)
  → 쌓으면: T_all ∈ R^(N_class × d)  ← 분류기 가중치 역할

(2) 이미지 추론 (슬라이드 아래쪽 그림)

  입력 이미지 → Image Encoder → I₁ ∈ R^(1 × d)

  유사도 계산:
  ┌──────────────────────────────────────┐
  │  I₁·T₁  I₁·T₂  I₁·T₃  ···  I₁·T_N │
  │  (1 × N_class) 유사도 벡터           │
  └──────────────────────────────────────┘

  argmax → 가장 유사한 텍스트 클래스 = 예측 레이블
```

**Shape 요약:**

```
Text Embeddings:   (N_class, d)      ← 클래스 수 × 임베딩 차원
Image Embedding:   (1, d)           ← 쿼리 이미지 1장
Similarity:        (1, N_class)     ← 각 클래스와의 유사도
Predicted class:   argmax           → scalar (클래스 인덱스)
```

```python
import torch
import torch.nn.functional as F

# ── Prompt Engineering ──────────────────────────────────────
classnames = ["cat", "dog", "bird"]
templates  = ["A photo of a {}.", "A {} in the wild.", "A picture of a {}."]

# 여러 template의 텍스트 임베딩 평균 (Ensembling)
text_embeds = []
for cls in classnames:
    embeds = []
    for tmpl in templates:
        prompt = tmpl.format(cls)
        # tokens = tokenize(prompt)         # (1, l)
        # embed  = text_encoder(tokens)     # (1, d)
        # embeds.append(F.normalize(embed)) # L2 정규화
        pass
    # cls_embed = torch.stack(embeds).mean(0)  # (1, d) — template 평균
    # text_embeds.append(cls_embed)

# text_features = torch.cat(text_embeds)      # (N_class, d)

# ── Zero-shot Inference ─────────────────────────────────────
# image_features = image_encoder(image)       # (1, d)
# image_features = F.normalize(image_features)
# logits = image_features @ text_features.T   # (1, N_class)
# pred   = logits.argmax(dim=-1)              # (1,) — 예측 클래스
```

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

#### 슬라이드 그림 해설 — Architecture (1/6): Student-Teacher Framework

```
슬라이드 그림 설명:
  x (원본 이미지)
    │
    ├── augment → x₁ ──────────────→  student g_θs  ──→ p₁ = softmax(z_s / τ_s)
    │                                      ↑                          │
    └── augment → x₂ ──────────────→  teacher g_θt  ──→ p₂ = softmax((z_t - c) / τ_t)
                                           │                          │
                                          ema   Loss = -p₂ log p₁  ◄─┘
                                           │         (Cross-Entropy)
                                      sg (stop gradient)

핵심 특징:
  ① Student ↔ Teacher: 완전히 동일한 구조 (ViT-S/16 등)
  ② Teacher가 더 강력한 게 아니라 단지 EMA 파라미터
  ③ Teacher는 역전파 없음 (sg: stop gradient)
  ④ Teacher 출력에만 centering 적용
```

**전체 Shape 흐름 (ViT-S/16 기준: D=384, K=65536):**

```
입력 이미지 x:             (B, 3, H, W)
augment(x) → x₁, x₂:     (B, 3, 224, 224) ← global crop (≥50%)

ViT Backbone (Student/Teacher 공통):
  Patch Embed:   (B, 196, 384)    ← N=196, D=384 (ViT-S)
  + CLS token:   (B, 197, 384)
  Encoder × 12: (B, 197, 384)
  CLS 추출:      (B, 384)         ← z_s (student) 또는 z_t (teacher)

MLP Projection Head (3층):
  (B, 384) → (B, 2048) → (B, 2048) → (B, K=65536)
  z_s ∈ R^(B × 65536)
  z_t ∈ R^(B × 65536)

Student:  p₁ = softmax(z_s / τ_s)          → (B, 65536)   τ_s=0.1 (높음)
Teacher:  p₂ = softmax((z_t - c) / τ_t)    → (B, 65536)   τ_t=0.04 (낮음)
          c: centering 벡터 ∈ R^(65536)

Loss: L = -Σ p₂ · log(p₁)  → scalar
```

---

#### 슬라이드 그림 해설 — Architecture (2/6): Multi-crop Training

```
슬라이드 왼쪽 사진: 기린 이미지에서 크롭 방식 설명

  원본 이미지 (기린 전체)
  ┌──────────────────────────┐
  │  ┌──────────────────┐   │
  │  │  Global Crop ①   │   │  ← ≥50% 면적 (빨간 박스)
  │  │  (224×224)       │   │
  │  └──────────────────┘   │
  │       ┌────┐ ┌────┐     │
  │       │Loc1│ │Loc2│     │  ← <50% 면적 (파란 박스)
  │       └────┘ └────┘     │
  └──────────────────────────┘

Multi-crop 전략:
  Global crop (2개): ≥50% 면적 → 전역 정보 포함
    → Student ✓, Teacher ✓ 모두 입력
  Local crop (여러 개): <50% 면적 → 부분 정보만
    → Student ✓ 만 입력 (Teacher는 global만 봄)

학습 목표 (슬라이드 오른쪽 구조도):
  Student(local crop)  →  Teacher(global crop)를 예측
  = "작은 조각에서 전체 의미를 추론"
  → local-to-global correspondence 학습
```

**Multi-crop Shape:**

```
Global crops (x_g1, x_g2): 각 (B, 3, 224, 224)
Local crops  (x_l1 ~ x_lm): 각 (B, 3, 96, 96)   ← 작은 크기

Teacher 입력: [x_g1, x_g2]
  → t_g1, t_g2 각 (B, K)   ← 2개의 global crop 출력

Student 입력: [x_g1, x_g2, x_l1, ..., x_lm]
  → s_g1, s_g2, s_l1, ..., s_lm 각 (B, K)

Loss: Σ_{x∈student crops} Σ_{t∈teacher outputs} H(t, s)
     (동일한 crop 쌍은 건너뜀)
```

---

#### 슬라이드 그림 해설 — Architecture (3/6): EMA & Centering

```
슬라이드 수식 설명:

① EMA (Exponential Moving Average): 최근의 것을 가장 크게 weighting, 오래된 것을 적게 weighting
   θ_teacher ← m · θ_teacher + (1-m) · θ_student
   m ≈ 0.996  (1에 가까울수록 teacher가 느리게 변함)

   왜 EMA?
   ─────────────────────────────────────────────
   m=0.996 → 매 스텝 student를 0.4%만 반영
   → Teacher는 최근 student들의 "시간 평균"
   → 안정적이고 일관된 감독 신호 제공
   → Collapse 방지 (student와 teacher가 동시에 나쁜 해로 수렴 방지)

② Centering (Teacher logit 정규화):
   c ← λ·c + (1-λ) · (1/B) · Σᵢ z_teacher⁽ⁱ⁾   ← EMA로 배치 평균 추적

   z̃_teacher = z_teacher - c                       ← centering 적용

   왜 Centering?
   ─────────────────────────────────────────────
   Teacher가 특정 차원에 편향되면 → Collapse (하나의 클래스로만 수렴)
   배치 평균을 빼면 → 모든 차원이 균등하게 활성화
   → Sharp한 τ_t=0.04와 조합해도 안정적 학습 가능
```

```python
import torch

class DINO_CenteringAndEMA:
    def __init__(self, out_dim=65536, center_momentum=0.9, ema_momentum=0.996):
        self.center = torch.zeros(1, out_dim)  # c ∈ R^(1 × K)
        self.center_m = center_momentum        # λ
        self.ema_m    = ema_momentum           # m

    def update_center(self, teacher_outputs):
        # teacher_outputs: list of (B, K) tensors
        batch = torch.cat(teacher_outputs, dim=0)   # (total_B, K)
        batch_mean = batch.mean(dim=0, keepdim=True) # (1, K)
        # EMA 업데이트
        self.center = self.center_m * self.center + (1 - self.center_m) * batch_mean

    def update_teacher(self, student_params, teacher_params):
        for p_s, p_t in zip(student_params, teacher_params):
            p_t.data = self.ema_m * p_t.data + (1 - self.ema_m) * p_s.data
```

---

#### 슬라이드 그림 해설 — Architecture (4/6): Temperature Softmax 차이

```
Student Temperature (τ_s = 0.1, temperature 높음) → 분포가 평탄(Smooth):

  logits = [2.0, 0.5, 1.0, 0.3]
  softmax(logits / 0.1) ≈ [0.93, 0.00, 0.07, 0.00]  → 높은 entropy 유지
  → Collapse 방지, 다양한 클래스를 탐색

Teacher Temperature (τ_t = 0.04, temperature 낮음) → 분포가 집중(Sharp):

  softmax(logits / 0.04) ≈ [1.00, 0.00, 0.00, 0.00]  → 거의 one-hot
  → 고신뢰 pseudo-label 역할
  → Student에게 명확한 학습 신호 제공

Softmax with Temperature 수식:
  σᵢ = exp(zᵢ / τ) / Σⱼ exp(zⱼ / τ)

  τ → 0:  one-hot 분포 (완전히 집중)
  τ → ∞:  uniform 분포 (완전히 평탄)
```

```python
import torch
import torch.nn.functional as F

z = torch.tensor([2.0, 0.5, 1.0, 0.3])

print("Student (τ=0.1, smooth):")
print(F.softmax(z / 0.1, dim=0).numpy().round(3))
# [0.933, 0.001, 0.066, 0.001]

print("Teacher (τ=0.04, sharp):")
print(F.softmax(z / 0.04, dim=0).numpy().round(3))
# [1.000, 0.000, 0.000, 0.000]
```

---

#### 슬라이드 그림 해설 — Architecture (5/6): ViT 출력 → MLP Projection → Logit

```
슬라이드 수식:
  CLS token → MLP projection → Logit vector

Shape 흐름 (ViT-S/16 기준):

ViT Backbone:
  입력:      (B, 3, 224, 224)
  Encoder:   (B, 197, 384)     ← D=384 (ViT-S)
  CLS 추출:  (B, 384)

MLP Projection Head (3층 MLP, BN 포함):
  (B, 384)  → Linear → BN → GELU → (B, 2048)
  (B, 2048) → Linear → BN → GELU → (B, 2048)
  (B, 2048) → Linear                → (B, 65536)  ← K=65536 (프로토타입 수)
  마지막 출력: L2 정규화 적용

z_s (student logit):  (B, 65536)
z_t (teacher logit):  (B, 65536)

Temperature Softmax 적용:
  p₁ = softmax(z_s / τ_s)             → (B, 65536)  ← Student 확률 분포
  p₂ = softmax((z_t - c) / τ_t)       → (B, 65536)  ← Teacher 확률 분포

Cross-Entropy Loss:
  L = -Σₖ p₂[k] · log(p₁[k])          → scalar
    = H(p₂, p₁)  (Cross-Entropy)
=> Loss는 student에만 반영하고, teacher에는 반영하지 않음. student는 이 결과를 ema (exponential moving avg)를 통해서만 teacher에게 전달함

"Classes in self-supervised learning" = Teacher가 발견한 의미적 클러스터
  → 레이블 없이 등장, 학습 과정에서 자연스럽게 형성됨
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    """DINO MLP Projection Head"""
    def __init__(self, in_dim=384, out_dim=65536,
                 hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),  # bottleneck
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        # x: CLS token (B, in_dim=384)
        x = self.mlp(x)                      # (B, bottleneck_dim=256)
        x = F.normalize(x, dim=-1)           # L2 정규화
        logit = self.last_layer(x)           # (B, out_dim=65536)
        return logit

# shape 확인
head = DINOHead(in_dim=384, out_dim=65536)
cls_token = torch.randn(4, 384)             # (B=4, D=384)
logit = head(cls_token)
print(f"logit shape: {logit.shape}")        # (4, 65536)
```

---

#### 슬라이드 그림 해설 — Architecture (6/6): Stop Gradient

```
슬라이드 수식 sg (stop gradient):

  Student:  역전파 O → 파라미터 갱신됨
  Teacher:  sg 표시 → 역전파 X → EMA로만 갱신

  왜 Stop Gradient?
  ─────────────────────────────────────────────────────
  만약 Teacher도 역전파를 받으면:
  → Student와 Teacher가 같은 방향으로 동시에 업데이트
  → 두 모델이 하나의 trivial solution으로 수렴 (collapse)

  Stop Gradient + EMA의 조합:
  → Teacher가 일관된 "느린" 감독자 역할
  → Student는 teacher의 분포를 따라가며 학습
  → 서로 당기는 힘의 균형 → 표현 다양성 유지
```

```python
# Stop gradient 구현 (PyTorch)
# 역전파 시 teacher 쪽으로 gradient가 흐르지 않도록 detach

teacher_output = teacher_net(x2)         # (B, K)
teacher_output = teacher_output.detach() # ← stop gradient

student_output = student_net(x1)         # (B, K)

# teacher_output: gradient 없음 (fixed target)
# student_output: gradient 있음 (학습 대상)
loss = cross_entropy(student_output, teacher_output)
loss.backward()  # student만 업데이트
```

---

#### 슬라이드 그림 해설 — Architecture Summary: 전체 알고리즘

```
슬라이드 Algorithm 1 (PyTorch pseudocode, multi-crop 제외 버전):

매 iteration마다:
─────────────────────────────────────────────────────
① 데이터 로드 및 두 가지 augment 적용
   x1 = augment(x),  x2 = augment(x)    각 (B, 3, 224, 224)

② Student 출력 계산 (High Temperature τ_s=0.1)
   s1 = g_θs(x1)    (B, K)
   s2 = g_θs(x2)    (B, K)

③ Teacher 출력 계산 (Low Temperature τ_t=0.04, centering c)
   t1 = g_θt(x1)    (B, K)
   t2 = g_θt(x2)    (B, K)

④ Loss 계산 (대칭)
   loss = H(t1, s2)/2 + H(t2, s1)/2
   H(t, s) = -(softmax((t-C)/τ_t)) · log(softmax(s/τ_s))

⑤ Student만 역전파
   loss.backward()
   update(g_θs)   ← SGD/AdamW로 student 갱신

⑥ Teacher EMA 업데이트 (역전파 없이)
   θ_t ← l·θ_t + (1-l)·θ_s   l=EMA momentum

⑦ Center 업데이트
   C ← m·C + (1-m)·mean([t1, t2])
─────────────────────────────────────────────────────
```

**전체 Shape 변환 요약 (ViT-S/16, K=65536):**

```
단계                   Student shape          Teacher shape
───────────────────────────────────────────────────────────────
입력 이미지             (B, 3, 224, 224)       (B, 3, 224, 224)
Patch Embed             (B, 196, 384)          (B, 196, 384)
+ CLS token             (B, 197, 384)          (B, 197, 384)
Encoder × 12           (B, 197, 384)          (B, 197, 384)
CLS 추출                (B, 384)               (B, 384)
MLP Head               (B, 65536)             (B, 65536)
─── Centering (Teacher만) ───────────────────────────────────
z_t - c                                       (B, 65536)
─── Temperature Softmax ─────────────────────────────────────
softmax(z / τ)          (B, 65536) τ=0.1      (B, 65536) τ=0.04
─── Loss ────────────────────────────────────────────────────
Cross-Entropy           scalar
─── 업데이트 ────────────────────────────────────────────────
역전파                  Student만 ✓            sg: Teacher ✗
EMA                     —                     θ_t ← m·θ_t + (1-m)·θ_s
Center 갱신             —                     c ← λ·c + (1-λ)·batch_mean
```

**학습 결과 — Attention Map 시각화 (슬라이드 Figure 3):**

```
DINO로 학습된 ViT의 [CLS] 토큰 Self-Attention Map:
  → 레이블 없이도 객체 영역을 정확히 분리
  → 각 헤드(head)가 서로 다른 의미 영역에 집중
  (예: head1=몸통, head2=배경, head3=얼굴 등)

추출 방법:
  attention_map = A[:, :, 0, 1:]   ← CLS 토큰(0번)이 패치들(1~196)에 주목하는 가중치
                                       shape: (B, heads, 196)
  reshape → (B, heads, 14, 14)     ← 14×14 = 196 패치 격자
  upsample → (B, heads, 224, 224)  ← 원본 이미지 해상도로 업샘플링
```

```python
import torch
import torch.nn.functional as F

def visualize_dino_attention(model, img, patch_size=16):
    """DINO ViT의 마지막 레이어 CLS Attention Map 시각화"""
    # img: (1, 3, H, W)
    H, W = img.shape[-2:]
    h_patches = H // patch_size   # 예: 224//16 = 14
    w_patches = W // patch_size   # 예: 224//16 = 14

    # 마지막 Encoder 블록의 Attention Weight 추출
    with torch.no_grad():
        # hook 또는 forward에서 attn_weights 수집
        # attn: (1, heads, N+1, N+1)
        attn = model.get_last_attention(img)

    # CLS 토큰(0번)이 패치들(1~N)에 주목하는 가중치
    attn_cls = attn[0, :, 0, 1:]    # (heads, N) = (heads, 196)
    attn_cls = attn_cls.reshape(-1, h_patches, w_patches)  # (heads, 14, 14)

    # 원본 해상도로 업샘플링
    attn_map = F.interpolate(
        attn_cls.unsqueeze(0),       # (1, heads, 14, 14)
        size=(H, W),
        mode='bilinear', align_corners=False
    )[0]                             # (heads, H, W)

    return attn_map  # 각 헤드별 attention map

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

*참고 논문*

- Vaswani et al., _Attention Is All You Need_, NeurIPS 2017  
- Dosovitskiy et al., _An Image Is Worth 16×16 Words_, ICLR 2021  
- Touvron et al., _Training data-efficient image transformers_, 2020  
- Liu et al., _Swin Transformer_, ICCV 2021  
- Radford et al., _Learning Transferable Visual Models From Natural Language Supervision_, ICML 2021  
- Caron et al., _Emerging Properties in Self-Supervised Vision Transformers_, ICCV 2021
