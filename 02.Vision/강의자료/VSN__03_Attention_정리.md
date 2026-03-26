# VSN 03 — Attention & Vision Transformer 정리

> AI Specialist 양성과정 | 강의 슬라이드 + 실습 코드(02_ViT_CIFAR10.ipynb) 통합 정리

---

## 📋 전체 목차 (Hierarchy)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Transformer (기초 리뷰)                                          │
│     1.1  Encoder — Input Embedding & Linear Projection              │
│     1.2  Encoder — Scaled Dot-Product Attention                     │
│     1.3  Attention의 의미 — 값 벡터의 가중합                         │
│     1.4  Global Attention — 전역 수용 필드                           │
│                                                                     │
│  2. Vision Transformer (ViT)                                        │
│     2.1  ViT 등장 배경 — CNN의 한계                                  │
│     2.2  전체 아키텍처 (Big Picture)                                 │
│     2.3  Patch Embedding     — 이미지 → 토큰 시퀀스                  │
│     2.4  Linear Projection   — 패치 → 임베딩 공간                    │
│     2.5  Position Embedding  — 위치 정보 주입                        │
│     2.6  Query / Key / Value — 공간 분해                             │
│     2.7  Self-Attention Map  — 패치 간 유사도                        │
│     2.8  Self-Attention (SA) — 1개 head                             │
│     2.9  Multi-Head SA (MSA) — 병렬 head                            │
│     2.10 Classification      — MLP Head                             │
│     2.11 텐서 흐름 한눈에 보기                                        │
│     2.12 실험 결과                                                   │
│                                                                     │
│  3. ViT 변형 모델들 (Post-ViTs)                                      │
│     3.1  DeiT  — 지식 증류 (Knowledge Distillation)                 │
│     3.2  Swin  — 윈도우 기반 계층적 구조                             │
│     3.3  CLIP  — 이미지-텍스트 대조 학습                             │
│     3.4  DINO  — 레이블 없는 자기 지도 학습                          │
│                                                                     │
│  4. 실습 코드 (02_ViT_CIFAR10.ipynb)                                │
│     4.1  환경 설정 및 데이터 로딩                                    │
│     4.2  ViT 모델 구현 전체                                          │
│     4.3  학습 루프 / 평가                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Transformer (기초 리뷰)

> ViT(Vision Transformer)를 이해하기 위한 NLP Transformer의 핵심 복습

---

### 1.1 Encoder — Input Embedding & Linear Projection

토큰이 세 가지 공간을 순서대로 이동한다.

```
 ╔══════════════╗      ╔══════════════════╗      ╔═════════════════════╗
 ║  Word Space  ║      ║ Embedding Space  ║      ║   Q / K / V Space   ║
 ║              ║      ║                  ║      ║                     ║
 ║  "children"  ║      ║  [0.2, -0.8,    ║      ║  Q: [...]  d_qkv    ║
 ║  → one-hot   ║─WE──▶║   0.5, ...]     ║─WQV─▶║  K: [...]  d_qkv   ║
 ║  (n_vocab,1) ║      ║  (d_embed, 1)   ║      ║  V: [...]  d_qkv   ║
 ╚══════════════╝      ╚══════════════════╝      ╚═════════════════════╝
   이산 정수 토큰         밀집 연속 벡터            어텐션 연산 공간
   (학습 불가)            WE: 학습 파라미터         WQ,WK,WV: 학습 파라미터
```

**행렬 크기 시각화**:
```
                  n_vocab
           ┌───────────────┐
           │               │  d_embedding
d_embedding│      WE       │──────────────▶ [ 임베딩 벡터 (d_emb × 1) ]
           │  (학습 가능)   │
           └───────────────┘
                                    │  WQ  │  WK  │  WV  │   (각 d_emb × d_qkv)
                                    └──────┴──────┴──────┘
                                         ↓       ↓       ↓
                                         Q       K       V    (d_qkv × 1)
```

---

### 1.2 Encoder — Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)\!V$$

**단계별 데이터 흐름**:
```
  Q (N × d_k)         K (N × d_k)         V (N × d_v)
  ┌─────────┐         ┌─────────┐         ┌─────────┐
  │ q₁ ···  │         │ k₁ ···  │         │ v₁ ···  │
  │ q₂ ···  │         │ k₂ ···  │         │ v₂ ···  │
  │   ⋮     │         │   ⋮     │         │   ⋮     │
  │ qN ···  │         │ kN ···  │         │ vN ···  │
  └────┬────┘         └────┬────┘         └────┬────┘
       │                   │                   │
       └──────── × Kᵀ ─────┘                   │
                   │                            │
                   ▼                            │
           QKᵀ  (N × N)                        │
           ┌─────────────┐                     │
           │ q₁·k₁  ··· │                     │
           │ q₂·k₁  ··· │  ÷ √d_k             │
           │    ⋮        │──────────▶          │
           │ qN·k₁  ··· │         │            │
           └─────────────┘         ▼            │
                               Softmax          │
                           ┌─────────────┐      │
                           │ a₁₁  a₁₂ ···│      │
                           │ a₂₁  a₂₂ ···│──────┘ × V
                           │    ⋮         │
                           │ aN₁  aN₂ ···│
                           └──────┬──────┘
                                  │
                                  ▼
                            V'  (N × d_v)   ← 최종 출력
```

> ⚠️ **Attention map은 N×N** — 시퀀스가 길어질수록 메모리·계산이 **O(N²)** 증가

**핵심 코드**:
```python
import math

def scaled_dot_product_attention(Q, K, V):
    d_k   = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (N,N)
    attn   = scores.softmax(dim=-1)                     # (N,N)
    output = attn @ V                                   # (N,d_v)
    return output, attn
```

---

### 1.3 Attention의 의미 — 값 벡터의 가중합

"themselves" 라는 단어를 표현할 때 다른 단어들의 V 벡터를 **중요도 비례로 합산**:

```
  Value 벡터 공간 (2D 단순화)
  ↑
  │     V_N (themselves)              aᴺ·V_N  ← 높은 가중치
  │    ╱                             ╱
  │   ╱    V₁ (Young)           a¹·V₁  ← 낮은 가중치
  │  ╱    ╱                ╱
  │ ╱    ╱    V₂(children)   a²·V₂  ← 중간 가중치
  │╱    ╱    ╱               ╱
  ┼───────────────────────────────────────── →
  │    V₃ (often)         a³·V₃  ← 낮은 가중치

  [① 원래 벡터들]     [② attention 가중]     [③ 가중합 → V'_N]

  V'_N = a¹·V₁ + a²·V₂ + a³·V₃ + ··· + aᴺ·V_N
       = Σᵢ aⁱ·Vᵢ          (Σaⁱ = 1, softmax 보장)
```

**직관**: Attention은 "어떤 단어의 정보를 얼마나 가져올까?"를 학습하는 **가중 평균 메커니즘**

---

### 1.4 Global Attention — 전역 수용 필드

```
  문장: "Young  children  often  solve  problems  by  themselves."
          [1]      [2]     [3]    [4]      [5]    [6]     [7]

  "themselves"(7번) 입장에서 attention:
  ┌──────────────────────────────────────────────────┐
  │                                          ╭───╮   │
  │                              ╭───────────╯   │   │
  │               ╭──────────────╯               │   │
  │  Young  children  often  solve  problems  by  themselves
  │  [낮음]  [높음]  [낮음]  [중간]   [낮음]  [낮음]  [자신]  │
  └──────────────────────────────────────────────────┘
  ↑ 모든 토큰 쌍이 직접 연결 (거리 무관) = Global Receptive Field

  ─────────────────────────────────────────────────────────
        CNN                   Transformer (Attention)
  ─────────────────────────────────────────────────────────
  수용 필드  │ 3×3 kernel (지역)  │  전체 시퀀스 (전역)
  장거리 의존 │ 층 ↑ 해야 획득     │  1 layer로 직접 획득
  위치 불변  │ ✅ (자동)          │  ❌ (위치 임베딩 필요)
  계산 복잡도 │  O(N)             │  O(N²)
  ─────────────────────────────────────────────────────────
```

---

## 2. Vision Transformer (ViT)

> "An Image Is Worth 16×16 Words" — ICLR 2021

---

### 2.1 ViT 등장 배경 — CNN의 한계

```
  데이터·모델 규모 확장에 따른 성능 변화

  정확도
    ↑
 90%│                                        ★ ViT-H (JFT-300M)
    │                             ╭──────────╯
 85%│              ╭──────────────╯  ViT 계속 상승
    │  ╭───────────╯
 80%│──╯  CNN(BiT) → 포화(Saturation)
    │
 75%│
    └────────────────────────────────────────────▶ 데이터 규모
       ImageNet-1K       ImageNet-21K         JFT-300M

  CNN의 Inductive Bias (사전 가정)
  ┌─────────────────────────────────────────────────────┐
  │  Locality          : 3×3 kernel → 지역 패턴만 봄    │
  │  Translation Equiv : 어디서든 같은 패턴 = 같은 응답  │
  │  → 장거리 의존성 약함, 규모 확장 시 포화             │
  └─────────────────────────────────────────────────────┘
            ↓ 핵심 질문
  "Self-Attention으로 Convolution을 완전히 대체할 수 있을까?"
```

---

### 2.2 전체 아키텍처 (Big Picture)

```
  입력 이미지
  ┌─────────────────────────────────────┐
  │  🖼  (3 × 32 × 32)   CIFAR-10 예시  │
  └───────────────┬─────────────────────┘
                  │ ① 패치 분할
                  ▼
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │P₁│P₂│P₃│P₄│··│··│··│P₆₄│  64개 패치 (4×4 픽셀)
  └──┴──┴──┴──┴──┴──┴──┴──┘
                  │ ② Linear Projection
                  ▼
  [p₁E][p₂E][p₃E] ··· [p₆₄E]    N=64 토큰, 각 D=64차원
                  │ ③ CLS 토큰 추가 + 위치 임베딩
                  ▼
  [★CLS][p₁E+pos₁][p₂E+pos₂] ··· [p₆₄E+pos₆₄]   (N+1=65 토큰)
                  │
    ┌─────────────▼──────────────────────────┐
    │   Transformer Encoder  ×  depth(=6)    │
    │ ┌─────────────────────────────────────┐│
    │ │ ┌───────┐                           ││
    │ │ │  LN   │ ← Pre Layer Normalization ││
    │ │ └───┬───┘                           ││
    │ │     ▼                               ││
    │ │ ┌───────────────────────────┐       ││
    │ │ │  Multi-Head Self-Attention│       ││
    │ │ │  (Q,K,V × 4 heads)       │       ││
    │ │ └───────────┬───────────────┘       ││
    │ │             │  + Skip Connection    ││
    │ │         ────┴────                   ││
    │ │ ┌───────┐                           ││
    │ │ │  LN   │                           ││
    │ │ └───┬───┘                           ││
    │ │     ▼                               ││
    │ │ ┌───────────────────────────┐       ││
    │ │ │  FFN: Linear→ReLU→Linear  │       ││
    │ │ └───────────┬───────────────┘       ││
    │ │             │  + Skip Connection    ││
    │ └─────────────┴───────────────────────┘│
    │           (× 6 반복)                   │
    └─────────────┬──────────────────────────┘
                  │ ④ CLS 토큰만 추출
                  ▼
              [★CLS]  (1 × 64)
                  │ ⑤ MLP Head
                  ▼
  ┌────────────────────────────────────┐
  │  LayerNorm → Linear(64, 10)        │
  └────────────────────────────────────┘
                  │
                  ▼
        [0.1, 0.05, 0.7, ···]   ← 10개 클래스 logits
              "plane"  "cat"      → argmax → 예측 클래스
```

**핵심 수식 (전체 흐름)**:
```
z₀   = [x_cls ; x¹_pE ; x²_pE ; ··· ; xᴺ_pE] + E_pos   ← 입력 임베딩
z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}                       ← MSA Block
z_ℓ  = MLP(LN(z'_ℓ))   + z'_ℓ                           ← FFN Block
y    = LN(z⁰_L)                                           ← 분류 출력
```

---

### 2.3 Patch Embedding

32×32 이미지가 4×4 패치 64개로 쪼개지는 과정:

```
  원본 이미지 (32×32)                    패치 그리드 (8×8 = 64개)
  ┌────────────────────────┐            ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │████████████████████████│            │P₁│P₂│P₃│P₄│P₅│P₆│P₇│P₈│
  │████████████████████████│  분할(P=4) │──┼──┼──┼──┼──┼──┼──┼──│
  │████████████████████████│ ─────────▶ │  │  │  │  │  │  │  │  │
  │████████████████████████│            │──┼──┼──┼──┼──┼──┼──┼──│
  │   (C=3, H=32, W=32)   │            │  │  │  │  │  │  │  │  │
  └────────────────────────┘            │  ···   (8행 × 8열)  ···│
                                        │  │  │  │  │  │  │  │  │
                                        └──┴──┴──┴──┴──┴──┴──┴──┘
                                              각 패치: 4×4×3 = 48픽셀

  1개 패치 내부 (P₁, 4×4 RGB):
  ┌────────────────┐
  │R R R R│G G G G│  ← Red 채널 (4×4)
  │R R R R│G G G G│
  │B B B B│        │  ← Blue 채널 (4×4)
  │B B B B│        │
  └────────────────┘
  flatten → [R R R··· G G G··· B B B···]
                       P²×C = 4×4×3 = 48 차원 벡터

  수식:
    x ∈ ℝ^{H×W×C}  →  x_p ∈ ℝ^{N×(P²C)}
    N = HW/P² = (32×32)/(4×4) = 64
```

**코드**:
```python
# (B, 3, 32, 32) → (B, 64, 48) → (B, 64, 64)
to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
    nn.Linear(48, 64)   # patch_dim=48, dim=64
)
```

---

### 2.4 Linear Projection

플랫 패치 벡터(48차원)를 Transformer hidden dim(D=64)으로 투영:

```
  x_p  ∈ ℝ^{N × P²C}         E  ∈ ℝ^{P²C × D}         xE  ∈ ℝ^{N × D}
  ┌──────────────────┐        ┌─────────────┐           ┌──────────┐
  │──────────────────│ P²C=48 │░░░░░░░░░░░░░│ D=64      │──────────│
  │ 패치 1 (flatten) │        │░            │           │ 토큰 1   │
  │──────────────────│        │░  학습 가능  │           │──────────│
  │ 패치 2 (flatten) │  ×     │░  파라미터  │   =       │ 토큰 2   │
  │──────────────────│  N=64  │░  (WE)      │           │──────────│
  │        ⋮         │        │░░░░░░░░░░░░░│           │    ⋮     │
  │──────────────────│        └─────────────┘           │──────────│
  │ 패치 64 (flatten)│                                   │ 토큰 64  │
  └──────────────────┘                                   └──────────┘
      (64 × 48)                   (48 × 64)                (64 × 64)
```

---

### 2.5 Position Embedding

```
  ① 패치 임베딩 결과                     ② CLS 토큰 prepend
  ┌────┬────┬────┬────┬────┐           ┌────┬────┬────┬────┬────┬────┐
  │ p₁E│ p₂E│ p₃E│ ···│p₆₄E│   +CLS → │★CLS│ p₁E│ p₂E│ p₃E│···│p₆₄E│
  └────┴────┴────┴────┴────┘           └────┴────┴────┴────┴────┴────┘
        N=64 토큰 (64 × D)                       N+1=65 토큰 (65 × D)

  ③ + 위치 임베딩 (element-wise 덧셈)
  ┌────┬────┬────┬────┬────┬────┐
  │pos₀│pos₁│pos₂│pos₃│···│pos₆₄│   ← 학습 파라미터 E_pos (65 × D)
  └────┴────┴────┴────┴────┴────┘
          +        +        +  ···
  ┌────┬────┬────┬────┬────┬────┐
  │★CLS│ p₁E│ p₂E│ p₃E│···│p₆₄E│
  └────┴────┴────┴────┴────┴────┘
          =
  ┌────┬────┬────┬────┬────┬────┐
  │ z₀ │ z₁ │ z₂ │ z₃ │···│z₆₄ │   ← Transformer 입력 z₀
  └────┴────┴────┴────┴────┴────┘
         (65 × D=64)

  수식: z₀ = [x_cls ; x¹_pE ; ··· ; xᴺ_pE] + E_pos
```

**[CLS] 토큰의 역할**:
```
  ★ CLS 토큰은 "모든 패치를 바라보며 전역 정보를 수집하는 대표 토큰"

  Attention 학습 후:
  ★CLS ──attend──▶ 모든 패치들 (global context 집약)
   └──────────────────────────────────────────────────▶ MLP Head → 분류
```

**코드**:
```python
cls_token     = nn.Parameter(torch.randn(1, 1, dim))              # (1,1,64)
pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))  # (1,65,64)

cls = repeat(cls_token, '1 1 d -> b 1 d', b=batch_size)  # (B,1,64)
x   = torch.cat([cls, x], dim=1)                          # (B,65,64)
x   = x + pos_embedding                                   # (B,65,64)
```

---

### 2.6 Query / Key / Value 계산

```
  입력 z (B × (N+1) × D)
  예: (B=2, 65, 64)
          │
          │ LayerNorm (Pre-LN)
          │ LN(x) = γ·(x-μ)/√(σ²+ε) + β
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │ Linear(D, 3·D_h·heads) → chunk(3) → reshape per head    │
  │                                                          │
  │  z (65 × 64)  ×  U_qkv (64 × 192)  =  (65 × 192)       │
  │                             ↓                            │
  │                    chunk 3등분 → (65 × 64)씩             │
  │                             ↓                            │
  │              reshape: (65 × 64) → (heads=4, 65, D_h=16) │
  └──────────────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
      Q (4,65,16)   K (4,65,16)   V (4,65,16)
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ head1    │  │ head1    │  │ head1    │
    │ head2    │  │ head2    │  │ head2    │
    │ head3    │  │ head3    │  │ head3    │
    │ head4    │  │ head4    │  │ head4    │
    └──────────┘  └──────────┘  └──────────┘
    각 head가 서로 다른 관점의 유사도를 학습

  D_h = D / heads = 64 / 4 = 16   (head당 차원)
```

---

### 2.7 Self-Attention Map — 패치 간 유사도

```
  A = softmax(q · kᵀ / √D_h)

  q  (N+1, D_h)          kᵀ  (D_h, N+1)
  ┌──────────┐            ┌─────────────────────┐
  │ z_CLS    │            │ k_CLS k₁ k₂ ··· k₆₄│
  │ z₁       │     ×      └─────────────────────┘
  │ z₂       │
  │   ⋮      │
  │ z₆₄      │
  └──────────┘
       = (N+1, N+1) = (65, 65) attention map (1 head)

                        k_CLS  k₁    k₂   ···  k₆₄
  softmax →  q_CLS  ┌─ [0.03  0.08  0.02  ···  0.01 ]  ← CLS가 모든 패치에 attend
             q₁     │  [0.05  0.20  0.03  ···  0.02 ]
             q₂     │  [0.01  0.03  0.15  ···  0.04 ]
              ⋮     │       ⋮
             q₆₄    └─ [0.02  0.01  0.04  ···  0.18 ]

  열합 = 1 (softmax 보장)   → 각 행 = "이 토큰이 다른 토큰에 얼마나 주목하는가"
```

---

### 2.8 Self-Attention (SA) — 1개 head 출력

```
  SA(z) = A · v

  A  (N+1, N+1)          v  (N+1, D_h)
  ┌─────────────────┐    ┌─────────┐
  │ 0.03  0.08  ··· │    │ v_CLS   │
  │ 0.05  0.20  ··· │ ×  │ v₁      │
  │    ⋮            │    │   ⋮     │
  │ 0.02  0.01  ··· │    │ v₆₄     │
  └─────────────────┘    └─────────┘
        (65, 65)             (65, 16)
                   = SA(z)  (65, 16)

  각 출력 토큰 = 입력 V 벡터들의 가중합
  SA(z)₁ = a₁₁·v_CLS + a₁₂·v₁ + a₁₃·v₂ + ··· + a₁,₆₅·v₆₄
```

**코드**:
```python
# Attention.forward() 핵심 부분
dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (B,heads,65,65)
attn = self.attend(dots)                                           # softmax
out  = einsum('b h i j, b h j d -> b h i d', attn, v)            # (B,heads,65,16)
out  = rearrange(out, 'b h n d -> b n (h d)')   # 헤드 합치기: (B,65,64)
```

---

### 2.9 Multi-Head Self-Attention (MSA)

```
  입력 z  (65 × 64)
      │
      ├──────────┬──────────┬──────────┬──────────┐
      │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼
   Head 1     Head 2     Head 3     Head 4    (heads=4)
  (65×16)    (65×16)    (65×16)    (65×16)

  각 head: 서로 다른 Q,K,V 파라미터 → 다른 패턴 학습
  ┌───────────┬───────────┬───────────┬───────────┐
  │SA₁(z)    │SA₂(z)    │SA₃(z)    │SA₄(z)    │
  │(65×16)   │(65×16)   │(65×16)   │(65×16)   │
  └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
        │           │           │           │
        └───────────┴─── concat ┴───────────┘
                         │
                    (65 × 64)   ← k × D_h = 4 × 16 = 64
                         │ × U_msa (64 × 64, 학습 파라미터)
                         ▼
                    MSA(z) (65 × 64)  ← 최종 MSA 출력

  MSA(z) = [SA₁(z) ; SA₂(z) ; SA₃(z) ; SA₄(z)] · U_msa

  왜 Multi-Head?
  ┌──────────────────────────────────────────────────────────────┐
  │ Head 1: 공간적 근접 패치 간 관계 학습 (질감, 경계)           │
  │ Head 2: 색상 일관성 관계 학습 (같은 물체의 다른 패치)        │
  │ Head 3: 전역 맥락 관계 학습 (배경 vs 전경)                  │
  │ Head 4: 의미적 관계 학습 (물체의 부분-전체 관계)             │
  └──────────────────────────────────────────────────────────────┘
```

---

### 2.10 Classification — MLP Head

```
  Transformer 마지막 출력 z_L  (B × 65 × 64)
          │
          │ [★CLS 토큰만 추출] ← z_L[:, 0, :]
          ▼
      (B × 64)   ← 전역 이미지 표현 벡터
          │
          │ LayerNorm
          ▼
      (B × 64)
          │
          │ Linear(64, 10)
          ▼
      (B × 10)   ← 클래스별 로짓

      y = LN(z⁰_L)  →  [0.1, 0.05, 0.8, 0.01, ···]
                          plane  car  bird   dog ...
                                   ↑
                               argmax = "bird" ✔

  [CLS] 토큰이 선택되는 이유:
  ┌───────────────────────────────────────────────────────────┐
  │ Attention을 통해 모든 패치 정보를 집약했기 때문           │
  │ 고정 크기(D차원) → 가변 패치 수와 무관한 분류 가능        │
  │ 학습하며 "좋은 요약" 표현을 자동으로 형성                 │
  └───────────────────────────────────────────────────────────┘
```

**코드**:
```python
x = self.transformer(x)                               # (B, 65, 64)
x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1) # (B, 64)
return self.mlp_head(x)                               # (B, 10)
```

---

### 2.11 텐서 흐름 한눈에 보기 (CIFAR-10, patch=4, dim=64, heads=4)

```
  단계                      Shape              설명
  ─────────────────────────────────────────────────────────────────
  입력 이미지              (B, 3, 32, 32)      RGB 32×32 이미지

  ─── Rearrange ───────────────────────────────────────────────────
  패치 분할                (B, 64, 48)         N=64, P²C=4×4×3=48

  ─── Linear(48→64) ───────────────────────────────────────────────
  패치 임베딩              (B, 64, 64)         D=64

  ─── CLS prepend ─────────────────────────────────────────────────
  + CLS 토큰               (B, 65, 64)

  ─── + pos_embedding ─────────────────────────────────────────────
  + 위치 임베딩            (B, 65, 64)         z₀ 완성

  ═══ Transformer Block × 6 ══════════════════════════════════════
  │  LN                    (B, 65, 64)
  │  to_qkv Linear         (B, 65, 192)        3 × (heads × D_h) = 3×64
  │  chunk → Q,K,V         (B, 65, 64) × 3
  │  reshape per head      (B, 4, 65, 16)      D_h = 64/4 = 16
  │  Q·Kᵀ (attn map)      (B, 4, 65, 65)      ← O(N²) 병목
  │  softmax               (B, 4, 65, 65)
  │  attn·V                (B, 4, 65, 16)
  │  concat heads          (B, 65, 64)
  │  + skip                (B, 65, 64)
  │  LN                    (B, 65, 64)
  │  FFN Linear(64→128)    (B, 65, 128)        mlp_dim=128
  │  FFN Linear(128→64)    (B, 65, 64)
  │  + skip                (B, 65, 64)
  ═══════════════════════════════════════════════════════════════

  CLS 추출                 (B, 64)             z_L[:, 0]
  LN                       (B, 64)
  Linear(64→10)            (B, 10)             ← 최종 logits
  ─────────────────────────────────────────────────────────────────
```

---

### 2.12 실험 결과

**데이터셋 규모와 성능 비교**:
```
  ViT vs CNN (BiT) — 데이터셋 크기별

  Top-1 Acc (%)
    90 │                                        ● ViT-H/14
       │                              ●         ● ViT-L/16
    85 │                    ● ● ●              ● ViT-B/16
       │          ─────────────────────────── ← CNN(BiT) 포화
    80 │─────────/
       │
    75 │
       └────────────────────────────────────▶
          ImageNet-1K     ImageNet-21K    JFT-300M
          (~1.2M)         (~14M)          (~300M)

  ✅  대규모 데이터 → ViT가 CNN을 앞도적으로 능가
  ❌  소규모 데이터 → CNN 우세 (inductive bias 덕분)
```

**ViT 모델 패밀리**:
```
  ┌──────────────┬────────┬───────┬───────┬────────┐
  │    모델      │ dim(D) │ heads │ depth │ Params │
  ├──────────────┼────────┼───────┼───────┼────────┤
  │ ViT-Ti/16    │  192   │   3   │  12   │  5.7M  │
  │ ViT-S/16     │  384   │   6   │  12   │  22M   │
  │ ViT-B/16 ⭐  │  768   │  12   │  12   │  86M   │
  │ ViT-L/16     │ 1024   │  16   │  24   │  307M  │
  │ ViT-H/14     │ 1280   │  16   │  32   │  632M  │
  └──────────────┴────────┴───────┴───────┴────────┘
  ⭐ 기준 모델 (BERT-Large와 동일 크기)
```

**강점 vs 한계**:
```
  ┌─────────────────────────┬────────────────────────────────┐
  │         강점            │            한계                │
  ├─────────────────────────┼────────────────────────────────┤
  │ 전역 패턴 효과적 학습   │ 소규모 데이터 과적합 취약      │
  │ 행렬 연산 → GPU 친화적  │ Attention 비용 O(N²)           │
  │ 대규모 데이터서 CNN 초월│ 대용량 사전학습 필요           │
  │ 다양한 태스크 전이 우수 │ CNN보다 해석이 어려움          │
  └─────────────────────────┴────────────────────────────────┘
```

---

## 3. ViT 변형 모델들 (Post-ViTs)

```
  연구 방향별 주요 모델 타임라인

  2020 ─────────────────────────────────────────────────────── 2022
    │                                                             │
   DeiT          CLIP   DINO       ConvNeXt
   (데이터효율)  Swin   (자기지도)  (CNN 재발명)
                (계층/효율)
    │             │                             │
    ▼             ▼                             ▼
  지식 증류     윈도우 어텐션             Transformer 철학 → CNN
  1.2M 이미지  O(N) 복잡도            CNN과 Transformer 공진화
```

---

### 3.1 DeiT — 지식 증류 (Knowledge Distillation)

**문제**: ViT는 JFT-300M(3억장)이 없으면 성능이 낮음 → 실용적이지 않음

```
  기존 ViT:   ImageNet-1K만으로 학습 → 성능 저하
  DeiT:       강한 CNN 교사(RegNet)에게서 지식을 증류 → 성능 회복
```

**아키텍처 비교**:
```
  ─── 기존 ViT ──────────────────────────────────────────────────
  [★CLS][p₁][p₂]···[pN]  →  Encoder  →  [★CLS]  →  MLP  →  logits
                                          CrossEntropy(GT label)

  ─── DeiT ──────────────────────────────────────────────────────
                                ┌──────────────────────────────┐
                                │    교사 모델 (RegNet, Frozen) │
                                │    강력한 CNN 기반 모델       │
                                └──────────────┬───────────────┘
                                               │ 교사 예측 y_t
  [★CLS][p₁][p₂]···[pN][🎓DIST]              │
         ↑                  ↑                  │
    패치 토큰           증류 토큰               │
                                               │
         │   Encoder   │                       │
         ▼             ▼                       ▼
      [★CLS]        [🎓DIST]              교사 로짓
         │               │
         ▼               ▼
    CrossEntropy    KL-Div (Soft)
    (GT label y)   or CrossEntropy (Hard)
         │               │
         └───── + ────────┘
                 총 손실
```

**손실 함수 비교**:
```
  ─── Hard Distillation ──────────────────────────────────────
  L = ½·CE(student, GT_label) + ½·CE(student, teacher_argmax)
       └── 정답지 기준 ──────┘  └── 교사 최고 예측 기준 ───┘

  ─── Soft Distillation ──────────────────────────────────────
  L = (1-λ)·CE(student, y) + λ·τ²·KL(student/τ, teacher/τ)
                              └── 교사 확률 분포 전체 학습 ──┘
  τ(temperature): 높으면 분포가 부드러워짐 → 더 많은 정보 전달
```

**결과**:
```
  ┌────────────────────────────┬────────────────────────────────┐
  │  방법                      │  ImageNet Top-1 Acc (DeiT-B)   │
  ├────────────────────────────┼────────────────────────────────┤
  │  DeiT (증류 없음)          │  81.8%                         │
  │  DeiT + Hard Distillation  │  83.0%  (+1.2%)                │
  │  DeiT + CNN 교사(RegNet)   │  84.2%  (+2.4%) ← 최고        │
  └────────────────────────────┴────────────────────────────────┘
  → CNN의 Inductive Bias가 Distillation Token을 통해 ViT로 이전됨
```

---

### 3.2 Swin Transformer — 윈도우 기반 계층적 구조

**ViT의 두 가지 문제**:
```
  ① O(N²) 복잡도: 패치 수(N) 증가 시 Attention 비용 폭발
  ② 단일 해상도: CNN처럼 Feature Pyramid 없음 → Detection/Seg 불리
```

**핵심 해결책: Shifted Window Attention**:
```
  ─── W-MSA (Window Multi-Head SA) ──────────────────────────────
  이미지를 고정 윈도우(M×M)로 분할, 각 윈도우 내부에서만 Attention

  ┌────┬────┬────┬────┐
  │    │    │    │    │   각 박스 = 윈도우 (예: 7×7 패치)
  │ W₁ │ W₂ │ W₃ │ W₄ │   윈도우 내부끼리만 Attention
  ├────┼────┼────┼────┤
  │ W₅ │ W₆ │ W₇ │ W₈ │   → 전역 정보 교환 ❌
  └────┴────┴────┴────┘

  ─── SW-MSA (Shifted Window Multi-Head SA) ──────────────────────
  윈도우를 M/2만큼 이동 → 이전 윈도우 경계를 넘어 Attention 가능

  ┌──┬─────┬─────┬──┐
  │  │     │     │  │   경계가 이동 → 인접 윈도우 정보 교환 ✅
  │  │ W'₂ │ W'₃ │  │
  ├──┼─────┼─────┼──┤   → 전역 맥락 간접 획득
  │  │     │     │  │
  └──┴─────┴─────┴──┘

  두 레이어를 번갈아 사용:
  [W-MSA] → [SW-MSA] → [W-MSA] → [SW-MSA] ···
  (지역 attention) (경계 교환) (지역 attention) ...
```

**복잡도 비교**:
```
  ┌──────────────────────┬──────────────────────────┬────────────┐
  │  Attention 종류      │  계산 공식               │  복잡도    │
  ├──────────────────────┼──────────────────────────┼────────────┤
  │ Global MSA (ViT)     │ 4HWC² + 2(HW)²C          │  O((HW)²)  │
  │ Window MSA (Swin)    │ 4HWC² + 2M²HWC            │  O(HW)     │
  └──────────────────────┴──────────────────────────┴────────────┘
  M = 윈도우 크기 (상수) → HW가 커져도 선형 증가!
```

**계층적 구조 (Feature Pyramid)**:
```
  입력 (H×W×3)
      │  4×4 패치 분할 + Linear Embedding
      ▼
  Stage 1:  H/4  × W/4  × C     ← Swin Block ×2
      │  Patch Merging (2×2 합치기, 채널 2배)
      ▼
  Stage 2:  H/8  × W/8  × 2C    ← Swin Block ×2
      │  Patch Merging
      ▼
  Stage 3:  H/16 × W/16 × 4C    ← Swin Block ×6
      │  Patch Merging
      ▼
  Stage 4:  H/32 × W/32 × 8C    ← Swin Block ×2
      │
      ▼
  분류: Global Average Pool → Linear
  Detection/Segmentation: FPN Head 연결 가능 ← ViT와의 핵심 차이
```

**ViT vs Swin 종합 비교**:
```
  ┌──────────────────┬────────────────┬────────────────────────┐
  │  특성            │     ViT        │     Swin               │
  ├──────────────────┼────────────────┼────────────────────────┤
  │  구조            │  Flat (단일)   │  Hierarchical (4단계)  │
  │  Attention       │  Global        │  Window + Shifted      │
  │  복잡도          │  O((HW)²)      │  O(HW)                 │
  │  적합 태스크     │  분류          │  분류/검출/분할 모두   │
  │  위치 인코딩     │  학습 가능     │  상대 위치 편향        │
  └──────────────────┴────────────────┴────────────────────────┘
```

---

### 3.3 CLIP — 이미지-텍스트 대조 학습

**문제**: 수동 레이블 = 비싸고 제한적, Zero-shot 불가

**핵심 아이디어**: 인터넷의 이미지+텍스트 쌍(4억 개)을 그대로 학습 신호로 사용

```
  ─── 학습 단계 ──────────────────────────────────────────────────
  미니배치 N=4 예시:

  텍스트         Text Encoder    임베딩 (d차원)
  "강아지 사진"  ──────────────▶  T₁ ◉
  "고양이 사진"  ──────────────▶  T₂ ◉
  "자동차 사진"  ──────────────▶  T₃ ◉
  "새 사진"      ──────────────▶  T₄ ◉

                                   ↕ ↕ ↕ ↕  (내적으로 유사도 계산)

  이미지         Image Encoder   임베딩 (d차원)
  🐕 강아지      ──────────────▶  I₁ ◉
  🐈 고양이      ──────────────▶  I₂ ◉
  🚗 자동차      ──────────────▶  I₃ ◉
  🐦 새          ──────────────▶  I₄ ◉

  유사도 행렬 (N×N):
         T₁    T₂    T₃    T₄
  I₁  [ 높음  낮음  낮음  낮음 ]  ← 강아지+강아지 → ↑
  I₂  [ 낮음  높음  낮음  낮음 ]  ← 고양이+고양이 → ↑
  I₃  [ 낮음  낮음  높음  낮음 ]
  I₄  [ 낮음  낮음  낮음  높음 ]
       대각선(매칭쌍): 유사도 최대화 ↑
       비대각선(불일치): 유사도 최소화 ↓

  InfoNCE Loss:
  L = -1/N · Σᵢ log[ exp(sim(Iᵢ,Tᵢ)/τ) / Σⱼexp(sim(Iᵢ,Tⱼ)/τ) ]
```

**Zero-shot 추론**:
```
  ─── 추론 단계 (레이블 없이!) ───────────────────────────────────
  ① 클래스 이름을 프롬프트로 변환
     "plane" → "A photo of a plane."
     "car"   → "A photo of a car."
     "dog"   → "A photo of a dog."
                    │ Text Encoder
                    ▼
     T_plane, T_car, T_dog, ···  (n_class × d)

  ② 테스트 이미지 임베딩
     🐕이미지 → Image Encoder → I  (1 × d)

  ③ 유사도로 분류
     [I·T_plane, I·T_car, I·T_dog, ···]
     = [0.10,     0.08,    0.85,  ···]
                              ↑
                           argmax = "dog" ✔

  → 새로운 클래스도 재학습 없이 즉시 분류 가능! (Zero-shot)
```

---

### 3.4 DINO — 레이블 없는 자기 지도 학습

**목표**: 레이블 없이 ViT가 물체 분할을 자동으로 학습

```
  학습 후 CLS 어텐션 맵이 물체 경계를 자동으로 분리:
  [원본 기린 이미지] → [CLS attention] → [기린 마스크] (레이블 없이!)
```

**Student-Teacher 프레임워크**:
```
  원본 이미지  x
       │
       │ 랜덤 증강 (2가지 이상의 crop)
       ├──────────────────────────────┐
       ▼                              ▼
  x₁ (글로벌 뷰: >50%)          x₂ (글로벌+로컬 뷰)
       │                              │
       ▼                              ▼
  ┌──────────────┐           ┌──────────────────┐
  │  Student     │           │   Teacher        │
  │  g_θs (ViT) │           │   g_θt (ViT)     │
  └──────┬───────┘           └────────┬─────────┘
         │                            │
         │ softmax(높은 τ=0.1)        │ centering + softmax(낮은 τ=0.04)
         │ → 완만한 분포              │ → 날카로운 분포 (고확신 pseudo-label)
         ▼                            ▼
        p₁                           p₂
         │                            │ stop-gradient ⛔
         └──── CrossEntropy(p₁, p₂) ──┘
                       │
                       ▼ 역전파 (Student만!)
              Student 파라미터 업데이트

  Teacher 업데이트: EMA (역전파 없음)
  θ_teacher ← m·θ_teacher + (1-m)·θ_student    m ≈ 0.996 (느리게 이동)
```

**각 기법의 역할**:
```
  ┌──────────────────────┬────────────────────────────────────────┐
  │   기법               │  역할                                  │
  ├──────────────────────┼────────────────────────────────────────┤
  │  EMA Teacher         │  안정적 학습 타겟, collapse 방지        │
  │  Centering           │  teacher 출력 편향 제거 → 붕괴 방지    │
  │  고온도(Student)     │  분포 완만 → 다양한 패턴 학습          │
  │  저온도(Teacher)     │  고확신 soft-label 생성                │
  │  Stop-gradient       │  teacher로 gradient 흐름 차단          │
  │  Multi-crop          │  로컬→글로벌 대응 관계 학습             │
  └──────────────────────┴────────────────────────────────────────┘
```

**Collapse 방지 메커니즘**:
```
  Centering 없으면:
  teacher → 항상 같은 차원(예: 차원 0)만 큰 값
  → student도 항상 같은 출력 → 모든 이미지가 같아 보임 (collapse)

  Centering:
  z̃_teacher = z_teacher - c     c = running mean of teacher outputs
  → 특정 차원에 편향 없이 균등한 학습 신호 유지
```

---

## 4. 실습 코드 (02_ViT_CIFAR10.ipynb)

### 4.1 환경 설정 및 데이터 로딩

```python
import torch, torchvision
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ─── CIFAR-10 정규화 값 (채널별 평균/표준편차) ─────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ─── Train: 증강 포함 / Test: 정규화만 ──────────────────────────────
train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),    # 랜덤 crop (데이터 증강)
    T.RandomHorizontalFlip(),       # 좌우 반전
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_tfms = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

train_loader = DataLoader(
    torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tfms),
    batch_size=256, shuffle=True, num_workers=8
)
test_loader = DataLoader(
    torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tfms),
    batch_size=256, shuffle=False, num_workers=8
)
# 입력 텐서 shape: (B, 3, 32, 32)
```

---

### 4.2 ViT 모델 구현 전체

#### ① PreNorm — Pre-Layer Normalization

```
  Post-LN (기존):  Attention → + → LN    (학습 불안정)
  Pre-LN  (ViT):  LN → Attention → +     (학습 안정 ✅)
```

```python
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 토큰 차원(dim) 기준 정규화
        self.fn   = fn                  # Attention 또는 FFN

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # LN 먼저, fn 나중
```

#### ② FeedForward — FFN (MLP Block)

```
  입력 (dim=64)
      │ Linear(64→128)   ← 차원 확장
      │ ReLU
      │ Dropout
      │ Linear(128→64)   ← 차원 복원
      │ Dropout
      ▼ 출력 (dim=64)
```

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
```

#### ③ Attention — Multi-Head Self-Attention

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads            # 전체 head 차원 (4×16=64)

        self.heads   = heads
        self.scale   = dim_head ** -0.5         # 1/√D_h (스케일링)
        self.attend  = nn.Softmax(dim=-1)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)  # Q,K,V 동시 변환
        self.to_out  = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # x: (B, N+1, D)  예: (2, 65, 64)
        b, n, _, h = *x.shape, self.heads

        # ① Q,K,V 한번에 선형 변환 후 분리
        qkv = self.to_qkv(x).chunk(3, dim=-1)            # 3 × (B,65,64)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv
        )  # 각각 (B, 4, 65, 16)

        # ② Scaled Dot-Product
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (B,4,65,65)
        attn = self.attend(dots)            # softmax → (B,4,65,65)

        # ③ 가중합
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (B,4,65,16)
        out = rearrange(out, 'b h n d -> b n (h d)')           # (B,65,64)
        return self.to_out(out)             # (B,65,64)
```

#### ④ Transformer — Encoder Block × depth

```python
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):              # depth=6번 반복
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x    # MSA + Skip Connection
            x = ff(x)   + x    # FFN + Skip Connection
        return x
```

#### ⑤ ViT 본체 — Config + Model

```python
from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size:  int   = 32     # H=W=32 (CIFAR-10)
    patch_size:  int   = 4      # P=4 → N=(32/4)²=64개 패치
    channels:    int   = 3      # RGB
    dim:         int   = 64     # 토큰 임베딩 차원 D
    depth:       int   = 6      # Encoder 블록 수
    heads:       int   = 4      # MHSA head 수
    dim_head:    int   = 64     # head당 차원 D_h
    mlp_dim:     int   = 128    # FFN 중간 차원
    num_classes: int   = 10     # CIFAR-10 클래스 수
    pool:        str   = "cls"  # CLS 토큰으로 분류
    dropout:     float = 0.0
    emb_dropout: float = 0.0


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        ph, pw = cfg.patch_size, cfg.patch_size
        num_patches = (cfg.image_size // ph) * (cfg.image_size // pw)  # 64
        patch_dim   = cfg.channels * ph * pw                            # 48

        # ─── ① 패치 토큰화 + 선형 투영 ──────────────────────────────
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.Linear(patch_dim, cfg.dim)
        )

        # ─── ② CLS 토큰 + 위치 임베딩 (학습 파라미터) ──────────────
        self.cls_token     = nn.Parameter(torch.randn(1, 1, cfg.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.dim))
        self.dropout       = nn.Dropout(cfg.emb_dropout)

        # ─── ③ Transformer Encoder ──────────────────────────────────
        self.transformer = Transformer(cfg.dim, cfg.depth, cfg.heads,
                                       cfg.dim_head, cfg.mlp_dim, cfg.dropout)
        self.pool = cfg.pool

        # ─── ④ 분류 헤드 ────────────────────────────────────────────
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)    # (B,3,32,32) → (B,64,64)
        b, n, _ = x.shape

        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x   = torch.cat([cls, x], dim=1)   # (B,65,64)
        x   = x + self.pos_embedding        # (B,65,64)
        x   = self.dropout(x)

        x   = self.transformer(x)           # (B,65,64)
        x   = x[:, 0]                       # (B,64)  ← CLS 추출
        return self.mlp_head(x)             # (B,10)
```

---

### 4.3 학습 루프 / 평가

```python
# ─── 모델 초기화 ────────────────────────────────────────────────────
vit_cfg   = ViTConfig(image_size=32, patch_size=4, dim=64, depth=6,
                      heads=4, dim_head=64, mlp_dim=128, num_classes=10)
model     = ViT(vit_cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)


# ─── 학습 루프 ──────────────────────────────────────────────────────
def train_epoch(model, optimizer, loader):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)   # log 확률
        loss   = F.nll_loss(output, target)           # NLL = CrossEntropy
        loss.backward()
        optimizer.step()
        # ※ F.log_softmax + F.nll_loss = nn.CrossEntropyLoss 와 동일


# ─── 평가 ───────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, len(loader.dataset)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output      = F.log_softmax(model(data), dim=1)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            correct    += output.argmax(dim=1).eq(target).sum().item()

    print(f"Loss: {total_loss/n:.4f}  Acc: {correct/n*100:.2f}%")
```

**학습 루프 흐름**:
```
  for epoch in range(1, num_epochs+1):
      ┌─── train_epoch ──────────────────────────────────────┐
      │  for batch in train_loader:                          │
      │    zero_grad → forward → loss → backward → step     │
      └──────────────────────────────────────────────────────┘
      ┌─── evaluate ─────────────────────────────────────────┐
      │  for batch in test_loader:                           │
      │    with no_grad: forward → loss, acc 누적            │
      └──────────────────────────────────────────────────────┘
      if best_acc 갱신: torch.save(checkpoint)
```

---

## 📌 핵심 개념 최종 요약

```
  ViT 학습의 3가지 핵심 요소

  ┌──────────────────────────────────────────────────────────────────┐
  │  1. 패치 토큰화 (Patch Tokenization)                             │
  │     이미지 → 패치 시퀀스 → 선형 투영 → NLP Transformer와 동일   │
  │                                                                  │
  │  2. 전역 Self-Attention                                          │
  │     모든 패치 쌍이 서로를 직접 참조 → 장거리 의존성 1 layer 획득│
  │     단, O(N²) 복잡도가 확장성 병목                              │
  │                                                                  │
  │  3. CLS 토큰                                                     │
  │     전역 이미지 표현을 집약하는 대표 토큰 → MLP Head로 분류      │
  └──────────────────────────────────────────────────────────────────┘

  Post-ViT 흐름 요약

  ViT → DeiT  : 데이터 효율 (지식 증류)
  ViT → Swin  : 계산 효율 (윈도우 어텐션) + 범용성 (계층 구조)
  ViT → CLIP  : 감독 신호 확장 (이미지+텍스트)
  ViT → DINO  : 레이블 제거 (자기 지도)
```

**개념-코드 대응표**:

| 개념 | 수식/설명 | 코드 |
|------|-----------|------|
| Patch Embedding | x → x_p E | `Rearrange + Linear` |
| CLS Token | x_class prepend | `cls_token + torch.cat` |
| Position Embedding | + E_pos | `pos_embedding + x` |
| Pre-LN | LN → fn | `PreNorm` 클래스 |
| Scaled Dot-Product | QKᵀ/√d | `einsum + scale` |
| Multi-Head | 병렬 head | `rearrange(h d)` |
| Skip Connection | x = f(x) + x | `attn(x) + x` |
| MLP Head | LN → Linear | `mlp_head` |

---

## 📚 참고 논문

| 논문 | 학회 | 연도 |
|------|------|------|
| Attention Is All You Need | NeurIPS | 2017 |
| An Image Is Worth 16×16 Words (ViT) | ICLR | 2021 |
| Training data-efficient image transformers (DeiT) | ICML | 2021 |
| Swin Transformer | ICCV | 2021 |
| Learning Transferable Visual Models (CLIP) | ICML | 2021 |
| Emerging Properties in Self-Supervised ViTs (DINO) | ICCV | 2021 |
