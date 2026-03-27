# 01. 핵심 개념 정리

> 초보자도 이해할 수 있도록 예시 중심으로 정리한 LLM 핵심 개념 모음

-----

## 📌 필수 수학 개념

### 활성화 함수 (Activation Function)

* **왜 필요한가?** 선형 함수(직선)만 쌓으면 아무리 깊어도 결국 직선 하나와 같음 → 복잡한 패턴 학습 불가
* 활성화 함수로 **비선형성**을 부여 → 구불구불한 복잡한 경계도 학습 가능

```
비유: 전구 스위치
  선형만 사용: 밝기를 조금씩만 조절 (어두움 ↔ 밝음, 중간값만)
  활성화 함수: "일정 밝기 이하면 꺼라" 같은 규칙 → 훨씬 다양한 표현 가능

주요 활성화 함수:
  ReLU:    f(z) = max(0, z)      ← 음수는 0, 양수는 그대로 (가장 많이 사용)
  Sigmoid: σ(z) = 1/(1+e⁻ᶻ)    ← 0~1 사이 (이진 분류)
  GELU:    GPT에서 사용, ReLU의 부드러운 버전 (FFN 내부에서 사용)
```

```python
import torch
import torch.nn.functional as F

z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

print(F.relu(z))          # [0.0, 0.0, 0.0, 1.0, 3.0]  ← 음수는 모두 0
print(torch.sigmoid(z))   # [0.05, 0.27, 0.5, 0.73, 0.95] ← 0~1 사이
print(F.gelu(z))          # [-0.004, -0.159, 0.0, 0.841, 2.996]
```

---

### Softmax

* **목적**: 숫자 나열을 **합이 1인 확률 분포**로 변환
* 지수 함수(eˣ)를 사용 → 작은 차이를 극적으로 벌려줌 → 가장 큰 값이 두드러짐

```
예시: 다음 단어 후보 점수
  "cat"   → 2.0
  "dog"   → 1.0
  "pizza" → 0.1

Softmax 적용 후:
  "cat"   → e²·⁰ / (e²·⁰ + e¹·⁰ + e⁰·¹) = 7.39 / 10.23 ≈ 0.659
  "dog"   → e¹·⁰ / 10.23 ≈ 0.242
  "pizza" → e⁰·¹ / 10.23 ≈ 0.099
  합계: 0.659 + 0.242 + 0.099 = 1.0 ✓

→ "cat"이 가장 높은 확률 → 다음 단어로 선택
```

```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()   # 합이 1이 되도록 정규화

scores = np.array([2.0, 1.0, 0.1])
print(softmax(scores))   # [0.659, 0.242, 0.099]
```

---

### Sigmoid

* **목적**: 하나의 값을 0~1 사이로 변환 → "예/아니오" 확률로 해석
* 이진 분류(스팸 메일인가? 긍정 리뷰인가?)에 사용

```
σ(z) = 1 / (1 + e⁻ᶻ)

  z 매우 큰 값  → 1.0  (예)
  z = 0        → 0.5  (반반)
  z 매우 작은 값 → 0.0  (아니오)

예시: 이메일 스팸 분류
  모델 출력 z = 3.0  → σ(3.0) = 0.95 → 95% 확률로 스팸!
  모델 출력 z = -2.0 → σ(-2.0) = 0.12 → 12% 확률로 스팸 (정상 메일)
```

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(3.0))    # 0.953  ← 스팸일 확률 95%
print(sigmoid(0.0))    # 0.5    ← 판단 불가
print(sigmoid(-2.0))   # 0.119  ← 스팸 아닐 확률 높음
```

> 💡 클래스가 2개인 Softmax = Sigmoid (수학적으로 동일)

---

### Cross Entropy

* **목적**: 모델의 예측이 정답과 얼마나 다른가를 수치화 → **클수록 나쁜 예측**
* LLM 학습의 핵심 Loss 함수 — 이 값을 줄이는 방향으로 학습

```
비유: 기상예보관 시험
  실제 날씨: [맑음=1.0, 흐림=0.0, 비=0.0]

  예보관 A: [0.9, 0.05, 0.05] ← 거의 정확 → Cross Entropy 낮음 (좋은 예측)
  예보관 B: [0.1, 0.5,  0.4 ] ← 완전히 틀림 → Cross Entropy 높음 (나쁜 예측)

공식: H(P, Q) = -Σ P(x) × log Q(x)
  P = 실제 정답 분포
  Q = 모델의 예측 분포
```

```python
def cross_entropy(p_true, q_pred):
    # -Σ P(x) log Q(x)
    return -np.sum(p_true * np.log(q_pred + 1e-10))  # 1e-10: log(0) 방지

p_true = np.array([1.0, 0.0, 0.0])   # 실제 정답: 맑음
q_A    = np.array([0.9, 0.05, 0.05]) # 예보관 A 예측 (좋음)
q_B    = np.array([0.1, 0.5,  0.4])  # 예보관 B 예측 (나쁨)

print(f"A의 Cross Entropy: {cross_entropy(p_true, q_A):.3f}")  # 0.105 (낮음)
print(f"B의 Cross Entropy: {cross_entropy(p_true, q_B):.3f}")  # 2.303 (높음)
```

---

### KL-Divergence

* **Cross Entropy에서 실제 데이터 자체의 불확실성(Entropy)을 뺀 값**
* "모델이 추가로 틀린 만큼의 순수한 벌점"
* RLHF, DPO 등 LLM 파인튜닝 과정에서 자주 등장

```
공식: KL(P||Q) = Cross Entropy(P, Q) - Entropy(P)

직관:
  Cross Entropy: 모델의 전체 불확실성
  Entropy(P):    데이터 자체가 갖는 불확실성 (모델이 어쩔 수 없는 부분)
  KL-Divergence: 모델이 추가로 틀린 만큼 (진짜 실력 차이)

예시:
  주사위 6면체 → 실제: 각 면 1/6 (완전 균등)
  모델이 [0.5, 0.1, 0.1, 0.1, 0.1, 0.1] 로 예측하면
  KL-Divergence가 커짐 → 이 차이를 줄이도록 학습
```

```python
def kl_divergence(p_true, q_pred):
    # KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
    return np.sum(p_true * np.log((p_true + 1e-10) / (q_pred + 1e-10)))

p = np.array([1/6] * 6)                           # 실제: 균등 분포
q = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])     # 예측: 편향된 분포

print(f"KL-Divergence: {kl_divergence(p, q):.3f}")  # 클수록 나쁜 예측
```

-----

## 📌 언어 모델 (Language Model)

### Encoder — 문맥 이해, 요약

* 입력 문장을 읽고 **의미와 문맥을 압축**한 벡터(context vector)를 만드는 역할
* 양방향으로 문장 전체를 참조 → 앞뒤 문맥 모두 이해
* 대표 모델: **BERT**

```
비유: 독해 전문가
  입력: "Je suis étudiant" (프랑스어: 나는 학생이다)
  Encoder: 이 문장 전체를 읽고 의미를 압축
  출력: context vector (압축된 의미 덩어리)

Encoder 내부:
  입력 → [Self-Attention] → [FFN] → context vector
  (문장 전체를 한 번에 보고 각 단어의 의미를 파악)
```

---

### Decoder — 맥락을 바탕으로 생성

* context vector를 받아 **새로운 텍스트를 한 단어씩 생성**하는 역할
* 앞쪽 문맥만 참조 (아직 생성하지 않은 미래 단어는 볼 수 없음)
* 대표 모델: **GPT**

```
비유: 작가
  입력: context vector + 이미 생성한 단어들
  Decoder: 다음에 올 가장 자연스러운 단어를 예측

  "I"     → 다음 예측 → "am"
  "I am"  → 다음 예측 → "a"
  "I am a"→ 다음 예측 → "student"
  ...한 단어씩 순서대로 생성!
```

```
                   context vector
INPUT ──→ [ENCODER] ───────────────→ [DECODER] ──→ OUTPUT
"Je suis   (양방향 이해,                 (단방향 생성,   "I am
étudiant"   의미 압축)                   한 단어씩)      a student"
```

---

### RNN (Recurrent Neural Network)

* 단어를 **순서대로 하나씩** 처리하면서 이전 정보를 기억하는 구조
* hidden state(기억)를 다음 단계로 전달

```
비유: 릴레이 달리기
  각 주자가 배턴(기억)을 받아서 달리고, 다음 주자에게 전달
  → 앞 주자의 정보가 계속 이어짐

"I"      → [RNN] → h₁ (I를 기억)
"am"     → [RNN] → h₂ (I + am을 기억)
"a"      → [RNN] → h₃ (I + am + a를 기억)
"student"→ [RNN] → h₄ → 최종 출력
```

```python
# RNN pseudo code
def rnn_step(x_t, h_prev, Wx, Wh, b):
    # x_t: 현재 단어 임베딩, h_prev: 이전 기억
    h_t = np.tanh(Wx @ x_t + Wh @ h_prev + b)
    return h_t

h = np.zeros(hidden_size)           # 초기 기억: 없음
for word in ["I", "am", "a", "student"]:
    x = embedding(word)
    h = rnn_step(x, h, Wx, Wh, b)  # 이전 기억 + 현재 단어 → 새 기억
```

* **단점**
  * 문장이 길수록 초반 정보를 잊어버림 (긴 문장의 첫 단어를 기억 못 함)
  * 순서대로 처리 → 병렬 처리 불가 → 느림

---

### RNN + Attention

* RNN의 "먼 단어를 잊는" 단점 보완
* **어떤 단어에 주목할지**를 직접 학습 → 먼 단어도 직접 참조 가능

```
번역 시: "Tom이 고기를 먹었다" → "Tom ate the meat"

Decoder가 "Tom" 생성 시:
  일반 RNN:     맨 마지막 hidden만 참고 (Tom 정보가 희미)
  RNN+Attention: "Tom"에 해당하는 Encoder hidden을 직접 참고!

  Attention 가중치:
    "Tom"   → 0.85 ← 높음 (지금 "Tom" 번역 중!)
    "고기를" → 0.10
    "먹었다" → 0.05
```

* **단점**: 여전히 순차 처리, 메모리 증가, hidden에 다른 정보가 섞임

---

### Transformer: Attention + FFN

* RNN 없이 **Attention만으로** 문맥 파악 + 모든 단어를 **동시에(병렬)** 처리
* 현재 모든 LLM(GPT, BERT, Claude 등)의 기반 구조

```
RNN (순차):                       Transformer (병렬):
time → [RNN] →                    ┌──────────────────────┐
flies→ [RNN] →                    │  Attention + FFN      │
like → [RNN] →                    │  time flies like an   │
an   → [RNN] → 출력               └──────────────────────┘
                                    ↑ 모든 단어를 동시에!

"RNN is hard to parallelize and not time-efficient!"
```

```
Transformer Block 구조 (아래 그림 참조):

  입력
   │
   ▼
[층 정규화 1]           ← Layer Normalization
   │
   ▼
[마스크드 멀티헤드 어텐션]  ← Masked Multi-head Attention
   │
   + ← 숏컷 연결 (Residual Connection, 입력을 그대로 더함)
   │
[층 정규화 2]
   │
   ▼
[피드 포워드]            ← FFN: 선형 → GELU → 선형
   │
   + ← 숏컷 연결
   │
  출력
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn       = FeedForward(d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Attention + 숏컷 연결
        x = x + self.attention(self.norm1(x), mask)
        # 2. FFN + 숏컷 연결
        x = x + self.ffn(self.norm2(x))
        return x
```

---

### BERT

* **Encoder만** 사용 → 문장을 **이해**하는 데 특화
* 양방향 학습(MLM): 빈칸 채우기 → 앞뒤 문맥 모두 참조

```
학습 방식 (Masked Language Model):
  원문:     "나는 [MASK] 강의를 수강합니다"
  정답:     "딥러닝"  ← 앞뒤 문맥 모두 보고 예측

주요 활용: 문서 분류, 감성 분석, 질의응답, 검색
→ 생성보다는 "이해"가 필요한 작업에 강함
```

---

### GPT

* **Decoder만** 사용 → 텍스트를 **생성**하는 데 특화
* 단방향 학습(CLM): 다음 단어 맞추기 → 앞쪽 문맥만 참조

```
학습 방식 (Causal Language Model):
  입력:    "나는 이번 학기에 딥러닝 강의를"
  정답:    "수강합니다"  ← 앞 단어들만 보고 다음 단어 예측

주요 활용: 텍스트 생성, 대화, 코드 작성, 번역
→ "이해"보다 "생성"이 필요한 작업에 강함

GPT 모델 전체 구조 (이미지 참조):
  토큰화된 텍스트
       ↓
  [임베딩 층] (Token Embedding + Positional Encoding)
       ↓
  [트랜스포머 블록] × N  (마스크드 멀티헤드 어텐션 + FFN)
       ↓
  [출력 층] (Linear + Softmax → 다음 토큰 확률)
       ↓
  "Every effort moves you forward"
```

| | BERT | GPT |
|--|------|-----|
| **구조** | Encoder only | Decoder only |
| **학습** | MLM (빈칸 채우기) | CLM (다음 단어) |
| **문맥** | 양방향 | 단방향 |
| **용도** | 이해 (분류, 검색) | 생성 (글쓰기, 대화) |

-----

## 📌 Data 입력

### Embedding (임베딩)

* **단어(Token)를 숫자 벡터로 변환**하는 것
* 비슷한 의미의 단어는 벡터 공간에서 가까운 위치에 있음 → 의미 연산 가능

```
변환 예시:
  "cat"    → [0.2,  0.8, -0.3,  0.1, ...]  ← 768차원 벡터
  "dog"    → [0.3,  0.7, -0.2,  0.1, ...]  ← "cat"과 유사!
  "king"   → [0.9,  0.1,  0.5, -0.2, ...]
  "queen"  → [0.9,  0.1,  0.5,  0.6, ...]  ← "king"과 유사!
  "pizza"  → [-0.1, 0.3,  0.9,  0.2, ...]  ← cat/dog와 거리가 멈

의미 연산:
  king - man + woman ≈ queen  (벡터 뺄셈/덧셈으로 의미 추론!)
```

```python
import torch.nn as nn

# 어휘 사전의 각 Token에 벡터를 하나씩 할당
embedding = nn.Embedding(num_embeddings=50000,  # 어휘 사전 크기
                         embedding_dim=768)     # 벡터 차원 수

token_ids = torch.tensor([40, 716, 257])        # "I", "am", "a"의 Token ID
vectors   = embedding(token_ids)
print(vectors.shape)   # [3, 768] ← 3개 토큰 × 768차원
```

---

### Vector 공간

* 단어들이 위치하는 **고차원 수학 공간**
* 의미적으로 비슷한 단어끼리 공간에서 가까이 위치

```
2D로 단순화한 벡터 공간 예시:

     ↑ 왕족 속성
     │
  왕(King) ─── 여왕(Queen)
     │              │
     │          여성 방향 →
     │
  남자(Man) ─── 여자(Woman)
     │
     └──────────────────→ 성별 속성

관계 보존:
  King - Man + Woman ≈ Queen
  Paris - France + Japan ≈ Tokyo

내적(Dot Product)으로 유사도 측정:
  비슷한 벡터 → 내적 값 큼(+)
  다른 벡터   → 내적 값 작거나 음수
```

```python
# 단어 벡터 간 유사도 측정
cat   = np.array([0.2,  0.8, -0.3])
dog   = np.array([0.3,  0.7, -0.2])
pizza = np.array([-0.1, 0.3,  0.9])

sim_cat_dog   = np.dot(cat, dog)    # 크다 → 유사함
sim_cat_pizza = np.dot(cat, pizza)  # 작다 → 다름
print(f"cat-dog 유사도: {sim_cat_dog:.3f}")    # 높은 값
print(f"cat-pizza 유사도: {sim_cat_pizza:.3f}") # 낮은 값
```

---

### Tokenizer — BPE, Wordpiece, Unigram

* 텍스트를 모델이 처리할 수 있는 **Token 조각으로 분리**하고 고유 번호(ID) 부여
* 원칙: 자주 쓰이는 단어는 길게(통째로), 드문 단어는 조각조각 분리

#### BPE Tokenizer (GPT 계열) — fastest
* 가장 많이 등장하는 **글자 쌍을 합쳐** 어휘를 구성

```
예시: "low", "lower", "lowest" 처리

초기: ["l","o","w"] ["l","o","w","e","r"] ["l","o","w","e","s","t"]

Step 1: "lo"가 가장 많이 등장 → "lo" 하나로 합침
Step 2: "low"가 가장 많이 등장 → "low" 하나로 합침
...반복...

결과: "low" / "low"+"er" / "low"+"est"  ← 압축됨!

장점: 빠름, GPT-2/3/4 등 GPT 계열에서 사용
```

```python
# BPE pseudo code
def bpe(text, num_merges):
    tokens = list(text)                       # 초기: 글자 하나씩
    for _ in range(num_merges):
        pairs = count_pairs(tokens)           # 모든 인접 쌍의 빈도 계산
        best  = max(pairs, key=pairs.get)     # 가장 많이 등장한 쌍
        if pairs[best] < 2:
            break
        tokens = merge_pair(tokens, best)     # 그 쌍을 하나로 합침
    return tokens
```

#### Wordpiece Tokenizer (BERT) — 중간 속도
* 각각 등장할 확률보다 **함께 등장할 확률이 더 높은 쌍**을 합침

```
판단 기준:
  P("ng"가 쌍으로 등장)
  ──────────────────────  > 1  이면 → "ng"를 하나의 Token으로
  P("n") × P("g")

예시: "playing" → ["play", "##ing"]
  "##"는 앞에 다른 단어가 있음을 표시 (BERT 규칙)

단순 빈도가 아닌, '함께 나올 때 의미가 생기는 쌍'을 찾음
```

#### Unigram Tokenizer (SentencePiece) — slowest
* 가능한 모든 조각 조합 중 **확률이 가장 높은 조합을 선택**, 쓸모없으면 제거

```
"playing" 처리:
  후보 1: "play" + "ing"        확률: 0.7  → 선택!
  후보 2: "pl" + "ay" + "ing"   확률: 0.2
  후보 3: "p" + "l" + "a" + ... 확률: 0.1

특징:
  ✓ 확률에 따라 토큰화 결과를 조금씩 다르게 샘플링 가능 (데이터 증강)
  ✓ 통계적으로 효율적인 어휘 사전 구축
  ✗ 속도가 느림
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")   # BPE 기반

text   = "I am a student learning deep learning"
tokens = tokenizer.tokenize(text)
ids    = tokenizer.encode(text)

print(tokens)   # ['I', 'Ġam', 'Ġa', 'Ġstudent', 'Ġlearning', ...]
print(ids)      # [40, 716, 257, 3710, 4673, ...]
```

| Tokenizer | 방식 | 속도 | 사용 모델 |
|-----------|------|------|-----------|
| BPE | 가장 많이 등장한 쌍을 합침 | fastest | GPT-2/3/4 |
| Wordpiece | 쌍으로 등장할 확률이 높으면 합침 | 중간 | BERT |
| Unigram | 최선의 조합을 확률로 탐색 | slowest | SentencePiece |

---

### 위치 인코딩 (Positional Encoding)

* **문제**: Transformer는 병렬 처리 → Token에 순서 정보가 없음!
* `"Tom ate the meat"` ≠ `"The meat ate Tom"` — 위치 없이는 구분 불가
* 해결: **위치 정보를 임베딩에 더함**

```
최종 입력 = 단어 임베딩 + 위치 인코딩

"Tom"    [임베딩 벡터] + [위치1 인코딩]
"ate"    [임베딩 벡터] + [위치2 인코딩]
"the"    [임베딩 벡터] + [위치3 인코딩]
"meat"   [임베딩 벡터] + [위치4 인코딩]
```

#### 절대 위치 인코딩 (Absolute Positional Encoding)
* 각 위치(1번, 2번, 3번...)에 고유한 숫자 벡터를 직접 할당
* 학습 가능한 파라미터로 구성 (BERT, GPT-2 방식)

```
위치 1 → [0.1, 0.3, 0.7, ...]
위치 2 → [0.4, 0.1, 0.2, ...]
위치 3 → [0.9, 0.6, 0.3, ...]
```

#### 삼각함수 인코딩 (Sinusoidal Positional Encoding)
* 사인(sin)/코사인(cos) 함수로 위치 패턴 생성 → **학습 없이도 작동**
* 낮은 차원: 주기 짧음 → 세밀한 위치 구분
* 높은 차원: 주기 매우 김 → 긴 문장에서도 반복 없이 유일한 위치 지문 생성

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))  ← 짝수 차원
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))  ← 홀수 차원

위치 1, 차원 0: sin(1/1)     = 0.841
위치 2, 차원 0: sin(2/1)     = 0.909
위치 1, 차원 2: sin(1/100)   = 0.010  ← 천천히 변함
위치 2, 차원 2: sin(2/100)   = 0.020
```

```python
def sinusoidal_pe(max_len, d_model):
    PE  = np.zeros((max_len, d_model))
    pos = np.arange(max_len).reshape(-1, 1)
    i   = np.arange(0, d_model, 2)

    PE[:, 0::2] = np.sin(pos / (10000 ** (i / d_model)))  # 짝수 차원
    PE[:, 1::2] = np.cos(pos / (10000 ** (i / d_model)))  # 홀수 차원
    return PE   # shape: [max_len, d_model]
```

#### 상대 위치 인코딩 (Relative Positional Encoding)
* 절대 위치("나는 3번째다") 대신 **단어 간의 거리**를 인코딩
* "Tom과 meat 사이는 2칸 차이다"처럼 상대적 거리를 Attention에 반영
* 단점: 매 스텝마다 거리를 계산해야 함 → 캐싱 어려움 → 느림

---

### ROPE (Rotary Positional Embeddings)

* 절대 위치 + 상대 위치의 장점만 합친 **현대 LLM의 표준** 위치 인코딩
* LLaMA, Claude, PaLM 등 최신 모델에서 사용

```
핵심 아이디어: 위치에 따라 벡터를 '회전'시킴

비유: 시계 바늘
  위치 1 → 12시 방향으로 회전
  위치 2 → 24시 방향으로 회전
  위치 3 → 36시 방향으로 회전

→ 두 단어의 내적을 계산하면 자동으로 '몇 칸 차이인지'가 반영됨!
  (상대 거리가 자동 보존)

특징:
  ✓ 2개씩 짝지어 각도를 다르게 회전 → 다양한 위치 패턴 포착
  ✓ KV Cache와 효율적으로 결합 가능 (캐싱 쉬움)
  ✓ 삼각함수 인코딩보다 학습 속도 빠름
```

```python
def apply_rope(x, pos, theta=10000.0):
    dim   = x.shape[-1]
    i     = torch.arange(0, dim, 2)
    freq  = 1.0 / (theta ** (i / dim))   # 차원마다 다른 주파수(회전 속도)
    angle = pos * freq                    # 위치 × 주파수 = 회전 각도

    x_even = x[..., 0::2]               # 짝수 차원
    x_odd  = x[..., 1::2]               # 홀수 차원

    # 2D 회전 공식 적용
    rotated_even = x_even * angle.cos() - x_odd * angle.sin()
    rotated_odd  = x_even * angle.sin() + x_odd * angle.cos()

    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
```

| 방식 | 특징 | 사용 모델 |
|------|------|-----------|
| 절대 위치 (학습형) | 학습으로 최적화 | BERT, GPT-2 |
| 삼각함수 (Sinusoidal) | 학습 불필요, 수식 기반 | 원조 Transformer |
| 상대 위치 | 단어 간 거리 반영, 느림 | T5, TransformerXL |
| ROPE | 회전 방식, 빠름, 캐싱 유리 | LLaMA, Claude, PaLM |

-----

## 📌 학습 (Training)

### Text 생성 — Autoregressive (자기회귀)

* GPT가 텍스트를 생성하는 방식: **한 번에 토큰 하나씩**, 이전에 생성한 것을 보면서 다음 예측
* 자기가 생성한 것을 다시 입력으로 사용 → "자기회귀(Autoregressive)"

```
비유: 문장 완성하기
  "Every"           → 다음 예측 → "effort"
  "Every effort"    → 다음 예측 → "moves"
  "Every effort moves" → 다음 예측 → "you"
  "Every effort moves you" → 다음 예측 → "forward"
  "Every effort moves you forward" → 다음 예측 → <EOS> → 종료!
```

```python
def generate(model, tokenizer, prompt, max_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_tokens):
        logits     = model(input_ids).logits
        next_logit = logits[:, -1, :]                           # 마지막 위치의 예측
        probs      = torch.softmax(next_logit, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)  # 가장 높은 확률 토큰

        input_ids = torch.cat([input_ids, next_token], dim=-1)  # 생성 토큰 추가

        if next_token.item() == tokenizer.eos_token_id:         # 종료 토큰이면 중지
            break

    return tokenizer.decode(input_ids[0])

result = generate(model, tokenizer, "Every effort moves")
print(result)   # "Every effort moves you forward"
```

---

### 데이터셋

* **Self-Supervised Learning**: 레이블(정답 태그) 없이 텍스트 자체에서 학습 데이터를 자동 생성
* 인터넷의 모든 텍스트가 학습 데이터가 될 수 있음!

```
원문 텍스트: "Every effort moves you forward"

자동 생성되는 학습 데이터 (Sliding Window):
  입력:  ["Every",  "effort", "moves", "you"    ]
  정답:  ["effort", "moves",  "you",   "forward" ]
                                        ↑ 바로 다음 단어가 정답!

→ 정답 레이블을 사람이 직접 붙일 필요 없음!
   텍스트 자체가 입력이자 정답
```

```python
def create_dataset(text, tokenizer, block_size=128):
    tokens   = tokenizer.encode(text)
    examples = []

    for i in range(0, len(tokens) - block_size, block_size):
        chunk     = tokens[i : i + block_size + 1]
        input_ids = chunk[:-1]   # 마지막 토큰 제외 (입력)
        labels    = chunk[1:]    # 첫 토큰 제외  (정답: 한 칸 밀린 다음 토큰)
        examples.append((input_ids, labels))

    return examples
```

---

### Token화

* 텍스트를 **모델이 처리할 수 있는 숫자(Token ID)로 변환**하는 과정
* 특수 Token(Special Token)도 함께 삽입

```
원문: "I am a student"

Tokenizer 처리:
  "I"       → Token ID: 40
  " am"     → Token ID: 716
  " a"      → Token ID: 257
  " student"→ Token ID: 3710

Special Token 추가:
  [BOS] "I" "am" "a" "student" [EOS]
   50256  40  716  257   3710   50256

Special Token 종류:
  <BOS>  : 문장 시작
  <EOS>  : 문장 끝 → 생성 종료 신호
  <PAD>  : 길이 맞추기용 패딩 (학습에서 무시!)
  <MASK> : BERT 빈칸 채우기용
```

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text    = "I am a student"
encoded = tokenizer(text, return_tensors="pt")

print(encoded["input_ids"])       # tensor([[40, 716, 257, 3710]])
print(encoded["attention_mask"])  # tensor([[1, 1, 1, 1]])  ← 1=실제 토큰, 0=PAD
```

---

### Token 임베딩

* Token ID를 **연속 벡터 공간의 점**으로 변환 (숫자 ID → 의미 있는 벡터)
* 학습을 통해 의미 있는 벡터 공간이 스스로 구성됨
* 입력과 출력의 길이가 같아야 함 (한 칸씩 밀린 다음 토큰이 정답)

```
Token ID → 임베딩 벡터 변환:

  ID: 40   ("I")       → [0.12, -0.34,  0.78, ...]  ← 768차원
  ID: 716  (" am")     → [0.45,  0.12, -0.23, ...]
  ID: 257  (" a")      → [0.67, -0.56,  0.34, ...]
  ID: 3710 (" student")→ [0.23,  0.89, -0.12, ...]

입력/타겟 쌍:
  입력:  "In"  "the"  "heart"  "of"
  타겟:  "the" "heart" "of"   "the"   ← 한 칸 밀린 다음 토큰
```

```python
vocab_size = 50257   # GPT-2 어휘 크기
embed_dim  = 768     # 임베딩 차원

token_embedding = nn.Embedding(vocab_size, embed_dim)
# Token ID → 768차원 벡터로 변환
# 처음엔 랜덤, 학습하면서 의미 있는 벡터 공간으로 수렴
```

---

### Attention — 문맥 파악

* **"지금 처리하는 단어가 다른 어떤 단어에 얼마나 주목해야 하나"** 를 계산
* Embedding에서 가중치(Weight)를 곱해서 **Q, K, V**를 구함

```
비유: 도서관 검색
  Q (Query) : 내가 찾는 키워드 → "딥러닝 관련 책"
  K (Key)   : 책 라벨(태그)    → "머신러닝", "딥러닝", "요리"
  V (Value) : 책의 실제 내용   → 해당 책 내용

  Q와 K를 비교(내적) → 관련된 K의 V를 가져옴

실제 예시: "The animal didn't cross the street because it was tired"
  "it"이 Q
  모든 단어가 K
  → "animal"과 가장 유사 → "animal"의 V를 많이 가져옴
  → "it = animal"을 이해!
```

```
Q, K, V 계산:
  Q = 입력 임베딩 X × Wq  (Query 가중치 행렬)
  K = 입력 임베딩 X × Wk  (Key 가중치 행렬)
  V = 입력 임베딩 X × Wv  (Value 가중치 행렬)

  임베딩 X (1×4) × Query Layer (4×2) = Q_student (1×2)
  임베딩 X (1×4) × Key Layer   (4×2) = K_student (1×2)
  임베딩 X (1×4) × Value Layer (4×2) = V_student (1×2)
```

```python
def compute_qkv(x, Wq, Wk, Wv):
    # x: 입력 임베딩 [seq_len, d_model]
    Q = x @ Wq   # [seq_len, d_k]
    K = x @ Wk   # [seq_len, d_k]
    V = x @ Wv   # [seq_len, d_v]
    return Q, K, V
```

> **Q, K를 분리하는 이유**: 분리하지 않으면 항상 자기 자신을 가장 주목
> → `Tom·meat == meat·Tom` (내적 교환법칙) → 방향성 없음
> → Q, K 분리 시 `Tom→meat` ≠ `meat→Tom` (방향성 있는 주목 가능)
>
> **K, V를 분리하는 이유**: 분리하지 않으면 K에 앞 단어들의 값이 섞임
> → 순수한 "키" 역할을 못 함

---

### Attention Score

* Q와 K의 **내적(Dot Product)으로 유사도를 계산**한 값
* 연관되어 있으면 값이 크게 나타남
* 수식: `Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) × V`

```
"Your journey begins with one step" 처리 시

Step 1: 각 단어에서 Q, K, V 생성
  "Your":    x⁽¹⁾ → k⁽¹⁾=[0.3,0.7], v⁽¹⁾=[0.1,0.8]
  "journey": x⁽²⁾ → q⁽²⁾=[0.4,1.4], k⁽²⁾=[0.4,1.1], v⁽²⁾=[0.3,1.0]
  ...

Step 2: "journey" 입장에서 모든 단어와 유사도 계산
  ω₂₁ = q⁽²⁾ · k⁽¹⁾ = 0.4×0.3 + 1.4×0.7 = 1.10
  ω₂₂ = q⁽²⁾ · k⁽²⁾ = 0.4×0.4 + 1.4×1.1 = 1.70
  ...

Step 3: Softmax → 합이 1인 확률(Attention Weight)로
  α₂₁ = 0.10, α₂₂ = 0.20, ...

Step 4: Value의 가중합 → 문맥이 반영된 새 벡터
  z⁽²⁾ = 0.10×[0.1,0.8] + 0.20×[0.3,1.0] + ... + 0.10×[0.3,0.7]
       = [0.30, 0.80]  ← "journey"의 문맥 반영 벡터
```

```python
import math

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # Step 1: Q·Kᵀ → Attention Score
    scores = torch.matmul(Q, K.transpose(-2, -1))   # [seq_len, seq_len]

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)

    # Step 3: Masking (선택)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Step 5: Value 가중합
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

---

### Scale (스케일링)

* **문제**: 차원 수(`dₖ`)가 커질수록 Q·K 내적값이 매우 커짐 → Softmax가 한쪽으로 쏠림 → gradient 소멸

```
왜 커지는가?
  dₖ개의 값을 더하면 분산이 dₖ배로 커짐
  → 값이 매우 크거나 작아짐 → Softmax 출력이 [0,0,...,1,0,0]처럼 극단적

해결: √dₖ 로 나눔 → 분산을 다시 1로 맞춤

예시:
  dₖ = 64 → √64 = 8로 나눔
  dₖ = 512 → √512 ≈ 22.6으로 나눔

Before Scale: [102.3, -50.1, 89.4] → softmax → [0.99, 0.00, 0.01]  (극단적)
After Scale:  [12.8,  -6.3, 11.2]  → softmax → [0.62, 0.02, 0.36]  (안정적)
```

---

### Attention Map

* Attention Score(Weight)를 **시각화한 행렬** — 어떤 단어가 어떤 단어를 주목하는지 한눈에 확인

```
"I am a student" Attention Map:
(값이 클수록 강하게 주목, 색이 진할수록 높은 주목)

         I    am    a   student
  I    [0.70  0.10  0.10   0.10]
  am   [0.40  0.40  0.10   0.10]
  a    [0.20  0.20  0.40   0.20]
  stu  [0.80  0.10  0.00   0.10]  ← "student"가 "I"를 강하게 주목!

해석:
  → "student"는 "I"와 가장 연관됨 (모델이 스스로 학습한 결과)
  → GPT는 Causal Masking으로 미래 위치는 항상 0
```

```python
# Attention Map 시각화
import matplotlib.pyplot as plt
import seaborn as sns

words        = ["I", "am", "a", "student"]
attn_weights = attn_weights[0].detach().numpy()  # shape: [seq_len, seq_len]

plt.figure(figsize=(6, 5))
sns.heatmap(attn_weights, xticklabels=words, yticklabels=words,
            annot=True, fmt=".2f", cmap="Blues")
plt.title("Attention Map")
plt.show()
```

---

### Softmax (Attention에서의 역할)

* Q·Kᵀ 결과(점수 나열)를 **합이 1인 확률 분포(Attention Weight)**로 변환
* 미래 단어 위치는 **−∞로 치환 후 Softmax** → 해당 위치의 가중치 = 0

```
Softmax 전:  [1.2, 0.8, -∞,  -∞ ]   ← -∞: 미래 위치 (보면 안 됨)
Softmax 후:  [0.6, 0.4, 0.0, 0.0]   ← -∞는 0으로 수렴 → 주목 안 함
              ↑    ↑    ↑    ↑
           과거  과거  미래  미래  (GPT Causal Masking)
```

---

### Masking

* GPT(Decoder)는 **미래 단어를 볼 수 없어야** 함 → Causal Masking(인과적 마스킹)

```
"I am a student" 학습 시:

"I" 예측 시    → "I"만 볼 수 있음         ("am"은 아직 미래!)
"am" 예측 시   → "I", "am"만 볼 수 있음
"a" 예측 시    → "I", "am", "a"만 볼 수 있음

Attention Score 행렬:
       I     am     a    student
  I  [1.2   -∞     -∞    -∞  ]
  am [0.8   1.5    -∞    -∞  ]
  a  [0.3   0.9   1.1    -∞  ]
  stu[0.1   0.4   0.8   1.3  ]
        ↑
  Softmax 후 -∞ 위치는 0 → 미래 단어에 대한 주목 없음
```

```python
def create_causal_mask(seq_len):
    # 하삼각 행렬: 자신과 이전 위치만 1(볼 수 있음), 미래는 0(볼 수 없음)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 예시 (seq_len=4):
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    return mask
```

---

### Multi-head Attention

* **각각의 Head가 다른 관점에서 문맥을 파악** → 다양한 관계를 동시에 포착
* 하나의 Attention으로는 하나의 관점밖에 못 봄 → 여러 Head를 병렬로!

```
비유: 여러 명의 독자가 같은 글을 읽음
  독자 1(Head 0): 문법적 관계에 집중 → "주어-동사 관계 파악"
  독자 2(Head 1): 의미적 관계에 집중 → "단어의 의미 연관성 파악"
  독자 3(Head 2): 위치적 관계에 집중 → "가까운 단어 관계 파악"
  → 세 독자의 의견을 모아서 최종 판단!

구조:
  입력: "I am a student" [4 × d_model]

  Head 0 (Q0,K0,V0) → a0  (문법적 관점)
  Head 1 (Q1,K1,V1) → a1  (의미적 관점)
  Head 2 (Q2,K2,V2) → a2  (위치적 관점)

  [a0 | a1 | a2] → Concatenate → 선형 변환(Wo) → 최종 출력
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # 각 head의 차원

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)   # 최종 출력 변환

    def forward(self, x, mask=None):
        B, T, D = x.shape

        Q = self.Wq(x)   # [B, T, D]
        K = self.Wk(x)
        V = self.Wv(x)

        # num_heads 개로 분리 → 각 head가 독립적으로 Attention
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            # shape: [B, num_heads, T, d_k]

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention Score 계산 (이미지의 MatMul → Scale → Mask → SoftMax → MatMul)
        scores  = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # MatMul + Scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # Mask
        weights = torch.softmax(scores, dim=-1)                   # SoftMax
        out     = weights @ V                                     # MatMul with V

        # Concatenate: head들을 다시 합침
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 최종 선형 변환
        return self.Wo(out)
```

```
Attention 연산 흐름 (이미지 우측 파란 박스):

  Q  K  V
  ↓  ↓  ↓
[MatMul] ← Q × Kᵀ: 유사도 계산
  ↓
[Scale]  ← / √dₖ: 분산 안정화
  ↓
[Mask]   ← -∞로 미래 차단 (opt.)
  ↓
[SoftMax]← 확률 분포로 변환
  ↓
[MatMul] ← × V: Value 가중합
  ↓
  출력 (Attention Value)
```

-----

## 📌 전체 흐름 요약

```
텍스트 입력: "Every effort moves you"
      │
      ▼  Token화
  Token IDs: [4,  5,  3,  2]
      │
      ▼  Token 임베딩
  Embedding: [[0.1,...], [0.4,...], ...]  + 위치 인코딩(ROPE 등)
      │
      ▼  Transformer Block × N
  ┌──────────────────────────────────────────────┐
  │  층 정규화 → 마스크드 멀티헤드 어텐션 → 숏컷  │
  │  층 정규화 → 피드 포워드(선형→GELU→선형) → 숏컷│
  └──────────────────────────────────────────────┘
      │
      ▼  출력 층 (Linear + Softmax)
  확률 분포: {"forward":0.7, "backward":0.1, ...}
      │
      ▼  가장 높은 확률의 토큰 선택 (Autoregressive)
  출력: "forward"

  → "Every effort moves you forward" 생성 완료!
```
