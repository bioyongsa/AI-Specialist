# LLM from Scratch — Chapter 2~7 코드 완전 해설

> 이 문서는 "Build a Large Language Model From Scratch" 실습 노트북(Chapter 2~7)의 핵심 코드를 단계별로 상세히 설명합니다.
> 각 챕터가 **왜 필요한지**, **어떻게 동작하는지**, **핵심 포인트**가 무엇인지를 중심으로 서술합니다.

---

## [claude] 검토 결과 요약

이 문서는 전반적으로 정확합니다. 아래 항목만 수정·보강했으며, 본문에서 **[claude]** 표시로 찾을 수 있습니다.

### 명백한 오류 (6건)

| 위치 | 원문 | 수정 |
|---|---|---|
| Q15 FeedForward | 4배 확장을 **"병목(Bottleneck)" 구조**라고 서술 | 반대입니다. 좁게→넓게→좁게이므로 **확장(expansion) / inverted bottleneck**. 4배 관행도 GPT 논문이 아니라 **원 Transformer 논문**에서 시작 |
| Q25 분류 shape | `[batch, seq_len, vocab_size]` → `[batch, vocab_size]` → "분류 헤드에 넣음" | `out_head`가 이미 `Linear(768, 2)`로 교체된 뒤이므로 **`num_classes`**. 그리고 분류 헤드는 모델 안에 있어 슬라이싱은 헤드 **통과 후**에 일어남 (순서 반대) |
| LoRA 개념 | "가중치 `W`를 `A×B`로 **근사**" | LoRA는 `W`를 근사하지 않습니다. `W`는 **동결**되고 **변화량 ΔW ≈ A×B**만 학습 |
| LoRA 파라미터 비율 | "전체의 **~0.1%** 수준" | rank=16으로 전체 Linear에 적용 시 2,654,208개 = **약 2.13%** |
| Decoder 도식 Step 4/6 | "Add & Norm" (Post-LN) | 이 코드는 **Pre-LN**이므로 `Norm → Sublayer → Add` 순서 |
| Decoder 도식 Step 7 | "마지막 토큰의 출력 벡터만 사용 `[:, -1, :]`" | `out_head`는 **모든 위치**에 적용됩니다. `[:, -1, :]`는 모델 밖 **생성 루프**에서 골라내는 단계 |

### 부정확·오해 소지 (5건)

| 위치 | 내용 |
|---|---|
| LayerNorm "왜 필요한가" | **Internal Covariate Shift**는 반박된 설명 (Santurkar et al., 2018) |
| Q11 마스킹 값 | "-1000을 넣으면 완전히 0이 되지 않아 정보 누출" → float32에서 언더플로로 **정확히 0**이 됨 |
| 5-3 `generate` 제목 | "고급 텍스트 생성"이라 했으나 노트북 구현에는 **temperature·top-k가 없음**. `generate_text_simple`과의 차이는 EOS 종료뿐 |
| Q21 `zero_grad` | 누락 시 "exploding gradient" → 정확히는 **기울기 누적** 문제 (별개 현상) |
| Encoder 도식 "I love Python" | `[314, 2012, 11361]` → `314`는 앞 공백이 붙은 `" I"`. 문장 첫머리는 **`40`** |

### 누락되어 추가한 내용

- **파라미터 수 163M vs 124M** — weight tying(`tok_emb`와 `out_head` 공유)으로 38.6M이 줄어드는 원리 (시험 단골)
- **GPT에는 Cross-Attention이 없음** — 원 Transformer의 Decoder와 혼동하기 쉬운 지점
- `final_norm`이 필요한 이유, `device=` 인자가 필수인 이유, `[block] * 12`가 위험한 이유
- `contiguous()`가 필요한 이유, `MultiHeadAttentionWrapper`와 통합 버전의 **출력 차원 차이**
- LayerNorm의 `keepdim`, `unbiased=False`, `eps` 위치, `norm1`/`norm2`를 따로 두는 이유
- FeedForward의 파라미터 비중(어텐션의 2배)과 GELU가 없으면 층이 무의미해지는 이유

---

## 전체 학습 흐름 한눈에 보기

```
Chapter 2: 텍스트 데이터 준비 (Tokenization + Dataset + Embedding)
    ↓
Chapter 3: Attention 메커니즘 구현 (Self-Attention → Multi-Head Attention)
    ↓
Chapter 4: GPT 모델 조립 (LayerNorm + FeedForward + TransformerBlock + GPTModel)
    ↓
Chapter 5: 사전 학습 (Pre-training Loop: Loss 계산 → 역전파 → 가중치 업데이트)
    ↓
Chapter 6: 분류 미세 조정 (Fine-tuning for Classification + LoRA)
    ↓
Chapter 7: 지시사항 미세 조정 (Instruction Fine-tuning / SFT)
```

---

## Chapter 2 — 텍스트 데이터 준비

### 2-1. 토크나이저(Tokenizer)란?

모델은 문자열(string)을 직접 이해하지 못합니다. 텍스트를 **숫자(토큰 ID)** 로 변환해줘야 하는데, 이 역할을 하는 것이 **토크나이저**입니다.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2용 BPE 토크나이저 로드

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# 결과: [15496, 11, 466, 345, ...] ← 각 단어/조각이 숫자로 변환됨

original = tokenizer.decode(token_ids)
# 결과: 원래 문자열로 복원됨
```

- `encode()` : 문자열 → 토큰 ID 리스트
- `decode()` : 토큰 ID 리스트 → 문자열
- `allowed_special` : `<|endoftext|>` 같은 특수 토큰도 처리 허용

**BPE(Byte Pair Encoding)** 방식을 사용하며, 자주 쓰이는 문자 조합을 하나의 토큰으로 묶습니다.

---

### 2-2. 입력(x)과 타겟(y)의 관계 — "1 shift"

GPT는 **다음 단어를 예측**하는 모델입니다. 따라서 정답(target)은 항상 입력보다 **1칸 뒤**로 밀려있어야 합니다.

```python
context_size = 4
enc_sample = tokenizer.encode(raw_text)[:50]

x = enc_sample[:context_size]           # [40, 367, 2885, 1464]  ← 입력
y = enc_sample[1:context_size + 1]      # [367, 2885, 1464, 1807] ← 정답 (1 shift!)
```

| 위치 | 입력(x) | 정답(y, 다음 토큰) |
|------|---------|-----------------|
| 0 | 40 | 367 |
| 1 | 367 | 2885 |
| 2 | 2885 | 1464 |
| 3 | 1464 | 1807 |

---

### 2-3. GPTDatasetV1 — 슬라이딩 윈도우 데이터셋

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. 전체 텍스트를 토큰화
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 2. 슬라이딩 윈도우로 데이터 조각 생성
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]        # 입력 구간
            target_chunk = token_ids[i + 1 : i + max_length + 1] # 타겟 (1 shift)

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

**핵심 파라미터:**
- `max_length` : 한 번에 모델에 넣을 토큰 수 (=context window 크기)
- `stride` : 윈도우를 얼마나 이동할지 (stride=1이면 한 칸씩, stride=max_length이면 겹침 없음)

```
stride=1인 경우:
  [40, 367, 2885, 1464] → [367, 2885, 1464, 1807]
  [367, 2885, 1464, 1807] → [2885, 1464, 1807, 3619]  ← 한 칸씩 이동
```

---

### 2-4. DataLoader 생성

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 한 번에 처리할 샘플 수
        shuffle=shuffle,         # 학습 시 순서 섞기
        drop_last=drop_last,     # 마지막 불완전 배치 버리기
        num_workers=num_workers  # 병렬 데이터 로딩 워커 수
    )
    return dataloader
```

---

### 2-5. 임베딩(Embedding) — 숫자 → 벡터

토큰 ID(정수)를 **의미 있는 벡터**로 변환합니다.

```python
# 토큰 임베딩: 각 토큰 ID를 output_dim 크기의 벡터로 변환
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# vocab_size=50257 (GPT-2 단어장 크기), output_dim=256 (벡터 차원)

token_embeddings = token_embedding_layer(inputs)
# inputs shape: [batch, seq_len]  →  출력 shape: [batch, seq_len, output_dim]

# 위치 임베딩: 각 위치(0, 1, 2, ...)도 벡터로 표현 (순서 정보 부여)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

# 최종 입력 = 토큰 임베딩 + 위치 임베딩
input_embeddings = token_embeddings + pos_embeddings
```

**왜 위치 임베딩이 필요한가?**
Transformer 구조는 순서를 자체적으로 인식하지 못합니다. 위치 정보를 벡터로 인코딩해서 더해줘야 "이 토큰이 몇 번째에 있는지"를 모델이 알 수 있습니다.

---

### 2-6. Tokenizer 비교: Qwen3 vs EXAONE

```python
from transformers import AutoTokenizer
qwen   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
exaone = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B")
```

**실험 결과 요약:**

| 텍스트 | Qwen3 토큰 수 | EXAONE 토큰 수 | 승자 |
|--------|-------------|---------------|------|
| 한국어 ("안녕하세요, 반갑습니다!") | 8 | 7 | EXAONE |
| 한국어 ("인공지능 기술이 발전하고 있습니다.") | 12 | 9 | EXAONE (3개 더 효율) |
| 일본어 | 8 | 15 | Qwen3 |
| 중국어 | 5 | 17 | Qwen3 (12개 차이!) |
| 영어 | 동일 | 동일 | - |

**결론:** 한국어는 EXAONE이, 일본어/중국어는 Qwen3가 더 효율적입니다. 토큰 수가 적을수록 같은 내용을 더 짧게 처리하므로 **학습 효율과 추론 속도**에 유리합니다.

---

## Chapter 3 — Attention 메커니즘

Attention은 "어떤 토큰이 현재 토큰을 이해하는 데 중요한가?"를 수치로 계산하는 핵심 메커니즘입니다.

### 3-1. CausalAttention (인과적 어텐션)

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        # Q, K, V 투영 레이어 (입력 벡터를 다른 공간으로 변환)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Causal Mask: 미래 정보 차단용 상삼각 행렬
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Q, K, V 계산
        queries = self.W_query(x)  # [b, num_tokens, d_out]
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # 2. 어텐션 스코어 계산 (Q·Kᵀ)
        attn_scores = queries @ keys.transpose(1, 2)  # [b, num_tokens, num_tokens]

        # 3. 미래 토큰 마스킹 (-inf로 채워서 softmax 후 0이 되게)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # 4. 스케일링 + Softmax → 어텐션 가중치 (확률값)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Value와 가중 합산 → 문맥 벡터
        context_vec = attn_weights @ values
        return context_vec
```

**Q, K, V의 역할:**

| 개념 | 비유 | 역할 |
|------|------|------|
| **Query (Q)** | "내가 지금 찾는 것" | 현재 토큰이 무엇을 알고 싶은지 |
| **Key (K)** | "내가 가진 정보 태그" | 다른 토큰이 어떤 정보를 갖고 있는지 |
| **Value (V)** | "실제 정보 내용" | 어텐션 가중치에 따라 가져올 실제 값 |

**Causal Masking이란?**

```
마스크 예시 (4×4):
[[0, 1, 1, 1],   ← 토큰0은 자기자신만 봄
 [0, 0, 1, 1],   ← 토큰1은 0,1만 봄
 [0, 0, 0, 1],   ← 토큰2는 0,1,2만 봄
 [0, 0, 0, 0]]   ← 토큰3은 모두 봄

1인 위치 = -inf → softmax 후 0이 됨 = "미래는 보지 못한다"
```

GPT는 자기회귀(Autoregressive) 모델이므로 **미래 토큰을 보면 안 됩니다**. 생성 시엔 아직 존재하지 않는 정보이기 때문입니다.

---

### 3-2. MultiHeadAttentionWrapper — 헤드를 단순 병렬 결합

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # CausalAttention을 num_heads개 만들어서 나란히 실행
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # 각 헤드의 출력을 dim=-1(마지막 차원)로 이어 붙임
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

각 헤드는 **서로 다른 관점(aspect)**에서 관계를 학습합니다. (ex: 한 헤드는 문법적 관계, 다른 헤드는 의미적 관계)

---

### 3-3. MultiHeadAttention — 최적화된 통합 버전

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 각 헤드의 차원

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 헤드 결합 후 최종 투영
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Q, K, V 한꺼번에 계산 (d_out = num_heads × head_dim)
        keys    = self.W_key(x)    # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # 2. 헤드 나누기 (view + transpose)
        # [b, num_tokens, d_out] → [b, num_tokens, num_heads, head_dim]
        #                        → [b, num_heads, num_tokens, head_dim]
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention (각 헤드 독립적으로 계산)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Value 가중 합산 후 헤드 결합
        # [b, num_heads, num_tokens, head_dim] → [b, num_tokens, d_out]
        context_vec = (attn_weights @ values).transpose(1, 2)
        # [claude] contiguous(): transpose는 메모리를 옮기지 않고 stride만 바꾼 '뷰'라
        #          논리적 순서와 물리적 순서가 어긋납니다. view()는 메모리가 연속일 때만
        #          동작하므로, 빼면 RuntimeError가 납니다. .reshape(...) 한 줄로 대체 가능.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 5. 최종 선형 투영
        return self.out_proj(context_vec)
```

**Wrapper 방식 vs 통합 방식의 차이:**

```
Wrapper: [head1 forward] + [head2 forward] + ... → 순차 실행 (느림)
통합:    W_query로 전체 d_out 계산 → view로 헤드 분리 → 병렬 행렬 연산 (빠름)
```

> **[claude] 보충 — 출력 차원이 다릅니다**
> - `MultiHeadAttentionWrapper`: 각 헤드가 `d_out`을 내고 concat하므로 최종 출력은 **`num_heads × d_out`**
> - `MultiHeadAttention`: `d_out`을 `num_heads`로 **나눠 쓰므로** 최종 출력은 **`d_out`**
>
> 그래서 통합 버전에는 `assert d_out % num_heads == 0`이 있고, `head_dim = d_out // num_heads`입니다. 같은 결과를 얻으려면 Wrapper의 `d_out`에 통합 버전의 `head_dim` 값을 넣어야 합니다.
>
> 또한 마지막 `view(b, num_tokens, d_out)`가 논문의 `Concat(head₁,...,head_h)`에 해당합니다. 메모리상 헤드 조각들이 이미 나란히 놓여 있어, 실제 복사 없이 shape만 바꿔 concat과 같은 결과를 얻습니다.

---

### 3-4. Multi-Head Attention 시각화 (Chapter_3_Excercise_Viz)

```python
# Forward Hook을 사용해 각 레이어, 각 헤드의 Attention Weight를 수집
def make_hook():
    def hook_fn(module, input, output):
        # Q, K 계산 후 attention weight 추출
        attn_weights = torch.softmax(attn_scores / ..., dim=-1)
        all_attentions.append(attn_weights.detach())
    return hook_fn

# 모든 트랜스포머 블록에 hook 등록
hooks = [block.att.register_forward_hook(make_hook()) for block in model.trf_blocks]
```

- **히트맵**: 특정 레이어의 모든 헤드 어텐션 패턴을 시각화
- **선 그래프**: 어떤 토큰이 어떤 토큰에 집중하는지 시각적으로 표현
- **인터랙티브 위젯**: Layer/Head를 드롭다운으로 바꿔가며 확인 가능

---

## Chapter 4 — GPT 모델 조립

### 4-1. LayerNorm (층 정규화)

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ (학습 가능)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β (학습 가능)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)   # 정규화
        return self.scale * norm_x + self.shift              # 스케일 & 시프트
```

**왜 필요한가?**
층이 깊어지면 각 층의 출력 스케일이 흔들려 기울기가 폭발하거나 소실되기 쉽습니다. LayerNorm은 각 토큰의 특성 벡터를 평균=0, 분산=1로 정규화해 **학습을 안정시키고 더 큰 학습률을 쓸 수 있게** 합니다.

> **[claude] 수정** — 원문은 "Internal Covariate Shift(ICS)를 줄이기 때문"이라고 설명했는데, 이는 BatchNorm 논문(2015)의 주장으로 이후 반박되었습니다. Santurkar et al.(2018)은 정규화가 ICS를 줄이지 않으며 실제 효과는 **손실 지형(loss landscape)을 매끄럽게 만드는 것**임을 보였습니다.

**정규화 축이 `dim=-1`인 이유**: 배치나 다른 토큰과 무관하게 토큰 하나의 `emb_dim`차원 벡터 안에서만 통계를 냅니다. 배치 크기·시퀀스 길이에 의존하지 않아 LLM에 적합합니다(BatchNorm과의 결정적 차이).

**`keepdim=True`인 이유**: 축소된 축을 크기 1로 남겨 `(b, num_tokens, 1)`이 되게 합니다. `False`면 `x - mean`에서 브로드캐스팅이 깨져 shape 에러가 납니다.

**`unbiased=False`인 이유**: 분산을 `n-1`이 아닌 `n`으로 나눕니다(모분산). GPT-2 원본(TensorFlow) 구현과 계산을 일치시켜야 사전학습 가중치를 그대로 불러올 수 있기 때문입니다.

**`eps`의 위치**: 반드시 루트 **안**(`torch.sqrt(var + eps)`)에 넣습니다. 밖에 더하면 `var=0`일 때 `sqrt(0)`의 도함수가 무한대라 역전파에서 NaN이 발생합니다.

**`norm1`, `norm2`를 따로 두는 이유**: LayerNorm은 학습 파라미터 γ, β를 가지므로 재사용하면 두 위치가 스케일을 강제로 공유하게 됩니다. 반면 `drop_shortcut`은 학습 파라미터가 없어 하나를 두 번 써도 됩니다.

---

### 4-2. GELU (활성화 함수)

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
```

**읽는 법:** "겔루" (Gaussian Error Linear Unit)

**ReLU와의 차이:**
- ReLU는 0에서 미분 불가능하지만 GELU는 모든 점에서 미분 가능해 기울기가 안정적입니다
- ReLU는 음수 입력의 기울기가 정확히 0이라 뉴런이 영구 비활성화되는 **dying ReLU** 문제가 있는데, GELU는 작은 음수 값을 통과시켜 회복 여지를 남깁니다

> **[claude] 보충** — 코드의 식은 GELU의 **tanh 근사식**입니다. 정확한 정의는 `x · Φ(x)`(Φ는 표준정규 누적분포함수)인데, 당시 `erf` 연산이 느려 근사식을 썼고 GPT-2 원본이 이를 사용하므로 가중치 호환을 위해 그대로 따릅니다.

---

### 4-3. FeedForward (피드 포워드 네트워크)

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 차원 4배 확장
            GELU(),                                           # 비선형 활성화
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 원래 차원으로 축소
        )

    def forward(self, x):
        return self.layers(x)
```

**역할:** Attention이 "어떤 토큰을 참조할지" 결정한다면, FeedForward는 "참조한 정보를 어떻게 변환할지"를 담당합니다. 각 토큰에 대해 **독립적으로**(position-wise) 처리되며, 4배 확장을 통해 더 풍부한 표현을 학습합니다.

| | 역할 | 작용 축 |
|---|---|---|
| Attention | 토큰 **간** 정보 교환 | `num_tokens` 축을 가로질러 섞음 |
| FeedForward | 토큰 **내** 정보 가공 | `emb_dim` 축만 변환 |

**파라미터 비중**: emb_dim=768 기준 블록 하나당 FFN은 약 4.72M(= 768×3072+3072 + 3072×768+768)으로, 같은 블록의 어텐션(약 2.36M)의 **2배**입니다. 파라미터 기준으로는 FFN이 모델의 주력입니다.

**GELU가 없으면**: Linear를 두 번 쌓아도 `W₂(W₁x+b₁)+b₂ = (W₂W₁)x + b'`가 되어 결국 `Linear(768,768)` 하나와 수학적으로 동일해집니다. 3072차원으로 확장한 의미가 사라지므로, 활성화 함수는 이 층이 존재하는 이유 그 자체입니다.

---

### 4-4. TransformerBlock (트랜스포머 블록)

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = MultiHeadAttention(...)
        self.ff   = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # [Attention 블록] Pre-LayerNorm + Residual Connection
        shortcut = x
        x = self.norm1(x)        # 1. 정규화 먼저
        x = self.att(x)          # 2. Attention
        x = self.drop_shortcut(x)
        x = x + shortcut         # 3. 잔차 연결 (원본 더하기)

        # [FeedForward 블록] Pre-LayerNorm + Residual Connection
        shortcut = x
        x = self.norm2(x)        # 1. 정규화 먼저
        x = self.ff(x)           # 2. FeedForward
        x = self.drop_shortcut(x)
        x = x + shortcut         # 3. 잔차 연결 (원본 더하기)

        return x
```

**Residual Connection(잔차 연결)이란?**
`x = f(x) + x` 형태로, 변환된 값에 **원본 입력을 더해줍니다**. 이렇게 하면 기울기가 깊은 층까지 잘 전달되어 **기울기 소실(Vanishing Gradient) 문제**를 방지합니다.

---

### 4-5. GPTModel — 전체 모델

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])    # 토큰 임베딩
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 위치 임베딩
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 트랜스포머 블록 n_layers개 쌓기
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 각 어휘에 대한 점수

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)                             # 토큰 → 벡터
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 위치 → 벡터

        x = tok_embeds + pos_embeds    # 두 임베딩 합산
        x = self.drop_emb(x)
        x = self.trf_blocks(x)         # 트랜스포머 블록들 통과
        x = self.final_norm(x)
        logits = self.out_head(x)      # [batch, seq_len, vocab_size] ← 각 단어의 예측 점수

        return logits
```

**GPT-2 Small 설정값:**

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 단어장 크기
    "context_length": 1024, # 한 번에 볼 수 있는 최대 토큰 수
    "emb_dim": 768,         # 임베딩 차원 크기
    "n_heads": 12,          # 어텐션 헤드 수
    "n_layers": 12,         # 트랜스포머 블록 수
    "drop_rate": 0.1,       # 드롭아웃 비율
    "qkv_bias": False       # Q/K/V Linear에 bias 사용 여부
}
```

> **[claude] 보충 — "124M"인데 실제로는 163M인 이유 (시험 단골)**
>
> | 구성 | 계산 | 개수 |
> |---|---|---|
> | `tok_emb` | 50257 × 768 | 38.6M |
> | `pos_emb` | 1024 × 768 | 0.8M |
> | TransformerBlock × 12 | (attn 2.36M + ffn 4.72M) × 12 | 85.0M |
> | `out_head` | 768 × 50257 | 38.6M |
> | **합계** | | **163,009,536** |
>
> GPT-2 원본은 `tok_emb`와 `out_head`의 가중치를 **공유**합니다(weight tying). 둘 다 `768 × 50257` 크기이고 "단어→벡터", "벡터→단어"로 서로 역방향 연산이라 같은 행렬을 쓰는 게 자연스럽습니다. 38.6M을 빼면 **124,412,160 ≈ 124M**이 됩니다. 교재 코드는 이해를 위해 일부러 분리해 두었으므로 `sum(p.numel() for p in model.parameters())`를 찍으면 163M이 나옵니다.
>
> `out_head`에 `bias=False`인 이유: logits에 상수를 더해도 softmax 결과는 변하지 않고, weight tying과의 정합성 때문입니다.

**`final_norm`이 필요한 이유**: Pre-LN 구조에서는 마지막 블록의 출력이 정규화를 거치지 않은 상태입니다. 12번의 잔차 덧셈이 누적되어 스케일이 커져 있으므로 출력 헤드에 넘기기 전에 한 번 정리합니다. Post-LN이었다면 불필요합니다.

**`torch.arange(seq_len, device=in_idx.device)`에서 `device=`가 필수인 이유**: 빠뜨리면 항상 CPU에 생성되어, 모델이 GPU에 있을 때 `RuntimeError: Expected all tensors to be on the same device`가 발생합니다.

**블록을 `[TransformerBlock(cfg) for _ in range(n)]`로 만드는 이유**: `[block] * 12`처럼 쓰면 **같은 객체 12개**가 담겨 파라미터를 공유하게 되므로 완전히 다른 모델이 됩니다. 반드시 매번 새로 생성해야 합니다.

---

### 4-6. generate_text_simple — 텍스트 생성

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 1. 최대 길이 초과 방지: 뒤쪽 context_size 토큰만 사용
        idx_cond = idx[:, -context_size:]

        # 2. 모델 예측 (기울기 계산 필요 없음)
        with torch.no_grad():
            logits = model(idx_cond)

        # 3. 마지막 위치의 예측값만 추출
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # 4. 가장 높은 점수의 토큰 선택 (Greedy Decoding)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 5. 기존 시퀀스에 이어 붙이기
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

---

## Chapter 5 — 사전 학습 (Pre-training)

### 5-1. calc_loss_batch — 배치 손실 계산

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 모델 순전파
    logits = model(input_batch)  # [batch, seq_len, vocab_size]

    # CrossEntropyLoss 계산
    # flatten(0, 1): [batch, seq_len, vocab_size] → [batch*seq_len, vocab_size]
    # .flatten():    [batch, seq_len]              → [batch*seq_len]
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss
```

**CrossEntropyLoss란?**
손실 = `-log P(정답 토큰)`입니다. 즉 정답 토큰에 부여한 **확률**(softmax 후)이 1에 가까울수록 손실이 0에 가까워지고, 0에 가까울수록 손실이 무한대로 발산합니다.

> **[claude] 보충** — "로짓 점수가 높을수록"이 아니라 **다른 토큰 대비 상대적으로** 높아야 합니다. 로짓 전체에 같은 값을 더해도 손실은 변하지 않습니다. 참고로 `F.cross_entropy`는 내부에서 log_softmax를 수행하므로 모델 출력에 softmax를 따로 적용하면 안 됩니다.

---

### 5-2. train_model_simple — 학습 메인 루프

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    for epoch in range(num_epochs):
        model.train()  # ← 학습 모드 (Dropout 활성화)

        for input_batch, target_batch in train_loader:
            # Step 1: 이전 기울기 초기화 (필수!)
            optimizer.zero_grad()

            # Step 2: 순전파 + 손실 계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Step 3: 역전파 (각 파라미터별 기울기 계산)
            loss.backward()

            # Step 4: 가중치 업데이트 (기울기 방향으로 파라미터 이동)
            optimizer.step()

    return train_losses, val_losses, tokens_seen
```

**학습의 4단계 (매 배치마다 반복):**

```
① optimizer.zero_grad()  → 이전 기울기 초기화
② loss = calc_loss(...)  → 순전파 + 손실 계산
③ loss.backward()        → 역전파 (기울기 계산)
④ optimizer.step()       → 가중치 업데이트
```

---

### 5-3. generate — EOS 종료 조건이 추가된 생성

> **[claude] 수정** — 원문 제목은 "고급 텍스트 생성"이지만, 노트북의 `generate()` 구현에는 **temperature와 top-k 샘플링이 들어 있지 않습니다**(docstring에만 언급되어 있고 인자에도 없음). `generate_text_simple`과의 실질적 차이는 **EOS 토큰 종료 조건 하나뿐**입니다. 교재 원본의 `generate()`는 `temperature=0.0, top_k=None` 인자를 받아 `torch.multinomial`로 샘플링하지만, 이 노트북 버전은 greedy만 구현되어 있습니다.

```python
def generate(model, idx, max_new_tokens, context_size, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # 마지막 위치만

        # Greedy Decoding
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # EOS 토큰 발생 시 생성 중단
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

---

### 5-4. 학습 전체 설정 및 실행

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # ← 빠른 학습을 위해 1024 → 256으로 줄임
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,  # 학습률
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1     # 과적합 방지용 L2 정규화
}

# AdamW 옵티마이저 사용 (Weight Decay가 올바르게 적용되는 Adam 변형)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=settings["learning_rate"],
    weight_decay=settings["weight_decay"]
)
```

**모델 저장 및 로드:**
```python
# 저장
torch.save(model.state_dict(), "outputs/model.pth")

# 로드
model = GPTModel(GPT_CONFIG_124M)                          # 1. 빈 모델 생성
model.load_state_dict(torch.load("outputs/model.pth", weights_only=True))  # 2. 가중치 로드
model.to(device)                                           # 3. GPU로 이동
```

---

## Chapter 6 — 분류 미세 조정 (Fine-tuning for Classification)

### 6-1. 스팸 분류 데이터셋 준비

```python
def create_balanced_dataset(df):
    """스팸/햄 불균형 해결 — 언더샘플링"""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])

# 라벨 인코딩
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 70% 학습 / 10% 검증 / 20% 테스트로 분할
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
```

---

### 6-2. SpamDataset — 패딩과 트런케이션

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 1. 토큰화
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # 2. 트런케이션 (너무 긴 문장 자르기)
        if max_length is not None:
            self.encoded_texts = [t[:max_length] for t in self.encoded_texts]

        # 3. 패딩 (모든 문장을 같은 길이로 맞추기)
        # 짧은 문장 끝에 pad_token_id(50256)를 채워 넣음
        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t))
            for t in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label   = self.data.iloc[index]["Label"]  # 0 또는 1
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
```

**패딩이 왜 필요한가?**
DataLoader는 여러 샘플을 하나의 배치로 묶어 처리하는데, 배치 내 모든 샘플의 길이가 같아야 텐서로 변환할 수 있습니다. 짧은 시퀀스는 패딩으로 채워 길이를 맞춥니다.

---

### 6-3. 모델 출력 헤드 교체 + 파라미터 동결

```python
# 1. 모든 파라미터 동결 (기존 지식 보존)
for param in model.parameters():
    param.requires_grad = False

# 2. 출력 헤드 교체: 50257개 단어 예측 → 2개 클래스(스팸/햄) 예측
model.out_head = torch.nn.Linear(in_features=768, out_features=2)
# 새 레이어는 requires_grad=True 상태로 자동 생성됨

# 3. 마지막 트랜스포머 블록 + 최종 정규화 층도 학습 허용 (잠금 해제)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

**전략의 의미:**
- 대부분의 가중치는 고정 → 사전 학습에서 얻은 언어 이해 능력 보존
- 마지막 레이어만 학습 → 스팸 분류 과제에 맞게 조정
- 이 방식이 전체 학습보다 **빠르고 과적합도 적음**

---

### 6-4. 분류 손실 및 정확도 계산

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    # GPT는 시퀀스 전체를 출력하므로, 분류에는 마지막 토큰의 출력만 사용
    # [claude] out_head를 교체한 뒤이므로 model(...)의 shape은
    #          [batch, seq_len, num_classes(=2)] 이고, 슬라이싱 결과는 [batch, 2]
    logits = model(input_batch)[:, -1, :]  # ← -1이 핵심! 마지막 위치
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    for input_batch, target_batch in data_loader:
        logits = model(input_batch)[:, -1, :]  # 마지막 토큰 출력
        predicted_labels = torch.argmax(logits, dim=-1)  # 가장 높은 클래스 선택

        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples
```

---

## Chapter 6 (LoRA) — 효율적인 파라미터 미세 조정

### LoRA란?

**Low-Rank Adaptation(LoRA)** 는 대형 모델의 가중치를 그대로 두고, 훨씬 작은 **저랭크(Low-Rank)** 행렬만 추가로 학습하는 기법입니다.

```
기존 방식: W (전체 학습) — 수억 개 파라미터
LoRA 방식: W + A×B (A, B만 학습) — 수천~수만 개 파라미터
```

---

### 6-LoRA-1. LoRALayer 구현

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # A 행렬: (in_dim, rank) — kaiming 초기화
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # B 행렬: (rank, out_dim) — 0으로 초기화 (학습 초기엔 효과 없음)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

        self.alpha = alpha  # 스케일 팩터
        self.rank  = rank   # 저랭크 차원 (rank << min(in_dim, out_dim))

    def forward(self, x):
        # LoRA 출력 = (alpha / rank) × (x @ A @ B)
        return (self.alpha / self.rank) * (x @ self.A @ self.B)
```

**왜 B를 0으로 초기화하나?**
학습 시작 시 LoRA 어댑터의 출력이 0이 되어, 기존 모델 동작을 그대로 유지하면서 점진적으로 학습됩니다.

---

### 6-LoRA-2. LinearWithLoRA — 기존 레이어에 LoRA 추가

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 기존 고정된 Linear 레이어
        self.lora   = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        # 출력 = 기존 Linear 출력 + LoRA 어댑터 출력
        return self.linear(x) + self.lora(x)
```

---

### 6-LoRA-3. 모든 Linear 레이어 자동 교체

```python
def replace_linear_with_lora(model, rank, alpha):
    """모델 내 모든 nn.Linear를 LinearWithLoRA로 재귀적으로 교체"""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)  # 재귀 호출
```

**LoRA 적용 순서:**
```python
# 1. 모든 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 2. Linear → LinearWithLoRA로 교체 (새 LoRA 파라미터는 requires_grad=True)
replace_linear_with_lora(model, rank=16, alpha=16)

# 결과: 전체 파라미터 중 극히 일부(LoRA A, B)만 학습
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 파라미터 수: {total_trainable:,}")  # ← 전체의 약 2% 수준
```

> **[claude] 수정** — 원문은 "전체의 ~0.1% 수준"이라고 했으나 실제로는 **약 2%**입니다. rank=16으로 모든 Linear를 감쌌을 때 블록당 LoRA 파라미터는
> `4×(768·16+16·768)` (Q/K/V/out_proj) `+ (768·16+16·3072) + (3072·16+16·768)` (FFN) `= 221,184`
> 이고, 12개 블록이면 **2,654,208개**입니다. 분류용 베이스 모델(약 124M) 대비 **2.13%**입니다.
> rank를 4~8로 낮추거나 어텐션 Q/V에만 적용하면 0.1%대까지 내려갈 수 있습니다.

---

## Chapter 7 — 지시사항 미세 조정 (Instruction Fine-tuning / SFT)

### 7-1. Alpaca 스타일 프롬프트 포맷

```python
def format_input(entry):
    """
    Alpaca 템플릿: 모델에게 명확한 역할과 형식을 부여합니다.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

**프롬프트 구조:**
```
Below is an instruction that describes a task. ...

### Instruction:
Classify the sentiment of the following text as positive or negative.

### Input:
I love this product!

### Response:
Positive
```

---

### 7-2. InstructionDataset

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded_texts = []

        for entry in data:
            # 지시사항 + 입력 텍스트
            instruction_plus_input = format_input(entry)
            # 응답 텍스트
            response_text = f"\n\n### Response:\n{entry['output']}"

            # 모델은 "질문+답변" 전체 텍스트로 학습 (다음 토큰 예측)
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
```

**왜 질문과 답변을 이어붙여서 학습하나?**
GPT는 다음 토큰 예측으로 학습합니다. 질문 뒤에 오는 `### Response:\n...` 부분을 예측하면서 자연스럽게 지시사항을 따르는 능력을 습득합니다.

---

### 7-3. custom_collate_fn — 배치 패딩 처리

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # EOS 토큰 추가

        # 패딩으로 최대 길이 맞추기
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # 입력: 마지막 토큰 제외 (0 ~ n-1)
        inputs  = torch.tensor(padded[:-1])
        # 타겟: 첫 번째 토큰 제외 (1 ~ n)  ← 1 shift!
        targets = torch.tensor(padded[1:])

        # 패딩 부분 마스킹: 타겟에서 패딩은 -100으로 설정
        # CrossEntropyLoss는 ignore_index=-100인 위치를 손실 계산에서 제외
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 첫 EOS는 유지, 나머지 패딩은 -100

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
```

**패딩 마스킹이 중요한 이유:**
패딩은 실제 데이터가 아니므로 손실 계산에 포함되면 안 됩니다. `ignore_index=-100`으로 설정하면 PyTorch의 `CrossEntropyLoss`가 해당 위치를 자동으로 무시합니다.

---

### 7-4. 학습 흐름 요약

```python
def main(test_mode=False):
    # 1. 데이터 로드 (Alpaca instruction-data.json)
    data = download_and_load_file(file_path, url)

    # 2. 분할: Train 85% / Test 10% / Val 5%
    train_data = data[:train_portion]

    # 3. Dataset + DataLoader (custom_collate_fn 사용)
    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        collate_fn=customized_collate_fn,
        batch_size=8, shuffle=True
    )

    # 4. 사전 학습 모델 로드 (GPT-2 Medium 355M)
    model = load_gpt2_model("gpt2-medium-355M.pth", BASE_CONFIG)

    # 5. 미세 조정 (train_model_simple 재사용)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs=2, ...)

    # 6. 테스트 데이터로 응답 생성 및 저장
    for entry in test_data:
        token_ids = generate(model, text_to_token_ids(format_input(entry), tokenizer), ...)
        response  = token_ids_to_text(token_ids, tokenizer)
        entry["model_response"] = response

    # 결과를 JSON으로 저장
    json.dump(test_data, open("outputs/instruction-data-with-response.json", "w"), indent=4)
```

---

## 전체 설계 패턴 요약

### 각 챕터의 역할

```
Chapter 2  → 입력 파이프라인: 텍스트 → 토큰 ID → 임베딩 벡터
Chapter 3  → 핵심 연산:      토큰 간 관계 학습 (Attention)
Chapter 4  → 모델 조립:      Attention + FFN + Norm → TransformerBlock → GPTModel
Chapter 5  → 학습:           손실 계산 → 역전파 → 가중치 갱신
Chapter 6  → 미세 조정:      분류 헤드 교체 + 일부 파라미터만 학습
Chapter 6  → 효율화:         LoRA로 극소수 파라미터만 학습
Chapter 7  → 지시 따르기:    Instruction 포맷 학습 → 원하는 응답 생성
```

### 학습 vs 추론(생성) 모드

| | 학습(Train) | 추론(Inference) |
|--|------------|----------------|
| 코드 | `model.train()` | `model.eval()` |
| Dropout | 활성화 (일부 뉴런 끔) | 비활성화 |
| 기울기 계산 | 필요 (`loss.backward()`) | 불필요 (`torch.no_grad()`) |
| 목적 | 가중치 업데이트 | 텍스트 생성/예측 |

---

*이 문서는 Sebastian Raschka의 "Build a Large Language Model From Scratch" 실습 코드를 기반으로 작성되었습니다.*


---
---

# 📝 시험 출제 포인트 총정리 — `????` 빈칸 완전 해설

> 각 노트북에서 `????`로 표시된 **빈칸 채우기 문제**를 챕터별로 정리합니다.
> "정답 코드 → 왜 이 코드인가?" 순서로 상세히 설명합니다.

---

## Chapter 2 빈칸 문제

### ❓ Q1. 토크나이저 encode / decode 메서드명

```python
# 빈칸 위치:
tokenids = tokenizer.????(text, allowed_special={"<|endoftext|>"})
strings  = tokenizer.????(tokenids)
```

**정답:**
```python
tokenids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings  = tokenizer.decode(tokenids)
```

**이유:**
- `encode()` : 문자열(str) → 정수 리스트(List[int]) 변환. 모델이 처리할 수 있는 형태로 바꿈
- `decode()` : 정수 리스트 → 문자열 역변환. 모델 출력을 사람이 읽을 수 있게 복원
- `allowed_special={"<|endoftext|>"}` : 이 특수 토큰을 단순 텍스트가 아닌 **진짜 특수 토큰**으로 처리하도록 허용. 없으면 오류 발생

---

### ❓ Q2. target(y)의 인덱스 — 1 shift

```python
# 빈칸 위치:
x = enc_sample[:context_size]
y = enc_sample[????:?????]
```

**정답:**
```python
y = enc_sample[1:context_size + 1]
```

**이유:**
GPT는 "현재 토큰이 주어졌을 때 다음 토큰이 무엇인가?"를 예측합니다.
따라서 정답(y)은 입력(x)보다 **정확히 1칸 오른쪽으로 밀려** 있어야 합니다.

```
x = [token_0, token_1, token_2, token_3]  ← 인덱스 0~3
y = [token_1, token_2, token_3, token_4]  ← 인덱스 1~4 (1 shift)
     ↑ x[0]의 정답  ↑ x[1]의 정답 ...
```
- 시작 인덱스: `0` → `1` (한 칸 뒤)
- 끝 인덱스: `context_size` → `context_size + 1`

---

### ❓ Q3. GPTDatasetV1 슬라이딩 윈도우 슬라이싱

```python
# 빈칸 위치 (input_chunk, target_chunk):
input_chunk  = token_ids[???? : ???? + max_length]
target_chunk = token_ids[???? + ???? : ???? + max_length + ????]
```

**정답:**
```python
input_chunk  = token_ids[i     : i + max_length    ]
target_chunk = token_ids[i + 1 : i + max_length + 1]
```

**이유:**
- `i` : 윈도우의 현재 시작 위치
- `input_chunk`  → 위치 `i`부터 `max_length`개 토큰 (i ~ i+max_length-1)
- `target_chunk` → `i+1`부터 `max_length`개 토큰 (i+1 ~ i+max_length) ← 1 shift!
- `for i in range(0, len(token_ids) - max_length, stride)` : stride만큼 점프하며 반복

---

### ❓ Q4. `__getitem__` 반환값

```python
# 빈칸 위치:
def __getitem__(self, idx):
    return self.????[idx], self.????[idx]
```

**정답:**
```python
return self.input_ids[idx], self.target_ids[idx]
```

**이유:**
DataLoader는 `__getitem__(idx)`을 호출해 샘플을 가져옵니다.
반드시 **(입력, 정답) 쌍**을 튜플로 반환해야 합니다. 순서가 바뀌면 모델이 입력과 정답을 반대로 인식하므로 주의합니다.

---

### ❓ Q5. DataLoader 첫 번째 인자

```python
# 빈칸 위치:
dataloader = DataLoader(
    ????,
    batch_size=batch_size, shuffle=shuffle, ...
)
```

**정답:**
```python
dataloader = DataLoader(dataset, batch_size=batch_size, ...)
```

**이유:**
`DataLoader`의 첫 번째 인자는 반드시 **`torch.utils.data.Dataset` 인스턴스**여야 합니다.
위에서 생성한 `GPTDatasetV1(txt, tokenizer, max_length, stride)` 객체가 그 역할을 합니다.

---

### ❓ Q6. Embedding 레이어 생성 인자

```python
# 빈칸 위치:
embedding_layer     = torch.nn.Embedding(????, ????)   # 토큰 임베딩
pos_embedding_layer = torch.nn.Embedding(????, ????)   # 위치 임베딩
```

**정답:**
```python
embedding_layer     = torch.nn.Embedding(vocab_size,      output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length,  output_dim)
```

**이유:**

| 레이어 | 첫 번째 인자 | 두 번째 인자 |
|--------|------------|------------|
| 토큰 임베딩 | `vocab_size` (단어장 크기, 예: 50257) | `output_dim` (벡터 차원) |
| 위치 임베딩 | `context_length` (최대 위치 수) | `output_dim` (벡터 차원) |

- 토큰 임베딩: 0~vocab_size-1 사이의 ID를 벡터로 변환 → `vocab_size`개 행이 필요
- 위치 임베딩: 0~context_length-1 위치를 벡터로 변환 → `context_length`개 행이 필요
- `torch.arange(max_length)` : [0, 1, 2, ..., max_length-1] 위치 인덱스 자동 생성

---

### ❓ Q7. 토큰 임베딩 + 위치 임베딩 합산

```python
# 빈칸 위치:
input_embeddings = ???? + ????
```

**정답:**
```python
input_embeddings = token_embeddings + pos_embeddings
```

**이유:**
Transformer는 순서(위치)를 스스로 인식하지 못합니다. 위치 임베딩을 **덧셈**으로 합쳐줘야 모델이 각 토큰의 위치 정보를 학습할 수 있습니다.
두 텐서의 shape이 동일하므로 element-wise 덧셈이 가능합니다.
- `token_embeddings` shape: `[batch, seq_len, emb_dim]`
- `pos_embeddings` shape: `[seq_len, emb_dim]` → 브로드캐스팅으로 batch 차원 자동 확장

---

## Chapter 3 빈칸 문제

### ❓ Q8. Q/K/V 투영 레이어 타입

```python
# 빈칸 위치:
self.W_query = nn.????(d_in, d_out, bias=qkv_bias)
self.W_key   = nn.????(d_in, d_out, bias=qkv_bias)
self.W_value = nn.????(d_in, d_out, bias=qkv_bias)
```

**정답:**
```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
```

**이유:**
Query, Key, Value는 모두 입력 벡터(`d_in` 차원)를 다른 공간(`d_out` 차원)으로 **선형 변환**하는 것입니다. 이것이 바로 `nn.Linear`(완전 연결층)의 역할입니다.
- `d_in`: 입력 임베딩 차원
- `d_out`: 어텐션 출력 차원 (Q, K, V 모두 동일하게 설정)
- `bias=qkv_bias`: 보통 GPT에서는 편향 없이 사용 (`False`)

---

### ❓ Q9. Q, K, V 벡터 계산

```python
# 빈칸 위치:
keys    = self.????(x)
queries = self.????(x)
values  = self.????(x)
```

**정답:**
```python
keys    = self.W_key(x)
queries = self.W_query(x)
values  = self.W_value(x)
```

**이유:**
`__init__`에서 정의한 투영 레이어에 입력 `x`를 통과시켜 Q, K, V를 계산합니다.
각 레이어는 **서로 다른 가중치 행렬**을 가지므로, 동일한 `x`에서 세 가지 다른 벡터가 만들어집니다.

---

### ❓ Q10. 어텐션 스코어 계산 (Q·Kᵀ)

```python
# 빈칸 위치:
attn_scores = ???? @ ????.transpose(1, 2)
```

**정답:**
```python
attn_scores = queries @ keys.transpose(1, 2)
```

**이유:**
어텐션 스코어는 **Query와 Key의 내적(dot product)**입니다.
- `queries` shape: `[b, num_tokens, d_out]`
- `keys.transpose(1, 2)` shape: `[b, d_out, num_tokens]`
- 결과 `attn_scores` shape: `[b, num_tokens, num_tokens]` ← 모든 토큰 쌍의 유사도

`keys.transpose(1, 2)`: 행렬 곱이 되도록 Keys의 1번, 2번 차원을 뒤집는 것입니다.
수식: `score(i,j) = Q_i · K_j` (i번째 토큰이 j번째 토큰과 얼마나 관련있는지)

---

### ❓ Q11. 마스킹 값

```python
# 빈칸 위치:
attn_scores.masked_fill_(
    self.mask.bool()[:num_tokens, :num_tokens],
    ????
)
```

**정답:**
```python
attn_scores.masked_fill_(..., -torch.inf)
```

**이유:**
`-torch.inf`(= 음의 무한대)를 넣어야 하는 이유:
```
softmax(-inf) = e^(-inf) / Σe^(scores) = 0 / Σ = 0
```
즉, 마스킹된 위치는 softmax 이후 **확률이 정확히 0**이 됩니다.

> **[claude] 보충** — 원문은 "다른 값(예: -1000)을 넣으면 완전히 0이 되지 않아 미세한 정보 누출이 발생한다"고 했는데, 실제로는 float32에서 `exp(-1000)`이 언더플로하여 **정확히 0**이 됩니다. 오히려 실무에서는 큰 음수 유한값(`torch.finfo(dtype).min` 등)을 쓰는 구현이 많습니다. `-inf`는 한 행이 **전부** 마스킹될 경우 `0/0 → NaN`을 만들 수 있기 때문입니다. causal mask에서는 대각선이 항상 열려 있어 이 문제가 생기지 않으므로 `-torch.inf`가 안전합니다.

---

### ❓ Q12. Softmax + 스케일링

```python
# 빈칸 위치:
attn_weights = torch.????(
    attn_scores / keys.shape[-1]????
    , dim=-1
)
```

**정답:**
```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

**이유:**
- `torch.softmax`: 점수를 0~1 사이의 확률로 변환 (모든 값의 합 = 1)
- `/ keys.shape[-1]**0.5`: **스케일링(Scaling)** 처리
  - `keys.shape[-1]` = `d_k` (Key 벡터의 차원)
  - `**0.5` = 제곱근(√)
  - 차원이 클수록 내적 값이 커져 softmax가 극단적으로 뾰족해지는 현상 방지
  - **Scaled Dot-Product Attention**이라 불리는 이유가 바로 이 스케일링 때문
- `dim=-1`: 마지막 차원(각 행)을 기준으로 softmax 계산

---

### ❓ Q13. Context Vector 계산

```python
# 빈칸 위치:
context_vec = ???? @ ????
```

**정답:**
```python
context_vec = attn_weights @ values
```

**이유:**
어텐션 가중치(확률)를 Value 벡터에 **가중 평균**하는 연산입니다.
- `attn_weights` shape: `[b, num_tokens, num_tokens]`
- `values` shape: `[b, num_tokens, d_out]`
- 결과 shape: `[b, num_tokens, d_out]`

예: 토큰 i의 context vector = Σⱼ(attn_weight[i,j] × value[j])
→ "내가 관심 있는 토큰들의 정보를 가중 합산"한 벡터

---

## Chapter 4 빈칸 문제

### ❓ Q14. LayerNorm 정규화 수식

```python
# 빈칸 위치:
norm_x = (x - ????) / torch.sqrt(???? + self.eps)
return self.???? * norm_x + self.????
```

**정답:**
```python
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

**이유:**
LayerNorm의 수식: `y = γ × (x - μ) / √(σ² + ε) + β`

| 변수 | 코드 | 의미 |
|------|------|------|
| `μ` (평균) | `mean` | `x.mean(dim=-1, keepdim=True)` |
| `σ²` (분산) | `var` | `x.var(dim=-1, keepdim=True, unbiased=False)` |
| `ε` (엡실론) | `self.eps = 1e-5` | 분모가 0이 되는 것 방지 |
| `γ` (스케일) | `self.scale` | 학습 가능한 파라미터 (초기값: 1) |
| `β` (시프트) | `self.shift` | 학습 가능한 파라미터 (초기값: 0) |

`self.scale`과 `self.shift`가 학습 가능한 이유: 정규화 후 모델이 필요에 따라 분포를 다시 조정할 수 있는 자유도를 주기 위함입니다.

---

### ❓ Q15. FeedForward의 확장 배수와 활성화 함수

```python
# 빈칸 위치:
self.layers = nn.Sequential(
    nn.Linear(cfg["emb_dim"], ???? * cfg["emb_dim"]),
    ????(),
    nn.Linear(???? * cfg["emb_dim"], cfg["emb_dim"]),
)
```

**정답:**
```python
self.layers = nn.Sequential(
    nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
    GELU(),
    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
)
```

**이유:**
- 확장 배수 `4`: 원 Transformer 논문("Attention Is All You Need", 512→2048)에서 확립된 관행으로 GPT·BERT·LLaMA가 모두 따릅니다. 좁은 공간에서는 비선형 변환의 표현력이 제한되므로, 넓은 공간으로 올려 비선형을 적용한 뒤 되돌립니다.

> **[claude] 수정** — 원문은 이 구조를 "병목(Bottleneck)"이라고 표현했으나 반대입니다. Bottleneck은 ResNet처럼 **넓게 → 좁게 → 넓게** 가는 구조입니다. 여기는 **좁게 → 넓게 → 좁게**이므로 정확한 명칭은 **확장(expansion) 구조** 또는 **inverted bottleneck**입니다. 또한 4배 관행은 GPT 논문이 아니라 원 Transformer 논문에서 시작됐습니다.
- `GELU()`: GPT 계열이 선택한 활성화 함수. ReLU보다 부드러운 비선형성을 제공하고 음수 구간에서도 작은 기울기를 유지합니다.
- 3개 레이어로 구성: `확장 → 비선형 → 축소`

```
emb_dim=768인 경우:
768 → 3072 (4배 확장) → GELU → 768 (원래 크기로)
```

---

### ❓ Q16. TransformerBlock의 att/ff 호출

```python
# 빈칸 위치:
# [Attention 블록]
shortcut = x
x = self.norm1(x)
x = self.????(x)   # ← 여기
x = self.drop_shortcut(x)
x = x + shortcut

# [FeedForward 블록]
shortcut = x
x = self.norm2(x)
x = self.????(x)   # ← 여기
x = self.drop_shortcut(x)
x = x + shortcut
```

**정답:**
```python
x = self.att(x)   # Attention 호출
x = self.ff(x)    # FeedForward 호출
```

**이유:**
`self.att`와 `self.ff`는 `__init__`에서 정의한 인스턴스입니다.
- `self.att` = `MultiHeadAttention(...)` → 토큰 간 관계 계산
- `self.ff` = `FeedForward(cfg)` → 각 토큰의 표현 변환

**잔차 연결(Residual Connection) 구조 핵심:**
```
shortcut = x         ← 원본 저장
x = Norm(x)          ← 먼저 정규화 (Pre-LN 방식)
x = SubLayer(x)      ← Attention 또는 FeedForward
x = Dropout(x)
x = x + shortcut     ← 원본 더하기 → 기울기 소실 방지
```

---

### ❓ Q17. GPTModel의 Embedding 차원 인자

```python
# 빈칸 위치:
self.tok_emb = nn.Embedding(cfg["????"], cfg["emb_dim"])
self.pos_emb = nn.Embedding(cfg["????"], cfg["emb_dim"])
self.out_head = nn.Linear(cfg["emb_dim"], cfg["????"], bias=False)
```

**정답:**
```python
self.tok_emb  = nn.Embedding(cfg["vocab_size"],      cfg["emb_dim"])
self.pos_emb  = nn.Embedding(cfg["context_length"],  cfg["emb_dim"])
self.out_head = nn.Linear(   cfg["emb_dim"],          cfg["vocab_size"], bias=False)
```

**이유와 각 크기의 의미:**

| 레이어 | 입력 크기 | 출력 크기 | 이유 |
|--------|---------|---------|------|
| `tok_emb` | `vocab_size=50257` | `emb_dim=768` | 5만 개 단어 각각을 768차원 벡터로 |
| `pos_emb` | `context_length=1024` | `emb_dim=768` | 최대 1024개 위치 각각을 768차원 벡터로 |
| `out_head` | `emb_dim=768` | `vocab_size=50257` | 768차원 → 5만 개 단어 각각의 점수(logit)로 |

`out_head`의 출력이 `vocab_size`인 이유: "다음에 올 단어 50257개 중 어느 것일 확률이 가장 높은가?"를 계산해야 하기 때문입니다.

---

### ❓ Q18. GPTModel forward 흐름 빈칸

```python
# 빈칸 위치:
x = tok_embeds + ????         # 임베딩 합산
x = self.????(x)              # 트랜스포머 블록 통과
logits = self.????(x)         # 최종 출력
```

**정답:**
```python
x = tok_embeds + pos_embeds   # 토큰 임베딩 + 위치 임베딩
x = self.trf_blocks(x)        # TransformerBlock 12개 순차 통과
logits = self.out_head(x)     # vocab_size 차원의 로짓 출력
```

**이유:**
- `pos_embeds`: `pos_emb(torch.arange(seq_len, device=...))` 로 생성한 위치 임베딩
- `self.trf_blocks`: `nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])` — Sequential이므로 그냥 `(x)` 호출만으로 n_layers번 순차 처리
- `self.out_head`: 최종 `nn.Linear(emb_dim, vocab_size)` — 각 위치의 다음 토큰 예측 점수

---

### ❓ Q19. generate_text_simple 빈칸 4개

```python
# 빈칸 위치:
with torch.????():           # ① 기울기 계산 비활성화
    logits = model(idx_cond)

logits = logits[:, ????, :]  # ② 마지막 시점 선택

idx_next = torch.????(logits, dim=-1, keepdim=True)  # ③ 최대값 인덱스

idx = torch.????((idx, idx_next), dim=1)  # ④ 이어붙이기
```

**정답:**
```python
with torch.no_grad():              # ①
logits = logits[:, -1, :]          # ②
idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # ③
idx = torch.cat((idx, idx_next), dim=1)  # ④
```

**이유:**

| 번호 | 코드 | 이유 |
|------|------|------|
| ① | `torch.no_grad()` | 추론 시 기울기 계산 불필요 → 메모리 절약, 속도 향상 |
| ② | `[:, -1, :]` | GPT는 시퀀스 전체를 출력하지만 **마지막 위치**만이 "다음 토큰"을 예측한 값 |
| ③ | `torch.argmax` | Greedy Decoding — 가장 높은 점수(확률)의 토큰 ID 선택 |
| ④ | `torch.cat(..., dim=1)` | 기존 시퀀스에 새 토큰을 뒤에 이어붙여 다음 스텝의 입력으로 사용 |

`logits[:, -1, :]`에서 `-1`이 핵심: shape `[batch, seq_len, vocab_size]`에서 `seq_len` 차원의 마지막 원소만 선택합니다.

---

## Chapter 5 빈칸 문제

### ❓ Q20. CrossEntropyLoss flatten 처리

```python
# 빈칸 위치:
loss = torch.nn.functional.????(
    logits.????(0, 1),
    target_batch.????()
)
```

**정답:**
```python
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),
    target_batch.flatten()
)
```

**이유:**
PyTorch의 `cross_entropy`는 입력을 `[N, C]` 형태로 받습니다.
- `N`: 샘플 수 / `C`: 클래스 수(vocab_size)

하지만 모델 출력 logits의 shape은 `[batch, seq_len, vocab_size]`이므로:
```
logits.flatten(0, 1)  → [batch × seq_len, vocab_size]   (N=batch×seq_len, C=vocab_size)
target_batch.flatten() → [batch × seq_len]                (N개 정답 레이블)
```
`flatten(0, 1)`: 0번째와 1번째 차원을 하나로 합칩니다 (batch × seq_len)
`flatten()`: 모든 차원을 1차원으로 펼칩니다

---

### ❓ Q21. 학습 루프 4단계 빈칸

```python
# 빈칸 위치:
model.????()              # ① 학습 모드 전환
optimizer.????()          # ② 기울기 초기화
loss = ????(...)          # ③ 손실 계산
loss.????()               # ④ 역전파
optimizer.????()          # ⑤ 가중치 업데이트
```

**정답:**
```python
model.train()                                               # ①
optimizer.zero_grad()                                       # ②
loss = calc_loss_batch(input_batch, target_batch, model, device)  # ③
loss.backward()                                             # ④
optimizer.step()                                            # ⑤
```

**이유 (각 단계가 중요한 이유):**

```
① model.train()
   → Dropout을 활성화 상태로 전환 (과적합 방지 효과 on)
   → model.eval()과 항상 쌍으로 사용
   
② optimizer.zero_grad()
   → PyTorch는 기본적으로 기울기를 누적(accumulate)합니다
   → 매 배치마다 반드시 초기화해야 이전 배치의 기울기가 섞이지 않음
   → 빠뜨리면 이전 배치들의 기울기가 누적된 채로 업데이트되어 학습이 망가짐
   → [claude] 정확히는 "exploding gradient"(깊은 층의 연쇄 곱으로 기울기가 발산하는 현상)와는
      별개의 문제입니다. 다만 의도적으로 누적시켜 큰 배치를 흉내내는 gradient accumulation 기법도 있습니다
   
③ calc_loss_batch(...)
   → 순전파(forward pass) 수행 후 손실값 계산
   
④ loss.backward()
   → 역전파(backpropagation): 연쇄 법칙으로 각 파라미터의 기울기 계산
   → 계산된 기울기는 각 파라미터의 .grad 속성에 저장됨
   
⑤ optimizer.step()
   → 계산된 기울기를 이용해 파라미터 업데이트
   → W = W - lr × ∇W (경사 하강법)
```

---

### ❓ Q22. text_to_token_ids / token_ids_to_text 차원 처리

```python
# 빈칸 위치:
encoded_tensor = torch.tensor(encoded).????(0)  # 배치 차원 추가
flat = token_ids.????(0)                        # 배치 차원 제거
```

**정답:**
```python
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # [seq_len] → [1, seq_len]
flat = token_ids.squeeze(0)                           # [1, seq_len] → [seq_len]
```

**이유:**
- `unsqueeze(0)`: 0번 차원에 크기 1인 차원 추가 → 모델이 요구하는 `[batch, seq_len]` 형태로 변환 (batch=1)
- `squeeze(0)`: 0번 차원의 크기 1인 차원 제거 → `tokenizer.decode(list)`가 요구하는 1차원 리스트로 변환

```
[12, 34, 56]           ← 일반 텐서 [seq_len]
[[12, 34, 56]]         ← unsqueeze(0) 후 [1, seq_len] (모델 입력 형태)
[12, 34, 56]           ← squeeze(0) 후 [seq_len] (decode 입력 형태)
```

---

### ❓ Q23. AdamW 옵티마이저 인자

```python
# 빈칸 위치:
optimizer = torch.optim.AdamW(
    model.????(), ????=settings["learning_rate"], weight_decay=settings["weight_decay"]
)
```

**정답:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
)
```

**이유:**
- `model.parameters()`: 학습 가능한 모든 파라미터를 반환하는 이터레이터. 옵티마이저가 어떤 파라미터를 업데이트할지 알 수 있도록 전달
- `lr=` : 학습률(learning rate) 키워드 인자명
- `weight_decay`: L2 규제 강도. AdamW에서는 가중치 감쇠를 기울기 업데이트와 분리하여 올바르게 적용 (일반 Adam과의 핵심 차이)

---

## Chapter 6 빈칸 문제

### ❓ Q24. SpamDataset — 트런케이션과 패딩

```python
# 빈칸 위치:
# 트런케이션 (자르기):
self.encoded_texts = [encoded_text[:????] for ...]

# 패딩 (채우기):
self.encoded_texts = [
    encoded_text + [????] * (self.max_length - len(encoded_text))
    for ...
]

# 레이블 접근:
label = self.data.iloc[index]["????"]
```

**정답:**
```python
encoded_text[:self.max_length]           # 트런케이션: max_length까지만 유지
+ [pad_token_id] * (self.max_length - len(encoded_text))  # 패딩: 나머지를 패딩으로
label = self.data.iloc[index]["Label"]   # 레이블 컬럼명
```

**이유:**
- 트런케이션(`[:self.max_length]`): 너무 긴 문장은 모델의 context_length를 초과하므로 잘라냄
- 패딩(`[pad_token_id] * N`): DataLoader의 배치 처리는 모든 시퀀스 길이가 같아야 함. 짧은 것은 `pad_token_id`(GPT-2에서는 50256, `<|endoftext|>`)로 채움
- `"Label"`: CSV 파일의 정답 컬럼 이름. `0`(ham) 또는 `1`(spam)

---

### ❓ Q25. 분류에서 마지막 토큰 출력 사용

```python
# 빈칸 위치:
logits = model(input_batch)[:, ????, :]
```

**정답:**
```python
logits = model(input_batch)[:, -1, :]
```

**이유:**
GPT는 언어 모델이기 때문에 각 위치마다 출력을 생성합니다.
분류 작업에서는 **전체 문장의 의미를 집약한 표현**이 필요한데, causal mask 때문에 시퀀스 전체를 참조할 수 있는 토큰은 **마지막 토큰(`-1`)뿐**입니다. 그래서 마지막 위치의 출력을 사용합니다.

```
[batch, seq_len, num_classes]      ← out_head를 Linear(768, 2)로 교체한 뒤의 출력
             ↓ [:, -1, :]
[batch, num_classes]  ← 이 값이 곧 클래스 로짓 (cross_entropy에 바로 투입)
```

> **[claude] 수정** — 원문은 shape을 `[batch, seq_len, vocab_size]` → `[batch, vocab_size]`로 적고 "이걸 분류 헤드에 넣음"이라고 했는데, 두 가지가 틀렸습니다.
> 1. 이 시점에는 이미 `model.out_head`가 `Linear(768, 2)`로 **교체된 뒤**이므로 마지막 차원은 `vocab_size(50257)`가 아니라 `num_classes(2)`입니다.
> 2. 분류 헤드는 모델 **안**에 있습니다. `[:, -1, :]` 슬라이싱은 헤드를 **통과한 뒤**에 일어나므로 "분류 헤드에 넣는다"는 순서가 반대입니다.

---

### ❓ Q26. 파라미터 동결(Freeze) / 해제(Unfreeze)

```python
# 빈칸 위치:
for param in model.parameters():
    param.requires_grad = ????          # 전체 동결

for param in model.trf_blocks[????].parameters():
    param.requires_grad = ????          # 마지막 블록 해제

for param in model.final_norm.parameters():
    param.requires_grad = ????          # 최종 정규화 층 해제

model.out_head = torch.nn.Linear(..., out_features=????)  # 출력 헤드 교체
```

**정답:**
```python
param.requires_grad = False   # 전체 동결
model.trf_blocks[-1]          # 마지막 트랜스포머 블록 (인덱스 -1)
param.requires_grad = True    # 마지막 블록 해제
param.requires_grad = True    # 최종 정규화 층 해제
out_features=num_classes      # num_classes = 2 (ham/spam)
```

**이유:**

```
전략: "사전 학습된 언어 이해 능력은 유지, 분류 작업만 새로 학습"

requires_grad = False (동결):
  → 해당 파라미터는 backward() 계산 시 기울기가 계산되지 않음
  → optimizer.step()이 호출돼도 값이 바뀌지 않음

requires_grad = True (해제):
  → 기울기 계산 대상이 되어 학습됨

trf_blocks[-1] (마지막 블록만 해제):
  → 모델 깊이가 12층이라면, 12번째(마지막) 블록만 조금 조정
  → 이 블록이 최종 출력과 가장 직접 연결되어 있으므로 효과적

out_features=2:
  → 기존 GPT의 out_head는 vocab_size=50257개 출력
  → 스팸 분류는 2개 클래스(0/1)만 있으므로 교체 필요
  → 새 Linear 레이어는 기본적으로 requires_grad=True 상태
```

---

## Chapter 6 (LoRA) 빈칸 문제

### ❓ Q27. LoRALayer — A, B 행렬 크기

```python
# 빈칸 위치:
self.A = nn.Parameter(torch.empty(in_dim, ????))   # A 행렬
self.B = nn.Parameter(torch.zeros(????, out_dim))  # B 행렬
```

**정답:**
```python
self.A = nn.Parameter(torch.empty(in_dim, rank))   # [in_dim, rank]
self.B = nn.Parameter(torch.zeros(rank, out_dim))  # [rank, out_dim]
```

**이유:**
LoRA의 핵심 아이디어: 기존 가중치 `W`는 **얼어붙은 채 그대로 두고**, 학습으로 생기는 **변화량 ΔW**만 두 작은 행렬의 곱 `A × B`로 근사합니다.

> **[claude] 수정** — 원문은 "가중치 행렬 `W`를 `A × B`로 근사한다"고 썼는데, 이는 LoRA가 아니라 일반적인 저랭크 행렬 분해입니다. LoRA는 `W`를 근사하지 않습니다. `W`는 동결된 채 남고 `ΔW ≈ A × B`만 학습하며, 최종 출력은 `y = Wx + (A×B)x`입니다. 이 문서 뒤쪽 Q29의 `output = W·x + ΔW·x` 설명이 정확한 쪽입니다.

```
ΔW (in_dim × out_dim) ≈ A (in_dim × rank) × B (rank × out_dim)

rank << min(in_dim, out_dim)이면 파라미터 수가 크게 줄어듦

예: in_dim=768, out_dim=768, rank=16인 경우
기존: 768 × 768 = 589,824 파라미터
LoRA: 768 × 16 + 16 × 768 = 24,576 파라미터 (약 96% 절감)
```

- `A`: kaiming_uniform으로 초기화 (non-zero → 학습 신호 전달 가능)
- `B`: 0으로 초기화 (학습 시작 시 LoRA 효과 = 0 → 기존 모델 동작 보존)

---

### ❓ Q28. LoRA forward 수식

```python
# 빈칸 위치:
x = (self.???? / self.rank) * (x @ self.???? @ self.????)
```

**정답:**
```python
x = (self.alpha / self.rank) * (x @ self.A @ self.B)
```

**이유:**
- `x @ self.A`: `[..., in_dim] × [in_dim, rank]` → `[..., rank]` (차원 축소)
- `@ self.B`: `[..., rank] × [rank, out_dim]` → `[..., out_dim]` (차원 복원)
- `self.alpha / self.rank`: 스케일링 팩터. `alpha`는 하이퍼파라미터로 LoRA의 학습 강도를 조절함. `alpha=rank`이면 스케일링 = 1

---

### ❓ Q29. LinearWithLoRA forward

```python
# 빈칸 위치:
return self.linear(x) + self.????(x)
```

**정답:**
```python
return self.linear(x) + self.lora(x)
```

**이유:**
기존 `Linear` 레이어의 출력에 LoRA 어댑터의 출력을 **더합니다**.
- `self.linear(x)`: 고정된 원래 가중치의 출력 (역전파 시 기울기 없음)
- `self.lora(x)`: 학습 가능한 LoRA 어댑터의 출력 (역전파 시 기울기 계산)

기존 동작을 방해하지 않으면서 작은 조정(delta)만 더하는 방식입니다:
`output = W·x + ΔW·x = W·x + (A×B)·x`

---

### ❓ Q30. replace_linear_with_lora — 타입 체크

```python
# 빈칸 위치:
if isinstance(module, torch.nn.????):
    setattr(model, name, LinearWithLoRA(module, rank, alpha))
else:
    replace_linear_with_lora(module, rank, alpha)
```

**정답:**
```python
if isinstance(module, torch.nn.Linear):
```

**이유:**
- `torch.nn.Linear`: 교체 대상이 되는 타입. 모델 내 모든 Linear 레이어를 LoRA 버전으로 바꿉니다.
- `isinstance()`: Python 내장 함수. 객체가 특정 클래스(또는 그 하위 클래스)의 인스턴스인지 확인
- `else` 절의 재귀 호출: Linear가 아닌 모듈(예: TransformerBlock, MultiHeadAttention 등)은 내부를 탐색해야 하므로 재귀적으로 처리

---

## Chapter 7 빈칸 문제

### ❓ Q31. custom_collate_fn — EOS 토큰 추가 및 패딩

```python
# 빈칸 위치:
new_item += [????]   # EOS 토큰 추가
padded = new_item + [????] * (batch_max_length - len(new_item))  # 패딩
```

**정답:**
```python
new_item += [pad_token_id]   # pad_token_id = 50256
padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
```

**이유:**
- `pad_token_id = 50256`: GPT-2의 `<|endoftext|>` 토큰 ID. 문장 끝과 패딩 모두 같은 토큰 사용
- 먼저 EOS 토큰 1개를 추가해 문장이 끝났음을 표시한 뒤, 나머지를 패딩으로 채움
- 배치 내 모든 시퀀스 길이를 `batch_max_length`로 동일하게 맞춰야 텐서로 변환 가능

---

### ❓ Q32. custom_collate_fn — 입력/타겟 슬라이싱

```python
# 빈칸 위치:
inputs  = torch.tensor(padded[????])  # 입력: 마지막 토큰 제외
targets = torch.tensor(padded[????])  # 타겟: 첫 번째 토큰 제외
```

**정답:**
```python
inputs  = torch.tensor(padded[:-1])  # 0 ~ n-1 번째 토큰
targets = torch.tensor(padded[1:])   # 1 ~ n 번째 토큰
```

**이유:**
다음 토큰 예측 학습의 기본 원리 (1 shift):
```
padded = [t0, t1, t2, t3, t4, PAD, PAD]
inputs  = [t0, t1, t2, t3, t4, PAD]   ← [:-1]: 마지막 제외
targets = [t1, t2, t3, t4, PAD, PAD]  ← [1:]:  첫 번째 제외

위치 0의 예측 문제: 입력=t0, 정답=t1
위치 1의 예측 문제: 입력=t1, 정답=t2
...
```

---

### ❓ Q33. custom_collate_fn — 패딩 마스킹 인덱스

```python
# 빈칸 위치:
if indices.numel() > 1:
    targets[indices[????:]] = ignore_index  # ignore_index = -100
```

**정답:**
```python
targets[indices[1:]] = ignore_index
```

**이유:**
이 처리가 왜 필요한지, `1:`이 왜 정답인지:

```
예시:
targets  = [t1, t2, t3, PAD, PAD, PAD]
indices (PAD 위치) = [3, 4, 5]

indices[0] = 3  → 첫 번째 PAD (실제 EOS 역할, 유지!)
indices[1:] = [4, 5]  → 나머지 패딩 (ignore_index로 대체)

결과:
targets = [t1, t2, t3, PAD, -100, -100]
```

- `indices[0]` (첫 번째 패딩): EOS 토큰에 해당 → 모델이 "문장 끝"을 학습하도록 **손실 계산에 포함**
- `indices[1:]` (두 번째 이후 패딩): 의미 없는 패딩 → `ignore_index=-100`으로 마킹
- PyTorch `CrossEntropyLoss`는 `ignore_index=-100`인 위치를 **자동으로 손실 계산에서 제외**

---

## 전체 빈칸 한눈에 정리

| 챕터 | 빈칸 | 정답 | 핵심 개념 |
|------|------|------|---------|
| Ch2 | `tokenizer.????` | `encode` / `decode` | BPE 토크나이저 |
| Ch2 | y 인덱스 | `1:context_size+1` | 1 shift (다음 토큰 예측) |
| Ch2 | input/target 슬라이스 | `i:i+max_length` / `i+1:i+max_length+1` | 슬라이딩 윈도우 |
| Ch2 | `__getitem__` 반환 | `self.input_ids[idx], self.target_ids[idx]` | 입력/정답 쌍 |
| Ch2 | DataLoader 첫 인자 | `dataset` | PyTorch Dataset |
| Ch2 | Embedding 인자 | `vocab_size, output_dim` / `context_length, output_dim` | 임베딩 크기 |
| Ch2 | 임베딩 합산 | `token_embeddings + pos_embeddings` | 위치 정보 부여 |
| Ch3 | QKV 레이어 타입 | `nn.Linear(d_in, d_out)` | 선형 투영 |
| Ch3 | Q,K,V 계산 | `self.W_query(x)` 등 | 투영 적용 |
| Ch3 | 어텐션 스코어 | `queries @ keys.transpose(1,2)` | Q·Kᵀ 내적 |
| Ch3 | 마스킹 값 | `-torch.inf` | softmax 후 0 보장 |
| Ch3 | Softmax | `torch.softmax(.../ keys.shape[-1]**0.5, dim=-1)` | Scaled Dot-Product |
| Ch3 | Context Vector | `attn_weights @ values` | 가중 합산 |
| Ch4 | LayerNorm 수식 | `(x-mean)/sqrt(var+eps)` | 정규화 공식 |
| Ch4 | LayerNorm 파라미터 | `self.scale * norm_x + self.shift` | γ, β |
| Ch4 | FeedForward 확장 | `4 * cfg["emb_dim"]`, `GELU()` | 4배 확장 + GELU |
| Ch4 | TransformerBlock | `self.att(x)`, `self.ff(x)` | Residual 연결 |
| Ch4 | GPTModel Embedding | `cfg["vocab_size"]`, `cfg["context_length"]` | 크기 선택 |
| Ch4 | GPTModel out_head | `cfg["vocab_size"]` | 단어 예측 출력 |
| Ch4 | GPTModel forward | `pos_embeds`, `self.trf_blocks(x)`, `self.out_head(x)` | 순전파 흐름 |
| Ch4 | generate 빈칸 | `no_grad`, `[:,-1,:]`, `argmax`, `cat` | 자기회귀 생성 |
| Ch5 | CrossEntropy | `cross_entropy`, `flatten(0,1)`, `flatten()` | 손실 계산 |
| Ch5 | 학습 루프 | `train()`, `zero_grad()`, `backward()`, `step()` | 4단계 학습 |
| Ch5 | 차원 변환 | `unsqueeze(0)`, `squeeze(0)` | 배치 차원 처리 |
| Ch5 | AdamW | `model.parameters()`, `lr=` | 옵티마이저 |
| Ch6 | SpamDataset | `[:self.max_length]`, `pad_token_id`, `"Label"` | 패딩/트런케이션 |
| Ch6 | 마지막 토큰 | `[:, -1, :]` | 분류용 출력 선택 |
| Ch6 | Freeze/Unfreeze | `False`, `trf_blocks[-1]`, `True` | 선택적 학습 |
| Ch6 | 출력 헤드 교체 | `out_features=num_classes` | 분류 헤드 |
| Ch6 LoRA | A, B 행렬 크기 | `(in_dim, rank)`, `(rank, out_dim)` | 저랭크 분해 |
| Ch6 LoRA | LoRA forward | `alpha/rank * (x @ A @ B)` | 저랭크 근사 |
| Ch6 LoRA | LinearWithLoRA | `linear(x) + lora(x)` | 잔차 방식 추가 |
| Ch6 LoRA | 타입 체크 | `torch.nn.Linear` | 교체 대상 |
| Ch7 | EOS + 패딩 | `pad_token_id` | 시퀀스 종료 표시 |
| Ch7 | 입력/타겟 슬라이스 | `padded[:-1]`, `padded[1:]` | 1 shift |
| Ch7 | 패딩 마스킹 | `indices[1:]` = `-100` | 손실 계산 제외 |


---
---

# 🔷 Encoder / Decoder 구조 도식화 해설

> 트랜스포머(Transformer)는 크게 **Encoder**와 **Decoder**로 구성됩니다.
> 이 코드에서 다루는 GPT는 **Decoder Only** 모델입니다. 두 구조의 차이를 명확히 이해하면 전체 흐름이 보입니다.

---

## 1. Encoder vs Decoder — 한눈에 보기

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        트랜스포머 전체 구조                                  │
│                                                                           │
│    ┌─────────────┐              ┌─────────────────┐                      │
│    │   ENCODER   │              │    DECODER      │                      │
│    │             │              │                 │                      │
│    │ "입력 문장   │              │ "출력 문장       │                      │
│    │  전체를      │  Context ──▶│  을 한 토큰씩    │                      │
│    │  동시에      │  Vector     │  생성"           │                      │
│    │  이해"      │              │                 │                      │
│    └─────────────┘              └─────────────────┘                      │
│                                                                           │
│    예) BERT, RoBERTa            예) GPT, ChatGPT                         │
│    용도: 분류, 검색, NLU        용도: 텍스트 생성, 번역 출력부분            │
└──────────────────────────────────────────────────────────────────────────┘
```

> **[claude] 주의 — 헷갈리기 쉬운 지점**
> 위 그림의 화살표(Encoder → Context Vector → Decoder)는 **원 Transformer(번역 모델)**의 구조입니다. 그 Decoder에는 인코더 출력을 받는 **Cross-Attention** 층이 블록마다 하나 더 들어 있어, 한 블록이 `Masked Self-Attn → Cross-Attn → FFN` 3단으로 구성됩니다.
> 반면 **GPT는 Decoder-Only**라서 Encoder도, Cross-Attention도 **없습니다**. 이 노트북의 `TransformerBlock`은 `Masked Self-Attn → FFN` 2단뿐입니다. 아래 "Decoder 내부 처리 절차"는 원 Transformer의 Decoder가 아니라 **GPT 블록**을 설명한 것으로 읽어야 합니다.

---

## 2. Encoder 내부 처리 절차 (단계별 도식)

Encoder는 **입력 문장 전체를 양방향으로 이해**하는 구조입니다 (BERT 계열).

```
입력 텍스트: "I love Python"
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 1: 토크나이저 (Tokenizer)                         │
│                                                        │
│  "I love Python"                                       │
│       ↓ encode()                                       │
│  [40, 1842, 11361]  ← 토큰 ID 리스트                    │
│  [claude] 원문의 314는 앞에 공백이 붙은 " I"입니다.        │
│          문장 첫머리의 "I"는 40. (정확한 값은            │
│          tiktoken으로 직접 확인 권장)                    │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 2: 토큰 임베딩 + 위치 임베딩                        │
│                                                        │
│  토큰 임베딩:   [I]→[0.3,0.1,...] [love]→[0.7,0.2,...]  │
│  위치 임베딩:   pos0→[p0] pos1→[p1] pos2→[p2]           │
│                        +                               │
│  최종 입력:   [I+p0] [love+p1] [Python+p2]              │
│               shape: [batch, seq_len, emb_dim]          │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 3: Multi-Head Self-Attention (양방향)              │
│                                                        │
│  "I"가 "love", "Python"을 모두 참조 가능                  │
│  "love"가 "I", "Python"을 모두 참조 가능                  │
│  "Python"이 "I", "love"를 모두 참조 가능                  │
│                                                        │
│  마스크 없음 → 과거/미래 모두 봄 (양방향)                   │
│                                                        │
│  ┌──────┐  ┌──────┐  ┌──────┐                         │
│  │  I   │←→│ love │←→│Python│  ← 서로 참조             │
│  └──────┘  └──────┘  └──────┘                         │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 4: Add & Norm (잔차 연결 + LayerNorm)              │
│                                                        │
│  출력 = LayerNorm(Attention(x) + x)   ← Post-LN 방식     │
│                              ↑                         │
│                          잔차 연결 (원본 더하기)           │
│  [claude] 원 논문(BERT 포함)은 Post-LN.                  │
│          GPT-2 코드는 Pre-LN이라 순서가 반대입니다        │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 5: Feed-Forward Network                          │
│                                                        │
│  각 토큰 독립적으로:                                      │
│  emb_dim → 4×emb_dim → GELU → emb_dim                 │
│                                                        │
│  [I벡터]   [love벡터]   [Python벡터]  (독립 처리)         │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 6: Add & Norm (다시 잔차 연결 + LayerNorm)          │
└───────────────────────────────────────────────────────┘
        │
  (Step 3~6 N번 반복)
        │
        ▼
   Context Vector (각 토큰의 문맥적 표현)
```

---

## 3. Decoder 내부 처리 절차 (단계별 도식)

Decoder는 **이전에 생성한 토큰들만 참조하며 다음 토큰을 예측**하는 구조입니다 (GPT 계열).

```
생성 목표: "I love" 다음 단어 예측
현재 문맥: ["I", "love"]
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 1: 토크나이저 (Tokenizer)                         │
│                                                        │
│  ["I", "love"]                                         │
│       ↓ encode()                                       │
│  [314, 2012]  ← 토큰 ID 리스트                          │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 2: 토큰 임베딩 + 위치 임베딩                        │
│                                                        │
│  [I+pos0]  [love+pos1]                                  │
│  shape: [batch, seq_len, emb_dim]                      │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 3: LayerNorm → Masked Multi-Head Self-Attention   │
│          (단방향) ← GPT의 핵심! Encoder와 다른 점!        │
│          [claude] 정규화가 어텐션 '앞'에 옵니다 (Pre-LN)  │
│                                                        │
│  Causal Mask (인과적 마스크) 적용:                        │
│                                                        │
│  "I"    →  자기자신만 참조 가능                            │
│  "love" →  "I", "love" 만 참조 가능 (미래 불가)            │
│                                                        │
│  어텐션 마스크 행렬:                                       │
│       I    love                                        │
│  I  [ 0    -inf ]    0 = 참조 가능                       │
│  love[ 0     0  ]  -inf = 차단 (softmax 후 0이 됨)       │
│                                                        │
│  ┌──────┐  ┌──────┐                                    │
│  │  I   │──│ love │   love → I 참조 가능                 │
│  └──────┘  └──────┘   I → love 참조 불가 (미래)           │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 4: Dropout → Add (잔차 연결)                       │
│          x = x + shortcut                              │
│  [claude] 이 코드는 Pre-LN이므로 "Add & Norm"이 아니라    │
│          Norm이 Step 3 앞에 오고 여기서는 Add만 합니다     │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 5: Feed-Forward Network                          │
│                                                        │
│  emb_dim → 4×emb_dim → GELU → emb_dim                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 6: Dropout → Add (잔차 연결)                       │
│          x = x + shortcut                              │
└───────────────────────────────────────────────────────┘
        │
  (Step 3~6 N번 반복)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 7: Final LayerNorm + Linear (출력 헤드)            │
│                                                        │
│  Linear(emb_dim → vocab_size) 를 '모든 위치'에 적용        │
│  → logits shape: [batch, seq_len, vocab_size]          │
│                                                        │
│  [claude] 수정: [:, -1, :] 는 모델 내부가 아니라           │
│  생성 루프에서 '이후에' 골라내는 단계입니다.                │
│  학습 때는 모든 위치의 로짓을 손실 계산에 사용합니다.        │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 7-2: 마지막 위치 로짓만 추출 (생성 시에만)            │
│  logits[:, -1, :]  → [batch, vocab_size]               │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Step 8: 다음 토큰 선택                                  │
│                                                        │
│  argmax(logits) = 가장 높은 점수의 토큰 ID                 │
│  예: ID 11361 → "Python"                               │
└───────────────────────────────────────────────────────┘
        │
        ▼
   ["I", "love", "Python"]  ← 생성된 결과
```

---

## 4. Encoder vs Decoder 핵심 차이 비교표

```
┌──────────────────┬────────────────────────┬────────────────────────┐
│     항목          │      Encoder           │      Decoder (GPT)     │
├──────────────────┼────────────────────────┼────────────────────────┤
│ 어텐션 방향       │ 양방향 (Bidirectional)  │ 단방향 (Causal/Left)    │
│                  │ 과거 + 미래 모두 참조     │ 과거만 참조              │
├──────────────────┼────────────────────────┼────────────────────────┤
│ Causal Mask      │ 없음                    │ 있음 (상삼각 -inf 마스크) │
├──────────────────┼────────────────────────┼────────────────────────┤
│ 주요 용도        │ 문장 이해/분류/검색        │ 텍스트 생성              │
├──────────────────┼────────────────────────┼────────────────────────┤
│ 학습 방식        │ MLM (마스크 토큰 예측)    │ CLM (다음 토큰 예측)     │
│                  │ [MASK] → 원래 단어 예측  │ t_n → t_(n+1) 예측     │
├──────────────────┼────────────────────────┼────────────────────────┤
│ 대표 모델        │ BERT, RoBERTa, ALBERT   │ GPT-2, GPT-3, ChatGPT  │
├──────────────────┼────────────────────────┼────────────────────────┤
│ 입출력 관계      │ 입력 전체 → 문맥 벡터     │ 이전 출력 → 다음 토큰    │
└──────────────────┴────────────────────────┴────────────────────────┘
```

---

## 5. GPT(Decoder Only)의 자기회귀 생성 전체 흐름

```
시작 입력: "Hello, I am"
                │
                ▼
           [토크나이저]
      [15496, 11, 314, 716]
                │
                ▼
    ┌─────────────────────┐
    │ GPT 모델 (Decoder)  │
    │                     │
    │ 토큰 임베딩          │
    │     +               │
    │ 위치 임베딩          │
    │     │               │
    │     ▼               │
    │ TransformerBlock ×12│
    │ (각 블록 내부:       │
    │  LayerNorm           │
    │  → MaskedAttention   │
    │  → Add              │
    │  → LayerNorm         │
    │  → FeedForward       │
    │  → Add)              │
    │     │               │
    │     ▼               │
    │ Final LayerNorm     │
    │     │               │
    │     ▼               │
    │ Linear (out_head)   │
    │ [emb_dim → 50257]   │
    └─────────────────────┘
                │
                ▼
    logits shape: [1, 4, 50257]
                │
                ▼
    logits[:, -1, :]  ← 마지막 위치만 추출 [1, 50257]
                │
                ▼
    argmax → 예: 토큰 ID 318 ("a")
                │
                ▼
    새 시퀀스: [15496, 11, 314, 716, 318]
    "Hello, I am a"
                │
             (반복)
                ▼
    최종: "Hello, I am a software engineer"
```

---

## 6. TransformerBlock 내부 흐름 (코드와 매핑)

```
입력 x (shape: [batch, seq_len, emb_dim])
    │
    ├──── shortcut = x  ← 잔차 연결을 위해 원본 저장
    │
    ▼
┌──────────────────────────────────────────┐
│  LayerNorm 1   (self.norm1)              │
│  x = (x - mean) / √(var + ε) × γ + β    │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Multi-Head Attention  (self.att)        │
│                                          │
│  Q = W_query(x)                          │
│  K = W_key(x)                            │
│  V = W_value(x)                          │
│                    ↓                     │
│  scores = Q @ K^T / √d_k                 │
│                    ↓                     │
│  scores.masked_fill_(mask, -inf)         │
│                    ↓                     │
│  weights = softmax(scores)               │
│                    ↓                     │
│  output = weights @ V                    │
│                    ↓                     │
│  output = out_proj(output)               │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Dropout  (self.drop_shortcut)           │
└──────────────────────────────────────────┘
    │
    ▼
x = x + shortcut  ← 잔차 연결 (원본 더하기)
    │
    ├──── shortcut = x  ← 두 번째 잔차 연결용 저장
    │
    ▼
┌──────────────────────────────────────────┐
│  LayerNorm 2   (self.norm2)              │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Feed-Forward  (self.ff)                 │
│                                          │
│  Linear(emb_dim → 4×emb_dim)            │
│         ↓                               │
│       GELU()                             │
│         ↓                               │
│  Linear(4×emb_dim → emb_dim)            │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Dropout  (self.drop_shortcut)           │
└──────────────────────────────────────────┘
    │
    ▼
x = x + shortcut  ← 잔차 연결
    │
    ▼
출력 x (shape: [batch, seq_len, emb_dim])  ← 입력과 shape 동일
```

---

## 7. 학습(Training) vs 생성(Inference) 흐름 비교

```
┌─────────────────────────────────────────────────────────────┐
│                   학습(Training) 흐름                         │
└─────────────────────────────────────────────────────────────┘

텍스트 데이터
    │
    ▼
GPTDatasetV1 (슬라이딩 윈도우)
    │ input: [t0, t1, t2, t3]
    │ target:[t1, t2, t3, t4]   ← 1 shift
    ▼
DataLoader (배치 단위로 묶기)
    │
    ▼
model.train()  → Dropout ON
    │
    ▼
optimizer.zero_grad()  → 기울기 초기화
    │
    ▼
logits = model(inputs)  → 순전파
    │ logits shape: [batch, seq_len, vocab_size]
    ▼
loss = cross_entropy(logits.flatten(0,1), targets.flatten())
    │
    ▼
loss.backward()  → 역전파: 각 파라미터의 기울기 계산
    │
    ▼
optimizer.step()  → W = W - lr × ∇W
    │
    ▼
(반복 → 손실 감소)


┌─────────────────────────────────────────────────────────────┐
│                   생성(Inference) 흐름                        │
└─────────────────────────────────────────────────────────────┘

시작 프롬프트 텍스트
    │
    ▼
text_to_token_ids()  → 토큰 ID 텐서 [1, seq_len]
    │
    ▼
model.eval()  → Dropout OFF
    │
    ▼
with torch.no_grad():  → 기울기 계산 OFF
    │
    ▼
logits = model(idx[:, -context_size:])  → 순전파
    │ shape: [1, seq_len, vocab_size]
    ▼
logits[:, -1, :]  → 마지막 위치 로짓 [1, vocab_size]
    │
    ▼
argmax(logits)  → 가장 높은 점수의 토큰 ID 선택
    │
    ▼
torch.cat([idx, idx_next], dim=1)  → 시퀀스에 추가
    │
    ▼
(max_new_tokens번 반복)
    │
    ▼
token_ids_to_text()  → 최종 텍스트 출력
```
