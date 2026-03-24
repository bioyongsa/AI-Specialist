# LLM from Scratch — Chapter 2~7 코드 완전 해설

> 이 문서는 "Build a Large Language Model From Scratch" 실습 노트북(Chapter 2~7)의 핵심 코드를 단계별로 상세히 설명합니다.
> 각 챕터가 **왜 필요한지**, **어떻게 동작하는지**, **핵심 포인트**가 무엇인지를 중심으로 서술합니다.

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
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 5. 최종 선형 투영
        return self.out_proj(context_vec)
```

**Wrapper 방식 vs 통합 방식의 차이:**

```
Wrapper: [head1 forward] + [head2 forward] + ... → 순차 실행 (느림)
통합:    W_query로 전체 d_out 계산 → view로 헤드 분리 → 병렬 행렬 연산 (빠름)
```

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
학습 중 각 층의 출력값 분포가 크게 달라지면 학습이 불안정해집니다 (Internal Covariate Shift). LayerNorm은 각 토큰의 특성 벡터를 평균=0, 분산=1로 정규화하여 **학습 안정성**을 높입니다.

---

### 4-2. GELU (활성화 함수)

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
```

**ReLU와의 차이:** ReLU는 x<0이면 완전히 0으로 만들지만, GELU는 **부드럽게 0으로** 만들어 기울기 흐름이 더 자연스럽습니다. GPT 계열 모델이 선호하는 활성화 함수입니다.

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

**역할:** Attention이 "어떤 토큰을 참조할지" 결정한다면, FeedForward는 "참조한 정보를 어떻게 변환할지"를 담당합니다. 각 토큰에 대해 **독립적으로** 처리되며, 4배 확장을 통해 더 풍부한 표현을 학습합니다.

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
정답 토큰의 로짓 점수가 높을수록 손실이 낮아집니다. 모델이 틀릴수록 큰 패널티를 주는 방식입니다.

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

### 5-3. generate — 고급 텍스트 생성

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
print(f"학습 파라미터 수: {total_trainable:,}")  # ← 전체의 ~0.1% 수준
```

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
