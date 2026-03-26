# Chapter 4 — GPT 아키텍처 코드 핵심 설명

---

## GPT 전체 구조 흐름

```
입력 토큰 ID  [B, seq_len]
      │
      ├─ tok_emb  → [B, seq_len, 768]   토큰 임베딩 (어떤 단어?)
      ├─ pos_emb  → [B, seq_len, 768]   위치 임베딩 (몇 번째 위치?)
      └─ tok + pos → [B, seq_len, 768]
      │
      ▼ drop_emb (Dropout)
      │
      ▼ TransformerBlock × 12
      │  ┌─────────────────────────────────────┐
      │  │  LayerNorm → MultiHeadAttention      │
      │  │  + Residual Connection               │
      │  │  LayerNorm → FeedForward             │
      │  │  + Residual Connection               │
      │  └─────────────────────────────────────┘
      │
      ▼ final_norm (LayerNorm)
      │
      ▼ out_head: Linear(768 → 50257)
      │
      [B, seq_len, 50257]  ← logits (각 단어의 점수)
```

---

## 1. LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        self.eps   = 1e-5                                # 분모가 0이 되는 것 방지
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ (gamma): 학습 가능 스케일
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # β (beta):  학습 가능 이동

    def forward(self, x):
        mean  = x.mean(dim=-1, keepdim=True)                  # 마지막 차원(emb_dim) 평균
        var   = x.var(dim=-1, keepdim=True, unbiased=False)   # 분산
        norm_x = (x - mean) / torch.sqrt(var + self.eps)      # 정규화
        return self.scale * norm_x + self.shift               # 복원력 부여
```

**왜 LayerNorm이 필요한가?**

```
깊은 네트워크에서 레이어를 거칠수록 값이 폭발하거나 소실됨
→ 각 레이어 출력을 평균 0, 표준편차 1로 정규화 → 학습 안정화

dim=-1 (마지막 차원=emb_dim) 방향으로 정규화:
  → 각 토큰의 768차원 벡터를 독립적으로 정규화
  → 배치 크기나 시퀀스 길이에 무관 (BatchNorm과 차이)

scale(γ), shift(β)는 학습 가능:
  → 정규화 후에도 모델이 필요한 범위로 값 복원 가능
```

---

## 2. GELU 활성화 함수

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

**ReLU vs GELU 비교**

```
ReLU:  f(x) = max(0, x)   → 0 이하 완전히 0 (불연속)
GELU:  음수에서도 작은 값 통과 → 더 부드러운 비선형성
→ GPT/BERT 등 Transformer 계열에서 표준으로 사용
```

---

## 3. FeedForward (FFN)

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072 (4배 확장)
            GELU(),                                           # 비선형 활성화
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768 (압축)
        )
```

**차원 변화**

```
[B, seq_len, 768]
      ↓  Linear(768→3072)
[B, seq_len, 3072]   ← 4배 확장: 더 넓은 공간에서 특징 추출
      ↓  GELU
[B, seq_len, 3072]
      ↓  Linear(3072→768)
[B, seq_len, 768]    ← 다시 원래 크기로 압축
```

**왜 4배 확장하는가?**

```
Attention: 토큰 간 관계(전역적 처리)
FFN: 각 토큰의 내용을 개별적으로 변환(국소적 처리)
4배 확장 → 더 풍부한 특징 표현 공간 → 복잡한 패턴 학습 가능
(경험적으로 4배가 성능과 효율의 균형점)
```

---

## 4. TransformerBlock — Pre-LayerNorm + Residual

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # ── Attention 블록 ──────────────────────────────
        shortcut = x            # 원본 입력 저장 (잔차 연결용)
        x = self.norm1(x)       # Pre-LayerNorm (정규화 먼저!)
        x = self.att(x)         # Multi-Head Attention
        x = self.drop_shortcut(x)
        x = x + shortcut        # ✅ Residual Connection

        # ── FeedForward 블록 ────────────────────────────
        shortcut = x
        x = self.norm2(x)       # Pre-LayerNorm
        x = self.ff(x)          # FeedForward
        x = self.drop_shortcut(x)
        x = x + shortcut        # ✅ Residual Connection

        return x
```

**Pre-LayerNorm vs Post-LayerNorm**

```
Post-LayerNorm (원본 Transformer):
  x → Attention → + x → LayerNorm
  → 깊은 네트워크에서 학습 불안정

Pre-LayerNorm (GPT-2 이후):
  x → LayerNorm → Attention → + x
  → 학습 더 안정적 (현재 표준)
```

**Residual Connection의 역할**

```
shortcut = x
x = self.att(x)
x = x + shortcut   ← "변화량(잔차)"만 학습하도록 유도

효과:
1. Vanishing gradient 방지: 역전파 시 shortcut 경로로 gradient 직접 흐름
2. 초기 학습 안정성: 초기에 att(x)≈0 → x+0=x (identity mapping)
3. 더 깊은 네트워크 가능 (ResNet과 동일한 원리)
```

---

## 5. GPTModel — 전체 조립

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        # 토큰 임베딩: vocab_size개 단어 각각에 emb_dim 벡터
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # → [50257, 768] 룩업 테이블

        # 위치 임베딩: context_length개 위치 각각에 emb_dim 벡터
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # → [1024, 768] 룩업 테이블

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer 블록 n_layers개 쌓기
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 출력: emb_dim → vocab_size (각 단어의 점수)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):   # in_idx: [B, seq_len]
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        # [B, seq_len] → [B, seq_len, 768]

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # torch.arange(seq_len) = [0, 1, ..., seq_len-1]
        # → [seq_len, 768]

        x = tok_embeds + pos_embeds  # [B, seq_len, 768]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)       # 12개 블록 통과
        x = self.final_norm(x)
        logits = self.out_head(x)    # [B, seq_len, 50257]
        return logits
```

**tok_emb vs pos_emb 차이**

```
tok_emb:  nn.Embedding(vocab_size=50257, emb_dim=768)
          → 어떤 단어인지 (내용 정보)

pos_emb:  nn.Embedding(context_length=1024, emb_dim=768)
          → 몇 번째 위치인지 (순서 정보)

vocab_size ≠ context_length 주의!
tok_emb의 인덱스: 토큰 ID (0~50256)
pos_emb의 인덱스: 위치 인덱스 (0~1023)
```

---

## 6. generate_text_simple — 텍스트 생성

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 1. 최대 길이 초과 시 뒤에서부터 자름
        idx_cond = idx[:, -context_size:]
        # [B, min(seq_len, context_size)]

        # 2. 추론 (gradient 계산 불필요 → 메모리 절약)
        with torch.no_grad():
            logits = model(idx_cond)
        # logits: [B, seq_len, vocab_size]

        # 3. 마지막 위치의 logit만 사용
        logits = logits[:, -1, :]
        # [B, vocab_size]

        # 4. Greedy Decoding: 가장 높은 점수의 토큰 선택
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # [B, 1]

        # 5. 새 토큰을 시퀀스 뒤에 이어붙임
        idx = torch.cat((idx, idx_next), dim=1)
        # [B, seq_len + 1]

    return idx
```

**Greedy Decoding의 한계와 대안**

```
Greedy: 매번 가장 확률 높은 토큰 선택
→ 결정적(deterministic), 단조로운 문장 생성

대안:
Top-K Sampling  → 상위 K개 중 랜덤 선택
Top-p (Nucleus) → 누적 확률 p% 이내에서 선택
Temperature     → 확률 분포 날카로움 조절
```

---

## 7. GPT-2 Small 설정값 의미

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 단어장: BPE 기반 50K+ 서브워드
    "context_length": 1024, # 최대 처리 토큰 수
    "emb_dim": 768,         # 임베딩 차원 (hidden size)
    "n_heads": 12,          # Attention 헤드 수 → head_dim = 768/12 = 64
    "n_layers": 12,         # TransformerBlock 개수
    "drop_rate": 0.1,       # Dropout 확률
    "qkv_bias": False       # Q,K,V Linear에 bias 사용 여부
}
```

**파라미터 수 계산 (GPT-2 Small 기준)**

```
Embedding:   50257 × 768 = 38.6M  (tok_emb)
           + 1024  × 768 = 0.8M   (pos_emb)

각 TransformerBlock × 12:
  Attention QKV:  3 × 768 × 768 = 1.77M
  out_proj:       768 × 768 = 0.59M
  FFN:            768×3072 + 3072×768 = 4.72M
  LayerNorm: 2 × 768 × 2 = 0.003M
  ── 블록당 약 7M × 12 = 84M

out_head: 768 × 50257 = 38.6M

총합: 약 124M 파라미터
```

---

## 8. 주요 메서드 요약

| 메서드/기법 | 역할 | 핵심 포인트 |
|------------|------|------------|
| `nn.Embedding` | 정수 ID → 밀집 벡터 | 룩업 테이블, 학습 가능 |
| `register_buffer` | 마스크 등 고정값 저장 | gradient 없음, state_dict 포함 |
| `masked_fill_(-inf)` | 미래 토큰 차단 | Softmax 후 확률 0 |
| `contiguous().view()` | 차원 재배열 | transpose 후 메모리 연속화 필요 |
| `torch.no_grad()` | 추론 시 gradient 차단 | 메모리 절약 |
| `torch.argmax` | 가장 큰 값의 인덱스 | Greedy Decoding |
| `torch.cat(dim=1)` | 시퀀스 이어붙이기 | 생성 루프에서 사용 |
| `unsqueeze(0)` | 배치 차원 추가 | [seq] → [1, seq] |
| `squeeze(0)` | 배치 차원 제거 | [1, seq] → [seq] |

---

*작성일: 2026-03-25*
