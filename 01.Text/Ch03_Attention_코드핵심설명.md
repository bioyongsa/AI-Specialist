# Chapter 3 — Attention 메커니즘 코드 핵심 설명

---

## 전체 흐름 (CausalAttention 기준)

```
입력 x [B, num_tokens, d_in]
      │
      ├─ W_query(x) → queries [B, num_tokens, d_out]
      ├─ W_key(x)   → keys    [B, num_tokens, d_out]
      └─ W_value(x) → values  [B, num_tokens, d_out]
                │
                ▼ queries @ keys.T
      attn_scores [B, num_tokens, num_tokens]  ← 토큰 간 유사도
                │
                ▼ masked_fill_(-inf, 미래 위치)
      인과적 마스킹 (미래 토큰 차단)
                │
                ▼ softmax( / √d_out )
      attn_weights [B, num_tokens, num_tokens]  ← 확률 (합=1)
                │
                ▼ dropout
                │
                ▼ @ values
      context_vec [B, num_tokens, d_out]  ← 최종 출력
```

---

## 1. CausalAttention

### `__init__` — 핵심 구성 요소

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):

        # Q, K, V 투영 레이어 (각각 독립적인 학습 파라미터)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # → 입력 벡터(d_in)를 각각 다른 공간(d_out)으로 변환

        self.dropout = nn.Dropout(dropout)

        # 인과적 마스크: 상삼각행렬 (대각선 위쪽 = 미래 위치)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        # mask:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]  ← 1인 위치 → -inf로 마스킹
```

**`register_buffer` vs `nn.Parameter` 차이**

| 구분 | register_buffer | nn.Parameter |
|------|-----------------|-------------|
| 학습 여부 | 학습 안 됨 | 학습됨 (gradient 계산) |
| state_dict 저장 | 저장됨 | 저장됨 |
| 용도 | mask, running stats 등 고정값 | 가중치 |

---

### `forward` — 단계별 연산

```python
def forward(self, x):   # x: [B, num_tokens, d_in]
    b, num_tokens, d_in = x.shape

    # Step 1: Q, K, V 계산
    queries = self.W_query(x)  # [B, num_tokens, d_out]
    keys    = self.W_key(x)    # [B, num_tokens, d_out]
    values  = self.W_value(x)  # [B, num_tokens, d_out]

    # Step 2: 어텐션 점수 계산 (Query · Key^T)
    attn_scores = queries @ keys.transpose(1, 2)
    # [B, num_tokens, d_out] @ [B, d_out, num_tokens]
    # = [B, num_tokens, num_tokens]
    # → (i, j) 위치 = i번 토큰이 j번 토큰에 얼마나 주목하는지

    # Step 3: 인과적 마스킹 (미래 토큰 차단)
    attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens],
        -torch.inf
    )
    # 미래 위치(-inf) → softmax 후 확률 0이 됨

    # Step 4: 스케일링 + Softmax
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5,  # √d_out으로 나눔
        dim=-1
    )
    # keys.shape[-1]**0.5 = √d_out
    # → 차원이 커질수록 내적값이 커져 기울기 소실 발생 → 스케일링으로 방지

    # Step 5: Dropout
    attn_weights = self.dropout(attn_weights)

    # Step 6: Value 가중합 → Context Vector
    context_vec = attn_weights @ values
    # [B, num_tokens, num_tokens] @ [B, num_tokens, d_out]
    # = [B, num_tokens, d_out]

    return context_vec
```

**Scaled Dot-Product 스케일링 이유**

```
d_out이 크면 → 내적값이 커짐 → Softmax 후 특정 위치에 확률 집중
→ 기울기 소실 (vanishing gradient)
→ √d_out으로 나눠서 값 범위 안정화
```

---

## 2. MultiHeadAttentionWrapper (단순 버전)

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # 각 헤드 출력: [B, num_tokens, d_out]
        # cat(dim=-1): [B, num_tokens, d_out * num_heads]
```

**단점**: 헤드 수만큼 별도 연산 → 비효율적 → MultiHeadAttention으로 개선

---

## 3. MultiHeadAttention (효율적 버전)

### 핵심 아이디어: view + transpose로 헤드 분할

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0  # d_out이 num_heads로 나누어 떨어져야 함

        self.head_dim = d_out // num_heads  # 각 헤드의 차원
        # 예: d_out=768, num_heads=12 → head_dim=64

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 헤드 합친 후 최종 투영
```

### forward — 차원 변환 과정

```python
def forward(self, x):   # x: [B, num_tokens, d_in]
    b, num_tokens, d_in = x.shape

    # Step 1: Q, K, V 계산
    keys    = self.W_key(x)    # [B, num_tokens, d_out]
    queries = self.W_query(x)
    values  = self.W_value(x)

    # Step 2: 헤드별로 분할 (view)
    # [B, num_tokens, d_out] → [B, num_tokens, num_heads, head_dim]
    keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

    # Step 3: 차원 순서 변경 (transpose)
    # [B, num_tokens, num_heads, head_dim] → [B, num_heads, num_tokens, head_dim]
    keys    = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values  = values.transpose(1, 2)
    # → num_heads개의 [num_tokens, head_dim] 행렬이 독립적으로 존재

    # Step 4: 어텐션 점수
    attn_scores = queries @ keys.transpose(2, 3)
    # [B, num_heads, num_tokens, head_dim] @ [B, num_heads, head_dim, num_tokens]
    # = [B, num_heads, num_tokens, num_tokens]

    # Step 5: 마스킹 + Softmax + Dropout
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # Step 6: Value 가중합
    context_vec = (attn_weights @ values).transpose(1, 2)
    # [B, num_heads, num_tokens, head_dim] → [B, num_tokens, num_heads, head_dim]

    # Step 7: 헤드 합치기
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    # [B, num_tokens, num_heads * head_dim] = [B, num_tokens, d_out]

    # Step 8: 최종 선형 투영
    context_vec = self.out_proj(context_vec)
    # [B, num_tokens, d_out]

    return context_vec
```

### 차원 변화 요약

```
입력:           [B, num_tokens, d_in]
W_key(x):       [B, num_tokens, d_out]        d_out = num_heads × head_dim
view:           [B, num_tokens, num_heads, head_dim]
transpose(1,2): [B, num_heads, num_tokens, head_dim]  ← 헤드별 독립 처리
@ keys.T:       [B, num_heads, num_tokens, num_tokens] ← 어텐션 스코어
@ values:       [B, num_heads, num_tokens, head_dim]
transpose(1,2): [B, num_tokens, num_heads, head_dim]
.view:          [B, num_tokens, d_out]
out_proj:       [B, num_tokens, d_out]                ← 최종 출력
```

---

## 4. Wrapper vs 효율 버전 비교

| 항목 | MultiHeadAttentionWrapper | MultiHeadAttention |
|------|--------------------------|-------------------|
| 구현 방식 | 헤드 수만큼 CausalAttention 반복 | view+transpose로 한번에 처리 |
| Q,K,V 행렬 | 헤드당 별도 (num_heads 세트) | 전체 합쳐서 1세트 (내부에서 분할) |
| 연산 효율 | 낮음 (반복 for-loop) | 높음 (병렬 행렬 연산) |
| out_proj | 없음 | 있음 (헤드 결합 후 투영) |

---

## 5. `contiguous().view()` — 왜 함께 쓰는가?

```python
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
```

```
transpose() 후 메모리가 불연속(non-contiguous)이 됨
view()는 연속 메모리를 요구함
→ contiguous()로 메모리를 연속 배치로 복사 후 view() 적용
```

---

## 6. `out_proj` (출력 선형 투영)의 역할

```python
self.out_proj = nn.Linear(d_out, d_out)
```

```
역할: 헤드들의 정보를 합친 후 추가로 학습 가능한 선형 변환 적용
효과: 헤드 간 정보를 최종적으로 믹싱 (cross-head interaction)
선택적이지만 GPT-2에서는 사용
```

---

*작성일: 2026-03-25*
