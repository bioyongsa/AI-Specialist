# Chapter 2 — Dataset & Tokenizer 코드 핵심 설명

---

## 전체 흐름

```
원본 텍스트 (str)
      │
      ▼ tokenizer.encode()
토큰 ID 리스트 [int, int, ...]
      │
      ▼ GPTDatasetV1 (슬라이딩 윈도우)
input_ids / target_ids 쌍 생성
      │
      ▼ DataLoader
배치 텐서 [B, seq_len]
      │
      ▼ nn.Embedding
토큰 임베딩 [B, seq_len, emb_dim]  +  위치 임베딩 [B, seq_len, emb_dim]
      │
      ▼
입력 임베딩 [B, seq_len, emb_dim]  → GPT 모델 입력
```

---

## 1. Tokenizer — tiktoken

```python
tokenizer = tt.get_encoding("gpt2")

# 인코딩: 문자열 → 토큰 ID 리스트
tokenids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# → [15496, 11, 466, 345, 588, ...]

# 디코딩: 토큰 ID 리스트 → 문자열
strings = tokenizer.decode(tokenids)
# → "Hello, do you like tea? ..."
```

**`allowed_special={"<|endoftext|>"}` 이유**

```
<|endoftext|>는 문서 경계를 표시하는 특수 토큰
기본적으로 이 토큰은 인코딩 불가능 → 허용 목록에 명시해야 함
없으면 에러 발생
```

---

## 2. Input / Target 1-shift 관계

GPT는 **"다음 토큰 예측"** 과제로 학습 → target은 input에서 1칸 뒤로 밀린 것

```python
context_size = 4
x = enc_sample[:context_size]       # [t0, t1, t2, t3]
y = enc_sample[1:context_size + 1]  # [t1, t2, t3, t4]
#                  ↑
#               1 shift!
```

```
입력 x: [t0, t1, t2, t3]
정답 y: [t1, t2, t3, t4]

→ 모델은 t0를 보고 t1을 예측
          t0,t1을 보고 t2를 예측
          t0,t1,t2를 보고 t3를 예측  ...
```

---

## 3. GPTDatasetV1 — 슬라이딩 윈도우 방식

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i     : i + max_length]      # [seq_len]
            target_chunk = token_ids[i + 1 : i + max_length + 1]  # [seq_len] (+1 shift)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

**stride 파라미터 의미**

```
max_length=4, stride=1:         max_length=4, stride=4:
[t0,t1,t2,t3]                  [t0,t1,t2,t3]
  [t1,t2,t3,t4]   ← 1칸 이동    [t4,t5,t6,t7]  ← 겹침 없음
    [t2,t3,t4,t5]
→ 데이터 풍부, 많은 중복        → 빠른 처리, 중복 없음
```

**`__len__` / `__getitem__` 필수 구현 이유**

```
DataLoader가 Dataset을 사용하려면 두 메서드가 반드시 있어야 함
__len__      : 전체 샘플 수 (DataLoader가 반복 횟수 계산에 사용)
__getitem__  : idx 번째 샘플 반환 (DataLoader가 배치 구성 시 호출)
```

---

## 4. create_dataloader_v1 — DataLoader 생성

```python
dataloader = DataLoader(
    dataset,            # GPTDatasetV1 인스턴스
    batch_size=batch_size,  # 한 번에 처리할 샘플 수
    shuffle=shuffle,        # True: 매 epoch 순서 섞기 (train에 필요)
    drop_last=drop_last,    # True: 마지막 불완전 배치 버리기
    num_workers=num_workers # 데이터 로딩 병렬 프로세스 수
)
```

**출력 텐서 shape**

```
batch_size=1, max_length=4:
  inputs:  [1, 4]  → [batch, seq_len]
  targets: [1, 4]  → [batch, seq_len]

batch_size=8, max_length=256:
  inputs:  [8, 256]
  targets: [8, 256]
```

---

## 5. nn.Embedding — 토큰 임베딩

```python
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#                                     ↑            ↑
#                                  6(단어장 크기)  3(벡터 차원)

# 내부 구조: [vocab_size, output_dim] 크기의 룩업 테이블(가중치 행렬)
# 입력: 정수 토큰 ID → 해당 행(row)을 그대로 꺼내옴 (행렬 곱 없음)

token_embeddings = embedding_layer(input_ids)
# input_ids: [4]         → token_embeddings: [4, 3]
# input_ids: [B, seq]    → token_embeddings: [B, seq, emb_dim]
```

---

## 6. Positional Embedding — 위치 임베딩

```python
context_length = max_length  # 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#                                         ↑
#                                   위치 수 (0~context_length-1)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# torch.arange(4) = [0, 1, 2, 3]  ← 위치 인덱스
# pos_embeddings shape: [4, emb_dim]

# 최종 입력 임베딩 = 토큰 임베딩 + 위치 임베딩
input_embeddings = token_embeddings + pos_embeddings
# [B, seq_len, emb_dim] + [seq_len, emb_dim]  → 브로드캐스트
# = [B, seq_len, emb_dim]
```

**왜 위치 임베딩이 필요한가?**

```
Attention은 순서 정보가 없음
"I love you" vs "you love I" → Attention만으로는 동일하게 처리됨
→ 위치 임베딩으로 각 토큰의 '몇 번째 위치인지' 정보를 추가
```

---

## 7. 주요 파라미터 정리

| 파라미터 | 설명 | GPT-2 Small 기준 |
|---------|------|-----------------|
| `vocab_size` | 단어장 크기 (토큰 임베딩 행 수) | 50,257 |
| `context_length` | 최대 처리 토큰 수 (위치 임베딩 행 수) | 1,024 |
| `emb_dim` | 임베딩 벡터 차원 | 768 |
| `max_length` | 데이터셋 윈도우 크기 | 모델의 context_length 이하 |
| `stride` | 윈도우 이동 간격 | max_length와 같으면 중복 없음 |
| `batch_size` | DataLoader 배치 크기 | 실험에 따라 다름 |

---

## 8. Hugging Face 토크나이저 비교 (`AutoTokenizer`)

```python
from transformers import AutoTokenizer
qwen   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
exaone = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B")

tokens = qwen.encode(text, add_special_tokens=True)
```

**모델 파일 구성 요약**

| 파일 | 역할 |
|------|------|
| `config.json` | 모델 구조 설계도 (레이어 수, 헤드 수 등) |
| `model.safetensors` | 학습된 가중치 (가장 큰 파일) |
| `tokenizer.json` | 토크나이저 통합 설정 |
| `vocab.json` | 단어 ↔ ID 매핑 테이블 |
| `merges.txt` | BPE 병합 규칙 |
| `generation_config.json` | 생성 옵션 기본값 |

---

*작성일: 2026-03-25*
