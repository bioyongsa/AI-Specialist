# Chapter 6 — Fine-tuning (분류) 코드 핵심 설명

---

## 전체 흐름

```
SMS 스팸 데이터 (ham/spam)
      │
      ▼ create_balanced_dataset()  50:50 균형
      │
      ▼ random_split()             Train 70% / Val 10% / Test 20%
      │
      ▼ SpamDataset                토큰화 + 패딩 → [seq_len] 텐서
      │
      ▼ DataLoader                 배치 생성
      │
      ▼ GPT-2 Small 로드 (사전학습 가중치)
      │
      ▼ out_head 교체              Linear(768→1000) → Linear(768→2)
      │
      ▼ 일부 레이어 동결 (마지막 블록만 학습)
      │
      ▼ train_classifier_simple()  학습
      │
      ▼ calc_accuracy_loader()     평가
```

---

## 1. 데이터 준비 유틸리티

### create_balanced_dataset — 클래스 균형

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    # spam 개수에 맞춰 ham을 언더샘플링
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df
```

**왜 균형을 맞추는가?**

```
실제 데이터: ham 4825개 vs spam 747개 (약 6:1 불균형)
불균형 상태로 학습하면:
  → 모델이 항상 "ham"이라 예측해도 87% 정확도
  → 실제로 스팸을 잘 잡는 모델이 아님
→ 1:1 균형으로 맞춰 각 클래스 동등 학습
```

---

### random_split — 데이터 분할

```python
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # 셔플
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df      = df[:train_end]           # 70%
    validation_df = df[train_end:validation_end]  # 10%
    test_df       = df[validation_end:]      # 나머지 20%
    return train_df, validation_df, test_df
```

---

### SpamDataset — 토큰화 + 패딩

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 1. 토큰화
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # 2. max_length 설정
        if max_length is None:
            self.max_length = self._longest_encoded_length()  # 가장 긴 문장 기준
        else:
            self.max_length = max_length
            # 길면 잘라냄 (Truncation)
            self.encoded_texts = [
                encoded_text[:max_length]
                for encoded_text in self.encoded_texts
            ]

        # 3. 패딩: 짧으면 pad_token_id로 채움
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
```

**pad_token_id = 50256 이유**

```
GPT-2의 <|endoftext|> 토큰 ID = 50256
→ 문서 끝 토큰을 패딩으로 재사용
→ 모델이 이 토큰 위치에서 분류 헤드 출력을 사용
```

**val/test dataset에서 train의 max_length를 사용하는 이유**

```python
val_dataset = SpamDataset(
    max_length=train_dataset.max_length,  # ← train 기준 길이 사용
    ...
)
```

```
일관성: 모든 데이터셋이 동일한 시퀀스 길이 → 모델 입력 크기 일치
정보 누수 방지: val/test의 max_length를 따로 계산하면
               val/test 데이터 분포 정보가 영향을 줄 수 있음
```

---

## 2. 사전학습 모델 로드 및 수정

```python
# 1. GPT-2 모델 설정
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,     # Fine-tuning 시 dropout 끔 (이미 사전학습됨)
    "qkv_bias": True      # GPT-2 원본 가중치에 bias 있음
}
BASE_CONFIG.update({"emb_dim": 768, "n_layers": 12, "n_heads": 12})

# 2. 모델 초기화 + 가중치 로드
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2-small-124M.pth", weights_only=True))

# 3. 출력 헤드 교체 (핵심!)
model.out_head = nn.Linear(
    in_features=768,    # emb_dim 고정
    out_features=2      # 분류 클래스 수 (ham=0, spam=1)
)
# 원본: Linear(768 → 50257) → 교체: Linear(768 → 2)
```

**왜 out_head만 교체하는가?**

```
GPT-2의 기존 out_head: 다음 토큰 예측용 (50257 클래스)
분류 태스크:           ham vs spam (2 클래스)
→ 출력 크기만 바꾸면 됨
→ 나머지 12개 블록의 언어 이해 능력은 그대로 활용
```

---

## 3. 레이어 동결 전략

```python
# 방법 1: 전체 동결 후 마지막 블록 + head만 학습
for param in model.parameters():
    param.requires_grad = False  # 전체 동결

# 마지막 Transformer 블록만 학습 허용
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

# LayerNorm + out_head 학습 허용
for param in model.final_norm.parameters():
    param.requires_grad = True

model.out_head.weight.requires_grad = True
```

**동결하는 이유**

```
사전학습 가중치는 이미 좋은 언어 표현을 가짐
전체 학습 시:
  - 과적합 위험 (소규모 데이터)
  - 오랜 학습 시간
  - Catastrophic Forgetting (기존 지식 손상)

마지막 블록만 학습:
  - 빠름, 메모리 효율
  - 태스크 특화 조정 가능
  - 사전학습 지식 유지
```

---

## 4. 분류 예측 방식

```
입력 시퀀스:  [t0, t1, t2, ..., t_n, PAD, PAD]
                                    ↑
                         마지막 실제 토큰 (또는 마지막 PAD)

GPT 출력:    [logit_0, logit_1, ..., logit_n, ...]
                                       ↑
                              마지막 위치의 logit 사용

→ out_head(last_token_embedding)
→ [B, 2]  ← ham/spam 각각의 점수
→ argmax → 0(ham) or 1(spam) 예측
```

**왜 마지막 토큰을 사용하는가?**

```
GPT는 왼쪽→오른쪽 방향으로 정보가 누적됨
→ 마지막 위치의 hidden state = 전체 시퀀스 정보를 요약한 벡터
→ 이를 분류 헤드에 입력하면 문장 전체를 기반으로 분류 가능
```

---

## 5. DataLoader 파라미터 비교

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,      # 학습: 매 epoch 순서 섞기
    drop_last=True,    # 마지막 불완전 배치 버리기 (안정적 학습)
    num_workers=0,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False,     # 평가: 순서 고정 (재현성)
    drop_last=False,   # 평가: 모든 샘플 사용
    num_workers=0,
)
```

---

## 6. GPT-2 모델 크기 옵션

| 모델 | 파라미터 | emb_dim | n_layers | n_heads |
|------|---------|---------|---------|---------|
| gpt2-small | 124M | 768 | 12 | 12 |
| gpt2-medium | 355M | 1024 | 24 | 16 |
| gpt2-large | 774M | 1280 | 36 | 20 |
| gpt2-xl | 1558M | 1600 | 48 | 25 |

---

## 7. Fine-tuning vs Pretraining 차이

| 항목 | Pretraining | Fine-tuning (분류) |
|------|------------|-------------------|
| 목적 | 다음 토큰 예측 | 스팸/햄 분류 |
| 데이터 | 수억 토큰 텍스트 | 수천 개 레이블 데이터 |
| Loss | Cross Entropy (vocab_size) | Cross Entropy (2 클래스) |
| 학습 파라미터 | 전체 | 일부 (last block + head) |
| Dropout | 있음 (0.1) | 없음 (0.0) |
| 출력 | logits [B, seq, vocab] | logits [B, 2] |

---

*작성일: 2026-03-25*
