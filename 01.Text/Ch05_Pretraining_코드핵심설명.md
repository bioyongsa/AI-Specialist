# Chapter 5 — Pretraining (사전학습) 코드 핵심 설명

---

## 전체 학습 흐름

```
텍스트 파일
      │
      ▼ create_dataloader_v1()
DataLoader [B, seq_len] 배치
      │
      for epoch in range(num_epochs):
      │    model.train()
      │    for input_batch, target_batch in train_loader:
      │         │
      │         ├─ optimizer.zero_grad()        gradient 초기화
      │         ├─ calc_loss_batch()            forward + loss
      │         ├─ loss.backward()              backpropagation
      │         └─ optimizer.step()             weight 업데이트
      │
      ▼ evaluate_model() 주기적 평가
      │
      ▼ generate_and_print_sample() 생성 확인
      │
      ▼ torch.save(model.state_dict())  모델 저장
```

---

## 1. calc_loss_batch — 배치 손실 계산

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)   # CPU → GPU 이동
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    # logits shape: [B, seq_len, vocab_size]  예: [2, 256, 50257]

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),   # [B×seq_len, vocab_size]  예: [512, 50257]
        target_batch.flatten()  # [B×seq_len]              예: [512]
    )
    return loss
```

**`flatten(0, 1)` 이유**

```
CrossEntropyLoss 입력 형식:
  - 예측: [N, C]  N=샘플 수, C=클래스 수
  - 정답: [N]

3D 텐서 → 2D로 변환 필요:
  logits:       [2, 256, 50257]
  .flatten(0,1) [512, 50257]    ← dim 0,1을 합침
  
  target_batch: [2, 256]
  .flatten()    [512]
```

**Loss와 Perplexity 관계**

```
Loss (Cross Entropy): -log(p_correct)
  → 낮을수록 좋음 (모델이 정답을 높은 확률로 예측)

Perplexity = e^Loss
  → 모델이 다음 단어 예측 시 "몇 가지 단어 중에 고민하는지"
  → 낮을수록 좋음
```

---

## 2. train_model_simple — 학습 루프

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    for epoch in range(num_epochs):
        model.train()  # Dropout 활성화, BN 학습 통계 사용

        for input_batch, target_batch in train_loader:

            # ① gradient 초기화 (필수! 누적 방지)
            optimizer.zero_grad()

            # ② forward + loss 계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # ③ backward (각 파라미터의 gradient 계산)
            loss.backward()

            # ④ weight 업데이트
            optimizer.step()

            tokens_seen += input_batch.numel()
            # numel() = 텐서의 전체 원소 수 = B × seq_len

            # eval_freq 스텝마다 평가
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(...)

        # 에포크마다 샘플 생성 확인
        generate_and_print_sample(model, tokenizer, device, start_context)
```

**`optimizer.zero_grad()` 왜 매번 호출하는가?**

```
PyTorch는 기본적으로 gradient를 누적함
→ 매 배치마다 초기화하지 않으면 이전 배치의 gradient가 더해짐
→ 잘못된 방향으로 업데이트됨

zero_grad() 호출 위치: loss.backward() 이전 반드시 호출
```

---

## 3. evaluate_model — 평가 함수

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # Dropout 비활성화, BN 고정 통계 사용

    with torch.no_grad():  # gradient 계산 끔 → 메모리 절약
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)

    model.train()  # 평가 후 반드시 train 모드 복귀!
    return train_loss, val_loss
```

**model.eval() 후 model.train() 복귀 이유**

```
eval() 상태 그대로 학습 루프 돌아가면:
  - Dropout이 꺼진 채 학습 → 과적합 위험
  - BatchNorm이 running stats 사용 → 학습 통계 업데이트 안 됨
→ 평가 후 반드시 train() 모드로 복귀
```

---

## 4. generate — 고급 텍스트 생성

```python
def generate(model, idx, max_new_tokens, context_size, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # 최대 길이 자르기

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]           # 마지막 위치만 사용

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy

        if idx_next == eos_id:  # EOS 토큰이면 중단
            break

        idx = torch.cat((idx, idx_next), dim=1)  # 이어붙이기
    return idx
```

---

## 5. text_to_token_ids / token_ids_to_text

```python
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # "Hello" → [15496] → [[15496]]
    # shape: [seq_len] → [1, seq_len]  (배치 차원 추가)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    # [[15496, 11, ...]] → [15496, 11, ...]  (배치 차원 제거)
    return tokenizer.decode(flat.tolist())
```

```
unsqueeze(0): [seq] → [1, seq]  ← 배치 차원 추가 (모델 입력에 맞춤)
squeeze(0):   [1, seq] → [seq]  ← 배치 차원 제거 (디코딩에 맞춤)
```

---

## 6. 모델 저장 / 로드

```python
# 저장
torch.save(model.state_dict(), "outputs/model.pth")

# 로드 (3단계 필수)
model = GPTModel(GPT_CONFIG_124M)                              # 1. 빈 모델 생성
model.load_state_dict(torch.load("outputs/model.pth", weights_only=True))  # 2. 가중치 주입
model.to(device)                                               # 3. GPU로 이동
```

**왜 3단계인가?**

```
torch.load()만으로는 모델 구조를 알 수 없음
→ 먼저 GPTModel 인스턴스를 만들어 구조를 정의한 후
→ load_state_dict()로 가중치를 채워 넣어야 함
→ to(device): 가중치 로드는 CPU에서 → GPU로 이동 필요
```

---

## 7. 학습 설정 파라미터

```python
GPT_CONFIG_124M = {
    "context_length": 256,  # 원본 1024보다 작게 설정 (학습 속도 개선)
    "drop_rate": 0.1,       # 10% Dropout
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,  # AdamW 기본 lr
    "num_epochs": 10,
    "batch_size": 2,        # 소규모 텍스트이므로 작게
    "weight_decay": 0.1     # L2 정규화
}
```

**AdamW vs SGD**

| 항목 | SGD | AdamW |
|------|-----|-------|
| 학습률 | 고정 | 파라미터별 자동 적응 |
| momentum | 수동 설정 | 내부에서 처리 (β1=0.9) |
| weight_decay | L2 패널티 직접 | gradient와 분리 처리 (더 정확) |
| LLM 적합성 | 보통 | 표준 (GPT, BERT 모두 사용) |

---

## 8. plot_losses — 이중 X축 시각화

```python
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses,   label="Validation loss")
    ax1.set_xlabel("Epochs")

    # 보조 X축: 처리한 토큰 수 기준
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 투명 (축만 생성)
    ax2.set_xlabel("Tokens seen")
```

```
twiny(): 상단에 보조 X축 추가 (Y축 공유)
alpha=0: 그래프는 투명하게 → 축 눈금만 표시
→ Epoch 기준과 토큰 수 기준을 동시에 확인 가능
```

---

*작성일: 2026-03-25*
