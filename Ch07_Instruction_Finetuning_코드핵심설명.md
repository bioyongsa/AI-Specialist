# Chapter 7 — Instruction Fine-tuning 코드 핵심 설명

---

## 전체 흐름

```
Instruction 데이터 (JSON)
  { "instruction": "...", "input": "...", "output": "..." }
        │
        ▼ format_input()
  "Below is an instruction...\n\n### Instruction:\n...\n\n### Input:\n..."
        │
        ▼ + Response 붙이기
  전체 텍스트 = 프롬프트 + "\n\n### Response:\n" + output
        │
        ▼ tokenizer.encode()
  토큰 ID 리스트
        │
        ▼ custom_collate_fn()
  inputs:  [padded[:-1]]  ← 마지막 토큰 제외
  targets: [padded[1:]]   ← 첫 토큰 제외 (1-shift)
  (패딩 부분은 -100으로 마스킹)
        │
        ▼ train_model_simple()  학습
        │
        ▼ generate()            추론
```

---

## 1. 데이터 포맷 — Alpaca 스타일

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

**실제 포맷 예시**

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Translate the following sentence to French.

### Input:
Hello, how are you?

### Response:
Bonjour, comment allez-vous?
```

**학습 시 전체 텍스트(질문+답변)를 입력으로 사용하는 이유**

```
GPT 방식: "다음 토큰 예측" 방식으로 학습
→ 모델이 프롬프트를 읽고 응답을 생성하는 방법을 학습
→ 프롬프트 부분: context 제공
→ 응답 부분: 실제 예측 대상 (loss 계산)
```

---

## 2. InstructionDataset

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 가변 길이 리스트 반환
```

**SpamDataset과 차이점**

```
SpamDataset:         패딩을 Dataset 내부에서 처리
InstructionDataset:  패딩을 custom_collate_fn에서 처리

이유: 배치 내 최대 길이에 맞춰 동적 패딩
→ 불필요한 패딩 최소화 → 메모리 효율
```

---

## 3. custom_collate_fn — 배치 패딩 및 마스킹

```python
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,       # CrossEntropyLoss가 무시하는 특수 값
    allowed_max_length=None,
    device="cpu"
):
    # 1. 배치 내 최대 길이 계산 (+1: EOS 토큰 공간)
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # EOS 토큰 추가

        # 최대 길이에 맞춰 패딩
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Input/Target 생성 (1-shift)
        inputs  = torch.tensor(padded[:-1])  # [0 ~ n-1]
        targets = torch.tensor(padded[1:])   # [1 ~ n]

        # 패딩 부분 마스킹: -100으로 설정
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 첫 PAD는 유지, 이후 -100
        # endregion

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

**ignore_index = -100 이유**

```python
# PyTorch CrossEntropyLoss는 target=-100인 위치를 loss 계산에서 제외
loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(), ignore_index=-100)

왜 필요한가?
  패딩 토큰에 대해 loss를 계산하면:
  → 모델이 패딩을 예측하도록 학습됨 (의미 없는 학습)
  → -100으로 마스킹하여 패딩 위치는 loss 기여 없음
```

**targets[indices[1:]] = ignore_index — 첫 PAD는 왜 유지하는가?**

```
시퀀스:    [...실제 토큰..., EOS, PAD, PAD, PAD]
targets:   [...실제 토큰..., EOS, PAD, PAD, PAD]
                                    ↑
                              첫 PAD 유지 → EOS 다음에 PAD가 오는 것을 학습
                                     이후 → -100으로 마스킹

이유: 모델이 "EOS 토큰 뒤에는 종료"를 학습하도록
```

---

## 4. Input/Target 1-shift 상세

```
원본 토큰:  [I, love, Paris, EOS, PAD, PAD]
padded:     [I, love, Paris, EOS, PAD, PAD, PAD]  ← +1 길이

inputs  = padded[:-1]: [I, love, Paris, EOS, PAD, PAD]
targets = padded[1:]:  [love, Paris, EOS, PAD, -100, -100]
                                              ↑↑
                                        마스킹됨

학습:
  I → love 예측
  love → Paris 예측
  Paris → EOS 예측
  EOS → PAD 예측 (첫 PAD, 유지)
  PAD → -100 (loss 무시)
```

---

## 5. partial을 사용한 DataLoader 설정

```python
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=customized_collate_fn,  # 커스텀 배치 처리 함수
    shuffle=True,
    drop_last=True
)
```

**`partial` 이란?**

```python
# partial은 함수의 일부 인자를 미리 고정한 새 함수를 반환
partial(custom_collate_fn, device=device)
# = device=device가 이미 설정된 custom_collate_fn 함수

# DataLoader의 collate_fn은 (batch,)만 받아야 함
# → partial로 나머지 인자를 미리 고정
```

**`collate_fn` 역할**

```
기본 DataLoader: 같은 크기 텐서를 torch.stack으로 합침
→ 가변 길이 시퀀스는 처리 불가

custom_collate_fn:
→ 배치 내 가변 길이를 패딩으로 통일
→ 마스킹 처리
→ device 이동까지 한번에 처리
```

---

## 6. Instruction Fine-tuning vs 분류 Fine-tuning 비교

| 항목 | 분류 Fine-tuning (Ch6) | Instruction Fine-tuning (Ch7) |
|------|----------------------|------------------------------|
| 출력 | 클래스 레이블 (2개) | 자연어 텍스트 |
| out_head | Linear(768→2) | 원본 유지 (768→50257) |
| 학습 데이터 | (텍스트, 레이블) 쌍 | (명령, 응답) 쌍 |
| Loss 계산 | 마지막 토큰만 사용 | 응답 전체 토큰에서 계산 |
| 패딩 처리 | Dataset 내부 | collate_fn에서 동적 처리 |
| 추론 방식 | argmax → 클래스 인덱스 | generate() → 토큰 생성 |

---

## 7. generate() 함수 사용 시 중요 파라미터

```python
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=256,       # 최대 생성 토큰 수
    context_size=1024,        # 모델의 최대 컨텍스트 길이
    eos_id=50256              # EOS 토큰 → 이 토큰 생성 시 중단
)
```

**eos_id=50256 역할**

```
50256 = <|endoftext|> 토큰 ID
→ 모델이 이 토큰을 생성하면 "응답 완료" 신호
→ max_new_tokens에 도달하지 않아도 조기 종료 가능
→ 불필요한 반복 생성 방지
```

---

*작성일: 2026-03-25*
