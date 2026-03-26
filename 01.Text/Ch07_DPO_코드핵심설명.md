# Chapter 7 — DPO (Direct Preference Optimization) 코드 핵심 설명

---

## DPO란?

```
목표: 사람의 선호도를 반영하도록 모델을 조정
방법: "좋은 답변"의 확률은 높이고, "나쁜 답변"의 확률은 낮춤

RLHF vs DPO:
  RLHF: 보상 모델 학습 → PPO로 policy 최적화 (복잡)
  DPO:  보상 모델 없이 직접 preference 데이터로 최적화 (단순)
```

---

## 전체 흐름

```
Preference 데이터:
  { "instruction": "...", "input": "...",
    "chosen": "좋은 답변",
    "rejected": "나쁜 답변" }
        │
        ▼ PreferenceDataset
  {prompt: tokens, chosen: tokens, rejected: tokens}
        │
        ▼ custom_collate_fn (DPO용)
  padding + mask 생성
        │
        ├─── policy_model    (학습됨)
        └─── reference_model (동결, 비교 기준)
        │
        ▼ compute_dpo_loss()
  chosen 확률 ↑, rejected 확률 ↓
        │
        ▼ optimizer.step()
```

---

## 1. PreferenceDataset

```python
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        for entry in data:
            prompt   = format_input(entry)
            chosen   = entry["chosen"]    # 👍 선호 응답
            rejected = entry["rejected"]  # 👎 비선호 응답

            prompt_tokens = tokenizer.encode(prompt)

            chosen_full_tokens   = tokenizer.encode(f"{prompt}\n\n### Response:\n{chosen}")
            rejected_full_tokens = tokenizer.encode(f"{prompt}\n\n### Response:\n{rejected}")

            self.encoded_texts.append({
                "prompt":   prompt_tokens,         # 프롬프트만
                "chosen":   chosen_full_tokens,    # 프롬프트 + 선호 답변
                "rejected": rejected_full_tokens,  # 프롬프트 + 비선호 답변
            })
```

**왜 prompt를 따로 저장하는가?**

```
DPO Loss 계산 시:
  - 응답 부분의 log probability만 필요
  - 프롬프트 부분은 loss에서 제외 (마스킹)

→ prompt 길이를 알아야 어디서부터 마스킹 해제할지 결정 가능
```

---

## 2. DPO용 custom_collate_fn

```python
def custom_collate_fn(batch, pad_token_id=50256,
                      allowed_max_length=None,
                      mask_prompt_tokens=True,  # ← DPO 핵심 옵션
                      device="cpu"):
    batch_data = {
        "prompt": [],
        "chosen": [], "rejected": [],
        "chosen_mask": [], "rejected_mask": []  # Loss 계산 마스크
    }

    for item in batch:
        prompt = torch.tensor(item["prompt"])

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            # 패딩 추가 후 tensor로 변환
            padded = sequence + [pad_token_id] * (max_length - len(sequence))
            tensor = torch.tensor(padded[:allowed_max_length])

            # 마스크: True = loss 계산, False = 무시
            mask = tensor != pad_token_id  # 패딩 위치는 False
            if mask_prompt_tokens:
                # 프롬프트 길이만큼 앞부분도 False (마스킹)
                mask[:len(item["prompt"])] = False
```

**mask_prompt_tokens=True 효과**

```
시퀀스:  [프롬프트 토큰들] [응답 토큰들] [PAD PAD PAD]
마스크:  [False False ...]  [True True ...]  [False False ...]
          ↑ 프롬프트 제외     ↑ 응답만 loss 계산  ↑ 패딩 제외

이유: 프롬프트(질문)를 얼마나 잘 생성하는지가 아니라
      응답(답변)의 품질만 평가하기 위함
```

---

## 3. DPO Loss 계산 원리

```python
def compute_dpo_loss(
    model_chosen_logprobs,    # policy_model의 chosen log prob
    model_rejected_logprobs,  # policy_model의 rejected log prob
    reference_chosen_logprobs,   # reference_model의 chosen log prob
    reference_rejected_logprobs, # reference_model의 rejected log prob
    beta=0.1                  # KL divergence 페널티 강도
):
    # 상대적 선호도 계산
    chosen_rewards   = model_chosen_logprobs   - reference_chosen_logprobs
    rejected_rewards = model_rejected_logprobs - reference_rejected_logprobs

    # DPO Loss: chosen을 높이고 rejected를 낮추도록
    loss = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards)).mean()
    return loss
```

**수식 직관**

```
chosen_rewards - rejected_rewards가 클수록:
  → policy가 reference 대비 chosen을 더 선호
  → loss 감소 (좋은 방향)

beta:
  작을수록: reference_model과 크게 달라도 허용 (자유로운 학습)
  클수록:   reference_model에 가깝게 유지 (보수적 학습)
```

---

## 4. Policy Model vs Reference Model

```python
# Policy Model: 학습 대상
policy_model = GPTModel(BASE_CONFIG)
policy_model.load_state_dict(torch.load(model_path, weights_only=True))
policy_model.to(device)
# → optimizer.step()으로 가중치 업데이트됨

# Reference Model: 고정 기준점
reference_model = GPTModel(BASE_CONFIG)
reference_model.load_state_dict(torch.load(model_path, weights_only=True))
reference_model.to(device)
reference_model.eval()    # 항상 eval 모드
# → gradient 계산 없음, 가중치 업데이트 없음
```

**같은 가중치로 시작하는 이유**

```
reference_model = SFT(Supervised Fine-tuning) 완료 모델
→ 이미 지시 따르기를 배운 상태

DPO는 이 기준점에서
  - 얼마나 chosen을 선호하는지
  - 얼마나 rejected를 기피하는지
를 상대적으로 측정하여 학습
```

---

## 5. 두 모델의 학습/추론 모드

```python
# 학습 루프
policy_model.train()         # dropout 활성화
reference_model.eval()       # 항상 고정

# gradient 계산
with torch.no_grad():
    ref_chosen_log_prob = reference_model(chosen_inputs)
    # reference model: no_grad (gradient 계산 불필요)

policy_chosen_log_prob = policy_model(chosen_inputs)
# policy model: gradient 계산됨 → backward → update
```

---

## 6. 학습 설정

```python
optimizer = torch.optim.AdamW(
    policy_model.parameters(),  # policy_model만 학습
    lr=5e-6,                    # 매우 낮은 학습률 (미세 조정)
    weight_decay=0.01
)

# beta = KL 페널티 강도
# 작은 값(0.1): reference로부터 자유롭게 이탈 허용
# 큰 값(0.5+): reference에 가깝게 유지

tracking = train_model_dpo_simple(
    policy_model=policy_model,
    reference_model=reference_model,
    ...
    beta=0.1,
    eval_freq=5,
    eval_iter=5,
)
```

---

## 7. SFT → DPO 학습 순서

```
1단계: Pretraining
   대규모 텍스트 → 다음 토큰 예측 학습
   → 언어 능력 획득

2단계: SFT (Supervised Fine-tuning)
   (instruction, response) 쌍 → 지시 따르기 학습
   → 명령 수행 능력 획득
   → 이것이 reference_model의 기반

3단계: DPO
   (prompt, chosen, rejected) → 선호도 정렬
   → 더 안전하고 유용한 응답 생성
```

---

## 8. Instruction FT vs DPO 비교

| 항목 | Instruction FT (SFT) | DPO |
|------|---------------------|-----|
| 데이터 | (instruction, response) | (instruction, chosen, rejected) |
| 모델 수 | 1개 | 2개 (policy + reference) |
| Loss | Cross Entropy | DPO Loss (-logsigmoid) |
| 목표 | 올바른 응답 생성 학습 | 선호 vs 비선호 구분 학습 |
| 학습 속도 | 빠름 | 느림 (두 모델 forward) |
| 보상 모델 | 불필요 | 불필요 (RLHF 대비 장점) |

---

*작성일: 2026-03-25*
