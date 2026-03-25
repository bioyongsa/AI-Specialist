# Chapter 6 — LoRA (Low-Rank Adaptation) 코드 핵심 설명

---

## LoRA란?

```
기존 Fine-tuning: 전체 파라미터 업데이트 → 비용 큼
LoRA:             원본 가중치는 동결 + 작은 저차원 행렬만 학습

원본 Linear:  W  (d_in × d_out)  ← 동결
LoRA 추가:   ΔW = A × B
                  A: (d_in × rank)   rank << d_in
                  B: (rank × d_out)  rank << d_out

최종 출력 = W·x + (α/rank) × (A·B)·x
                  ↑
           LoRA 어댑터 (학습되는 부분)
```

**왜 효율적인가?**

```
GPT-2 Small의 W_query shape: [768, 768]
파라미터 수: 768 × 768 = 589,824

LoRA rank=16 적용 시:
  A: 768 × 16  = 12,288
  B: 16  × 768 = 12,288
  합계:          24,576  ← 원본의 4.2%만 학습!

전체 모델에서 학습 가능 파라미터 비율: 약 1~5%
```

---

## 1. LoRALayer

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

        # A 행렬: (in_dim, rank) — kaiming 초기화
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # kaiming_uniform_: 입력 크기에 따라 적절한 범위로 초기화
        # → gradient 소실/폭발 방지

        # B 행렬: (rank, out_dim) — 0으로 초기화
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # 0으로 초기화하는 이유:
        # 학습 시작 시 LoRA 기여분 = 0 → 원본 모델과 동일하게 시작
        # → 안정적인 학습 시작

        self.alpha = alpha
        self.rank  = rank

    def forward(self, x):
        # x @ A: [B, seq, in_dim] @ [in_dim, rank]  = [B, seq, rank]
        # @ B:   [B, seq, rank]   @ [rank, out_dim] = [B, seq, out_dim]
        return (self.alpha / self.rank) * (x @ self.A @ self.B)
```

**alpha/rank 스케일링 의미**

```
alpha: LoRA의 영향력 크기를 조절하는 스케일 파라미터
rank:  저차원 크기

alpha/rank = 스케일 팩터
  - alpha=rank일 때: 스케일=1 → lr과 동일하게 동작
  - alpha > rank:    LoRA 영향력 강화
  - alpha < rank:    LoRA 영향력 감소

관례: alpha = rank로 설정 (스케일=1)
→ 학습률만으로 LoRA 강도 조절 가능
```

---

## 2. LinearWithLoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 원본 Linear (동결됨)
        self.lora   = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank, alpha
        )

    def forward(self, x):
        # 원본 출력 + LoRA 보정값
        return self.linear(x) + self.lora(x)
        #       ↑ 고정           ↑ 학습됨
```

**구조 도식**

```
입력 x
  │
  ├──────────────────────┐
  │  원본 Linear(동결)   │  LoRALayer
  │  W·x                │  (α/r) × x@A@B
  │                      │
  └──────────┬───────────┘
             ↓
        W·x + (α/r)·x@A@B
             ↓
          출력
```

---

## 3. replace_linear_with_lora

```python
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Linear 레이어를 LinearWithLoRA로 교체
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 자식 모듈도 재귀적으로 탐색
            replace_linear_with_lora(module, rank, alpha)
```

**`named_children()` vs `named_modules()` 차이**

```
named_children(): 직접 자식만 반환 (깊이 1)
named_modules():  모든 하위 모듈 반환 (모든 깊이)

→ 재귀 호출을 직접 구현하기 때문에 named_children() 사용
→ named_modules() 쓰면 이미 교체된 모듈을 또 교체하는 문제 발생
```

**`setattr(model, name, ...)` 역할**

```python
setattr(model, 'W_query', LinearWithLoRA(W_query, rank, alpha))
# = model.W_query = LinearWithLoRA(W_query, rank, alpha)

# getattr(model, 'W_query') → 현재 W_query 객체 반환
# setattr(model, 'W_query', new_obj) → model.W_query를 new_obj로 교체
```

---

## 4. LoRA 적용 순서 (main 함수)

```python
# 1단계: 전체 파라미터 동결
for param in model.parameters():
    param.requires_grad = False
# → 기존 GPT-2 가중치 모두 고정

# 2단계: Linear 레이어를 LoRA로 교체
replace_linear_with_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
# → 새로 생성된 LoRALayer의 A, B는 requires_grad=True (기본값)

# 3단계: 학습 가능 파라미터 확인
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능 파라미터: {total_params:,}")
# → LoRA 파라미터만 출력됨
```

**파라미터 동결/해제 흐름**

```
before replace:
  GPT-2 모든 파라미터 → requires_grad = False

after replace_linear_with_lora:
  GPT-2 원본 W_query 등 → requires_grad = False (그대로)
  새 LoRALayer A, B     → requires_grad = True  (새로 생성)

결과: LoRA 파라미터만 학습!
```

---

## 5. 하이퍼파라미터

```python
LORA_RANK  = 16   # 저차원 병목 크기 (보통 4, 8, 16, 32 중 선택)
LORA_ALPHA = 16   # 스케일 팩터 (보통 rank와 동일하게 설정)
LEARNING_RATE = 8e-4  # LoRA만 학습 → 일반 fine-tuning보다 높은 lr 가능
```

**rank 선택 기준**

```
rank=1~4:   극도로 적은 파라미터, 단순 태스크에 적합
rank=8~16:  일반적으로 좋은 성능, 균형점
rank=32~64: 많은 파라미터, 복잡한 태스크

→ LORA_RANK=16: 균형 잡힌 기본값
```

---

## 6. LoRA vs 전체 Fine-tuning vs 레이어 동결 비교

| 항목 | 전체 Fine-tuning | 레이어 동결 | LoRA |
|------|-----------------|------------|------|
| 학습 파라미터 | 전체 (124M) | 일부 (마지막 블록) | LoRA만 (~2M) |
| 메모리 | 매우 큼 | 중간 | 적음 |
| 학습 속도 | 느림 | 중간 | 빠름 |
| 성능 | 최고 | 좋음 | 거의 동등 |
| Catastrophic Forgetting | 위험 | 중간 | 낮음 |
| 원본 가중치 보존 | 손상됨 | 부분 보존 | 완전 보존 |

---

## 7. `nn.init.kaiming_uniform_` — A 행렬 초기화

```python
nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
```

```
Kaiming (He) 초기화:
  ReLU 계열 활성화 함수에 최적화된 초기화 방법
  분산 = 2 / (1 + a^2) / fan_in

a=math.sqrt(5): PyTorch nn.Linear의 기본 초기화와 동일한 설정
→ LoRA A 행렬이 기존 Linear와 유사한 분포로 초기화됨

B는 zeros:
→ 학습 초기에 LoRA 기여 = A@B = 0
→ 원본 모델과 동일한 출력으로 시작 (안정적 학습)
```

---

*작성일: 2026-03-25*
