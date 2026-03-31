# On-Device AI 응용 강의 정리 (ODAI-2nd)
> 성균관대 신동군 교수 | 초보자 친화 개념 정리

---

## 목차

1. [LLM Pruning (가지치기)](#1-llm-pruning-가지치기)
   - 1.1 Taylor Expansion & OBD
   - 1.2 OBS (Optimal Brain Surgeon)
   - 1.3 SparseGPT
   - 1.4 Wanda
   - 1.5 SPP (Sparsity-Preserved PEFT)
2. [PEFT - LoRA](#2-peft---lora)
3. [LLM Quantization (양자화)](#3-llm-quantization-양자화)
   - 3.1 양자화 개요
   - 3.2 Weight-Activation 공동 양자화
   - 3.3 Weight-Only 양자화
   - 3.4 QLoRA & NF4
4. [Efficient Inference (효율적 추론)](#4-efficient-inference-효율적-추론)
   - 4.1 Sparse Attention
   - 4.2 Speculative Decoding
   - 4.3 KV Cache 최적화 (H2O)
   - 4.4 Operator Fusion & FlashAttention
   - 4.5 Patch-based Inference

---

## 1. LLM Pruning (가지치기)

### 🔑 핵심 개념: Pruning이란?

> **비유:** 회사에서 100명이 일하는데, 실제로 핵심 업무를 하는 사람은 30명뿐이라면?
> 나머지 70명을 줄여도 업무 성과가 크게 떨어지지 않는다면, 그게 바로 **Pruning(가지치기)**입니다.

신경망도 마찬가지입니다. 수많은 파라미터(가중치, weight) 중에서 **별로 중요하지 않은 값들을 0으로 만들어 버리는 기법**이 Pruning입니다.

- **목적:** 모델 크기 ↓, 연산 속도 ↑, 메모리 사용량 ↓
- **Sparsity(희소성):** 전체 가중치 중 0인 비율. 50% sparsity = 절반이 0

---

### 1.1 Taylor Expansion Analysis & OBD

#### 🧮 기본 아이디어: 제거했을 때 손실이 얼마나 늘어나는가?

가중치 하나를 제거하면 모델의 Loss(오차)가 얼마나 바뀌는지를 **테일러 전개(Taylor Expansion)**로 수식화합니다.

```
δL ≈ g_i * δw_i  +  (1/2) * h_ii * δw_i²
     (1차항: 기울기)   (2차항: 헤시안 대각)
```

| 항목 | 의미 | 비유 |
|------|------|------|
| `g_i` (1차 미분) | 기울기 | 산에서 지금 얼마나 가파른가 |
| `h_ii` (2차 미분) | 곡률 (헤시안) | 그 기울기가 얼마나 빠르게 변하는가 |

#### OBD (Optimal Brain Damage, LeCun 1989) 가정

1. 학습이 수렴 → **1차 항 무시** (`g_i ≈ 0`)
2. 파라미터 간 독립 → **교차 항 무시** (`h_ij = 0, i≠j`)
3. 목적함수는 거의 2차 함수 → 3차 이상 무시

결론적으로 각 가중치 `w_i`의 중요도:

```
δL_i ≈ (1/2) * h_ii * w_i²
```

> **비유:** 직원의 중요도 = (그 직원이 얼마나 예민한 역할인지) × (현재 얼마나 열심히 일하는가)²
> → 이 값이 낮은 직원부터 내보냄

**문제점:** 헤시안(Hessian) 행렬 계산이 매우 어렵고 비쌈.

---

### 1.2 OBS (Optimal Brain Surgeon)

#### OBD의 한계 극복

OBD는 파라미터들이 서로 독립이라고 가정했지만, 실제로는 **파라미터들이 서로 영향**을 줍니다.

OBS는 가중치 하나를 제거할 때, **나머지 가중치들을 최적으로 보정**해줌으로써 손실을 최소화합니다.

```
최적 가중치 변화량: δw = -(w_q / H⁻¹_qq) * H⁻¹_{:,q}
제거로 인한 손실:   L_q = (1/2) * w_q² / H⁻¹_qq
```

**절차:**
1. 역 헤시안 H⁻¹ 계산
2. 손실 L_q가 가장 작은 가중치 q 선택
3. 나머지 가중치 δw로 보정
4. 반복

> **비유:** 직원 한 명을 내보낼 때, 그냥 내보내는 게 아니라 남은 팀원들의 역할을 재조정해서 업무 공백을 최소화하는 방식

**후속 연구:** WoodFisher, M-FAC, oBERT 등이 헤시안 근사 방법을 개선

---

### 1.3 SparseGPT (ICML 2023)

#### LLM에서 Pruning이 왜 어려운가?

| 문제 | 설명 |
|------|------|
| 모델이 너무 큼 | LLM은 100B+ 파라미터 → OBS 수백 시간 소요 |
| Magnitude Pruning 한계 | 10% sparsity 이상이면 정확도 급락 |
| 재학습 불가 | LLaMA-2 학습비용 ~278억원, 재학습은 현실적으로 불가 |

#### SparseGPT의 핵심 아이디어: **One-Shot + Layer-wise**

**핵심 문제 - Row-Hessian Challenge:**
- 각 행(row)마다 다른 프루닝 마스크(M_i)를 쓰면, 헤시안 역행렬을 행마다 새로 계산해야 함
- 즉, (H_{M_i})⁻¹ ≠ (H⁻¹)_{M_i} → 재사용 불가

**해결책 - Hessian Synchronization:**

```
[열(Column) 단위로 왼쪽에서 오른쪽으로 가지치기]

W 행렬:
| col1 | col2 | col3 | col4 |
  ↑prune  ↑update  ↑update  ↑update
  (freeze) → → →

역 헤시안: H_U1⁻¹ → H_U2⁻¹ → H_U3⁻¹ → H_U4⁻¹
            (elimination으로 갱신)
```

- 열(column)을 하나씩 pruning하면서 오른쪽만 업데이트, 왼쪽은 고정
- → 모든 행이 **같은 역 헤시안**을 공유 가능!

**성과:**
- OPT-175B, BLOOM-176B를 **4.5시간 이내에** 60% unstructured sparsity 달성
- Magnitude Pruning 대비 훨씬 높은 정확도 유지

---

### 1.4 Wanda (ICLR 2024)

#### 더 간단하면서 더 강한 Pruning

**핵심 아이디어:** 가중치의 중요도 = |가중치 크기| × ||입력 활성화의 L2 norm||

```
Magnitude Pruning:  S_ij = |W_ij|
Wanda:              S_ij = |W_ij| × ||X_j||₂
```

> **비유 (Magnitude Pruning):** 직원의 급여만 보고 중요도 판단 (급여가 낮으면 덜 중요?)
> **비유 (Wanda):** 직원의 급여 × 그 직원이 담당하는 업무량을 보고 판단 (급여가 낮아도 핵심 업무 담당이면 중요!)

**예시 (숫자로 보기):**

```
가중치 W = [4,  0,  1, -1]
입력 놈   = [1,  2,  8,  3]   ← 8번 채널이 매우 활발!

Magnitude: S = [4,  0,  1,  1]  → 4열 제거 (값=1)
Wanda:     S = [4,  0,  8,  3]  → 2,4열 제거 (값=0,1)
              ↑ 8번 채널은 값=1이어도 활성화가 크니 중요!
```

**장점:**
- Weight Update 없음 → SparseGPT보다 단순
- 복잡도: O(d²_hidden) vs SparseGPT O(d³_hidden)
- 성능은 SparseGPT와 동등하거나 우수 (LLaMA 기준)

---

### 1.5 SPP: Sparsity-Preserved PEFT (ICML 2024)

#### 문제: Pruning 후 재학습(Fine-tuning)이 필요한데...

| 방법 | 문제점 |
|------|--------|
| SparseGPT, Wanda | 재학습 없이 pruning → 50% 이상 sparsity에서 성능 저하 |
| 일반 LoRA + 재학습 | LoRA가 **dense 행렬**을 추가 → 모델의 sparsity(희소성) 파괴! |

#### SPP의 해결책: Sparsity를 유지하면서 Fine-tuning

**LoRA vs SPP 핵심 차이:**

```
LoRA:  W_pruned + B × A  (dense 행렬 추가 → sparsity 파괴)
SPP:   W_pruned ⊙ (W_β 반복) ⊙ (W_α 반복)  (element-wise 곱셈 → sparsity 유지!)
```

- `W_α ∈ R^(r×n)`: 열(column) 방향 스케일 파라미터
- `W_β ∈ R^(m×1)`: 행(row) 방향 스케일 파라미터

> **비유:** LoRA는 새 직원을 추가해서 결원을 채우는 방식 → 0이었던 자리가 채워짐
> SPP는 기존 직원의 역할 비중(스케일)만 조정 → 이미 쉬고 있는 직원(=0)은 그대로 쉬게 함

**효과:**
- SparseGPT + SPP → Dense 모델 대비 성능 거의 복구
- Sparsity 유지하면서 downstream task 성능 향상

---

## 2. PEFT - LoRA

### 🔑 핵심 개념: Parameter-Efficient Fine-Tuning

GPT-3 같은 거대 모델을 특정 태스크(번역, 요약 등)에 맞게 **전체 파라미터를 업데이트하지 않고**, 아주 작은 부분만 학습하는 기법.

#### LoRA (ICLR 2024)

**아이디어:** 가중치 업데이트 행렬 ΔW를 두 개의 작은 행렬 곱으로 분해

```
기존 Fine-tuning: y = W₀x + ΔW·x     (ΔW: d×d → 엄청 큼)
LoRA:             y = W₀x + B·A·x     (B: d×r, A: r×d, r << d)
```

**파라미터 비교 (d=1024, r=4):**

```
일반 Fine-tuning: ΔW = 1024 × 1024 = 1,048,576개
LoRA:             A + B = 1024×4 + 4×1024 = 8,192개  (128배 절약!)
```

> **비유:** 큰 그림(ΔW)을 그리는 대신, 기다란 세로 붓(B)과 가로 붓(A)으로 표현. 
> 붓의 움직임(r=rank)이 단순할수록(r이 작을수록) 파라미터가 적어짐.

**LoRA의 4가지 장점:**

| 장점 | 설명 |
|------|------|
| 학습 속도 ↑ | 전체 가중치의 gradient 계산 불필요 |
| 메모리 ↓ | optimizer state가 작은 A, B에만 유지 |
| Catastrophic Forgetting 방지 | W₀는 frozen → 기존 지식 보존 |
| 배포 편의 | 학습 후 B×A를 W₀에 합산 가능 |

**성능:** GPT-3 175B에서 Adapter, PrefixTuning보다 높은 정확도 + 낮은 파라미터 수

---

## 3. LLM Quantization (양자화)

### 🔑 핵심 개념: 양자화란?

> **비유:** 악보에서 음표의 위치를 정밀하게 기록하는 대신, 몇 가지 표준 위치 중 하나로 반올림해서 저장하는 것.
> 정보가 약간 손실되지만, 파일 크기가 크게 줄어듦.

- **FP32 → INT8:** 32비트 실수 → 8비트 정수 (메모리 4배 절약)
- **양자화 오차(Quantization Error):** 반올림으로 인한 오차
- **핵심 도전:** LLM의 **Outlier(이상치)** 처리 — 소수의 극단적으로 큰 값들이 양자화 품질을 망침

---

### 3.1 양자화 개요

| 방식 | 설명 | 대표 방법 |
|------|------|-----------|
| **QAT** (Quantization Aware Training) | 학습 중에 양자화 효과를 시뮬레이션 | CNN, BERT에 적합 |
| **PTQ** (Post-Training Quantization) | 학습 완료 후 양자화 | LLM에 주로 사용 |
| **Weight-Activation Co-Quant** | 가중치+활성화 둘 다 양자화 | ZeroQuant, LLM.int8(), SmoothQuant |
| **Weight-Only Quant** | 가중치만 양자화, 활성화는 FP16 | GPTQ, AWQ, SpQR |

**왜 LLM에는 PTQ가 주류인가?**
- LLaMA-2 학습비용 ≈ 278억원
- 재학습(QAT)은 현실적으로 불가능

---

### 3.2 Weight-Activation 공동 양자화

#### ZeroQuant (NeurIPS 2022, Microsoft DeepSpeed)

**문제 인식:**
- 레이어 간 활성화값(activation)의 범위가 **토큰마다 매우 다름** → static quantization 적용 불가
- 가중치 행렬도 어텐션 헤드별로 분포가 크게 다름

**해결책:**

```
가중치 → Group-wise Quantization
         W 행렬을 g개 그룹으로 나누고, 각 그룹마다 별도의 scale 적용

활성화 → Token-wise Quantization
         추론 중에 각 토큰의 min/max를 동적으로 캘리브레이션
```

> **비유 (Group-wise):** 회사 전체에 하나의 급여 기준이 아니라, 부서별로 다른 기준 적용
> **비유 (Token-wise):** 모든 문장에 동일한 기준이 아니라, 각 문장의 특성에 맞게 조정

**추가 기법 - LKD (Layer-by-layer Knowledge Distillation):**
- 원본(FP16) 모델이 **선생(teacher)**, 양자화 모델이 **학생(student)**
- 레이어별로 출력 차이를 최소화하도록 학습 → 4bit 양자화도 정확도 유지

**성과:** FP16 대비 5.2배 효율, 메모리 3배 절약

---

#### LLM.int8() / GPT3.int8() (NeurIPS 2022)

**핵심 발견:** 6.7B 이상 대형 모델에서 **Outlier(이상치) 특성 출현**
- 특정 채널에 절대값이 비정상적으로 큰 값들이 나타남
- 이를 일반 INT8로 양자화하면 → 나머지 정상값들의 해상도가 극도로 낮아짐

```
예시:
정상 채널: [0.1, 0.2, 0.3, ...]  → INT8 범위에 잘 맞음
이상치 채널: [0.1, 87.3, 0.2, ...] → 87.3 때문에 다른 값들이 전부 0에 몰림!
```

**해결책 - 혼합 정밀도 분해(Mixed-Precision Decomposition):**

```
입력 X와 가중치 W를 두 그룹으로 분리:

[이상치 열] × [해당 W 행] → FP16으로 곱셈 (정밀도 유지)
[정상 값들] × [나머지 W] → INT8으로 곱셈 (효율 확보)

최종 결과 = FP16 결과 + INT8 결과
```

**양자화 절차 (Vector-wise):**
1. 행(row)별 최대 절대값 C_x, 열(col)별 최대 절대값 C_w 계산
2. X / C_x → INT8, W / C_w → INT8
3. INT8 행렬 곱셈 → INT32 출력
4. C_x × C_w / 127² 로 역정규화

---

#### SmoothQuant (ICML 2023, NVIDIA + MIT)

**핵심 관찰:** 이상치(outlier)는 **항상 같은 채널에서 발생**

```
이상치가 고정된 채널에 있다면:
- 활성화 X: 특정 채널에 큰 값 → 양자화 어려움
- 가중치 W: 균일하게 분포 → 양자화 쉬움
```

**아이디어 - 이상치의 어려움을 activation에서 weight로 이전:**

```
Y = X * W
  = (X * diag(s)⁻¹) * (diag(s) * W)
  = X̂ * Ŵ

여기서 s_j = sqrt(max|X_j|) / sqrt(max|W_j|)
```

> **비유:** 두 사람이 무거운 짐을 나를 때, 한 사람(X)이 너무 힘들면 짐 일부를 다른 사람(W)에게 넘겨주는 것.
> 둘 다 적당히 힘든 상태가 되면, 전체 작업이 더 효율적!

**실제 숫자 예시:**

```
원본 X의 특정 열: [1, -16, 2, 6]  ← 이상치: -16 (범위 22)
원본 W의 같은 열: 균일하게 분포   ← 범위가 작음

SmoothQuant 적용 후:
X̂의 해당 열: [1, -4, 2, 2]   ← 범위 6으로 줄어듦 (양자화 쉬움)
Ŵ의 해당 열: 살짝 증가        ← 범위 늘었지만 여전히 쉬움
```

**구현 특징:**
- Linear, BMM(Batched Matrix Multiplication): **INT8** 사용
- LayerNorm, SoftMax: **FP16** 유지 (이 레이어는 계산량이 적어 병목이 아님)

**성과:** FP16과 동등한 정확도 + 추론 속도 향상 + 메모리 절반

---

#### OmniQuant (ICLR 2024)

SmoothQuant을 발전시켜 두 가지 **학습 가능한(Learnable)** 파라미터 도입:

**1) LWC (Learnable Weight Clipping):**
```
가중치 clipping 범위를 고정된 min/max 대신 학습으로 최적화
W_q = clamp(⌊W/h⌋ + z, 0, 2^N - 1)
γ, β ∈ [0,1]: 상한/하한 clipping 강도 (학습 파라미터)
```

**2) LET (Learnable Equivalent Transformation):**
```
채널별 scaling과 shifting을 학습으로 최적화
→ SmoothQuant의 s를 데이터 기반으로 자동으로 찾아줌
```

---

#### QuaRot & SpinQuant (NeurIPS 2024 / ICLR 2025)

**문제:** SmoothQuant은 A8W8까지는 되지만, **A4W4 (4비트 활성화)는 불가**

**해결:** 랜덤 Hadamard 변환(회전 행렬 R)으로 활성화를 회전시켜 outlier 제거

```
Y = X * W
  = (X * R) * (R⁻¹ * W⁻¹)   ← 수학적으로 동일
  = X' * W'                   ← 회전 후 outlier가 분산됨!

R은 직교 행렬: R*Rᵀ = I, |R| = 1 (정보 손실 없음)
```

> **비유:** 이상치가 한쪽에 몰린 데이터를 회전시키면, 특정 방향에만 집중되던 이상치가 여러 방향으로 분산됨.
> → 모든 채널이 비슷한 범위 → 4bit 양자화 가능

**성과:** 가중치 + 활성화 + KV 캐시 **모두 4bit** 양자화 가능

---

### 3.3 Weight-Only 양자화

#### 왜 Weight-Only가 필요한가?

```
W8A8: 배치 추론(batch=128)에 유리 → 계산(Compute) 병목
W4A16: 단일 쿼리(batch=1) 추론에 유리 → 메모리(Memory) 병목
```

| 배치 크기 | 병목 | 적합한 양자화 |
|-----------|------|--------------|
| 큰 배치 (128+) | 계산 (TFLOPS) | W8A8 |
| 단일 쿼리 (1) | 메모리 대역폭 | W4A16 |

W4A16은 계산은 FP16 그대로이지만, **가중치를 4bit로 줄여** GPU 메모리에 올릴 수 있는 모델 크기를 늘림
→ GPTQ 덕분에 OPT-175B를 단일 A100 GPU에서 최초 실행!

---

#### GPTQ (ICLR 2023, Neural Magic)

**OBS를 LLM 양자화에 적용:** 각 가중치를 양자화할 때, 나머지 가중치를 보정

```
[레이어별 독립 처리 - Layer-wise]

각 레이어 내에서:
- W의 행(row)들을 독립적으로 처리
- 열(column)을 왼쪽 → 오른쪽으로 순서대로 양자화

역 헤시안: H_F⁻¹ = (2XX^T)⁻¹  (Cholesky 분해로 계산)
한 열 양자화 후: 오른쪽 미양자화 열들의 가중치 보정
```

**Lazy Batch-Updates:** 열 하나씩이 아니라 블록 단위로 업데이트 → GPU 효율 극대화

**성과:**
- 1.3B 모델: 수 분 / 175B 모델: 4.2시간 (ZeroQuant은 1.3B에 3시간!)
- W3, W4 양자화에서도 성능 유지 (RTN은 붕괴)

---

#### SpQR (ICLR 2024, UW)

**핵심 아이디어:** 이상치 가중치를 따로 고정밀도(16bit)로 저장

**민감도 계산 (GPTQ 방식):**
```
s_ij = (w_ij - quant(w_ij))² / (2 * (XX^T)⁻¹_jj)
```

**민감한 패턴 발견:**
- 행/열 이상치 (Row/Column outliers)
- 민감한 어텐션 헤드
- Rotary embedding 패턴
- 비구조적 이상치

**Bi-level 양자화:**
```
1레벨: 소규모 그룹(8~32개 가중치)별 양자화 → 3~4bit
2레벨: 양자화 scale과 zero-point도 양자화 (2nd order)
+ 이상치: 1% 미만을 16bit로 별도 저장 → CSR 형식으로 메모리 절약
```

---

#### AWQ (MLSys 2024, NVIDIA + MIT)

**핵심 발견:** 가중치의 중요도는 **가중치 크기**가 아니라 **입력 활성화 크기**로 결정됨

```
실험: OPT-6.7B (W3 양자화 기준)
- 전체 INT3 (RTN): PPL 43.2  ← 매우 나쁨
- 상위 1% 채널만 FP16 유지: PPL 13.0  ← 크게 개선!
- 어떤 채널이 중요? → 활성화 크기가 큰 채널!
```

**문제:** 혼합 정밀도(FP16 + INT3)는 하드웨어 비효율적

**해결 - Per-channel Scaling:**
```
WX → Q(W * s) * (s⁻¹ * X)

중요한 채널에 s > 1을 곱하면:
→ 양자화 후 오차가 s배 줄어듦 (더 정밀하게 표현)
→ 반대쪽 s⁻¹은 활성화에 흡수 (수학적으로 동일)
→ 결과적으로 모든 가중치가 INT3이지만, 중요한 채널은 더 정확!
```

> **비유:** 중요한 악기(활성화가 큰 채널)의 음량을 미리 키워두고 녹음하면,
> 나중에 압축했을 때도 그 악기의 음이 덜 손실됨.

**성과:** GPTQ보다 낮은 perplexity + Jetson Orin에서 실시간 실행 가능

---

### 3.4 QLoRA & NF4 (NeurIPS 2023)

#### QLoRA = 양자화된 모델 + LoRA Fine-tuning

```
Full Fine-tuning: 전체 파라미터 FP32 업데이트 → 메모리 최대
LoRA:             W₀(FP16) + A,B(FP16) → 메모리 중간
QLoRA:            W₀(NF4) + A,B(BF16) → 메모리 최소, paging 활용
```

**3가지 핵심 기술:**

**① NF4 (NormalFloat 4-bit) - 새로운 데이터 타입:**
```
LLM 가중치는 정규분포 N(0,1)에 가까움 → 이를 활용!
각 양자화 구간에 동일한 수의 값이 배치되도록 설계
→ 정보이론적으로 최적 (Information-theoretically optimal)

q_i = (1/2) * [Q_X(i / 2^(k+1)) + Q_X((i+1) / 2^(k+1))]
Q_X: 표준 정규분포의 분위수 함수
```

**② Double Quantization (이중 양자화):**
```
기존:   파라미터당 4bit + 스케일 0.5bit = 4.5bit/param
이중:   파라미터당 4bit + 스케일 0.127bit = 4.127bit/param

방법: 양자화 상수(scale factor, FP32)를 또 양자화(FP8)!
```

**③ Paged Optimizer (CPU Offloading):**
- GPU 메모리가 부족하면 Optimizer state를 CPU RAM으로 내보냄
- 필요할 때만 GPU로 불러옴

**성과:** 65B 모델을 단일 48GB GPU에서 fine-tuning 가능!

---

## 4. Efficient Inference (효율적 추론)

### 4.1 Sparse Attention

#### 기본 Attention의 문제

표준 Attention은 **모든 토큰 쌍**을 계산 → O(n²) 계산량
- 긴 문서(n이 크면) → 계산 폭발

#### Longformer & Big Bird

**Longformer (2020):**
```
[Local Attention]     : 각 토큰이 윈도우(2w+1) 안의 토큰만 봄
[Global Attention]    : 특수 토큰([CLS])이 모든 토큰을 봄
[Dilated Attention]   : 간격을 두어 더 넓은 범위를 봄
                        (낮은 레이어: 작은 윈도우, 높은 레이어: 큰 윈도우)
```

**Big Bird (NeurIPS 2020):**
- Longformer에서 Dilated attention을 **랜덤 attention**으로 교체
- = Local + Global + Random → O(n) 복잡도로 긴 문서 처리

> **비유 (Longformer):** 수업 중 질문 시, 항상 모든 학생이 듣는 대신
> → 주변 몇 명(local), 선생님(global)만 들음

---

#### Dynamic Sparsity - SpAtten (HPCA 2021)

**Cascade Pruning: 중요하지 않은 토큰과 헤드를 실시간 제거**

```
입력 문장: "As a visual treat, the film is almost perfect."
           11 Tokens, 12 Heads

Layer 1 (100% 계산)
    ↓ 누적 attention score 낮은 토큰 제거
Layer 2 (38% 계산): "As treat, film perfect."  5 Tokens, 10 Heads
    ↓ 또 제거
Layer 3 (12% 계산): "film perfect"  2 Tokens, 8 Heads

결과: Positive (정확도 유지!)
```

**추가 기법:**
- **Value Pruning:** QK 점수가 낮으면 V(Value)를 아예 메모리에서 안 가져옴
- **Progressive Quantization:** 처음엔 저정밀도로 계산, 확신이 없으면 고정밀도로 재계산

---

#### Deja Vu (ICML 2023)

**Contextual Sparsity:** 입력 문맥에 따라 동적으로 사용할 뉴런/헤드를 예측

```
[일반 방식]
Input → Attention_k → MLP_k → Attention_{k+1} → ...

[Deja Vu]
Input → Attention_k → MLP_k → Attention_{k+1}
              ↓                      ↓
         Predictor_k            Predictor_{k+1}
         (어떤 뉴런을            (어떤 헤드를
          활성화할지 예측)        활성화할지 예측)
```

**Asynchronous Execution (비동기 실행):**
- 임베딩이 레이어 간에 천천히 변한다는 특성 활용
- k번째 블록의 출력으로 k+1번째 블록의 sparsity를 **미리 예측(Look-ahead)**
- 예측과 계산을 병렬로 실행 → 지연 없음

---

#### Conditional Computation - Switch Transformer (JMLR 2022)

**Mixture of Experts (MoE):** 모든 입력이 같은 가중치를 통과하는 대신, **입력마다 다른 전문가(Expert)**를 선택

```
[기존 Transformer]
Token → Self-Attention → FFN (동일한 가중치) → 출력

[Switch Transformer]
Token → Self-Attention → Router → Expert 1 (선택됨)  → 출력
                                  Expert 2 (미선택)
                                  Expert 3 (미선택)
                                  Expert 4 (미선택)
```

- **Router:** 토큰이 어떤 Expert로 갈지 결정
- **장점:** 전체 파라미터는 많지만, 토큰당 계산하는 파라미터는 적음
- **실제:** Mixtral 8x7B, GPT-4 등이 MoE 구조 채택

---

### 4.2 Speculative Decoding (ICML 2023)

#### 문제: LLM의 Auto-regressive 생성은 왜 느린가?

```
"The cat sat on the" → "mat"  (1번 forward pass)
"The cat sat on the mat" → "."  (2번 forward pass)
→ 토큰 하나 생성에 매번 대형 모델 1회 실행 = 느림!
```

#### 해결: 작은 모델이 초안 작성, 큰 모델이 검토

```
[작은 모델 M_q - 초안 작성] (γ=5개 토큰 한번에 생성)
"japan 's benchmark" → "bond n" (틀릴 수 있음)

[큰 모델 M_p - 검토] (γ개 토큰을 병렬로 평가)
- "bond": 확률 분포 불일치 → 거부(빨간색)
- "n": 맞지 않음 → 거부, 큰 모델이 수정(파란색)
```

**핵심 원리:**
- 맞는 토큰은 그대로 수용 (작은 모델 덕에 공짜로 얻음!)
- 틀린 토큰은 큰 모델이 수정 샘플링
- **출력 분포는 순수 M_p와 동일** (수학적으로 증명됨)

```
속도 향상 계수:
α = 작은 모델이 큰 모델을 얼마나 잘 모방하는가 (0~1)
γ = 한 번에 제안하는 토큰 수

실험 결과: T5-XXL 11B 기준 최대 3.4배 속도 향상!
```

> **비유:** 대기업 임원(큰 모델)의 결재가 필요한 문서를
> 팀원(작은 모델)이 초안을 쭉 작성하고,
> 임원은 훑어보고 맞는 부분은 OK, 틀린 부분만 수정.
> → 임원이 처음부터 쓰는 것보다 훨씬 빠름!

---

### 4.3 KV Cache 최적화 - H2O (NeurIPS 2024)

#### KV Cache란?

Auto-regressive 생성 시, 이미 생성한 토큰의 Key/Value를 저장해두어 재계산을 방지:
```
토큰 생성 시마다: Q(새 토큰) × K(모든 이전 토큰) 계산
→ KV Cache에 저장하면 K를 재계산 불필요!
```

**문제:** 긴 시퀀스 → KV Cache가 GPU 메모리를 모두 차지

#### H2O (Heavy-Hitter Oracle)

**핵심 발견:** KV Cache의 5%만 있어도 동일한 토큰 생성 가능

**어떤 5%를 유지할까?**
- **누적 attention score**가 높은 토큰 = "Heavy Hitters(H2)"
- → 많은 쿼리가 이 토큰을 자주 참조한다는 의미

```
[KV Cache 관리 전략]
Heavy Hitter 토큰들 + 최근 토큰들만 유지
누적 attention score가 낮은 오래된 토큰들은 제거

선택 기준:
A_cum[j] = Σ_i softmax(Q_i * K_j^T)   (j번째 토큰의 누적 attention)
```

> **비유:** 도서관에서 자주 빌려가는 책(H2)과 최근에 들어온 신간만 서가에 두고,
> 오래되고 안 빌려가는 책은 창고로 이동.

**성과:** 20% KV Cache만으로도 Full Cache와 거의 동일한 정확도!

---

### 4.4 Operator Fusion & FlashAttention

#### Operator Fusion (연산 융합)

**문제:** GPU는 연산보다 메모리 접근이 더 느린 경우가 많음

```
[일반 방식 - Linear + SiLU]
HBM(GPU 메모리)        SRAM(캐시)
W, X → Load → Compute Linear → Store → Load → Compute SiLU → Store

[Fusion 방식]
W, X → Load → Compute Linear → Compute SiLU → Store
              (중간 결과를 SRAM에서 바로 처리)
```

> **비유:** 음식 배달에서 각 단계마다 (조리 → 냉장고 → 전자레인지 → 냉장고 → 배달) 하는 대신
> → 조리 후 바로 배달 (중간 불필요한 이동 제거)

**효과:** Memory-bound 연산에서 특히 효과적 → 처리 속도 크게 향상

---

#### FlashAttention (NeurIPS 2022)

**표준 Attention 연산의 문제:**

```
Attention = softmax(QKᵀ) * V

[메모리 접근 순서]
1. Q, K 읽기 → QKᵀ 계산 → S 저장(HBM)
2. S 읽기 → Softmax 계산 → P 저장(HBM)
3. P, V 읽기 → PV 계산 → O 저장(HBM)

HBM 읽기/쓰기: 5번! (S, P 행렬이 매우 큼)
```

**FlashAttention의 해결:**

```
[Tiling + Recomputation]
KV를 작은 블록(tile)으로 잘라서 SRAM에 올려 처리
→ 중간 결과(S, P)를 HBM에 저장하지 않음!

HBM 읽기/쓰기: Q, K, V 읽기 + O 쓰기 (4번!)
```

**핵심 수학 - 타일된 Softmax (온라인 알고리즘):**

```
[표준 Softmax - 전체를 한번에 봐야 함]
m(x) = max(x₁, ..., xₙ)     ← 전체 스캔 필요
softmax(xᵢ) = exp(xᵢ - m(x)) / Σ exp(xⱼ - m(x))

[Tiled Softmax - 부분적으로 계산, 나중에 합치기]
x = [x⁽¹⁾, x⁽²⁾] 두 블록으로 나누면:
m(x) = max(m(x⁽¹⁾), m(x⁽²⁾))
ℓ(x) = e^(m(x⁽¹⁾)-m(x)) * ℓ(x⁽¹⁾) + e^(m(x⁽²⁾)-m(x)) * ℓ(x⁽²⁾)
```

**성과:**
- 시퀀스 길이 N에서 메모리 O(N²) → O(N) 절약
- 실제 속도: 2~4배 향상 (A100 기준)
- GPT-4, LLaMA 등 사실상 모든 최신 LLM에 기본 탑재

---

### 4.5 Patch-based Inference (NeurIPS 2021, MCUNetV2)

#### 초소형 디바이스(MCU)에서 추론 문제

MCU(마이크로컨트롤러)는 SRAM이 극히 제한됨 (예: 256kB)

**문제:** MobileNetV2의 초반 레이어들이 256kB를 초과

```
MobileNetV2 SRAM 사용량:
Block 0: 600kB ← 초과!
Block 1: 620kB ← 초과!
Block 2: 1372kB ← 훨씬 초과!  (Peak!)
...
Block 5~: 200kB 이하 ← OK
```

#### 해결: Per-patch Inference

전체 feature map을 한번에 처리하는 대신, **작은 패치(patch) 단위로** 처리

```
[Per-layer inference - 기존]
전체 입력 ──────────────────→ 전체 출력
(메모리에 전체 feature map 유지)

[Per-patch inference - MCUNetV2]
패치1 → conv1 → 중간결과 → conv2 → 패치1 출력
패치2 → conv1 → 중간결과 → conv2 → 패치2 출력
...
(한번에 패치 하나씩, 메모리 대폭 절약)
```

**효과:** Peak SRAM 1372kB → 172kB (8배 절감!)

**문제 - Halo(겹침 영역):**
```
conv 연산에는 주변 픽셀이 필요함 → 패치 간 겹치는 영역(halo) 발생
→ 겹치는 부분을 중복 계산해야 함 (계산량 약간 증가)
→ 하지만 메모리 절약이 훨씬 크므로 실용적!
```

---

## 📊 전체 기술 비교 요약

### Pruning 방법 비교

| 방법 | 방식 | Retraining | 복잡도 | 핵심 아이디어 |
|------|------|------------|--------|--------------|
| Magnitude | Unstructured | O | O(1) | 절댓값이 작은 것 제거 |
| OBD | Unstructured | O | O(n) | 2차 미분 기반 중요도 |
| OBS | Unstructured | X | O(n³) | 역 헤시안 + 가중치 보정 |
| SparseGPT | Unstructured | X | O(d³) | 열 단위 Hessian Sync |
| Wanda | Unstructured | X | O(d²) | 가중치 × 활성화 놈 |
| SPP | Fixed Mask | O | - | Sparsity 유지 PEFT |

### Quantization 방법 비교

| 방법 | 대상 | 방식 | 핵심 기법 |
|------|------|------|-----------|
| ZeroQuant | W+A | Group/Token-wise | LKD 지식 증류 |
| LLM.int8() | W+A | Vector-wise | Outlier 분리 |
| SmoothQuant | W+A | Channel-wise | Outlier를 W로 이전 |
| OmniQuant | W+A | Learnable | LWC + LET 학습 |
| QuaRot | W+A+KV | Rotation | Hadamard 변환 |
| GPTQ | W-only | Column-wise | OBQ 기반 보정 |
| SpQR | W-only | Bi-level | 이상치 FP16 분리 |
| AWQ | W-only | Channel scaling | 활성화 기반 보호 |
| QLoRA | W+LoRA | NF4 | 이중 양자화 + Paging |

### Efficient Inference 방법 비교

| 방법 | 해결 문제 | 핵심 기법 | 속도 향상 |
|------|-----------|-----------|-----------|
| Longformer/BigBird | O(n²) Attention | Sparse Pattern | O(n) |
| SpAtten | 연산량 | Token/Head Pruning | 동적 |
| Switch Transformer | 파라미터 활용 | MoE Routing | - |
| Speculative Decoding | 순차 생성 | 초안+검토 | 최대 3.4× |
| H2O | KV Cache 메모리 | Heavy-Hitter 유지 | 메모리 20× |
| FlashAttention | HBM 접근 | Tiling+Recompute | 2~4× |
| Patch Inference | MCU 메모리 | 패치 단위 처리 | 메모리 8× |

---

## 🔗 기술 간 연관 관계

```
[Pruning 계보]
OBD (1989) → OBS (1992) → SparseGPT (2023) → SPP + LoRA (2024)
                          Wanda (2024) ──────┘

[Quantization 계보]
INT8 기초 → ZeroQuant/LLM.int8() (outlier 발견)
              ↓
          SmoothQuant (채널 이전) → OmniQuant (학습 가능)
              ↓
          QuaRot (회전 변환) → A4W4 가능
              ↓
       GPTQ (weight-only) → SpQR (outlier 분리) → AWQ (활성화 기반)
              ↓
       QLoRA (PEFT + 양자화 결합)

[Efficient Inference 계보]
Sparse Attention → SpAtten → Deja Vu (동적 희소성)
KV Cache → H2O (heavy-hitter)
Operator Fusion → FlashAttention (tiled softmax)
Speculative Decoding (draft + verify)
```

---

*정리: AI전문가교육 26년 5차 | 참고: 성균관대 신동군 교수 강의자료*
