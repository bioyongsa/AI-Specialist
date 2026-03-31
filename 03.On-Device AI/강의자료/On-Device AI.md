### LLM Operations
* Prefill phase: matrix 연산 (M * M)
* Decode phase: vector 연산 (V * M)
* On-Device에서의 batch size는 보통 1
## Memory
* Gemma2: param: 2Billion: FP16 (=2bytes, Billion당 1GB, so total 4GB), INT4로 양자화하면, (1GB)
* Llama2: params: 7Billion: FP16 (=2bytes, Billion당 1GB, so total 14GB), INT4로 양자화하면, (3.5GB)
* LLM InferenceL KV Cache = 2 * Batch Size * Sequence Length * # of Layers * Hidden Dimension * Precision (=>Quantization)
  * if batch size = 1, 2 x 1 x 4096 x 32 x 4096 x 2B
---
* Quantization
* Pruning: matrix에서 중요하지 않은 부분을 0으로
* Low-rank Approximation
* Knowledge Distillation
---
* Llama.cpp (open source AI framework)
*   GGUF/GGML: tensor library
*   다양한 SLM 모델 지원
*   UIs: ollama

## Pruning
* 학습에서는 model이 커야 하고, 추론에서는 일부의 정보만 있으면 된다!
* 학습에서는 model이 크므로 redundancy가 있음 (over-parameterized) => 중복된 parameter들은 삭제!! (이것이 pruning!!)
```
Pruning은 어디서 사용되나?
Pruning은 "Training 완료 후 → Inference 배포 전" 사이에 위치한 Post-training 경량화 기법!

Pre-training  →  Fine-tuning  →  [Pruning]  →  Deployment
                                     │
                          over-parameterized된
                          불필요한 가중치 제거
```

* Magnitude-based Pruning
  * 핵심 아이디어: 가중치의 절댓값(magnitude)이 작으면 → 기여도가 낮다 → 제거해도 된다 (unstructured pruning)

* Sparse matrix compressed format
  * CSR (Compressed sparse row): input vector가 여러 번 사용
  * CSC (Compressed sparse column): output vector가 여러 번 사용

* Sparsity: 전체 가중치 중 0인 것의 비율

```
# 예: 가중치가 10개인 레이어
weight = [0.8, 0.0, 0.0, -0.3, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0]
#                                     0이 6개, 전체 10개

sparsity = 0이 아닌 것 / 전체  가 아니라
sparsity = 0인 것 / 전체 = 6/10 = 0.6
```

```
Inner-vector wise Pruning — "벡터 안에서" 제거
벡터를 [v0, v1, v2, v3] 으로 나눴을 때,
각 벡터 내부의 원소 중 작은 것을 선택적으로 제거

원본:
행0: [ 0.8   0.2   0.9   0.1 │ 0.7   0.3   0.5   0.2 ]
      ←── 벡터V0 ──────────→   ←── 벡터V1 ──────────→

Inner-vector wise 적용 후 (각 벡터 안에서 작은 값 2개 제거):
행0: [ 0.8    0    0.9    0  │ 0.7    0    0.5    0  ]
             ↑제거      ↑제거        ↑제거       ↑제거

핵심:
벡터의 경계는 그대로 ── 벡터 안에서만 솎아냄
벡터마다 독립적으로 판단
→ 앞서 설명한 N:M Sparsity (2:4 등)가 대표적인 Inner-vector wise 기법

Outer-vector wise Pruning — "벡터 통째로" 제거
벡터 자체의 중요도를 판단해서,
중요하지 않은 벡터는 통째로 제거

각 벡터의 norm 계산:
행0: V0 norm=1.21 ✅   V1 norm=0.78 ✅
행1: V0 norm=0.09 ❌   V1 norm=1.35 ✅
행2: V0 norm=1.44 ✅   V1 norm=0.07 ❌
행3: V0 norm=0.11 ❌   V1 norm=1.21 ✅

Outer-vector wise 적용 후:
행0: [ 0.8   0.2   0.9   0.1 │ 0.7   0.3   0.5   0.2 ]  ← 둘 다 유지
행1: [  0     0     0     0  │ 0.9   0.1   0.8   0.6 ]  ← V0 통째로 제거
행2: [ 0.9   0.8   0.7   0.6 │  0     0     0     0  ]  ← V1 통째로 제거
행3: [  0     0     0     0  │ 0.8   0.7   0.9   0.5 ]  ← V0 통째로 제거

핵심:
벡터 내부는 손대지 않음 ── 벡터 단위로 살리거나 통째로 날림
→ 블록 단위 Sparsity, CSR 포맷과 궁합이 좋음
```

## Quantization
* Uniform Quantization (Linear): r = S(q - Z)
  * floating point range: 원래의 범위: r_min ~ 0 ~ r_max
  * integer로 quantized: q_min ~ Z (zero point) ~ Q_max
  * floating-point scale: S라고 설정
  * S (scale값) = (r_max - r_min) / (q_max - q_min)
  * Z (zero-point: 원래의 0이 변환된 값) = round (q_min - r_min / S)
* Non-Uniform Quantization (e.g: Logarithmic)

* Asymmetric Quatization: Z(zero point)가 0이 아님, 구현 복잡
* Symmetric Quantization: Z가 0. 구현 단순, but 8-bit quantizaiton을 예를 들면, -127 ~ +127 의 범위를 갖게 되므로 +128이 mapping되지 않아서 메모리 낭비 발생 가능
* Clipping: -256 ~ 256의 범위를 -127 ~ 127로 mapping하면, 0 부근의 값들이 모두 0으로 mapping되어서 이를 구분해야 할 필요성 대두
  * -256 ~ 256의 범위를 -5 ~ 5범위로 cliping해서 mapping해서 0 부근의 값들을 조금 쉽게 구분할 수 있도록 함. 이 때, 5 이상의 값은 모두 5로, -5 이하의 값은 모두 -5로 매핑
  * 이런 outlier에 대해서도 따로 고려해야 할 수 있어야 함.

---

### PTQ (Post-Training Quantization) vs QAT (Quantization-Aware Training)

<img width="1440" height="1102" alt="image" src="https://github.com/user-attachments/assets/33a44d74-5ae6-47ef-b387-5f086e0e5be5" />

### weight equalization
```
Weight Equalization은 PTQ 파이프라인의 전처리 단계에서 사용됩니다. 실제 INT8 변환이 일어나기 전에 weight 값 자체를 재조정합니다.
[사전 학습된 모델] → [Weight Equalization] → [Calibration] → [INT8 변환]
                      ↑
                   여기서 사용 (Training과 무관, Inference 전 1회)
```
<img width="1440" height="848" alt="image" src="https://github.com/user-attachments/assets/d3605d94-63fb-4cdd-9226-4e83f8b2970e" />

**한계**
ReLU 계열 (ReLU, ReLU6) 에서만 수학적으로 보장됩니다. 
GELU, SiLU 같은 비선형 함수는 scale-equivariant하지 않기 때문에, Transformer 계열 (LLM, ViT) 에는 직접 적용하기 어렵습니다. 
그래서 LLM 양자화에서는 Weight Equalization보다 SmoothQuant나 AWQ 같은 다른 기법이 주로 사용됩니다.

### Quantization Function
* Straight Through Estimator (STE)
  * 계단식 quantization 함수: 미분 불가능.. 그러므로 forward는 그대로 사용하고, backward에서는 미분을 위해 비스듬한 형태의 모양의 quantization function 사용

<img width="1440" height="720" alt="image" src="https://github.com/user-attachments/assets/f46e4e17-752d-48e0-ae92-d1d396de243b" />

```
Forward:  실수 weight → Quantize (계단 함수) → INT 값으로 연산
Backward: gradient가 Quantizer를 역으로 통과할 때 STE 적용 (gradient = 1로 통과)
          → 실수 weight가 업데이트됨

Inference (배포 후): STE 불필요. 이미 잘 학습된 weight를 INT8로 고정해서 사용
```

**ReLU와의 관계**
```
두 가지 측면에서 연관됩니다.
1. 기원이 같은 아이디어 — ReLU의 미분도 x < 0인 구간에서는 0입니다 (dead neuron 문제). 이걸 허용하면서도 학습이 되는 이유가 "0 아닌 구간의 gradient로 충분히 학습된다"는 논리인데, STE도 같은 철학에서 출발합니다. 수학적으로 완벽하지 않아도 경험적으로 작동하면 쓴다는 딥러닝의 실용주의적 접근입니다.
2. 구현 관점에서의 연관 — STE를 구현할 때 실제로 clamp() + ReLU 계열 연산을 활용합니다. clip range [α, β] 안에서만 gradient를 통과시키는 것이 ReLU의 "양수 구간만 통과" 구조와 완전히 동일합니다.
ReLU:  x > 0  → gradient 통과 (=1)
       x ≤ 0  → gradient 차단 (=0)

STE:   α < r < β  → gradient 통과 (=1)
       그 외       → gradient 차단 (=0)
즉 STE는 "양방향으로 clip된 ReLU의 derivative" 와 수식 구조가 동일합니다. ReLU가 0 이하를 잘라내듯, STE는 clip range 밖을 잘라내는 것입니다.
```

## Knowledge Distillation
* small model로는 학습시키기가 어려움. teacher(complex mode)가 가지고 있는 지식을 student(tiny model)에 전달...
* Logit: neural network의 layer를 통과한 이미지의 score (logit)를 softmax를 통과시켜서 output probability를 얻음.
* Distillation에서는 T(temperature)를 1 이상의 크기로 설정해서 큰 값과 작은 값의 편차를 줄이는 방식으로 진행함
* Distillation Loss: teacher model의 logits 과 student model의 logits의 차이 (= 확률분포 차이), 즉 cross-entropy와 같은 개념

## Hessian
* Hessian은 손실 함수(Loss)를 가중치들로 두 번 미분한 2차 편미분 행렬
* Hij=∂2L∂wi∂wjH_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}Hij​=∂wi​∂wj​∂2L​
```
**직관적 의미**
손실(Loss)
    │
    │  ← 1차 미분 = Gradient (경사): 어느 방향으로 내려가야 하나?
    │
    │  ← 2차 미분 = Hessian (곡률): 얼마나 가파른 곡면인가?
    ▼
가중치 공간
Hessian 값의미크다이 가중치를 조금만 바꿔도 Loss가 크게 변함 → 민감한 가중치작다이 가중치를 많이 바꿔도 Loss가 별로 안 변함 → 둔감한 가중치
```

## Optimal Brain Surgeon (OBS)
* 가중치를 양자화(quantization)할 때 발생하는 오차를 다른 가중치들로 보상해주는 기법

## SparseGPT (=> PTP(Post-Trainint Pruning) 기법임)
* mask selection, weight reconstruction을 분할
* 핵심 아이디어: OBS의 원리는 유지하되, 연산을 GPT 스케일에서 동작하도록 근사(Approximation) 한 기법
```
For each Transformer 레이어:
  ├─ 1. 해당 레이어의 가중치 행렬만 추출
  ├─ 2. 입력 데이터로 작은 Hessian만 계산 (전체 X, 레이어 단위 O)
  ├─ 3. 열(column) 단위로 순서대로 처리:
  │     ├─ 가장 "덜 중요한" 가중치 선택 (Hessian 기준)
  │     ├─ 해당 가중치를 0으로 제거 (Pruning)
  │     └─ 같은 열의 나머지 가중치로 오차 보상 (OBS 방식)
  └─ 4. 다음 열로 이동
```
---
```
Hessian
  └─▶ 가중치의 중요도/민감도를 수치화
         │
         ├─▶ OBS: "중요도 기반으로 양자화 오차를 분산시키자"
         │
         └─▶ SparseGPT: "OBS 원리를 GPT 규모에서 열 단위로 근사하여 적용하자"
```

## Wanda (=> 이것도 PTP(Post-Trainint Pruning) 기법)
```
SparseGPT:  Hessian 계산 → 오차 보상(weight update) → Pruning
               ↑ 복잡하고 무거움

Wanda:      가중치 × 활성화 크기만 보고 → 바로 Pruning
               ↑ 단순하고 가벼움
```
```
Wanda의 핵심 아이디어
왜 활성화(Activation)를 함께 본 걸까?
SparseGPT 이전 pruning들은 가중치의 절댓값(magnitude) 만 보고 제거했음.

"작은 가중치 = 중요하지 않다" → 제거

그런데 LLM을 분석해보니 문제가 있었어:
입력 활성화가 매우 큰 특정 feature가 존재 (= "massive activation")

   가중치가 작더라도,
   활성화가 크면 → 출력에 미치는 영향이 큼!

   ∴ 가중치 크기만 봐서는 중요도를 잘못 판단할 수 있음
```
```
OBS (1993)
  └─▶ Hessian 기반 오차 보상 pruning (소형 모델)
         │
         └─▶ SparseGPT (2023)
               └─▶ OBS를 LLM 스케일로 근사 (열 단위 분할)
                        │
                        └─▶ Wanda (2023)
                              └─▶ Hessian 없이 활성화만으로 근사
                                   → 더 빠르고 단순하게 비슷한 성능
```

## PEFT (Parameter-Eficient Fune-Tuning)
* Parameter Efficient Fine-Tuning.
* 모델의 전체 파라미터를 건드리지 말고, 아주 일부만 학습시켜서 효율적으로 파인튜닝하자"는 개념/방향성.
* PEFT 계열 기법들:
  * LoRA ← 오늘의 주인공
  * Prefix Tuning
  * Prompt Tuning
  * Adapter 등...

### LoRA
* Low-Rank Adaptation. PEFT를 구현하는 방법 중 하나
```
W_new = W_original + (A × B)
          ↑              ↑
       건드리지 않음    이 두 작은 행렬만 학습!
```

### SPP (Sparsity-Preserved PEFT)
* 문제의 발단: LoRA가 Sparsity를 깨뜨린다.
* Pruning의 맥락에서 보면...
```
[Pruning 이후 W_original 상태]

W_original =  0.5   0    0    0.3
               0    0    0.7   0
               0   0.2   0    0
              0.4   0    0    0

→ mask가 적용되어 0인 자리는 "죽은 파라미터"
  (N:M sparsity, CSR 포맷 등으로 저장)

여기에 LoRA 업데이트를 더하면,,

W_new = W_original + (A × B)

W_new =  0.5   0.1  0.2  0.3
          0.3   0   0.7   0.1    ← 0이었던 자리에 값이 생겨버림!
          0.1  0.2   0    0.2
          0.4   0.3  0.1  0

→ Sparsity Mask가 완전히 파괴됨
→ 압축했던 의미가 없어짐
```

* Sparsity-Preserved PEFT — "Sparsity를 유지한 채로 PEFT를 하자"
* 핵심 아이디어: Mask를 같이 학습에 반영

```
[일반 LoRA]
W_new = W_original + (A × B)
                     ↑
                     mask 무시하고 그냥 더함

[SPP]
W_new = W_original + Mask ⊙ (A × B)
                     ↑
                     ⊙ = element-wise 곱 (마스크 적용!)
                     → 0이었던 자리는 여전히 0 유지
```

```
사전학습 모델 (Dense)
      │
      ▼
  Pruning 적용
  → Sparse W + Mask 생성
      │
      ▼
  Fine-tuning 필요
      │
      ├── LoRA만 쓰면? → Mask 파괴 → Sparse 의미 없음
      │
      └── SPP 사용 → Mask 보존 + 효율적 Fine-tuning ✅
```

### PTQ (Post-Training Quantization)
* 학습이 완료된 모델의 가중치(및 활성화)를 낮은 비트폭으로 변환하는 기법.
* 재학습 없이 추론 속도와 메모리 효율을 높이는 것이 목표

```
FP32 Model (학습 완료)
        │
        ▼
  [Calibration]  ← 소량의 대표 데이터(Calibration Dataset)로 통계 수집
        │          (min/max, 분포, outlier 등)
        ▼
  [Quantization] ← FP32 → INT8 / INT4 등으로 변환
        │
        ▼
  INT Model (추론용)
```
* 핵심 개념은 Scale(s) 와 Zero-point(z) 를 통한 선형 매핑
```
Q(x) = round( x / s + z )          # 양자화
x̂    = s * (Q(x) - z)               # 역양자화 (dequantization)

s = (x_max - x_min) / (2^b - 1)    # b = bit width
```

**1. Weight-Activation Quantization**
가중치(W)와 활성화(Activation, X) 를 동시에 양자화하는 방식.
```
W (INT)          X (INT)
            │                │
            └────────┬────────┘
                     │
              INT Arithmetic
              (e.g. INT8 GEMM)
                     │
                     ▼
                 Output (INT or FP)
```

**2. Weight-Only Quantization**
가중치(W)만 낮은 비트폭으로 저장하고, 연산 시에는 FP16/BF16으로 역양자화해서 사용하는 방식.
```
저장:   W (INT4/INT8)  ← 메모리 절감
           │
           ▼ (dequantize at runtime)
연산:   W (FP16) × X (FP16)  ← FP 연산
           │
           ▼
        Output (FP16)
```
**두 방식 비교 요약**
```
Weight-Activation          Weight-Only
                ─────────────────          ───────────
저장 형식          INT (W + A)               INT (W만)
추론 연산          INT GEMM                  FP16 GEMM
속도 이점          Compute-bound에 강함       Memory-bound에 강함
정확도 손실        상대적으로 큼              상대적으로 작음
구현 복잡도        높음 (A 보정 필요)         낮음
대표 타겟          엣지 디바이스, 서버        LLM 대규모 모델
```

**ZeroQuant**
Microsoft가 제안한 Weight-Activation PTQ 기법으로, LLM의 대규모 모델에서 INT8 추론을 효율적으로 실현하기 위해 설계

**핵심 아이디어 3가지**
* Group-wise Quantization (가중치)
  * 가중치를 작은 그룹 단위로 나누어 각각 독립적인 scale을 적용.
```
W (전체) 를 g개씩 그룹 분할
  ┌────────────────────────────┐
  │ [g0: s0] [g1: s1] [g2: s2] │  ← 그룹마다 다른 scale
  └────────────────────────────┘

- 그룹 크기 g가 작을수록 정밀도 ↑, 오버헤드 ↑
- 전체 단일 scale보다 outlier에 강건함
```
* Token-wise Quantization (활성화)
  * Activation을 토큰 단위로 scale을 개별 계산.
```
입력 시퀀스: [Token_0, Token_1, ..., Token_T]
                │          │             │
              scale_0    scale_1      scale_T  ← 토큰마다 동적 계산

- 각 토큰의 분포가 다르므로 토큰별 scale이 정확도에 유리
- Static calibration 대비 outlier 대응력 향상
```

**SmoothQuant**
* outlier...???

**OmniQuant**
* Learnable Weight Clipping (LWC)
* Learnable Equivalent Transformation (LET)
  * (p.31 그림) tensor로 quantization하면, channel별로 scattering되는데, 이들을 모아주고 (shift), 다시 channel-wise로 scaling해서 quantization함

**QuaRot**
* Quantization Rotation(?)
  * quantization matrix가 타원형으로 되어있는 모양을 회전시켜서 x와 y의 편차를 적게함으로써 (정사각형 모양으로 만듦) quantization을 쉽게 하도록.. (SmoothQuant와 비슷??)
  * 
    
