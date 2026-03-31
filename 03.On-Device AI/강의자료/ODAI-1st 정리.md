# On-Device AI 기본 (ODAI-1st) 정리

> 성균관대 신동군 교수 강의 기반 정리  
> 대상: AI/ML 입문자 및 On-Device AI에 관심 있는 분

---

## 목차

1. [On-Device AI 개요](#1-on-device-ai-개요)
2. [Network Pruning (가지치기)](#2-network-pruning-가지치기)
3. [Quantization (양자화)](#3-quantization-양자화)
4. [Knowledge Distillation (지식 증류)](#4-knowledge-distillation-지식-증류)

---

## 1. On-Device AI 개요

### 1-1. 딥러닝 모델의 종류

AI 모델은 크게 세 가지 구조로 발전해 왔습니다.

#### DNN (Deep Neural Network)
- 여러 층의 뉴런으로 이루어진 기본 신경망
- 각 뉴런: `y = ReLU(Σwᵢxᵢ + b)` — 입력에 가중치를 곱하고, 합산 후 활성화 함수 통과
- 🍰 **비유**: 케이크를 만들 때 여러 단계(밀가루 → 반죽 → 굽기)를 거치듯, DNN도 입력 데이터가 여러 층을 통과하며 최종 결과를 만듦

#### CNN (Convolutional Neural Network)
- 이미지 처리에 특화된 신경망
- Conv(특징 추출) → Pooling(축소) → FC(분류) 구조
- 각 레이어가 점점 더 고수준의 특징(선 → 모양 → 물체)을 추출
- 🔍 **비유**: 처음엔 픽셀의 선을 보고, 그다음엔 눈·코를 인식하고, 마지막엔 "이건 고양이!"라고 분류

#### Transformer & LLM
- 문장의 **어느 부분에 집중할지** 학습하는 Attention 메커니즘 기반
- GPT, BERT 등 대형 언어 모델(LLM)의 기반 구조
- **Multi-modal LLM**: 텍스트뿐 아니라 이미지·음성도 처리 (예: GPT-4V)

### 1-2. LLM 추론 동작 방식

LLM이 답변을 생성할 때 두 단계로 동작합니다.

| 단계 | 이름 | 설명 |
|------|------|------|
| 1단계 | **Prefill** | 입력 토큰 전체를 한 번에 처리. GEMM 연산(행렬×행렬) |
| 2단계 | **Decode** | 토큰을 하나씩 순서대로 생성. GEMV 연산(행렬×벡터) |

- **KV Cache**: 이전에 계산한 Key/Value 값을 메모리에 저장해 두어 반복 계산을 피하는 기법
  - Llama2-7B 기준: 배치=1, 시퀀스 길이=4096이면 약 **2GB** 메모리 필요
- Decode 단계는 GPU를 비효율적으로 사용 → **배치 처리**로 개선 가능

### 1-3. AI 모델은 왜 이렇게 클까?

- **LLM 모델 크기 증가 속도 > GPU 메모리 증가 속도** → 큰 간격(gap) 존재
- Llama2-7B를 FP16으로 로드하면 **~14GB** 필요
- iPhone 15 Pro의 RAM은 **8GB** → 스마트폰에서 직접 실행 불가!

| 모델 | 파라미터 | FP16 | INT4 |
|------|---------|------|------|
| Gemma2 | 2B | 4GB | 1GB |
| Phi-3.5-mini | 3.8B | 7.6GB | 1.9GB |
| Llama2-7B | 7B | 14GB | 3.5GB |

→ **압축 기법(Pruning, Quantization, Distillation)이 필수!**

### 1-4. Cloud AI vs On-Device AI

```
[Cloud AI]                        [On-Device AI]
대용량 모델          →         경량화된 모델
고성능 서버          →         스마트폰/엣지 기기
네트워크 필요         →         네트워크 불필요
높은 지연시간         →         빠른 응답 (저지연)
보안 취약 가능        →         데이터가 기기 안에만 존재
```

**On-Device AI 장점 5가지**
1. 빠른 응답 (서버 왕복 없음)
2. 보안 강화 (내 데이터가 외부로 나가지 않음)
3. 네트워크 연결 불필요 (오프라인 동작 가능)
4. 비용 절감 (클라우드 서버 비용 없음)
5. 개인화된 서비스 (내 기기에서 나만의 AI)

### 1-5. On-Device AI를 가능하게 하는 3가지 요소

```
+--------------------+
| Efficient AI Models |  ← 작은 모델, 모델 압축 기법
+--------------------+
| Edge AI Frameworks |  ← LiteRT, ONNX Runtime, llama.cpp, MLC-LLM 등
+--------------------+
| Hardware Accelerators |← GPU, NPU, PIM 등
+--------------------+
```

#### 대표 온디바이스 소형 모델

| 회사 | 모델 |
|------|------|
| Google | Gemma2 (2B, 9B) |
| Meta | Llama 3.2 (1B, 3B) |
| Microsoft | Phi-3.5-mini (3.82B) |
| Apple | OpenELM (270M~3B) |
| Alibaba | Qwen2 (0.5B, 1.5B) |

#### llama.cpp
- Meta의 LLaMA 구조를 C/C++로 구현한 오픈소스 프로젝트
- CPU/GPU 하이브리드 추론 지원
- 정수 양자화 (1.5b~8b) 지원
- LMStudio, Ollama 등 다양한 UI가 이 위에 구축됨

#### 하드웨어 가속기 (NPU 예시)
| 칩 | NPU 성능 |
|----|---------|
| Qualcomm Snapdragon X | 40 TOPS |
| Intel Core Ultra (Lunar Lake) | 45 TOPS |
| AMD Ryzen AI (Strix Point) | 50 TOPS |
| Apple M4 | 38 TOPS |

> TOPS = Tera Operations Per Second (초당 1조 연산)

### 1-6. 모델 복잡도 지표: Parameters, Activations, FLOPs

| 지표 | 의미 | 비유 |
|------|------|------|
| **Parameters** | 학습된 가중치 수 | 레시피의 재료 개수 |
| **Activations** | 각 레이어의 출력값 크기 | 요리 중간 재료 담는 그릇 크기 |
| **FLOPs** | 연산 횟수 | 요리 단계 수 |

- AlexNet 기준: 파라미터 61M(224MB), 연산량 1.4G FLOPs
- **주의**: MobileNetV2처럼 파라미터를 줄여도 **peak 메모리(activation)는 줄지 않을 수 있음**

---

## 2. Network Pruning (가지치기)

### 2-1. 기본 개념

**왜 Pruning(가지치기)이 가능할까?**

- DNN은 학습을 쉽게 하기 위해 의도적으로 **과다하게 파라미터를 설계**함 (over-parameterized)
- 즉, **중요하지 않은 가중치(weight)가 많음** → 제거해도 성능 유지 가능
- 🌳 **비유**: 나무를 가꿀 때 불필요한 가지를 쳐내도 나무는 잘 자라듯, 신경망도 불필요한 가중치를 제거해도 성능 유지 가능

**Pruning의 역사**

| 연도 | 방법 | 핵심 아이디어 |
|------|------|-------------|
| 1989 | Optimal Brain Damage (LeCun) | 각 가중치가 학습 손실에 미치는 영향(saliency) 계산 → 낮은 것 제거 |
| 2015 | Magnitude-based Pruning | 절댓값이 작은 가중치는 중요도가 낮다고 가정 → 단순히 크기로 판단 |

### 2-2. Pruning 절차

```
[원본 모델] → [중요도 낮은 가중치 제거] → [Fine-tuning(재학습)] → [압축된 모델]
```

**반복적(Iterative) Pruning**
- 한 번에 많이 제거하면 성능 급락 → 조금씩 여러 번 반복
- 각 반복마다 목표 희소성(sparsity)을 점진적으로 높임

**Fine-tuning 팁**
- 학습률(LR)을 원래의 1/10 ~ 1/100로 설정
- 지나치게 많이 제거하면 Fine-tuning으로도 회복 불가

### 2-3. 어떤 가중치를 제거할까? (파라미터 선택)

```
예시: W = [15, 0.2, -9]
      y = ReLU(15·x₀ + 0.2·x₁ - 9·x₂)

→ 0.2(w₁)를 제거하면? 출력에 거의 영향 없음 ✓
→ 15(w₀)를 제거하면? 출력에 큰 영향 → 제거 금지!
```

**원칙**: 제거되는 파라미터가 **덜 중요할수록** 가지치기 후 성능이 유지됨

### 2-4. Pruning 세부 설계 선택지

#### (A) Pruning 단위(Granularity)

| 종류 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **비구조적(Unstructured)** | 개별 가중치를 임의로 제거 | 정확도 유지 높음 | 하드웨어 가속 어려움 |
| **구조적(Structured)** | 채널/필터/행 전체를 제거 | 하드웨어 가속 쉬움 | 정확도 손실 가능 |

> 🔑 **비구조적 Pruning**: 여기저기 산재된 0이 많은 희소(sparse) 행렬 생성  
> 🔑 **구조적 Pruning**: 아예 작은 행렬 → 바로 가속 가능

#### (B) 중요도 기준

**1) Magnitude-based (크기 기반) — 원소 단위**
```
가중치 행렬:   중요도(절댓값):   가지치기 후:
[ 5  -7  1]   [5  7  1]        [ 5  -7   0]  ← 1 제거
[-4   2  6]   [4  2  6]        [-4   0   6]  ← 2 제거
[ 3  -1 -3]   [3  1  3]        [ 3   0  -3]  ← 1 제거
```

**2) Row Vector L1/L2-norm 기반 — 행 단위**
```
L1-norm 계산: 행별 절댓값 합산
행 1: |5|+|-7|+|1| = 13
행 2: |-4|+|2|+|6| = 12
행 3: |3|+|-1|+|-3| = 7 ← 가장 작으므로 이 행 전체 제거

결과: 행 3이 [0, 0, 0]으로 대체됨
```

**3) Scaling-based Channel Pruning (배치 정규화 활용)**
- BN(Batch Normalization) 레이어의 스케일 파라미터 γ에 L1 정규화 추가
- 학습 중 γ가 0에 가까워진 채널 → 출력이 거의 0 → 해당 채널 제거
- 🍰 **비유**: 요리 레시피에서 "이 재료는 거의 쓰이지 않네" → 레시피에서 삭제

#### (C) 레이어별 Pruning 비율 결정 (Sensitivity Analysis)

서로 다른 레이어는 가지치기에 민감도가 다름!

```
방법:
1. 레이어 Lᵢ를 선택
2. 가지치기 비율 r ∈ {0, 0.1, 0.2, ..., 0.9} 시도
3. 각 비율에서 정확도 저하(Δacc) 측정
4. 모든 레이어 반복
5. 허용 가능한 성능 저하 임계값 T 설정 → 비율 결정

※ 첫 번째 레이어는 민감 (많이 제거 못함)
   FC 레이어는 중복성 많음 (많이 제거 가능)
```

### 2-5. 희소 행렬 저장 형식 (Sparse Matrix Format)

가지치기 후 대부분의 값이 0인 **희소(Sparse) 행렬** 생성  
→ 0을 저장하는 건 낭비! → 압축 저장 형식 필요

#### CSR (Compressed Sparse Row) 형식

```
원본 행렬 (A~M은 0이 아닌 값, 나머지는 0):

    A B . .
C D . . .
    E . F .
        G .
H I . . . J
K . . L M .

저장:
Row Pointer:    [0, 2, 4, 6, 7, 10, 13]  ← 각 행의 시작 인덱스
Column Indices: [2, 3, 0, 1, 2, 4, 3, ...]  ← 비영 원소의 열 위치
Values:         [A, B, C, D, E, F, G, ...]  ← 실제 값
```

- **Row Pointer**: 행 i의 비영 원소들은 Column Indices[RP[i] ~ RP[i+1]-1]에 있음
- 장점: 행 단위 접근 효율적
- 단점: 입력 벡터를 여러 번 읽어야 함

#### CSC (Compressed Sparse Column) 형식
- 열 단위 저장
- 출력 갱신이 여러 번 발생하지만, 입력은 한 번씩만 읽음
- 출력이 입력보다 작을 때 메모리 대역폭 절약

### 2-6. 고급 Pruning 패턴

#### Vector-Wise Sparsity
- **IVW (Inner-Vector Wise)**: 벡터 안에서 일정 비율만 제거 (세밀)
- **OVW (Outer-Vector Wise)**: 벡터 전체를 통째로 제거 (굵게)
- **채널 재배열(Permutation)**: 가지치기 전 가중치 순서를 재정렬해 효율 향상

#### N:M Sparsity (NVIDIA 지원)
- **연속된 M개 원소 중 N개를 0으로 만드는** 규칙적 희소 패턴
- 대표 예시: **2:4 sparsity** (4개 중 2개 제거 = 50% 희소성)
- NVIDIA Ampere GPU의 Tensor Core에서 하드웨어 가속 지원
- 정확도 손실 최소화하면서 2배 속도 향상 가능

```
예시 (2:4 sparsity):
원본:  [3.7, 0.2, -1.1, 5.0, 0.9, -4.2, 0.1, 2.8]
적용:  [3.7,   0, -1.1, 5.0,   0, -4.2,   0, 2.8]
         ↑     ↑          ↑     ↑          ↑
       유지  제거        유지  제거        제거
```

### 2-7. 구조적 Pruning에서 압축 효과 분석

**AlexNet 가지치기 결과 예시:**

```
Conv 레이어: 입력에 민감 → 제거 많이 못함
FC 레이어:  중복성 많음 → 대부분의 크기 감소가 여기서 발생
```

- 파라미터 수 ↓, 연산량(FLOPs) ↓
- 하지만 메모리 peak(activation)는 줄지 않을 수 있음 주의

---

## 3. Quantization (양자화)

### 3-1. 기본 개념

**Quantization이란?**

- 연속적인 실수(float) 데이터를 더 **적은 비트의 이산값(정수)**으로 표현하는 기법
- 🌡️ **비유**: 온도를 "23.456℃"가 아니라 "23℃"로 반올림해 저장하는 것
- 저장 용량 ↓, 연산 속도 ↑, 전력 소비 ↓ (대신 약간의 정확도 손실)

**비트폭(bit-width) 종류:**

```
FP32  ────────────────────────────── 정확도 최고, 크기 큼
FP16  ─────────────────────────
BF16  ─────────────────────────
INT8  ──────────────────
INT4  ────────────
INT2  ──────
Binary ─── (1 or -1만)  ←─── 가장 작지만 정확도 손실 큼
```

**숫자 데이터 타입 비교:**

| 타입 | 부호 | 지수 | 가수 | 특징 |
|------|------|------|------|------|
| FP32 | 1bit | 8bit | 23bit | 표준 학습용 |
| FP16 | 1bit | 5bit | 10bit | 빠른 추론 |
| BF16 | 1bit | 8bit | 7bit | FP32와 같은 지수 범위, 학습 안정성 |
| INT8 | 1bit | - | 7bit | 정수, 하드웨어 친화적 |

### 3-2. 균일 양자화 (Uniform / Linear Quantization)

가장 널리 쓰이는 양자화 방식.

**핵심 수식:**
```
양자화:   q = round(r / S) + Z
역양자화: r̃ = S(q - Z)

r: 원본 실수값
q: 양자화된 정수값
S: 스케일 팩터 (scale factor)
Z: 영점 (zero point)
```

**스케일 팩터 계산:**
```
S = (r_max - r_min) / (q_max - q_min)

예시 (8-bit, r범위: -7.59 ~ 10.8):
S = (10.8 - (-7.59)) / (127 - (-128))
  = 18.39 / 255 ≈ 0.072

Z = round(q_min - r_min / S)
  = round(-128 - (-7.59/0.072))
  = round(-22.58) = -23
```

**예시 계산:**
```
원본값 r = 5.47 을 INT8로 양자화:
  q = round(5.47 / 0.072) + (-23)
    = round(75.97) - 23 = 53

역양자화:
  r̃ = 0.072 × (53 - (-23))
    = 0.072 × 76 = 5.472

양자화 오차: 5.472 - 5.47 = 0.002 (매우 작음!)
```

### 3-3. 비대칭 vs 대칭 양자화

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **비대칭(Asymmetric)** | 0을 중심으로 안 함. Z≠0 | 양자화 범위를 최대한 활용 | 구현 복잡 |
| **대칭(Symmetric)** | 0을 중심으로 대칭. Z=0 | 구현 단순 | 범위 낭비 가능 |

**대칭 양자화 예시:**
```
S = r_max / q_max = 10.8 / 127 ≈ 0.085

양자화: q = round(5.47 / 0.085) = 64
역양자화: r̃ = 0.085 × 64 = 5.44
```

### 3-4. 균일 vs 비균일 양자화

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **균일(Uniform)** | 인접 레벨 간 간격 동일 | 하드웨어 구현 간단 | 실제 데이터 분포와 안 맞을 수 있음 |
| **비균일(Non-uniform)** | 간격이 가변적 | 정확도 높음 | 하드웨어 구현 어려움 |

**로그 양자화 (비균일):**
- 곱셈을 비트 시프트(shift)로 대체 가능 → 빠름
- VGG-16: 4비트 균일양자화 27.8% 성능손실 vs 로그양자화 5% 손실

### 3-5. 학습된 양자화 (Learned Quantization)

#### Weight Sharing (가중치 공유)
- K-means 클러스터링으로 비슷한 가중치들을 **하나의 대표값으로 묶음**
- 코드북(Codebook)에 대표값 저장, 각 위치에는 **인덱스**만 저장

```
예시:
원본 가중치: [2.1, 2.3, -0.5, 5.1, 2.0, -0.4]
클러스터링:   [  2,   2,    0,   3,   2,    0]  ← 인덱스만 저장
코드북:       {0: -0.45, 1: ..., 2: 2.13, 3: 5.1}

FP32 6개(24byte) → 인덱스 6개(2bit×6=1.5byte) + 코드북 4개(16byte)
→ 저장 공간 크게 절약!
```

### 3-6. Per-Tensor vs Per-Channel 양자화

**Per-Tensor Quantization:**
- 전체 가중치 텐서에 하나의 S, Z 사용
- 큰 모델에는 잘 작동하지만, 작은 모델에서는 정확도 손실 크게 발생

**Per-Channel Quantization:**
- 출력 채널별로 각각 다른 S, Z 사용
- 채널마다 가중치 범위가 100배 이상 차이 날 수 있음 → 채널별 처리 필요
- 정확도 향상, 하지만 계산 오버헤드 증가

```
Per-Tensor:  전체에 S=0.1 적용 → 채널별 범위 차이로 오차 발생
Per-Channel: 채널1에 S=0.5, 채널2에 S=0.01, ... 각각 적용 → 정확도 ↑
```

### 3-7. Weight Equalization (가중치 균등화)

- 채널마다 가중치 범위가 크게 달라서 생기는 문제 해결
- 아이디어: 레이어 k의 채널 출력 스케일을 줄이고, 레이어 k+1의 채널 입력 스케일을 올려서 **전체 동작은 유지**하되 가중치 범위를 균등화
- 🍰 **비유**: 요리 레시피에서 재료 A를 1/10로 줄이고 재료 B를 10배 늘려서 맛은 같지만 측정이 쉬워지는 것

### 3-8. Static vs Dynamic Quantization

| 방식 | Clipping 범위 결정 | 정확도 | 속도 |
|------|-----------------|--------|------|
| **Static** | 추론 전에 미리 계산 (고정) | 낮음 | 빠름 |
| **Dynamic** | 추론 중 실시간으로 계산 | 높음 | 느림 |

- Static: 배포 전 보정 데이터로 범위 계산 → 실시간 오버헤드 없음
- Dynamic: 각 activation map의 min/max를 실시간으로 계산

### 3-9. Fine-tuning 방법: PTQ vs QAT

#### PTQ (Post-Training Quantization, 학습 후 양자화)
- **이미 학습된 모델**을 가져와서 양자화만 적용
- 소규모 보정 데이터(calibration data)로 S, Z 계산
- 빠르고 편하지만, 저비트(4bit 이하)에서 정확도 손실 큼

```
흐름:
학습완료 모델 → 보정 데이터로 범위 계산 → 양자화 적용 → 배포
```

#### QAT (Quantization-Aware Training, 양자화 인식 학습)
- 학습 도중 양자화 효과를 **시뮬레이션**하며 학습
- 가중치는 FP32로 유지하되, Forward pass에서 양자화 시뮬레이션
- Backward pass에서는 **STE(Straight Through Estimator)** 사용

**STE란?**
- 양자화 함수(반올림)는 미분이 불가능(기울기=0 또는 ±∞)
- → STE로 근사: "반올림을 그냥 항등함수로 취급하고 기울기 그대로 통과"
- 실제로 잘 동작함 (단, 이진 양자화 같은 극단적 경우는 예외)

```
QAT 학습 흐름:
Forward:  FP32 → 양자화 시뮬레이션(INT8처럼) → 손실 계산
Backward: 기울기 역전파 (STE 적용) → FP32 가중치 업데이트
결과:     INT8 가중치만 배포
```

**성능 비교:**
```
PTQ: 소형 모델에서 부정확 (표현력 부족)
QAT: 학습이 더 걸리지만 훨씬 정확, 소형 모델에도 효과적
```

### 3-10. Linear Quantized 행렬 곱셈

실제 추론에서 가중치(W)와 입력(X)의 곱셈을 정수 연산으로 수행:

```
Y = WX (실수 연산) → 정수 연산으로 변환

Sᵧqᵧ = Sᵥ(qᵥ - Zᵥ) · Sₓ(qₓ - Zₓ)

→ qᵧ = (SᵥSₓ/Sᵧ)(qᵥqₓ - Zᵥqₓ - Zₓqᵥ + ZᵥZₓ) + Zᵧ

※ SᵥSₓ/Sᵧ는 [0,1) 범위 → 비트 시프트로 구현 가능
※ 정수 곱셈 + 정수 덧셈만 사용 → 하드웨어 친화적!
```

---

## 4. Knowledge Distillation (지식 증류)

### 4-1. 왜 필요한가?

**작은 모델을 직접 학습시키면 생기는 문제:**
- 큰 데이터셋에서 **언더피팅(underfitting)** 발생
- 모델 표현력이 부족해 제대로 학습이 안 됨

**해결책:** 먼저 크고 정확한 **선생 모델(Teacher)**을 학습시키고,  
그 지식을 작은 **학생 모델(Student)**에게 전달!

> 🎓 **비유**: 경험 많은 교수님(Teacher)이 학생(Student)에게 강의하면,  
> 학생이 혼자 공부하는 것보다 훨씬 효과적으로 배울 수 있음

### 4-2. 핵심 아이디어: Soft Label vs Hard Label

**Hard Label (정답 레이블):**
```
입력: 강아지 사진
정답: [0, 0, 1, 0, 0]  ← 강아지=1, 나머지=0
```

**Soft Label (Teacher의 예측 확률):**
```
입력: 강아지 사진
Teacher 출력: [0.01, 0.02, 0.85, 0.05, 0.07]
                               ↑강아지  ↑고양이
```

→ "이 사진은 85% 강아지인데, 고양이랑도 7% 닮았어" 라는 **풍부한 정보** 전달!  
→ 학생 모델이 클래스 간 **유사성(관계)**도 함께 학습 가능

### 4-3. 온도(Temperature) 개념

**Softmax with Temperature:**
```
p(zᵢ, T) = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)

T = 1:  일반 softmax → 확률 분포가 날카로움 (한 클래스에 집중)
T > 1:  분포가 부드러워짐 (soft labels) → 클래스 간 유사성 더 잘 표현
T → ∞: 균등 분포
```

```
예시 logits: [5, 2, 0.5]

T=1:   [0.94, 0.05, 0.01]  ← 첫 번째에 집중
T=3:   [0.68, 0.23, 0.09]  ← 더 균등 분포 → 관계 정보 풍부
```

### 4-4. Knowledge Distillation 손실 함수

```
전체 손실 = α × KL(Teacher || Student)    ← soft label로부터 학습
          + (1-α) × CrossEntropy(Student, 정답)  ← hard label 학습

α: 두 손실의 균형 조절 하이퍼파라미터
```

**KL Divergence (Forward vs Reverse):**
- Forward KL: KL[Pₜ||Pₛ] — Teacher 분포 전체를 포함하려 함 (mode-covering)
- Reverse KL: KL[Pₛ||Pₜ] — Teacher의 주요 모드에 집중 (mode-seeking)

### 4-5. KD 방법 분류

#### (A) Response-based KD (출력 기반)

```
Teacher: [이미지] → [특징 추출] → [0.85, 0.07, 0.05, ...]
                                           ↓ 지식 전달
Student: [이미지] → [작은 특징 추출] → 비슷한 분포 학습
```

- 최종 출력(로짓) 또는 확률 분포를 맞추는 것
- 가장 간단한 방법

#### (B) Feature-based KD (중간 특징 기반)

- 최종 출력뿐 아니라 **중간 레이어의 특징 맵(feature map)**도 전달
- Teacher의 중간 표현을 Student가 모방

**대표 방법들:**

| 방법 | 핵심 아이디어 |
|------|-------------|
| **FitNets** | Teacher의 중간 feature map을 직접 회귀(regression)로 맞춤. 크기 다르면 변환 레이어 추가 |
| **AT (Attention Transfer)** | Spatial attention map(공간 집중도)을 전달 → 채널 크기 불일치 해결 |
| **NST** | Feature map 분포를 통계적으로 맞춤 (MMD 최소화) |
| **Factor Transfer** | Autoencoder로 Teacher 특징을 압축 후 전달 → Student 친화적 |

#### FitNets 상세:

```
Teacher:  [입력] → [레이어1~중간] → [Hint Layer] → [레이어~끝] → 출력
                                         ↓ 회귀 학습
Student:  [입력] → [레이어1~중간] → [Guided Layer] → [레이어~끝] → 출력
                                    (Teacher hint를 모방)

크기가 다를 경우: Regressor(변환 레이어) 추가로 shape 맞춤
```

#### NST (Neuron Selectivity Transfer):

```
목표: Teacher와 Student의 feature distribution이 비슷해야 함
방법: MMD(Maximum Mean Discrepancy) 최소화

distillation 전: Teacher/Student feature 분포가 크게 다름
distillation 후: 분포가 유사해짐 → 성능 향상
```

#### (C) Structural / Functional KD (구조적 KD)

- 특징값 자체가 아니라 **특징들 사이의 관계(구조)** 전달
- 아키텍처 불일치에 강인함

| 방법 | 핵심 아이디어 |
|------|-------------|
| **FSP (Flow of Solution Procedure)** | 레이어 간 변환 행렬(내적)을 맞춤 |
| **Activation Boundary KD** | Teacher의 ReLU 활성화 패턴(켜짐/꺼짐)을 복제 |
| **Relational KD** | 샘플 간의 거리와 각도 관계를 보존 |

#### Relational KD 예시:

```
Teacher가 학습한 것:
  - "강아지와 고양이는 가깝다 (둘 다 동물)"
  - "강아지와 자동차는 멀다"
  - "두 강아지 이미지는 매우 가깝다"

→ 이 관계 정보를 Student가 학습
→ Student도 같은 공간 기하 구조를 갖게 됨
```

### 4-6. Transformer/LLM에서의 KD

#### MobileBERT
- Teacher: IB-BERT (Inverted-Bottleneck BERT)
- Student: MobileBERT (매우 작은 모델)
- **Feature map 크기를 Teacher와 Student가 동일하게** 설계 → 직접 전달 가능
- Feature map transfer + Attention transfer 동시 적용

#### LLM Pruning + KD (Compact Language Models, NeurIPS'24)
- 대형 LLM을 Pruning으로 먼저 축소
- Pruning 후 재학습(retraining) 시 KD 적용
- 원래 LLM을 Teacher로 사용 → 축소된 모델(Student)의 성능 회복

```
흐름:
대형 LLM → [Pruning] → 축소된 LLM → [KD로 재학습] → 최종 소형 모델
  (Teacher)                                               (Student)
```

---

## 요약 비교표

| 기법 | 핵심 아이디어 | 장점 | 주의점 |
|------|-------------|------|--------|
| **Pruning** | 중요하지 않은 가중치 제거 | 연산량/크기 감소 | Fine-tuning 필요, 구조적 vs 비구조적 선택 |
| **Quantization** | 고정밀 숫자 → 저비트 정수 | 메모리/속도 크게 개선 | PTQ는 간단하지만 손실 큼, QAT는 정확하지만 학습 필요 |
| **Knowledge Distillation** | Teacher → Student 지식 전달 | 작은 모델도 높은 정확도 | Teacher 모델 필요, 학습 시간 추가 |

---

## 핵심 용어 정리

| 용어 | 설명 |
|------|------|
| **Sparsity** | 전체 파라미터 중 0의 비율 (50% sparsity = 절반이 0) |
| **TOPS** | Tera Operations Per Second, 초당 1조 연산 (NPU 성능 단위) |
| **KV Cache** | LLM 추론 시 Key/Value를 메모리에 저장해 재사용하는 기법 |
| **Calibration** | PTQ에서 S, Z를 계산하기 위해 소규모 데이터를 사용하는 과정 |
| **STE** | Straight Through Estimator, 양자화의 미분 불가능 문제를 우회하는 기법 |
| **Soft Label** | Teacher 모델의 확률 출력 (클래스 간 관계 정보 포함) |
| **Temperature** | Softmax 분포의 날카로움을 조절하는 파라미터 |
| **CSR/CSC** | 희소 행렬을 압축 저장하는 형식 (Compressed Sparse Row/Column) |
| **N:M Sparsity** | M개 중 N개를 0으로 만드는 규칙적 희소 패턴 (예: 2:4) |

---

*참고: 성균관대 신동군 교수 On-Device AI 기본 강의 슬라이드 기반*
