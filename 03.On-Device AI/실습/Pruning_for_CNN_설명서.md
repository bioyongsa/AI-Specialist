# 📘 Assignment 1. Pruning for CNN — 초보자를 위한 코드 설명서

> **대상 독자**: Python과 AI를 처음 배우는 분  
> **목표**: 각 코드 셀이 무엇을 하는지, 왜 필요한지를 쉽게 이해한다

---

## 📋 목차

1. [전체 실습 목표](#1-전체-실습-목표)
2. [Setup — 준비 단계](#2-setup--준비-단계)
3. [모델 크기·정확도 평가](#3-모델-크기와-정확도-평가)
4. [가중치 분포 시각화](#4-가중치-분포-시각화)
5. [Reconstruction Error 함수](#5-reconstruction-error-함수)
6. [1.1 Pruning Granularity (세분성)](#6-11-pruning-granularity-세분성)
7. [1.2 Pruning Ratio (비율)](#7-12-pruning-ratio-비율)
8. [1.3 Pruning Schedule (일정)](#8-13-pruning-schedule-일정)

---

## 1. 전체 실습 목표

**Pruning(가지치기)** 이란, 신경망 모델에서 **중요도가 낮은 가중치(weight)를 0으로 만들어 없애는 것**입니다.  
나무를 가지치기하듯, 쓸모없는 부분을 잘라내어 모델을 더 가볍고 빠르게 만듭니다.

```
원본 모델 (크고 무거움)
  ↓ Pruning
압축 모델 (작고 빠름) → 성능은 최대한 유지!
```

이 실습에서 배우는 3가지 핵심:

| 주제 | 내용 |
|------|------|
| **Pruning Granularity** | 얼마나 세밀하게 가지치기할까? (개별 → 채널 단위) |
| **Pruning Ratio** | 레이어마다 얼마나 잘라낼까? |
| **Pruning Schedule** | 언제, 어떤 순서로 잘라낼까? |

---

## 2. Setup — 준비 단계

### 2-1. 라이브러리 불러오기 (Cell 4)

```python
import math
import random
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from tqdm.auto import tqdm
```

**각 라이브러리의 역할:**

| 라이브러리 | 역할 |
|-----------|------|
| `torch` | 딥러닝의 핵심. 텐서(다차원 배열) 연산 및 신경망 구현 |
| `torch.nn` | 신경망 레이어(Conv2d, Linear 등) 모음 |
| `matplotlib.pyplot` | 그래프·그림 그리기 |
| `torchvision` | 이미지 데이터셋(CIFAR10), 변환 도구 제공 |
| `tqdm` | 진행률 표시줄 (`████░░ 60%` 같은 것) |
| `numpy` | 수치 계산용 배열 라이브러리 |

---

### 2-2. Seed 설정 (Cell 5)

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

**왜 필요한가?**  
딥러닝은 내부적으로 난수(랜덤 숫자)를 많이 사용합니다.  
Seed를 고정하면 실험을 다시 실행해도 **항상 같은 결과**가 나옵니다 → 재현성(Reproducibility) 보장.

```
Seed = 0으로 고정
  → 실행 1번: 정확도 93.4%
  → 실행 2번: 정확도 93.4% (동일!)
  
Seed 없으면:
  → 실행 1번: 정확도 93.4%
  → 실행 2번: 정확도 91.8% (다를 수 있음)
```

---

### 2-3. CIFAR-10 데이터셋 준비 (Cell 7)

```python
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),   # 무작위로 이미지 잘라내기
        RandomHorizontalFlip(),               # 무작위로 좌우 반전
        ToTensor(),                           # PIL 이미지 → PyTorch 텐서
    ]),
    "test": Compose([
        ToTensor(),                           # 테스트는 변환 없이 텐서로만
    ]),
}

dataset = {split: CIFAR10(...) for split in ["train", "test"]}

dataloader = {
    split: DataLoader(dataset[split], batch_size=512, ...)
    for split in ['train', 'test']
}
```

**CIFAR-10이란?**  
32×32 픽셀 크기의 컬러 이미지 60,000장으로 이루어진 데이터셋.  
비행기, 자동차, 새, 고양이 등 **10가지 종류**로 분류됩니다.

```
[훈련용 50,000장] → 모델 학습에 사용
[테스트용 10,000장] → 학습 후 성능 평가에 사용
```

**Data Augmentation(데이터 증강)이란?**  
같은 이미지를 랜덤하게 잘라내거나 뒤집어서 학습 데이터를 다양하게 만드는 기법.  
→ 모델이 더 잘 일반화(처음 보는 데이터에도 잘 맞춤)되도록 도움.

**DataLoader란?**  
데이터를 **배치(batch)** 단위로 나눠서 모델에 공급하는 역할.  
`batch_size=512`는 한 번에 512장씩 처리한다는 의미.

---

### 2-4. 사전 학습된 모델 로딩 (Cell 9)

```python
TORCH_HUB_REPO = "SKKU-ESLAB/pytorch-models"
MODEL_NAME = "cifar10_vgg9_bn"

model = torch.hub.load(TORCH_HUB_REPO, MODEL_NAME, pretrained=True)
if torch.cuda.is_available():
    model = model.cuda()

print(model)
```

**핵심 개념:**
- `torch.hub.load`: 인터넷에서 미리 학습된 모델을 다운로드
- `pretrained=True`: 이미 CIFAR-10으로 훈련된 가중치를 불러옴
- `model.cuda()`: 모델을 GPU 메모리로 이동 (GPU가 없으면 CPU에서 실행)
- 사용 모델: **VGG9 + BatchNorm** 구조 기반의 CNN

```
VGG9 구조 개요:
입력 이미지(32×32×3)
  → Conv 레이어들 (특징 추출)
  → BatchNorm (학습 안정화)
  → ReLU (비선형성)
  → Pooling (크기 축소)
  → Fully Connected (분류)
  → 출력(10개 클래스)
```

---

## 3. 모델 크기와 정확도 평가

### 3-1. 모델 크기 계산 (Cell 11)

```python
def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """모델의 총 파라미터 수 계산"""
    num_counted_elements = 0
    for param in model.parameters():
        num_counted_elements += param.count_nonzero() if count_nonzero_only else param.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """모델의 크기를 비트 단위로 계산"""
    return get_num_parameters(model, count_nonzero_only) * data_width

dense_model_size = get_model_size(model)
print(f"dense model has size={dense_model_size / (1024**2) / 8:.2f} MiB")
```

**용어 설명:**

| 용어 | 의미 |
|------|------|
| `param.numel()` | 파라미터의 총 원소 개수 (num elements) |
| `param.count_nonzero()` | 0이 아닌 원소의 개수 (Pruning 후 사용) |
| `data_width=32` | float32 = 32비트 = 4바이트 |
| `dense model` | Pruning 전 원본 모델 (빽빽하게 가중치가 차있는 상태) |

**크기 계산 예시:**
```
파라미터 수 = 1,000,000개
각 파라미터 = 32비트(float32)
모델 크기 = 1,000,000 × 32 bits = 32,000,000 bits
           = 32,000,000 / 8 bytes = 4,000,000 bytes
           = 4,000,000 / 1024 / 1024 ≈ 3.8 MiB
```

---

### 3-2. 학습(train)과 평가(evaluate) 함수 (Cell 13)

```python
def train(model, dataloader, criterion, optimizer, scheduler, callbacks=None):
    model.train()  # 학습 모드로 전환 (Dropout 등 활성화)
    for inputs, targets in tqdm(dataloader, desc='Train', leave=False):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU로 이동
        optimizer.zero_grad()       # 이전 gradient 초기화
        outputs = model(inputs)     # 모델에 입력 전달 (순전파)
        loss = criterion(outputs, targets)  # 손실 계산
        loss.backward()             # 역전파 (gradient 계산)
        optimizer.step()            # 가중치 업데이트
        scheduler.step()            # 학습률 조정
        if callbacks is not None:
            for callback in callbacks:
                callback()          # 추가 동작 (예: Pruning 마스크 적용)
```

**딥러닝 학습 한 사이클 흐름:**
```
① 데이터 입력(inputs)
      ↓
② 모델 예측(outputs) — 순전파(Forward Pass)
      ↓
③ 정답(targets)과 비교 → 손실(loss) 계산
      ↓
④ 손실을 역방향으로 전파 → 각 가중치의 기여도(gradient) 계산 — 역전파(Backward Pass)
      ↓
⑤ Optimizer가 gradient를 보고 가중치 업데이트
      ↓
⑥ 반복...
```

```python
@torch.no_grad()  # ← 중요! gradient 계산 비활성화 (메모리·속도 절약)
def evaluate(model, dataloader, verbose=True) -> float:
    model.eval()  # 평가 모드 (Dropout 비활성화)
    num_samples, num_correct = 0, 0
    for inputs, targets in tqdm(dataloader, desc="Eval", ...):
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)  # 가장 높은 확률의 클래스 선택
        num_samples += targets.size(0)
        num_correct += (preds == targets).sum()
    return (num_correct / num_samples * 100).item()  # 정확도(%) 반환
```

**`@torch.no_grad()`란?**  
평가할 때는 가중치를 업데이트할 필요가 없으므로, gradient 계산을 꺼서 메모리와 속도를 아낍니다.

---

## 4. 가중치 분포 시각화

### 4-1. 전체 모델 가중치 히스토그램 (Cell 16)

```python
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()

    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 2차원 이상의 파라미터만 (Conv, Linear)
            ax = axes[plot_index]
            param_data = param.detach().view(-1).cpu()  # 1차원으로 펼치기

            if count_nonzero_only:
                param_data = param_data[param_data != 0]

            ax.hist(param_data, bins=bins, density=True, color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    plt.show()
```

**왜 `param.dim() > 1` 조건을 쓰는가?**

| 레이어 | 파라미터 | 차원 | Pruning 대상? |
|--------|---------|------|--------------|
| Conv2d | weight | 4차원 (out_ch, in_ch, kH, kW) | ✅ 예 |
| Conv2d | bias | 1차원 | ❌ 아니오 |
| Linear | weight | 2차원 (out_feat, in_feat) | ✅ 예 |
| Linear | bias | 1차원 | ❌ 아니오 |
| BatchNorm2d | weight/bias | 1차원 | ❌ 아니오 |

→ `dim() > 1`이면 Pruning 핵심 대상인 Conv/Linear weight만 골라집니다.

---

### 4-2. 개별 가중치 텐서 히트맵 시각화 (Cell 26)

```python
def draw_weight_distribution(weight_tensor, title="Weight Distribution"):
    weights = weight_tensor.detach().cpu().numpy()

    # 4D 텐서(Conv2d weight)인 경우 2D로 변환해서 이미지처럼 표시
    if len(weights.shape) == 4:
        co, ci, kh, kw = weights.shape
        weights = weights.transpose(0, 2, 1, 3).reshape(co * kh, ci * kw)

    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude of Weight Values')
    plt.title(title)
    plt.show()
```

**텐서 형태 변환 시각화:**
```
Conv2d weight 원래 형태: (out_ch=128, in_ch=64, kH=3, kW=3)
   ↓ transpose + reshape
2D 배열: (128×3, 64×3) = (384, 192)
   ↓
히트맵으로 표시 (밝을수록 값이 큼)
```

---

## 5. Reconstruction Error 함수

### (Cell 19)

```python
def get_reconstruction_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    return torch.sum((tensor1 - tensor2) ** 2).item()
```

**Reconstruction Error란?**  
Pruning 전·후 모델 출력이 얼마나 달라졌는지를 숫자로 표현한 것.

수식: $\| WX - \hat{W}X \|^2_2$

```
원본 출력 tensor1 = [1.2, 3.4, 0.5]
Pruning 후 tensor2 = [1.1, 3.2, 0.4]
차이 = [0.1, 0.2, 0.1]
차이의 제곱 = [0.01, 0.04, 0.01]
합계(error) = 0.06  ← 이 값이 작을수록 Pruning 영향이 적음
```

---

## 6. 1.1 Pruning Granularity (세분성)

Pruning을 어느 단위로 수행하느냐에 따라 4가지로 나뉩니다.

```
[Fine-grained]  → 개별 원소 하나하나
[Vector-level]  → 행(row) 한 줄
[Kernel-level]  → 3×3 필터 하나 전체
[Channel-level] → 입력 채널 전체
```

---

### 6-1. Fine-grained Pruning (Cell 28) — [실습 1]

**개념:** 가중치 하나하나의 절댓값을 보고, 작은 것부터 제거.

```python
def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    sparsity = min(1.0, max(0.0, sparsity))  # 0~1 범위 클리핑

    if sparsity == 1.0:   # 전부 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0: # 전부 유지
        return torch.ones_like(weight)

    # ① 제거할 원소 개수 계산
    num_pruned_elements = round(weight.numel() * sparsity)
    # weight.numel() = 전체 원소 수
    # sparsity=0.5이면 절반을 제거

    # ② 중요도 = 절댓값
    importance = torch.abs(weight)

    # ③ threshold 계산: num_pruned_elements번째로 작은 값
    threshold = torch.kthvalue(importance.flatten(), num_pruned_elements)[0]

    # ④ 마스크 생성: threshold보다 큰 것만 1(유지), 나머지는 0(제거)
    mask = importance > threshold

    # ⑤ 마스크 적용: weight에 0/1 곱하기
    weight.mul_(mask)

    return mask
```

**단계별 예시 (sparsity=0.5, 총 6개 원소):**
```
weight  = [-0.8, 0.1, 0.5, -0.2, 0.9, 0.05]
importance = [0.8, 0.1, 0.5,  0.2, 0.9, 0.05]

num_pruned = round(6 × 0.5) = 3개 제거

오름차순 정렬: [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
3번째 값(1-indexed) = 0.2 → threshold = 0.2

mask = importance > 0.2
     = [True, False, True, False, True, False]
     = [  1,     0,    1,     0,    1,     0]

결과 weight = [-0.8, 0, 0.5, 0, 0.9, 0]
```

**`torch.kthvalue`의 주의사항:**  
k는 **1-indexed** (1부터 시작). k=0이면 오류 발생.  
→ `num_pruned_elements=0`일 때 문제가 생길 수 있어서,  
이전 대화에서처럼 `torch.sort`를 쓰는 것이 더 안전합니다.

---

### 6-2. Vector-level Pruning (Cell 30)

**개념:** Conv2d weight의 마지막 차원(kW=가로 방향 한 줄)을 벡터로 보고, 벡터 단위로 제거.

```
Conv2d weight shape: (out_ch=128, in_ch=64, kH=3, kW=3)
                                                     ^^^
                                              이 방향(kW=3)이 벡터

각 벡터의 중요도 = 벡터 원소들의 절댓값 합 (L1 norm)
```

```python
# 벡터 개수: out_ch × in_ch × kH
num_vectors = weight.shape[0] * weight.shape[1] * weight.shape[2]

# 중요도: kW 방향으로 합산 (dim=3)
importance = weight.abs().sum(dim=(3,), keepdim=True)
# shape: (128, 64, 3, 1) — 각 벡터의 중요도 스칼라

# threshold 계산 후 마스크 생성 및 확장
mask = importance > threshold
mask = mask.expand_as(weight)  # (128, 64, 3, 1) → (128, 64, 3, 3)로 확장
```

**Fine-grained vs Vector-level 차이:**
```
Fine-grained 마스크 예시 (3×3 커널 하나):
  1 0 1
  0 1 0   ← 들쭉날쭉 (비구조적)
  1 1 0

Vector-level 마스크 예시 (열 단위):
  1 0 1
  1 0 1   ← 열이 통째로 살거나 제거됨 (구조적)
  1 0 1
```

---

### 6-3. Kernel-level Pruning (Cell 32)

**개념:** Conv2d의 3×3 커널 전체를 하나의 단위로 보고 제거.

```
Conv2d weight: (out_ch=128, in_ch=64, kH=3, kW=3)
커널 수 = 128 × 64 = 8,192개
각 커널 = 3×3 = 9개 원소

각 커널의 중요도 = 9개 원소 절댓값의 합 (dim=2,3 방향 합산)
```

```python
num_kernels = weight.shape[0] * weight.shape[1]

importance = weight.abs().sum(dim=(2, 3), keepdim=True)
# shape: (128, 64, 1, 1)

mask = importance > threshold
mask = mask.expand_as(weight)  # (128, 64, 1, 1) → (128, 64, 3, 3)
```

---

### 6-4. Channel-level Pruning (Cell 34)

**개념:** 입력 채널(in_channel) 전체를 하나의 단위로 제거.  
채널을 제거하면 그 채널과 관련된 **모든 연산이 사라지므로** 실질적인 속도 향상 가능.

```
Conv2d weight: (out_ch=128, in_ch=64, kH=3, kW=3)
                              ^^^^^
                         이 방향이 채널

각 입력 채널의 중요도 = 해당 채널의 모든 가중치 절댓값 합
                     = dim=(0, 2, 3) 방향으로 합산 (out_ch, kH, kW 제거)
```

```python
num_channels = weight.shape[1]  # in_ch = 64

importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)
# shape: (1, 64, 1, 1) — 채널별 중요도

mask = importance > threshold
mask = mask.expand_as(weight)  # (1, 64, 1, 1) → (128, 64, 3, 3)
```

---

### 6-5. 4가지 방법 비교 정리

| 방법 | 제거 단위 | 마스크 패턴 | HW 가속 효율 | 정확도 보존 |
|------|----------|-----------|------------|-----------|
| Fine-grained | 개별 원소 | 불규칙 (비구조적) | 낮음 | 높음 |
| Vector-level | 행/열 | 행/열 단위 패턴 | 중간 | 중간 |
| Kernel-level | 3×3 커널 | 커널 단위 | 중간 |중간 |
| Channel-level | 입력 채널 전체 | 채널 통째로 | 높음 | 낮음 |

**일반적 경향:**  
Fine-grained ≈ 정확도 가장 좋음, 하지만 실제 속도 개선 어려움  
Channel-level ≈ 실제 속도 개선 크지만, 정확도 손실 클 수 있음

---

### 6-6. Reconstruction Error 비교 (Cell 38)

```python
# 랜덤 입력으로 각 Pruning 방법의 출력 차이 비교
input_sample = torch.randn(1, 64, 32, 32, device=weight.device)

# 원본 Conv 레이어
conv_original = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False, ...)
conv_original.weight.data = weight.clone()
output_original = conv_original(input_sample)

# Fine-grained pruned Conv
conv_pruned_fine_grained = nn.Conv2d(...)
conv_pruned_fine_grained.weight.data = weight.clone() * mask_fine_grained
output_pruned_fine_grained = conv_pruned_fine_grained(input_sample)

# Reconstruction Error 계산
error = get_reconstruction_error(output_original, output_pruned_fine_grained)
```

같은 sparsity(50%) 기준으로:
- **Fine-grained**: 중요도 낮은 원소만 골라 제거 → Reconstruction Error 가장 작음
- **Channel-level**: 통째로 제거 → 중요한 것도 함께 사라져 Error 커질 수 있음

---

## 7. 1.2 Pruning Ratio (비율)

### 7-1. Sensitivity 분석 (Cell 42)

**목표:** 각 레이어가 Pruning에 얼마나 민감한지 파악.

```python
@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    # sparsities = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    named_conv_weights = [(name, param) for (name, param)
                          in model.named_parameters() if param.dim() > 1]

    accuracies = []
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()  # 원본 백업
        accuracy = []

        for sparsity in tqdm(sparsities):
            prune_weight_fine_grained(param.detach(), sparsity=sparsity)  # 해당 레이어만 Pruning
            acc = evaluate(model, dataloader, verbose=False)               # 전체 모델 정확도 평가
            param.copy_(param_clone)  # 원본 복원! (다음 sparsity 테스트를 위해)
            accuracy.append(acc)

        accuracies.append(accuracy)

    return sparsities, accuracies
```

**동작 원리:**
```
레이어 A만 sparsity=0.4로 Pruning → 정확도 측정 → 복원
레이어 A만 sparsity=0.5로 Pruning → 정확도 측정 → 복원
...
레이어 A만 sparsity=0.9로 Pruning → 정확도 측정 → 복원

레이어 B만 sparsity=0.4로 Pruning → 정확도 측정 → 복원
...
```
→ **한 번에 하나의 레이어만 Pruning**하면서 민감도를 측정.

---

### 7-2. Sensitivity 시각화 (Cell 44)

```python
def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = dense_model_accuracy * 0.96  # 기준: 원본의 96% 수준

    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax.plot(sparsities, accuracies[plot_index], label='accuracy after pruning')
            ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities), ...)
            # 하한선: 이 아래로 내려가면 Pruning 너무 과한 것
```

**결과 해석:**
```
정확도가 sparsity 증가에도 잘 안 떨어지는 레이어 → 덜 민감 → 높은 sparsity 가능
정확도가 sparsity 조금만 올려도 뚝 떨어지는 레이어 → 민감 → 낮은 sparsity 유지
```

---

### 7-3. FineGrainedPruner 클래스 (Cell 46) — [실습 2]

```python
class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        # 초기화 시 즉시 Pruning 수행 및 마스크 저장
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        """저장된 마스크를 모델에 다시 적용 (학습 중 마스크 유지용)"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        """레이어별 sparsity에 따라 Pruning 수행"""
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:
                masks[name] = prune_weight_fine_grained(param, sparsity_dict[name])
        return masks
```

**왜 `apply` 메서드가 필요한가?**  
Fine-tuning(재학습) 중에 optimizer가 가중치를 업데이트하면,  
이미 0으로 만든 가중치가 다시 0이 아닌 값을 가질 수 있습니다.  
→ `apply()`를 매 스텝마다 호출해서 **마스크를 강제로 유지**.

**레이어별 sparsity 지정 예시:**
```python
custom_sparsity_dict = {
    'backbone.conv0.weight': 0.5,  # 50% 제거
    'backbone.conv1.weight': 0.7,  # 70% 제거 (덜 민감한 레이어)
    'backbone.conv7.weight': 0.3,  # 30%만 제거 (민감한 레이어)
    'classifier.weight':     0.5,
}
```

---

### 7-4. Uniform vs Non-uniform Sparsity (Cell 49~51)

```python
# Uniform: 모든 레이어에 동일한 sparsity
sparsity_dict = {name: 0.6 for name, param in model.named_parameters() if param.dim() > 1}

# Non-uniform: sensitivity 분석 결과를 반영한 레이어별 다른 sparsity
custom_sparsity_dict = {
    'backbone.conv0.weight': 0.4,  # 민감 → 낮게
    'backbone.conv3.weight': 0.8,  # 덜 민감 → 높게
    ...
}
```

**비교 결과 (일반적 경향):**
```
같은 전체 Sparsity 기준:
  Uniform Pruning       → 정확도 X%
  Non-uniform Pruning   → 정확도 X+α% (일반적으로 더 높음)
```

---

### 7-5. Global Magnitude Pruning (Cell 53) — [실습 3]

**개념:** 모든 레이어의 가중치를 **하나의 풀(pool)**로 모아서, 전체 기준으로 threshold 결정.

```python
class FineGrainedPrunerV2:
    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity, global_prune):
        masks = dict()
        if global_prune:
            # ① 모든 레이어의 weight를 1차원으로 펼쳐서 하나로 합치기
            parameters_to_prune = []
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    parameters_to_prune.append(param.view(-1))

            all_weights = torch.cat(parameters_to_prune)  # 전체를 하나의 벡터로

            # ② 전체 기준 threshold 계산
            num_elements = all_weights.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(all_weights)
            threshold = torch.kthvalue(importance, num_zeros)[0]

            # ③ 각 레이어에 동일한 global threshold 적용
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    masks[name] = mask
        else:
            # 레이어별 독립 Pruning (기존 방식)
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    masks[name] = prune_weight_fine_grained(param, sparsity)
        return masks
```

**Layer-wise vs Global 비교:**
```
[Layer-wise Pruning]
  레이어 A: 상위 50%만 유지
  레이어 B: 상위 50%만 유지  ← 레이어별로 독립적 threshold

[Global Magnitude Pruning]
  모든 가중치를 하나로 합침 → 전체 상위 50%만 유지
  → 레이어 A: 실제로 30%만 제거될 수도 (중요한 가중치가 많으면)
  → 레이어 B: 실제로 70%가 제거될 수도 (중요하지 않은 가중치가 많으면)
  ← 자동으로 민감한 레이어는 적게, 덜 민감한 레이어는 많이 Pruning됨!
```

---

## 8. 1.3 Pruning Schedule (일정)

### 8-1. One-shot Pruning (Cell 62)

**개념:** 목표 sparsity를 **한 번에** 적용하고, 그 후 fine-tuning으로 성능 회복.

```python
num_finetune_epochs = 5
target_sparsity = 0.95  # 95% 제거 (매우 공격적!)

pruner = FineGrainedPrunerV2(model, target_sparsity, global_prune=True)  # 한 번에 Pruning!

for epoch in range(num_finetune_epochs):
    train(model, dataloader['train'], criterion, optimizer, scheduler,
          callbacks=[lambda: pruner.apply(model)])  # 매 스텝마다 마스크 재적용
    accuracy = evaluate(model, dataloader['test'])
    print(f"Epoch {epoch+1}/{num_finetune_epochs}: accuracy={accuracy:.2f}%")
```

```
One-shot Pruning 흐름:
  원본 모델(93%) → 95% Pruning → 정확도 급락(예: 70%) → 5 epoch 학습 → 회복(예: 88%)
```

---

### 8-2. Sparsity Scheduler (Cell 64) — [실습 4]

**개념:** 학습이 진행될수록 Pruning 비율을 점진적으로 늘림.

**수식:**

$$v^{(t)} = v_f + (v_i - v_f)\left(1 - \frac{t - t_i}{t_f - t_i}\right)^n$$

| 기호 | 의미 |
|------|------|
| $v^{(t)}$ | t번째 epoch의 sparsity |
| $v_i$ | 시작 sparsity (예: 0.9) |
| $v_f$ | 최종 목표 sparsity (예: 0.95) |
| $t_i$ | 스케줄링 시작 epoch |
| $t_f$ | 스케줄링 종료 epoch |
| $n=1$ | Linear (선형) |
| $n=3$ | Cubic (세제곱) |

```python
def get_sparsity_schedule(num_epochs, epoch_start, epoch_end,
                           sparsity_start, sparsity_end, exponent):
    sparsity_schedule = []
    for epoch in range(num_epochs):
        if epoch < epoch_start:
            sparsity = 0  # 아직 Pruning 시작 전

        elif epoch >= epoch_end:
            sparsity = sparsity_end  # 목표 sparsity 유지

        else:
            # 수식 적용
            progress = (epoch - epoch_start) / (epoch_end - epoch_start)
            sparsity = sparsity_end + (sparsity_start - sparsity_end) * (1 - progress) ** exponent

        sparsity_schedule.append(sparsity)
    return sparsity_schedule
```

**Linear vs Cubic 비교 (sparsity_start=0.9, sparsity_end=0.95, epoch 0~2):**
```
Epoch  Linear   Cubic
  0    0.900    0.900
  1    0.925    0.944   ← Cubic이 초반에 더 빠르게 증가
  2    0.950    0.950
  3    0.950    0.950
  4    0.950    0.950
```

---

### 8-3. Iterative Pruning (Cell 68, 70)

**개념:** 매 epoch마다 스케줄에 따라 sparsity를 조금씩 올리며 Pruning + 학습을 반복.

```python
# Linear 스케줄로 Iterative Pruning
for epoch in range(num_finetune_epochs):
    # 이번 epoch의 sparsity로 Pruner 생성
    pruner = FineGrainedPrunerV2(model, linear_sparsity_schedule[epoch], global_prune=True)

    # 학습하면서 매 스텝마다 마스크 적용
    train(model, dataloader['train'], criterion, optimizer, scheduler,
          callbacks=[lambda: pruner.apply(model)])

    accuracy = evaluate(model, dataloader['test'])
    print(f"Epoch {epoch+1}: accuracy={accuracy:.2f}%")
```

**One-shot vs Iterative 비교:**
```
[One-shot Pruning]
  원본 → 95% 한번에 → 학습 → 회복

[Iterative Pruning (Linear)]
  원본 → 90% → 학습 → 92.5% → 학습 → 95% → 학습 → 회복
  (점진적으로 올려가며 모델이 적응할 시간을 줌)

[Iterative Pruning (Cubic)]
  원본 → 90% → 학습 → 94.4% → 학습 → 95% → 학습 → 회복
  (초반에 빠르게, 후반에 천천히 → 일반적으로 성능이 더 좋음)
```

---

## 🔑 핵심 용어 정리

| 용어 | 뜻 |
|------|-----|
| **Weight (가중치)** | 신경망이 학습을 통해 얻은 숫자 파라미터 |
| **Sparsity (희소성)** | 전체 가중치 중 0으로 만든 비율 (0.5 = 50% 제거) |
| **Pruning (가지치기)** | 중요도 낮은 가중치를 0으로 만드는 과정 |
| **Mask (마스크)** | 0(제거)과 1(유지)로 이루어진 같은 크기의 텐서 |
| **Magnitude** | 가중치의 크기 = 절댓값. 작은 것이 덜 중요하다고 판단 |
| **Threshold** | 이 값보다 작으면 제거, 크면 유지하는 기준값 |
| **Reconstruction Error** | Pruning 전후 모델 출력의 차이를 수치로 표현한 것 |
| **Fine-tuning** | Pruning 후 성능 회복을 위한 추가 학습 |
| **Sensitivity** | 레이어가 Pruning에 얼마나 민감한가 |
| **Granularity** | Pruning을 수행하는 단위 크기 |
| **Dense model** | Pruning 전 상태 (가중치가 빽빽히 차있음) |
| **Sparse model** | Pruning 후 상태 (많은 가중치가 0) |

---

## 📌 실습 과제 요약

| 실습 | 파일 위치 | 핵심 구현 |
|------|----------|----------|
| **실습 1** | Cell 28 | `prune_weight_fine_grained` — threshold 계산 및 마스크 생성 |
| **실습 2** | Cell 47 | `custom_sparsity_dict` — sensitivity 분석 결과로 레이어별 sparsity 직접 입력 |
| **실습 3** | Cell 53 | `FineGrainedPrunerV2.prune` — global threshold 계산 (`torch.cat`, `torch.kthvalue`) |
| **실습 4** | Cell 64 | `get_sparsity_schedule` — Linear/Cubic 스케줄 수식 구현 |
