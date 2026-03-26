# Vision Transformer (ViT) + CIFAR-10 코드 설명

> **참고 자료:** `03_04_03_Attention.pdf` (AI Specialist 양성과정 — Attention 강의자료)  
> **대상 독자:** Python 및 Vision LLM 입문자  
> **다루는 범위:** 노트북 처음 ~ `8) 오분류(실패) 샘플만 시각화하기` 까지

---

## 목차

1. [ViT란 무엇인가?](#1-vit란-무엇인가)
2. [0) Setup — 환경 준비](#2-0-setup--환경-준비)
3. [1) 데이터 로딩 & 전처리](#3-1-데이터-로딩--전처리)
4. [2) ViT 모델 구성 개요 — 핵심 블록 구현](#4-2-vit-모델-구성-개요--핵심-블록-구현)
5. [2-1) ViTConfig + ViT 모델 본체](#5-2-1-vitconfig--vit-모델-본체)
6. [2-2) 하이퍼파라미터 설정](#6-2-2-하이퍼파라미터-설정)
7. [3) 모델 구조 요약 & 파라미터 확인](#7-3-모델-구조-요약--파라미터-확인)
8. [4) 체크포인트 저장/이어학습](#8-4-체크포인트-저장이어학습)
9. [5) Training — 학습 루프](#9-5-training--학습-루프)
10. [6) Attention 시각화](#10-6-attention-시각화)
11. [7) Confusion Matrix로 오분류 패턴 확인](#11-7-confusion-matrix로-오분류-패턴-확인)
12. [8) 오분류 샘플 시각화](#12-8-오분류-샘플-시각화)

---

## 섹션별 Python 유틸리티 한눈에 보기

![Python 유틸리티 맵](./fig_09_utils_map.png)

---

## 1. ViT란 무엇인가?

> **강의자료 참고: PDF p.9 — "Vision Transformer"**

CNN(합성곱 신경망)은 이미지의 **국소적(local) 패턴**만 학습합니다.  
반면 ViT는 이미지를 **패치(patch, 작은 조각)** 단위로 잘라 NLP(자연어 처리)의 Transformer에 그대로 넣습니다.

핵심 질문: **"Self-Attention이 Convolution을 대체할 수 있는가?"** (PDF p.9)

| 항목 | CNN | ViT |
|------|-----|-----|
| 입력 단위 | 픽셀 단위 합성곱 | 패치(예: 4×4) 토큰 |
| 수용 영역 | 국소(local) | 전역(global) Self-Attention |
| 장점 | 작은 데이터에서도 강함 | 대규모 데이터에서 뛰어남 |

### ViT 전체 파이프라인 흐름도

![ViT 전체 파이프라인](./fig_01_vit_pipeline.png)

---

## 2. 0) Setup — 환경 준비

> **강의자료 참고: PDF p.9 (ViT 등장 배경)**

```python
import importlib.util, subprocess, sys

def ensure_package(pkg_name, import_name=None):
    name = import_name or pkg_name
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

packages = [("einops", "einops"), ("torchinfo", "torchinfo")]
for pkg, imp in packages:
    ensure_package(pkg, imp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | 비유 |
|---------|------|------|
| `importlib.util.find_spec(name)` | 패키지 설치 여부 확인. `None`이면 미설치 | "이 재료가 냉장고에 있나요?" |
| `subprocess.check_call([...])` | Python 코드 안에서 외부 명령(pip install) 실행 | "없으면 마트에서 사 오기" |
| `sys.executable` | 현재 실행 중인 Python 인터프리터 경로 | "어느 Python을 쓰고 있나요?" |
| `torch.cuda.is_available()` | GPU 사용 가능 여부. `True`/`False` 반환 | "GPU 라는 고성능 계산기가 있나요?" |
| `torch.device('cuda'/'cpu')` | 연산 장치 객체 생성 | "계산을 어디서 할지 선택" |

**왜 두 패키지가 필요한가?**
```
einops    → 텐서 형태 변환을 직관적으로 (rearrange, repeat)
torchinfo → 모델 레이어와 파라미터 수를 테이블로 요약 출력
```

---

## 3. 1) 데이터 로딩 & 전처리

> **강의자료 참고: PDF p.12 — "Patch embedding"**

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # R, G, B 채널별 평균
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),    # 데이터 증강
    T.RandomHorizontalFlip(),       # 데이터 증강
    T.ToTensor(),                   # 텐서 변환
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, ...)
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | 비유 |
|---------|------|------|
| `T.Compose([변환1, 변환2, ...])` | 여러 변환을 순서대로 연결하는 파이프라인 | 공장 컨베이어 벨트 |
| `T.RandomCrop(32, padding=4)` | padding 후 랜덤하게 잘라내기 (데이터 증강) | 사진을 조금씩 다르게 잘라보기 |
| `T.RandomHorizontalFlip()` | 50% 확률로 좌우 반전 (데이터 증강) | 거울에 비친 이미지 |
| `T.ToTensor()` | PIL 이미지 (H,W,C) → PyTorch 텐서 (C,H,W), 0~255 → 0~1 | "사진을 숫자 행렬로" |
| `T.Normalize(mean, std)` | 채널별 `(x - mean) / std` 정규화 | "학교 성적을 표준화" |
| `DataLoader(shuffle=True)` | 배치 단위로 데이터 제공. 매 epoch마다 순서 섞기 | "카드를 섞어서 조금씩 나눠주기" |
| `torchvision.datasets.CIFAR10` | CIFAR-10 자동 다운로드 및 로드 | "데이터셋 다운로더" |

**왜 train은 증강하고 test는 안 할까?**
```
학습 = 다양하게 경험시킨다  →  RandomCrop, RandomFlip 포함
평가 = 공정하게 측정한다    →  ToTensor + Normalize만
```

**텐서 shape 이해:**
```
이미지 한 장: (C, H, W) = (3, 32, 32)
배치 단위:    (B, C, H, W) = (256, 3, 32, 32)
```

**PDF p.12 연결:** 이 32×32 이미지가 이후 패치로 분할됩니다. `x ∈ ℝ^{H×W×C}`

---

## 4. 2) ViT 모델 구성 개요 — 핵심 블록 구현

> **강의자료 참고: PDF p.5, p.11, p.15~18**

### Scaled Dot-Product Attention 흐름도

![Attention 흐름도](./fig_03_attention_flow.png)

### Transformer Encoder 블록 흐름도

![Transformer Block](./fig_04_transformer_block.png)

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | PDF 연결 |
|---------|------|---------|
| `nn.Module` | 모델의 기본 클래스. `__init__`에서 레이어 정의, `forward`에서 실행 흐름 | 모든 블록의 기반 |
| `nn.LayerNorm(dim)` | 레이어 정규화. 토큰 차원 기준으로 평균 0, 표준편차 1 | p.11 `LN(zℓ₋₁)` |
| `nn.Linear(in, out)` | 완전연결층. `y = xW + b` | p.13 Linear Projection |
| `nn.Softmax(dim=-1)` | 마지막 차원을 확률(합계=1)로 변환 | p.5 `softmax(QKᵀ/√d_k)` |
| `nn.Dropout(p)` | 학습 시 p 확률로 뉴런 끄기 (과적합 방지) | - |
| `einops.rearrange(x, pattern)` | 텐서 형태 변환을 직관적인 문자열로 표현 | p.12 이미지→패치 변환 |
| `torch.einsum(notation, q, k)` | 아인슈타인 표기로 효율적 행렬 곱 계산 | p.16 `Q·Kᵀ` 계산 |
| `nn.ModuleList([])` | 여러 레이어를 리스트로 관리 (파라미터 자동 추적) | depth만큼 블록 쌓기 |

**PreNorm 수식 매핑:**
```python
# 코드
x = attn(x) + x   # ← Residual Connection ①
x = ff(x) + x     # ← Residual Connection ②

# PDF p.11 수식
z'ℓ = MSA(LN(zℓ₋₁)) + zℓ₋₁  ← 위 코드의 첫 줄
zℓ  = MLP(LN(z'ℓ))  + z'ℓ   ← 위 코드의 둘째 줄
```

**차원 흐름 (Attention.forward):**
```
입력 x:       (B, N+1, dim)       = (1, 65, 64)
Q, K, V:      (B, heads, N+1, dh) = (1, 4, 65, 64)
dots (Q·Kᵀ):  (B, heads, N+1, N+1) = (1, 4, 65, 65)  ← 패치 간 유사도
attn:         (1, 4, 65, 65)        ← softmax 가중치
출력:         (1, 65, 64)           ← 가중합 후 투영
```

---

## 5. 2-1) ViTConfig + ViT 모델 본체

> **강의자료 참고: PDF p.10, p.12, p.13, p.14, p.19**

### Patch Embedding 상세 흐름도

![Patch Embedding](./fig_02_patch_embedding.png)

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | PDF 연결 |
|---------|------|---------|
| `@dataclass` | 클래스에 `__init__`, `__repr__` 자동 생성. 설정값 묶기에 편리 | ViTConfig |
| `nn.Parameter(torch.randn(...))` | 학습 가능한 파라미터 선언 (역전파로 업데이트됨) | p.14 CLS토큰, 위치임베딩 |
| `einops.Rearrange(pattern)` | `nn.Sequential`에 넣을 수 있는 형태 변환 레이어 | p.12 패치 분할 |
| `einops.repeat(x, '1 1 d -> b 1 d', b=b)` | CLS 토큰(1,1,d)을 배치 크기(b)만큼 복제 | p.14 x_class |
| `torch.cat((cls, x), dim=1)` | CLS 토큰과 패치 시퀀스를 토큰 차원으로 이어붙임 | p.14 [x_class; 패치들] |
| `Union[int, Tuple[int, int]]` | 타입 힌트. 정수 또는 정수 쌍(H, W) 모두 허용 | - |
| `assert 조건, "메시지"` | 조건이 False이면 에러 발생. 입력 검증에 사용 | - |

**전체 forward 차원 흐름:**
```
입력:          (B, 3, 32, 32)
Rearrange:     (B, 64, 48)     ← 64개 패치, 각 4×4×3=48 픽셀 펼침
nn.Linear:     (B, 64, 64)     ← 48차원 → 64차원(dim) 투영
CLS 추가:      (B, 65, 64)     ← 64+1 = 65개 토큰
위치 임베딩:   (B, 65, 64)     ← element-wise 더하기
Transformer:   (B, 65, 64)
CLS 추출:      (B, 64)         ← index 0번 토큰만
MLP Head:      (B, 10)         ← 10개 클래스 logit
```

**PDF p.14 연결:** `z₀ = [x_class; x¹_pE; ... ; xᴺ_pE] + E_pos`

---

## 6. 2-2) 하이퍼파라미터 설정

```python
vit_cfg = ViTConfig(
    image_size=32,    # CIFAR-10: 32×32
    patch_size=4,     # 4×4 패치 → (32/4)² = 64개 패치 토큰
    dim=64,           # 토큰 임베딩 차원
    depth=6,          # Encoder 블록 6개 반복
    heads=4,          # MHSA head 4개
    num_classes=10,
    pool="cls",
)

model = ViT(vit_cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `model.to(device)` | 모델의 모든 파라미터를 GPU 또는 CPU로 이동 |
| `optim.Adam(model.parameters(), lr=...)` | Adam 옵티마이저. `model.parameters()`로 학습할 파라미터 자동 수집 |

**patch_size와 연산량 관계 (PDF p.34 Swin 연결):**
```
patch_size=4  → 패치 64개  → attention map: 65×65   ← 이 실습
patch_size=2  → 패치 256개 → attention map: 257×257 ← 약 15배 연산량 증가!

ViT 복잡도: O((HW)²) ← 패치 수의 제곱에 비례
```

---

## 7. 3) 모델 구조 요약 & 파라미터 확인

```python
from torchinfo import summary

_summary = summary(model, input_size=(1, 3, 32, 32), device=str(device))

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print("Trainable parameters:", count_parameters(model))
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `torchinfo.summary(model, input_size)` | 레이어별 출력 shape, 파라미터 수를 테이블로 출력 |
| `model.parameters()` | 모든 학습 가능한 파라미터를 순회하는 이터레이터 반환 |
| `p.numel()` | 텐서의 총 원소 수 반환 (예: shape (64,64) → 4096) |
| `p.requires_grad` | 이 파라미터가 역전파(학습)에 참여하는지 여부 |

---

## 8. 4) 체크포인트 저장/이어학습

```python
ckpt_path = "vit_cifar10.pth"

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0)
    extra_epochs = 10
else:
    extra_epochs = 20

# 정확도 개선 시에만 저장
if test_acc > best_acc:
    torch.save({"epoch": epoch, "model": model.state_dict(), ...}, ckpt_path)
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `os.path.exists(path)` | 파일이 존재하는지 True/False 반환 |
| `torch.load(path, map_location=device)` | 저장된 텐서/모델 로드. `map_location`으로 GPU/CPU 선택 가능 |
| `model.load_state_dict(dict)` | 저장된 가중치를 모델에 복원 |
| `model.state_dict()` | 모든 파라미터를 `{이름: 텐서}` 딕셔너리로 반환 |
| `optimizer.state_dict()` | Adam의 내부 상태(momentum 등)를 딕셔너리로 반환 |
| `torch.save(dict, path)` | 딕셔너리를 `.pth` 파일로 저장 |
| `dict.get(key, default)` | 키가 없으면 default 값 반환 (KeyError 방지) |

---

## 9. 5) Training — 학습 루프

### train_epoch 흐름도

![train_epoch 흐름도](./fig_05_train_epoch.png)

### evaluate 흐름도

![evaluate 흐름도](./fig_06_evaluate.png)

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | 비유 |
|---------|------|------|
| `model.train()` | 학습 모드 전환. Dropout 활성화 | "시험 공부 모드" |
| `model.eval()` | 추론 모드 전환. Dropout 비활성화 | "실제 시험 모드" |
| `optimizer.zero_grad()` | 이전 배치의 기울기 초기화. **매 배치마다 필수!** | "칠판 지우기" |
| `F.log_softmax(logits, dim=1)` | 로그 소프트맥스. 수치적으로 안정적인 확률 출력 | "확률로 변환" |
| `F.nll_loss(output, target)` | NLL Loss. log_softmax와 합치면 = CrossEntropyLoss | "오답 벌점" |
| `loss.backward()` | 역전파. Chain Rule로 기울기(gradient) 계산 | "어디서 틀렸나 분석" |
| `optimizer.step()` | 기울기로 파라미터 업데이트 (Adam: w = w - lr·∇w) | "틀린 부분 교정" |
| `torch.no_grad()` | 이 블록 안에서 기울기 계산 중지. 메모리 절약 | "채점 시 지우개 들고 다니지 않기" |
| `torch.max(output, dim=1)` | dim=1 방향 최댓값과 그 인덱스(예측 클래스) 반환 | "가장 자신 있는 답 선택" |
| `pred.eq(target)` | 예측과 정답이 같으면 True | "채점하기" |

**학습 5단계 — 꼭 기억하기:**
```
① zero_grad()    ← 기울기 초기화 (안 하면 기울기가 쌓임!)
② forward()      ← 모델 통과 → 예측값 계산
③ loss 계산      ← 정답과 예측의 차이
④ backward()     ← 역전파: 어디를 얼마나 수정할지 계산
⑤ step()         ← 가중치 실제 업데이트
```

**F.log_softmax + F.nll_loss = nn.CrossEntropyLoss:**
```python
# 아래 두 줄은
output = F.log_softmax(model(data), dim=1)
loss = F.nll_loss(output, target)

# 이것과 동일합니다
loss = nn.CrossEntropyLoss()(model(data), target)
```

---

## 10. 6) Attention 시각화

> **강의자료 참고: PDF p.16, p.17, p.51**

### Attention 시각화 파이프라인

![Attention 시각화 파이프라인](./fig_07_attention_viz.png)

### 사용된 Python 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `attn.detach()` | 텐서를 계산 그래프에서 분리. 저장/시각화용 (기울기 계산 불필요) |
| `F.interpolate(tensor, size, mode='bilinear')` | 텐서 크기 변환. (8,8) attention map → (32,32) 이미지 크기로 업샘플링 |
| `tensor.unsqueeze(0)` | 차원 추가. (8,8) → (1,1,8,8) (interpolate 입력 형식 맞춤) |
| `tensor.permute(1,2,0)` | 차원 순서 변경. (C,H,W) → (H,W,C) (matplotlib imshow 형식) |
| `tensor.clamp(0, 1)` | 값을 0~1 범위로 클리핑 |
| `ax.imshow(attn_map, alpha=0.45)` | 반투명 히트맵을 원본 이미지 위에 오버레이 |
| `random.sample(list, k=n)` | 리스트에서 n개를 랜덤하게 비복원 추출 |

**Attention Map 차원 변환 흐름:**
```
last_attn 저장:  [B, H, N+1, N+1] = [1, 4, 65, 65]
CLS 행 추출:     [64]  ← CLS(index=0)가 각 패치(1~64)에 주는 attention
reshape:         (8, 8)  ← 8×8 패치 그리드로 복원
F.interpolate:   (32, 32)  ← 원본 이미지 크기로 업샘플링
ax.imshow:       이미지 위에 히트맵 오버레이
```

**왜 마지막 레이어를 보는 것이 좋을까?**
```
초반 레이어 (layer_idx=0) → 색상, 엣지 같은 저수준 패턴
마지막 레이어 (default)   → "고양이", "자동차" 같은 의미적 패턴 ← 이걸 봐야 직관적
```

---

## 11. 7) Confusion Matrix로 오분류 패턴 확인

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, data_loader):
    ...
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `sklearn.metrics.confusion_matrix(y_true, y_pred)` | 실제 vs 예측 레이블로 혼동 행렬(N×N) 계산 |
| `sns.heatmap(cm, annot=True, fmt='d')` | 행렬을 색상 히트맵으로 시각화. `annot=True`로 숫자 표시 |
| `preds.cpu().numpy()` | GPU 텐서 → CPU 이동 → NumPy 배열 변환 |
| `all_preds.extend(...)` | 리스트를 이어붙이기 (`append`는 중첩, `extend`는 펼침) |

**Confusion Matrix 읽는 법:**
```
행(row)     = 실제 정답 클래스
열(column)  = 모델의 예측 클래스
대각선      = 올바른 예측 (클수록 좋음)
대각선 외   = 오분류 (클수록 많이 헷갈림)

자주 헷갈리는 CIFAR-10 클래스 쌍:
  cat ↔ dog        (비슷한 동물 형태)
  automobile ↔ truck  (차량 유형 유사)
  deer ↔ horse     (4족 보행 동물)
```

---

## 12. 8) 오분류 샘플 시각화

### 오분류 시각화 흐름도

![오분류 시각화 흐름도](./fig_08_misclassified.png)

### 핵심 코드

```python
def plot_misclassified_samples(model, data_loader, max_samples=10):
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            logits = model(data)
            preds = logits.argmax(dim=1)       # 최고 확률 클래스
            wrong_mask = preds.ne(target)       # 오답 위치 Boolean Mask
            wrong_data = data[wrong_mask]       # Boolean Indexing
```

### 사용된 Python 유틸리티

| 유틸리티 | 역할 | 비유 |
|---------|------|------|
| `logits.argmax(dim=1)` | 각 샘플에서 최댓값의 인덱스(예측 클래스) 반환 | "가장 자신 있는 답" |
| `preds.ne(target)` | `ne` = Not Equal. 예측≠정답인 위치에 `True` | "틀린 문제 체크" |
| `wrong_mask.any()` | Boolean 텐서에서 하나라도 True가 있으면 True | "틀린 게 있나요?" |
| `data[wrong_mask]` | **Boolean Indexing**: True인 위치의 샘플만 추출 | "틀린 문제만 골라내기" |
| `tensor.detach().cpu()` | 계산 그래프 분리 → CPU 이동. 시각화 전 필수 | - |
| `img.numpy().transpose(1,2,0)` | (C,H,W) → (H,W,C): matplotlib imshow 형식 변환 | "채널 순서 바꾸기" |
| `np.array(axes).reshape(-1)` | axes 배열을 1D로 평탄화 | "2D 배열을 1줄로" |
| `zip(wrong_data, wrong_target, wrong_preds)` | 여러 이터러블을 묶어서 동시에 순회 | "세 열을 나란히 읽기" |

**Boolean Indexing 직관적 이해:**
```python
data   = tensor([이미지0, 이미지1, 이미지2, 이미지3])
mask   = tensor([False,   True,    False,   True  ])  # 오답 위치
result = data[mask]
# → tensor([이미지1, 이미지3])  ← True인 위치만 선택!
```

**오분류 샘플로 알 수 있는 것:**
```
배경과 객체가 비슷한 색인 경우     → 데이터 증강 강화
객체가 너무 작거나 잘린 경우       → 더 큰 모델/더 많은 데이터
학습 데이터에서 보기 드문 자세/각도 → 다양한 증강 추가
```

---

## 전체 흐름 요약

```
[CIFAR-10 이미지 32×32]
        ↓  T.ToTensor() + T.Normalize()
[Patch Embedding]  → (B, 64, 48) → Linear → (B, 64, 64)
        ↓  einops.Rearrange + nn.Linear
[Token Sequence]   → (B, 65, 64)  (CLS 추가 + 위치 임베딩)
        ↓  Transformer Encoder × 6회
   [LayerNorm] → [MHSA: Q·Kᵀ/√d → Softmax → ×V] → [Residual]
   [LayerNorm] → [FFN: Linear→ReLU→Linear]          → [Residual]
        ↓
[CLS Token 추출]   → (B, 64)
        ↓  nn.LayerNorm + nn.Linear
[MLP Head]         → (B, 10)  ← 10개 클래스 logit
        ↓
[F.nll_loss]  → loss.backward() → optimizer.step()
```

---

## 강의자료(PDF) 연결 요약

| 노트북 섹션 | PDF 페이지 | 핵심 개념 |
|------------|-----------|----------|
| Setup / 데이터 | p.9 | ViT 등장 배경 (CNN 한계) |
| Patch Embedding | p.12 | `N = HW/P²`, `xp ∈ ℝ^{N×(P²C)}` |
| Linear Projection | p.13 | `xp E` (패치 → dim 차원 투영) |
| Position Embedding | p.14 | `z₀ = [x_class; x¹_pE; ...] + E_pos` |
| QKV / Attention | p.5, p.15, p.16 | `A = softmax(qkᵀ/√D_h)` |
| Self-Attention | p.17 | `SA(z) = Av` |
| Multi-Head | p.18 | `MSA(z) = [SA₁(z); ...; SAₖ(z)] U_msa` |
| Classification | p.19 | `y = LN(z⁰_L)` → MLP Head |
| 오분류 분석 | p.22 | ViT 한계 (소규모 데이터, O(N²)) |
