# AI 인증시험 API 레퍼런스 — 5개 분야 백지 구현

> 시험에서 **빈칸이 아니라 한 줄 전체**를 써야 하는 상황을 대비한 문서입니다.
> "이 함수가 어느 패키지 소속인지"를 외우는 대신 **규칙으로 추론**하는 것이 목표입니다.

## 대상 범위 (29개 노트북)

| 디렉토리 | 노트북 | 코드량 |
|---|---|---|
| **01.Language** | `Chapter_2_Exercise_Dataset` ~ `Chapter_7_Exercise_Follow_Instructions(_dpo)` (9개) | 107K자 |
| **02.Vision** | `01_ResNet18_CIFAR10` ~ `06_Stable_Diffusion_v1_4` (6개) | 141K자 |
| **03.On-Device_AI** | `1. Pruning for CNN`, `2. Quantization for CNN`, `3. Knowledge Distillation`, `4. Pruning for LLM`, `5. Quantization for LLM_colab` (5개) | 168K자 |
| **04.Time_Series** | `ts_practice`, `RecSys_GCF_practice`, `RecSys_NCF` (3개) | 33K자 |
| **05.RAG_LLM** | `1. Llama_index`, `2. RAG`, `1. Data_preprocessing`, `2. Task_1`, `3. Task_2`, `4_RAG_framework_evaluation_with_MCP` (6개) | 84K자 |

이 문서의 모든 API 목록은 **위 29개 노트북의 코드 셀을 직접 스캔**해 추출한 것입니다.

> **범위에서 빠진 것**
> - `01.Language/Chapter_1_Exercise_Vector Space.ipynb` — `gensim` 단어벡터 실습으로 PyTorch 코드가 없어 제외
> - `01.Language/llm_from_scratch/` 하위 노트북 — 원본 교재본
> - `03.On-Device_AI/실습/*_answer.ipynb` — 정답본
> - `04.Time_Series/*_sol(ution).ipynb` — 정답본

## 분야별 라이브러리 지형

| 분야 | 핵심 라이브러리 |
|---|---|
| 01.Language | `torch`, `tiktoken`, `transformers` |
| 02.Vision | `torch`, **`torchvision`**, **`einops`**, `torchinfo`, **`sklearn`**, `PIL` |
| 03.On-Device_AI | `torch`, **`torch.quantization`**, `transformers`, `datasets`, `torchprofile`, `copy` |
| 04.Time_Series | `torch`, **`sklearn`**, `pandas`, **`torch_geometric`**, `yfinance` |
| 05.RAG_LLM | **`llama_index`**, `openai`, `bs4`, `blingfire`, `nltk` |

> ⚠️ **`sklearn`은 02.Vision과 04.Time_Series에서 실제로 사용됩니다.** 출제 포인트의 "sklearn 등 주요 library" 항목은 이 두 분야에 해당합니다.

---

# 제1부. 개념 정리 (모든 분야 공통)

## 1-1. 함수(function)와 메서드(method)는 같은 것인가?

**메서드는 함수의 한 종류입니다.** 정확히는 — **객체에 소속된 함수를 메서드라고 부릅니다.**

파이썬에게 직접 물어보면 이렇게 답합니다:

```python
type(torch.tensor)   # builtin_function_or_method
type(x.view)         # builtin_function_or_method   ← 똑같음!
```

**타입이 완전히 동일**합니다. 즉 본질적으로 같은 것이고, **어디에 소속되어 있느냐**만 다릅니다.

```
함수(function) ⊃ 메서드(method)
     │                │
  독립적으로 존재     객체에 붙어서 존재
  torch.tensor(...)   x.view(...)
```

### 결정적 차이 — 메서드는 자기 객체를 자동으로 받는다

```python
x.view(2, 3)          # 우리가 쓰는 형태
Tensor.view(x, 2, 3)  # 실제로 일어나는 일 — x가 첫 인자로 들어감
```

클래스 정의를 보면 이게 드러납니다:

```python
class LayerNorm(nn.Module):
    def forward(self, x):     # ← self 가 첫 번째 인자
        ...
model.forward(x)              # self=model 이 자동으로 채워짐
```

| | 함수 | 메서드 |
|---|---|---|
| 소속 | 모듈/패키지 | 객체(인스턴스) |
| 형태 | `모듈.이름(...)` | `객체.이름(...)` |
| 첫 인자 | 직접 다 넘김 | **객체 자신이 자동으로 들어감** |
| 예시 | `torch.tensor(lst)` | `x.view(2,3)` |

> **정리**: "객체에 붙어 있을 때만 메서드"라는 이해가 정확합니다.

## 1-2. 어떤 패키지 뒤에 클래스가 오고, 어떤 패키지 뒤에 함수가 오는가?

### 결론: **패키지가 결정하지 않습니다. 이름의 대소문자가 결정합니다.**

가장 확실한 증거 — **같은 패키지 안에 둘 다 있습니다**:

```python
torch.tensor   ->  function   (소문자)
torch.Tensor   ->  class      (대문자)
```

### 실제로 규칙이 지켜지는지 검사해봤습니다

`torch.nn` 안의 모든 이름을 조사한 결과:

| 이름 형태 | 클래스 | 클래스 아님 |
|---|---|---|
| **대문자로 시작** | **166개** | 0개 |
| 소문자로 시작 | **0개** | 14개 |

**예외가 단 하나도 없습니다.** 파이썬 공식 코딩 규약(PEP 8)이고, 이 시험 범위의 모든 라이브러리가 따릅니다.

```
대문자로 시작  →  클래스 (Linear, Conv2d, MinMaxScaler, VectorStoreIndex, DataLoader)
소문자로 시작  →  함수   (tensor, zeros, rearrange, train_test_split, save_image)
```

### 패키지는 "경향"을 알려준다

| 패키지 | 주로 들어있는 것 | 대표 예 |
|---|---|---|
| `torch` (최상위) | **함수** — 텐서 만들기·전역 연산 | `torch.tensor`, `torch.cat` |
| `torch.nn` | **클래스** — 가중치를 가진 층 | `nn.Linear`, `nn.Conv2d` |
| `torch.nn.functional` (`F`) | **함수 전용** — 가중치 없는 계산 | `F.relu`, `F.cross_entropy` |
| `torch.optim` | **클래스** — 옵티마이저 | `optim.Adam`, `optim.SGD` |
| `torch.utils.data` | **클래스** — 데이터 도구 | `Dataset`, `DataLoader`, `TensorDataset` |
| `torchvision.transforms` | **클래스** — 이미지 변환 | `Compose`, `ToTensor`, `Normalize` |
| `torchvision.datasets` | **클래스** — 데이터셋 | `CIFAR10` |
| `torchvision.models` | **함수** — 모델 생성기 | `resnet18()`, `resnet50()` |
| `sklearn.preprocessing` | **클래스** — 변환기 | `MinMaxScaler` |
| `sklearn.metrics` | **함수** — 평가 지표 | `mean_squared_error`, `confusion_matrix` |
| `einops` | **함수** — 축 재배열 | `rearrange`, `repeat`, `reduce` |
| `llama_index.core` | **클래스** — 인덱스·리더 | `VectorStoreIndex`, `SimpleDirectoryReader` |

> `torchvision.models.resnet18`은 **소문자 = 함수**입니다. 클래스가 아니라 **모델을 만들어 돌려주는 함수(팩토리)** 라는 뜻입니다.

### 더 근본적인 질문 — 왜 어떤 건 클래스로 만들었나?

**기억할 것(상태)이 있으면 클래스, 없으면 함수입니다.**

```python
self.dropout = nn.Dropout(0.1)   # 클래스: train/eval 상태를 기억해야 함
x = self.dropout(x)

x = F.dropout(x, 0.1)            # 함수: 기억할 것 없이 즉시 계산하고 끝

scaler = MinMaxScaler()          # 클래스: 학습 데이터의 min/max를 기억해야 함
scaler.fit_transform(train)      #         그래야 test에 같은 기준을 적용
```

## 1-3. 이름의 각 부분을 부르는 용어

```
torch  .  optim  .  AdamW  (  model.parameters()  ,  lr=5e-5  )
  │         │         │              │                  │
  │         │         │              │                  └ 키워드 인자 (keyword argument)
  │         │         │              └ 위치 인자 (positional argument)
  │         │         └ 클래스 (class)
  │         └ 서브패키지 / 하위 모듈 (subpackage, submodule)
  └ 패키지 / 최상위 모듈 (package, top-level module)
```

| 부분 | 용어 | 설명 |
|---|---|---|
| `torch` | **패키지(package)** | 최상위 이름공간. 폴더 하나에 해당 |
| `optim` | **서브패키지(subpackage)** | 패키지 안의 하위 폴더 |
| `AdamW` | **클래스(class)** | 대문자로 시작 |
| `torch.optim.AdamW` 전체 | **정규화된 이름(fully qualified name)** | "전체 경로" |
| `.` | **점 표기법(dot notation)** | 정확히는 속성 접근 연산자 |
| `(...)` 안의 값들 | **인자(argument)** | 넘겨주는 값 |
| 인자 목록 규격 | **시그니처(signature)** | `AdamW(params, lr, weight_decay)` |
| `import torch.nn as nn`의 `nn` | **별칭(alias)** | 짧게 부르는 이름 |
| `nn.Linear(768, 2)`의 결과물 | **인스턴스(instance) / 객체(object)** | 클래스로 찍어낸 실물 |
| `x.view(...)` | **메서드(method)** | 객체에 붙은 함수 |
| `x.shape` | **속성(attribute)** | 객체에 붙은 값 (괄호 없음) |
| `class GPTDatasetV1(Dataset)`의 `Dataset` | **부모/기반 클래스(base class)** | 상속(inheritance) |
| `p=0.5` | **기본값(default value)** | 안 주면 자동 적용 |

## 1-4. 판별 순서 (4단계)

```
① 점(.)이 아예 없는가?           ← iter(), next(), len(), rearrange()
     예  → 그룹 0 참조. 출처는 셋 중 하나
           · 내장 함수(builtins)      : iter, next, len, print, range
           · 이 파일에서 def로 정의   : create_dataloader_v1, evaluate
           · from X import Y 로 가져옴 : rearrange, DataLoader, MinMaxScaler
     아니오 → ②

② 이름이 대문자로 시작하는가?
     예  → 클래스. 인스턴스를 만들어 변수에 보관한 뒤 나중에 호출
     아니오 → ③

③ 점(.) 앞에 있는 것이 "객체"인가 "모듈"인가?
     객체(x, model, optimizer, scaler) → 메서드
     모듈(torch, F, np, transforms)    → 함수

④ 이름이 밑줄(_)로 끝나는가?
     예 → 제자리 수정 (in-place). 원본이 바뀜
```

---

# 제2부. 공통 핵심 API (5개 분야 전부에서 사용)

## 그룹 0 — 내장 함수 (점 없이 그냥 쓰는 것들)

### 소속이 없는 게 아니라 "생략"된 것

`iter()`, `next()`, `len()`, `print()` 처럼 **점(.) 없이 갑자기 나오는** 이름들은 전부 **`builtins` 모듈** 소속입니다.

```
iter  -> __module__ = 'builtins'      next  -> 'builtins'
len   -> 'builtins'                    print -> 'builtins'
```

파이썬이 시작할 때 `builtins`를 **자동으로 열어두기 때문에** import도, 점도 필요 없습니다.

```python
import builtins
builtins.next(builtins.iter([10, 20, 30]))      # → 10   (같은 동작)
```

### 점(.)이 없는 이름의 출처는 딱 3가지

| 출처 | 예 | 확인 방법 |
|---|---|---|
| ① **내장 함수** (`builtins`) | `iter`, `next`, `len`, `print`, `range`, `sum`, `zip` | `iter.__module__` → `'builtins'` |
| ② **이 파일에서 `def`로 정의** | `create_dataloader_v1`, `train`, `evaluate` | 위쪽 셀에 `def` 있음 |
| ③ **`from X import Y`로 가져옴** | `rearrange`, `DataLoader`, `MinMaxScaler`, `VectorStoreIndex` | 위쪽에 `from ... import` 있음 |

**모르는 이름을 만나면 이 3가지 중 무엇인지만 확인하면 됩니다.**

### 자주 쓰이는 내장 함수

| 함수 | 의미 | 실제 사용 |
|---|---|---|
| `print(...)` | 출력 | 전 분야 |
| `len(obj)` | 길이·개수 | `len(self.input_ids)`, `len(data_loader)` |
| `range(start, end, step)` | 반복용 정수 나열 | `range(0, len(token_ids) - max_length, stride)` |
| `enumerate(iterable)` | **인덱스와 값을 함께** 순회 | `for i, (x, y) in enumerate(loader)` |
| `zip(a, b)` | 두 시퀀스를 짝지어 순회 | 전 분야 |
| `super()` | 부모 클래스 호출 | `super().__init__()` |
| `open(path, mode, encoding)` | 파일 열기 | `open(path, "r", encoding="utf-8")` |
| `int(x)` / `float(x)` | 형 변환 | `int(train_ratio * len(text_data))` |
| `min(a,b)` / `max(iterable)` | 최소·최대 | `min(num_batches, len(loader))` |
| `sum(iterable)` | 합계 | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| `isinstance(obj, cls)` | 타입 판별 | `isinstance(module, torch.nn.Linear)` |
| `getattr` / `setattr` | 이름(문자열)으로 속성 읽기·쓰기 | `setattr(model, name, LinearWithLoRA(...))` |
| **`iter(obj)`** | **순회 장치 만들기** | `data_iter = iter(dataloader)` |
| **`next(it)`** | **다음 값 하나 꺼내기** | `first_batch = next(data_iter)` |
| `sorted(iterable)` | 정렬 | 05.RAG_LLM |

### `iter()` 와 `next()` — 자세히

**`iter(대상)`** 은 책에 끼우는 **책갈피**를 만듭니다. 대상은 그대로 두고, **"어디까지 읽었는지" 기억하는 별도의 물건**을 돌려줍니다.

```python
data = [1, 2, 3]
a, b = iter(data), iter(data)      # 책갈피를 두 개 만들면
next(a)   # → 1
next(b)   # → 1                     ← 각자 위치를 따로 기억
```

**`next(진행표시자)`** 는 하나 꺼내고 한 칸 전진합니다.

```python
it = iter([10, 20, 30])
next(it)         # → 10
next(it)         # → 20
next(it)         # → 30
next(it)         # → StopIteration 예외
next(it, None)   # 기본값을 주면 예외 대신 None
```

#### ★ `for`문은 사실 `iter` + `next` 다

```python
# 우리가 쓰는 형태
for batch in dataloader:
    print(batch)

# 파이썬이 내부에서 실제로 하는 일
it = iter(dataloader)          # ① 책갈피 생성
while True:
    try:
        batch = next(it)       # ② 하나 꺼내기
        print(batch)
    except StopIteration:      # ③ 다 꺼내면 종료
        break
```

`__iter__`를 가진 객체를 **iterable(순회 가능)** 이라고 합니다. `list`, `str`, `dict`, `range`, `DataLoader` 모두 갖고 있습니다.

```
Dataset (__len__, __getitem__)  →  DataLoader (__iter__)  →  for문 / iter+next
```

## 그룹 1 — 텐서 만들기 `torch.` (함수)

| 호출 형태 | 하는 일 | 사용 분야 |
|---|---|---|
| `torch.tensor(data, dtype=None)` | **리스트/숫자 → 텐서** | 전 분야 |
| `torch.zeros(*size)` / `torch.ones(*size)` | 전부 0 / 전부 1. **크기를 낱개 인자로** | 전 분야 |
| `torch.empty(*size)` | 초기화 안 된 빈 텐서 | 01 (LoRA) |
| `torch.arange(end)` / `(start, end, step)` | **연속 정수** `[0,1,2,...]` | 01, 02 |
| `torch.linspace(start, end, steps)` | 구간을 **균등 분할** | 01, 02 |
| `torch.randn(*size)` | 표준정규분포 난수 | 02 (DDPM 노이즈) |
| `torch.randint(low, high, size)` | 정수 난수 | 02 (timestep 샘플링) |
| `torch.full(size, value)` | 특정 값으로 채우기 | 02 |
| `torch.from_numpy(arr)` | NumPy → 텐서 | 04 |

**⚠️ `zeros`와 `tensor`는 역할이 다릅니다**
```python
torch.zeros(2, 3)        # "크기"를 받음 → 2×3 짜리 0 텐서
torch.tensor([[1,2,3]])  # "데이터"를 받음 → 값 그대로
```

## 그룹 2 — 모양 바꾸기

| 호출 형태 | 하는 일 | 사용 분야 |
|---|---|---|
| `x.view(*shape)` | **모양만 재해석**. 원소 수는 그대로 | 01, 02, 03 |
| `x.reshape(*shape)` | `view`와 유사하나 비연속이어도 동작 | 02, 04 |
| `x.transpose(dim0, dim1)` | 두 축의 **자리 교환** | 01, 02 |
| `x.permute(*dims)` | **여러 축을 한 번에** 재배열 | 02 |
| `x.contiguous()` | 메모리 **연속 정리**. transpose 후 view 전 필수 | 01, 02 |
| `x.unsqueeze(dim)` / `x.squeeze(dim)` | 크기 1인 축 **삽입 / 제거** | 전 분야 |
| `x.flatten(start, end)` | 지정 축들을 **하나로 합침** | 01, 02, 03 |
| `x.size()` / `x.shape` | 크기 조회 (메서드 / 속성) | 전 분야 |
| `torch.stack(tensors, dim=0)` | **새 축을 만들며** 쌓기 | 01, 02 |
| `torch.cat(tensors, dim=0)` | **기존 축에** 이어붙이기 | 01, 02 |
| `torch.einsum('...', a, b)` | 아인슈타인 표기 곱 | 02 (ViT) |

**⚠️ `stack`·`cat`은 리스트/튜플을 "하나의 인자"로 받습니다**
```python
torch.cat((idx, idx_next), dim=1)   # ✅ 괄호로 묶기
torch.cat(idx, idx_next, dim=1)     # ❌ 에러
```
`[2,3]` 두 개 → `stack`은 `[2,2,3]`(축 추가), `cat`은 `[4,3]`(축 확장).

## 그룹 3 — 계산 · 집계

| 호출 형태 | 하는 일 | 사용 분야 |
|---|---|---|
| `x.mean(dim, keepdim)` / `x.sum(dim)` | 평균 / 합계 | 전 분야 |
| `x.var(dim, keepdim, unbiased)` | 분산. `unbiased=False`면 n으로 나눔 | 01 (LayerNorm) |
| `x.item()` | 원소 1개 텐서 → **파이썬 숫자** | 전 분야 |
| `x.numel()` | 원소 **총 개수** | 01, 03 |
| `x.tolist()` / `x.numpy()` / `x.cpu()` | 리스트 / NumPy / CPU 이동 | 전 분야 |
| `torch.argmax(input, dim)` | 최댓값의 **위치** | 01, 02, 03 |
| `torch.softmax(input, dim)` | 확률화. **`dim` 필수** | 01, 02 |
| `torch.topk(input, k)` | 상위 k개 값·인덱스 | 03, 05 |
| `torch.kthvalue(input, k)` | **k번째로 작은 값** | **03 (Pruning 임계값)** |
| `torch.triu(input, diagonal=0)` | **상삼각**만 남김 | 01 |
| `torch.gather(input, dim, index)` | index가 가리키는 **값만 추출** | 01 (DPO) |
| `torch.nonzero(input)` | 0이 아닌 **위치 목록** | 01 |
| `x.masked_fill_(mask, value)` | mask가 True인 곳을 value로. **제자리** | 01 |
| `x.clone()` / `x.detach()` | 복사 / 계산그래프 분리 | 전 분야 |
| `x.to(device)` | GPU/CPU 이동 | 전 분야 |
| `torch.sqrt` / `tanh` / `pow` / `abs` | 수학 함수 | 01, 02, 03 |
| `torch.no_grad()` | 기울기 계산 끄기 (with문) | 전 분야 |

## 그룹 4 — 신경망 층 `nn.` (전부 클래스, 대문자)

### 공통

| 클래스 | 하는 일 | 사용 분야 |
|---|---|---|
| `nn.Module` | 모든 모델의 **부모 클래스** | 전 분야 |
| `nn.Linear(in_features, out_features, bias=True)` | **완전연결층** | 전 분야 |
| `nn.Sequential(*modules)` | 층을 **순서대로 자동 연결** | 전 분야 |
| `nn.ModuleList(modules)` | 층을 **리스트로 보관** (자동 실행 X) | 01, 02 |
| `nn.Dropout(p=0.5)` | 확률 p로 0 만들기 | 01, 02 |
| `nn.Parameter(data)` | 텐서를 **학습 대상으로 등록** | 01, 02, 03 |
| `nn.Embedding(num_embeddings, embedding_dim)` | **조회 표**. 정수 ID → 벡터 | 01, 04 |
| `nn.LayerNorm(dim)` | 층 정규화 | 01, 02 |
| `nn.Identity()` | 아무것도 안 함 (층 제거용 자리표시) | 02 |

### 02.Vision 전용 (CNN 계열)

| 클래스 | 하는 일 |
|---|---|
| `nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias)` | **2D 합성곱** |
| `nn.BatchNorm2d(num_features)` | 배치 정규화 (**채널 축** 기준) |
| `nn.GroupNorm(num_groups, num_channels)` | 그룹 정규화 (DDPM/U-Net) |
| `nn.MaxPool2d` / `nn.AvgPool2d` | 풀링 (다운샘플) |
| `nn.Upsample(scale_factor, mode)` | 업샘플 (U-Net 디코더) |
| `nn.ReLU()` / `nn.SiLU()` / `nn.GELU()` | 활성 함수 |

**⚠️ `Embedding`의 두 인자**
```python
nn.Embedding(vocab_size,     emb_dim)   # 토큰용: "단어 종류 수", "벡터 차원"
nn.Embedding(context_length, emb_dim)   # 위치용: "최대 길이",    "벡터 차원"
#            ↑ num_embeddings ↑ embedding_dim
```

**⚠️ `Sequential`에는 별표(`*`)가 필요합니다**
```python
nn.Sequential(*[Block() for _ in range(12)])   # ✅ 리스트를 낱개로 풀어줌
nn.Sequential([Block() for _ in range(12)])    # ❌ "리스트 1개"를 넘긴 셈
```

**`Sequential` vs `ModuleList`**: `Sequential`은 `x = self.blocks(x)` 한 줄로 전부 통과시켜 주지만, `ModuleList`는 **직접 for문을 돌려야** 합니다.

## 그룹 5 — 손실 · 활성 함수

### 클래스형 (`nn.` 대문자) — 미리 만들어두고 재사용

| 클래스 | 용도 | 사용 분야 |
|---|---|---|
| `nn.CrossEntropyLoss()` | 다중 분류 | 01, 02, **03** |
| `nn.MSELoss()` | 회귀 (제곱 오차) | **04**, 03 |
| `nn.BCEWithLogitsLoss()` | 이진 분류 (sigmoid 내장) | 02 |
| `nn.CosineEmbeddingLoss()` | 두 벡터의 방향 유사도 | **03 (Distillation)** |
| `nn.KLDivLoss()` | 분포 간 차이 | 03 (Distillation) |

### 함수형 (`F.` 소문자) — 즉시 계산

| 함수 | 용도 | 사용 분야 |
|---|---|---|
| `F.cross_entropy(input, target, ignore_index=-100)` | 분류 손실. input은 **로짓** | 01 |
| `F.log_softmax(input, dim)` / `F.softmax(input, dim)` | 로그 확률 / 확률 | 01, 02, 03 |
| `F.nll_loss(input, target)` | log_softmax와 짝 | 02 |
| `F.mse_loss` / `F.l1_loss` / `F.smooth_l1_loss` | 회귀 손실들 | 02 |
| `F.logsigmoid(input)` | `log(sigmoid(x))` | 01 (DPO) |
| `F.relu` / `F.silu` / `F.gelu` | 활성 함수 | 02, 03 |
| `F.interpolate(input, size, mode)` | 크기 조절 (업/다운샘플) | 02 |
| `F.pad(input, pad)` | 패딩 | 02, 03 |
| `F.conv2d` / `F.linear` | 합성곱 / 선형 (가중치 직접 전달) | 02, 03 |

**⚠️ `cross_entropy`는 내부에서 softmax를 합니다.** 앞에서 직접 softmax를 걸면 **두 번 적용**되어 학습이 망가집니다. 반드시 **날것의 로짓**을 넣으세요.

**⚠️ `nn.CrossEntropyLoss()` vs `F.cross_entropy()`** — 전자는 클래스라 `criterion = nn.CrossEntropyLoss()`로 만들어두고 `criterion(pred, y)`로 씁니다. 후자는 즉시 호출입니다. **결과는 같습니다.**

## 그룹 6 — 모델 관리 (`nn.Module`의 메서드)

| 호출 형태 | 하는 일 |
|---|---|
| `model.parameters()` | 학습 대상 전부를 넘겨줌 |
| `model.named_parameters()` | (이름, 파라미터) 쌍으로 순회 |
| `model.named_children()` / `named_modules()` | 자식 층 순회 (**LoRA 주입·Pruning**) |
| `model.train()` / `model.eval()` | 학습 / 평가 모드 전환 |
| `model.to(device)` | 모델을 GPU/CPU로 |
| `model.state_dict()` / `load_state_dict(sd)` | 가중치 저장용 딕셔너리 / 주입 |
| `self.register_buffer(name, tensor)` | **학습 안 되지만 저장되는** 값 등록 |
| `loss.backward()` | 역전파 (텐서 메서드) |
| `copy.deepcopy(model)` | **모델 완전 복제** (03에서 원본 보존용) |

## 그룹 7 — 학습 도구

| 호출 형태 | 하는 일 | 사용 분야 |
|---|---|---|
| `torch.optim.Adam(params, lr)` | 가장 일반적인 옵티마이저 | 02, 04 |
| `torch.optim.AdamW(params, lr, weight_decay)` | 가중치 감쇠 분리형 | 01 |
| `torch.optim.SGD(params, lr, momentum)` | 확률적 경사하강 | 02, 03 |
| `optimizer.zero_grad()` / `step()` | 기울기 초기화 / 가중치 갱신 | 전 분야 |
| `torch.optim.lr_scheduler.StepLR(opt, step_size, gamma)` | 계단식 학습률 감소 | 02 |
| `torch.optim.lr_scheduler.LambdaLR(opt, fn)` | 사용자 정의 학습률 | 03 |
| `scheduler.step()` | 학습률 갱신 | 02, 03 |
| `Dataset` / `DataLoader(dataset, batch_size, shuffle, drop_last, collate_fn, num_workers)` | 데이터 파이프라인 | 전 분야 |
| `TensorDataset(X, y)` | 텐서 두 개를 **바로 묶어** 데이터셋화 | **04** |
| `Subset(dataset, indices)` | 데이터셋 일부만 사용 | 03 |
| `nn.init.kaiming_uniform_(tensor, a=0)` | 가중치 초기화 (**제자리**) | 01 (LoRA) |
| `torch.manual_seed(seed)` | 재현성 고정 | 전 분야 |
| `torch.save(obj, path)` / `torch.load(path, weights_only=True)` | 저장 / 로드 | 전 분야 |
| `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | 장치 지정 | 전 분야 |

### 학습 5단계 정형구 (전 분야 공통 — 최빈출)

```python
model.train()                              # ① 학습 모드
for x, y in train_loader:
    optimizer.zero_grad()                  # ② 이전 기울기 삭제
    loss = criterion(model(x), y)          # ③ 예측 → 손실
    loss.backward()                        # ④ 역전파
    optimizer.step()                       # ⑤ 가중치 갱신
```

평가 시에는 `model.eval()` + `with torch.no_grad():`, 끝나면 **반드시 `model.train()` 복귀**.
`eval()`은 *동작 방식*(드롭아웃/BN)을 바꾸고, `no_grad()`는 *기울기 기록*을 끕니다 — **다른 일을 하므로 둘 다 필요**합니다.

---

# 제3부. 분야별 고유 API

## 3-1. 01.Language (LLM)

| 호출 | 형태 | 의미 |
|---|---|---|
| `tiktoken.get_encoding("gpt2")` | 함수 | BPE 토크나이저 로드 (vocab 50257, EOS 50256) |
| `tokenizer.encode(text, allowed_special={...})` | 메서드 | 글자 → 토큰 ID |
| `tokenizer.decode(ids)` | 메서드 | 토큰 ID → 글자 |
| `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")` | 클래스메서드 | HuggingFace 토크나이저 |
| `torch.triu(torch.ones(L,L), diagonal=1)` | 함수 | **인과 마스크** (미래 차단) |
| `attn.masked_fill_(mask, -torch.inf)` | 메서드 | 미래를 −무한대로 (softmax 후 0) |
| `torch.softmax(scores / d**0.5, dim=-1)` | 함수 | **√d_k 스케일링** + 확률화 |
| `keys.view(b,T,H,hd).transpose(1,2)` | 메서드 | 멀티헤드 분할 |
| `ctx.contiguous().view(b,T,d_out)` | 메서드 | 헤드 결합 |
| `logits.flatten(0,1)`, `targets.flatten()` | 메서드 | CE 입력용 평탄화 |
| `logits[:, -1, :]` | 슬라이싱 | 마지막 시점만 (생성·분류) |
| `idx[:, -context_size:]` | 슬라이싱 | 문맥을 **뒤에서** 자르기 |
| `param.requires_grad = False` | 속성 대입 | **파라미터 동결** |
| `setattr(model, name, LinearWithLoRA(...))` | 내장 함수 | LoRA 층 교체 |
| `F.log_softmax` + `torch.gather` | 함수 | DPO 로그확률 추출 |
| `F.logsigmoid(beta * logits)` | 함수 | DPO 손실 |
| `partial(custom_collate_fn, device=device)` | 함수 | collate_fn 인자 고정 |

## 3-2. 02.Vision

### torchvision — 데이터·모델·변환

| 호출 | 형태 | 의미 |
|---|---|---|
| `transforms.Compose([...])` | 클래스 | **변환들을 순서대로 묶음** |
| `transforms.ToTensor()` | 클래스 | PIL/ndarray → 텐서, **0~255 → 0~1** |
| `transforms.Normalize(mean, std)` | 클래스 | 채널별 정규화 |
| `transforms.RandomHorizontalFlip()` | 클래스 | **좌우 반전 (증강)** |
| `transforms.RandomCrop(size, padding)` | 클래스 | 랜덤 잘라내기 (증강) |
| `transforms.Resize(size)` / `CenterCrop(size)` | 클래스 | 크기 조절 / 중앙 자르기 |
| `transforms.Lambda(fn)` | 클래스 | **사용자 정의 변환** |
| `datasets.CIFAR10(root, train, download, transform)` | 클래스 | CIFAR-10 데이터셋 |
| `models.resnet18(weights=None)` | **함수** | ResNet18 생성 (소문자 = 팩토리 함수) |
| `models.resnet50(...)` | 함수 | ResNet50 생성 |
| `save_image(tensor, path, nrow)` | 함수 | 텐서를 이미지 파일로 저장 |

**⚠️ 변환 순서가 중요합니다** — `ToTensor()`가 있어야 그 뒤 `Normalize()`가 텐서에 적용됩니다.
```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # PIL 단계 증강
    transforms.ToTensor(),               # → 텐서 (0~1)
    transforms.Normalize(mean, std),     # → 정규화
])
```

### einops — 축 재배열 (모두 소문자 = 함수)

| 호출 | 의미 |
|---|---|
| `rearrange(x, 'b h n d -> b n (h d)')` | **축 재배열·병합**. `view`+`transpose`를 한 줄로 |
| `repeat(x, 'b n -> b n d', d=64)` | 축 복제 |
| `reduce(x, 'b c h w -> b c', 'mean')` | 축 축약 (평균 등) |
| `Rearrange('...')` | **클래스** — `nn.Sequential` 안에 넣는 층 버전 |

### 기타

| 호출 | 형태 | 의미 |
|---|---|---|
| `summary(model, input_size=...)` | 함수 | **모델 구조·파라미터 수 출력** (`torchinfo`) |
| `TSNE(n_components=2, perplexity=30, ...)` | 클래스 | 고차원 → 2D 시각화 (`sklearn.manifold`) |
| `confusion_matrix(y_true, y_pred)` | 함수 | **혼동 행렬** (`sklearn.metrics`) |
| `Image.open(path)` | 클래스메서드 | 이미지 로드 (`PIL`) |
| `load_dataset(name)` | 함수 | HuggingFace 데이터셋 (`datasets`) |

## 3-3. 03.On-Device_AI

### Pruning (가지치기) — 대부분 **수동 구현**

`torch.nn.utils.prune`을 쓰지 않고 **직접 마스크를 만드는 방식**입니다.

| 호출 | 형태 | 의미 |
|---|---|---|
| `importance = weight.abs()` | 메서드 | **중요도 = 절댓값(magnitude)** |
| `torch.kthvalue(importance.flatten(), k)[0]` | 함수 | **k번째로 작은 값 = 잘라낼 임계값** |
| `mask = importance > threshold` | 연산 | 남길 위치 표시 |
| `weight.mul_(mask)` | 메서드 (제자리) | 마스크 적용 = 실제 가지치기 |
| `self.masks = FineGrainedPruner.prune(model, sparsity_dict)` | 클래스메서드 | 마스크 보관 (재적용용) |
| `profile_macs(model, input)` | 함수 | **연산량(MACs) 측정** (`torchprofile`) |

**핵심 흐름**: 중요도(절댓값) → 임계값(`kthvalue`) → 마스크 → 곱하기

### Quantization (양자화)

| 호출 | 형태 | 의미 |
|---|---|---|
| `torch.quantization.QuantStub()` | 클래스 | **입력을 양자화**하는 자리표시 |
| `torch.quantization.DeQuantStub()` | 클래스 | **출력을 역양자화** |
| `torch.quantization.default_qconfig` | 속성 | PTQ 기본 설정 |
| `torch.quantization.default_qat_qconfig` | 속성 | QAT(양자화 인식 학습) 설정 |
| `torch.quantization.prepare(model, inplace=False)` | 함수 | **관측자(observer) 삽입** |
| `torch.quantization.convert(model)` | 함수 | **실제 int8 모델로 변환** |
| `copy.deepcopy(model)` | 함수 | 원본 보존용 복제 |
| `KMeans(n_clusters=4, mode='euclidean')` | 클래스 | **K-means 양자화** (`fast_pytorch_kmeans`) |

**핵심 흐름**: `QuantStub` 삽입 → `prepare` (관측) → 캘리브레이션 실행 → `convert` (int8)

### Knowledge Distillation (지식 증류)

| 호출 | 형태 | 의미 |
|---|---|---|
| `nn.CrossEntropyLoss()` | 클래스 | **정답(hard label)** 손실 |
| `F.log_softmax(student/T, dim=1)` + `F.softmax(teacher/T, dim=1)` | 함수 | **온도(T) 적용 soft label** |
| `nn.KLDivLoss()` | 클래스 | 교사·학생 분포 차이 |
| `nn.CosineEmbeddingLoss()` | 클래스 | 은닉 표현의 **방향** 일치 |
| `nn.MSELoss()` | 클래스 | 은닉 표현의 **값** 일치 |
| `teacher.eval()` + `with torch.no_grad()` | 메서드 | **교사는 고정** |

**핵심**: 최종 손실 = `α × (정답 손실)` + `(1−α) × (교사 모방 손실)`

### transformers (LLM 압축용)

| 호출 | 형태 | 의미 |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained(name)` | 클래스메서드 | 사전학습 LLM 로드 |
| `AutoTokenizer.from_pretrained(name)` | 클래스메서드 | 짝 토크나이저 |
| `AutoConfig.from_pretrained(name)` | 클래스메서드 | 설정만 로드 |
| `OPTDecoderLayer` / `BloomBlock` / `LlamaDecoderLayer` | 클래스 | 모델별 블록 타입 (교체 대상 판별용) |

## 3-4. 04.Time_Series & RecSys

### sklearn — 전처리·분할·평가

| 호출 | 형태 | 의미 |
|---|---|---|
| `MinMaxScaler()` | 클래스 | **0~1 범위로 정규화**. 학습 데이터의 min/max를 **기억** |
| `scaler.fit_transform(train)` | 메서드 | **학습 데이터**: 기준을 배우고 + 변환 |
| `scaler.transform(test)` | 메서드 | **평가 데이터**: 배운 기준으로 변환만 |
| `scaler.inverse_transform(pred)` | 메서드 | 원래 스케일로 복원 |
| `train_test_split(X, y, test_size, random_state, shuffle)` | 함수 | 학습/평가 분할 |
| `mean_squared_error(y_true, y_pred)` | 함수 | MSE |
| `root_mean_squared_error(y_true, y_pred)` | 함수 | **RMSE** (원래 단위로 해석 가능) |
| `mean_absolute_percentage_error(y_true, y_pred)` | 함수 | MAPE (백분율 오차) |

**⚠️ 시계열에서 가장 자주 나오는 함정**
```python
scaler.fit_transform(train)   # ✅ 학습 데이터에만 fit
scaler.transform(test)        # ✅ 평가 데이터는 transform만
scaler.fit_transform(test)    # ❌ 평가 데이터로 다시 fit하면 정보 누출(data leakage)
```

### PyTorch 데이터 구성

| 호출 | 형태 | 의미 |
|---|---|---|
| `TensorDataset(X_tensor, y_tensor)` | 클래스 | 텐서 두 개를 **바로 묶어** 데이터셋화 |
| `DataLoader(dataset, batch_size, shuffle=False)` | 클래스 | 시계열은 보통 **shuffle=False** |
| `nn.LSTM` / `nn.GRU` / `nn.RNN` | 클래스 | 순환 신경망 |
| `nn.Embedding(num_users, dim)` | 클래스 | **RecSys: 사용자/아이템 임베딩** |

### torch_geometric (그래프 추천)

| 호출 | 형태 | 의미 |
|---|---|---|
| `download_url(url, path)` | 함수 | 데이터 다운로드 |
| `extract_zip(path, folder)` | 함수 | 압축 해제 |

### pandas 유틸

| 호출 | 의미 |
|---|---|
| `df.nunique()` | 고유값 개수 (사용자·아이템 수 파악) |
| `df.values` | DataFrame → NumPy 배열 |
| `pd.read_csv(path)` | CSV 로드 |

## 3-5. 05.RAG_LLM

### llama_index 핵심 흐름

```
문서 로드 → 청킹 → 인덱스 생성 → 검색기/질의엔진 → 질의
Reader   → Splitter → VectorStoreIndex → as_retriever()/as_query_engine() → query()
```

| 호출 | 형태 | 의미 |
|---|---|---|
| `SimpleDirectoryReader(path).load_data()` | 클래스 → 메서드 | **폴더의 문서들을 읽어** Document 리스트로 |
| `WikipediaReader().load_data(pages=[...])` | 클래스 → 메서드 | 위키피디아 문서 로드 |
| `Document(text=...)` | 클래스 | 문서 한 건 |
| `SentenceSplitter(chunk_size, chunk_overlap)` | 클래스 | **문장 단위 청킹** |
| `VectorStoreIndex.from_documents(docs)` | 클래스메서드 | **문서 → 벡터 인덱스** 생성 |
| `index.as_retriever()` | 메서드 | **검색기** 생성 (문서 조각만 반환) |
| `index.as_query_engine()` | 메서드 | **질의 엔진** 생성 (LLM 답변까지) |
| `query_engine.query("질문")` | 메서드 | 질의 실행 |
| `retriever.retrieve("질문")` | 메서드 | 관련 청크 검색 |
| `node.get_content()` | 메서드 | 검색된 청크의 본문 |
| `Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")` | 속성 대입 | **전역 임베딩 모델 설정** |
| `Settings.llm = OpenAI(model=...)` | 속성 대입 | 전역 LLM 설정 |
| `PromptTemplate("...")` | 클래스 | 프롬프트 틀 |
| `get_response_synthesizer()` | 함수 | 응답 합성기 |

### 커스텀 확장 (상속용 부모 클래스)

| 클래스 | 상속해서 무엇을 만드나 |
|---|---|
| `BaseRetriever` | 나만의 검색 로직 |
| `BaseSynthesizer` | 나만의 응답 합성 |
| `CustomQueryEngine` | 나만의 질의 엔진 |

### MCP · Agent

| 호출 | 형태 | 의미 |
|---|---|---|
| `BasicMCPClient(url)` | 클래스 | **MCP 서버 연결** |
| `McpToolSpec(client)` | 클래스 | MCP 도구 목록을 LlamaIndex 도구로 |
| `FunctionAgent(tools=..., llm=...)` | 클래스 | 도구를 쓰는 에이전트 |
| `ToolCall` / `ToolCallResult` | 클래스 | 도구 호출 이벤트 |

### 전처리 유틸

| 호출 | 형태 | 의미 |
|---|---|---|
| `OpenAI(api_key=...)` | 클래스 | OpenAI 클라이언트 (`openai` 패키지) |
| `client.chat.completions.create(model, messages)` | 메서드 | LLM 호출 |
| `BeautifulSoup(html, 'html.parser')` | 클래스 | HTML 파싱 (`bs4`) |
| `text_to_sentences_and_offsets(text)` | 함수 | **문장 분리 + 위치** (`blingfire`) |
| `json.loads(s)` / `json.dumps(obj)` | 함수 | JSON 파싱 / 직렬화 |
| `bz2.open(path)` | 함수 | 압축 파일 읽기 |

---

# 제4부. 평가 Metric (출제 포인트 명시 항목)

## 분야별 지표

| 분야 | 지표 | 코드 |
|---|---|---|
| **01.Language** | 평균 Loss | `total_loss / num_batches` |
| | Accuracy | `(argmax(logits,-1) == target).sum().item() / num_examples` |
| | DPO Margin | `chosen_reward - rejected_reward` |
| **02.Vision** | Accuracy | `(preds == labels).sum().item() / total` |
| | **혼동 행렬** | `confusion_matrix(y_true, y_pred)` (`sklearn.metrics`) |
| | 시각화 | `TSNE(n_components=2).fit_transform(features)` |
| **03.On-Device** | 정확도 유지율 | 압축 전후 accuracy 비교 |
| | **모델 크기** | `sum(p.numel() for p in model.parameters())` |
| | **연산량** | `profile_macs(model, input)` |
| | **희소도(sparsity)** | `(weight == 0).sum() / weight.numel()` |
| **04.Time_Series** | **RMSE** | `root_mean_squared_error(y_true, y_pred)` |
| | MSE | `mean_squared_error(y_true, y_pred)` |
| | MAPE | `mean_absolute_percentage_error(y_true, y_pred)` |
| **05.RAG_LLM** | 검색 품질 / 답변 정확도 | LLM 기반 평가, MCP 프레임워크 |

## Loss와 Metric의 차이 (서술형 대비)

| | Loss | Metric |
|---|---|---|
| 목적 | **학습에 사용** | 사람이 성능 판단 |
| 미분 가능? | **필수** | 불필요 |
| 예 | `CrossEntropyLoss`, `MSELoss` | Accuracy, RMSE, 혼동 행렬 |

Accuracy는 미분이 안 되기 때문에 **직접 학습에 쓸 수 없습니다.** 그래서 학습은 Loss로 하고, 판단은 Metric으로 합니다.

## RMSE를 쓰는 이유 (04)

MSE는 제곱이라 **단위가 원래 값의 제곱**이 됩니다. 제곱근을 씌운 RMSE는 **원래 단위로 해석**할 수 있어 "평균 몇 원 틀렸다"처럼 말할 수 있습니다.

## 스케일 복원 후 평가 (04의 함정)

```python
pred_scaled = model(x)                          # 0~1 범위 예측
pred = scaler.inverse_transform(pred_scaled)    # ← 원래 단위로 되돌린 뒤
rmse = root_mean_squared_error(y_true, pred)    #   평가해야 의미 있는 값
```

---

# 제5부. 백지에서 쓸 때 자주 틀리는 것들

| # | 함정 | 올바른 형태 |
|---|---|---|
| 1 | `cat`/`stack`에 인자를 나열 | `torch.cat((a, b), dim=1)` — **튜플로 묶기** |
| 2 | `Sequential`에 별표 누락 | `nn.Sequential(*[...])` |
| 3 | `transpose` 축 번호 | 3차원 → `(1,2)`, 4차원 → **`(2,3)`** |
| 4 | `transpose` 후 바로 `view` | 사이에 **`.contiguous()`** |
| 5 | `softmax`에 `dim` 생략 | `dim`은 **기본값 없음** — 반드시 명시 |
| 6 | `cross_entropy` 앞에 softmax | **로짓 그대로** 넣기 |
| 7 | `arange`에 device 누락 | `torch.arange(n, device=x.device)` |
| 8 | `_` 유무 혼동 | `masked_fill_`(원본 수정) ↔ `masked_fill`(새 텐서) |
| 9 | **`ToTensor()` 위치** | `Normalize()` **앞**에 와야 함 |
| 10 | **테스트에 `fit_transform`** | 테스트는 **`transform`만** (정보 누출) |
| 11 | **교사 모델 학습됨** | `teacher.eval()` + `no_grad()` |
| 12 | **역정규화 없이 RMSE** | `inverse_transform` 후 평가 |

## 백지 구현 사고 절차 — "주어 → 소속 → 인자"

### 예시 1 — "어텐션 점수를 확률로 바꾸고 싶다"
```
① 새 텐서를 만드는 계산이다        → torch. 소속
② 확률화 = softmax                 → 소문자 = 함수
③ softmax(input, dim) 형식         → dim 필수
결론: torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

### 예시 2 — "합성곱 층을 만들고 싶다"
```
① 가중치를 기억해야 한다           → 클래스 → nn. 대문자
② 2D 합성곱 = Conv2d
③ Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias)
결론: nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
```

### 예시 3 — "학습 데이터를 0~1로 정규화하고 싶다"
```
① min/max를 기억해야 한다          → 클래스 → 대문자
② sklearn.preprocessing.MinMaxScaler
③ 학습은 fit_transform, 평가는 transform
결론: scaler = MinMaxScaler(); train_scaled = scaler.fit_transform(train)
```

### 예시 4 — "문서로 벡터 인덱스를 만들고 싶다"
```
① 인덱스를 보관해야 한다           → 클래스 → 대문자
② VectorStoreIndex
③ 문서 리스트에서 만드는 클래스메서드 = from_documents
결론: index = VectorStoreIndex.from_documents(documents)
```

---

# 부록 — 모를 때 찾는 법 : `help()`

> **결론부터**: 모든 환경·모든 대상에서 통하는 방법은 **`help()` 하나뿐**입니다.
> 다른 방법들(`F12`, `Shift+Tab`, `inspect.signature`)은 전부 조건부로만 동작합니다.

## 1. 왜 `help()` 뿐인가 — 측정 결과

torch API 45개에 `inspect.signature()`를 적용해본 결과입니다.

| 그룹 | 성공 | 실패 |
|---|---|---|
| `torch.*` 함수 | 3 | **13** |
| **텐서 메서드** (`x.view`, `x.mean` …) | **0** | **15** ← 전멸 |
| `nn.*` 클래스 | 6 | 0 |
| `F.*` 함수 | 3 | 1 |
| 기타 (`AdamW`, `DataLoader`) | 4 | 0 |
| **총계** | 16 | **29 (실패율 64%)** |

| `inspect.signature` 성공 | 실패 |
|---|---|
| `nn.Linear`, `nn.Conv2d`, `DataLoader`, `MinMaxScaler` | `torch.tensor`, `torch.cat`, `torch.softmax` … |
| — | `x.view`, `x.mean`, `x.transpose` … **전부** |
| **파이썬으로 구현된 것** | **C++로 구현된 것** |

가장 자주 쓰는 **텐서 메서드는 하나도 되지 않습니다.**
반면 `help()`는 C++ 구현도 **docstring 첫 줄에서 시그니처를 읽어와** 100% 보여줍니다.

## 2. 기본 사용법

```python
help(대상)
```

`import` 조차 필요 없는 **내장 함수**입니다. 스크립트·REPL·주피터·SSH 서버 어디서나 동일하게 동작합니다.

### 클래스에 쓸 때 — 맨 위 4번째 줄만 보면 된다

```python
help(DataLoader)
```
```
Help on class DataLoader in module torch.utils.data.dataloader:

class DataLoader(typing.Generic)
 |  DataLoader(dataset: 'Dataset[_T_co]', batch_size: 'int | None' = 1, shuffle: ...)
 |             ↑ 첫 번째 인자 = dataset
```

`self`는 표시되지 않으므로 **보이는 첫 이름이 곧 첫 번째 인자**입니다.

### 함수·메서드에 쓸 때

```python
help(torch.cat)              # cat(tensors, dim=0, *, out=None) -> Tensor
help(nn.Conv2d)              # Conv2d(in_channels, out_channels, kernel_size, ...)
help(MinMaxScaler)           # MinMaxScaler(feature_range=(0,1), *, copy=True, clip=False)
help(rearrange)              # rearrange(tensor, pattern, **axes_lengths)
```

### 텐서 메서드에 쓸 때 — 인스턴스를 통해 부른다

```python
x = torch.zeros(2, 2)
help(x.view)         # view(*shape) -> Tensor
help(x.mean)         # mean(dim=None, keepdim=False, *, dtype=None) -> Tensor
help(x.transpose)    # transpose(dim0, dim1) -> Tensor
help(x.masked_fill_) # masked_fill_(mask, value)
```

## 3. 실전 — 빈칸 채우기에 적용하는 절차

문제:
```python
dataloader = DataLoader(????, batch_size=batch_size, shuffle=shuffle,
                        drop_last=drop_last, num_workers=num_workers)
```

| 단계 | 할 일 |
|---|---|
| ① | 새 셀에 `help(DataLoader)` 실행 |
| ② | 맨 위 시그니처에서 **인자 이름을 순서대로** 읽는다 → `dataset, batch_size, shuffle, sampler, …` |
| ③ | 문제에 **이미 이름으로 주어진 것**을 지운다 → `batch_size`, `shuffle`, `drop_last`, `num_workers` 제외 |
| ④ | 남은 것 중 **첫 번째**가 답 → `dataset` |
| ⑤ | 그 이름의 **변수가 위쪽 코드에 있는지** 확인 → `dataset = GPTDatasetV1(...)` |

⑤단계가 중요합니다. 시그니처의 인자 **이름**과 실제로 넣을 **변수 이름**이 항상 같지는 않습니다.

## 4. 출력이 길 때 — 첫 줄만 보는 헬퍼

노트북 맨 위에 한 번만 정의해두면 한 줄로 끝납니다.

```python
import inspect

def sig(obj):
    """파이썬 구현이든 C++ 구현이든 시그니처 한 줄만 돌려준다."""
    try:
        return f"{obj.__name__}{inspect.signature(obj)}"
    except (ValueError, TypeError):
        return (inspect.getdoc(obj) or '').split('\n')[0]
```

```python
sig(torch.cat)     # 'cat(tensors, dim=0, *, out=None) -> Tensor'
sig(x.view)        # 'view(*shape) -> Tensor'
sig(nn.Conv2d)     # 'Conv2d(in_channels, out_channels, kernel_size, ...)'
```

## 5. 주피터를 쓴다면 `?` 가 더 편하다

```python
torch.cat?
x.view?
```

`help()`와 같은 내용이 **별도 패널**에 떠서 본문 스크롤을 건드리지 않습니다.
단, **IPython·주피터 전용**이라 일반 `python` 스크립트에서는 `SyntaxError`가 납니다.

## 6. 커버리지 비교

| 방법 | 커버리지 | 동작 환경 |
|---|---|---|
| **`help(obj)`** | **100%** | **모든 환경** |
| `sig(obj)` 헬퍼 | 100% | 모든 환경 (사전 정의 필요) |
| `obj?` | 100% | IPython·주피터 전용 |
| `inspect.signature(obj)` 단독 | **36%** | **텐서 메서드 전부 실패 — 단독 사용 금물** |
| `F12` / `Shift+Tab` | — | VSCode 등 에디터 전용 |

---

# 부록 B — 시그니처는 어디에 정의되어 있나 (C/C++ 출신을 위한 설명)

## B-1. 파이썬에는 헤더 파일이 없다

C/C++는 **선언(.h)** 과 **정의(.c/.cpp)** 가 분리되어 있습니다. 파이썬은 **정의가 곧 선언**입니다.

## B-2. 실제 예 — `DataLoader`의 첫 인자

```
파일     : ...\site-packages\torch\utils\data\dataloader.py
클래스   : 142줄부터
__init__ : 248줄부터
```

```python
248 |     def __init__(
249 |         self,                                    ← 파이썬이 자동으로 채움
250 |         dataset: Dataset[_T_co],                 ← ★ 사용자 기준 첫 번째 인자
251 |         batch_size: int | None = 1,
252 |         shuffle: bool | None = None,
...
263 |         *,                                       ← 이 뒤는 키워드 전용
264 |         prefetch_factor: int | None = None,
```

## B-3. `self`는 왜 세지 않나

```python
dataloader = DataLoader(dataset, batch_size=8)
#            └ 클래스 호출     →  __init__(self=새객체, dataset=dataset, batch_size=8)
```

## B-4. `*` 표시 — C에는 없는 문법

`*` **앞**은 위치로도 줄 수 있지만, **뒤**는 `prefetch_factor=2`처럼 키워드로만 가능합니다.

## B-5. 위치 인자 vs 키워드 인자 — 교재에 둘 다 나온다

```python
# Chapter 2, 4, 7 — 위치로 전달
DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Chapter 6 — 이름을 명시
DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

**둘 다 정답이고 결과도 같습니다.** 다만 시험에서는 **교재가 쓴 형태 그대로** 답하는 것이 안전합니다.

| | 위치 인자 | 키워드 인자 |
|---|---|---|
| 순서 | **시그니처 순서를 지켜야 함** | 순서 무관 |
| 제약 | `*` 뒤에는 못 씀 | 없음 |

## B-6. C/C++ 헤더에 가장 가까운 것 — `.pyi` 스텁

```python
inspect.getsourcefile(torch.tensor)
# → TypeError : C++ 구현이라 파이썬 소스 없음
```

```
torch\_C\_VariableFunctions.pyi   ← 30,876번째 줄에 def tensor(...)
torch\nn\functional.pyi   (28 KB)
```

| | C/C++ 헤더(.h) | 파이썬 스텁(.pyi) |
|---|---|---|
| 필수인가 | **필수** | **선택** |
| 용도 | 컴파일러 타입 검사 | **에디터 자동완성** |
| 강제력 | 어기면 컴파일 에러 | 어겨도 **실행은 됨** |

## B-7. ⚠️ 타입 힌트는 강제되지 않는다

```python
batch_size: int | None = 1        # "int 또는 None"이라 적혀 있지만 검사하지 않음
```

C에서 컴파일러가 잡아주던 것을 파이썬은 **런타임까지 미룹니다.**

## B-8. `F12`(정의로 이동)는 C의 "헤더로 점프"에 대응한다

> 시그니처 확인 방법 자체는 「부록 — 모를 때 찾는 법」을 보세요.

### `F12`는 "파이썬 기능"이 아니라 "에디터 기능"

| 환경 | `F12` / `Ctrl+클릭` |
|---|---|
| **VSCode의 `.ipynb`** | ✅ (Pylance) |
| VSCode의 `.py` | ✅ |
| **브라우저 JupyterLab** | ❌ (브라우저 개발자도구가 열림) |

### 정적 분석의 한계

```python
dataloader = create_dataloader_v1(raw_text, batch_size=1, ...)
```
`F12`는 코드를 **읽어서 추론**할 뿐입니다. C였다면 컴파일러가 타입을 확정해주지만, 파이썬은 **실행 전까지 타입이 정해지지 않습니다.** 이때는 셀을 실행한 뒤 `type(dataloader)` 나 `help(dataloader)` 를 쓰세요.

### 전제 조건

`Ctrl+Shift+P` → `Python: Select Interpreter` 로 커널과 동일한 환경을 지정해야 합니다.

---

# 최종 요약 — 6줄

1. **대문자로 시작하면 클래스, 소문자면 함수.** 패키지가 아니라 **이름**이 결정한다 (`torch.nn` 검사 결과 예외 0건).
2. **메서드는 객체에 소속된 함수.** 타입은 함수와 동일하며, 자기 객체를 첫 인자로 자동으로 받는다.
3. **기억할 상태가 있으면 클래스**(`nn.Dropout`, `MinMaxScaler`), **없으면 함수**(`F.dropout`, `rearrange`).
4. **점(.)이 없는 이름은 소속이 없는 게 아니라 생략된 것.** 내장 함수(`builtins`) · 이 파일의 `def` · `from X import Y` 셋 중 하나다.
5. **시그니처를 확인하는 유일한 범용 수단은 `help()`.** `inspect.signature`는 텐서 메서드에 전부 실패한다(커버리지 36%).
6. **5개 분야의 공통 뼈대는 동일하다** — `Dataset → DataLoader → 모델 → 손실 → zero_grad/backward/step`. 분야마다 다른 것은 **전처리·층 종류·평가 지표**뿐이다.

---

# 부록 C — 사용 API 전수 목록 (패키지별)

> 시험 범위 **29개 노트북 전체**에서 실제로 사용된 함수·클래스를 중복 없이 모은 목록입니다.
> 총 **308개** 항목이며, 소속 패키지별로 묶었습니다.
> 서브모듈(`torch.amp` 등)·자료형(`torch.float32` 등)·상수(`torch.pi` 등)는 함수·클래스가 아니므로 제외했습니다.

---

## 1. `torch.` — 텐서 생성 · 전역 연산 (57개)

### 1-1. 텐서 만들기

| API | 의미 |
|---|---|
| `torch.tensor(data)` | 리스트·숫자를 텐서로 변환 (데이터를 복사) |
| `torch.as_tensor(data)` | 텐서로 변환하되 가능하면 복사하지 않음 (메모리 절약) |
| `torch.from_numpy(arr)` | NumPy 배열 → 텐서 (메모리 공유) |
| `torch.zeros(*size)` | 전부 0인 텐서 |
| `torch.ones(*size)` | 전부 1인 텐서 |
| `torch.full(size, value)` | 지정한 값으로 채운 텐서 |
| `torch.empty(*size)` | 초기화하지 않은 빈 텐서 (곧 값을 채울 때) |
| `torch.arange(start, end, step)` | 연속 정수 나열 `[0,1,2,...]` |
| `torch.linspace(start, end, steps)` | 구간을 균등 분할한 값들 |
| `torch.randn(*size)` | 표준정규분포 난수 (평균 0, 분산 1) |
| `torch.randint(low, high, size)` | 정수 난수 |
| `torch.randperm(n)` | 0~n-1을 무작위로 섞은 순열 |
| `torch.zeros_like(x)` | x와 **같은 shape**의 0 텐서 |
| `torch.ones_like(x)` | x와 같은 shape의 1 텐서 |
| `torch.randn_like(x)` | x와 같은 shape의 난수 텐서 |
| `torch.diag(x)` | 대각 행렬 생성 / 대각 원소 추출 |
| `torch.Generator()` | 난수 생성기 객체 (시드를 따로 관리할 때) |

### 1-2. 모양 바꾸기 · 합치기

| API | 의미 |
|---|---|
| `torch.cat(tensors, dim)` | **기존 축**에 이어붙이기 (축 개수 그대로) |
| `torch.stack(tensors, dim)` | **새 축**을 만들며 쌓기 (축 개수 +1) |
| `torch.flatten(x, start, end)` | 지정 축들을 하나로 합침 |

### 1-3. 수학 · 집계

| API | 의미 |
|---|---|
| `torch.sum(x, dim)` | 합계 |
| `torch.mean(x, dim)` | 평균 |
| `torch.var(x, dim)` | 분산 |
| `torch.max(x)` | 최댓값 (값과 인덱스) |
| `torch.abs(x)` | 절댓값 — **Pruning의 중요도 계산에 사용** |
| `torch.sqrt(x)` | 제곱근 |
| `torch.pow(x, n)` | 거듭제곱 |
| `torch.exp(x)` | 지수 함수 |
| `torch.cos(x)` | 코사인 — 위치 인코딩·스케줄러에 사용 |
| `torch.tanh(x)` | 하이퍼볼릭 탄젠트 — GELU 근사식 |
| `torch.sigmoid(x)` | 시그모이드 (0~1로 압축) |
| `torch.sign(x)` | 부호만 추출 (-1, 0, +1) |
| `torch.round(x)` | 반올림 — **양자화에서 정수화** |
| `torch.clamp(x, min, max)` | 값을 범위 안으로 자르기 |
| `torch.clip(x, min, max)` | `clamp`의 별칭 |
| `torch.norm(x)` | 노름(크기) 계산 |
| `torch.cumprod(x, dim)` | 누적 곱 — **DDPM의 `alpha_bar` 계산** |
| `torch.matmul(a, b)` | 행렬 곱 (`@` 연산자와 동일) |
| `torch.einsum('...', a, b)` | 아인슈타인 표기법 곱 — 복잡한 축 연산을 한 줄로 |
| `torch.isnan(x)` | NaN 여부 판별 |

### 1-4. 선택 · 인덱싱

| API | 의미 |
|---|---|
| `torch.argmax(x, dim)` | 최댓값의 **위치(인덱스)** — 예측 클래스 선택 |
| `torch.argsort(x)` | 정렬했을 때의 인덱스 순서 |
| `torch.topk(x, k)` | 상위 k개의 값과 인덱스 |
| `torch.kthvalue(x, k)` | **k번째로 작은 값** — Pruning 임계값 계산 |
| `torch.gather(x, dim, index)` | index가 가리키는 위치의 값만 추출 — DPO 로그확률 |
| `torch.nonzero(x)` | 0이 아닌 원소의 위치 목록 |
| `torch.triu(x, diagonal)` | 상삼각 부분만 남김 — **인과 마스크** |
| `torch.softmax(x, dim)` | 점수를 합이 1인 확률로 변환 |

### 1-5. 저장 · 실행 환경

| API | 의미 |
|---|---|
| `torch.save(obj, path)` | 객체(주로 `state_dict`)를 파일로 저장 |
| `torch.load(path)` | 저장된 파일 불러오기 |
| `torch.device("cuda"/"cpu")` | 연산 장치 지정 |
| `torch.manual_seed(n)` | 난수 시드 고정 (재현성) |
| `torch.set_default_device(d)` | 이후 생성되는 텐서의 기본 장치 지정 |
| `torch.get_default_dtype()` | 기본 실수 자료형 조회 |
| `torch.no_grad()` | **기울기 계산 끄기** (with문) — 평가·추론 시 필수 |
| `torch.inference_mode()` | `no_grad`보다 더 최적화된 추론 전용 모드 |
| `torch.set_grad_enabled(bool)` | 기울기 계산을 조건부로 켜고 끄기 |

---

## 2. `torch.nn` (`nn.`) — 층 · 손실 클래스 (28개)

> 전부 **대문자로 시작 = 클래스**입니다. 인스턴스를 만들어두고 나중에 호출합니다.

### 2-1. 기본 구조

| API | 의미 |
|---|---|
| `nn.Module` | **모든 층·모델의 부모 클래스**. 상속해서 직접 정의 |
| `nn.Sequential(*layers)` | 층을 순서대로 자동 연결 |
| `nn.ModuleList([...])` | 층을 리스트로 보관 (자동 실행 X, for문 필요) |
| `nn.Parameter(tensor)` | 텐서를 **학습 대상**으로 등록 |
| `nn.Identity()` | 아무것도 하지 않음 — 층을 제거할 때 자리표시 |

### 2-2. 완전연결 · 임베딩

| API | 의미 |
|---|---|
| `nn.Linear(in_features, out_features, bias)` | 완전연결층. **마지막 축만** 바꿈 |
| `nn.Embedding(num_embeddings, embedding_dim)` | 정수 ID → 벡터 **조회 표(lookup table)** |

### 2-3. 합성곱 · 풀링 (Vision)

| API | 의미 |
|---|---|
| `nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)` | 2D 합성곱 — 이미지 특징 추출 |
| `nn.Conv1d(...)` | 1D 합성곱 — **시계열**에 사용 |
| `nn.ConvTranspose2d(...)` | 전치 합성곱 — 크기를 키움(업샘플), U-Net 디코더 |
| `nn.MaxPool2d(kernel_size)` | 최댓값 풀링 (다운샘플) |
| `nn.AvgPool2d(kernel_size)` | 평균 풀링 |
| `nn.Upsample(scale_factor, mode)` | 단순 확대 (학습 파라미터 없음) |

### 2-4. 정규화

| API | 의미 |
|---|---|
| `nn.LayerNorm(dim)` | **한 토큰의 feature끼리** 정규화 — Transformer |
| `nn.BatchNorm2d(num_features)` | **배치 방향**으로 정규화 — CNN |
| `nn.GroupNorm(num_groups, num_channels)` | 채널을 그룹으로 묶어 정규화 — DDPM/U-Net |

### 2-5. 활성 함수 · 드롭아웃

| API | 의미 |
|---|---|
| `nn.ReLU()` | 음수를 0으로 자름 |
| `nn.LeakyReLU(slope)` | 음수도 약간 통과시킴 |
| `nn.GELU()` | 부드러운 ReLU — GPT 계열 표준 |
| `nn.SiLU()` | Swish. `x * sigmoid(x)` — DDPM/U-Net |
| `nn.Softmax(dim)` | 확률화 (클래스 버전) |
| `nn.Dropout(p)` | 확률 p로 원소를 0으로 (과적합 방지) |

### 2-6. 순환 신경망 (시계열)

| API | 의미 |
|---|---|
| `nn.RNN(input_size, hidden_size)` | 기본 순환 신경망 |
| `nn.LSTM(input_size, hidden_size)` | 장기 기억이 가능한 순환 신경망 |

### 2-7. 손실 함수 (클래스형)

| API | 의미 |
|---|---|
| `nn.CrossEntropyLoss()` | 다중 분류 손실. **내부에 softmax 포함** → 로짓을 그대로 넣을 것 |
| `nn.MSELoss()` | 평균 제곱 오차 — 회귀 |
| `nn.BCEWithLogitsLoss()` | 이진 분류. sigmoid 내장 |
| `nn.CosineEmbeddingLoss()` | 두 벡터의 **방향** 유사도 — Knowledge Distillation |

---

## 3. `torch.nn.functional` (`F.`) — 함수형 (16개)

> 전부 **소문자 = 함수**. 학습 파라미터가 없어 즉시 호출합니다.

| API | 의미 |
|---|---|
| `F.relu(x)` | ReLU (함수 버전) |
| `F.softmax(x, dim)` | 확률화 |
| `F.log_softmax(x, dim)` | softmax 후 로그 — 수치적으로 안정 |
| `F.cross_entropy(input, target)` | 분류 손실 (로짓 입력) |
| `F.nll_loss(input, target)` | 음의 로그 우도. `log_softmax`와 짝으로 사용 |
| `F.mse_loss(a, b)` | 평균 제곱 오차 |
| `F.l1_loss(a, b)` | 절대값 오차 |
| `F.smooth_l1_loss(a, b)` | L1과 L2를 절충 — 객체 검출(DETR) |
| `F.logsigmoid(x)` | `log(sigmoid(x))` — **DPO 손실의 핵심** |
| `F.interpolate(x, size, mode)` | 크기 조절 (업/다운샘플) |
| `F.pad(x, pad)` | 패딩 추가 |
| `F.conv2d(x, weight)` | 합성곱 (가중치를 직접 넘김) — 양자화 실습 |
| `F.linear(x, weight)` | 선형 변환 (가중치를 직접 넘김) |
| `F.avg_pool1d(x, kernel)` | 1D 평균 풀링 |
| `F.embedding(input, weight)` | 임베딩 조회 (`nn.Embedding`의 내부 구현) |

---

## 4. `torch.nn.init` — 가중치 초기화 (2개)

> 이름이 **밑줄로 끝남 = 제자리 수정(in-place)**. 원본 텐서가 직접 바뀝니다.

| API | 의미 |
|---|---|
| `nn.init.kaiming_uniform_(tensor, a)` | ReLU 계열에 맞춘 초기화 — **LoRA의 A 행렬** |
| `nn.init.xavier_uniform_(tensor)` | 입출력 크기를 함께 고려한 초기화 |

---

## 5. `torch.optim` — 옵티마이저 · 학습률 스케줄러 (7개)

| API | 의미 |
|---|---|
| `optim.SGD(params, lr, momentum)` | 확률적 경사하강법 — Vision·On-Device |
| `optim.Adam(params, lr)` | 가장 널리 쓰이는 적응형 옵티마이저 |
| `optim.AdamW(params, lr, weight_decay)` | 가중치 감쇠를 분리한 Adam — **LLM 표준** |
| `optim.Optimizer` | 모든 옵티마이저의 부모 클래스 |
| `lr_scheduler.StepLR(opt, step_size, gamma)` | 일정 주기마다 학습률을 계단식으로 감소 |
| `lr_scheduler.CosineAnnealingLR(opt, T_max)` | 코사인 곡선을 따라 학습률 감소 |
| `lr_scheduler.LambdaLR(opt, fn)` | 사용자 정의 함수로 학습률 제어 |

---

## 6. `torch.utils.data` — 데이터 파이프라인 (4개)

| API | 의미 |
|---|---|
| `Dataset` | 데이터셋의 **부모 클래스**. `__len__`·`__getitem__`을 정의해 상속 |
| `DataLoader(dataset, batch_size, shuffle, drop_last, collate_fn)` | 배치 묶기·셔플·병렬 로딩 |
| `TensorDataset(X, y)` | 텐서 두 개를 바로 묶어 데이터셋화 — **시계열** |
| `Subset(dataset, indices)` | 데이터셋의 일부만 잘라 사용 |

---

## 7. `torch.quantization` — 양자화 (6개)

| API | 의미 |
|---|---|
| `QuantStub()` | 입력을 양자화하는 **자리표시 층** |
| `DeQuantStub()` | 출력을 다시 실수로 되돌리는 층 |
| `default_qconfig` | PTQ(사후 양자화) 기본 설정 |
| `default_qat_qconfig` | QAT(양자화 인식 학습) 설정 |
| `prepare(model)` | 관측자(observer)를 삽입해 값 분포를 수집할 준비 |
| `convert(model)` | 수집된 정보로 **실제 int8 모델로 변환** |

> **흐름**: `QuantStub` 삽입 → `prepare` → 캘리브레이션 실행 → `convert`

---

## 8. `torchvision` — 이미지 데이터 · 모델 (12개)

### 8-1. `transforms` — 이미지 변환 (전부 클래스)

| API | 의미 |
|---|---|
| `Compose([...])` | 변환들을 **순서대로** 묶음 |
| `ToTensor()` | PIL/ndarray → 텐서, **0~255를 0~1로** 변환 |
| `ToPILImage()` | 텐서 → PIL 이미지 (역변환) |
| `Normalize(mean, std)` | 채널별 정규화 — **`ToTensor()` 다음에** 위치 |
| `Resize(size)` | 크기 조절 |
| `CenterCrop(size)` | 중앙을 잘라냄 |
| `RandomCrop(size, padding)` | 무작위 위치로 잘라냄 — **데이터 증강** |
| `RandomHorizontalFlip()` | 무작위 좌우 반전 — **데이터 증강** |
| `Lambda(fn)` | 사용자 정의 변환 |

### 8-2. `datasets` · `models` · `utils`

| API | 의미 |
|---|---|
| `datasets.CIFAR10(root, train, download, transform)` | CIFAR-10 데이터셋 |
| `models.resnet18(weights)` | ResNet18 생성 — **소문자 = 팩토리 함수** |
| `models.resnet50(weights)` | ResNet50 생성 |
| `utils.save_image(tensor, path, nrow)` | 텐서를 이미지 파일로 저장 |

---

## 9. torch 생태계 부가 라이브러리 (8개)

| API | 소속 | 의미 |
|---|---|---|
| `summary(model, input_size)` | `torchinfo` | 층별 출력 shape·파라미터 수를 표로 출력 |
| `profile_macs(model, input)` | `torchprofile` | **연산량(MACs) 측정** — On-Device 경량화 평가 |
| `KMeans(n_clusters, mode)` | `fast_pytorch_kmeans` | K-means 군집 — **가중치 클러스터 양자화** |
| `rearrange(x, 'b h n d -> b n (h d)')` | `einops` | 축 재배열·병합. `view`+`transpose`를 한 줄로 |
| `repeat(x, pattern)` | `einops` | 축 복제 |
| `reduce(x, pattern, 'mean')` | `einops` | 축 축약 (평균·합 등) |
| `Rearrange(pattern)` | `einops.layers.torch` | `nn.Sequential` 안에 넣는 **층 버전** |
| `download_url(url, path)` / `extract_zip(path, folder)` | `torch_geometric.data` | 그래프 데이터 다운로드·압축 해제 |

---

## 10. `transformers` — 사전학습 모델 (8개)

| API | 의미 |
|---|---|
| `AutoTokenizer.from_pretrained(name)` | 모델에 맞는 토크나이저를 이름만으로 로드 |
| `AutoModelForCausalLM.from_pretrained(name)` | 사전학습 LLM 로드 |
| `AutoConfig.from_pretrained(name)` | 가중치 없이 **설정만** 로드 |
| `OPTDecoderLayer` / `OPTAttention` | OPT 모델의 블록 — 양자화·Pruning 대상 판별용 |
| `BloomBlock` | Bloom 모델의 블록 |
| `LlamaDecoderLayer` / `LlamaRMSNorm` | Llama 모델의 블록·정규화 |

---

## 11. `sklearn` — 전처리 · 분할 · 평가 (5개)

| API | 소속 | 의미 |
|---|---|---|
| `MinMaxScaler()` | `preprocessing` | 0~1 정규화. 학습 데이터의 min/max를 **기억**하는 클래스 |
| `train_test_split(X, y, test_size, random_state)` | `model_selection` | 학습/평가 데이터 분할 |
| `mean_squared_error(y_true, y_pred)` | `metrics` | MSE — 회귀 평가 지표 |
| `confusion_matrix(y_true, y_pred)` | `metrics` | **혼동 행렬** — 분류 평가 |
| `TSNE(n_components, perplexity)` | `manifold` | 고차원 특징을 2D로 시각화 |

> **`scaler` 사용 규칙**: 학습 데이터는 `fit_transform()`, 평가 데이터는 **`transform()`만**. 평가 데이터로 다시 `fit`하면 정보 누출(data leakage)입니다.

---

## 12. `llama_index` — RAG 프레임워크 (18개)

### 12-1. 문서 로드 · 청킹

| API | 의미 |
|---|---|
| `SimpleDirectoryReader(path).load_data()` | 폴더의 문서들을 읽어 Document 리스트로 |
| `WikipediaReader().load_data(pages)` | 위키피디아 문서 로드 |
| `Document(text=...)` | 문서 한 건을 나타내는 객체 |
| `SentenceSplitter(chunk_size, chunk_overlap)` | 문장 단위로 **청킹**(잘게 나누기) |

### 12-2. 인덱스 · 검색 · 질의

| API | 의미 |
|---|---|
| `VectorStoreIndex.from_documents(docs)` | 문서 → **벡터 인덱스** 생성 |
| `index.as_retriever()` | **검색기** 생성 (관련 청크만 반환) |
| `index.as_query_engine()` | **질의 엔진** 생성 (LLM 답변까지 생성) |
| `get_response_synthesizer()` | 검색 결과를 답변으로 합성하는 객체 |
| `PromptTemplate("...")` | 프롬프트 틀 정의 |
| `Settings` | **전역 설정** (`Settings.llm`, `Settings.embed_model`) |

### 12-3. 모델 연결

| API | 의미 |
|---|---|
| `OpenAI(model=...)` (`llms.openai`) | LlamaIndex용 LLM 래퍼 |
| `OpenAIEmbedding(model=...)` | 임베딩 모델 (문서를 벡터로) |

### 12-4. 커스텀 확장 (상속용 부모 클래스)

| API | 의미 |
|---|---|
| `BaseRetriever` | 나만의 검색 로직을 만들 때 상속 |
| `BaseSynthesizer` | 나만의 응답 합성기를 만들 때 상속 |
| `CustomQueryEngine` | 나만의 질의 엔진을 만들 때 상속 |

### 12-5. MCP · Agent

| API | 의미 |
|---|---|
| `BasicMCPClient(url)` | **MCP 서버**에 연결하는 클라이언트 |
| `McpToolSpec(client)` | MCP 도구 목록을 LlamaIndex 도구로 변환 |
| `FunctionAgent(tools, llm)` | 도구를 사용하는 에이전트 |
| `ToolCall` / `ToolCallResult` | 도구 호출 이벤트·결과 객체 |
| `Context` | 에이전트 실행 문맥 |

---

## 13. `openai` (1개)

| API | 의미 |
|---|---|
| `OpenAI(api_key=...)` | OpenAI API 클라이언트. `client.chat.completions.create(...)`로 호출 |

---

## 14. `numpy` (`np.`) — 수치 배열 (21개)

| API | 의미 |
|---|---|
| `np.array(data)` / `np.asarray(data)` | 배열 생성 (후자는 복사 최소화) |
| `np.zeros(shape)` / `np.zeros_like(x)` | 0 배열 |
| `np.arange(start, end, step)` | 연속 정수 |
| `np.linspace(start, end, num)` | 균등 분할 값 |
| `np.ogrid[...]` | 좌표 격자 생성 (이미지 마스크 등) |
| `np.concatenate(arrays, axis)` | 배열 이어붙이기 |
| `np.expand_dims(x, axis)` | 축 추가 (`unsqueeze`에 대응) |
| `np.roll(x, shift)` | 원소를 회전 이동 |
| `np.where(cond, a, b)` | 조건에 따라 선택 |
| `np.clip(x, min, max)` | 값을 범위로 자르기 |
| `np.mean(x)` / `np.sqrt(x)` / `np.abs(x)` | 평균 / 제곱근 / 절댓값 |
| `np.dot(a, b)` | 내적 |
| `np.linalg.norm(x)` | 노름(벡터 크기) |
| `np.random.*` | 난수 생성 |
| `np.float32` / `np.uint8` / `np.ndarray` | 자료형 · 배열 타입 |

---

## 15. `pandas` (`pd.`) — 표 데이터 (4개)

| API | 의미 |
|---|---|
| `pd.read_csv(path, sep, header, names)` | CSV/TSV 파일 로드 |
| `pd.concat([df1, df2])` | DataFrame 이어붙이기 |
| `pd.date_range(start, periods)` | 날짜 범위 생성 — 시계열 |
| `pd.DateOffset(...)` | 날짜 간격 객체 |

> 자주 쓰이는 **메서드**: `df.sample(frac=1)`(셔플), `df.reset_index(drop=True)`, `df.map({...})`(라벨 인코딩), `df.iloc[i]`, `df.nunique()`, `df.to_csv()`, `df.values`

---

## 16. `matplotlib` (`plt.`) — 시각화 (28개)

| 분류 | API |
|---|---|
| 그림 만들기 | `plt.figure()`, `plt.subplots()`, `plt.subplot()`, `plt.gca()` |
| 그리기 | `plt.plot()`, `plt.scatter()`, `plt.bar()`, `plt.imshow()`(이미지·히트맵) |
| 도형 | `plt.Circle()`, `plt.Rectangle()` |
| 꾸미기 | `plt.title()`, `plt.suptitle()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`, `plt.text()`, `plt.colorbar()`, `plt.axis()` |
| 눈금 | `plt.xticks()`, `plt.yticks()` |
| 색상 | `plt.get_cmap()`, `ListedColormap`, `Normalize` |
| 출력·저장 | `plt.show()`, `plt.savefig()`, `plt.close()`, `plt.tight_layout()`, `plt.ioff()`, `plt.rcParams` |
| 애니메이션 | `PillowWriter` |

> 축 관련 **메서드**: `ax.twiny()`(보조 x축), `ax.set_xlabel()`, `ax.imshow()`

---

## 17. `tiktoken` — BPE 토크나이저 (2개)

| API | 의미 |
|---|---|
| `tiktoken.get_encoding("gpt2")` | 인코딩 이름으로 토크나이저 로드 |
| `tiktoken.encoding_for_model(name)` | **모델 이름**으로 토크나이저 로드 |

> 주요 메서드: `tokenizer.encode(text)`, `tokenizer.decode(ids)`

---

## 18. 파이썬 표준 라이브러리

### 18-1. `os` · `pathlib` — 파일·환경

| API | 의미 |
|---|---|
| `os.path.exists/join/dirname` | 경로 확인·조합 |
| `os.makedirs(path, exist_ok=True)` | 폴더 생성 |
| `os.rename(a, b)` | 이름 변경 |
| `os.environ` | 환경 변수 (API 키 등) |
| `os.getcwd()` | 현재 작업 폴더 |
| `os.cpu_count()` | CPU 코어 수 (`num_workers` 설정용) |
| `os.system(cmd)` | 셸 명령 실행 |
| `Path(...)` (`pathlib`) | 객체 지향 경로 처리 |

### 18-2. `json` — 직렬화

| API | 의미 |
|---|---|
| `json.load(file)` / `json.loads(str)` | JSON 읽기 (파일 / 문자열) |
| `json.dump(obj, file)` / `json.dumps(obj)` | JSON 쓰기 |
| `json.JSONDecoder` / `JSONDecodeError` | 파서 객체 / 파싱 오류 |

### 18-3. `collections` — 자료구조

| API | 의미 |
|---|---|
| `defaultdict(type)` | 없는 키에 기본값을 자동 생성하는 딕셔너리 |
| `OrderedDict()` | 순서를 보장하는 딕셔너리 |
| `Counter(iterable)` | 원소 개수 세기 |
| `namedtuple(name, fields)` | 이름으로 접근 가능한 튜플 |

### 18-4. 기타 표준 라이브러리

| API | 소속 | 의미 |
|---|---|---|
| `partial(fn, **kwargs)` | `functools` | 인자를 미리 고정한 새 함수 — **`collate_fn`에 필수** |
| `deepcopy(obj)` | `copy` | 완전 복제 — **모델 원본 보존** |
| `dataclass` | `dataclasses` | 설정 클래스를 간결하게 정의 |
| `math.sqrt/log/log2/ceil` | `math` | 수학 함수 |
| `random.seed/choice/sample` | `random` | 파이썬 난수 |
| `re.sub(pattern, repl, s)` | `re` | 정규식 치환 |
| `date` / `timedelta` | `datetime` | 날짜·기간 |
| `parser` | `dateutil` | 유연한 날짜 문자열 파싱 |
| `version(pkg)` | `importlib.metadata` | 패키지 버전 조회 |
| `Any` / `Dict` / `List` / `Tuple` / `Union` | `typing` | 타입 힌트 (**실행에 영향 없음**) |

---

## 19. 기타 외부 라이브러리

| API | 소속 | 의미 |
|---|---|---|
| `tqdm(iterable)` | `tqdm` | **진행률 표시 바**. `tqdm.auto`/`tqdm.notebook`도 동일 |
| `Image.open(path)` | `PIL` | 이미지 파일 로드 |
| `load_dataset(name)` | `datasets` (HuggingFace) | 공개 데이터셋 로드 |
| `StableDiffusionPipeline` | `diffusers` | Stable Diffusion 전체 파이프라인 |
| `BeautifulSoup(html, parser)` | `bs4` | HTML 파싱 — RAG 전처리 |
| `text_to_sentences_and_offsets(text)` | `blingfire` | 문장 분리 + 위치 정보 |
| `display` / `clear_output` / `Image` | `IPython.display` | 노트북 출력 제어 |

---

## 20. 교재 자체 모듈 (표준 라이브러리 아님)

> 노트북과 같은 폴더의 `.py` 파일에서 가져오는 것들입니다. 시험에서는 **이미 정의되어 있다고 가정**합니다.

| API | 소속 | 의미 |
|---|---|---|
| `GPTModel` | `previous_chapters` | 앞 챕터에서 만든 GPT 모델 클래스 |
| `create_dataloader_v1` | `previous_chapters` | 데이터로더 생성 헬퍼 |
| `generate_text_simple` / `generate_and_print_sample` | `previous_chapters` | 텍스트 생성 |
| `load_gpt2_model` / `download_model` | `previous_chapters` | 사전학습 가중치 로드 |
| `plot_losses` | `previous_chapters` | 학습 곡선 시각화 |
| `Reader` / `LlamaIndexRetriever` / `KGQueryEngine` | `import_function` | RAG 실습용 커스텀 클래스 |

---

## 요약 — 어느 그룹을 먼저 볼 것인가

| 우선순위 | 그룹 | 이유 |
|---|---|---|
| ★★★ | 1(`torch.`), 2(`nn.`), 3(`F.`), 5(`optim`), 6(`utils.data`) | **5개 분야 전부**에서 공통으로 쓰임 |
| ★★ | 8(`torchvision`), 7(`quantization`), 11(`sklearn`), 12(`llama_index`) | 각 분야의 **핵심 차별점** |
| ★ | 14~19 | 보조 도구. 이름과 역할 정도만 |
