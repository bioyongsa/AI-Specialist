# PyTorch API 레퍼런스 — 백지 구현을 위한 정리

> 시험에서 **빈칸이 아니라 한 줄 전체**를 써야 하는 상황을 대비한 문서입니다.
> "이 함수가 어느 패키지 소속인지"를 외우는 대신 **규칙으로 추론**하는 것이 목표입니다.
>
> 모든 시그니처는 이 프로젝트에 설치된 PyTorch에서 직접 추출했습니다.

---

# 제1부. 개념 정리

## 1-1. 함수(function)와 메서드(method)는 같은 것인가?

**메서드는 함수의 한 종류입니다.** 정확히는 — **객체에 소속된 함수를 메서드라고 부릅니다.**

파이썬에게 직접 물어보면 이렇게 답합니다:

```python
type(torch.tensor)   # builtin_function_or_method
type(x.view)         # builtin_function_or_method   ← 똑같음!
```

둘의 **타입이 완전히 동일**합니다. 즉 본질적으로 같은 것이고, **어디에 소속되어 있느냐**만 다릅니다.

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

이 `self` 자리를 자동으로 채워주는 것이 메서드의 정체입니다. 그래서 `x.view(2,3)`은 **"x라는 텐서에게 모양을 바꾸라고 시키는 것"** 이고, `torch.tensor(data)`는 **"torch에게 텐서를 만들어달라고 하는 것"** 입니다.

| | 함수 | 메서드 |
|---|---|---|
| 소속 | 모듈/패키지 | 객체(인스턴스) |
| 형태 | `모듈.이름(...)` | `객체.이름(...)` |
| 첫 인자 | 직접 다 넘김 | **객체 자신이 자동으로 들어감** |
| 예시 | `torch.tensor(lst)` | `x.view(2,3)` |

> **정리**: "객체에 붙어 있을 때만 메서드"라는 이해가 정확합니다.

---

## 1-2. 어떤 패키지 뒤에 클래스가 오고, 어떤 패키지 뒤에 함수가 오는가?

### 결론: **패키지가 결정하지 않습니다. 이름의 대소문자가 결정합니다.**

가장 확실한 증거 — **같은 패키지 안에 둘 다 있습니다**:

```python
torch.tensor   ->  function   (소문자)
torch.Tensor   ->  class      (대문자)
```

이름이 딱 한 글자 다른데 성격이 완전히 다릅니다. 그러니 "torch 패키지는 함수만 있다" 같은 규칙은 존재할 수 없습니다.

### 실제로 규칙이 지켜지는지 검사해봤습니다

`torch.nn` 안의 모든 이름을 조사한 결과:

| 이름 형태 | 클래스 | 클래스 아님 |
|---|---|---|
| **대문자로 시작** | **166개** | 0개 |
| 소문자로 시작 | **0개** | 14개 |

**예외가 단 하나도 없습니다.** 이것은 파이썬의 공식 코딩 규약(PEP 8)이고, PyTorch도 예외 없이 따릅니다.

```
대문자로 시작  →  클래스 (Linear, Embedding, Dropout, Sequential, AdamW, DataLoader)
소문자로 시작  →  함수   (tensor, zeros, arange, softmax, cross_entropy)
```

### 그렇다면 패키지는 무엇을 알려주는가 — "경향"

패키지가 성격을 **결정**하지는 않지만, 어떤 종류가 모여 있는지의 **경향**은 있습니다:

| 패키지 | 주로 들어있는 것 | 대표 예 |
|---|---|---|
| `torch` (최상위) | **함수** — 텐서 만들기·전역 연산 | `torch.tensor`, `torch.zeros`, `torch.cat` |
| `torch.nn` | **클래스** — 가중치를 가진 층 | `nn.Linear`, `nn.Embedding` |
| `torch.nn.functional` (`F`) | **함수 전용** — 가중치 없는 계산 | `F.cross_entropy`, `F.log_softmax` |
| `torch.nn.init` | **함수** — 가중치 초기화 (`_`로 끝남) | `nn.init.kaiming_uniform_` |
| `torch.optim` | **클래스** — 옵티마이저 | `optim.AdamW` |
| `torch.utils.data` | **클래스** — 데이터 도구 | `Dataset`, `DataLoader` |

> 검사 결과 `torch.nn.functional` 안에도 클래스가 5개 있긴 합니다. 즉 **패키지만 믿으면 안 되고, 대소문자를 봐야 확실합니다.**

### 더 근본적인 질문 — 왜 어떤 건 클래스로 만들었나?

**기억할 것(상태)이 있으면 클래스, 없으면 함수입니다.**

```python
self.dropout = nn.Dropout(0.1)   # 클래스: train/eval 상태를 기억해야 함
x = self.dropout(x)              #         만들어두고 나중에 사용

x = F.dropout(x, 0.1)            # 함수: 기억할 것 없이 즉시 계산하고 끝
```

`nn.Linear`가 클래스인 이유도 같습니다 — **학습되는 가중치를 품고 있어야** 하니까요. 반면 `F.cross_entropy`는 입력을 받아 숫자 하나 뱉으면 끝이라 기억할 것이 없습니다.

### 판별 순서 (3단계)

```
① 이름이 대문자로 시작하는가?
     예  → 클래스. 인스턴스를 만들어 변수에 보관한 뒤 나중에 호출
     아니오 → ②

② 점(.) 앞에 있는 것이 "객체"인가 "모듈"인가?
     객체(x, model, optimizer) → 메서드
     모듈(torch, F, np)        → 함수

③ 이름이 밑줄(_)로 끝나는가?
     예 → 제자리 수정 (in-place). 원본이 바뀜
```

---

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

---

# 제2부. 그룹별 API 레퍼런스

> 시그니처의 `*` 뒤 인자는 **반드시 키워드로** 줘야 합니다.

## 그룹 1 — 텐서 만들기 `torch.` (함수)

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `torch.tensor(data, dtype=None)` | **리스트/숫자 → 텐서**. 데이터를 그대로 옮김 | `torch.tensor(input_chunk)` |
| `torch.zeros(*size)` | 전부 0. **크기를 낱개 인자로** | `torch.zeros(emb_dim)`, `torch.zeros(rank, out_dim)` |
| `torch.ones(*size)` | 전부 1 | `torch.ones(context_length, context_length)` |
| `torch.empty(*size)` | 초기화 안 된 빈 텐서 (곧 채울 때) | `torch.empty(in_dim, rank)` |
| `torch.arange(end)` / `(start, end, step)` | **연속 정수** `[0,1,2,...]` | `torch.arange(seq_len, device=in_idx.device)` |
| `torch.linspace(start, end, steps)` | 구간을 **균등 분할** | `torch.linspace(0, num_epochs, len(train_losses))` |
| `torch.randn(*size)` | 표준정규분포 난수 | `torch.randn(2048, 2048)` |

**⚠️ `zeros`와 `tensor`는 역할이 다릅니다**
```python
torch.zeros(2, 3)        # "크기"를 받음 → 2×3 짜리 0 텐서
torch.tensor([[1,2,3]])  # "데이터"를 받음 → 값 그대로
```

## 그룹 2 — 모양 바꾸기 (메서드 + torch 함수)

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `x.view(*shape)` | **모양만 재해석**. 원소 수는 그대로 | `keys.view(b, num_tokens, num_heads, head_dim)` |
| `x.transpose(dim0, dim1)` | 두 축의 **자리 교환** | `keys.transpose(1, 2)` / `keys.transpose(2, 3)` |
| `x.contiguous()` | 메모리 **연속 정리**. transpose 후 view 전 필수 | `context_vec.contiguous().view(...)` |
| `x.unsqueeze(dim)` | 크기 1인 축 **삽입** | `torch.tensor(encoded).unsqueeze(0)` |
| `x.squeeze(dim)` | 크기 1인 축 **제거** | `token_ids.squeeze(0)` |
| `x.flatten(start, end)` | 지정 축들을 **하나로 합침** | `logits.flatten(0, 1)` |
| `torch.stack(tensors, dim=0)` | **새 축을 만들며** 쌓기 | `torch.stack(inputs_lst)` |
| `torch.cat(tensors, dim=0)` | **기존 축에** 이어붙이기 | `torch.cat((idx, idx_next), dim=1)` |

**⚠️ `stack`·`cat`은 리스트/튜플을 "하나의 인자"로 받습니다**
```python
torch.cat((idx, idx_next), dim=1)   # ✅ 괄호로 묶기
torch.cat(idx, idx_next, dim=1)     # ❌ 에러
```

**`stack` vs `cat`** — `[2,3]` 두 개를 넣으면:
- `stack` → `[2,2,3]` (축이 하나 늘어남)
- `cat` → `[4,3]` (기존 축이 길어짐)

## 그룹 3 — 계산 · 집계

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `x.mean(dim, keepdim=False)` | 평균 | `x.mean(dim=-1, keepdim=True)` |
| `x.var(dim, keepdim, unbiased)` | 분산. `unbiased=False`면 n으로 나눔 | `x.var(dim=-1, keepdim=True, unbiased=False)` |
| `x.sum(dim)` | 합계 | `mask.sum(-1)`, `(pred == target).sum()` |
| `x.item()` | 원소 1개 텐서 → **파이썬 숫자** | `loss.item()` |
| `x.numel()` | 원소 **총 개수** | `input_batch.numel()` |
| `x.tolist()` | 텐서 → 파이썬 리스트 | `out.squeeze(0).tolist()` |
| `torch.argmax(input, dim, keepdim)` | 최댓값의 **위치** | `torch.argmax(logits, dim=-1, keepdim=True)` |
| `torch.softmax(input, dim)` | 확률화. **`dim` 필수** | `torch.softmax(scores / d**0.5, dim=-1)` |
| `torch.triu(input, diagonal=0)` | **상삼각**만 남김 | `torch.triu(torch.ones(L,L), diagonal=1)` |
| `torch.gather(input, dim, index)` | index가 가리키는 **값만 추출** | `torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1))` |
| `torch.nonzero(input)` | 0이 아닌 **위치 목록** | `torch.nonzero(mask).squeeze()` |
| `x.masked_fill_(mask, value)` | mask가 True인 곳을 value로. **제자리** | `attn_scores.masked_fill_(mask_bool, -torch.inf)` |
| `x.clone()` | 복사본 | `labels[:, 1:].clone()` |
| `x.detach()` | 계산그래프에서 분리 | `(chosen - ref).detach()` |
| `x.to(device)` | GPU/CPU 이동 | `input_batch.to(device)` |
| `x.bool()` | 0/1 → True/False | `self.mask.bool()` |
| `torch.sqrt` / `tanh` / `pow` | 제곱근 / 하이퍼볼릭탄젠트 / 거듭제곱 | GELU 수식에서 사용 |

## 그룹 4 — 신경망 층 `nn.` (전부 클래스, 대문자)

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `nn.Linear(in_features, out_features, bias=True)` | **완전연결층**. 가중치 학습됨 | `nn.Linear(d_in, d_out, bias=qkv_bias)` |
| `nn.Embedding(num_embeddings, embedding_dim)` | **조회 표**. 정수 ID → 벡터 | `nn.Embedding(vocab_size, emb_dim)` |
| `nn.Dropout(p=0.5)` | 확률 p로 0 만들기 | `nn.Dropout(cfg["drop_rate"])` |
| `nn.Parameter(data, requires_grad=True)` | 텐서를 **학습 대상으로 등록** | `nn.Parameter(torch.ones(emb_dim))` |
| `nn.Sequential(*modules)` | 층을 **순서대로 자동 연결** | `nn.Sequential(*[TransformerBlock(cfg) for _ in range(n)])` |
| `nn.ModuleList(modules)` | 층을 **리스트로 보관** (자동 실행 X) | `nn.ModuleList([CausalAttention(...) for _ in range(h)])` |
| `nn.Module` | 모든 모델의 **부모 클래스** | `class GPTModel(nn.Module):` |

**⚠️ `Embedding`의 두 인자**
```python
nn.Embedding(vocab_size,     emb_dim)   # 토큰용: "단어 종류 수", "벡터 차원"
nn.Embedding(context_length, emb_dim)   # 위치용: "최대 길이",    "벡터 차원"
#            ↑ num_embeddings ↑ embedding_dim
```
첫 인자는 **표의 행 수**, 둘째는 **각 행의 길이**입니다.

**⚠️ `Sequential`에는 별표(`*`)가 필요합니다**
```python
nn.Sequential(*[Block() for _ in range(12)])   # ✅ 리스트를 낱개로 풀어줌
nn.Sequential([Block() for _ in range(12)])    # ❌ "리스트 1개"를 넘긴 셈
```

**`Sequential` vs `ModuleList`**: `Sequential`은 `x = self.trf_blocks(x)` 한 줄로 전부 통과시켜 주지만, `ModuleList`는 **직접 for문을 돌려야** 합니다. 그래서 Ch3 Wrapper는 `torch.cat([head(x) for head in self.heads], dim=-1)`처럼 손으로 순회합니다.

## 그룹 5 — 손실 · 활성 함수 `F.` (전부 함수, 소문자)

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `F.cross_entropy(input, target, ignore_index=-100)` | 분류 손실. input은 **로짓**, target은 **정수 클래스** | `F.cross_entropy(logits.flatten(0,1), target.flatten())` |
| `F.log_softmax(input, dim)` | softmax 후 로그. 수치적으로 안정 | `F.log_softmax(logits, dim=-1)` |
| `F.logsigmoid(input)` | `log(sigmoid(x))` | `-F.logsigmoid(beta * logits)` |
| `F.softmax(input, dim)` | 확률화 (`torch.softmax`와 동일) | — |

**⚠️ `cross_entropy`는 내부에서 softmax를 합니다.** 앞에서 직접 softmax를 걸면 **두 번 적용**되어 학습이 망가집니다. 반드시 **날것의 로짓**을 넣으세요.

## 그룹 6 — 모델 관리 (`nn.Module`의 메서드)

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `model.parameters()` | 학습 대상 전부를 넘겨줌 | `AdamW(model.parameters(), lr=...)` |
| `model.train()` / `model.eval()` | 학습 / 평가 모드 전환 | `model.train()`, `model.eval()` |
| `model.to(device)` | 모델을 GPU/CPU로 | `model.to(device)` |
| `model.state_dict()` | 가중치를 딕셔너리로 | `torch.save(model.state_dict(), path)` |
| `model.load_state_dict(sd)` | 가중치 불러 넣기 | `model.load_state_dict(torch.load(path, weights_only=True))` |
| `self.register_buffer(name, tensor)` | **학습 안 되지만 저장되는** 값 등록 | `self.register_buffer("mask", torch.triu(...))` |
| `model.named_children()` | 자식 층을 (이름, 객체)로 순회 | `for name, module in model.named_children():` |
| `loss.backward()` | 역전파 (텐서 메서드) | `loss.backward()` |

## 그룹 7 — 학습 도구

| 호출 형태 | 하는 일 | 교재 실제 사용 |
|---|---|---|
| `torch.optim.AdamW(params, lr, weight_decay)` | 옵티마이저 (**클래스**) | `torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)` |
| `optimizer.zero_grad()` | 기울기 초기화 | `optimizer.zero_grad()` |
| `optimizer.step()` | 가중치 갱신 | `optimizer.step()` |
| `Dataset` | 상속받아 쓰는 **부모 클래스** | `class GPTDatasetV1(Dataset):` |
| `DataLoader(dataset, batch_size=1, shuffle=None, drop_last=False, collate_fn=None, num_workers=0)` | 배치로 묶기 (**클래스**) | `DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)` |
| `nn.init.kaiming_uniform_(tensor, a=0)` | 가중치 초기화 (**제자리**) | `nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))` |
| `torch.no_grad()` | 기울기 계산 끄기 (with문) | `with torch.no_grad():` |
| `torch.manual_seed(seed)` | 재현성 고정 | `torch.manual_seed(123)` |
| `torch.save(obj, path)` | 저장 | `torch.save(model.state_dict(), "outputs/model.pth")` |
| `torch.load(path, weights_only=True)` | 로드 | `torch.load(path, map_location="cpu", weights_only=True)` |
| `torch.device("cuda"/"cpu")` | 장치 지정 | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` |

---

# 제3부. 챕터별 실제 사용 API

> 각 노트북의 코드 셀을 스캔해 실제로 등장한 것만 정리했습니다.
> **형태** 열 — `클래스` / `함수` / `메서드` / `속성`

## Chapter 1 — Vector Space (gensim)

PyTorch를 쓰지 않고 `gensim`만 사용합니다.

| 호출 | 형태 | 의미 |
|---|---|---|
| `api.load("glove-wiki-gigaword-50")` | 함수 | 사전학습된 단어 벡터 모델 다운로드·로드 |
| `model.similarity(word1, word2)` | 메서드 | 두 단어의 **코사인 유사도** (−1~1) |
| `model.most_similar(word, topn=5)` | 메서드 | 벡터 공간에서 **가장 가까운 단어** N개 |
| `model.doesnt_match([w1, w2, ...])` | 메서드 | 그룹에서 **의미가 이질적인 단어** 찾기 |

## Chapter 2 — Dataset & Embedding

| 호출 | 형태 | 의미 |
|---|---|---|
| `torch.device("cuda")` | 함수 | 연산 장치 객체 생성 |
| `torch.cuda.is_available()` | 함수 | GPU 사용 가능 여부 |
| `torch.set_default_device(DEVICE)` | 함수 | 이후 모든 텐서를 자동으로 해당 장치에 생성 |
| `torch.manual_seed(123)` | 함수 | 난수 시드 고정 (재현성) |
| `torch.get_default_dtype()` | 함수 | 기본 실수 타입 확인 |
| `torch.randn(2048, 2048)` | 함수 | 정규분포 난수 텐서 (속도 측정용) |
| `torch.cuda.empty_cache()` | 함수 | GPU 메모리 캐시 반환 |
| `torch.cuda.synchronize()` | 함수 | GPU 연산 완료 대기 (정확한 시간 측정) |
| `tiktoken.get_encoding("gpt2")` | 함수 | BPE 토크나이저 로드 |
| `tokenizer.encode(text, allowed_special={...})` | 메서드 | 글자 → 토큰 ID 리스트 |
| `tokenizer.decode(ids)` | 메서드 | 토큰 ID → 글자 |
| `Dataset` | 클래스 | 상속받아 커스텀 데이터셋 정의 |
| `DataLoader(dataset, batch_size, shuffle, drop_last, num_workers)` | 클래스 | 배치 묶기·셔플 |
| `torch.tensor(input_chunk)` | 함수 | 리스트 → 텐서 |
| `list.append(x)` | 메서드 | 파이썬 리스트에 추가 |
| `torch.nn.Embedding(vocab_size, output_dim)` | 클래스 | 토큰 임베딩 테이블 |
| `torch.arange(max_length)` | 함수 | 위치 번호 `[0,1,2,3]` 생성 |
| `iter(dataloader)` / `next(data_iter)` | 함수(내장) | 배치 하나만 꺼내 확인 |
| `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")` | 메서드(클래스메서드) | HuggingFace 토크나이저 로드 |

## Chapter 3 — Attention

| 호출 | 형태 | 의미 |
|---|---|---|
| `nn.Module` | 클래스 | 모든 층의 부모 |
| `nn.Linear(d_in, d_out, bias=qkv_bias)` | 클래스 | Q/K/V 투영층 |
| `nn.Dropout(dropout)` | 클래스 | 어텐션 가중치 드롭아웃 |
| `nn.ModuleList([...])` | 클래스 | 여러 헤드를 파라미터로 등록 |
| `self.register_buffer('mask', ...)` | 메서드 | 학습 안 되지만 저장되는 마스크 등록 |
| `torch.ones(L, L)` | 함수 | 전부 1인 정사각 표 |
| `torch.triu(t, diagonal=1)` | 함수 | 대각선 **위쪽만** 남김 = 미래 위치 |
| `self.mask.bool()` | 메서드 | 0/1 → True/False (masked_fill용) |
| `attn_scores.masked_fill_(mask, -torch.inf)` | 메서드 | 미래 위치를 −무한대로 (**제자리**) |
| `torch.inf` | 속성 | 무한대 상수 |
| `torch.softmax(x, dim=-1)` | 함수 | 점수 → 확률 |
| `x.transpose(1, 2)` | 메서드 | 3차원에서 축 교환 |
| `x.transpose(2, 3)` | 메서드 | **4차원**에서 축 교환 |
| `x.view(b, T, H, hd)` | 메서드 | 헤드 분할 |
| `x.contiguous()` | 메서드 | 메모리 정리 (view 전 필수) |
| `torch.cat([...], dim=-1)` | 함수 | 헤드 결과 이어붙이기 (Wrapper) |
| `torch.stack((inputs, inputs), dim=0)` | 함수 | 배치 차원 생성 |
| `@` | 연산자 | 행렬 곱 (`torch.matmul`과 동일) |

## Chapter 3 (Viz) — Attention 시각화

| 호출 | 형태 | 의미 |
|---|---|---|
| `block.att.register_forward_hook(fn)` | 메서드 | **순전파 중간값 가로채기** (hook 등록) |
| `hook.remove()` | 메서드 | hook 해제 (안 하면 계속 쌓임) |
| `model.eval()` | 메서드 | 평가 모드 |
| `with torch.no_grad():` | 함수(컨텍스트) | 기울기 계산 끄기 |
| `attn_weights.detach()` | 메서드 | 계산그래프 분리 (저장용) |
| `tensor.numpy()` | 메서드 | 텐서 → NumPy (matplotlib용) |
| `x.masked_fill(mask, v)` | 메서드 | 밑줄 **없음** → 새 텐서 반환 |
| `plt.subplots(2, n)` | 함수 | 격자 그래프 생성 |
| `axes[i].imshow(attn, cmap="Blues")` | 메서드 | 히트맵 그리기 |

## Chapter 4 — GPT 아키텍처

| 호출 | 형태 | 의미 |
|---|---|---|
| `nn.Parameter(torch.ones(emb_dim))` | 클래스 | LayerNorm의 scale(γ) — 학습 대상 |
| `nn.Parameter(torch.zeros(emb_dim))` | 클래스 | LayerNorm의 shift(β) |
| `x.mean(dim=-1, keepdim=True)` | 메서드 | 토큰별 평균 |
| `x.var(dim=-1, keepdim=True, unbiased=False)` | 메서드 | 토큰별 분산 (n으로 나눔) |
| `torch.sqrt(var + eps)` | 함수 | 표준편차 |
| `torch.tanh(...)` / `torch.pow(x, 3)` / `torch.pi` | 함수 / 함수 / 속성 | GELU 근사식 구성요소 |
| `nn.Sequential(*[...])` | 클래스 | 층을 순서대로 연결 |
| `nn.Embedding(cfg["vocab_size"], emb_dim)` | 클래스 | 토큰 임베딩 |
| `nn.Embedding(cfg["context_length"], emb_dim)` | 클래스 | 위치 임베딩 |
| `torch.arange(seq_len, device=in_idx.device)` | 함수 | 위치 번호 + **장치 일치** |
| `nn.Linear(emb_dim, vocab_size, bias=False)` | 클래스 | 출력 헤드 (bias 없음) |
| `idx[:, -context_size:]` | 슬라이싱 | 문맥을 **뒤에서** 자르기 |
| `logits[:, -1, :]` | 슬라이싱 | 마지막 시점만 |
| `torch.argmax(logits, dim=-1, keepdim=True)` | 함수 | 최고 점수 토큰 선택 (greedy) |
| `torch.cat((idx, idx_next), dim=1)` | 함수 | 생성 토큰 이어붙이기 |
| `torch.tensor(encoded).unsqueeze(0)` | 함수 + 메서드 | 배치 차원 추가 |
| `out.squeeze(0).tolist()` | 메서드 | 배치 차원 제거 → 리스트 |

## Chapter 5 — Pretraining

| 호출 | 형태 | 의미 |
|---|---|---|
| `torch.nn.functional.cross_entropy(logits, targets)` | 함수 | 손실 계산 |
| `logits.flatten(0, 1)` | 메서드 | `(B,T,V)` → `(B*T,V)` 평탄화 |
| `target_batch.flatten()` | 메서드 | `(B,T)` → `(B*T,)` |
| `model.train()` | 메서드 | 학습 모드 |
| `model.eval()` | 메서드 | 평가 모드 (드롭아웃 끔) |
| `optimizer.zero_grad()` | 메서드 | 이전 기울기 삭제 |
| `loss.backward()` | 메서드 | 역전파 |
| `optimizer.step()` | 메서드 | 가중치 갱신 |
| `torch.optim.AdamW(model.parameters(), lr, weight_decay)` | 클래스 | 옵티마이저 |
| `model.parameters()` | 메서드 | 학습 대상 전부 |
| `with torch.no_grad():` | 함수(컨텍스트) | 평가 시 기울기 끄기 |
| `loss.item()` | 메서드 | 텐서 → 파이썬 실수 |
| `input_batch.numel()` | 메서드 | 원소 개수 (토큰 수 카운트) |
| `x.to(device)` | 메서드 | 장치 이동 |
| `torch.save(model.state_dict(), path)` | 함수 + 메서드 | 가중치 저장 |
| `torch.load(path, weights_only=True)` | 함수 | 가중치 로드 |
| `model.load_state_dict(sd)` | 메서드 | 모델에 가중치 주입 |
| `model.pos_emb.weight.shape[0]` | 속성 체인 | 최대 문맥 길이 역산 |
| `torch.linspace(0, epochs, n)` | 함수 | 에포크 축 생성 |
| `ax1.twiny()` | 메서드 | 보조 x축 (Tokens seen) |

## Chapter 6 — Classification Finetuning

| 호출 | 형태 | 의미 |
|---|---|---|
| `pd.read_csv(path, sep="\t", header=None, names=[...])` | 함수 | TSV 로딩 |
| `df[df["Label"] == "spam"].shape[0]` | 인덱싱 + 속성 | 조건에 맞는 행 개수 |
| `df.sample(num_spam, random_state=123)` | 메서드 | **언더샘플링** (무작위 추출) |
| `pd.concat([a, b])` | 함수 | DataFrame 합치기 |
| `df["Label"].map({"ham":0, "spam":1})` | 메서드 | **라벨 인코딩** (문자→정수) |
| `df.sample(frac=1, random_state=123)` | 메서드 | 전체 셔플 (`frac=1` = 100%) |
| `df.reset_index(drop=True)` | 메서드 | 인덱스 0부터 재부여 |
| `df.to_csv(path, index=None)` | 메서드 | CSV 저장 |
| `self.data.iloc[index]["Label"]` | 메서드 + 인덱싱 | 번호로 행 → 열 선택 |
| `torch.tensor(encoded, dtype=torch.long)` | 함수 | **정수형** 텐서 (임베딩/CE 필수) |
| `torch.long` | 속성 | 64비트 정수 타입 |
| `model(input_batch)[:, -1, :]` | 슬라이싱 | 분류는 **마지막 토큰**만 |
| `torch.argmax(logits, dim=-1)` | 함수 | 예측 클래스 |
| `(predicted == target).sum().item()` | 연산 + 메서드 | 맞춘 개수 |
| `param.requires_grad = False` | 속성 대입 | **파라미터 동결** |
| `model.trf_blocks[-1].parameters()` | 인덱싱 + 메서드 | 마지막 블록만 해제 |
| `torch.nn.Linear(in_features=..., out_features=num_classes)` | 클래스 | 분류 헤드 교체 |
| `torch.linspace(0, num_epochs, n)` | 함수 | 그래프 축 |

## Chapter 6 — LoRA

| 호출 | 형태 | 의미 |
|---|---|---|
| `nn.Parameter(torch.empty(in_dim, rank))` | 클래스 + 함수 | LoRA A 행렬 (학습 대상) |
| `nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))` | 함수 (**제자리**) | A를 랜덤 초기화 |
| `math.sqrt(5)` | 함수 | 제곱근 (파이썬 표준 라이브러리) |
| `nn.Parameter(torch.zeros(rank, out_dim))` | 클래스 + 함수 | LoRA B 행렬 — **0으로 시작** |
| `linear.in_features` / `linear.out_features` | 속성 | 원본 Linear의 입출력 크기 승계 |
| `model.named_children()` | 메서드 | 자식 층 (이름, 객체) 순회 |
| `isinstance(module, torch.nn.Linear)` | 함수(내장) | 층 종류 판별 |
| `setattr(model, name, LinearWithLoRA(...))` | 함수(내장) | **층을 동적으로 교체** |
| `sum(p.numel() for p in model.parameters() if p.requires_grad)` | 함수 + 메서드 | 학습 가능 파라미터 수 |
| `p.requires_grad` | 속성 | 학습 대상 여부 |

## Chapter 7 — Instruction Tuning (SFT)

| 호출 | 형태 | 의미 |
|---|---|---|
| `json.load(file)` | 함수 | JSON 파일 → 파이썬 객체 |
| `json.dump(data, file, indent=4)` | 함수 | 파이썬 객체 → JSON 저장 |
| `f"\n\n### Instruction:\n{entry['instruction']}"` | f-string | **Alpaca 프롬프트 템플릿** |
| `max(len(item)+1 for item in batch)` | 함수(내장) | 배치 내 최대 길이 (**동적 패딩**) |
| `item.copy()` | 메서드 | 리스트 복사 (원본 보호) |
| `targets == pad_token_id` | 연산 | 패딩 위치 True/False 지도 |
| `torch.nonzero(mask).squeeze()` | 함수 + 메서드 | True인 **인덱스 목록** |
| `indices.numel()` | 메서드 | 인덱스 개수 |
| `targets[indices[1:]] = ignore_index` | 인덱싱 대입 | 두 번째 패딩부터 −100 (손실 제외) |
| `torch.stack(inputs_lst).to(device)` | 함수 + 메서드 | 배치 텐서 생성 + 장치 이동 |
| `partial(custom_collate_fn, device=device, ...)` | 함수 | **인자를 미리 고정한 새 함수** |
| `DataLoader(..., collate_fn=customized_collate_fn)` | 클래스 | 커스텀 배치 구성 |
| `tqdm(enumerate(test_data), total=len(test_data))` | 클래스 | 진행률 표시 |
| `re.sub(r'[ ()]', '', CHOOSE_MODEL)` | 함수 | 파일명에서 특수문자 제거 |
| `generated_text[len(input_text):]` | 슬라이싱 | 프롬프트 제거, 응답만 추출 |

## Chapter 7 — DPO

| 호출 | 형태 | 의미 |
|---|---|---|
| `torch.ones(len(padded)).bool()` | 함수 + 메서드 | 전부 True인 마스크 |
| `mask[len(sequence):] = False` | 인덱싱 대입 | 패딩 구간 끄기 |
| `mask[:prompt.shape[0] + 2] = False` | 인덱싱 대입 | **프롬프트 구간 끄기** (답변만 학습) |
| `F.log_softmax(logits, dim=-1)` | 함수 | 로그 확률 (수치 안정) |
| `torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1))` | 함수 | **정답 토큰의 확률만** 추출 |
| `.squeeze(-1)` | 메서드 | gather 후 축 되돌리기 |
| `labels[:, 1:].clone()` | 슬라이싱 + 메서드 | 자기회귀 정렬 (라벨 1칸 밀기) |
| `logits[:, :-1, :]` | 슬라이싱 | 자기회귀 정렬 (로짓 뒤 자르기) |
| `selected.sum(-1) / mask.sum(-1)` | 메서드 | **유효 토큰만** 평균 (길이 보정) |
| `F.logsigmoid(beta * logits)` | 함수 | DPO 손실 핵심 |
| `(a - b).detach()` | 메서드 | reward는 로깅용 → 그래프 분리 |
| `reference_model.eval()` | 메서드 | 기준 모델 고정 |
| `with torch.no_grad():` | 함수(컨텍스트) | 기준 모델은 기울기 없음 |
| `losses.mean()` | 메서드 | 배치 손실 평균 (backward는 스칼라만) |
| `batch["chosen"].numel()` | 메서드 | 토큰 수 카운트 |

---

# 제4부. 백지에서 쓸 때 자주 틀리는 8가지

| # | 함정 | 올바른 형태 |
|---|---|---|
| 1 | `cat`/`stack`에 인자를 나열 | `torch.cat((a, b), dim=1)` — **튜플로 묶기** |
| 2 | `Sequential`에 별표 누락 | `nn.Sequential(*[...])` |
| 3 | `transpose` 축 번호 | 3차원 → `(1,2)`, 4차원 → **`(2,3)`** |
| 4 | `transpose` 후 바로 `view` | 사이에 **`.contiguous()`** |
| 5 | `softmax`에 `dim` 생략 | `dim`은 **기본값 없음** — 반드시 명시 |
| 6 | `cross_entropy` 앞에 softmax | **로짓 그대로** 넣기 |
| 7 | `arange`에 device 누락 | `torch.arange(n, device=x.device)` |
| 8 | `_` 유무 혼동 | `masked_fill_`(원본 수정) ↔ `masked_fill`(새 텐서 반환) |

---

# 제5부. 백지 구현 사고 절차

**"주어 → 소속 → 인자"** 순서로 생각합니다.

### 예시 1 — "어텐션 점수를 확률로 바꾸고 싶다"
```
① 새 텐서를 만드는 계산이다        → torch. 소속
② 확률화 = softmax                 → 소문자 = 함수
③ softmax(input, dim) 형식         → dim 필수
결론: torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

### 예시 2 — "Q 투영층을 만들고 싶다"
```
① 가중치를 기억해야 한다           → 클래스 → nn. 대문자
② 완전연결 = Linear
③ Linear(in_features, out_features, bias)
결론: nn.Linear(d_in, d_out, bias=qkv_bias)
```

### 예시 3 — "이 텐서의 모양을 바꾸고 싶다"
```
① 이미 있는 텐서를 변형한다        → 메서드 (텐서에 붙임)
② 모양 변경 = view
③ view(*shape)
결론: keys.view(b, num_tokens, self.num_heads, self.head_dim)
```

### 예시 4 — "가중치를 저장하고 싶다"
```
① 파일로 내보내는 전역 작업        → torch. 소속
② 저장 = save
③ save(obj, path) — obj는 모델이 아니라 state_dict
결론: torch.save(model.state_dict(), "outputs/model.pth")
```

---

# 부록 — 모를 때 찾는 법

| 방법 | 사용법 | 언제 |
|---|---|---|
| **자동완성** | `torch.` 입력 후 `Ctrl+Space` | "torch에 뭐가 있더라?" |
| **시그니처 힌트** | 괄호 안에서 `Shift+Tab` | "인자를 뭘 받지?" |
| **물음표** | 셀에 `torch.tensor?` 실행 | 설명 전체 보기 |
| **마우스 hover** | 함수 이름 위에 커서 | 가장 빠름 |
| **정의로 이동** | `F12` 또는 `Ctrl+클릭` | 실제 구현이 궁금할 때 |
| **목록 뽑기** | `[n for n in dir(torch) if 'sq' in n]` | 이름이 가물가물할 때 |
| **타입 확인** | `isinstance(nn.Linear, type)` | 클래스인지 함수인지 |

---

# 최종 요약 — 3줄

1. **대문자로 시작하면 클래스, 소문자면 함수.** 패키지가 아니라 **이름**이 결정한다 (검사 결과 `torch.nn`에서 예외 0건).
2. **메서드는 객체에 소속된 함수.** 타입은 함수와 동일하며, 자기 객체를 첫 인자로 자동으로 받는다.
3. **기억할 상태가 있으면 클래스**(`nn.Dropout`), **없으면 함수**(`F.dropout`).
