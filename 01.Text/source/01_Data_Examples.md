# LLM 실습(Chapter 1~7) — 포인트 정리



## 0. 출제 포인트 ↔ 코드 축 매핑

| 출제 포인트               | 대응 코드 축                                                                                                               | 핵심 챕터         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------- |
| 모델의 원리 (layer 의미/구성) | `nn.Linear` / `nn.Embedding` / `LayerNorm` / `GELU` / `nn.Sequential` / `register_buffer`                             | Ch3, Ch4      |
| Tensor 흐름 (shape 변환) | `.view()` / `.transpose()` / `.contiguous()` / `[:, -1, :]` / `unsqueeze` / `squeeze` / `dim=` 인자                     | Ch3, Ch4, Ch5 |
| 결과↔정답 → 학습 연결        | `cross_entropy` / `zero_grad→backward→step` / `train()`·`eval()` / `no_grad()` / `requires_grad`                      | Ch5, Ch6, Ch7 |
| 데이터 구성 원리            | sliding window(`stride`), padding, `ignore_index=-100`, mask, train/val split, `shuffle`/`drop_last`                  | Ch2, Ch6, Ch7 |
| 관련 Library 활용        | `tiktoken` / `torch.utils.data` / `pandas` / `torch.optim.AdamW` / `functools.partial` / `transformers.AutoTokenizer` | Ch2, Ch6, Ch7 |
| 평가 Metric            | Accuracy(`argmax→==→sum→/N`), 평균 Loss, DPO reward margin                                                              | Ch6, Ch5, Ch7 |

> ⚠️ 이 교재에는 **sklearn이 전혀 사용되지 않는다.** LLM 트랙의 "library 활용"은
> `tiktoken` / `torch.utils.data` / `pandas` / `functools.partial` / `transformers.AutoTokenizer`로 대체 대비할 것.

### 📌 시작 전에 알아둘 기본 용어

| 용어                  | 쉬운 설명                                                                            |
| ------------------- | -------------------------------------------------------------------------------- |
| **Tensor**          | 숫자를 담는 다차원 상자. `[3, 5]`는 3행 5열 표, `[2, 4, 256]`은 그런 표가 2장 있고 각 칸이 256개 숫자 벡터라는 뜻 |
| **shape**           | 텐서의 크기. LLM에서는 보통 `(batch, seq_len, emb_dim)` = (문장 몇 개, 토큰 몇 개, 벡터 몇 차원)        |
| **dim=N**           | "N번째 축을 기준으로 계산해라". `dim=-1`은 항상 **맨 마지막 축**을 뜻함                                 |
| **batch**           | 한 번에 처리하는 문장 묶음. 8개씩 묶으면 batch_size=8                                            |
| **logits**          | 모델이 뱉은 **가공 전 점수**. 확률이 아니고 softmax를 거쳐야 확률이 됨                                   |
| **파라미터(Parameter)** | 학습으로 값이 바뀌는 숫자들 (가중치). `nn.Parameter`로 감싸야 학습 대상이 됨                              |
| **gradient(기울기)**   | "이 파라미터를 조금 키우면 손실이 얼마나 줄어드는가"를 나타내는 값                                           |

---

# 1. 메인 표 — TODO 이외 구간의 출제 예상 빈칸

## Chapter 2 — Dataset & Embedding

파일: `Chapter_2_Exercise_Dataset.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                          | 빈칸 후보                                                  | 의미 / 출제 의도                                                                                                                                                                                                                                                                   | 확률  |
| ---- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 2-1  | `for i in range(0, len(token_ids) - max_length, stride):`                                               | `stride`                                               | `range(시작, 끝, **건너뛸 간격**)` 구조에서 세 번째 자리가 `stride`다. 긴 글에서 `max_length`짜리 창문(window)을 오려낼 때 **창문을 몇 칸씩 밀지**를 정한다.<br>· `stride=1` → 한 칸씩 밀어서 거의 똑같은 데이터가 잔뜩 생김(중복 多, 데이터 수 多)<br>· `stride=max_length` → 창문끼리 안 겹침(중복 無)<br>"왜 사전학습에서는 stride를 context_length와 같게 두나?"가 단골 질문 | ★★★ |
| 2-2  | `pos_embeddings = pos_embedding_layer(torch.arange(max_length))`                                        | `torch.arange`                                         | Transformer는 단어 순서를 스스로 모른다. 그래서 "너는 0번째, 너는 1번째" 하고 **위치 번호를 따로 알려줘야** 한다. `torch.arange(4)`는 `[0,1,2,3]`을 만들어주는 함수다. 토큰 임베딩은 **토큰 ID**를 넣지만, 위치 임베딩은 **위치 번호**를 넣는다는 게 핵심 차이                                                                                               | ★★★ |
| 2-3  | `input_embeddings = token_embeddings + pos_embeddings`                                                  | `+`                                                    | "무슨 단어인가(token)" + "몇 번째인가(position)" 두 정보를 **더해서** 하나의 벡터로 합친다. 이어붙이기(concat)가 아니라 **덧셈**인 것이 포인트. shape가 `[8,4,256] + [4,256]`으로 달라 보여도 파이토치가 자동으로 8개 문장에 똑같이 더해준다(브로드캐스팅)                                                                                                 | ★★★ |
| 2-4  | `print(token_embeddings.shape)` → `[8, 4, 256]`                                                         | 출력 shape 값                                             | `batch_size=8`(문장 8개), `max_length=4`(각 4토큰), `output_dim=256`(토큰당 256차원 벡터). **정수 ID였던 입력이 임베딩을 지나면서 차원이 하나 늘어난다**는 것을 보여주는 대표 지점                                                                                                                                           | ★★★ |
| 2-5  | `self.input_ids.append(torch.tensor(input_chunk))`                                                      | `torch.tensor`                                         | 지금까지는 그냥 파이썬 리스트(`[318, 22, ...]`)였는데, 여기서 **파이토치 텐서로 바뀐다**. 모델은 리스트를 못 먹고 텐서만 먹기 때문에, 이 줄이 "일반 파이썬 세계 → 딥러닝 세계"의 경계선이다                                                                                                                                                      | ★★  |
| 2-6  | `pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)`                                  | `context_length`                                       | 위치 임베딩 테이블의 행 개수는 **모델이 감당하는 최대 토큰 수**여야 한다. 토큰 임베딩이 `vocab_size`(단어 종류 수)인 것과 대비해서 외울 것. 둘 다 `nn.Embedding`이지만 첫 인자의 **의미가 완전히 다르다**                                                                                                                                        | ★★  |
| 2-7  | `tokenizer = tiktoken.get_encoding("gpt2")`                                                             | `"gpt2"` / `get_encoding`                              | 글자를 숫자로 바꾸는 사전(BPE)을 불러온다. `"gpt2"`를 고르면 자동으로 단어 종류 50,257개, 문장끝 토큰 번호 50,256이 따라온다. 뒤 챕터의 `vocab_size=50257`, `pad_token_id=50256`이 전부 여기서 나온 숫자                                                                                                                            | ★★  |
| 2-8  | `token_ids = tokenizer.encode(txt, allowed_special={"<\|endoftext\|>"})`                                | `allowed_special`                                      | 토크나이저는 기본적으로 `<\|endoftext\|>` 같은 특수 문자열을 보면 "이건 위험하다"며 에러를 낸다. 이 인자로 **"이 특수 토큰은 허용한다"** 고 명시해야 통과된다                                                                                                                                                                        | ★★  |
| 2-9  | `def __len__(self): return len(self.input_ids)`                                                         | `len(self.input_ids)`                                  | PyTorch `Dataset`은 세 메서드가 한 세트다. `__init__`(데이터 준비) / `__len__`(총 몇 개인지) / `__getitem__`(n번째 하나 주기). DataLoader가 `__len__`을 보고 몇 번 반복할지 정하므로 **빠지면 동작 자체가 안 된다**                                                                                                             | ★★  |
| 2-10 | `create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)` | `stride=max_length` / `shuffle=False`                  | 실습 확인용이라 결과를 매번 똑같이 보려고 `shuffle=False`로 둔 것. **"목적에 따라 옵션이 달라진다"** 는 데이터 구성 원리 문항의 전형                                                                                                                                                                                       | ★★  |
| 2-11 | `DEVICE = torch.device("cuda")` / `torch.set_default_device(DEVICE)`                                    | `torch.device` / `set_default_device`                  | 계산을 CPU에서 할지 GPU에서 할지 정한다. `set_default_device`를 쓰면 이후 만드는 모든 텐서가 자동으로 GPU에 올라가서 `.to(device)`를 깜빡해도 동작한다                                                                                                                                                                    | ★★  |
| 2-12 | `data_iter = iter(dataloader)` / `first_batch = next(data_iter)`                                        | `iter` / `next`                                        | DataLoader는 for문으로 도는 게 정석이지만, **배치 하나만 꺼내 눈으로 확인**할 때 이 두 함수를 쓴다. `iter`로 꺼내는 장치를 만들고 `next`로 한 개씩 뽑는다                                                                                                                                                                      | ★   |
| 2-13 | `assert len(token_ids) > max_length, "..."`                                                             | `>` / `max_length`                                     | 글이 창문 크기보다 짧으면 데이터를 한 개도 못 만든다. 학습 시작 후 이상한 에러가 나기 전에 **미리 막아주는 안전장치**                                                                                                                                                                                                       | ★   |
| 2-14 | `qwen_tokens = qwen.encode(text, add_special_tokens=True)`                                              | `AutoTokenizer.from_pretrained` / `add_special_tokens` | HuggingFace에서 모델별 토크나이저를 이름만으로 내려받는 방식. 같은 문장이라도 모델마다 토큰 개수가 다르다(한국어는 EXAONE, 중국어는 Qwen이 효율적)는 **토크나이저 성능 비교** 개념                                                                                                                                                            | ★   |

---

## Chapter 3 — Attention (최대 출제 구간)

파일: `Chapter_3_Excercise_Attention.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                     | 빈칸 후보                                           | 의미 / 출제 의도                                                                                                                                                                                                                                                                                                  | 확률  |
| ---- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 3-1  | `self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))` | `register_buffer` / `torch.triu` / `diagonal=1` | GPT는 "다음 단어 맞추기"로 학습하므로 **미래 단어를 미리 보면 반칙**이다. 그래서 미래 위치를 표시한 지도(mask)를 만든다.<br>· `torch.ones(L,L)` → 전부 1인 정사각 표<br>· `torch.triu(..., diagonal=1)` → **대각선 위쪽만 1**로 남김 = 미래 위치<br>· `register_buffer` → 이 표는 **학습되지 않지만**(gradient 없음) 모델과 함께 저장/GPU 이동되게 등록하는 것. `nn.Parameter`(학습 O)와의 차이를 묻는 문제가 자주 나옴 | ★★★ |
| 3-2  | `attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)`                          | `keys.shape[-1]**0.5` / `dim=-1`                | 두 가지가 한 줄에 들어있다.<br>· **√d_k 나누기**: 벡터 차원이 커질수록 내적 값이 커지는데, 그대로 softmax에 넣으면 한 곳만 1이고 나머지는 0이 되어 학습이 멈춘다. 그래서 차원의 제곱근(`keys.shape[-1]**0.5`)으로 나눠 크기를 눌러준다<br>· **`dim=-1`**: 마지막 축(=각 단어가 다른 모든 단어를 얼마나 볼지) 기준으로 확률화. 이 축이 바뀌면 완전히 다른 계산이 됨                                                                | ★★★ |
| 3-3  | `self.head_dim = d_out // num_heads`                                                               | `//` / `num_heads`                              | 멀티헤드는 **큰 벡터 하나를 여러 조각으로 나눠 각자 다른 관점으로 보게** 하는 기법. d_out=768, num_heads=12면 head_dim=64. `/`(실수 나눗셈)가 아니라 `//`(몫)여야 정수가 나온다                                                                                                                                                                                 | ★★★ |
| 3-4  | `assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"`                          | `%` / `num_heads`                               | 나눠 떨어지지 않으면 조각 크기가 안 맞아 shape 에러가 난다. `%`(나머지)가 0인지 확인하는 방어 코드                                                                                                                                                                                                                                              | ★★★ |
| 3-5  | `keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)`                                   | `view` / `self.num_heads, self.head_dim`        | `view`는 **숫자 개수는 그대로 두고 모양만 바꾸는** 함수. `[2, 6, 768]` → `[2, 6, 12, 64]`처럼 마지막 768을 12×64로 쪼갠다. 실제로 복사가 일어나는 게 아니라 "같은 데이터를 다르게 읽는다"는 개념                                                                                                                                                                      | ★★★ |
| 3-6  | `keys = keys.transpose(1, 2)`                                                                      | `transpose(1, 2)`                               | `transpose(a,b)`는 **a번 축과 b번 축의 자리를 바꾼다**. `(b, 토큰, 헤드, 차원)` → `(b, 헤드, 토큰, 차원)`. 이렇게 해야 "헤드별로 독립된 (토큰×차원) 행렬"이 되어 12개 헤드가 동시에 계산된다                                                                                                                                                                         | ★★★ |
| 3-7  | `attn_scores = queries @ keys.transpose(2, 3)`                                                     | `transpose(2, 3)`                               | **가장 헷갈리는 함정.** 행렬곱을 하려면 뒤쪽 텐서의 마지막 두 축을 뒤집어야 하는데,<br>· CausalAttention은 3차원 `(b, T, d)` → 마지막 두 축이 1,2 → `transpose(1,2)`<br>· MultiHeadAttention은 4차원 `(b, H, T, d)` → 마지막 두 축이 2,3 → **`transpose(2,3)`**<br>둘을 바꿔 적으면 오답. 항상 "뒤에서 두 개"라고 기억할 것                                                          | ★★★ |
| 3-8  | `context_vec = (attn_weights @ values).transpose(1, 2)`                                            | `transpose(1, 2)`                               | 헤드별 계산이 끝났으니 다시 사람이 읽기 좋은 순서 `(b, 토큰, 헤드, 차원)`으로 되돌린다. 3-6의 정확한 역동작                                                                                                                                                                                                                                         | ★★★ |
| 3-9  | `context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)`                           | `contiguous()` / `self.d_out`                   | `transpose`는 실제 메모리를 옮기지 않고 "읽는 순서만" 바꾸므로 메모리가 뒤죽박죽(비연속) 상태가 된다. 이 상태로 `view`를 부르면 **RuntimeError**가 난다. `contiguous()`가 메모리를 정리해주는 역할.<br>그리고 `view(b, T, d_out)`이 바로 **헤드들을 다시 하나로 이어붙이는(concat) 동작**의 실체다                                                                                                | ★★★ |
| 3-10 | `self.out_proj = nn.Linear(d_out, d_out)` / `context_vec = self.out_proj(context_vec)`             | `nn.Linear(d_out, d_out)` / `out_proj`          | 헤드들을 단순히 이어붙이기만 하면 각 헤드 정보가 섞이지 않는다. 마지막에 Linear를 한 번 더 통과시켜 **헤드끼리 정보를 섞어준다**. 입력=출력 차원이라 크기는 안 변함                                                                                                                                                                                                         | ★★  |
| 3-11 | `mask_bool = self.mask.bool()[:num_tokens, :num_tokens]`                                           | `.bool()` / `[:num_tokens, :num_tokens]`        | 두 가지 이유가 겹쳐있다. ① `masked_fill_`은 True/False 마스크만 받으므로 0/1 실수를 `.bool()`로 변환. ② 마스크는 최대 길이(예: 1024) 기준으로 만들어져 있는데 실제 입력이 6토큰이면 **6×6만 잘라 써야** shape가 맞는다                                                                                                                                                     | ★★  |
| 3-12 | `return torch.cat([head(x) for head in self.heads], dim=-1)`                                       | `torch.cat` / `dim=-1`                          | Wrapper 버전은 **작은 어텐션을 여러 개 만들어 결과를 옆으로 이어붙이는** 단순한 방식. `dim=-1`이므로 출력 차원이 `d_out × num_heads`로 늘어난다(2×2=4). 반면 진짜 MultiHeadAttention은 하나의 큰 Linear를 쪼개 쓰므로 출력이 `d_out` 그대로(2). **이 차이를 묻는 문제 주의**                                                                                                           | ★★  |
| 3-13 | `self.heads = nn.ModuleList([CausalAttention(...) for _ in range(num_heads)])`                     | `nn.ModuleList` / `range(num_heads)`            | 그냥 파이썬 `[]` 리스트에 레이어를 담으면 파이토치가 **파라미터로 인식하지 못해 학습이 안 된다**. `nn.ModuleList`로 감싸야 `model.parameters()`에 포함된다                                                                                                                                                                                                 | ★★  |
| 3-14 | `self.dropout = nn.Dropout(dropout)`                                                               | `nn.Dropout`                                    | 학습 중 어텐션 가중치 일부를 랜덤하게 0으로 만들어, 모델이 특정 단어에만 의존하지 않게 한다(과적합 방지). `model.eval()`을 부르면 자동으로 꺼진다                                                                                                                                                                                                                 | ★★  |
| 3-15 | `batch = torch.stack((inputs, inputs), dim=0)`                                                     | `torch.stack` / `dim=0`                         | `torch.stack`은 **새 축을 만들면서** 텐서를 쌓는다. `[6,3]` 두 개 → `[2,6,3]`. 같은 문장을 두 번 쌓아 "batch_size=2인 척"하는 실습용 트릭. (기존 축에 이어붙이는 `torch.cat`과 구분할 것)                                                                                                                                                                   | ★★  |
| 3-16 | `context_length = batch.shape[1]`                                                                  | `shape[1]`                                      | `batch.shape`는 `(2, 6, 3)` = (문장수, 토큰수, 차원). 그중 **1번 인덱스가 토큰 수**. 마스크 크기를 여기에 맞춰야 한다                                                                                                                                                                                                                        | ★★  |
| 3-17 | `b, num_tokens, d_in = x.shape`                                                                    | 언패킹 순서                                          | 파이썬의 튜플 언패킹으로 shape 3개를 한 번에 변수에 담는 관용구. **순서가 (배치, 토큰, 차원)** 임을 아는지 확인                                                                                                                                                                                                                                     | ★   |
| 3-18 | 출력 shape: Wrapper → `[2, 6, 4]` / MHA → `[2, 6, 2]`                                                | 서술형                                             | 3-12에서 설명한 구조 차이가 숫자로 드러나는 지점. 왜 4와 2로 갈리는지 설명할 수 있어야 함                                                                                                                                                                                                                                                     | ★★  |

---

## Chapter 4 — GPT 아키텍처

파일: `Chapter_4_Excercise_GPT.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                                                                        | 빈칸 후보                                           | 의미 / 출제 의도                                                                                                                                                                                        | 확률  |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 4-1  | `mean = x.mean(dim=-1, keepdim=True)`                                                                                                                 | `dim=-1` / `keepdim=True`                       | LayerNorm은 **한 토큰이 가진 768개 숫자끼리** 평균을 낸다(배치나 토큰 방향이 아님) → `dim=-1`.<br>`keepdim=True`는 계산 후에도 축을 없애지 않고 `[2,6,1]` 형태로 남기는 것. 이래야 원래 `[2,6,768]`에서 뺄 때 자동으로 768칸 전체에 적용된다. `False`면 shape가 안 맞아 에러 | ★★★ |
| 4-2  | `var = x.var(dim=-1, keepdim=True, unbiased=False)`                                                                                                   | `unbiased=False` / `dim=-1`                     | 분산을 구할 때 `n-1`로 나눌지(표본분산, `unbiased=True`) `n`으로 나눌지(모분산, `False`)의 선택. LayerNorm 논문과 원 구현이 **`n`으로 나누므로 `unbiased=False`**. 기본값이 `True`라서 **일부러 명시**한 것 → 그래서 빈칸으로 좋다                            | ★★★ |
| 4-3  | `self.scale = nn.Parameter(torch.ones(emb_dim))`                                                                                                      | `nn.Parameter` / `torch.ones`                   | 정규화로 평균0·분산1을 만들면 모든 층이 똑같은 분포가 되어 표현력이 떨어진다. 그래서 **모델이 스스로 배율을 다시 배우도록** 학습 가능한 값(γ)을 곱한다. `nn.Parameter`로 감싸야 학습 대상이 되고, 시작값은 "아무것도 안 바꾸는" **1**                                                | ★★★ |
| 4-4  | `self.shift = nn.Parameter(torch.zeros(emb_dim))`                                                                                                     | `torch.zeros`                                   | 위와 같은 이유의 이동값(β). 시작값은 "아무것도 안 더하는" **0**. `scale=1, shift=0`이면 정규화 결과 그대로라는 점이 핵심                                                                                                                | ★★★ |
| 4-5  | `shortcut = x` … `x = x + shortcut` (블록당 2회)                                                                                                          | `shortcut` / `+`                                | **잔차 연결(Residual)**. 층을 깊게 쌓으면 기울기가 뒤로 갈수록 사라져 학습이 안 되는데, 입력을 출력에 그대로 더해주면 기울기가 지나갈 **고속도로**가 생긴다. 한 블록에 어텐션용·FFN용 **두 번** 등장하는 것도 포인트                                                            | ★★★ |
| 4-6  | `x = self.norm1(x)` (att 앞) / `x = self.norm2(x)` (ff 앞)                                                                                              | `norm1` / `norm2` 순서                            | 정규화를 연산 **앞**에 두면 Pre-LN, **뒤**에 두면 Post-LN. 이 교재는 Pre-LN(학습이 더 안정적). 코드 순서를 뒤섞어 놓고 "올바른 순서로 배열하라"는 문제로 나오기 쉽다. 정답 순서: `shortcut 저장 → norm → att → dropout → +shortcut`                           | ★★★ |
| 4-7  | `self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])`                                                           | `nn.Sequential` / `*` / `cfg["n_layers"]`       | 똑같은 블록을 12개 만들어 순서대로 연결한다. `nn.Sequential`은 넣은 순서대로 자동 통과시켜주는 컨테이너.<br>앞의 `*`는 **리스트를 낱개 인자로 풀어주는** 파이썬 문법(언패킹). 이게 없으면 "리스트 1개"를 넘긴 셈이 되어 에러                                                     | ★★★ |
| 4-8  | `pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))`                                                                              | `torch.arange(seq_len` / `device=in_idx.device` | 두 포인트. ① 위치 번호 `[0,1,...,T-1]`을 만든다. ② **`device=`를 지정하지 않으면** 입력은 GPU인데 위치 번호는 CPU라서 `Expected all tensors to be on the same device` 에러가 난다                                                      | ★★★ |
| 4-9  | `idx_cond = idx[:, -context_size:]`                                                                                                                   | `-context_size:`                                | 문장을 생성하다 보면 길이가 계속 늘어나 모델 한계(1024)를 넘게 된다. 이때 **최근 것을 남기고 오래된 앞부분을 버려야** 한다 → 음수 인덱스로 뒤에서 자르기.<br>`[:context_size]`(앞에서 자르기)로 바꾸면 방금 생성한 단어가 사라져 문장이 이상해진다                                        | ★★★ |
| 4-10 | `encoded_tensor = torch.tensor(encoded).unsqueeze(0)`                                                                                                 | `unsqueeze(0)`                                  | 모델은 항상 `(배치, 토큰)` 2차원을 기대하는데 문장 하나면 `[15]`짜리 1차원뿐이다. `unsqueeze(0)`는 **맨 앞에 크기 1짜리 축을 끼워 넣어** `[1, 15]`로 만든다. "가짜 배치 차원 추가"                                                                       | ★★★ |
| 4-11 | `decoded_text = tokenizer.decode(out.squeeze(0).tolist())`                                                                                            | `squeeze(0)` / `tolist()`                       | 4-10의 정확한 반대. `squeeze(0)`로 배치 축을 없애 `[25]`로 만들고, `tolist()`로 텐서를 파이썬 리스트로 되돌려야 토크나이저가 받아준다. **입구(unsqueeze) ↔ 출구(squeeze)** 쌍으로 외울 것                                                             | ★★★ |
| 4-12 | `self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)`                                                                            | `bias=False`                                    | 768차원 벡터를 50,257개 단어 점수로 변환하는 마지막 층. GPT 계열은 이 층에 **편향(bias)을 쓰지 않는 것이 관례**(토큰 임베딩과 가중치를 공유할 수 있도록). 기본값이 `True`라 일부러 껐다는 게 출제 포인트                                                                | ★★  |
| 4-13 | `self.eps = 1e-5` / `torch.sqrt(var + self.eps)`                                                                                                      | `self.eps`                                      | 분산이 0이면 0으로 나누게 되어 결과가 무한대/NaN이 된다. 아주 작은 수를 더해 **0 나눗셈을 방지**하는 안전장치                                                                                                                              | ★★  |
| 4-14 | `0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))`                                             | `0.5` / `2.0 / torch.pi` / `0.044715` / `3`     | GELU 활성화 함수의 tanh 근사식. ReLU가 음수를 딱 잘라 0으로 만드는 것과 달리, GELU는 **부드럽게 눌러서** 음수도 조금은 통과시킨다. 상수 `0.044715`와 3제곱은 통째로 외우는 수밖에 없다                                                                         | ★★  |
| 4-15 | `self.drop_emb = nn.Dropout(cfg["drop_rate"])` / `self.drop_shortcut = nn.Dropout(cfg["drop_rate"])`                                                  | `nn.Dropout`                                    | 드롭아웃이 들어가는 **위치**를 묻는다. ① 임베딩 직후 ② 잔차를 더하기 **직전**. 순서가 `att → dropout → +shortcut`인 것이 중요                                                                                                         | ★★  |
| 4-16 | `batch_size, seq_len = in_idx.shape`                                                                                                                  | 언패킹                                             | GPT의 입력은 실수 벡터가 아니라 **정수 토큰 ID 2차원 배열** `(B, T)`라는 사실 확인. 임베딩을 통과해야 비로소 3차원 `(B, T, C)`가 된다                                                                                                       | ★★  |
| 4-17 | `model.eval()` (생성 전)                                                                                                                                 | `eval()`                                        | 드롭아웃을 끄고 결과를 안정적으로 만든다. 이걸 빼면 같은 입력에 매번 다른 출력이 나온다. **추론 전에는 항상 `eval()`**                                                                                                                        | ★★  |
| 4-18 | `GPT_CONFIG_124M = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False}` | 각 값                                             | GPT-2 Small의 스펙. 단어 5만개, 최대 1024토큰, 벡터 768차원, 헤드 12개, 층 12개. `768 ÷ 12 = 64`(head_dim)까지 연결해서 기억할 것                                                                                               | ★★  |
| 4-19 | `self.final_norm = LayerNorm(cfg["emb_dim"])`                                                                                                         | `final_norm`                                    | 블록을 다 지난 뒤 **출력 헤드에 넣기 직전** 한 번 더 정규화한다. Pre-LN 구조에서는 이 마지막 정규화가 반드시 필요하다                                                                                                                         | ★★  |

---

## Chapter 5 — Pretraining (학습 루프 & 데이터 분할)

파일: `Chapter_5_Excercise_Pretraining.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                     | 빈칸 후보                                   | 의미 / 출제 의도                                                                                                                                                                                                                              | 확률  |
| ---- | -------------------------------------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 5-1  | `model.eval()` … `with torch.no_grad():` … `model.train()`                                         | `eval()` / `no_grad()` / `train()`      | 세 개가 세트로 움직인다.<br>· `model.eval()` → 드롭아웃 끄기 (**동작 방식** 변경)<br>· `torch.no_grad()` → 기울기 기록 안 하기 (**메모리·속도** 절약)<br>· `model.train()` → 평가 끝나면 **반드시 원복**. 이걸 빠뜨리면 이후 학습에서 드롭아웃이 꺼진 채로 돌아 성능이 나빠진다<br>"eval()과 no_grad()의 차이"는 단골 서술 문항 | ★★★ |
| 5-2  | `split_idx = int(train_ratio * len(text_data))`                                                    | `int(...)` / `train_ratio`              | 전체 글자 수의 90% 지점을 계산한다. 곱셈 결과가 실수라서 **`int()`로 감싸야** 슬라이싱 인덱스로 쓸 수 있다                                                                                                                                                                    | ★★★ |
| 5-3  | `text_data[:split_idx]` (train) / `text_data[split_idx:]` (val)                                    | 슬라이싱 방향                                 | 앞 90%는 학습용, 뒤 10%는 검증용. **검증 데이터는 학습에 절대 쓰지 않아야** 모델이 진짜 실력인지 외운 것인지 구분할 수 있다                                                                                                                                                           | ★★★ |
| 5-4  | train: `shuffle=True, drop_last=True` / val: `shuffle=False, drop_last=False`                      | `True` / `False`                        | **데이터 구성 원리 최다 출제 지점.**<br>· 학습은 순서를 섞어야(`shuffle=True`) 모델이 데이터 순서를 외우지 않는다. 검증은 섞을 이유가 없고(`False`) 결과 재현이 중요<br>· `drop_last=True`는 마지막 자투리 배치(크기가 작음)를 버려 배치 크기를 일정하게 유지 → 학습 안정. 검증은 **모든 데이터를 다 봐야** 하므로 `False`                   | ★★★ |
| 5-5  | `stride=gpt_config["context_length"]`                                                              | `context_length`                        | 사전학습은 데이터가 방대하므로 굳이 겹칠 필요가 없다. stride를 창문 크기와 같게 두어 **중복 없이** 전체를 한 번씩 훑는다 (Ch2의 실습용 `stride=1`과 대비)                                                                                                                                    | ★★★ |
| 5-6  | `input_batch, target_batch = input_batch.to(device), target_batch.to(device)`                      | `.to(device)`                           | 모델이 GPU에 있으면 데이터도 GPU에 있어야 계산이 된다. 하나라도 CPU면 즉시 에러. 배치를 꺼낼 때마다 매번 옮겨줘야 하는 이유는 **DataLoader가 CPU에서 데이터를 만들기 때문**                                                                                                                         | ★★★ |
| 5-7  | `total_loss += loss.item()`                                                                        | `.item()`                               | `loss`는 텐서라서 그대로 더하면 **계산 그래프가 계속 쌓여 메모리가 터진다**. `.item()`으로 순수 파이썬 실수 하나만 뽑아내면 안전하다. "왜 그냥 `+= loss`가 아닌가"가 출제 포인트                                                                                                                     | ★★★ |
| 5-8  | `return total_loss / num_batches`                                                                  | `/ num_batches`                         | 배치별 손실을 다 더한 뒤 개수로 나눠 **평균 손실**을 만든다. 이 값이 이 챕터의 **평가 Metric**이고, 학습/검증 곡선의 y축이 된다                                                                                                                                                      | ★★★ |
| 5-9  | `torch.save(model.state_dict(), "outputs/model.pth")`                                              | `state_dict()`                          | 모델 객체 전체가 아니라 **가중치 숫자만** 딕셔너리로 저장한다. 파일이 작고, 파이토치 버전이 달라도 잘 열린다                                                                                                                                                                        | ★★★ |
| 5-10 | `model.load_state_dict(torch.load("outputs/model.pth", weights_only=True))`                        | `load_state_dict` / `weights_only=True` | 저장의 반대. **먼저 같은 구조의 빈 모델을 만든 뒤** 가중치를 부어넣는 2단계라는 점이 핵심. `weights_only=True`는 파일에 악성 코드가 있어도 실행하지 않는 안전 옵션                                                                                                                               | ★★★ |
| 5-11 | `model.to(device)` (로드 직후)                                                                         | `.to(device)`                           | `torch.load`는 기본적으로 CPU에 올린다. **로드 후 GPU로 옮기는 것을 빠뜨리면** 이후 추론에서 device 불일치 에러. 실제 노트북 주석에도 "핵심 수정"이라 표시된 함정                                                                                                                             | ★★★ |
| 5-12 | `context_size = model.pos_emb.weight.shape[0]`                                                     | `pos_emb.weight.shape[0]`               | 설정값을 따로 안 넘겨도, **위치 임베딩 테이블의 행 수 = 최대 문맥 길이**라는 성질을 이용해 모델에서 직접 알아낸다                                                                                                                                                                    | ★★  |
| 5-13 | `num_batches = min(num_batches, len(data_loader))`                                                 | `min`                                   | 평가에 5배치만 쓰고 싶은데 데이터가 3배치뿐일 수 있다. `min`으로 **둘 중 작은 값**을 택해 인덱스 초과를 막는다                                                                                                                                                                   | ★★  |
| 5-14 | `tokens_seen += input_batch.numel()`                                                               | `.numel()`                              | `numel()`은 텐서 안 원소의 **총 개수**(number of elements). `[2, 256]`이면 512. 지금까지 학습에 사용한 토큰 수를 세는 데 쓰인다                                                                                                                                         | ★★  |
| 5-15 | `if idx_next == eos_id: break`                                                                     | `eos_id` / `break`                      | 문장 끝 토큰(50256)이 나오면 더 생성하지 않고 멈춘다. 없으면 `max_new_tokens`를 다 채울 때까지 의미 없는 말을 계속 만든다                                                                                                                                                       | ★★  |
| 5-16 | `OTHER_SETTINGS = {"learning_rate": 5e-4, "num_epochs": 10, "batch_size": 2, "weight_decay": 0.1}` | 각 값                                     | 사전학습은 **바닥부터 배우므로 학습률이 크다(5e-4)**. 뒤 챕터 파인튜닝의 5e-5, DPO의 5e-6과 비교해서 "왜 점점 작아지는가"를 설명할 수 있어야 함                                                                                                                                           | ★★  |
| 5-17 | `GPT_CONFIG_124M["context_length"] = 256`                                                          | `256`                                   | 원본 GPT-2는 1024지만, 실습 시간을 줄이려고 256으로 낮췄다. **설정값 하나가 학습 속도/메모리에 직결**된다는 점                                                                                                                                                                 | ★★  |
| 5-18 | `if len(data_loader) == 0: return float("nan")`                                                    | `float("nan")`                          | 데이터가 하나도 없으면 0으로 나누게 되므로, 미리 "숫자 아님(NaN)"을 돌려주고 끝낸다                                                                                                                                                                                     | ★   |
| 5-19 | `epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))`               | `torch.linspace`                        | `linspace(시작, 끝, 개수)`는 구간을 **균등하게 나눈 값들**을 만든다. 손실 기록 개수에 맞춰 에포크 축을 만들 때 사용                                                                                                                                                             | ★   |
| 5-20 | `ax2 = ax1.twiny()` / `ax2.plot(tokens_seen, train_losses, alpha=0)`                               | `twiny()` / `alpha=0`                   | `twiny()`는 **위쪽에 x축을 하나 더** 만든다. 같은 그래프를 "에포크 기준"과 "본 토큰 수 기준"으로 동시에 읽기 위한 것. `alpha=0`은 선을 투명하게 만들어 **축만 생기게** 하는 트릭                                                                                                                   | ★   |

---

## Chapter 6 — Classification Finetuning (Metric / 데이터 구성)

파일: `Chapter_6_Excercise_Finetuning_Classification.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                                  | 빈칸 후보                                 | 의미 / 출제 의도                                                                                                                                                                             | 확률  |
| ---- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 6-1  | `predicted_labels = torch.argmax(logits, dim=-1)`                                                               | `torch.argmax` / `dim=-1`             | 모델이 뱉은 점수 `[-1.2, 3.4]`에서 **가장 큰 값의 위치(인덱스)** 를 고른다 → 1(spam). `max`는 값을, `argmax`는 **위치**를 준다는 차이가 핵심. Accuracy 계산의 1단계                                                               | ★★★ |
| 6-2  | `correct_predictions += (predicted_labels == target_batch).sum().item()`                                        | `==` / `.sum().item()`                | `==`로 예측과 정답을 비교하면 `[True, False, True]` 같은 불리언 텐서가 나온다. `.sum()`은 True를 1로 세어 맞춘 개수를 구하고, `.item()`으로 파이썬 숫자로 뽑는다. **한 줄에 3단계**가 들어있어 어디를 뚫어도 문제가 된다                                  | ★★★ |
| 6-3  | `return correct_predictions / num_examples`                                                                     | `/ num_examples`                      | **정확도(Accuracy)의 정의식**: 맞춘 개수 ÷ 전체 개수. "평가 Metric 이해" 항목의 정답 코드가 바로 이 줄                                                                                                                | ★★★ |
| 6-4  | `logits = model(input_batch)[:, -1, :]`                                                                         | `[:, -1, :]`                          | GPT는 토큰마다 출력을 내지만(`[8, 120, 2]`), 분류는 **문장 전체에 답 하나**만 필요하다. 그래서 마지막 토큰의 출력만 뽑아 `[8, 2]`로 만든다.<br>마지막 토큰을 쓰는 이유: 인과 마스킹 때문에 **마지막 토큰만이 문장 전체를 다 본 상태**이기 때문                            | ★★★ |
| 6-5  | `loss = torch.nn.functional.cross_entropy(logits, target_batch)`                                                | `cross_entropy`                       | 분류 손실 함수. Ch5(사전학습)에서는 토큰마다 정답이 있어 `flatten(0,1)`로 펴줬지만, 여기서는 문장당 정답 1개라 **평탄화 없이 바로** 넣는다. 두 버전을 나란히 놓고 비교하는 문제가 나오기 쉽다                                                               | ★★★ |
| 6-6  | `num_spam = df[df["Label"] == "spam"].shape[0]`                                                                 | `.shape[0]` / `"spam"`                | pandas에서 조건에 맞는 행만 골라낸 뒤 `.shape[0]`(행 개수)로 몇 개인지 센다. 스팸이 소수 클래스라 **이 개수가 기준**이 된다                                                                                                     | ★★★ |
| 6-7  | `ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)`                                      | `.sample(num_spam` / `random_state`   | 정상 메일이 스팸보다 훨씬 많으면, 모델이 무조건 "정상"이라고만 해도 정확도가 높게 나와버린다. 그래서 **많은 쪽을 적은 쪽 개수만큼 무작위로 줄인다(언더샘플링)**. `random_state=123`은 매번 같은 결과를 얻기 위한 시드                                                 | ★★★ |
| 6-8  | `balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])`                                              | `pd.concat`                           | 줄인 정상 메일 + 전체 스팸을 합쳐 **50:50 균형 데이터셋**을 만든다                                                                                                                                            | ★★  |
| 6-9  | `balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})`                                        | `.map({"ham": 0, "spam": 1})`         | 손실 함수는 문자열을 못 먹는다. `"ham"→0`, `"spam"→1`로 **라벨 인코딩**해야 한다. `.map()`은 딕셔너리 규칙대로 값을 바꿔주는 pandas 함수                                                                                       | ★★★ |
| 6-10 | `df = df.sample(frac=1, random_state=123).reset_index(drop=True)`                                               | `frac=1` / `reset_index(drop=True)`   | `frac=1`은 "**전체의 100%를 뽑아라**" = 순서만 뒤섞기. 합친 직후에는 앞쪽이 전부 ham, 뒤쪽이 전부 spam이라 그냥 자르면 한쪽만 몰린다.<br>섞으면 인덱스가 뒤죽박죽이므로 `reset_index(drop=True)`로 0부터 다시 매긴다(`drop=True`는 옛 인덱스를 열로 남기지 않겠다는 뜻) | ★★★ |
| 6-11 | `train_end = int(len(df) * train_frac)`                                                                         | `int(len(df) * train_frac)`           | 학습셋이 끝나는 위치. 실수 결과를 `int()`로 정수화                                                                                                                                                       | ★★★ |
| 6-12 | `validation_end = train_end + int(len(df) * validation_frac)`                                                   | `train_end +`                         | 검증셋 끝 위치는 **학습셋 끝에서 이어서** 계산해야 한다. `train_end +`를 빠뜨리면 학습셋과 검증셋이 겹쳐버린다 — 대표적인 함정                                                                                                       | ★★★ |
| 6-13 | `train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)`                                        | `0.7, 0.1`                            | 인자가 두 개뿐인데 결과는 세 개다. **테스트셋은 나머지 전부(0.2)** 로 자동 결정된다는 구조 이해                                                                                                                            | ★★  |
| 6-14 | `val_dataset = SpamDataset(csv_file=..., max_length=train_dataset.max_length, tokenizer=tokenizer)`             | `train_dataset.max_length`            | **가장 중요한 데이터 구성 함정.** 검증셋 안에서 최장 길이를 따로 구하면 학습셋과 입력 길이가 달라져 비교가 불공정해지고 shape도 흔들린다. 따라서 **학습셋 기준을 그대로 물려받아야** 한다                                                                       | ★★★ |
| 6-15 | `train_dataset = SpamDataset(..., max_length=None, ...)`                                                        | `None`                                | 학습셋만 `None`을 줘서 "**데이터 안의 가장 긴 문장에 맞춰 자동 결정**"하게 한다. 6-14와 짝으로 외울 것                                                                                                                    | ★★  |
| 6-16 | train_loader `shuffle=True, drop_last=True` / val·test `drop_last=False`                                        | `True` / `False`                      | 5-4와 같은 논리의 분류 버전. 검증·테스트에서 `drop_last=False`인 이유는 **한 샘플도 빠뜨리지 않고 평가**해야 정확도가 정확하기 때문                                                                                                 | ★★★ |
| 6-17 | `torch.tensor(encoded, dtype=torch.long)` / `torch.tensor(label, dtype=torch.long)`                             | `dtype=torch.long`                    | `long`은 64비트 **정수** 타입. 임베딩 층은 "몇 번 단어인지"를 정수로 받아야 하고, `cross_entropy`의 정답도 정수 클래스 번호여야 한다. `float`로 주면 즉시 에러                                                                          | ★★  |
| 6-18 | `df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])`                              | `sep="\t"` / `header=None` / `names=` | 파일이 콤마가 아닌 **탭으로 구분**된 형식이라 `sep="\t"`. 첫 줄이 제목이 아니라 바로 데이터라서 `header=None`, 대신 `names=`로 열 이름을 직접 붙여준다. pandas 활용 문항의 전형                                                              | ★★  |
| 6-19 | `optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)`                                  | `5e-5`                                | 이미 잘 학습된 모델을 **살짝만 조정**하는 것이므로 학습률을 작게 잡는다. 크게 주면 기존 지식이 망가진다(catastrophic forgetting)                                                                                                 | ★★  |
| 6-20 | `assert train_dataset.max_length <= BASE_CONFIG["context_length"]`                                              | `<=`                                  | 데이터 길이가 모델의 최대 문맥(1024)을 넘으면 위치 임베딩 범위를 벗어나 에러. 학습 시작 전에 확인                                                                                                                            | ★★  |
| 6-21 | `"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}` / `"gpt2-medium (355M)": {1024, 24, 16}` | 각 값                                   | 모델이 커진다는 것은 **벡터 차원·층 수·헤드 수가 함께 커지는 것**임을 보여주는 표                                                                                                                                      | ★★  |
| 6-22 | `train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)`                     | `num_batches=eval_iter`               | 매 에포크 끝에 정확도를 재는데, 전체를 다 보면 느리므로 **일부 배치만** 확인한다                                                                                                                                       | ★★  |
| 6-23 | `model.eval()` (calc_accuracy_loader 첫 줄)                                                                       | `eval()`                              | 정확도를 잴 때 드롭아웃이 켜져 있으면 매번 값이 달라진다. 평가 함수 첫 줄에 반드시 등장                                                                                                                                    | ★★  |

---

## Chapter 6 — LoRA

파일: `Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                 | 빈칸 후보                               | 의미 / 출제 의도                                                                                                                                    | 확률  |
| ---- | ------------------------------------------------------------------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| L-1  | `nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))`                             | `kaiming_uniform_` / `math.sqrt(5)` | A 행렬은 **랜덤한 작은 값**으로 시작해야 학습이 진행된다(둘 다 0이면 기울기도 0이라 영원히 안 배움). Kaiming 초기화는 층을 통과해도 값의 크기가 유지되도록 설계된 표준 방식                                    | ★★★ |
| L-2  | `self.B = nn.Parameter(torch.zeros(rank, out_dim))`                            | `torch.zeros`                       | **LoRA 최대 출제 포인트.** B를 0으로 두면 `A @ B = 0`이므로 학습 시작 시점에 **원래 모델과 출력이 완전히 동일**하다. 즉 "일단 원본을 망치지 않고 시작해서 서서히 배운다". A와 B 중 하나만 0인 이유를 설명할 수 있어야 함 | ★★★ |
| L-3  | `for name, module in model.named_children():`                                  | `named_children()`                  | 모델 안의 **바로 아래 자식 층들**을 (이름, 객체) 쌍으로 하나씩 꺼낸다. 이름이 있어야 나중에 그 자리에 새 층을 꽂아 넣을 수 있다                                                                | ★★★ |
| L-4  | `setattr(model, name, LinearWithLoRA(module, rank, alpha))`                    | `setattr`                           | `setattr(객체, "이름", 값)`은 `객체.이름 = 값`을 **문자열 이름으로** 하는 것. 층 이름을 미리 알 수 없으니 이 방식으로 **기존 Linear를 LoRA 버전으로 갈아끼운다**                                | ★★★ |
| L-5  | `else: replace_linear_with_lora(module, rank, alpha)`                          | `module` (재귀 호출)                    | 모델은 층 안에 층이 들어있는 **양파 구조**다. Linear가 아니면 그 안으로 한 겹 더 들어가 같은 작업을 반복 — 이것이 재귀. 이 줄이 없으면 최상위 층만 바뀌고 내부는 그대로 남는다                                  | ★★★ |
| L-6  | `total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)` | `p.numel()` / `p.requires_grad`     | `if p.requires_grad`로 **학습되는 파라미터만** 걸러내고 `numel()`로 개수를 세어 합산한다. LoRA의 장점(전체 1억2천만 중 수십만 개만 학습)을 숫자로 확인하는 코드                                 | ★★★ |
| L-7  | `self.A = nn.Parameter(torch.empty(in_dim, rank))`                             | `(in_dim, rank)`                    | 큰 행렬 `(in_dim × out_dim)`을 통째로 학습하는 대신, **가는 허리(rank=16)를 가진 두 행렬**로 쪼갠다. `(in,16) @ (16,out)`이라 파라미터가 훨씬 적다                                  | ★★★ |
| L-8  | `self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)`  | `in_features` / `out_features`      | 원래 Linear가 가진 입출력 크기를 그대로 물려받아야 **더할 때 shape가 맞는다**. 직접 숫자를 쓰지 않고 속성에서 읽어오는 것이 포인트                                                            | ★★  |
| L-9  | `self.linear = linear`                                                         | `linear`                            | 원본 층을 **버리지 않고 그대로 품는다**. 원본은 동결(freeze)된 채 남고 LoRA만 학습되므로, 나중에 LoRA만 떼어내면 원래 모델로 복귀 가능                                                       | ★★  |
| L-10 | `LORA_RANK = 16` / `LORA_ALPHA = 16` / `LEARNING_RATE = 8e-4`                  | 각 값                                 | `rank`는 허리 굵기(클수록 표현력↑, 파라미터↑), `alpha/rank`는 LoRA 출력의 세기 조절. 학습할 파라미터가 적고 새로 시작하는 값이라 **학습률을 크게(8e-4)** 잡는다 — 일반 파인튜닝 5e-5의 16배              | ★★  |
| L-11 | `model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)`  | `768` / `num_classes`               | 5만 단어를 예측하던 머리를 떼고 **2개 클래스 머리**로 교체. 새로 만든 층이라 자동으로 학습 대상(`requires_grad=True`)이 된다                                                          | ★★  |
| L-12 | `x @ self.A @ self.B` 의 shape 흐름: `(B,T,in) → (B,T,r) → (B,T,out)`             | 서술형                                 | 입력이 **좁은 통로(rank)를 통과했다가 다시 넓어지는** 구조. 최종 shape가 원래 Linear 출력과 같아야 `linear(x) + lora(x)` 덧셈이 성립한다                                             | ★★★ |

---

## Chapter 7 — Instruction Tuning (SFT)

파일: `Chapter_7_Exercise_Follow_Instructions.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                                                                  | 빈칸 후보                                       | 의미 / 출제 의도                                                                                                                                                  | 확률  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 7-1  | `f"\n\n### Instruction:\n{entry['instruction']}"`                                                                               | `### Instruction:` / `entry['instruction']` | **Alpaca 프롬프트 템플릿**. 모델에게 "여기부터가 지시사항이다"를 알려주는 고정 구분자. 학습 때와 추론 때 **똑같은 형식**을 써야 모델이 알아듣는다 — 형식이 조금이라도 다르면 성능이 급락                                           | ★★★ |
| 7-2  | `input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""`                                                    | `if entry["input"] else ""`                 | 어떤 지시는 추가 입력이 있고("이 문장을 번역하라: ~") 어떤 것은 없다("수도를 말하라"). 파이썬 삼항 연산자로 **있을 때만 섹션을 붙이고 없으면 빈 문자열**로 처리                                                          | ★★★ |
| 7-3  | `response_text = f"\n\n### Response:\n{entry['output']}"`                                                                       | `### Response:` / `entry['output']`         | 정답이 시작되는 지점을 알리는 구분자. 추론할 때 모델 출력에서 이 표시 뒤쪽만 잘라내 답으로 쓴다(7-15와 연결)                                                                                           | ★★★ |
| 7-4  | `full_text = instruction_plus_input + response_text`                                                                            | `+`                                         | **핵심 개념**: SFT는 질문과 답을 따로 주는 게 아니라 **하나의 긴 문장으로 이어붙여** 넣고, 평소처럼 "다음 토큰 맞추기"로 학습한다. 그 과정에서 모델이 자연스럽게 "지시 뒤에는 응답이 온다"를 배운다                                    | ★★★ |
| 7-5  | `batch_max_length = max(len(item)+1 for item in batch)`                                                                         | `max(...)` / `+1`                           | 문장마다 길이가 제각각이라 텐서로 못 묶는다. 그래서 **그 배치 안에서 가장 긴 것**에 길이를 맞춘다(동적 패딩). 전체 데이터 기준이 아니라 **배치별로** 정한다는 게 효율의 핵심. `+1`은 문장 끝 토큰 자리를 미리 확보하는 것                       | ★★★ |
| 7-6  | `mask = targets == pad_token_id`                                                                                                | `== pad_token_id`                           | 정답 텐서에서 **어디가 진짜 내용이고 어디가 채움(padding)인지** 표시한 True/False 지도를 만든다                                                                                            | ★★★ |
| 7-7  | `indices = torch.nonzero(mask).squeeze()`                                                                                       | `torch.nonzero` / `.squeeze()`              | `nonzero`는 **True인 위치의 인덱스 목록**을 돌려준다. 결과가 `[[5],[6],[7]]` 형태라 `squeeze()`로 `[5,6,7]`처럼 납작하게 편다                                                             | ★★★ |
| 7-8  | `def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, ...)`                                                      | `-100` / `50256`                            | 패딩 자리까지 정답으로 학습하면 모델이 "빈칸 뱉기"를 배워버린다. 그래서 그 자리를 `-100`으로 바꾸는데, **`-100`은 파이토치 `cross_entropy`의 기본 `ignore_index`** 라서 자동으로 손실 계산에서 제외된다. 왜 하필 -100인지가 단골 질문 | ★★★ |
| 7-9  | `inputs_tensor = torch.stack(inputs_lst).to(device)`                                                                            | `torch.stack` / `.to(device)`               | 길이를 맞춘 개별 텐서 8개를 `stack`으로 **새 배치 축을 만들어 쌓아** `[8, L]`로 만들고 GPU로 보낸다                                                                                        | ★★★ |
| 7-10 | `customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)`                                    | `partial`                                   | DataLoader는 `collate_fn`에 배치 하나만 넘겨주므로 `device` 같은 추가 인자를 전달할 방법이 없다. `functools.partial`은 **일부 인자를 미리 고정한 새 함수**를 만들어 이 문제를 푼다. "library 활용" 항목의 대표 코드     | ★★★ |
| 7-11 | `train_loader = DataLoader(train_dataset, batch_size=..., collate_fn=customized_collate_fn, shuffle=True, drop_last=True, ...)` | `collate_fn=`                               | 기본 DataLoader는 길이가 다른 샘플을 묶다가 에러를 낸다. `collate_fn`은 "**배치를 어떻게 묶을지 내가 직접 정하겠다**"는 인자                                                                        | ★★★ |
| 7-12 | `train_portion = int(len(data) * 0.85)` / `test_portion = int(len(data) * 0.1)`                                                 | `0.85` / `0.1`                              | 85% 학습 / 10% 테스트 / 나머지 5% 검증. 비율의 합이 1이 되도록 나머지를 계산하는 구조                                                                                                    | ★★  |
| 7-13 | `test_data = data[train_portion:train_portion + test_portion]`                                                                  | `train_portion + test_portion`              | 세 덩어리로 자를 때 **시작점은 앞 덩어리 끝** 이어야 겹치지 않는다. 6-12와 같은 유형                                                                                                       | ★★  |
| 7-14 | `inputs = inputs[:allowed_max_length]` / `targets = targets[:allowed_max_length]`                                               | `[:allowed_max_length]`                     | 너무 긴 샘플이 하나만 있어도 배치 전체가 커져 메모리가 터진다. 1024로 잘라 상한을 건다. **입력과 정답을 똑같이** 잘라야 짝이 유지된다                                                                           | ★★  |
| 7-15 | `response_text = generated_text[len(input_text):].replace("### Response:", "").strip()`                                         | `len(input_text):`                          | 모델 출력에는 우리가 넣은 프롬프트가 그대로 앞에 붙어 나온다. **입력 글자 수만큼 잘라내야** 순수 답변만 남는다                                                                                           | ★★  |
| 7-16 | `generate(model=..., max_new_tokens=256, context_size=BASE_CONFIG["context_length"], eos_id=50256)`                             | `eos_id=50256`                              | 답이 끝났는데도 계속 생성하지 않도록 종료 토큰을 알려준다. 50256은 GPT-2의 `<\|endoftext\|>`                                                                                           | ★★  |
| 7-17 | `optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)` / `num_epochs = 2`                            | `0.00005` / `2`                             | 이미 방대한 지식을 가진 모델이라 **작은 학습률로 짧게(2에포크)** 돌린다. 오래 돌리면 오히려 원래 능력을 잃는다                                                                                          | ★★  |
| 7-18 | `self.encoded_texts.append(tokenizer.encode(full_text))`                                                                        | `tokenizer.encode`                          | 학습 중에 매번 토큰화하면 느리므로 `__init__`에서 **미리 전부 토큰화**해 둔다(pre-tokenize). 속도 최적화 관용구                                                                                | ★★  |
| 7-19 | `CHOOSE_MODEL = "gpt2-medium (355M)"` / `model_name = "gpt2-medium-355M.pth"`                                                   | 값                                           | 지시 따르기는 단순 분류보다 어려워 **더 큰 모델(medium)** 을 쓴다. Ch6이 small인 것과 대비                                                                                              | ★   |

---

## Chapter 7 — DPO (Preference Tuning)

파일: `Chapter_7_Exercise_Follow_Instructions_dpo.ipynb`

| #    | 원본 코드 (그대로 출제)                                                                        | 빈칸 후보                                                | 의미 / 출제 의도                                                                                                                                                                                         | 확률  |
| ---- | ------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| D-1  | `mask = torch.ones(len(padded)).bool()`                                               | `torch.ones` / `.bool()`                             | 일단 **전부 True(=다 사용)** 인 지도를 만들어 두고, 이후에 필요 없는 곳만 False로 끄는 방식. `ones` → `.bool()`로 1을 True로 바꾼다                                                                                                    | ★★★ |
| D-2  | `mask[len(sequence):] = False`                                                        | `len(sequence):` / `False`                           | 원래 문장 길이 **이후는 전부 패딩**이므로 손실 계산에서 제외한다                                                                                                                                                             | ★★★ |
| D-3  | `if mask_prompt_tokens: mask[:prompt.shape[0] + 2] = False`                           | `prompt.shape[0] + 2`                                | **DPO의 핵심 아이디어.** 질문(prompt)은 chosen과 rejected가 완전히 동일하므로 학습해봐야 의미가 없다. **답변 부분만 비교**해야 선호도가 제대로 학습된다. `+2`는 프롬프트와 답변 사이의 `\n\n` 두 토큰만큼 더 건너뛰는 보정                                                  | ★★★ |
| D-4  | `log_probs = F.log_softmax(logits, dim=-1)`                                           | `log_softmax` / `dim=-1`                             | 확률을 그대로 쓰면 여러 토큰의 확률을 곱하다가 0에 수렴해 계산이 무너진다. **로그를 취하면 곱셈이 덧셈**이 되어 안정적이다. `log(softmax(x))`를 따로 하는 것보다 `log_softmax` 한 번이 수치적으로 안전                                                                 | ★★★ |
| D-5  | `torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)`       | `unsqueeze(-1)` / `.squeeze(-1)` / `input=log_probs` | 5만 개 단어의 확률 중 **실제 정답 단어 하나의 확률만** 뽑아내는 작업. `gather`는 "index가 가리키는 자리의 값만 꺼내라"는 함수.<br>`index`는 `input`과 차원 수가 같아야 해서 `unsqueeze(-1)`로 축을 하나 늘렸다가, 꺼낸 뒤 `squeeze(-1)`로 다시 줄인다 — **늘렸다 줄이는 쌍**으로 기억 | ★★★ |
| D-6  | `mask = selection_mask[:, 1:].clone()`                                                | `[:, 1:]`                                            | 라벨을 한 칸 밀었으므로(`labels[:, 1:]`) **마스크도 똑같이 한 칸 밀어야** 서로 가리키는 위치가 맞는다. 한쪽만 밀면 엉뚱한 토큰을 마스킹하게 됨                                                                                                        | ★★★ |
| D-7  | `avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)`                            | `sum(-1) / mask.sum(-1)`                             | 그냥 `.mean()`을 쓰면 패딩 자리(0)까지 개수에 포함되어, **긴 답변이 무조건 불리해진다**. 유효 토큰 수(`mask.sum`)로만 나눠야 길이에 공정한 비교가 된다                                                                                                | ★★★ |
| D-8  | `with torch.no_grad():` (reference 모델 계산 구간)                                          | `no_grad()`                                          | reference 모델은 **기준점이므로 절대 변하면 안 된다**. 기울기를 아예 계산하지 않아 메모리도 절반으로 아낀다. DPO에 모델이 두 개인 이유와 함께 묻는 문제가 많다                                                                                                | ★★★ |
| D-9  | `reference_model.eval()`                                                              | `eval()`                                             | 드롭아웃을 꺼서 기준 모델의 출력이 **항상 같은 값**이 되게 한다. 기준이 흔들리면 비교 자체가 무의미                                                                                                                                        | ★★★ |
| D-10 | `chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()`       | `.detach()`                                          | 이 값은 **로그 출력용**이지 학습에 쓰이지 않는다. `.detach()`로 계산 그래프에서 떼어내야 불필요한 역전파 경로가 생기지 않는다                                                                                                                     | ★★★ |
| D-11 | `padded = sequence + [pad_token_id] * (max_length_common - len(sequence))`            | `max_length_common - len(sequence)`                  | **모자란 만큼만** 패딩 토큰을 이어붙인다. `리스트 * 정수`는 파이썬에서 반복 연결 문법                                                                                                                                               | ★★★ |
| D-12 | `current_max = max(len(item[key]) + 1 for item in batch)`                             | `+ 1`                                                | chosen과 rejected는 길이가 다르지만 **같은 길이로 맞춰야** 나중에 비교가 쉽다. 두 그룹을 모두 훑어 공통 최대 길이를 구한다                                                                                                                    | ★★  |
| D-13 | `f"Train Margin {res['train_chosen_reward']-res['train_rejected_reward']:.3f}"`       | `chosen - rejected`                                  | **DPO의 평가 Metric = Margin**. "좋은 답 점수 − 나쁜 답 점수"가 클수록 모델이 선호를 잘 구분한다는 뜻. 정확도(Ch6)나 손실(Ch5)과는 다른 종류의 지표라는 점이 출제 포인트                                                                                 | ★★★ |
| D-14 | `PreferenceDataset` 저장 구조: `{"prompt":…, "chosen":…, "rejected":…}`                   | 키 이름                                                 | SFT는 (질문, 정답) 2개였지만 DPO는 **(질문, 좋은 답, 나쁜 답) 3개가 한 세트**. "데이터 구성 원리" 문항에서 SFT와의 차이를 묻기 좋다                                                                                                           | ★★★ |
| D-15 | `return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()`                | `.mean()`                                            | 배치 안 샘플별로 나온 값들을 평균 내 스칼라 하나로 만든다. `backward()`는 스칼라에만 호출할 수 있기 때문                                                                                                                                 | ★★  |
| D-16 | `optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, ...)` / `beta=0.1` | `5e-6` / `0.1`                                       | DPO는 이미 SFT까지 끝난 모델을 **아주 미세하게** 다듬는 단계라 학습률이 극도로 작다. `beta`는 "기준 모델에서 얼마나 벗어나도 되는지"를 조절하며, 크면 보수적·작으면 과감                                                                                          | ★★  |
| D-17 | `policy_model.train()` / `policy_model.eval()` (평가 진입·복귀)                             | `train()` / `eval()`                                 | **policy 모델만** 모드를 오간다. reference는 처음부터 끝까지 `eval()` 고정 — 둘의 취급이 다르다는 게 포인트                                                                                                                        | ★★  |
| D-18 | `tokens_seen += batch["chosen"].numel()`                                              | `.numel()`                                           | 학습 진행량을 토큰 수로 기록. 그래프의 보조 x축에 쓰인다                                                                                                                                                                  | ★   |
| D-19 | `policy_model`과 `reference_model` 둘 다 동일 `model_path`로 `load_state_dict`              | 서술형                                                  | 학습 시작 시점에 **두 모델은 완전히 같은 가중치**다. 이후 policy만 업데이트되며 둘 사이가 벌어지고, 그 차이가 곧 "얼마나 학습됐는가"가 된다                                                                                                             | ★★★ |

---

# 2. 별도 표 — 기존 `????`(TODO) 구간 정리

> 이미 노트북에 정답 주석이 노출된 부분. **그대로 나올 수도, 인접 코드로 옮겨 변형될 수도** 있다.

| 챕터   | TODO 코드 (정답)                                                                                                | 핵심 개념 (자세히)                                                                                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ch2  | `tokenizer.encode(...)` / `tokenizer.decode(...)`                                                           | **글자 ↔ 숫자 변환의 양방향.** `encode`는 사람 글 → 토큰 ID 리스트, `decode`는 그 반대. 모든 LLM 파이프라인의 입구와 출구이며, 두 메서드 이름을 바꿔 쓰면 즉시 에러                                                        |
| Ch2  | `y = enc_sample[1:context_size+1]`                                                                          | **GPT 학습의 정의 그 자체.** 입력이 `[A,B,C,D]`면 정답은 `[B,C,D,E]` — 즉 **한 칸 오른쪽으로 민 것**. "다음 토큰 예측"이 코드로는 그저 인덱스 1칸 이동이라는 사실이 핵심                                                  |
| Ch2  | `input_chunk = token_ids[i : i + max_length]`                                                               | 슬라이딩 윈도우로 잘라낸 **입력 조각**. 시작 `i`, 길이 `max_length`                                                                                                                      |
| Ch2  | `target_chunk = token_ids[i + 1 : i + max_length + 1]`                                                      | 같은 조각을 **통째로 1칸 밀어** 만든 정답 조각. 시작과 끝 **양쪽 모두 +1**이어야 길이가 유지된다 — 한쪽만 +1 하는 실수를 노린 문제                                                                                   |
| Ch2  | `return self.input_ids[idx], self.target_ids[idx]`                                                          | `__getitem__`은 **같은 idx의 (입력, 정답) 쌍**을 반환해야 한다. 인덱스가 어긋나면 엉뚱한 정답으로 학습된다                                                                                               |
| Ch2  | `DataLoader(dataset, ...)`                                                                                  | DataLoader의 **첫 번째 위치 인자는 항상 Dataset 객체**. 뒤의 batch_size·shuffle 등은 모두 키워드 인자                                                                                         |
| Ch2  | `torch.nn.Embedding(vocab_size, output_dim)` / `embedding_layer(input_ids)`                                 | `nn.Embedding`은 `(단어 종류 수 × 벡터 차원)` 크기의 **거대한 조회 표(lookup table)**. 정수 ID를 넣으면 해당 행 벡터를 꺼내준다. 곱셈이 아니라 **조회**라는 점이 핵심                                                  |
| Ch3  | `nn.Linear(d_in, d_out, bias=qkv_bias)` ×3                                                                  | 같은 입력 x를 **세 가지 역할**로 변환한다. Query=내가 찾는 것, Key=내가 가진 표지, Value=실제 내용. 세 층 모두 구조는 같고 **학습된 가중치만 다르다**                                                                  |
| Ch3  | `keys = self.W_key(x)` / `queries = self.W_query(x)` / `values = self.W_value(x)`                           | **셀프(self) 어텐션**인 이유가 여기 있다. Q·K·V가 전부 **같은 입력 x**에서 나온다. 번역기처럼 다른 문장에서 나오면 크로스 어텐션                                                                                   |
| Ch3  | `attn_scores = queries @ keys.transpose(1, 2)`                                                              | Q와 K의 **내적 = 두 단어가 얼마나 관련 있는가**. 방향이 비슷한 벡터끼리 내적이 크다. 결과는 `(토큰수 × 토큰수)` 관계 표                                                                                          |
| Ch3  | `masked_fill_(mask, -torch.inf)`                                                                            | 미래 위치를 **0이 아니라 −무한대**로 채운다. 0으로 채우면 softmax 후에도 확률이 남지만, −inf는 `exp(-inf)=0`이라 **확률이 정확히 0**이 된다. 이 이유를 묻는 문제가 자주 나온다                                                |
| Ch3  | `attn_weights = torch.softmax(...)`                                                                         | 점수를 **합이 1인 확률 분포**로 바꾼다. "이 단어가 각 단어를 몇 % 참고할지"의 배분표                                                                                                                 |
| Ch3  | `context_vec = attn_weights @ values`                                                                       | 확률만큼 Value를 **가중 평균**한다. 결과는 "주변 문맥이 섞여 들어간 새 벡터" — 어텐션의 최종 산출물                                                                                                       |
| Ch3  | `context_vecs = ca(batch)`                                                                                  | `nn.Module`은 `ca.forward(x)`가 아니라 **`ca(x)`로 호출**하는 것이 규칙. 직접 forward를 부르면 hook 등이 동작하지 않는다                                                                           |
| Ch4  | `norm_x = (x - mean) / torch.sqrt(var + self.eps)` / `self.scale * norm_x + self.shift`                     | **정규화 2단계**: ① 평균을 빼고 표준편차로 나눠 평균0·분산1로 만든다 ② 학습 가능한 `scale`·`shift`로 다시 자유롭게 조정. ②가 없으면 표현력이 제한된다                                                                    |
| Ch4  | `nn.Linear(emb_dim, 4*emb_dim)` / `GELU()` / `nn.Linear(4*emb_dim, emb_dim)`                                | **넓혔다 줄이는 구조**(768→3072→768). 중간에서 차원을 4배로 키워 복잡한 패턴을 표현할 공간을 만들고 다시 줄인다. 사이에 비선형(GELU)이 없으면 두 Linear가 하나로 합쳐져 **의미가 사라진다**                                           |
| Ch4  | `x = self.att(x)` / `x = self.ff(x)`                                                                        | 트랜스포머 블록의 두 기둥. **어텐션 = 토큰끼리 정보 교환**, **FFN = 각 토큰 안에서 정보 가공**. 역할이 완전히 다르다                                                                                           |
| Ch4  | `nn.Embedding(cfg["vocab_size"], ...)` / `cfg["context_length"]` / `out_head → cfg["vocab_size"]`           | 같은 `nn.Embedding`이라도 **토큰용은 `vocab_size`, 위치용은 `context_length`**. 그리고 출력 헤드는 다시 `vocab_size`로 돌아온다(어떤 단어가 나올지 예측하므로)                                                 |
| Ch4  | `x = tok_embeds + pos_embeds` → `self.trf_blocks(x)` → `self.out_head(x)`                                   | GPT 전체 흐름 3단계. **`(B,T)` 정수 → `(B,T,768)` 벡터 → 블록 12개 통과 → `(B,T,50257)` 점수**. 이 shape 변화를 말로 설명할 수 있어야 한다                                                            |
| Ch4  | `with torch.no_grad():` / `logits[:, -1, :]` / `torch.argmax(...)` / `torch.cat(...)`                       | **텍스트 생성 4단계**: ① 기울기 끄기 ② 마지막 위치 점수만 꺼내기 ③ 가장 높은 것 고르기(greedy) ④ 뒤에 이어붙여 다음 입력으로. 이 루프가 곧 "자기회귀 생성"                                                                  |
| Ch5  | `F.cross_entropy(logits.flatten(0,1), target_batch.flatten())`                                              | `cross_entropy`는 `(샘플수, 클래스수)` 2차원 입력만 받는데 우리 logits는 `(B,T,V)` 3차원이다. `flatten(0,1)`로 **배치와 토큰 축을 합쳐** `(B*T, V)`로 만들고, 정답도 `(B*T,)`로 편다. **모든 토큰을 독립 문제로 취급**한다는 의미 |
| Ch5  | `model.train()` → `optimizer.zero_grad()` → `calc_loss_batch(...)` → `loss.backward()` → `optimizer.step()` | **딥러닝 학습의 5단계 정형구(최빈출).** ① 학습모드 ② 이전 기울기 지우기 ③ 예측하고 오차 재기 ④ 오차를 거꾸로 흘려 기울기 계산 ⑤ 그 기울기로 가중치 수정. **순서가 하나라도 바뀌면 학습이 망가진다**                                             |
| Ch5  | `logits[:, -1, :]` / `torch.argmax` / `torch.cat((idx, idx_next), dim=1)`                                   | 생성 루프. `dim=1`로 이어붙이는 이유는 **토큰 축(길이 방향)** 으로 늘려야 하기 때문                                                                                                                |
| Ch5  | `torch.tensor(encoded).unsqueeze(0)` / `token_ids.squeeze(0)`                                               | 모델 입출력의 **배치 차원 관리**. 문장 1개도 `[1, L]`로 포장해야 하고, 결과를 사람에게 보여줄 땐 다시 벗겨야 한다                                                                                              |
| Ch5  | `AdamW(model.parameters(), lr=..., weight_decay=...)`                                                       | `model.parameters()`는 **학습할 모든 가중치를 옵티마이저에 넘기는** 관용구. `weight_decay`는 가중치가 지나치게 커지지 않게 눌러 과적합을 막는 규제                                                                  |
| Ch6  | `encoded_text[:self.max_length]` / `encoded_text + [pad_token_id] * (...)`                                  | **길이 맞추기 2단계**: 긴 건 자르고(truncation), 짧은 건 채운다(padding). 이 두 개가 있어야 서로 다른 문장들을 하나의 텐서로 묶을 수 있다                                                                         |
| Ch6  | `label = self.data.iloc[index]["Label"]`                                                                    | `iloc[index]`로 **번호로 행을 꺼내고**, `["Label"]`로 정답 열을 지정한다. 열 이름이 6-18의 `names=["Label","Text"]`에서 왔다는 연결고리                                                               |
| Ch6  | `logits = model(input_batch)[:, -1, :]`                                                                     | 분류에서 **마지막 토큰만 쓰는 이유**: 인과 마스킹 때문에 마지막 토큰만이 앞의 모든 단어를 본 상태라 문장 전체를 대표한다                                                                                               |
| Ch6  | `param.requires_grad = False` → 헤드 교체 → `trf_blocks[-1]` / `final_norm` → `True`                            | **선택적 파인튜닝 전략.** ① 전부 얼려서 사전학습 지식 보존 ② 새 분류 헤드는 자동으로 학습 대상 ③ 마지막 블록과 최종 정규화만 추가로 풀어줌. 전체를 학습하는 것보다 빠르고 과적합도 적다. `[-1]`이 **마지막** 블록을 뜻하는 파이썬 음수 인덱스                    |
| Ch6  | `out_features=num_classes`                                                                                  | 출력 크기를 50,257 → **2**로 바꾸는 것이 "언어모델 → 분류기" 전환의 전부. 몸통은 그대로 재사용                                                                                                        |
| LoRA | `torch.empty(in_dim, rank)` / `torch.zeros(rank, out_dim)`                                                  | A는 `(in × r)`, B는 `(r × out)`. **곱하면 원래 크기 `(in × out)`** 가 되지만 파라미터 수는 훨씬 적다. 가운데 `r`이 두 행렬을 잇는 허리                                                                   |
| LoRA | `x = (self.alpha / self.rank) * (x @ self.A @ self.B)`                                                      | `alpha/rank`는 **LoRA가 원본에 얼마나 강하게 개입할지**의 세기 조절 손잡이. rank를 바꿔도 세기가 일정하게 유지되도록 나눠준다. 곱하는 순서 `x → A → B`도 shape상 바꿀 수 없다                                                |
| LoRA | `return self.linear(x) + self.lora(x)`                                                                      | **원본 출력은 그대로 두고 보정값만 더한다.** 원본 `linear`는 동결되어 변하지 않으므로, 학습되는 것은 오직 더해지는 부분뿐                                                                                           |
| LoRA | `isinstance(module, torch.nn.Linear)` / `replace_linear_with_lora(...)`                                     | `isinstance`로 **이 층이 Linear인지 판별**해 교체 여부를 정한다. LayerNorm이나 Embedding은 건드리지 않는다                                                                                       |
| Ch7  | `new_item += [pad_token_id]` / `padded = new_item + [pad_token_id] * (...)`                                 | 같은 토큰(50256)이 **두 가지 역할**을 겸한다. 앞은 "문장 끝" 표시, 뒤는 "길이 채우기". 이 구분이 뒤의 마스킹 로직으로 이어진다                                                                                     |
| Ch7  | `inputs = torch.tensor(padded[:-1])` / `targets = torch.tensor(padded[1:])`                                 | 하나의 시퀀스에서 **앞을 잘라 입력, 뒤를 잘라 정답**을 동시에 만든다. `[:-1]`은 마지막 빼고, `[1:]`은 첫 번째 빼고 — 결과적으로 정답이 입력보다 1칸 앞선다                                                                   |
| Ch7  | `targets[indices[1:]] = ignore_index`                                                                       | **`[1:]`이 핵심.** 첫 번째 패딩은 "여기서 문장이 끝난다"를 가르쳐야 하므로 **남겨두고**, 두 번째부터의 패딩만 −100으로 무시시킨다. `[0:]`으로 바꾸면 모델이 문장을 끝내는 법을 못 배운다                                                |
| DPO  | `labels = labels[:, 1:].clone()` / `logits = logits[:, :-1, :]`                                             | 모델은 t번째 입력으로 t+1번째를 예측하므로 **둘의 위치를 맞춰야** 한다. 라벨은 앞을 하나 버리고, 로짓은 뒤를 하나 버려 길이와 대응을 동시에 맞춘다. `.clone()`은 원본 훼손 방지                                                        |
| DPO  | `torch.gather(input=log_probs, ...)`                                                                        | **raw logits가 아니라 log_probs를 넣어야** 한다. 정규화되지 않은 점수를 뽑으면 확률로서의 의미가 없어 DPO 수식이 성립하지 않는다 — 대표적인 오답 유도                                                                    |
| DPO  | `model_logratios = model_chosen_logprobs - model_rejected_logprobs`                                         | "이 모델이 **좋은 답을 나쁜 답보다 얼마나 더 선호하는가**"를 하나의 수로 압축. 순서를 뒤집으면 학습 방향이 정반대가 된다                                                                                              |
| DPO  | `logits = model_logratios - reference_logratios`                                                            | "학습 중인 모델이 **기준 모델보다 얼마나 더 잘 구분하게 되었는가**". DPO는 절대 점수가 아니라 **기준 대비 상대적 개선**을 학습한다                                                                                     |
| DPO  | `losses = -F.logsigmoid(beta * logits)`                                                                     | 위 차이가 클수록 손실이 작아지는 형태. `logsigmoid`는 `log(sigmoid(x))`를 안정적으로 계산하며, 앞의 `-`가 있어 **손실을 줄이는 것 = 선호 구분을 키우는 것**이 된다                                                       |
| DPO  | `chosen_full_tokens = tokenizer.encode(f"...{chosen_response}")` / `{rejected_response}`                    | **프롬프트는 공통, 뒤에 붙는 답변만 다르다.** 두 시퀀스가 같은 형식으로 만들어져야 뒤에서 공정하게 비교된다                                                                                                       |

### `????` 구간 대비 요령

- 정답이 주석에 그대로 노출돼 있으므로, 시험에서는 **같은 개념의 인접 코드**로 이동해 출제될 가능성이 크다.

| TODO에 노출된 것                                    | 실제 출제 예상 위치                                                   |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `attn_scores = queries @ keys.transpose(1, 2)` | MultiHeadAttention의 `transpose(2, 3)`                         |
| `logits[:, -1, :]`                             | `idx_cond = idx[:, -context_size:]`                           |
| `param.requires_grad = False`                  | `trf_blocks[-1]` 인덱스, `sum(p.numel() ... if p.requires_grad)` |
| `nn.Linear(d_in, d_out, bias=qkv_bias)`        | `self.out_proj = nn.Linear(d_out, d_out)`                     |
| `norm_x = (x - mean) / torch.sqrt(var + eps)`  | `x.var(dim=-1, keepdim=True, unbiased=False)`                 |
| `torch.softmax(...)`                           | `F.log_softmax(logits, dim=-1)` (DPO)                         |

- 반대로 **학습 5단계**, **LayerNorm 식**, **`flatten(0, 1)`** 은 TODO에 있어도 그대로 재출제될 확률이 매우 높은 필수 암기 항목이다.

---

# 3. 최종 우선순위 Top 12 — 상세 해설

> 시간이 없다면 이 12개만. 각 항목은 **여러 챕터에 반복 등장**하므로 하나를 이해하면 여러 문제를 커버한다.

| 순위  | 항목                             | 핵심 개념                                     | 위치            |
| --- | ------------------------------ | ----------------------------------------- | ------------- |
| 1   | 학습 5단계 + 모드 전환                 | 오차를 가중치 수정으로 연결하는 표준 절차. 순서가 곧 정답         | Ch5, Ch6, DPO |
| 2   | MHA shape 체인                   | 큰 벡터를 헤드로 쪼갰다가 다시 합치는 4단계 텐서 변형           | Ch3, Ch4      |
| 3   | 인과 마스킹 + 스케일드 softmax          | 미래를 못 보게 막고, 점수를 안정적인 확률로 바꾸는 어텐션의 심장     | Ch3           |
| 4   | LayerNorm 계산식                  | 토큰 하나 안에서 값 분포를 고르게 만들어 학습을 안정화           | Ch4           |
| 5   | Residual + Pre-LN 순서           | 깊은 층에서도 기울기가 흐르게 하는 우회로와 정규화 위치           | Ch4           |
| 6   | 토큰 임베딩 + 위치 임베딩 덧셈             | "무엇을" 과 "몇 번째" 를 한 벡터로 합치는 입력 구성          | Ch2, Ch4      |
| 7   | 두 종류의 cross_entropy            | 생성(모든 토큰) vs 분류(마지막 토큰) — 목적에 따른 손실 구성 차이 | Ch5 ↔ Ch6     |
| 8   | Accuracy 4단계                   | 로짓을 정답률이라는 숫자로 환산하는 평가 Metric의 정의         | Ch6           |
| 9   | 데이터 분할과 로더 옵션                  | 학습용/평가용 데이터를 목적에 맞게 구성하는 규칙               | Ch2, Ch5, Ch6 |
| 10  | 패딩 · ignore_index · collate_fn | 길이가 제각각인 실제 데이터를 배치로 묶는 기술                | Ch6, Ch7      |
| 11  | LoRA 저랭크 어댑터                   | 원본을 얼리고 작은 보정 행렬만 학습하는 효율적 파인튜닝           | LoRA          |
| 12  | DPO 선호 학습                      | 정답 대신 "A가 B보다 낫다"로 학습시키는 방식과 그 지표         | DPO           |

---

### 1위 — 학습 5단계 + 모드 전환

```python
model.train()                                          # ①
for input_batch, target_batch in train_loader:
    optimizer.zero_grad()                              # ②
    loss = calc_loss_batch(input_batch, target_batch, model, device)   # ③
    loss.backward()                                    # ④
    optimizer.step()                                   # ⑤
```

**왜 이 순서인가**

- **① `model.train()`** — 드롭아웃을 켜고 학습 모드로 진입. 평가 후 이걸 빠뜨리면 드롭아웃이 꺼진 채 학습된다.
- **② `zero_grad()`** — 파이토치는 기울기를 **덮어쓰지 않고 계속 더한다**. 지우지 않으면 이전 배치의 기울기가 섞여 학습이 엉망이 된다. **가장 자주 나오는 빈칸.**
- **③ 손실 계산** — 예측과 정답의 차이를 하나의 숫자로.
- **④ `backward()`** — 그 숫자에서 거꾸로 각 가중치의 기울기를 구한다. **`loss`(스칼라)에서 호출**한다는 점이 포인트.
- **⑤ `step()`** — 구한 기울기로 실제 가중치를 수정. ④ 없이 ⑤만 하면 아무 일도 일어나지 않는다.

**함께 묶어 외울 것**: 평가 시에는 `model.eval()` + `with torch.no_grad():`, 끝나면 **반드시 `model.train()`으로 복귀**.
`eval()`은 *동작 방식*(드롭아웃)을 바꾸고, `no_grad()`는 *기울기 기록*을 끈다 — **다른 일을 하므로 둘 다 필요**하다.

---

### 2위 — MultiHeadAttention shape 체인

```python
keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)   # ① 쪼개기
keys = keys.transpose(1, 2)                                      # ② 헤드를 앞으로
attn_scores = queries @ keys.transpose(2, 3)                     # ③ 4D 행렬곱
context_vec = (attn_weights @ values).transpose(1, 2)            # ④ 되돌리기
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # ⑤ 합치기
```

**흐름을 shape로 따라가기** (b=2, T=6, d_out=768, H=12, head_dim=64)

| 단계                  | shape            | 설명                        |
| ------------------- | ---------------- | ------------------------- |
| 시작                  | `(2, 6, 768)`    | Linear 통과 후               |
| ① view              | `(2, 6, 12, 64)` | 768을 12헤드 × 64차원으로 분해     |
| ② transpose(1,2)    | `(2, 12, 6, 64)` | 헤드별 독립 행렬이 되도록 축 교환       |
| ③ @ transpose(2,3)  | `(2, 12, 6, 6)`  | 헤드마다 토큰끼리의 관계 표           |
| ④ transpose(1,2)    | `(2, 6, 12, 64)` | 사람이 읽는 순서로 복귀             |
| ⑤ contiguous().view | `(2, 6, 768)`    | 헤드 결합 — **출력이 입력과 같은 크기** |

**시험에서 노리는 세 곳**

1. `transpose(1,2)`와 `transpose(2,3)`의 구분 → **"항상 뒤에서 두 축"**. 3차원이면 (1,2), 4차원이면 (2,3).
2. `contiguous()` 누락 → `view` 호출 시 RuntimeError. transpose가 메모리를 실제로 옮기지 않기 때문.
3. `d_out // num_heads` → `/`가 아니라 `//`. 실수가 나오면 view에서 에러.

---

### 3위 — 인과 마스킹 + 스케일드 softmax

```python
self.register_buffer("mask", torch.triu(torch.ones(L, L), diagonal=1))
...
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

**세 가지 개념이 겹쳐 있다**

1. **마스크 만들기** — `torch.ones(L,L)`로 전부 1인 표를 만들고 `triu(diagonal=1)`로 **대각선 위쪽만** 남긴다. 대각선 위 = 아직 오지 않은 미래 토큰. `diagonal=0`이면 자기 자신까지 가려버려 계산이 무너진다.
2. **왜 −inf인가** — softmax는 `exp(점수)`를 계산하는데 `exp(-inf) = 0`이다. 0으로 채우면 `exp(0)=1`이라 오히려 확률이 남는다. **미래를 완전히 차단하려면 −inf여야 한다.**
3. **왜 √d_k로 나누는가** — 차원이 커질수록 내적 값이 커지고, 큰 값을 softmax에 넣으면 한 곳만 1이고 나머지는 0인 극단적 분포가 된다. 그러면 기울기가 거의 0이 되어 학습이 멈춘다. 표준편차를 1 근처로 되돌리는 보정이 `/√d_k`.

**`register_buffer` vs `nn.Parameter`** — 자주 나오는 비교 문항.

|                     | `nn.Parameter`        | `register_buffer` |
| ------------------- | --------------------- | ----------------- |
| 학습되는가               | O (gradient 계산됨)      | X                 |
| `state_dict`에 저장되는가 | O                     | O                 |
| `.to(device)` 따라가는가 | O                     | O                 |
| 예시                  | `scale`, `shift`, 가중치 | `mask`            |

---

### 4위 — LayerNorm 계산식

```python
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

**한 줄씩 뜯어보기**

- `dim=-1` — **한 토큰이 가진 768개 숫자끼리** 평균/분산을 낸다. 배치나 토큰 방향이 아니다. (BatchNorm은 배치 방향 — 이 차이가 자주 출제된다.)
- `keepdim=True` — 결과 shape를 `(2,6,1)`로 유지해야 `(2,6,768)`에서 뺄 때 자동 확장(브로드캐스팅)된다. `False`면 `(2,6)`이 되어 에러.
- `unbiased=False` — `n`으로 나누는 모분산. 기본값이 `True`(n−1)라서 **일부러 명시**한 것 → 빈칸으로 좋은 자리.
- `+ self.eps` — 분산이 0일 때 0으로 나누는 것을 방지.
- `scale * norm_x + shift` — 정규화로 잃은 표현력을 되찾기 위한 **학습 가능한 재조정**. 초기값 `scale=1, shift=0`이면 아무것도 바꾸지 않는 상태에서 출발한다.

---

### 5위 — Residual Connection과 Pre-LN 순서

```python
shortcut = x           # ① 원본 보관
x = self.norm1(x)      # ② 정규화 먼저 (Pre-LN)
x = self.att(x)        # ③ 어텐션
x = self.drop_shortcut(x)   # ④ 드롭아웃
x = x + shortcut       # ⑤ 원본을 더해 복원
```

(같은 5단계가 `norm2` + `ff`로 **한 번 더** 반복된다.)

**왜 필요한가**

- 층을 12개, 24개 쌓으면 역전파 과정에서 기울기가 계속 곱해지며 0에 가까워진다(기울기 소실). 입력을 출력에 그대로 더해두면 **기울기가 그 길로 손실 없이 흘러갈 수 있다.**
- `x + shortcut`은 "어텐션이 계산한 **변화량만** 원본에 더한다"는 뜻이기도 하다. 어텐션이 아무 도움이 안 되면 0을 출력해 원본을 그대로 통과시킬 수도 있다.

**Pre-LN vs Post-LN**

|                   | 순서                       | 특징                      |
| ----------------- | ------------------------ | ----------------------- |
| **Pre-LN** (이 교재) | `norm → att → +shortcut` | 학습이 안정적, warmup 없이도 잘 됨 |
| Post-LN (원논문)     | `att → +shortcut → norm` | 깊게 쌓으면 학습이 불안정          |

→ **코드 순서를 뒤섞어 놓고 배열시키는 문제**로 나오기 쉽다.

---

### 6위 — 토큰 임베딩 + 위치 임베딩

```python
tok_embeds = self.tok_emb(in_idx)                                    # (B,T) → (B,T,768)
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T,) → (T,768)
x = tok_embeds + pos_embeds                                          # (B,T,768)
```

**세 가지 포인트**

1. **두 임베딩의 첫 인자가 다르다** — 토큰용은 `vocab_size`(50257: 단어 종류 수), 위치용은 `context_length`(1024: 최대 길이). 둘 다 `nn.Embedding`이라 헷갈리기 쉬운데 **의미가 완전히 다르다.**
2. **입력이 다르다** — 토큰 임베딩에는 **토큰 ID**(`in_idx`)를, 위치 임베딩에는 **위치 번호**(`torch.arange`)를 넣는다.
3. **concat이 아니라 덧셈** — 차원이 늘어나지 않는다. `(B,T,768) + (T,768)`은 shape가 달라 보이지만 파이토치가 배치 방향으로 자동 확장해준다.

**`device=in_idx.device`를 빼면?** 입력은 GPU인데 `torch.arange` 결과는 CPU라서 `Expected all tensors to be on the same device` 에러가 난다. 실전에서 가장 흔한 실수 중 하나.

---

### 7위 — 두 종류의 cross_entropy

|              | 사전학습/SFT (Ch5, Ch7)                                     | 분류 (Ch6)                                     |
| ------------ | ------------------------------------------------------- | -------------------------------------------- |
| 코드           | `cross_entropy(logits.flatten(0,1), targets.flatten())` | `cross_entropy(model(x)[:, -1, :], targets)` |
| logits shape | `(B, T, 50257)` → `(B*T, 50257)`                        | `(B, T, 2)` → `(B, 2)`                       |
| 정답 shape     | `(B, T)` → `(B*T,)`                                     | `(B,)`                                       |
| 정답 개수        | **토큰마다 1개**                                             | **문장마다 1개**                                  |
| 의미           | 모든 위치에서 다음 단어 맞히기                                       | 문장 전체를 한 클래스로 분류                             |

**왜 이런 차이가 생기나**

- `cross_entropy`는 `(샘플수, 클래스수)` 2차원 입력만 받는다. 사전학습에서는 토큰 하나하나가 독립된 문제이므로 `flatten(0,1)`로 배치·토큰 축을 합쳐 **B×T개의 문제**로 만든다.
- 분류에서는 문장당 답이 하나뿐이므로 마지막 토큰 위치만 뽑아 그대로 넣는다. **마지막 토큰을 쓰는 이유는 인과 마스킹 때문에 그 토큰만이 문장 전체를 본 상태**이기 때문이다.

---

### 8위 — Accuracy 계산 4단계

```python
logits = model(input_batch)[:, -1, :]                              # ① (B, 2)
predicted_labels = torch.argmax(logits, dim=-1)                    # ② (B,)
correct_predictions += (predicted_labels == target_batch).sum().item()  # ③
return correct_predictions / num_examples                          # ④
```

| 단계  | 하는 일                          | 주의점                                   |
| --- | ----------------------------- | ------------------------------------- |
| ①   | 마지막 토큰의 클래스 점수만 추출            | `[:, -1, :]` — 콤마 위치와 `-1`            |
| ②   | 가장 점수가 높은 클래스의 **인덱스** 선택     | `max`는 값, `argmax`는 **위치**. `dim=-1`  |
| ③   | 정답과 비교 → True를 합산 → 파이썬 수로 변환 | `==` 결과는 불리언 텐서, `.sum()`이 True를 1로 셈 |
| ④   | 맞춘 수 ÷ 전체 수                   | **Accuracy의 정의식**                     |

**"평가 Metric 이해" 항목의 정답 코드가 바로 이 네 줄이다.** 손실(Loss)과의 차이도 함께 정리해둘 것 — 손실은 학습에 쓰이는 미분 가능한 값, 정확도는 사람이 읽는 성능 지표이며 미분이 안 되어 학습에 직접 쓸 수 없다.

---

### 9위 — 데이터 분할과 DataLoader 옵션

```python
split_idx = int(train_ratio * len(text_data))
train_loader = create_dataloader_v1(text_data[:split_idx], ..., shuffle=True,  drop_last=True)
val_loader   = create_dataloader_v1(text_data[split_idx:], ..., shuffle=False, drop_last=False)
```

| 옵션           | 학습용                    | 평가용                        | 이유                                               |
| ------------ | ---------------------- | -------------------------- | ------------------------------------------------ |
| `shuffle`    | `True`                 | `False`                    | 학습은 데이터 순서를 외우면 안 됨 / 평가는 매번 같은 결과여야 비교 가능       |
| `drop_last`  | `True`                 | `False`                    | 학습은 배치 크기를 일정하게 유지해 안정화 / 평가는 **한 샘플도 빠뜨리면 안 됨** |
| `stride`     | `context_length`(사전학습) | —                          | 중복 없이 전체를 한 번씩                                   |
| `max_length` | 자동(`None`)             | `train_dataset.max_length` | **평가셋은 학습셋 기준을 물려받아야** 공정한 비교                    |

**분할 인덱스 계산의 함정** (Ch6)

```python
train_end      = int(len(df) * train_frac)
validation_end = train_end + int(len(df) * validation_frac)   # ← train_end를 더해야 함
```

`train_end +`를 빠뜨리면 학습셋과 검증셋이 **겹쳐버린다**. 그리고 분할 전에 반드시 `df.sample(frac=1)`로 섞어야 한다 — 안 섞으면 앞쪽은 전부 ham, 뒤쪽은 전부 spam이 되어 한쪽 클래스만 든 셋이 만들어진다.

---

### 10위 — 패딩 · ignore_index · collate_fn

```python
batch_max_length = max(len(item)+1 for item in batch)      # ① 배치별 최대 길이
padded  = new_item + [pad_token_id] * (batch_max_length - len(new_item))  # ② 채우기
inputs  = torch.tensor(padded[:-1])
targets = torch.tensor(padded[1:])
mask    = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index                     # ③ 두 번째 패딩부터 무시
```

**왜 이렇게까지 하는가**

1. **길이가 제각각인 문장을 하나의 텐서로 묶어야** 배치 연산이 가능하다. 그래서 짧은 것을 채운다(padding).
2. 그런데 채운 자리까지 정답으로 학습하면 모델이 "빈칸 뱉기"를 배운다. 그래서 손실 계산에서 빼야 한다.
3. `ignore_index=-100`인 이유 — **파이토치 `cross_entropy`의 기본값이 -100**이라, 정답을 -100으로 바꿔두기만 하면 자동으로 무시된다.
4. **`indices[1:]`의 의미** — 첫 번째 패딩 토큰은 "여기서 문장이 끝난다"를 가르치는 EOS 역할이므로 **남겨두고**, 두 번째부터만 무시한다. `[0:]`으로 바꾸면 모델이 문장을 끝내는 법을 배우지 못해 끝없이 생성한다.
5. **`partial`이 필요한 이유** — DataLoader는 `collate_fn`에 배치 하나만 넘겨준다. `device`나 `allowed_max_length` 같은 추가 인자를 전달할 방법이 없으므로, `functools.partial`로 **미리 고정한 새 함수**를 만들어 넘긴다.

**동적 패딩 vs 고정 패딩** — Ch6은 데이터셋 전체 최장 길이로 고정(`self.max_length`), Ch7은 **배치마다 다르게**(`batch_max_length`). 후자가 메모리 효율이 훨씬 좋다.

---

### 11위 — LoRA 저랭크 어댑터

```python
self.A = nn.Parameter(torch.empty(in_dim, rank))     # (in × 16)
nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))     # 랜덤 초기화
self.B = nn.Parameter(torch.zeros(rank, out_dim))    # (16 × out), 전부 0
...
x = (self.alpha / self.rank) * (x @ self.A @ self.B)
return self.linear(x) + self.lora(x)                 # 원본 + 보정
```

**핵심 아이디어**

- 큰 가중치 행렬 `(768 × 768)` = 589,824개를 전부 학습하는 대신, `(768×16)` + `(16×768)` = 24,576개만 학습한다. **약 24배 절약.**
- 원본 `linear`는 **동결(freeze)** 되어 그대로 남고, LoRA가 계산한 값만 더해진다.

**왜 B만 0인가 (최다 출제)**

- `A @ B`에서 **B가 0이면 곱이 0** → 학습 시작 시점에 `linear(x) + 0 = linear(x)`, 즉 **원본 모델과 완전히 동일**하다. 파인튜닝이 원본을 갑자기 망가뜨리지 않고 부드럽게 시작된다.
- 그렇다고 **A까지 0이면** 기울기도 0이 되어 **영원히 학습되지 않는다.** 그래서 하나는 랜덤, 하나는 0이다.

**주입 메커니즘**

```python
for name, module in model.named_children():
    if isinstance(module, torch.nn.Linear):
        setattr(model, name, LinearWithLoRA(module, rank, alpha))
    else:
        replace_linear_with_lora(module, rank, alpha)   # 재귀
```

모델은 층 안에 층이 있는 양파 구조라, Linear가 아니면 **한 겹 더 들어가 같은 작업을 반복**한다. 이 재귀가 없으면 최상위 층만 바뀐다.

**`alpha/rank`** — LoRA 출력의 세기 조절 손잡이. rank를 바꿔도 개입 강도가 일정하게 유지되도록 나눠준다.

---

### 12위 — DPO 선호 학습

```python
model_logratios     = model_chosen_logprobs - model_rejected_logprobs
reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
logits = model_logratios - reference_logratios
losses = -F.logsigmoid(beta * logits)
```

**SFT와 무엇이 다른가**

|       | SFT (Ch7)       | DPO                                   |
| ----- | --------------- | ------------------------------------- |
| 데이터   | (질문, **정답**) 2개 | (질문, **chosen**, **rejected**) 3개     |
| 모델    | 1개              | **2개** (policy + reference)           |
| 학습 목표 | 정답을 그대로 따라하기    | **좋은 답을 나쁜 답보다 선호**하게 만들기             |
| 지표    | Loss            | Loss + **Margin** (chosen − rejected) |

**수식의 의미를 말로 풀면**

1. `model_logratios` — "학습 중인 모델이 좋은 답을 나쁜 답보다 **얼마나 더** 선호하는가"
2. `reference_logratios` — "원래 모델은 얼마나 선호했는가"
3. `logits = ① − ②` — "**기준보다 얼마나 더 잘 구분하게 되었는가**". DPO는 절대 점수가 아니라 **기준 대비 상대적 개선**을 학습한다.
4. `-logsigmoid(beta * logits)` — 이 값이 클수록 손실이 작아지는 형태. `beta`는 기준 모델에서 벗어나는 정도를 제어(크면 보수적).

**구현상 반드시 지킬 것**

- reference 모델은 `eval()` + `with torch.no_grad()` — **절대 변하지 않는 기준점**이어야 하고, 기울기를 안 만들면 메모리도 절반으로 준다.
- `compute_logprobs`에서 `labels[:, 1:]` / `logits[:, :-1, :]` — 자기회귀 정렬. 여기서 마스크도 `[:, 1:]`로 **똑같이** 밀어야 한다.
- `sum(-1) / mask.sum(-1)` — 그냥 `.mean()`을 쓰면 패딩 개수까지 포함되어 **긴 답변이 불리해진다.**
- `.detach()` — reward는 로깅용이므로 계산 그래프에서 분리.

---

# 4. 셀프 체크리스트 (한 줄 답 가능?)

- [ ] `stride`가 `max_length`보다 작으면 데이터에 무슨 일이 일어나는가?
- [ ] 위치 임베딩의 입력이 `torch.arange(...)`인 이유는? 토큰 임베딩과 무엇이 다른가?
- [ ] `register_buffer`와 `nn.Parameter`의 차이는? mask는 왜 buffer인가?
- [ ] `transpose(1,2)`와 `transpose(2,3)`은 각각 몇 차원 텐서에서 쓰이는가?
- [ ] `contiguous()`를 빼면 어떤 에러가 나는가? 왜 나는가?
- [ ] LayerNorm에서 `unbiased=False`, `keepdim=True`인 이유는?
- [ ] 마스킹에 0이 아니라 `-torch.inf`를 쓰는 이유는?
- [ ] `idx[:, -context_size:]`를 `idx[:, :context_size]`로 바꾸면 무슨 일이 생기는가?
- [ ] `logits.flatten(0, 1)` 후 shape은? 분류에서는 왜 flatten을 안 하는가?
- [ ] 검증 로더에서 `shuffle=False`, `drop_last=False`인 이유는 각각 무엇인가?
- [ ] `ignore_index=-100`의 근거는? `indices[1:]`에서 왜 첫 패딩은 남기는가?
- [ ] LoRA의 B를 0으로, A를 랜덤으로 초기화하는 이유는?
- [ ] DPO에서 reference 모델에 `no_grad()`와 `eval()`을 쓰는 이유는?
- [ ] `model.eval()`과 `torch.no_grad()`는 각각 무엇을 끄는가?
- [ ] Accuracy와 Loss의 차이는? 왜 Accuracy로 직접 학습할 수 없는가?
