# Chapter 3 — Multi-Head Attention 시각화 코드 핵심 설명

---

## 전체 흐름

```
GPT-2 Small 모델 로드
        │
        ▼ register_forward_hook()
각 TransformerBlock의 Attention에 hook 등록
        │
        ▼ model(input_tensor)  ← forward 실행
각 레이어의 Attention Weight가 all_attentions에 자동 저장
        │
        ▼ hook.remove()
        │
        ├─ 1. 히트맵 시각화 (imshow)
        └─ 2. 연결선 시각화 (line plot)
```

---

## 1. Forward Hook으로 Attention Weight 수집

```python
all_attentions = []  # 레이어별 attention weight 저장

def make_hook():
    def hook_fn(module, input, output):
        # module: 해당 Attention 레이어
        # input:  레이어 입력 (tuple)
        # output: 레이어 출력 (우리는 직접 재계산)

        x = input[0]  # 실제 입력 텐서
        b, num_tokens, _ = x.shape

        # Q, K 직접 계산 (Attention Weight 추출 위해)
        keys    = module.W_key(x)
        queries = module.W_query(x)
        keys    = keys.view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)

        attn_scores  = queries @ keys.transpose(2, 3)
        mask_bool    = module.mask.bool()[:num_tokens, :num_tokens]
        attn_scores  = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        all_attentions.append(attn_weights.detach())
        # shape: [1, num_heads, num_tokens, num_tokens]

    return hook_fn

# 모든 TransformerBlock의 att에 hook 등록
hooks = [block.att.register_forward_hook(make_hook()) for block in model.trf_blocks]

with torch.no_grad():
    model(input_tensor)  # forward → 모든 hook 자동 실행

for hook in hooks:
    hook.remove()  # 메모리 누수 방지
```

**Hook이 input[0]을 사용하는 이유**

```
register_forward_hook의 callback: (module, input, output)
  input:  tuple 형태, 첫 번째 원소가 실제 텐서
  output: Attention 최종 출력 (context_vec)

→ Attention weight는 output에 포함되어 있지 않음
→ input[0]으로 x를 받아 직접 Q, K를 계산하여 weight 추출
```

---

## 2. Attention Weight 구조

```python
all_attentions
# → 리스트, 길이 = 레이어 수 (12)

all_attentions[layer_idx]
# shape: [1, num_heads, num_tokens, num_tokens]
#         ↑  ↑          ↑           ↑
#     배치  헤드수    현재 토큰   주목하는 토큰

all_attentions[4][0, 3]
# → shape: [num_tokens, num_tokens]
# → 5번째 레이어, 4번째 헤드의 attention matrix
```

---

## 3. 히트맵 시각화 (imshow)

```python
layer_idx = 4   # 시각화할 레이어 번호
num_heads = all_attentions[layer_idx].shape[1]  # 12 (GPT-2 Small)

fig, axes = plt.subplots(2, num_heads // 2, figsize=(24, 8))
# 2행 × 6열 = 12개 헤드 동시 시각화

for head_idx in range(num_heads):
    attn = all_attentions[layer_idx][0, head_idx].numpy()
    # shape: [num_tokens, num_tokens]

    axes[head_idx].imshow(attn, cmap="Blues")
    # Blues 컬러맵: 진할수록 높은 attention weight
    # (i, j): i번 토큰이 j번 토큰에 얼마나 주목하는지
```

**Attention Matrix 해석**

```
행(row) = 현재 예측 중인 토큰
열(col) = 참조하는 과거 토큰

attn[3, 1] = 0.8:
  4번째 토큰을 예측할 때 2번째 토큰에 80% 주목

인과적 마스킹으로 상삼각은 0:
  attn[1, 3] = 0  (과거 토큰은 미래를 볼 수 없음)
```

---

## 4. 연결선 시각화 (Interactive Widget)

```python
def plot_attention_lines(change=None):
    layer_idx = layer_w.value  # Dropdown 선택값
    head_idx  = head_w.value

    attn = all_attentions[layer_idx][0, head_idx].numpy()
    n    = len(tokens)

    # 왼쪽(현재 토큰) ↔ 오른쪽(참조 토큰) 연결선
    for i in range(n):
        for j in range(n):
            w = float(attn[i, j])
            if w > 0.005:  # 약한 연결은 표시 안 함
                ax.plot(
                    [x_left, x_right],
                    [y_pos[i], y_pos[j]],
                    color="#4A90D9",
                    alpha=min(w * 2, 1.0),  # weight가 클수록 불투명
                    linewidth=1.2
                )
```

**alpha=min(w * 2, 1.0) 이유**

```
w (attention weight): 0.0 ~ 1.0
alpha (선 불투명도): 0.0(투명) ~ 1.0(불투명)

w * 2: 약한 연결도 보이게 증폭
min(..., 1.0): 최대 1.0으로 클리핑

→ 강한 attention: 진한 선
→ 약한 attention: 투명한 선
```

---

## 5. ipywidgets — 인터랙티브 컨트롤

```python
layer_w = widgets.Dropdown(
    options=list(range(len(all_attentions))),  # 0~11
    value=4,              # 기본 선택값
    description="Layer:"
)
head_w = widgets.Dropdown(
    options=list(range(all_attentions[0].shape[1])),  # 0~11
    value=3,
    description="Head:"
)

# 값이 바뀔 때마다 plot 재실행
layer_w.observe(plot_attention_lines, names="value")
head_w.observe(plot_attention_lines, names="value")
```

**plot을 Image 위젯으로 렌더링하는 이유**

```python
buffer = io.BytesIO()
fig.savefig(buffer, format="png", ...)
plot_widget.value = buffer.getvalue()
```

```
일반 plt.show(): 새 창/셀에 출력 → Dropdown과 분리됨
Image 위젯:      같은 셀에 연결 → Dropdown 변경 시 즉시 업데이트
```

---

## 6. 레이어/헤드별 Attention 패턴 관찰 포인트

| 레이어 | 일반적인 특징 |
|--------|-------------|
| 얕은 레이어 (0~3) | 인접 토큰 위주 attention (지역적 패턴) |
| 중간 레이어 (4~7) | 구문 구조 (주어-동사, 수식 관계) |
| 깊은 레이어 (8~11) | 의미적 관계 (문장 전체 맥락) |

**헤드별 역할 다양성**

```
같은 레이어 내에서도 헤드마다 다른 패턴:
  Head 0: 이전 토큰 위주 attention
  Head 3: 특정 키워드 위주 attention
  Head 7: 문장 첫 토큰 (BOS) 위주 attention

→ 다양한 관점에서 정보를 병렬 처리하는 것이 Multi-Head의 핵심
```

---

*작성일: 2026-03-25*
