# LLM 심화 개념 정리 (02_Concepts.md)

> PDF 강의자료(p.97~) + AI_Advanced_Concepts.md 기반  
> 각 개념에 직관적인 비유, 설명, Python pseudo code 포함

---

## 목차

1. [기본 구조 및 활성화 함수](#1-기본-구조-및-활성화-함수)
   - [Dropout](#11-dropout)
   - [정규화 (Normalization / Layer Norm / RMS Norm)](#12-정규화-normalization--layer-normalization--rms-normalization)
   - [FFN (Feed Forward Network)](#13-ffn-feed-forward-network)
   - [활성화 함수 (Sigmoid / ReLU / GELU)](#14-활성화-함수-sigmoid--relu--gelu)
   - [Residual Connection](#15-residual-connection)
2. [AI 모델의 종류](#2-ai-모델의-종류)
   - [Vision Transformer (ViT)](#21-vision-transformer-vit)
   - [Whisper](#22-whisper)
   - [AlphaFold](#23-alphafold)
   - [GPT](#24-gpt)
3. [GPT 모델 입출력](#3-gpt-모델-입출력)
   - [Output Token 생성](#31-output-token-생성)
   - [Temperature](#32-temperature)
   - [Top-K Sampling](#33-top-k-sampling)
4. [지식의 저장](#4-지식의-저장)
   - [Attention](#41-attention)
   - [FFN 지식의 저장](#42-ffn-지식의-저장)
   - [Latent Space](#43-latent-space)
5. [Training (학습 과정)](#5-training-학습-과정)
   - [학습 사이클 (순전파 → Loss → 역전파 → 경사하강법)](#51-학습-사이클)
   - [비용 함수 - MSE / Cross Entropy](#52-비용-함수---mse--cross-entropy)
   - [Local Minimum](#53-local-minimum)
   - [Adam Optimizer](#54-adam-optimizer)
   - [Epoch / Step / Batch Size](#55-epoch--step--batch-size)
6. [미세조정 및 효율적 학습 (Fine-Tuning)](#6-미세조정-및-효율적-학습-fine-tuning)
   - [분류 미세조정](#61-분류-미세조정)
   - [LoRA](#62-lora)
   - [PEFT](#63-peft)
   - [QLoRA](#64-qlora)
   - [BF16](#65-bf16)
   - [지시 미세조정 (Instruction Tuning)](#66-지시-미세조정-instruction-tuning)
   - [RLHF / DPO (Loss 함수)](#67-rlhf--dpo-loss-함수)
   - [PPO](#68-ppo)
   - [KL Divergence](#69-kl-divergence)
   - [Dataset 구조 - Reward Model / DPO](#610-dataset-구조---reward-model--dpo)
7. [LLM 효율화 (Inference Efficiency)](#7-llm-효율화-inference-efficiency)
   - [KV Caching](#71-kv-caching)
   - [GQA](#72-gqa-grouped-query-attention)
   - [MLA](#73-mla-multi-head-latent-attention)
   - [MOE (Mixture of Experts)](#74-moe-mixture-of-experts)
   - [Sliding Window Attention](#75-sliding-window-attention)
   - [Flash Attention](#76-flash-attention)

---

## 1. 기본 구조 및 활성화 함수

### 1.1 Dropout

**설명**  
학습 중에 신경망의 일부 뉴런을 **무작위로 꺼버리는** 기법. Overfitting(과적합)을 방지하기 위해 사용. 주로 Softmax 이후에 적용하며, 데이터가 충분히 많은 LLM 사전학습에는 잘 사용하지 않음.

**비유 (주식 투자)**  
삼성전자 한 종목에만 몰빵하면 그 주식이 흔들릴 때 전체 계좌가 위험해짐. Dropout은 의도적으로 일부 종목 거래를 막아 **강제 분산투자**를 하게 만드는 것과 같음. 특정 뉴런에만 의존하지 않고 전체 패턴을 골고루 학습.

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.ffn = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Attention 계산 후 dropout 적용
        attn_output, _ = self.attention(x, x, x)

        # Softmax 이후 랜덤으로 일부 값을 0으로 마스킹
        attn_output = self.dropout(attn_output)  # ← 여기가 Dropout

        # Residual Connection
        x = x + attn_output
        return x


# 학습 vs 추론 모드에서 동작 차이
model = TransformerBlock(d_model=768)

model.train()   # 학습 모드: dropout 활성화 (일부 뉴런 OFF)
output_train = model(torch.randn(4, 10, 768))

model.eval()    # 추론 모드: dropout 비활성화 (모든 뉴런 ON)
with torch.no_grad():
    output_eval = model(torch.randn(4, 10, 768))
```

---

### 1.2 정규화 (Normalization / Layer Normalization / RMS Normalization)

**설명**  
데이터의 숫자 스케일이 제멋대로 튀지 않도록 **일정한 범위로 맞춰주는** 작업. 학습을 안정화시키지만, 과도하면 학습된 내용이 사라질 수 있음.

| 종류 | 정규화 방향 | 특징 |
|------|------------|------|
| **Batch Norm** | 배치(샘플) 방향 | CNN에 주로 사용 |
| **Layer Norm** | 특징(채널) 방향 | LLM에서 표준 사용, γ(감마)·β(베타) 학습 |
| **RMS Norm** | 특징 방향 | 평균 계산을 제외해 더 빠름 (LLaMA 등에서 사용) |

**비유**  
기지국 콜 로그를 분석할 때, 어떤 값은 100만 단위, 어떤 값은 0.1 단위라면 AI가 큰 값에만 압도됨. 오디오 믹싱처럼 **볼륨을 균일하게 맞춰주는 작업**과 같음.

```python
import torch
import torch.nn as nn

# ── Layer Normalization ──────────────────────────────────────
# 수식: y = (x - mean) / sqrt(var + eps) * gamma + beta
# LLM에서 표준 사용. gamma, beta는 학습 파라미터
d_model = 768
layer_norm = nn.LayerNorm(d_model)  # gamma=1, beta=0 으로 초기화

x = torch.randn(2, 10, d_model)  # (batch, seq_len, d_model)
out_ln = layer_norm(x)
# out_ln: 각 샘플의 d_model 차원에서 mean=0, std=1로 정규화

# ── RMS Normalization (직접 구현) ────────────────────────────
# 수식: y = x / sqrt(mean(x^2) + eps) * gamma
# 평균(mean)을 빼지 않아 계산 비용 절감
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # 학습 파라미터

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()  # RMS 계산
        x_norm = x / (rms + self.eps)                     # 스케일 조정
        return self.gamma * x_norm                         # 감마 적용

rms_norm = RMSNorm(d_model)
out_rms = rms_norm(x)

# ── Batch vs Layer 비교 ──────────────────────────────────────
# Batch Norm: 배치 방향 정규화 → 배치 크기에 민감, CNN에 적합
# Layer Norm: 특징 방향 정규화 → 배치 크기와 무관, LLM에 적합
batch_norm = nn.BatchNorm1d(d_model)  # (batch, d_model)
```

---

### 1.3 FFN (Feed Forward Network)

**설명**  
Attention이 단어 간 **관계(문맥)**를 파악한다면, FFN은 그 정보를 받아 **복잡한 패턴을 소화하고 지식을 저장**하는 역할. 트랜스포머 블록에서 Attention 다음에 위치하며, 내부적으로 차원을 크게 늘렸다가 다시 줄임(Expansion → Contraction).

**비유**  
Attention이 "이 문장에서 어떤 단어들이 서로 관련 있지?"를 분석하는 탐정이라면, FFN은 분석 결과를 바탕으로 **"그래서 정답은 이거야"를 도출하는 전문가 뇌**와 같음.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    """
    트랜스포머 FFN: d_model → d_ff(4배 확장) → d_model
    - 1번 선형 변환: 차원 확대 (지식을 세분화해서 저장)
    - 활성화 함수: 비선형 패턴 학습
    - 2번 선형 변환: 차원 축소 (핵심 정보 압축)
    """
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 확장: 768 → 3072
        self.fc2 = nn.Linear(d_ff, d_model)   # 축소: 3072 → 768
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, d_model=768)
        x = self.fc1(x)          # → (batch, seq_len, 3072)
        x = F.gelu(x)            # 활성화 (GELU 사용)
        x = self.dropout(x)
        x = self.fc2(x)          # → (batch, seq_len, 768)
        return x

ffn = FeedForwardNetwork()
x = torch.randn(2, 10, 768)  # (batch=2, seq_len=10, d_model=768)
out = ffn(x)
print(out.shape)  # torch.Size([2, 10, 768])
```

---

### 1.4 활성화 함수 (Sigmoid / ReLU / GELU)

**설명**  
신경망의 각 층에서 **다음 층으로 신호를 얼마나 보낼지 결정하는 문지기**.

| 함수 | 수식 | 특징 | 주요 사용처 |
|------|------|------|------------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | 출력 0~1, 깊은 층에서 기울기 소실 | 이진 분류, 오래된 모델 |
| **ReLU** | max(0, x) | 빠름, 음수 뉴런 사망 문제 | CNN, 간단한 DNN |
| **GELU** | x·Φ(x) | 부드러운 곡선, 최신 LLM 표준 | GPT, BERT 등 |

**비유**  
- **Sigmoid**: 무조건 0~1 사이로 찌그러뜨리는 엄격한 문지기 → 층이 깊어지면 신호가 약해짐
- **ReLU**: 양수면 통과, 음수면 무조건 0 → 단순하고 빠르지만 일부 뉴런이 영구히 꺼짐
- **GELU**: ReLU처럼 음수는 약하게 억제하되, 0 근처에서 부드럽게 → 최신 LLM에서 가장 애용

```python
import torch
import torch.nn.functional as F
import numpy as np

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ── Sigmoid ──────────────────────────────────────────────────
sigmoid_out = torch.sigmoid(x)
# tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
# 문제: 층이 깊어질수록 기울기가 0에 수렴 → Vanishing Gradient

# ── ReLU ─────────────────────────────────────────────────────
relu_out = F.relu(x)
# tensor([0., 0., 0., 1., 2.])
# 장점: 계산 빠름, 층 깊이 가능
# 단점: 음수 뉴런 완전 차단 → "Dying ReLU" 문제

# ── GELU (Gaussian Error Linear Unit) ────────────────────────
gelu_out = F.gelu(x)
# tensor([-0.0454, -0.1587,  0.0000,  0.8413,  1.9545])
# ReLU와 유사하지만 0 근처에서 부드럽게 처리
# GPT, BERT 등 현대 LLM의 FFN에서 표준으로 사용

# ── GELU 직접 구현 (근사식) ──────────────────────────────────
def gelu_approx(x):
    """GELU 근사 수식: x * sigmoid(1.702 * x)"""
    return x * torch.sigmoid(1.702 * x)

gelu_approx_out = gelu_approx(x)

# ── FFN에서의 실제 사용 ──────────────────────────────────────
class FFN_with_GELU(torch.nn.Module):
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # GELU 활성화
```

---

### 1.5 Residual Connection

**설명**  
이전 층의 입력값을 **변환된 출력에 그대로 더해주는 지름길(Shortcut)**. 깊은 네트워크에서 기울기 소실을 방지하고 학습을 안정화. 출력 = F(x) + x

**비유**  
스키를 타고 산을 내려올 때, 각 구간마다 처음부터 지형을 다시 파악하는 대신 **이전 지형 정보(x)를 그대로 가져와서 변화량(ΔF)만 추가로 학습**. 모델 층이 아무리 깊어도 원본 정보를 보존하며 내려갈 수 있음.

```python
import torch
import torch.nn as nn

class TransformerBlockWithResidual(nn.Module):
    """
    트랜스포머의 핵심 구조:
    - Pre-Norm: 정규화 → 연산 → Residual 덧셈 (최근 주류)
    - Post-Norm: 연산 → Residual 덧셈 → 정규화 (원래 논문)
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # ── Attention + Residual ──────────────────────────────
        residual = x                              # 원본 보존
        x = self.norm1(x)                        # Pre-Norm
        attn_out, _ = self.attention(x, x, x)
        x = residual + attn_out                  # ← Residual 덧셈

        # ── FFN + Residual ────────────────────────────────────
        residual = x                              # 원본 보존
        x = self.norm2(x)                        # Pre-Norm
        ffn_out = self.ffn(x)
        x = residual + ffn_out                   # ← Residual 덧셈

        return x

block = TransformerBlockWithResidual()
x = torch.randn(2, 10, 768)  # (batch, seq_len, d_model)
out = block(x)
print(out.shape)  # torch.Size([2, 10, 768]) ← 입출력 차원 동일!
```

---

## 2. AI 모델의 종류

### 2.1 Vision Transformer (ViT)

**설명**  
이미지를 작은 조각(Patch)으로 잘라서 마치 텍스트의 단어처럼 취급하고, Transformer Encoder로 문맥을 파악하듯 이미지를 이해하는 모델.

**비유**  
사진을 퍼즐 조각처럼 잘라서 순서대로 나열한 뒤, "이 조각들이 전체적으로 어떤 그림을 형성하는지" 글 읽듯이 파악하는 것.

```
이미지 (224×224)
    │
    ▼
패치로 분할 (16×16 patch → 196개 패치)
    │
    ▼
각 패치 → 선형 변환 → 1D 벡터 (토큰화)
    │
[CLS] 토큰 + 196개 패치 토큰 + Position Embedding
    │
    ▼
Transformer Encoder (L layers)
    │
    ▼
[CLS] 토큰 → 분류 Head → 정답 클래스
```

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """이미지를 패치로 분할하고 임베딩 벡터로 변환"""
    def __init__(self, img_size=224, patch_size=16, d_model=768, in_channels=3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196개
        # Conv로 패치 추출 + 선형 투영을 한번에 처리
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.proj(x)          # → (batch, 768, 14, 14)
        x = x.flatten(2)          # → (batch, 768, 196)
        x = x.transpose(1, 2)    # → (batch, 196, 768)  ← 토큰 시퀀스
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        d_model = 768
        self.patch_embed = PatchEmbedding()
        num_patches = 196

        # CLS 토큰: 전체 이미지 표현을 담당
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 위치 임베딩: 패치 순서 정보 추가
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=12, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.head = nn.Linear(d_model, num_classes)  # 분류 헤드

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)    # (B, 196, 768)

        # CLS 토큰 앞에 붙이기
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                   # (B, 197, 768)
        x = x + self.pos_embed                            # 위치 정보 추가

        x = self.transformer(x)       # Transformer 처리
        cls_output = x[:, 0, :]       # CLS 토큰만 추출
        return self.head(cls_output)  # 분류 결과
```

---

### 2.2 Whisper

**설명**  
OpenAI가 만든 음성 인식 모델. **음성(오디오) → 텍스트** 변환에 Encoder-Decoder Transformer 구조 사용. 다양한 언어, 억양, 잡음에서도 뛰어난 인식 성능.

**비유**  
아무리 시끄러운 장소에서도, 사투리가 심해도 통역을 척척 해내는 만능 통역사.

```python
# Whisper 사용 예시 (openai-whisper 라이브러리)
# pip install openai-whisper

import whisper

# 모델 로드 (tiny/base/small/medium/large)
model = whisper.load_model("base")

# 음성 파일 → 텍스트 변환
result = model.transcribe("audio.mp3", language="ko")
print(result["text"])  # "안녕하세요, 반갑습니다."

# ── Whisper 내부 구조 (pseudo code) ──────────────────────────
class WhisperModel(nn.Module):
    """
    Encoder: 오디오 Mel-spectrogram → 특징 벡터 추출
    Decoder: 특징 벡터 → 텍스트 토큰 생성 (Autoregressive)
    """
    def __init__(self):
        super().__init__()
        # Encoder: 오디오 → 컨텍스트 벡터
        self.audio_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6
        )
        # Decoder: 이전 텍스트 + 컨텍스트 → 다음 텍스트 토큰
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6
        )

    def forward(self, mel_spectrogram, text_tokens):
        # 1. 오디오 특징 추출
        audio_features = self.audio_encoder(mel_spectrogram)
        # 2. 텍스트 토큰 디코딩 (Cross-Attention으로 오디오 참조)
        text_output = self.text_decoder(text_tokens, audio_features)
        return text_output
```

---

### 2.3 AlphaFold

**설명**  
구글 딥마인드가 개발한 단백질 구조 예측 AI. 아미노산 배열만 보고 단백질의 **3차원 입체 구조**를 예측. Transformer Attention으로 아미노산 간의 관계(거리, 각도)를 파악.

**비유**  
레고 블록(아미노산)의 목록만 보고 완성된 3D 레고 구조물의 형태를 예측하는 것. 수십 년 걸리던 단백질 구조 연구를 몇 분 만에 해결.

```python
# AlphaFold 핵심 개념 (pseudo code)

class Evoformer(nn.Module):
    """
    AlphaFold의 핵심 블록
    - MSA (Multiple Sequence Alignment) 표현: 진화적 관계 정보
    - Pair 표현: 아미노산 쌍의 거리/각도 관계
    두 표현이 서로 정보를 교환하며 업데이트
    """
    def __init__(self, d_msa=256, d_pair=128):
        super().__init__()
        # MSA 처리: 아미노산 배열 내 문맥 파악
        self.msa_attention = nn.MultiheadAttention(d_msa, num_heads=8, batch_first=True)
        # Pair 처리: 아미노산 쌍(i, j) 관계 파악
        self.pair_ffn = nn.Linear(d_pair, d_pair)
        # Outer product mean: MSA → Pair 정보 전달
        self.outer_product = nn.Linear(d_msa, d_pair)

    def forward(self, msa_repr, pair_repr):
        # 1. MSA 내부 Attention (행 방향: 같은 위치 아미노산 비교)
        msa_out, _ = self.msa_attention(msa_repr, msa_repr, msa_repr)
        msa_repr = msa_repr + msa_out  # Residual

        # 2. MSA → Pair 정보 전달 (외적 평균)
        pair_update = self.outer_product(msa_repr.mean(dim=1, keepdim=True))
        pair_repr = pair_repr + pair_update  # Residual

        # 3. Pair FFN
        pair_repr = pair_repr + self.pair_ffn(pair_repr)
        return msa_repr, pair_repr

# 실제 AlphaFold 사용
# pip install alphafold  또는  ColabFold 환경에서
# from alphafold.model import model as af_model
```

---

### 2.4 GPT

**설명**  
**Generative Pre-trained Transformer**. 앞 문맥을 보고 다음에 올 가장 자연스러운 단어를 반복 생성하는 **디코더 전용 언어 모델**. 텍스트 생성, 번역, 요약, 질의응답 등에 범용적으로 사용.

**비유**  
"오늘 날씨가 맑아서..." 뒤에 올 단어를 5만 개 단어장에서 가장 자연스러운 것을 골라 계속 이어 쓰는 스마트 자동완성.

```
입력: "인공지능은"
    ↓
Token Embedding + Position Embedding
    ↓
Transformer Decoder Block × N (Masked Self-Attention + FFN)
    ↓
Linear Projection → Vocab Size (50,257)
    ↓
Softmax → 확률 분포
    ↓
"매우" (가장 높은 확률의 토큰 선택)
    ↓
다시 입력으로 → "인공지능은 매우" → 반복
```

```python
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    """GPT의 디코더 블록 (Masked Self-Attention + FFN)"""
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        seq_len = x.size(1)
        # Causal Mask: 미래 토큰 보지 못하게 마스킹
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        # Masked Self-Attention + Residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           attn_mask=mask)[0]
        # FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        # token_ids: (B, T) - B=배치, T=시퀀스 길이
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)

        x = self.embed(token_ids) + self.pos_embed(pos)  # (B, T, 768)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)  # (B, T, 50257) ← vocab 크기의 로짓
        return logits
```

---

## 3. GPT 모델 입출력

### 3.1 Output Token 생성

**설명**  
GPT는 입력 텍스트를 처리한 뒤, 마지막 위치의 벡터를 **50,257개 단어 사전에 대한 확률 분포**로 변환하고, 가장 높은 확률의 토큰을 선택해 순차적으로 출력.

입출력 텐서 차원:
```
Source Input:  (B, T, d_model) → B=Batch, T=시퀀스 길이, d=768
Output Logits: (B, T, V)       → V=Vocab size=50,257
```

```python
import torch
import torch.nn.functional as F

# GPT 출력: logits (B=1, T=4, V=50257)
logits = torch.randn(1, 4, 50257)

# 마지막 토큰의 로짓만 사용 (다음 단어 예측)
last_logits = logits[:, -1, :]         # (1, 50257)
probs = F.softmax(last_logits, dim=-1) # 확률 분포로 변환

# Greedy Decoding: 가장 높은 확률 토큰 선택
next_token = torch.argmax(probs, dim=-1)

# Autoregressive 생성 (pseudo code)
def generate(model, input_ids, max_new_tokens=50):
    """입력 → 반복적으로 다음 토큰 생성"""
    for _ in range(max_new_tokens):
        logits = model(input_ids)        # 전체 시퀀스 처리
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)  # 시퀀스에 추가
        if next_token.item() == EOS_TOKEN_ID:  # 종료 토큰이면 중단
            break
    return input_ids
```

---

### 3.2 Temperature

**설명**  
Softmax 계산 시 로짓을 **온도(T)로 나눠** 확률 분포의 날카로움을 조절.
- **T → 0**: 특정 토큰으로 확률 집중 → 항상 같은 답 (결정적)
- **T = 1**: 기본 확률 분포 그대로
- **T > 1**: 확률 평탄화 → 다양하고 창의적인 답 (무작위성 증가)

**비유**  
커피숍 메뉴 고르기. 온도가 낮으면 항상 아메리카노(익숙한 선택), 온도가 높으면 오늘은 아일랜드 크림 소다도 도전해볼까? (모험적 선택).

```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    """
    temperature=0.1 → 가장 높은 확률 토큰만 거의 확정적으로 선택
    temperature=1.0 → 모델의 기본 확률 분포 그대로
    temperature=2.0 → 확률 분포가 평탄해져 다양한 토큰 선택 가능
    """
    # logits: (vocab_size,)
    scaled_logits = logits / temperature        # 온도로 스케일 조정
    probs = F.softmax(scaled_logits, dim=-1)    # 확률로 변환
    next_token = torch.multinomial(probs, num_samples=1)  # 확률 기반 샘플링
    return next_token

# 예시 비교
logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

print("T=0.1 (결정적):", F.softmax(logits / 0.1, dim=-1))
# tensor([0.9999, 0.0001, ...])  ← 첫 번째 토큰에 확률 집중

print("T=1.0 (기본):",   F.softmax(logits / 1.0, dim=-1))
# tensor([0.5765, 0.2122, 0.1288, 0.0825])

print("T=2.0 (창의적):", F.softmax(logits / 2.0, dim=-1))
# tensor([0.3894, 0.2671, 0.2219, 0.1216])  ← 고르게 분포
```

---

### 3.3 Top-K Sampling

**설명**  
전체 어휘 중 **확률 상위 K개만 추려내고** 그 안에서 확률 기반으로 랜덤 선택. 엉뚱한 단어(Hallucination) 출현을 방지하면서도 다양성 유지.

**비유**  
퇴직연금(DC형) 운용 시, 시장의 수천 개 ETF 중 **수익률 상위 50개(Top-K)만 추려낸 뒤** 그 안에서 하나를 고르는 방식. 엉뚱한 투자를 차단하면서도 최선의 다양성 확보.

```python
import torch
import torch.nn.functional as F

def top_k_sampling(logits, k=50, temperature=1.0):
    """
    1. 온도 조정
    2. 상위 K개만 남기고 나머지는 -무한대로 마스킹
    3. 남은 K개 중에서 확률 기반 샘플링
    """
    # 온도 조정
    logits = logits / temperature

    # 상위 K개를 제외한 나머지를 -inf로 마스킹
    top_k_values, top_k_indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(0, top_k_indices, top_k_values)

    # 확률 분포로 변환 후 샘플링
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# Top-P (Nucleus Sampling) - Top-K의 발전형
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """누적 확률이 p를 넘는 지점까지만 허용"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 누적 확률이 p를 초과하는 토큰 제거
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

---

## 4. 지식의 저장

### 4.1 Attention

**설명**  
단어 하나를 처리할 때 **문장 전체에서 어떤 단어를 얼마나 참조할지** 가중치를 계산해 문맥 정보를 흡수. "creature"라는 단어가 앞에 "fluffy blue"가 있다면, 그 맥락을 흡수해 새로운 표현 벡터를 생성.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q (Query): 현재 내가 찾고 싶은 정보
    K (Key):   각 토큰이 가진 색인(어떤 정보가 있는지)
    V (Value): 실제 전달할 정보 내용
    
    수식: Attention(Q,K,V) = softmax(QK^T / √d_k) * V
    """
    d_k = Q.size(-1)  # 키 차원

    # Q와 K의 유사도 계산 (내적)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (B, heads, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax로 가중치 정규화
    attn_weights = F.softmax(scores, dim=-1)

    # 가중합으로 문맥 정보 수집
    context = torch.matmul(attn_weights, V)
    return context, attn_weights

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 64
        # Q, K, V 각각의 선형 투영
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        # Q, K, V 계산 후 head 분리
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # (B, n_heads, T, d_head)

        context, _ = scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(context)
```

---

### 4.2 FFN 지식의 저장

**설명**  
FFN의 가중치 행렬에는 훈련 데이터에서 습득한 **사실 지식이 저장**됨. 고차원 공간에서 각 개념은 직교(Orthogonal)에 가까운 방향으로 저장되어 서로 간섭 없이 수많은 지식을 담을 수 있음. 층이 깊어질수록 새로운 추상 정보를 생성.

```python
# FFN이 지식을 저장하는 방식 직관적 이해 (pseudo code)
import torch

d_ff = 3072  # 확장 차원
d_model = 768

# FFN의 가중치 = 학습된 패턴/지식의 저장소
W1 = torch.randn(d_model, d_ff)  # (768, 3072)
W2 = torch.randn(d_ff, d_model)  # (3072, 768)

# 각 FFN 뉴런(d_ff의 각 차원)은 특정 개념/패턴을 담당
# 예: 뉴런 #42 → "파리는 프랑스의 수도" 패턴에 반응
# 예: 뉴런 #137 → "동사의 과거형" 패턴에 반응

# 직교성(Orthogonality): 3072 차원에서 각 패턴들이 서로 간섭하지 않음
# → 수백만 개의 독립적인 사실을 저장 가능

def ffn_knowledge_lookup(x, W1, W2):
    """
    x: 입력 문맥 벡터
    W1@x: 어떤 '지식 뉴런'이 활성화되는지 확인
    GELU: 관련 뉴런만 활성화 (희소 활성화)
    W2: 활성화된 지식을 최종 출력 벡터로 조합
    """
    hidden = torch.gelu(x @ W1)  # 관련 지식 뉴런 활성화
    output = hidden @ W2          # 지식 조합 → 출력
    return output
```

---

### 4.3 Latent Space

**설명**  
AI가 단어와 개념을 표현하는 **고차원 추상 공간**. 텍스트가 이 공간의 좌표(벡터)로 변환되며, 의미적으로 유사한 개념은 가까이, 관련 없는 개념은 멀리 위치. "왕" - "남자" + "여자" ≈ "여왕" 같은 벡터 연산이 가능한 이유.

```python
import torch
import torch.nn as nn

# 단어 → 잠재 공간 좌표 변환
vocab_size = 50257
d_model = 768  # 잠재 공간의 차원 수

embedding = nn.Embedding(vocab_size, d_model)

# 각 단어는 768차원 벡터 = 잠재 공간 내의 한 좌표
king_id = torch.tensor([1000])     # "왕"의 토큰 ID
queen_id = torch.tensor([2000])    # "여왕"의 토큰 ID
man_id = torch.tensor([500])       # "남자"의 토큰 ID
woman_id = torch.tensor([600])     # "여자"의 토큰 ID

king_vec  = embedding(king_id)     # (1, 768)
queen_vec = embedding(queen_id)    # (1, 768)
man_vec   = embedding(man_id)      # (1, 768)
woman_vec = embedding(woman_id)    # (1, 768)

# 잠재 공간에서의 벡터 연산
# "왕" - "남자" + "여자" ≈ "여왕"
analogy_vec = king_vec - man_vec + woman_vec

# 코사인 유사도로 유사 단어 검색
def find_nearest(query_vec, all_vecs, top_k=5):
    query_norm = query_vec / query_vec.norm(dim=-1, keepdim=True)
    all_norm = all_vecs / all_vecs.norm(dim=-1, keepdim=True)
    similarities = (query_norm @ all_norm.T).squeeze()
    return similarities.topk(top_k)

# Latent Space의 차원 압축/확장
class LatentProjection(nn.Module):
    """차원을 줄여 핵심 특징만 압축 (인코더), 다시 확장 (디코더)"""
    def __init__(self, d_model=768, d_latent=256):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_latent)  # 압축
        self.decoder = nn.Linear(d_latent, d_model)  # 복원

    def forward(self, x):
        z = self.encoder(x)  # 잠재 공간으로
        return self.decoder(z)
```

---

## 5. Training (학습 과정)

### 5.1 학습 사이클

**설명**  
AI가 오차(Loss)를 줄이며 가중치(Weight)를 찾아가는 반복 과정:

```
순전파(Forward Pass) → Loss 계산 → 역전파(Backpropagation) → 경사하강법(Gradient Descent) → 반복
```

**비유 (스키장)**  
짙은 안개 속 스키장 정상에서 가장 낮은 골짜기(Loss=0)를 찾아 내려가는 과정. 순전파=예측, Loss=현재 위치 확인, 역전파=어느 방향으로 내려가야 하는지 계산, 경사하강=한 걸음 내려가기.

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = GPT(vocab_size=50257, d_model=256, n_layers=4)  # 작은 GPT
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

def train_one_step(model, optimizer, input_ids, target_ids):
    """한 번의 학습 사이클 (1 Step)"""

    # ① 순전파 (Forward Pass): 예측값 계산
    logits = model(input_ids)         # (B, T, vocab_size)

    # ② Loss 계산: 예측과 정답의 차이
    B, T, V = logits.shape
    loss = criterion(logits.view(B*T, V), target_ids.view(B*T))

    # ③ 역전파 (Backpropagation): 각 가중치의 기울기 계산
    optimizer.zero_grad()   # 이전 기울기 초기화
    loss.backward()         # 오차를 역방향으로 전파 → 기울기 계산

    # ④ 경사하강법: 기울기 방향으로 가중치 업데이트
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 기울기 폭발 방지
    optimizer.step()        # weight = weight - lr * gradient

    return loss.item()

# 학습 루프
for epoch in range(10):
    for batch_input, batch_target in dataloader:
        loss = train_one_step(model, optimizer, batch_input, batch_target)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

---

### 5.2 비용 함수 - MSE / Cross Entropy

**설명**  
모델의 예측이 정답과 얼마나 다른지를 **숫자 하나(Loss)**로 표현하는 함수.

| 함수 | 용도 | 수식 |
|------|------|------|
| **MSE** | 연속 수치 예측 | mean((y_pred - y_true)²) |
| **Cross Entropy** | 분류 (카테고리) | -Σ y_true × log(y_pred) |

**비유**  
- **MSE**: 예측 기온 22℃ vs 실제 25℃ → 차이²을 평균낸 점수
- **Cross Entropy**: "이 문장의 다음 단어는?" 에서 정답 단어의 확률만 높이면 됨. 틀렸을 때 벌점을 크게, 맞았을 때 벌점을 작게 줌

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MSE Loss (수치 예측) ─────────────────────────────────────
mse_loss = nn.MSELoss()

y_pred = torch.tensor([22.0, 18.0, 30.0])  # 예측 기온
y_true = torch.tensor([25.0, 20.0, 28.0])  # 실제 기온
loss_mse = mse_loss(y_pred, y_true)
print(f"MSE: {loss_mse:.4f}")  # (9 + 4 + 4) / 3 = 5.67

# ── Cross Entropy Loss (분류) ────────────────────────────────
cross_entropy = nn.CrossEntropyLoss()

# LLM에서의 Cross Entropy: 다음 토큰 예측
# 입력: logits (B×T, vocab_size), 정답: target token IDs
logits = torch.randn(8, 50257)  # 8개 토큰의 예측 로짓
targets = torch.randint(0, 50257, (8,))  # 정답 토큰 IDs
loss_ce = cross_entropy(logits, targets)

# 내부 계산:
# 1. Softmax → 확률 변환
# 2. 정답 토큰의 log 확률만 추출
# 3. 평균의 음수 (-log(P(정답)))
probs = F.softmax(logits, dim=-1)
correct_probs = probs[range(8), targets]
manual_ce = -torch.log(correct_probs).mean()
print(f"Cross Entropy: {loss_ce:.4f} ≈ {manual_ce:.4f}")

# 직관: 정답 토큰 확률 = 1.0 → loss = 0 (완벽)
#       정답 토큰 확률 = 0.01 → loss = 4.6 (매우 나쁨)
```

---

### 5.3 Local Minimum

**설명**  
경사하강법이 최저점(Global Minimum)이 아닌 **중간의 작은 웅덩이(Local Minimum)에 빠져버리는** 문제. 단, 고차원(수십억 파라미터)에서는 Local Minimum보다 기울기가 0에 가까운 **안장점(Saddle Point)과 평탄 구간**이 더 큰 문제.

**비유**  
안개 낀 스키장에서 눈앞만 보고 내려가다가 작은 웅덩이를 진짜 바닥으로 착각하고 멈춰버리는 상황.

```python
import torch
import matplotlib.pyplot as plt

# Loss 함수 시각화 (1D 예시)
def loss_landscape(w):
    """여러 개의 Local Minimum이 있는 복잡한 Loss 함수"""
    return (w**4 - 4*w**2 + w + 0.5 * torch.sin(5*w))

w_range = torch.linspace(-2.5, 2.5, 500)
loss_values = loss_landscape(w_range)

# 경사하강법으로 최솟값 탐색
def gradient_descent(start_w, lr=0.01, steps=200):
    w = torch.tensor([start_w], requires_grad=True)
    trajectory = []
    for _ in range(steps):
        loss = loss_landscape(w)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad       # 기울기 방향으로 이동
            w.grad.zero_()
        trajectory.append(w.item())
    return trajectory

# 시작점에 따라 다른 Local Minimum에 수렴!
traj_a = gradient_descent(start_w=-2.0)  # Local Min에 빠질 수 있음
traj_b = gradient_descent(start_w=1.5)   # 다른 위치에 수렴

# 해결책
print("Local Minimum 탈출 전략:")
print("1. Momentum 추가 (관성으로 웅덩이 탈출)")
print("2. 큰 학습률 사용 (큰 보폭으로 웅덩이 넘기)")
print("3. Adam Optimizer (각 파라미터에 적응형 학습률)")
print("4. 배치 노이즈 활용 (SGD의 노이즈가 오히려 도움)")
```

---

### 5.4 Adam Optimizer

**설명**  
**Momentum(관성)**과 **RMSprop(적응형 학습률)**을 결합한 최적화 알고리즘.
- **Momentum**: 이전에 이동하던 방향과 속도를 기억해 Local Minimum을 훅 치고 지나감
- **RMSprop**: 기울기가 컸던 파라미터는 보폭을 줄이고, 작았던 파라미터는 보폭을 키움

**비유**  
단순한 경사하강법이 그때그때 발끝 경사만 보고 내려가는 초보 스키어라면, Adam은 이전까지 탄 관성을 기억하는 경험 많은 스키어. 웅덩이를 만나도 가속도로 치고 올라가 진짜 최저점을 찾음.

```python
import torch
import torch.optim as optim

model = nn.Linear(768, 768)

# Adam 기본 사용
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,       # 학습률
    betas=(0.9, 0.999),  # (beta1=관성 계수, beta2=RMS 계수)
    eps=1e-8       # 수치 안정화
)

# ── Adam 내부 동작 (직접 구현) ───────────────────────────────
class AdamOptimizer:
    """Adam Optimizer 직접 구현"""
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1   # 1차 모멘트 (방향/관성) 계수
        self.beta2 = beta2   # 2차 모멘트 (속도/보폭) 계수
        self.eps = eps
        self.t = 0           # 타임스텝
        # 1차 모멘트 (평균 기울기): 방향
        self.m = [torch.zeros_like(p) for p in self.params]
        # 2차 모멘트 (기울기 제곱 평균): 보폭 조절
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad  # 현재 기울기

            # 모멘트 업데이트 (지수 이동 평균)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # 편향 보정 (초기 단계에서 m, v가 0에 치우치는 현상 보정)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # 파라미터 업데이트
            # m_hat: 어디로 갈지 (방향)
            # v_hat: 얼마나 갈지 (보폭) - 기울기 컸던 방향은 보폭 줄임
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
```

---

### 5.5 Epoch / Step / Batch Size

**설명**  
전체 데이터를 어떻게 쪼개서 학습하는지 단위 개념.

```
전체 데이터: 1,000개
Batch Size:  100개씩
Step:        한 Batch 처리 = 가중치 1회 업데이트 → 10 Steps/Epoch
Epoch:       전체 1,000개를 1회 다 훑음 = 1 Epoch
```

| 크기 | 장점 | 단점 |
|------|------|------|
| **작은 Batch** | 메모리 적게 사용, 일반화 성능↑ | 학습 시간 길어짐, 불안정 |
| **큰 Batch** | 학습 안정적, GPU 효율↑ | 과적합 위험, 많은 메모리 필요 |

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 데이터셋 생성
total_data = 1000
input_data  = torch.randn(total_data, 768)
target_data = torch.randint(0, 50257, (total_data,))

dataset = TensorDataset(input_data, target_data)
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 총 Step 수 계산
steps_per_epoch = total_data // batch_size  # 1000 / 100 = 10
print(f"Steps per Epoch: {steps_per_epoch}")

num_epochs = 5
total_steps = steps_per_epoch * num_epochs   # 50
print(f"Total Steps: {total_steps}")

# 학습 루프
for epoch in range(num_epochs):          # Epoch 반복
    for step, (x_batch, y_batch) in enumerate(dataloader):  # Step 반복
        # x_batch: (100, 768) ← Batch Size = 100
        loss = train_one_step(model, optimizer, x_batch, y_batch)
        if step % 5 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Loss: {loss:.4f}")
```

---

## 6. 미세조정 및 효율적 학습 (Fine-Tuning)

### 6.1 분류 미세조정

**설명**  
사전학습된 GPT 모델에서 **마지막 레이어만 수정**해 특정 분류 작업에 맞게 조정. 짧은 입력은 Padding으로 길이를 맞추고, 정답은 0/1 레이블로 처리.

```python
import torch
import torch.nn as nn

class GPTForClassification(nn.Module):
    """GPT 기반 분류 모델 (마지막 레이어만 교체)"""
    def __init__(self, gpt_model, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        d_model = 768

        # 기존 LM 헤드 제거, 분류 헤드 추가
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # GPT 출력에서 마지막 유효 토큰의 벡터 사용
        gpt_output = self.gpt(input_ids)       # (B, T, d_model)
        # 마지막 토큰 벡터 → 분류
        last_hidden = gpt_output[:, -1, :]     # (B, d_model)
        logits = self.classifier(last_hidden)  # (B, num_classes)
        return logits

# Padding 처리 (짧은 입력의 경우)
def pad_sequence(token_ids, max_len=128, pad_token=0):
    """입력 시퀀스를 max_len으로 패딩"""
    padded = token_ids + [pad_token] * (max_len - len(token_ids))
    return padded[:max_len]

# 학습: 분류 레이어만 업데이트
base_gpt = GPT(vocab_size=50257, d_model=768, n_layers=12)
classifier = GPTForClassification(base_gpt, num_classes=2)

# GPT 파라미터 고정, 분류기만 학습
for param in classifier.gpt.parameters():
    param.requires_grad = False
for param in classifier.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 결과: softmax → argmax로 최종 클래스 선택
output = classifier(torch.randint(0, 50257, (4, 128)))  # (4, 2)
pred_class = output.argmax(dim=-1)  # (4,)
print(pred_class)  # tensor([0, 1, 0, 1])
```

---

### 6.2 LoRA

**설명**  
**Low-Rank Adaptation**. 수십억 파라미터의 거대 가중치 행렬 전체를 학습하는 대신, **작은 두 행렬(A, B)의 곱으로 변화량(ΔW)**만 학습. 학습 파라미터 수를 90% 이상 줄이면서도 Full Fine-tuning 수준의 성능 달성.

**비유**  
스키복 전체를 새로 맞추는 대신, 기존 스키복은 그대로 두고 공기저항 줄이는 **작은 패치(LoRA 어댑터)**만 덧대는 방식.

```
W (원본 가중치, 고정)  +  BA (LoRA 어댑터, 학습)
(d × d)                    (d × r) × (r × d)

d = 768, r = 8 (rank)
학습 파라미터: 768×768 = 589,824  →  2 × 768×8 = 12,288 (약 50배 감소)
```

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA가 적용된 선형 레이어
    출력 = x @ W.T + x @ A.T @ B.T * (alpha/r)
    
    초기화:
    - A: 랜덤 초기화 (정규분포)
    - B: 0으로 초기화 → 학습 시작시 LoRA 출력 = 0 (원본 모델과 동일하게 시작)
    """
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        # 원본 가중치 (고정)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )
        # LoRA 어댑터 (학습 대상)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))    # 랜덤 초기화
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # 0으로 초기화
        self.scaling = alpha / r   # 스케일링 계수

    def forward(self, x):
        # 원본 출력 (고정)
        original = x @ self.weight.T
        # LoRA 변화량
        lora_update = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original + lora_update  # 합산

# 실제 모델에 LoRA 적용
def apply_lora(model, r=8, alpha=16):
    """모델의 모든 Linear 레이어에 LoRA 적용"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f  = module.in_features
            out_f = module.out_features
            lora_layer = LoRALinear(in_f, out_f, r=r, alpha=alpha)
            # 원본 가중치 복사 후 고정
            lora_layer.weight.data = module.weight.data.clone()
            # 모듈 교체
            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]] \
                     if '.' in name else model
            setattr(parent, name.split('.')[-1], lora_layer)

# 학습 파라미터 확인
base_model = nn.Linear(768, 768)
lora_model = LoRALinear(768, 768, r=8)

total = sum(p.numel() for p in lora_model.parameters())
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"전체 파라미터: {total:,}")
print(f"학습 파라미터: {trainable:,}")   # A + B만
```

---

### 6.3 PEFT

**설명**  
**Parameter-Efficient Fine-Tuning**. 전체 파라미터 중 **극히 일부만 업데이트**해 효율적으로 미세조정하는 기법들의 총칭. LoRA가 대표적이며, Prefix Tuning, Adapter 등도 포함.

```python
# HuggingFace PEFT 라이브러리 사용 예시
# pip install peft transformers

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 1. 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 언어 생성 태스크
    r=8,                             # Rank (작을수록 파라미터 적음)
    lora_alpha=32,                   # 스케일링 계수
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # LoRA 적용할 모듈
)

# 3. PEFT 모델 생성
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.24%

# 4. 일반 학습과 동일하게 사용
optimizer = torch.optim.Adam(peft_model.parameters(), lr=2e-4)
```

---

### 6.4 QLoRA

**설명**  
**Quantized LoRA**. 모델 전체 가중치를 **4-bit로 양자화**해 메모리를 대폭 절감하고, 그 위에 LoRA 어댑터를 학습. 단일 GPU에서 70B 모델도 파인튜닝 가능하게 함. 학습은 역양자화(Dequantization) 후 진행, 결과는 BF16.

```
기존 LoRA:
  70B 모델 → 140GB (FP16) 필요 → A100 2장 필요

QLoRA:
  70B 모델 → 4-bit 양자화 → ~35GB → A100 1장으로 가능!
```

```python
# QLoRA 사용 예시 (bitsandbytes 라이브러리)
# pip install bitsandbytes transformers peft accelerate

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 1. 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 양자화 로드
    bnb_4bit_quant_type="nf4",              # NormalFloat4 양자화 방식
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산은 BF16으로
    bnb_4bit_use_double_quant=True,         # 이중 양자화 (추가 메모리 절약)
)

# 2. 모델을 4-bit로 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. LoRA 적용 (학습 파라미터만 BF16)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 양자화-역양자화 흐름
# 저장: FP16 → 4-bit (NF4)  ← 메모리 절약
# 연산: 4-bit → BF16 (역양자화) → LoRA 연산 → BF16 결과
```

---

### 6.5 BF16

**설명**  
**Brain Float 16**. 32비트 부동소수점(FP32)에서 정밀도를 줄인 16비트 형식. FP16과 달리 **FP32와 동일한 지수 범위**를 유지해 오버플로우/언더플로우 위험 없이 학습 안정성 확보.

```
FP32:  1비트 부호 + 8비트 지수 + 23비트 가수  = 범위 큼, 정밀도 높음
FP16:  1비트 부호 + 5비트 지수 + 10비트 가수  = 범위 작음, 오버플로우 위험
BF16:  1비트 부호 + 8비트 지수 +  7비트 가수  = FP32와 같은 범위, 학습 안정적
```

```python
import torch

# BF16 텐서 생성
x_fp32 = torch.randn(4, 768)
x_bf16 = x_fp32.to(torch.bfloat16)

print(f"FP32 메모리: {x_fp32.element_size() * x_fp32.numel()} bytes")  # 4 bytes/element
print(f"BF16 메모리: {x_bf16.element_size() * x_bf16.numel()} bytes")  # 2 bytes/element

# 혼합 정밀도 학습 (Mixed Precision Training)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):  # BF16으로 순전파
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()  # FP32로 역전파 (정밀도 유지)
    scaler.step(optimizer)
    scaler.update()
```

---

### 6.6 지시 미세조정 (Instruction Tuning)

**설명**  
AI가 사용자의 명령(Prompt)을 따라 챗봇처럼 답변하도록 훈련. 특정 형식의 데이터(지시-입력-출력)로 Fine-tuning. Instruction 부분은 Loss 계산에서 제외(마스킹).

```python
# 지시 미세조정 데이터 형식 (Alpaca 스타일)
instruction_data = {
    "instruction": "다음 문장을 영어로 번역해줘.",
    "input": "오늘 날씨가 정말 좋다.",
    "output": "The weather is really nice today."
}

# Prompt 템플릿
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def format_instruction(sample):
    return PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample.get("input", ""),
        output=sample["output"]
    )

# 핵심: Instruction 부분은 Loss 계산에서 제외
def create_masked_labels(input_ids, response_start_pos):
    """
    - Instruction 부분: -100으로 마스킹 (Loss 계산 제외)
    - Response 부분: 실제 토큰 ID (Loss 계산 포함)
    """
    labels = input_ids.clone()
    labels[:, :response_start_pos] = -100  # Instruction 마스킹
    return labels

# 훈련 (Response만 학습)
formatted = format_instruction(instruction_data)
tokens = tokenizer(formatted, return_tensors="pt")

response_start = formatted.find("### Response:") + len("### Response:\n")
response_start_pos = len(tokenizer.encode(formatted[:response_start]))

labels = create_masked_labels(tokens["input_ids"], response_start_pos)
outputs = model(**tokens, labels=labels)
loss = outputs.loss  # Instruction 제외, Response만 평가
```

---

### 6.7 RLHF / DPO (Loss 함수)

**설명**  
AI의 답변이 인간이 선호하는 방향이 되도록 **인간의 피드백으로 학습**하는 기법.

- **RLHF**: 별도의 Reward Model을 훈련한 뒤 강화학습(PPO)으로 정책 업데이트
- **DPO**: Reward Model 없이 선호 데이터(chosen vs rejected)만으로 직접 최적화

```python
import torch
import torch.nn.functional as F

# ── DPO Loss 구현 ─────────────────────────────────────────────
def dpo_loss(model, ref_model, chosen_ids, rejected_ids, beta=0.1):
    """
    DPO: 두 응답 중 더 좋은 것(chosen)의 확률을 높이고,
         더 나쁜 것(rejected)의 확률을 낮춤.
    
    β(beta): Reference 모델을 얼마나 반영할지 결정
             β가 크면 원본 모델과 크게 다르지 않게 유지
    """
    # 현재 모델의 log 확률
    logits_chosen   = model(chosen_ids)
    logits_rejected = model(rejected_ids)

    log_prob_chosen   = compute_log_prob(logits_chosen, chosen_ids)
    log_prob_rejected = compute_log_prob(logits_rejected, rejected_ids)

    # Reference 모델의 log 확률 (frozen)
    with torch.no_grad():
        ref_logits_chosen   = ref_model(chosen_ids)
        ref_logits_rejected = ref_model(rejected_ids)
        ref_log_prob_chosen   = compute_log_prob(ref_logits_chosen, chosen_ids)
        ref_log_prob_rejected = compute_log_prob(ref_logits_rejected, rejected_ids)

    # Implicit Reward 계산 (현재 모델 - Reference 모델)
    reward_chosen   = log_prob_chosen   - ref_log_prob_chosen
    reward_rejected = log_prob_rejected - ref_log_prob_rejected

    # DPO Loss: chosen reward가 rejected보다 크도록
    # Bradley-Terry 모델 기반: sigmoid(-β * (r_chosen - r_rejected))
    loss = -F.logsigmoid(beta * (reward_chosen - reward_rejected)).mean()
    return loss

def compute_log_prob(logits, token_ids):
    """시퀀스 전체의 log 확률 합산"""
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)  # 시퀀스 전체 합산
```

---

### 6.8 PPO

**설명**  
**Proximal Policy Optimization**. RLHF에서 사용하는 강화학습 알고리즘. Reward Model의 점수를 보상으로 받아 정책(LLM)을 업데이트. 업데이트 폭을 제한해 학습 안정성 확보.

```python
# PPO in RLHF (pseudo code)

class RLHF_PPO:
    """
    RLHF PPO 학습 과정
    1. SFT 모델로 응답 생성
    2. Reward Model로 점수 계산
    3. PPO로 정책 업데이트 (KL 패널티 포함)
    """
    def __init__(self, policy_model, reward_model, ref_model):
        self.policy  = policy_model   # 학습 대상 LLM
        self.reward  = reward_model   # 인간 선호도 점수 예측
        self.ref     = ref_model      # 원본 SFT 모델 (고정)

    def train_step(self, prompts):
        # 1. 현재 정책으로 응답 생성
        responses = self.policy.generate(prompts)

        # 2. Reward Model로 점수 계산
        rewards = self.reward(prompts, responses)

        # 3. KL Divergence 패널티 추가 (원본 모델과 너무 달라지지 않게)
        with torch.no_grad():
            ref_logprobs = self.ref.log_prob(prompts, responses)
        policy_logprobs = self.policy.log_prob(prompts, responses)
        kl_penalty = policy_logprobs - ref_logprobs  # KL(π || π_ref)

        # 4. 최종 보상 = Reward - β * KL
        beta = 0.1
        total_reward = rewards - beta * kl_penalty

        # 5. PPO Clip Loss (급격한 정책 변화 방지)
        ratio = torch.exp(policy_logprobs - ref_logprobs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)  # ε=0.2
        ppo_loss = -torch.min(
            ratio * total_reward,
            clipped_ratio * total_reward
        ).mean()

        return ppo_loss
```

---

### 6.9 KL Divergence

**설명**  
**두 확률 분포 사이의 차이**를 측정하는 지표. DPO/PPO에서 학습된 모델이 원본 Reference 모델과 **너무 많이 달라지지 않도록** 제약(패널티)으로 사용.

```python
import torch
import torch.nn.functional as F

def kl_divergence(p_logprobs, q_logprobs):
    """
    KL(P || Q) = Σ P * log(P/Q)
    P: 현재 모델 분포
    Q: Reference(원본) 모델 분포
    
    값이 0에 가까울수록 두 분포가 유사
    값이 크면 모델이 원본과 너무 달라진 것 → 패널티
    """
    p = torch.exp(p_logprobs)
    kl = (p * (p_logprobs - q_logprobs)).sum(dim=-1)
    return kl

# 직관적 예시
# 원본 모델 분포: [0.7, 0.2, 0.1] (A단어 70%, B 20%, C 10%)
p_probs = torch.tensor([0.7, 0.2, 0.1])
p_log = torch.log(p_probs)

# 학습 후 분포 (약간 변화): [0.6, 0.3, 0.1]
q1_probs = torch.tensor([0.6, 0.3, 0.1])
q1_log = torch.log(q1_probs)

# 학습 후 분포 (크게 변화): [0.1, 0.8, 0.1]
q2_probs = torch.tensor([0.1, 0.8, 0.1])
q2_log = torch.log(q2_probs)

kl1 = kl_divergence(p_log, q1_log)
kl2 = kl_divergence(p_log, q2_log)
print(f"약간 변화한 경우 KL: {kl1:.4f}")   # 작음 (유사)
print(f"크게 변화한 경우 KL: {kl2:.4f}")   # 큼 (패널티 높음)

# PyTorch 내장 함수
kl_loss = F.kl_div(q1_log, p_probs, reduction='sum')
```

---

### 6.10 Dataset 구조 - Reward Model / DPO

**설명**  

| 방식 | 데이터 구조 | 특징 |
|------|------------|------|
| **Reward Model** | Prompt + Response + Feedback(점수) | 채점 AI를 별도 학습 후 PPO에 활용 |
| **DPO** | Prompt + Chosen + Rejected | Reward Model 없이 직접 선호도 학습 |

```python
# ── Reward Model 데이터셋 ────────────────────────────────────
reward_model_dataset = [
    {
        "prompt":    "여름 휴가지를 추천해줘",
        "response":  "제주도의 함덕 해변은 어떠신가요? 맑은 바다와 모래사장...",
        "feedback":  4.5  # 사람이 직접 채점한 점수 (1~5)
    },
    {
        "prompt":    "여름 휴가지를 추천해줘",
        "response":  "바다로 가세요.",
        "feedback":  1.2  # 성의 없는 답변
    }
]

# ── DPO 데이터셋 ─────────────────────────────────────────────
dpo_dataset = [
    {
        "prompt":   "여름 휴가지를 추천해줘",
        "chosen":   "제주도의 함덕 해변은 어떠신가요? 맑은 바다와 모래사장이 아름다워...",
        "rejected": "바다로 가세요."
    },
    {
        "prompt":   "파이썬으로 정렬 알고리즘을 구현해줘",
        "chosen":   "버블 정렬의 예시입니다. def bubble_sort(arr): ...",
        "rejected": "정렬 알고리즘은 복잡합니다."
    }
]

from torch.utils.data import Dataset

class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # chosen과 rejected를 각각 토크나이징
        chosen_text  = item["prompt"] + item["chosen"]
        rejected_text = item["prompt"] + item["rejected"]
        chosen_ids  = self.tokenizer(chosen_text,  max_length=self.max_len, truncation=True)
        rejected_ids = self.tokenizer(rejected_text, max_length=self.max_len, truncation=True)
        return {
            "chosen_ids":   torch.tensor(chosen_ids["input_ids"]),
            "rejected_ids": torch.tensor(rejected_ids["input_ids"])
        }
```

---

## 7. LLM 효율화 (Inference Efficiency)

### 7.1 KV Caching

**설명**  
토큰을 하나씩 생성할 때마다 이전 모든 토큰의 **Key, Value를 다시 계산하는 낭비**를 방지. 한 번 계산한 K, V를 캐시에 저장하고 재사용. 속도 대폭 향상, 단 메모리 사용량 증가.

**비유**  
책을 읽을 때 앞 내용을 기억하지 못해 매 페이지마다 처음부터 다시 읽는 것을 막기 위해 **독서 노트(캐시)**에 핵심 내용을 적어두고 참조하는 것.

```python
import torch
import torch.nn as nn

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # KV 캐시 저장소
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=True):
        B, T, d = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if use_cache:
            if self.cache_k is not None:
                # 새 토큰의 K, V만 계산 후 캐시에 추가
                K = torch.cat([self.cache_k, K], dim=1)  # 이전 K + 새 K
                V = torch.cat([self.cache_v, V], dim=1)  # 이전 V + 새 V
            # 캐시 업데이트
            self.cache_k = K.detach()
            self.cache_v = V.detach()

        # Attention 계산 (Q는 새 토큰만, K/V는 전체)
        scale = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return self.W_o(context)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

# KV 캐시 메모리 계산
def kv_cache_memory(batch_size, seq_len, n_layers, n_heads, d_head, dtype_bytes=2):
    """KV 캐시 총 메모리 계산"""
    # K, V 각각 (B, heads, seq_len, d_head)
    per_layer = 2 * batch_size * n_heads * seq_len * d_head * dtype_bytes
    total = per_layer * n_layers
    return total / (1024**3)  # GB 단위

mem_gb = kv_cache_memory(1, 4096, 32, 32, 128)
print(f"KV Cache 메모리: {mem_gb:.2f} GB")
```

---

### 7.2 GQA (Grouped Query Attention)

**설명**  
Multi-Head Attention에서 **여러 Query Head가 K, V를 공유**하도록 그룹화. KV 캐시 메모리를 획기적으로 줄이면서 성능 유지. (LLaMA 2, Mistral 등에서 사용)

```
MHA (Multi-Head Attention):  Q1-K1-V1, Q2-K2-V2, ..., Q32-K32-V32
GQA (Grouped Query Attention): Q1,Q2,Q3,Q4 → K1-V1  (4개 Head가 1개 KV 공유)
MQA (Multi-Query Attention):   Q1,...,Q32  → K1-V1  (모든 Head가 1개 KV 공유)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """
    GQA: n_heads Query, n_kv_heads Key/Value
    n_kv_heads가 n_heads보다 작으면 여러 Query가 같은 K,V 공유
    """
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads  # 각 KV를 몇 번 반복?
        self.d_head     = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads    * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_head, d_model,    bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # K, V를 n_rep번 반복해서 n_heads개로 확장
        K = K.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T, d_head)
        V = V.repeat_interleave(self.n_rep, dim=1)

        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V)
        out    = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

# KV 캐시 절약 효과
print(f"MHA KV 파라미터: {32 * 128}  (32 heads × 128 d_head)")
print(f"GQA KV 파라미터: {8  * 128}  (8 kv_heads × 128 d_head) → 4배 절약")
```

---

### 7.3 MLA (Multi-Head Latent Attention)

**설명**  
DeepSeek에서 제안. K, V 전체를 캐시하는 대신 **압축된 Latent Vector(저차원)**만 캐시하고, 필요할 때 복원. GQA보다 더 큰 배치와 긴 문맥 처리 가능.

```
기존 KV 캐시: K (B, heads, T, d_head) + V (B, heads, T, d_head) → 큰 메모리
MLA 캐시:    c_kv (B, T, d_latent)                               → 훨씬 작은 메모리
             d_latent << heads × d_head
```

```python
import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    """
    MLA: KV를 Latent 벡터로 압축해 캐시
    - 압축: x → c_kv (d_latent, 작은 차원)
    - 복원: c_kv → K, V (필요할 때)
    - 모든 Head가 동일한 Latent Vector 공유
    """
    def __init__(self, d_model=4096, n_heads=32, d_latent=512):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_latent = d_latent  # d_latent << n_heads × d_head

        self.W_q   = nn.Linear(d_model, n_heads * self.d_head)
        # KV 압축: d_model → d_latent (캐시 대상)
        self.W_dkv = nn.Linear(d_model, d_latent)
        # KV 복원: d_latent → K, V
        self.W_uk  = nn.Linear(d_latent, n_heads * self.d_head)
        self.W_uv  = nn.Linear(d_latent, n_heads * self.d_head)
        self.W_o   = nn.Linear(n_heads * self.d_head, d_model)
        # Latent 캐시
        self.cache_c = None

    def forward(self, x, use_cache=True):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # KV 압축 → 이것만 캐시
        c_kv = self.W_dkv(x)  # (B, T, d_latent) ← 캐시 대상

        if use_cache and self.cache_c is not None:
            c_kv = torch.cat([self.cache_c, c_kv], dim=1)
        if use_cache:
            self.cache_c = c_kv.detach()

        # 필요할 때 K, V 복원 (연산 비용 < 메모리 비용)
        K = self.W_uk(c_kv).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_uv(c_kv).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

---

### 7.4 MOE (Mixture of Experts)

**설명**  
전체 모델의 모든 뉴런을 가동하는 대신, 입력마다 **관련된 소수의 전문가(Expert) 네트워크만 선택적으로 활성화**. 파라미터 수는 많지만 실제 연산량은 적어 Inference 속도 향상.

**비유**  
고객센터에 전화할 때 모든 상담원이 모든 업무를 다 아는 게 아니라, "5G 기지국 전문가", "요금제 전문가" 등 담당자를 Router가 연결해주는 것.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    """
    MOE 레이어: 
    - Router: 어떤 Expert를 선택할지 결정
    - Experts: 각각 독립된 FFN
    - Top-K 선택: 보통 Top-2 Expert만 활성화
    """
    def __init__(self, d_model=768, num_experts=8, top_k=2, d_ff=3072):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Router: 어떤 Expert가 이 토큰을 처리할지 결정
        self.router = nn.Linear(d_model, num_experts)
        # Expert 네트워크들 (각각 독립적인 FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, d = x.shape
        x_flat = x.view(B * T, d)  # (B*T, d_model)

        # 1. Router: 각 토큰에 대해 Expert 점수 계산
        router_logits = self.router(x_flat)       # (B*T, num_experts)
        router_probs  = F.softmax(router_logits, dim=-1)

        # 2. Top-K Expert 선택
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 정규화

        # 3. 선택된 Expert만 활성화하여 출력 계산
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = top_k_indices[:, k]   # 각 토큰의 k번째 Expert ID
            weights    = top_k_probs[:, k:k+1]  # 해당 가중치

            for expert_id in range(self.num_experts):
                mask = (expert_ids == expert_id)  # 이 Expert를 선택한 토큰들
                if mask.any():
                    expert_input  = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += weights[mask] * expert_output  # 가중합

        return output.view(B, T, d)

# 효율성: 8개 Expert 중 2개만 활성화 → 연산량 25%로 절약
moe = MixtureOfExperts(num_experts=8, top_k=2)
x = torch.randn(2, 10, 768)
out = moe(x)
print(out.shape)  # torch.Size([2, 10, 768])
```

---

### 7.5 Sliding Window Attention

**설명**  
긴 문장을 처리할 때 모든 이전 토큰을 다 참조하면 O(T²) 연산이 폭발. **최근 N개 토큰만 참조하는 창문(Window)**을 설정해 연산량을 O(T×W)로 줄임. (Mistral 7B 등에서 사용)

**비유**  
신문 기사를 읽을 때 30페이지 분량의 이전 내용을 전부 기억하는 대신, **최근 3페이지만 기억**하며 읽어나가는 것.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention:
    각 토큰은 자신으로부터 window_size 이내의 토큰만 참조
    """
    def __init__(self, d_model=768, n_heads=12, window_size=512):
        super().__init__()
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.window_size = window_size
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, d = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Sliding Window 마스크 생성
        window_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        for i in range(T):
            # i번 토큰은 [i-window_size, i] 범위만 참조
            start = max(0, i - self.window_size)
            window_mask[i, :start] = False  # 창문 밖은 참조 불가
            window_mask[i, i+1:]  = False   # 미래 토큰도 참조 불가 (Causal)

        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)

# 연산량 비교
T = 32768  # 32K 토큰
W = 4096   # Window 크기
print(f"Full Attention:   {T * T:,} 연산")
print(f"Sliding Window:   {T * W:,} 연산  ({T//W}배 절약)")
```

---

### 7.6 Flash Attention

**설명**  
GPU의 느린 메모리(HBM)와 빠른 메모리(SRAM) 사이의 **I/O 병목**을 수학적으로 해결한 혁신 알고리즘. Tiling과 Online Softmax를 활용해 Attention 행렬(N²)을 HBM에 저장하지 않고 SRAM 내에서 처리. 학습 속도 2~3배 향상, 메모리 O(N²) → O(N) 절감.

**비유**  
편의점(SRAM, 빠름)과 멀리 있는 대형마트(HBM, 느림) 중에서, 기존 Attention은 재료를 대형마트에 계속 왔다갔다 가져오는 방식. Flash Attention은 편의점에서 작은 블록(Tile)씩 꺼내서 바로바로 처리해 이동 횟수를 획기적으로 줄임.

```
기존 Attention 메모리 흐름:
HBM → SRAM: Q, K 로드     → HBM: S(=QK^T) 저장
HBM → SRAM: S 로드         → HBM: P(=softmax(S)) 저장
HBM → SRAM: P, V 로드      → HBM: O(=PV) 저장
총 HBM 접근: N² 크기 데이터 × 여러 번

Flash Attention Tiling:
작은 블록씩 SRAM에 올려 → Online Softmax로 한 번에 처리
HBM 접근: O(N) 으로 대폭 감소
```

```python
import torch
import torch.nn.functional as F
import math

# ── Flash Attention (개념 구현) ──────────────────────────────
def flash_attention_tiling(Q, K, V, block_size=64):
    """
    Tiling 기반 Flash Attention 구현
    핵심: 전체 N×N 행렬을 메모리에 올리지 않고
          작은 블록(block_size)씩 처리
    """
    B, H, T, d = Q.shape
    scale = d ** -0.5
    O = torch.zeros_like(Q)       # 출력 초기화
    l = torch.zeros(B, H, T, 1)   # 분모 누적 (softmax 정규화용)
    m = torch.full((B, H, T, 1), float('-inf'))  # 최대값 추적 (수치 안정)

    # K, V를 블록 단위로 처리
    for j in range(0, T, block_size):
        K_block = K[:, :, j:j+block_size, :]  # (B, H, block, d) ← SRAM에 올림
        V_block = V[:, :, j:j+block_size, :]

        # Q 블록과 K 블록의 Attention Score 계산 (SRAM 내에서)
        S_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Online Softmax: 새 블록의 max 값으로 누적 합 보정
        m_new = torch.maximum(m, S_block.max(dim=-1, keepdim=True).values)
        exp_S = torch.exp(S_block - m_new)      # 수치 안정 softmax

        # 누적 합 보정 (이전 블록 결과 × 보정 계수)
        l = torch.exp(m - m_new) * l + exp_S.sum(dim=-1, keepdim=True)
        O = torch.exp(m - m_new) * O + torch.matmul(exp_S, V_block)
        m = m_new

    # 최종 정규화
    O = O / l
    return O  # HBM에 한 번만 씀 ← 핵심!

# ── 실제 사용 (PyTorch 2.0+) ────────────────────────────────
# PyTorch 2.0부터 Flash Attention이 내장됨
Q = torch.randn(2, 12, 128, 64, device='cuda')
K = torch.randn(2, 12, 128, 64, device='cuda')
V = torch.randn(2, 12, 128, 64, device='cuda')

# scaled_dot_product_attention이 내부적으로 Flash Attention 사용
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V)

# 또는 xformers / flash-attn 라이브러리 사용
# pip install flash-attn
# from flash_attn import flash_attn_func
# output = flash_attn_func(Q, K, V, causal=True)

print("Flash Attention 핵심 최적화:")
print("1. Tiling: 큰 행렬을 작은 블록으로 쪼개 SRAM에서 처리")
print("2. Online Softmax: 블록별 softmax를 수학적으로 통합")
print("3. Recomputation: 역전파 시 Attention 행렬을 재계산 (메모리 절약)")
print("결과: HBM I/O O(N²) → O(N), 속도 2-3배 향상")
```

---

*본 문서는 AI Application Specialist 강의자료(p.97~) 및 AI_Advanced_Concepts.md를 바탕으로 작성되었습니다.*
