# Chapter 4-4: Applications — Object Detection · Tracking · Segmentation 정리

> AI Specialist 양성과정 | CV Applications 핵심 정리  
> 참고 실습: `03_DETR.ipynb`, `04_Unet.ipynb`

---

## 전체 구조 한눈에 보기

PDF 전체는 **3개 주제**로 구성되며, 각 주제는 공통적으로 **개념 → Metric → 모델 진화(고전 → CNN → Transformer)** 순서로 전개됩니다.

| 주제 | 핵심 질문 | 슬라이드 |
|------|-----------|---------|
| Object Detection | 이미지에서 "무엇이 어디에" | 1–26 |
| Object Tracking | 시간축으로 "동일 객체 유지" | 36–61 |
| Segmentation | 픽셀 단위로 "어디까지가 그 객체" | 62–100 |

> 모델 간 연결 고리: Detection의 **DETR** → Tracking의 **TrackFormer**, Detection의 **Faster R-CNN** → Segmentation의 **Mask R-CNN**

### 1. Object Detection (슬라이드 1–26)

```
1-1. 개념 및 배경
     ├── What? → Classification (클래스 + confidence score)
     ├── Where? → Localization (bounding box + IoU)
     ├── 실세계 응용: 자율주행, 의료, 감시, 농업 등
     └── 어려운 이유: scale 변화, occlusion, pose 변화, 조명

1-2. Detection Metric
     ├── IoU (Intersection over Union)
     ├── Confusion matrix: TP / FP / FN (TN은 무의미)
     ├── Precision / Recall / F1
     ├── AP: PR curve 면적 (단일 클래스)
     ├── mAP: 전체 클래스 AP 평균
     └── COCO metric: AP@[.50:.05:.95], Small/Medium/Large

1-3. 1-stage vs 2-stage
     ├── 2-stage: Region Proposal → 분류/회귀 (정확도 우선)
     └── 1-stage: 이미지 전체 단일 pass (속도 우선)

1-4. 주요 모델 (연대순)
     ├── [Classical]
     │    └── Viola-Jones (2001): Haar feature + AdaBoost + cascade
     ├── [Two-stage]
     │    ├── R-CNN (2014): Selective search + AlexNet + SVM
     │    ├── NMS: 중복 박스 제거 알고리즘
     │    ├── Fast R-CNN (2015): 공유 feature map + RoI Pooling
     │    └── Faster R-CNN (2015): RPN으로 end-to-end 학습
     ├── [One-stage / Anchor-free]
     │    └── CenterNet (2019): center point 표현, heatmap + offset + size
     └── [Transformer-based]
          └── DETR (2020): Set prediction + Hungarian matching
               ├── 구성: CNN backbone → Encoder → Decoder (object query 100개)
               ├── 출력: pred_logits (B,100,92) + pred_boxes (B,100,4)
               ├── GIoU loss: IoU=0 상황도 gradient 생성
               └── NMS 완전 제거, anchor 설계 불필요
```

### 2. Object Tracking (슬라이드 36–61)

```
2-1. 개념 및 어려움
     ├── Occlusion, 재출현, 유사 객체, 빠른 움직임
     ├── Scale 변화, 카메라 움직임, 조명·외형 변화
     └── 실시간 요구사항: 30~60 FPS

2-2. Tracking Metric
     ├── MOTA: 1 − (FN+FP+IDsw)/GT — 종합 오류율
     ├── IDF1: identity 일관성 (IDTP 기반 F1)
     └── HOTA: √(DetA × AssA) — 탐지+연관 균형

2-3. 방법론
     ├── Tracking-by-Detection: 탐지 후 IoU/ReID로 연관 (2-stage)
     └── Tracking-by-Regression: 이전 박스에서 직접 회귀 (1-stage)

2-4. 주요 모델 (연대순)
     ├── [Classical trackers]
     │    ├── Lucas-Kanade (1998): optical flow — 픽셀 밝기 보존 가정
     │    └── Particle filter (2006): 비선형·비가우시안 상태 추정
     ├── [Discriminative trackers]
     │    └── TLD (2010): Tracking + Learning + Detection 통합
     │         └── Median Flow + 온라인 학습 + sliding window 재탐지
     ├── [Tracking-by-Detection]
     │    └── ByteTrack (2022): low-confidence 탐지도 활용 (2단계 매칭)
     │         ├── 1차: high-score ↔ track (IoU 기반)
     │         ├── 2차: low-score ↔ 남은 track (occlusion 구출)
     │         └── Kalman filter 위치 예측, MOT17 80.3 MOTA @ 30FPS
     └── [Transformer-based]
          └── TrackFormer (2021): DETR 확장 — 탐지+추적 통합
               ├── Detection queries: 새 객체 발견
               ├── Track queries: 이전 프레임 임베딩 → 슬롯 번호로 identity 유지
               └── Hungarian matching 학습, NMS/ReID 불필요
```

### 3. Segmentation (슬라이드 62–100)

```
3-1. 개념 및 종류
     ├── Semantic: 픽셀마다 클래스 부여 (Things + Stuff)
     ├── Instance: 인스턴스별 개별 마스크 (Things only)
     ├── Panoptic: Semantic + Instance 통합
     └── 필요 이유: 정밀도, 인스턴스 분리, 장면 이해, 픽셀 단위 결정

3-2. Segmentation Metric
     ├── IoU per class: |P∩G| / |P∪G|
     ├── mIoU: 전체 클래스 IoU 평균 (주요 벤치마크 지표)
     └── FWIoU: 클래스 빈도 가중 IoU

3-3. Classical segmentation
     ├── Thresholding / Otsu
     ├── K-means clustering
     ├── GrabCut (graph-based)
     └── Canny edge + Watershed

3-4. 주요 모델 (연대순)
     ├── FCN (2015): 최초 E2E DL segmentation
     │    ├── FC layer → 1×1 Conv (공간 정보 유지)
     │    └── Skip fusion: FCN-32s / 16s / 8s (해상도 복원)
     ├── U-Net (2015): 의료 영상 특화
     │    ├── Encoder: MaxPool → 채널↑, 해상도↓
     │    ├── Decoder: UpConv → 채널↓, 해상도↑
     │    ├── Skip connection: encoder feature를 decoder에 concat
     │    └── Overlap-tile + elastic transform + 경계 가중 loss
     ├── Mask R-CNN (2017): instance segmentation
     │    ├── Faster R-CNN + FCN on RoI
     │    ├── RoI Align: bilinear interpolation (양자화 오류 제거)
     │    └── Mask head: 클래스별 binary mask (sigmoid + BCE)
     ├── SegFormer (2021): efficient Transformer segmentation
     │    ├── Hierarchical encoder: Efficient Self-Attention (O(N²/R))
     │    ├── Mix-FFN: depth-wise conv → positional encoding 대체
     │    └── All-MLP Decoder: 경량 + 빠른 추론
     └── SAM (2023): Vision Foundation Model
          ├── Promptable: point / box / text / mask 입력
          ├── Image encoder (ViT+MAE) + Prompt encoder + Mask decoder
          ├── SA-1B: 11M 이미지, 1.1B 마스크
          └── → SAM 2 (비디오 지원) → SAM 3 (개념 프롬프트)
```

---

## 목차

1. [Object Detection 개요](#1-object-detection-개요)
2. [Detection Metric](#2-detection-metric)
3. [주요 Object Detection 모델](#3-주요-object-detection-모델)
4. [DETR — Transformer 기반 Detection](#4-detr--transformer-기반-detection)
5. [Object Tracking](#5-object-tracking)
6. [Segmentation 개요](#6-segmentation-개요)
7. [U-Net — 의료 영상 Segmentation](#7-u-net--의료-영상-segmentation)
8. [기타 Segmentation 모델 요약](#8-기타-segmentation-모델-요약)

---

## 1. Object Detection 개요

### 핵심 질문 두 가지

| 질문 | 출력 | 핵심 개념 |
|------|------|-----------|
| **What?** (무엇?) | Class label | Classification + Confidence score |
| **Where?** (어디?) | Bounding box | Localization + IoU |

```
Detection 난이도를 높이는 주요 요인:
- Scale change    : 크기 변화 (가깝고 먼 객체)
- Occlusion       : 부분 가려짐
- Pose change     : 자세/회전 변화
- Background complexity : 유사한 배경/객체
- Light condition : 조명 변화
```

### One-stage vs. Two-stage 비교

```
[One-stage Detector]
Image → Backbone → Regression Box + Classification → 결과
       (빠름, 속도 우선 — YOLO, SSD, RetinaNet, CenterNet)

[Two-stage Detector]
Image → Backbone → Object Proposal → Classification + Regression → 결과
       (정확함, 정밀도 우선 — R-CNN, Fast R-CNN, Faster R-CNN)
```

---

## 2. Detection Metric

### 2-1. IoU (Intersection over Union)

$$IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}$$

```python
def compute_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)
```

### 2-2. Confusion Matrix (Object Detection)

| | Prediction True | Prediction False |
|---|---|---|
| **GT True** | TP | FN |
| **GT False** | FP | TN (무의미) |

**TP 판정 조건**: (Correct class) AND (IoU ≥ threshold) AND (Unique match)

```
Confidence score < threshold → Prediction 제거
IoU threshold: 보통 0.5 (PASCAL VOC), 0.50~0.95 (COCO)
```

### 2-3. Precision / Recall / F1

$$Precision = \frac{TP}{TP+FP}, \quad Recall = \frac{TP}{TP+FN}, \quad F1 = \frac{2 \cdot P \cdot R}{P + R}$$

### 2-4. AP / mAP

```
Average Precision (AP) 계산 절차:
  1) confidence score 내림차순 정렬
  2) threshold별로 TP/FP/FN 계산
  3) Precision-Recall Curve 그리기
  4) AUC (곡선 아래 면적) 계산

mAP = 전체 클래스 AP의 평균

COCO Evaluation:
  - AP@[.50:.05:.95]  : 10개 IoU threshold에서 평균
  - AP@small/medium/large : 크기별 평가
    - Small  : < 32² pixels
    - Medium : 32² ~ 96² pixels
    - Large  : > 96² pixels
```

### 2-5. NMS (Non-Maximum Suppression)

중복 박스 제거, 가장 높은 confidence만 남기는 알고리즘.

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes  : List of [x1, y1, x2, y2]
    scores : confidence scores
    """
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while sorted_idx:
        # 가장 높은 score의 box 선택
        best = sorted_idx.pop(0)
        keep.append(best)

        # 나머지와 IoU 계산 → 높으면 제거
        sorted_idx = [
            idx for idx in sorted_idx
            if compute_iou(boxes[best], boxes[idx]) < iou_threshold
        ]
    return keep
```

---

## 3. 주요 Object Detection 모델

### 3-1. 진화 타임라인

```
[Traditional] VJ Detector(2001) → HOG(2005) → DPM(2008) → AlexNet(2012)

[Two-stage]
  R-CNN(2014) → Fast R-CNN(2015) → Faster R-CNN(2015)

[One-stage]
  YOLO(2015) → SSD(2016) → RetinaNet(2017) → CenterNet(2019)

[Transformer]
  Transformer(2017) → DETR(2020) → Swin Transformer(2021)
```

### 3-2. R-CNN 계열 요약

| 모델 | 속도 | mAP (VOC'07) | 핵심 개선 |
|------|------|-------------|-----------|
| R-CNN | 50 sec/img | 66.0 | 최초 딥러닝 탐지기, Selective Search |
| Fast R-CNN | 2 sec/img | 66.9 | 공유 feature map, RoI Pooling |
| Faster R-CNN | **0.2 sec/img** | 66.9 | **RPN** 도입 — CPU-bound 제거 |

```python
# Faster R-CNN 핵심: RPN Anchor 생성 pseudo code
# 9 anchors = 3 scales × 3 ratios

ANCHOR_SCALES = [8, 16, 32]    # 실제 픽셀 크기로 환산 시 stride 곱
ANCHOR_RATIOS = [0.5, 1, 2]    # w:h 비율

def generate_anchors(feature_map_h, feature_map_w, stride=16):
    anchors = []
    for y in range(feature_map_h):
        for x in range(feature_map_w):
            for scale in ANCHOR_SCALES:
                for ratio in ANCHOR_RATIOS:
                    # 중심점 (원본 이미지 좌표계)
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    w = scale * stride * (ratio ** 0.5)
                    h = scale * stride / (ratio ** 0.5)
                    anchors.append([cx, cy, w, h])
    return anchors  # 총 H/stride * W/stride * 9 개

# RPN 출력 shape
# Classification: [2 × 9] = 18  (object vs background)
# Bounding box:   [4 × 9] = 36  (offset 회귀)
```

### 3-3. CenterNet (2019)

Anchor-free 접근 — 각 객체를 **중심점(Center point)**으로 표현.

```python
# CenterNet 출력 3가지
# 1) Keypoint heatmap  : (B, num_classes, H/4, W/4)
# 2) Local offset      : (B, 2, H/4, W/4)  — quantization error 보정
# 3) Object size       : (B, 2, H/4, W/4)  — width, height

# 손실 함수
# - Heatmap  : Modified Focal Loss (Gaussian kernel 기반)
# - Offset   : L1 Loss
# - Size     : L1 Loss

def decode_centermap(heatmap, offset, size, topk=100, threshold=0.3):
    """CenterNet inference pseudo code"""
    # 1) Peak 탐지 (local max)
    peaks = find_local_maxima(heatmap)
    # 2) confidence threshold 필터링
    peaks = [p for p in peaks if p.score > threshold][:topk]
    # 3) 박스 복원
    boxes = []
    for p in peaks:
        cx = p.x + offset[0, p.y, p.x]
        cy = p.y + offset[1, p.y, p.x]
        w  = size[0, p.y, p.x]
        h  = size[1, p.y, p.x]
        boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2, p.score, p.class_id])
    return boxes
```

---

## 4. DETR — Transformer 기반 Detection

> 논문: "End-to-End Object Detection with Transformers" (ECCV 2020)

### 4-1. 핵심 아이디어

```
기존 방식의 문제점:
  - Hand-crafted anchor 설계 필요
  - NMS(post-processing) 의존
  - 복잡한 multi-stage (RPN, FPN 등)

DETR의 해결책:
  - "Object detection as Set Prediction"
  - NMS 불필요 → Bipartite Matching (Hungarian Algorithm)
  - End-to-End Transformer 구조
```

### 4-2. 전체 아키텍처

```
Input Image (3 × H₀ × W₀)
    │
    ▼
[Backbone: ResNet-50]
    f ∈ ℝ^{C×h×w}   (C=2048, h=H₀/32, w=W₀/32)
    │
    ▼ 1×1 Conv (C→d=256)
    z₀ ∈ ℝ^{d×hw}  + Positional Encoding P ∈ ℝ^{d×hw}
    │
    ▼
[Transformer Encoder] (N layers)
    Multi-Head Self-Attention + FFN
    oₑ ∈ ℝ^{d×hw}
    │
    ▼
[Transformer Decoder] (M layers)
    Object Queries q ∈ ℝ^{d×N}  (N=100, learnable)
    Cross-Attention with encoder output
    │
    ▼
[Prediction Heads]
    FFN → Class p(cᵢ) ∈ ℝ^{N×(C+1)}   (+1은 no-object)
    FFN → BBox  b̂ ∈ ℝ^{N×4}
```

### 4-3. 실습 코드 (DETR 추론 파이프라인)

```python
import torch
import torchvision.transforms as T
from PIL import Image
import requests

# ── 1) 전처리 파이프라인 ─────────────────────────────────────────
transform = T.Compose([
    T.Resize(800),                                         # 짧은 변 기준 800px
    T.ToTensor(),                                          # (C, H, W), 0~1
    T.Normalize([0.485, 0.456, 0.406],                    # ImageNet mean
                [0.229, 0.224, 0.225])                    # ImageNet std
])

# ── 2) 모델 로드 (torch.hub) ──────────────────────────────────────
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()
torch.set_grad_enabled(False)

# ── 3) 이미지 준비 ───────────────────────────────────────────────
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im  = Image.open(requests.get(url, stream=True).raw)
img = transform(im).unsqueeze(0)  # (1, 3, H, W)

# ── 4) 추론 ──────────────────────────────────────────────────────
outputs = model(img)
# pred_logits: (1, 100, 92)  — 100 object queries, 92 classes (91+no-obj)
# pred_boxes : (1, 100,  4)  — 정규화된 (cx, cy, w, h)

# ── 5) 후처리: confidence 필터링 + 박스 복원 ──────────────────────
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([(x_c - 0.5*w), (y_c - 0.5*h),
                        (x_c + 0.5*w), (y_c + 0.5*h)], dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # no-object 제외
keep   = probas.max(-1).values > 0.9                     # confidence > 0.9

bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
# keep.shape        → (100,)
# bboxes_scaled.shape → (N_detected, 4)
```

**출력 텐서 Dimension 정리**

```
pred_logits  : (B=1, N_queries=100, C+1=92)
                └─ softmax(-1) 후 마지막 차원이 no-object
pred_boxes   : (B=1, N_queries=100, 4)
                └─ 정규화 (cx, cy, w, h) → 픽셀 변환 필요
```

### 4-4. Hungarian Loss

```python
# Bipartite Matching (Hungarian Algorithm) pseudo code

def hungarian_matching(pred_logits, pred_boxes, targets):
    """
    GT와 예측 간 최적 1:1 매칭을 찾아 loss 계산
    - 매칭 비용 = 클래스 cost + bounding box cost (L1 + GIoU)
    """
    N = len(pred_logits)       # num_queries (e.g. 100)
    M = len(targets['labels']) # num_gt_objects

    # Cost matrix: (N, M)
    cost_class = -pred_logits[:, targets['labels']]  # 음수 로짓 (낮을수록 좋음)
    cost_bbox  = torch.cdist(pred_boxes, targets['boxes'], p=1)  # L1
    cost_giou  = -generalized_iou(pred_boxes, targets['boxes'])  # GIoU

    # λ 가중치 적용
    C = cost_class + 5 * cost_bbox + 2 * cost_giou  # (N, M)

    # Hungarian Algorithm: 최솟값 매칭
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
    return row_ind, col_ind  # 매칭된 (pred_idx, gt_idx) 쌍

# 최종 Hungarian Loss
# L_Hungarian = Σ [-log p̂_{σ(i)}(c_i) + L_box(b_i, b̂_{σ(i)})]
# ※ NMS 불필요 — 1:1 매칭이 보장되므로 중복 탐지 없음
```

### 4-5. GIoU vs. IoU

```
IoU 문제: 박스가 겹치지 않으면 항상 0 → gradient 없음
           겹치지 않는 여러 배치가 같은 IoU=0 → 방향 정보 없음

GIoU = IoU - (C \ (A∪B)) / |C|
        └─ 두 박스를 감싸는 최소 박스(C) 대비 빈 공간 패널티

Loss_GIoU = 1 - GIoU   (범위: 0 < Loss ≤ 2)
```

### 4-6. Attention 시각화 (Forward Hook)

```python
# Encoder self-attention 및 Decoder cross-attention 추출

conv_features  = {}
enc_attn_weights = []
dec_attn_weights = []

# Hook 등록 예시
def make_hook(storage):
    def hook(module, input, output):
        storage.append(output[1])  # attention weight (index 1)
    return hook

# Encoder 마지막 레이어 hook
model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
    make_hook(enc_attn_weights)
)

# Decoder 마지막 레이어 cross-attention hook
model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
    make_hook(dec_attn_weights)
)

# 추론 수행
outputs = model(img)

# Encoder self-attention: (h*w, h*w) → reshape to (h, w, h, w)
h, w = f_map.tensors.shape[-2:]  # feature map 해상도
enc_attn = enc_attn_weights[0]   # (850, 850) — 25*34=850
enc_attn_2d = enc_attn.reshape(h, w, h, w)  # (25, 34, 25, 34)

# Decoder cross-attention: (num_queries, h*w)
# keep에 해당하는 query의 attention map을 2D로 복원
dec_attn = dec_attn_weights[0]  # (num_queries, h*w)
```

---

## 5. Object Tracking

### 5-1. 추적의 어려움

```
Occlusion(가림), 재출현, 유사한 객체, 빠른 움직임,
스케일 변화, 카메라 움직임, 조명변화, 외형 급변
```

### 5-2. 추적 방법론 비교

| 특징 | Tracking-by-Detection | Tracking-by-Regression |
|------|----------------------|----------------------|
| 구조 | 2-stage | 1-stage |
| 방식 | 탐지 후 연관(IoU/ReID) | 이전 박스에서 회귀 |
| Identity 유지 | 프레임 간 매칭 | 프레임 간 전파 |
| 탐지 에러 영향 | 치명적 | 덜 민감 |
| Occlusion 복구 | Re-identification | 예측(Prediction) |

### 5-3. 추적 Metric

```
MOTA = 1 - (FN + FP + IDsw) / GT     ← 종합 오류율
IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)  ← Identity 일관성
HOTA = √(DetA × AssA)                ← 탐지+연관+위치 균형
```

### 5-4. ByteTrack (ECCV 2022)

**핵심 아이디어**: 낮은 confidence 탐지도 버리지 말고 활용

```python
# ByteTrack pseudo code
def byte_track(frames, detector, threshold_high=0.7, threshold_low=0.1):
    tracks = []

    for frame in frames:
        # ① 탐지 실행
        detections = detector(frame)

        # ② confidence 기준 분리
        D_high = [d for d in detections if d.score > threshold_high]
        D_low  = [d for d in detections if threshold_low < d.score <= threshold_high]

        # ③ Kalman Filter로 기존 트랙 위치 예측
        for t in tracks:
            t.predict()  # 현재 프레임 위치 추정

        # ④ 1차 매칭: 기존 트랙 ↔ high-confidence 탐지
        matched, unmatched_tracks, unmatched_high = \
            associate(tracks, D_high, metric='iou')

        # ⑤ 2차 매칭: 미매칭 트랙 ↔ low-confidence 탐지
        #    occlusion/blur로 낮아진 score의 실제 객체 구출
        matched_low, lost_tracks, _ = \
            associate(unmatched_tracks, D_low, metric='iou')

        # ⑥ 완전 미매칭 트랙 → buffer 보관
        #    (잠시 후 재출현 가능)

        # ⑦ 새 객체 초기화 (unmatched high detections)
        for d in unmatched_high:
            tracks.append(Track(d))

    return tracks

# 핵심 통찰:
# "Low confidence detections are not noise, they are temporal evidence."
# ByteTrack: MOT17 80.3 MOTA, 77.3 IDF1 @ 30 FPS 달성
```

### 5-5. TrackFormer (2021)

DETR 확장 → 탐지와 추적을 하나의 Transformer로 처리.

```
[DETR]
  Object Queries → 각 프레임에서 독립 탐지

[TrackFormer]
  Detection Queries : 새 객체 발견
  Track Queries     : 이전 프레임의 탐지 임베딩 → 다음 프레임으로 전달
  
  Frame t-1 decoder output  →  Frame t decoder input (Track Queries)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Autoregressive 방식으로 identity가 자연스럽게 전파됨
  NMS 불필요, Hand-crafted association 불필요
```

---

## 6. Segmentation 개요

### 6-1. 세그멘테이션이 필요한 이유

- **정밀도**: 정확한 경계 필요 (의료 영상, 로봇)
- **인스턴스 분리**: 겹치는 동일 클래스 객체 구분
- **장면 이해**: 도로/하늘/보행자 전체 픽셀 파악
- **픽셀 단위 의사결정**: 미세 결함 검출, AR 배경 제거

### 6-2. 세그멘테이션 종류

```
Semantic Segmentation : 픽셀마다 클래스 부여 (Things + Stuff)
Instance Segmentation : 각 인스턴스 개별 마스크 (Things only)
Panoptic Segmentation : Semantic + Instance 통합 (Things + Stuff)
```

### 6-3. 주요 Metric

$$IoU_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|}, \quad mIoU = \frac{1}{C}\sum_{c=1}^{C}IoU_c$$

```python
def compute_miou(preds, targets, num_classes):
    """
    preds   : (N, H, W) — 예측 class index
    targets : (N, H, W) — GT class index
    """
    iou_list = []
    for c in range(num_classes):
        pred_c  = (preds == c)
        gt_c    = (targets == c)
        inter   = (pred_c & gt_c).sum().float()
        union   = (pred_c | gt_c).sum().float()
        if union == 0:
            continue
        iou_list.append((inter / union).item())
    return sum(iou_list) / len(iou_list)
```

### 6-4. FCN (2015) — 최초 딥러닝 Segmentation

```
핵심 변경: FC layer → 1×1 Conv
           (공간 정보 유지 가능)

출력: 입력과 동일한 H×W 크기의 segmentation map

Skip Fusion:
  FCN-32s : deepest layer만 (가장 거칠음)
  FCN-16s : + pool4 skip
  FCN-8s  : + pool3 skip (가장 정밀)
```

---

## 7. U-Net — 의료 영상 Segmentation

> 논문: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

### 7-1. 아키텍처 개요

```
[Contracting Path — Encoder]
Input (B, 1, H, W)
  → DoubleConv → (B, 64, H, W)      [x1, skip 저장]
  → MaxPool + DoubleConv → (B, 128, H/2, W/2)   [x2]
  → MaxPool + DoubleConv → (B, 256, H/4, W/4)   [x3]
  → MaxPool + DoubleConv → (B, 512, H/8, W/8)   [x4]
  → MaxPool + DoubleConv → (B, 1024, H/16, W/16) [bottleneck, x5]

[Expanding Path — Decoder]
  → UpConv + concat(x4) + DoubleConv → (B, 512, H/8, W/8)
  → UpConv + concat(x3) + DoubleConv → (B, 256, H/4, W/4)
  → UpConv + concat(x2) + DoubleConv → (B, 128, H/2, W/2)
  → UpConv + concat(x1) + DoubleConv → (B, 64, H, W)
  → 1×1 Conv → (B, n_classes, H, W)

Skip Connection: Encoder 각 단계의 feature map을 Decoder에 concat
                 → 공간 정보(boundary detail) 복원
```

### 7-2. 실습 코드 (PyTorch 구현)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 기본 블록: (Conv3×3 → ReLU) × 2 ─────────────────────────────
class DoubleConv(nn.Module):
    """
    입력: (B, C_in, H, W)
    출력: (B, C_out, H, W)   ← padding=1이라 H,W 유지
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


# ── 인코더 블록 ───────────────────────────────────────────────────
class Down(nn.Module):
    """
    입력: (B, C_in, H, W)
    출력: (B, C_out, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)   # 해상도 1/2
        x = self.conv(x)   # 채널 확장 + 특징 추출
        return x


# ── 디코더 블록 ───────────────────────────────────────────────────
class Up(nn.Module):
    """
    x1: 디코더 feature (B, C_dec, H/2, W/2)
    x2: 인코더 skip    (B, C_skip, H, W)
    출력: (B, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transpose Conv: 파라미터 포함 업샘플, 채널 동시 축소
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)                   # 해상도 2배

        # 크기 정렬 (홀수 입력 대응)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x//2,
                         diff_y//2, diff_y - diff_y//2])

        x = torch.cat([x2, x1], dim=1)     # Skip concat (채널 방향)
        return self.conv(x)


# ── 출력 헤드 ─────────────────────────────────────────────────────
class OutConv(nn.Module):
    """1×1 Conv으로 채널 → 클래스 수 변환"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


# ── 전체 U-Net ────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    채널 구성 (bilinear=False):
      Encoder: C → 64 → 128 → 256 → 512 → 1024
      Decoder: 1024 → 512 → 256 → 128 → 64 → n_classes
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super().__init__()
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512,  256 // factor, bilinear)
        self.up3   = Up(256,  128 // factor, bilinear)
        self.up4   = Up(128,  64,            bilinear)

        # Output
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder (skip 저장)
        x1 = self.inc(x)       # (B, 64,   H,    W)
        x2 = self.down1(x1)    # (B, 128,  H/2,  W/2)
        x3 = self.down2(x2)    # (B, 256,  H/4,  W/4)
        x4 = self.down3(x3)    # (B, 512,  H/8,  W/8)
        x5 = self.down4(x4)    # (B, 1024, H/16, W/16)

        # Decoder (skip concat)
        x = self.up1(x5, x4)   # (B, 512, H/8,  W/8)
        x = self.up2(x,  x3)   # (B, 256, H/4,  W/4)
        x = self.up3(x,  x2)   # (B, 128, H/2,  W/2)
        x = self.up4(x,  x1)   # (B, 64,  H,    W)

        return self.outc(x)    # (B, n_classes, H, W)
```

### 7-3. 학습 루프

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = UNet(n_channels=1, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()   # Sigmoid + BCE (binary segmentation)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# ── 학습 1 epoch ──────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)              # (B, 1, H, W)
        loss   = criterion(logits, masks) # 마스크는 float (0/1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# ── 검증 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# ── 메인 학습 루프 ────────────────────────────────────────────────
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss   = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch [{epoch}/{num_epochs}]  train: {train_loss:.4f}  val: {val_loss:.4f}")

# ── 예측 시각화 ───────────────────────────────────────────────────
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    img_sample, mask_sample = next(iter(val_loader))
    logit = model(img_sample.to(device))
    pred  = torch.sigmoid(logit) > 0.5    # 이진화

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_sample[0, 0].cpu(), cmap='gray'); axes[0].set_title("Input")
axes[1].imshow(mask_sample[0, 0].cpu(), cmap='gray'); axes[1].set_title("GT Mask")
axes[2].imshow(pred[0, 0].cpu(), cmap='gray'); axes[2].set_title("Prediction")
plt.tight_layout(); plt.show()
```

### 7-4. U-Net 학습 특징

```
Overlap-tile strategy:
  - 고해상도 의료 이미지를 tile로 분할 처리
  - 각 tile을 미러링 패딩(mirroring extrapolation)으로 확장 → 경계 아티팩트 방지

Elastic transformation (데이터 증강):
  - 의료 이미지의 형태 변형(deformation)에 불변하도록 학습
  - 라벨링 비용이 높아 소규모 데이터셋에 특히 효과적

가중치 맵 손실 (Weighted Loss):
  w(x) = w_c(x) + w₀·exp(-(d₁(x)+d₂(x))²/2σ²)
  ① w_c : 클래스 빈도 역수 → 클래스 불균형 보정
  ② 지수 항 : 인접 셀 경계 픽셀에 높은 가중치 → 경계 분리 강조
```

---

## 8. 기타 Segmentation 모델 요약

### 8-1. Mask R-CNN (ICCV 2017)

```
= Faster R-CNN + FCN on RoI

핵심 변경점:
  - RoI Pooling → RoI Align (bilinear interpolation, 정확도 향상)
  - Mask head 추가: 각 클래스별 binary mask 예측 (sigmoid)

Multi-task Loss:
  L = L_cls + L_bbox + L_mask
  L_mask: pixel-wise BCE (클래스별 독립 이진 분류)

RoI Align vs. RoI Pooling:
  Pooling : 정수 좌표로 반올림 → misalignment → 부정확한 마스크
  Align   : bilinear interpolation → 소수 좌표 처리 → 정확한 마스크
```

### 8-2. SegFormer (2021)

```
= Hierarchical Transformer Encoder + Lightweight All-MLP Decoder

Encoder 특징:
  - Overlap Patch Merging (kernel > stride): 경계 아티팩트 감소
  - Efficient Self-Attention: K,V 압축 (O(N²) → O(N²/R))
    reduction ratio R = [64, 16, 4, 1] (깊을수록 작아짐)
  - Mix-FFN: depth-wise conv 포함 (위치 정보 자연 인코딩, PE 불필요)

Decoder:
  - 4단계 feature map을 MLP로 통합 (H/4 해상도 기준)
  - FPN이나 복잡한 conv 없이 빠르고 정확

성능 (ADE20K):
  SegFormer-B0: 37.4 mIoU, 3.7M params, 50.5 FPS
  SegFormer-B5: 51.0 mIoU, 84.7M params, 15.2 FPS
```

### 8-3. SAM — Segment Anything Model (2023)

```
목표: 어떤 객체든 추론 없이 즉시 분할 (Vision Foundation Model)

Task: Promptable Segmentation
  Prompt 종류: Point / Bounding Box / Text / Mask

Architecture:
  Image Encoder  : ViT (MAE 사전학습)
  Prompt Encoder : Point/Box → Positional embedding
                   Text → CLIP text encoder
                   Mask → Conv + pixel-wise sum
  Mask Decoder   : Lightweight Transformer
                   Cross-attention (image ↔ prompt)
                   출력: 3개 마스크 후보 + IoU score

Dataset: SA-1B (11M 이미지, 1.1B 마스크 — COCO 대비 약 400배)

활용:
  Zero-shot segmentation / Edge detection / Object proposal
  Instance segmentation / Text-based segmentation

진화:
  SAM (2023) → SAM 2 (2024, 비디오 지원) → SAM 3 (2025, 개념 프롬프트)
```

---

## 핵심 정리 표

| 모델 | 연도 | 태스크 | 핵심 기여 |
|------|------|--------|----------|
| R-CNN | 2014 | Detection | 딥러닝 기반 최초 탐지 |
| Faster R-CNN | 2015 | Detection | RPN — End-to-End 학습 |
| YOLO | 2015 | Detection | One-stage, 실시간 |
| CenterNet | 2019 | Detection | Anchor-free, 중심점 표현 |
| **DETR** | 2020 | Detection | Transformer + Bipartite Matching, NMS 제거 |
| ByteTrack | 2022 | Tracking | Low-conf 탐지 활용 |
| TrackFormer | 2021 | Tracking | DETR 기반 E2E 추적 |
| FCN | 2015 | Segmentation | 최초 DL Segmentation |
| **U-Net** | 2015 | Segmentation | Skip connection, 의료 영상 |
| Mask R-CNN | 2017 | Instance Seg | RoI Align + Mask head |
| SegFormer | 2021 | Semantic Seg | Efficient Transformer |
| SAM | 2023 | Universal Seg | Promptable, Foundation Model |

---

> **실습 노트북 연결**
> - `03_DETR.ipynb`: DETR 추론 파이프라인, pred_logits/pred_boxes 해석, Encoder/Decoder Attention 시각화
> - `04_Unet.ipynb`: U-Net PyTorch 구현 (DoubleConv → Down → Up → OutConv), 학습/검증 루프, 예측 마스크 시각화
