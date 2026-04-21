# RRC State Predictor 기능 및 분석 내용 정리

---

## 1. 프로젝트 개요

```
목적:
  gNB-CU-CP VCCB SW에 GRU 기반 AI를 연동하여
  UE별 RRC Inactivity Timer를 동적으로 최적화

핵심 아이디어:
  정적 타이머(고정값) → AI 동적 타이머(UE별 예측값)
  → 불필요한 RRCSetup 감소
  → Resume 성공률 향상
```

---

## 2. 아키텍처

```
vccb.xcc 내부에 GRU 추론 직접 포함
  (별도 프로세스 없음, SHM 없음, 외부 라이브러리 없음)

Python (PyTorch 학습)
  → state_dict() 가중치 추출
  → export_weights.py로 VccbAiWeights.h 자동 생성
  → C++에서 GRU 순전파 직접 구현
  → vccb.xcc 빌드에 포함

모델 갱신:
  VccbAiWeights.h 재생성 → VCCB 재빌드 → 재배포
```

---

## 3. GRU 모델 구조

```
입력:
  시퀀스 Feature: seq[10][2]
    - durationSec   (INACTIVE 지속 시간)
    - resumeCause   (Resume 원인: 0~3)

  정적 Feature: stat[15]
    - primary_5qi, drb_count, ue_ambr_dl_mbps
    - has_gbr_flow, cn_expected_idle_period
    - cn_stationary, last_rsrp, neighbor_cell_count
    - rna_cell_count, avg_cell_stay_sec
    - resume_count_24h, resume_success_rate
    - avg_inactive_duration, hour_of_day, day_of_week

출력:
  resume_prob  (Resume 확률: 0.0~1.0)
  resume_time  (예상 Resume 시간: 초)

모델 크기:
  hidden_size = 16
  총 파라미터 = 1,282개
```

---

## 4. 타이머 결정 로직

```
resume_prob >= 0.6 (KEEP):
  → recommendedTimer = resume_time × 1.2, 올림

resume_prob <= 0.3 (EARLY_RELEASE):
  → recommendedTimer = 1초 (최솟값)

중간 범위 (DEFAULT):
  → 설정 기반 정적 타이머 사용

공통:
  → clamp(result, 1초, 3600초)
```

---

## 5. AI 개입 포인트 (VCCB 코드)

| 포인트 | 파일 | 함수 | 개입 내용 |
|---|---|---|---|
| ① | MsgE1apMake.cpp | `VEIL_fillE1_Inactivity_Timer()` | AI 추론 타이머값으로 교체 |
| ② | FeatRrcInactive.cpp | `FnVCCB_IsAbleToProceedRRCInactive()` | INACTIVE 천이 판단 |
| ③ | StateSaRrcInact.cpp | `FnVCCB_Inact_ProcRRCResumeRequest1()` | Resume 이력 갱신 |

---

## 6. Fallback 정책

```
AI 추론 실패 시 → 기존 정적 타이머 그대로 사용
연속 10회 실패  → Fallback 모드 전환
최대 연장 3회 초과 → 즉시 INACTIVE 천이 강제
IS_REDCAP_UE    → AI 미적용
IS_NRDC         → AI 미적용
```

---

## 7. 신규 파일 목록

| 파일 | 위치 | 역할 |
|---|---|---|
| `VccbAiEngine.h` | VCCB_MAIN/hdr/ | 구조체 정의, 함수 선언 |
| `VccbAiEngine.cpp` | VCCB_MAIN/src/ | GRU 순전파 구현 |
| `VccbAiWeights.h` | VCCB_MAIN/hdr/ | Python 자동 생성 가중치 |
| `VccbAiClient.h` | VCCB_MAIN/hdr/ | AI Client 함수 선언 (미완) |
| `VccbAiClient.cpp` | VCCB_MAIN/src/ | AI Client 구현 (미완) |

---

## 8. UE별 AI 컨텍스트 구조체

```cpp
struct VccbAiCallContext_s {
    // 이력 순환 버퍼
    struct VccbAiHistory_s history[10]; // [durationSec, resumeCause]
    int    writeIdx;
    int    count;

    // 통계
    float  avgDurationSec;
    float  resumeSuccessRate;  // 초기값 1.0
    int    resumeCount24h;

    // 현재 상태
    long   inactiveEntryTs;    // INACTIVE 진입 타임스탬프
    int    extendCount;        // 연속 타이머 연장 횟수
    float  lastInferTimerSec;  // 마지막 추론 결과

    // GRU 은닉 상태
    float  h[16];
};
```

---

## 9. 작업 진행 현황

| 순서 | 작업 내용 | 완료 |
|---|---|---|
| 1 | GRU 모델 설계 및 학습 (train_gru.py) | ✅ |
| 2 | 가중치 추출 → VccbAiWeights.h 자동 생성 | ✅ |
| 3 | C++ GRU 순전파 엔진 구현 (VccbAiEngine.cpp) | ✅ |
| 4 | UE별 AI 컨텍스트 구조체 정의 | ✅ |
| 5 | vccb.xcc 빌드 성공 확인 | ✅ |
| 6 | callInfo_type에 aiCtx 필드 추가 | ⬜ |
| 7 | VccbAiClient.h / VccbAiClient.cpp 구현 | ⬜ |
| 8 | MainInit.cpp에 AI 초기화 추가 | ⬜ |
| 9 | 포인트②③① 순서로 VCCB 코드 수정 | ⬜ |
| 10 | vccb.sim 빌드 및 gtest 검증 | ⬜ |
| 11 | 로그취득절차 (나중에 별도 진행) | ⬜ |

---

## 10. 빌드 환경

```
컴파일러:    /usr/bin/g++ (C++17)
OS:          RedHat 8.10 (Docker container: access_rhel8.10)
워크스페이스: /p4_ws/sangku/26b_ina2_vzw
VCCB 경로:   Application/NR_CALL_APP/VCCB
빌드 명령:   make -f Makefile.sim
외부 라이브러리: 없음 (시나리오 B — C++ 직접 포팅)
```

---

## 11. 로그취득절차 (나중에 별도 진행)

```
1단계: gtest 로그에서 RRC 상태 전환 패턴 추출
2단계: 5QI별 3GPP 규격 기반 합성 데이터 생성 스크립트 작성
3단계: 둘을 합쳐서 GRU 모델 재학습
       → VccbAiWeights.h 재생성
       → VCCB 재빌드 / 재배포
```

---

## 12. 주요 상수

| 상수 | 값 | 의미 |
|---|---|---|
| `VCCB_AI_HISTORY_LEN` | 10 | UE별 이력 버퍼 크기 |
| `VCCB_AI_MAX_EXTEND_COUNT` | 3 | 최대 타이머 연장 횟수 |
| `VCCB_AI_KEEP_THRESHOLD` | 0.6 | INACTIVE 유지 판단 임계값 |
| `VCCB_AI_RELEASE_THRESHOLD` | 0.3 | 조기 해제 판단 임계값 |
| `VCCB_AI_MARGIN_FACTOR` | 1.2 | 타이머 여유 배율 |
| `VCCB_AI_TIMER_MIN_SEC` | 1 | 최소 타이머 값 (초) |
| `VCCB_AI_TIMER_MAX_SEC` | 3600 | 최대 타이머 값 (초) |
| `VCCB_AI_FALLBACK_COUNT` | 10 | Fallback 전환 연속 실패 횟수 |

---

## 13. Python 학습 파일 위치

```
~/vccb_ai/train_gru.py        → GRU 모델 학습
~/vccb_ai/export_weights.py   → 가중치 추출 → VccbAiWeights.h 생성
~/vccb_ai/gru_model.pth       → 학습된 모델 가중치
~/vccb_ai/VccbAiWeights.h     → 생성된 C++ 헤더 (컨테이너로 복사됨)
```
