# VCCB AI 추론 기능 구현 현황

> **기준일**: 2026-05-21 (최근 갱신: 2026-07-05 — 코드 리뷰 후속 결함 수정 11건: resumeCause train/serve 인코딩 매핑, 연장 성공 시 extendCount 리셋 훅, 0초 이력 오염 차단, pendingExtendTimerSec, 통계 정합 등. §7 세션 로그 참조)
>
> **설계 기반**: `VCCB_AI_Integration_DetailedDesign.md`
>
> **구현 방식**: 시나리오 B (C++ 직접 포팅, GRU 기반, 외부 라이브러리 없음)

---

## 목차

1. [수정된 파일 목록](#1-수정된-파일-목록)
2. [완료된 항목](#2-완료된-항목)
3. [남은 항목](#3-남은-항목)
4. [Feature 벡터 매핑 현황](#4-feature-벡터-매핑-현황)
5. [AI 개입 포인트 현황](#5-ai-개입-포인트-현황)
6. [2026-07-04 세션 작업 로그](#6-2026-07-04-세션-작업-로그)

---

## 1. 수정된 파일 목록

| 파일 | 위치 | 주요 변경 내용 |
|---|---|---|
| `VccbAiClient.cpp` | `VCCB_MAIN/src/` | 핵심 AI 정책 로직 전면 재구현 |
| `VccbAiClient.h` | `VCCB_MAIN/hdr/` | `VCCB_AI_INFER_TIMEOUT_US` 상수 추가 |
| `FeatRrcInactive.cpp` | `VCCB_FEAT/RRC_INACTIVE/src/` | 포인트② BearerContextModification 전송 연동 |
| `MsgE1apMake.cpp` | `VCCB_MSG/E1AP/src/` | 포인트① AI 타이머 추천 함수 연결 |
| `MainInit.cpp` | `VCCB_MAIN/src/` | `VccbAiClient_Reset()` 초기화 호출 추가 |

---

## 2. 완료된 항목

### 2-1. Critical — 핵심 로직

| 항목 | 파일 | 내용 |
|---|---|---|
| **포인트② `IsAbleToProceedInactive` 재구현** | `VccbAiClient.cpp` | AI 추론(`GRU + MLP`) 실제 호출. `resume_prob ≥ 0.6` 시 타이머·타임스탬프 저장 후 `false` 반환(INACTIVE 보류). `extendCount ≥ 3` 시 카운터 리셋 후 강제 INACTIVE 천이(`true`). Fallback 시 `true`(기존 동작 유지). |
| **포인트② BearerContextModification 전송** | `FeatRrcInactive.cpp` | AI가 `false` 반환 시 `VEIL_sendE1SetupModRelRequest(E1Procedure_BearerContextModification)` fire-and-forget 호출. 오류 로그 포함. `#include "MsgE1apSend.h"` 추가. |
| **포인트① E1AP inactivityTimer IE 연결** | `MsgE1apMake.cpp` | `VEIL_fillE1_Inactivity_Timer()` 내 `FnVCCB_GetRRCInactiveTimerValue()` 직접 호출 → `VccbAiClient_RecommendInactivityTimer()` 래퍼로 교체. `#include "VccbAiClient.h"` 추가. |

### 2-2. 안정성 보강

| 항목 | 파일 | 내용 |
|---|---|---|
| **`IS_REDCAP_UE` / `IS_NRDC_MN,SN` skip 활성화** | `VccbAiClient.cpp` | `shouldSkipAi()` 내 주석 처리된 코드 활성화. 매크로 위치 `ComMacro.h` 확인 완료. |
| **글로벌 카운터 멀티스레드 안전 처리** | `VccbAiClient.cpp` | `int32_t` / `bool` → `std::atomic<int32_t>` / `std::atomic<bool>` 교체. `fetch_add()`, `store()` 사용. C++11 환경(`Makefile.body` 확인됨). |
| **`VccbAiClient_Reset()` 초기화 호출 추가** | `MainInit.cpp` | `FnVCCB_InitBlock()` 내 `VccbAiEngine_Init()` 직후 `VccbAiClient_Reset()` 호출. `#include "VccbAiClient.h"` 추가. |
| **추론 타임아웃(1ms) 측정 및 Fallback 연동** | `VccbAiClient.cpp` | `clock_gettime(CLOCK_MONOTONIC)` 기반 `runInferWithTimeout()` 헬퍼 구현. `VCCB_AI_INFER_TIMEOUT_US(1000us)` 초과 시 `noteInferFailure()` 처리. |

### 2-3. Feature 벡터 매핑 (확인된 필드)

| stat[] | Feature 이름 | 접근 경로 |
|---|---|---|
| `stat[0]` | primary_5qi | `drb[i].drbQosIdxInCall` → `qosFlow[idx].qosParam.nonDyna.fiveQi` |
| `stat[1]` | drb_count | `drb[i].usage == CI_use` 순회 카운트 |
| `stat[2]` | ue_ambr_dl_mbps | `ueInfo.ambr.aggrMbrDl / 1,000,000` (bps → Mbps) |
| `stat[3]` | has_gbr_flow | `qosFlow[i].qosParam.gbrQosInformationPresent != 0` |
| `stat[6]` | last_rsrp | `servCellSsbRsrp` (값 `255` = invalid → 학습 분포 평균 `75.0f`로 mean-fill, 2026-07-05) |
| `stat[8]` | rna_cell_count | `ueInfo.RrcInactiveInfo.numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | `aiCtx.avgDurationSec` (EMA) |
| `stat[10]` | resume_count_24h | `aiCtx.resumeCount24h` |
| `stat[11]` | resume_success_rate | `aiCtx.resumeSuccessRate` (EMA) |
| `stat[12]` | avg_inactive_duration | `aiCtx.avgDurationSec` |
| `stat[13]` | hour_of_day | `localtime(time(NULL))->tm_hour` |
| `stat[14]` | day_of_week | `localtime(time(NULL))->tm_wday` |

---

## 3. 남은 항목

> **2026-07-04 정리**: 이전에 "미확인"으로 분류됐던 `stat[4]`(handover_count_24h)/`stat[7]`(secondary_5qi)는 실제 signaling-plane 필드로 완전히 구현·검증 완료(§4 표 참조). `stat[5]`(cn_stationary)만 NGAP `ExpectedUEBehaviour` 미배선으로 예약 상태(`0.0f` 고정) — 이번 범위 밖.

### 3-1. 🟡 중간 — 기능 보완

| 항목 | 파일 | 내용 |
|---|---|---|
| **포인트④ RNA Update 시 AI 권고 로그** | `FeatRrcInactive.cpp` | `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` 내 `resumeCause == NRRRC_rna_Update && dlDataFlag == VCCB_FLAG_OFF` 분기에서 AI RNA 크기 권고값 조회 및 로그 출력. |
| **포인트⑤ XnAP HO 시 AI Context seed** | `StateHandleSa.cpp` 등 | XnAP HandoverRequest 수신 후 `VLIB_InitCallInfo()` 직후, UE History(`Time UE Stayed in Cell`)를 `aiCtx.history` 버퍼에 seed. 코드베이스에서 `XN_HandoverRequest` 또는 `RetrieveUEContextRequest` 검색하여 핸들러 위치 확인 필요. |
| **24h 슬라이딩 윈도우 Resume 카운트** | `VccbAiClient.cpp` | `resumeCount24h` 현재 단순 increment. 24h 이내 이벤트만 카운트하는 슬라이딩 윈도우 구현 필요. 타임스탬프 링 버퍼(`aiCtx` 내 추가) 또는 이벤트 시각 배열 활용. |
| **Resume 성공률 실패 판단 로직** | `VccbAiClient.cpp` | `resumeSuccessRate` 현재 모든 Resume을 성공으로 처리(`isSuccess = 1.0f`). `resumeCause` 값에 따라 실패 여부 구분 필요. `NRRRC_emergency`, `NRRRC_rna_Update` 등 설계서 섹션 1-3 매핑 테이블 참조. |

### 3-2. 🟢 낮음 — 디버그 및 운영

| 항목 | 파일 | 내용 |
|---|---|---|
| **추론 통계 출력 (`itscp_vccb`)** | 기존 itscp 핸들러 | 설계서 섹션 2-6 항목. AI Client 상태(Active/Fallback/Init), 총 요청·성공·실패·Fallback 수, 트리거별 요청 수(포인트①②③④), 평균 추론 소요시간(us). 전역 통계 구조체 추가 후 itscp 출력 함수에 연결. |

---

## 4. Feature 벡터 매핑 현황

| stat[] | Feature 이름 | 상태 | 비고 |
|---|---|---|---|
| `stat[0]` | primary_5qi | ✅ 완료 | Non-Dynamic 5QI 직접 접근 |
| `stat[1]` | drb_count | ✅ 완료 | |
| `stat[2]` | ue_ambr_dl_mbps | ✅ 완료 | bps → Mbps 변환 |
| `stat[3]` | has_gbr_flow | ✅ 완료 | |
| `stat[4]` | handover_count_24h | ✅ 완료 | `ctx->handoverCount24h` (OnHandover@HO_Succ, 24h 링, persist) |
| `stat[5]` | cn_stationary | ⏸ 예약 | ExpectedUEBehaviour 미배선, `0.0f` |
| `stat[6]` | last_rsrp | ✅ 완료 | `servCellSsbRsrp`, 255=invalid 처리 |
| `stat[7]` | secondary_5qi | ✅ 완료 | primary와 다른 첫 DRB 5QI(혼합 서비스군) |
| `stat[8]` | rna_cell_count | ✅ 완료 | `numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | ✅ 완료 | `avgDurationSec` 재사용 |
| `stat[10]` | resume_count_24h | ⚠️ 부분 | 단순 increment, 슬라이딩 윈도우 미구현 |
| `stat[11]` | resume_success_rate | ⚠️ 부분 | EMA 구조 있으나 항상 `1.0f` |
| `stat[12]` | avg_inactive_duration | ✅ 완료 | `avgDurationSec` |
| `stat[13]` | hour_of_day | ✅ 완료 | `tm_hour` |
| `stat[14]` | day_of_week | ✅ 완료 | `tm_wday` |

> **완료**: 12/15 (80%) | **부분**: 2/15 | **미완료**: 3/15 (20%)

---

## 5. AI 개입 포인트 현황

| 포인트 | 위치 | 설계 의도 | 구현 상태 |
|---|---|---|---|
| **포인트①** | `MsgE1apMake.cpp` → `VEIL_fillE1_Inactivity_Timer()` | E1AP BearerContextSetup/Modify 시 AI 추천 타이머 값 주입 | ✅ **완료** — `VccbAiClient_RecommendInactivityTimer()` 연결 |
| **포인트②** | `FeatRrcInactive.cpp` → `FnVCCB_IsAbleToProceedRRCInactive()` | InactivityNotification 수신 시 즉시 천이 or 타이머 연장 판단 | ✅ **완료** — AI 추론 호출 + BearerContextModification 전송 |
| **포인트③** | `StateSaRrcInact.cpp` → `FnVCCB_Inact_ProcRRCResumeRequest1()` | RRCResumeRequest 수신 시 INACTIVE 지속 시간·원인 이력 기록 | ✅ **완료** (기존 구현) — `VccbAiClient_OnResumeRequest()` 연결됨 |
| **포인트④** | `FeatRrcInactive.cpp` → `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` | RNA Update 수신 시 AI RNA 크기 권고값 로그 출력 | ❌ **미구현** |
| **포인트⑤** | `StateHandleSa.cpp` 등 (XnAP HO 핸들러) | XnAP HO 수신 시 소스 셀 체류 시간을 aiCtx 이력에 seed | ❌ **미구현** |

---

## 부록 — 주요 상수 현황 (`VccbAiClient.h`)

| 상수 | 값 | 의미 |
|---|---|---|
| `VCCB_AI_KEEP_THRESHOLD` | `0.6f` | INACTIVE 유지 판단 확률 임계값 |
| `VCCB_AI_RELEASE_THRESHOLD` | `0.3f` | 조기 해제 판단 확률 임계값 |
| `VCCB_AI_MARGIN_FACTOR` | `1.2f` | 타이머 여유 배율 |
| `VCCB_AI_TIMER_MIN_SEC` | `1` | 최소 타이머 값 (초) |
| `VCCB_AI_TIMER_MAX_SEC` | `3600` | 최대 타이머 값 (초) |
| `VCCB_AI_MAX_EXTEND_COUNT` | `3` | 최대 타이머 연장 횟수 |
| `VCCB_AI_FALLBACK_COUNT` | `10` | Fallback 전환 연속 실패 횟수 |
| `VCCB_AI_INFER_TIMEOUT_US` | `1000` | 추론 타임아웃 (μs) |

---

## 6. 2026-07-04 세션 작업 로그

### 6-1. 코드 변경 (완료)

| 항목 | 파일 | 내용 |
|---|---|---|
| **`handoverCount24h` 필드 추가** | `VccbAiTypes.h` | `VccbAiCallContext_type`에 `handoverCount24h`, `handoverTs[10]`, `handoverTsWriteIdx` 3개 필드 추가 |
| **`FnVCCBAiClient_OnHandover()` 신규 구현** | `VccbAiClient.cpp/.h` | HO 타임스탬프 push → `calcCount24hFromTs()`(기존 `calcResumeCount24h`를 공유 헬퍼로 리팩토링)로 24h 카운트 갱신 → `FnVCCBAiPersist_Save()`. master-enable 무관 항시 관찰(observation-only) |
| **HO 훅 삽입 (2곳)** | `StateSaHo.cpp`(~4795행), `StateCommon.cpp`(~3833행, `#include` 추가) | 기존 `VHCM_SA_HO_STAT(ptCi, HO_Succ)` 직후 `FnVCCBAiClient_OnHandover(ptCi)` 호출 — HO 성공 peg와 동일 지점이라 exactly-once 보장 |
| **영속 저장소 확장** | `VccbAiPersistStore.cpp` | `VccbAiPersistEntry_type`에 HO 관련 3필드 추가, `Restore()`/`Save()`에 memcpy 반영 |
| **`stat[7]` secondary_5qi 구현** | `VccbAiClient.cpp` | primary와 다른 첫 DRB의 5QI를 primary_5qi 추출 로직과 동일한 패턴으로 미러링 |
| **ML 파이프라인 4곳 LOCKSTEP 반영** | `preprocess.py`, `synth_data_gen.py`, `gen_realish_csv.py`, `analyze_features.py`, `test_SaRrcInactiveAI.cpp` | stat[4]/[7] 실값 채움 + 라벨 공식에 `svc_factor`/`mob_factor` 곱셈 결합 (아래 6-2) |
| **seedAiContext JSON 하니스 동기화** | `test_SaRrcInactiveAI.json`, `VccbBtCommonTestJson.cpp` (파서) | 미사용 `rnaCellCount` 제거, 신규 `handoverCount24h` 파라미터 추가. profile A/B의 DRB-level QoS flow(5QI 2/3/6)와 dwell/AMBR/resumeCause 값 재정합 |
| **AI 매트릭스 테스트 스위트 신규 작성** | `Test/ut/tool/gen_ai_matrix_tc.py`(신규), `test_SaRrcInactiveAI.json`(병합), `VccbBtCommonTestChecker.cpp`(`checkUe_Proc_AiInfo` 체커 추가) | 5개 그룹(A~E) 16케이스로 primary_5qi/secondary_5qi/handover_count/행동이력/rsrp 각 feature를 단독 격리해 timer 변화를 관찰. `skip:1`(재학습 후 활성화 예정) |

상세: `VCCB_AI_Integration_DetailedDesign.md` §D-4-보충, §6-10 참조.

### 6-2. 라벨 결합 공식 및 문헌 검증 (완료)

```
label_dwell = base(primary_5qi) × rsrp_factor(rsrp) × hist_factor(succ) × svc_factor(secondary_5qi) × mob_factor(ho_count_24h)
  rsrp_factor = 1 + (75 − rsrp) / 150         # 3GPP TS 38.331/38.304
  hist_factor = 1 − 0.3 × (succ − 0.5)        # 3GPP TS 38.133
  svc_factor  = 0.7 (secondary∈{1,2}) else 1  # 3GPP TS 23.501 + Nokia whitepaper
  mob_factor  = 1 − 0.05 × min(ho_count, 6)   # US Patent 11,109,442
```

3개 에이전트를 동원해 4개 계수 전체(개념·범위·계수값)를 3GPP 표준/Nokia·Ericsson 백서/특허 대비 1차 검증 — **결론: 공식 구조는 문헌적으로 정당화됨, 정확한 계수값(0.7·0.05 등)은 실측 calibration 대상**으로 확정. **LOCKSTEP 3곳** (`preprocess.py`, `synth_data_gen.py`, `test_SaRrcInactiveAI.cpp genEvalSet`) 동시 수정 필수 — 한 곳만 바꾸면 학습/평가 라벨 분포가 어긋남.

> **⚠️ 후속 정정 (같은 날, 사용자 제공 URL 10개 원문 대조 후)**: 1차 검증은 원문 링크 없는 agent 조사였다. 이후 실제 URL 10개(ScienceDirect/IEEE/USPTO/R&S 등)를 직접 fetch한 결과, `svc_factor`는 근거가 강화(신뢰도 85%→88%, R&S VoNR 백서로 5QI1 PDB100ms 원문 재확인)됐지만 **`mob_factor`가 인용했던 US 특허 11,109,442는 실제로 읽어보니 반대 방향 로직**이었다(신뢰도 80%→55%로 하향) — 특허는 "고속이동+실시간앱→timer 증가"인데 우리 공식은 "고이동성→timer 감소"로 별개 메커니즘. 개념적 근거로만 유효하고 계수 방향의 직접 근거로는 부적절함을 확인. 상세 대조표는 `VCCB_AI_Integration_DetailedDesign.md` 부록 E 참조. 가장 강력한 신규 근거는 ScienceDirect 2025 논문(*Predictive modeling of RRC inactive transitions...*) — 실제 상용 NR SA 데이터로 RRC Inactive→Resume/Fallback을 ML로 예측하는, 본 프로젝트와 문제 설정이 거의 동일한 2025년 선행연구.

### 6-3. 재학습 진행 상태 (진행 중 → 검증 대기)

전처리(Method A+B 병합) 완료, 학습(`python train.py`, 환경변수 없는 **Phase 0 full training**) 진행 중:

```
[Method B] real CSV : 92,977 unique samples from 8 file(s)
  (vccb_ai_dataset.csv 79,477 + eval_fixed.csv 3,000 + round_1~5.csv 각 1,500 + shift.csv 3,000)
[Method A] synthetic : 20,000 samples (3GPP 5QI 규칙)
Total : 112,977  (real 82.3% / synthetic 17.7%)
  resume_prob=1 (resume<60s): 52.0%   (40–60% 목표 범위 내 ✅)
  resume_time: mean=448.6s std=791.9s
```

**다음 단계 (재학습 완료 후)**:
1. `eval_report.txt`에서 Phase-1 floor(AUROC/F1/MAE, §6-9 기준) 통과 확인
2. 통과 시 → VccbAiWeights.h 반영, VCCB→itscp_vccb→UT 순서 재빌드, 매트릭스 테스트(`skip:1→0`) 활성화·tighten
3. 실패 시 → `svc_factor`/`mob_factor` 계수를 3곳 LOCKSTEP으로 동시 조정 후 **Phase 0부터 재실행** (fine-tune 아님 — 계수 변경은 라벨 자체가 바뀌므로 warm-start 부적절)
4. Phase 0 통과가 확정된 이후에만, 향후 실측 데이터가 추가로 쌓이면 `VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py`(Phase 2, 선택)로 점진 미세조정

> 매트릭스 16케이스는 재학습 acceptance gate가 **아니다** — 4개 base case가 유일한 학습/검증 입력이며, 매트릭스는 모델 완성 후 feature 민감도를 시연/회귀 확인하는 참조 세트다.

상세 절차: `VCCB_AI_Verification_Runbook.md` Part G 참조.

---

## 7. 2026-07-05 세션 작업 로그 — 코드 리뷰 후속 결함 수정

전체 코드 리뷰에서 발견된 결함 11건을 일괄 수정했다. **⚠️ 이 변경으로 라벨/이력 인코딩·수집 CSV 의미가 바뀌므로, 진행 중이던 재학습은 기존 실측 CSV의 cause 재검증(아래 7-4) 후 Phase 0부터 재실행해야 한다.**

### 7-1. 🔴 모델 품질 직접 영향 (4건)

| # | 결함 | 수정 |
|---|---|---|
| 1 | **resumeCause train/serve 인코딩 불일치** — 서빙은 NRRRC ASN.1 원시값(mo_Data=4, mt_Access=2, rna=8...)을 history/CSV에 저장, 학습은 0..5 인코딩(mo_Data=1, mt_Access=3...) → seq[][1] 의미 왜곡 | `FnVCCBAiClient_MapResumeCause()` 신설(`VccbAiClient.cpp`, 공개 API). `OnResumeRequest`의 history push·data-collect emit·성공률 판정 모두 매핑값 사용. 공유 상수 `VCCB_AI_CAUSE_*`를 `VccbAiTypes.h`에 정의(ML CAUSE_*와 lockstep) |
| 2 | **연장 성공 시 extendCount 미리셋** — Point2 defer 후 UE가 CONNECTED 유지 중 트래픽 재개(연장의 성공 케이스)하면 RRCResume이 없어 extendCount가 잔존 → 다음 idle 구간의 연장 예산 잠식 | `FnVCCBAiClient_OnDataActivityResumed()` 신설. `StateE1ap.cpp::FnVCCB_Proc_E1InactivityNotification_ByRRCInactive()`의 `ActivityStatus_ACTIVE`(DATA_RESTART) 분기에서 호출 — extendCount/pending 리셋 + 성공 로그(관측 전용) |
| 3 | **0초 이력 오염** — `inactiveEntryTs==0`(inter-gNB resume 타깃 등)인데 durationSec=0.0을 history/EMA에 push → "즉시 resume" 허위 신호가 persist로 호 경계 넘어 지속 | `OnResumeRequest`가 `inactiveEntryTs==0`이면 history/EMA/CSV/persist 전부 skip, 런타임 카운터(extendCount)만 리셋 (설계서 §1-3대로) |
| 4 | **`localtime()` 스레드 비안전** — Call Task 멀티스레드에서 정적 버퍼 레이스 | `buildAiInput()` → `localtime_r()` 교체 |

### 7-2. 🟡 정책/통계 정합 (5건)

| # | 결함 | 수정 |
|---|---|---|
| 5 | Fallback/skip 경로에서 default 타이머를 [1,3600]으로 clamp — AI off인데 운영 설정을 변조 | Point1 skip/실패/mid-range(DEFAULT) 경로 모두 `defaultTimerSec` **무가공 반환**. clamp는 AI 산출값(KEEP/EARLY_RELEASE)에만 적용 |
| 6 | 연장 결정과 실제 전송 타이머가 별도 추론 — defer 직후 BearerContextMod가 Point1을 재호출해 재추론(자기모순 가능) | `aiCtx.pendingExtendTimerSec` 신설(`VccbAiTypes.h`, 런타임 전용·비영속). Point2 KEEP이 set → Point1이 창구간 동안 재추론 없이 그 값 반환. proceed/cap/OnInactiveEntry/OnResume/OnDataActivityResumed에서 clear (설계서 §1-2 우선순위 1 구현) |
| 7 | 타임아웃 통계 이중 계상(fallback 2회, timeout이 failure로도) | `AiInferResult` enum(OK/FAIL/TIMEOUT) + `noteInferFailure(bool isTimeout)` — 시도당 failure **또는** timeout 1회 + fallback 1회만 peg |
| 8 | Pt4 `totalRequests` 계상 시점이 Pt1/2와 상이(분모 의미 불일치) | `OnRnaUpdate` 진입 즉시 증가로 통일, skip 시 fallback peg도 동일 패턴 |
| 9 | XnAP seed의 `resumeCause=-1` OOD 입력 | `VCCB_AI_CAUSE_OTHERS`(5)로 대체 — 학습된 catch-all 코드로 in-distribution 유지 |

### 7-3. 🟢 낮음 (기타)

- **persist 복원 재시도**: 첫 AI 접점에 5G-S-TMSI 미할당(key==0)이면 `persistRestored`를 세우지 않아 이후 추론에서 복원 재시도 (`maybeRestorePersisted`)
- **invalid RSRP(255) → 75.0 mean-fill** (0.0은 "극약전계"와 혼동, 드리프트 bin0 오염)
- **DISABLED ≠ FALLBACK 통계 분리**: master-disable 상태의 skip은 fallbackCount 미계상(`noteSkipFallback`) → 대시보드 FallbackRate가 실제 장애만 반영
- `getCmd.cpp` "1282 params" → **1266** 정정, `train.py` docstring `Linear(15,16)` → `Linear(14,16)` 정정
- `VccbAiTypes.h` `h[]` 주석 정정: 현 엔진은 stateless(매 추론 h=0에서 재계산) — 예약 필드임을 명시. `VccbAiEngine.h`에 C++-only(참조 파라미터) 명시
- **synth_data_gen.py**: rna_Update 이벤트는 타임라인만 진행하고 행/이력 미배출 — 서빙 Point3의 rna skip과 분포 정합 (§7-4 참조)

### 7-4. ⚠️ 재학습 파급 (필독)

1. **기존 on-device 수집 CSV(실측·sweep)는 cause 컬럼이 원시 NRRRC 인코딩**이라 신규 수집분(매핑된 0..5)과 혼합 불가 — 재매핑(mo_Data 4→1, mt_Access 2→3, mo_Signalling 3→2, rna 8→4, VoiceCall/SMS 등→5) 또는 폐기 후 재수집.
2. `synth_data_gen.py` 재실행 시 rna_Update 행이 사라지므로 데이터 구성비가 변한다 (fig05 등 보고 자료 재생성 필요).
3. UT 변경: `Point3_*` 테스트가 `FnVCCBAiClient_OnInactiveEntry()` 선행 stamp 필수화, 신규 `Point3_resume_without_entry_stamp_skips_history` 추가, sweep `causeSet`이 NRRRC enum 값으로 변경.
4. **빌드**: `VccbAiTypes.h`(aiCtx 필드 추가)가 `VccbStruct.h`에 포함되므로 **VCCB → itscp_vccb → UT 동시 재빌드** 필수 (§5-2 순서).
5. Windows 참조 사본 기준 수정이므로, Linux p4 워크스페이스의 `Base/VccbItscpDefine/include/VccbAiTypes.h`에 동일 반영 확인.

---

## 8. 2026-07-05 세션 작업 로그(2) — ML 파이프라인 lockstep 복원 + 시각화·fine-tuning 체계

### 8-1. synth_data_gen.py 라벨 lockstep 복원 (§D-4 위반 수정)

`analyze_synth_literature.py`(신규 진단 스크립트)로 생성 데이터를 §D-4 문헌 근거 4계수와 대조한 결과, 이 파일만 `rsrp_factor`/`hist_factor`를 dwell에 곱하지 않는 lockstep 위반이 확인되어 수정했다.

- **수정**: 이벤트별 `dur = sample_iat(persona) × rsrp_factor × hist_factor × svc_factor × mob_factor` (rsrp는 세션별 샘플을 stat[6]과 라벨에 동일 적용, hist는 진화하는 EMA 사용 — preprocess.py/genEvalSet과 동일 공식).
- **검증**: 수정 후 4계수 모두 REPRODUCED — svc 층화비율 0.684(목표 0.70), rsrp/hist/mob 페르소나 내부 Spearman 전부 음수. 전체 모집단 상관은 페르소나 교란(고정형=신호좋음+긴dwell) 때문에 여전히 양수이며, 판정은 반드시 페르소나 통제 후 수행(`synth_literature_check_stratified.png`).
- **fine-tuning 데이터**: `--profile field`(분포 이동된 실계망 모사 페르소나) + `--outlier-frac`(수집 아티팩트 주입: label spike/zero, AMBR ×1000, stale EMA) 옵션 신설.

### 8-2. preprocess.py outlier 처리 (Method B 실측 행 전용)

- `VCCB_AI_OUTLIER_MODE=iqr(기본)|pctl|off`, `VCCB_AI_OUTLIER_IQR_K=3.0`.
- 도메인 룰(비유한/`<1s` stamp race/`>3600s` clock glitch/RSRP 범위/EMA>24h/seq>24h) + 통계 룰(log-IQR 펜스: label, ambr, avg_dur 상단).
- before/after 분포를 `outlier_report.png`로 자동 출력. 검증: 2% 주입 fine-tune CSV에서 785행(1.81%) 정확히 제거(4개 룰 모두 발화).

### 8-3. train.py v1.7.0 — 학습 리포트 + fine-tune before/after

- epoch별 loss history 수집 → `train_report_{full|finetune}.png`(학습곡선/예측vs실제/분류 분리도/MAE vs baseline) + `train_metrics_{mode}.json`(메트릭·history 기계가독 덤프).
- **fine-tune 모드**: warm-start 직후 동일 val split로 사전 평가(`pre-FT`)를 수행해 리포트 패널(4)에 before/after MAE·AUROC 비교 표기.
- 실측 사이클 검증(Windows/CUDA): base 학습 AUROC 0.970·MAE 10.63s(naive 12.35/persist 22.97 상회) → field 셋 fine-tune 후 MAE 11.08→9.63s, AUROC 0.973→0.976.

### 8-4. visualize_ut_results.py — gtest 결과 시각화 (신규)

- 사용: `./vccb_ut --gtest_filter='*Ai*:*AI*' --gtest_output=json:ut_results.json` → `python3 visualize_ut_results.py ut_results.json [--baseline prev.json]`.
- 출력: `ut_results.png`(suite별 pass/fail, 느린 테스트 top15, Phase-1 품질 vs floor, 종합 pass rate/실패 목록) + `_summary.txt`. `--baseline` 지정 시 newly failing/fixed 및 AUROC/F1/MAE 추이 비교. Phase-1 수치는 C++ 테스트의 기존 `RecordProperty`(auroc/f1/mae ×1000)를 파싱하며 XML/JSON 모두 지원.

### 8-4-보충. 터미널 ASCII 시각화 (`termplot.py`, 신규, 2026-07-05 추가 요청)

PNG 파일은 그대로 두고, **같은 데이터를 스크립트 실행 shell(bash/PowerShell/redhat 컨테이너 tty)에 즉시 텍스트 그래프로도 출력**하도록 4개 스크립트에 연결했다. 외부 패키지 추가 없이(numpy만 사용) Unicode block 문자(█▏▎▍▌▋▊▉, 스파크라인 ▁▂▃▄▅▆▇)로 막대그래프·히스토그램·loss 스파크라인을 그린다. stdout 인코딩이 유니코드를 못 받으면 자동으로 ASCII(`#`/`-`)로 폴백한다.

- `preprocess.py`: outlier 필터 전/후 라벨 duration 히스토그램(log-bin) + 룰별 제거건수 막대그래프 + 최종 `dataset.npz` 라벨 분포.
- `train.py`: val/train loss 스파크라인(최적 epoch 표시) + MAE vs baseline 막대(fine-tune 시 before/after 포함) + AUROC/F1 막대(fine-tune 시 before/after) + resume_prob 양성/음성 분리 이중 히스토그램.
- `analyze_synth_literature.py`: 5QI 분포, svc_factor naive vs 층화 vs 이론값 막대, rsrp/hist/mob 페르소나별 상관관계 막대(0 기준선 표시로 REPRODUCED 여부 시각 확인), 전체모집단 상관(교란 비교용) 막대.
- `visualize_ut_results.py`: suite별 pass/fail 막대, 느린 테스트 top12 막대, Phase-1 AUROC/F1/MAE vs floor 막대(baseline 비교 포함), 종합 pass rate.

이 PC에서 4개 스크립트 모두 재실행해 실제 bash 화면에 렌더링되는 것을 확인했다(Windows Git Bash, PYTHONIOENCODING=utf-8).

### 8-5. 권장 운영 절차 (fine-tuning 사이클)

```bash
# 1) base 데이터/학습
python3 synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0
python3 preprocess.py && python3 train.py            # -> train_report_full.png
cp model.pt model_base.pt
# 2) field(실측 또는 모사) 데이터 + outlier 필터
python3 synth_data_gen.py --out vccb_ai_finetune.csv --ues 3000 --seed 7 --profile field --outlier-frac 0.02
VCCB_AI_DATA_FILE=vccb_ai_finetune.csv VCCB_AI_SYNTH_N=0 python3 preprocess.py   # -> outlier_report.png
# 3) fine-tune (before/after 리포트 자동 생성)
VCCB_AI_FINETUNE=1 python3 train.py                  # -> train_report_finetune.png
# 4) 배포 전 UT + 시각화
python3 export_weights.py && (UT 재빌드/실행)
python3 visualize_ut_results.py ut_results.json --baseline prev.json
```
