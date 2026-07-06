# VCCB AI 추론 기능 구현 현황

> **기준일**: 2026-05-21 (최근 갱신: 2026-07-06 — §3 "남은 항목"에 있던 5개 항목(포인트④/⑤, resume_count_24h 슬라이딩 윈도우, resumeSuccessRate 실패 분기, itscp `get stat ai`) 전부 구현 완료 코드 확인, UT 43/43 전체 통과(§9-3). `cn_stationary`(stat[5])는 이전에 "AMF 미제공으로 구조적 구현 불가"로 기록됐으나, 같은 날 후속 코드 검증에서 **이 결론이 근거 부족으로 판명**됨 — NGAP 코덱은 이미 `ExpectedUEBehaviour`를 unpack하는데 VCCB 애플리케이션이 그 값을 읽지 않고 버리는 것이 실제 원인. 중요도 평가 결과 한계효용이 낮을 것으로 판단되어 **`0.0f` 유지로 결정**하되 "영구 미배선"이 아니라 "낮은 우선순위로 보류, 재평가 가능"으로 재분류(§3-3/§9-2). §9 세션 로그 참조)
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
9. [2026-07-06 세션 작업 로그 — 잔존 항목 완료 검증](#9-2026-07-06-세션-작업-로그--잔존-항목-완료-검증)

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

> **2026-07-06 최종 정리**: 이 섹션에 있던 5개 항목(포인트④/⑤, 24h 슬라이딩 윈도우, Resume 성공률 실패 분기, itscp 추론 통계 출력)은 모두 구현 완료로 확인됨 — 상세는 §9 참조. 아래 표는 과거 기록이며, 실제 구현 위치는 §4/§5의 ✅ 표시와 §9를 따른다.

### 3-1. ✅ 완료됨 (과거 🟡 중간 — 기능 보완)

| 항목 | 파일 | 내용 | 완료 근거 |
|---|---|---|---|
| **포인트④ RNA Update 시 AI 권고 로그** | `FeatRrcInactive.cpp` | `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` 내 `resumeCause == NRRRC_rna_Update && dlDataFlag == VCCB_FLAG_OFF` 분기에서 AI RNA 크기 권고값 조회 및 로그 출력. | `FnVCCBAiClient_OnRnaUpdate(ptCi)` 호출 확인 (§9) |
| **포인트⑤ XnAP HO 시 AI Context seed** | `StateSaHo.cpp` | XnAP HandoverRequest 수신 후 UE History(`Time UE Stayed in Cell`)를 `aiCtx` 이력에 seed. | `FnVCCBAiClient_SeedFromXnapHistory(ptCi)` — store 성공 게이트 뒤로 순서 수정 완료 (§9) |
| **24h 슬라이딩 윈도우 Resume 카운트** | `VccbAiClient.cpp` | `resumeCount24h`를 24h 이내 이벤트만 카운트하는 슬라이딩 윈도우로 구현. | `calcResumeCount24h(ctx)` 타임스탬프 링 버퍼 구현 확인 (§9) |
| **Resume 성공률 실패 판단 로직** | `VccbAiClient.cpp` | `resumeCause` 값에 따라 성공/실패 구분(`NRRRC_emergency`→실패). | emergency=0.0/그 외=1.0 EMA 판정 구현 확인 (§9) |

### 3-2. ✅ 완료됨 (과거 🟢 낮음 — 디버그 및 운영)

| 항목 | 파일 | 내용 | 완료 근거 |
|---|---|---|---|
| **추론 통계 출력 (`itscp_vccb`)** | `getCmd.cpp`/`main.cpp` | AI Client 상태(Active/Fallback/Init), 총 요청·성공·실패·Fallback 수, 트리거별 요청 수, 평균 추론 소요시간(us). | `GetStatAiReport()` 구현 + `ITSCP_SET_FUNC` 등록 확인 (§9) |

### 3-3. ⏸ 낮은 우선순위로 보류 (2026-07-06 재평가 — 이전 "구현 불가" 결론 정정)

| 항목 | 내용 |
|---|---|
| **`stat[5]` cn_stationary** | 이전 기록("AMF가 IE 자체를 제공하지 않아 구조적으로 구현 불가")은 **코드 검증 결과 근거 부족으로 판명**됐다. NGAP 코덱(`Middleware/NR_CALL_LIB/IF_NGAP`)은 `CoreNetworkAssistanceInformation` IE 안의 `ExpectedUEBehaviour`(`expectedUEMobility` 포함, stationary/mobile)를 이미 unpack하도록 구현되어 있다(`ngapUnpackIE.c:143031 fnNGAP_saveIENGAP_ExpectedUEBehaviour_type`). 실제 원인은 VCCB 애플리케이션 계층의 `VNIL_StoreNGAP_CoreNetworkAssistanceInformation()`(`VCCB_MSG/NGAP/src/MsgNgapStore.cpp:4690`)이 이 필드를 아예 읽지 않고 버리는 것이었다(저장할 struct 필드도 `VccbUeRrcInactiveInfo_s`에 없음). 다만 **중요도 평가 결과**(라벨 공식이 이 값에 의존하지 않고, `primary_5qi`/`has_gbr_flow`/`ambr` 등 STRONG feature가 persona의 "고정형 여부"를 이미 상당 부분 대변하며, `stat[4]` handover_count_24h와도 개념적으로 겹침) 한계효용이 낮을 것으로 판단되어, **지금 당장 struct 필드·저장 함수 전체 구현은 보류하고 `0.0f` 유지로 결정**(2026-07-06, 사용자 확인). "영구 미배선/재검토 대상 아님"이 아니라 **"검증 결과 낮은 우선순위로 보류, 필요 시 재평가 가능"**으로 재분류. 저비용 관측(수신 카운터 1줄)조차 아직 착수하지 않은 상태. |

---

## 4. Feature 벡터 매핑 현황

| stat[] | Feature 이름 | 상태 | 비고 |
|---|---|---|---|
| `stat[0]` | primary_5qi | ✅ 완료 | Non-Dynamic 5QI 직접 접근 |
| `stat[1]` | drb_count | ✅ 완료 | |
| `stat[2]` | ue_ambr_dl_mbps | ✅ 완료 | bps → Mbps 변환 |
| `stat[3]` | has_gbr_flow | ✅ 완료 | |
| `stat[4]` | handover_count_24h | ✅ 완료 | `ctx->handoverCount24h` (OnHandover@HO_Succ, 24h 링, persist) |
| `stat[5]` | cn_stationary | ⏸ 낮은 우선순위(보류) | NGAP 코덱은 `ExpectedUEBehaviour` unpack을 이미 지원하나 VCCB가 안 읽음(수신 경로 미배선) + 중요도 평가상 한계효용 낮음 → 당분간 `0.0f` 유지, 재평가 가능 |
| `stat[6]` | last_rsrp | ✅ 완료 | `servCellSsbRsrp`, 255=invalid 처리 |
| `stat[7]` | secondary_5qi | ✅ 완료 | primary와 다른 첫 DRB 5QI(혼합 서비스군) |
| `stat[8]` | rna_cell_count | ✅ 완료 | `numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | ✅ 완료 | `avgDurationSec` 재사용 |
| `stat[10]` | resume_count_24h | ✅ 완료 | `calcResumeCount24h(ctx)` 24h 슬라이딩 윈도우(타임스탬프 링) |
| `stat[11]` | resume_success_rate | ✅ 완료 | `NRRRC_emergency`=실패(0.0)/그 외=성공(1.0) EMA(α=0.1) 판정 |
| `stat[12]` | avg_inactive_duration | ✅ 완료 | `avgDurationSec` |
| `stat[13]` | hour_of_day | ✅ 완료 | `tm_hour` |
| `stat[14]` | day_of_week | ✅ 완료 | `tm_wday` |

> **완료**: 14/15 (93%) | **낮은 우선순위로 보류(재평가 가능)**: 1/15 (`stat[5]` cn_stationary, 2026-07-06 "구현 불가" 결론 정정)

---

## 5. AI 개입 포인트 현황

| 포인트 | 위치 | 설계 의도 | 구현 상태 |
|---|---|---|---|
| **포인트①** | `MsgE1apMake.cpp` → `VEIL_fillE1_Inactivity_Timer()` | E1AP BearerContextSetup/Modify 시 AI 추천 타이머 값 주입 | ✅ **완료** — `VccbAiClient_RecommendInactivityTimer()` 연결 |
| **포인트②** | `FeatRrcInactive.cpp` → `FnVCCB_IsAbleToProceedRRCInactive()` | InactivityNotification 수신 시 즉시 천이 or 타이머 연장 판단 | ✅ **완료** — AI 추론 호출 + BearerContextModification 전송 |
| **포인트③** | `StateSaRrcInact.cpp` → `FnVCCB_Inact_ProcRRCResumeRequest1()` | RRCResumeRequest 수신 시 INACTIVE 지속 시간·원인 이력 기록 | ✅ **완료** (기존 구현) — `VccbAiClient_OnResumeRequest()` 연결됨 |
| **포인트④** | `FeatRrcInactive.cpp` → `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` | RNA Update 수신 시 AI RNA 크기 권고값 로그 출력 | ✅ **완료** — `FnVCCBAiClient_OnRnaUpdate()` 연결 |
| **포인트⑤** | `StateSaHo.cpp` (XnAP HO 핸들러) | XnAP HO 수신 시 소스 셀 체류 시간을 aiCtx 이력에 seed | ✅ **완료** — `FnVCCBAiClient_SeedFromXnapHistory()` 연결 (store 성공 게이트 이후 순서) |

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

---

## 9. 2026-07-06 세션 작업 로그 — 잔존 항목 완료 검증

이전 세션 기록(§3)에 "미구현"으로 남아있던 5개 항목이 실제로는 이미 구현 완료된 상태였음을 소스 코드 직접 확인으로 검증했다. 문서가 실제 구현 완료 시점(2026-07-05 저녁)의 후속 갱신을 놓치고 있었다.

### 9-1. 코드 직접 검증 결과

| 항목 | 파일:라인 | 확인 내용 |
|---|---|---|
| 포인트④ RNA Update AI 로그 | `FeatRrcInactive.cpp:737` | `FnVCCBAiClient_OnRnaUpdate(ptCi)` 호출 |
| 포인트⑤ XnAP HO 이력 seed | `StateSaHo.cpp:5890` | `FnVCCBAiClient_SeedFromXnapHistory(ptCi)`, `if (GOOD != status) return status;` 게이트 **뒤**에 위치(순서 버그 수정 반영) |
| `resume_count_24h` 슬라이딩 윈도우 | `VccbAiClient.cpp:846` | `calcResumeCount24h(ctx)` — 타임스탬프 링 버퍼 방식 |
| `resumeSuccessRate` 실패 분기 | `VccbAiClient.cpp:864` | `isSuccess = (aiCause == VCCB_AI_CAUSE_EMERGENCY) ? 0.0f : 1.0f` EMA(α=0.1) |
| itscp `get stat ai` | `getCmd.cpp:22180`, `main.cpp:2364` | `GetStatAiReport()` 구현 + `ITSCP_SET_FUNC` 등록 |
| `VccbAiWeights.h` 재생성 여부 | 파일 mtime | Jul 5 21:55 — `VccbAiClient.cpp`(21:02)/`StateSaHo.cpp` 수정 이후 시점 → 최신 코드 반영한 재학습·export 완료로 판단 |
| `test_SaRrcInactiveAI.json` | 18개 testcase | 전부 `"skip": 0`, `tc_filter_use: 0`(필터 비활성 → 전체 실행) |

### 9-2. 유일한 남은 항목 — "구현 불가" 결론을 같은 날 재검증 후 정정 (낮은 우선순위로 보류)

- **`stat[5]` cn_stationary**: `VccbAiClient.cpp:228`에 `input->stat[5] = 0.0f` 고정. 당초 이 세션 초반에는 "AMF가 이 파라미터를 CU-CP로 내려주지 않는 것으로 확인" → 구조적으로 구현 불가로 기록했으나, **같은 날 후속 세션에서 코드를 직접 대조한 결과 이 판단이 틀렸음을 확인**했다:
  - NGAP 코덱(`Middleware/NR_CALL_LIB/IF_NGAP/UNPACK/src/ngapUnpackIE.c:143031 fnNGAP_saveIENGAP_ExpectedUEBehaviour_type`)은 `CoreNetworkAssistanceInformation` IE 안의 `ExpectedUEBehaviour`(`expectedUEMobility` stationary/mobile 포함)를 **이미 unpack하도록 구현되어 있다** — "AMF가 보내도 받을 방법이 없다"는 전제가 코드상 사실이 아니었다.
  - 진짜 원인은 VCCB `VNIL_StoreNGAP_CoreNetworkAssistanceInformation()`(`VCCB_MSG/NGAP/src/MsgNgapStore.cpp:4690`)이 `uEIdentityIndexValue`/`pagingDrx`/`periodicRegistrationUpdateTimer`/`tAIListForInactive`/Ext-IE만 읽고 `expectedUEBehaviour`는 아예 참조하지 않는 것 — `VccbAiClient.cpp:227` 원래 주석("not wired yet")과 정확히 일치한다. 저장할 struct 필드도 `VccbUeRrcInactiveInfo_s`(`Base/VccbItscpDefine/include/VccbStruct.h:7724`)에 없다.
  - 다만 이 값이 AI 추론에서 차지할 **중요도를 함께 평가**한 결과: 라벨 공식(§D-4)이 cn_stationary에 의존하지 않고, `feature_signal.txt` 기준 STRONG feature(primary_5qi/has_gbr_flow/ambr)가 persona의 "고정형 여부"를 이미 상당 부분 대변하며, `stat[4]`(handover_count_24h, 애초에 cn_stationary 대체용으로 도입된 이동성 proxy)와도 개념적으로 겹쳐 **한계효용이 낮을 것으로 판단**된다.
  - **결정(2026-07-06, 사용자 확인)**: struct 필드 추가·저장 함수 구현·LOCKSTEP 재학습까지 가는 전체 배선은 지금 하지 않고 `0.0f` 유지. 다만 "AMF 미제공, 영구 미배선, 재검토 대상 아님"이라는 기존 문구는 근거가 없으므로 폐기하고, **"검증 결과 낮은 우선순위로 보류 — 필요 시(예: AMF가 실제로 이 IE를 보내는지 관측 카운터로 확인한 뒤) 재평가 가능"**으로 재분류한다. 관측 카운터 자체도 아직 착수하지 않았다(향후 옵션으로만 기록).

### 9-3. UT 실행 결과 (RHEL8.10, i686 UT 바이너리) — 전체 통과 확인

`./OBJ/ACPF/i686/UT_VCCB.i686` 실행 결과, 6 suite 43 tests 전부 `PASSED`:

```
[==========] 43 tests from 6 test suites ran. (4674 ms total)
[  PASSED  ] 43 tests.
```

- `test_SaRrcInactive`(1) / `VccbAiEngine`(7) / `VccbAiWeights`(5) / `VccbAiContext`(1) / `test_SaRrcInactive_AI`(11) / `test_SaRrcInactive_Json`(18, base 4 + matrix 14) 전부 통과.
- 최초 1회 실행에서 `AI_matrix_B_secondary_video2`가 `lastInferTimerSec=42`로 밴드([1,3]) 초과 실패했으나, 재실행 시 `resume_prob=0.249`(RELEASE, timer=1)로 정상 통과 — DRB 활동 상태(activityStatus) 타이밍에 따라 직전 라운드의 defer 상태가 남아있을 수 있음(픽스처 격리 자체는 `SetUp()`에서 보장되므로 회귀는 아니고, 케이스 내부 이벤트 순서/상태머신 타이밍 민감성으로 판단됨 — 실패 재현 시 후속 조사 대상).

### 9-4. 코드 밖 후속 판단 사항 (미검증)

- Windows 참조 워크스페이스 → Linux p4 빌드 워크스페이스로의 add/submit/sync 여부 (§7-4-5 참조, 미확인). §9-3의 UT 실행은 RHEL8.10 워크스페이스에서 확인됐으므로 최소한 소스는 Linux 측에 반영되어 있음이 간접 확인됨.
- **i686 UT 바이너리는 호스트(x86) 빌드** — 실제 타겟인 ARM xcc 크로스컴파일(`VCCB` 프로덕션 바이너리) 빌드 통과 여부는 별도 확인 필요.
- Phase-2 실측 데이터 fine-tuning 사이클은 설계상 후속 단계(§D-0)이며 현재 블로킹 아님.

---

## 10. 2026-07-06 세션 작업 로그(3) — BuildServer 재학습 실행 절차 리뷰

사용자가 `BuildServer`(`/p4_ws/sangku/26b_ai_vzw/.../VCCB/ML`) 에서 직접 실행한 재학습 로그를 검토했다. **큰 틀은 §8-5 절차와 일치하나, fine-tune 단계에 신규 데이터가 반영되지 않는 절차 누락이 발견됐다.**

### 10-1. 실행 순서 및 단계별 평가

| # | 실행 명령 | 평가 |
|---|---|---|
| 1 | `synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0` | ✅ base 합성 데이터 5,000 UE → 64,465행 |
| 2 | `source venv/bin/activate` | ✅ |
| 3 | `VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py` | ✅ Method B(기존 `real/*.csv` 8개 파일, 77,965행) + Method A(합성 20,000) = **97,965행**, `dataset.npz` 저장 |
| 4 | `python train.py` | ✅ Phase 0 scratch 학습(50 epoch). **AUROC=0.966, F1@0.6=0.913, MAE=10.69s** (naive 12.87s·persistence 21.36s 모두 상회) — Phase-1 floor 통과 |
| 5 | `python export_weights.py` | ⚠️ **시점이 이르다** — 바로 다음에 fine-tune을 하므로 이 시점의 export는 fine-tune 전 가중치를 잠깐 내보낸 것에 불과하다(뒤에서 다시 export하며 덮어씀). 기능적 문제는 없으나, 이 사이에 VCCB를 재빌드했다면 fine-tune 반영 전 가중치가 배포될 위험이 있었다. **export_weights.py는 파이프라인 맨 마지막에 1회만 실행 권장.** |
| 6 | `cp model.pt model.pt.phase1` | ✅ Phase 0 모델 백업 |
| 7 | `mkdir -p real` | 무해(이미 존재하는 디렉터리, no-op) |
| 8 | `gen_realish_csv.py --out real/shift.csv --n 3000 --dur-scale 2.2` | ✅ 신규 필드 모사 데이터 생성(`append=False`로 기존 `shift.csv` 덮어씀). 스크립트 출력이 그 다음 할 일을 정확히 안내함: `"Next (dry-run): preprocess.py -> train.py ... -> VCCB_AI_FINETUNE=1 python train.py"` |
| 9 | `evaluate.py` | 참고용 — 이 시점 `dataset.npz`(8번 이전 버전 `shift.csv` 기준)에 대한 Phase 0 모델 평가. 정보성 리포트만 생성, 파이프라인에 영향 없음 |
| 10 | `cp model.pt model.pt.phase1` | 6번과 동일 명령 중복(무해, 불필요) |
| 11 | `VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py` | ❌ **핵심 문제** — 8번에서 만든 신규 `real/shift.csv`를 반영하려면 **`preprocess.py`를 재실행해 `dataset.npz`를 갱신**해야 하는데 이 단계가 누락됨. 그 결과 fine-tune이 3번에서 만든 것과 **동일한(구 `shift.csv` 포함) `dataset.npz`**로 재실행됐다 — 8번에서 생성한 신규 필드 데이터가 **전혀 학습에 반영되지 않았다.** |
| 12 | `export_weights.py` | 최종 export. 그러나 11번이 사실상 "같은 데이터로 재학습"이었으므로, 최종 `VccbAiWeights.h`는 **Phase 0과 실질적으로 동일**한 가중치다. |

### 10-2. 증거 — before/after 지표가 거의 그대로임

| 지표 | Phase 0 (4번) | Fine-tune 후 (11번) | 차이 |
|---|---|---|---|
| Val loss | 0.2384 | 0.2381 | −0.0003 |
| AUROC | 0.9660 | 0.9660 | 0 |
| F1@0.6 | 0.913 | 0.913 | 0 |
| MAE | 10.69s | 10.68s | −0.01s |

신규 데이터가 전혀 유입되지 않았으므로 모델이 **이미 학습한 것과 동일한 분포를 재학습**한 것과 같다 — 위 표의 미세한 변화(부동소수 반복·셔플 순서 차이 수준)는 그 진단과 정확히 일치한다. **지금 배포된 `VccbAiWeights.h`는 fine-tune이 아니라 Phase 0(base) 모델이라고 보는 게 정확하다.**

### 10-3. 실행 가능한 명령어 (복붙용) — 옵션·출력·시각화 해설 포함

> **핵심 규칙**: `preprocess.py`가 만드는 `dataset.npz`는 **스냅샷**이다. `real/*.csv`에 새 파일을 넣거나 기존 파일을 갱신해도 `preprocess.py`를 다시 돌리지 않으면 `train.py`는 그 변화를 절대 볼 수 없다. **CSV를 새로 만들거나 갱신한 직후에는 항상 `preprocess.py`를 재실행**한다. `export_weights.py`는 배포할 최종 가중치가 확정된 뒤 **파이프라인 맨 마지막에 1회만** 실행한다.
>
> **시각화는 이미 자동이다** — 아래 각 단계는 별도 플래그 없이 (a) 터미널에 유니코드/ASCII 막대그래프를 바로 찍고 (b) 동시에 PNG 파일로도 저장한다(`termplot.py` 연동, 2026-07-05 구현). 실행 위치는 모두 `ML/` 디렉터리 기준이다.

#### (0) 사전 확인 — base 모델이 이미 있다면 생략 가능

이미 `model.pt`/`VccbAiWeights.h`가 존재한다면 아래 (1)은 건너뛰고 (2)부터 시작한다. 처음부터 새로 만드는 경우에만 (1)부터 실행한다.

#### (1) base 데이터 생성 + 학습 (Phase 0, scratch)

```bash
python3 synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0
VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py
python train.py
cp model.pt model.pt.phase1
```

| 명령/옵션 | 의미 |
|---|---|
| `synth_data_gen.py --out <path>` | 생성한 합성 CSV를 저장할 경로. |
| `--ues 5000` | 가상 UE(단말) 5,000대를 시뮬레이션 — 늘리면 데이터가 많아지지만 실행 시간도 늘어남. |
| `--seed 0` | 난수 시드. 같은 시드면 항상 동일한 데이터가 재현됨(디버깅/비교용). |
| `VCCB_AI_DATA_FILE=<path>` (env) | `preprocess.py`가 "실측/보고용 CSV(Method B)"로 병합할 **단일 파일** 경로. 기본값은 `vccb_ai_dataset.csv`. |
| `VCCB_AI_DATA_DIR=<dir>` (env) | 위와 별개로, 이 디렉터리 안의 **모든 `*.csv`를 자동으로 함께 병합**(수집 배치마다 파일이 쌓이는 구조). 기본값은 `./real`. |
| `train.py` | 인자 없음 — 환경변수(`VCCB_AI_FINETUNE` 등)가 없으면 항상 **처음부터(scratch) 50 epoch 전체 학습**을 수행. |
| `cp model.pt model.pt.phase1` | 방금 만든 base 모델을 이름 바꿔 백업 — 이후 fine-tune이 잘못돼도 되돌아올 지점. |

**출력에서 볼 것**
- `preprocess.py`: `[Method B] real CSV : N unique samples` = 실측/보고용 CSV에서 몇 행이 들어갔는지. `[Method A] synthetic : 20,000` = 내장 3GPP 규칙 합성 데이터(항상 추가됨). `resume_prob=1 (resume < 60s): NN.N%` — **40~60% 사이면 정상**(한쪽으로 쏠리면 라벨이 편향돼 학습이 왜곡됨).
- `train.py`: `Epoch` 표의 `Val` 열이 매 epoch 줄어들다 평평해지면 정상 수렴. `AUROC`(1에 가까울수록 좋음, ≥0.9면 우수), `F1@0.6`(임계값 0.6에서의 정밀도·재현율 조화평균), `MAE`(평균 초 단위 예측 오차 — `naive`/`persistence` baseline보다 작아야 "모델이 의미 있다"는 뜻).

**시각화**
- 터미널: `preprocess.py` 실행 시 최종 라벨 분포 히스토그램(로그 스케일) 자동 출력. `train.py` 실행 시 loss 스파크라인 + MAE/AUROC 막대 + resume_prob 분리 히스토그램 자동 출력.
- 파일: `train_report_full.png`(학습곡선/예측-실제/MAE/AUROC 4패널), `train_metrics_full.json`(수치 원본).

#### (2) 신규 필드(실측 모사) 데이터 생성

```bash
python3 gen_realish_csv.py --out real/shift.csv --n 3000 --dur-scale 2.2
```

| 옵션 | 의미 |
|---|---|
| `--out real/shift.csv` | 저장 경로. **주의: 기본이 덮어쓰기**(`--append` 안 주면 기존 파일이 사라짐). |
| `--n 3000` | 생성할 행 수. |
| `--dur-scale 2.2` | INACTIVE 지속시간을 기존 합성 데이터 대비 **2.2배로 늘려서** "분포가 이동한(shift) 필드 데이터"를 흉내냄(1.0=이동 없음, 순수 재현 확인용). |
| `--seed 7`(기본값, 생략 가능) | 재현용 난수 시드. |
| `--rsrp-bias 0.15`(기본값) | 기존 합성 데이터에 없던 "RSRP가 낮을수록 지속시간이 길어지는" 결합관계를 추가로 주입하는 강도. |
| `--lab-mode`(옵션, 기본 off) | 실제 랩 UE를 흉내낼 때: AMBR(stat2)·RNA(stat8)를 고정값으로 못박고 5QI/RSRP/시간대만 바꿈(실측 수집 특성에 더 가깝게). |
| `--append` | 기존 파일에 이어서 추가(기본은 덮어쓰기이므로, 누적하려면 반드시 이 플래그 필요). |

> 이 스크립트는 **"진짜 실측 데이터"가 아니라 fine-tune 배관(machinery) 자체가 제대로 동작하는지 증명하기 위한 의도적으로 분포가 다른 더미 데이터**다(스크립트 최상단 주석 참조). 진짜 필드 로그가 쌓이면 그 CSV를 같은 스키마로 `real/`에 넣고 이 스크립트는 건너뛰어도 된다.

**출력에서 볼 것**: `label mean : 177.1s   resume<60s = 29.6%` — 방금 만든 데이터의 라벨 평균/양성비율. `written : .../real/shift.csv (append=False)` — `append=False`면 **덮어썼다는 뜻**이므로 이전에 이 파일에 들어있던 데이터는 사라졌음을 인지할 것.

#### (3) ★ 반드시 재실행 — 신규 CSV를 `dataset.npz`에 반영

```bash
VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py
```

(1)과 완전히 동일한 명령이지만, **`real/`에 (2)에서 만든 새 `shift.csv`가 이미 있으므로 이번엔 그 내용이 `dataset.npz`에 포함된다.** 이 단계를 건너뛰면 아래 (4)는 신규 데이터를 전혀 못 보고 예전 데이터로만 재학습하게 된다(2026-07-06 BuildServer 실행에서 실제로 발생한 문제, §10-1/§10-2 참조).

**출력에서 볼 것**: `[Method B] real CSV : N unique samples from K file(s)` 목록에 `./real/shift.csv`가 **새 행 수로** 다시 나열되는지 확인. `outlier_report.png`가 새로 생겼다면 = 방금 넣은 데이터 중 일부가 이상치 규칙(1초 미만/1시간 초과/AMBR 튐 등)에 걸려 자동 제거됐다는 뜻(정상 동작, 걸러진 비율만 확인).

**시각화**: 라벨 분포 히스토그램이 터미널에 다시 출력됨(이번엔 신규 데이터가 섞인 최종 분포). 이상치가 하나라도 제거됐을 때만 `outlier_report.png` + 터미널 전/후 비교 막대그래프가 추가로 나온다(제거 건수 0이면 생성 안 됨 — 정상).

#### (4) fine-tune 실행

```bash
VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py
```

| 환경변수 | 의미 | 기본값 |
|---|---|---|
| `VCCB_AI_FINETUNE=1` | scratch(처음부터) 학습이 아니라, **직전 `model.pt`를 불러와(warm-start) 이어서 조금만 더 학습**하도록 전환. | `0`(꺼짐=scratch) |
| `VCCB_AI_FREEZE_GRU=1` | 시퀀스(GRU) 부분과 정적 feature(MLP) 부분은 **고정**하고 출력 head 2개만 미세조정. 신규 데이터가 적을 때 기존에 학습한 구조를 깨뜨리지 않기 위한 안전장치. | `0`(전체 갱신) |
| `VCCB_AI_FINETUNE_LR`(미지정, 기본 `2e-4`) | fine-tune 학습률 — scratch보다 훨씬 작게(=조금씩만 바꿈). | `2e-4` |
| `VCCB_AI_FINETUNE_EPOCHS`(미지정, 기본 `15`) | fine-tune epoch 수(scratch의 50보다 적음). | `15` |
| `VCCB_AI_EARLY_STOP_PATIENCE`(미지정) | 검증 손실이 N epoch 연속 개선 안 되면 조기 종료. fine-tune 모드에서는 기본 `5`(자동으로 과적합 방지), scratch 모드에서는 기본 `0`(끝까지 50 epoch 실행). | 모드별 자동 |

**출력에서 볼 것**: 맨 위 `[FINETUNE] pre-FT on this val: AUROC=... F1=... MAE=...`가 **직전 (1)의 최종 지표와 다르면 정상**(신규 데이터가 반영돼 val set 자체가 달라졌다는 뜻). 학습이 끝난 뒤 `[timer] premature-release/wasted-hold` 줄까지 나오면 완료. **before/after 지표(AUROC/F1/MAE)가 소수점 3~4자리 이상 유의미하게 움직여야 fine-tune이 제대로 데이터를 반영했다는 증거** — 거의 완전히 동일하면 (3)을 빼먹었다는 신호(2026-07-06 실사례, §10-2 참조).

**시각화**
- 터미널: `mode=finetune`으로 표시되는 loss 스파크라인 + MAE/AUROC의 **before/after 막대 비교**(fine-tune 전용) + resume_prob 분리 히스토그램.
- 파일: `train_report_finetune.png`(before/after 비교 패널 포함), `train_metrics_finetune.json`.

#### (5) 평가 리포트/시각화 (권장, 스킵해도 파이프라인엔 무해)

```bash
python3 evaluate.py
```

옵션 없이 그대로 실행하면 된다(`--show`를 주면 화면에 그래프 창을 띄우는데, 헤드리스 빌드서버에서는 필요 없음 — 주지 않아도 PNG는 그대로 저장됨).

**출력에서 볼 것**: `beats naive=Y beats persistence=Y`가 둘 다 `Y`여야 "이 모델이 단순 규칙보다 낫다"는 뜻. `avg gap vs oracle`이 작을수록(예: 4.2s) "지금 가진 feature로 낼 수 있는 최선치에 근접했다"는 뜻.

**시각화**
- 터미널: MAE 비교 막대그래프(모델 vs naive vs persistence vs oracle) 자동 출력.
- 파일: `eval_pred_vs_true.png`(예측 vs 실제 산점도), `eval_baseline_mae.png`(baseline 비교 막대), `eval_model_vs_oracle.png`(oracle 근접도), `eval_report.txt`(텍스트 리포트 전체).

#### (6) ★ export — 여기서 딱 한 번만

```bash
python3 export_weights.py
```

인자 없음. `model.pt`(방금 fine-tune된 최종본)를 읽어 C++ 헤더 `VccbAiWeights.h`로 변환한다.

**출력에서 볼 것**: `Total parameters : 1,266`(항상 이 값이어야 함 — 다르면 모델 구조가 바뀐 것). `Inference @feature-mean : resume_prob=0.33xx`처럼 **1.000이 아닌 값**이어야 정상(모델이 상수로 붕괴하지 않았다는 뜻). `Exported → .../VccbAiWeights.h` 경로 확인 후, VCCB/itscp_vccb/UT를 순서대로 재빌드한다.

> **주의**: export_weights.py를 (4) fine-tune **이전에** 먼저 실행해두면, 그 사이 누군가 VCCB를 재빌드할 경우 fine-tune 반영 전 가중치가 배포될 수 있다(2026-07-06 BuildServer 실행에서 실제로 이 순서로 조기 실행된 사례 있음, §10-1 참조). **fine-tune이 확정된 뒤 마지막에 1회만** 실행할 것.

### 10-4. 권장 후속 조치

이번 실행분(§10-1)은 절차상 fine-tune이 무의미했으므로, 신규 필드 데이터(`real/shift.csv`)를 실제로 반영하려면 §10-3 절차대로 **`preprocess.py` 재실행 → fine-tune → export_weights.py**를 다시 수행하는 것을 권장한다. 다만 현재 배포된 가중치(Phase 0 기준)도 Phase-1 floor는 통과한 상태라 급하게 막힌 것은 아니다.
