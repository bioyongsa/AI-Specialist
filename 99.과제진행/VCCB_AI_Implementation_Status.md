# VCCB AI 추론 기능 구현 현황

> **기준일**: 2026-05-23
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

---

## 1. 수정된 파일 목록

| 파일 | 위치 | 주요 변경 내용 |
|---|---|---|
| `VccbAiClient.cpp` | `VCCB_MAIN/src/` | 핵심 AI 정책 로직 전면 재구현 + 포인트④⑤ 신규 API + 전역 통계 + 24h 슬라이딩 윈도우 |
| `VccbAiClient.h` | `VCCB_MAIN/hdr/` | 포인트④⑤ 함수 선언, 전역 통계 구조체, `VccbAiClient_PrintStats()` 추가 |
| `VccbAiEngine.h` | `VCCB_MAIN/hdr/` | `VccbAiCallContext_type`에 `resumeTs[10]`, `resumeTsWriteIdx` 필드 추가 |
| `FeatRrcInactive.cpp` | `VCCB_FEAT/RRC_INACTIVE/src/` | 포인트② BearerContextModification 전송 연동, 포인트④ RNA Update AI 로그 추가 |
| `MsgE1apMake.cpp` | `VCCB_MSG/E1AP/src/` | 포인트① AI 타이머 추천 함수 연결 |
| `StateSaRrcInact.cpp` | `VCCB_STATE/FEAT_SA/src/` | 포인트③ `VccbAiClient_OnResumeRequest()` 연결 |
| `StateSaHo.cpp` | `VCCB_STATE/FEAT_HO/src/` | 포인트⑤ `VccbAiClient_SeedFromXnapHistory()` 호출 추가 |
| `MainInit.cpp` | `VCCB_MAIN/src/` | `VccbAiClient_Reset()` 초기화 호출 추가 |

---

## 2. 완료된 항목

### 2-1. Critical — 핵심 로직

| 항목 | 파일 | 내용 |
|---|---|---|
| **포인트② `IsAbleToProceedInactive` 재구현** | `VccbAiClient.cpp` | AI 추론(`GRU + MLP`) 실제 호출. `resume_prob ≥ 0.6` 시 타이머·타임스탬프 저장 후 `false` 반환(INACTIVE 보류). `extendCount ≥ 3` 시 카운터 리셋 후 강제 INACTIVE 천이(`true`). Fallback 시 `true`(기존 동작 유지). |
| **포인트② BearerContextModification 전송** | `FeatRrcInactive.cpp` | AI가 `false` 반환 시 `VEIL_sendE1SetupModRelRequest(E1Procedure_BearerContextModification)` fire-and-forget 호출. `#include "MsgE1apSend.h"` 추가. |
| **포인트① E1AP inactivityTimer IE 연결** | `MsgE1apMake.cpp` | `VEIL_fillE1_Inactivity_Timer()` 내 `FnVCCB_GetRRCInactiveTimerValue()` 직접 호출 → `VccbAiClient_RecommendInactivityTimer()` 래퍼로 교체. |
| **포인트③ Resume 이력 기록** | `StateSaRrcInact.cpp` | `FnVCCB_Inact_ProcRRCResumeRequest1()` 에서 `VccbAiClient_OnResumeRequest()` 호출. RNA Update 분기 제외. |
| **포인트④ RNA Update AI 권고 로그** | `FeatRrcInactive.cpp` | `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` 의 RNA Update(`NRRRC_rna_Update && dlDataFlag == OFF`) 분기에서 `VccbAiClient_OnRnaUpdate()` 호출. resume_prob 기반 RNA 크기 권고(`KEEP_OR_EXPAND` / `NO_CHANGE` / `SHRINK`) 로그 출력. |
| **포인트⑤ XnAP HO AI Context seed** | `StateSaHo.cpp` | `VCCB_ProcMsgXnap_HandoverRequest()` 내 `FnVCCB_StoreXNMessage()` 직후 `VccbAiClient_SeedFromXnapHistory()` 호출. `visitedCell[lastIdx].uRat.nrCell.stayTime` (+ `stayTimeEnhanced`) 을 `aiCtx.history[]` 에 seed. |

### 2-2. 안정성 보강

| 항목 | 파일 | 내용 |
|---|---|---|
| **`IS_REDCAP_UE` / `IS_NRDC_MN,SN` skip** | `VccbAiClient.cpp` | `shouldSkipAi()` 내 REDCAP/NRDC 분기 활성화. |
| **글로벌 카운터 멀티스레드 안전** | `VccbAiClient.cpp` | `std::atomic<int64_t>` 전역 통계 카운터 사용 (C++11). |
| **`VccbAiClient_Reset()` 초기화** | `MainInit.cpp` | `FnVCCB_InitBlock()` 내 `VccbAiEngine_Init()` 직후 호출. 통계 카운터 전체 리셋. |
| **추론 타임아웃(1ms) 측정 및 Fallback 연동** | `VccbAiClient.cpp` | `clock_gettime(CLOCK_MONOTONIC)` 기반. `VCCB_AI_INFER_TIMEOUT_US(1000us)` 초과 시 `timeoutCount` 증가 + Fallback. |

### 2-3. 기능 보완

| 항목 | 파일 | 내용 |
|---|---|---|
| **24h 슬라이딩 윈도우 resumeCount24h** | `VccbAiClient.cpp`, `VccbAiEngine.h` | `aiCtx`에 `resumeTs[10]` 타임스탬프 링 버퍼 추가. `OnResumeRequest()` 호출마다 현재 시각 기록 후 24h 이내 항목 카운트. |
| **resumeSuccessRate 실패 판단 로직** | `VccbAiClient.cpp` | `NRRRC_emergency` cause = 실패(0.0f), 그 외 = 성공(1.0f). EMA(α=0.1) 갱신. |
| **전역 AI 추론 통계 구조체** | `VccbAiClient.h`, `.cpp` | `VccbAiGlobalStats_type`: 총 요청·성공·실패·타임아웃·Fallback·트리거별·평균 추론 시간. |
| **통계 출력 (`VccbAiClient_PrintStats`)** | `VccbAiClient.cpp` | `VCCB_PRINTF` 기반 포맷 출력. itscp 핸들러 또는 디버그 커맨드에서 호출 가능. |

### 2-4. Feature 벡터 매핑 (확인된 필드)

| stat[] | Feature 이름 | 접근 경로 |
|---|---|---|
| `stat[0]` | primary_5qi | `drb[i].drbQosIdxInCall` → `qosFlow[idx].qosParam.nonDyna.fiveQi` |
| `stat[1]` | drb_count | `drb[i].usage == CI_use` 순회 카운트 |
| `stat[2]` | ue_ambr_dl_mbps | `ueInfo.ambr.aggrMbrDl / 1,000,000` (bps → Mbps) |
| `stat[3]` | has_gbr_flow | `qosFlow[i].qosParam.gbrQosInformationPresent != 0` |
| `stat[6]` | last_rsrp | `servCellSsbRsrp` (값 `255` = invalid → `0.0f` 처리) |
| `stat[8]` | rna_cell_count | `ueInfo.RrcInactiveInfo.numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | `aiCtx.avgDurationSec` (EMA) |
| `stat[10]` | resume_count_24h | `aiCtx.resumeCount24h` (24h 슬라이딩 윈도우) |
| `stat[11]` | resume_success_rate | `aiCtx.resumeSuccessRate` (EMA, NRRRC_emergency = 실패) |
| `stat[12]` | avg_inactive_duration | `aiCtx.avgDurationSec` |
| `stat[13]` | hour_of_day | `localtime(time(NULL))->tm_hour` |
| `stat[14]` | day_of_week | `localtime(time(NULL))->tm_wday` |

---

## 3. 남은 항목

### 3-1. 🟡 중간 — Feature 벡터 미확인 필드

| stat[] | Feature 이름 | 현재 값 | 해결 방법 |
|---|---|---|---|
| `stat[4]` | cn_expected_idle_period | `0.0f` (고정) | 5G NR NGAP CoreNetworkAssistanceInformation에 해당 필드 미존재. 향후 NGAP 규격 확장 시 `VccbStruct.h` `ueInfo.RrcInactiveInfo` 계열에서 확인. |
| `stat[5]` | cn_stationary | `0.0f` (고정) | 동상 — NGAP stationaryIndication 필드 경로 미확인. |
| `stat[7]` | neighbor_cell_count | `0.0f` (고정) | `NRRRC_MeasurementReport` 처리 후 ptCi에 저장되는 인접 셀 카운트 필드 미확인. `VccbStruct.h` 내 `measResult`, `nbrCellCount` 계열 검색 필요. |

### 3-2. 🟢 낮음 — 통합 및 운영

| 항목 | 파일 | 내용 |
|---|---|---|
| **itscp_vccb 디버그 커맨드 연결** | itscp 핸들러 | `VccbAiClient_PrintStats()` 를 itscp 디버그 커맨드 핸들러에 연결. 현재는 단독 함수로 구현 완료. 핸들러 식별 후 연결 작업 필요. |
| **stat[4][5][7] 실제 필드 매핑** | `VccbAiClient.cpp` | 위 3-1 항목 해결 후 `buildAiInput()` 의 `stat[4]`, `stat[5]`, `stat[7]` 코드 교체. |

---

## 4. Feature 벡터 매핑 현황

| stat[] | Feature 이름 | 상태 | 비고 |
|---|---|---|---|
| `stat[0]` | primary_5qi | ✅ 완료 | Non-Dynamic 5QI 직접 접근 |
| `stat[1]` | drb_count | ✅ 완료 | |
| `stat[2]` | ue_ambr_dl_mbps | ✅ 완료 | bps → Mbps 변환 |
| `stat[3]` | has_gbr_flow | ✅ 완료 | |
| `stat[4]` | cn_expected_idle_period | ⚠️ 0.0f | 5G NR NGAP에 해당 필드 미존재, 향후 규격 확장 시 대응 |
| `stat[5]` | cn_stationary | ⚠️ 0.0f | 동상 |
| `stat[6]` | last_rsrp | ✅ 완료 | `servCellSsbRsrp`, 255=invalid 처리 |
| `stat[7]` | neighbor_cell_count | ❌ TODO | MeasurementReport 저장 필드 경로 미확인 |
| `stat[8]` | rna_cell_count | ✅ 완료 | `numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | ✅ 완료 | `avgDurationSec` 재사용 |
| `stat[10]` | resume_count_24h | ✅ 완료 | 24h 슬라이딩 윈도우 구현 (`resumeTs[10]` 링 버퍼) |
| `stat[11]` | resume_success_rate | ✅ 완료 | EMA + NRRRC_emergency = 실패 판단 |
| `stat[12]` | avg_inactive_duration | ✅ 완료 | `avgDurationSec` |
| `stat[13]` | hour_of_day | ✅ 완료 | `tm_hour` |
| `stat[14]` | day_of_week | ✅ 완료 | `tm_wday` |

> **완료**: 13/15 (87%) | **조건부(NGAP 규격 한계)**: 2/15 | **미완료**: 1/15

---

## 5. AI 개입 포인트 현황

| 포인트 | 위치 | 설계 의도 | 구현 상태 |
|---|---|---|---|
| **포인트①** | `MsgE1apMake.cpp` → `VEIL_fillE1_Inactivity_Timer()` | E1AP BearerContextSetup/Modify 시 AI 추천 타이머 값 주입 | ✅ **완료** — `VccbAiClient_RecommendInactivityTimer()` 연결 |
| **포인트②** | `FeatRrcInactive.cpp` → `FnVCCB_IsAbleToProceedRRCInactive()` | InactivityNotification 수신 시 즉시 천이 or 타이머 연장 판단 | ✅ **완료** — AI 추론 호출 + BearerContextModification 전송 |
| **포인트③** | `StateSaRrcInact.cpp` → `FnVCCB_Inact_ProcRRCResumeRequest1()` | RRCResumeRequest 수신 시 INACTIVE 지속 시간·원인 이력 기록 | ✅ **완료** — `VccbAiClient_OnResumeRequest()` 연결 |
| **포인트④** | `FeatRrcInactive.cpp` → `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` | RNA Update 수신 시 AI RNA 크기 권고값 로그 출력 | ✅ **완료** — `VccbAiClient_OnRnaUpdate()` 연결 + resume_prob 기반 RNA 권고 로그 |
| **포인트⑤** | `StateSaHo.cpp` → `VCCB_ProcMsgXnap_HandoverRequest()` | XnAP HO 수신 시 소스 셀 체류 시간을 aiCtx 이력에 seed | ✅ **완료** — `VccbAiClient_SeedFromXnapHistory()` 연결 |

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
| `VCCB_AI_24H_WINDOW_SEC` | `86400` | 24h 슬라이딩 윈도우 크기 (초) |

---

## 부록 — 전역 통계 구조체 (`VccbAiGlobalStats_type`)

`VccbAiClient_GetStats()` 또는 `VccbAiClient_PrintStats()` 로 조회 가능.

| 필드 | 의미 |
|---|---|
| `totalRequests` | 총 추론 요청 수 (포인트①②④ 합산) |
| `successCount` | 추론 성공 수 |
| `failureCount` | 추론 실패 (NaN, buildInput 오류) 수 |
| `timeoutCount` | 1ms 초과 타임아웃 수 |
| `fallbackCount` | Fallback 응답 총 수 |
| `triggerPt1Count` ~ `triggerPt5Count` | 포인트별 호출 횟수 |
| `totalInferUs` | 성공한 추론의 총 소요 시간 (us) |
| `consecutiveFailCount` | 현재 연속 실패 횟수 |
| `isFallbackMode` | 현재 Fallback 모드 여부 |
