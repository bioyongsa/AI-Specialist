# VCCB AI 추론 기능 구현 현황

> **기준일**: 2026-05-21
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
| `stat[6]` | last_rsrp | `servCellSsbRsrp` (값 `255` = invalid → `0.0f` 처리) |
| `stat[8]` | rna_cell_count | `ueInfo.RrcInactiveInfo.numInactTAI` |
| `stat[9]` | avg_cell_stay_sec | `aiCtx.avgDurationSec` (EMA) |
| `stat[10]` | resume_count_24h | `aiCtx.resumeCount24h` |
| `stat[11]` | resume_success_rate | `aiCtx.resumeSuccessRate` (EMA) |
| `stat[12]` | avg_inactive_duration | `aiCtx.avgDurationSec` |
| `stat[13]` | hour_of_day | `localtime(time(NULL))->tm_hour` |
| `stat[14]` | day_of_week | `localtime(time(NULL))->tm_wday` |

---

## 3. 남은 항목

### 3-1. 🔴 높음 — Feature 벡터 미확인 필드

| stat[] | Feature 이름 | 현재 값 | 해결 방법 |
|---|---|---|---|
| `stat[4]` | cn_expected_idle_period | `0.0f` (고정) | `VccbStruct.h` 및 NGAP InitialContextSetupRequest 처리 코드에서 `coreNetworkAssistance` → `expectedIdlePeriod` 저장 위치 추적 |
| `stat[5]` | cn_stationary | `0.0f` (고정) | 동상 — `stationaryIndication` 필드 경로 확인 |
| `stat[7]` | neighbor_cell_count | `0.0f` (고정) | MeasurementReport 저장 구조체(`measResultNeighCell` 계열) 내 인접 셀 수 필드 확인. `VccbStruct.h` 내 `nbrCell`, `neighborCell` 검색. |

### 3-2. 🟡 중간 — 기능 보완

| 항목 | 파일 | 내용 |
|---|---|---|
| **포인트④ RNA Update 시 AI 권고 로그** | `FeatRrcInactive.cpp` | `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` 내 `resumeCause == NRRRC_rna_Update && dlDataFlag == VCCB_FLAG_OFF` 분기에서 AI RNA 크기 권고값 조회 및 로그 출력. |
| **포인트⑤ XnAP HO 시 AI Context seed** | `StateHandleSa.cpp` 등 | XnAP HandoverRequest 수신 후 `VLIB_InitCallInfo()` 직후, UE History(`Time UE Stayed in Cell`)를 `aiCtx.history` 버퍼에 seed. 코드베이스에서 `XN_HandoverRequest` 또는 `RetrieveUEContextRequest` 검색하여 핸들러 위치 확인 필요. |
| **24h 슬라이딩 윈도우 Resume 카운트** | `VccbAiClient.cpp` | `resumeCount24h` 현재 단순 increment. 24h 이내 이벤트만 카운트하는 슬라이딩 윈도우 구현 필요. 타임스탬프 링 버퍼(`aiCtx` 내 추가) 또는 이벤트 시각 배열 활용. |
| **Resume 성공률 실패 판단 로직** | `VccbAiClient.cpp` | `resumeSuccessRate` 현재 모든 Resume을 성공으로 처리(`isSuccess = 1.0f`). `resumeCause` 값에 따라 실패 여부 구분 필요. `NRRRC_emergency`, `NRRRC_rna_Update` 등 설계서 섹션 1-3 매핑 테이블 참조. |

### 3-3. 🟢 낮음 — 디버그 및 운영

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
| `stat[4]` | cn_expected_idle_period | ❌ TODO | 필드 경로 미확인, 현재 `0.0f` |
| `stat[5]` | cn_stationary | ❌ TODO | 필드 경로 미확인, 현재 `0.0f` |
| `stat[6]` | last_rsrp | ✅ 완료 | `servCellSsbRsrp`, 255=invalid 처리 |
| `stat[7]` | neighbor_cell_count | ❌ TODO | 필드 경로 미확인, 현재 `0.0f` |
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
