# VCCB AI 연동 상세 설계서

> **목적**: gNB-CU-CP VCCB SW에 LSTM 기반 RRC Inactivity Timer 최적화 AI를 연동하기 위한 Cursor IDE 작업 가이드
>
> **대상 독자**: Cursor IDE + Claude LLM 환경에서 코드 작성을 수행하는 개발자
>
> **코드베이스**: 삼성 5G gNB-CU-CP VCCB (C++, 사내 MDS/POSIX 기반)
>
> **아키텍처 전제**: AI 추론은 `vccb.xcc` 내부에서 직접 수행한다. 별도 프로세스 및 SHM IPC는 사용하지 않는다.

---

## 목차

1. [메시지 처리 영역 — 기존 VCCB 코드 개입](#1-메시지-처리-영역--기존-vccb-코드-개입)
2. [AI 연동 영역 — 신규 개발](#2-ai-연동-영역--신규-개발)
3. [추론 엔진 구현 시나리오](#3-추론-엔진-구현-시나리오)
4. [callInfo_type 연동 영역](#4-callinfo_type-연동-영역)
5. [빌드 및 환경 영역](#5-빌드-및-환경-영역)

---

## 사전 컨텍스트 — Cursor에서 반드시 먼저 읽어야 할 파일

Cursor에서 각 섹션 작업 시작 전, 아래 파일들을 **반드시 먼저 열어서 컨텍스트에 포함**해야 한다.

| 파일 | 이유 |
|---|---|
| `VccbAiInfer.h` | AI 연동 구조체 및 상수 정의 전체 포함 |
| `VccbStruct.h` | `callInfo_type`, `callDRBInfo_type`, `ueInfo_type` 등 VCCB 핵심 구조체 |
| `FeatRrcInactive.h` / `.cpp` | RRC Inactive 처리 함수 집합, AI 개입 포인트 대부분 포함 |
| `MsgE1apMake.cpp` | E1AP 메시지 생성 함수, 타이머 IE 주입 위치 포함 |
| `StateSaRrcInact.cpp` | RRC Inactive 상태머신, RRCResumeRequest 처리 포함 |
| `FeatCallProc.cpp` | InactivityNotification → RRC Inactive 천이 체인 포함 |
| `StateHandleSa.cpp` | STATE_Conn 핸들러, Inact_Suspend 핸들러 포함 |

---

## 1. 메시지 처리 영역 — 기존 VCCB 코드 개입

이 영역은 **기존 VCCB 코드를 최소한으로 수정**하여 AI 개입 지점을 삽입하는 작업이다. 기존 동작을 깨지 않는 것이 최우선이다.

---

### 1-1. E1AP InactivityNotification 수신 → INACTIVE 천이 판단 (AI 개입 포인트 ②)

#### 개입 대상 함수

`FeatRrcInactive.cpp` 내의 `FnVCCB_IsAbleToProceedRRCInactive()`

#### 현재 동작

이 함수는 E1AP InactivityNotification 수신 후 호출 체인의 끝에서 최종적으로 RRC INACTIVE 천이 여부를 `bool`로 반환한다. 호출 체인은 다음과 같다.

```
E1AP InactivityNotification 수신
  → RscCall::setDataActivityStatus(ActivityStatus_INACTIVE)
  → FnVCCB_UpdateUserInactStateAndDoActionIfExpiry(DATA_EXPIRY)
  → FnVCCB_DoInactivityActionInIdleState()       [FeatCallProc.cpp]
  → FnVCCB_CheckRRCInactive()                    [FeatRrcInactive.cpp]
  → FnVCCB_IsAbleToProceedRRCInactive()          ← 개입 위치
```

SA basic call(`E_CALLTYPE_SA`) 분기에서 `activityStatus == ActivityStatus_INACTIVE` 조건 충족 시 현재는 무조건 `true`를 반환하여 즉시 INACTIVE 천이를 진행한다.

#### 개입 설계

- `activityStatus == ActivityStatus_INACTIVE` 조건이 만족된 시점에, 무조건 `true`를 반환하는 대신 **AI Client 판단 함수를 호출**하여 그 반환값을 사용한다.
- AI 판단 결과가 **"즉시 천이"** 이면 기존과 동일하게 `true` 반환.
- AI 판단 결과가 **"타이머 연장"** 이면 두 가지 동작을 수행한다.
  - `VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification)` 를 호출하여 갱신된 타이머 값을 포함한 BearerContextModificationRequest를 CU-UP에 전송한다 (fire-and-forget).
  - `false`를 반환하여 INACTIVE 천이를 보류한다.
- AI Client가 **Fallback 상태**이거나 **추론 실패** 시에는 반드시 `true`를 반환하여 기존 동작을 보장한다.

#### 주의사항

- `IS_NRDC(ptCi)` 분기 (NRDC MN/SN 케이스) 는 AI 개입 대상이 아니다. SA basic call 분기에만 적용한다.
- `IS_REDCAP_UE(ptCi)` 인 경우, 초기 구현에서는 AI 적용 없이 기존 로직을 유지한다.
- 타이머 연장 선택 시 STATE 전환은 발생하지 않으며 `STATE_Conn` 을 유지한다. BearerContextModify 응답 대기를 위한 별도 STATE 추가는 불필요하다 (fire-and-forget).
- 연속 연장 횟수에 상한을 두어야 한다. `ptCi->aiCtx.extendCount` 가 설정된 임계값(`VCCB_AI_MAX_EXTEND_COUNT`) 이상이면 AI 판단과 무관하게 즉시 천이를 강제하고 카운터를 리셋한다.
- 연장 선택 시 INACTIVE 진입 타임스탬프(`ptCi->aiCtx.inactiveEntryTs`)를 반드시 기록해야 한다. 이 타임스탬프는 추후 포인트③에서 실제 INACTIVE 지속 시간을 계산하는 데 사용된다.

---

### 1-2. E1AP BearerContextSetupRequest / BearerContextModifyRequest 내 inactivityTimer IE 주입 (AI 개입 포인트 ①)

#### 개입 대상 함수

`MsgE1apMake.cpp` 내의 `VEIL_fillE1_Inactivity_Timer(callInfo_type*, callDRBInfo_type*, E1_Inactivity_Timer*)`

이 함수가 DRB 레벨 inactivityTimer를 E1AP IE에 채우는 **최종 함수**이다.

#### 현재 동작

`IS_SUPPORT_RRC_INACTIVE(ptCi)` 분기 내에서 `ptCi->qosFlow[ptDi->drbQosIdxInCall]` 에서 5QI 값을 추출하고, `FnVCCB_GetRRCInactiveTimerValue(call, qos5qi)` 를 호출하여 정적으로 설정된 타이머 값을 반환받아 IE에 설정한다.

#### 5QI 접근 방법 (확인됨)

```
ptDi->drbQosIdxInCall  →  ptCi->qosFlow[] 의 인덱스
ptCi->qosFlow[drbQosIdxInCall].usage == CI_use 인지 확인 후
VLIB_getFiveQi(ptQi, &qos5qi) 로 uint8_t qos5qi 획득
```

#### 개입 설계

- `IS_SUPPORT_RRC_INACTIVE(ptCi)` 분기에서, `FnVCCB_GetRRCInactiveTimerValue()` 호출을 AI 래퍼 호출로 교체한다.
- AI 래퍼는 내부적으로 다음 우선순위로 타이머 값을 결정한다.
  1. AI Client가 Active 상태이고 `ptCi->aiCtx.lastInferTimerSec` 이 유효한 값(0 초과)이면 해당 값 사용.
  2. AI 추론을 새로 수행할 수 있으면 추론 결과 사용.
  3. 위 두 경우 모두 아니면 기존 `FnVCCB_GetRRCInactiveTimerValue()` 반환값 사용 (Fallback).
- 최종 타이머 값은 반드시 3GPP TS 38.413 허용 범위(1초 ~ 3600초) 내로 클램핑해야 한다.
- `IS_REDCAP_UE(ptCi)` 인 경우 초기 구현에서는 AI 적용 없이 기존 로직 유지.

#### 이 함수가 호출되는 맥락

이 함수는 INACTIVE 천이 과정 중 아래 두 시점에 호출된다.

- **BearerContextModification (타이머 포함)**: `VLIB_StartRRCInactive()` → F1 ModQuery 응답 수신 → `VEIL_sendE1SetupModRelRequest(E1Procedure_BearerContextModification)` 내부.
- **BearerContextSetup (초기 타이머)**: 일부 절차에서 Bearer를 새로 설정할 때.

---

### 1-3. RRCResumeRequest1 수신 핸들러 — AI 이력 기록 (AI 개입 포인트 ③)

#### 개입 대상 함수

`StateSaRrcInact.cpp` 내의 `FnVCCB_Inact_ProcRRCResumeRequest1()`

#### 현재 동작

이 함수는 `FnVCCB_Inact_Suspend_CallResumeInd()` 또는 `FnVCCB_Inact_ResumeReq_CallResumeInd()` 에서 호출된다. 내부에서 다음 순서로 동작한다.

1. F1 메시지 unpack 및 store.
2. `RscCall::getDataActivityStatus(*ptCi)` 로 현재 data activity 상태 확인.
3. `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest(ptCi, dataActivityStatus)` 로 다음 상태 결정 (`stateCmd` 변경: RRCResume_cmd / RRCSuspend_cmd / RRCIdle_cmd).
4. stateCmd에 따라 각 절차 진행.

#### 개입 설계

- `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()` **호출 직전**에 AI 이력 갱신 처리를 삽입한다.
- 이 시점에 INACTIVE 지속 시간을 계산한다. 계산 방법: 현재 시각 - `ptCi->aiCtx.inactiveEntryTs` (포인트②에서 기록한 타임스탬프). 단, `inactiveEntryTs` 가 기록되지 않은 경우(포인트②를 거치지 않은 RNA Update 등)는 이력 갱신을 skip한다.
- `ptCi->ueInfo.RrcInactiveInfo.resumeCause` 에서 Resume 원인값을 읽어 이력에 함께 기록한다.
- 이 삽입 코드는 `ptCi->stateCmd == RRCIdle_cmd` 분기(E1/NG reset 케이스) 에서는 실행하지 않는다.

#### Resume Cause 매핑

`ptCi->ueInfo.RrcInactiveInfo.resumeCause` 의 타입은 NRRRC enum이다. AI 이력 구조체에는 uint8_t로 저장하며, 내부 변환 테이블을 통해 `enumVccbAiResumeCause_type` 으로 매핑한다. 주요 값은 다음과 같다.

| NRRRC 값 | 의미 |
|---|---|
| `NRRRC_rna_Update` | RNA Update (위치 등록 목적, 데이터 전송 없음) |
| `NRRRC_mo_Data` | UE 발신 데이터 |
| `NRRRC_mt_Access` | 네트워크 발신 데이터 (Paging에 의한 Resume) |
| `NRRRC_emergency` | 긴급 통화 |

---

### 1-4. RNA Update 수신 시 AI 처리 (AI 개입 포인트 ④)

#### 개입 대상 함수

`FeatRrcInactive.cpp` 내의 `FnVCCB_SetRRCInactiveAction_after_RRCResumeRequest()`

#### 현재 동작

`resumeCause == NRRRC_rna_Update` 이고 `dlDataFlag == VCCB_FLAG_OFF` 인 경우, 다음 stateCmd를 `RRCSuspend_cmd`로 설정하여 RRC INACTIVE 상태를 유지한다.

#### 개입 설계

- 해당 분기에서 `SET_STATE_CMD(RRCSuspend_cmd)` 이전에 AI RNA 크기 권고값을 조회하고 로그로 기록한다.
- 현 시점에서는 RNA 크기를 직접 변경하는 메커니즘이 VCCB 코드에 없으므로, AI 권고값은 **로그 기록에만 사용**한다.
- 향후 RNA 크기 파라미터 조정 메커니즘이 확인되면 실제 적용 로직을 추가한다.

---

### 1-5. XnAP HandoverRequest 수신 시 AI Context 초기화 (AI 개입 포인트 ⑤)

#### 개입 대상 함수

XnAP HandoverRequest 처리 핸들러. `StateHandleSa.cpp` 또는 관련 파일에서 Cursor에서 직접 검색한다.

```
Codebase Search: "XN_HandoverRequest" 또는 "RetrieveUEContextRequest"
```

#### 개입 설계

- 신규 UE Context가 초기화된 직후(`VLIB_InitCallInfo()` 호출 이후), XnAP 메시지에서 UE History 정보(소스 셀 체류 시간 등)를 읽어 AI Context의 이력 버퍼에 seed 데이터로 기록한다.
- XnAP UE History에서 획득 가능한 `Time UE Stayed in Cell` 값을 활용한다. 이 필드의 실제 접근 방법은 Cursor에서 코드베이스를 검색하여 확인해야 한다.

---

## 2. AI 연동 영역 — 신규 개발

이 영역은 `vccb.xcc` 내부에 포함될 신규 AI Client 코드를 작성하는 작업이다. 추론은 Call Task 내에서 **동기 방식으로 직접 호출**되며, 별도 프로세스나 SHM은 사용하지 않는다.

---

### 2-1. 신규 파일 구성

| 파일 | 역할 | 포함 대상 |
|---|---|---|
| `VccbAiInfer.h` | 구조체 및 상수 정의 (이미 작성됨) | Feature 벡터, 추론 결과, enum 등 |
| `VccbAiClient.h` | AI Client 함수 선언 | VCCB 코드에서 호출할 모든 AI 인터페이스 |
| `VccbAiClient.cpp` | AI Client 공통 구현 | 초기화, Feature 벡터 구성, 이력 관리, Fallback 처리 |
| `VccbAiEngine.h` / `.cpp` | 추론 엔진 구현 | 섹션 3의 두 시나리오 중 하나를 선택하여 구현 |

---

### 2-2. AI Client 초기화

#### 구현 위치

`MainInit.cpp` 의 `FnVCCB_InitBlock()` 내부.

#### 동작 요구사항

- 추론 엔진 초기화를 시도한다 (섹션 3의 시나리오에 따라 내용이 다름).
- 초기화 실패 시 AI Client를 **Fallback 모드**로 설정하고, 에러 로그를 출력한 후 계속 진행한다. VCCB 기동에는 영향을 주지 않는다.
- 초기화 성공 시 AI Client 내부 상태를 **Active 모드**로 설정하고, 연속 실패 카운터, 통계 카운터 등을 초기화한다.
- 모듈 내부에는 싱글턴 방식의 전역 제어 구조체를 하나만 유지한다.

---

### 2-3. 추론 호출 설계

#### 호출 위치

Call Task의 메시지 처리 흐름 내에서, 섹션 1의 각 개입 포인트에서 직접 호출된다.

#### 스레드 안전 요구사항

VCCB의 Call Task는 `MAX_CALL_THREAD_NUM_VCCB` 개의 스레드가 독립적으로 동작한다. 각 스레드는 `ptCi->myThreadId` 로 고유 식별된다.

- **시나리오 A (정적 라이브러리)**: ONNX Runtime Session의 스레드 안전 여부를 확인해야 한다. 스레드 안전하지 않으면 스레드별 Session을 별도 생성하거나 mutex로 보호한다.
- **시나리오 B (직접 포팅)**: 가중치 배열은 읽기 전용이므로 스레드 안전하다. 중간 연산 버퍼만 스레드별로 분리하면 된다.

#### 블로킹 금지 원칙

Call Task는 블로킹이 허용되지 않는다. 추론 자체가 1ms 이내에 완료되어야 한다. 추론이 이 시간을 초과할 경우 Fallback 처리로 즉시 전환한다.

#### Fallback 동작

AI Client가 Fallback 상태이거나 추론 실패 시, 기존 `FnVCCB_GetRRCInactiveTimerValue()` 를 직접 호출하여 정적 타이머 값을 반환한다. VCCB 서비스에는 영향이 없다.

---

### 2-4. Feature 벡터 구성

#### 시퀀스 Feature (과거 이력, 10 x 2)

`ptCi->aiCtx.inactivityHistory` 의 순환 버퍼에서 최신 순으로 최대 10개의 이력을 읽는다. 이력이 10개 미만이면 앞 부분은 0으로 패딩한다. 각 타임스텝은 `[durationSec, resumeCause]` 2개 값으로 구성된다.

#### 정적 Feature (15개)

아래 Feature들의 실제 `ptCi` 필드명은 `VccbStruct.h` 를 Cursor에서 직접 열어 확인해야 한다.

| Feature 이름 | 의미 | 접근 방법 힌트 |
|---|---|---|
| `primary_5qi` | 주 DRB의 5QI 값 | `ptCi->drb[]` 순회, `drbQosIdxInCall` → `ptCi->qosFlow[]` → `VLIB_getFiveQi()` |
| `drb_count` | 활성 DRB 수 | `ptCi->drb[]` 순회, `usage == CI_use` 카운트 |
| `ue_ambr_dl_mbps` | UE DL AMBR (Mbps) | `VccbStruct.h` 에서 `ambr`, `ueAmbr`, `ambrDl` 검색 |
| `has_gbr_flow` | GBR Flow 존재 여부 | `ptCi->qosFlow[]` 의 `gbrQosInfo` 또는 유사 필드 |
| `cn_expected_idle_period` | AMF 제공 예상 유휴 시간 | `ptCi->ueInfo` 의 `coreNetAssistInfo` 계열 필드 (섹션 4 참조) |
| `cn_stationary` | UE Stationary 여부 | 동상 |
| `last_rsrp` | 최근 RSRP 측정값 | `ptCi->qInfo` 또는 측정 저장 구조체 |
| `neighbor_cell_count` | 이웃 셀 수 | MeasurementReport 저장 구조체 |
| `rna_cell_count` | RNA 내 셀 수 | `ptCi->ueInfo.RrcInactiveInfo.numInactTAI` (확인됨) |
| `avg_cell_stay_sec` | 평균 셀 체류 시간 | `ptCi->aiCtx.avgInactiveDurationSec` |
| `resume_count_24h` | 24시간 내 Resume 횟수 | `ptCi->aiCtx.resumeCount24h` |
| `resume_success_rate` | Resume 성공률 | `ptCi->aiCtx.resumeSuccessRate` |
| `avg_inactive_duration` | 평균 INACTIVE 지속 시간 | `ptCi->aiCtx.avgInactiveDurationSec` |
| `hour_of_day` | 현재 시각 (0~23) | `time(NULL)` → `localtime()` → `tm_hour` |
| `day_of_week` | 요일 (0=일, 6=토) | `time(NULL)` → `localtime()` → `tm_wday` |

---

### 2-5. UE Context AI 이력 관리

#### 초기화

`VLIB_InitCallInfo()` 호출 시점에 `ptCi->aiCtx` 전체를 `memset` 으로 초기화하고, `resumeSuccessRate` 만 초기값 1.0f로 설정한다.

#### 이력 갱신 (순환 버퍼)

크기 10의 순환 버퍼에 이력을 추가하며, `writeIdx = (writeIdx + 1) % 10` 으로 갱신한다.

#### 통계 갱신

평균 INACTIVE 지속 시간을 **지수 이동 평균(α = 0.1)** 으로 업데이트한다.

- 첫 번째 기록: 초기값으로 그대로 사용.
- 이후: `avg = 0.9 × avg + 0.1 × newDuration`.

---

### 2-6. 상태 출력 (디버그용)

`itscp_vccb` 디버그 커맨드에서 출력할 항목은 다음과 같다.

- 현재 AI Client 상태 (Active / Fallback / Init)
- 사용 중인 추론 엔진 종류 (시나리오 A 또는 B)
- 연속 실패 횟수
- 총 요청 수, 성공 수, 실패 수, Fallback 수
- 트리거별 요청 수 (포인트①②③④)
- 평균 추론 소요 시간 (us)

---

## 3. 추론 엔진 구현 시나리오

AI 추론 엔진은 **빌드 환경의 정적 라이브러리 링크 가능 여부**에 따라 두 가지 방식 중 하나를 선택한다. 두 방식 모두 `VccbAiEngine.h` / `VccbAiEngine.cpp` 라는 동일한 인터페이스로 추상화하여, `VccbAiClient.cpp` 가 어느 방식이든 동일하게 호출할 수 있도록 설계한다.

#### 시나리오 선택 방법

섹션 5-1의 정적 라이브러리 링크 테스트를 먼저 수행한 후 결과에 따라 결정한다.

```
정적 라이브러리 링크 가능  →  시나리오 A 선택
정적 라이브러리 링크 불가  →  시나리오 B 선택
```

---

### 시나리오 A — 정적 라이브러리 사용 가능한 경우

#### 개요

```
Python (PyTorch 학습)
  → ONNX Export
  → rrc_predictor.onnx 생성
  → ONNX Runtime ARM 크로스 컴파일 정적 라이브러리 확보
  → vccb.xcc 빌드 시 정적 링크
  → vccb.xcc 내부에서 ONNX Session으로 직접 추론
```

#### 추론 엔진 초기화

- `FnVCCB_InitBlock()` 호출 시 ONNX Runtime Environment 및 Session 객체를 생성한다.
- 모델 파일 경로는 설정 파일 또는 컴파일 타임 상수로 지정한다.
- Session 생성 옵션: 단일 스레드 추론 (`intra_op_num_threads = 1`), 로그 레벨 최소화.
- 초기화 실패(모델 파일 없음, 포맷 불일치 등) 시 Fallback 모드로 전환하고 VCCB 기동을 계속한다.

#### 입출력 텐서 구조

| 텐서 이름 | Shape | 타입 | 설명 |
|---|---|---|---|
| `seq` | `[1, 10, 2]` | float32 | 시퀀스 Feature (duration, cause) |
| `static` | `[1, 15]` | float32 | 정적 Feature |
| `resume_prob` | `[1, 1]` | float32 | 출력: Resume 확률 (0.0~1.0) |
| `resume_time` | `[1, 1]` | float32 | 출력: 예상 Resume 시간 (초) |

#### 추론 후 타이머 계산

```
resume_prob >= VCCB_AI_KEEP_CONTEXT_THRESHOLD (기본 0.6):
  → recommendedTimer = resume_time × VCCB_AI_TIMER_MARGIN_FACTOR (기본 1.2), 올림

resume_prob <= VCCB_AI_EARLY_RELEASE_THRESHOLD (기본 0.3):
  → recommendedTimer = VCCB_AI_TIMER_MIN_SEC (기본 1초)

중간 범위:
  → recommendedTimer = 설정 기반 정적 타이머 (Fallback 값)

공통 후처리:
  → recommendedTimer = clamp(result, VCCB_AI_TIMER_MIN_SEC, VCCB_AI_TIMER_MAX_SEC)
```

#### 스레드 안전 처리

ONNX Runtime Session의 `Run()` 메서드가 스레드 안전한지를 사용하는 버전 문서에서 확인한다.

- **스레드 안전한 경우**: Session 하나를 공유하여 사용.
- **스레드 안전하지 않은 경우**: `MAX_CALL_THREAD_NUM_VCCB` 개의 Session을 초기화 시점에 미리 생성하여 `myThreadId` 를 인덱스로 사용.

#### 모델 갱신

재학습된 모델을 반영하려면 새 `.onnx` 파일로 교체 후 VCCB를 재기동해야 한다.

#### 빌드 시 추가 사항

```
Makefile에 추가 필요:
  CXXFLAGS에 ONNX Runtime include 경로 추가
    -I $(ONNX_RUNTIME_ROOT)/include

  LDFLAGS에 정적 라이브러리 추가
    -L $(ONNX_RUNTIME_ROOT)/lib -lonnxruntime
    또는
    $(ONNX_RUNTIME_ROOT)/lib/libonnxruntime.a 직접 명시

  vccb.xcc 소스 목록에 추가:
    VccbAiClient.cpp
    VccbAiEngine.cpp   ← ONNX Runtime 사용 구현체
```

---

### 시나리오 B — 정적 라이브러리 사용 불가능한 경우

#### 개요

```
Python (PyTorch 학습)
  → 학습 완료 후 가중치(weight, bias) 수치 추출
  → Python 스크립트로 C++ 헤더 파일 자동 생성 (VccbAiWeights.h)
  → C++에서 LSTM 순전파 연산 직접 구현
  → vccb.xcc에 포함 (외부 라이브러리 의존 없음)
```

#### LSTM 구조 제약

외부 라이브러리 없이 직접 구현하므로 모델을 최대한 단순하게 유지해야 한다.

| 항목 | 권장값 |
|---|---|
| LSTM hidden size | 16 또는 32 |
| LSTM 레이어 수 | 1 |
| 정적 Feature MLP | 1~2층 |
| 총 파라미터 수 | 5,000 이하 |
| 추론 소요 시간 목표 | < 0.5ms |

#### Python → C++ 가중치 변환 흐름

학습 완료 후 Python에서 아래 가중치들을 NumPy 배열로 추출하고, 자동 생성 스크립트로 C++ 헤더 파일(`VccbAiWeights.h`)로 변환한다.

```
추출 대상 가중치:
  LSTM:
    weight_ih (input-hidden), weight_hh (hidden-hidden)
    bias_ih, bias_hh
    (각각 4개 gate: input, forget, cell, output 순)

  Fully Connected (Static Feature 처리용):
    weight, bias

  Output Head:
    resume_prob용: weight, bias (+ sigmoid 적용)
    resume_time용: weight, bias (linear)
```

자동 생성된 `VccbAiWeights.h` 의 형태 (직접 수정 금지):

```cpp
// 자동 생성 파일 — Python 스크립트로만 수정할 것
// 생성 일시: YYYY-MM-DD, 모델 버전: vX.X
namespace VccbAiWeights {
    static const int   HIDDEN_SIZE = 16;
    static const float LSTM_WEIGHT_IH[4 * 16][12] = { ... };
    static const float LSTM_BIAS_IH[4 * 16] = { ... };
    // ... 이하 생략
}
```

#### C++ 직접 구현 대상 연산

| 연산 | 구현 방법 |
|---|---|
| 행렬 × 벡터 곱 | 이중 for 루프 |
| 벡터 덧셈 | 단순 for 루프 |
| sigmoid | `1.0f / (1.0f + expf(-x))` |
| tanh | `tanhf(x)` (C 표준 라이브러리) |
| LSTM 게이트 연산 | 위 연산의 조합 |

#### 중간 연산 버퍼 관리

LSTM 은닉 상태(`h_t`, `c_t`) 및 중간 연산 배열은 `ptCi->aiCtx` 내부에 보관한다. Call Task별로 독립된 `ptCi` 를 사용하므로 추가적인 동기화는 불필요하다.

#### 모델 갱신

재학습 후 가중치를 변경하려면 Python 변환 스크립트를 재실행하여 `VccbAiWeights.h` 를 새로 생성하고, VCCB를 재빌드 및 재배포해야 한다.

#### 빌드 시 추가 사항

외부 라이브러리 링크가 필요 없다.

```
Makefile에 추가 필요:
  vccb.xcc 소스 목록에 추가:
    VccbAiClient.cpp
    VccbAiEngine.cpp   ← C++ 직접 포팅 구현체
  (VccbAiWeights.h 는 헤더이므로 별도 추가 불필요)
```

---

## 4. callInfo_type 연동 영역

이 영역은 기존 VCCB 핵심 구조체에 AI Context 필드를 추가하고, 관련 초기화/해제 코드를 수정하는 작업이다.

---

### 4-1. VccbAiCallContext_type 필드 추가 위치

#### 대상 파일

`VccbStruct.h` 의 `callInfo_type` 구조체.

#### 추가 위치 결정 방법

Cursor에서 `VccbStruct.h` 를 열고 `callInfo_type` 구조체 내에서 다음 기준으로 삽입 위치를 결정한다.

- 구조체 크기 변화가 기존 멤버 정렬에 영향을 최소화하는 위치.
- 기능적으로 관련된 블록 (`ueInfo`, `CallUserInactivity` 등) 의 인근.
- 멤버 이름은 `aiCtx` 로 하며, 타입은 `VccbAiCallContext_type` (정의는 `VccbAiInfer.h` 에 있음).
- `VccbStruct.h` 상단에 `#include "VccbAiInfer.h"` 를 추가하거나, Forward declaration + 포인터 방식을 검토한다. 구조체 크기가 critical한 경우 포인터 방식(`VccbAiCallContext_type *ptAiCtx`) 을 사용하고 별도 메모리 풀에서 할당하는 방식도 고려한다.

---

### 4-2. VLIB_InitCallInfo() 수정

#### 파일

`FeatCallInfo.cpp` 또는 해당 함수가 구현된 파일. Cursor에서 다음으로 검색한다.

```
Codebase Search: "VLIB_InitCallInfo" 또는 "InitCallInfo"
```

#### 수정 내용

`VLIB_InitCallInfo()` 함수 내에서 기존 초기화 코드 이후에 `ptCi->aiCtx` 전체를 `memset(0)` 으로 초기화하는 코드를 추가한다. 이후 `resumeSuccessRate = 1.0f` 만 별도로 설정한다.

---

### 4-3. VLIB_FreeCallInfo() 수정 여부

#### 판단 기준

- `VccbAiCallContext_type` 가 값 타입(비포인터)으로 embedded된 경우: 별도 처리 불필요.
- 포인터 방식으로 구현한 경우: `VLIB_FreeCallInfo()` 에서 메모리를 반납하는 코드를 추가해야 한다.

---

### 4-4. ptCi 실제 필드명 확인 — Cursor 작업 가이드

아래 필드들은 `VccbStruct.h` 를 Cursor에서 열어 실제 이름을 확인한 후 Feature 벡터 구성 코드에 반영해야 한다.

#### 5QI 접근 방법 (확인됨)

```
ptDi->drbQosIdxInCall                            → qosFlow[] 인덱스
ptCi->qosFlow[drbQosIdxInCall].usage == CI_use   → 유효성 확인
VLIB_getFiveQi(&ptCi->qosFlow[idx], &qos5qi)     → uint8_t 5QI 획득
```

#### UE AMBR DL — 확인 필요

```
VccbStruct.h 내 Search: "ambr", "ueAmbr", "ueDlAmbr", "ambrDl"
단위(bps / Kbps / Mbps) 확인 후 Mbps로 변환
```

#### GBR Flow 여부 — 확인 필요

```
VccbStruct.h 내 Search: "gbrQosInfo", "gbrFlow", "bearerType", "isGbr"
```

#### CN Assistance Information (AMF 제공) — 확인 필요

```
VccbStruct.h 내 Search: "coreNetAssistInfo", "expectedIdlePeriod", "stationaryIndication"
NGAP InitialContextSetupRequest 처리 코드에서 ptCi->ueInfo 저장 위치 추적
```

#### RSRP 측정값 — 확인 필요

```
VccbStruct.h 내 Search: "rsrp", "measResult", "qInfo"
NRRRC_MeasurementReport 처리 함수에서 ptCi에 저장하는 위치 추적
```

#### Neighbor Cell Count — 확인 필요

```
VccbStruct.h 내 Search: "nbrCell", "neighborCell", "measResultNeighCell"
```

---

## 5. 빌드 및 환경 영역

이 영역은 신규 파일 추가와 환경 호환성 확인 작업이다. 코드 작성에 앞서 반드시 확인해야 한다.

---

### 5-1. 정적 라이브러리 링크 가능 여부 확인 (최우선)

시나리오 A/B 결정을 위한 가장 중요한 선행 작업이다.

#### 단계 1 — 툴체인 확인

```bash
# Makefile에서 크로스 컴파일러 변수 확인
grep -E "^CC|^CXX|^AR" Makefile | head -10
```

#### 단계 2 — 테스트용 정적 라이브러리 생성

```bash
# 테스트용 소스 작성
cat > libtest.cpp << 'EOF'
int testFunc() { return 42; }
EOF

# 반드시 Makefile의 CXX (크로스 컴파일러)로 컴파일
# x86용 g++로 컴파일하면 아키텍처 불일치로 실패
$(CXX) -c libtest.cpp -o libtest.o

# 정적 라이브러리 생성
ar rcs libtest.a libtest.o
```

#### 단계 3 — vccb.xcc 빌드에 링크 추가

생성된 `libtest.a` 를 Makefile의 링크 옵션에 추가하고 `vccb.xcc` 빌드를 시도한다.

```
빌드 성공  →  시나리오 A 선택
빌드 실패  →  시나리오 B 선택
```

---

### 5-2. 신규 파일 빌드 시스템 추가

#### Makefile 확인

```
Project Root Search: "Makefile", "CMakeLists.txt", "*.mk"
기존 FeatRrcInactive.cpp 가 등록된 방식을 확인하고 동일 패턴으로 신규 파일 추가
```

#### 공통 추가 대상

```
vccb.xcc 소스 목록에 추가:
  VccbAiClient.cpp
  VccbAiEngine.cpp

헤더 Include 경로에 추가:
  VccbAiInfer.h, VccbAiClient.h, VccbAiEngine.h 위치 디렉토리
```

시나리오 A의 경우 추가로 ONNX Runtime 라이브러리 링크 옵션을 추가한다.

---

### 5-3. Atomic 헤더 사용 가능 여부 확인

```bash
echo "#include <atomic>" | $(CXX) -std=c++11 -x c++ - -o /dev/null 2>&1
```

`<atomic>` 을 사용할 수 없는 경우 GCC 빌트인 함수(`__sync_fetch_and_add`, `__atomic_store_n`)로 대체하고, `VccbAiInfer.h` 의 atomic 타입 선언 부분을 `#ifdef` 로 분기 처리한다.

---

### 5-4. C++ 버전 확인

```bash
grep -r "std=c++" Makefile
```

최소 C++11 이상이어야 한다. C++11 미만이면 `<atomic>` 대신 POSIX mutex로 대체하고 lambda 표현식을 일반 함수로 교체한다.

---

### 5-5. 추론 시간 측정 방법

AI 추론이 Call Task 내에서 직접 수행되므로, 추론 소요 시간이 길면 해당 Call 스레드 전체가 지연된다. 반드시 추론 시간을 측정하고 목표(1ms 이내)를 충족하는지 검증해야 한다.

```
측정 방법:
  clock_gettime(CLOCK_MONOTONIC, &t_start)
  [추론 실행]
  clock_gettime(CLOCK_MONOTONIC, &t_end)
  latency_us = (t_end.tv_nsec - t_start.tv_nsec) / 1000

측정 결과를 AI Client 통계 구조체에 누적하고
2-6의 상태 출력 함수에서 평균값으로 확인
```

---

## 부록 A — 작업 우선순위 권장

| 순서 | 영역 | 작업 내용 | 이유 |
|---|---|---|---|
| 1 | 섹션 5-1 | 정적 라이브러리 링크 가능 여부 테스트 | **시나리오 A/B 방향 결정 — 이후 모든 작업의 전제** |
| 2 | 섹션 4 | `VccbStruct.h` 에서 미확정 필드명 전수 확인 | Feature 벡터 코드의 선행 조건 |
| 3 | 섹션 5 | C++ 버전, atomic 지원 여부 확인 | AI Client 코드 방향 결정 |
| 4 | 섹션 4 | `callInfo_type` 에 `aiCtx` 필드 추가 | 이후 모든 코드의 전제 |
| 5 | 섹션 3 | 선택된 시나리오에 따라 추론 엔진 구현 | 추론 호출의 기반 |
| 6 | 섹션 2 | AI Client 초기화 구현 | 섹션 3 완료 후 |
| 7 | 섹션 2 | Feature 벡터 구성 함수 구현 | 섹션 4 완료 후 |
| 8 | 섹션 1 | `VEIL_fillE1_Inactivity_Timer()` 개입 (포인트①) | 가장 단순한 개입 |
| 9 | 섹션 1 | `FnVCCB_IsAbleToProceedRRCInactive()` 개입 (포인트②) | 타이머 연장 포함 |
| 10 | 섹션 1 | `FnVCCB_Inact_ProcRRCResumeRequest1()` 개입 (포인트③) | 이력 갱신 |
| 11 | 섹션 1 | RNA Update 포인트④, XnAP 포인트⑤ | 낮은 우선순위 |

---

## 부록 B — Fallback 정책 요약

| 조건 | Fallback 동작 |
|---|---|
| AI Client Fallback 모드 | 기존 정적 타이머 값 그대로 사용 |
| 추론 엔진 초기화 실패 | AI Client Fallback 모드 설정, VCCB 기동 계속 |
| 추론 소요 시간 > 1ms | Fallback 응답 생성, 실패 카운터 증가 |
| 연속 10회 추론 실패 | AI Client Fallback 모드 전환 |
| 최대 연장 횟수 초과 | 즉시 INACTIVE 천이 강제, 카운터 리셋 |
| IS_REDCAP_UE 인 경우 | 초기 구현에서 AI 미적용, 기존 로직 유지 |
| IS_NRDC 케이스 | AI 미적용, 기존 로직 유지 |

---

## 부록 C — 주요 상수 정의 위치

모든 상수는 `VccbAiInfer.h` 에서 정의되며, 운영 환경에 맞게 조정 가능해야 한다.

| 상수 이름 | 의미 | 기본값 |
|---|---|---|
| `VCCB_AI_INFER_TIMEOUT_US` | 추론 타임아웃 (us) | 1000 |
| `VCCB_AI_TIMEOUT_FALLBACK_THRESHOLD` | Fallback 전환 연속 실패 횟수 | 10 |
| `VCCB_AI_MAX_EXTEND_COUNT` | 최대 타이머 연장 횟수 | 3 |
| `VCCB_AI_KEEP_CONTEXT_THRESHOLD` | INACTIVE 유지 판단 확률 임계값 | 0.6 |
| `VCCB_AI_EARLY_RELEASE_THRESHOLD` | 조기 해제 판단 확률 임계값 | 0.3 |
| `VCCB_AI_TIMER_MARGIN_FACTOR` | 타이머 여유 배율 | 1.2 |
| `VCCB_AI_TIMER_MIN_SEC` | 최소 타이머 값 (초) | 1 |
| `VCCB_AI_TIMER_MAX_SEC` | 최대 타이머 값 (초) | 3600 |
| `VCCB_AI_FALLBACK_TIMER_SEC` | Fallback 시 기본 타이머 (초) | 설정 기반 |
| `VCCB_AI_INACTIVITY_HISTORY_LEN` | UE별 이력 버퍼 크기 | 10 |

---

## 부록 D — 시나리오별 Python 작업 요약

두 시나리오 모두 Python으로 LSTM을 학습하지만, 모델 변환 단계가 다르다.

| 단계 | 시나리오 A | 시나리오 B |
|---|---|---|
| 학습 | PyTorch LSTM 학습 | PyTorch LSTM 학습 (hidden size 축소 권장) |
| 변환 | `torch.onnx.export()` 로 `.onnx` 파일 생성 | `state_dict()` 로 가중치 추출 → C++ 헤더 자동 생성 스크립트 실행 |
| 배포 | `.onnx` 파일을 CPP 보드 파일시스템에 배포 | 생성된 `VccbAiWeights.h` 로 VCCB 재빌드 |
| 모델 갱신 | `.onnx` 파일 교체 + VCCB 재기동 | `VccbAiWeights.h` 재생성 + VCCB 재빌드 / 재배포 |
