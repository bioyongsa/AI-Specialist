# VCCB AI 연동 상세 설계서

> **목적**: gNB-CU-CP VCCB SW에 LSTM 기반 RRC Inactivity Timer 최적화 AI를 연동하기 위한 Cursor IDE 작업 가이드
>
> **대상 독자**: Cursor IDE + Claude LLM 환경에서 코드 작성을 수행하는 개발자
>
> **코드베이스**: 삼성 5G gNB-CU-CP VCCB (C++, 사내 MDS/POSIX 기반)

---

## 목차

1. [메시지 처리 영역 — 기존 VCCB 코드 개입](#1-메시지-처리-영역--기존-vccb-코드-개입)
2. [AI 연동 영역 — 신규 개발](#2-ai-연동-영역--신규-개발)
3. [AI 데몬 영역 — 별도 프로세스 신규 개발](#3-ai-데몬-영역--별도-프로세스-신규-개발)
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

XnAP HandoverRequest 처리 핸들러. `StateHandleSa.cpp` 또는 관련 파일에서 `XN_HandoverRequest` 처리 함수를 Cursor에서 직접 검색한다.

**Cursor에서 수행할 검색 명령**:
```
Codebase Search: "XN_HandoverRequest" 또는 "RetrieveUEContextRequest"
```

#### 개입 설계

- 신규 UE Context가 초기화된 직후(`VLIB_InitCallInfo()` 호출 이후), XnAP 메시지에서 UE History 정보(소스 셀 체류 시간 등)를 읽어 AI Context의 이력 버퍼에 seed 데이터로 기록한다.
- XnAP UE History에서 획득 가능한 `Time UE Stayed in Cell` 값을 활용한다. 이 필드의 실제 접근 방법은 Cursor에서 코드베이스를 검색하여 확인해야 한다.

---

## 2. AI 연동 영역 — 신규 개발

이 영역은 VCCB 프로세스(`vccb.xcc`) 에 포함될 신규 AI Client 코드(`VccbAiClient.cpp`)를 작성하는 작업이다.

---

### 2-1. 신규 파일 구성

| 파일 | 역할 | 포함 대상 |
|---|---|---|
| `VccbAiInfer.h` | 구조체 및 상수 정의 (이미 작성됨) | 공유 메모리 레이아웃, IPC 구조체, enum 등 |
| `VccbAiClient.h` | AI Client 함수 선언 | VCCB 코드에서 호출할 모든 AI 인터페이스 |
| `VccbAiClient.cpp` | AI Client 함수 구현 | SHM attach, 추론 요청/응답, Fallback 처리 등 |

---

### 2-2. AI Client 초기화

#### 구현 위치

`MainInit.cpp` 의 `FnVCCB_InitBlock()` 내부.

#### 동작 요구사항

- POSIX `shm_open()` 으로 `/rrc_ai_shm` 경로의 SHM에 `O_RDWR` 모드로 attach를 시도한다.
- attach 실패 시 AI Client를 **Fallback 모드**로 설정하고, 에러 로그를 출력한 후 `ERROR`를 반환한다. VCCB 기동에는 영향을 주지 않는다.
- attach 성공 시 AI Client 내부 상태를 **Active 모드**로 설정하고, 연속 타임아웃 카운터, 통계 카운터 등을 초기화한다.
- 모듈 내부에는 싱글턴 방식의 전역 제어 구조체를 하나만 유지한다.

---

### 2-3. SHM 기반 추론 호출 (핵심 구현)

#### 스레드 안전 설계

VCCB의 Call Task는 `MAX_CALL_THREAD_NUM_VCCB` 개의 스레드가 독립적으로 동작한다. 각 스레드는 `ptCi->myThreadId` 로 고유 식별된다. SHM 내의 Request/Response Slot을 스레드 ID 기반으로 분리하여 **스레드 간 경합 없이** 동작해야 한다.

```
SHM Layout:
  reqRing[0]  ← myThreadId == 0 전용
  resRing[0]  ← myThreadId == 0 전용
  reqRing[1]  ← myThreadId == 1 전용
  resRing[1]  ← myThreadId == 1 전용
  ...
  reqRing[N]  ← myThreadId == N 전용
  resRing[N]
  daemonHeartbeat  (atomic uint32)
  daemonReady      (uint8)
```

#### 타임아웃 처리

- Call Task는 블로킹이 허용되지 않는다. 추론 요청 후 최대 `VCCB_AI_INFER_TIMEOUT_US` (기본값: 2000us = 2ms) 동안만 응답을 폴링(50us 간격)한다.
- 타임아웃 발생 시 Fallback 응답을 생성하고 연속 타임아웃 카운터를 증가시킨다.
- 연속 타임아웃 횟수가 `VCCB_AI_TIMEOUT_FALLBACK_THRESHOLD` (기본값: 10회) 이상이면 AI Client를 Fallback 모드로 전환한다.

#### Fallback 동작

AI Client가 Fallback 상태일 때 추론 요청 시, 기존 정적 타이머 값을 그대로 반환하거나 기존 `FnVCCB_GetRRCInactiveTimerValue()` 를 직접 호출한다. VCCB 서비스에는 영향이 없다.

#### Request ID 관리

- `atomic_uint32_t` 타입의 전역 카운터를 사용하여 단조 증가하는 Request ID를 생성한다.
- Response 구조체의 `requestId` 필드가 전송한 Request ID와 일치하고, `isValid` 플래그가 세팅되어 있을 때만 유효한 응답으로 처리한다.

---

### 2-4. Feature 벡터 구성

#### 개요

`ptCi` (callInfo_type) 의 실제 필드에서 Feature 값을 추출하여 `VccbAiFeatureVector_type` 구조체를 채운다.

#### 시퀀스 Feature (과거 이력, 10 x 2)

`ptCi->aiCtx.inactivityHistory` 의 순환 버퍼에서 최신 순으로 최대 10개의 이력을 읽는다. 이력이 10개 미만이면 앞 부분은 0으로 패딩한다. 각 타임스텝은 `[durationSec, resumeCause]` 2개 값으로 구성된다.

#### 정적 Feature (15개)

아래 Feature들의 실제 `ptCi` 필드명은 `VccbStruct.h` 를 Cursor에서 직접 열어 확인해야 한다.

| Feature 이름 | 의미 | 접근 방법 힌트 |
|---|---|---|
| `primary_5qi` | 주 DRB의 5QI 값 | `ptCi->drb[]` 순회, `CI_use` 인 DRB의 `drbQosIdxInCall` → `ptCi->qosFlow[]` → `VLIB_getFiveQi()` |
| `drb_count` | 활성 DRB 수 | `ptCi->drb[]` 순회, `usage == CI_use` 카운트 |
| `ue_ambr_dl_mbps` | UE DL AMBR (Mbps) | `VccbStruct.h` 에서 `ambr` 또는 `ueAmbr` 관련 필드 검색 |
| `has_gbr_flow` | GBR Flow 존재 여부 | `ptCi->qosFlow[]` 의 `gbrQosInfo` 또는 유사 필드 확인 |
| `cn_expected_idle_period` | AMF 제공 예상 유휴 시간 | `ptCi->ueInfo` 의 `coreNetAssistInfo` 계열 필드 확인 (별도 섹션 참조) |
| `cn_stationary` | UE Stationary 여부 | 동상 |
| `last_rsrp` | 최근 RSRP 측정값 | `ptCi->qInfo` 또는 측정 저장 구조체 확인 |
| `neighbor_cell_count` | 이웃 셀 수 | MR (Measurement Report) 저장 구조체 확인 |
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

VCCB의 `VLIB_InitCallInfo()` 호출 시점에 `ptCi->aiCtx` 전체를 `memset`으로 초기화하고, `resumeSuccessRate` 만 초기값 1.0f (100%)로 설정한다.

#### 이력 갱신 (순환 버퍼)

이력 구조체는 크기 10의 순환 버퍼로 구성된다. 이력 추가 시 `writeIdx` 위치에 기록하고, `writeIdx = (writeIdx + 1) % 10` 으로 갱신하며, `count` 를 최대 10까지 증가시킨다.

#### 통계 갱신

이력 추가 시 평균 INACTIVE 지속 시간을 **지수 이동 평균(α = 0.1)** 으로 업데이트한다.

- 첫 번째 기록: 초기값으로 그대로 사용.
- 이후: `avg = 0.9 × avg + 0.1 × newDuration`.

---

### 2-6. Heartbeat 기반 데몬 생존 확인

- SHM의 `daemonHeartbeat` 필드(atomic uint32)는 AI 데몬이 1초마다 증가시킨다.
- AI Client는 호출 시점에 이 값을 읽어 이전 호출 시 기록한 값과 비교한다. 값이 변경되지 않았으면 데몬 비정상으로 판단하고 Fallback 모드로 전환한다.
- 이 체크는 매 추론 요청마다 수행하지 않고, 일정 주기(예: 1초마다 또는 연속 타임아웃 발생 시)에만 수행하여 오버헤드를 최소화한다.

---

### 2-7. 상태 출력 (디버그용)

`itscp_vccb` 또는 동등한 VCCB 디버그 커맨드에서 AI Client 상태를 출력하는 함수를 구현한다. 출력 항목은 다음과 같다.

- 현재 AI Client 상태 (Active / Fallback / Init)
- SHM attach 주소
- 연속 타임아웃 횟수
- 총 요청 수, 성공 수, 타임아웃 수, Fallback 수
- 트리거별 요청 수 (포인트①②③④)
- 평균 추론 지연 시간 (us)

---

## 3. AI 데몬 영역 — 별도 프로세스 신규 개발

이 영역은 `vccb.xcc` 와 별도로 기동되는 `ai_infer.xcc` 프로세스를 작성하는 작업이다. 언어는 C++(VCCB와 동일 환경)이며 ONNX Runtime을 정적 또는 동적 링크하여 사용한다.

---

### 3-1. 프로세스 구조

```
ai_infer.xcc 프로세스
  │
  ├── 메인 스레드: 초기화 및 감시
  │     - SHM 생성 및 mmap
  │     - ONNX Runtime Session 생성 및 모델 로드
  │     - daemonReady = 1 설정
  │     - 종료 신호 처리 (SIGTERM, SIGINT)
  │
  ├── 추론 루프 스레드 (별도 pthread)
  │     - 모든 Thread Slot을 순서대로 폴링
  │     - Ready 요청 발견 시 즉시 추론 수행
  │     - 응답을 Response Slot에 기록
  │
  └── Heartbeat 스레드 (별도 pthread)
        - 1초마다 daemonHeartbeat atomic 카운터 증가
```

---

### 3-2. SHM 생성 및 관리

#### SHM 이름 및 크기

- 이름: `/rrc_ai_shm` (`VccbAiInfer.h` 의 `VCCB_AI_SHM_PATH` 상수 사용)
- 크기: `sizeof(VccbAiSharedMemory_type)`
- 접근 권한: `0666`

#### 생성 순서

1. `shm_open(O_CREAT | O_RDWR, 0666)` 로 SHM 생성 또는 오픈.
2. `ftruncate()` 로 크기 설정.
3. `mmap(PROT_READ | PROT_WRITE, MAP_SHARED)` 로 매핑.
4. 전체 영역을 `memset(0)` 으로 초기화.
5. `daemonReady = 0` 상태를 유지하다가 모델 로드 완료 후 `daemonReady = 1` 설정.

#### 종료 시 정리

- `munmap()` 으로 매핑 해제.
- `shm_unlink()` 로 SHM 삭제. (ai_infer.xcc가 SHM Owner이므로 종료 시 삭제 책임)

---

### 3-3. ONNX Runtime 초기화 및 추론

#### 환경 요구사항

- ONNX Runtime 버전: 해당 CPP 보드에서 지원하는 버전으로 결정 (일반적으로 1.16 이상 권장).
- 모델 경로: 설정 파일 또는 실행 인자로 주입 가능하도록 설계. 기본값은 `/opt/vccb/models/rrc_predictor.onnx`.
- Session 옵션: 단일 스레드 추론 (`intra_op_num_threads = 1`), 로그 레벨 최소화.

#### 입력/출력 텐서 구조

| 텐서 이름 | Shape | 타입 | 설명 |
|---|---|---|---|
| `seq` | `[1, 10, 2]` | float32 | 시퀀스 Feature (duration, cause) |
| `static` | `[1, 15]` | float32 | 정적 Feature |
| `resume_prob` | `[1, 1]` | float32 | 출력: Resume 확률 (0.0~1.0) |
| `resume_time` | `[1, 1]` | float32 | 출력: 예상 Resume 시간 (초) |

#### 추론 후 타이머 계산

```
resume_prob >= KEEP_CONTEXT_THRESHOLD (기본 0.6):
  → recommendedTimer = resume_time × TIMER_MARGIN_FACTOR (기본 1.2), 올림

resume_prob <= EARLY_RELEASE_THRESHOLD (기본 0.3):
  → recommendedTimer = TIMER_MIN_SEC (기본 1초)

중간:
  → recommendedTimer = FALLBACK_TIMER_SEC (설정값 기반 정적 타이머)

공통:
  → recommendedTimer = clamp(result, TIMER_MIN_SEC, TIMER_MAX_SEC)
```

---

### 3-4. 추론 루프 스레드 동작

```
while (daemonRunning)
{
    for (threadId = 0; threadId < MAX_CALL_THREAD_NUM_VCCB; threadId++)
    {
        reqSlot = shm->reqRing[threadId]
        resSlot = shm->resRing[threadId]

        if (reqSlot.isReady == 0) continue;
        if (resSlot.isValid == 1) continue;  // 미소비 응답 있으면 skip

        // 요청 복사 (atomic load)
        localReq = reqSlot
        reqSlot.isReady = 0  (atomic store)

        // 추론 수행
        sessionRun(localReq.features → output)

        // 응답 기록
        resSlot.requestId = localReq.requestId
        resSlot.resumeProb = output.resumeProb
        resSlot.resumeTimeSec = output.resumeTimeSec
        resSlot.recommendedTimer = calcTimer(output)
        resSlot.isValid = 1  (atomic store, 반드시 마지막에)
    }
    nanosleep(10us)  // CPU 과부하 방지
}
```

---

### 3-5. 모델 Hot-swap

#### 트리거

`SIGUSR1` 시그널 수신 시 모델 Hot-swap을 수행한다.

#### 동작 방식

1. 새 모델 파일 경로를 사전에 정해진 위치(예: `/opt/vccb/models/rrc_predictor_new.onnx`)에서 읽는다.
2. 새 ONNX Session을 생성하여 새 모델을 로드한다. 실패 시 기존 세션 유지.
3. 추론 루프 스레드가 현재 추론 중이 아닌 시점에 세션 포인터를 교체한다. 이때 `std::atomic` 또는 `std::mutex`를 사용하여 추론 중 교체로 인한 충돌을 방지한다.
4. 교체 완료 후 기존 세션을 해제한다.
5. 교체 완료 여부를 로그로 기록한다.

---

### 3-6. 기동 순서 및 장애 처리

#### CPP 보드 기동 순서

1. `ai_infer.xcc` 먼저 기동 → SHM 생성 → 모델 로드 → `daemonReady = 1`.
2. `vccb.xcc` 기동 → `FnVCCB_AiClientInit()` → SHM attach 시도.
3. VCCB가 먼저 기동되더라도 SHM attach 실패 → Fallback 모드로 서비스 지속.

#### 장애 시나리오

| 시나리오 | VCCB 처리 | ai_infer 처리 |
|---|---|---|
| ai_infer 미기동 | SHM attach 실패 → Fallback 모드 | 해당 없음 |
| ai_infer 크래시 | Heartbeat 미갱신 감지 → Fallback 모드 전환 | 재시작 스크립트에 의해 자동 재기동 |
| 추론 지연 > 2ms | 타임아웃 → Fallback 응답 | 처리 지속 |
| 연속 10회 타임아웃 | Fallback 모드 전환 | 무관 |
| ai_infer 재기동 | Heartbeat 정상화 감지 → Active 모드 복귀 | SHM 재생성 또는 기존 SHM 재사용 |

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

#### 파일

`FeatCallInfo.cpp` 또는 해당 함수가 구현된 파일.

#### 판단 기준

- `VccbAiCallContext_type` 가 값 타입(비포인터)으로 `callInfo_type` 에 embedded된 경우: `callInfo_type` 메모리 해제 시 자동으로 같이 해제되므로 별도 처리 불필요.
- `VccbAiCallContext_type *ptAiCtx` 포인터 방식으로 구현한 경우: `VLIB_FreeCallInfo()` 에서 `ptAiCtx` 가 가리키는 메모리를 반납하는 코드를 추가해야 한다.

---

### 4-4. ptCi 실제 필드명 확인 — Cursor 작업 가이드

아래 필드들은 `VccbStruct.h` 를 Cursor에서 열어 실제 이름을 확인한 후 `VccbAiClient.cpp` 의 Feature 벡터 구성 코드에 반영해야 한다.

#### 5QI 접근 방법 (확인됨)

```
ptDi->drbQosIdxInCall                            → qosFlow[] 인덱스
ptCi->qosFlow[drbQosIdxInCall].usage == CI_use   → 유효성 확인
VLIB_getFiveQi(&ptCi->qosFlow[idx], &qos5qi)     → uint8_t 5QI 획득
```

#### UE AMBR DL (Mbps) — 확인 필요

Cursor에서 다음을 검색한다.

```
VccbStruct.h 내 Search: "ambr", "ueAmbr", "ueDlAmbr", "ambrDl"
```

`ueInfo_type` 또는 `callInfo_type` 의 AMBR 관련 필드를 찾아 단위(bps인지 Kbps인지)를 확인하고 Mbps로 변환하는 코드를 작성한다.

#### GBR Flow 여부 — 확인 필요

Cursor에서 다음을 검색한다.

```
VccbStruct.h 내 Search: "gbrQosInfo", "gbrFlow", "bearerType", "isGbr"
qosFlowInfo_type 또는 callDRBInfo_type 내 GBR 관련 필드 확인
```

#### CN Assistance Information (AMF 제공) — 확인 필요

3GPP TS 23.501에 정의된 CN Assistance Information에는 `expectedUEBehaviourParameters` 가 포함되며, 이 중 `expectedIdlePeriod` 와 `stationaryIndication` 이 Feature로 활용 대상이다.

Cursor에서 다음을 검색한다.

```
VccbStruct.h 내 Search: "coreNetAssistInfo", "expectedIdlePeriod", "stationaryIndication", "CnAssist", "assistInfo"
```

NGAP Initial Context Setup Request 처리 코드 또는 UEContextModificationRequest 처리 코드에서 이 정보가 `ptCi->ueInfo` 의 어느 멤버에 저장되는지 추적한다.

#### RSRP 측정값 — 확인 필요

Cursor에서 다음을 검색한다.

```
VccbStruct.h 내 Search: "rsrp", "measResult", "qInfo"
NRRRC_MeasurementReport 처리 함수에서 ptCi에 RSRP를 저장하는 위치 추적
```

#### Neighbor Cell Count — 확인 필요

Cursor에서 다음을 검색한다.

```
VccbStruct.h 내 Search: "nbrCell", "neighborCell", "measResultNeighCell"
MeasurementReport 저장 구조체에서 이웃 셀 수를 세는 방법 확인
```

---

## 5. 빌드 및 환경 영역

이 영역은 신규 파일 추가와 환경 호환성 확인 작업이다. 코드 작성에 앞서 반드시 확인해야 한다.

---

### 5-1. 신규 파일 빌드 시스템 추가

#### Makefile 또는 CMake 확인

Cursor에서 다음을 검색하여 빌드 시스템 종류와 파일 추가 방법을 확인한다.

```
Project Root Search: "Makefile", "CMakeLists.txt", "*.mk", "build.sh"
```

#### 파일 추가 패턴

기존 소스 파일(예: `FeatRrcInactive.cpp`) 이 Makefile에 등록된 방식을 확인하고, 동일한 패턴으로 아래 파일들을 추가한다.

- VCCB 빌드 대상에 추가할 파일: `VccbAiClient.cpp`
- AI 데몬 별도 빌드 대상: `AiDaemonMain.cpp`, `AiDaemonInfer.cpp`

#### 헤더 파일 Include 경로

`VccbAiInfer.h` 와 `VccbAiClient.h` 가 위치할 디렉토리를 Include 경로에 추가한다. 기존 헤더 파일(`FeatRrcInactive.h` 등) 이 위치한 디렉토리와 동일하게 배치하는 것이 가장 간단하다.

#### ONNX Runtime 링크 (AI 데몬 빌드)

```
AI 데몬 빌드 시 추가 필요:
  -I /path/to/onnxruntime/include
  -L /path/to/onnxruntime/lib
  -lonnxruntime
```

ONNX Runtime 라이브러리가 CPP 보드에 설치된 경로를 먼저 확인해야 한다.

---

### 5-2. POSIX SHM 사용 가능 여부 확인

#### 확인 방법

CPP 보드의 OS(Linux 기반 임베디드 OS)에서 POSIX SHM이 지원되는지 확인한다.

```bash
# 보드에서 직접 실행
ls /dev/shm
# 또는
cat /proc/filesystems | grep tmpfs
```

#### 링크 옵션

POSIX SHM 함수(`shm_open`, `mmap` 등) 사용 시 `-lrt` 링크 옵션이 필요할 수 있다.

#### 대안 (POSIX SHM 미지원 시)

POSIX SHM이 지원되지 않는 경우, 사내 IPC 프레임워크(MDS 등) 의 공유 메모리 API를 대신 사용하는 방식을 검토해야 한다. 이 경우 SHM 관련 코드를 추상화 레이어로 감싸서 구현체를 교체할 수 있도록 설계한다.

---

### 5-3. Atomic 헤더 사용 가능 여부 확인

#### 확인 방법

C++11의 `<atomic>` 헤더 사용 가능 여부를 확인한다.

```bash
# 컴파일러 버전 확인
g++ --version

# C++11 지원 확인
echo "#include <atomic>" | g++ -std=c++11 -x c++ - -o /dev/null 2>&1
```

#### 대안

`<atomic>` 을 사용할 수 없는 경우, GCC 빌트인 함수(`__sync_fetch_and_add`, `__atomic_store_n` 등)를 사용한다. `VccbAiInfer.h` 의 atomic 타입 선언 부분을 `#ifdef` 로 분기 처리한다.

---

### 5-4. C++ 버전 확인

#### 확인 방법

```bash
g++ --version
# 또는 기존 Makefile에서 -std=c++XX 옵션 확인
grep -r "std=c++" Makefile CMakeLists.txt
```

#### 요구사항

- 최소: C++11 (`<atomic>`, `<thread>`, range-based for 등 사용)
- 권장: C++14 이상

#### 대응

C++11 미만인 경우, `<atomic>` 대신 POSIX mutex + 일반 변수로 대체하고, lambda 표현식을 일반 함수로 교체한다.

---

### 5-5. 보드 CPU 코어 수 및 Affinity 확인

VCCB는 `MAX_CALL_THREAD_NUM_VCCB` 개의 Call 스레드를 운용하며, 코어별 Affinity를 설정한다 (`MainTask.cpp` 의 `FnVCCB_SetCallTaskAffinity()`, `FnVCCB_SetTaskAffinityForVNF()` 참조).

AI 데몬 프로세스의 추론 루프 스레드가 Call 스레드와 동일 코어를 경합하지 않도록 CPU Affinity를 설정하는 것이 권장된다.

```
Cursor에서 확인:
  - CORE_NUM_VCCB 매크로 정의 위치
  - MAX_CALL_THREAD_NUM_VCCB 값
  - CPU Affinity 설정 패턴 (sched_setaffinity 사용법)
```

---

## 부록 A — 작업 우선순위 권장

| 순서 | 영역 | 작업 내용 | 이유 |
|---|---|---|---|
| 1 | 섹션 4 | `VccbStruct.h` 에서 미확정 필드명 전수 확인 | 모든 Feature 벡터 코드의 선행 조건 |
| 2 | 섹션 5 | C++ 버전, SHM, atomic 지원 여부 확인 | AI Client 코드 방향 결정 |
| 3 | 섹션 4 | `callInfo_type` 에 `aiCtx` 필드 추가 | 이후 모든 코드의 전제 |
| 4 | 섹션 2 | AI Client 초기화 및 SHM attach 구현 | 추론 호출의 기반 |
| 5 | 섹션 2 | Feature 벡터 구성 함수 구현 | 섹션 4 완료 후 |
| 6 | 섹션 2 | SHM 기반 추론 호출 구현 | 섹션 2-4 완료 후 |
| 7 | 섹션 3 | AI 데몬 기본 구조 구현 (SHM 생성, 추론 루프) | E2E 연동 테스트 위해 |
| 8 | 섹션 1 | `VEIL_fillE1_Inactivity_Timer()` 개입 (포인트①) | 가장 단순한 개입 |
| 9 | 섹션 1 | `FnVCCB_IsAbleToProceedRRCInactive()` 개입 (포인트②) | 타이머 연장 포함 |
| 10 | 섹션 1 | `FnVCCB_Inact_ProcRRCResumeRequest1()` 개입 (포인트③) | 이력 갱신 |
| 11 | 섹션 3 | AI 데몬 Hot-swap, Heartbeat 구현 | 운영 안정성 |
| 12 | 섹션 1 | RNA Update 포인트④, XnAP 포인트⑤ | 낮은 우선순위 |

---

## 부록 B — Fallback 정책 요약

AI 연동 코드의 모든 개입 지점에서 다음 Fallback 정책을 일관되게 적용해야 한다.

| 조건 | Fallback 동작 |
|---|---|
| AI Client Fallback 모드 | 기존 정적 타이머 값 그대로 사용 |
| SHM attach 실패 | AI Client Fallback 모드 설정, VCCB 기동 계속 |
| 추론 타임아웃 | Fallback 응답 생성, 타임아웃 카운터 증가 |
| 연속 10회 타임아웃 | AI Client Fallback 모드 전환 |
| 최대 연장 횟수 초과 | 즉시 INACTIVE 천이 강제, 카운터 리셋 |
| IS_REDCAP_UE 인 경우 | 초기 구현에서 AI 미적용, 기존 로직 유지 |
| IS_NRDC 케이스 | AI 미적용, 기존 로직 유지 |

---

## 부록 C — 주요 상수 정의 위치

모든 상수는 `VccbAiInfer.h` 에서 정의되어 있으며, 실제 값은 운영 환경에 맞게 조정 가능해야 한다.

| 상수 이름 | 의미 | 기본값 |
|---|---|---|
| `VCCB_AI_SHM_PATH` | POSIX SHM 이름 | `/rrc_ai_shm` |
| `VCCB_AI_INFER_TIMEOUT_US` | 추론 타임아웃 (us) | 2000 |
| `VCCB_AI_TIMEOUT_FALLBACK_THRESHOLD` | Fallback 전환 연속 타임아웃 횟수 | 10 |
| `VCCB_AI_MAX_EXTEND_COUNT` | 최대 타이머 연장 횟수 | 3 |
| `VCCB_AI_KEEP_CONTEXT_THRESHOLD` | INACTIVE 유지 판단 확률 임계값 | 0.6 |
| `VCCB_AI_EARLY_RELEASE_THRESHOLD` | 조기 해제 판단 확률 임계값 | 0.3 |
| `VCCB_AI_TIMER_MARGIN_FACTOR` | 타이머 여유 배율 | 1.2 |
| `VCCB_AI_TIMER_MIN_SEC` | 최소 타이머 값 (초) | 1 |
| `VCCB_AI_TIMER_MAX_SEC` | 최대 타이머 값 (초) | 3600 |
| `VCCB_AI_FALLBACK_TIMER_SEC` | Fallback 시 기본 타이머 (초) | 설정 기반 |
| `VCCB_AI_INACTIVITY_HISTORY_LEN` | UE별 이력 버퍼 크기 | 10 |
