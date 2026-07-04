# VCCB RRC State Predictor — 검증 절차서 (Verification Runbook)

이 문서는 VCCB에 통합된 RRC State Transition Predictor(GRU 기반 AI)를 **처음부터 끝까지 직접 확인**하기 위한 절차서다.
대상 독자는 미래의 나(sangku) — 이 세션에서 어떤 작업을 했는지 잊었을 때 이 문서만 보고 재현할 수 있어야 한다.

다루는 범위:

- **Part A** — VCCB 컴파일 (AI 코드 포함/미포함 빌드)
- **Part B** — Dataset 생성 + 초기 학습 + 가중치 반영
- **Part C** — 가상 Call을 인가해서 InactivityTimer 추론값을 실제로 확인
- **Part D** — Fine-tuning 수행 (프로덕션 경로 + 회차별 개선 증명 하니스)
- **Part E** — 이번 세션에서 수정한 4개 항목(A-1/B-1/B-2/B-3) 검증
- **Part F** — 트러블슈팅 / 산출물 경로 총정리
- **Part G** — 2026-07-04 실측 데이터 프로덕션 재학습 + 자주 묻는 질문(FAQ)

---

## 0. 사전 준비 — 환경과 경로

| 구분          | 위치                                                      | 용도                                               |
| ----------- | ------------------------------------------------------- | ------------------------------------------------ |
| p4 워크스페이스   | `/p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB` | 소스 루트 (이 문서의 `$VCCB`)                            |
| gtest 빌드/실행 | **docker 안** (크로스컴파일 toolchain)                         | C++ 컴파일 · gtest 실행                               |
| ML 파이프라인    | **docker 밖, VM 호스트의 Python venv**                       | `preprocess.py`/`train.py`/`export_weights.py` 등 |
| 마운트         | p4 워크스페이스가 docker에 volume mount                         | docker에서 만든 파일이 호스트에도 그대로 보임 → 파일 복사 불필요         |

```bash
export VCCB=/p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB
```

venv가 없다면 최초 1회만 (docker **밖**, 호스트에서):

```bash
cd $VCCB/ML
python3 -m venv venve venv/bin/activate

## VCCB AI fine-tuning 파이프라인(학습, 평가, 가중치 변환 등)을 위해
## python 가상환경을 활성화 함
source venv/bin/activate
```

---

## Part A — VCCB 컴파일

### A-1. 데이터 수집 없이 일반 빌드 (평상시 / 추론 검증용)

```bash
cd $VCCB/Test/ut
make clean
make -sj7 RT_OPTION=test_SaRrcInactive
```

- AI 추론 코드(`VccbAiClient.cpp`, `VccbAiEngine.cpp`, `VccbAiPersistStore.cpp`)는 항상 빌드에 포함된다 (와일드카드로 `VCCB_MAIN/src/*.cpp` 전체를 잡음).
- `-DVCCB_AI_DATA_COLLECT`가 없으므로 CSV 로거·sweep 테스트는 컴파일조차 안 됨 (운영과 동일 조건).

### A-2. AI 유닛 테스트만 별도로 빌드 (Point 1~5 로직 검증)

```bash
make -sj7 RT_OPTION=test_SaRrcInactiveAI
```

### A-3. 학습 데이터 수집용 빌드 (Method B, §Part D에서 사용)

```bash
export VCCB_AI_DATA_FILE=$VCCB/ML/vccb_ai_dataset.csv
rm -f "$VCCB_AI_DATA_FILE"     # 깨끗한 셋을 구성하기 위해서 기존의 것 삭제 (append 모드이므로)
make clean; make -sj7 RT_OPTION=test_SaRrcInactive VCCB_AI_DATA_COLLECT=1
```

### 빌드 성공 체크리스트

- [ ] 컴파일 에러 없음 (`SetEnabled was not declared` 발생 시 → `VccbAiClient.h` 최신본 미러링 확인)

- [ ] 링크 에러 없음 (`undefined symbol: FnVCCBAiClient_SetEnabled` 발생 시 → 아래 stale lib 삭제 후 재빌드)
  
  ```bash
  OBJ=$VCCB/OBJ/ACPF/i686
  rm -f $OBJ/VccbAiClient.o $OBJ/vccbx.a
  make -sj7 RT_OPTION=test_SaRrcInactive VCCB_AI_DATA_COLLECT=1
  # 로그에 "Compiling .../VccbAiClient.o" 가 다시 찍히는지 확인
  ```

- [ ] `ITSCP_VCCB`도 정상 빌드되는지 확인 (`No rule to make target 'VccbAiEngine.h'` 발생 시 → `ITSCP_VCCB/Makefile.body`의 `-I$(NR_CP_DIR)/VCCB/VCCB_MAIN/hdr` 확인)

---

## Part B — Dataset 생성 + 초기 학습 + 가중치 반영

### B-1. (선택) Method B 실측형 데이터 수집

Part A-3으로 빌드한 바이너리로 gtest 실행 → CSV에 실측 범위 데이터 누적:

```bash
cd $VCCB/Test/ut
make -sj7 RT_OPTION=test_SaRrcInactive VCCB_AI_DATA_COLLECT=1
# → $VCCB_AI_DATA_FILE 에 (a) 기존 TC 실제 흐름 + (b) AiClient_DataCollect_Sweep 데이터 누적
wc -l "$VCCB_AI_DATA_FILE"
```

### B-2. 전처리 → 학습 → 가중치 export (docker 밖, venv)

```bash
cd $VCCB/ML
source venv/bin/activate

python preprocess.py          # Method A(합성) + Method B(있으면 자동 병합) -> dataset.npz
python train.py                # dataset.npz -> model.pt
python export_weights.py       # model.pt -> $VCCB/VCCB_MAIN/hdr/VccbAiWeights.h
# 또는 한 번에:
bash run_all.sh
```

### B-3. 검증

| 확인 항목                                          | 정상 기준                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| `preprocess.py` 로그                             | `resume_prob=1 (...)%`가 한쪽으로 심하게 쏠리지 않음 (이상적 ~50%)                         |
| `train.py` 끝의 Phase-1 블록                       | `[verdict] beats naive=Y beats persistence=Y`                              |
| `train.py`/`export_weights.py` 로그              | `SCRIPT_VERSION`이 서로 짝이 맞는지 (`export_weights.py vX.Y.Z (train.py vX.Y.Z)`) |
| `export_weights.py`의 `Inference @feature-mean` | `resume_prob`가 1.000으로 고정되지 않음, `resume_time_raw`가 수십 초 수준 (sigmoid 포화 아님) |
| `head -6 $VCCB/VCCB_MAIN/hdr/VccbAiWeights.h`  | 파라미터 수 1266, 버전 스탬프 확인                                                     |

### B-4. 가중치 반영 재빌드 (추론 검증용 = 데이터 수집 토글 OFF)

```bash
cd $VCCB/Test/ut
make clean; make -sj7 RT_OPTION=test_SaRrcInactive
```

---

## Part C — 가상 Call을 인가해서 InactivityTimer 추론값 확인

여기가 "실제로 어떤 타이머 값이 나오는지"를 눈으로 보는 단계다. **두 가지 관측 경로**가 있다.

### C-1. 로그로 직접 보기 (가장 확실한 방법)

AI 유닛 테스트를 실행하면 각 개입 포인트가 남기는 정보성 로그가 그대로 콘솔(stdout)에 출력된다.

```bash
cd $VCCB/Test/ut
make -sj7 RT_OPTION=test_SaRrcInactiveAI 2>&1 | tee ai_test.log
```

찾아야 할 로그 패턴 (`VccbAiClient.cpp`에서 정의된 정확한 포맷):

| Point           | 로그 패턴                                                                     | 의미                                                            |
| --------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------- |
| ① 타이머 추천        | `[AI 1] resume_prob=0.XXX resume_time=YY.Ys -> timer=Zd (latency=...us)`  | 추론된 `resume_prob`/`resume_time`으로부터 최종 **timer=Z초**가 어떻게 나왔는지 |
| ② 연장/진행 판단 (연장) | `[AI 2] resume_prob=... >= 0.6 -> Defer INACTIVE. extendCount=N timer=Zs` | INACTIVE 진입을 미루고 timer=Z초로 재설정                                |
| ② 연장/진행 판단 (진행) | `[AI 2] resume_prob=... < 0.6 -> Proceed INACTIVE.`                       | 즉시 INACTIVE 전환                                                |
| ② 캡 도달          | `[AI 2] extendCount(3) >= MAX(3). Forced INACTIVE transition.`            | 무한 연장 방지 가드 발동                                                |

```bash
grep -E "\[AI 1\]|\[AI 2\]" ai_test.log
```

- `timer=` 뒤의 값이 **[1, 3600]** 범위인지 (클램프 정상 동작), 입력(5QI/AMBR/RSRP 등)을 바꾼 TC마다 값이 달라지는지 확인한다.
- `test_SaRrcInactiveAI.cpp`의 `Point1_active_inference_returns_valid_timer`, `Point2_active_decision_is_consistent_and_bounded` 등 테스트가 이 값을 assert하므로, **테스트 PASS 자체가 1차 검증**이다.

### C-2. gNB 프로세스 실행 중 실시간 조회 (itscp 명령)

빌드된 VCCB가 실제로 떠 있는 상태에서 itscp 콘솔로 접속:

```
get stat ai
```

(내부적으로 `get` → `stat` → `ai` 리프가 `GetStatAiReport()`를 호출 — `ITSCP_VCCB/src/main.cpp:2453` `get→stat`, `main.cpp:2360-2366` `stat→ai`)

출력에 포함되는 정보:

- 누적 추론 요청 수 / 성공 / 실패 / 타임아웃 / fallback 횟수
- Point 1~5 각각의 트리거 횟수 (`triggerPt1Count` ~ `triggerPt5Count`)
- extendCap 발동 횟수 (`extendCapHitCount`)
- 영속 저장소 통계 (restore hit/miss, saves, evictions)
- drift 히스토그램 (5QI/RSRP/prob 분포)

> 정확한 콘솔 진입 경로(예: telnet 포트, itscp 접속 방법)는 배포 환경마다 다르므로 운영 문서를 참고. 명령 트리 자체는 위와 같이 소스로 확인됨.

### C-3. 파라미터를 바꿔가며 sweep — "입력에 따라 값이 달라지는가"

`test_SaRrcInactive.cpp`의 `AiClient_DataCollect_Sweep`(§Part A-3 빌드로만 컴파일됨)이 5QI×DRB×AMBR×GBR×RSRP×RNA×cause 조합을 바꿔가며 `buildAiInput()`을 실제로 태운다. 이 테스트의 stdout에서 `[AI 1]`/`[AI 2]` 로그를 같은 방식으로 grep하면, **조합별로 timer 값이 달라지는지**를 직접 눈으로 확인할 수 있다.

---

## Part D — Fine-tuning 수행 절차

두 갈래로 나뉜다: **(D-1) 실제 production 경로**(gtest 실측 로그 → 재학습)와, **(D-2) 이번 세션에 만든 회차별 개선 증명 하니스**(`finetune_rounds.py`, 실측 데이터가 아직 없을 때 파이프라인 자체가 실제로 모델을 움직이는지 증명하는 도구).

### D-1. Production fine-tuning (실측 CSV가 쌓였을 때)

```bash
cd $VCCB/ML
source venv/bin/activate

# 1) 실측 CSV 병합 -> dataset.npz (기존 model.pt는 그대로 둠)
python preprocess.py

# 2) fine-tune (warm-start, 기존 model.pt 이어서 학습)
VCCB_AI_FINETUNE=1 python train.py
#   소량 데이터라 catastrophic forgetting이 걱정되면 head만 학습:
VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py

# 3) 가중치 재export + 재빌드
python export_weights.py
cd $VCCB/Test/ut && make clean; make -sj7 RT_OPTION=test_SaRrcInactive
```

| 옵션                            | 기본값           | 의미                                                        |
| ----------------------------- | ------------- | --------------------------------------------------------- |
| `VCCB_AI_FINETUNE`            | 0(OFF)        | 1이면 `model.pt`에서 warm-start (처음부터 학습 대신)                  |
| `VCCB_AI_FINETUNE_LR`         | 2e-4          | full training(1e-3)보다 낮은 LR — 급격한 망각 방지                   |
| `VCCB_AI_FINETUNE_EPOCHS`     | 15            | full training(50)보다 적은 epoch                              |
| `VCCB_AI_FREEZE_GRU`          | 0(OFF)        | 1이면 GRU+MLP(1200개 파라미터) 동결, head 2개(66개)만 학습 — 소량 데이터에 권장 |
| `VCCB_AI_EARLY_STOP_PATIENCE` | fine-tune 시 5 | val loss 정체 시 조기 종료                                       |

### D-2. 회차별 개선 증명 하니스 (`finetune_rounds.py`) — 실측 데이터 없이 파이프라인 검증

**위치**: `$VCCB/ML/finetune_rounds.py` (이번 세션에 신규 작성)

**무엇을 하는가**: Round-0(합성데이터만 학습한 베이스라인)을 만든 뒤, N회차에 걸쳐 "새로 들어온 실측형 배치"를 순차적으로 fine-tune하면서, **매 회차 고정된 held-out 평가셋**에 대해 MAE/AUROC/premature-release%/wasted-hold%가 어떻게 바뀌는지 그래프로 보여준다. `full_finetune`(전체 미세조정) vs `head_only_finetune`(GRU/MLP 동결) 두 전략을 동시에 비교한다.

```bash
cd $VCCB/ML
source venv/bin/activate

# 기본 실행 (5회차, dur_scale=1.6 shift)
python finetune_rounds.py

# 다른 shift 강도로 재실행 시 --tag로 기존 결과를 보존
python finetune_rounds.py --tag dur1.2 --dur-scale 1.2
```

**주요 옵션**
| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--rounds` | 5 | fine-tune 회차 수 |
| `--n-per-round` | 1500 | 회차별 새 배치 행 수 |
| `--n-eval` | 3000 | 고정 held-out 평가셋 행 수 (모든 회차·전략이 공통으로 사용) |
| `--dur-scale` | 1.6 | 실측형 데이터의 duration shift 배율 (1.0=shift 없음) |
| `--rsrp-bias` | 0.15 | 합성 prior가 모르는 새 RSRP→duration 결합 강도 |
| `--tag` | (없음) | 결과물 파일명 접미사 — 다른 파라미터로 재실행해도 이전 결과 보존 |
| `--show` | OFF | 실행 끝나면 PNG 3개를 OS 기본 뷰어로 자동 오픈 (desktop 세션에서만 의미 있음, headless docker/SSH에서는 그냥 무시됨) |
| `--no-checkpoints` | OFF | `.pt` 체크포인트 저장을 생략 (디스크 절약용) |
| `--out-dir` | `real/` (또는 `--tag` 지정 시 `real_<tag>/`) | CSV를 저장할 디렉토리 직접 지정 |

**같은 `--tag`(또는 tag 없이)로 재실행하면 이전 결과를 덮어쓴다** — 실행 시작 시 아래처럼 경고가 먼저 출력된다:

```
[WARN] 8 file(s) from a previous run will be OVERWRITTEN (same tag='...').
       Use --tag <name> to keep both runs side by side. Affected, e.g.:
         ...finetune_rounds_metrics_viztest.csv
         ...
```

자동으로 파일명에 번호가 붙지는 않으므로(디렉토리에 결과가 무한정 쌓이는 것을 막기 위한 설계), 결과를 보존하려면 실행마다 다른 `--tag`를 지정해야 한다 (예: `--tag 2026-07-01` 처럼 날짜로).

### D-2-부록. 터미널에서 PNG 없이 바로 결과 훑어보기

일반 터미널(bash/PowerShell/SSH/docker)은 PNG를 화면에 직접 그리지 못한다. 그래서 스크립트가 실행 끝에 **ASCII 스파크라인 + 자동 진단 문구**를 항상 터미널에 출력한다 (별도 명령 불필요, PNG 저장과 동시에 자동으로 나옴):

```
MAE (s), lower=better
  full_finetune        11.6 @=... 9.8   (round0 -> round5)
  head_only_finetune   11.6 @=... 9.9   (round0 -> round5)
...
[이번 실행 결과에 대한 자동 진단]
  [WARN] [full_finetune] premature-release가 25.7pp 줄었지만 wasted-hold가 25.7pp 늘었습니다. ...
```

- `[WARN]`/`[OK]`/`[FLAT]`/`[INFO]` 태그는 **실제 이번 실행의 수치**에 따라 동적으로 결정된다 (고정 문구가 아님) — premature-release/wasted-hold가 반대 방향으로 같이 움직이면 `[WARN]`, 한쪽만 개선되면 `[OK]`.
- 그림으로 직접 보고 싶으면 `--show` 옵션을 주면 desktop 세션에서 PNG 3개가 자동으로 열린다. headless(docker/SSH) 환경에서는 이 옵션이 조용히 무시된다 — PNG 파일 자체는 정상 저장되므로 나중에 다른 방법(scp, mount 등)으로 열어서 확인.
- `evaluate.py`도 동일한 방식으로 ASCII 막대(모델 vs naive/persistence/oracle MAE) + 해석 문구 + `--show`를 지원한다.

### D-2-부록. 파일 인코딩 (한글이 깨질 때)

`finetune_rounds_report*.txt` / `finetune_rounds_metrics*.csv` / `eval_report.txt`는 **UTF-8 with BOM**(`utf-8-sig`)으로 저장된다 — Windows 메모장 구버전이나 Windows PowerShell 5.1의 `Get-Content`가 BOM 없는 UTF-8을 시스템 코드페이지(cp949)로 오인해서 한글이 깨지는 것을 막기 위함이다. (`finetune_rounds_config*.json`은 JSON 스펙상 BOM을 넣지 않음 — `json.load()`로 재파싱할 때 문제 없도록.) 터미널 출력 자체도 UTF-8로 강제 설정되어 있어(`sys.stdout.reconfigure`), Windows 한국어 로케일 콘솔에서도 크래시 없이 한글이 정상 출력된다.

**생성되는 파일** (모두 `$VCCB/ML/` 아래, `--tag` 지정 시 파일명에 `_<tag>` 접미사):

```
real/eval_fixed.csv                 고정 평가셋 (학습에 절대 안 씀)
real/round_1.csv ~ round_N.csv       회차별 새 배치 (실제 aiDataCollectEmit() 스키마와 동일
                                     → VCCB_AI_DATA_DIR에 넣으면 production 파이프라인에도 재사용 가능)
model_round0_baseline.pt             Round-0 체크포인트
model_round{i}_{strategy}.pt         회차·전략별 체크포인트
finetune_rounds_config.json          이 실행에 사용된 모든 파라미터 기록
finetune_rounds_metrics.csv          회차별 전체 수치 표
finetune_rounds_report.txt           before(round0) -> after(마지막 회차) 요약
finetune_rounds_mae.png              MAE(모델/naive/persistence) vs 회차
finetune_rounds_classification.png   AUROC / F1 vs 회차
finetune_rounds_timer_kpi.png        premature-release% / wasted-hold% vs 회차
```

**결과 해석 시 주의** — 반드시 확인할 것:

- MAE만 보고 "좋아졌다"고 판단하지 말 것. `timer_kpi.png`의 **두 서브플롯을 함께** 봐야 한다 — premature-release가 줄었는데 wasted-hold가 그만큼 늘었다면 "정확도는 개선됐지만 편향 방향이 바뀐 것"이다.
- `full_finetune`과 `head_only_finetune`의 AUROC/F1 추이를 비교해서, 소량 데이터에서 head-only가 더 안정적인지(=catastrophic forgetting 회피) 확인.
- shift 강도(`--dur-scale`)를 낮췄을 때 트레이드오프 크기(pp 단위)가 같이 줄어드는지 — 줄어든다면 "베이스라인과 실측 분포 간 격차가 클수록 트레이드오프가 커진다"는 가설이 뒷받침됨.

**실측 사례 (2026-07-01 실행)** — 위 세 가지가 실제로 확인된 재현 가능한 결과:

| shift 강도                           | premature-release (전→후) | wasted-hold (전→후) | 트레이드오프 크기   |
| ---------------------------------- | ----------------------- | ----------------- | ----------- |
| `--dur-scale 1.6` (기본)             | 53.9% → 28.2%           | 46.1% → 71.8%     | **±25.7pp** |
| `--dur-scale 1.2` (`--tag dur1.2`) | 36.5% → 27.7%           | 63.5% → 72.3%     | **±8.8pp**  |

→ shift를 1.6에서 1.2로 낮추자 트레이드오프 폭이 약 1/3로 줄었다 — 가설이 뒷받침됨. 또한 두 실행 모두 `head_only_finetune`이 `full_finetune`보다 AUROC를 더 안정적으로(또는 더 높게) 유지했다 — 소량 데이터엔 head-only freeze를 권장하는 근거로 재사용 가능.

> ⚠️ 이 하니스는 **실측 라벨이 아직 없는 상태에서 fine-tune 배관(warm-start/freeze/early-stop/export) 자체가 올바르게 동작하는지**를 증명하는 도구다. 여기서 나온 개선폭은 "합성 shift 분포에 대한 개선"이며, 실제 필드 KPI가 아니다 (`gen_realish_csv.py` 헤더 참고).

---

## Part E — 이번 세션 수정 항목 검증 (A-1 / B-1 / B-2 / B-3)

| #       | 무엇을 고쳤나                                                                                 | 어디서 확인하나                                                                                                                                                                                                                                                                             |
| ------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A-1** | Point⑤ XnAP HO 시드가 store 실패 게이트보다 앞서 실행되던 순서 버그                                         | `StateSaHo.cpp`의 `VCCB_ProcMsgXnap_HandoverRequest` 내부에서 `FnVCCBAiClient_SeedFromXnapHistory(ptCi)` 호출이 `if (GOOD != status) return status;` **뒤에** 있는지 코드 확인                                                                                                                        |
| **B-1** | Wasted-hold KPI(premature-release와 대칭) 추가                                               | `python train.py` 또는 `python evaluate.py` 실행 후 출력의 `[timer] premature-release ... wasted-hold ...` 줄 확인. `finetune_rounds.py` 실행 시 `_timer_kpi.png`에도 반영됨                                                                                                                            |
| **B-2** | `inactiveEntryTs` 스탬프를 실제 INACTIVE 진입 지점(`FnVCCB_FinalizeRRCSuspend`)으로 이동 (과대계상 라벨 수정) | ①`VccbAiClient.cpp`의 Point①/②에 더는 `inactiveEntryTs` 대입이 없는지, ②`StateSaRrcInact.cpp`의 `FnVCCB_FinalizeRRCSuspend` 양쪽 케이스(CONNECTED→INACTIVE, 재suspend)에서 `FnVCCBAiClient_OnInactiveEntry(ptCi)` 호출 확인                                                                                 |
| **B-3** | 영속 저장소 키에 배포 secret salt 결합 (PII 보호)                                                    | `VccbAiPersistStore.cpp`의 `FnVCCBAiPersist_DeriveKey()`가 `fnv1a64(t->data, sizeof(t->data), g_persistSalt ^ (uint64_t)t->n)` 형태인지 확인. **운영 시**: `export VCCB_AI_PERSIST_SALT=<site-secret>` 설정 후 빌드 → 재기동해도 동일 salt를 넣어야 기존 영속 엔트리가 살아남음 (스모크 테스트: salt 변경 후 restore가 miss로 바뀌는지 확인) |

빠른 코드 확인 커맨드 (grep):

```bash
grep -n "inactiveEntryTs" $VCCB/VCCB_MAIN/src/VccbAiClient.cpp        # Point1/2에 대입 없어야 함
grep -n "OnInactiveEntry" $VCCB/VCCB_STATE/FEAT_SA/src/StateSaRrcInact.cpp  # 2곳(CONNECTED/재suspend) 나와야 함
grep -n "g_persistSalt\|VCCB_AI_PERSIST_SALT" $VCCB/VCCB_MAIN/src/VccbAiPersistStore.cpp $VCCB/VCCB_MAIN/hdr/VccbAiPersistStore.h
```

---

## Part F — 트러블슈팅 / 산출물 경로 총정리

### 자주 겪는 문제

| 증상                                                                              | 원인                                                  | 조치                                                                                                                                                                                         |
| ------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `resume_prob` 항상 1.000                                                          | 입력이 raw인데 정규화 없이 학습 (구버전)                           | `train.py`/`export_weights.py`가 정규화+fold를 지원하는 최신본인지 `SCRIPT_VERSION` 확인                                                                                                                   |
| `SetEnabled was not declared`                                                   | `VccbAiClient.h` 구버전                                | 최신 `.h` 미러링                                                                                                                                                                                |
| `undefined symbol: FnVCCBAiClient_...`                                          | `vccbx.a`가 stale                                    | Part A 체크리스트의 lib 삭제 후 재빌드                                                                                                                                                                 |
| `No rule to make target 'VccbAiEngine.h'` (ITSCP)                               | include 경로 누락                                       | `ITSCP_VCCB/Makefile.body`의 `-I.../VCCB_MAIN/hdr` 확인                                                                                                                                       |
| fine-tune 후 salt restore가 다 miss                                                | salt가 재기동 전후로 달라짐                                   | `VCCB_AI_PERSIST_SALT` 환경변수를 고정값으로 유지                                                                                                                                                      |
| `finetune_rounds.py`/`evaluate.py` 실행 시 `UnicodeEncodeError: 'cp949' codec ...` | Windows 한국어 로케일 콘솔 기본 코드페이지가 cp949라 이모지/블록문자를 못 인코딩 | 두 스크립트 모두 이미 수정됨(stdout UTF-8 강제 + ASCII 전용 시각화) — 최신본인지 확인. 다른 스크립트에서 재발하면 동일하게 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` 패턴 적용                                        |
| `.txt`/`.csv` 리포트를 메모장/엑셀로 열면 한글이 깨짐                                            | BOM 없는 UTF-8을 시스템 코드페이지로 오인                         | 이미 `utf-8-sig`(BOM)로 저장되게 수정됨 — 구버전 산출물이면 재실행해서 새로 생성                                                                                                                                      |
| `.../pyenv/shims/python3: ... Permission denied` (BuildServer, venv 활성화 전)      | `PYENV_ROOT`가 `/root/.pyenv`를 가리켜 sangku 권한으로 접근 불가 | **`sudo`로 우회하지 말 것** (산출물이 root 소유가 돼 이후 단계가 다시 막힘). `source venv/bin/activate`를 먼저 실행 후 재시도 — venv의 `python3`이 PATH 우선순위를 가져가 깨진 shim을 건너뜀. 이미 sudo로 만든 파일이 있으면 `chown sangku:sangku`로 복구 |

### 전체 산출물 경로 지도

```
$VCCB/
├── ML/
│   ├── preprocess.py / train.py / export_weights.py / evaluate.py / metrics.py
│   ├── gen_realish_csv.py            # 실측형 dry-run 데이터 생성기
│   ├── finetune_rounds.py            # ★ 이번 세션 신규: 회차별 개선 증명 하니스
│   ├── dataset.npz / model.pt        # production 파이프라인 산출물
│   ├── real/                         # finetune_rounds.py가 저장한 회차별 CSV
│   ├── model_round*_*.pt             # finetune_rounds.py 체크포인트
│   └── finetune_rounds_*.{csv,txt,png,json}
├── VCCB_MAIN/
│   ├── hdr/VccbAiClient.h / VccbAiEngine.h / VccbAiWeights.h / VccbAiPersistStore.h
│   └── src/VccbAiClient.cpp / VccbAiEngine.cpp / VccbAiPersistStore.cpp
├── VCCB_STATE/FEAT_SA/src/StateSaRrcInact.cpp   (Point③, B-2 스탬프 위치)
├── VCCB_STATE/FEAT_HO/src/StateSaHo.cpp          (Point⑤, A-1 수정 위치)
├── VCCB_FEAT/RRC_INACTIVE/src/FeatRrcInactive.cpp (Point②④)
└── Test/ut/FlowTC/JSON/src/test_SaRrcInactive.cpp / test_SaRrcInactiveAI.cpp
```

---

## 부록 — 실측 dry-run fine-tune 절차 (2026-06-12 BuildServer 실행 기록)

> ⚠️ **현재 기준 아님**: 아래 수치(5,000 UE 합성, 80,498행, 단일 shift.csv)는 2026-06-12 시점의 소규모 dry-run 기록이다. 실제 8개 real CSV(92,977행) 기반의 현재 프로덕션 재학습 결과는 **Part G**를 볼 것. 이 부록은 파이프라인 각 단계(synth_data_gen→gen_realish_csv→preprocess→train→export)를 저수준으로 손으로 돌리는 절차 자체를 이해하기 위한 참고 자료로만 유지한다.

Part D-2의 `finetune_rounds.py`(다회차 자동 비교 하니스)와는 별개로, **실제로 한 번 손으로 돌렸던 절차**를 명령어 순서 그대로 남긴다. `synth_data_gen.py` + `gen_realish_csv.py` + `train.py`/`export_weights.py`를 **개별적으로 직접 호출**하는 저수준(low-level) 절차이므로, `finetune_rounds.py`가 내부적으로 무엇을 자동화하고 있는지 이해할 때도 참고가 된다. (`torch.load`의 `FutureWarning`은 매 실행마다 반복 출력되므로 아래에서는 생략.)

### 1. 학습 데이터 재생성 (docker **외부**)

**1.1 기존 ut 생성 파일 삭제**

```bash
sudo rm -rf /p4_ws/sangku/26b_ai_vzw/Output/ut_results/
cd /p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB/ML
```

**1.2 학습데이터 재생성**

> ⚠️ **venv를 먼저 활성화할 것.** 활성화 전에 `python3`/`./run_all.sh`를 실행하면 셸의 `PATH`가 `~/.pyenv/shims/python3`를 먼저 잡을 수 있고, 이 shim이 (환경에 따라) `PYENV_ROOT`가 `/root/.pyenv`로 잘못 설정돼 있으면 `Permission denied`로 죽는다. `sudo`로 우회하면 산출물이 root 소유가 돼 이후 `sangku` 권한 단계에서 다시 막힌다 — 절대 `sudo python3`로 우회하지 말 것. venv를 먼저 활성화하면 `venv/bin`이 `PATH` 맨 앞에 와서 이 shim을 자연스럽게 건너뛴다.

```bash
source venv/bin/activate   # 반드시 synth_data_gen.py/run_all.sh 실행 전에!
which python3               # .../VCCB/ML/venv/bin/python3 이어야 정상

python3 ./synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0
# [synth_data_gen] wrote 80498 rows from 5000 virtual UEs -> vccb_ai_dataset.csv

./run_all.sh   # 산출물: dataset.npz, model.pt (baseline), VccbAiWeights.h
# ==============================
# ..... Preprocess, Train, Export Weights .....
# ==============================
#  Done!
#  VccbAiWeights.h updated.
#  Rebuild VCCB to apply.
# ==============================

cp -v model.pt model.pt.phase1   # 합성 14차원 복원 (fine-tune 전 baseline 보관)
```

### 2. dry-run CSV 생성 (docker **외부**)

```bash
mkdir -p real
python gen_realish_csv.py --out real/shift.csv --n 3000 --dur-scale 2.2   # 강한 shift CSV 생성
```

```
================================================================
gen_realish_csv — DRY-RUN real-ish dataset (NOT real UE data)
================================================================
  rows           : 3,000
  dur_scale      : 2.2  (1.0 = matched to prior)
  rsrp_bias      : 0.15
  lab_mode       : ON  (AMBR=100.0 Mbps, RNA=1 pinned; only 5QI/RSRP/time/idle vary)
  label mean     : 175.9s   resume<60s = 30.8%
  written        : .../VCCB/ML/real/shift.csv  (append=False)
----------------------------------------------------------------
Next (dry-run): preprocess.py -> train.py (note BEFORE Phase-1) ->
  VCCB_AI_FINETUNE=1 python train.py (AFTER Phase-1 should improve)
```

```bash
ls -lh real/shift.csv
head -1 real/shift.csv
python -c "import numpy; print(numpy.__version__)"
```

> ※ 여기서 **VccbAiWeights.h가 이미 갱신됐으므로 `p4 sync`** 시점 (뒤에서 재빌드할 다른 워크스페이스/VM이 최신 헤더를 받도록).

```bash
python3 preprocess.py   # Method A(합성 20,000) + B(vccb_ai_dataset.csv 80,498 + shift.csv 3,000) 병합
```

```
================================================================
VCCB RRC State Predictor — training data builder (Method A + B)
================================================================
[Method B]   + 80,498 rows  (.../VCCB/ML/vccb_ai_dataset.csv)
[Method B]   + 3,000 rows  (.../VCCB/ML/real/shift.csv)
[Method B] real CSV     : 83,498 unique samples from 2 file(s)
[Method A] synthetic    : 20,000 samples (3GPP 5QI rules)
----------------------------------------------------------------
Total samples : 103,498  (real 83,498 = 80.7% / synthetic 20,000)
  resume_prob=1 (resume < 60s): 51.8%
  resume_time   : mean=554.3s  std=939.1s
Saved → .../VCCB/ML/dataset.npz
  X_seq  (103498, 10, 2)  X_stat (103498, 14)
```

```bash
python3 evaluate.py   # BEFORE 기준선 기록 (아직 model.pt = phase1, fine-tune 전)
```

```
======================================================================
VCCB AI evaluation report  (evaluate.py v1.1.0)
======================================================================
  samples       : train=87,974 val=15,524 (val pos 8,013=51.6%)
  [class] AUROC=0.9518 PR-AUC=0.9311 ECE=0.0139  @0.6: P=0.911 R=0.924 F1=0.917
  [regr s] method                   MAE    RMSE     p90    bias
           model                  11.75   14.36   22.55   -0.02
           naive                  12.79   15.34   23.59    0.09
           persistence            21.77   73.23   36.58   11.29
  [verdict] beats naive=Y  beats persistence=Y
  [timer] early-release  model=32.6%  persistence=28.2%
======================================================================
report saved -> .../VCCB/ML/eval_report.txt
plots saved -> eval_pred_vs_true.png, eval_baseline_mae.png, eval_model_vs_oracle.png
```

> 이 실행의 `evaluate.py`는 `[timer] early-release`만 출력하는 구버전 — 현재는 B-1 개선으로 `premature-release`/`wasted-hold`가 함께 출력된다 (본문 §Part B-3, §Part D-2 참고).

### 3. fine-tuning 실행 + training

```bash
VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py   # AFTER (끝의 Phase-1 MAE를 2단계 evaluate.py와 비교)
```

```
[train.py v1.6.0]
  Total samples: 103,498
  [FINETUNE] warm start from .../VCCB/ML/model.pt
  [FINETUNE] lr=0.0002 epochs=15 freeze_gru=True
  Early stop: patience=5 (val loss)
Epoch      Train        Val   P_loss   T_loss
--------------------------------------------------
    1     0.2621     0.2556   0.2413   0.0286 *
    ...
   15     0.2614     0.2551   0.2408   0.0286 *

Best val loss : 0.2551
Model saved   → .../VCCB/ML/model.pt

── Phase-1 quality metrics (val) ──
  [class] AUROC=0.9523 PR-AUC=0.9323 ECE=0.0132  @0.6: P=0.914 R=0.919 F1=0.916
  [regr s] method                   MAE    RMSE     p90    bias
           model                  11.77   14.35   22.53    0.12
           naive                  12.79   15.34   23.59    0.09
           persistence            21.77   73.23   36.58   11.29
  [verdict] beats naive=Y  beats persistence=Y
```

### 4. 가중치 재 export

```bash
python3 export_weights.py
```

```
[export_weights.py v1.2.0]  (train.py v1.6.0)
  Normalization folded into GRU_ih / MLP / head_time (C++ stays raw)
  Total parameters : 1,266
  Inference @feature-mean : resume_prob=0.7957  resume_time_raw≈39.0s
  (If resume_prob is still ~1.000 across varied inputs, revisit data balance.)

Exported → .../VCCB/VCCB_MAIN/hdr/VccbAiWeights.h
Next: rebuild VCCB (no Makefile changes needed)
```

> ※ 여기서도 **VccbAiWeights.h가 갱신됐으므로 `p4 sync`** (docker 안에서 빌드할 워크스페이스에 반영).

### 5. VCCB 재빌드 (docker **내부**)

```bash
cd /p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/ITSCP_VCCB
make clean; make -sj7
```

### 6. ut 실행 (docker **내부**)

```bash
cd /p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB/Test/ut
make clean; make -sj7 RT_OPTION=test_SaRrcInactiveAI OUTPUT=jenkins
```

### 6.1 drift 검증용 배포-후 baseline 캡처 (배포 후, 운영 진입 시 1회)

```bash
python drift_monitor.py --capture-baseline --state /var/log/vccb_ai/drift_state.json
```

### 6.2 drift 검증 — 배포 후 주기 실행 (cron, 매일 1회 등)

```bash
python3 drift_monitor.py \
    --train-baseline /p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB/ML/drift_baseline.json \
    --state /var/log/vccb_ai/drift_state.json \
    >> /var/log/vccb_ai/drift.log 2>&1
```

**요약 흐름**

```
[docker 밖] synth_data_gen.py -> run_all.sh(baseline) -> model.pt.phase1 백업
      │
[docker 밖] gen_realish_csv.py(shift.csv) -> p4 sync(VccbAiWeights.h) -> preprocess.py(A+B 병합)
      │       -> evaluate.py(BEFORE) -> VCCB_AI_FINETUNE=1 [FREEZE_GRU=1] train.py(AFTER)
      │       -> export_weights.py -> p4 sync(VccbAiWeights.h)
      │
[docker 안] ITSCP_VCCB make -> VCCB/Test/ut make(RT_OPTION=test_SaRrcInactiveAI)
      │
[운영]      drift_monitor.py --capture-baseline (1회) -> cron으로 주기 실행
```

---

## Part G — 2026-07-04 실측 데이터 프로덕션 재학습 + FAQ

이번 세션은 `stat[4]`/`stat[7]` 신규 매핑(§Implementation_Status 6-1) 반영 후 **실제 필드에서 쌓인 8개 real CSV**로 진행한 최초의 프로덕션 재학습이다. Part D-2 부록의 2026-06-12 dry-run(단일 shift.csv, 5,000 UE 합성)과는 데이터 규모·출처가 다르므로 수치를 혼동하지 말 것 — **이 Part G가 현재 기준(current)**, D-2 부록은 파이프라인 동작 원리를 이해하기 위한 저수준 참고 자료로만 유지한다.

### G-1. 실행한 명령 그대로

```bash
cd $VCCB/ML
source venv/bin/activate

# 전처리: 실측 8개 CSV + 합성 20,000 병합
VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py

# 학습: Phase 0 Full Training (환경변수 없음 — GRU+Head 처음부터)
python train.py
```

### G-2. preprocess.py 출력 해석

```
[Method B]   + 79,477 rows  (./vccb_ai_dataset.csv)          # ut 실행 중 누적된 실측 로그
[Method B]   + 3,000 rows   (./real/eval_fixed.csv)          # 고정 held-out 평가셋 (학습에 안 씀 표기와 별개로 preprocess 병합 대상)
[Method B]   + 1,500 rows × 5  (./real/round_1~5.csv)        # finetune_rounds.py 등에서 누적된 회차별 배치
[Method B]   + 3,000 rows   (./real/shift.csv)               # dry-run 실측형 shift 데이터
[Method B] real CSV     : 92,977 unique samples from 8 file(s)
[Method A] synthetic    : 20,000 samples (3GPP 5QI rules)
Total samples : 112,977  (real 92,977 = 82.3% / synthetic 20,000)
  resume_prob=1 (resume < 60s): 52.0%      ← 40–60% 목표 범위 내 (라벨 균형 OK)
  resume_time   : mean=448.6s  std=791.9s
```

**Method A vs Method B**
| 구분 | 내용 | 목적 |
|---|---|---|
| **Method A (synthetic)** | 3GPP 5QI 규칙 기반 생성 데이터, 20,000행 | 5QI×RSRP×HO 조합의 systematic coverage 보장 |
| **Method B (real)** | 실측/현장 CSV 8개 병합, 92,977행 | 실제 UE 동작 분포 반영 |

**dataset.npz란**: NumPy 압축 아카이브(`.npy` 여러 개를 zip). 내용:
```
X_seq  (112977, 10, 2)   # 시계열: 샘플 × 과거10스텝 × [durationSec, resumeCause]
X_stat (112977, 14)      # 정적 특성: 샘플 × stat[0..13] (§DetailedDesign 2-4)
Y_prob (112977,)         # 라벨: resume_prob(dwell<60s 여부, 0/1)
Y_time (112977,)         # 라벨: resume_time(dwell 초)
```
샘플 1개 = `(X_seq[i], X_stat[i]) → (Y_prob[i], Y_time[i])` 한 쌍. `X_stat`의 14개 값은 `VCCB_AI_Integration_DetailedDesign.md` §2-4의 idx 0~13 표와 완전히 동일한 정의.

### G-3. train.py 출력 해석 — P_loss / T_loss

```
Epoch      Train        Val   P_loss   T_loss
    1     0.4141     0.2994   0.2829   0.0331 *
   10     0.2614     0.2580   0.2445   0.0269 *
```

| 컬럼 | 의미 | 낮을수록 |
|---|---|---|
| `P_loss` | Probability loss — `Y_prob` 예측 정확도(BCE류) | resume 확률 예측이 정교함 |
| `T_loss` | Time loss — `Y_time` 예측 정확도(MSE류) | dwell 시간 예측이 정교함 |
| `Train`/`Val` | `P_loss+T_loss` 가중합의 학습/검증셋 값 | 과적합 없이 수렴 중(Val≈Train 유지 확인) |
| `*` | 이번 epoch가 best validation → 체크포인트 갱신 | — |

### G-4. Phase 0 → Phase 1 → Phase 2 순서 (자주 헷갈리는 부분)

| Phase | 명령 | 시점 |
|---|---|---|
| **0. Full Training** | `python train.py` (환경변수 없음) | **stat[]/라벨 공식이 바뀐 직후 항상 여기부터.** GRU+Head를 처음부터 학습 |
| **1. Validate** | `eval_report.txt`의 `resume_prob=1` 비율(40–60%)과 Phase-1 floor(AUROC/F1/MAE, §DetailedDesign 6-9) 확인 | Phase 0 직후, 자동 게이트 없음 — **수동 판정** |
| **2. Fine-tune (선택)** | `VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py` | Phase 1 **통과 후**, 이후 소량의 신규 실측 배치만 추가로 반영하고 싶을 때만. GRU 동결 + head만 재학습(warm-start) |

**주의**: 계수(`svc_factor`/`mob_factor`) 조정처럼 **라벨 자체가 바뀌는 변경**은 Phase 2(fine-tune)가 아니라 **Phase 0부터 재실행**해야 한다 — warm-start는 이전 라벨 분포를 전제로 하므로 라벨이 바뀌면 부적절.

### G-5. 검증 체크리스트 (Phase 1)

1. `eval_report.txt`(또는 `preprocess.py`/`train.py` 콘솔) → `resume_prob=1` 비율 40–60%인지
2. `eval_report.txt`의 `[verdict] beats naive=Y beats persistence=Y` 확인
3. Phase-1 floor(§DetailedDesign 6-9): AUROC≥0.85, F1@0.6≥0.78, MAE≤15s
4. 실패 시 → `svc_factor(0.7)`/`mob_factor(0.05)`를 LOCKSTEP 3곳에서 동시 조정 후 Phase 0 재실행
5. 통과 시 → `export_weights.py` → VCCB→itscp_vccb→UT 순서 재빌드 → AI 매트릭스 테스트(`skip:1→0`) 활성화하여 tighten

### G-6. 매트릭스 테스트(16케이스)는 재학습에 관여하지 않음

`gen_ai_matrix_tc.py`가 생성한 16개 feature 민감도 케이스(`AI_matrix_*`, §DetailedDesign 6-10)는 **acceptance 학습 루프의 입력이 아니다.** 4개 base case(profile A/B 등, `skip:0`)만이 실제 학습/검증에 쓰인다. 매트릭스는 재학습된 모델이 "5QI/HO/RSRP를 바꾸면 timer가 의도한 방향으로 변하는가"를 사람이 확인하는 **참조/시연 세트**이며, 평소엔 `skip:1`로 비활성 상태를 유지하다가 탐색적 검증이 필요할 때만 수동으로 켠다.
