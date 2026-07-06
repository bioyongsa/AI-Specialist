# VCCB AI 파이프라인 실행 결과 해설서 (Running Procedure)

이 문서는 `VCCB_AI_Verification_Runbook.md` **Part H**(원샷 재학습 파이프라인)를 실제로 실행했을 때
터미널에 찍히는 **모든 출력 줄의 의미**와, 생성되는 **`*.png` 그림을 읽는 법**을 한 줄 한 줄 설명한다.

- 대상 독자: 미래의 나(sangku) — 로그 숫자를 보고 "이게 정상인가?"를 즉시 판단하기 위한 참조서.
- 짝 문서: 절차(무엇을 어떤 순서로) = `VCCB_AI_Verification_Runbook.md`, 결과 해석(그 출력이 뭘 뜻하는지) = **이 문서**.
- 예시 로그는 2026-07-06 BuildServer 실행분(재학습→UT 43/43 PASSED)을 그대로 인용한다.

---

## 0. 이 실행 한 판의 데이터 성격 (가장 먼저 알아야 할 것)

이번 실행에서 학습에 들어간 데이터는 **전부 시뮬레이션**이다. 진짜 gNB 실측 로그는 아직 없다.

**데이터 소스**

1. **`vccb_ai_dataset.csv`** — `synth_data_gen.py` 생성
   - preprocess: "Method B 'real CSV'" 분류
   - 실제: **시뮬레이션**(가상 UE 5000개)

2. **`real/shift.csv`** — `gen_realish_csv.py` 생성
   - preprocess: "Method B 'real CSV'" 분류
   - 실제: **시뮬레이션**(DRY-RUN, dur 분포만 이동)

3. **20,000 samples** — `preprocess.py` 내부 생성
   - preprocess: "Method A synthetic"
   - 실제: **3GPP 5QI 규칙 합성**

> ⚠️ **용어 함정**: preprocess.py가 "real 76.3%"라고 찍어도, 그건 "**CSV 파일에서 온 비율**"이라는 뜻이지
> "실측 필드 데이터"라는 뜻이 아니다. `VCCB_AI_DATA_FILE`/`VCCB_AI_DATA_DIR`로 읽힌 것은 전부 "Method B=real"로 집계된다.
> 지금 배포된 `VccbAiWeights.h`는 실질적으로 **Phase-0(합성) 모델**이다. (메모리 [[feedback-retrain-preprocess-snapshot]] 참조.)

---

## 1. 전체 파이프라인 한눈에 (H단계 ↔ 스크립트 ↔ 산출물)

```
H-0  환경변수/정리     export VCCB, ITSCP  +  rm 산출물 전체
H-1  base 학습        synth_data_gen.py → (rm real/shift.csv) → preprocess.py → train.py → cp model.pt.phase1
H-2  shift 생성       gen_realish_csv.py --out real/shift.csv --dur-scale 2.2
H-3  재-preprocess    preprocess.py  (이번엔 shift.csv 포함 → dataset.npz 갱신)
H-4  fine-tune+export VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 train.py → export_weights.py
H-5  VCCB 재빌드      (docker) ITSCP_VCCB make
H-5b bound 재보정     (재학습 후 AI_matrix_* 실패 시) gen_ai_matrix_tc.py
H-6  UT 실행          make RT_OPTION=test_SaRrcInactiveAI → UT_VCCB.i686
H-7  drift 검증       drift_monitor.py  (VCCB 운영 중일 때만 의미)
```

**산출 파일**

| 파일 | 단계 | 설명 |
|---|---|---|
| `vccb_ai_dataset.csv` | H-1 | 합성 학습 원본 CSV |
| `real/shift.csv` | H-2 | 분포 이동시킨 fine-tune용 CSV |
| `dataset.npz` | H-1, H-3 | 전처리 완료 텐서 (X_seq/X_stat/Y_prob/Y_time) |
| `model.pt` | H-1, H-4 | 학습 가중치 (PyTorch state_dict, 최신=H-4) |
| `model.pt.phase1` | H-1 | base 모델 백업 |
| `train_report_full.png` | H-1 | base 학습 4-패널 리포트 ★ |
| `train_report_finetune.png` | H-4 | fine-tune 4-패널 (before/after) ★ |
| `train_metrics_*.json` | H-1/H-4 | 지표 기계판독본 (JSON) |
| `outlier_report.png` | preprocess | outlier 필터 전/후 (drop>0일 때만) |
| `VccbAiWeights.h` | H-4 | C++ 임베드 가중치 (1,266 param) |

---

## 2. Line-by-line 해설

### 2-1. H-0 환경변수 & 정리

> 🖥️ **실행 명령어**
>
> ```bash
> export VCCB=/p4_ws/.../VCCB
> export ITSCP=/p4_ws/.../ITSCP_VCCB
> sudo rm -rf $VCCB/../../../Output/ut_results/     # 이전 UT 결과 XML 제거
> cd $VCCB/ML
> rm -f dataset.npz model.pt model.pt.phase1        # Group 1
> rm -f train_report_*.png train_metrics_*.json     # Group 1
> rm -f eval_*.png eval_report.txt                  # Group 2
> rm -f finetune_rounds_*.* model_round*_*.pt        # Group 3
> rm -f real/eval_fixed.csv real/round_*.csv real/shift.csv  # Group 4
> ```

- 4개 Group을 전부 지우는 이유: preprocess.py가 `real/` 아래 **모든 `*.csv`를 무조건 병합**한다.
  이전 세션 CSV가 남아 있으면 base 모델이 그 분포를 미리 학습해버려 H-3 fine-tune이 no-op이 된다. (메모리 [[project-part-h-pipeline-rules]])

### 2-2. H-1 ① synth_data_gen.py

> 🖥️ **실행 명령어**
>
> ```bash
> python3 ./synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0
> ```

| 옵션 | 이 실행에서 준 값 | 의미 (소스: `synth_data_gen.py` argparse) |
|---|---|---|
| `--out` | `vccb_ai_dataset.csv` | 생성한 CSV를 저장할 파일 경로(상대경로면 스크립트 실행 위치 기준). |
| `--ues` | `5000` | 시뮬레이션할 가상 UE 수. 기본값도 5000이라 사실 안 줘도 같은 결과지만, 명시적으로 적어서 의도를 분명히 한 것. |
| `--seed` | `0` | 재현성 시드(아래 출력 설명의 `seed=0` 참고). 기본값도 0. |
| `--profile` | (안 줌 → 기본값 `base`) | `base`(정상 인구 집단) / `field`(fine-tune용으로 뒤틀린 인구 집단) 중 선택. 이번엔 안 줬으므로 기본값 `base`가 적용됐다. |
| `--outlier-frac` | (안 줌 → 기본값 `0.0`) | 행마다 이상치를 주입할 확률. 안 줬으므로 0.0(깨끗한 데이터). |
| `--append` | (안 줌 → 기본값 꺼짐) | 켜면 기존 파일에 이어붙이고, 꺼져 있으면(기본) 새로 덮어쓴다. 이번엔 꺼져 있으므로 `vccb_ai_dataset.csv`가 매번 새로 만들어진다. |

**실행 결과:**

```
[synth_data_gen] wrote 64465 rows from 5000 virtual UEs -> vccb_ai_dataset.csv
[synth_data_gen] profile=base outlier_frac=0.000 seed=0
```

| 토큰 | 실제 동작 (소스: `synth_data_gen.py`) | 쉬운 설명 |
|---|---|---|
| `64465 rows` | 가상 UE 5,000명이 각각 여러 번 "RRC_INACTIVE로 있다가 resume"하는 사건을 시뮬레이션하며, 사건 하나마다 CSV 한 줄을 쓴다. 단 resumeCause가 "RNA Update"인 사건은 실제 C++ 코드(Point-3, `StateSaRrcInact.cpp`)가 애초에 처리를 건너뛰므로, 이 스크립트도 그 경우엔 **줄을 쓰지 않는다**(시간은 흘러가지만 행이 안 남음). | "행(row)" = CSV 한 줄 = "UE 하나가 RRC_INACTIVE로 있다가 resume한 사건" 하나. UE 1명이 페르소나에 따라 4~40번(아래 `sessions_range`) 반복 resume하므로, 5,000명이 64,465줄이 됐다 — 평균 UE당 약 12.9번 resume한 셈. |
| `5000 virtual UEs` | `--ues 5000`(기본값). 각 UE는 4종류 **페르소나(persona)** 중 하나를 뽑아 배정받는다: `iot_sensor`(45%, 거의 안 움직이는 IoT 센서, dwell 매우 길음), `bursty_phone`(35%, 스마트폰처럼 자주 깨어남, dwell 짧음), `fwa_fixed`(12%, 고정형 광대역, dwell 길고 트래픽 큼), `commuter`(8%, 이동하며 신호 약한 통근자). | "페르소나"는 그 UE의 "성격"을 정하는 틀이다 — 5QI 선호도, DRB 개수, AMBR(속도), RSRP(신호 세기), 얼마나 자주 resume하는지까지 전부 페르소나별로 다르게 미리 정해져 있어서, 실제 UE 인구 집단의 다양성을 흉내 낸다. |
| `profile=base` | `--profile base`(기본값). 위 4개 페르소나의 "정상판"(`PERSONAS`)을 쓴다. `--profile field`를 주면 대신 인위적으로 뒤틀린 `PERSONAS_FIELD`를 쓰는데, `bursty_phone` 비중을 35%→47%로 올리고 dwell을 더 짧게, `commuter`는 8%→15%로 올리고 RSRP를 더 약하게 설정해 둔 것(코드 주석: "새로 수집된 field 데이터를 흉내"). | `base` = 학습 시작점(§2-6에서 말한 "prior")을 만드는 "정상" 데이터. `field` = fine-tune 훈련용으로 "약간 다른 실측처럼 보이는" 데이터를 만들 때 쓰는 옵션. 이번 실행은 H-1에서 `base`만 썼고(fine-tune 역할은 `gen_realish_csv.py`가 대신 맡음), `field` 프로필은 이번엔 안 썼다. |
| `outlier_frac=0.000` | `--outlier-frac`(기본 0.0). 0보다 크게 주면, 찍히는 행마다 그 확률로 4가지 "실제 로거 버그"를 흉내 낸 값 오염을 주입한다: `label_spike`(라벨이 수천~86400초로 폭주, 40% 확률), `label_zero`(라벨=0초, 25%), `ambr_spike`(AMBR을 1000배로 부풀림, 20%), `avgdur_spike`(누적 평균 dwell을 이상값으로 교체, 15%). | 이건 §2-3의 `[outlier] mode=iqr` 필터가 **실제로 걸러내는지 시험**하기 위한 장치다. base 데이터는 처음부터 깨끗해야 하므로 0.0 — `field` 프로필로 fine-tune 세트를 만들 때만 보통 ~0.02 정도로 켠다. |
| `seed=0` | `--seed 0`(기본값). Python `random.Random(seed)`에 넣는 난수 시드(값). | 같은 seed + 같은 옵션이면 **항상 똑같은 CSV**가 나온다(재현 가능·결정적). "방금 그 이상한 결과가 우연인지 재현되는 버그인지" 확인할 때, 같은 seed로 다시 돌려서 똑같이 나오면 코드 문제, 다르게 나오면 난수 문제라는 걸 구분할 수 있다. |

#### ⚠️ (누락 주의) venv 활성화 — H-1①과 ② 사이 필수 단계

> 🖥️ **실행 명령어**
>
> ```bash
> source venv/bin/activate
> which python3
> ```

| 명령어 | 의미 |
|---|---|
| `source venv/bin/activate` | 이 디렉토리(`$VCCB/ML/venv`)에 미리 만들어 둔 **Python 가상환경(virtual environment)**을 켠다. 가상환경 = 시스템 전체에 영향 주지 않고 이 프로젝트 전용 패키지(numpy/pandas/torch 등)만 따로 설치해 둔 격리된 공간. | 
| `which python3` | 지금 `python3`/`python` 명령이 **어느 실행파일을 가리키는지** 확인한다. |

**실행 결과:**

```
(venv) sangku@BuildServer:ML $ which python3
/p4_ws/sangku/26b_ai_vzw/Application/NR_CALL_APP/VCCB/ML/venv/bin/python3
```

- 프롬프트 맨 앞에 `(venv)`가 붙은 것과, `which python3` 결과가 시스템 경로(`/usr/bin/python3`)가 아니라 **`.../ML/venv/bin/python3`**를 가리키는 것으로 활성화 성공을 확인한다.
- **왜 이 시점에 켜는가**: `synth_data_gen.py`는 파일 맨 위 주석에 "Dependencies: Python 3 standard library only"라고 적혀 있는 대로 표준 라이브러리만 쓰므로, 시스템 `python3`(venv 없이)로도 실행됐다. 하지만 **이 뒤로 나오는 `preprocess.py`/`train.py`/`export_weights.py`/`drift_monitor.py`는 numpy·pandas·torch 등 외부 패키지가 필요**해서, 반드시 이 venv가 켜진 상태에서 실행해야 한다. venv를 안 켜고 이 뒤 스크립트들을 돌리면 `ModuleNotFoundError: No module named 'torch'` 같은 에러가 난다.
- 이후 H-1②부터 H-7까지 모든 python 명령어는 **이 venv가 계속 켜진 상태**(같은 쉘 세션)에서 실행된다고 가정한다.

### 2-3. H-1 ② preprocess.py (1차, base — shift.csv 없음)

> 🖥️ **실행 명령어**
>
> ```bash
> rm -f real/shift.csv
> VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py
> ```

| 명령/환경변수 | 이 실행에서 준 값 | 의미 (소스: `preprocess.py`) |
|---|---|---|
| `rm -f real/shift.csv` | — | H-1이 실행되는 **이 시점에** `real/shift.csv`가 남아있지 않도록 미리 지운다. §2-1에서 이미 한 번 지웠지만, H-1 직전에 한 번 더 명시하는 것 — 이게 없으면 base 모델이 shift 분포를 미리 학습해버려 H-3(§2-7)의 fine-tune 효과가 사라진다(§2-1 설명 참고). |
| `VCCB_AI_DATA_FILE` | `./vccb_ai_dataset.csv` | preprocess.py가 Method B(§2-3 표 참고)로 읽을 **단일 CSV 파일** 경로. 스크립트 기본값도 이미 `vccb_ai_dataset.csv`라서(`GTEST_CSV` 기본값), 이번엔 사실 지정 안 해도 결과가 같다 — 다른 이름의 CSV를 쓸 때를 위해 항상 명시적으로 적어주는 습관. |
| `VCCB_AI_DATA_DIR` | `./real` | Method B로 읽을 **디렉토리**(그 안의 `*.csv`를 전부 병합). 기본값도 `real/`이라 마찬가지로 이번엔 명시가 필수는 아니다. 이 디렉토리에 `shift.csv`, `round_*.csv` 등이 쌓이는 게 §2-1에서 계속 언급된 "real/ 정리"의 대상이다. |
| `python preprocess.py` | — | 옵션(argument) 없이 바로 실행. 이 스크립트는 CLI 플래그가 아니라 **전부 환경변수**로 동작을 제어한다(`VCCB_AI_SYNTH_N`, `VCCB_AI_OUTLIER_MODE`, `VCCB_AI_PROB_HORIZON` 등도 전부 환경변수이며, 이번엔 전부 기본값 그대로 사용됨). |

**실행 결과:**

```
[Method B]   + 64,465 rows  (./vccb_ai_dataset.csv)
[Method B] real CSV     : 64,465 unique samples from 1 file(s)
[outlier] mode=iqr  real rows 64,465 -> 64,465 (dropped 0 = 0.00%)
[Method A] synthetic    : 20,000 samples (3GPP 5QI rules)
----------------------------------------------------------------
Total samples : 84,465  (real 64,465 = 76.3% / synthetic 20,000)
  resume_prob=1 (resume < 60s): 55.1%
  resume_time   : mean=343.8s  std=579.8s
```

| 줄 | 실제 동작 (소스: `preprocess.py`) | 쉬운 설명 |
|---|---|---|
| `[Method B] + 64,465 rows (./vccb_ai_dataset.csv)` | `discover_real_csvs()`가 `VCCB_AI_DATA_FILE`(단일 파일)과 `VCCB_AI_DATA_DIR`(디렉토리, 기본 `./real/`) 안의 모든 `*.csv`를 pandas로 읽어들인다. 앞서 `rm -f real/shift.csv`를 했으므로 이 시점엔 `vccb_ai_dataset.csv` 하나만 있다. | "Method B"는 이 스크립트가 CSV **파일**에서 읽어온 데이터를 부르는 이름이다 — 합성이든 실측이든 CSV에 들어있으면 전부 Method B로 취급된다(§0의 "용어 함정" 참고). |
| `real CSV : 64,465 unique samples from 1 file(s)` | 여러 CSV를 합친 뒤 `drop_duplicates()`로 완전히 똑같은 행을 제거한다. | 중복 제거 후 남은 "고유(unique)" 행 수와, 그걸 만든 파일 개수. 파일이 1개뿐이라 중복 제거로 줄어든 게 없다(64,465 그대로). |
| `[outlier] mode=iqr ... dropped 0 (0.00%)` | **이상치(outlier) 필터**: 도메인 규칙(라벨이 1초 미만/3600초 초과, RSRP가 0~127 범위 밖 등)과 통계 규칙(`IQR fence`)으로 "있을 수 없는 값"을 가진 행을 학습 전에 제거한다. `IQR(Interquartile Range, 사분위 범위)` = 데이터를 크기순으로 4등분했을 때 25%~75% 지점 사이의 폭. 이 폭의 3배(`k=3.0`, 기본값)를 넘어서는 값을 "너무 멀리 떨어진 값"으로 보고 버린다. | base 데이터는 3GPP 규칙으로 처음부터 깨끗하게 만들어졌으니 버릴 게 0개인 게 정상이다. **실측 CSV를 넣었을 때 이 %가 갑자기 커지면** 로거 자체에 문제가 있다는 신호다(예: 타임스탬프 버그로 음수/초대형 dwell이 찍힘 — `synth_data_gen.py`의 `outlier_frac`이 흉내 내는 게 바로 이런 버그). |
| `[Method A] synthetic : 20,000 samples (3GPP 5QI rules)` | CSV와 무관하게, `preprocess.py`가 자체적으로 3GPP 5QI 프로필(`FIVEQI_PROFILES`) 규칙에 따라 새 샘플 20,000개를 즉석에서 만들어 추가한다(`VCCB_AI_SYNTH_N` 환경변수로 개수 조절, 기본 20,000). | "Method A"는 CSV 없이 이 스크립트가 그 자리에서 만들어내는 데이터를 부르는 이름. 학습 데이터의 절대량과 5QI별 다양성을 보장하는 "안전판" 역할 — CSV(Method B)가 특정 5QI만 몰려 있어도 Method A가 균형을 채워준다. |
| `Total samples : 84,465` | 64,465(Method B) + 20,000(Method A) = 84,465. 이후 두 그룹을 무작위로 뒤섞어(`rng.permutation`) 하나의 데이터셋으로 합친다. | 최종적으로 모델이 학습하게 될 전체 샘플 수. |
| `real ... = 76.3%` | Method B(64,465) ÷ 전체(84,465) × 100 = 76.3%. | Method B(CSV)가 전체의 몇 %를 차지하는지. **이 비율이 높을수록 "CSV 기반" 학습 의존도가 높다**는 뜻이지만, §0에서 짚었듯 지금은 그 CSV 자체도 시뮬레이션이라 "실측 의존도"로 착각하면 안 된다. |
| `resume_prob=1 (resume < 60s): 55.1%` | `Y_prob = (label_dur < 60초)`로 만드는 **분류 라벨(정답)**의 양성 비율. `PROB_HORIZON_SEC`(환경변수 `VCCB_AI_PROB_HORIZON`, 기본 60초) 안에 resume하면 양성(1), 아니면 음성(0). | 전체 샘플 중 "60초 이내에 금방 다시 깨어난" 샘플이 몇 %인지. 55.1%면 양성/음성이 거의 반반이라 **분류 학습에 유리한 균형 데이터**라는 뜻(한쪽으로 90% 이상 쏠리면 학습이 어려워진다). |
| `resume_time mean=343.8s std=579.8s` | **회귀 라벨**(`Y_time`, dwell 원시 초)의 평균/표준편차. | dwell이 평균 343.8초(약 5.7분)이고, 표준편차(std)가 평균보다 더 큰 건 "로그정규분포(log-normal)"라서 그렇다 — 대부분은 짧게 끝나지만 아주 가끔 매우 길게 끄는 값(수천 초)들이 평균·표준편차를 크게 끌어올린다. 이런 긴 꼬리(롱테일)는 무선망 트래픽에서 흔한 정상 패턴이다. |

이어서 나오는 **막대 히스토그램**(터미널 ASCII):

```
-- final dataset.npz ? Y_time (label_durationSec, log-scaled bins)
  [      1.0,      2.2) |###########                         | 3,150
  [     19.0,     30.8) |####################################| 9,864   ← 최빈 bin
  ...
  [   2008.3,   3184.6) |#########                           | 2,470
  n=84,465  real+synthetic merged
```

- 이건 **히스토그램(histogram)** — 값을 여러 구간(bin)으로 나눠서 각 구간에 몇 개가 들어가는지 세어 그린 그래프다. 각 행 = 구간 `[lo, hi)` 하나(lo 이상 hi 미만). `#` 길이는 그 구간에 속한 샘플 수에 비례한다(가장 긴 게 "최빈값 구간" = 가장 흔한 dwell 범위).
- **"로그 스케일(log-scaled)"**: 구간 경계(`1.0, 2.2, 4.0, 7.0, ...`)가 일정한 간격이 아니라 곱셈으로 커진다(1.0→2.2→4.0은 대략 ×1.8씩). dwell이 1초부터 수천 초까지 워낙 넓게 퍼져 있어서, 보통의 등간격(1초씩) 구간으로 나누면 짧은 값들이 전부 한 구간에 뭉쳐 버린다. 로그 스케일을 쓰면 **짧은 dwell(초 단위)과 긴 dwell(수천 초)을 한 화면에서 같이 자세히 볼 수 있다.**
- 오른쪽 숫자 = 그 구간에 실제로 들어간 행 수.
- 읽는 법: **10~30초 근처가 가장 흔한 dwell 구간(봉우리)**이고, 수천 초짜리 긴 꼬리도 소수 존재 → InactivityTimer를 예측하는 문제로서는 자연스러운 분포.

```
Saved → .../dataset.npz
  X_seq  (84465, 10, 2)  X_stat (84465, 14)
```

- `X_seq (84465, 10, 2)` = 샘플수 × **시퀀스 길이 10(H)** × **[durationSec, resumeCause] 2피처**.
- `X_stat (84465, 14)` = 샘플수 × **정적 피처 14(S)**. → **모델 계약 S=14, H=10 확인 포인트.** 여기 숫자가 다르면 즉시 중단.

### 2-4. H-1 ③ train.py (Phase 0, 전체 학습)

> 🖥️ **실행 명령어**
>
> ```bash
> python train.py
> ```

| 명령/환경변수 | 이 실행에서 준 값 | 의미 (소스: `train.py`) |
|---|---|---|
| `python train.py` | 환경변수 없이 실행 | `VCCB_AI_FINETUNE`을 안 줬으므로 기본값 `"0"` → `FINETUNE=False` 분기를 탄다. 즉 **처음부터(scratch) 학습** 모드다(랜덤 초기화된 새 모델로 시작, §2-8의 "warm start"와 대조). |
| `VCCB_AI_FINETUNE` | (안 줌 → `"0"`) | `"1"`이면 §2-8처럼 기존 `model.pt`를 불러와 이어 학습(fine-tune)하고, 안 주면(기본) 새로 학습한다. 이 단계는 fine-tune 이전이라 당연히 안 줌. |
| `VCCB_AI_EARLY_STOP_PATIENCE` | (안 줌 → scratch 기본값 `"0"`) | 조기종료 patience. FINETUNE이 꺼져 있을 때 기본값이 0(=조기종료 OFF)이라, 50 에폭을 항상 다 돈다. |
| (내부 상수, CLI/환경변수 아님) | `EPOCHS=50`, `LR=1e-3`, `BATCH_SIZE=256`, `VAL_RATIO=0.15`, `SEED=42` | 이 값들은 스크립트 상단에 **고정 상수**로 박혀 있어 커맨드라인이나 환경변수로 바꿀 수 없다(scratch 학습은 항상 이 값). fine-tune만 `VCCB_AI_FINETUNE_LR`/`VCCB_AI_FINETUNE_EPOCHS`로 오버라이드 가능(§2-8). |

**실행 결과 — 에폭 진행 테이블:**

```
Epoch      Train        Val   P_loss   T_loss
    1     0.4320     0.2873   0.2708   0.0330 *
   ...
   40     0.2198     0.2189   0.2068   0.0242 *
   ...
   50     0.2197     0.2189   0.2068   0.0242
```

| 열 | 실제 계산 (소스: `train.py`) | 쉬운 설명 |
|---|---|---|
| `Epoch` (에폭) | 학습 데이터 전체를 처음부터 끝까지 한 바퀴 다 보는 것을 1**에폭**이라 부른다. `for epoch in range(1, eff_epochs+1)` 루프의 반복 횟수. | 사람으로 비유하면 "문제집 한 권을 통째로 다 풀어보는 한 회차"다. 1에폭에 그친 것 vs 50에폭을 반복한 것을 비교하면, 반복할수록(보통) 모델이 점점 더 잘 맞히게 된다. |
| `Train` | 그 에폭 동안 **학습셋(전체의 85%)**에 대해 계산한 손실의 평균. 손실 = `BCE(분류) + 0.5 × SmoothL1(회귀)`. | **손실(loss)**은 "모델이 얼마나 틀렸는지"를 숫자 하나로 나타낸 것 — 낮을수록 더 잘 맞춘다는 뜻이다. `BCE`(Binary Cross-Entropy, 이진 교차 엔트로피)는 "KEEP이냐 RELEASE냐" 둘 중 하나를 맞히는 **분류** 문제에 쓰는 표준 손실 공식이고, `SmoothL1`(=Huber loss)은 "몇 초로 예측했나"를 맞히는 **회귀** 문제에 쓰는 손실 공식인데 유난히 큰 오차(이상치)에는 덜 민감하게(완만하게) 반응하도록 설계돼 있다. 이 둘을 0.5:1 비율로 더해서 "분류 반, 회귀 반" 신경 쓰게 만든 것. |
| `Val` | 학습에는 전혀 쓰지 않고 따로 떼어둔 **검증셋(전체의 15%)**에 대해 같은 방식으로 계산한 손실. | **Val(validation, 검증) loss가 가장 중요한 숫자다.** Train loss는 "이미 공부한 문제"를 얼마나 잘 푸는지고, Val loss는 "한 번도 안 본 문제"를 얼마나 잘 푸는지다 — 실전에서 중요한 건 후자. **Train은 계속 내려가는데 Val이 다시 오르면 "과적합(overfitting)"** = 모델이 학습 데이터를 답 통째로 외워버려서 새 데이터에는 오히려 약해지는 현상. |
| `P_loss` | Val 손실 중 분류(`BCE`) 부분만 따로 뽑은 값. | resume_prob(KEEP/RELEASE 결정) 예측이 얼마나 정확한지에 대한 손실. |
| `T_loss` | Val 손실 중 회귀(`SmoothL1`) 부분만 따로 뽑은 값, 단 라벨이 양성(60초 이내 resume)인 샘플에서만 계산(`mask = y_prob > 0.5`). | resume_time(몇 초 뒤 resume할지) 예측이 얼마나 정확한지에 대한 손실. 음성 샘플(60초 넘도록 안 돌아올 UE)은 "몇 초 뒤"라는 숫자 자체가 의미 없으므로 회귀 손실 계산에서 제외한다. |
| `*` | 그 에폭의 Val loss가 지금까지 나온 것 중 **가장 낮으면(최고 성능 갱신)** 붙는다. 이 순간 모델 가중치 전체를 메모리에 복사해 둔다(`best_state`). | 학습이 끝나면 **마지막 에폭의 가중치가 아니라, `*`가 마지막으로 붙었던 그 "제일 좋았던 순간"의 가중치**를 최종 `model.pt`로 저장한다. 그래서 50번째 줄에 `*`가 없어도 문제가 아니다 — 이미 더 좋았던 40번째 결과가 저장돼 있다. |

- 읽는 법: `*`이 뒤쪽 에폭까지 계속 붙다가 어느 지점부터 멈추면 "수렴"(더 이상 좋아지지 않는 지점에 도달)했다는 뜻. 여기선 40 근처에서 사실상 수렴(0.2189)하고 이후 평평 → 정상.
- **조기종료(early stopping)**: Val loss가 `patience`(기본값, 예: 5)번 연속으로 개선되지 않으면(=`*`가 5번 연속 안 붙으면) 남은 에폭을 다 채우지 않고 학습을 일찍 끝내는 기능(`EARLY_STOP_PATIENCE`). 시간 낭비(더 이상 좋아지지 않을 게 뻔한데 계속 도는 것)를 막는다. **full 학습(scratch)은 기본 OFF(patience=0)**라 50 에폭을 항상 다 돌리고, **fine-tune은 기본 ON(patience=5)**이라 필요하면 일찍 멈춘다.

```
Best val loss : 0.2188
Model saved   → .../model.pt
```

- 전체 에폭 중 최저 Val loss와, 그 best 가중치를 `model.pt`로 저장했다는 뜻(마지막 에폭이 아니라 **best**를 저장).

#### Phase-1 품질 지표 블록 (매 실행 자동 출력)

```
── Phase-1 quality metrics (val) ──
  samples       : train=71,796 val=12,669 (val pos 6,866=54.2%)
  [class] AUROC=0.9696 PR-AUC=0.9678 ECE=0.0097  @0.6: P=0.941 R=0.913 F1=0.927
  [regr s] method                   MAE    RMSE     p90    bias
           model                  10.64   13.20   20.42    0.34
           naive                  12.35   14.96   23.32    0.08
           persistence            22.97   76.99   38.89   12.87
           oracle/core3           11.61   14.33   22.62   -0.04
           oracle/full5           11.56   14.34   23.08   -0.34
  [oracle core3] groups=63 cov=99.5% MAE(model,oracle)=3.62
  [oracle full5] groups=190 cov=96.6% MAE(model,oracle)=3.44
  [timer] premature-release model=30.2% persist=29.1%  |  wasted-hold model=69.8% (over ~8.5s p90 19.5s) persist=70.8%
  [verdict] beats naive=Y  beats persistence=Y
```

**samples 줄**
- `train=71,796 val=12,669` : 85/15 분할. `val pos 6,866=54.2%` : 검증셋 양성(resume<60s) 개수와 비율.

**[class] 분류 지표** (resume_prob 품질, KEEP/RELEASE 결정용)

| 지표 | 계산 (소스: `metrics.py`) | 쉬운 설명 | 기준 |
|---|---|---|---|
| AUROC | "Area Under the ROC Curve". 검증셋에서 양성 샘플 하나와 음성 샘플 하나를 무작위로 한 쌍씩 뽑는 걸 모든 조합에 대해 반복했을 때, **모델이 양성에 더 높은 점수를 준 비율**. 1.0=완벽, 0.5=동전 던지기(순전히 무작위)와 같음. | "이 모델이 KEEP 대상과 RELEASE 대상을 얼마나 잘 구별하는가"를 특정 임계값(0.6 등)과 상관없이 종합적으로 평가한 점수다. 0.97이면 "100번 비교 중 97번은 진짜 양성 쪽에 더 높은 점수를 준다"는 뜻. | > 0.90 |
| PR-AUC | "Precision-Recall AUC" — precision(정밀도)-recall(재현율) 곡선 아래 면적. | AUROC와 취지는 비슷하지만, 양성/음성 비율이 한쪽으로 심하게 쏠린("불균형") 데이터에서 더 엄격하게 반응하도록 설계된 버전. 지금은 양성 54%로 이미 균형이 잡혀 있어서 AUROC(0.9696)와 거의 같은 값(0.9678)이 나온다. | 높을수록 좋음 |
| ECE | "Expected Calibration Error(기대 보정 오차)". 모델이 "resume_prob=0.7"이라고 말한 샘플들을 한 그룹으로 모아서, 그중 실제로 몇 %가 진짜 60초 이내 resume했는지 확인하고, 그 실제 비율과 0.7의 차이를 평균 낸 값. | **모델이 뱉는 확률 숫자를 그대로 믿을 수 있는지**의 척도다. ECE=0.01이면 "0.7이라고 말했을 때 실제로도 대략 69~71%는 맞다"는 뜻 — 숫자 자체가 신뢰할 만하다는 것. ECE가 크면 "모델이 0.9라고 자신 있게 말해도 실제로는 60%밖에 안 맞을 수 있다"는 위험 신호다. | < 0.03 |
| @0.6 P (precision, 정밀도) | resume_prob≥0.6인 샘플을 전부 KEEP으로 판정했을 때, **KEEP이라 판정한 것 중 실제로 맞은 비율** = TP/(TP+FP). | "모델이 KEEP이라고 한 것 중에 진짜 맞은 게 몇 %인가." 잘못 KEEP 판정(오탐, false positive)이 적을수록 이 값이 높다. | > 0.93 |
| @0.6 R (recall, 재현율) | 같은 기준에서, **실제 양성(진짜 60초 이내 resume한 대상) 중 모델이 잡아낸 비율** = TP/(TP+FN). | "진짜 KEEP 대상 중에 모델이 놓치지 않고 잡아낸 게 몇 %인가." 놓친 것(미탐, false negative)이 적을수록 이 값이 높다. | > 0.90 |
| F1 | Precision과 Recall의 조화평균 = `2PR/(P+R)`. | Precision과 Recall은 보통 한쪽을 올리면 다른 쪽이 내려가는 trade-off 관계다(임계값을 낮추면 Recall↑Precision↓, 올리면 반대). 둘을 하나의 숫자로 균형 있게 요약한 게 F1. | > 0.85 |

**[regr s] 회귀 지표** (resume_time 초 예측, 양성 샘플만 — 음성은 "몇 초 뒤"가 의미 없어서 제외)

예측 방법 (행):
- **model**: 우리 GRU 모델의 예측. **이게 제일 낮아야(정확해야) 정상**(MAE 10.64s).
- **naive**: 가장 멍청한 기준선 — 학습셋 양성 샘플들의 dwell **전역 평균**을 모든 샘플에 똑같이 예측값으로 씀. "아무것도 안 배운 모델도 이 정도는 한다"는 최소 하한선.
- **persistence**: `stat[9]`(그 UE의 직전 resume들 평균 dwell, `avgDurationSec`)를 그대로 이번 예측값으로 씀 — "방금까지 하던 대로 또 그럴 것"이라는 가장 단순한 예측 방식. C++ 엔진에 AI가 없다면 대신 쓸 법한 방식이라 **모델이 이걸 못 이기면 AI를 도입할 가치가 없다.**
- **oracle/core3**: 검증이 아니라 "치팅"에 가까운 상한선 — (5QI, RSRP 구간, 성공률 구간) 3개 축이 똑같은 학습셋 샘플들의 dwell **평균**을 그 조합의 "정답"처럼 미리 계산해 놓고, 검증셋의 같은 조합에 그 평균값을 꺼내 쓴다. 모델이 "입력이 같으면 평균적으로 얼마나 걸리는지"를 완벽히 배웠다면 도달할 수 있는 **이론적 상한**에 가깝다.
- **oracle/full5**: 위와 같은 방식이지만 조합 축을 5개(5QI, DRB 수, GBR 여부, RSRP 구간, 성공률 구간)로 더 세밀하게 나눈 버전. 더 촘촘한(그래서 더 낮은 MAE가 나오는) 상한선.

성능 지표 (열):
| 지표 | 계산 | 쉬운 설명 |
|---|---|---|
| MAE (Mean Absolute Error) | `mean(\|예측−실제\|)` — 오차의 절댓값을 평균. | "평균적으로 몇 초씩 틀리는가." 가장 직관적인 지표. 낮을수록 좋음. |
| RMSE (Root Mean Squared Error) | `sqrt(mean((예측−실제)²))` — 오차를 제곱해서 평균 낸 뒤 다시 제곱근. | MAE와 비슷하지만 **큰 오차에 훨씬 더 민감**하다(제곱을 하기 때문). RMSE가 MAE보다 훨씬 크면(예: persistence의 22.97 vs 76.99) "가끔 아주 크게 틀리는 샘플이 섞여 있다"는 신호. |
| p90 (90th percentile) | 절대 오차를 크기순으로 줄 세웠을 때 하위 90% 지점의 값. | "최악의 10%를 제외한 나머지는 이 초 이내로 맞힌다"는 뜻 — 꼬리(가장 잘 못 맞히는 부분) 성능을 보는 지표. |
| bias (평균 부호오차) | `mean(예측−실제)` — 절댓값 없이 그대로 평균(부호 유지). | **+면 평균적으로 과대예측**(실제보다 길게 예측 → 타이머를 필요보다 오래 유지, 자원 낭비), **−면 평균적으로 과소예측**(실제보다 짧게 예측 → 타이머가 너무 빨리 끝나 조기 해제, 재접속 비용 증가). 0에 가까울수록 어느 방향으로도 치우치지 않았다는 뜻(model=+0.34로 거의 무편향). |

**[oracle …] 줄**
- `groups=63 cov=99.5%` : oracle이 만든 (5QI,rsrp,succ) 조합 그룹 개수(63개) / 검증셋 중 그 그룹들로 커버되는 비율(99.5%). oracle은 학습셋에서 그 조합이 **30개 이상 나온 그룹만** "믿을 만하다"고 채택하므로(`N_MIN_GROUP=30`), 너무 드문 조합은 제외되고 그만큼 커버율이 100%보다 낮아진다.
- `MAE(model,oracle)=3.62` : **우리 모델의 예측과 oracle(이론 상한)의 예측이 서로 얼마나 다른가**(초). 이 값이 작을수록 모델이 "이론적으로 가능한 최선"에 가깝게 학습됐다는 뜻. 3.6초면 매우 우수한 수준.

**[timer] 타이머 KPI** (모델 오차를 "실제 서비스에 어떤 영향을 주는가"로 번역한 지표)
- `premature-release model=30.2%` : 모델이 추천한 타이머(`ceil(예측초 × MARGIN_FACTOR)`, MARGIN_FACTOR=1.2)가 실제 resume 시점보다 **짧게 끝나서** UE를 너무 일찍 IDLE로 강제 전환시킨 비율. 이러면 UE가 다시 접속할 때 RRC 셋업 절차를 처음부터 다시 밟아야 해서 자원·시간이 낭비된다. persist(기준선)의 29.1%와 비슷한 수준 — 모델이 이 부분에서는 기준선보다 특별히 낫지도 나쁘지도 않다.
- `wasted-hold model=69.8% (over ~8.5s p90 19.5s)` : 반대로 타이머가 실제보다 **길게 유지돼서** gNB가 그 UE의 context를 필요 이상으로 붙잡아 둔 비율과, 그 낭비 시간(평균 8.5초, 최악 10%는 19.5초까지). `MARGIN_FACTOR=1.2`가 "예측값의 1.2배"로 **의도적으로** 여유를 주는 설계라서 이렇게 late(늦게 해제)가 많은 건 정상이다. 눈여겨봐야 할 건 비율이 아니라 **낭비되는 초의 크기**(8.5초면 작아서 괜찮음).

**[verdict]**
- `beats naive=Y beats persistence=Y` : 앞서 설명한 두 기준선을 MAE 기준으로 모두 이겼는지(모델 MAE < 기준선 MAE). **둘 다 Y여야 이 AI 모델을 실제로 배포할 가치가 있다** — naive를 못 이기면 "아무것도 안 배운 것보다 못한" 모델이고, persistence를 못 이기면 "AI 없이 단순 평균값 쓰는 것보다 못한" 모델이라는 뜻이기 때문. 하나라도 N이면 학습 실패를 의심해야 한다.

#### 터미널 시각화 (PNG의 ASCII 미러)

```
-- regression MAE vs baselines (val positives) --
  model       |#################                   | 10.64s
  naive       |###################                 | 12.35s
  persistence |####################################| 22.97s
-- classification quality (Phase-1) --
  AUROC  |####################################| 0.970
  F1@0.6 |##################################  | 0.927
-- resume_prob separability (model output) --
  range                 resume<horizon   no-resume
  [  0.00,  0.08)               |###########        44/4092
  ...
  [  0.92,  1.00) ##############|#                5214/191
```

- **MAE 막대**: model 막대가 naive/persistence보다 짧아야 정상(그림으로 verdict 확인).
- **분류 품질 막대**: AUROC/F1을 Phase-1 floor 대비 시각화.
- **separability(분리도)**: 세로선(`|`) = KEEP 임계값 0.6. 왼쪽 숫자쌍 `44/4092` = 그 prob 구간의 (양성 수 / 음성 수).
  - 잘 학습됐다면 **음성(no-resume)은 왼쪽(prob 낮음)에, 양성은 오른쪽(prob 높음)에** 쏠린다.
  - 여기선 `[0.00,0.08)`에 음성 4092개, `[0.92,1.00)`에 양성 5214개 → **깔끔하게 분리됨**(AUROC 0.97과 일치).

> ⚠️ **"MAE 막대"와 "분류 품질 막대"의 `|` 시작 위치가 서로 다른 건 오타/버그가 아니다.** 이 스크립트(`termplot.py`)는 막대그래프를 그릴 때마다 **그 그래프 안의 가장 긴 라벨 길이**에 맞춰 `|` 위치를 정한다. "MAE 막대"는 라벨 중 `persistence`(11글자)가 가장 길어서 `|`가 뒤로 밀리고, "분류 품질 막대"는 `F1@0.6`(6글자)이 가장 길어서 `|`가 앞에 온다. **같은 그래프 안(예: model/naive/persistence 세 줄)에서는 항상 정렬이 맞아야 하고**, 여기 예시들도 실제 그렇다 — 다른 그래프끼리는 애초에 맞출 필요가 없는 설계다.

### 2-5. H-1 ④ model.pt.phase1 백업

> 🖥️ **실행 명령어**
>
> ```bash
> cp -v model.pt model.pt.phase1
> ```

`cp`(파일 복사)에 `-v`(verbose, 상세 출력) 옵션을 붙여서, 복사가 실제로 일어났는지 화면에 확인 문구를 찍게 한 것뿐이다(옵션 없이 `cp model.pt model.pt.phase1`만 해도 파일은 똑같이 복사되지만 아무 출력도 없다).

**실행 결과:**

```
'model.pt' -> 'model.pt.phase1'
```

- base 모델을 `.phase1`이라는 별도 이름으로 복사해 둔다. fine-tune이 망가지면 이걸로 되돌린다. **export 전의 안전한 기준점.**

### 2-6. H-2 gen_realish_csv.py (분포 이동 CSV 생성)

먼저 이 스크립트가 **무엇을 만드는지** 그림부터. H-1까지는 `vccb_ai_dataset.csv`(가상 UE 5,000명이 만든 64,465행)로 학습한 "base 모델"이 있다. `gen_realish_csv.py`는 이 base 모델이 **한 번도 보지 못한 새 3,000행**을 추가로 만들어서, "이 새 데이터를 보여주면 fine-tune이 실제로 모델을 움직이는가?"를 검증하기 위한 용도다. 스크립트 맨 위 주석에도 "이건 진짜 UE 데이터가 아니다. fine-tune 배관(warm start+freeze+early-stop+export)이 실제로 모델을 움직인다는 걸 증명하는 게 유일한 목적"이라고 스스로 밝히고 있다.

> 🖥️ **실행 명령어**
>
> ```bash
> mkdir -p real
> python3 gen_realish_csv.py --out real/shift.csv --n 3000 --dur-scale 2.2
> ```

| 옵션 | 이 실행에서 준 값 | 의미 (소스: `gen_realish_csv.py` argparse) |
|---|---|---|
| `mkdir -p real` | — | `real/` 디렉토리가 없으면 만든다(`-p`=이미 있으면 에러 없이 통과). `--out`이 그 안에 파일을 쓰므로 디렉토리가 먼저 있어야 한다. |
| `--out` | `real/shift.csv` | 출력 CSV 경로. `preprocess.py`가 `VCCB_AI_DATA_DIR=./real`을 보므로, 이 디렉토리 안에 써야 H-3에서 자동으로 읽힌다. |
| `--n` | `3000` | 생성할 행 수(아래 표의 `rows` 항목). |
| `--dur-scale` | `2.2` | dwell에 곱하는 배율(아래 표에서 자세히 설명). 기본값은 `1.6`인데 이번엔 `2.2`로 더 크게 줘서 shift를 더 뚜렷하게 만들었다. |
| `--seed` | (안 줌 → 기본값 `7`) | 재현성 시드. `synth_data_gen.py`의 시드(0)와는 별개의 값(7)이 기본으로 잡혀 있다 — 서로 다른 스크립트이므로 우연히 같은 데이터가 나오지 않도록. |
| `--rsrp-bias` | (안 줌 → 기본값 `0.15`) | 아래 표의 `rsrp_bias` 항목. 기본값 그대로 사용됨. |
| `--lab-mode` | (안 줌 → 꺼짐) | 켜면 AMBR/RNA를 고정값으로 박는다(아래 표의 `lab_mode` 항목). 안 줬으므로 꺼진 상태(=OFF)로 실행됨. |
| `--ambr-fixed`, `--rna-fixed` | (안 줌, `--lab-mode`가 꺼져 있어 무의미) | `--lab-mode`를 켰을 때만 쓰이는 고정값들이라, 이번엔 적용되지 않았다. |
| `--append` | (안 줌 → 기본값 꺼짐) | 켜면 기존 CSV에 이어붙이고, 꺼져 있으면(기본) 덮어쓴다 — 아래 표의 `append=False`와 연결됨. |

**실행 결과:**

```
  rows           : 3,000
  dur_scale      : 2.2  (1.0 = matched to prior)
  rsrp_bias      : 0.15
  lab_mode       : OFF (AMBR/RNA vary like the synthetic prior)
  label mean     : 177.1s   resume<60s = 29.6%
  written        : .../real/shift.csv  (append=False)
```

**먼저 용어 2개**

- **dwell (드웰, 체류시간)**: UE가 RRC_INACTIVE 상태로 "아무 것도 안 하고 대기하는 시간"(초). 이 값이 곧 `label_durationSec`이고, 모델이 맞혀야 하는 정답(`resume_time`)의 원천이다. 예: "이 UE는 40초 동안 INACTIVE로 있다가 다시 resume했다" → dwell = 40초.
- **prior (사전 분포/사전 가정)**: 통계학에서 "미리 가정해 둔 분포"라는 뜻. 여기서는 **base 모델(H-1의 model.pt)이 학습 때 이미 배운 dwell 분포**를 가리킨다. base 모델은 `preprocess.py`의 3GPP 규칙(5QI별 평균 dwell + RSRP·성공률 보정)으로 만든 dwell을 정답으로 보고 학습했다. `gen_realish_csv.py`는 소스코드에서 **이 3GPP 규칙을 거의 그대로 재사용**해서 일단 dwell을 계산한 뒤(`sample_duration()`), 거기에 "인위적인 왜곡"을 하나 더 얹는다. 그 왜곡이 바로 `dur_scale`과 `rsrp_bias`다.

| 토큰 | 실제 계산 (소스: `gen_realish_csv.py`) | 쉬운 설명 |
|---|---|---|
| `rows : 3,000` | `--n 3000`(기본값). `synth_data_gen.py`처럼 "가상 UE 5,000명"을 굴리는 게 아니라, **한 번에 독립적인 샘플 3,000개**를 곧바로 뽑는다(UE라는 개념 자체가 없다). | "행(row)" = CSV 한 줄 = 학습 데이터 한 건("UE 하나가 RRC_INACTIVE로 있다가 resume한 사건" 하나). 3,000개를 새로 만들었다는 뜻 — 실측이라면 한 번에 이 정도밖에 못 모을 거라는 "작은 배치"를 흉내 낸 것. |
| `dur_scale : 2.2` | `d = next_dur * 2.2` — 3GPP 규칙으로 계산한 dwell(`next_dur`)에 **그대로 2.2를 곱한다**. | prior(base 모델이 아는 계산식)가 "이 조건이면 dwell=20초"라고 말하는 걸, 이 스크립트는 "아니, 실측이면 44초(20×2.2)일 것"이라며 라벨을 바꿔치기하는 것. "실측은 base 모델의 가정보다 dwell이 더 길더라"는 가짜 신호를 강제로 주입해 fine-tune이 배울 거리를 만든다. |
| `rsrp_bias : 0.15` | `d += 0.15 * (75.0 - last_rsrp)` — dwell(초)에 **"0.15 × (75 − 그 UE의 RSRP)"를 더한다.** ⚠️ RSRP 값 자체를 곱하는 게 아니라, dwell(라벨)에 더하는 보정항이다. | RSRP는 신호 세기 지표(0~127, 클수록 신호 좋음, 75가 기준점). RSRP가 75보다 **낮으면**(신호 약함) 이 항이 양수가 되어 dwell이 **늘어나고**, 75보다 **높으면**(신호 좋음) 음수가 되어 dwell이 **줄어든다**. 예: RSRP=40(약한 신호) → (75−40)=35 → dwell에 +5.25초 추가. base 모델도 RSRP↔dwell 관계를 곱하기 형태로 이미 알고 있지만(`preprocess.py`의 `rsrp_factor`), 이 스크립트는 그 위에 **덧셈으로 된 새 관계를 하나 더** 얹어서 "실측에서는 RSRP 영향이 base가 배운 것보다 더 세더라"는 신호를 만든다. |
| `lab_mode : OFF` | `--lab-mode` 플래그를 안 줬으므로, AMBR(stat[2])·RNA(stat[8])도 5QI 프로필 범위 안에서 무작위로 다양하게 생성(`synth_data_gen.py`와 같은 방식). | 실제 물리 시험실(lab)에서 UE로 테스트하면 AMBR(가입 요금제로 정해지는 다운로드 속도 상한)이나 RNA(셀에 설정된 값)는 **테스트 중 바꾸기 어려운 고정값**이다(스크립트 주석: "SIM 구독/셀 설정으로 고정"). `--lab-mode`를 켜면 이 두 값을 상수로 박아 그 제약을 흉내 낸다. 지금은 OFF라 "고정 안 하고 다양화"한 것 = 실측 lab보다는 좀 더 이상적인(다양성 있는) 데이터에 가깝다. |
| `label mean : 177.1s` | 이 3,000행의 dwell(`dur_scale`+`rsrp_bias` 반영 후 최종값) 평균. | ⚠️ 이 숫자를 base의 "343.8s"(§2-3, 84,465개 혼합 세트 전체 평균)와 **직접 비교하면 안 된다** — 표본 크기·5QI 조합 비율이 전혀 다른 두 그룹의 평균이라 우연히 더 작게 나올 수 있다. `dur_scale=2.2`는 "**같은 조건(5QI/RSRP 등)일 때** base보다 2.2배 길다"는 뜻이지, "전체 평균이 무조건 base보다 커진다"는 뜻이 아니다. |
| `resume<60s = 29.6%` | 이 3,000행 중 60초 이내 resume(=양성 라벨)한 비율. | 이게 실제로 믿을 수 있는 비교 기준이다: base는 55.1%(§2-3)였는데 이 CSV는 29.6% — **훨씬 더 많은 샘플이 60초를 넘겨서야 resume**했다는 뜻 = dwell이 전반적으로 길어지는 방향으로 분포가 이동했다는 증거다. (mean 절대값보다 이 비율로 "shift가 제대로 됐는지" 확인하는 게 정확하다.) |
| `written : real/shift.csv (append=False)` | `write_csv()`가 `mode="a" if append else "w"`로 파일을 연다. `--append` 옵션을 안 줬으므로 `append=False` → **"w" 모드 = 기존 내용을 지우고 새로 쓴다(덮어쓰기).** | 이 스크립트를 몇 번 실행해도 CSV가 누적되지 않고 항상 최신 3,000행만 남는다는 뜻. (참고: 이건 "H-1 전에 미리 지워야 하는 이유"와는 별개다 — H-1은 H-2보다 **먼저** 실행되므로, 문제가 되는 건 이 시점이 아니라 **이전 세션에서 이미 만들어져 있던** shift.csv다. §2-1의 `rm -f real/shift.csv`가 그 대비책.) |

### 2-7. H-3 preprocess.py (2차, shift.csv 포함)

> 🖥️ **실행 명령어**
>
> ```bash
> VCCB_AI_DATA_FILE=./vccb_ai_dataset.csv VCCB_AI_DATA_DIR=./real python preprocess.py
> ```

§2-3과 **완전히 똑같은 명령어**를 다시 실행하는 것뿐이다(옵션도 동일). 코드가 달라진 게 아니라, 그 사이(H-2)에 `real/` 디렉토리 안에 `shift.csv`가 새로 생겼기 때문에 **입력이 달라져서 결과가 달라진다** — `VCCB_AI_DATA_DIR=./real`이 그 디렉토리 안의 `*.csv`를 전부 훑는다는 걸 다시 상기할 것(§2-3 표 참고).

**실행 결과:**

```
[Method B]   + 64,465 rows  (./vccb_ai_dataset.csv)
[Method B]   + 3,000 rows  (./real/shift.csv)
[Method B] real CSV     : 67,465 unique samples from 2 file(s)
...
Total samples : 87,465  (real 67,465 = 77.1% / synthetic 20,000)
  resume_prob=1 (resume < 60s): 54.2%
  resume_time   : mean=338.1s  std=571.6s
  X_seq  (87465, 10, 2)  X_stat (87465, 14)
```

- 이번엔 **CSV 2개**(base + shift) → `from 2 file(s)`, total 87,465(=84,465+3,000). **H-3이 유효하게 동작한 증거.**
- shift.csv가 반영돼 분포가 소폭 이동(resume<60s 55.1%→54.2%, mean 343.8→338.1). 이 미세 변화가 fine-tune이 배울 새 신호.
- 만약 여기 total이 H-1과 **똑같으면** H-3 no-op → `real/shift.csv` 삭제 여부 재확인(§2-1).

### 2-8. H-4 ① train.py (fine-tune, FREEZE_GRU)

> 🖥️ **실행 명령어**
>
> ```bash
> VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py
> ```

| 환경변수 | 이 실행에서 준 값 | 의미 (소스: `train.py`) |
|---|---|---|
| `VCCB_AI_FINETUNE` | `1` | §2-4에서는 이게 없어서 scratch 학습이었는데, 이번엔 `1`로 켜서 `FINETUNE=True` 분기를 탄다 → 기존 `model.pt`를 불러와(warm start) 이어 학습한다. |
| `VCCB_AI_FREEZE_GRU` | `1` | GRU+MLP를 동결하고 출력 head 2개만 학습(아래 표에서 자세히 설명). |
| `VCCB_AI_FINETUNE_LR` | (안 줌 → 기본값 `2e-4`) | fine-tune 학습률. 안 줬으므로 기본값 사용(아래 `lr=0.0002` 출력과 일치). |
| `VCCB_AI_FINETUNE_EPOCHS` | (안 줌 → 기본값 `15`) | fine-tune 에폭 수. 안 줬으므로 기본값 사용(아래 `epochs=15` 출력과 일치). |
| `VCCB_AI_EARLY_STOP_PATIENCE` | (안 줌 → FINETUNE일 때 기본값 `5`) | §2-4와 달리 `FINETUNE=1`이면 기본값 자체가 5(=조기종료 ON)로 바뀐다(아래 `Early stop: patience=5` 출력과 일치). |

**실행 결과:**

```
  [FINETUNE] warm start from .../model.pt
  [FINETUNE] lr=0.0002 epochs=15 freeze_gru=True
  [FINETUNE] pre-FT on this val: AUROC=0.9622 F1=0.921 MAE=10.78s
  Early stop: patience=5 (val loss)
```

먼저 이 모델이 왜 "출력 head가 2개"인지부터 짚고 가야 `freeze_gru`를 이해할 수 있다. `train.py`의 `RrcGruModel`은 매 예측마다 이런 구조로 동작한다(§2-4 X_seq/X_stat과 연결):

```
seq[10,2] → GRU(16) ──┐
                       ├→ concat(32) → head_prob(32→1, sigmoid) → resume_prob   (분류: KEEP이냐 RELEASE냐)
stat[14] → MLP(16)  ──┘             └→ head_time(32→1, 선형)   → resume_time   (회귀: 몇 초 뒤 resume?)
```

즉 이 모델은 한 번 예측할 때 **두 가지 다른 질문에 동시에 답한다**: "이 UE는 곧 돌아올까?"(resume_prob, 분류)와 "돌아온다면 몇 초 뒤일까?"(resume_time, 회귀). 질문이 2개라서 **정확히 2개의 출력 head**(`head_prob`, `head_time`)가 있는 것 — 앞단(GRU+MLP)은 두 질문에 공통으로 쓰이는 "특징 요약기"이고, 뒷단(head 2개)은 그 요약된 정보를 각자의 질문에 맞게 최종 숫자 하나로 뽑아내는 부분이다.

| 줄 | 실제 동작 (소스: `train.py`) | 쉬운 설명 |
|---|---|---|
| `warm start from model.pt` | `model.load_state_dict(torch.load(MODEL_FILE))` — 새 모델을 무작위 값으로 시작하는 게 아니라, **H-1에서 이미 학습된 base 모델의 가중치를 통째로 불러와서 그 상태에서 이어 학습**한다. | "빈 백지에서 다시 배우기 시작" (=scratch 학습, H-1이 한 것) 이 아니라, "이미 어느 정도 아는 사람이 새 자료 몇 개만 더 보고 지식을 업데이트하는 것" (=warm start)이라는 뜻. |
| `lr=0.0002 epochs=15` | fine-tune 전용 값(`VCCB_AI_FINETUNE_LR` 기본 2e-4, `VCCB_AI_FINETUNE_EPOCHS` 기본 15) — base 학습 때 쓰던 `lr=1e-3`, `epochs=50`보다 훨씬 작고 짧다. | `lr`(learning rate, 학습률)은 "한 번에 얼마나 크게 가중치를 바꿀지"를 정하는 값이다. base 학습처럼 큰 폭으로 계속 바꾸면, 이미 잘 배워둔 지식(3GPP 규칙 기반 prior)을 **덮어써서 잊어버리는** 현상이 생길 수 있다 — 이를 "**catastrophic forgetting(치명적 망각)**"이라 부른다. 학습률을 낮추고(0.0002) 에폭도 짧게(15) 잡아서, "기존 지식은 최대한 유지하면서 새 신호만 살짝 반영"하도록 조심스럽게 조정하는 것. |
| `freeze_gru=True` | 환경변수 `VCCB_AI_FREEZE_GRU=1`을 켜면, 파라미터 이름이 `"gru."` 또는 `"mlp."`로 시작하는 것들을 전부 `requires_grad = False`로 만들어 **학습에서 제외**한다(옵티마이저에도 안 넘김). 그 결과 **위 그림의 앞단(GRU+MLP)은 그대로 얼려두고, 뒷단(`head_prob`, `head_time`) 2개짜리 작은 선형층만 학습**된다. | ⚠️ 이름은 "FREEZE_**GRU**"지만 실제로는 GRU뿐 아니라 **MLP까지 같이 동결**된다(변수 이름이 그 사실을 다 담지 못하는 것뿐). "구조/특징 추출 능력은 base가 배운 걸 그대로 믿고, 마지막에 숫자를 뽑아내는 부분만 새 데이터에 맞춰 미세조정"하는 전략이다. 데이터가 3,000행처럼 적을 때, 전체를 다 풀어놓고 학습시키면 그 적은 데이터로 GRU까지 망가뜨릴 위험이 있어서, 위험이 적은 head 2개(각각 32개 가중치+1개 편향뿐)만 건드리는 것. |
| `pre-FT on this val: ...` | fine-tune 학습(에폭 루프)이 **시작되기 전에**, warm-start된(=base) 모델을 이번에 새로 나뉜 검증셋(shift.csv가 포함된 87,465개 중 15%)으로 먼저 채점해 둔다(`eval_on_val(model)`을 학습 전에 한 번 호출). | ★**"정직한 before" 스냅샷**★ — 이렇게 해야 "학습 전(pre-FT)"과 "학습 후(after)"를 **완전히 같은 검증 데이터**로 비교할 수 있다. (base 모델을 H-1 때의 val 결과와 비교하면 안 되는 이유는 §2-8 하단 "해석" 항목에서 다룬다 — 데이터셋 자체가 다르기 때문.) |
| `Early stop: patience=5` | Val loss가 5에폭 연속으로 지금까지의 최저치를 못 넘으면(=`*`가 5번 연속 안 붙으면) 15에폭을 다 채우지 않고 학습을 중단한다. | §2-4에서 설명한 "조기종료"와 같은 기능. fine-tune은 데이터가 적어서 오래 돌릴수록 오히려 과적합 위험이 커지므로 기본으로 켜져 있다. |

에폭 테이블(1~15) 후:

```
Best val loss : 0.2390
...
  [class] AUROC=0.9631 PR-AUC=0.9579 ECE=0.0092  @0.6: P=0.932 R=0.907 F1=0.919
  model 10.76 / naive 12.64 / persistence 23.17 ...
  [verdict] beats naive=Y  beats persistence=Y
```

터미널 before/after 막대 (실제 출력 그대로, 생략 없이):

```
-- regression MAE vs baselines (val positives) --
  before FT   |#################                   | 10.78s
  after FT    |#################                   | 10.76s
  naive       |####################                | 12.64s
  persistence |####################################| 23.17s
-- classification quality (Phase-1) --
  AUROC before |####################################| 0.962
  AUROC after  |####################################| 0.963
  F1 before    |##################################  | 0.921
  F1 after     |##################################  | 0.919
```

- 첫 번째 그래프는 라벨 중 `persistence`(11글자)가 가장 길어서 `|`가 뒤에서 시작하고, 두 번째 그래프는 `AUROC before`(12글자)가 가장 길어서 그 기준으로 다시 정렬된다 — 위 §2-4 경고와 같은 이유(그래프별로 독립적인 라벨 폭 계산).
- `AUROC before`와 `AUROC after`의 막대 길이가 시각적으로 거의 같은 건 이상이 아니다: 0.962→0.963의 차이(0.001)가 36칸짜리 막대 해상도로는 표현이 안 될 만큼 작아서다. **숫자(0.962/0.963)로 실제 변화를 확인**하고 막대 길이 차이는 크게 신경 쓰지 말 것.

**해석 (매우 중요):**
- fine-tune 개선폭은 **작다**: MAE 10.78→10.76, AUROC 0.962→0.963, F1 0.921→0.919(F1은 살짝 감소).
- 이건 정상 범위다. 이유: ① `freeze_gru=True`라 head 2개만 움직임, ② shift가 3,000행/전체 87,465행으로 미미, ③ 데이터가 이미 합성이라 분포가 급변하지 않음.
- **핵심 판정: `pre-FT`와 after가 "완전히 소수점까지 동일"하면 H-3 누락(no-op) 신호.** 여기선 ECE·MAE가 아주 조금이라도 움직였으므로 fine-tune이 실제로 새 데이터를 봤다는 증거.
- ⚠️ `Best val loss 0.2390`은 base의 0.2188보다 **높다**. 이건 나빠진 게 아니라 **다른 데이터셋**(shift 포함, 더 어려운 롱테일)의 val이라 절대비교 불가. 정직한 비교는 오직 **같은 split의 pre-FT vs after**(위 before/after 막대)다.

### 2-9. H-4 ② export_weights.py

> 🖥️ **실행 명령어**
>
> ```bash
> python3 export_weights.py
> ```

이 스크립트는 CLI 옵션도, 환경변수도 받지 않는다(`argparse` 자체가 없음) — 항상 같은 디렉토리의 `model.pt`를 읽어서 `../VCCB_MAIN/hdr/VccbAiWeights.h`에 쓰는 고정된 동작만 한다. 그래서 **어떤 `model.pt`가 지금 이 디렉토리에 있는지가 곧 결과를 결정한다** — H-4①(fine-tune)이 방금 덮어쓴 `model.pt`(=fine-tune 후 가중치)를 그대로 읽는다.

**실행 결과:**

```
  Normalization folded into GRU_ih / MLP / head_time (C++ stays raw)
  Total parameters : 1,266
  Inference @feature-mean : resume_prob=0.6685  resume_time_raw≈40.2s
  (If resume_prob is still ~1.000 across varied inputs, revisit data balance.)
Exported → .../VCCB_MAIN/hdr/VccbAiWeights.h
```

이 스크립트가 하는 일을 한 줄로 말하면: **PyTorch가 학습한 `model.pt`(파이썬 전용 포맷)를, C++ VCCB 엔진이 그대로 읽을 수 있는 순수 숫자 배열(`VccbAiWeights.h`)로 옮겨 담는 것.** C++ 쪽에는 PyTorch가 없으므로, 학습된 가중치 숫자들을 그냥 `static const float[]` 배열로 통째로 복사해서 헤더 파일에 박아넣는다.

| 줄 | 실제 동작 (소스: `export_weights.py`) | 쉬운 설명 |
|---|---|---|
| `Normalization folded into GRU_ih / MLP / head_time` | 학습 때는 입력값을 `(x - 평균μ) / 표준편차σ`로 **정규화(normalization)**해서 모델에 넣었다(§2-4의 train.py `SEQ_MEAN/STAT_MEAN` 등). 이 스크립트는 그 μ/σ 상수를 모델의 첫 번째 층(GRU 입력 가중치, MLP 가중치)과 마지막 층(`head_time`)의 가중치·편향 안에 수식으로 **미리 계산해 접어 넣는다**(`W/σ`, `b - (W/σ)·μ` 형태). | **정규화**란 "5QI(1~9)"와 "AMBR(10~2000)"처럼 단위·범위가 완전히 다른 숫자들을 비슷한 크기로 맞춰주는 전처리다(안 하면 큰 숫자가 학습을 지배해버린다). 문제는 C++ 엔진(`buildAiInput()`)은 이 정규화를 전혀 모른 채 **원본 그대로의 raw 값**(5QI=5, AMBR=350.0 등)을 넘긴다는 것. 그래서 "정규화를 접어 넣는다" = **C++ 코드는 하나도 안 바꾸고, 대신 가중치 숫자 자체를 수학적으로 조정**해서, raw 입력을 넣어도 마치 정규화된 입력을 넣은 것과 똑같은 결과가 나오게 만드는 트릭이다. |
| `Total parameters : 1,266` | 모델의 모든 가중치·편향 개수를 합산(`sum(p.numel() for p in model.parameters())`). GRU(3×16×2 + 3×16×16 + 3×16 + 3×16) + MLP(16×14+16) + head_prob(32+1) + head_time(32+1) = 1,266. | ★**모델 계약 확인 포인트**★ — S(정적 피처 수)=14, H(시퀀스 길이)=10이라는 계약이 지켜졌다면 항상 정확히 1,266이 나와야 한다. 이 숫자가 다르면 C++ 쪽(`VccbAiEngine.h`)과 파이썬 쪽 모델 구조가 어긋났다는 뜻이므로 **뒤 단계(VCCB 재빌드)로 넘어가지 말고 즉시 원인을 찾아야 한다.** |
| `Inference @feature-mean : resume_prob=0.6685` | 모든 입력 피처를 정규화 기준으로 "평균값"(정규화하면 0)에 놓고 모델에 한 번 돌려본 결과(`dummy_seq/stat = zeros`). 일종의 **건강검진용 더미 입력**. | "가장 평범한 조건의 UE"를 넣었을 때 모델이 뭐라고 답하는지 보는 것. 만약 이 값이 어떤 입력을 넣어도 항상 1.000에 가깝게 나온다면, 모델이 "무조건 KEEP"만 외쳐서 아무것도 구별 못 하고 있다는 신호(스크립트 출력의 경고 문구 "resume_prob이 여전히 ~1.000이면 데이터 균형을 재검토하라"가 이걸 말한다). 0.6685는 KEEP 임계값(0.6) 근처의 중간값이라 **건강하다**. |
| `resume_time_raw≈40.2s` | 같은 "평균 조건" 입력에 대한 회귀 head의 예측(초 단위로 역정규화까지 마친 값). | 가장 평범한 UE라면 대략 40초 뒤에 돌아올 거라고 예측한다는 뜻. 상식적인 범위(수 초~수백 초)에 있으면 정상. |
| `Exported → VccbAiWeights.h` | 위 배열들을 문자열로 합쳐서 `../VCCB_MAIN/hdr/VccbAiWeights.h` 파일에 그대로 덮어쓴다. | C++ 쪽 헤더 파일이 최신 학습 결과로 갱신됐다는 뜻. 이 파일은 **C++ 소스 취급**이라 이후 p4 sync → VCCB(ITSCP_VCCB) 재빌드를 해야 실제 바이너리에 반영된다(파이썬 실행만으로는 VCCB 프로세스가 바뀌지 않는다). |

> **export는 파이프라인 맨 끝 1회만.** base 직후 조기 실행하면 fine-tune 전 가중치가 헤더에 잠깐 쓰이는 위험 창이 생긴다. (메모리 [[feedback-retrain-preprocess-snapshot]] 규칙 3)

### 2-10. H-7 drift_monitor.py

> 🖥️ **실행 명령어**
>
> ```bash
> sudo mkdir -p /var/log/vccb_ai
> sudo chown sangku:sangku /var/log/vccb_ai
>
> python drift_monitor.py --capture-baseline --state /var/log/vccb_ai/drift_state.json
> ```

| 명령/옵션 | 이 실행에서 준 값 | 의미 (소스: `drift_monitor.py` argparse) |
|---|---|---|
| `sudo mkdir -p /var/log/vccb_ai` | — | drift 로그·상태 파일을 저장할 디렉토리를 최초 1회 만든다(`/var/log/`는 root 소유라 `sudo` 필요). |
| `sudo chown sangku:sangku ...` | — | 방금 만든 디렉토리의 소유자를 `sangku`로 바꿔서, 이후 `python drift_monitor.py`를 **sudo 없이** 실행해도 그 안에 파일을 쓸 수 있게 한다. |
| `--capture-baseline` | (켜짐, 값 없음) | 지금 이 순간의 `resume_prob` 분포 히스토그램을 "정상 상태의 기준(reference)"으로 `--state` 파일에 저장하고 **바로 종료**한다(뒤에 나오는 drift 계산은 전혀 안 함). 코드 주석: "정상적으로 운영되는 날 1회만 실행". |
| `--state` | `/var/log/vccb_ai/drift_state.json` | 이전 스냅샷·히스토리·`--capture-baseline`으로 저장한 기준값을 보관하는 JSON 파일 경로. 매 실행마다 이 파일을 읽고 다시 갱신해서 쓴다. |
| `--cmd` | (안 줌 → 기본값 `"itscp_vccb get stat ai"`) | AI 추론 통계를 가져오는 실제 명령. 기본값 그대로 사용됨 — 이게 바로 아래 "실행 결과"에서 실패하는 그 명령이다. |

**실행 결과:**

```
[drift] command failed (255): itscp_vccb get stat ai
[drift] AI snapshot invalid/unpublished (valid=0). VCCB not running or no inference yet.
```

- **이건 에러가 아니라 예상된 실패.** BuildServer에서 UT만 돌린 상태라 VCCB 프로세스가 없어 `itscp_vccb get stat ai`가 exit 255.
- drift 모니터는 **VCCB가 실제 운영 중일 때만** 의미. (메모리 [[project-part-h-pipeline-rules]])

---

**참고: 실제 운영 환경(VCCB가 살아있을 때)의 주기 실행 명령어**

BuildServer에서는 VCCB가 안 돌고 있어서 위처럼 실패하지만, 실제 gNB 운영 환경에서는 아래 명령을 cron(예: 매일 새벽 3시)으로 등록해서 주기적으로 돌린다:

> 🖥️ **실행 명령어 (cron 등록용)**
>
> ```bash
> python3 drift_monitor.py \
>     --train-baseline $VCCB/ML/drift_baseline.json \
>     --state /var/log/vccb_ai/drift_state.json \
>     >> /var/log/vccb_ai/drift.log 2>&1
> ```

| 옵션 | 의미 (소스: `drift_monitor.py`) |
|---|---|
| `--train-baseline` | `export_baseline.py`가 학습 데이터로부터 미리 뽑아둔 "정상 입력 분포"(5QI/RSRP 히스토그램) 파일. 지금 들어오는 실제 트래픽의 5QI/RSRP 분포를 이 파일과 비교해서, **"입력 자체가 학습 때와 달라졌는지"**(input drift)를 PSI(Population Stability Index, 모집단 안정성 지수)로 계산한다. |
| `--state` | 위와 같음(이전 스냅샷 + `--capture-baseline`으로 저장한 기준값 보관). |
| `>> .../drift.log 2>&1` | 이건 `drift_monitor.py`의 옵션이 아니라 **쉘(bash) 리다이렉션**이다 — 표준출력(stdout)과 표준에러(stderr, `2>&1`로 stdout에 합침)를 파일 끝에 이어서 계속 추가(`>>`)한다. cron으로 매일 도는 스크립트라, 실행할 때마다 로그가 이 파일에 계속 쌓인다. |
| (내부 동작) | 매 실행마다 ① 이전 실행 대비 `fallback/fail/timeout` 비율 급증(health drift), ② 학습 baseline 대비 입력 분포 PSI(input drift), ③ `--capture-baseline`으로 저장해 둔 기준 대비 출력(`resume_prob`) 분포 PSI(output drift) 3가지를 확인하고, 문제가 있으면 `[WARN]`/`[ALARM]`을 찍고 비정상 종료코드(1/2)를 반환한다 — cron 감시 도구가 이 종료코드로 사람에게 알릴 수 있다. |

---

## 3. UT 실행 결과 해설 (H-6)

```
=======================================================================
 3. CHECKER : UE STATE
-----------------------------------------------------------------------
Checker UE State (callId: 80)
+ use Ue State Check ( name: ptCi->usage, ue: 1/0x1, val:1/0x1 )
+ aiInfo::lastInferTimerSec >= 1.0 ( ue: 1.0 )
+ aiInfo::lastInferTimerSec <= 3.0 ( ue: 1.0 )
[       OK ] .../AI_matrix_E_rsrp_high (179 ms)
[----------] 18 tests from test_SaRrcInactive_Json (3169 ms total)
[==========] 43 tests from 6 test suites ran. (4816 ms total)
[  PASSED  ] 43 tests.
[VccbBtMain.cpp   164] RUN_ALL_TESTS: 0
```

이 로그는 C++ 단위테스트 프레임워크인 **gtest**가 찍는 출력이다. `test_SaRrcInactive_Json`이라는 스위트(suite, 테스트 묶음)가 JSON 파일(`test_SaRrcInactiveAI.json`)에 적힌 시나리오들을 하나씩 재생하면서, 시나리오가 끝난 뒤 "실제 UE 상태가 JSON에 적어둔 기대값과 맞는지"를 **checker**가 검사한다.

| 줄 | 실제 동작 | 쉬운 설명 |
|---|---|---|
| `Checker UE State (callId: 80)` | 시나리오 안에서 만들어진 가상 UE 하나(`callId=80`이라는 내부 식별자)의 최종 상태를 검사하는 단계로 들어간다는 표시. | `callId`는 이 테스트 하니스가 시뮬레이션 중인 UE 하나하나에 붙이는 번호다(실제 무선 규격 용어가 아니라 테스트 코드 내부 식별자). 이 UE에 대한 검증을 시작한다는 로그. |
| `use Ue State Check ( name: ptCi->usage, ue: 1/0x1, val:1/0x1 )` | C++ 구조체 `ptCi->usage`(포인터 `ptCi`가 가리키는 UE 컨텍스트의 `usage` 필드) 값을 읽어서, JSON에 적힌 기대값과 비교한다. `ue: 1/0x1`=실제 관측값, `val:1/0x1`=기대값(둘 다 10진수/16진수로 표기). | "이 UE가 지금 활성 상태(usage=1)인가"를 확인하는 것. 값이 서로 일치하므로(1==1) 이 항목은 통과. |
| `aiInfo::lastInferTimerSec >= 1.0 ( ue: 1.0 )` | JSON의 `aiInfo:{lastInferTimerSecMin}` 값(하한)과, 실제로 `ptCi->aiCtx.lastInferTimerSec`에 저장된 값을 비교한다. `ue: 1.0` = 실제 관측값이 1.0이라는 뜻. | `lastInferTimerSec`은 **AI가 마지막 추론(Point①)에서 계산해 E1AP 메시지에 실어 보낸 InactivityTimer 값(초)**이다. 여기서는 "이 값이 최소 1.0초 이상이어야 한다"는 규칙을 검사 중이고, 실제로 1.0이 나와서 통과(`>=`이므로 1.0도 통과). |
| `aiInfo::lastInferTimerSec <= 3.0 ( ue: 1.0 )` | 같은 값을 JSON의 `lastInferTimerSecMax`(상한, 3.0)와 비교. | 이 테스트 케이스(`rsrp_high`)는 기대 범위가 `[1,3]`초로 좁게 잡혀 있다 — 이건 **모델이 "RELEASE"로 판단한 경로**라는 뜻이다(뒤에서 설명하는 resume_prob<0.6 → 타이머는 확장되지 않고 최솟값 1초 그대로 유지됨). 관측값 1.0이 [1,3] 범위 안에 들어서 통과. |
| `[ OK ]` | 이 개별 테스트 케이스(`AI_matrix_E_rsrp_high`)의 모든 checker 항목이 통과했다는 gtest 표시, 괄호 안은 실행 소요시간(179ms). | 초록 불(pass) 하나. |
| `18 tests from test_SaRrcInactive_Json` | 이 스위트 하나에 포함된 개별 테스트 케이스 수(JSON 파일의 `testcases` 배열 길이 = base 4개 + `gen_ai_matrix_tc.py`가 만든 matrix 14개). | AI 관련 시나리오만 모아놓은 테스트 묶음의 크기. |
| `43 tests from 6 test suites` | 이 프로그램(`UT_VCCB.i686`) 전체에 등록된 모든 스위트(6개)의 테스트 케이스 총합. | AI 스위트(18개)뿐 아니라 VCCB의 다른 기능(핸드오버, 세션관리 등)을 검사하는 스위트들까지 합친 전체 규모. |
| `[ PASSED ] 43 tests.` | 43개 테스트 케이스가 **전부** 실패 없이 통과했다는 gtest의 최종 판정. | ★**전체 회귀(regression) 없음 확인**★ — AI 로직뿐 아니라 기존 VCCB 기능도 이번 변경(재학습)으로 망가지지 않았다는 뜻. |
| `RUN_ALL_TESTS: 0` | gtest 메인 함수(`RUN_ALL_TESTS()`)의 반환값. 0=성공, 0이 아니면 실패한 테스트가 있었다는 뜻(유닉스 프로세스 종료코드 규칙과 동일). | CI/스크립트가 "성공했는지"를 자동으로 판단할 때 보는 숫자. 사람이 로그를 안 읽어도 `echo $?`로 이 값만 확인하면 pass/fail을 알 수 있다. |

> `lastInferTimerSec` 자세한 의미: Point①에서 AI가 계산해 E1AP에 설정한 InactivityTimer 값(초, 런타임 전용). KEEP이면 확장된 큰 값(예 35초), RELEASE면 MIN=1. UT는 `aiInfo:{lastInferTimerSecMin/Max}`로 이 값의 범위를 assert. (메모리 [[project-ai-matrix-test-suite]])

> **재학습 후 이 band가 안 맞아 실패하면** → H-5b(`gen_ai_matrix_tc.py` 재실행)로 보정. 실패 메시지의 `is XX`가 새 모델의 실제 timer 값이므로 `skip:1`로 감추지 말 것. (2026-07-06 A_primary5qi_5 / E_rsrp_low가 이 경로로 [30,40] 보정됨)

---

## 4. PNG 그림 해석법

### 4-1. `train_report_full.png` — base 학습 4-패널

**(1) 좌상 — 학습 곡선** `학습 곡선 — mode=full`
- 파랑=train loss, **빨강=val loss** (과적합 감시 핵심)
- 초록/주황 점선=val BCE(prob) / val SmoothL1(time)
- 회색 세로점선=best 에폭
- 읽는 법: 빨강이 내려가다 평평 → 정상. 다시 오르면 과적합.

**(2) 우상 — 예측 vs 실제** `예측 vs 실제 (양성 N건) — MAE / bias`
- x축=실제 resume_time, y축=예측
- 빨강 점선 `y=x`=완벽선
- **점들이 이 선에 몰릴수록 정확.**
- 선 위=과대예측, 아래=과소예측

**(3) 좌하 — 분류 분리도** `분류 분리도 — AUROC / F1 / ECE`
- 초록=실제 양성(resume<60s), 빨강=실제 음성
- 검정 점선=KEEP 임계값 0.6
- **초록은 오른쪽(prob↑), 빨강은 왼쪽(prob↓)으로 갈라져야 좋음**

**(4) 우하 — MAE vs baseline** `회귀 MAE vs baseline`
- 막대: model(파랑) vs naive/persistence(회색)
- **model 막대가 제일 낮아야 함** (낮을수록 예측 정확)

### 4-2. `train_report_finetune.png` — fine-tune 4-패널

`train_report_full.png`와 같은 4-패널이되 **fine-tune 관점 추가**:

- 패널 (1): 에폭이 15개(짧음), `mode=finetune (freeze_gru=True)`.
- 패널 (4): **보라색 `before FT` 막대**가 추가되어 `after FT`와 나란히. 제목에 `AUROC 0.962 → 0.963` 델타 표기.
  → **before ≥ after로 나빠지지 않고, 조금이라도 개선/유지되는지** 한눈에 확인하는 그림.

### 4-3. `outlier_report.png` — 이상치 필터 전/후 (이번 실행은 미생성)

- preprocess.py가 **행을 실제로 버렸을 때만**(`dropped > 0`) 생성. 이번엔 base 데이터라 drop=0 → **파일 없음**(정상).
- **실측 CSV 투입 시** 생성되면: 좌=label duration 분포(빨강 필터 전 / 초록 필터 후), 우=AMBR 분포. 빨강만 있고 초록이 없는 꼬리 = 걸러낸 이상치. **초록이 원 분포를 심하게 깎으면 필터가 과함**(IQR k 재조정 검토).

### 4-4. (선택) `eval_*.png` — evaluate.py 산출

- 이번 파이프라인에선 실행 안 함(H-4에서 `evaluate.py`는 선택 단계). 돌리면 동일 지표의 상세 플롯 + `eval_report.txt` 생성. 해석 기준은 §2-4 지표와 동일.

---

## 5. "건강한 실행" 빠른 체크리스트

한 판 돌린 뒤 아래를 위→아래로 훑으면 이상 여부를 5초에 판단할 수 있다.

| 항목 | 정상 기준 | 상태 |
|---|---|---|
| preprocess X_stat 열 | 14 | ✅ |
| preprocess X_seq 차원 | (N, 10, 2) | ✅ |
| H-3 sample 증가 | shift.csv 반영 | ✅ |
| train [verdict] | naive/persist 모두 Y | ✅ |
| AUROC / F1 | > 0.90 / > 0.85 | ✅ |
| ECE | < 0.03 | ✅ |
| fine-tune B/A | 동일 아님 | ✅ |
| export params | 1,266 | ✅ |
| export prob@mean | 0.6~0.7 | ✅ |
| UT result | 43/43 PASSED | ✅ |

**장애 진단**:
- 4/5/6 실패 → 데이터/학습 문제 (§2-4 지표 재확인)
- 3 실패 → real/ 미정리 (§2-1 cleanup 재실행)
- 8 실패 → 구조 불일치 (LOCKSTEP 확인)
- 10 실패 → H-5b bound 보정 필요 (§3 또는 메모리 [[project-part-h-pipeline-rules]] H-5b)

---

## 관련 문서 / 메모리

- 절차서: `VCCB_AI_Verification_Runbook.md` (Part H = 이 실행의 순서)
- 설계: `VCCB_AI_Integration_DetailedDesign.md`, 상태: `VCCB_AI_Implementation_Status.md`
- 메모리: [[project-part-h-pipeline-rules]], [[feedback-retrain-preprocess-snapshot]], [[project-ai-matrix-test-suite]], [[project-vccb-ai-overview]]
