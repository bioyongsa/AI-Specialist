"""
preprocess.py
Build GRU training data (dataset.npz) for the VCCB RRC State Predictor.

DESIGN: Method A + Method B combined
  - Method B (real ranges): load + merge CSV(s) produced by the logger
      (#ifdef VCCB_AI_DATA_COLLECT in VccbAiClient.cpp).  Each Resume event
      emits one row whose features are EXACTLY what buildAiInput() builds.
      Point at a single file (VCCB_AI_DATA_FILE) or a whole directory of
      accumulated runs (VCCB_AI_DATA_DIR, every *.csv merged + de-duped).
  - Method A (synthetic): generate additional samples using 3GPP domain rules
      so the GRU has enough volume + sequence diversity to learn from.

  The run prints the real-vs-synthetic ratio so Phase-2 progress (more real
  data, less synthetic dependence) is visible at a glance.

CRITICAL CONTRACT
  The feature layout, ordering, and *raw* scaling MUST match
  VccbAiClient.cpp::buildAiInput() and VccbAiEngine.h exactly, otherwise the
  trained weights are meaningless at inference time.

  seq[t][0] = durationSec      (raw seconds)
  seq[t][1] = resumeCause      (raw int, NR-RRC ResumeCause)
  stat[0]   = primary_5qi              (raw, e.g. 1/2/5/9)
  stat[1]   = drb_count                (raw count)
  stat[2]   = ue_ambr_dl_mbps          (raw Mbps)
  stat[3]   = has_gbr_flow             (0/1)
  stat[4]   = handover_count_24h       (successful SA handovers in 24h window;
                                        mobility proxy, 0..VCCB_AI_HISTORY_LEN)
  stat[5]   = cn_stationary            (currently always 0.0)
  stat[6]   = last_rsrp                (raw 0..127, SSB-RSRP index)
  stat[7]   = secondary_5qi            (5QI of the first active DRB whose 5QI
                                        differs from primary_5qi; 0 = single
                                        service-class session. Mixed sessions,
                                        e.g. VoNR 5QI=1 + eMBB data 5QI=9.)
  stat[8]   = rna_cell_count           (raw, numInactTAI)
  stat[9]   = avg_inactive_duration_sec (raw seconds, == avgDurationSec)
  stat[10]  = resume_count_24h         (raw count)
  stat[11]  = resume_success_rate      (0..1)
  stat[12]  = hour_of_day              (0..23)
  stat[13]  = day_of_week              (0..6)
  (Former stat[12] avg_inactive_duration duplicated stat[9] and was removed;
   STATIC_INPUT is now 14, hour/dow shifted from 13/14 to 12/13.)

Labels
  Y_time = realized next INACTIVE duration in SECONDS (raw).
           C++ uses output.resume_time directly as seconds:
             timer = resume_time * VCCB_AI_MARGIN_FACTOR
  Y_prob = 1.0 if the UE resumed "soon" (duration < PROB_HORIZON_SEC) else 0.0
           -> high prob == keep/defer release was the correct decision.

Output: dataset.npz  (X_seq, X_stat, Y_prob, Y_time)
"""

import os
import glob
import math
import numpy as np

import termplot

try:
    import pandas as pd
except ImportError:
    pd = None  # only needed when a gtest CSV is present

# ──────────────────────────────────────────────
# Paths / config
# ──────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dataset.npz")

# gtest / live logger output (Method B). Two ways to point at real CSVs:
#   VCCB_AI_DATA_FILE : a single CSV (back-compat default).
#   VCCB_AI_DATA_DIR  : a directory — every *.csv inside is merged (one file per
#                       collection run / PC, appended over time). Phase-2 2-1.
# Both may be set; all discovered CSVs are merged + de-duplicated together.
GTEST_CSV   = os.environ.get(
    "VCCB_AI_DATA_FILE",
    os.path.join(SCRIPT_DIR, "vccb_ai_dataset.csv"),
)
GTEST_DIR   = os.environ.get("VCCB_AI_DATA_DIR", os.path.join(SCRIPT_DIR, "real"))

# Number of synthetic samples to generate (Method A).
N_SYNTHETIC = int(os.environ.get("VCCB_AI_SYNTH_N", "20000"))

# ── Outlier handling for REAL (Method B) rows ──
# Field-collected CSVs contain logging artifacts (timestamp glitches -> 0s or
# impossible-huge labels, mis-parsed AMBR, stale persisted EMA). These are
# removed BEFORE training so a handful of corrupt rows cannot dominate the
# regression loss — especially important for small fine-tuning batches.
#   VCCB_AI_OUTLIER_MODE : iqr (default) | pctl | off
#     iqr  : domain rules + IQR fence on log1p(label) and log1p(ambr) (k below)
#     pctl : domain rules + keep [0.5%, 99.5%] percentile band of the label
#     off  : no filtering (previous behaviour)
#   VCCB_AI_OUTLIER_IQR_K: IQR fence multiplier (default 3.0 = only far tails)
# Synthetic (Method A) rows are generated clean and are never filtered.
# A before/after report is printed and plotted to outlier_report.png.
OUTLIER_MODE  = os.environ.get("VCCB_AI_OUTLIER_MODE", "iqr").lower()
OUTLIER_IQR_K = float(os.environ.get("VCCB_AI_OUTLIER_IQR_K", "3.0"))
OUTLIER_PNG   = os.path.join(SCRIPT_DIR, "outlier_report.png")

# Hard domain-validity ceilings (3GPP/TS 38.413 timer range + collector clamps).
LABEL_MIN_SEC  = 1.0      # <1s = suspend/resume stamp race (C++ now skips these)
LABEL_MAX_SEC  = 3600.0   # generator/timer ceiling; beyond = clock glitch
AVGDUR_MAX_SEC = 86400.0  # persisted EMA can never legitimately exceed 24h

SEED = 42

# ──────────────────────────────────────────────
# Constants (must match VccbAiEngine.h)
# ──────────────────────────────────────────────
SEQ_LEN      = 10   # VCCB_AI_HISTORY_LEN
SEQ_INPUT    = 2    # VCCB_AI_SEQ_INPUT
STATIC_INPUT = 14   # VCCB_AI_STATIC_INPUT

# Label policy: UE resuming within this many seconds counts as "imminent resume".
PROB_HORIZON_SEC = float(os.environ.get("VCCB_AI_PROB_HORIZON", "60"))

# resumeCause encoding — the ML 0..5 codes (VCCB_AI_CAUSE_* in VccbAiTypes.h).
# NOTE: these are NOT the raw over-the-air NRRRC_ResumeCause enum values (ASN.1
# order differs: mo_Data=4, mt_Access=2, ...). The C++ side converts via
# FnVCCBAiClient_MapResumeCause() before storing into aiCtx.history[] or
# emitting data-collect CSV rows, so every CSV reaching this script is already
# in this encoding. LOCKSTEP: VccbAiClient.cpp map / synth_data_gen.py /
# gen_realish_csv.py / test_SaRrcInactiveAI.cpp genEvalSet.
CAUSE_EMERGENCY = 0   # treated as "prediction failure" by resumeSuccessRate
CAUSE_MO_DATA   = 1
CAUSE_MO_SIGNAL = 2
CAUSE_MT_ACCESS = 3
CAUSE_RNA_UPDATE = 4
CAUSE_OTHERS    = 5   # catch-all for every other NRRRC cause

# ──────────────────────────────────────────────
# 3GPP domain knowledge: 5QI traffic-class profiles
#   mean/std of inactive duration (sec), GBR, typical resumeCause
# ──────────────────────────────────────────────
# 5QI reference (TS 23.501 Table 5.7.4-1):
#   1  : conversational voice (GBR)
#   2  : conversational video (GBR)
#   5  : IMS signalling (non-GBR)
#   7  : voice/video/interactive (non-GBR)
#   8/9: default bearer, buffered streaming / TCP (non-GBR, eMBB)
FIVEQI_PROFILES = {
    1: dict(dur_mean=20,  dur_std=10,  gbr=1, cause=CAUSE_MO_SIGNAL, drb=(1, 2), ambr=(50, 300)),
    2: dict(dur_mean=30,  dur_std=15,  gbr=1, cause=CAUSE_MO_DATA,   drb=(1, 3), ambr=(100, 600)),
    5: dict(dur_mean=45,  dur_std=25,  gbr=0, cause=CAUSE_MO_SIGNAL, drb=(1, 2), ambr=(10, 100)),
    7: dict(dur_mean=90,  dur_std=45,  gbr=0, cause=CAUSE_MT_ACCESS, drb=(1, 3), ambr=(50, 500)),
    8: dict(dur_mean=150, dur_std=80,  gbr=0, cause=CAUSE_MO_DATA,   drb=(1, 4), ambr=(100, 1000)),
    9: dict(dur_mean=200, dur_std=110, gbr=0, cause=CAUSE_MO_DATA,   drb=(1, 6), ambr=(100, 2000)),
}
FIVEQI_CHOICES = list(FIVEQI_PROFILES.keys())

MAX_DUR_SEC = 86400.0  # clip ceiling


# ──────────────────────────────────────────────
# Method B: load + merge real logger CSV(s)
# ──────────────────────────────────────────────
def discover_real_csvs():
    """Collect every real CSV path from VCCB_AI_DATA_FILE + VCCB_AI_DATA_DIR/*.csv."""
    paths = []
    if GTEST_CSV and os.path.exists(GTEST_CSV):
        paths.append(GTEST_CSV)
    if GTEST_DIR and os.path.isdir(GTEST_DIR):
        paths += sorted(glob.glob(os.path.join(GTEST_DIR, "*.csv")))
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp); uniq.append(p)
    return uniq


def load_real_csvs(paths):
    """
    Merge one or more logger CSVs into (X_seq, X_stat, label_dur, label_cause)
    or None if nothing usable is found.

    Schema produced by aiDataCollectEmit():
        callId,
        seq0_dur,seq0_cause, ... seq9_dur,seq9_cause,
        stat0 ... stat14,
        label_durationSec, label_resumeCause
    """
    if not paths:
        return None
    if pd is None:
        raise RuntimeError("pandas required to read logger CSV; pip install pandas")

    frames = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception as e:  # tolerate a partial/locked file mid-collection
            print(f"[Method B]   skip {os.path.basename(p)} ({e})")
            continue
        if len(d):
            print(f"[Method B]   + {len(d):,} rows  ({p})")
            frames.append(d)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    if len(df) == 0:
        return None

    n = len(df)
    X_seq  = np.zeros((n, SEQ_LEN, SEQ_INPUT), dtype=np.float32)
    for t in range(SEQ_LEN):
        X_seq[:, t, 0] = df[f"seq{t}_dur"].to_numpy(dtype=np.float32)
        X_seq[:, t, 1] = df[f"seq{t}_cause"].to_numpy(dtype=np.float32)

    X_stat = np.zeros((n, STATIC_INPUT), dtype=np.float32)
    for i in range(STATIC_INPUT):
        X_stat[:, i] = df[f"stat{i}"].to_numpy(dtype=np.float32)

    label_dur   = df["label_durationSec"].to_numpy(dtype=np.float32)
    label_cause = df["label_resumeCause"].to_numpy(dtype=np.int32)

    print(f"[Method B] real CSV     : {n:,} unique samples from {len(frames)} file(s)")
    return X_seq, X_stat, label_dur, label_cause


# ──────────────────────────────────────────────
# Outlier filter (real/Method-B rows only)
# ──────────────────────────────────────────────
def _iqr_fence(values, k):
    """Return (lo, hi) fence in the given (already transformed) space."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def filter_outliers(X_seq, X_stat, label_dur, label_cause):
    """Drop rows with collection artifacts. Returns filtered arrays + a report
    dict {rule_name: dropped_count, ...} and keeps a copy of the pre-filter
    label/ambr arrays for the before/after report plot."""
    n0 = len(label_dur)
    report = {}
    keep = np.ones(n0, dtype=bool)

    def apply(rule, mask_bad):
        nonlocal keep
        newly = mask_bad & keep
        report[rule] = int(newly.sum())
        keep &= ~mask_bad

    # ── domain-validity rules (always applied unless mode=off) ──
    finite = (np.isfinite(label_dur)
              & np.isfinite(X_stat).all(axis=1)
              & np.isfinite(X_seq).all(axis=(1, 2)))
    apply("non_finite", ~finite)
    apply(f"label<{LABEL_MIN_SEC:g}s (stamp race)", label_dur < LABEL_MIN_SEC)
    apply(f"label>{LABEL_MAX_SEC:g}s (clock glitch)", label_dur > LABEL_MAX_SEC)
    apply("rsrp out of [0,127]", (X_stat[:, 6] < 0) | (X_stat[:, 6] > 127))
    apply(f"avg_dur>{AVGDUR_MAX_SEC:g}s (stale EMA)", X_stat[:, 9] > AVGDUR_MAX_SEC)
    apply("seq_dur>86400s", (X_seq[:, :, 0] > 86400.0).any(axis=1))

    # ── statistical rules on the survivors ──
    if OUTLIER_MODE == "iqr" and keep.sum() > 50:
        log_dur = np.log1p(label_dur)
        lo, hi = _iqr_fence(log_dur[keep], OUTLIER_IQR_K)
        apply(f"label IQR fence (k={OUTLIER_IQR_K:g}, log)",
              (log_dur < lo) | (log_dur > hi))
        log_ambr = np.log1p(np.maximum(X_stat[:, 2], 0.0))
        lo, hi = _iqr_fence(log_ambr[keep], OUTLIER_IQR_K)
        apply(f"ambr IQR fence (k={OUTLIER_IQR_K:g}, log)",
              (log_ambr < lo) | (log_ambr > hi))
        # stale persisted EMA (stat[9]): plausible-range corruption that the
        # 24h domain ceiling cannot catch (e.g. 10000..86400s) — fence it
        # statistically like the label, upper side only (0 = cold start, valid).
        log_avg = np.log1p(np.maximum(X_stat[:, 9], 0.0))
        _, hi = _iqr_fence(log_avg[keep], OUTLIER_IQR_K)
        apply(f"avg_dur IQR fence (k={OUTLIER_IQR_K:g}, log, upper)",
              log_avg > hi)
    elif OUTLIER_MODE == "pctl" and keep.sum() > 50:
        lo, hi = np.percentile(label_dur[keep], [0.5, 99.5])
        apply("label pctl band [0.5%,99.5%]",
              (label_dur < lo) | (label_dur > hi))

    dropped = n0 - int(keep.sum())
    print(f"[outlier] mode={OUTLIER_MODE}  real rows {n0:,} -> {int(keep.sum()):,} "
          f"(dropped {dropped:,} = {100.0*dropped/max(n0,1):.2f}%)")
    for rule, cnt in report.items():
        if cnt:
            print(f"[outlier]   - {rule:<32}: {cnt:,}")

    before = {"label": label_dur.copy(), "ambr": X_stat[:, 2].copy()}
    return (X_seq[keep], X_stat[keep], label_dur[keep], label_cause[keep],
            report, before)


def plot_outlier_report(before, label_after, ambr_after, report):
    """2-panel before/after PNG so the filtering effect is visually auditable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "C:/Windows/Fonts/malgun.ttf"):
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
                matplotlib.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"[outlier] plot skipped ({e})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.logspace(0, np.log10(90000), 60)

    ax = axes[0]
    ax.hist(np.maximum(before["label"], 1.0), bins=bins, alpha=0.55,
            color="#dc2626", label=f"필터 전 (n={len(before['label']):,})")
    ax.hist(np.maximum(label_after, 1.0), bins=bins, alpha=0.75,
            color="#16a34a", label=f"필터 후 (n={len(label_after):,})")
    ax.axvline(LABEL_MAX_SEC, color="#dc2626", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("label_durationSec (log)"); ax.set_ylabel("rows")
    ax.set_title("라벨 duration 분포 — outlier 필터 전/후", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    bins2 = np.logspace(-1, 7, 60)
    ax.hist(np.maximum(before["ambr"], 0.1), bins=bins2, alpha=0.55,
            color="#dc2626", label="필터 전")
    ax.hist(np.maximum(ambr_after, 0.1), bins=bins2, alpha=0.75,
            color="#16a34a", label="필터 후")
    ax.set_xscale("log")
    ax.set_xlabel("ue_ambr_dl_mbps (log)"); ax.set_ylabel("rows")
    ax.set_title("AMBR 분포 — outlier 필터 전/후", fontsize=11)
    ax.legend(fontsize=9)

    total_drop = sum(report.values())
    fig.suptitle(f"preprocess.py outlier report — mode={OUTLIER_MODE}, "
                 f"dropped {total_drop:,} rows", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTLIER_PNG, dpi=150)
    plt.close(fig)
    print(f"[outlier] report plot -> {OUTLIER_PNG}")


def termplot_outlier_report(before, label_after, ambr_after, report):
    """Terminal ASCII mirror of plot_outlier_report() -- same before/after
    label and AMBR distributions, printed directly to this shell."""
    total_drop = sum(report.values())
    termplot.rule(f"outlier report (terminal) — mode={OUTLIER_MODE}, "
                  f"dropped {total_drop:,} rows", width=64)
    termplot.bar_chart(
        [(rule_name, cnt) for rule_name, cnt in report.items() if cnt],
        fmt="{:,}", title="dropped rows by rule")
    termplot.histogram(before["label"], bins=14, log=True,
                       title="label_durationSec — BEFORE filter (log-scaled bins)")
    termplot.histogram(label_after, bins=14, log=True,
                       title="label_durationSec — AFTER filter (log-scaled bins)")


# ──────────────────────────────────────────────
# Method A: 3GPP domain-rule synthetic generation
# ──────────────────────────────────────────────
def sample_duration(profile, rng) -> float:
    """Log-normal inactive duration from a 5QI profile (seconds)."""
    m, s = profile["dur_mean"], profile["dur_std"]
    sigma2 = math.log(1 + (s / m) ** 2)
    mu     = math.log(m) - sigma2 / 2
    d = rng.lognormal(mu, math.sqrt(sigma2))
    return float(np.clip(d, 1.0, MAX_DUR_SEC))


def gen_synthetic(n: int, rng):
    X_seq  = np.zeros((n, SEQ_LEN, SEQ_INPUT), dtype=np.float32)
    X_stat = np.zeros((n, STATIC_INPUT),       dtype=np.float32)
    label_dur   = np.zeros(n, dtype=np.float32)
    label_cause = np.zeros(n, dtype=np.int32)

    # secondary_5qi (stat[7]): mixed service-class model. ~40% of samples carry a
    # 2nd DRB with a *different* 5QI (e.g. VoNR/video + default data). Decisions
    # are drawn from an INDEPENDENT rng so the primary feature/label stream is
    # byte-for-byte unchanged (keeps the Phase-1 eval distribution stable).
    sec_rng   = np.random.default_rng(SEED + 777)
    mix_flags = sec_rng.random(n) < 0.40
    data_comp = sec_rng.choice([1, 2], size=n)   # voice/video companion when primary is data
    # handover_count_24h (stat[4]): mobility proxy. Poisson(mean 1) capped at the
    # 24h ring size -> most UEs low-mobility (0-2), a tail of high-mobility movers.
    ho_counts = np.minimum(sec_rng.poisson(1.0, size=n), SEQ_LEN)

    for k in range(n):
        five_qi = rng.choice(FIVEQI_CHOICES)
        prof    = FIVEQI_PROFILES[five_qi]
        cause   = prof["cause"]

        drb_count   = int(rng.integers(prof["drb"][0], prof["drb"][1] + 1))
        ue_ambr_dl  = float(rng.uniform(*prof["ambr"]))
        has_gbr     = float(prof["gbr"])
        # SSB-RSRP index 0..127 (higher = stronger). Centre around mid, wide.
        last_rsrp   = float(np.clip(rng.normal(75, 25), 0, 127))
        rna_cells   = float(rng.integers(1, 17))
        succ_rate   = float(np.clip(rng.normal(0.85, 0.15), 0.0, 1.0))
        count_24h   = int(rng.integers(0, 11))
        hour        = float(rng.integers(0, 24))
        dow         = float(rng.integers(0, 7))

        # ---- secondary_5qi (mirror buildAiInput: first DRB 5QI != primary, else 0) ----
        if mix_flags[k]:
            secondary_5qi = int(data_comp[k]) if five_qi in (8, 9) else 9
            drb_count     = max(drb_count, 2)   # a 2nd DRB must exist to carry it
        else:
            secondary_5qi = 0

        # ---- sequence history: 0..SEQ_LEN past resumes (cold start possible) ----
        hist_len = int(rng.integers(0, SEQ_LEN + 1))
        hist_durs = []
        for t in range(hist_len):
            d = sample_duration(prof, rng)
            X_seq[k, t, 0] = d
            X_seq[k, t, 1] = float(cause)
            hist_durs.append(d)
        avg_dur = float(np.mean(hist_durs)) if hist_durs else prof["dur_mean"]

        # ---- static feature vector (raw, buildAiInput order) ----
        X_stat[k, 0]  = float(five_qi)
        X_stat[k, 1]  = float(drb_count)
        X_stat[k, 2]  = ue_ambr_dl
        X_stat[k, 3]  = has_gbr
        X_stat[k, 4]  = float(ho_counts[k])  # handover_count_24h (mobility proxy)
        X_stat[k, 5]  = 0.0               # cn_stationary           (unused)
        X_stat[k, 6]  = last_rsrp
        X_stat[k, 7]  = float(secondary_5qi)  # secondary_5qi (0 = single service-class)
        X_stat[k, 8]  = rna_cells
        X_stat[k, 9]  = avg_dur          # avg_inactive_duration_sec (EMA)
        X_stat[k, 10] = float(count_24h)
        X_stat[k, 11] = succ_rate
        X_stat[k, 12] = hour
        X_stat[k, 13] = dow

        # ---- label: next inactive duration, modulated by domain rules ----
        base = sample_duration(prof, rng)
        # weak signal: poor RSRP -> resume harder/later; high history -> sooner
        rsrp_factor = 1.0 + (75.0 - last_rsrp) / 150.0   # ~0.65 .. 1.5
        hist_factor = 1.0 - 0.3 * (succ_rate - 0.5)       # high success -> shorter
        # service-group differentiation (Nokia): a latency-critical secondary
        # service (5QI 1/2 GBR voice/video alongside data) keeps the session
        # "warmer" -> shorter inactive dwell than a pure-bulk session.
        svc_factor = 0.7 if secondary_5qi in (1, 2) else 1.0
        # mobility (US 11,109,442): higher 24h handover count -> higher
        # reconnection likelihood -> shorter inactive dwell (bounded -30%).
        mob_factor = 1.0 - 0.05 * min(int(ho_counts[k]), 6)
        next_dur = float(np.clip(
            base * rsrp_factor * hist_factor * svc_factor * mob_factor, 1.0, MAX_DUR_SEC))

        label_dur[k]   = next_dur
        label_cause[k] = cause

    print(f"[Method A] synthetic    : {n:,} samples (3GPP 5QI rules)")
    return X_seq, X_stat, label_dur, label_cause


# ──────────────────────────────────────────────
# Label derivation (shared by both sources)
# ──────────────────────────────────────────────
def make_labels(label_dur: np.ndarray):
    Y_time = label_dur.astype(np.float32)                       # raw seconds
    Y_prob = (label_dur < PROB_HORIZON_SEC).astype(np.float32)  # imminent resume
    return Y_prob, Y_time


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 64)
    print("VCCB RRC State Predictor — training data builder (Method A + B)")
    print("=" * 64)
    rng = np.random.default_rng(SEED)

    seq_parts, stat_parts, dur_parts = [], [], []
    n_real = 0

    real_paths = discover_real_csvs()
    real = load_real_csvs(real_paths)
    if real is not None:
        s, st, d, c = real
        if OUTLIER_MODE != "off":
            s, st, d, c, rep, before = filter_outliers(s, st, d, c)
            if sum(rep.values()) > 0:
                plot_outlier_report(before, d, st[:, 2], rep)
                termplot_outlier_report(before, d, st[:, 2], rep)
        n_real = len(d)
        seq_parts.append(s); stat_parts.append(st); dur_parts.append(d)
    else:
        print(f"[Method B] real CSV     : none found "
              f"(file={GTEST_CSV}, dir={GTEST_DIR}) — synthetic only")

    if N_SYNTHETIC > 0:
        s, st, d, _c = gen_synthetic(N_SYNTHETIC, rng)
        seq_parts.append(s); stat_parts.append(st); dur_parts.append(d)

    if not seq_parts:
        raise RuntimeError("No data: provide a logger CSV or set VCCB_AI_SYNTH_N > 0")

    X_seq  = np.concatenate(seq_parts,  axis=0)
    X_stat = np.concatenate(stat_parts, axis=0)
    label_dur = np.concatenate(dur_parts, axis=0)

    # shuffle real + synthetic together
    perm = rng.permutation(len(X_seq))
    X_seq, X_stat, label_dur = X_seq[perm], X_stat[perm], label_dur[perm]

    Y_prob, Y_time = make_labels(label_dur)

    np.savez_compressed(
        OUTPUT_FILE,
        X_seq=X_seq, X_stat=X_stat, Y_prob=Y_prob, Y_time=Y_time,
    )

    total = len(X_seq)
    real_ratio = (n_real / total * 100.0) if total else 0.0
    print("-" * 64)
    print(f"Total samples : {total:,}  "
          f"(real {n_real:,} = {real_ratio:.1f}% / synthetic {total - n_real:,})")
    print(f"  resume_prob=1 (resume < {PROB_HORIZON_SEC:.0f}s): {Y_prob.mean()*100:.1f}%")
    print(f"  resume_time   : mean={Y_time.mean():.1f}s  std={Y_time.std():.1f}s")
    termplot.histogram(Y_time, bins=16, log=True,
                       title="final dataset.npz — Y_time (label_durationSec, log-scaled bins)",
                       note="real+synthetic merged")
    print(f"Saved → {OUTPUT_FILE}")
    print(f"  X_seq  {X_seq.shape}  X_stat {X_stat.shape}")
    if n_real == 0:
        print("  NOTE: synthetic-only. Collect real CSVs (-DVCCB_AI_DATA_COLLECT)")
        print(f"        into {GTEST_DIR}/ to raise the real ratio (Phase 2-1).")
    print("Next: python train.py   (or analyze_features.py for signal check)")


if __name__ == "__main__":
    main()
