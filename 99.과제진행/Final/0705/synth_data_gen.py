#!/usr/bin/env python3
# ============================================================
# synth_data_gen.py
# Realistic synthetic training-data generator for the VCCB RRC-state
# (inactivity-timer) GRU model.
# ------------------------------------------------------------
# Why this exists
#   The on-device collector (-DVCCB_AI_DATA_COLLECT, see VccbAiClient.cpp
#   aiDataCollectEmit) emits one CSV row per real Resume event, but real logs
#   are scarce. The previous gtest "sweep" (AiClient_DataCollect_Sweep) filled
#   the gap with an independent cross-product of feature values + heuristic
#   labels, which has three weaknesses:
#     1) features are sampled independently (no per-UE correlation),
#     2) labels are a hand-written formula (not behavioral),
#     3) the sequence (seq[]) has no real temporal dynamics.
#
#   This generator instead simulates a *population of virtual UEs*. Each UE is
#   drawn from a behavioral persona and its INACTIVE/Resume timeline is produced
#   by a renewal process, so:
#     - the per-row label (label_durationSec) = the *next* inactive duration,
#       sampled from the persona's inter-arrival-time (IAT) distribution,
#     - seq[] holds that UE's PRIOR events (oldest-first, zero-padded) exactly
#       as buildAiInput() would have seen them at inference time,
#     - the aggregate stats (avg duration, 24h resume count, success rate) are
#       computed with the SAME EMA recurrences as VccbAiClient.cpp.
#
# Output schema  (MUST match aiDataCollectEmit in VccbAiClient.cpp):
#   callId,
#   seq0_dur,seq0_cause, ... , seq{H-1}_dur,seq{H-1}_cause,
#   stat0, ... , stat13,                 (14 static features; was 15)
#   label_durationSec,label_resumeCause
#
# Usage:
#   python3 synth_data_gen.py --out vccb_ai_dataset.csv --ues 5000 --seed 0
#   # append synthetic rows to a file that already has real/collected rows:
#   python3 synth_data_gen.py --out vccb_ai_dataset.csv --ues 2000 --append
#   # fine-tuning set: distribution-shifted "field" population + realistic
#   # collection noise (outliers) that preprocess.py's filter must handle:
#   python3 synth_data_gen.py --out vccb_ai_finetune.csv --ues 3000 --seed 7 \
#       --profile field --outlier-frac 0.02
#
# LOCKSTEP (design doc SS D-4): the per-event dwell applies ALL FOUR documented
# label factors -- rsrp_factor, hist_factor, svc_factor, mob_factor -- exactly
# as ML/preprocess.py gen_synthetic() and test_SaRrcInactiveAI.cpp genEvalSet()
# do. (Before 2026-07-05 this file only applied svc/mob; the persona-level
# RSRP<->dwell correlation was an accidental confound, not the documented
# per-session effect. analyze_synth_literature.py verifies this empirically.)
#
# Dependencies: Python 3 standard library only (random, math, argparse, os).
# Feed the resulting CSV to the existing training pipeline (design doc App. D),
# then export weights with export_weights.py -> VccbAiWeights.h.
# ============================================================

import argparse
import math
import os
import random

# ---- model contract constants (keep in sync with VccbAiEngine.h) ----
H = 10          # VCCB_AI_HISTORY_LEN
S = 14          # VCCB_AI_STATIC_INPUT (was 15; dropped duplicate stat[12])
WINDOW_SEC = 86400      # VCCB_AI_24H_WINDOW_SEC

# ---- resumeCause codes (CANONICAL: aligned with ML/preprocess.py CAUSE_*) ----
# IMPORTANT: these integers must match preprocess.py / gen_realish_csv.py and the
# NRRRC_ResumeCause family fed to buildAiInput(). Only EMERGENCY is treated as a
# "prediction failure" (resume_success_rate) by the C++ side.
# Reporting (fig05) shows four headline buckets + "others"; any cause that is not
# one of the four headline causes is emitted as CAUSE_OTHERS.
#   (Previously this file used a DIFFERENT, conflicting encoding
#    {1:high_Prio,2:mt_Access,3:mo_Signalling,4:mo_Data,5:rna_Update};
#    it is now unified with preprocess.py.)
CAUSE_EMERGENCY     = 0
CAUSE_MO_DATA       = 1   # reported under "others"
CAUSE_MO_SIGNALLING = 2
CAUSE_MT_ACCESS     = 3
CAUSE_RNA_UPDATE    = 4
CAUSE_OTHERS        = 5   # catch-all (former high_Prio / mo_Data / etc.) -> "others"

# Dataset-wide resumeCause target mix. Sampled per Resume event (independent of
# persona) so the generated CSV matches these proportions regardless of which
# persona produced the event:
#   emergency 6% / mo_Signalling 32% / mt_Access 29% / rna_Update 19% / others 14%
# NOTE: rna_Update events are simulated on the timeline but emit NO row and NO
# history entry (see simulate_ue) because the production Point-3 call site skips
# RNA-update resumes — the emitted mix therefore excludes rna_Update by design.
GLOBAL_CAUSE_MIX = [
    (CAUSE_EMERGENCY,     0.06),
    (CAUSE_MO_SIGNALLING, 0.32),
    (CAUSE_MT_ACCESS,     0.29),
    (CAUSE_RNA_UPDATE,    0.19),
    (CAUSE_OTHERS,        0.14),
]


class Persona:
    """A behavioral archetype: how often this kind of UE resumes, and its QoS /
    radio characteristics."""
    def __init__(self, name, weight, iat_shape, iat_scale,
                 five_qi_choices, drb_range, ambr_dl_mbps, has_gbr_prob,
                 rsrp_mean_std, rna_range, cause_choices,
                 sessions_range, active_gap_sec):
        self.name = name
        self.weight = weight
        # Weibull(shape=beta, scale=alpha); random.weibullvariate(alpha, beta)
        self.iat_shape = iat_shape
        self.iat_scale = iat_scale
        self.five_qi_choices = five_qi_choices      # [(value, prob), ...]
        self.drb_range = drb_range                  # (min, max)
        self.ambr_dl_mbps = ambr_dl_mbps            # (mean, std)
        self.has_gbr_prob = has_gbr_prob
        self.rsrp_mean_std = rsrp_mean_std          # (mean, std) over 0..127
        self.rna_range = rna_range                  # (min, max)
        self.cause_choices = cause_choices          # [(cause, prob), ...]
        self.sessions_range = sessions_range        # (min, max) events per UE
        self.active_gap_sec = active_gap_sec        # (mean) CONNECTED time between resumes


# Population: weights need not sum to 1 (normalized at pick time).
PERSONAS = [
    # name          w     k    scale  5qi                         drb     ambr        gbr  rsrp     rna     causes                                                            sess     gap
    Persona("iot_sensor",   0.45, 1.6,  1800.0, [(9, 0.8), (8, 0.2)],            (1, 2), (1.0, 0.5),  0.00, (98, 8),  (1, 2),
            GLOBAL_CAUSE_MIX,            (4, 20),  600.0),
    Persona("bursty_phone", 0.35, 1.1,  25.0,   [(8, 0.5), (7, 0.3), (9, 0.2)],  (2, 4), (150.0, 80.0), 0.20, (75, 16), (3, 9),
            GLOBAL_CAUSE_MIX,                                  (8, 40),  90.0),
    Persona("fwa_fixed",    0.12, 2.0,  3000.0, [(9, 1.0)],                      (3, 8), (800.0, 250.0), 0.40, (108, 5), (1, 2),
            GLOBAL_CAUSE_MIX,                                       (3, 12),  1200.0),
    Persona("commuter",     0.08, 1.2,  120.0,  [(7, 0.5), (8, 0.5)],            (2, 3), (200.0, 120.0), 0.10, (60, 22), (4, 12),
            GLOBAL_CAUSE_MIX,               (6, 30),  45.0),
]

# "field" profile (--profile field): a plausibly DRIFTED live-network population
# used to emulate freshly collected field data for the fine-tuning cycle
# (design doc SS7: base model = synthetic prior, fine-tune = incremental real
# batches). Differences vs the base lab prior:
#   - more short-dwell smartphone traffic (bursty_phone up, dwell scale down)
#   - more commuters with weaker radio (mobility/RSRP shift)
#   - fewer idle IoT/FWA rows
# This is intentionally a moderate covariate+label shift: big enough that
# fine-tuning has something to learn, small enough that the synthetic prior
# still applies.
PERSONAS_FIELD = [
    Persona("iot_sensor",   0.28, 1.6,  1500.0, [(9, 0.8), (8, 0.2)],            (1, 2), (1.0, 0.5),  0.00, (95, 9),  (1, 2),
            GLOBAL_CAUSE_MIX,            (4, 20),  600.0),
    Persona("bursty_phone", 0.47, 1.1,  18.0,   [(8, 0.5), (7, 0.3), (9, 0.2)],  (2, 4), (180.0, 90.0), 0.25, (72, 17), (3, 9),
            GLOBAL_CAUSE_MIX,                                  (8, 40),  75.0),
    Persona("fwa_fixed",    0.10, 2.0,  2600.0, [(9, 1.0)],                      (3, 8), (850.0, 260.0), 0.40, (107, 5), (1, 2),
            GLOBAL_CAUSE_MIX,                                       (3, 12),  1200.0),
    Persona("commuter",     0.15, 1.2,  90.0,   [(7, 0.5), (8, 0.5)],            (2, 3), (220.0, 130.0), 0.12, (55, 24), (5, 14),
            GLOBAL_CAUSE_MIX,               (6, 30),  40.0),
]

PROFILES = {"base": PERSONAS, "field": PERSONAS_FIELD}


def weighted_choice(choices, rng):
    """choices: list of (value, prob); prob need not be normalized."""
    total = sum(p for _, p in choices)
    r = rng.uniform(0.0, total)
    upto = 0.0
    for value, prob in choices:
        upto += prob
        if r <= upto:
            return value
    return choices[-1][0]


def pick_persona(rng, personas=None):
    if personas is None:
        personas = PERSONAS
    return weighted_choice([(p, p.weight) for p in personas], rng)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def sample_iat(p, rng):
    """Next INACTIVE duration (seconds) = label_durationSec. Clamped to the
    model's valid timer range [1, 3600] (VCCB_AI_TIMER_MIN/MAX_SEC)."""
    d = rng.weibullvariate(p.iat_scale, p.iat_shape)
    return clamp(d, 1.0, 3600.0)


def inject_outlier(stat, rng, dur):
    """Emulate a realistic field-collection artifact on ONE row. Mutates stat
    in place when the artifact is feature-side; returns the (possibly
    corrupted) label duration. Artifact taxonomy mirrors what a live logger
    actually produces:
      - label_spike : suspend/resume timestamp glitch -> impossible duration
                      way past the 3600s timer ceiling (e.g. clock jump)
      - label_zero  : resume raced the suspend stamp -> 0s duration
      - ambr_spike  : mis-parsed AMBR field (unit bug) -> x1000 magnitude
      - avgdur_spike: corrupted persisted EMA restored from a stale entry
    preprocess.py's outlier filter is expected to remove/repair these rows."""
    kind = rng.random()
    if kind < 0.40:
        return rng.uniform(4000.0, 86400.0)          # label_spike
    if kind < 0.65:
        return 0.0                                    # label_zero
    if kind < 0.85:
        stat[2] = stat[2] * 1000.0                    # ambr_spike (Mbps -> bps-like)
        return dur
    stat[9] = rng.uniform(10000.0, 86400.0)           # avgdur_spike
    return dur


def simulate_ue(callid, p, rng, outlier_frac=0.0):
    """Yield CSV rows (list of fields) for one virtual UE's session timeline.

    outlier_frac: probability per emitted row of injecting a realistic
    collection artifact (see inject_outlier) -- used to build fine-tuning sets
    that exercise preprocess.py's outlier filter. 0.0 = clean (default).
    """
    n_sessions = rng.randint(*p.sessions_range)

    # ---- per-UE (mostly static) characteristics ----
    five_qi  = weighted_choice(p.five_qi_choices, rng)
    drb_cnt  = rng.randint(*p.drb_range)
    ambr     = max(0.1, rng.gauss(*p.ambr_dl_mbps))
    has_gbr  = 1.0 if rng.random() < p.has_gbr_prob else 0.0
    rna_cnt  = rng.randint(*p.rna_range)
    rsrp_mu, rsrp_sd = p.rsrp_mean_std

    # secondary_5qi (stat[7]): mixed service-class session model. ~40% of UEs
    # carry a 2nd DRB with a *different* 5QI (e.g. VoNR/video + default data), the
    # signal behind service-group differentiated timers. Must mirror the
    # buildAiInput() extraction "first active DRB whose 5QI != primary_5qi, else 0".
    if rng.random() < 0.40:
        secondary_5qi = rng.choice([1, 2]) if five_qi in (8, 9) else 9
        drb_cnt = max(drb_cnt, 2)   # a 2nd DRB must exist to carry the 2nd class
    else:
        secondary_5qi = 0

    # handover_count_24h (stat[4]): per-UE mobility proxy. Exponential(mean 1)
    # capped at the 24h ring size -> most UEs low-mobility, a high-mobility tail.
    ho_count_24h = min(int(rng.expovariate(1.0)), H)

    # ---- evolving state (mirrors VccbAiClient.cpp recurrences) ----
    hist = []                       # [(dur, cause, ts)] chronological
    ema_avg = 0.0                   # avgDurationSec (EMA a=0.1), cold = 0
    ema_succ = 1.0                  # resumeSuccessRate (EMA a=0.1), cold = 1.0
    t = rng.randint(0, WINDOW_SEC)  # rolling wall-clock (sec)

    # Label factors (LOCKSTEP with preprocess.py gen_synthetic / genEvalSet,
    # design doc SS D-4): label_dwell = base * rsrp_factor * hist_factor
    # * svc_factor * mob_factor.
    # svc/mob depend on per-UE-frozen drivers (secondary_5qi, ho_count_24h) so
    # they are constant across this UE's events; rsrp/hist depend on the
    # PER-SESSION rsrp sample and the EVOLVING success-rate EMA and therefore
    # must be recomputed inside the event loop below.
    svc_factor = 0.7 if secondary_5qi in (1, 2) else 1.0
    mob_factor = 1.0 - 0.05 * min(ho_count_24h, 6)

    for _ in range(n_sessions):
        # --- per-session radio sample FIRST: it feeds both stat[6] and the
        #     label's rsrp_factor (same value, as in preprocess.py) ---
        rsrp = clamp(rng.gauss(rsrp_mu, rsrp_sd), 0.0, 127.0)
        rsrp_factor = 1.0 + (75.0 - rsrp) / 150.0     # ~0.65 .. 1.5 (3GPP RSRP)
        hist_factor = 1.0 - 0.3 * (ema_succ - 0.5)    # high success -> shorter

        dur = clamp(sample_iat(p, rng) * rsrp_factor * hist_factor
                    * svc_factor * mob_factor, 1.0, 3600.0)
        cause = weighted_choice(p.cause_choices, rng)

        # SERVE-PATH MIRROR: the Point-3 call site (StateSaRrcInact.cpp) skips
        # RNA-update resumes entirely — production never pushes them into
        # aiCtx.history[] and the data collector never emits a row for them.
        # So an rna_Update event here advances the timeline (the dwell still
        # elapsed) but produces no row and no history/EMA update. Keeping rna
        # rows/seq entries would create a train/serve distribution mismatch.
        if cause == CAUSE_RNA_UPDATE:
            t += int(dur) + int(max(1.0, rng.expovariate(1.0 / p.active_gap_sec)))
            continue

        hour = (t // 3600) % 24
        wday = (t // 86400) % 7

        # --- seq[]: PRIOR events only, oldest-first, zero-padded (as buildAiInput) ---
        recent = hist[-H:]
        seq = [(0.0, 0.0)] * H
        for i, (d, c, _ts) in enumerate(recent):
            seq[i] = (d, float(c))

        # --- 24h resume count from prior events ---
        rc24 = sum(1 for (_d, _c, ts) in hist if (t - ts) <= WINDOW_SEC)

        stat = [0.0] * S
        stat[0]  = float(five_qi)
        stat[1]  = float(drb_cnt)
        stat[2]  = float(ambr)
        stat[3]  = has_gbr
        stat[4]  = float(ho_count_24h)  # handover_count_24h (mobility proxy)
        stat[5]  = 0.0                 # cn_stationary (not in NGAP today)
        stat[6]  = float(rsrp)
        stat[7]  = float(secondary_5qi)  # secondary_5qi (0 = single service-class)
        stat[8]  = float(rna_cnt)
        stat[9]  = ema_avg             # avg_inactive_duration_sec (== avgDurationSec)
        stat[10] = float(rc24)
        stat[11] = ema_succ
        stat[12] = float(hour)         # (former stat[12] avg_inactive_duration removed)
        stat[13] = float(wday)

        # --- optional collection-artifact injection (fine-tuning realism) ---
        # Corrupts only the EMITTED row, exactly like a field logging glitch;
        # the UE's internal timeline/EMA state stays clean so one bad row does
        # not cascade into every subsequent row of this UE.
        emit_dur = dur
        if outlier_frac > 0.0 and rng.random() < outlier_frac:
            emit_dur = inject_outlier(stat, rng, emit_dur)

        # --- emit row (snapshot taken BEFORE this event mutates state) ---
        row = [str(callid)]
        for d, c in seq:
            row.append("%.6f" % d)
            row.append("%.6f" % c)
        for x in stat:
            row.append("%.6f" % x)
        row.append("%.6f" % emit_dur)
        row.append("%d" % cause)
        yield row

        # --- post-event state update (same order as OnResumeRequest) ---
        if not hist:
            ema_avg = dur
        else:
            ema_avg = 0.9 * ema_avg + 0.1 * dur
        is_success = 0.0 if cause == CAUSE_EMERGENCY else 1.0
        ema_succ = 0.1 * is_success + 0.9 * ema_succ

        hist.append((dur, cause, t))
        # advance wall-clock: CONNECTED gap + this INACTIVE duration
        t += int(dur) + int(max(1.0, rng.expovariate(1.0 / p.active_gap_sec)))


def header():
    cols = ["callId"]
    for tnum in range(H):
        cols.append("seq%d_dur" % tnum)
        cols.append("seq%d_cause" % tnum)
    for i in range(S):
        cols.append("stat%d" % i)
    cols.append("label_durationSec")
    cols.append("label_resumeCause")
    return ",".join(cols) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate realistic synthetic VCCB AI training data.")
    ap.add_argument("--out", default="vccb_ai_dataset_synth.csv", help="output CSV path")
    ap.add_argument("--ues", type=int, default=5000, help="number of virtual UEs")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing file (no header if file already exists)")
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="base",
                    help="persona population: 'base' = lab prior (default), "
                         "'field' = drifted live-network mix for fine-tuning sets")
    ap.add_argument("--outlier-frac", type=float, default=0.0,
                    help="per-row probability of injecting a collection artifact "
                         "(label spike/zero, AMBR/EMA glitch); use ~0.02 for "
                         "fine-tuning sets to exercise preprocess.py's filter")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    personas = PROFILES[args.profile]

    mode = "a" if args.append else "w"
    need_header = True
    if args.append and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        need_header = False

    n_rows = 0
    with open(args.out, mode) as f:
        if need_header:
            f.write(header())
        for callid in range(args.ues):
            p = pick_persona(rng, personas)
            for row in simulate_ue(callid, p, rng, outlier_frac=args.outlier_frac):
                f.write(",".join(row) + "\n")
                n_rows += 1

    print("[synth_data_gen] wrote %d rows from %d virtual UEs -> %s"
          % (n_rows, args.ues, args.out))
    print("[synth_data_gen] profile=%s outlier_frac=%.3f seed=%d"
          % (args.profile, args.outlier_frac, args.seed))


if __name__ == "__main__":
    main()
