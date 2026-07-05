#!/usr/bin/env python3
# ============================================================
# analyze_synth_literature.py
# ------------------------------------------------------------
# Checks how well the RAW synth_data_gen.py output (vccb_ai_dataset.csv)
# reproduces the literature-cited label factors documented in
# VCCB_AI_Integration_DetailedDesign.md §D-4 / §D-4-보충:
#
#   label_dwell = base(primary_5qi) x rsrp_factor x hist_factor
#                 x svc_factor x mob_factor
#
#   rsrp_factor = 1 + (75 - rsrp) / 150          (3GPP TS 38.331/38.304)
#   hist_factor = 1 - 0.3 * (succ - 0.5)         (3GPP TS 38.133)
#   svc_factor  = 0.7 if secondary_5qi in {1,2} else 1.0   (R&S VoNR whitepaper,
#                 5QI1 Conversational Voice / PDB 100ms)
#   mob_factor  = 1 - 0.05 * min(ho_count_24h, 6)          (US 11,109,442, concept
#                 only -- direction disputed, see doc warning)
#
# This is a DIAGNOSTIC, not a training script: it reads the CSV that
# `synth_data_gen.py` produced and empirically measures whether each factor's
# documented relationship is actually present in the generated labels.
#
# Usage:
#   python analyze_synth_literature.py [vccb_ai_dataset.csv]
# Output:
#   synth_literature_check.png   (6-panel figure)
#   synth_literature_check.txt   (numeric summary)
# ============================================================
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_data_gen as sg  # re-simulate with a persona tag for stratified diagnostics
import termplot  # zero-dep terminal ASCII mirror of the PNG panels

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "vccb_ai_dataset.csv")
PNG_PATH = os.path.join(SCRIPT_DIR, "synth_literature_check.png")
PNG_PATH2 = os.path.join(SCRIPT_DIR, "synth_literature_check_stratified.png")
TXT_PATH = os.path.join(SCRIPT_DIR, "synth_literature_check.txt")

STAT_NAMES = [
    "primary_5qi", "drb_count", "ue_ambr_dl_mbps", "has_gbr_flow",
    "handover_count_24h", "cn_stationary", "last_rsrp", "secondary_5qi",
    "rna_cell_count", "avg_inactive_duration_sec", "resume_count_24h",
    "resume_success_rate", "hour_of_day", "day_of_week",
]


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.ptp(x) == 0:
        return float("nan")
    if _HAVE_SCIPY:
        r, _ = spearmanr(x, y)
        return float(r)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def resimulate_with_persona_tag(n_ues=2000, seed=0):
    """Re-run the exact same generator logic (same PERSONAS/simulate_ue), but
    also tag each row with its persona name. NOT written to the production
    CSV/schema -- used only here to control for the persona confound when
    checking rsrp_factor / hist_factor / svc_factor."""
    rng = random.Random(seed)
    rows = []
    for callid in range(n_ues):
        p = sg.pick_persona(rng)
        for row in sg.simulate_ue(callid, p, rng):
            rows.append(row + [p.name])
    cols = ["callId"]
    for t in range(sg.H):
        cols += [f"seq{t}_dur", f"seq{t}_cause"]
    cols += [f"stat{i}" for i in range(sg.S)]
    cols += ["label_durationSec", "label_resumeCause", "persona"]
    df = pd.DataFrame(rows, columns=cols)
    for i, name in enumerate(STAT_NAMES):
        df.rename(columns={f"stat{i}": name}, inplace=True)
    for c in df.columns:
        if c not in ("persona",):
            df[c] = pd.to_numeric(df[c])
    return df


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run: python synth_data_gen.py --out {os.path.basename(CSV_PATH)} --ues 5000 --seed 0"
        )
    df = pd.read_csv(CSV_PATH)
    for i, name in enumerate(STAT_NAMES):
        df.rename(columns={f"stat{i}": name}, inplace=True)
    n = len(df)
    dur = df["label_durationSec"].to_numpy(dtype=np.float64)

    lines = []
    lines.append("=" * 78)
    lines.append("synth_data_gen.py output vs literature-cited label factors (design doc SS D-4)")
    lines.append("=" * 78)
    lines.append(f"dataset : {CSV_PATH}")
    lines.append(f"rows    : {n:,}")
    lines.append("-" * 78)

    # ---- 1) primary_5qi (base duration per 5QI / persona archetype) ----
    fiveqi_group = df.groupby("primary_5qi")["label_durationSec"].agg(["mean", "count"])
    lines.append("[1] primary_5qi -> mean label_durationSec (persona-driven base):")
    for qi, row in fiveqi_group.iterrows():
        lines.append(f"    5QI={int(qi):<3d} mean_dur={row['mean']:8.1f}s  n={int(row['count']):,}")
    lines.append("    NOTE: unlike preprocess.py's FIVEQI_PROFILES table (fixed base per 5QI),")
    lines.append("    synth_data_gen.py's duration comes from the PERSONA's Weibull(iat_scale),")
    lines.append("    and personas overlap across 5QI values (e.g. 5QI=9 appears in")
    lines.append("    iot_sensor[scale=1800s], fwa_fixed[scale=3000s] AND bursty_phone[scale=25s]).")
    lines.append("    So 5QI <-> duration is persona-mediated, not a direct 1:1 base table.")

    # ---- 2) secondary_5qi -> svc_factor (expected x0.7 when in {1,2}) ----
    # NAIVE (whole-population) comparison -- confounded, see note below.
    is_mixed = df["secondary_5qi"].isin([1, 2])
    mean_mixed = dur[is_mixed.to_numpy()].mean()
    mean_single = dur[(~is_mixed).to_numpy()].mean()
    ratio = mean_mixed / mean_single if mean_single else float("nan")

    # STRATIFIED comparison -- secondary_5qi can only be in {1,2} for UEs whose
    # primary_5qi in (8,9) (see synth_data_gen.py simulate_ue). Restricting the
    # "single" comparison group to the SAME primary_5qi subset removes the
    # persona-composition confound (5QI=7 personas have short dwell scales and
    # can never be "mixed", which drags the naive "single" mean down).
    same_qi = df["primary_5qi"].isin([8, 9])
    is_mixed_s = is_mixed & same_qi
    is_single_s = (~is_mixed) & same_qi
    mean_mixed_s = dur[is_mixed_s.to_numpy()].mean()
    mean_single_s = dur[is_single_s.to_numpy()].mean()
    ratio_s = mean_mixed_s / mean_single_s if mean_single_s else float("nan")

    lines.append("-" * 78)
    lines.append("[2] secondary_5qi in {1,2} (svc_factor, documented x0.7):")
    lines.append(f"    [naive, whole population]")
    lines.append(f"    mixed  (secondary in {{1,2}}): mean_dur={mean_mixed:8.1f}s  n={int(is_mixed.sum()):,}")
    lines.append(f"    single (secondary==0)       : mean_dur={mean_single:8.1f}s  n={int((~is_mixed).sum()):,}")
    lines.append(f"    naive ratio mixed/single = {ratio:.3f}  <- CONFOUNDED (see below)")
    lines.append(f"    [stratified: primary_5qi in {{8,9}} only -- secondary in {{1,2}} is IMPOSSIBLE")
    lines.append(f"     for 5QI=7 UEs, so comparing across all 5QI mixes in the persona 5QI=7")
    lines.append(f"     dwell scale into the 'single' baseline and masks svc_factor]")
    lines.append(f"    mixed  (5QI in {{8,9}} & secondary in {{1,2}}): mean_dur={mean_mixed_s:8.1f}s  n={int(is_mixed_s.sum()):,}")
    lines.append(f"    single (5QI in {{8,9}} & secondary==0)       : mean_dur={mean_single_s:8.1f}s  n={int(is_single_s.sum()):,}")
    lines.append(f"    stratified ratio mixed/single = {ratio_s:.3f}  (design doc target: 0.7)")

    # ---- 3) handover_count_24h -> mob_factor (expected 1-0.05*min(ho,6)) ----
    ho_bucket = df["handover_count_24h"].clip(upper=6)
    ho_group = pd.DataFrame({"ho": ho_bucket, "dur": dur}).groupby("ho")["dur"].agg(["mean", "count"])
    lines.append("-" * 78)
    lines.append("[3] handover_count_24h (clipped 6) -> mean duration (mob_factor, documented 1-0.05*ho):")
    base_ho0 = ho_group.loc[0, "mean"] if 0 in ho_group.index else float("nan")
    for ho, row in ho_group.iterrows():
        theo = 1.0 - 0.05 * ho
        empirical_ratio = row["mean"] / base_ho0 if base_ho0 else float("nan")
        lines.append(f"    ho={int(ho)}  mean_dur={row['mean']:8.1f}s  n={int(row['count']):,}"
                     f"   empirical_ratio_vs_ho0={empirical_ratio:.3f}  theory_ratio={theo:.3f}")
    rho_ho = spearman(df["handover_count_24h"], dur)
    lines.append(f"    Spearman(ho_count_24h, duration) = {rho_ho:.3f}  (expect negative if mob_factor applied)")

    # ---- 4) last_rsrp -> rsrp_factor (documented 1+(75-rsrp)/150) ----
    rsrp = df["last_rsrp"].to_numpy(dtype=np.float64)
    rho_rsrp = spearman(rsrp, dur)
    edges = np.quantile(rsrp, np.linspace(0, 1, 7))
    edges = np.unique(edges)
    rsrp_bin = np.digitize(rsrp, edges[1:-1])
    lines.append("-" * 78)
    lines.append("[4] last_rsrp -> mean duration (rsrp_factor, documented 1+(75-rsrp)/150):")
    for b in np.unique(rsrp_bin):
        m = rsrp_bin == b
        lo, hi = rsrp[m].min(), rsrp[m].max()
        lines.append(f"    rsrp in [{lo:5.1f},{hi:5.1f}]  mean_dur={dur[m].mean():8.1f}s  n={int(m.sum()):,}")
    lines.append(f"    Spearman(last_rsrp, duration) = {rho_rsrp:.3f}  (expect negative if rsrp_factor applied)")

    # ---- 5) resume_success_rate -> hist_factor (documented 1-0.3*(succ-0.5)) ----
    succ = df["resume_success_rate"].to_numpy(dtype=np.float64)
    rho_succ = spearman(succ, dur)
    edges2 = np.quantile(succ, np.linspace(0, 1, 7))
    edges2 = np.unique(edges2)
    succ_bin = np.digitize(succ, edges2[1:-1])
    lines.append("-" * 78)
    lines.append("[5] resume_success_rate -> mean duration (hist_factor, documented 1-0.3*(succ-0.5)):")
    for b in np.unique(succ_bin):
        m = succ_bin == b
        lo, hi = succ[m].min(), succ[m].max()
        lines.append(f"    succ in [{lo:.3f},{hi:.3f}]  mean_dur={dur[m].mean():8.1f}s  n={int(m.sum()):,}")
    lines.append(f"    Spearman(resume_success_rate, duration) = {rho_succ:.3f}  (expect negative if hist_factor applied)")

    # ---- 6) persona-controlled check: is the whole-population rsrp/succ
    # correlation a genuine per-session effect, or an artifact of persona
    # composition (personas differ in BOTH mean RSRP and dwell scale)? ----
    pdf = resimulate_with_persona_tag(n_ues=2000, seed=0)
    pdur = pdf["label_durationSec"].to_numpy(dtype=np.float64)
    lines.append("-" * 78)
    lines.append("[6] Persona-controlled check (re-simulated, n_ues=2000, seed=0, persona-tagged):")
    lines.append("    Per-persona: RSRP mean / iat_scale (dwell base) -- shows the confound directly:")
    for p in sg.PERSONAS:
        lines.append(f"      {p.name:<14} rsrp_mean_std={p.rsrp_mean_std}  iat_scale={p.iat_scale:.0f}s")
    lines.append("    Within-persona Spearman(rsrp, duration) and (succ, duration) -- should be ~0")
    lines.append("    if rsrp_factor/hist_factor are truly NOT wired into the per-session dwell:")
    within_rsrp = {}
    within_succ = {}
    within_ho = {}
    for pname in pdf["persona"].unique():
        m = (pdf["persona"] == pname).to_numpy()
        r_r = spearman(pdf["last_rsrp"].to_numpy()[m], pdur[m])
        r_s = spearman(pdf["resume_success_rate"].to_numpy()[m], pdur[m])
        r_h = spearman(pdf["handover_count_24h"].to_numpy()[m], pdur[m])
        within_rsrp[pname] = r_r
        within_succ[pname] = r_s
        within_ho[pname] = r_h
        lines.append(f"      {pname:<14} n={int(m.sum()):6,}  rho(rsrp,dur)={r_r:+.3f}   rho(succ,dur)={r_s:+.3f}   rho(ho,dur)={r_h:+.3f}")
    max_abs_within_rsrp = max(abs(v) for v in within_rsrp.values())
    max_abs_within_succ = max(abs(v) for v in within_succ.values())
    max_abs_within_ho = max(abs(v) for v in within_ho.values())
    lines.append(f"    max |within-persona rho| : rsrp={max_abs_within_rsrp:.3f}  succ={max_abs_within_succ:.3f}  ho={max_abs_within_ho:.3f}")
    lines.append(f"    vs whole-population rho  : rsrp={rho_rsrp:.3f}  succ={rho_succ:.3f}  ho={rho_ho:.3f}")
    lines.append("    Interpretation guide: the whole-population rho mixes persona composition")
    lines.append("    (stationary personas have BOTH better RSRP and longer dwell scales) with the")
    lines.append("    per-session factor effect; the within-persona rho isolates the factor itself.")
    lines.append("    A factor is 'applied per session' iff its within-persona rho is consistently")
    lines.append("    negative across personas -- regardless of what the confounded population rho says.")

    # ---- data-driven verdict: a factor counts as applied per-session when its
    # within-persona correlation is negative for (nearly) every persona ----
    def applied(within):
        vals = list(within.values())
        neg = sum(1 for v in vals if v < -0.005)
        return neg >= max(1, len(vals) - 1)   # allow one noisy persona

    rsrp_ok = applied(within_rsrp)
    succ_ok = applied(within_succ)
    ho_ok = applied(within_ho)
    svc_ok = 0.55 <= ratio_s <= 0.85

    def fmt_within(within):
        return ", ".join(f"{k}={v:+.3f}" for k, v in within.items())

    lines.append("-" * 78)
    lines.append("VERDICT (data-driven; REPRODUCED = within-persona rho consistently negative):")
    lines.append(f"  svc_factor (secondary_5qi)  : {'REPRODUCED' if svc_ok else 'MISMATCH'} "
                 f"(stratified ratio={ratio_s:.3f}, target~0.70; naive whole-pop {ratio:.3f} is confounded, see [2])")
    lines.append(f"  mob_factor (handover_24h)   : {'REPRODUCED (per-session)' if ho_ok else 'NOT DETECTED'} "
                 f"[{fmt_within(within_ho)}]")
    lines.append(f"  rsrp_factor (last_rsrp)     : {'REPRODUCED (per-session)' if rsrp_ok else 'NOT APPLIED per-session'} "
                 f"[{fmt_within(within_rsrp)}]")
    lines.append(f"  hist_factor (success_rate)  : {'REPRODUCED (per-session)' if succ_ok else 'NOT APPLIED per-session'} "
                 f"[{fmt_within(within_succ)}]")
    if rsrp_ok and succ_ok and ho_ok and svc_ok:
        lines.append("  -> All 4 documented label factors (design doc SS D-4) are present in the")
        lines.append("     generated data: SS D-4 'Lockstep: 3 places' holds for synth_data_gen.py,")
        lines.append("     preprocess.py gen_synthetic() and gtest genEvalSet().")
    else:
        lines.append("  -> At least one documented factor is missing from the per-session dwell:")
        lines.append("     SS D-4 'Lockstep: 3 places' is violated for synth_data_gen.py. Fix the")
        lines.append("     dwell formula (dur = sample_iat * rsrp_factor * hist_factor * svc_factor")
        lines.append("     * mob_factor) to match preprocess.py / genEvalSet.")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\ntext report saved -> {TXT_PATH}")

    termplot_report(df, ratio, ratio_s, within_rsrp, within_succ, within_ho,
                    rho_rsrp, rho_succ, rho_ho, svc_ok, rsrp_ok, succ_ok, ho_ok)

    _plot(df, dur, is_mixed_s, is_single_s, ho_group, rsrp, rsrp_bin, succ, succ_bin,
          ratio_s, rho_ho, rho_rsrp, rho_succ)
    _plot_stratified(pdf, pdur, within_rsrp, within_succ)


def termplot_report(df, ratio, ratio_s, within_rsrp, within_succ, within_ho,
                    rho_rsrp, rho_succ, rho_ho, svc_ok, rsrp_ok, succ_ok, ho_ok):
    """Terminal ASCII mirror of the two PNG figures -- 5QI distribution,
    svc_factor naive-vs-stratified-vs-theory, and per-persona rho bars for
    rsrp/hist/mob (the same confound-vs-real-effect comparison as the
    synth_literature_check_stratified.png small multiples)."""
    termplot.rule("synth_data_gen.py literature-factor check (terminal)", width=64)

    counts = df["primary_5qi"].value_counts().sort_index()
    termplot.bar_chart([(f"5QI={int(q)}", int(c)) for q, c in counts.items()],
                       fmt="{:,}", title="primary_5qi row distribution")

    termplot.bar_chart(
        [("naive (whole-pop)", ratio), ("stratified (5QI 8/9)", ratio_s),
         ("theory target", 0.70)],
        fmt="{:.3f}", ref=0.70, ref_label="theory 0.70",
        title=f"svc_factor mixed/single ratio — {'REPRODUCED' if svc_ok else 'MISMATCH'}")

    def persona_bar(within, ok, name):
        termplot.bar_chart(
            [(p, v) for p, v in within.items()], fmt="{:+.3f}", ref=0.0,
            ref_label="zero (no effect)",
            title=f"{name} within-persona Spearman rho — "
                  f"{'REPRODUCED (all negative)' if ok else 'NOT DETECTED'}")

    persona_bar(within_rsrp, rsrp_ok, "rsrp_factor")
    persona_bar(within_succ, succ_ok, "hist_factor")
    persona_bar(within_ho, ho_ok, "mob_factor")

    termplot.bar_chart(
        [("rsrp (whole-pop)", rho_rsrp), ("succ (whole-pop)", rho_succ),
         ("ho (whole-pop)", rho_ho)],
        fmt="{:+.3f}",
        title="whole-population rho (CONFOUNDED -- compare against the per-persona bars above)")


def _setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ):
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _plot(df, dur, is_mixed_s, is_single_s, ho_group, rsrp, rsrp_bin, succ, succ_bin,
          ratio_s, rho_ho, rho_rsrp, rho_succ):
    _setup_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    # [0,0] primary_5qi distribution
    ax = axes[0, 0]
    counts = df["primary_5qi"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#2563eb")
    ax.set_title("(1) primary_5qi 분포 (페르소나 혼합)", fontsize=10)
    ax.set_xlabel("5QI"); ax.set_ylabel("row count")

    # [0,1] 5QI vs mean duration (base)
    ax = axes[0, 1]
    g = df.groupby("primary_5qi")["label_durationSec"].mean().sort_index()
    ax.bar(g.index.astype(str), g.values, color="#16a34a")
    ax.set_title("(2) 5QI 별 평균 dwell (persona-driven base)", fontsize=10)
    ax.set_xlabel("5QI"); ax.set_ylabel("mean duration (s)")

    # [0,2] svc_factor check (STRATIFIED: primary_5qi in {8,9} only -- see [2])
    ax = axes[0, 2]
    means = [dur[is_single_s.to_numpy()].mean(), dur[is_mixed_s.to_numpy()].mean()]
    theo = [means[0], means[0] * 0.7]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, means, width=w, label="empirical", color="#16a34a")
    ax.bar(x + w / 2, theo, width=w, label="theory (x0.7)", color="#9ca3af")
    ax.set_xticks(x); ax.set_xticklabels(["single-class\n(5QI 8/9, secondary=0)", "mixed-class\n(5QI 8/9, secondary in {1,2})"])
    ax.set_title(f"(3) svc_factor(5QI 8/9 내 통제) 비율={ratio_s:.2f}\n(문헌=0.70)", fontsize=10)
    ax.set_ylabel("mean duration (s)"); ax.legend(fontsize=8)

    # [1,0] handover_count_24h vs duration
    ax = axes[1, 0]
    ho_x = ho_group.index.to_numpy()
    ho_y = ho_group["mean"].to_numpy()
    ax.plot(ho_x, ho_y, "o-", color="#16a34a", label="empirical mean dwell")
    ax2 = ax.twinx()
    theo_ratio = 1.0 - 0.05 * ho_x
    ax2.plot(ho_x, theo_ratio, "s--", color="#dc2626", label="theory ratio (1-0.05*ho)")
    ax2.set_ylabel("theory ratio", color="#dc2626")
    ax.set_xlabel("handover_count_24h (clipped 6)"); ax.set_ylabel("mean duration (s)")
    ax.set_title(f"(4) mob_factor(전체모집단) rho={rho_ho:.3f}\n(문헌 방향: 음수, 페르소나 혼합에 신호 매몰 — 하단 참조)", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # [1,1] RSRP vs duration
    ax = axes[1, 1]
    rsrp_means = pd.Series(dur).groupby(rsrp_bin).mean()
    rsrp_centers = pd.Series(rsrp).groupby(rsrp_bin).mean()
    ax.plot(rsrp_centers, rsrp_means, "o-", color="#2563eb", label="empirical mean dwell")
    ax2 = ax.twinx()
    rr = np.linspace(rsrp.min(), rsrp.max(), 50)
    theo_curve = 1.0 + (75.0 - rr) / 150.0
    ax2.plot(rr, theo_curve, "--", color="#dc2626", label="theory rsrp_factor")
    ax2.set_ylabel("theory factor", color="#dc2626")
    ax.set_xlabel("last_rsrp (0-127 scale)"); ax.set_ylabel("mean duration (s)")
    ax.set_title(f"(5) rsrp_factor(전체모집단) rho={rho_rsrp:+.3f}\n(문헌 방향: 음수 — 페르소나 교란, 하단 참조)", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # [1,2] resume_success_rate vs duration
    ax = axes[1, 2]
    succ_means = pd.Series(dur).groupby(succ_bin).mean()
    succ_centers = pd.Series(succ).groupby(succ_bin).mean()
    ax.plot(succ_centers, succ_means, "o-", color="#2563eb", label="empirical mean dwell")
    ax2 = ax.twinx()
    ss = np.linspace(succ.min(), succ.max(), 50)
    theo_curve2 = 1.0 - 0.3 * (ss - 0.5)
    ax2.plot(ss, theo_curve2, "--", color="#dc2626", label="theory hist_factor")
    ax2.set_ylabel("theory factor", color="#dc2626")
    ax.set_xlabel("resume_success_rate"); ax.set_ylabel("mean duration (s)")
    ax.set_title(f"(6) hist_factor(전체모집단) rho={rho_succ:+.3f}\n(문헌 방향: 음수 — 페르소나 교란, 하단 참조)", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle("synth_data_gen.py 출력 vs 설계문서(§D-4) 문헌 근거 라벨 계수 — 재현 여부 검증 (모집단 전체)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"plot saved -> {PNG_PATH}")


def _plot_stratified(pdf, pdur, within_rsrp, within_succ):
    """Persona-controlled follow-up: shows the population-level rsrp/succ vs
    duration correlation is a persona confound, not a real per-session
    rsrp_factor/hist_factor effect (§[6] in the text report)."""
    _setup_font()
    import matplotlib.pyplot as plt

    personas = list(pdf["persona"].unique())
    colors = {"iot_sensor": "#2563eb", "bursty_phone": "#dc2626",
              "fwa_fixed": "#16a34a", "commuter": "#9333ea"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) persona-level scatter of mean RSRP vs mean dwell -> the confound source
    ax = axes[0]
    for pname in personas:
        m = (pdf["persona"] == pname).to_numpy()
        ax.scatter(pdf["last_rsrp"].to_numpy()[m].mean(), pdur[m].mean(),
                   s=200, color=colors.get(pname, "#666"), label=pname, zorder=3)
    ax.set_xlabel("페르소나 평균 RSRP"); ax.set_ylabel("페르소나 평균 dwell (s)")
    ax.set_title("(a) 교란의 근원: 페르소나별 (평균RSRP, 평균dwell)\n"
                 "고정형(fwa_fixed/iot_sensor)=신호 좋음+dwell 김,\n이동형(bursty_phone/commuter)=신호 약함+dwell 짧음")
    ax.legend(fontsize=8)

    # (b) within-persona RSRP vs duration (small multiples, should be ~flat)
    ax = axes[1]
    for pname in personas:
        m = (pdf["persona"] == pname).to_numpy()
        r = pdf["last_rsrp"].to_numpy()[m]
        d = pdur[m]
        edges = np.quantile(r, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        bins = np.digitize(r, edges[1:-1])
        xs, ys = [], []
        for b in np.unique(bins):
            bm = bins == b
            xs.append(r[bm].mean()); ys.append(d[bm].mean())
        ax.plot(xs, ys, "o-", color=colors.get(pname, "#666"),
                label=f"{pname} (rho={within_rsrp[pname]:+.2f})")
    ax.set_xlabel("last_rsrp"); ax.set_ylabel("mean duration (s)")
    ax.set_title("(b) 페르소나 내부 RSRP vs dwell\n"
                 "(우하향 = rsrp_factor 적용됨 / 평탄 = 미적용)", fontsize=10)
    ax.legend(fontsize=7)

    # (c) within-persona resume_success_rate vs duration
    ax = axes[2]
    for pname in personas:
        m = (pdf["persona"] == pname).to_numpy()
        s = pdf["resume_success_rate"].to_numpy()[m]
        d = pdur[m]
        edges = np.quantile(s, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        bins = np.digitize(s, edges[1:-1])
        xs, ys = [], []
        for b in np.unique(bins):
            bm = bins == b
            xs.append(s[bm].mean()); ys.append(d[bm].mean())
        ax.plot(xs, ys, "o-", color=colors.get(pname, "#666"),
                label=f"{pname} (rho={within_succ[pname]:+.2f})")
    ax.set_xlabel("resume_success_rate"); ax.set_ylabel("mean duration (s)")
    ax.set_title("(c) 페르소나 내부 success_rate vs dwell\n"
                 "(우하향 = hist_factor 적용됨 / 평탄 = 미적용)", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("페르소나 통제 진단 — 전체 모집단 상관관계가 교란인지 실제 계수 효과인지 분리", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH2, dpi=150)
    plt.close(fig)
    print(f"plot saved -> {PNG_PATH2}")


if __name__ == "__main__":
    main()
