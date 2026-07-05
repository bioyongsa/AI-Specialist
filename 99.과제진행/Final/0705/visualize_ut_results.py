#!/usr/bin/env python3
# ============================================================
# visualize_ut_results.py
# ------------------------------------------------------------
# Parse + visualize a gtest run of the VCCB AI unit-test suite
# (test_SaRrcInactiveAI.cpp) so the full result set can be compared and
# analyzed at a glance instead of scrolling raw gtest stdout.
#
# 1) Run the UT binary with structured output (Linux build PC):
#      ./vccb_ut --gtest_filter='*Ai*:*AI*' --gtest_output=json:ut_results.json
#    (XML also accepted: --gtest_output=xml:ut_results.xml)
#
# 2) Visualize:
#      python3 visualize_ut_results.py ut_results.json
#      python3 visualize_ut_results.py ut_results.json --baseline prev.json
#
# Output:
#   ut_results.png       : 4-panel figure
#     (1) per-suite pass/fail stacked bars
#     (2) slowest tests (top 15)
#     (3) Phase-1 quality metrics vs floors (from RecordProperty: auroc,
#         f1_at_0_6, mae_model_ms/naive/persist -- all x1000 integers emitted
#         by Phase1_quality_floors_on_eval_set)
#     (4) overall pass-rate + failed-test list
#   ut_results_summary.txt : console/text summary (also printed)
#   With --baseline: panel (3) overlays the baseline run's metrics and the
#   summary lists newly-failing / newly-fixed tests vs the baseline.
#
# Dependencies: stdlib + matplotlib (plot skipped gracefully if missing).
# ============================================================
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import termplot  # zero-dep terminal ASCII mirror of the PNG report panels

# Phase-1 floors — keep in lockstep with test_SaRrcInactiveAI.cpp asserts.
FLOOR_AUROC = 0.85
FLOOR_F1 = 0.78
FLOOR_MAE_ABS = 15.0

# RecordProperty keys emitted by Phase1_quality_floors_on_eval_set (x1000 ints).
METRIC_KEYS = ("auroc", "f1_at_0_6", "mae_model_ms", "mae_naive_ms",
               "mae_persist_ms", "eval_positives")


def _sec(s):
    """gtest time strings look like '0.123s'."""
    try:
        return float(str(s).rstrip("s"))
    except Exception:
        return 0.0


def load_gtest_json(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    tests = []
    for suite in doc.get("testsuites", []):
        for t in suite.get("testsuite", []):
            rec = {
                "suite": suite.get("name", "?"),
                "name": t.get("name", "?"),
                "full": f"{suite.get('name','?')}.{t.get('name','?')}",
                "time": _sec(t.get("time", "0s")),
                "skipped": t.get("result") == "SKIPPED" or t.get("status") == "NOTRUN",
                "failed": bool(t.get("failures")),
                "failure_msgs": [f.get("failure", "") for f in t.get("failures", [])],
                "props": {},
            }
            for k in METRIC_KEYS:          # RecordProperty -> extra JSON fields
                if k in t:
                    try:
                        rec["props"][k] = float(t[k])
                    except (TypeError, ValueError):
                        pass
            tests.append(rec)
    return tests


def load_gtest_xml(path):
    root = ET.parse(path).getroot()
    tests = []
    for suite in root.iter("testsuite"):
        for t in suite.iter("testcase"):
            failures = [f.get("message", "") for f in t.findall("failure")]
            props = {}
            pr = t.find("properties")
            if pr is not None:
                for p in pr.findall("property"):
                    if p.get("name") in METRIC_KEYS:
                        try:
                            props[p.get("name")] = float(p.get("value"))
                        except (TypeError, ValueError):
                            pass
            tests.append({
                "suite": suite.get("name", "?"),
                "name": t.get("name", "?"),
                "full": f"{suite.get('name','?')}.{t.get('name','?')}",
                "time": _sec(t.get("time", "0")),
                "skipped": t.find("skipped") is not None,
                "failed": bool(failures),
                "failure_msgs": failures,
                "props": props,
            })
    return tests


def load_any(path):
    if path.lower().endswith(".xml"):
        return load_gtest_xml(path)
    return load_gtest_json(path)


def extract_phase1(tests):
    """Return de-scaled Phase-1 metrics dict or None (looks for the quality
    test's RecordProperty payload on any test in the run)."""
    for t in tests:
        p = t["props"]
        if "auroc" in p:
            return {
                "auroc": p["auroc"] / 1000.0,
                "f1": p.get("f1_at_0_6", float("nan")) / 1000.0,
                "mae_model": p.get("mae_model_ms", float("nan")) / 1000.0,
                "mae_naive": p.get("mae_naive_ms", float("nan")) / 1000.0,
                "mae_persist": p.get("mae_persist_ms", float("nan")) / 1000.0,
                "positives": int(p.get("eval_positives", 0)),
                "test": t["full"],
            }
    return None


def summarize(tests, baseline=None, base_tests=None):
    L = []
    n = len(tests)
    failed = [t for t in tests if t["failed"]]
    skipped = [t for t in tests if t["skipped"]]
    total_time = sum(t["time"] for t in tests)
    L.append("=" * 72)
    L.append("VCCB AI UT run summary (test_SaRrcInactiveAI)")
    L.append("=" * 72)
    L.append(f"tests   : {n}  passed {n - len(failed) - len(skipped)}  "
             f"failed {len(failed)}  skipped {len(skipped)}   "
             f"total {total_time:.2f}s")

    suites = {}
    for t in tests:
        s = suites.setdefault(t["suite"], {"pass": 0, "fail": 0, "time": 0.0})
        s["fail" if t["failed"] else "pass"] += 0 if t["skipped"] else 1
        s["time"] += t["time"]
    L.append("-" * 72)
    L.append(f"{'suite':<38}{'pass':>6}{'fail':>6}{'time(s)':>10}")
    for name, s in sorted(suites.items()):
        L.append(f"{name:<38}{s['pass']:>6}{s['fail']:>6}{s['time']:>10.2f}")

    if failed:
        L.append("-" * 72)
        L.append("FAILED:")
        for t in failed:
            L.append(f"  ✗ {t['full']}")
            for m in t["failure_msgs"][:2]:
                first = str(m).strip().splitlines()[0] if m else ""
                L.append(f"      {first[:100]}")

    ph = extract_phase1(tests)
    if ph:
        L.append("-" * 72)
        L.append(f"Phase-1 quality (from {ph['test']}, n_pos={ph['positives']}):")
        L.append(f"  AUROC      {ph['auroc']:.3f}  (floor {FLOOR_AUROC})")
        L.append(f"  F1@0.6     {ph['f1']:.3f}  (floor {FLOOR_F1})")
        L.append(f"  MAE model  {ph['mae_model']:.2f}s  (floor {FLOOR_MAE_ABS}s; "
                 f"naive {ph['mae_naive']:.2f}s / persistence {ph['mae_persist']:.2f}s)")

    if base_tests is not None:
        base_fail = {t["full"] for t in base_tests if t["failed"]}
        cur_fail = {t["full"] for t in failed}
        newly_failing = sorted(cur_fail - base_fail)
        newly_fixed = sorted(base_fail - cur_fail)
        L.append("-" * 72)
        L.append(f"vs baseline ({baseline}):")
        L.append(f"  newly failing : {', '.join(newly_failing) if newly_failing else '(none)'}")
        L.append(f"  newly fixed   : {', '.join(newly_fixed) if newly_fixed else '(none)'}")
        bph = extract_phase1(base_tests)
        if ph and bph:
            L.append(f"  AUROC {bph['auroc']:.3f} → {ph['auroc']:.3f}   "
                     f"F1 {bph['f1']:.3f} → {ph['f1']:.3f}   "
                     f"MAE {bph['mae_model']:.2f}s → {ph['mae_model']:.2f}s")
    L.append("=" * 72)
    return "\n".join(L)


def termplot_report(tests, base_tests=None):
    """Terminal ASCII mirror of plot()'s 4 panels."""
    termplot.rule("VCCB AI UT results (terminal)", width=64)

    suites = {}
    for t in tests:
        s = suites.setdefault(t["suite"], [0, 0])
        s[1 if t["failed"] else 0] += 0 if t["skipped"] else 1
    termplot.bar_chart([(name, s[0]) for name, s in suites.items()],
                       fmt="{:,}", title="(1) suite별 pass 수")
    fails = {name: s[1] for name, s in suites.items() if s[1]}
    if fails:
        termplot.bar_chart(list(fails.items()), fmt="{:,}",
                           title="(1b) suite별 fail 수")

    slow = sorted(tests, key=lambda t: -t["time"])[:12]
    termplot.bar_chart([(t["name"][:30], t["time"] * 1000) for t in slow],
                       fmt="{:.1f}", unit="ms", title="(2) 느린 테스트 top 12")

    ph = extract_phase1(tests)
    if ph:
        bph = extract_phase1(base_tests) if base_tests else None
        items = [("AUROC", ph["auroc"]), ("F1@0.6", ph["f1"])]
        if bph:
            items += [("AUROC(baseline)", bph["auroc"]), ("F1(baseline)", bph["f1"])]
        termplot.bar_chart(items, fmt="{:.3f}", ref=FLOOR_AUROC,
                           ref_label=f"AUROC floor {FLOOR_AUROC}",
                           title="(3) Phase-1 분류 품질")
        mae_items = [("model MAE", ph["mae_model"]), ("naive MAE", ph["mae_naive"]),
                    ("persist MAE", ph["mae_persist"])]
        if bph:
            mae_items.insert(1, ("baseline MAE", bph["mae_model"]))
        termplot.bar_chart(mae_items, fmt="{:.2f}", unit="s", ref=FLOOR_MAE_ABS,
                           ref_label=f"MAE floor {FLOOR_MAE_ABS}s",
                           title="(3b) Phase-1 회귀 MAE vs baseline")

    n = len([t for t in tests if not t["skipped"]])
    nf = len([t for t in tests if t["failed"]])
    rate = 100.0 * (n - nf) / max(n, 1)
    termplot.rule(f"(4) 종합: pass rate {rate:.1f}% ({n - nf}/{n})", width=64)
    if nf:
        for t in tests:
            if t["failed"]:
                print(f"  FAIL: {t['full']}")
    else:
        print("  all AI unit tests green")


def plot(tests, out_png, base_tests=None):
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
        print(f"(plot skipped: {e})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))

    # (1) per-suite pass/fail
    ax = axes[0, 0]
    suites = {}
    for t in tests:
        s = suites.setdefault(t["suite"], [0, 0])
        s[1 if t["failed"] else 0] += 0 if t["skipped"] else 1
    names = list(suites.keys())
    passes = [suites[k][0] for k in names]
    fails = [suites[k][1] for k in names]
    ax.barh(names, passes, color="#1baf7a", label="pass")
    ax.barh(names, fails, left=passes, color="#e34948", label="fail")
    for i, (p, f) in enumerate(zip(passes, fails)):
        ax.annotate(f"{p}P" + (f" / {f}F" if f else ""), xy=(p + f, i),
                    va="center", fontsize=9)
    ax.set_title("(1) suite별 pass/fail", fontsize=11)
    ax.set_xlabel("tests")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # (2) slowest tests
    ax = axes[0, 1]
    slow = sorted(tests, key=lambda t: -t["time"])[:15]
    ax.barh([t["name"][:44] for t in slow], [t["time"] * 1000 for t in slow],
            color=["#e34948" if t["failed"] else "#2a78d6" for t in slow])
    ax.set_xlabel("ms")
    ax.set_title("(2) 느린 테스트 top 15 (빨강 = 실패)", fontsize=11)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)

    # (3) Phase-1 metrics vs floors (+ baseline overlay)
    ax = axes[1, 0]
    ph = extract_phase1(tests)
    if ph:
        import numpy as np
        bph = extract_phase1(base_tests) if base_tests else None
        labels = ["AUROC", "F1@0.6"]
        cur = [ph["auroc"], ph["f1"]]
        floors = [FLOOR_AUROC, FLOOR_F1]
        x = np.arange(len(labels))
        w = 0.3 if bph else 0.45
        ax.bar(x - (w / 2 if bph else 0), cur, width=w, color="#2a78d6",
               label="this run")
        if bph:
            ax.bar(x + w / 2, [bph["auroc"], bph["f1"]], width=w,
                   color="#9085e9", label="baseline run")
        for xi, fl in zip(x, floors):
            ax.hlines(fl, xi - 0.35, xi + 0.35, color="#e34948", ls="--", lw=1.4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        mae_txt = (f"MAE model {ph['mae_model']:.1f}s "
                   f"(floor {FLOOR_MAE_ABS:.0f}s / naive {ph['mae_naive']:.1f}s / "
                   f"persist {ph['mae_persist']:.1f}s)")
        if bph:
            mae_txt += f"\nbaseline MAE {bph['mae_model']:.1f}s"
        ax.set_title(f"(3) Phase-1 품질 vs floor(빨간 점선)\n{mae_txt}", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Phase-1 metrics not found\n(run includes "
                "Phase1_quality_floors_on_eval_set?)", ha="center", va="center")

    # (4) overall pass rate + failed list
    ax = axes[1, 1]
    n = len([t for t in tests if not t["skipped"]])
    nf = len([t for t in tests if t["failed"]])
    ax.axis("off")
    rate = 100.0 * (n - nf) / max(n, 1)
    color = "#1baf7a" if nf == 0 else "#e34948"
    ax.text(0.5, 0.82, f"{rate:.1f}%", ha="center", fontsize=44, color=color)
    ax.text(0.5, 0.68, f"pass rate ({n - nf}/{n})", ha="center", fontsize=12)
    if nf:
        failed_names = [t["full"] for t in tests if t["failed"]][:8]
        ax.text(0.5, 0.5, "FAILED:\n" + "\n".join(failed_names),
                ha="center", va="top", fontsize=9, color="#e34948")
    else:
        ax.text(0.5, 0.5, "all AI unit tests green", ha="center", fontsize=12,
                color="#1baf7a")
    ax.set_title("(4) 종합", fontsize=11)

    fig.suptitle("VCCB AI UT 결과 리포트 (gtest)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    print(f"plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Visualize a gtest run of the VCCB AI UT suite.")
    ap.add_argument("results", nargs="?", default="ut_results.json",
                    help="gtest --gtest_output json/xml file")
    ap.add_argument("--baseline", default=None,
                    help="previous run's json/xml to compare against")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside input)")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"{args.results} not found.\nRun the UT first, e.g.:\n"
                 "  ./vccb_ut --gtest_filter='*Ai*:*AI*' "
                 "--gtest_output=json:ut_results.json")

    tests = load_any(args.results)
    if not tests:
        sys.exit("no test cases found in the results file")
    base_tests = load_any(args.baseline) if args.baseline else None

    report = summarize(tests, args.baseline, base_tests)
    print(report)
    txt = os.path.splitext(args.results)[0] + "_summary.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nsummary -> {txt}")

    termplot_report(tests, base_tests)

    out_png = args.out or os.path.splitext(args.results)[0] + ".png"
    plot(tests, out_png, base_tests)


if __name__ == "__main__":
    main()
