"""
termplot.py
Zero-extra-dependency (stdlib + numpy, both already required by every caller)
terminal ASCII plotting for the VCCB ML scripts.

Purpose: preprocess.py / train.py / analyze_synth_literature.py /
visualize_ut_results.py already save PNG reports. This module prints the SAME
underlying data as bar charts / histograms / sparklines directly into
whatever shell the script is run from (bash, PowerShell, a redhat docker
container's tty) -- no image protocol, no extra pip package, works over SSH.
PNG generation is unchanged; these are an additional textual view.

Uses Unicode block characters (█▏▎▍▌▋▊▉, ▁▂▃▄▅▆▇). If the terminal's stdout
encoding cannot represent them (seen on some Windows consoles using cp949),
falls back to plain ASCII ('#'/'-') automatically.
"""
import sys
import numpy as np

_FULL = "█"                      # █
_PARTIALS = " ▏▎▍▌▋▊▉"  # 1/8..7/8 block, index 0=blank
_SPARK_LEVELS = " ▁▂▃▄▅▆▇█"  # ' ▁▂▃▄▅▆▇█'

_ASCII_FULL = "#"
_ASCII_PARTIAL = "-"


def _unicode_ok():
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


_UNICODE = _unicode_ok()


def _safe_print(s):
    if _UNICODE:
        try:
            print(s)
            return
        except UnicodeEncodeError:
            pass
    print(s.encode("ascii", "replace").decode("ascii"))


def _block_bar(frac, width):
    frac = max(0.0, min(1.0, frac))
    if not _UNICODE:
        n = int(round(frac * width))
        return (_ASCII_FULL * n).ljust(width)
    cells = frac * width
    full = int(cells)
    rem = cells - full
    s = _FULL * full
    if full < width:
        idx = int(round(rem * 7))
        s += _PARTIALS[idx] if idx > 0 else " "
        s = s.ljust(width)
    return s[:width]


def rule(title=None, width=64, char="-"):
    if title:
        pad = max(0, width - len(title) - 2)
        _safe_print(f"{char * 2} {title} " + char * pad)
    else:
        _safe_print(char * width)


def bar_chart(items, width=36, fmt="{:,.2f}", unit="", title=None,
             ref=None, ref_label=None):
    """items: iterable of (label, value). ref: optional reference value
    (e.g. a floor/threshold or a baseline) drawn as a '|' marker on every bar
    so the eye can compare magnitude at a glance, same intent as the dashed
    reference lines in the matplotlib PNG panels."""
    items = list(items)
    if not items:
        return
    if title:
        rule(title)
    maxv = max(abs(v) for _, v in items) or 1.0
    if ref is not None:
        maxv = max(maxv, abs(ref))
    label_w = max(len(str(l)) for l, _ in items)
    for label, v in items:
        bar = _block_bar(abs(v) / maxv, width)
        if ref is not None:
            mi = min(width - 1, int(abs(ref) / maxv * width))
            bar = bar[:mi] + "|" + bar[mi + 1:]
        _safe_print(f"  {str(label).ljust(label_w)} |{bar}| {fmt.format(v)}{unit}")
    if ref is not None:
        note = f"('|' marker = {ref_label} " if ref_label else "('|' marker = "
        _safe_print(f"  {' ' * label_w}   {note}{fmt.format(ref)}{unit})")


def histogram(data, bins=16, width=36, title=None, log=False, note=""):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if title:
        rule(title)
    if len(data) == 0:
        _safe_print("  (no finite data)")
        return
    work = np.log1p(np.clip(data, 0, None)) if log else data
    counts, edges = np.histogram(work, bins=bins)
    maxc = counts.max() or 1
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        if log:
            lo, hi = np.expm1(lo), np.expm1(hi)
        bar = _block_bar(c / maxc, width)
        _safe_print(f"  [{lo:9.1f},{hi:9.1f}) |{bar}| {c:,}")
    tail = f"  n={len(data):,}"
    if note:
        tail += f"  {note}"
    _safe_print(tail)


def sparkline(values, width=None):
    """Single-line unicode sparkline (e.g. loss curve across epochs)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return ""
    if width and len(values) > width:
        idx = np.linspace(0, len(values), width + 1).astype(int)
        values = np.array([values[idx[i]:max(idx[i] + 1, idx[i + 1])].mean()
                           for i in range(width)])
    lo, hi = float(values.min()), float(values.max())
    rng = (hi - lo) or 1.0
    levels = _SPARK_LEVELS if _UNICODE else " .:-=+*#"
    chars = [levels[int(round((v - lo) / rng * (len(levels) - 1)))] for v in values]
    return "".join(chars)


def dual_histogram(pos, neg, bins=16, width=30, title=None,
                   pos_label="positive", neg_label="negative"):
    """Two side-by-side ASCII histograms sharing one x-domain -- terminal
    analogue of the overlaid-color matplotlib separability plot."""
    pos = np.asarray(pos, dtype=np.float64); pos = pos[np.isfinite(pos)]
    neg = np.asarray(neg, dtype=np.float64); neg = neg[np.isfinite(neg)]
    if title:
        rule(title)
    if len(pos) == 0 and len(neg) == 0:
        _safe_print("  (no data)")
        return
    lo = min(pos.min() if len(pos) else 0.0, neg.min() if len(neg) else 0.0)
    hi = max(pos.max() if len(pos) else 1.0, neg.max() if len(neg) else 1.0)
    edges = np.linspace(lo, hi, bins + 1)
    pc, _ = np.histogram(pos, bins=edges) if len(pos) else (np.zeros(bins), edges)
    nc, _ = np.histogram(neg, bins=edges) if len(neg) else (np.zeros(bins), edges)
    maxc = max(pc.max() if len(pc) else 0, nc.max() if len(nc) else 0) or 1
    _safe_print(f"  {'range':<22}{pos_label:>10}{'':>2}{neg_label:>10}")
    for i in range(bins):
        b1 = _block_bar(pc[i] / maxc, width // 2)
        b2 = _block_bar(nc[i] / maxc, width // 2)
        _safe_print(f"  [{edges[i]:6.2f},{edges[i+1]:6.2f}) {b1}|{b2}  "
                   f"{int(pc[i]):>5}/{int(nc[i]):<5}")
