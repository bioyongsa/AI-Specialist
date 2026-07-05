"""
train.py
GRU model training for VCCB RRC Inactivity Timer prediction.

Architecture (must match VccbAiEngine.cpp exactly):
  GRU  : input_size=2, hidden_size=16, num_layers=1, batch_first=True
  MLP  : Linear(14, 16) + ReLU
  Head1: Linear(32, 1)  + Sigmoid  -> resume_prob
  Head2: Linear(32, 1)             -> resume_time  (regression)

Usage:
  python train.py                          # full training from scratch
  VCCB_AI_FINETUNE=1 python train.py       # warm-start fine-tune from model.pt
  VCCB_AI_FINETUNE=1 VCCB_AI_FREEZE_GRU=1 python train.py
                                           # fine-tune heads only (small real set)
  VCCB_AI_EARLY_STOP_PATIENCE=5 python train.py
                                           # stop when val loss stalls (default 5
                                           # for fine-tune, 0/off for full train)

Output:
  model.pt  (saved PyTorch state_dict, used by export_weights.py)
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import metrics  # shared Phase-1 quality metrics (numpy-only, no circular import)
import termplot  # zero-dep terminal ASCII mirror of the PNG report panels

# ──────────────────────────────────────────────
# Script version — BUMP THIS ON EVERY FUNCTIONAL CHANGE.
# Printed at run start and (via export_weights.py) embedded into VccbAiWeights.h
# so a generated weight header can be traced back to the exact script revision.
#   1.0.0 : initial GRU trainer (raw inputs, raw-seconds time target)
#   1.1.0 : normalize inputs + time target (constants folded back in export)
#   1.2.0 : clip regression target to a realistic ceiling + Huber(SmoothL1) loss
#           (tames heavy-tailed duration outliers so time-loss no longer spikes)
#   1.3.0 : opt-in fine-tuning (warm start from model.pt + lower LR + fewer
#           epochs, optional GRU/MLP freeze) for incremental retrain on freshly
#           collected real labels without forgetting the synthetic prior.
#   1.4.0 : auto-print Phase-1 quality metrics on the held-out val split at end
#           of every run (shared ML/metrics.py: AUROC/PR-AUC/ECE/F1 + regression
#           MAE vs naive/persistence/oracle baselines) so model changes are
#           measured-and-proven each run.
#   1.5.0 : optional early stopping on val loss (patience env var; default 5
#           when VCCB_AI_FINETUNE=1, else 0) — Phase 3-3 guard against
#           overfitting on small real-label batches.
#   1.6.0 : drop duplicate static feature (old stat[12] avg_inactive_duration
#           == stat[9]); STATIC_INPUT 15->14, MLP 16x14, total params 1282->1266.
#           Requires re-export of VccbAiWeights.h (incompatible with 15-dim model.pt).
#   1.7.0 : visual training report. Collects per-epoch loss history and renders
#           train_report_{full|finetune}.png (loss curves / pred-vs-actual /
#           prob separability / MAE-vs-baselines) + dumps the run's metrics to
#           train_metrics_{mode}.json. In fine-tune mode the warm-started model
#           is ALSO scored on the val split BEFORE training so the report shows
#           a true before/after comparison of the fine-tuning cycle.
# ──────────────────────────────────────────────
SCRIPT_VERSION = "1.7.0"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "dataset.npz")
MODEL_FILE   = os.path.join(SCRIPT_DIR, "model.pt")

# ──────────────────────────────────────────────
# Hyperparameters  (match VccbAiEngine.h constants)
# ──────────────────────────────────────────────
HIDDEN_SIZE   = 16    # VCCB_AI_HIDDEN_SIZE
SEQ_INPUT     = 2     # VCCB_AI_SEQ_INPUT
STATIC_INPUT  = 14    # VCCB_AI_STATIC_INPUT
SEQ_LEN       = 10    # VCCB_AI_HISTORY_LEN

BATCH_SIZE    = 256
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_RATIO     = 0.15
SEED          = 42

# ──────────────────────────────────────────────
# Fine-tuning (opt-in via env vars; default OFF = full training from scratch).
#   VCCB_AI_FINETUNE=1     : warm-start from existing model.pt instead of random init
#   VCCB_AI_FINETUNE_LR    : smaller LR for incremental update (default 2e-4)
#   VCCB_AI_FINETUNE_EPOCHS: fewer epochs (default 15)
#   VCCB_AI_FREEZE_GRU=1   : freeze GRU+MLP, train only the output heads
#                            (recommended when the fresh real-label set is small)
# Rationale: once a usable model exists (synthetic prior), incremental real-label
# batches should *adjust* it, not overwrite it. Low LR + few epochs + optional
# freeze limits catastrophic forgetting of the synthetic-learned structure.
# ──────────────────────────────────────────────
FINETUNE        = os.environ.get("VCCB_AI_FINETUNE", "0") == "1"
FINETUNE_LR     = float(os.environ.get("VCCB_AI_FINETUNE_LR", "2e-4"))
FINETUNE_EPOCHS = int(os.environ.get("VCCB_AI_FINETUNE_EPOCHS", "15"))
FREEZE_GRU      = os.environ.get("VCCB_AI_FREEZE_GRU", "0") == "1"

# Early stopping: halt when val loss fails to beat the running best for
# `patience` consecutive epochs. Default ON (patience=5) for fine-tune,
# OFF (0) for full scratch training — preserves existing 50-epoch behaviour.
# Override explicitly: VCCB_AI_EARLY_STOP_PATIENCE=N  (0 disables).
_EARLY_STOP_DEFAULT = "5" if FINETUNE else "0"
EARLY_STOP_PATIENCE = int(os.environ.get("VCCB_AI_EARLY_STOP_PATIENCE",
                                         _EARLY_STOP_DEFAULT))

# ──────────────────────────────────────────────
# Fixed feature/label normalization (domain-informed, NOT data-fitted).
# ------------------------------------------------------------------
# Raw features span wildly different scales (AMBR 10..2000, RSRP 0..127,
# durationSec 0..400, 5QI 1..9, 0/1 flags). Feeding them raw saturates the
# sigmoid -> resume_prob collapses to a constant ~1.0 and resume_time diverges.
# We normalize here for stable training; export_weights.py folds these exact
# constants back into the first affine layers (GRU_ih / MLP / head_time) so the
# C++ engine keeps consuming RAW inputs and emitting RAW seconds (no change to
# VccbAiEngine.cpp / buildAiInput()). Keep these in lockstep with export.
#
# Order MUST match preprocess.py / buildAiInput():
#   seq[t]  = [durationSec, resumeCause]
#   stat[*] = see preprocess.py header (14 features)
# σ must be > 0; for always-zero features (stat 4/5/7) use μ=0, σ=1.
# (Former stat[12] avg_inactive_duration was a duplicate of stat[9] and was
#  removed; STATIC_INPUT=14, hour/dow are now indices 12/13.)
SEQ_MEAN  = np.array([60.0, 2.0], dtype=np.float32)          # durationSec, resumeCause
SEQ_STD   = np.array([60.0, 2.0], dtype=np.float32)
STAT_MEAN = np.array([5, 2, 500, 0.5, 0, 0, 64, 0, 8, 60, 5, 0.8, 12, 3],
                     dtype=np.float32)
STAT_STD  = np.array([3, 2, 500, 0.5, 1, 1, 32, 1, 5, 60, 3, 0.2,  7, 2],
                     dtype=np.float32)
TIME_MEAN = np.float32(60.0)   # resume_time target center (seconds)
TIME_STD  = np.float32(60.0)

# Regression-target ceiling (seconds). The synthetic duration is log-normal with
# a long tail (clipped at 86400s in preprocess), which makes a few samples
# dominate the time MSE. An inactivity timer is never set that high anyway, so we
# clip the *training target* to a realistic ceiling to stabilize the regression.
# (Affects Y_time only; Y_prob's <60s threshold is unaffected.)
REG_CLIP_SEC = np.float32(600.0)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class VccbDataset(Dataset):
    def __init__(self, X_seq, X_stat, Y_prob, Y_time):
        # Normalize inputs (broadcast over the last/feature axis) and the time
        # target. Y_prob is already 0/1 (BCE) and is left untouched.
        # Clip the time target to a realistic ceiling first (v1.2.0) so heavy-tail
        # outliers don't dominate the regression loss.
        X_seq  = (X_seq.astype(np.float32)  - SEQ_MEAN)  / SEQ_STD
        X_stat = (X_stat.astype(np.float32) - STAT_MEAN) / STAT_STD
        Y_time = np.clip(Y_time.astype(np.float32), 1.0, REG_CLIP_SEC)
        Y_time = (Y_time - TIME_MEAN) / TIME_STD

        self.X_seq  = torch.tensor(X_seq,  dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat, dtype=torch.float32)
        self.Y_prob = torch.tensor(Y_prob, dtype=torch.float32).unsqueeze(1)
        self.Y_time = torch.tensor(Y_time, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.Y_prob)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_stat[idx], self.Y_prob[idx], self.Y_time[idx]


# ──────────────────────────────────────────────
# Model (must match VccbAiEngine.cpp forward pass)
# ──────────────────────────────────────────────
class RrcGruModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ① GRU: processes seq[SEQ_LEN][SEQ_INPUT]
        self.gru = nn.GRU(
            input_size=SEQ_INPUT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        # ② MLP: processes stat[STATIC_INPUT]
        self.mlp = nn.Sequential(
            nn.Linear(STATIC_INPUT, HIDDEN_SIZE),
            nn.ReLU(),
        )

        # ③ Output heads: combined = concat(gru_out, mlp_out) -> dim 32
        self.head_prob = nn.Linear(HIDDEN_SIZE * 2, 1)   # + sigmoid in forward
        self.head_time = nn.Linear(HIDDEN_SIZE * 2, 1)   # linear (regression)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p if p.dim() >= 2 else p.unsqueeze(0))
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, seq, stat):
        # seq  : (B, SEQ_LEN, SEQ_INPUT)
        # stat : (B, STATIC_INPUT)

        # ① GRU
        _, h_n = self.gru(seq)          # h_n: (1, B, HIDDEN_SIZE)
        gru_out = h_n.squeeze(0)        # (B, HIDDEN_SIZE)

        # ② MLP
        mlp_out = self.mlp(stat)        # (B, HIDDEN_SIZE)

        # ③ Combine + heads
        combined = torch.cat([gru_out, mlp_out], dim=1)   # (B, 32)
        prob = torch.sigmoid(self.head_prob(combined))     # (B, 1)
        time = self.head_time(combined)                    # (B, 1) linear

        return prob, time


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    BCE for resume_prob (classification) +
    Huber/SmoothL1 for resume_time (regression, only when prob=1).
    SmoothL1 (v1.2.0) is robust to the remaining duration outliers — large
    residuals contribute linearly instead of quadratically, so the time term
    no longer spikes the total loss.
    """
    def __init__(self, lambda_time: float = 0.5):
        super().__init__()
        self.bce    = nn.BCELoss()
        self.reg    = nn.SmoothL1Loss(reduction="none")
        self.lam    = lambda_time

    def forward(self, pred_prob, pred_time, y_prob, y_time):
        loss_prob = self.bce(pred_prob, y_prob)

        # Weight time regression only on positive samples (resume_prob = 1)
        time_loss_raw = self.reg(pred_time, y_time)
        mask = (y_prob > 0.5).float()
        loss_time = (time_loss_raw * mask).sum() / (mask.sum() + 1e-6)

        return loss_prob + self.lam * loss_time, loss_prob.item(), loss_time.item()


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train():
    print(f"[train.py v{SCRIPT_VERSION}]")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load dataset
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"{DATASET_FILE} not found.\n"
            "Run preprocess.py first."
        )

    print(f"Loading {DATASET_FILE} ...")
    data = np.load(DATASET_FILE)
    dataset = VccbDataset(
        data["X_seq"], data["X_stat"],
        data["Y_prob"], data["Y_time"],
    )
    print(f"  Total samples: {len(dataset):,}")

    # Train / val split
    n_val   = int(len(dataset) * VAL_RATIO)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    model     = RrcGruModel().to(device)

    # ── helper: Phase-1 metrics of a model state on the held-out val split ──
    tr_idx  = np.array(train_ds.indices)
    va_idx  = np.array(val_ds.indices)
    horizon = float(os.environ.get("VCCB_AI_PROB_HORIZON", "60"))

    def eval_on_val(m):
        m.eval()
        with torch.no_grad():
            p_t, t_t = m(dataset.X_seq[va_idx].to(device),
                         dataset.X_stat[va_idx].to(device))
        pp = p_t.squeeze(1).cpu().numpy()
        pt = t_t.squeeze(1).cpu().numpy() * float(TIME_STD) + float(TIME_MEAN)
        r = metrics.compute_all(
            X_stat_tr=data["X_stat"][tr_idx], y_time_tr=data["Y_time"][tr_idx],
            y_prob_tr=data["Y_prob"][tr_idx],
            X_stat_va=data["X_stat"][va_idx], y_time_va=data["Y_time"][va_idx],
            y_prob_va=data["Y_prob"][va_idx],
            pred_prob_va=pp, pred_time_va=pt, horizon=horizon)
        return r, pp, pt

    # ── Fine-tune: warm-start from existing model.pt (opt-in) ──
    eff_lr     = LR
    eff_epochs = EPOCHS
    pre_res    = None   # pre-fine-tune metrics (before/after comparison)
    if FINETUNE:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"VCCB_AI_FINETUNE=1 but {MODEL_FILE} not found.\n"
                "Run a full training once before fine-tuning."
            )
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        eff_lr     = FINETUNE_LR
        eff_epochs = FINETUNE_EPOCHS
        print(f"  [FINETUNE] warm start from {MODEL_FILE}")
        print(f"  [FINETUNE] lr={eff_lr} epochs={eff_epochs} freeze_gru={FREEZE_GRU}")

        # Score the warm-started model on THIS dataset's val split before any
        # update — the honest "before" of the fine-tuning cycle.
        pre_res, _, _ = eval_on_val(model)
        pc, pr = pre_res["classification"], pre_res["regression"]
        print(f"  [FINETUNE] pre-FT on this val: AUROC={pc['AUROC']:.4f} "
              f"F1={pc['f1']:.3f} MAE={pr['model']['MAE']:.2f}s")

        if FREEZE_GRU:
            # Train only the output heads; keep the synthetic-learned GRU/MLP fixed.
            for pname, pp in model.named_parameters():
                if pname.startswith("gru.") or pname.startswith("mlp."):
                    pp.requires_grad = False

    criterion = CombinedLoss(lambda_time=0.5)
    trainable = [pp for pp in model.parameters() if pp.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=eff_lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eff_epochs)

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    stale_epochs  = 0
    history = {"epoch": [], "train": [], "val": [], "p_loss": [], "t_loss": []}

    if EARLY_STOP_PATIENCE > 0:
        print(f"  Early stop: patience={EARLY_STOP_PATIENCE} (val loss)")

    print(f"{'Epoch':>5} {'Train':>10} {'Val':>10} {'P_loss':>8} {'T_loss':>8}")
    print("-" * 50)

    for epoch in range(1, eff_epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for seq, stat, y_p, y_t in train_loader:
            seq, stat, y_p, y_t = (x.to(device) for x in (seq, stat, y_p, y_t))
            optimizer.zero_grad()
            pred_p, pred_t = model(seq, stat)
            loss, lp, lt = criterion(pred_p, pred_t, y_p, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        sum_lp = sum_lt = 0.0
        with torch.no_grad():
            for seq, stat, y_p, y_t in val_loader:
                seq, stat, y_p, y_t = (x.to(device) for x in (seq, stat, y_p, y_t))
                pred_p, pred_t = model(seq, stat)
                loss, lp, lt = criterion(pred_p, pred_t, y_p, y_t)
                val_loss += loss.item()
                sum_lp   += lp
                sum_lt   += lt
        val_loss /= len(val_loader)
        sum_lp   /= len(val_loader)
        sum_lt   /= len(val_loader)

        scheduler.step()

        marker = " *" if val_loss < best_val_loss else ""
        print(f"{epoch:>5} {tr_loss:>10.4f} {val_loss:>10.4f} "
              f"{sum_lp:>8.4f} {sum_lt:>8.4f}{marker}")

        history["epoch"].append(epoch)
        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        history["p_loss"].append(sum_lp)
        history["t_loss"].append(sum_lt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            stale_epochs  = 0
        else:
            stale_epochs += 1

        if (EARLY_STOP_PATIENCE > 0
                and stale_epochs >= EARLY_STOP_PATIENCE):
            print(f"\n[EARLY STOP] no val improvement for {EARLY_STOP_PATIENCE} "
                  f"epoch(s) — stopped at {epoch}/{eff_epochs}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Model saved   → {MODEL_FILE}")

    # ── Phase-1 quality metrics on the held-out val split (best checkpoint) ──
    res, pred_prob, pred_time = eval_on_val(model)
    print("\n── Phase-1 quality metrics (val) ──")
    for line in metrics.format_block(res):
        print(line)

    # ── visual training report + machine-readable metrics dump (v1.7.0) ──
    mode = "finetune" if FINETUNE else "full"
    dump_metrics_json(mode, res, pre_res, history, best_epoch, best_val_loss)
    plot_train_report(mode, history, best_epoch, res, pre_res,
                      pred_prob, pred_time,
                      y_prob_va=data["Y_prob"][va_idx],
                      y_time_va=data["Y_time"][va_idx])
    termplot_train_report(mode, history, best_epoch, res, pre_res,
                          pred_prob, y_prob_va=data["Y_prob"][va_idx])

    print("\nNext: run  export_weights.py  (or evaluate.py for plots/full report)")


def termplot_train_report(mode, history, best_epoch, res, pre_res,
                          pred_prob, y_prob_va):
    """Terminal ASCII mirror of plot_train_report() -- loss sparkline, MAE vs
    baselines bar chart, AUROC/F1 vs Phase-1 floor bar chart, and the
    resume_prob positive/negative separability histogram."""
    c, r = res["classification"], res["regression"]

    termplot.rule(f"train report (terminal) — mode={mode}"
                  + (f", freeze_gru={FREEZE_GRU}" if mode == "finetune" else ""),
                  width=64)
    spark = termplot.sparkline(history["val"], width=60)
    print(f"  val loss   {spark}  (best @ epoch {best_epoch}, "
          f"{history['val'][0]:.3f} -> {min(history['val']):.3f})")
    spark_t = termplot.sparkline(history["train"], width=60)
    print(f"  train loss {spark_t}")

    mae_items = []
    if pre_res is not None:
        mae_items.append(("before FT", pre_res["regression"]["model"]["MAE"]))
        mae_items.append(("after FT", r["model"]["MAE"]))
    else:
        mae_items.append(("model", r["model"]["MAE"]))
    mae_items.append(("naive", r["naive"]["MAE"]))
    mae_items.append(("persistence", r["persistence"]["MAE"]))
    termplot.bar_chart(mae_items, fmt="{:.2f}", unit="s",
                       title="regression MAE vs baselines (val positives)")

    auroc_items = [("AUROC", c["AUROC"]), ("F1@0.6", c["f1"])]
    if pre_res is not None:
        pc = pre_res["classification"]
        auroc_items = [("AUROC before", pc["AUROC"]), ("AUROC after", c["AUROC"]),
                       ("F1 before", pc["f1"]), ("F1 after", c["f1"])]
    termplot.bar_chart(auroc_items, fmt="{:.3f}",
                       title="classification quality (Phase-1)")

    yv = np.asarray(y_prob_va)
    pv = np.asarray(pred_prob)
    termplot.dual_histogram(pv[yv > 0.5], pv[yv <= 0.5], bins=12, width=28,
                            title="resume_prob separability (model output)",
                            pos_label="resume<horizon", neg_label="no-resume")


# ──────────────────────────────────────────────
# Visual training report (v1.7.0)
# ──────────────────────────────────────────────
def _strip_arrays(res):
    """compute_all() result minus the numpy plotting arrays (JSON-safe)."""
    if res is None:
        return None
    return {k: v for k, v in res.items() if k != "_arrays"}


def dump_metrics_json(mode, res, pre_res, history, best_epoch, best_val_loss):
    out = os.path.join(SCRIPT_DIR, f"train_metrics_{mode}.json")
    payload = {
        "script_version": SCRIPT_VERSION,
        "mode": mode,
        "freeze_gru": FREEZE_GRU if mode == "finetune" else False,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "metrics": _strip_arrays(res),
        "metrics_before_finetune": _strip_arrays(pre_res),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"metrics json  → {out}")


def plot_train_report(mode, history, best_epoch, res, pre_res,
                      pred_prob, pred_time, y_prob_va, y_time_va):
    """4-panel PNG: loss curves / pred-vs-actual / prob separability /
    MAE vs baselines (fine-tune mode adds the before-FT bar + AUROC delta)."""
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
        print(f"(train report plot skipped: {e})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    c = res["classification"]; r = res["regression"]

    # (1) loss curves
    ax = axes[0, 0]
    ax.plot(history["epoch"], history["train"], "-", color="#2a78d6", lw=2, label="train loss")
    ax.plot(history["epoch"], history["val"], "-", color="#e34948", lw=2, label="val loss")
    ax.plot(history["epoch"], history["p_loss"], "--", color="#1baf7a", lw=1.2, label="val BCE(prob)")
    ax.plot(history["epoch"], history["t_loss"], "--", color="#eda100", lw=1.2, label="val SmoothL1(time)")
    if best_epoch:
        ax.axvline(best_epoch, color="#888", ls=":", lw=1)
        ax.annotate(f"best @ {best_epoch}", xy=(best_epoch, min(history["val"])),
                    fontsize=9, color="#555")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"(1) 학습 곡선 — mode={mode}", fontsize=11)
    ax.legend(fontsize=8)

    # (2) predicted vs actual resume_time (val positives)
    ax = axes[0, 1]
    pos = np.asarray(y_prob_va) > 0.5
    t_true = np.asarray(y_time_va)[pos]
    t_pred = np.asarray(pred_time)[pos]
    n_show = min(3000, len(t_true))
    sel = np.random.default_rng(0).choice(len(t_true), n_show, replace=False) \
        if len(t_true) > n_show else np.arange(len(t_true))
    ax.scatter(t_true[sel], t_pred[sel], s=6, alpha=0.25, color="#2a78d6")
    lim = max(1.0, float(np.percentile(t_true, 99)))
    ax.plot([0, lim], [0, lim], "--", color="#e34948", lw=1.2, label="y = x (완벽 예측)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("실제 resume_time (s)"); ax.set_ylabel("예측 resume_time (s)")
    ax.set_title(f"(2) 예측 vs 실제 (양성 {int(pos.sum()):,}건) — "
                 f"MAE {r['model']['MAE']:.1f}s / bias {r['model']['bias']:+.1f}s", fontsize=11)
    ax.legend(fontsize=8)

    # (3) resume_prob separability by true class
    ax = axes[1, 0]
    yv = np.asarray(y_prob_va)
    pv = np.asarray(pred_prob)
    bins = np.linspace(0, 1, 41)
    ax.hist(pv[yv > 0.5], bins=bins, alpha=0.6, color="#1baf7a",
            label=f"실제 resume<{int(res['horizon'])}s (양성)")
    ax.hist(pv[yv <= 0.5], bins=bins, alpha=0.6, color="#e34948", label="실제 no-resume (음성)")
    ax.axvline(res["threshold"], color="#333", ls="--", lw=1.2,
               label=f"KEEP threshold {res['threshold']}")
    ax.set_xlabel("model resume_prob"); ax.set_ylabel("rows")
    ax.set_title(f"(3) 분류 분리도 — AUROC {c['AUROC']:.3f} / F1 {c['f1']:.3f} / "
                 f"ECE {c['ECE']:.3f}", fontsize=11)
    ax.legend(fontsize=8)

    # (4) MAE vs baselines (+ before-FT bar in fine-tune mode)
    ax = axes[1, 1]
    names  = ["model"]
    maes   = [r["model"]["MAE"]]
    colors = ["#2a78d6"]
    if pre_res is not None:
        names.insert(0, "before FT")
        maes.insert(0, pre_res["regression"]["model"]["MAE"])
        colors.insert(0, "#9085e9")
        names[1] = "after FT"
    names += ["naive\n(global mean)", "persistence\n(avg_dur)"]
    maes  += [r["naive"]["MAE"], r["persistence"]["MAE"]]
    colors += ["#898781", "#898781"]
    bars = ax.bar(names, maes, color=colors)
    for b, v in zip(bars, maes):
        ax.annotate(f"{v:.1f}s", xy=(b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    title = f"(4) 회귀 MAE vs baseline (val 양성)"
    if pre_res is not None:
        pa = pre_res["classification"]["AUROC"]
        title += f"\nAUROC {pa:.3f} → {c['AUROC']:.3f}"
    ax.set_ylabel("MAE (s)")
    ax.set_title(title, fontsize=11)

    fig.suptitle(f"train.py v{SCRIPT_VERSION} 학습 리포트 — mode={mode}"
                 + (f" (freeze_gru={FREEZE_GRU})" if mode == "finetune" else ""),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(SCRIPT_DIR, f"train_report_{mode}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"train report  → {out}")


if __name__ == "__main__":
    train()
