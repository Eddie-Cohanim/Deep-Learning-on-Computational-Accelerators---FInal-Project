"""
Alcohol Image Classifier — Full Evaluation Suite
=================================================
Reads your experiment JSON and produces 4 full-size plots + CSV.

JSON modes supported
--------------------
MODE A (your current format) — aggregated per-class metrics only:
    Produces: plots 1 (metrics bar), 2 (summary scorecards), 3 (IoU vs F1),
              4 (hyperparameter info card)
    NOT possible without raw data: confusion matrix, ROC curves, PR curves

MODE B (recommended additions) — add these fields to your JSON:
    "raw_predictions": [
        {"ground_truth": "Malbec", "predicted": "Pinot Noir", "confidence": 0.72},
        ...
    ]
    Unlocks: confusion matrix + ROC/PR curves

Usage
-----
    pip install scikit-learn pandas matplotlib numpy
    python confusion_matrix_eval.py                          # uses results_example.json
    python confusion_matrix_eval.py my_results.json          # custom path
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD JSON
# ──────────────────────────────────────────────────────────────────────────────

json_path = sys.argv[1] if len(sys.argv) > 1 else "results_example.json"
with open(json_path) as f:
    data = json.load(f)

timestamp          = data.get("timestamp", "N/A")
hyperparams        = data.get("hyperparameters", {})
model_cfg          = hyperparams.get("model", {})
training_cfg       = hyperparams.get("training", {})
cross_val_summary  = data.get("cross_validation", {})
results        = data["results"]
if "per_class_results" not in results:
    print(f"[SKIPPED] {json_path} — missing 'per_class_results' (unsupported format)")
    sys.exit(0)
per_class_raw  = results["per_class_results"]
raw_preds      = results.get("raw_predictions", data.get("raw_predictions", None))   # optional

# ──────────────────────────────────────────────────────────────────────────────
# 2.  CLASSES & COLORS
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = model_cfg.get("class_names", list(per_class_raw.keys()))

# Auto-assign colors — wine palette
PALETTE = [
    "#FFB6C1", "#FFA500", "#F0E68C", "#FFFACD",
    "#A93226", "#C0392B", "#7B241C", "#66BB6A",
    "#42A5F5", "#AB47BC", "#FFA726", "#EF5350",
]
CLASS_COLORS = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(CLASSES)}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  BUILD stats DICT FROM JSON
#     Derive what we can; mark what requires raw predictions
# ──────────────────────────────────────────────────────────────────────────────

def iou_from_f1(f1):
    """IoU = F1 / (2 - F1)  — exact algebraic identity when TP+FP+FN > 0."""
    return f1 / (2 - f1) if f1 < 1.0 else 1.0

stats = {}
for cls in CLASSES:
    if cls not in per_class_raw:
        print(f"[WARNING] Class '{cls}' in class_names but missing from per_class_results — skipping.")
        continue
    c   = per_class_raw[cls]
    p   = c["precision"]
    r   = c["recall"]
    f1  = c["f1"]
    n   = c["total_samples_tested"]
    tp  = round(r * n)          # TP  = recall * total_true_positives
    fn  = n - tp                # FN  = missed true positives
    iou = round(iou_from_f1(f1), 4)

    stats[cls] = dict(
        Precision = round(p,  4),
        Recall    = round(r,  4),
        F1        = round(f1, 4),
        IoU       = iou,
        Samples   = n,
        TP        = tp,
        FN        = fn,
        # FP & TN require cross-class info — filled below if raw_preds available
        FP        = None,
        TN        = None,
        AUC       = None,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4.  IF raw_predictions PRESENT — compute confusion matrix + AUC
# ──────────────────────────────────────────────────────────────────────────────

has_raw      = raw_preds is not None and len(raw_preds) > 0
has_conf     = has_raw and "confidence" in raw_preds[0]
ground_truths, predictions, confidences = [], [], []

if has_raw:
    for row in raw_preds:
        ground_truths.append(row["ground_truth"])
        predictions.append(row["predicted"])
        if has_conf:
            confidences.append(row.get("confidence", 1.0))

    cm_matrix = confusion_matrix(ground_truths, predictions, labels=CLASSES)
    n_total   = len(ground_truths)

    for i, cls in enumerate(CLASSES):
        tp = cm_matrix[i, i]
        fp = cm_matrix[:, i].sum() - tp
        fn = cm_matrix[i, :].sum() - tp
        tn = n_total - tp - fp - fn
        stats[cls].update(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

    if has_conf:
        y_true_bin   = label_binarize(ground_truths, classes=CLASSES)
        score_matrix = np.zeros((n_total, len(CLASSES)))
        cls_to_idx   = {c: i for i, c in enumerate(CLASSES)}
        for row_i, (row, conf) in enumerate(zip(raw_preds, confidences)):
            pred_idx = cls_to_idx[row["predicted"]]
            score_matrix[row_i, pred_idx] = conf
            remaining = (1 - conf) / max(len(CLASSES)-1, 1)
            for j in range(len(CLASSES)):
                if j != pred_idx:
                    score_matrix[row_i, j] = remaining
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], score_matrix[:, i])
            stats[cls]["AUC"] = round(auc(fpr, tpr), 4)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  CONSOLE TABLE
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_val_accuracy():
    """Return val accuracy, falling back to cross-validation mean when absent."""
    direct = results.get("val_accuracy")
    if direct is not None:
        return direct
    return cross_val_summary.get("mean_val_accuracy")

def _resolve_val_loss():
    """Return val loss, falling back to cross-validation mean when absent."""
    direct = results.get("val_loss")
    if direct is not None:
        return direct
    return cross_val_summary.get("mean_val_loss")

def print_table():
    resolved_val_accuracy = _resolve_val_accuracy()
    resolved_val_loss     = _resolve_val_loss()
    print(f"\n{'='*80}")
    print(f"  RESULTS — {timestamp}")
    print(f"  Model: {training_cfg.get('optimizer','?')} lr={training_cfg.get('learning_rate','?')}  "
          f"epochs={training_cfg.get('num_epochs','?')}  batch={training_cfg.get('batch_size','?')}")
    val_acc_display  = f"{resolved_val_accuracy:.4f}" if resolved_val_accuracy is not None else "?"
    val_loss_display = f"{resolved_val_loss:.4f}"     if resolved_val_loss     is not None else "?"
    test_acc_display = results.get('test_accuracy')
    test_acc_display = f"{test_acc_display:.4f}" if test_acc_display is not None else "?"
    print(f"  Val accuracy: {val_acc_display}   "
          f"Test accuracy: {test_acc_display}   "
          f"Val loss: {val_loss_display}")
    print(f"{'='*80}")

    has_fp = any(stats[c]["FP"] is not None for c in CLASSES if c in stats)
    has_auc= any(stats[c]["AUC"] is not None for c in CLASSES if c in stats)

    cols = ["Class","Samples","TP","FN"]
    if has_fp: cols += ["FP","TN"]
    cols += ["Precision","Recall","F1","IoU"]
    if has_auc: cols.append("AUC")

    widths = {"Class":22,"Samples":8,"TP":5,"FN":5,"FP":5,"TN":6,
              "Precision":10,"Recall":8,"F1":8,"IoU":8,"AUC":8}
    header = "  ".join(f"{c:>{widths[c]}}" for c in cols)
    sep    = "-" * len(header)
    print(header); print(sep)

    for cls in CLASSES:
        if cls not in stats: continue
        s  = stats[cls]
        row_vals = [cls, s["Samples"], s["TP"], s["FN"]]
        if has_fp: row_vals += [s["FP"] or "—", s["TN"] or "—"]
        row_vals += [s["Precision"], s["Recall"], s["F1"], s["IoU"]]
        if has_auc: row_vals.append(s["AUC"] or "—")

        parts = []
        for col, val in zip(cols, row_vals):
            w = widths[col]
            if col == "Class":
                parts.append(f"{val:<{w}}")
            elif isinstance(val, int):
                parts.append(f"{val:>{w}}")
            elif isinstance(val, float):
                parts.append(f"{val:>{w}.4f}")
            else:
                parts.append(f"{str(val):>{w}}")
        print("  ".join(parts))

    print(sep)
    macro_p   = np.mean([stats[c]["Precision"] for c in CLASSES if c in stats])
    macro_r   = np.mean([stats[c]["Recall"]    for c in CLASSES if c in stats])
    macro_f1  = np.mean([stats[c]["F1"]        for c in CLASSES if c in stats])
    macro_iou = np.mean([stats[c]["IoU"]       for c in CLASSES if c in stats])
    print(f"\n  Macro Avg  — Precision: {macro_p:.4f}  Recall: {macro_r:.4f}  "
          f"F1: {macro_f1:.4f}  IoU: {macro_iou:.4f}")
    if not has_raw:
        print("\n  [INFO] Confusion matrix & AUC-ROC require 'raw_predictions' in JSON.")
    print()

print_table()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  SAVE CSV
# ──────────────────────────────────────────────────────────────────────────────

pd.DataFrame([{"class": c, **stats[c]} for c in CLASSES if c in stats]
             ).to_csv("per_class_metrics.csv", index=False)
print("Saved: per_class_metrics.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SHARED STYLE
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D27"
ACCENT   = "#6C63FF"
TEXT_COL = "#E8EAF0"
GRID_COL = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor":   GRID_COL,"axes.labelcolor": TEXT_COL,
    "xtick.color":      TEXT_COL,"ytick.color":     TEXT_COL,
    "text.color":       TEXT_COL,"grid.color":      GRID_COL,
    "grid.linewidth":   0.6,     "font.family":     "DejaVu Sans",
})

def save(fig, fname):
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved: {fname}")

colors = [CLASS_COLORS[c] for c in CLASSES if c in stats]
valid_classes = [c for c in CLASSES if c in stats]

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Per-Class Metrics Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 8), facecolor=DARK_BG)
fig.suptitle("Per-Class Metrics: Precision / Recall / F1 / IoU",
             fontsize=18, fontweight="bold", color=TEXT_COL, y=1.01)
ax.set_facecolor(CARD_BG)

metric_list   = ["Precision", "Recall", "F1", "IoU"]
metric_colors = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]
width         = 0.18
x             = np.arange(len(valid_classes))
n_m           = len(metric_list)
offsets       = np.linspace(-(n_m-1)/2*width, (n_m-1)/2*width, n_m)

for k, (metric, mc) in enumerate(zip(metric_list, metric_colors)):
    vals = [stats[cls][metric] for cls in valid_classes]
    bars = ax.bar(x + offsets[k], vals, width, color=mc, alpha=0.85,
                  label=metric, edgecolor=DARK_BG, linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5,
                color=mc, fontweight="bold")

# Class color dots on x-axis
for i, cls in enumerate(valid_classes):
    ax.add_patch(plt.Circle((i, -0.07), 0.06, color=CLASS_COLORS[cls],
                             transform=ax.get_xaxis_transform(), clip_on=False, zorder=5))

ax.set_xticks(x)
ax.set_xticklabels([c.replace(" ", "\n") for c in valid_classes], fontsize=11)
ax.tick_params(axis="x", pad=16)
ax.set_ylim(0, 1.20)
ax.set_ylabel("Score", fontsize=13)
ax.axhline(1.0, color=GRID_COL, lw=1)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=11, loc="lower right", framealpha=0.3,
          facecolor=CARD_BG, edgecolor=GRID_COL, ncol=4)

# Overall accuracy annotation
oa = results.get("test_accuracy") or _resolve_val_accuracy()
if oa:
    ax.axhline(oa, color=ACCENT, lw=1.5, ls="--", alpha=0.8)
    ax.text(len(valid_classes)-0.4, oa+0.012, f"Test Acc {oa:.3f}",
            color=ACCENT, fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig, "plot_1_metrics_bar_chart.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Summary Scorecards
# ──────────────────────────────────────────────────────────────────────────────

macro_p   = np.mean([stats[c]["Precision"] for c in valid_classes])
macro_r   = np.mean([stats[c]["Recall"]    for c in valid_classes])
macro_f1  = np.mean([stats[c]["F1"]        for c in valid_classes])
macro_iou = np.mean([stats[c]["IoU"]       for c in valid_classes])
test_acc  = results.get("test_accuracy") or _resolve_val_accuracy() or 0
val_loss  = _resolve_val_loss()

summary = [
    ("Test Accuracy",   test_acc,   "Overall fraction of test samples predicted correctly"),
    ("Macro Precision", macro_p,    "Avg: of all predicted as class X, how many truly are X"),
    ("Macro Recall",    macro_r,    "Avg: of all true class X, how many did the model catch"),
    ("Macro F1",        macro_f1,   "Avg: harmonic mean of Precision and Recall"),
    ("Macro IoU",       macro_iou,  "Avg: TP / (TP+FP+FN) — strictest overlap score"),
]
if val_loss is not None:
    # Invert loss to fit 0-1 scorecard (lower is better — show as 1-normalised)
    summary.append(("Val Loss", val_loss, "Validation cross-entropy loss (lower = better)"))

fig, ax = plt.subplots(figsize=(13, 10), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG); ax.axis("off")
fig.suptitle("Overall Model Summary", fontsize=22, fontweight="bold",
             color=TEXT_COL, y=0.97)

card_h   = 0.115
card_gap = 0.018
y_start  = 0.88

for idx, (label, val, desc) in enumerate(summary):
    y     = y_start - idx * (card_h + card_gap)
    # For loss: red=high, green=low  (inverted)
    if label == "Val Loss":
        color    = "#4CAF50" if val<=0.5 else "#FF9800" if val<=1.0 else "#F44336"
        fill_val = max(0, 1 - val)           # bar fills inversely with loss
        disp     = f"{val:.4f}"
    else:
        color    = "#4CAF50" if val>=0.85 else "#FF9800" if val>=0.70 else "#F44336"
        fill_val = val
        disp     = f"{val:.4f}"

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, y - card_h*0.5), 0.94, card_h,
        boxstyle="round,pad=0.012", facecolor=CARD_BG, edgecolor=color,
        linewidth=2.5, transform=ax.transAxes, clip_on=False, zorder=2))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, y - card_h*0.5), 0.94*fill_val, card_h,
        boxstyle="round,pad=0.012", facecolor=color, alpha=0.18,
        transform=ax.transAxes, clip_on=False, zorder=1))
    ax.text(0.07, y + 0.018, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=TEXT_COL, va="center", zorder=3)
    ax.text(0.07, y - 0.025, desc, transform=ax.transAxes,
            fontsize=9.5, color="#8A8FA8", va="center", style="italic", zorder=3)
    ax.text(0.93, y, disp, transform=ax.transAxes,
            fontsize=22, fontweight="bold", color=color,
            va="center", ha="right", zorder=3)

ax.text(0.50, 0.03,
        "  🟢  >= 0.85   excellent          🟡  >= 0.70   acceptable          🔴  < 0.70   needs work  ",
        transform=ax.transAxes, fontsize=11, color=TEXT_COL,
        ha="center", va="center", alpha=0.75)

plt.tight_layout()
save(fig, "plot_2_summary_scorecards.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 — IoU vs F1 Scatter  (shows which classes are hardest)
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
fig.suptitle("IoU vs F1 per Class  (ideal = top-right corner)",
             fontsize=18, fontweight="bold", color=TEXT_COL, y=1.01)
ax.set_facecolor(CARD_BG)

for cls in valid_classes:
    s   = stats[cls]
    sz  = s["Samples"] * 10
    ax.scatter(s["F1"], s["IoU"], s=sz, color=CLASS_COLORS[cls],
               alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(cls.replace(" ", "\n"),
                (s["F1"], s["IoU"]),
                textcoords="offset points", xytext=(10, 4),
                fontsize=9, color=CLASS_COLORS[cls], fontweight="bold")

ax.axhline(macro_iou, color=ACCENT, lw=1.5, ls="--", alpha=0.7,
           label=f"Macro IoU = {macro_iou:.3f}")
ax.axvline(macro_f1,  color="#FFA726", lw=1.5, ls="--", alpha=0.7,
           label=f"Macro F1  = {macro_f1:.3f}")

# Quadrant shading
ax.axhspan(macro_iou, 1.05, xmin=0, xmax=1, alpha=0.04, color="#4CAF50")
ax.axhspan(0, macro_iou,  xmin=0, xmax=1, alpha=0.04, color="#F44336")

ax.set_xlabel("F1 Score",  fontsize=13, labelpad=8)
ax.set_ylabel("IoU Score", fontsize=13, labelpad=8)
ax.set_xlim(0.4, 1.05); ax.set_ylim(0.25, 1.05)
ax.grid(True, alpha=0.25)
ax.tick_params(labelsize=11)

# Bubble size legend
for n_s in [20, 25, 30]:
    ax.scatter([], [], s=n_s*10, color="grey", alpha=0.5, label=f"n={n_s} samples")

ax.legend(fontsize=10, loc="lower right", framealpha=0.3,
          facecolor=CARD_BG, edgecolor=GRID_COL)
ax.text(0.98, 0.98, "Bubble size ∝ sample count",
        transform=ax.transAxes, fontsize=8.5, color=TEXT_COL,
        ha="right", va="top", alpha=0.6)

plt.tight_layout()
save(fig, "plot_3_iou_vs_f1_scatter.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Hyperparameter Info Card
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 8), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG); ax.axis("off")
fig.suptitle("Experiment Configuration", fontsize=20, fontweight="bold",
             color=TEXT_COL, y=0.97)

def info_card(ax, title, items, x, y, w, h, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015",
        facecolor=CARD_BG, edgecolor=color, linewidth=2,
        transform=ax.transAxes, clip_on=False))
    ax.text(x + 0.015, y + h - 0.035, title,
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=color, va="top")
    for i, (k, v) in enumerate(items):
        ax.text(x + 0.02, y + h - 0.075 - i*0.052, f"{k}:",
                transform=ax.transAxes, fontsize=9.5, color="#8A8FA8", va="top")
        ax.text(x + w - 0.015, y + h - 0.075 - i*0.052, str(v),
                transform=ax.transAxes, fontsize=9.5, color=TEXT_COL,
                va="top", ha="right", fontweight="bold")

model_items = [
    ("Input size",    "×".join(map(str, model_cfg.get("in_size",[])))),
    ("Channels",      str(model_cfg.get("channels",[]))),
    ("Hidden dims",   str(model_cfg.get("hidden_dims",[]))),
    ("Activation",    model_cfg.get("activation","?")),
    ("Batch norm",    str(model_cfg.get("use_batchnorm","?"))),
    ("Dropout",       model_cfg.get("dropout_probability","?")),
    ("Pooling",       model_cfg.get("pooling_type","?")),
    ("Conv kernel",   model_cfg.get("conv_kernel_size","?")),
]
train_items = [
    ("Epochs",        training_cfg.get("num_epochs","?")),
    ("Batch size",    training_cfg.get("batch_size","?")),
    ("Learning rate", training_cfg.get("learning_rate","?")),
    ("Optimizer",     training_cfg.get("optimizer","?")),
    ("Weight decay",  training_cfg.get("weight_decay","?")),
    ("Loss fn",       training_cfg.get("loss_function","?")),
    ("Early stop",    training_cfg.get("early_stopping_patience","?")),
]
result_items = [
    ("Val accuracy",  f"{(_resolve_val_accuracy() or 0):.4f}"),
    ("Test accuracy", f"{(results.get('test_accuracy') or 0):.4f}"),
    ("Val loss",      f"{(_resolve_val_loss() or 0):.4f}"),
    ("# Classes",     len(valid_classes)),
    ("Timestamp",     timestamp[:10]),
]

info_card(ax, "Model Architecture", model_items,  0.03, 0.08, 0.44, 0.82, "#42A5F5")
info_card(ax, "Training Config",    train_items,  0.53, 0.08, 0.44, 0.72, "#66BB6A")
info_card(ax, "Results",            result_items, 0.53, 0.00, 0.44, 0.10 + 5*0.052, "#FFA726")

# Overwrite results card properly
ax.patches[-1].set_visible(False)
info_card(ax, "Results", result_items, 0.53, -0.04, 0.44, 0.45, "#FFA726")

plt.tight_layout()
save(fig, "plot_4_experiment_config.png")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 5 (conditional) — Confusion Matrix  [only if raw_predictions present]
# ──────────────────────────────────────────────────────────────────────────────

if has_raw:
    fig, ax = plt.subplots(figsize=(13, 11), facecolor=DARK_BG)
    fig.suptitle("Confusion Matrix", fontsize=20, fontweight="bold",
                 color=TEXT_COL, y=1.01)

    cmap = LinearSegmentedColormap.from_list("dark_blue",
           [CARD_BG, "#1565C0", "#42A5F5", "#E3F2FD"])
    im   = ax.imshow(cm_matrix, cmap=cmap, aspect="auto", vmin=0)
    cb   = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Count")
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)

    thresh = cm_matrix.max() * 0.55
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm_matrix[i,j]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm_matrix[i,j]>thresh else TEXT_COL)

    tl = [c.replace(" ", "\n") for c in CLASSES]
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(tl, fontsize=10, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(tl, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=13, labelpad=8)
    plt.tight_layout()
    save(fig, "plot_5_confusion_matrix.png")
else:
    print("\n  [SKIPPED] plot_5_confusion_matrix.png — add 'raw_predictions' to JSON to unlock.")

print("\nDone.")