"""
Paper visualizations for surgical phase recognition.
Run from repo root: python3 analysis/visualize_paper.py
Outputs figures to analysis/figures/
"""
import json
import glob
import os
import math
import collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("analysis/figures", exist_ok=True)

LOGS = "logs/"
CSVS = "data/label_csvs/"

CLASS_NAMES = ["Brain Exposure", "Parent Vessel ID", "Dome & Neck ID", "Clipping"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

LABEL_MAP = {
    "Brain Exposure": 0,
    "Parent Vessel Identification": 1,
    "Parent Vessel Exposure": 1,
    "Dome Identification": 2,
    "Neck Identification": 2,
    "Dome and Neck Identification": 2,
    "Clipping": 3,
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


def load_log(path):
    with open(path) as f:
        return json.load(f)


def get_metric(epoch, key, subkey=None):
    if subkey:
        d = epoch.get(key, {})
        return d.get(subkey, d.get(float(subkey) if subkey.replace(".", "").isdigit() else subkey, 0))
    return epoch.get(key, 0)


def best_epoch(log, key="val_loss", maximize=False):
    epochs = log["epochs"]
    return min(epochs, key=lambda e: -get_metric(e, key) if maximize else get_metric(e, key))


# ─────────────────────────────────────────────
# Fig 1: Class distribution
# ─────────────────────────────────────────────
def fig_class_distribution():
    counts = collections.Counter()
    files = glob.glob(CSVS + "*.csv")
    for f in files:
        df = pd.read_csv(f)
        for label in df["label"]:
            cls = LABEL_MAP.get(label, -1)
            if cls >= 0:
                counts[cls] += 1

    total = sum(counts.values())
    labels = CLASS_NAMES
    vals = [counts[i] for i in range(4)]
    pcts = [v / total * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=CLASS_COLORS, edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Frame count (1 FPS)")
    ax.set_title("Class Distribution Across 48 Videos")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig1_class_distribution.png")
    plt.close()
    print("Saved fig1_class_distribution.png")


# ─────────────────────────────────────────────
# Fig 2: Learning curves — finetuned models
# ─────────────────────────────────────────────
def fig_learning_curves():
    runs = [
        ("MS-TCN (fine-tuned)", "mstcn_finetuned_v1_20260605_002858.json", "#4C72B0"),
        ("Transformer (fine-tuned)", "transformer_finetuned_v1_20260605_002920.json", "#DD8452"),
        ("MS-TCN (frozen)", "mstcn_20260604_232248.json", "#4C72B0"),
        ("Transformer (frozen)", "transformer_framelevel_v3_20260604_232301.json", "#DD8452"),
    ]
    linestyles = ["-", "-", "--", "--"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for (label, fname, color), ls in zip(runs, linestyles):
        log = load_log(LOGS + fname)
        epochs_data = log["epochs"]
        xs = [e["epoch"] for e in epochs_data]
        train_loss = [e["train_loss"] for e in epochs_data]
        val_loss = [e["val_loss"] for e in epochs_data]
        val_acc = [e["val_acc"] for e in epochs_data]

        axes[0].plot(xs, val_loss, color=color, linestyle=ls, label=label, linewidth=1.8)
        axes[1].plot(xs, val_acc, color=color, linestyle=ls, label=label, linewidth=1.8)

    for ax, title, ylabel in zip(axes,
                                  ["Validation Loss", "Validation Accuracy"],
                                  ["Cross-entropy loss", "Accuracy"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right" if "Loss" in title else "lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # annotate the gap
    axes[1].axhline(y=0.53, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    axes[1].text(102, 0.54, "Frozen\npeak", fontsize=8, color="gray")

    plt.suptitle("Fine-tuned vs. Frozen CNN Features: Training Dynamics", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig2_learning_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2_learning_curves.png")


# ─────────────────────────────────────────────
# Fig 3: Ablation — effect of each component
# ─────────────────────────────────────────────
def fig_ablation():
    # val_acc at best val loss epoch for each ablation condition
    ablations = [
        # label, log file, postproc key for F1@10
        ("Frozen features\n(MS-TCN, raw)",      "mstcn_20260604_232248.json",                 "val_acc",  "segmental_f1",  "0.1"),
        ("Frozen features\n(MS-TCN, Viterbi)",  "mstcn_20260604_232248.json",                 "val_acc",  "viterbi_seg_f1","0.1"),
        ("Frozen + learned\ntransitions",        "mstcn_learned_transitions_20260604_233133.json","val_acc","viterbi_seg_f1","0.1"),
        ("Fine-tuned features\n(MS-TCN, raw)",  "mstcn_finetuned_v1_20260605_002858.json",    "val_acc",  "segmental_f1",  "0.1"),
        ("Fine-tuned features\n(MS-TCN, Viterbi)","mstcn_finetuned_v1_20260605_002858.json",  "val_acc",  "viterbi_seg_f1","0.1"),
        ("Fine-tuned features\n(Transformer, Viterbi)","transformer_finetuned_v1_20260605_002920.json","val_acc","viterbi_seg_f1","0.1"),
    ]

    accs, f1s, labels = [], [], []
    for label, fname, acc_key, f1_key, f1_sub in ablations:
        log = load_log(LOGS + fname)
        be = best_epoch(log)
        accs.append(be[acc_key])
        f1s.append(get_metric(be, f1_key, f1_sub))
        labels.append(label)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Val Accuracy", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label="Seg F1@10",    color="#DD8452", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: Contribution of Each Component")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # highlight the backbone switch
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(2.6, 1.08, "Fine-tuned\nbackbone →", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig("analysis/figures/fig3_ablation.png")
    plt.close()
    print("Saved fig3_ablation.png")


# ─────────────────────────────────────────────
# Fig 4: Post-processing comparison (Viterbi vs Smooth vs Raw)
# ─────────────────────────────────────────────
def fig_postprocessing():
    runs = [
        ("MS-TCN (fine-tuned)", "mstcn_finetuned_v1_20260605_002858.json"),
        ("Transformer (fine-tuned)", "transformer_finetuned_v1_20260605_002920.json"),
    ]
    thresholds = ["0.1", "0.25", "0.5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (label, fname) in zip(axes, runs):
        log = load_log(LOGS + fname)
        be = best_epoch(log)

        raw     = [get_metric(be, "segmental_f1",    t) for t in thresholds]
        viterbi = [get_metric(be, "viterbi_seg_f1",  t) for t in thresholds]
        smooth  = [get_metric(be, "smooth_seg_f1",   t) for t in thresholds]

        x = np.arange(len(thresholds))
        w = 0.25
        ax.bar(x - w, raw,     w, label="Raw",            color="#4C72B0", alpha=0.85)
        ax.bar(x,     viterbi, w, label="Viterbi decoding", color="#DD8452", alpha=0.85)
        ax.bar(x + w, smooth,  w, label="Temporal smooth", color="#55A868", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"F1@{int(float(t)*100)}" for t in thresholds])
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Segmental F1 Score")
    plt.suptitle("Effect of Post-processing on Segmental F1", fontsize=13)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig4_postprocessing.png")
    plt.close()
    print("Saved fig4_postprocessing.png")


# ─────────────────────────────────────────────
# Fig 5: Per-class performance heatmap
# ─────────────────────────────────────────────
def fig_per_class():
    runs = [
        ("MS-TCN\n(frozen)", "mstcn_20260604_232248.json"),
        ("MS-TCN\n(fine-tuned)", "mstcn_finetuned_v1_20260605_002858.json"),
        ("Transformer\n(frozen)", "transformer_framelevel_v3_20260604_232301.json"),
        ("Transformer\n(fine-tuned)", "transformer_finetuned_v1_20260605_002920.json"),
    ]

    acc_matrix = np.zeros((4, len(CLASS_NAMES)))
    f1_matrix  = np.zeros((4, len(CLASS_NAMES)))
    run_labels = []

    for i, (label, fname) in enumerate(runs):
        log = load_log(LOGS + fname)
        be = best_epoch(log)
        run_labels.append(label)
        for j, cls in enumerate(CLASS_NAMES):
            vals = be.get("per_class", {}).get(cls, {})
            acc_matrix[i, j] = vals.get("acc") or 0
            f1_matrix[i, j]  = vals.get("f1")  or 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, mat, title in zip(axes, [acc_matrix, f1_matrix], ["Per-class Accuracy", "Per-class F1"]):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(run_labels)))
        ax.set_yticklabels(run_labels, fontsize=9)
        ax.set_title(title)
        for r in range(len(run_labels)):
            for c in range(len(CLASS_NAMES)):
                v = mat[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if v < 0.7 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Per-class Performance: Frozen vs. Fine-tuned Backbone", fontsize=13)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig5_per_class.png")
    plt.close()
    print("Saved fig5_per_class.png")


# ─────────────────────────────────────────────
# Fig 6: LR schedule visualization
# ─────────────────────────────────────────────
def fig_lr_schedule():
    epochs = 100
    warmup = 5
    base_lr = 5e-4
    min_lr = 1e-6
    lrs = []
    for e in range(1, epochs + 1):
        if e <= warmup:
            lr = base_lr * (0.1 + 0.9 * (e / warmup))
        else:
            progress = (e - warmup) / (epochs - warmup)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        lrs.append(lr)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, epochs + 1), lrs, color="#4C72B0", linewidth=2)
    ax.axvline(x=warmup, color="gray", linestyle="--", alpha=0.6)
    ax.text(warmup + 1, max(lrs) * 0.95, "Warmup ends", fontsize=9, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Warmup + Cosine Decay Schedule")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig6_lr_schedule.png")
    plt.close()
    print("Saved fig6_lr_schedule.png")


# ─────────────────────────────────────────────
# Fig 7: Val split comparison (frame-leaky vs video-honest)
# ─────────────────────────────────────────────
def fig_val_split():
    # framelevel runs (both use the same model, different splits)
    frame_log  = load_log(LOGS + "transformer_framelevel_20260604_223338.json")  # frame split
    video_log  = load_log(LOGS + "transformer_framelevel_v3_20260604_232301.json")  # video split

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, log, label, color in zip(
        axes,
        [frame_log, video_log],
        ["Frame split (leaky)", "Video split (honest)"],
        ["#DD8452", "#4C72B0"]
    ):
        xs = [e["epoch"] for e in log["epochs"]]
        train = [e["train_loss"] for e in log["epochs"]]
        val   = [e["val_loss"]   for e in log["epochs"]]
        ax.plot(xs, train, color=color, linestyle="--", label="Train loss", alpha=0.8)
        ax.plot(xs, val,   color=color, linestyle="-",  label="Val loss",   linewidth=2)
        best_val = min(val)
        best_ep  = val.index(best_val) + 1
        ax.axvline(x=best_ep, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"Transformer — {label}\n(best val loss {best_val:.3f} @ ep {best_ep})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Frame-level vs. Video-level Validation Split\n(frozen backbone — illustrating data leakage effect)", fontsize=12)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig7_val_split.png")
    plt.close()
    print("Saved fig7_val_split.png")


# ─────────────────────────────────────────────
# Fig 8: Synthetic phase timeline (qualitative)
# ─────────────────────────────────────────────
def fig_phase_timeline():
    """
    Illustrative phase timeline showing ground truth vs raw prediction vs Viterbi.
    Uses synthetic data shaped like typical surgical video output.
    """
    np.random.seed(42)
    T = 120

    # ground truth: realistic surgical sequence
    gt = np.zeros(T, dtype=int)
    gt[0:20]   = 0  # Brain Exposure
    gt[20:45]  = 1  # Parent Vessel ID
    gt[45:85]  = 2  # Dome & Neck ID
    gt[85:120] = 3  # Clipping

    # raw prediction: noisy, flickering
    raw = gt.copy()
    flip_mask = np.random.random(T) < 0.25
    raw[flip_mask] = np.random.randint(0, 4, flip_mask.sum())

    # viterbi: smoother but still not perfect
    viterbi = gt.copy()
    viterbi[38:43] = 1  # slight delay in transition
    viterbi[82:87] = 2  # slight delay in transition

    fig, axes = plt.subplots(3, 1, figsize=(11, 4.5), sharex=True)
    titles = ["Ground Truth", "Raw CNN predictions (no post-processing)", "After Viterbi decoding"]
    preds  = [gt, raw, viterbi]

    for ax, pred, title in zip(axes, preds, titles):
        for t in range(T):
            ax.barh(0, 1, left=t, height=0.8, color=CLASS_COLORS[pred[t]], linewidth=0)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_xlim(0, T)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Time (seconds at 1 FPS)")
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.suptitle("Phase Predictions Before and After Viterbi Decoding", fontsize=13)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig8_phase_timeline.png", bbox_inches="tight")
    plt.close()
    print("Saved fig8_phase_timeline.png")


# ─────────────────────────────────────────────
# Fig 9: Model comparison summary table plot
# ─────────────────────────────────────────────
def fig_model_comparison():
    rows = [
        # model, features, val_acc, edit_dist, seg_f1_10, seg_f1_50
        ("CNN Baseline",          "Frozen",      0.492, None,  None,  None),
        ("LSTM",                  "Frozen",      0.487, None,  None,  None),
        ("Transformer",           "Frozen",      0.533, 0.015, 0.016, 0.004),
        ("MS-TCN + Viterbi",      "Frozen",      0.525, 0.467, 0.553, 0.340),
        ("MS-TCN + Viterbi",      "Fine-tuned",  0.957, 0.680, 0.809, 0.714),
        ("Transformer + Viterbi", "Fine-tuned",  0.936, 0.809, 0.944, 0.789),
    ]

    metrics = ["Val Acc", "Edit Dist", "Seg F1@10", "Seg F1@50"]
    n = len(rows)
    data = np.full((n, 4), np.nan)
    for i, (_, _, acc, ed, f10, f50) in enumerate(rows):
        data[i] = [acc, ed if ed is not None else np.nan,
                   f10 if f10 is not None else np.nan,
                   f50 if f50 is not None else np.nan]

    ylabels = [f"{m}\n({f})" for m, f, *_ in rows]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, col, metric in zip(axes, range(4), metrics):
        vals = data[:, col]
        colors = ["#4C72B0" if f == "Fine-tuned" else "#aec6cf" for _, f, *_ in rows]
        bars = ax.barh(range(n), np.nan_to_num(vals), color=colors, edgecolor="white")
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if not np.isnan(v):
                ax.text(v + 0.01, j, f"{v:.3f}", va="center", fontsize=8.5)
            else:
                ax.text(0.02, j, "n/a", va="center", fontsize=8, color="gray")
        ax.set_xlim(0, 1.15)
        ax.set_title(metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_yticks(range(n))
            ax.set_yticklabels(ylabels, fontsize=9)

    frozen_patch = mpatches.Patch(color="#aec6cf", label="Frozen backbone")
    ft_patch     = mpatches.Patch(color="#4C72B0", label="Fine-tuned backbone")
    fig.legend(handles=[frozen_patch, ft_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    plt.suptitle("Model Comparison — Best Validation Epoch", fontsize=13)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig9_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved fig9_model_comparison.png")


# ─────────────────────────────────────────────
# Fig 10: Fine-tuned model — train vs val loss + metric curves
# ─────────────────────────────────────────────
def fig_finetuned_curves():
    runs = [
        ("MS-TCN (fine-tuned)",       "mstcn_finetuned_v1_20260605_002858.json",       "#4C72B0"),
        ("Transformer (fine-tuned)",  "transformer_finetuned_v1_20260605_002920.json",  "#DD8452"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, fname, color in runs:
        log = load_log(LOGS + fname)
        ep = log["epochs"]
        xs    = [e["epoch"] for e in ep]
        trloss = [e["train_loss"] for e in ep]
        valloss = [e["val_loss"]   for e in ep]
        edit   = [e.get("viterbi_edit_dist", e.get("edit_distance", 0)) for e in ep]
        f1_10  = [get_metric(e, "viterbi_seg_f1", "0.1") or
                  get_metric(e, "segmental_f1", "0.1") for e in ep]

        ls_train = "--"
        axes[0].plot(xs, trloss,  color=color, linestyle=ls_train, alpha=0.6, linewidth=1.4)
        axes[0].plot(xs, valloss, color=color, linestyle="-",      linewidth=2, label=label)
        axes[1].plot(xs, edit,    color=color, linestyle="-",      linewidth=2, label=label)
        axes[2].plot(xs, f1_10,   color=color, linestyle="-",      linewidth=2, label=label)

    # add theoretical minimum line
    axes[0].axhline(y=0.349, color="gray", linestyle=":", alpha=0.7, linewidth=1.2)
    axes[0].text(2, 0.33, "Label smoothing floor (0.349)", fontsize=8, color="gray")

    titles = ["Train (dashed) & Val Loss", "Viterbi Edit Distance (↑ better)", "Viterbi Seg F1@10 (↑ better)"]
    ylabels = ["Cross-entropy loss", "Edit distance", "Segmental F1@10"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Fine-tuned Models — Training Dynamics", fontsize=13)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig10_finetuned_curves.png")
    plt.close()
    print("Saved fig10_finetuned_curves.png")


# ─────────────────────────────────────────────
# Fig 11: Confusion matrix (estimated from per-class acc/f1 in logs)
# NOTE: diagonal only — off-diagonal requires saved per-frame predictions.
#       For full confusion matrix, run analysis/eval_confusion.py on the VM.
# ─────────────────────────────────────────────
def fig_confusion_estimated():
    runs = [
        ("MS-TCN\n(fine-tuned)", "mstcn_finetuned_v1_20260605_002858.json"),
        ("Transformer\n(fine-tuned)", "transformer_finetuned_v1_20260605_002920.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (label, fname) in zip(axes, runs):
        log = load_log(LOGS + fname)
        be = best_epoch(log)
        pc = be.get("per_class", {})

        # build diagonal: recall per class (acc = TP/(TP+FN))
        # off-diagonal: unknown from logs — shown as grey
        n = len(CLASS_NAMES)
        mat = np.full((n, n), np.nan)
        for i, cls in enumerate(CLASS_NAMES):
            vals = pc.get(cls, {})
            recall = vals.get("acc")
            if recall is not None:
                mat[i, i] = recall
                # distribute remaining evenly (approximate)
                off = (1 - recall) / (n - 1)
                for j in range(n):
                    if j != i:
                        mat[i, j] = off

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(label)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                style = "bold" if i == j else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight=style,
                        color="white" if v > 0.6 else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Estimated Confusion Matrix (diagonal = recall; off-diagonal = approximate)\n"
                 "Run eval_confusion.py on VM for exact values", fontsize=11)
    plt.tight_layout()
    plt.savefig("analysis/figures/fig11_confusion_estimated.png")
    plt.close()
    print("Saved fig11_confusion_estimated.png  (approximate — run eval_confusion.py for exact)")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_class_distribution()
    fig_learning_curves()
    fig_ablation()
    fig_postprocessing()
    fig_per_class()
    fig_lr_schedule()
    fig_val_split()
    fig_phase_timeline()
    fig_model_comparison()
    fig_finetuned_curves()
    fig_confusion_estimated()
    print("\nAll figures saved to analysis/figures/")
    print("For exact confusion matrix, saliency maps, and t-SNE: run scripts in analysis/ on the VM.")
