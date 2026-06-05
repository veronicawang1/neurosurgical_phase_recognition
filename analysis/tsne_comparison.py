"""
t-SNE of frozen vs fine-tuned CNN features side by side.
Run on VM from repo root:
    python3 analysis/tsne_comparison.py
Outputs: analysis/figures/tsne_comparison.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

os.makedirs("analysis/figures", exist_ok=True)

CLASS_NAMES  = ["Brain Exposure", "Parent Vessel ID", "Dome & Neck ID", "Clipping"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

FROZEN_DIR   = "data/features"
FINETUNED_DIR = "data/features_finetuned"
LABELS_DIR   = "data/labels"
MAX_FRAMES   = 4000  # subsample so t-SNE is fast


def load_features_and_labels(feat_dir, label_dir, max_frames):
    all_feats, all_labels = [], []
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    for ff in feat_files:
        stem = os.path.basename(ff).replace(".pt", "")
        lf   = os.path.join(label_dir, stem + ".pt")
        if not os.path.exists(lf):
            continue
        feats  = torch.load(ff, map_location="cpu")
        labels = torch.load(lf, map_location="cpu")
        valid  = labels != -100
        feats  = feats[valid[:len(feats)]]
        labels = labels[valid[:len(labels)]]
        all_feats.append(feats)
        all_labels.append(labels)
    all_feats  = torch.cat(all_feats,  dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    if len(all_feats) > max_frames:
        idx = np.random.choice(len(all_feats), max_frames, replace=False)
        all_feats  = all_feats[idx]
        all_labels = all_labels[idx]
    return all_feats, all_labels


np.random.seed(42)
print("Loading frozen features...")
f_frozen,  l_frozen  = load_features_and_labels(FROZEN_DIR,    LABELS_DIR, MAX_FRAMES)
print("Loading fine-tuned features...")
f_finetuned, l_finetuned = load_features_and_labels(FINETUNED_DIR, LABELS_DIR, MAX_FRAMES)

print(f"Frozen: {len(f_frozen)} frames  |  Fine-tuned: {len(f_finetuned)} frames")
print("Running t-SNE on frozen features...")
import sklearn
_tsne_kwargs = {"max_iter": 1000} if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4) else {"n_iter": 1000}
tsne_frozen    = TSNE(n_components=2, perplexity=40, random_state=42, **_tsne_kwargs).fit_transform(f_frozen)
print("Running t-SNE on fine-tuned features...")
tsne_finetuned = TSNE(n_components=2, perplexity=40, random_state=42, **_tsne_kwargs).fit_transform(f_finetuned)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, embed, labels, title in zip(
    axes,
    [tsne_frozen, tsne_finetuned],
    [l_frozen, l_finetuned],
    ["Frozen ImageNet Features", "Fine-tuned Backbone Features"],
):
    for cls in range(4):
        mask = labels == cls
        ax.scatter(embed[mask, 0], embed[mask, 1],
                   c=CLASS_COLORS[cls], s=4, alpha=0.5, label=CLASS_NAMES[cls])
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=11)
plt.suptitle("t-SNE of CNN Features: Before and After Backbone Fine-tuning", fontsize=14)
plt.tight_layout()
plt.savefig("analysis/figures/tsne_comparison.png", bbox_inches="tight")
print("Saved analysis/figures/tsne_comparison.png")
