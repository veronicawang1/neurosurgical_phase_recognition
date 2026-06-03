"""
t-SNE visualization of CNN features colored by surgical phase.
Run on the VM after feature extraction:
    python3 visualize_features.py
Saves a plot to visualizations/tsne_features.png
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

from data.splits import build_samples

FEATURES_DIR = "data/features"
LABELS_DIR = "data/labels"
OUT_DIR = "visualizations"

CLASS_NAMES = [
    "Brain Exposure",
    "Parent Vessel ID",
    "Dome and Neck ID",
    "Clipping",
]
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading features and labels...")
samples = build_samples(FEATURES_DIR, LABELS_DIR)

all_feats = []
all_labels = []

for s in samples:
    feats = torch.load(s["features"])
    labels = torch.load(s["labels"])
    T = min(feats.shape[0], labels.shape[0])
    feats, labels = feats[:T], labels[:T]

    mask = labels != -100
    feats = feats[mask]
    labels = labels[mask]

    all_feats.append(feats)
    all_labels.append(labels)

all_feats = torch.cat(all_feats, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

print(f"Total frames: {len(all_feats)}")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(all_labels == i).sum()}")

# subsample if large to keep t-SNE fast
MAX_SAMPLES = 5000
if len(all_feats) > MAX_SAMPLES:
    idx = np.random.choice(len(all_feats), MAX_SAMPLES, replace=False)
    all_feats = all_feats[idx]
    all_labels = all_labels[idx]
    print(f"Subsampled to {MAX_SAMPLES} frames for t-SNE")

print("Running t-SNE (this takes a few minutes)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedded = tsne.fit_transform(all_feats)

print("Plotting...")
fig, ax = plt.subplots(figsize=(10, 8))

for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    mask = all_labels == i
    ax.scatter(
        embedded[mask, 0],
        embedded[mask, 1],
        c=color,
        label=name,
        alpha=0.5,
        s=10,
        linewidths=0,
    )

ax.set_title("t-SNE of CNN Features by Surgical Phase", fontsize=14)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(markerscale=3, fontsize=11)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "tsne_features.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
