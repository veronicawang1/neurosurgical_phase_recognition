"""
Exact confusion matrix from best checkpoint on val set.
Run on VM from repo root:
    python3 analysis/eval_confusion.py --model transformer --features_dir data/features_finetuned
    python3 analysis/eval_confusion.py --model mstcn       --features_dir data/features_finetuned
Outputs: analysis/figures/confusion_<model>.png
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.opera_transformer import NeuroOperA
from models.mstcn import MSTCN
from utils.postprocess import viterbi_decode
from torch.utils.data import DataLoader

os.makedirs("analysis/figures", exist_ok=True)

CLASS_NAMES = ["Brain Exposure", "Parent Vessel ID", "Dome & Neck ID", "Clipping"]

parser = argparse.ArgumentParser()
parser.add_argument("--model",        choices=["transformer", "mstcn"], default="transformer")
parser.add_argument("--features_dir", default="data/features_finetuned")
parser.add_argument("--labels_dir",   default="data/labels")
parser.add_argument("--num_classes",  type=int, default=4)
args = parser.parse_args()

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

samples = build_samples(args.features_dir, args.labels_dir)
_, val_samples = train_val_split(samples)
val_loader = DataLoader(FeatureDataset(val_samples), batch_size=1,
                        shuffle=False, collate_fn=collate_variable_length)

if args.model == "transformer":
    model = NeuroOperA(num_classes=args.num_classes).to(device)
    ckpt  = "checkpoints/best_transformer_framelevel.pt"
else:
    model = MSTCN(num_classes=args.num_classes, num_filters=128).to(device)
    ckpt  = "checkpoints/best_mstcn.pt"

model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print(f"Loaded {ckpt}")

cm = np.zeros((args.num_classes, args.num_classes), dtype=int)

with torch.no_grad():
    for features, labels, padding_mask in val_loader:
        features = features.to(device)
        labels   = labels.to(device)

        if args.model == "transformer":
            logits, _ = model(features)
        else:
            logits, _ = model(features)

        for b in range(features.shape[0]):
            seq_labels = labels[b].cpu()
            seq_logits = logits[b].cpu()
            valid = seq_labels != -100
            if valid.sum() == 0:
                continue
            probs = F.softmax(seq_logits[valid], dim=-1).numpy()
            gt    = seq_labels[valid].tolist()
            preds = viterbi_decode(probs, num_classes=args.num_classes).tolist()
            for g, p in zip(gt, preds):
                cm[g, p] += 1

# normalize rows to recall
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mat, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Confusion Matrix (frame counts)", "Confusion Matrix (normalized recall)"],
    [True, False],
):
    im = ax.imshow(mat, vmin=0, vmax=(None if fmt else 1), cmap="Blues")
    ax.set_xticks(range(args.num_classes))
    ax.set_yticks(range(args.num_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            v = mat[i, j]
            text = f"{v:.2f}" if not fmt else str(int(v))
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if (cm_norm[i, j] > 0.6) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"{args.model.upper()} — Viterbi predictions on val set (fine-tuned features)", fontsize=12)
plt.tight_layout()
out = f"analysis/figures/confusion_{args.model}.png"
plt.savefig(out)
print(f"Saved {out}")
print("\nRaw confusion matrix:")
print(cm)
