"""
MS-TCN trainer with video-level val split, Viterbi post-processing, and temporal smoothing.
"""
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.mstcn import MSTCN
from utils.class_weights import compute_class_weights
from utils.logger import RunLogger
from utils.metrics import edit_distance, segmental_f1, apply_boundary_mask
from utils.postprocess import viterbi_decode, temporal_smooth, compute_transition_matrix


CLASS_NAMES = [
    "Brain Exposure",
    "Parent Vessel ID",
    "Dome and Neck ID",
    "Clipping",
]


def _acc(logits, labels):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    if mask.sum() == 0:
        return 0, 0
    correct = (flat_logits[mask].argmax(1) == flat_labels[mask]).sum().item()
    return correct, mask.sum().item()


def _per_class_stats(logits, labels, num_classes):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    preds = flat_logits.argmax(1)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for c in range(num_classes):
        tp[c] = ((preds == c) & (flat_labels == c)).sum()
        fp[c] = ((preds == c) & (flat_labels != c)).sum()
        fn[c] = ((preds != c) & (flat_labels == c)).sum()
    return tp, fp, fn


def train_one_epoch(model, loader, optimizer, device, class_weights=None,
                    grad_clip=1.0, label_smoothing=0.1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for features, labels, padding_mask in loader:
        features = features.to(device)
        labels = labels.to(device)

        _, all_stage_logits = model(features)

        # multi-stage loss: sum loss from every stage
        loss = 0
        for stage_logits in all_stage_logits:
            loss += F.cross_entropy(
                stage_logits.reshape(-1, stage_logits.shape[-1]),
                labels.reshape(-1),
                weight=class_weights,
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() / len(all_stage_logits)
        c, n = _acc(all_stage_logits[-1], labels)
        correct += c
        total += n
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, device, num_classes, transition_matrix=None,
                   smooth_window=15, boundary_ignore_secs=5):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    all_preds, all_gts = [], []
    viterbi_preds, smooth_preds = [], []

    for features, labels, padding_mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        _, all_stage_logits = model(features)
        logits = all_stage_logits[-1]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        c, n = _acc(logits, labels)
        correct += c
        total += n
        btp, bfp, bfn = _per_class_stats(logits.cpu(), labels.cpu(), num_classes)
        tp += btp; fp += bfp; fn += bfn

        flat_labels = labels.reshape(-1).cpu()
        flat_preds = logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()
        mask = flat_labels != -100
        all_preds.extend(flat_preds[mask].tolist())
        all_gts.extend(flat_labels[mask].tolist())

        # per-video Viterbi and smoothing
        for b in range(features.shape[0]):
            seq_labels = labels[b].cpu()
            seq_logits = logits[b].cpu()
            valid_mask = seq_labels != -100
            if valid_mask.sum() == 0:
                continue
            probs = F.softmax(seq_logits[valid_mask], dim=-1).numpy()
            gt_seq = seq_labels[valid_mask].tolist()

            v_preds = viterbi_decode(probs, transition_matrix, num_classes)
            s_preds = temporal_smooth(probs, smooth_window)
            viterbi_preds.extend(v_preds.tolist())
            smooth_preds.extend(s_preds.tolist())

    per_class = []
    for c in range(num_classes):
        c_acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
        per_class.append((c_acc, f1))

    bp, bg = apply_boundary_mask(all_preds, all_gts, boundary_ignore_secs)
    boundary_acc = sum(p == g for p, g in zip(bp, bg)) / len(bp) if bp else 0.0

    extra = {
        "edit_distance": round(edit_distance(all_preds, all_gts), 4),
        "segmental_f1": segmental_f1(all_preds, all_gts),
        "boundary_aware_acc": round(boundary_acc, 4),
        "viterbi_edit_dist": round(edit_distance(viterbi_preds, all_gts), 4),
        "viterbi_seg_f1": segmental_f1(viterbi_preds, all_gts),
        "smooth_edit_dist": round(edit_distance(smooth_preds, all_gts), 4),
        "smooth_seg_f1": segmental_f1(smooth_preds, all_gts),
    }
    return total_loss / len(loader), correct / total, per_class, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--run_name", default="mstcn")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--smooth_window", type=int, default=15)
    parser.add_argument("--boundary_ignore_secs", type=int, default=5)
    parser.add_argument("--use_learned_transitions", action="store_true",
                        help="Estimate Viterbi transition matrix from training data")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos with features+labels")

    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples, args.num_classes, device)

    if args.use_learned_transitions:
        transition_matrix = compute_transition_matrix(
            args.labels_dir, args.num_classes, train_samples)
        print("Using learned transition matrix")
    else:
        transition_matrix = None
        print("Using default surgical phase transition matrix")

    train_loader = DataLoader(
        FeatureDataset(train_samples), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        FeatureDataset(val_samples), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_variable_length,
    )

    model = MSTCN(
        num_stages=args.num_stages, num_layers=args.num_layers,
        num_filters=args.num_filters, num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    logger = RunLogger(args.log_dir, args.run_name, vars(args))

    for epoch in range(1, args.epochs + 1):
        logger.start_epoch()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            class_weights=class_weights,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
        )
        scheduler.step()
        val_loss, val_acc, per_class, extra = eval_one_epoch(
            model, val_loader, device, args.num_classes,
            transition_matrix=transition_matrix,
            smooth_window=args.smooth_window,
            boundary_ignore_secs=args.boundary_ignore_secs,
        )

        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")
        print(f"    edit_dist={extra['edit_distance']:.3f}  "
              f"seg_f1@10={extra['segmental_f1'][0.1]:.3f}  "
              f"seg_f1@25={extra['segmental_f1'][0.25]:.3f}  "
              f"seg_f1@50={extra['segmental_f1'][0.5]:.3f}")
        print(f"    viterbi: edit={extra['viterbi_edit_dist']:.3f}  "
              f"seg_f1@10={extra['viterbi_seg_f1'][0.1]:.3f}")
        print(f"    smooth:  edit={extra['smooth_edit_dist']:.3f}  "
              f"seg_f1@10={extra['smooth_seg_f1'][0.1]:.3f}")
        print(f"    boundary_aware_acc={extra['boundary_aware_acc']:.3f}")

        per_class_metrics = {}
        for name, (acc, f1) in zip(CLASS_NAMES, per_class):
            acc_str = f"{acc:.3f}" if not math.isnan(acc) else " n/a"
            f1_str  = f"{f1:.3f}"  if not math.isnan(f1)  else " n/a"
            print(f"    {name:<22} acc={acc_str}  f1={f1_str}")
            per_class_metrics[name] = {"acc": None if math.isnan(acc) else round(acc, 4),
                                       "f1": None if math.isnan(f1) else round(f1, 4)}

        logger.log_epoch(epoch, {
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            **{k: v for k, v in extra.items()},
            "per_class": per_class_metrics,
        })

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_mstcn.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
