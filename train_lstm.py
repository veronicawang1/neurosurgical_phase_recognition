import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.cnn_lstm import CnnLstm
from utils.class_weights import compute_class_weights


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


def run_epoch(model, loader, optimizer, device, train=True, class_weights=None, num_classes=5):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for features, labels, padding_mask in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=class_weights if train else None,
                ignore_index=-100,
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            c, n = _acc(logits, labels)
            correct += c
            total += n
            if not train:
                btp, bfp, bfn = _per_class_stats(logits.cpu(), labels.cpu(), num_classes)
                tp += btp
                fp += bfp
                fn += bfn

    acc = correct / total if total > 0 else 0.0
    per_class = None
    if not train:
        per_class = []
        for c in range(num_classes):
            c_acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
            denom = 2 * tp[c] + fp[c] + fn[c]
            f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
            per_class.append((c_acc, f1))
    return total_loss / len(loader), acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos with features+labels")

    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples, args.num_classes, device)

    train_loader = DataLoader(
        FeatureDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        FeatureDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
    )

    model = CnnLstm(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, optimizer, device, train=True,
                                              class_weights=class_weights, num_classes=args.num_classes)
        val_loss, val_acc, per_class = run_epoch(model, val_loader, optimizer, device, train=False,
                                                  num_classes=args.num_classes)
        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  val loss={val_loss:.4f} acc={val_acc:.3f}")
        for name, (acc, f1) in zip(CLASS_NAMES, per_class):
            acc_str = f"{acc:.3f}" if not math.isnan(acc) else " n/a"
            f1_str  = f"{f1:.3f}"  if not math.isnan(f1)  else " n/a"
            print(f"    {name:<22} acc={acc_str}  f1={f1_str}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_lstm.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
