"""
Transformer trainer with window-based training.

Two val split modes (--val_split):
  frame  : windows split randomly across train/val (leaky but optimistic, good for debugging)
  video  : videos split at video level, val windows come from held-out videos (honest evaluation)
"""
import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from data.dataset import FeatureDataset, collate_variable_length
from utils.logger import RunLogger
from data.splits import build_samples, train_val_split
from models.opera_transformer import NeuroOperA
from utils.class_weights import compute_class_weights


CLASS_NAMES = [
    "Brain Exposure",
    "Parent Vessel ID",
    "Dome and Neck ID",
    "Clipping",
]


class WindowDataset(Dataset):
    def __init__(self, samples, window=64):
        self.windows = []
        for s in samples:
            feats = torch.load(s["features"])
            labels = torch.load(s["labels"])
            T = min(feats.shape[0], labels.shape[0])
            feats, labels = feats[:T], labels[:T]
            for start in range(0, T - window + 1, window):
                self.windows.append((feats[start:start+window], labels[start:start+window]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def collate(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.stack(labels)


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


def run_train(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits, _ = model(feats)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=class_weights,
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        c, n = _acc(logits, labels)
        correct += c
        total += n
    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def run_val_windows(model, loader, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits, _ = model(feats)
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
    per_class = []
    for c in range(num_classes):
        c_acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
        per_class.append((c_acc, f1))
    return total_loss / len(loader), correct / total if total > 0 else 0.0, per_class


@torch.no_grad()
def run_val_videos(model, loader, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for features, labels, padding_mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)
        logits, _ = model(features, padding_mask)
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
    per_class = []
    for c in range(num_classes):
        c_acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
        per_class.append((c_acc, f1))
    return total_loss / len(loader), correct / total if total > 0 else 0.0, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--val_split", choices=["frame", "video"], default="video",
                        help="frame: leaky window split (debug); video: honest held-out video split")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--run_name", default="transformer_framelevel")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  val_split={args.val_split}")

    all_samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(all_samples)} videos")

    if args.val_split == "video":
        train_samples, val_samples = train_val_split(all_samples)
        print(f"Videos — train: {len(train_samples)}  val: {len(val_samples)}")

        train_dataset = WindowDataset(train_samples, window=args.window)
        print(f"Train windows: {len(train_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate)
        val_loader = DataLoader(
            FeatureDataset(val_samples),
            batch_size=4, shuffle=False,
            collate_fn=collate_variable_length,
        )
        val_fn = lambda m: run_val_videos(m, val_loader, device, args.num_classes)

    else:
        dataset = WindowDataset(all_samples, window=args.window)
        n_val = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(42))
        print(f"Windows — train: {n_train}  val: {n_val}  (leaky frame split)")

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate)
        val_fn = lambda m: run_val_windows(m, val_loader, device, args.num_classes)

    class_weights = compute_class_weights(
        train_samples if args.val_split == "video" else all_samples,
        args.num_classes, device
    )

    model = NeuroOperA(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    logger = RunLogger(args.log_dir, args.run_name, vars(args))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_train(model, train_loader, optimizer, device,
                                          class_weights=class_weights)
        val_loss, val_acc, per_class = val_fn(model)
        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")
        per_class_metrics = {}
        for name, (acc, f1) in zip(CLASS_NAMES, per_class):
            acc_str = f"{acc:.3f}" if not math.isnan(acc) else " n/a"
            f1_str  = f"{f1:.3f}"  if not math.isnan(f1)  else " n/a"
            print(f"    {name:<22} acc={acc_str}  f1={f1_str}")
            per_class_metrics[name] = {"acc": None if math.isnan(acc) else round(acc, 4),
                                       "f1": None if math.isnan(f1) else round(f1, 4)}
        logger.log_epoch(epoch, {"train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                                  "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                                  "per_class": per_class_metrics})

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_transformer_framelevel.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
