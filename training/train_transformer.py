import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.augmentation import AugmentedFeatureDataset
from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.opera_transformer import NeuroOperA
from utils.class_weights import compute_class_weights
from utils.logger import RunLogger


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


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for features, labels, padding_mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)
        logits, attn = model(features, padding_mask)
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
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    num_classes = model.classifier.out_features
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
        tp += btp
        fp += bfp
        fn += bfn

    per_class = []
    for c in range(num_classes):
        acc = tp[c].item() / (tp[c] + fn[c]).item() if (tp[c] + fn[c]) > 0 else float("nan")
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else float("nan")
        per_class.append((acc, f1))

    return total_loss / len(loader), correct / total, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--run_name", default="transformer")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos with features+labels")

    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples, args.num_classes, device)

    train_dataset = (AugmentedFeatureDataset(train_samples) if args.augment
                     else FeatureDataset(train_samples))
    if args.augment:
        print("Data augmentation: ON (temporal crop + gaussian noise)")
    train_loader = DataLoader(
        train_dataset,
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

    model = NeuroOperA(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    logger = RunLogger(args.log_dir, args.run_name, vars(args))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, class_weights=class_weights)
        val_loss, val_acc, per_class = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  val loss={val_loss:.4f} acc={val_acc:.3f}")
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
            ckpt = os.path.join(args.checkpoint_dir, "best_transformer.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
