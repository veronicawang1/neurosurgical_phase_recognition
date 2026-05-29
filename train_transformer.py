import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset, collate_variable_length
from data.splits import build_samples, train_val_split
from models.opera_transformer import NeuroOperA


def _acc(logits, labels):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    correct = (flat_logits[mask].argmax(1) == flat_labels[mask]).sum().item()
    return correct, mask.sum().item()


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
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos with features+labels")

    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

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

    model = NeuroOperA(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  val loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_transformer.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
