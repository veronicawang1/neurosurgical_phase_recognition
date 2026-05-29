"""
Transformer trainer with frame-level train/val split (for debugging/exploration).
Not a rigorous evaluation — use train_transformer.py (video-level split) for real results.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from data.splits import build_samples
from models.opera_transformer import NeuroOperA


class FlatFrameSequenceDataset(Dataset):
    """Wraps per-video samples into a flat list of (feature, label) frame pairs,
    then re-chunks them into fixed-length windows for the transformer."""

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


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits, _ = model(feats)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
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
    acc = correct / total if total > 0 else 0.0
    return total_loss / len(loader), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = build_samples(args.features_dir, args.labels_dir)
    print(f"Found {len(samples)} videos")

    dataset = FlatFrameSequenceDataset(samples, window=args.window)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    print(f"Windows — train: {n_train}  val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = NeuroOperA(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"Epoch {epoch:03d}  train loss={train_loss:.4f} acc={train_acc:.3f}  val loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_transformer_framelevel.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
