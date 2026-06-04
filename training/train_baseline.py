import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from models.cnn_backbone import FrameBackbone
from utils.logger import RunLogger


class FrameDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.samples = []
        for fname in os.listdir(features_dir):
            if not fname.endswith(".pt"):
                continue
            label_path = os.path.join(labels_dir, fname)
            if not os.path.exists(label_path):
                continue
            feats = torch.load(os.path.join(features_dir, fname))
            labels = torch.load(label_path)
            T = min(feats.shape[0], labels.shape[0])
            for i in range(T):
                if labels[i].item() == -100:
                    continue
                self.samples.append((feats[i], labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--run_name", default="baseline_cnn")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = FrameDataset(args.features_dir, args.labels_dir)
    print(f"Total labeled frames: {len(dataset)}")

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"Train: {n_train}  Val: {n_val}")

    model = FrameBackbone(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = RunLogger(args.log_dir, args.run_name, vars(args))
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model.classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model.classifier(feats)
                val_loss += F.cross_entropy(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        tl = train_loss / len(train_loader)
        ta = correct / total
        vl = val_loss / len(val_loader)
        va = val_correct / val_total
        print(f"Epoch {epoch:03d} | train loss={tl:.4f} acc={ta:.3f} | val loss={vl:.4f} acc={va:.3f}")
        logger.log_epoch(epoch, {"train_loss": round(tl, 4), "train_acc": round(ta, 4),
                                  "val_loss": round(vl, 4), "val_acc": round(va, 4)})

        if vl < best_val:
            best_val = vl
            ckpt = os.path.join(args.checkpoint_dir, "baseline_cnn.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
