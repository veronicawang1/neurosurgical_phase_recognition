"""
Fine-tunes the ResNet50 backbone directly on raw video frames.
After running this, re-extract features with the fine-tuned checkpoint:
    python3 extract_features.py --checkpoint checkpoints/finetuned_backbone.pt
"""
import argparse
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from models.cnn_backbone import FrameBackbone


class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, labels_dir, transform):
        self.samples = []
        self.transform = transform

        for fname in os.listdir(video_dir):
            if not fname.endswith(".mp4"):
                continue
            stem = fname.replace(".mp4", "")
            label_path = os.path.join(labels_dir, stem + ".pt")
            if not os.path.exists(label_path):
                continue

            labels = torch.load(label_path)
            cap = cv2.VideoCapture(os.path.join(video_dir, fname))
            fps = cap.get(cv2.CAP_PROP_FPS)
            step = max(1, int(round(fps)))

            i = 0
            frame_idx = 0
            success, _ = cap.read()
            while success:
                if i % step == 0:
                    if frame_idx < labels.shape[0] and labels[frame_idx].item() != -100:
                        self.samples.append((
                            os.path.join(video_dir, fname),
                            i,
                            labels[frame_idx].item(),
                        ))
                    frame_idx += 1
                success, _ = cap.read()
                i += 1
            cap.release()

        print(f"  Loaded {len(self.samples)} labeled frames from {video_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_num, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame), torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/raw_videos")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Lower LR for backbone layers to avoid forgetting ImageNet knowledge")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Building dataset (reads frames from video files, takes a moment)...")
    dataset = VideoFrameDataset(args.video_dir, args.labels_dir, tfm)

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_tfm

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train: {n_train}  Val: {n_val}")

    model = FrameBackbone(num_classes=args.num_classes).to(device)

    # use a lower LR for the backbone, higher for the new classifier head
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            logits, _ = model(frames, return_features=True)
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
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                logits, _ = model(frames, return_features=True)
                val_loss += F.cross_entropy(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss/len(train_loader):.4f} acc={correct/total:.3f} | "
            f"val loss={val_loss/len(val_loader):.4f} acc={val_correct/val_total:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "finetuned_backbone.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")


if __name__ == "__main__":
    main()
