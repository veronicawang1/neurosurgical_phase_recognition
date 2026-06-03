import argparse
import os
import cv2
import torch
from torchvision import transforms
from models.cnn_backbone import FrameBackbone

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None,
                    help="Path to fine-tuned backbone checkpoint. "
                         "If not provided, uses frozen ImageNet weights.")
parser.add_argument("--video_dir", default="data/raw_videos")
parser.add_argument("--out_dir", default="data/features")
args = parser.parse_args()

VIDEO_DIR = args.video_dir
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = FrameBackbone(num_classes=4).to(device)
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded fine-tuned checkpoint: {args.checkpoint}")
else:
    print("Using frozen ImageNet weights")
model.eval()

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

label_stems = {f.replace(".pt", "") for f in os.listdir("data/labels") if f.endswith(".pt")}
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4") and f.replace(".mp4", "") in label_stems]
print(f"Found {len(video_files)} videos with matching labels\n")

with torch.no_grad():
    for vid_idx, fname in enumerate(video_files, 1):
        path = os.path.join(VIDEO_DIR, fname)
        cap = cv2.VideoCapture(path)

        out_name = fname.replace(".mp4", ".pt")

        if os.path.exists(os.path.join(OUT_DIR, out_name)):
            print(f"[{vid_idx}/{len(video_files)}] {fname}  skipping (already extracted)")
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(round(fps)))
        expected = total_frames // step

        print(f"[{vid_idx}/{len(video_files)}] {fname}  ({expected} frames to extract)")

        feats = []
        i = 0
        extracted = 0
        success, frame = cap.read()

        while success:
            if i % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = tfm(frame_rgb).unsqueeze(0).to(device)
                _, feat = model(x, return_features=True)
                feats.append(feat.cpu())
                extracted += 1

                if extracted % 100 == 0:
                    print(f"  {extracted}/{expected} frames", flush=True)

            success, frame = cap.read()
            i += 1

        cap.release()

        feats = torch.cat(feats, dim=0)
        torch.save(feats, os.path.join(OUT_DIR, out_name))
        print(f"  Done -> {out_name}  shape={feats.shape}\n")

print("All videos processed.")
