# extract_features.py

import os
import cv2
import torch
from torchvision import transforms
from models.cnn_backbone import FrameBackbone

VIDEO_DIR = "data/raw_videos"
OUT_DIR = "data/features"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FrameBackbone(num_classes=5).to(device)
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

with torch.no_grad():
    for fname in os.listdir(VIDEO_DIR):
        if not fname.endswith(".mp4"):
            continue

        path = os.path.join(VIDEO_DIR, fname)
        cap = cv2.VideoCapture(path)

        feats = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(round(fps))  # extract 1 fps

        i = 0
        success, frame = cap.read()

        while success:
            if i % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = tfm(frame).unsqueeze(0).to(device)
                _, feat = model(x, return_features=True)
                feats.append(feat.cpu())

            success, frame = cap.read()
            i += 1

        cap.release()

        feats = torch.cat(feats, dim=0)
        out_name = fname.replace(".mp4", ".pt")
        torch.save(feats, os.path.join(OUT_DIR, out_name))