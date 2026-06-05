"""
Grad-CAM saliency maps on surgical video frames.
Run on VM from repo root:
    python3 analysis/saliency_map.py --video data/raw_videos/<video>.mp4 --checkpoint checkpoints/finetuned_backbone.pt
Outputs: analysis/figures/saliency_<phase>.png for each phase present in the video.
"""
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.cnn_backbone import FrameBackbone

os.makedirs("analysis/figures", exist_ok=True)

CLASS_NAMES  = ["Brain Exposure", "Parent Vessel ID", "Dome & Neck ID", "Clipping"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

parser = argparse.ArgumentParser()
parser.add_argument("--video",      required=True)
parser.add_argument("--checkpoint", default="checkpoints/finetuned_backbone.pt")
parser.add_argument("--num_frames", type=int, default=4,
                    help="Number of example frames to visualize per class")
args = parser.parse_args()

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

model = FrameBackbone(num_classes=4).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

# Hook into the last conv layer of ResNet50 (layer4)
activations, gradients = {}, {}

def forward_hook(module, input, output):
    activations["value"] = output.detach()

def backward_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0].detach()

target_layer = model.backbone.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(round(fps)))

frames_by_class = {c: [] for c in range(4)}

i = 0
success, frame = cap.read()
while success:
    if i % step == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tfm(frame_rgb).unsqueeze(0).to(device)
        x.requires_grad_(True)

        logits, _ = model(x, return_features=True)
        pred_class = logits.argmax(1).item()

        model.zero_grad()
        logits[0, pred_class].backward()

        # Grad-CAM
        grads   = gradients["value"]          # [1, C, H, W]
        acts    = activations["value"]        # [1, C, H, W]
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if len(frames_by_class[pred_class]) < args.num_frames:
            frames_by_class[pred_class].append((frame_rgb, cam, pred_class))

    if all(len(v) >= args.num_frames for v in frames_by_class.values()):
        break
    success, frame = cap.read()
    i += 1
cap.release()

# Plot: one row per class, args.num_frames columns (original | heatmap overlay)
for cls_idx, examples in frames_by_class.items():
    if not examples:
        continue
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = [axes]
    for row, (frame_rgb, cam, pred) in enumerate(examples):
        # original
        axes[row][0].imshow(frame_rgb)
        axes[row][0].axis("off")
        axes[row][0].set_title(f"Frame — predicted: {CLASS_NAMES[pred]}", fontsize=9)
        # overlay
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        overlay = 0.5 * frame_rgb / 255.0 + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        axes[row][1].imshow(overlay)
        axes[row][1].axis("off")
        axes[row][1].set_title("Grad-CAM", fontsize=9)

    plt.suptitle(f"Grad-CAM — {CLASS_NAMES[cls_idx]}", fontsize=12, color=CLASS_COLORS[cls_idx])
    plt.tight_layout()
    out = f"analysis/figures/saliency_{cls_idx}_{CLASS_NAMES[cls_idx].replace(' ', '_')}.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")
