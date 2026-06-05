# Neurosurgery Phase Recognition

Surgical phase recognition for aneurysm clipping procedures. Classifies video frames into 4 phases using a CNN feature extractor + temporal model pipeline.

## Phases

| ID | Phase |
|----|-------|
| 0 | Brain Exposure |
| 1 | Parent Vessel Identification |
| 2 | Dome and Neck Identification |
| 3 | Clipping |

## Pipeline

```
Raw video → extract frames at 1 FPS → CNN features (ResNet50) → temporal model → phase labels
```

1. **Label preparation** — convert per-video CSV annotations to frame-level `.pt` label tensors
2. **Backbone fine-tuning** — fine-tune ResNet50 on labeled frames before feature extraction
3. **Feature extraction** — extract 2048-dim CNN features at 1 FPS, saved as `.pt` files
4. **Temporal model training** — train MS-TCN, Transformer, or LSTM on the feature sequences
5. **Post-processing** — Viterbi decoding or temporal smoothing to enforce valid phase ordering

## Setup

```bash
pip install -r requirements.txt
```

Trained on a GCP VM (n1-standard-8 + Tesla T4). If CUDA fails with a cublas error, run:

```bash
echo "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib" | sudo tee /etc/ld.so.conf.d/nvidia-cublas.conf && sudo ldconfig
```

## Usage

All training scripts must be run from the repo root with `python3 -m`.

### 1. Prepare labels

```bash
python3 prepare_labels.py
```

Reads CSVs from `data/label_csvs/`, writes frame-level label tensors to `data/labels/`.

### 2. Fine-tune backbone

```bash
python3 -m training.finetune_backbone --epochs 10 --run_name finetune_v1
```

Pre-saves frames as JPEGs to `data/frames/`, then fine-tunes ResNet50. Saves checkpoint to `checkpoints/finetuned_backbone.pt`.

### 3. Extract features

```bash
# With fine-tuned backbone (recommended)
python3 extract_features.py \
    --checkpoint checkpoints/finetuned_backbone.pt \
    --out_dir data/features_finetuned

# With frozen ImageNet weights (baseline only)
python3 extract_features.py --out_dir data/features
```

### 4. Train temporal models

```bash
# MS-TCN (best overall — use learned transitions with fine-tuned features)
python3 -m training.train_mstcn \
    --features_dir data/features_finetuned \
    --run_name mstcn_v1 \
    --use_learned_transitions

# Causal Transformer (video-level val split)
python3 -m training.train_transformer_framelevel \
    --features_dir data/features_finetuned \
    --val_split video \
    --run_name transformer_v1

# LSTM baseline
python3 -m training.train_lstm \
    --features_dir data/features_finetuned \
    --run_name lstm_v1
```

## Models

| File | Description |
|------|-------------|
| `models/cnn_backbone.py` | ResNet50 feature extractor (2048-dim output) |
| `models/mstcn.py` | MS-TCN: multi-stage dilated temporal convolutions |
| `models/opera_transformer.py` | NeuroOperA: causal transformer with multi-head attention |
| `models/cnn_lstm.py` | CNN-LSTM baseline |

## Post-processing

Both methods are applied automatically during validation and logged separately.

- **Viterbi decoding** — enforces valid phase ordering via a transition matrix. Use `--use_learned_transitions` to estimate the matrix from training data instead of using the default surgical ordering prior.
- **Temporal smoothing** — median filter over softmax probabilities (window size controlled by `--smooth_window`).

## Metrics

| Metric | Description |
|--------|-------------|
| Val accuracy | Frame-level accuracy on held-out videos |
| Edit distance | Segment-level ordering correctness (1.0 = perfect) |
| Seg F1@10/25/50 | Segmental F1 at 10%, 25%, 50% overlap thresholds |
| Boundary-aware acc | Frame accuracy ignoring frames within 5s of phase transitions |

## Data layout

```
data/
  label_csvs/       # raw annotation CSVs (one per video)
  labels/           # prepared frame-level label tensors (.pt)
  raw_videos/       # source .mp4 files at 1 FPS
  frames/           # pre-saved JPEG frames for backbone fine-tuning
  features/         # CNN features from frozen ImageNet backbone
  features_finetuned/  # CNN features from fine-tuned backbone (use this)
```

## Results (best runs on fine-tuned features)

| Model | Val Acc | Viterbi F1@10 | Viterbi F1@50 |
|-------|---------|---------------|---------------|
| MS-TCN + learned transitions | 0.957 | 0.889 | 0.714 |
| Transformer + Viterbi | 0.936 | 0.944 | 0.789 |
