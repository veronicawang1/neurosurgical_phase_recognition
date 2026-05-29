import os
import random


def build_samples(features_dir: str, labels_dir: str) -> list[dict]:
    """Return list of {features, labels} dicts for all videos that have both."""
    samples = []
    for fname in os.listdir(features_dir):
        if not fname.endswith(".pt"):
            continue
        label_path = os.path.join(labels_dir, fname)
        if not os.path.exists(label_path):
            print(f"  WARNING: no label file for {fname}, skipping")
            continue
        samples.append({
            "features": os.path.join(features_dir, fname),
            "labels": label_path,
        })
    return samples


def train_val_split(samples: list[dict], val_frac: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]
