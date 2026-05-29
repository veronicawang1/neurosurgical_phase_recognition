"""
Convert per-video label CSVs (exported from Google Sheets) into .pt label tensors.

Expected CSV format (one file per video, filename = video name):
    frame index, label
    0, Brain Exposure
    1, Brain Exposure
    ...

Usage:
    python prepare_labels.py --csv_dir data/label_csvs --out_dir data/labels

Each CSV named <video>.csv produces data/labels/<video>.pt — a LongTensor of
length T where each value is the integer class index for that frame.
Frames missing from the CSV get label -100 (ignored in cross-entropy loss).
"""

import argparse
import os
import pandas as pd
import torch

# Define your class mapping here — order determines the integer index.
CLASS_NAMES = [
    "Brain Exposure",
    "Parent Vessel Identification",
    "Neck Identification",
    "Dome Identification",
    "Clipping",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def csv_to_label_tensor(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    max_frame = int(df["frame index"].max())
    labels = torch.full((max_frame + 1,), -100, dtype=torch.long)

    for _, row in df.iterrows():
        frame_idx = int(row["frame index"])
        label_str = str(row["label"]).strip()
        if label_str not in CLASS_TO_IDX:
            continue  # ignore frames with out-of-scope labels (e.g. "Liquid B&W")
        labels[frame_idx] = CLASS_TO_IDX[label_str]

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="data/label_csvs")
    parser.add_argument("--out_dir", default="data/labels")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(args.csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {args.csv_dir}")
        return

    for fname in csv_files:
        csv_path = os.path.join(args.csv_dir, fname)
        out_name = fname.replace(".csv", ".pt")
        out_path = os.path.join(args.out_dir, out_name)

        labels = csv_to_label_tensor(csv_path)
        torch.save(labels, out_path)
        print(f"  {fname} -> {out_path}  (T={labels.shape[0]})")

    print(f"\nDone. {len(csv_files)} label files written to {args.out_dir}/")
    print("Class mapping:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
