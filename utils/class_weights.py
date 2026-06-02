import torch


def compute_class_weights(samples, num_classes, device="cpu"):
    counts = torch.zeros(num_classes)
    for s in samples:
        labels = torch.load(s["labels"])
        valid = labels[labels != -100]
        for c in range(num_classes):
            counts[c] += (valid == c).sum()

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    weights = weights / weights.sum() * num_classes

    print("Class weights:")
    for i, w in enumerate(weights):
        print(f"  class {i}: count={int(counts[i])}  weight={w:.3f}")

    return weights.to(device)
