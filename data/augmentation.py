import torch
from torch.utils.data import Dataset


class AugmentedFeatureDataset(Dataset):
    def __init__(self, samples, noise_std=0.01, min_crop_frac=0.7):
        self.samples = samples
        self.noise_std = noise_std
        self.min_crop_frac = min_crop_frac

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        features = torch.load(item["features"])
        labels = torch.load(item["labels"])
        T = min(features.shape[0], labels.shape[0])
        features, labels = features[:T], labels[:T]

        # random temporal crop
        crop_len = int(T * (self.min_crop_frac + torch.rand(1).item() * (1 - self.min_crop_frac)))
        crop_len = max(1, crop_len)
        start = torch.randint(0, T - crop_len + 1, (1,)).item()
        features = features[start:start + crop_len]
        labels = labels[start:start + crop_len]

        # gaussian noise
        features = features + torch.randn_like(features) * self.noise_std

        return features, labels
