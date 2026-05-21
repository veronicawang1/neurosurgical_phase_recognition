# data/dataset.py

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of dicts:
        {
          "features": path to .pt tensor [T, D],
          "labels": path to .pt tensor [T]
        }
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        features = torch.load(item["features"])
        labels = torch.load(item["labels"])
        return features, labels


def collate_variable_length(batch):
    features, labels = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in features])
    max_len = max(lengths).item()
    dim = features[0].shape[1]

    padded_features = torch.zeros(len(batch), max_len, dim)
    padded_labels = torch.full((len(batch), max_len), -100)

    padding_mask = torch.ones(len(batch), max_len).bool()

    for i, (x, y) in enumerate(zip(features, labels)):
        T = x.shape[0]
        padded_features[i, :T] = x
        padded_labels[i, :T] = y
        padding_mask[i, :T] = False

    return padded_features, padded_labels, padding_mask