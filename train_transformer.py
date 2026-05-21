import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset, collate_variable_length
from models.opera_transformer import NeuroOperA


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0

    for features, labels, padding_mask in loader:
        features = features.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        logits, attn = model(features, padding_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=class_weights,
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)