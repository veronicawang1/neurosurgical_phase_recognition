import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class FrameBackbone(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x).flatten(1)
        logits = self.classifier(feat)

        if return_features:
            return logits, feat

        return logits