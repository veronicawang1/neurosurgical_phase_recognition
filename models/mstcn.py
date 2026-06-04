"""
Multi-Scale Temporal Convolutional Network (MS-TCN).
Designed specifically for surgical phase recognition.
Reference: MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation (CVPR 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, num_filters: int, dilation: int, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Conv1d(
            num_filters, num_filters,
            kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.LayerNorm(num_filters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = F.relu(self.conv(x))
        out = self.dropout(out.transpose(1, 2)).transpose(1, 2)
        out = (residual + out).transpose(1, 2)
        out = self.norm(out).transpose(1, 2)
        return out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers: int, num_filters: int, input_dim: int,
                 num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, num_filters, kernel_size=1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_filters, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.classifier = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = x.transpose(1, 2)        # [B, input_dim, T]
        x = self.input_proj(x)       # [B, num_filters, T]
        for layer in self.layers:
            x = layer(x)
        logits = self.classifier(x)  # [B, num_classes, T]
        return logits.transpose(1, 2)  # [B, T, num_classes]


class MSTCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        num_stages: int = 2,
        num_layers: int = 8,
        num_filters: int = 64,
        num_classes: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        # first stage takes raw features
        self.stages.append(SingleStageModel(num_layers, num_filters, input_dim,
                                            num_classes, dropout))
        # subsequent stages refine predictions (take softmax of prev stage + features)
        for _ in range(num_stages - 1):
            self.stages.append(SingleStageModel(num_layers, num_filters,
                                                num_classes, num_classes, dropout))

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, input_dim]
        Returns: logits [B, T, num_classes], list of per-stage logits for multi-stage loss
        """
        all_logits = []
        out = self.stages[0](x)
        all_logits.append(out)

        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=-1))
            all_logits.append(out)

        # return final stage logits + all stage logits for loss computation
        return all_logits[-1], all_logits
