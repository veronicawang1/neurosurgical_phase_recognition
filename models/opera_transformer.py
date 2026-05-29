import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask, key_padding_mask=None):
        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


class NeuroOperA(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        d_model=256,
        num_classes=5,
        num_layers=2,
        n_heads=4,
        dropout=0.3,
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features, padding_mask=None):
        """
        features: [B, T, input_dim]
        padding_mask: [B, T], True where padded
        """

        B, T, _ = features.shape
        x = self.proj(features)

        causal_mask = torch.triu(
            torch.ones(T, T, device=features.device),
            diagonal=1
        ).bool()

        all_attn = []

        for layer in self.layers:
            x, attn = layer(
                x,
                causal_mask=causal_mask,
                key_padding_mask=padding_mask,
            )
            all_attn.append(attn)

        logits = self.classifier(x)
        return logits, all_attn