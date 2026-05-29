import torch.nn as nn


class CnnLstm(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits
