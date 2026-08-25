
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU}


class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 hidden_sizes: Sequence[int] = (256, 128, 64),
                 dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation}")
        act = _ACT[activation]
        layers: List[nn.Module] = []
        prev = int(in_features)
        for h in hidden_sizes:


            layers += [nn.Linear(prev, int(h)), nn.LayerNorm(int(h)), act()]
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg, in_features: int, num_classes: int) -> nn.Module:
    m = cfg.model
    return MLP(in_features, num_classes,
               hidden_sizes=list(m.hidden_sizes),
               dropout=float(m.dropout),
               activation=str(m.activation))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
