from typing import Tuple
import torch
import torch.nn as nn
from utils.registry import register_model


class _CNNBackbone(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            # Optional: keep adaptive pooling to stabilize across envs
            nn.AdaptiveAvgPool2d((10, 10)),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


@register_model("CNNPolicy")
class CNNPolicy(nn.Module):
    def __init__(self, obs_shape: Tuple[int,int,int], n_actions: int):
        super().__init__()
        c, h, w = obs_shape
        self.backbone = _CNNBackbone(c)

        # LazyLinear figures out in_features the first time you call forward
        self.pi = nn.LazyLinear(n_actions)
        self.v  = nn.LazyLinear(1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.pi(z)
        value  = self.v(z).squeeze(-1)
        return logits, value

