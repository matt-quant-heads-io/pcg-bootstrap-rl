from typing import Tuple
import torch
import torch.nn as nn
from utils.registry import register_model

class _CNNBackbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.out_dim = 64

    def forward(self, x):
        return self.net(x).flatten(1)

@register_model("CNNPolicy")
class CNNPolicy(nn.Module):
    def __init__(self, obs_shape: Tuple[int,int,int], n_actions: int):
        super().__init__()
        c,h,w = obs_shape
        self.backbone = _CNNBackbone(c)
        self.pi = nn.Linear(self.backbone.out_dim, n_actions)
        self.v  = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    def policy_value(self, x):
        return self.forward(x)
