from typing import Tuple
import torch
import torch.nn as nn
from utils.registry import register_model

@register_model("MLPPolicy")
class MLPPolicy(nn.Module):
    def __init__(self, obs_shape: Tuple[int,int,int], n_actions: int, hidden: int = 256):
        super().__init__()
        c,h,w = obs_shape
        in_dim = c*h*w
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    # unify with agents that call policy_value
    def policy_value(self, x):
        return self.forward(x)
