import torch
import torch.nn as nn

class ActorCategorical(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c,h,w = obs_shape
        self.feat = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.logits = nn.Linear(64, n_actions)

    def forward(self, x):
        z = self.feat(x).flatten(1)
        return self.logits(z)
