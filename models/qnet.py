import torch
import torch.nn as nn

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

class QNet(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c,h,w = obs_shape
        self.feat = _CNNBackbone(c)
        self.head = nn.LazyLinear(n_actions)

    def forward(self, x):
        z = self.feat(x).flatten(1)
        return self.head(z)
