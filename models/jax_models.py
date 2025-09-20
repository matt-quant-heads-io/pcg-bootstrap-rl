# jax_models.py
from __future__ import annotations
from typing import Tuple
import jax.numpy as jnp
from flax import linen as nn


class CNNBackbone(nn.Module):
    """Simple CNN; final adaptive avg pool -> (10,10); flatten."""
    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, C, H, W]
        def conv(ch): 
            return nn.Conv(ch, kernel_size=(3, 3), padding='SAME')
        x = conv(32)(x); x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

        x = conv(64)(x); x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

        x = conv(64)(x); x = nn.relu(x)

        # Adaptive pooling to (10,10) regardless of H,W
        _, _, H, W = x.shape
        tgt_h, tgt_w = 10, 10
        stride_h = max(1, H // tgt_h)
        stride_w = max(1, W // tgt_w)
        kernel_h = max(1, H - (tgt_h - 1) * stride_h)
        kernel_w = max(1, W - (tgt_w - 1) * stride_w)
        x = nn.avg_pool(x,
                        window_shape=(1, kernel_h, kernel_w, 1),
                        strides=(1, stride_h, stride_w, 1),
                        padding='VALID')
        return x.reshape((x.shape[0], -1))  # [B, C*10*10]


class CNNPolicy(nn.Module):
    """Shared backbone + policy/value heads (JAX version of your Torch CNNPolicy)."""
    obs_shape: Tuple[int, int, int]  # (C,H,W)
    n_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # x: [B, C, H, W]
        c, _, _ = self.obs_shape
        z = CNNBackbone(in_channels=c)(x)
        logits = nn.Dense(self.n_actions)(z)
        value  = nn.Dense(1)(z)
        return logits, jnp.squeeze(value, axis=-1)  # [B,A], [B]
