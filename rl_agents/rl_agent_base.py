from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gymnasium import spaces

class RLAgentBase:
    def __init__(self, env, algo_config: Dict[str, Any], map_pad_fill: Optional[int]=0, crop_size: Optional[int]=None):
        self.env = env
        self.algo_config = algo_config
        self.map_pad_fill = map_pad_fill
        self._obs_space = getattr(env, "observation_space", None)
        self._act_space = getattr(env, "action_space", None)
        if hasattr(env, "prob_config") and "tiles" in env.prob_config:
            self.num_classes = int(len(env.prob_config["tiles"]))
        elif isinstance(self._obs_space, spaces.Box):
            self.num_classes = int(np.max(self._obs_space.high)) + 1
        else:
            self.num_classes = int(algo_config.get("num_tile_classes", 6))
        self.crop_size = crop_size

    @torch.no_grad()
    def preprocess_observation(self, obs):
        assert isinstance(obs, dict) and "map" in obs and "pos" in obs, "obs must have 'map' and 'pos'"
        y, x = obs["pos"]
        map_np = np.asarray(obs["map"], dtype=np.float32)  # (1,H,W)
        assert map_np.ndim == 3 and map_np.shape[0] == 1, f"Expected (1,H,W), got {map_np.shape}"
        _, H, W = map_np.shape
        if self.crop_size is None:
            self.crop_size = int(W)
        half = self.crop_size // 2
        padded = np.pad(map_np[0], ((half,half),(half,half)), mode="constant", constant_values=self.map_pad_fill)
        top, left = int(y), int(x)
        cropped = padded[top: top+self.crop_size, left: left+self.crop_size]
        if cropped.shape != (self.crop_size, self.crop_size):
            cropped = np.pad(cropped,
                             ((0, max(0, self.crop_size - cropped.shape[0])),
                              (0, max(0, self.crop_size - cropped.shape[1]))),
                             mode="constant", constant_values=self.map_pad_fill)
            cropped = cropped[:self.crop_size, :self.crop_size]
        cropped_long = torch.as_tensor(cropped, dtype=torch.long)
        cropped_long.clamp_(0, self.num_classes - 1)
        state = F.one_hot(cropped_long, num_classes=self.num_classes).float()
        state = rearrange(state, "h w c -> 1 c h w")
        return state

    @torch.no_grad()
    def act_eval(self, proc_state: torch.Tensor) -> int:
        raise NotImplementedError
