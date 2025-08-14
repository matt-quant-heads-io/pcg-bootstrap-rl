from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ActionSpaceAdapter(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if hasattr(env, "prob_config") and "tiles" in env.prob_config:
            n_tiles = len(env.prob_config["tiles"])
        else:
            high = getattr(getattr(env, "observation_space", None), "high", None)
            n_tiles = int(np.max(high)) + 1 if high is not None else 6
        self.action_space = spaces.Discrete(n_tiles)

    def step(self, action):
        return self.env.step(int(action))

class SafeEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps_override: int | None = None):
        super().__init__(env)
        self._max_steps_override = max_steps_override

    def reset(self, **kwargs):
        if self._max_steps_override is not None and hasattr(self.env, "_max_steps"):
            self.env._max_steps = int(self._max_steps_override)
        return self.env.reset(**kwargs)

    def step(self, action):
        try:
            return self.env.step(action)
        except Exception as e:
            raise RuntimeError(f"Env step failed: {e}")
