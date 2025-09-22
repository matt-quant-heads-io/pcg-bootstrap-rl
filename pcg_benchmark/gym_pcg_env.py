# pcg_benchmark/gym_pcg_env.py
import copy
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from .pcg_env import PCGEnv, _recursiveDiversity


class GymPCGEnv(PCGEnv, gym.Env):
    """
    Single-environment PCG Gymnasium env with a Dict observation:
        {"map": (1,H,W) int64 tiles, "pos": (2,) int64 (y,x)}
    Action space is Discrete(n_tiles): action writes a tile id at current pos.
    """
    metadata = {"render_modes": []}

    def __init__(self, name, problem):
        super().__init__(name, problem)
        cfg_path = str(Path(__file__).parent.parent / "configs" / "envs" / f"{name}.yaml")
        self.prob_config = yaml.safe_load(open(cfg_path, "r"))
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        self._current_content = None
        self._positions_queue = []
        self._pos = (0, 0)
        self._step_count = 0
        self._max_steps = 1000
        self._done = False

    def _create_action_space(self):
        n_tiles = max(1, int(len(self.prob_config.get("tiles", []))))
        return spaces.Discrete(n_tiles)

    def _create_observation_space(self):
        self._height = int(self._problem._height)
        self._width = int(self._problem._width)

        map_space = spaces.Box(
            low=0,
            high=max(0, len(self.prob_config.get("tiles", [])) - 1),
            shape=(1, self._height, self._width),
            dtype=np.int64,
        )
        pos_space = spaces.Box(
            low=0,
            high=max(self._height, self._width) - 1,
            shape=(2,),
            dtype=np.int64,
        )
        return spaces.Dict({"map": map_space, "pos": pos_space})

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(int(seed))

        self._current_content = self._get_initial_content()
        self._positions_queue = [(y, x) for y in range(self._height) for x in range(self._width)]
        self._step_count = 0
        self._done = False

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self._done:
            obs, info = self.reset()
            return obs, 0.0, True, False, info

        prev_content = self._current_content.copy()
        self._current_content = self._apply_action(action)
        self._step_count += 1

        obs = self._get_observation()
        reward = self._calculate_reward(self._current_content, prev_content)
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_steps
        self._done = terminated or truncated
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    def _get_initial_content(self):
        n_tiles = max(1, int(len(self.prob_config.get("tiles", []))))
        return np.random.randint(0, n_tiles, size=(self._height, self._width), dtype=np.int64)

    def _apply_action(self, action):
        updated = self._current_content.copy()
        y, x = self._pos
        updated[y, x] = int(action)
        return updated

    def _get_observation(self):
        if self._positions_queue:
            self._pos = self._positions_queue.pop()
        obs_map = self._current_content.astype(np.int64).reshape(1, self._height, self._width)
        return {"map": obs_map, "pos": np.asarray(self._pos, dtype=np.int64)}

    def _calculate_reward(self, new_content, old_content):
        new_info = self._problem.info(new_content)
        old_info = self._problem.info(old_content)
        new_quality_score = self._problem.quality(new_info)
        old_quality_score = self._problem.quality(old_info)
        return float(new_quality_score - old_quality_score) * 100.0

    def _is_terminated(self):
        info = self._problem.info(self._current_content.reshape(self._height, self._width))
        quality_score = self._problem.quality(info)
        return bool(quality_score >= 1.0 or len(self._positions_queue) == 0)

    def _get_info(self):
        if self._current_content is None:
            return {}
        info_dict = self._problem.info(self._current_content)
        quality_score = self._problem.quality(info_dict)
        return {
            "quality": float(quality_score),
            "step_count": int(self._step_count),
            "content_info": info_dict,
        }

    @property
    def content_space(self):
        return self._problem._content_space

    @property
    def control_space(self):
        return self._problem._control_space

    def seed(self, seed):
        self._problem._random = np.random.default_rng(seed)
        if self.content_space is None:
            raise AttributeError("self._content_space is not initialized")
        self.content_space.seed(seed)
        if self.control_space is None:
            raise AttributeError("self._control_space is not initialized")
        self.control_space.seed(seed)
