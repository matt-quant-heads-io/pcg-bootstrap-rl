# pcg_adapters/vector_env.py
from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


def _stack_obs_dict(obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Stack [{"map": (1,H,W), "pos": (2,)}]*N into:
        {"map": (N,1,H,W), "pos": (N,2)}
    """
    maps = np.stack([o["map"] for o in obs_list], axis=0)
    poss = np.stack([o["pos"] for o in obs_list], axis=0)
    return {"map": maps, "pos": poss}


class PCGSyncVectorEnv(VectorEnv):
    """
    Synchronous VectorEnv wrapper for any single-env factory.

    Usage:
        def make_single():
            return GymPCGEnv(name, problem)
        venv = PCGSyncVectorEnv(make_single, num_envs=8, seed=123)
    """

    metadata = {"render_modes": []}

    def __init__(self, make_env_fn: Callable[[], gym.Env], num_envs: int, seed: Optional[int] = None):
        assert num_envs >= 1
        self.envs: List[gym.Env] = []
        for i in range(num_envs):
            env = make_env_fn()
            if seed is not None:
                try:
                    env.seed(int(seed) + i)
                except Exception:
                    pass
            self.envs.append(env)

        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        self.observation_space = batch_space(self.single_observation_space, n=num_envs)
        self.action_space = batch_space(self.single_action_space, n=num_envs)

    # -------------- VectorEnv API --------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_list, infos_list = [], []
        for i, env in enumerate(self.envs):
            if seed is not None:
                try:
                    env.seed(int(seed) + i)
                except Exception:
                    pass
            obs, info = env.reset(seed=None, options=options)
            obs_list.append(obs)
            infos_list.append(info)
        batched_obs = _stack_obs_dict(obs_list)
        return batched_obs, infos_list

    def step(self, actions: np.ndarray):
        # actions: (N,) for Discrete or (N,...) for Box
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            a_i = actions[i]
            obs, rew, term, trunc, info = env.step(a_i)
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        batched_obs = _stack_obs_dict(obs_list)
        rewards = np.asarray(rew_list, dtype=np.float32)
        terminated = np.asarray(term_list, dtype=bool)
        truncated = np.asarray(trunc_list, dtype=bool)
        return batched_obs, rewards, terminated, truncated, info_list

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass

    def render(self):
        # Optional: render first env's content; adapt as needed
        if self.num_envs > 0 and hasattr(self.envs[0], "render"):
            return self.envs[0].render(self.envs[0]._current_content)
        return None


class BatchDimCompatEnv(gym.Env):
    """
    Wrap a VectorEnv(num_envs==1) with a single-env API:
      - reset/step return unbatched observations:
            {"map": (1,H,W), "pos": (2,)}
      - reward: float, terminated/truncated: bool, info: dict
    """

    metadata = {"render_modes": []}

    def __init__(self, venv: PCGSyncVectorEnv):
        assert venv.num_envs == 1, "BatchDimCompatEnv only supports num_envs==1"
        self.venv = venv
        self.observation_space = venv.single_observation_space
        self.action_space = venv.single_action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self.venv.reset(seed=seed, options=options)
        return {"map": obs["map"][0], "pos": obs["pos"][0]}, infos[0]

    def step(self, action):
        batched_action = np.asarray([action])
        obs, rew, term, trunc, infos = self.venv.step(batched_action)
        return (
            {"map": obs["map"][0], "pos": obs["pos"][0]},
            float(rew[0]),
            bool(term[0]),
            bool(trunc[0]),
            infos[0],
        )

    def close(self):
        self.venv.close()

    def render(self):
        return self.venv.render()
