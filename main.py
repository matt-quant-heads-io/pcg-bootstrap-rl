# main.py
import os, argparse, shutil, json
import numpy as np
import torch

from utils.config import load_yaml, derive_run_dir
from utils.registry import MODEL_REGISTRY
from pcg_adapters.wrappers import ActionSpaceAdapter, SafeEnvWrapper  # keep if you use them

# Expect pcg_benchmark to be installed and discoverable
import pcg_benchmark
import models

from gymnasium import spaces
import numpy as np

from pcg_adapters.vector_env import PCGSyncVectorEnv, BatchDimCompatEnv

from gymnasium import spaces
import numpy as np


def get_n_actions(env) -> int:
    """
    Robustly determine the per-env action count, even when env is vectorized.
    Preference order:
      1) env.single_action_space (from PCGVectorRunner / compat wrapper)
      2) env.action_space (Discrete)
      3) MultiDiscrete (pick the per-dimension size; assume uniform or take max)
      4) integer Box (derive from high)
      5) prob_config['tiles']
      6) _problem._content_space.range()['max']
    """
    # 1) Prefer single_action_space if present (vector wrappers expose this)
    space = getattr(env, "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    if space is None:
        raise ValueError("Env has no action space")

    # Discrete
    if isinstance(space, spaces.Discrete):
        return int(space.n)

    # MultiDiscrete (common when a Discrete was batched)
    if isinstance(space, spaces.MultiDiscrete):
        nvec = np.asarray(space.nvec).astype(int)
        # If it's from batch_space(Discrete(n), N), nvec is [n, n, ..., n]
        # Use the first (or max as a safe fallback)
        return int(nvec[0]) if nvec.size > 0 else int(nvec.max())

    # Integer Box: try to infer (max + 1)
    if isinstance(space, spaces.Box) and np.issubdtype(space.dtype, np.integer):
        # Box.high can be scalar or array
        high = np.asarray(space.high)
        if np.all(np.isfinite(high)):
            return int(high.max()) + 1

    # Prob config tiles
    if hasattr(env, "prob_config") and isinstance(getattr(env, "prob_config"), dict):
        tiles = env.prob_config.get("tiles")
        if isinstance(tiles, (list, tuple)) and len(tiles) > 0:
            return int(len(tiles))

    # Content space range fallback
    try:
        rng = env._problem._content_space.range()
        if isinstance(rng, dict) and isinstance(rng.get("max"), (int, np.integer)):
            return int(rng["max"])
    except Exception:
        pass

    raise ValueError("Could not infer number of actions from env.")



def get_game_name(env_cfg: dict) -> str:
    def _extract(obj):
        if isinstance(obj, str): return obj
        if isinstance(obj, dict):
            for k in ("name","id","env","problem","game"):
                v = obj.get(k)
                if isinstance(v, str): return v
        return None
    for top in ("game","problem"):
        if top in env_cfg:
            name = _extract(env_cfg[top])
            if name: return name
    raise ValueError(f"Could not determine game name from env_config keys={list(env_cfg.keys())}")


def infer_n_actions(env) -> int:
    # First choice: tiles in prob_config
    if hasattr(env, "prob_config") and isinstance(getattr(env, "prob_config"), dict):
        tiles = env.prob_config.get("tiles")
        if isinstance(tiles, (list, tuple)): return len(tiles)
    # Try content_space range
    try:
        rng = env._problem._content_space.range()
        if isinstance(rng, dict) and isinstance(rng.get("max"), int):
            return int(rng["max"])
    except Exception:
        pass
    # Fallback: observation_space.high + 1
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "high"):
        return int(np.max(env.observation_space.high)) + 1
    raise ValueError("Could not infer number of actions from env.")


def make_single_env(game_name: str):
    """
    If pcg_benchmark.make returns your GymPCGEnv already, this is enough.
    Otherwise, import your class and construct it explicitly.
    """
    return pcg_benchmark.make(game_name)


def make_env_any(env_cfg: dict):
    """
    Build either a single env wrapped to be vector-compatible or
    an actual VectorEnv when num_envs > 1.
    """
    game = get_game_name(env_cfg)
    num_envs = int(env_cfg.get("num_envs", 1))
    seed = env_cfg.get("seed")

    def _factory():
        env = make_single_env(game)
        # Optional: preserve your existing wrappers
        env = ActionSpaceAdapter(env)
        env = SafeEnvWrapper(env, max_steps_override=env_cfg.get("max_steps"))
        # Harden: ensure Discrete action space
        if not hasattr(env.action_space, "n"):
            n_act = infer_n_actions(env)
            env.action_space = spaces.Discrete(n_act)
        if seed is not None:
            try: env.seed(seed)
            except Exception: pass
        return env

    venv = PCGSyncVectorEnv(_factory, num_envs=num_envs, seed=seed)
    if num_envs == 1:
        return BatchDimCompatEnv(venv)  # keeps your old shape assumptions
    return venv


def build_model(model_name: str, env, algo_cfg):
    import importlib
    importlib.import_module("models")

    obs, _ = env.reset()

    from rl_agents.rl_agent_base import RLAgentBase
    tmp = RLAgentBase(env, algo_cfg)
    state = tmp.preprocess_observation(obs)
    if state.ndim == 3:  # (C,H,W) -> (1,C,H,W)
        state = state.unsqueeze(0)
    c, h, w = state.shape[1:]

    from utils.registry import MODEL_REGISTRY
    ModelCls = MODEL_REGISTRY.get(model_name)
    if ModelCls is None:
        raise ValueError(f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY.keys())}")

    n_actions = get_n_actions(env)  # <-- robust per-env action count
    return ModelCls((c, h, w), n_actions).to("cuda")



def build_agent(algorithm: str, env, model, algo_cfg, run_dir: str):
    if algorithm.upper() == "PPO":
        from rl_agents.ppo import PPOAgent as Agent
    elif algorithm.upper() == "DQN":
        from rl_agents.dqn import DQNAgent as Agent
    elif algorithm.upper() == "A2C":
        from rl_agents.a2c import A2CAgent as Agent
    elif algorithm.upper() == "REINFORCE":
        from rl_agents.reinforce import REINFORCEAgent as Agent
    elif algorithm.upper() == "SAC":
        from rl_agents.sac_discrete import SACDiscreteAgent as Agent
    elif algorithm.upper() == "TD3":
        from rl_agents.td3_discrete import TD3DiscreteAgent as Agent
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")
    return Agent(env, model, algo_cfg, run_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--env_config", required=True)
    p.add_argument("--algo_config", required=True)
    p.add_argument("--seed", default=None)
    p.add_argument("--num_envs", default=8)
    args = p.parse_args()

    env_cfg = load_yaml(args.env_config)
    algo_cfg = load_yaml(args.algo_config)

    if args.seed:
        env_cfg["seed"] = int(args.seed)

    env_cfg["num_envs"] = args.num_envs

    game_name = get_game_name(env_cfg)
    run_dir = derive_run_dir(game_name, args.algorithm)
    print(f"run_dir: {run_dir}")

    os.makedirs(os.path.join(run_dir, "configs"), exist_ok=True)
    shutil.copy(args.env_config, os.path.join(run_dir, "configs", os.path.basename(args.env_config)))
    shutil.copy(args.algo_config, os.path.join(run_dir, "configs", os.path.basename(args.algo_config)))

    # ---- Use vectorized env (num_envs in env_config; default 1) ----
    env = make_env_any(env_cfg)

    model = build_model(args.model, env, algo_cfg)
    agent = build_agent(args.algorithm, env, model.to("cuda"), algo_cfg, run_dir)

    agent.train()
    eval_report = agent.evaluate(algo_cfg.get("eval_episodes", 5))
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics", "eval.json"), "w") as f:
        json.dump(eval_report, f, indent=2)


if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"total time: {int((end - start).total_seconds()/60.0)} mins")
