import os, argparse, shutil, json
import numpy as np
import torch
from utils.config import load_yaml, derive_run_dir
from utils.registry import MODEL_REGISTRY
from pcg_adapters.wrappers import ActionSpaceAdapter, SafeEnvWrapper

# Expect pcg_benchmark to be installed and discoverable
import pcg_benchmark
import models

from gymnasium import spaces
import numpy as np

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
    # Last resort
    raise ValueError("Could not infer number of actions from env.")

# Build env
def make_env(env_cfg):
    game = get_game_name(env_cfg)
    env = pcg_benchmark.make(game)
    # Wrap
    env = ActionSpaceAdapter(env)
    env = SafeEnvWrapper(env, max_steps_override=env_cfg.get("max_steps"))
    # Seed if provided
    if env_cfg.get("seed") is not None:
        try: env.seed(env_cfg["seed"])
        except Exception: pass
    # HARDEN: if action_space is still an int, convert to Discrete
    if not hasattr(env.action_space, "n"):
        n_act = infer_n_actions(env)
        env.action_space = spaces.Discrete(n_act)
    return env

def build_model(model_name: str, env, algo_cfg):
    import importlib
    # ensure models package is imported so registry populates
    importlib.import_module("models")

    obs, _ = env.reset()
    from rl_agents.rl_agent_base import RLAgentBase
    tmp = RLAgentBase(env, algo_cfg)
    state = tmp.preprocess_observation(obs)
    c, h, w = state.shape[1:]
    n_actions = env.action_space.n if hasattr(env.action_space, "n") else infer_n_actions(env)

    from utils.registry import MODEL_REGISTRY
    ModelCls = MODEL_REGISTRY.get(model_name)
    if ModelCls is None:
        raise ValueError(f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY.keys())}")

    return ModelCls((c, h, w), n_actions)


def build_agent(algorithm: str, env, model, algo_cfg, run_dir: str):
    # Lazy import to avoid circulars
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
    args = p.parse_args()

    env_cfg = load_yaml(args.env_config)
    algo_cfg = load_yaml(args.algo_config)

    if args.seed:
        env_cfg["seed"] = args.seed
    
    game_name = env_cfg["problem"]["name"]
    run_dir = derive_run_dir(game_name, args.algorithm)
    print(f"run_dir: {run_dir}")

    os.makedirs(os.path.join(run_dir, "configs"), exist_ok=True)
    shutil.copy(args.env_config, os.path.join(run_dir, "configs", os.path.basename(args.env_config)))
    shutil.copy(args.algo_config, os.path.join(run_dir, "configs", os.path.basename(args.algo_config)))

    env = pcg_benchmark.make(game_name)
    model = build_model(args.model, env, algo_cfg)
    # agent = build_agent(args.algorithm, env, model, algo_cfg, "/home/ubuntu/pcg_bootstrap_rl/"+run_dir)
    agent = build_agent(args.algorithm, env, model, algo_cfg, run_dir)
    

    agent.train()
    eval_report = agent.evaluate(algo_cfg["eval_episodes"])
    with open(os.path.join(run_dir, "metrics", "eval.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

if __name__ == "__main__":
    main()
