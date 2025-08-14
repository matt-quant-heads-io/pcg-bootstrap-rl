import os, argparse, shutil, json
import numpy as np
import torch
from utils.config import load_yaml, derive_run_dir
from utils.registry import MODEL_REGISTRY
from pcg_adapters.wrappers import ActionSpaceAdapter, SafeEnvWrapper

# Expect pcg_benchmark to be installed and discoverable
import pcg_benchmark

# Build env
def make_env(env_cfg):
    game = env_cfg.get("game")
    assert game, "env_config must contain 'game' (e.g., 'zelda-v0')"
    env = pcg_benchmark.make(game)
    env = ActionSpaceAdapter(env)
    env = SafeEnvWrapper(env, max_steps_override=env_cfg.get("max_steps"))
    if env_cfg.get("seed") is not None:
        try:
            env.seed(env_cfg["seed"])
        except Exception:
            pass
    return env

def build_model(model_name: str, env, algo_cfg):
    obs, _ = env.reset()
    from rl_agents.rl_agent_base import RLAgentBase
    tmp = RLAgentBase(env, algo_cfg)
    state = tmp.preprocess_observation(obs)
    c,h,w = state.shape[1:]
    n_actions = env.action_space.n
    ModelCls = MODEL_REGISTRY.get(model_name)
    assert ModelCls is not None, f"Unknown model '{model_name}'. Known: {list(MODEL_REGISTRY.keys())}"
    model = ModelCls((c,h,w), n_actions)
    return model

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
    return Agent(env=env, model=model, algo_config=algo_cfg, run_dir=run_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--env_config", required=True)
    p.add_argument("--algo_config", required=True)
    args = p.parse_args()

    env_cfg = load_yaml(args.env_config)
    algo_cfg = load_yaml(args.algo_config)

    game_name = env_cfg.get("game", "game")
    run_dir = derive_run_dir(game_name, args.algorithm)

    os.makedirs(os.path.join(run_dir, "configs"), exist_ok=True)
    shutil.copy(args.env_config, os.path.join(run_dir, "configs", os.path.basename(args.env_config)))
    shutil.copy(args.algo_config, os.path.join(run_dir, "configs", os.path.basename(args.algo_config)))

    env = make_env(env_cfg)
    model = build_model(args.model, env, algo_cfg)
    agent = build_agent(args.algorithm, env, model, algo_cfg, run_dir)

    agent.train()
    eval_report = agent.evaluate()
    with open(os.path.join(run_dir, "metrics", "eval.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

if __name__ == "__main__":
    main()
