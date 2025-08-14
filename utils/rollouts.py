from typing import Dict, Any
import torch

@torch.no_grad()
def evaluate_agent(env, agent, episodes: int = 10) -> Dict[str, Any]:
    results = {"returns": [], "steps": [], "qualities": []}
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0
        while not done:
            state = agent.preprocess_observation(obs)
            action = agent.act_eval(state)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_ret += float(reward)
            ep_steps += 1
        results["returns"].append(ep_ret)
        results["steps"].append(ep_steps)
        if isinstance(info, dict) and "quality" in info:
            results["qualities"].append(float(info["quality"]))
    return results
