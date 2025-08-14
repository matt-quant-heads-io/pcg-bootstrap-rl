from typing import Dict, Any, Optional
import os
from PIL import Image
import torch

@torch.no_grad()
def evaluate_agent(env, agent, episodes: int = 10, render_out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compatibility eval helper. Prefer RLAgentBase.evaluate()."""
    results = {"returns": [], "steps": [], "qualities": []}
    if render_out_dir is not None:
        os.makedirs(render_out_dir, exist_ok=True)

    for ep in range(episodes):
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

        if render_out_dir is not None:
            try:
                final_content = obs["map"][0]
                render_result = env.render(final_content)
                out_path = os.path.join(render_out_dir, f"episode_{ep+1:03d}.png")
                if isinstance(render_result, Image.Image):
                    render_result.save(out_path)
                else:
                    try:
                        Image.fromarray(render_result).save(out_path)
                    except Exception:
                        pass
                results.setdefault("render_paths", []).append(out_path)
            except Exception as e:
                results.setdefault("render_errors", []).append(str(e))
    return results
