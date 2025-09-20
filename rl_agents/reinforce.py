from typing import Dict, Any
import numpy as np
import torch
from utils.logging import JSONLLogger
from .rl_agent_base import RLAgentBase

class REINFORCEAgent(RLAgentBase):
    def __init__(self, env, model, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(algo_config.get("learning_rate", 3e-4)))
        self.gamma = algo_config.get("gamma", 0.99)
        self.total_steps = algo_config.get("total_steps", 200000)
        self.run_dir = run_dir

    @torch.no_grad()
    def act_eval(self, state):
        logits, _ = self.model(state.to(self.device))
        return torch.argmax(logits, dim=-1).item()

    def train(self):
        logger = JSONLLogger(self.run_dir)
        if self.algo_config["do_pretrain"]:
            self.run_pretrain()
        obs, info = self.env.reset()
        ep_return, ep_len, steps = 0.0, 0, 0
        logps, rewards = [], []

        # rolling checkpoint boundary (reach-or-pass)
        next_ckpt = int(self.checkpoint_interval) if int(self.checkpoint_interval) > 0 else None

        while steps < self.total_steps:
            s = self.preprocess_observation(obs)
            logits, _ = self.model(s.to(self.device))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            # keep log-prob ON GRAPH (no detach/cpu here)
            logps.append(dist.log_prob(a))

            # safe env step
            obs, r, term, trunc, info = self.env_step_safe(int(a.item()))
            rewards.append(float(r))

            ep_return += float(r)
            ep_len += 1
            steps += 1

            # checkpoint (reach-or-pass boundary)
            if next_ckpt is not None and steps >= next_ckpt:
                ck = {
                    "checkpoint_path": self._save_checkpoint(steps, {
                        "algo": "REINFORCE",
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.opt.state_dict(),
                    }),
                    "ckpt_step": int(steps),
                }
                qd = self._compute_interval_qd()
                if qd is not None:
                    q_mean, d_mean = qd
                    ck["interval_quality_mean"] = q_mean
                    ck["interval_diversity_mean"] = d_mean
                self._interval_contents.clear()

                logger.log({"step": steps, **ck})
                eval_eps = int(self.algo_config.get("eval_episodes", 10))
                eval_res = self._evaluate_for_checkpoint(eval_eps)
                self._write_eval_json(steps, eval_res, extra=ck)
                next_ckpt += int(self.checkpoint_interval)

            if term or trunc:
                # Monte Carlo returns (normalize for variance reduction)
                G = 0.0
                returns = []
                for rr in reversed(rewards):
                    G = rr + self.gamma * G
                    returns.append(G)
                returns.reverse()
                returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

                logps_t = torch.stack(logps).to(self.device)  # keep grads
                loss = -(logps_t * returns_t).mean()

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # record final content for interval Q/D
                try:
                    self._append_final_content(obs)
                except Exception:
                    pass

                logger.log({
                    "step": steps,
                    "ep_return": ep_return,
                    "ep_len": ep_len,
                    "quality": float(info.get("quality", 0.0))
                })

                # reset episode buffers
                obs, info = self.env.reset()
                ep_return, ep_len = 0.0, 0
                logps, rewards = [], []

        logger.close()


