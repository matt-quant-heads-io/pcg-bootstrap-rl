from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import JSONLLogger
from .rl_agent_base import RLAgentBase

class A2CAgent(RLAgentBase):
    def __init__(self, env, model, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(algo_config.get("learning_rate", 3e-4)))
        self.gamma = algo_config.get("gamma", 0.99)
        self.ent_coef = algo_config.get("ent_coef", 0.01)
        self.vf_coef = algo_config.get("vf_coef", 0.5)
        self.rollout_length = algo_config.get("rollout_length", 128)
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
        ep_return, ep_len, global_steps = 0.0, 0, 0

        while global_steps < self.total_steps:
            states, actions, rewards, dones, values, logps = [], [], [], [], [], []

            for _ in range(self.rollout_length):
                s = self.preprocess_observation(obs)
                logits, v = self.model(s.to(self.device))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)

                # obs2, r, term, trunc, info = self.env.step(int(a.item()))
                obs2, r, term, trunc, info = self.env_step_safe(int(a.item()))
                done = term or trunc

                states.append(s)
                actions.append(a)
                rewards.append(torch.tensor([r], dtype=torch.float32))
                dones.append(torch.tensor([float(done)], dtype=torch.float32))
                values.append(v.detach().cpu())
                logps.append(logp.detach().cpu())

                ep_return += float(r)
                ep_len += 1
                obs = obs2
                global_steps += 1

                # checkpoint check
                if self.checkpoint_interval > 0 and (global_steps % self.checkpoint_interval) == 0:
                    print(f"[A2C] checkpoint_interval = {self.checkpoint_interval}, run_dir = {self.run_dir}")
                    ck = {
                        "checkpoint_path": self._save_checkpoint(global_steps, {
                            "algo": "A2C",
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.opt.state_dict(),
                        }),
                        "ckpt_step": int(global_steps),
                    }
                    # attach interval Q/D means (and clear interval bucket)
                    qd = self._compute_interval_qd()
                    if qd is not None:
                        q_mean, d_mean = qd
                        ck["interval_quality_mean"] = q_mean
                        ck["interval_diversity_mean"] = d_mean
                    self._interval_contents.clear()

                    # log + eval and write metrics keyed by checkpoint
                    logger.log({"step": global_steps, **ck})
                    eval_eps = int(self.algo_config.get("eval_episodes", 10))
                    eval_res = self._evaluate_for_checkpoint(eval_eps)
                    self._write_eval_json(global_steps, eval_res, extra=ck)

                if done:
                    # bucket the final content of this episode for interval Q/D
                    try:
                        self._append_final_content(obs)
                    except Exception:
                        pass
                    logger.log({
                        "step": global_steps,
                        "ep_return": ep_return,
                        "ep_len": ep_len,
                        "quality": float(info.get("quality", 0.0))
                    })
                    obs, info = self.env.reset()
                    ep_return, ep_len = 0.0, 0

                if global_steps >= self.total_steps:
                    break

            with torch.no_grad():
                last_v = self.model(self.preprocess_observation(obs).to(self.device))[1].squeeze(-1)

            rewards_t = torch.cat(rewards).to(self.device)
            values_t = torch.cat(values).to(self.device)
            dones_t = torch.cat(dones).to(self.device)
            adv = rewards_t.squeeze(-1) + self.gamma * last_v * (1.0 - dones_t.squeeze(-1)) - values_t
            returns = adv + values_t

            states_t = torch.cat([s for s in states]).to(self.device)
            actions_t = torch.stack(actions).squeeze(-1).to(self.device)
            logits, v_pred = self.model(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(actions_t)
            ent = dist.entropy().mean()
            pg_loss = -(adv.detach() * new_logps).mean()
            v_loss = F.mse_loss(v_pred, returns)
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()
        logger.close()


