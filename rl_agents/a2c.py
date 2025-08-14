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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=algo_config.get("learning_rate", 3e-4))
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
                obs2, r, term, trunc, info = self.env.step(int(a.item()))
                done = term or trunc
                states.append(s); actions.append(a); rewards.append(torch.tensor([r], dtype=torch.float32))
                dones.append(torch.tensor([float(done)], dtype=torch.float32))
                values.append(v.cpu()); logps.append(logp.cpu())
                ep_return += float(r); ep_len += 1
                obs = obs2; global_steps += 1
                if done:
                    logger.log({"step": global_steps, "ep_return": ep_return, "ep_len": ep_len, "quality": float(info.get("quality", 0.0))})
                    obs, info = self.env.reset(); ep_return, ep_len = 0.0, 0
                if global_steps >= self.total_steps:
                    break
            with torch.no_grad():
                last_v = self.model(self.preprocess_observation(obs).to(self.device))[1].cpu()
            rewards_t = torch.cat(rewards); values_t = torch.cat(values); dones_t = torch.cat(dones)
            adv = rewards_t.squeeze(-1) + self.gamma * last_v * (1.0 - dones_t.squeeze(-1)) - values_t
            returns = adv + values_t
            states_t = torch.cat([s for s in states]).to(self.device)
            actions_t = torch.stack(actions).squeeze(-1)
            logits, v_pred = self.model(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(actions_t.to(self.device))
            ent = dist.entropy().mean()
            pg_loss = -(adv.to(self.device).detach() * new_logps).mean()
            v_loss = F.mse_loss(v_pred, returns.to(self.device))
            loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()
        logger.close()

    def evaluate(self):
        from utils.rollouts import evaluate_agent
        return evaluate_agent(self.env, self, episodes=self.algo_config.get("eval_episodes", 10))
