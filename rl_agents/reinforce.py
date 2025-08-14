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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=algo_config.get("learning_rate", 3e-4))
        self.gamma = algo_config.get("gamma", 0.99)
        self.total_steps = algo_config.get("total_steps", 200000)
        self.run_dir = run_dir

    @torch.no_grad()
    def act_eval(self, state):
        logits, _ = self.model(state.to(self.device))
        return torch.argmax(logits, dim=-1).item()

    def train(self):
        logger = JSONLLogger(self.run_dir)
        obs, info = self.env.reset()
        ep_return, ep_len, steps = 0.0, 0, 0
        logps, rewards = [], []
        while steps < self.total_steps:
            s = self.preprocess_observation(obs)
            logits, _ = self.model(s.to(self.device))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            obs, r, term, trunc, info = self.env.step(int(a.item()))
            rewards.append(r)
            ep_return += float(r); ep_len += 1; steps += 1
            if term or trunc:
                G = 0.0; returns = []
                for r in reversed(rewards):
                    G = r + self.gamma * G
                    returns.append(G)
                returns.reverse()
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                logps_t = torch.stack(logps).to(self.device)
                loss = -(logps_t * returns).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                logger.log({"step": steps, "ep_return": ep_return, "ep_len": ep_len, "quality": float(info.get("quality", 0.0))})
                obs, info = self.env.reset(); ep_return, ep_len = 0.0, 0; logps, rewards = [], []
        logger.close()

    def evaluate(self):
        from utils.rollouts import evaluate_agent
        return evaluate_agent(self.env, self, episodes=self.algo_config.get("eval_episodes", 10))
