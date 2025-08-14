from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .rl_agent_base import RLAgentBase
from models.actor_categorical import ActorCategorical
from models.qnet import QNet
from utils.replay_buffer import ReplayBuffer

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

class TD3Backbone(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.policy = ActorCategorical(obs_shape, n_actions)
        self.q1 = QNet(obs_shape, n_actions)
        self.q2 = QNet(obs_shape, n_actions)

class TD3DiscreteAgent(RLAgentBase):
    def __init__(self, env, model_unused, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs0, _ = env.reset()
        s = self.preprocess_observation(obs0)
        c,h,w = s.shape[1:]
        self.n_actions = env.action_space.n
        self.model = TD3Backbone((c,h,w), self.n_actions).to(self.device)
        self.target = TD3Backbone((c,h,w), self.n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.gamma = float(algo_config.get("gamma", 0.99))
        self.tau = float(algo_config.get("tau", 0.005))
        self.lr = float(algo_config.get("learning_rate", 3e-4))
        self.batch_size = int(algo_config.get("batch_size", 256))
        self.total_steps = int(algo_config.get("total_steps", 200000))
        self.init_random = int(algo_config.get("init_random_steps", 10000))
        self.policy_delay = int(algo_config.get("policy_delay", 2))
        self.logit_noise_std = float(algo_config.get("logit_noise_std", 0.2))
        self.q_opt = optim.Adam(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=self.lr)
        self.pi_opt = optim.Adam(self.model.policy.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(algo_config.get("replay_size", 200000))
        self.run_dir = run_dir
        self._step = 0

    @torch.no_grad()
    def act_eval(self, s):
        logits = self.model.policy(s.to(self.device))
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def _act_train(self, s, eps=0.1):
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        logits = self.model.policy(s.to(self.device))
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        return s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

    def train(self):
        from utils.logging import JSONLLogger
        logger = JSONLLogger(self.run_dir)
        obs, info = self.env.reset()
        ep_return, ep_len, updates = 0.0, 0, 0
        while self._step < self.total_steps:
            s = self.preprocess_observation(obs)
            a = self._act_train(s)
            obs2, r, term, trunc, info = self.env.step(a)
            d = term or trunc
            s2 = self.preprocess_observation(obs2)
            self.buffer.push(s, a, r, s2, d)
            batch = self._sample_batch()
            if batch is not None and self._step >= self.init_random:
                q_loss = self._update_critics(*batch)
                updates += 1
                if updates % self.policy_delay == 0:
                    pi_loss = self._update_actor(batch[0])
                    soft_update(self.target, self.model, self.tau)
                if updates % 1000 == 0:
                    logger.log({"step": self._step, "q_loss": float(q_loss), "pi_loss": float(pi_loss if updates % self.policy_delay == 0 else 0.0)})
            ep_return += float(r); ep_len += 1
            obs = obs2; self._step += 1
            if d:
                logger.log({"step": self._step, "ep_return": ep_return, "ep_len": ep_len, "quality": float(info.get("quality", 0.0))})
                obs, info = self.env.reset(); ep_return, ep_len = 0.0, 0
        logger.close()

    def _update_critics(self, s, a, r, s2, d):
        with torch.no_grad():
            logits_next = self.target.policy(s2)
            noise = torch.randn_like(logits_next) * self.logit_noise_std
            logits_next = logits_next + noise
            a2 = torch.argmax(logits_next, dim=-1, keepdim=True)
            q1_t = self.target.q1(s2).gather(1, a2).squeeze(1)
            q2_t = self.target.q2(s2).gather(1, a2).squeeze(1)
            q_t = torch.min(q1_t, q2_t)
            y = r + (1.0 - d) * self.gamma * q_t
        q1 = self.model.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.model.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
        self.q_opt.zero_grad(); loss.backward(); self.q_opt.step()
        return float(loss.detach().item())

    def _update_actor(self, s):
        logits = self.model.policy(s)
        pi = torch.softmax(logits, dim=-1)
        q1 = self.model.q1(s); q2 = self.model.q2(s)
        q_min = torch.min(q1, q2)
        loss = -(pi * q_min).sum(dim=-1).mean()
        self.pi_opt.zero_grad(); loss.backward(); self.pi_opt.step()
        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self):
        from utils.rollouts import evaluate_agent
        return evaluate_agent(self.env, self, episodes=self.algo_config.get("eval_episodes", 10))
