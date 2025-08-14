from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from .rl_agent_base import RLAgentBase
from models.actor_categorical import ActorCategorical
from models.qnet import QNet
from utils.logging import JSONLLogger
from utils.replay_buffer import ReplayBuffer

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

class SACBackbone(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.policy = ActorCategorical(obs_shape, n_actions)
        self.q1 = QNet(obs_shape, n_actions)
        self.q2 = QNet(obs_shape, n_actions)

class SACDiscreteAgent(RLAgentBase):
    def __init__(self, env, model_unused, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- normalize env.action_space to Discrete(n) ---
        n_act = getattr(getattr(env, "action_space", None), "n", None)
        if n_act is None or isinstance(env.action_space, int):
            n_act = None
            if hasattr(env, "prob_config") and isinstance(env.prob_config, dict):
                tiles = env.prob_config.get("tiles")
                if isinstance(tiles, (list, tuple)):
                    n_act = len(tiles)
            if n_act is None:
                try:
                    n_act = int(np.max(env.observation_space.high)) + 1
                except Exception:
                    n_act = None
            if n_act is None:
                try:
                    rng = env._problem._content_space.range()
                    n_act = int(rng["max"])
                except Exception:
                    pass
            if n_act is None:
                raise ValueError("SACDiscreteAgent: could not infer discrete action size from env.")
            env.action_space = spaces.Discrete(n_act)
        self.n_actions = env.action_space.n
        # -------------------------------------------------

        obs0, _ = env.reset()
        s = self.preprocess_observation(obs0)
        c, h, w = s.shape[1:]

        self.model = SACBackbone((c, h, w), self.n_actions).to(self.device)
        self.target = SACBackbone((c, h, w), self.n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.gamma = float(algo_config.get("gamma", 0.99))
        self.tau = float(algo_config.get("tau", 0.005))
        self.lr = float(algo_config.get("learning_rate", 3e-4))
        self.total_steps = int(algo_config.get("total_steps", 200000))
        self.batch_size = int(algo_config.get("batch_size", 256))
        self.init_random = int(algo_config.get("init_random_steps", 10000))
        self.learn_alpha = bool(algo_config.get("learn_alpha", True))
        self.alpha = float(algo_config.get("alpha", 0.2))
        self.target_entropy = float(algo_config.get("target_entropy", -1.0))

        self.q_opt = optim.Adam(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=self.lr)
        self.pi_opt = optim.Adam(self.model.policy.parameters(), lr=self.lr)
        if self.learn_alpha:
            self.log_alpha = torch.tensor(np.log(self.alpha), device=self.device, requires_grad=True)
            self.a_opt = optim.Adam([self.log_alpha], lr=self.lr)

        self.buffer = ReplayBuffer(algo_config.get("replay_size", 200000))
        self.run_dir = run_dir
        self._step = 0

    @torch.no_grad()
    def act_eval(self, s):
        logits = self.model.policy(s.to(self.device))
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def _act_train(self, s):
        logits = self.model.policy(s.to(self.device))
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        # safety: ensure 4D [B,C,H,W]
        if s.ndim == 5:  s  = s.squeeze(1)
        if s2.ndim == 5: s2 = s2.squeeze(1)
        return s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

    def train(self):
        logger = JSONLLogger(self.run_dir)
        obs, info = self.env.reset()
        ep_return, ep_len = 0.0, 0

        while self._step < self.total_steps:
            s = self.preprocess_observation(obs)

            if self._step < self.init_random:
                a = int(np.random.randint(self.n_actions))
            else:
                a = self._act_train(s)

            obs2, r, term, trunc, info = self.env.step(a)
            d = term or trunc
            s2 = self.preprocess_observation(obs2)
            self.buffer.push(s, a, r, s2, d)

            batch = self._sample_batch()
            if batch is not None and self._step >= self.init_random:
                loss_dict = self._update(*batch)
                if self._step % 1000 == 0:
                    logger.log({"step": self._step, **{k: float(v) for k, v in loss_dict.items()}})

            ep_return += float(r)
            ep_len += 1
            obs = obs2
            self._step += 1

            # checkpoint check
            payload = {
                "algo": "SAC-Discrete",
                "policy_state_dict": self.model.policy.state_dict(),
                "q1_state_dict": self.model.q1.state_dict(),
                "q2_state_dict": self.model.q2.state_dict(),
                "target_policy_state_dict": self.target.policy.state_dict(),
                "target_q1_state_dict": self.target.q1.state_dict(),
                "target_q2_state_dict": self.target.q2.state_dict(),
                "pi_opt_state_dict": self.pi_opt.state_dict(),
                "q_opt_state_dict": self.q_opt.state_dict(),
            }
            if getattr(self, "learn_alpha", False) and hasattr(self, "log_alpha"):
                payload["log_alpha"] = self.log_alpha
                if hasattr(self, "a_opt"):
                    payload["a_opt_state_dict"] = self.a_opt.state_dict()
            ck = self.maybe_checkpoint(self._step, payload)
            if ck is not None:
                logger.log({"step": self._step, **ck})
                eval_eps = int(self.algo_config.get("eval_episodes", 10))
                eval_res = self._evaluate_for_checkpoint(eval_eps)
                self._write_eval_json(self._step, eval_res, extra=ck)

            if d:
                # bucket the final content of this episode for interval Q/D
                try:
                    self._append_final_content(obs)
                except Exception:
                    pass
                logger.log({
                    "step": self._step,
                    "ep_return": ep_return,
                    "ep_len": ep_len,
                    "quality": float(info.get("quality", 0.0))
                })
                obs, info = self.env.reset()
                ep_return, ep_len = 0.0, 0
        logger.close()


    def _update(self, s, a, r, s2, d):
        # Critic
        with torch.no_grad():
            logits_next = self.target.policy(s2)
            pi_next = torch.softmax(logits_next, dim=-1)
            log_pi_next = torch.log_softmax(logits_next, dim=-1)
            q1_t = self.target.q1(s2); q2_t = self.target.q2(s2)
            q_min = torch.min(q1_t, q2_t)
            alpha = self.log_alpha.exp() if self.learn_alpha else torch.tensor(self.alpha, device=self.device)
            v_next = (pi_next * (q_min - alpha * log_pi_next)).sum(dim=-1)
            y = r + (1.0 - d) * self.gamma * v_next

        q1 = self.model.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.model.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # Actor
        logits = self.model.policy(s)
        pi = torch.softmax(logits, dim=-1)
        log_pi = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q1_pi = self.model.q1(s); q2_pi = self.model.q2(s)
            q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp() if self.learn_alpha else torch.tensor(self.alpha, device=self.device)
        pi_loss = (pi * (alpha * log_pi - q_pi)).sum(dim=-1).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.learn_alpha:
            target_H = self.target_entropy if self.target_entropy != -1.0 else -np.log(pi.shape[-1])
            alpha_loss = -(self.log_alpha * (log_pi.detach() + target_H).mean())
            self.a_opt.zero_grad(); alpha_loss.backward(); self.a_opt.step()

        soft_update(self.target, self.model, self.tau)
        return {"loss/q": q_loss.detach(), "loss/pi": pi_loss.detach(),
                "alpha": (self.log_alpha.exp().detach() if self.learn_alpha else torch.tensor(self.alpha))}
