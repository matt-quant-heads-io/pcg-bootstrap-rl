from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from utils.logging import JSONLLogger
from utils.replay_buffer import ReplayBuffer
from .rl_agent_base import RLAgentBase
from models.qnet import QNet

class DQNAgent(RLAgentBase):
    def __init__(self, env, model_unused, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- normalize env.action_space to Discrete(n), done INSIDE __init__ ---
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
                raise ValueError("DQNAgent: could not infer discrete action size from env.")
            env.action_space = spaces.Discrete(n_act)
        self.n_actions = env.action_space.n
        # ----------------------------------------------------------------------

        # derive shapes
        dummy_obs, _ = env.reset()
        s = self.preprocess_observation(dummy_obs)  # [1,C,H,W]
        c, h, w = s.shape[1:]

        # nets
        self.q = QNet((c, h, w), self.n_actions).to(self.device)
        self.qt = QNet((c, h, w), self.n_actions).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=algo_config.get("learning_rate", 3e-4))

        # hparams
        self.gamma = algo_config.get("gamma", 0.99)
        self.batch_size = algo_config.get("batch_size", 256)
        self.target_update = algo_config.get("target_update", 10_000)
        self.total_steps = algo_config.get("total_steps", 200_000)
        self.start_learning = algo_config.get("start_learning", 10_000)
        self.eps_start = algo_config.get("eps_start", 1.0)
        self.eps_end = algo_config.get("eps_end", 0.05)
        self.eps_decay = algo_config.get("eps_decay", 500_000)
        self.buffer = ReplayBuffer(algo_config.get("replay_size", 200_000))
        self.run_dir = run_dir
        self._step = 0

    def _eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self._step / self.eps_decay)

    @torch.no_grad()
    def act_eval(self, proc_state):
        q = self.q(proc_state.to(self.device))
        return int(torch.argmax(q, dim=-1).item())

    @torch.no_grad()
    def _act_train(self, proc_state):
        if np.random.rand() < self._eps():
            return int(np.random.randint(self.n_actions))
        return self.act_eval(proc_state)

    def _learn(self):
        if len(self.buffer) < self.start_learning:
            return
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        # s, s2 should be [B,C,H,W] after your ReplayBuffer fix (cat, not stack)
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        s2 = s2.to(self.device); d = d.to(self.device)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.qt(s2).max(dim=1).values
            tgt = r + self.gamma * (1.0 - d) * q2
        loss = F.mse_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

    def train(self):
        logger = JSONLLogger(self.run_dir)
        obs, info = self.env.reset()
        ep_return, ep_len = 0.0, 0

        while self._step < self.total_steps:
            s = self.preprocess_observation(obs)
            # epsilon-greedy
            a = self._act_train(s)

            obs2, r, term, trunc, info = self.env.step(int(a))
            d = term or trunc
            s2 = self.preprocess_observation(obs2)

            # store
            self.buffer.push(s, a, r, s2, d)

            # learn
            self._learn()

            # target network sync
            if self._step % self.target_update == 0:
                self.qt.load_state_dict(self.q.state_dict())

            ep_return += float(r)
            ep_len += 1
            obs = obs2
            self._step += 1

            # checkpoint check
            ck = self.maybe_checkpoint(self._step, {
                "algo": "DQN",
                "q_state_dict": self.q.state_dict(),
                "qt_state_dict": self.qt.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            })
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

        
