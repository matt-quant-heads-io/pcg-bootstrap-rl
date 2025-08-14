from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import JSONLLogger
from .rl_agent_base import RLAgentBase

class PPOAgent(RLAgentBase):
    def __init__(self, env, model, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=algo_config.get("learning_rate", 3e-4))
        self.gamma = algo_config.get("gamma", 0.99)
        self.lam = algo_config.get("gae_lambda", 0.95)
        self.clip_coef = algo_config.get("clip_coef", 0.2)
        self.vf_coef = algo_config.get("vf_coef", 0.5)
        self.ent_coef = algo_config.get("ent_coef", 0.01)
        self.batch_size = algo_config.get("batch_size", 4096)
        self.minibatch_size = algo_config.get("minibatch_size", 256)
        self.update_epochs = algo_config.get("update_epochs", 4)
        self.rollout_length = algo_config.get("rollout_length", 128)
        self.total_steps = algo_config.get("total_steps", 200000)
        self.run_dir = run_dir

    @torch.no_grad()
    def _policy(self, state):
        logits, value = self.model(state.to(self.device))
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        logp = probs.log_prob(action)
        return action.item(), logp.cpu(), value.cpu(), probs.entropy().mean().item()

    @torch.no_grad()
    def act_eval(self, proc_state):
        logits, _ = self.model(proc_state.to(self.device))
        return torch.argmax(logits, dim=-1).item()

    def _compute_gae(self, rewards, values, dones, next_value):
        adv = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            next_values = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_values * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def train(self):
        logger = JSONLLogger(self.run_dir)
        obs, info = self.env.reset()
        ep_return, ep_len, global_steps = 0.0, 0, 0
        while global_steps < self.total_steps:
            states, actions, logps, rewards, dones, values = [], [], [], [], [], []
            for _ in range(self.rollout_length):
                state = self.preprocess_observation(obs)
                a, logp, v, _ = self._policy(state)
                next_obs, r, term, trunc, info = self.env.step(int(a))
                done = term or trunc
                states.append(state)
                actions.append(a)
                logps.append(logp)
                rewards.append(torch.tensor([r], dtype=torch.float32))
                dones.append(torch.tensor([float(done)], dtype=torch.float32))
                values.append(v)
                ep_return += float(r); ep_len += 1
                obs = next_obs
                global_steps += 1
                if done:
                    logger.log({"step": global_steps, "ep_return": ep_return, "ep_len": ep_len, "quality": float(info.get("quality", 0.0))})
                    obs, info = self.env.reset(); ep_return, ep_len = 0.0, 0
                if global_steps >= self.total_steps:
                    break
            with torch.no_grad():
                last_state = self.preprocess_observation(obs)
                _, last_v = self.model(last_state.to(self.device))
                last_v = last_v.cpu()
            states_t = torch.cat([s for s in states]).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            logps_t = torch.stack(logps).to(self.device)
            rewards_t = torch.cat(rewards).to(self.device)
            dones_t = torch.cat(dones).to(self.device)
            values_t = torch.cat(values).to(self.device)
            adv, returns = self._compute_gae(rewards_t.squeeze(-1), values_t, dones_t.squeeze(-1), last_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            B = len(actions)
            idxs = np.arange(B)
            for _ in range(self.update_epochs):
                np.random.shuffle(idxs)
                for start in range(0, B, self.minibatch_size):
                    mb = idxs[start:start+self.minibatch_size]
                    mb_states = states_t[mb]
                    mb_actions = actions_t[mb]
                    mb_old_logps = logps_t[mb]
                    mb_adv = adv[mb]
                    mb_returns = returns[mb]
                    logits, values_pred = self.model(mb_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logps = dist.log_prob(mb_actions)
                    ratio = (new_logps - mb_old_logps).exp()
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = torch.nn.functional.mse_loss(values_pred, mb_returns)
                    ent = dist.entropy().mean()
                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
        logger.close()

    @torch.no_grad()
    def evaluate(self):
        from utils.rollouts import evaluate_agent
        return evaluate_agent(self.env, self, episodes=self.algo_config.get("eval_episodes", 10))
