from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import JSONLLogger
from .rl_agent_base import RLAgentBase

import os, json


import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch


def _save_eval_levels(self, step: int, contents_list):
    """
    Save level images for a given checkpoint step under:
      <run_dir>/levels/ckpt_<step>/level_XXXX.png
    """
    # Normalize step to int (support strings like "step_2000")
    if isinstance(step, str):
        try:
            step = int(str(step).split("_")[-1])
        except Exception:
            step = int(step) if str(step).isdigit() else 0

    out_dir = Path(self.run_dir) / "levels" / f"ckpt_{step:08d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, content in enumerate(contents_list, start=1):
        try:
            # env.render can accept a single content or list; we use one-at-a-time
            rendered = self.env.render(content)
            # If your render already returns a PIL.Image, keep it;
            # if it's a numpy array, convert to Image
            if isinstance(rendered, Image.Image):
                img = rendered
            else:
                # assume HxW or HxWxC ndarray -> to uint8
                import numpy as np
                arr = np.asarray(rendered)
                if arr.dtype != np.uint8:
                    # simple normalization if not uint8
                    arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-9)).astype("uint8")
                if arr.ndim == 2:
                    img = Image.fromarray(arr, mode="L")
                else:
                    img = Image.fromarray(arr)

            img.save(out_dir / f"level_{i:04d}.png")
        except Exception as e:
            print(f"[warn] failed to save level {i} for step {step}: {e}")

    print(f"[levels] wrote {len(contents_list)} image(s) to {out_dir}")


class PPOAgent(RLAgentBase):
    def __init__(self, env, model, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(algo_config.get("learning_rate", 3e-4)))
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

    def _format_action_for_env(self, a):
        """
        Accepts:
        - scalar int
        - torch.Tensor or np.ndarray with shape () or (B,)
        Returns:
        - int for single env
        - np.ndarray (B,) for vectorized envs
        """
        # normalize to numpy
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        if isinstance(a, (list, tuple)):
            a = np.asarray(a)

        # scalar case
        if not isinstance(a, np.ndarray):
            return int(a)

        # ndarray cases
        if a.ndim == 0:
            return int(a)

        # batched vs single
        num_envs = getattr(self.env, "num_envs", 1)
        if num_envs > 1:
            return a.astype(np.int64, copy=False)
        else:
            # single env (compat wrapper): take first element as scalar
            return int(a[0])

    @torch.no_grad()
    def _policy(self, state):
        logits, value = self.model(state.to(self.device))
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        logp = probs.log_prob(action)
        act = action.detach().cpu()
        act = act.numpy() if act.numel() > 1 else act.item()
        return act, logp.detach().cpu(), value.detach().cpu(), probs.entropy().mean().item()

    @torch.no_grad()
    def act_eval(self, state):
        """
        Greedy action for eval. Returns:
        - int for single env
        - np.ndarray of shape (B,) for vectorized envs
        """
        import numpy as np
        import torch

        device = getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu")

        # If a dict slips in, preprocess first
        if isinstance(state, dict):
            state = self.preprocess_observation(state)

        state = state.to(device)
        if state.ndim == 3:  # (C,H,W) -> (1,C,H,W)
            state = state.unsqueeze(0)

        logits, _ = self.model(state)              # (B, n_actions)
        acts = torch.argmax(logits, dim=-1)        # (B,)

        if acts.numel() == 1:
            return int(acts.item())                # single env
        return acts.detach().cpu().numpy().astype(np.int64)  # vectorized



    def _compute_gae(self, rewards, values, dones, next_value):
        # Ensure all tensors are on the same device
        device = values.device
        rewards = rewards.to(device)
        values = values.to(device)
        dones = dones.to(device)
        next_value = next_value.to(device)

        adv = torch.zeros_like(rewards, device=device)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            next_values = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_values * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def _compute_dones(self, term, trunc):
        """Return bool for single env, or (N,) bool array for vector envs."""
        term = np.asarray(term)
        trunc = np.asarray(trunc)
        if term.ndim == 0:  # single env
            return bool(term or trunc)
        return np.logical_or(term, trunc)

    def _maybe_reset_done_envs(self, dones, next_obs):
        """
        For vector envs, reset only the done indices and splice their fresh obs
        back into the batched next_obs dict. For single env, reset if done.
        """
        num_envs = getattr(self.env, "num_envs", 1)

        # Single env
        if num_envs == 1:
            if bool(dones):
                next_obs, _ = self.env.reset()
            return next_obs

        # Vector env
        dones = np.asarray(dones, dtype=bool)
        if dones.ndim == 0:
            # occasional odd cases; normalize to vector length 1
            dones = np.asarray([bool(dones)], dtype=bool)

        idx = np.nonzero(dones)[0]
        if idx.size == 0:
            return next_obs

        # Reset those envs individually and overwrite their slices in next_obs
        # NOTE: our PCGVectorRunner exposes .envs
        for i in idx:
            o_i, _ = self.env.envs[i].reset()
            # next_obs is a dict with numpy arrays shaped (N, ...)
            if isinstance(next_obs, dict):
                for k in next_obs.keys():
                    # all obs entries must be indexable by env index
                    next_obs[k][i] = o_i[k]
            else:
                # if your obs isn't a dict, adapt this indexing accordingly
                next_obs[i] = o_i
        return next_obs

    def _format_action_for_env(self, a):
        """(From previous step) Normalize actions to scalar (single) or (N,) (vector)."""
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        if isinstance(a, (list, tuple)):
            a = np.asarray(a)
        num_envs = getattr(self.env, "num_envs", 1)
        if isinstance(a, np.ndarray):
            if a.ndim == 0:
                return int(a) if num_envs == 1 else np.asarray([int(a)])
            return a.astype(np.int64, copy=False) if num_envs > 1 else int(a[0])
        return int(a)


    def train(self):
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Categorical

        logger = JSONLLogger(self.run_dir)
        if self.algo_config.get("do_pretrain"):
            self.run_pretrain()

        device = getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # ---- Algo hyperparams ----
        gamma          = float(self.algo_config.get("gamma", 0.99))
        gae_lambda     = float(self.algo_config.get("gae_lambda", self.algo_config.get("lam", 0.95)))
        self.clip_coef = float(self.algo_config.get("clip_coef", 0.2))
        self.ent_coef  = float(self.algo_config.get("ent_coef", 0.0))
        self.vf_coef   = float(self.algo_config.get("vf_coef", 0.5))
        self.rollout_length = int(self.algo_config.get("rollout_length", 2048))
        self.update_epochs  = int(self.algo_config.get("update_epochs", 8))
        self.minibatch_size = int(self.algo_config.get("minibatch_size", 256))

        # ---- helpers (batch-safe) ----
        def _format_action_for_env(a):
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            if isinstance(a, (list, tuple)):
                a = np.asarray(a)
            num_envs = getattr(self.env, "num_envs", 1)
            if isinstance(a, np.ndarray):
                if a.ndim == 0:
                    return int(a) if num_envs == 1 else np.asarray([int(a)], dtype=np.int64)
                return a.astype(np.int64, copy=False) if num_envs > 1 else int(a[0])
            return int(a)

        def _compute_dones(term, trunc):
            term = np.asarray(term)
            trunc = np.asarray(trunc)
            if term.ndim == 0:
                return bool(term or trunc)
            return np.logical_or(term, trunc)

        def _maybe_reset_done_envs(dones, next_obs):
            num_envs = getattr(self.env, "num_envs", 1)
            if num_envs == 1:
                if bool(dones):
                    next_obs, _ = self.env.reset()
                return next_obs
            dones = np.asarray(dones, dtype=bool)
            idx = np.nonzero(dones)[0]
            if idx.size == 0:
                return next_obs
            for i in idx:
                o_i, _ = self.env.envs[i].reset()
                if isinstance(next_obs, dict):
                    for k in next_obs.keys():
                        next_obs[k][i] = o_i[k]
                else:
                    next_obs[i] = o_i
            return next_obs

        # ---- init episode accounting ----
        obs, info = self.env.reset()
        state = self.preprocess_observation(obs)            # (B,C,H,W) or (1,C,H,W)
        if state.ndim == 3:
            state = state.unsqueeze(0)
        B = state.shape[0]
        ep_return = np.zeros(B, dtype=np.float32)
        ep_len    = np.zeros(B, dtype=np.int64)
        global_steps = 0

        # ---- training loop ----
        while global_steps < self.total_steps:
            # Decide per-iteration horizon so total transitions ~= rollout_length
            T = max(1, self.rollout_length // max(1, B))

            # storages: (T,B,...) on device
            states    = torch.zeros((T, B) + state.shape[1:], dtype=torch.float32, device=device)
            actions   = torch.zeros((T, B), dtype=torch.int64, device=device)
            logps     = torch.zeros((T, B), dtype=torch.float32, device=device)
            rewards   = torch.zeros((T, B), dtype=torch.float32, device=device)
            dones_buf = torch.zeros((T, B), dtype=torch.bool,   device=device)
            values    = torch.zeros((T, B), dtype=torch.float32, device=device)

            for t in range(T):
                states[t] = state

                # sample from policy (your _policy already batch-safe)
                a, logp, v, _ = self._policy(state)

                # store action/logp/value (as vectors of len B)
                if torch.is_tensor(logp): logp_t = logp.view(-1).to(device)
                else:                     logp_t = torch.as_tensor(np.asarray(logp).reshape(-1), dtype=torch.float32, device=device)
                if torch.is_tensor(v):    v_t    = v.view(-1).to(device)
                else:                     v_t    = torch.as_tensor(np.asarray(v).reshape(-1), dtype=torch.float32, device=device)

                actions[t] = torch.as_tensor(a, dtype=torch.int64, device=device).view(-1)
                logps[t]   = logp_t
                values[t]  = v_t

                # env step
                env_action = _format_action_for_env(a)
                obs_next, r, term, trunc, info = self.env.step(env_action)

                # rewards & dones to vectors
                r_np = np.asarray(r, dtype=np.float32)
                if r_np.ndim == 0: r_np = np.asarray([r_np], dtype=np.float32)
                dones = _compute_dones(term, trunc)
                d_np = np.asarray(dones, dtype=bool)
                if d_np.ndim == 0: d_np = np.asarray([bool(d_np)], dtype=bool)

                # accumulate episode stats per env
                ep_return += r_np
                ep_len    += 1

                # store rewards/dones
                rewards[t]   = torch.from_numpy(r_np).to(device)
                dones_buf[t] = torch.from_numpy(d_np).to(device)

                # per-env episode end logging + final content hook
                done_idx = np.nonzero(d_np)[0]
                for i in done_idx:
                    try:
                        # provide single-env-shaped obs to your hook if it expects that
                        if isinstance(obs_next, dict):
                            one_obs = {k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in obs_next.items()}
                            self._append_final_content(one_obs)
                        else:
                            self._append_final_content(obs_next)
                    except Exception:
                        pass
                    logger.log({
                        "step": global_steps + (t + 1) * B,
                        "env_id": int(i),
                        "ep_return": float(ep_return[i]),
                        "ep_len": int(ep_len[i]),
                        "quality": float(info[i].get("quality", 0.0)) if isinstance(info, (list, tuple)) and i < len(info) else float(info.get("quality", 0.0)) if isinstance(info, dict) else 0.0
                    })
                    ep_return[i] = 0.0
                    ep_len[i] = 0

                # reset only done envs
                obs_next = _maybe_reset_done_envs(dones, obs_next)

                # next state
                state = self.preprocess_observation(obs_next)
                if state.ndim == 3:
                    state = state.unsqueeze(0)

            # advance global step count by number of environment steps collected
            global_steps += T * B

            # ---- bootstrap last value ----
            with torch.no_grad():
                logits_last, v_last = self.model(state.to(device))
                v_last = v_last.view(-1)  # (B,)

            # ---- compute GAE over (T,B) ----
            advantages = torch.zeros_like(rewards, device=device)
            last_adv = torch.zeros(B, dtype=torch.float32, device=device)
            not_done = (~dones_buf).float()

            for t in reversed(range(T)):
                next_value = v_last if t == T - 1 else values[t + 1]
                delta = rewards[t] + gamma * next_value * not_done[t] - values[t]
                last_adv = delta + gamma * gae_lambda * not_done[t] * last_adv
                advantages[t] = last_adv

            returns = advantages + values

            # ---- flatten (T,B,...) -> (N,...) ----
            def _flat(x):
                return x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1)

            flat_states   = _flat(states)        # (N,C,H,W)
            flat_actions  = _flat(actions)       # (N,)
            flat_logps    = _flat(logps)         # (N,)
            flat_adv      = _flat(advantages)    # (N,)
            flat_returns  = _flat(returns)       # (N,)
            flat_values   = _flat(values)        # (N,)

            # normalize adv
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            N = flat_states.size(0)
            mb_size = min(self.minibatch_size, N)
            num_minibatches = max(1, N // mb_size)

            # ---- PPO updates ----
            for _ in range(self.update_epochs):
                idx = torch.randperm(N, device=device)
                for s in range(0, N, mb_size):
                    mb_idx = idx[s:s + mb_size]
                    s_mb = flat_states[mb_idx]
                    a_mb = flat_actions[mb_idx]
                    old_logp_mb = flat_logps[mb_idx]
                    adv_mb = flat_adv[mb_idx]
                    ret_mb = flat_returns[mb_idx]

                    logits, v_pred = self.model(s_mb)
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(a_mb)
                    entropy = dist.entropy().mean()

                    ratio = (new_logp - old_logp_mb).exp()
                    pg1 = ratio * adv_mb
                    pg2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_mb
                    policy_loss = -torch.min(pg1, pg2).mean()

                    # value loss (clipped)
                    v_pred = v_pred.view(-1)
                    v_clipped = flat_values[mb_idx] + (v_pred - flat_values[mb_idx]).clamp(-self.clip_coef, self.clip_coef)
                    v_loss_unclipped = (v_pred - ret_mb).pow(2)
                    v_loss_clipped   = (v_clipped - ret_mb).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

            # ---- checkpoint & optional eval ----
            ck = self.maybe_checkpoint(global_steps, {
                "algo": "PPO",
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            })
            if ck is not None:
                logger.log({"step": global_steps, **ck})
                eval_eps = int(self.algo_config.get("eval_episodes", 10))
                eval_res = self._evaluate_for_checkpoint(eval_eps, save_images=True, step=global_steps)
                self._write_eval_json(global_steps, eval_res, extra=ck)

        logger.close()


        
