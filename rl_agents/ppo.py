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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def train(self):
        logger = JSONLLogger(self.run_dir)
        if self.algo_config["do_pretrain"]:
            self.run_pretrain()
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
                logps.append(logp.detach().cpu())
                rewards.append(torch.tensor([r], dtype=torch.float32))
                dones.append(torch.tensor([float(done)], dtype=torch.float32))
                values.append(v.detach().cpu())

                ep_return += float(r)
                ep_len += 1
                obs = next_obs
                global_steps += 1

                # checkpoint check
                ck = self.maybe_checkpoint(global_steps, {
                    "algo": "PPO",
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                })
                if ck is not None:
                    logger.log({"step": global_steps, **ck})
                    # NEW: evaluate at this checkpoint and write eval.json[step]
                    eval_eps = int(self.algo_config.get("eval_episodes", 10))
                    eval_res = self._evaluate_for_checkpoint(eval_eps, save_images=True, step=global_steps)
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
                last_state = self.preprocess_observation(obs).to(self.device)
                _, last_v = self.model(last_state)
                last_v = last_v.squeeze(-1)

            states_t = torch.cat([s for s in states]).to(self.device)  # [B,C,H,W]
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            logps_t = torch.stack(logps).to(self.device)
            rewards_t = torch.cat(rewards).to(self.device)
            dones_t = torch.cat(dones).to(self.device)
            values_t = torch.cat(values).to(self.device)

            adv, returns = self._compute_gae(
                rewards_t.squeeze(-1),
                values_t,
                dones_t.squeeze(-1),
                last_v
            )

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            B = len(actions)
            idxs = np.arange(B)
            for _ in range(self.update_epochs):
                np.random.shuffle(idxs)
                for start in range(0, B, self.minibatch_size):
                    mb = idxs[start:start + self.minibatch_size]
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

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
        logger.close()

        
