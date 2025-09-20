# train_ppo_jax.py
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints

import pcg_benchmark
from jax_ppo import (PPOConfig, make_ppo_train_state, forward_logits_value,
                     ppo_update, gae_scan)


# -------------------- Cropped window preprocessing --------------------

def crop_window(map_np: np.ndarray, pos: Tuple[int, int], size: int, pad_value: int = 1) -> np.ndarray:
    """
    Mirror of your PyTorch transform:
      - pad full map with constant 'pad_value'
      - extract size×size window with top-left at (y, x) around pos
      - pos = (y, x) in original map indices
    """
    assert size % 2 == 1, "crop_size should be odd"
    y, x = int(pos[0]), int(pos[1])
    pad = size // 2
    padded = np.pad(map_np, pad_width=pad, mode='constant', constant_values=pad_value)
    # Shift indices by pad to account for padding offset
    y0, x0 = y, x
    return padded[y0:y0+size, x0:x0+size]


def obs_to_cropped_onehot(obs: Dict[str, Any], n_channels: int, crop_size: int) -> np.ndarray:
    """Crop -> one-hot -> CHW float32."""
    tile = crop_window(obs['map'], obs['pos'], crop_size, pad_value=1)  # H=W=crop_size
    oh = np.eye(n_channels, dtype=np.float32)[tile.reshape(-1)].reshape(crop_size, crop_size, n_channels)
    return np.transpose(oh, (2, 0, 1))  # (C,H,W)


# -------------------- Files / metrics helpers --------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate(env, params, apply_fn, n_actions: int, crop_size: int,
             episodes: int, step: int, run_dir: Path) -> Dict[str, Any]:
    returns, lens, qualities, infos_dump = [], [], [], []
    for ep in range(episodes):
        try:
            obs, info = env.reset(seed=step + ep)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(step + ep)
            obs, info = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            x = obs_to_cropped_onehot(obs, n_actions, crop_size)[None, ...]  # [1,C,H,W]
            logits, _ = apply_fn({'params': params}, jnp.asarray(x))
            a = int(jax.random.categorical(jax.random.PRNGKey(step + ep + ep_len), logits[0]))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += float(r)
            ep_len += 1
        returns.append(ep_ret)
        lens.append(ep_len)
        qualities.append(float(info.get("quality", 0.0)))
        infos_dump.append({
            "quality": float(info.get("quality", 0.0)),
            "step_count": ep_len,
            "content_info": info.get("content_info", {})
        })
    res = {
        "episodes": episodes,
        "len_mean": float(np.mean(lens)),
        "len_std": float(np.std(lens)),
        "quality_mean": float(np.mean(qualities) if qualities else 0.0),
        "quality_std": float(np.std(qualities) if qualities else 0.0),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "infos": infos_dump
    }
    # aggregate and per-ckpt mirrors
    metrics_dir = ensure_dir(run_dir / "metrics")
    agg_path = metrics_dir / "eval.json"
    agg = {}
    if agg_path.exists():
        try:
            agg = json.loads(agg_path.read_text())
        except Exception:
            agg = {}
    agg[f"step_{int(step)}"] = res
    write_json(agg_path, agg)
    ck_dir = ensure_dir(metrics_dir / "checkpoint" / f"ckpt_{int(step):08d}")
    write_json(ck_dir / "eval.json", res)
    return res


# -------------------- Training --------------------

def train_ppo(env_name: str,
              run_dir: Path,
              config: PPOConfig,
              crop_size: int,
              eval_episodes: int = 10) -> None:
    # Env
    env = pcg_benchmark.make(env_name)
    H, W, n_actions = env.observation_space  # (H, W, C) in your env wrapper
    obs_shape = (n_actions, crop_size, crop_size)  # CHW (cropped + one-hot)

    # RNG & model state
    rng = jax.random.PRNGKey(config.seed)
    rng, init_key = jax.random.split(rng)
    train_state = make_ppo_train_state(init_key, obs_shape, n_actions, config.learning_rate)

    # Buffers
    T = config.rollout_length
    obs_buf   = np.zeros((T, *obs_shape), np.float32)
    act_buf   = np.zeros((T,), np.int32)
    logp_buf  = np.zeros((T,), np.float32)
    rew_buf   = np.zeros((T,), np.float32)
    done_buf  = np.zeros((T,), np.float32)
    val_buf   = np.zeros((T,), np.float32)

    # Start
    obs, info = env.reset()
    global_steps = 0
    t0 = time.time()

    # JITed forward for rollout
    jitted_forward = forward_logits_value

    while global_steps < config.total_steps:
        # -------- Rollout (env stays in Python) --------
        for t in range(T):
            x = obs_to_cropped_onehot(obs, n_actions, crop_size)  # (C,H,W)
            obs_buf[t] = x

            logits, value = jitted_forward(train_state.params, train_state.apply_fn, jnp.asarray(x[None, ...]))
            logits = np.asarray(logits[0])
            value  = float(np.asarray(value)[0])

            # Sample action on host for simplicity
            a = int(np.random.default_rng().choice(n_actions, p=np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()))
            # Logprob of chosen action:
            logp = float((logits - jax.nn.logsumexp(logits))[a])

            next_obs, r, term, trunc, info = env.step(a)
            done = term or trunc

            act_buf[t]  = a
            logp_buf[t] = logp
            rew_buf[t]  = float(r)
            done_buf[t] = float(done)
            val_buf[t]  = value

            obs = next_obs
            global_steps += 1
            if done:
                obs, info = env.reset()

            if global_steps >= config.total_steps:
                # truncate buffers if needed
                T = t + 1
                obs_buf   = obs_buf[:T]
                act_buf   = act_buf[:T]
                logp_buf  = logp_buf[:T]
                rew_buf   = rew_buf[:T]
                done_buf  = done_buf[:T]
                val_buf   = val_buf[:T]
                break

        # -------- Bootstrap and GAE (JIT + scan) --------
        x_last = obs_to_cropped_onehot(obs, n_actions, crop_size)[None, ...]
        _, last_v = jitted_forward(train_state.params, train_state.apply_fn, jnp.asarray(x_last))
        last_v = float(np.asarray(last_v)[0])

        adv, rets = gae_scan(jnp.asarray(rew_buf), jnp.asarray(val_buf),
                             jnp.asarray(done_buf), jnp.asarray(last_v),
                             config.gamma, config.gae_lambda)
        adv = np.asarray(adv)
        rets = np.asarray(rets)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------- PPO update (JIT), with epoch×minibatch loops --------
        B = T
        idxs = np.arange(B)
        for _ in range(config.update_epochs):
            np.random.shuffle(idxs)
            for s in range(0, B, config.minibatch_size):
                mb = idxs[s:s+config.minibatch_size]
                mb_obs = jnp.asarray(obs_buf[mb])
                mb_act = jnp.asarray(act_buf[mb])
                mb_old = jnp.asarray(logp_buf[mb])
                mb_adv = jnp.asarray(adv[mb])
                mb_ret = jnp.asarray(rets[mb])

                train_state, metrics = ppo_update(
                    train_state,
                    mb_obs, mb_act, mb_old, mb_adv, mb_ret,
                    config.clip_coef, config.vf_coef, config.ent_coef
                )

        # -------- Periodic eval + checkpoint --------
        if (global_steps % config.save_every_steps == 0) or (global_steps >= config.total_steps):
            res = evaluate(env,
                           params=train_state.params,
                           apply_fn=train_state.apply_fn,
                           n_actions=n_actions,
                           crop_size=crop_size,
                           episodes=10,
                           step=global_steps,
                           run_dir=run_dir)
            print(f"[eval] step={global_steps} return={res['return_mean']:.3f} quality={res['quality_mean']:.3f}")

            # Save Flax params checkpoint
            ckpt_dir = ensure_dir(run_dir / "checkpoints")
            checkpoints.save_checkpoint(ckpt_dir, target=train_state.params,
                                        step=global_steps, overwrite=True, keep=3)

        # progress
        if global_steps % 10_000 == 0 or global_steps >= config.total_steps:
            elapsed = time.time() - t0
            print(f"[ppo] step={global_steps}/{config.total_steps} elapsed={elapsed/60:.1f}m")

    # Final eval already handled in the loop if just crossed total_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True, help="pcg_benchmark env name, e.g., zelda-v0")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--crop_size", type=int, default=21, help="Odd window size for cropping around pos")
    # PPO knobs
    ap.add_argument("--total_steps", type=int, default=200_000)
    ap.add_argument("--rollout_length", type=int, default=128)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--minibatch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every_steps", type=int, default=50_000)
    ap.add_argument("--eval_episodes", type=int, default=10)
    args = ap.parse_args()

    cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        save_every_steps=args.save_every_steps,
    )

    run_dir = Path(args.run_dir)
    ensure_dir(run_dir / "metrics")

    train_ppo(env_name=args.env,
              run_dir=run_dir,
              config=cfg,
              crop_size=args.crop_size,
              eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
