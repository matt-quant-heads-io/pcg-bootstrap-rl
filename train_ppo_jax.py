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
from rl_agents.jax_ppo import (
    PPOConfig,
    make_ppo_train_state,
    ppo_update,
    gae_scan,
)

# -------------------- JIT forward (apply_fn static) --------------------

def forward_logits_value(params, x, *, apply_fn):
    """Pure function for JIT; apply_fn is static, params/x are traced."""
    return apply_fn({'params': params}, x)

# JIT with apply_fn as static kwarg
JIT_FORWARD = jax.jit(forward_logits_value, static_argnames=('apply_fn',))


# -------------------- Cropped window preprocessing --------------------

def _normalize_map_2d(arr: np.ndarray) -> np.ndarray:
    """
    Accepts map shapes: (H,W), (1,H,W), (H,W,1), and returns (H,W).
    Raises if shape is otherwise incompatible.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # (1,H,W) -> (H,W)
        if a.shape[0] == 1 and a.shape[1] >= 1 and a.shape[2] >= 1:
            return a[0]
        # (H,W,1) -> (H,W)
        if a.shape[2] == 1 and a.shape[0] >= 1 and a.shape[1] >= 1:
            return a[:, :, 0]
    raise ValueError(f"Expected map of shape (H,W) or (1,H,W) or (H,W,1); got {a.shape}")

def crop_window(map_np: np.ndarray, pos: Tuple[int, int], size: int, pad_value: int) -> np.ndarray:
    """
    Centered crop with explicit center indexing on a padded map.
    Always returns (size, size).
    """
    assert size % 2 == 1, "crop_size must be odd"
    y, x = int(pos[0]), int(pos[1])
    pad = size // 2

    arr2d = _normalize_map_2d(map_np)
    arr2d = np.asarray(arr2d)

    # pad with the provided tile id
    padded = np.pad(arr2d, ((pad, pad), (pad, pad)), mode="constant", constant_values=int(pad_value))

    cy, cx = y + pad, x + pad
    y0, y1 = cy - pad, cy + pad + 1
    x0, x1 = cx - pad, cx + pad + 1
    tile = padded[y0:y1, x0:x1]

    # hard guarantee
    if tile.shape != (size, size):
        fix = np.full((size, size), int(pad_value), dtype=arr2d.dtype)
        ys = slice(max(0, y0), min(y1, padded.shape[0]))
        xs = slice(max(0, x0), min(x1, padded.shape[1]))
        fy0 = max(0, - (y0 - 0)); fx0 = max(0, - (x0 - 0))
        fy1 = fy0 + (ys.stop - ys.start); fx1 = fx0 + (xs.stop - xs.start)
        fix[fy0:fy1, fx0:fx1] = padded[ys, xs]
        tile = fix
    return tile

def obs_to_cropped_onehot(obs: Dict[str, Any], n_channels: int, crop_size: int, pad_value: int = 1) -> np.ndarray:
    """
    Returns CHW float32 one-hot crop.
    - Accepts map in (H,W), (1,H,W), or (H,W,1).
    - Clamps tile ids into [0, n_channels-1].
    """
    tile = crop_window(obs["map"], obs["pos"], crop_size, pad_value=pad_value)
    tile = tile.astype(np.int64)
    tile = np.clip(tile, 0, n_channels - 1)

    oh = np.eye(n_channels, dtype=np.float32)[tile.reshape(-1)]
    oh = oh.reshape(crop_size, crop_size, n_channels)   # HWC
    chw = np.transpose(oh, (2, 0, 1))                   # CHW
    return chw


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
            x_b = jnp.asarray(x, dtype=jnp.float32)
            logits, _ = JIT_FORWARD(params, x_b, apply_fn=apply_fn)
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
    # Observation shape from env is unused here; we build CHW from crop/onehot.
    # H, W = env.observation_space.shape
    n_actions = env.action_space  # your wrapper exposes an int

    print(f"crop_size: {crop_size}")
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

    while global_steps < config.total_steps:
        # -------- Rollout (env stays in Python) --------
        for t in range(T):
            x = obs_to_cropped_onehot(obs, n_actions, crop_size)  # (C,H,W)
            obs_buf[t] = x

            x_b = jnp.asarray(x[None, ...], dtype=jnp.float32)  # (1,C,H,W)
            logits, value = JIT_FORWARD(train_state.params, x_b, apply_fn=train_state.apply_fn)

            logits = np.asarray(logits[0])
            value  = float(np.asarray(value)[0])

            # Sample action on host for simplicity (numerically stable softmax)
            logits_s = logits - logits.max()
            probs = np.exp(logits_s) / np.exp(logits_s).sum()
            a = int(np.random.default_rng().choice(n_actions, p=probs))
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
        x_last_b = jnp.asarray(x_last, dtype=jnp.float32)
        _, last_v = JIT_FORWARD(train_state.params, x_last_b, apply_fn=train_state.apply_fn)
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
                mb_obs = jnp.asarray(obs_buf[mb], dtype=jnp.float32)
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
                           episodes=eval_episodes,
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
    ap.add_argument("--crop_size", type=int, default=22, help="Odd window size for cropping around pos")
    # PPO knobs
    ap.add_argument("--total_steps", type=int, default=200_000)
    ap.add_argument("--rollout_length", type=int, default=2048)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--minibatch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every_steps", type=int, default=50_000)
    ap.add_argument("--eval_episodes", type=int, default=50)
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
