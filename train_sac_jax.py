# train_sac_jax.py
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints

import pcg_benchmark
from models.jax_sac import SACConfig, make_sac_states, sac_update


# ---------- Cropped window & one-hot ----------

import numpy as np
from typing import Any, Dict, Tuple

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




# ---------- Files / metrics ----------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)

def evaluate(env, params_bundle, n_actions, crop_size, episodes, step, run_dir: Path):
    actor_params, actor_apply = params_bundle
    returns, lens, qualities, infos_dump = [], [], [], []
    for ep in range(episodes):
        try:
            obs, info = env.reset(seed=step + ep)
        except TypeError:
            if hasattr(env, "seed"): env.seed(step + ep)
            obs, info = env.reset()
        done = False; ep_ret=0.0; ep_len=0
        while not done:
            x = obs_to_cropped_onehot(obs, n_actions, crop_size)[None, ...]
            a, *_ = actor_apply({'params': actor_params}, jnp.asarray(x), rng=jax.random.PRNGKey(step+ep+ep_len), sample=False)
            a = np.asarray(a[0])
            # Map continuous [-1,1] -> discrete action: argmax over channels
            act = int(np.argmax(a))
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            ep_ret += float(r); ep_len += 1
        returns.append(ep_ret); lens.append(ep_len); qualities.append(float(info.get("quality",0.0)))
        infos_dump.append({"quality": float(info.get("quality",0.0)), "step_count": ep_len, "content_info": info.get("content_info", {})})
    res = {
        "episodes": episodes,
        "len_mean": float(np.mean(lens)), "len_std": float(np.std(lens)),
        "quality_mean": float(np.mean(qualities)) if qualities else 0.0,
        "quality_std": float(np.std(qualities)) if qualities else 0.0,
        "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
        "infos": infos_dump
    }
    metrics_dir = ensure_dir(run_dir / "metrics")
    agg_path = metrics_dir / "eval.json"
    agg = {}
    if agg_path.exists():
        try: agg = json.loads(agg_path.read_text())
        except Exception: agg = {}
    agg[f"step_{int(step)}"] = res
    write_json(agg_path, agg)
    ck_dir = ensure_dir(metrics_dir / "checkpoint" / f"ckpt_{int(step):08d}")
    write_json(ck_dir / "eval.json", res)
    return res


# ---------- Replay Buffer (simple, host) ----------

class Replay:
    def __init__(self, capacity, obs_shape, n_actions):
        self.N = int(capacity)
        self.obs = np.zeros((self.N, *obs_shape), np.float32)
        self.next = np.zeros((self.N, *obs_shape), np.float32)
        self.act = np.zeros((self.N, n_actions), np.float32)  # continuous [-1,1]
        self.rew = np.zeros((self.N,), np.float32)
        self.done= np.zeros((self.N,), np.float32)
        self.ptr = 0; self.size = 0
    def add(self, o, a, r, n, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next[self.ptr]= n
        self.done[self.ptr]= d
        self.ptr = (self.ptr + 1) % self.N
        self.size = min(self.size + 1, self.N)
    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=(batch,))
        return dict(
            obs=jnp.asarray(self.obs[idx]),
            act=jnp.asarray(self.act[idx]),
            rew=jnp.asarray(self.rew[idx]),
            obs_next=jnp.asarray(self.next[idx]),
            done=jnp.asarray(self.done[idx]),
        )


# ---------- Train ----------

def train_sac(env_name: str,
              run_dir: Path,
              crop_size: int = 21,
              total_steps: int = 200_000,
              warmup: int = 5_000,
              batch_size: int = 256,
              save_every: int = 50_000,
              eval_every: int = 50_000,
              seed: int = 0):
    env = pcg_benchmark.make(env_name)
    H, W, n_actions = env.observation_space
    obs_shape = (n_actions, crop_size, crop_size)
    cfg = SACConfig(batch_size=batch_size, seed=seed)

    rng = jax.random.PRNGKey(seed)
    (rng,
     actor_state, critic1_state, critic2_state,
     t1_params, t2_params,
     log_alpha, alpha_opt_state, target_entropy, alpha_opt) = make_sac_states(rng, obs_shape, n_actions, cfg)

    # Simple replay (host)
    replay = Replay(capacity=200_000, obs_shape=obs_shape, n_actions=n_actions)

    # Reset env
    obs, info = env.reset()
    cur = obs_to_cropped_onehot(obs, n_actions, crop_size)
    global_steps = 0
    ensure_dir(run_dir / "metrics"); ensure_dir(run_dir / "checkpoints")

    while global_steps < total_steps:
        # Action (sample from actor; map to discrete to step env)
        a_cont, *_ = actor_state.apply_fn({'params': actor_state.params},
                                          jnp.asarray(cur[None, ...]),
                                          rng=jax.random.PRNGKey(global_steps),
                                          sample=True)
        a_cont = np.asarray(a_cont[0])
        a_disc = int(np.argmax(a_cont))
        next_obs, r, term, trunc, info = env.step(a_disc)
        done = term or trunc
        nxt = obs_to_cropped_onehot(next_obs, n_actions, crop_size)

        replay.add(cur, a_cont, float(r), nxt, float(done))

        cur = nxt if not done else obs_to_cropped_onehot(env.reset()[0], n_actions, crop_size)
        global_steps += 1

        # Learn
        if replay.size >= max(batch_size, warmup):
            batch = replay.sample(batch_size)
            (rng,
             actor_state, critic1_state, critic2_state,
             t1_params, t2_params,
             log_alpha, alpha_opt_state, metrics) = sac_update(
                rng, actor_state, critic1_state, critic2_state,
                t1_params, t2_params,
                log_alpha, alpha_opt_state, target_entropy, alpha_opt,
                batch, cfg
            )

        # Eval + ckpt
        if (global_steps % eval_every == 0) or (global_steps >= total_steps):
            res = evaluate(env,
                           params_bundle=(actor_state.params, actor_state.apply_fn),
                           n_actions=n_actions, crop_size=crop_size,
                           episodes=10, step=global_steps, run_dir=run_dir)
            print(f"[SAC] step={global_steps} return={res['return_mean']:.3f} quality={res['quality_mean']:.3f}")

        if (global_steps % save_every == 0) or (global_steps >= total_steps):
            checkpoints.save_checkpoint(run_dir / "checkpoints", target={
                'actor': actor_state.params,
                'critic1': critic1_state.params,
                'critic2': critic2_state.params,
                'tcritic1': t1_params,
                'tcritic2': t2_params,
                'log_alpha': log_alpha
            }, step=global_steps, overwrite=True, keep=3)
            print(f"[SAC] saved checkpoint @ step {global_steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--crop_size", type=int, default=21)
    ap.add_argument("--total_steps", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=50000)
    ap.add_argument("--eval_every", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_sac(args.env, Path(args.run_dir),
              crop_size=args.crop_size,
              total_steps=args.total_steps,
              warmup=args.warmup,
              batch_size=args.batch_size,
              save_every=args.save_every,
              eval_every=args.eval_every,
              seed=args.seed)

if __name__ == "__main__":
    main()
