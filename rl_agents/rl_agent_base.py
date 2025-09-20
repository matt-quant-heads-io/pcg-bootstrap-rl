from typing import Optional, Dict, Any
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
from einops import rearrange
from gymnasium import spaces
from PIL import Image  # needed for saving renders
import matplotlib.pyplot as plt

from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import copy

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

from .hooks import _HOOK_LIBRARY


def to_json_safe(obj):
    """
    Recursively convert an object into JSON-serializable form:
      - numpy scalars → Python scalars
      - numpy arrays → lists
      - tuples → lists
      - dicts → recurse on values
      - lists → recurse on elements
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):  # catches np.float32, np.int64, etc.
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


Hook = Callable[["RLAgentBase", Dict[str, Any]], None]

@dataclass
class HookSpec:
    name: str
    fn: Hook
    enabled: bool = True
    order: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class RLAgentBase:
    def __init__(self, env, algo_config: Dict[str, Any],
                 map_pad_fill: Optional[int] = 0,
                 crop_size: Optional[int] = None):
        self.env = env
        self.algo_config = algo_config
        self.map_pad_fill = map_pad_fill
        self._obs_space = getattr(env, "observation_space", None)
        self._act_space = getattr(env, "action_space", None)
        self._pretrain_hooks: List[HookSpec] = []

        # infer number of tile classes
        if hasattr(env, "prob_config") and isinstance(env.prob_config, dict) and "tiles" in env.prob_config:
            self.num_classes = int(len(env.prob_config["tiles"]))
        elif isinstance(self._obs_space, spaces.Box):
            self.num_classes = int(np.max(self._obs_space.high)) + 1
        else:
            self.num_classes = int(algo_config.get("num_tile_classes", 6))

        self.crop_size = crop_size
        self.run_dir = getattr(self, "run_dir", getattr(algo_config, "run_dir", None))

        # --- NEW: checkpoint + interval-metrics state ---
        self.checkpoint_interval = int(self.algo_config.get("checkpoint_interval", 0))
        self._interval_contents = []      # list of final maps (HxW) within the current interval
        self._last_ckpt_step = 0

    # === Pretrain hooks API ===
    def register_pretrain_hook(self, name: str, fn: Hook, *, enabled: bool=True, order: int=0, config: Optional[Dict[str,Any]]=None):
        self._pretrain_hooks.append(HookSpec(name=name, fn=fn, enabled=enabled, order=order, config=config or {}))

    def run_pretrain(self):
        # Allows YAML-driven config: algo_config["pretrain"] can define hooks and params
        cfg = self.algo_config.get("pretrain", {})
        # Merge YAML-declared hooks into registry if any
        for h in cfg.get("hooks", []):
            # h: {"name": "...", "type": "replay_warmup"/"behavior_cloning"/..., "enabled": true, "order": 10, "config": {...}}
            hook_type = h.get("type")
            spec = HookSpec(
                name=h.get("name", hook_type),
                fn=_HOOK_LIBRARY[hook_type],
                enabled=h.get("enabled", True),
                order=h.get("order", 0),
                config=h.get("config", {}),
            )
            self._pretrain_hooks.append(spec)

        self._pretrain_hooks.sort(key=lambda s: s.order)
        ctx = {"agent": self, "now": time.time()}
        for spec in self._pretrain_hooks:
            if not spec.enabled: 
                continue
            t0 = time.time()
            try:
                local_ctx = {**ctx, **spec.config}
                print(f"[pretrain] start: {spec.name}")
                spec.fn(self, local_ctx)
                print(f"[pretrain] done:  {spec.name} ({time.time()-t0:.2f}s)")
            except Exception as e:
                print(f"[pretrain] ERROR in {spec.name}: {e}")
                if cfg.get("fail_fast", True):
                    raise

    @torch.no_grad()
    def preprocess_observation(self, obs):
        assert isinstance(obs, dict) and "map" in obs and "pos" in obs, "obs must have 'map' and 'pos'"
        y, x = obs["pos"]
        map_np = np.asarray(obs["map"], dtype=np.float32)  # (1,H,W)
        assert map_np.ndim == 3 and map_np.shape[0] == 1, f"Expected (1,H,W), got {map_np.shape}"
        _, H, W = map_np.shape

        # NEW: full-position context — model sees a 2*max(H,W) crop centered at (y,x)
        crop_ref = max(H, W)
        self.crop_size = crop_ref * 2
        half = self.crop_size // 2

        # pad, then crop centered at current pos
        padded = np.pad(map_np[0], ((half, half), (half, half)),
                        mode="constant", constant_values=self.map_pad_fill)
        top, left = int(y), int(x)
        cropped = padded[top: top + self.crop_size, left: left + self.crop_size]

        # edge guard (should rarely trigger, but safe)
        if cropped.shape != (self.crop_size, self.crop_size):
            cropped = np.pad(
                cropped,
                ((0, max(0, self.crop_size - cropped.shape[0])) ,
                (0, max(0, self.crop_size - cropped.shape[1]))),
                mode="constant",
                constant_values=self.map_pad_fill,
            )
            cropped = cropped[:self.crop_size, :self.crop_size]

        # one-hot [1, C, H, W]
        cropped_long = torch.as_tensor(cropped, dtype=torch.long).clamp(0, self.num_classes - 1)
        state = F.one_hot(cropped_long, num_classes=self.num_classes).float()
        state = rearrange(state, "h w c -> 1 c h w")
        return state


    # -------- NEW helpers for interval metrics & checkpoints --------
    def _append_final_content(self, obs) -> None:
        arr = np.asarray(obs["map"][0])
        arr = np.rint(arr).astype(np.int64)
        arr = np.clip(arr, 0, self.num_classes - 1)
        self._interval_contents.append(arr)

    def _compute_interval_qd(self):
        if not self._interval_contents:
            return None
        contents = self._interval_contents
        try:
            q_score, d_score, _, details, _ = self.env.evaluate(contents, controls=None)
            q_vals = details.get("quality", [])
            d_vals = details.get("diversity", [])
            if len(q_vals) == 0 or len(d_vals) == 0:
                return None
            return float(np.mean(q_vals)), float(np.mean(d_vals))
        except Exception:
            return None

    def _save_checkpoint(self, step: int, payload: dict) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        ckpt_dir = os.path.join(run_dir_abs, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_{step:08d}.pt")
        torch.save(payload, path)
        return path

    def maybe_checkpoint(self, step: int, payload: dict):
        if self.checkpoint_interval <= 0:
            return None
        if step > 0 and (step - self._last_ckpt_step) >= self.checkpoint_interval:
            qd = self._compute_interval_qd()
            ckpt_path = self._save_checkpoint(step, payload)
            self._last_ckpt_step = step
            self._interval_contents.clear()
            out = {
                "checkpoint_path": ckpt_path,
                "ckpt_step": int(step),
            }
            if qd is not None:
                q_mean, d_mean = qd
                out["interval_quality_mean"] = q_mean
                out["interval_diversity_mean"] = d_mean
            return out
        return None
    # ----------------------------------------------------------------

    @torch.no_grad()
    def act_eval(self, proc_state: torch.Tensor) -> int:
        """Agents must override this: return greedy action for eval."""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Run episodes and save final renders to <run_dir>/levels/ (absolute, safe-join)."""
        results = {"returns": [], "steps": [], "qualities": []}
        if episodes is None:
            episodes = int(self.algo_config.get("eval_episodes", 10))

        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else None
        levels_dir = None
        if run_dir_abs:
            levels_dir = os.path.join(run_dir_abs, "levels")
            os.makedirs(levels_dir, exist_ok=True)

        def _save_in_levels(filename: str) -> str:
            assert levels_dir is not None
            out_path = os.path.abspath(os.path.join(levels_dir, filename))
            levels_abs = os.path.abspath(levels_dir) + os.sep
            if not out_path.startswith(levels_abs):
                out_path = os.path.abspath(os.path.join(levels_dir, os.path.basename(filename)))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            return out_path

        for ep in range(episodes):
            obs, info = self.env.reset()
            done = False
            ep_ret, ep_steps = 0.0, 0

            while not done:
                state = self.preprocess_observation(obs)
                action = self.act_eval(state)
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                done = terminated or truncated
                ep_ret += float(reward)
                ep_steps += 1

            results["returns"].append(ep_ret)
            results["steps"].append(ep_steps)
            if isinstance(info, dict) and "quality" in info:
                results["qualities"].append(float(info["quality"]))

            # record final content into the current interval bucket
            try:
                self._append_final_content(obs)
            except Exception as e:
                results.setdefault("render_errors", []).append(f"interval_collect_failed: {e}")

            # render final map
            try:
                final_content = np.asarray(obs["map"][0])
                final_content = np.rint(final_content).astype(np.int64)
                final_content = np.clip(final_content, 0, self.num_classes - 1)
                render_result = self.env.render(final_content)

                if levels_dir is not None:
                    out_path = _save_in_levels(f"episode_{ep+1:03d}.png")
                    if isinstance(render_result, Image.Image):
                        render_result.save(out_path)
                    else:
                        try:
                            Image.fromarray(render_result).save(out_path)
                        except Exception:
                            arr = final_content.astype(np.uint8)
                            rng = float(arr.max() - arr.min())
                            if rng > 0:
                                arr = ((arr - arr.min()) * (255.0 / rng)).astype(np.uint8)
                            Image.fromarray(arr).save(out_path)
                    results.setdefault("render_paths", []).append(out_path)
            except Exception as e:
                results.setdefault("render_errors", []).append(str(e))

        try:
            self._plot_metrics_by_checkpoint()
        except Exception as e:
            print(f"Error in evaluate: {e}")

        return results

    def _save_eval_levels(self, step: int, contents_list):
        """
        Save level images for a given checkpoint step under:
        <run_dir>/levels/ckpt_<step>/level_XXXX.png
        `contents_list` is a list of final contents (e.g., ndarray or whatever env.render expects).
        """
        out_dir = self._run_dir() / "levels" / f"ckpt_{int(step):08d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, content in enumerate(contents_list, start=1):
            try:
                rendered = self.env.render(content)
                # Accept PIL.Image or ndarray
                if isinstance(rendered, Image.Image):
                    img = rendered
                else:
                    arr = np.asarray(rendered)
                    if arr.dtype != np.uint8:
                        # simple normalization to uint8
                        mn, mx = float(arr.min()), float(arr.max())
                        scale = 255.0 / (mx - mn + 1e-9)
                        arr = ((arr - mn) * scale).clip(0, 255).astype(np.uint8)
                    if arr.ndim == 2:
                        img = Image.fromarray(arr, mode="L")
                    else:
                        img = Image.fromarray(arr)
                img.save(out_dir / f"level_{i:04d}.png")
            except Exception as e:
                print(f"[warn] _save_eval_levels: failed for item {i} @ step {step}: {e}")

        print(f"[levels] wrote {len(contents_list)} image(s) to {out_dir}")


    def _metrics_dir(self) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        mdir = os.path.join(run_dir_abs, "metrics")
        os.makedirs(mdir, exist_ok=True)
        return mdir

    def _plots_dir(self) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        pdir = os.path.join(run_dir_abs, "plots")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def _metrics_dir(self) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        mdir = os.path.join(run_dir_abs, "metrics")
        os.makedirs(mdir, exist_ok=True)
        return mdir

    def _eval_json_path(self) -> str:
        return os.path.join(self._metrics_dir(), "eval.json")

    def _write_eval_json(self, step: int, eval_res: dict, extra: dict | None = None):
        """
        Append/overwrite aggregate metrics in metrics/eval.json and also write a
        per-checkpoint snapshot to metrics/checkpoint/ckpt_<step>/eval.json
        """
        metrics_dir = self._metrics_dir()
        agg_path = metrics_dir / "eval.json"
        data = {}
        if agg_path.exists():
            try:
                with open(agg_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        key = f"step_{int(step):08d}"
        row = dict(eval_res)
        if extra and "checkpoint_path" in extra:
            row["checkpoint_path"] = extra["checkpoint_path"]
        data[key] = row

        with open(agg_path, "w") as f:
            json.dump(data, f, indent=2)

        # per-ckpt snapshot
        ck_dir = self._ckpt_dir(metrics_dir / "checkpoint", int(step))
        with open(ck_dir / "eval.json", "w") as f:
            json.dump(row, f, indent=2)
        if extra:
            with open(ck_dir / "manifest.json", "w") as f:
                json.dump({"step": int(step), **extra}, f, indent=2)

    def _plot_metrics_by_checkpoint(self, step: int | None = None):
        """Render plots into plots/ and mirror into plots/checkpoint/ckpt_<step>/ if step given."""
        eval_path = self._eval_json_path()
        if not os.path.exists(eval_path):
            print(f"Eval path: {eval_path} does not exist!")
            return
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            items = sorted(((int(k.split("_")[-1]), v) for k, v in data.items()), key=lambda kv: kv[0])
            if not items:
                return
            steps = [k for k, _ in items]
            mean_returns   = [v.get("return_mean") for _, v in items]
            mean_qualities = [v.get("quality_mean") for _, v in items]
            mean_steps     = [v.get("len_mean") for _, v in items]
            interval_q     = [v.get("interval_quality_mean") for _, v in items]
            interval_d     = [v.get("interval_diversity_mean") for _, v in items]
            have_iqd = any((q is not None) or (d is not None) for q, d in zip(interval_q, interval_d))

            import matplotlib.pyplot as plt
            plots_dir = self._plots_dir()
            ck_plots_dir = self._ckpt_dir(plots_dir / "checkpoint", int(step)) if step is not None else None

            def _save(fig, filename):
                fig.savefig(plots_dir / filename, bbox_inches="tight")
                if ck_plots_dir is not None:
                    fig.savefig(ck_plots_dir / filename, bbox_inches="tight")

            def _plot(y, ylabel, title, fname):
                fig = plt.figure()
                plt.plot(steps, y, marker="o")
                plt.xlabel("Checkpoint step"); plt.ylabel(ylabel); plt.title(title)
                _save(fig, fname); plt.close(fig)

            _plot(mean_returns,   "Mean return (eval)",  "Mean Return vs Checkpoint",   "return_vs_checkpoint.png")
            _plot(mean_qualities, "Mean quality (eval)", "Mean Quality vs Checkpoint",  "quality_vs_checkpoint.png")
            _plot(mean_steps,     "Mean steps/episode",  "Mean Episode Length vs Checkpoint", "steps_vs_checkpoint.png")

            if have_iqd:
                fig = plt.figure()
                if any(v is not None for v in interval_q):
                    plt.plot(steps, interval_q, marker="o", label="Interval mean quality")
                if any(v is not None for v in interval_d):
                    plt.plot(steps, interval_d, marker="s", label="Interval mean diversity")
                plt.xlabel("Checkpoint step"); plt.ylabel("Interval metric")
                plt.title("Interval Quality/Diversity vs Checkpoint"); plt.legend()
                _save(fig, "interval_qd_vs_checkpoint.png")
                plt.close(fig)
        except Exception as e:
            print(f"Error in _plot_metrics_by_checkpoint: {e}")

    @torch.no_grad()
    def _evaluate_for_checkpoint(self, eval_episodes: int = 10, save_images: bool = False, step: int = 0):
        returns = []
        ep_lens = []
        qualities = []
        finals = []  # <--- collect final contents per episode
        last_content = None
        infos = []
        base_seed = int(step) if isinstance(step, (int, np.integer)) else 0
        for ep in range(eval_episodes):
            obs, info = self.env.reset(seed=base_seed + ep)
            done = False
            ep_ret, ep_len = 0.0, 0
            last_content = None

            while not done:
                state = self.preprocess_observation(obs)
                a = self.act_eval(state)
                
                obs, r, term, trunc, info = self.env.step(int(a))
                # import pdb; pdb.set_trace()
                # print(f"")
                done = term or trunc
                ep_ret += float(r)
                ep_len += 1
                # last_content = self.env._current_content  # final content
                last_content = np.array(self.env._current_content, copy=True)

            # quality from info at episode end (or recompute)
            q = float(info.get("quality", 0.0))
            returns.append(ep_ret)
            ep_lens.append(ep_len)
            qualities.append(q)
            infos.append(to_json_safe(info))
            finals.append(last_content)
                

        result = {
            "episodes": eval_episodes,
            "len_mean":    float(np.mean(ep_lens)),
            "len_std":     float(np.std(ep_lens)),
            "quality_mean":float(np.mean(qualities)),
            "quality_std": float(np.std(qualities)),
            "infos": infos
        }

        if save_images and finals:
            self._save_eval_levels(step, finals)

        return result

    # --- run/dir helpers ---
    def _run_dir(self) -> Path:
        return Path(self.run_dir)

    def _metrics_dir(self) -> Path:
        d = self._run_dir() / "metrics"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _plots_dir(self) -> Path:
        d = self._run_dir() / "plots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ckpt_dir(self, base: Path, step: int) -> Path:
        ck = base / "checkpoint" / f"ckpt_{int(step):08d}"
        ck.mkdir(parents=True, exist_ok=True)
        return ck

    def _eval_json_path(self) -> str:
        return str(self._metrics_dir() / "eval.json")




    def _write_eval_json(self, step: int, eval_res: dict, extra: dict | None = None):
        """
        Append/update evaluation results in:
        {run_dir}/metrics/eval.json
        Structure:
        {
            "step_12345": { ...metrics... },
            "step_25000": { ...metrics... },
            ...
        }
        """
        metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        eval_path = os.path.join(metrics_dir, "eval.json")

        # load existing if present
        data = {}
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    data = json.load(f)
            except Exception:
                # if file is corrupted, start fresh
                data = {}

        key = f"step_{int(step)}"
        payload = dict(eval_res)
        if extra:
            payload.update({k: v for k, v in extra.items() if k.endswith("_path") or "checkpoint" in k or "algo" in k})

        data[key] = payload

        with open(eval_path, "w") as f:
            json.dump(data, f, indent=2)

    def _plot_metrics_by_checkpoint(self):
        """Render plots from metrics/eval.json into metrics/plots/. Non-fatal on errors."""
        import os, json, math

        eval_path = self._eval_json_path()
        if not os.path.exists(eval_path):
            print(f"Eval path: {eval_path} does not exist!")
            return
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            # items like: [ (500, {...}), (1000, {...}), ... ]
            items = sorted(
                ((int(k.split("_")[-1]), v) for k, v in data.items()),
                key=lambda kv: kv[0]
            )
            if not items:
                return

            steps = [k for k, _ in items]

            # Means
            mean_returns   = [v.get("return_mean")        for _, v in items]
            mean_qualities = [v.get("quality_mean")       for _, v in items]
            mean_lengths   = [v.get("len_mean")           for _, v in items]

            # Stds (optional)
            std_returns    = [v.get("return_std")         for _, v in items]
            std_qualities  = [v.get("quality_std")        for _, v in items]
            std_lengths    = [v.get("len_std")            for _, v in items]

            # Optional interval series
            interval_q     = [v.get("interval_quality_mean")  for _, v in items]
            interval_d     = [v.get("interval_diversity_mean") for _, v in items]
            interval_q_std = [v.get("interval_quality_std")   for _, v in items]
            interval_d_std = [v.get("interval_diversity_std") for _, v in items]
            have_iqd = any((q is not None) or (d is not None) for q, d in zip(interval_q, interval_d))

            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"Matplotlib not available: {e}")
                return  # plotting not available; silently skip

            plots_dir = self._plots_dir()
            os.makedirs(plots_dir, exist_ok=True)

            def _as_float_array(x):
                out = []
                for v in x:
                    try:
                        if v is None:
                            out.append(np.nan)
                        else:
                            out.append(float(v))
                    except Exception:
                        out.append(np.nan)
                return np.asarray(out, dtype=np.float32)

            def _plot_with_band(x, y_mean, y_std, ylabel, title, fname):
                x = np.asarray(x, dtype=np.int64)
                y = _as_float_array(y_mean)
                s = _as_float_array(y_std) if y_std is not None else None

                # mask out NaNs to avoid empty plots
                if s is not None:
                    mask = ~np.isnan(y) & ~np.isnan(s)
                else:
                    mask = ~np.isnan(y)

                x_m, y_m = x[mask], y[mask]
                s_m = s[mask] if s is not None else None

                fig = plt.figure()
                if x_m.size > 0:
                    plt.plot(x_m, y_m, marker="o", linewidth=1.5)
                    if s_m is not None and np.any(s_m > 0):
                        y_lo = y_m - s_m
                        y_hi = y_m + s_m
                        plt.fill_between(x_m, y_lo, y_hi, alpha=0.2, linewidth=0)
                else:
                    # draw axes so file isn't empty even with no data
                    pass

                plt.xlabel("Checkpoint step")
                plt.ylabel(ylabel)
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
                plt.close(fig)

            # Core plots with error bands
            _plot_with_band(
                steps, mean_returns, std_returns,
                ylabel="Mean return (eval)",
                title="Mean Return vs Checkpoint",
                fname="return_vs_checkpoint.png"
            )
            _plot_with_band(
                steps, mean_qualities, std_qualities,
                ylabel="Mean quality (eval)",
                title="Mean Quality vs Checkpoint",
                fname="quality_vs_checkpoint.png"
            )
            _plot_with_band(
                steps, mean_lengths, std_lengths,
                ylabel="Mean steps/episode (eval)",
                title="Mean Episode Length vs Checkpoint",
                fname="steps_vs_checkpoint.png"
            )

            # Interval Q/D (with optional bands if stds provided)
            if have_iqd:
                x = np.asarray(steps, dtype=np.int64)
                iq = _as_float_array(interval_q)
                idv = _as_float_array(interval_d)
                iq_s = _as_float_array(interval_q_std) if any(v is not None for v in interval_q_std) else None
                id_s = _as_float_array(interval_d_std) if any(v is not None for v in interval_d_std) else None

                fig = plt.figure()

                def _plot_series(x, y, s, label, marker):
                    mask = ~np.isnan(y) & (~np.isnan(s) if s is not None else True)
                    if not np.any(mask):
                        return
                    xm, ym = x[mask], y[mask]
                    plt.plot(xm, ym, marker=marker, label=label, linewidth=1.5)
                    if s is not None:
                        sm = s[mask]
                        if np.any(sm > 0):
                            plt.fill_between(xm, ym - sm, ym + sm, alpha=0.2, linewidth=0)

                _plot_series(x, iq, iq_s, "Interval mean quality", "o")
                _plot_series(x, idv, id_s, "Interval mean diversity", "s")

                plt.xlabel("Checkpoint step")
                plt.ylabel("Interval metric")
                plt.title("Interval Quality/Diversity vs Checkpoint")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, "interval_qd_vs_checkpoint.png"), bbox_inches="tight")
                plt.close(fig)

        except Exception as e:
            print(f"Error in _plot_metrics_by_checkpoint: {e}")


    @torch.no_grad()
    def env_step_safe(self, action):
        """
        Call env.step(int(action)) and always return a Gymnasium-style 5-tuple:
        (obs, reward, terminated, truncated, info).
        Handles None returns and legacy 4-tuple envs.
        """
        out = self.env.step(int(action))

        # Env swallowed an error and returned None → treat as terminal and reset
        if out is None:
            obs, info = self.env.reset()
            return obs, 0.0, True, False, info

        # Some envs might still return the legacy 4-tuple (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
            return obs, float(reward), terminated, truncated, info

        # Normal Gymnasium 5-tuple
        return out