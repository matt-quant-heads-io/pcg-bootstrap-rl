from typing import Optional, Dict, Any
import os
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gymnasium import spaces
from PIL import Image  # needed for saving renders

class RLAgentBase:
    def __init__(self, env, algo_config: Dict[str, Any],
                 map_pad_fill: Optional[int] = 0,
                 crop_size: Optional[int] = None):
        self.env = env
        self.algo_config = algo_config
        self.map_pad_fill = map_pad_fill
        self._obs_space = getattr(env, "observation_space", None)
        self._act_space = getattr(env, "action_space", None)

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
        # ------------------------------------------------

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
        import torch
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
        except Exception:
            pass

        return results

    def _metrics_dir(self) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        mdir = os.path.join(run_dir_abs, "metrics")
        os.makedirs(mdir, exist_ok=True)
        return mdir

    def _plots_dir(self) -> str:
        pdir = os.path.join(self._metrics_dir(), "plots")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def _plot_metrics_by_checkpoint(self):
        """
        Load metrics/eval.json and plot summary metrics vs. checkpoint step into metrics/plots/.
        Non-fatal if file is missing or malformed.
        """
        import json
        import matplotlib.pyplot as plt

        eval_path = os.path.join(self._metrics_dir(), "eval.json")
        if not os.path.exists(eval_path):
            return  # nothing to plot yet

        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
        except Exception:
            return

        # Expect keys like "50000", values: {"results": {...}, "summary": {...}, ...}
        try:
            # Sort by numeric step
            items = sorted(((int(k), v) for k, v in data.items()), key=lambda kv: kv[0])
        except Exception:
            return

        if not items:
            return

        # Collect series
        steps = [k for k, _ in items]
        mean_returns = []
        mean_qualities = []
        mean_steps = []
        interval_q = []
        interval_d = []
        have_iqd = False

        for _, v in items:
            summ = v.get("summary", {})
            mean_returns.append(summ.get("mean_return", None))
            mean_qualities.append(summ.get("mean_quality", None))
            mean_steps.append(summ.get("mean_steps", None))
            iq = v.get("interval_quality_mean", None)
            idv = v.get("interval_diversity_mean", None)
            have_iqd = have_iqd or (iq is not None or idv is not None)
            interval_q.append(iq)
            interval_d.append(idv)

        plots_dir = self._plots_dir()

        # Return vs checkpoint
        fig = plt.figure()
        plt.plot(steps, mean_returns, marker="o")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Mean return (eval)")
        plt.title("Mean Return vs Checkpoint")
        out1 = os.path.join(plots_dir, "return_vs_checkpoint.png")
        fig.savefig(out1, bbox_inches="tight")
        plt.close(fig)

        # Quality vs checkpoint
        fig = plt.figure()
        plt.plot(steps, mean_qualities, marker="o")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Mean quality (eval)")
        plt.title("Mean Quality vs Checkpoint")
        out2 = os.path.join(plots_dir, "quality_vs_checkpoint.png")
        fig.savefig(out2, bbox_inches="tight")
        plt.close(fig)

        # Steps-per-episode vs checkpoint
        fig = plt.figure()
        plt.plot(steps, mean_steps, marker="o")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Mean steps per episode (eval)")
        plt.title("Mean Episode Length vs Checkpoint")
        out3 = os.path.join(plots_dir, "steps_vs_checkpoint.png")
        fig.savefig(out3, bbox_inches="tight")
        plt.close(fig)

        # Interval Q/D (from training buckets), if available
        if have_iqd:
            fig = plt.figure()
            # Plot what we have; missing values will show gaps
            plt.plot(steps, interval_q, marker="o", label="Interval mean quality")
            plt.plot(steps, interval_d, marker="s", label="Interval mean diversity")
            plt.xlabel("Checkpoint step")
            plt.ylabel("Interval metric")
            plt.title("Interval Quality/Diversity vs Checkpoint")
            plt.legend()
            out4 = os.path.join(plots_dir, "interval_qd_vs_checkpoint.png")
            fig.savefig(out4, bbox_inches="tight")
            plt.close(fig)

    def _metrics_dir(self) -> str:
        run_dir_abs = os.path.abspath(self.run_dir) if getattr(self, "run_dir", None) else "."
        mdir = os.path.join(run_dir_abs, "metrics")
        os.makedirs(mdir, exist_ok=True)
        return mdir

    def _eval_json_path(self) -> str:
        return os.path.join(self._metrics_dir(), "eval.json")

    def _write_eval_json(self, step: int, results: Dict[str, Any], extra: Dict[str, Any] | None = None):
        try:
            path = self._eval_json_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
            else:
                data = {}
            key = str(step)
            returns = results.get("returns", [])
            qualities = results.get("qualities", [])
            steps_arr = results.get("steps", [])
            summary = {
                "mean_return": float(np.mean(returns)) if len(returns) else None,
                "mean_quality": float(np.mean(qualities)) if len(qualities) else None,
                "mean_steps": float(np.mean(steps_arr)) if len(steps_arr) else None,
            }
            payload = {"results": results, "summary": summary}
            if extra:
                payload.update(extra)
            data[key] = payload
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # non-fatal

    @torch.no_grad()
    def _evaluate_for_checkpoint(self, episodes: int):
        """Run evaluation without polluting the training interval bucket."""
        save_bucket = list(self._interval_contents)
        try:
            self._interval_contents = []
            return self.evaluate(episodes=episodes)
        finally:
            self._interval_contents = save_bucket

    def _plots_dir(self) -> str:
        p = os.path.join(self._metrics_dir(), "plots")
        os.makedirs(p, exist_ok=True)
        return p

    def _plot_metrics_by_checkpoint(self):
        """Render plots from metrics/eval.json into metrics/plots/. Non-fatal on errors."""
        eval_path = self._eval_json_path()
        if not os.path.exists(eval_path):
            return
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            items = sorted(((int(k), v) for k, v in data.items()), key=lambda kv: kv[0])
            if not items:
                return
            steps = [k for k, _ in items]
            mean_returns = [v.get("summary", {}).get("mean_return") for _, v in items]
            mean_qualities = [v.get("summary", {}).get("mean_quality") for _, v in items]
            mean_steps = [v.get("summary", {}).get("mean_steps") for _, v in items]
            interval_q = [v.get("interval_quality_mean") for _, v in items]
            interval_d = [v.get("interval_diversity_mean") for _, v in items]
            have_iqd = any(q is not None or d is not None for q, d in zip(interval_q, interval_d))

            try:
                import matplotlib.pyplot as plt
            except Exception:
                return  # plotting not available; silently skip

            plots_dir = self._plots_dir()

            def _plot(y, ylabel, title, fname):
                fig = plt.figure()
                plt.plot(steps, y, marker="o")
                plt.xlabel("Checkpoint step"); plt.ylabel(ylabel); plt.title(title)
                fig.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
                plt.close(fig)

            _plot(mean_returns, "Mean return (eval)", "Mean Return vs Checkpoint", "return_vs_checkpoint.png")
            _plot(mean_qualities, "Mean quality (eval)", "Mean Quality vs Checkpoint", "quality_vs_checkpoint.png")
            _plot(mean_steps, "Mean steps/episode (eval)", "Mean Episode Length vs Checkpoint", "steps_vs_checkpoint.png")

            if have_iqd:
                fig = plt.figure()
                plt.plot(steps, interval_q, marker="o", label="Interval mean quality")
                plt.plot(steps, interval_d, marker="s", label="Interval mean diversity")
                plt.xlabel("Checkpoint step"); plt.ylabel("Interval metric")
                plt.title("Interval Quality/Diversity vs Checkpoint"); plt.legend()
                fig.savefig(os.path.join(plots_dir, "interval_qd_vs_checkpoint.png"), bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

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