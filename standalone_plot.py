#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

# Use a headless backend so PNGs save on servers without DISPLAY
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt


def _log(verbose, *msg):
    if verbose:
        print("[plot]", *msg)


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def plot_eval_json(eval_json_path: str, out_dir: str, verbose: bool = True) -> int:
    """
    Reads eval.json with structure:
    {
      "step_500000": {
        "episodes": 50,
        "len_mean": 76.0,
        "len_std": 0.0,
        "quality_mean": 0.49,
        "quality_std": 0.01,
        "infos": [
          { "quality": ..., "step_count": ..., "content_info": {...} },
          ...
        ]
      },
      "step_1000000": { ... }
    }
    Writes:
      out_dir/quality_vs_checkpoint.png
      out_dir/content_info_means.png
    Returns 0 on success, nonzero on hard failure.
    """
    # ---------- load ----------
    if not os.path.exists(eval_json_path):
        print(f"[plot] ERROR: file not found: {eval_json_path}")
        return 2

    try:
        with open(eval_json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[plot] ERROR: invalid JSON: {e}")
        return 3

    if not isinstance(data, dict) or not data:
        print("[plot] ERROR: top-level JSON must be a non-empty object.")
        return 4

    os.makedirs(out_dir, exist_ok=True)

    # ---------- parse / sort by step ----------
    items = []
    for k, v in data.items():
        if not isinstance(v, dict):
            _log(verbose, f"skip non-dict value at key {k}")
            continue
        try:
            step = int(str(k).split("_")[-1])
        except Exception:
            _log(verbose, f"skip key that doesn't look like step_*: {k}")
            continue
        items.append((step, v))

    items.sort(key=lambda kv: kv[0])

    if not items:
        print("[plot] ERROR: no valid step_* entries found.")
        return 5

    steps = [k for k, _ in items]
    q_mean = [_safe_float(v.get("quality_mean")) for _, v in items]
    q_std  = [_safe_float(v.get("quality_std"), 0.0) for _, v in items]

    _log(verbose, f"found {len(items)} checkpoints, steps={steps[:5]}{'...' if len(steps)>5 else ''}")
    _log(verbose, f"quality_mean (first 5) = {q_mean[:5]}")
    _log(verbose, f"quality_std  (first 5) = {q_std[:5]}")

    # ---------- Plot 1: quality vs checkpoint (with error bars) ----------
    wrote_any = False
    try:
        fig = plt.figure()
        plt.errorbar(steps, q_mean, yerr=q_std, fmt="-o", capsize=4)
        plt.xlabel("Checkpoint step")
        plt.ylabel("quality_mean (± quality_std)")
        plt.title("Mean Quality vs Checkpoint")
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(out_dir, "quality_vs_checkpoint.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        _log(verbose, f"wrote {out_path}")
        wrote_any = True
    except Exception as e:
        print(f"[plot] WARN: failed to write quality plot: {e}")

    # ---------- Plot 2: content_info — mean ± std across checkpoints (infos[0]) ----------
    # Pull only the first infos element and look at its content_info dict
    target_fields = ["regions", "players", "keys", "doors", "enemies", "player_key", "key_door"]
    field_values = {f: [] for f in target_fields}

    for _, v in items:
        infos = v.get("infos", [])
        if not isinstance(infos, list) or not infos:
            _log(verbose, "missing or empty 'infos'; skip this checkpoint")
            continue
        first = infos[0]
        if not isinstance(first, dict):
            _log(verbose, "infos[0] is not a dict; skip")
            continue
        ci = first.get("content_info", {})
        if not isinstance(ci, dict):
            _log(verbose, "infos[0].content_info is not a dict; skip")
            continue
        # accept alias
        if "key_door" not in ci and "player_door" in ci:
            ci = dict(ci)
            ci["key_door"] = ci["player_door"]

        for f in target_fields:
            if f in ci and ci[f] is not None:
                v = _safe_float(ci[f], None)
                if v is not None and not np.isnan(v):
                    field_values[f].append(v)

    labels, means, stds = [], [], []
    for f in target_fields:
        arr = np.asarray(field_values[f], dtype=float)
        if arr.size > 0:
            labels.append(f)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
    _log(verbose, f"content_info fields present: {labels}")

    if labels:
        try:
            x = np.arange(len(labels))
            fig = plt.figure(figsize=(10, 4))
            plt.bar(x, means, yerr=stds, capsize=4)
            plt.xticks(x, labels, rotation=20)
            plt.ylabel("Mean across checkpoints (± std)")
            plt.title("content_info (from infos[0]) — mean ± std across checkpoints")
            plt.grid(axis="y", alpha=0.3)
            out_path = os.path.join(out_dir, "content_info_means.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            _log(verbose, f"wrote {out_path}")
            wrote_any = True
        except Exception as e:
            print(f"[plot] WARN: failed to write content_info plot: {e}")
    else:
        _log(verbose, "no usable content_info fields found; skipping content_info plot.")

    if not wrote_any:
        print("[plot] WARN: no plots were written (likely no valid data).")
        return 6

    return 0


def main():
    ap = argparse.ArgumentParser(description="Plot eval.json into PNGs.")
    ap.add_argument("--eval_json", required=True, help="Path to metrics/eval.json")
    ap.add_argument("--out_dir",   required=True, help="Directory to write PNG plots")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = ap.parse_args()

    rc = plot_eval_json(args.eval_json, args.out_dir, verbose=not args.quiet)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()


# python standalone_plot.py \
#   --eval_json /home/ubuntu/pcg-bootstrap-rl/results/zelda-v0/PPO/20250911_025542/metrics/eval.json \
#   --out_dir   /home/ubuntu/pcg-bootstrap-rl/results/zelda-v0/PPO/20250911_025542/metrics/plots