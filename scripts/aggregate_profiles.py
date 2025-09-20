#!/usr/bin/env python3
import argparse, os, json, csv, re
import numpy as np

TIME_KEY_WALL = "Elapsed (wall clock) time"
TIME_KEY_RSS  = "Maximum resident set size"

def parse_time_log(path):
    if not os.path.exists(path):
        return {}
    wall = None
    rss = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(TIME_KEY_WALL):
                wall = line.strip().split(": ", 1)[1]
            elif line.startswith(TIME_KEY_RSS):
                rss = line.strip().split(": ", 1)[1]
    # convert wall to seconds "H:MM:SS" or "M:SS" format
    wall_sec = None
    if wall:
        parts = wall.split(":")
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 3: wall_sec = parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2: wall_sec = parts[0]*60 + parts[1]
            else: wall_sec = float(parts[0])
        except:
            pass
    rss_kb = None
    if rss:
        try:
            rss_kb = float(rss)
        except:
            pass
    return {"wall_time_sec": wall_sec, "max_rss_kb": rss_kb}

def parse_gpu_csv(path):
    if not os.path.exists(path):
        return {}
    # header: timestamp, name, util.gpu [%], util.mem [%], mem.used [MiB], mem.total [MiB]
    util_gpu, util_mem, mem_used, mem_total = [], [], [], []
    with open(path, "r", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6: 
                continue
            try:
                ug = float(parts[2].rstrip(" %"))
                um = float(parts[3].rstrip(" %"))
                mu = float(parts[4].split()[0])
                mt = float(parts[5].split()[0])
            except:
                continue
            util_gpu.append(ug)
            util_mem.append(um)
            mem_used.append(mu)
            mem_total.append(mt)
    res = {}
    if util_gpu:
        res["gpu_util_avg"] = float(np.mean(util_gpu))
        res["gpu_util_max"] = float(np.max(util_gpu))
    if mem_used:
        res["gpu_mem_used_max_mib"] = float(np.max(mem_used))
        res["gpu_mem_used_avg_mib"] = float(np.mean(mem_used))
        res["gpu_mem_total_mib"] = float(np.max(mem_total))
    return res

def read_eval_json(run_dir):
    path = os.path.join(run_dir, "metrics", "eval.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # pick the latest key (e.g., max step or alphabetically last)
        if isinstance(data, dict) and data:
            last_key = sorted(data.keys())[-1]
            return {"eval_key": last_key, **data[last_key]}
    except Exception:
        pass
    return {}

def infer_env_alg_from_run_dir(run_dir):
    # .../results/<env>/<alg>/<stamp>
    parts = run_dir.strip(os.sep).split(os.sep)
    env, alg, stamp = None, None, None
    try:
        env = parts[-3]
        alg = parts[-2]
        stamp = parts[-1]
    except:
        pass
    return env, alg, stamp

def append_csv(csv_path, row, field_order):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in field_order})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--profile_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    time_log = os.path.join(args.profile_dir, "time.log")
    gpu_log  = os.path.join(args.profile_dir, "gpu.csv")
    cprof    = os.path.join(args.profile_dir, "cprofile.pstats")

    env, alg, stamp = infer_env_alg_from_run_dir(args.run_dir)
    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "profile_dir": os.path.abspath(args.profile_dir),
        "env": env, "algorithm": alg, "stamp": stamp,
        "cprofile_path": cprof if os.path.exists(cprof) else ""
    }
    summary.update(parse_time_log(time_log))
    summary.update(parse_gpu_csv(gpu_log))
    summary.update(read_eval_json(args.run_dir))

    # write per-run JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # append global CSV
    fields = [
        "env","algorithm","stamp","run_dir","profile_dir",
        "wall_time_sec","max_rss_kb",
        "gpu_util_avg","gpu_util_max","gpu_mem_used_max_mib","gpu_mem_used_avg_mib","gpu_mem_total_mib",
        "eval_key","episodes","return_mean","return_std","len_mean","len_std","quality_mean","quality_std",
        "cprofile_path"
    ]
    append_csv(args.out_csv, summary, fields)

    print("Wrote:", args.out_json)
    print("Appended:", args.out_csv)

if __name__ == "__main__":
    main()
