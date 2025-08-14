import os, json, argparse
import matplotlib.pyplot as plt

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def plot_reward_curve(records, out_path=None, title="Reward vs Steps"):
    steps = [r.get("step") for r in records if "step" in r and "ep_return" in r]
    rets  = [r.get("ep_return") for r in records if "step" in r and "ep_return" in r]
    if not steps:
        print("No step/ep_return pairs found in metrics.")
        return
    plt.figure()
    plt.plot(steps, rets)
    plt.xlabel("Steps")
    plt.ylabel("Episode Return")
    plt.title(title)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to a metrics.jsonl file")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Reward vs Steps")
    args = ap.parse_args()
    recs = load_jsonl(args.log)
    plot_reward_curve(recs, args.out, title=args.title)

if __name__ == "__main__":
    main()
