import os, yaml, datetime
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def derive_run_dir(game: str, algo: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("results", game, algo, ts)
    for d in ["", "metrics", "plots", "checkpoints", "levels", "configs"]:
        os.makedirs(os.path.join(out, d) if d else out, exist_ok=True)
    return out
