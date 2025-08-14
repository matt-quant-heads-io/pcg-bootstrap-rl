import os, json

class JSONLLogger:
    def __init__(self, out_dir: str, fname: str = "metrics.jsonl"):
        self.path = os.path.join(out_dir, "metrics", fname)
        self._f = open(self.path, "a", buffering=1)

    def log(self, record):
        self._f.write(json.dumps(record) + "\n")

    def close(self):
        self._f.close()
