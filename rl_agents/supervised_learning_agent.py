
from typing import Dict, Any
import os
import numpy as np
import torch
import torch.nn as nn
from .rl_agent_base import RLAgentBase


class SupervisedLearningAgent(RLAgentBase):
    """Simple behavior cloning-style pretrainer for the given model.

    Expects a dataset .npz with 'states' [N,C,H,W] and 'actions' [N]."""
    def __init__(self, env, model, algo_config: Dict[str, Any], run_dir: str):
        super().__init__(env, algo_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.run_dir = run_dir
        self.lr = float(algo_config.get("learning_rate", 3e-4))
        self.epochs = int(algo_config.get("epochs", 5))
        self.batch_size = int(algo_config.get("batch_size", 128))
        self.dataset_path = str(algo_config.get("dataset", "data/zelda_bc.npz"))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def act_eval(self, state):
        logits, _ = self.model(state.to(self.device))
        return int(torch.argmax(logits, dim=-1).item())

    def _iter_batches(self, X, y, bs):
        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        for start in range(0, N, bs):
            sl = idx[start:start+bs]
            xb = torch.as_tensor(X[sl], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(y[sl], dtype=torch.long, device=self.device)
            yield xb, yb

    def train(self):
        data = np.load(self.dataset_path)
        X = data["states"]  # [N,C,H,W]
        y = data["actions"] # [N]
        for ep in range(self.epochs):
            total, correct, loss_sum = 0, 0, 0.0
            for xb, yb in self._iter_batches(X, y, self.batch_size):
                logits, _ = self.model(xb)
                loss = self.ce(logits, yb)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                loss_sum += float(loss.detach().item()) * xb.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == yb).sum().item())
                total += int(xb.size(0))
            acc = correct / max(1, total)
            print(f"[BC] epoch {ep+1}/{self.epochs} loss={loss_sum/max(1,total):.4f} acc={acc:.3f}")
        # Save pretrained weights
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "supervised.pt"))

    @torch.no_grad()
    def evaluate(self, episodes: int = 5):
        # reuse RLAgentBase.evaluate for env-based eval
        return super().evaluate(episodes=episodes)
