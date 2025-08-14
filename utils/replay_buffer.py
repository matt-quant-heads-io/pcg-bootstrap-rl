from typing import Any, List, Tuple
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []
        self.idx = 0

    def push(self, s: torch.Tensor, a: int, r: float, s2: torch.Tensor, d: bool):
        data = (s.cpu(), int(a), float(r), s2.cpu(), bool(d))
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.idx] = data
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(list(s)),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(list(s2)),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.storage)
