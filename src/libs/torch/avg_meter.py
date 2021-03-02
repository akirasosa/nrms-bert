from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    value: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    @torch.no_grad()
    def update(self, value, n: int = 1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        avg = self.avg
        self.reset()
        return avg
