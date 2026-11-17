import math

import numpy as np


class Adam:
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, *betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, lr=None, clip=None):
        lr = self.lr if lr is None else lr
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = clip / gnorm if clip and gnorm > clip else 1.0
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p.data -= lr * ((m / bc1) / (np.sqrt(v / bc2) + self.eps) + self.wd * p.data)
            p.grad = None
        return gnorm

    def state(self):
        return {"t": self.t, **{f"m{i}": m for i, m in enumerate(self.m)}, **{f"v{i}": v for i, v in enumerate(self.v)}}

    def load(self, st):
        self.t = int(st["t"])
        for i in range(len(self.m)):
            self.m[i][...] = st[f"m{i}"]
            self.v[i][...] = st[f"v{i}"]


def cosine_lr(it, total, base, warmup=200, floor=1e-6):
    if it < warmup:
        return base * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * min(1.0, t)))
