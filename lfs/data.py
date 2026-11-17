"""LOL-v2 Synthetic loading: pairs, a one-off uint8 cache, random patch batches."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _imgs(d):
    return {p.stem: p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in EXT}


def find_pairs(split_dir):
    """<split_dir>/{Input|Low}/ and {GT|Normal}/. Handles the HF layout where test inputs are
    named 'r<id>' but test GTs '<id>'."""
    split_dir = Path(split_dir)
    low_d = next(split_dir / n for n in ("Input", "Low", "input", "low") if (split_dir / n).is_dir())
    gt_d = next(split_dir / n for n in ("GT", "Normal", "gt", "normal") if (split_dir / n).is_dir())
    low, gt = _imgs(low_d), _imgs(gt_d)
    pairs = []
    for k, p in low.items():
        g = gt.get(k) or (gt.get(k[1:]) if k.startswith("r") else None)
        if g is not None:
            pairs.append((str(p), str(g)))
    if not pairs:
        raise RuntimeError(f"no low/GT pairs found under {split_dir}")
    return pairs


def read_pair_list(path, root=None):
    """Read tab-separated image pairs, optionally resolving paths under root."""
    from pathlib import Path

    pairs = []
    base = Path(root).expanduser() if root else None

    with open(path) as f:
        for line in f:
            columns = line.rstrip("\n").split("\t")
            if len(columns) >= 2:
                low, gt = columns[0], columns[1]
                if base:
                    low = str(base / low) if not Path(low).is_absolute() else low
                    gt = str(base / gt) if not Path(gt).is_absolute() else gt
                pairs.append((low, gt))

    if not pairs:
        raise RuntimeError(f"empty pair list: {path}")
    return pairs


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def cached_stack(pairs, cache_dir):
    """Stack all pairs into two uint8 arrays (N,H,W,3), cached as .npy (memory-mapped)."""
    key = hashlib.md5("\n".join(a + b for a, b in pairs).encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    fl, fg = (os.path.join(cache_dir, f"{key}_{s}.npy") for s in ("low", "gt"))
    if not (os.path.exists(fl) and os.path.exists(fg)):
        h, w, _ = load_rgb(pairs[0][0]).shape
        L = np.lib.format.open_memmap(fl + ".tmp.npy", "w+", np.uint8, (len(pairs), h, w, 3))
        G = np.lib.format.open_memmap(fg + ".tmp.npy", "w+", np.uint8, (len(pairs), h, w, 3))
        for i, (a, b) in enumerate(pairs):
            la, gb = load_rgb(a), load_rgb(b)
            if la.shape != (h, w, 3) and la.shape[:2] == (w, h):  # portrait image: rotate to landscape
                la, gb = np.rot90(la), np.rot90(gb)
            L[i], G[i] = la, gb
            if (i + 1) % 100 == 0:
                print(f"  cached {i + 1}/{len(pairs)}", flush=True)
        L.flush(), G.flush()
        del L, G
        os.replace(fl + ".tmp.npy", fl)
        os.replace(fg + ".tmp.npy", fg)
    return np.load(fl, mmap_mode="r"), np.load(fg, mmap_mode="r")


def split_train_val(pairs, n_val, seed):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    val = sorted(idx[:n_val])
    tr = sorted(idx[n_val:])
    return tr, val


def to_chw(a):
    """uint8 (N,H,W,3) -> float32 (N,3,H,W) in [0,1]."""
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class PatchSampler:
    """Random patch batches with flips / 90-degree rotations (same for input and target)."""

    def __init__(self, low, gt, indices, patch, batch, seed):
        self.low, self.gt, self.idx = low, gt, list(indices)
        self.patch, self.batch = patch, batch
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        p = self.patch
        H, W = self.low.shape[1:3]
        xs, ys = [], []
        for i in self.rng.choice(self.idx, self.batch, replace=False):
            y0, x0 = self.rng.integers(0, H - p + 1), self.rng.integers(0, W - p + 1)
            a = np.asarray(self.low[i, y0:y0 + p, x0:x0 + p])
            b = np.asarray(self.gt[i, y0:y0 + p, x0:x0 + p])
            if self.rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            if self.rng.random() < 0.5:
                a, b = a[::-1], b[::-1]
            k = int(self.rng.integers(0, 4))
            a, b = np.rot90(a, k), np.rot90(b, k)
            xs.append(a)
            ys.append(b)
        return to_chw(np.stack(xs)), to_chw(np.stack(ys))
