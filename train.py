#!/usr/bin/env python3
"""Train FreqSSM-Lite on LOL-v2 Synthetic (NumPy only; runs on any CPU).

  python train.py --data_root ../expomamba-llie/datasets/LOLv2_syn --name lolv2syn_c16

Model selection uses 50 images held out from Train (val); Test is only used by test.py.
Re-running the same command resumes from runs/<name>/last.npz.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from lfs import autograd as ag
from lfs.autograd import Tensor
from lfs.data import PatchSampler, cached_stack, find_pairs, read_pair_list, split_train_val, to_chw
from lfs.metrics import psnr, ssim
from lfs.model import FreqSSMLite
from lfs.optim import Adam, cosine_lr


def evaluate(net, low, gt, idx, bs=2, crop=0):
    """Mean PSNR/SSIM on val images (optionally a centre crop, for a fast check during training)."""
    ps, ss = [], []
    H, W = low.shape[1:3]
    t, l = (H - crop) // 2, (W - crop) // 2
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        x, y = to_chw(np.asarray(low[j])), to_chw(np.asarray(gt[j]))
        if crop:
            x, y = x[:, :, t:t + crop, l:l + crop], y[:, :, t:t + crop, l:l + crop]
        out = net.enhance(x)
        ps += list(psnr(out, y))
        ss += list(ssim(out, y))
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../expomamba-llie/datasets/LOLv2_syn")
    ap.add_argument("--pairs", default="", help="pair list (low<TAB>gt) to train on instead of <data_root>/Train")
    ap.add_argument("--pairs_root", default="", help="dataset root for relative pair-list paths")
    ap.add_argument("--name", default="lolv2syn_c16")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/lowlight-freqssm"))
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fft_weight", type=float, default=0.05)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_images", type=int, default=50, help="val images used at each check")
    ap.add_argument("--val_crop", type=int, default=0, help="centre crop for val (0 = full image)")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop (and save) after this long; 0 = no limit")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--init", default="", help="start from these weights (best.npz/last.npz of another run)")
    a = ap.parse_args()
    if a.patch % 4:
        sys.exit("--patch must be divisible by 4")

    run = os.path.join(a.out, a.name)
    os.makedirs(run, exist_ok=True)
    pairs = read_pair_list(a.pairs, a.pairs_root or None) if a.pairs else find_pairs(os.path.join(a.data_root, "Train"))
    print(f"train pairs: {len(pairs)}  (caching to {a.cache_dir})", flush=True)
    low, gt = cached_stack(pairs, a.cache_dir)
    tr_idx, val_idx = split_train_val(pairs, a.n_val, a.seed)
    val_idx = val_idx[: a.val_images]
    sampler = PatchSampler(low, gt, tr_idx, a.patch, a.batch, a.seed)

    net = FreqSSMLite(c=a.channels, seed=a.seed)
    params = net.params()
    opt = Adam(params, lr=a.lr)
    n_params = sum(p.data.size for p in params)
    start, best = 0, {"psnr": -1.0, "iter": -1}
    last = os.path.join(run, "last.npz")
    if a.init and not os.path.exists(last):
        ck = np.load(a.init)
        net.load_state_dict({k[2:] if k.startswith("p:") else k: ck[k] for k in ck.files
                             if not k.startswith(("o:", "iter", "best"))})
        print(f"initialised weights from {a.init}", flush=True)
    if os.path.exists(last):
        ck = np.load(last, allow_pickle=True)
        net.load_state_dict({k[2:]: ck[k] for k in ck.files if k.startswith("p:")})
        opt.load({k[2:]: ck[k] for k in ck.files if k.startswith("o:")})
        start = int(ck["iter"]) + 1
        best = json.loads(str(ck["best"]))
        sampler.rng = np.random.default_rng(a.seed + start)
        print(f"resumed from {last} at iter {start}", flush=True)
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump({**vars(a), "params": n_params, "train_pairs": len(tr_idx), "val_pairs": len(val_idx),
                   "numpy": np.__version__}, f, indent=2)
    print(f"FreqSSM-Lite c={a.channels}: {n_params / 1e3:.1f}K params | train {len(tr_idx)} val {len(val_idx)}", flush=True)

    new_log = not os.path.exists(os.path.join(run, "log.csv"))
    log_f = open(os.path.join(run, "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "lr", "loss", "grad_norm", "sec_per_iter", "val_psnr", "val_ssim"])
        log_f.flush()

    def save(path, it):
        np.savez(path, iter=it, best=json.dumps(best), **{"p:" + k: v for k, v in net.state_dict().items()},
                 **{"o:" + k: v for k, v in opt.state().items()})

    t_start, t0, acc, n = time.time(), time.time(), 0.0, 0
    for it in range(start, a.iters):
        x, y = sampler()
        out = net(Tensor(x))
        loss = ag.charbonnier(out, y)
        if a.fft_weight:
            loss = ag.add(loss, ag.mul(ag.fft_amp_l1(out, y), a.fft_weight))
        if not np.isfinite(loss.data):
            sys.exit(f"non-finite loss at iter {it}")
        loss.backward()
        lr = cosine_lr(it, a.iters, a.lr)
        gn = opt.step(lr=lr, clip=1.0)
        acc += float(loss.data)
        n += 1
        last_iter = it + 1 == a.iters
        timeout = a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes
        if (it + 1) % 50 == 0 or last_iter or timeout:
            spi = (time.time() - t0) / n
            vp = vs = ""
            if (it + 1) % a.val_every == 0 or last_iter:
                vp, vs = evaluate(net, low, gt, val_idx, crop=a.val_crop)
                if vp > best["psnr"]:
                    best = {"psnr": vp, "ssim": vs, "iter": it}
                    np.savez(os.path.join(run, "best.npz"), **net.state_dict())
            log.writerow([it, f"{lr:.2e}", f"{acc / n:.5f}", f"{gn:.3f}", f"{spi:.3f}", vp, vs])
            log_f.flush()
            print(f"it {it + 1:6d}/{a.iters} lr {lr:.2e} loss {acc / n:.4f} |g| {gn:.2f} {spi:.2f}s/it"
                  + (f" | val PSNR {vp:.2f} SSIM {vs:.4f} (best {best['psnr']:.2f} @ {best['iter'] + 1})" if vp != "" else ""),
                  flush=True)
            save(last, it)
            t0, acc, n = time.time(), 0.0, 0
        if timeout:
            print(f"stopped after {a.max_minutes} min at iter {it + 1}; re-run the same command to resume", flush=True)
            return
    with open(os.path.join(run, "best.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(f"done. best val PSNR {best['psnr']:.3f} at iter {best['iter'] + 1} -> {run}/best.npz", flush=True)


if __name__ == "__main__":
    main()
