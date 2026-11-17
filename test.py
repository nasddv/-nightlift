#!/usr/bin/env python3
"""Evaluate a checkpoint on the LOL-v2 Synthetic Test split (100 pairs).

  python test.py --ckpt runs/lolv2syn_c16/best.npz --save_images
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

import time

from lfs.data import find_pairs, load_rgb, read_pair_list, to_chw
from lfs.metrics import psnr, ssim
from lfs.model import FreqSSMLite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="../expomamba-llie/datasets/LOLv2_syn")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--pairs", default="", help="pair list (low<TAB>gt) to test on instead of <data_root>/<split>")
    ap.add_argument("--pairs_root", default="", help="dataset root for relative pair-list paths")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop after this long; re-run to continue")
    ap.add_argument("--channels", type=int, default=None, help="default: read from the run's config.json")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    run = os.path.dirname(a.ckpt)
    c = a.channels
    if c is None:
        cfg = os.path.join(run, "config.json")
        c = json.load(open(cfg))["channels"] if os.path.exists(cfg) else 16
    net = FreqSSMLite(c=c)
    sd = np.load(a.ckpt)
    net.load_state_dict({k[2:] if k.startswith("p:") else k: sd[k] for k in sd.files if not k.startswith(("o:", "iter", "best"))})

    pairs = read_pair_list(a.pairs, a.pairs_root or None) if a.pairs else find_pairs(os.path.join(a.data_root, a.split))
    if a.limit:
        pairs = pairs[: a.limit]
    out_dir = os.path.join(run, f"{a.split.lower()}_images")
    if a.save_images:
        os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(run, f"{a.split.lower()}_partial.json")  # per-image results so far (resume)
    rows = json.load(open(partial)) if os.path.exists(partial) else []
    done, t0 = {r["image"] for r in rows}, time.time()
    for i, (lo, hi) in enumerate(pairs):
        if os.path.basename(lo) in done:
            continue
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"stopped after {a.max_minutes} min: {len(rows)}/{len(pairs)} done; re-run to continue")
            return
        x, y = to_chw(load_rgb(lo)[None]), to_chw(load_rgb(hi)[None])
        out = net.enhance(x)
        p_in, p = float(psnr(x, y)[0]), float(psnr(out, y)[0])
        s = float(ssim(out, y)[0])
        rows.append({"image": os.path.basename(lo), "psnr_input": p_in, "psnr": p, "ssim": s})
        if a.save_images:
            Image.fromarray((out[0].transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(out_dir, os.path.basename(lo)))
        json.dump(rows, open(partial, "w"))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)}  running PSNR {np.mean([r['psnr'] for r in rows]):.3f}", flush=True)
    res = {"split": a.split, "n": len(rows), "ckpt": a.ckpt,
           "psnr_input": float(np.mean([r["psnr_input"] for r in rows])),
           "psnr": float(np.mean([r["psnr"] for r in rows])),
           "ssim": float(np.mean([r["ssim"] for r in rows]))}
    with open(os.path.join(run, f"{a.split.lower()}_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run, f"{a.split.lower()}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"{a.split}: {res['n']} images | input PSNR {res['psnr_input']:.3f} -> "
          f"PSNR {res['psnr']:.3f} dB, SSIM {res['ssim']:.4f}")


if __name__ == "__main__":
    main()
