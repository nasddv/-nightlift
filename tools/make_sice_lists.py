#!/usr/bin/env python3
"""Pair lists for SICE-Grad: input <root>/SICE_Grad/<name>, target <root>/SICE_Reshape/<name> (900x600).

Test = the scenes in FECNet's SICE test split (scene id k for Dataset_Part1_k, 360+k for
Dataset_Part2_k), the protocol ExpoMamba's code also uses; everything else is train
(train.py holds out its own validation images).

  python tools/make_sice_lists.py --root /path/to/SICE --test_ids ../expomamba-llie/data/splits/sice/fecnet_test_ids.txt
"""
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--test_ids", default="../expomamba-llie/data/splits/sice/fecnet_test_ids.txt")
ap.add_argument("--out", default="splits/sice_grad")
a = ap.parse_args()

root = Path(a.root).expanduser().resolve()
gt = {p.stem: p for p in (root / "SICE_Reshape").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
test_ids = {int(x) for x in open(a.test_ids).read().split()}


def scene_id(stem):
    part, k = stem.split("_")[-2:]
    return int(k) + (360 if part == "Part2" else 0)


train, test = [], []
for p in sorted((root / "SICE_Grad").iterdir()):
    if p.stem not in gt:
        continue
    row = f"{p.relative_to(root)}\t{gt[p.stem].relative_to(root)}\n"
    (test if scene_id(p.stem) in test_ids else train).append(row)
os.makedirs(a.out, exist_ok=True)
for name, rows in (("train", train), ("test", test)):
    open(os.path.join(a.out, name + ".txt"), "w").writelines(rows)
    print(f"{name}: {len(rows)} pairs -> {a.out}/{name}.txt")
