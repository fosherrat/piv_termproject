"""Stage 3: apply to real PIV recordings (OpenPIV example data).

Real recordings are double-frame (A, B).  We build a pulse-coded double exposure
    I = A_n + r * B_n + bg          (A_n, B_n: background-subtracted, normalized frames)
and compare every double-exposure method with the reference obtained from the real
separate frames (cross-correlation of A_n and B_n).  The reference is only trusted where
it passes the universal outlier test.

python scripts/evaluate_real.py --model wide_bce=results/sep_unet_wide_bce.pt
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tifffile
from PIL import Image
from scipy import ndimage

from pivsep.correlation import cross_piv, auto_piv, asym_auto_piv
from pivsep.separation import SepUNet, threshold_split, unet_split
from pivsep.postproc import uod_median, pinn_fit

WIN, STEP = 32, 16
DATA = "data/real"


def load(f):
    a = tifffile.imread(f) if f.endswith(".tif") else np.array(Image.open(f).convert("L"))
    return a.astype(np.float64)


def pairs():
    out = {"test1": ("test1/exp1_001_a.bmp", "test1/exp1_001_b.bmp")}
    for i in range(6):
        out[f"test2_{i}"] = (f"test2/2image_{i}0.tif", f"test2/2image_{i}1.tif")
    return {k: tuple(os.path.join(DATA, f) for f in v) for k, v in out.items()}


def preprocess(a):
    """Background removal (local min, smoothed)."""
    bg = ndimage.uniform_filter(ndimage.minimum_filter(a, 15), 31)
    return np.clip(a - bg, 0, None)


def normalize_pair(A, B, peak=0.6):
    s = peak / np.percentile(np.concatenate([A.ravel(), B.ravel()]), 99.7)
    return A * s, B * s


def make_double(A, B, r, bg=0.02):
    return np.clip(A + r * B + bg, 0, 1).astype(np.float32)


def crop(a):
    H, W = a.shape
    h, w = H // 16 * 16, W // 16 * 16
    return a[:h, :w]


def evaluate_pair(models, A, B, r):
    I = make_double(A, B, r)
    cx, cy, ru, rv = cross_piv(A, B, WIN, STEP)              # reference from real frames
    _, _, bad = uod_median(ru, rv)
    valid = ~bad & (np.hypot(ru, rv) > 2.0)                 # double exposure needs |d| > ~1.5 d_p
    res = {}
    ua, va = auto_piv(I, WIN, STEP)[2:]
    pick = (ua * ru + va * rv) < 0
    res["auto_or"] = (np.where(pick, -ua, ua), np.where(pick, -va, va))
    res["asym"] = asym_auto_piv(I, WIN, STEP)[2:]
    res["thr_cc"] = cross_piv(*threshold_split(I), WIN, STEP)[2:]
    for name, m in models.items():
        u, v = cross_piv(*unet_split(m, I), WIN, STEP)[2:]
        res[f"unet_cc:{name}"] = (u, v)
        res[f"unet_cc:{name}+pinn"] = pinn_fit(cx, cy, u, v, steps=800)
    out = {}
    for k, (u, v) in res.items():
        e = np.hypot(u - ru, v - rv)[valid]
        sg = ((u * ru + v * rv) < 0)[valid]
        out[k] = dict(err=e, sign=sg)
    return out, dict(I=I, cx=cx, cy=cy, ref=(ru, rv), valid=valid, res=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["wide_bce=results/sep_unet_wide_bce.pt"])
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--sets", nargs="+", default=None, help="subset of pair names")
    ap.add_argument("--out", default="results/eval_real.json")
    a = ap.parse_args()
    models = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
        models[name] = m
    P = pairs()
    names = a.sets or list(P)
    table = {}
    for r in a.r:
        for group in ("test1", "test2"):
            outs = []
            for n in [n for n in names if n.startswith(group)]:
                A, B = normalize_pair(*(crop(preprocess(load(f))) for f in P[n]))
                outs.append(evaluate_pair(models, A, B, r)[0])
            if not outs:
                continue
            t = {}
            for k in outs[0]:
                e = np.concatenate([o[k]["err"] for o in outs]); sg = np.concatenate([o[k]["sign"] for o in outs])
                t[k] = dict(n=int(e.size), median=float(np.median(e)), outlier=float(np.mean(e > 1)),
                            sign_err=float(np.mean(sg)))
            table[f"{group}/r{r}"] = t
            print(f"{group} r={r} (n={t['asym']['n']}): " + "  ".join(
                f"{k}: out {v['outlier']:.3f} sgn {v['sign_err']:.3f} med {v['median']:.3f}" for k, v in t.items()),
                flush=True)
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
