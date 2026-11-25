"""Compare velocity estimation methods on pulse-coded double exposures.

Methods
  gt_cc     : cross-correlation of the TRUE frames A, B           (upper bound)
  auto      : classical auto-correlation (sign picked by convention dx>0 -> ambiguous)
  auto_or   : auto-correlation with oracle sign (magnitude accuracy only)
  asym      : nonlinear auto-correlation  sum I^2(x) I(x+s)        (classical sign fix)
  thr_cc    : blob-intensity threshold split  -> cross-correlation
  unet_cc   : U-Net split                     -> cross-correlation  (proposed)

python scripts/evaluate.py --model results/sep_unet.pt
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pivsep.synth import SynthParams, make_sample, gt_window_displacement
from pivsep.correlation import cross_piv, auto_piv, asym_auto_piv
from pivsep.separation import SepUNet, threshold_split, unet_split

METHODS = ["gt_cc", "auto", "auto_or", "asym", "thr_cc", "unet_cc"]
OUTLIER_PX = 1.0


def run_image(model, s, win=32, step=16):
    img, A, B = s["img"], s["A"], s["B"]
    cx, cy, *_ = cross_piv(A, B, win, step)
    gu, gv = gt_window_displacement(s["flow"], cx, cy, win)
    res = {}
    res["gt_cc"] = cross_piv(A, B, win, step)[2:]
    ua, va = auto_piv(img, win, step)[2:]
    flip = ua < 0                               # convention: dx > 0
    res["auto"] = (np.where(flip, -ua, ua), np.where(flip, -va, va))
    pick = (ua * gu + va * gv) < 0               # oracle
    res["auto_or"] = (np.where(pick, -ua, ua), np.where(pick, -va, va))
    res["asym"] = asym_auto_piv(img, win, step)[2:]
    tA, tB = threshold_split(img)
    res["thr_cc"] = cross_piv(tA, tB, win, step)[2:]
    uA, uB = unet_split(model, img)
    res["unet_cc"] = cross_piv(uA, uB, win, step)[2:]
    out = {}
    for k, (u, v) in res.items():
        e = np.hypot(u - gu, v - gv)
        out[k] = dict(err=e.ravel(), sign=((u * gu + v * gv) < 0).ravel())
    sep = {}
    for k, (a, b) in {"thr": (tA, tB), "unet": (uA, uB)}.items():
        # fraction of particle intensity assigned to the wrong frame
        wrong = np.minimum(a, B).sum() + np.minimum(b, A).sum()
        sep[k] = wrong / (A.sum() + B.sum())
    return out, sep, (cx, cy, gu, gv, res, tA, tB, uA, uB)


def summarize(outs):
    s = {}
    for k in METHODS:
        e = np.concatenate([o[k]["err"] for o in outs])
        sg = np.concatenate([o[k]["sign"] for o in outs])
        s[k] = dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e**2))),
                    outlier=float(np.mean(e > OUTLIER_PX)), sign_err=float(np.mean(sg)))
    return s


def sweep(model, name, values, make_params, n_img, seed0=1000):
    table = {}
    for val in values:
        outs, seps = [], []
        for i in range(n_img):
            rng = np.random.default_rng(seed0 + i)       # same flow/particles across values
            o, sp, _ = run_image(model, make_sample(rng, make_params(val)))
            outs.append(o); seps.append(sp)
        t = summarize(outs)
        t["sep_wrong"] = {k: float(np.mean([s[k] for s in seps])) for k in ("thr", "unet")}
        table[str(val)] = t
        print(f"{name}={val}: " + "  ".join(
            f"{k}: med {t[k]['median']:.3f} out {t[k]['outlier']:.3f} sgn {t[k]['sign_err']:.3f}"
            for k in METHODS) + f"  | wrong-assign thr {t['sep_wrong']['thr']:.3f} "
            f"unet {t['sep_wrong']['unet']:.3f}", flush=True)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet.pt")
    ap.add_argument("--n-img", type=int, default=12)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    model = SepUNet(base=ck["base"]); model.load_state_dict(ck["state"]); model.eval()
    torch.set_num_threads(4)

    res = {}
    res["r"] = sweep(model, "r", [0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
                     lambda r: SynthParams(r=r), a.n_img)
    res["ppp"] = sweep(model, "ppp", [0.015, 0.03, 0.05, 0.07],
                       lambda p: SynthParams(r=0.7, ppp=p), a.n_img)
    res["noise"] = sweep(model, "noise", [0.005, 0.015, 0.03, 0.05],
                         lambda n: SynthParams(r=0.7, noise=n), a.n_img)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
