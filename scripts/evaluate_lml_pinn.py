"""PINN (divergence-free stream-function fit) post-processing on the LML samples,
uniform-translation flow only (the LML "plasma" field is not divergence-free, so the PINN
assumption does not hold there).  Uses the cache written by evaluate_lml.py.

Inputs: U-Net (zero-shot) split -> multi-pass CC, with and without UOD before the fit.
python scripts/evaluate_lml_pinn.py
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from pivsep.correlation import multipass_piv
from pivsep.separation import SepUNet, unet_split
from pivsep.postproc import uod_median, pinn_fit
import evaluate_lml as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet_wide_bce.pt")
    ap.add_argument("--cache", default="data/lml/lml_samples.npz")
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--ppp", type=float, default=0.01)
    ap.add_argument("--n-img", type=int, default=12)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default="results/eval_lml_pinn.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    ck = torch.load(a.model, map_location="cpu")
    m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    C = np.load(a.cache)
    out = {}
    for r in a.r:
        acc = {k: [] for k in ("mp", "mp+uod", "mp+pinn", "mp+uod+pinn")}
        t0 = time.time()
        for i in range(a.n_img):
            s = {k: C[f"translation|{r}|{a.ppp}|{i}|{k}"] for k in ("I", "P1", "P2")}
            I = E.to_model_scale(s["I"])
            cx, cy, gu, gv = E.gt_windows(s["P1"], s["P2"], I.shape)
            val = np.isfinite(gu) & (np.hypot(gu, gv) > 2)
            u, v = multipass_piv(*unet_split(m, I), mask_r=3)[2:]
            uu, vu = uod_median(u, v)[:2]
            res = {"mp": (u, v), "mp+uod": (uu, vu),
                   "mp+pinn": pinn_fit(cx, cy, u, v, steps=a.steps),
                   "mp+uod+pinn": pinn_fit(cx, cy, uu, vu, steps=a.steps)}
            for k, (x, y) in res.items():
                acc[k].append(np.hypot(x - gu, y - gv)[val])
            print(f"  r={r} image {i + 1}/{a.n_img} ({time.time() - t0:.0f}s)", flush=True)
        out[f"translation/r{r}"] = {}
        print(f"translation r={r}:", flush=True)
        for k, v in acc.items():
            e = np.concatenate(v)
            out[f"translation/r{r}"][k] = dict(n=int(e.size), outlier=float(np.mean(e > 1)),
                                               median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))))
            print(f"   {k:12s} outlier {np.mean(e > 1):.3f}  median {np.median(e):.3f}  rms {np.sqrt(np.mean(e ** 2)):.3f}",
                  flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
