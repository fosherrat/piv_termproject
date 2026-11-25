"""Stage 5: masking + vector validation on real recordings.

Validation chain (all decided from the measurement itself, never from the reference):
  1. content mask   - saturated objects / windows with weak particle signal are excluded
  2. peak ratio     - vectors whose correlation peak is not clearly the highest are removed
  3. UOD 5x5        - universal outlier detection on a 5x5 neighbourhood, ignoring removed vectors
  (+ optional fill  - removed vectors interpolated from valid neighbours, 100 % coverage)
Thresholds are chosen on the fine-tuning training pairs (test2_0..2) at r = 0.7 and then
frozen for the test pairs (test2_3..5, test1).  The classical baselines get the same mask and
UOD (they have no comparable peak-ratio for the separated frames).

Coverage = kept vectors / vectors with a trusted reference.

python scripts/evaluate_validation.py
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import evaluate_real as R
from pivsep.separation import SepUNet, unet_split
from pivsep.correlation import multipass_piv, asym_auto_piv, auto_piv, window_quality, content_mask
from pivsep.postproc import uod_median, uod_flags, fill_nan

TRAIN = ["test2_0", "test2_1", "test2_2"]
TEST = {"test2": ["test2_3", "test2_4", "test2_5"], "test1": ["test1"]}
GRID = dict(rel_signal=[0.0, 0.4, 0.5, 0.6], ppr=[0.0, 0.9, 1.0, 1.1], uod5=[False, True])
MIN_COVERAGE = 0.7


def prepare(model, name, r):
    """Everything that does not depend on the validation thresholds."""
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()[name]))
    I = R.make_double(A, B, r)
    cx, cy, ru, rv = multipass_piv(A, B)
    ref_ok = ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2.0)
    sA, sB = unet_split(model, I)
    u, v = multipass_piv(sA, sB, mask_r=3)[2:]
    _, ppr = window_quality(sA, sB, u, v)
    au, av = asym_auto_piv(I)[2:]
    ou, ov = auto_piv(I)[2:]
    flip = (ou * ru + ov * rv) < 0                      # oracle sign
    ou, ov = np.where(flip, -ou, ou), np.where(flip, -ov, ov)
    return dict(I=I, cx=cx, cy=cy, ref=(ru, rv), ref_ok=ref_ok, unet=(u, v), ppr=ppr,
                asym=(au, av), auto_or=(ou, ov))


def validate(d, field, rel_signal, ppr_t, uod5, use_ppr=True):
    u, v = d[field]
    keep = content_mask(d["I"], rel_signal=rel_signal) if rel_signal > 0 else np.ones_like(u, bool)
    if use_ppr and ppr_t > 0:
        keep &= d["ppr"] >= ppr_t
    if uod5:
        keep &= ~uod_flags(u, v, keep)
    return keep


def score(ds, field, params, use_ppr=True, fill=False):
    errs, n_ref, n_keep, signs = [], 0, 0, []
    for d in ds:
        u, v = d[field]; ru, rv = d["ref"]; ok = d["ref_ok"]
        keep = validate(d, field, *params, use_ppr=use_ppr)
        if fill:
            u, v = fill_nan(u, v, keep)
            keep = np.ones_like(keep)
        sel = ok & keep
        errs.append(np.hypot(u - ru, v - rv)[sel]); signs.append(((u * ru + v * rv) < 0)[sel])
        n_ref += ok.sum(); n_keep += sel.sum()
    e = np.concatenate(errs); s = np.concatenate(signs)
    return dict(coverage=float(n_keep / n_ref), outlier=float(np.mean(e > 1)), sign_err=float(np.mean(s)),
                median=float(np.median(e)), n=int(e.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet_ft_real.pt")
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--out", default="results/eval_validation.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.model, map_location="cpu")
    model = SepUNet(base=ck["base"]); model.load_state_dict(ck["state"]); model.eval()

    # ---- threshold selection on training pairs, r = 0.7
    tr = [prepare(model, n, 0.7) for n in TRAIN]
    best = None
    for params in itertools.product(GRID["rel_signal"], GRID["ppr"], GRID["uod5"]):
        s = score(tr, "unet", params)
        if s["coverage"] >= MIN_COVERAGE and (best is None or s["outlier"] < best[1]["outlier"]):
            best = (params, s)
    params = best[0]
    base = (params[0], 0.0, params[2])            # same mask + UOD for baselines, no peak ratio
    print(f"selected on {TRAIN} (r=0.7): rel_signal={params[0]}, ppr>={params[1]}, uod5={params[2]}  "
          f"-> train coverage {best[1]['coverage']:.2f}, outlier {best[1]['outlier']:.3f}", flush=True)

    out = {"selected": dict(rel_signal=params[0], ppr=params[1], uod5=params[2], train=best[1])}
    for r in a.r:
        for g, names in TEST.items():
            ds = [prepare(model, n, r) for n in names]
            t = {
                "asym raw": score(ds, "asym", (0.0, 0.0, False)),
                "asym + mask + UOD5": score(ds, "asym", base, use_ppr=False),
                "auto (oracle sign) + mask + UOD5": score(ds, "auto_or", base, use_ppr=False),
                "unet mp raw": score(ds, "unet", (0.0, 0.0, False)),
                "unet mp + mask": score(ds, "unet", (params[0], 0.0, False)),
                "unet mp + mask + ppr": score(ds, "unet", (params[0], params[1], False)),
                "unet mp + mask + ppr + UOD5": score(ds, "unet", params),
                "unet mp + validation + fill": score(ds, "unet", params, fill=True),
            }
            out[f"{g}/r{r}"] = t
            print(f"\n{g} r={r}")
            for k, s in t.items():
                print(f"   {k:36s} coverage {100 * s['coverage']:5.1f}%  outlier {100 * s['outlier']:5.1f}%  "
                      f"sign {100 * s['sign_err']:5.1f}%  med {s['median']:.3f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
