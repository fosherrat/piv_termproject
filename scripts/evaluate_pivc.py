"""Validation on PIV Challenge 2001, case C (Stanislas): strong wall reflection in a
centrifugal impeller, 1000 x 512 px, real recording with background images and mask.

Like stage 3/4, the real double-frame recording (A, B) is turned into a pulse-coded
double exposure I = A + r*B, every double-exposure method is run on I, and the result is
compared with the cross-correlation of the real frames A, B (multi-pass, the reference).
Nothing of case C was used for training: this is a pure cross-domain test.

Pre-processing variants for the double exposure:
  bgsub : provided background images (Cbg_1, Cbg_2) subtracted first (case C instructions)
  raw   : no background images, only the local-minimum background removal of stage 3 ->
          the wall reflection / flare stays in the double exposure (and is doubled)
The reference always uses the background-subtracted frames.

Validity (like stage 3/4): vector centre outside the mask (Cmask_1, case C rule), reference
passes UOD, |d_ref| > 2 px (double exposure needs |d| > ~1.5 d_p).

python scripts/evaluate_pivc.py --model zeroshot=results/sep_unet_wide_bce.pt \
    finetuned=results/sep_unet_ft_real.pt
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tifffile
import torch

from pivsep.correlation import cross_piv, auto_piv, asym_auto_piv, multipass_piv
from pivsep.separation import SepUNet, threshold_split, unet_split
from pivsep.postproc import uod_median
import evaluate_real as R

DATA = "data/pivchallenge_C"
WIN, STEP = 32, 16
SAMPLES = [(624, 256), (288, 352), (160, 160), (896, 432)]   # case C correlation-sample points


def load_case(data=DATA):
    L = lambda f: tifffile.imread(os.path.join(data, f + ".tif")).astype(np.float64)
    A0, B0, bA, bB = L("C001_1"), L("C001_2"), L("Cbg_1"), L("Cbg_2")
    M = L("Cmask_1") > 127                                   # True = fluid
    frames = {}
    for var, (a, b) in {"bgsub": (np.clip(A0 - bA, 0, None), np.clip(B0 - bB, 0, None)),
                        "raw": (A0, B0)}.items():
        a, b = R.preprocess(a * M), R.preprocess(b * M)
        frames[var] = R.normalize_pair(a, b)
    return frames, M


def centre_mask(M, cx, cy):
    return M[np.round(cy).astype(int), np.round(cx).astype(int)]


def stats(u, v, ok):
    s = np.hypot(u, v)[ok]
    return dict(n=int(ok.sum()), umean=float(u[ok].mean()), vmean=float(v[ok].mean()),
                urms=float(u[ok].std()), vrms=float(v[ok].std()), mag_median=float(np.median(s)))


def run_methods(models, I):
    res, tim = {}, {}

    def t(name, f):
        t0 = time.time(); res[name] = f(); tim[name] = time.time() - t0

    ua, va = auto_piv(I, WIN, STEP)[2:]
    res["auto"] = (ua, va)
    t("asym", lambda: asym_auto_piv(I, WIN, STEP)[2:])
    t("thr_ccm", lambda: cross_piv(*threshold_split(I), WIN, STEP, mask_r=3)[2:])
    for name, m in models.items():
        t0 = time.time(); sA, sB = unet_split(m, I); ts = time.time() - t0
        t(f"unet_ccm:{name}", lambda: cross_piv(sA, sB, WIN, STEP, mask_r=3)[2:])
        t(f"unet_mp:{name}", lambda: multipass_piv(sA, sB, mask_r=3)[2:])
        res[f"unet_mp:{name}+uod"] = uod_median(*res[f"unet_mp:{name}"])[:2]
        tim[f"unet_ccm:{name}"] += ts; tim[f"unet_mp:{name}"] += ts
        res[f"_split:{name}"] = (sA, sB)
    return res, tim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["zeroshot=results/sep_unet_wide_bce.pt",
                                                    "finetuned=results/sep_unet_ft_real.pt"])
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--variants", nargs="+", default=["bgsub", "raw"])
    ap.add_argument("--out", default="results/eval_pivc.json")
    ap.add_argument("--dump", default="results/pivc_fields.npz", help="fields for the figure script")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    models = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
        models[name] = m

    frames, M = load_case()
    A, B = frames["bgsub"]
    t0 = time.time()
    cx, cy, ru, rv = multipass_piv(A, B)
    t_ref = time.time() - t0
    su, sv = cross_piv(A, B, WIN, STEP)[2:]
    inm = centre_mask(M, cx, cy)
    _, _, bad = uod_median(ru, rv)
    valid = inm & ~bad & (np.hypot(ru, rv) > 2.0)
    out = {"grid": list(cx.shape), "n_fluid": int(inm.sum()), "n_valid": int(valid.sum()),
           "ref_mp": stats(ru, rv, inm & ~bad), "ref_sp_vs_mp_outlier":
               float(np.mean(np.hypot(su - ru, sv - rv)[valid] > 1)),
           "ref_time_per_vector_ms": 1e3 * t_ref / cx.size,
           "samples": {f"{x},{y}": [float(ru[np.argmin(abs(cy[:, 0] - y)), np.argmin(abs(cx[0] - x))]),
                                    float(rv[np.argmin(abs(cy[:, 0] - y)), np.argmin(abs(cx[0] - x))])]
                       for x, y in SAMPLES},
           "results": {}}
    print(f"grid {cx.shape}, fluid {inm.sum()}, valid {valid.sum()}, ref {out['ref_mp']}", flush=True)
    dump = dict(cx=cx, cy=cy, ru=ru, rv=rv, valid=valid, inm=inm, A=A, B=B)
    for var in a.variants:
        Av, Bv = frames[var]
        for r in a.r:
            I = R.make_double(Av, Bv, r)
            res, tim = run_methods(models, I)
            ua, va = res.pop("auto")
            pick = (ua * ru + va * rv) < 0
            res["auto_or"] = (np.where(pick, -ua, ua), np.where(pick, -va, va))
            t = {}
            for k, (u, v) in res.items():
                if k.startswith("_"):
                    continue
                e = np.hypot(u - ru, v - rv)[valid]
                sg = ((u * ru + v * rv) < 0)[valid]
                t[k] = dict(n=int(e.size), median=float(np.median(e)), outlier=float(np.mean(e > 1)),
                            sign_err=float(np.mean(sg)))
                if k in tim:
                    t[k]["ms_per_vector"] = 1e3 * tim[k] / cx.size
            out["results"][f"{var}/r{r}"] = t
            print(f"{var} r={r} (n={valid.sum()}):\n   " + "\n   ".join(
                f"{k:24s} out {v['outlier']:.3f} sgn {v['sign_err']:.3f} med {v['median']:.3f}"
                for k, v in t.items()), flush=True)
            if r == 0.7:
                dump[f"I_{var}"] = I
                for k, (u, v) in res.items():
                    dump[f"{var}|{k}|u"] = u; dump[f"{var}|{k}|v"] = v
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(a.dump, **{k: np.asarray(v, np.float32) for k, v in dump.items()})
    print("saved", a.out, a.dump)


if __name__ == "__main__":
    main()
