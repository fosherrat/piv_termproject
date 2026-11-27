"""Unified, matched-processing comparison for the paper.

Every method sees the same pulse-coded double exposure I and is processed with the same
pipeline, so that the only difference between methods is how the direction / pairing is
obtained:
  * same pass schedule 64/32 -> 32/16 -> 32/16 (final grid 32 px windows, 50 % overlap),
    same between-pass UOD + smoothing, same centre mask (radius 3 px), same sub-pixel fit
  * same final post-processing: reported raw and after the same UOD + median replacement
  * same evaluation vectors per dataset (decided from the reference only):
    |d_ref| > 2 px (double exposures need |d| > ~1.5 d_p) and, for real recordings,
    reference passes UOD and (case C) window centre inside the flow mask

Methods
  gt     : multi-pass CC of the true separate frames A, B (upper bound; synthetic only)
  asym   : nonlinear auto-correlation sum I^2(x) I(x+s), multi-pass (classical direction fix)
  oracle : auto-correlation, multi-pass, direction taken from the reference (idealised)
  unet   : U-Net split -> multi-pass CC with symmetric window deformation (proposed)
Single-pass (32/16) versions of asym / oracle / unet are stored as well (processing ablation).

U-Net model fixed per domain before evaluation: synthetic datasets -> model trained on our
synthetic images only (zero-shot); real recordings -> model fine-tuned on test2 frames
(test1 and case C were never used for training).  Both models are stored for every dataset.

Datasets
  synth : pivsep/synth.py, 12 images 256^2 per r (seeds 1000..1011, same as stage 1)
  lml   : LML generator (PGlml/DGlml/SIGlml), uniform translation, 12 images per r, ppp 0.01
  test1 : OpenPIV test1 (real), reference = multi-pass CC of the real frames
  test2 : OpenPIV test2 (real, cylinder wake), pairs 3-5 pooled; pairs 0-2 were used for
          fine-tuning, so this is a held-out-instant test of the fine-tuning set-up, not an
          unseen recording.  Reference as test1
  caseC : PIV Challenge 2001 case C (real, background subtracted), reference as test1

python scripts/evaluate_paper.py [--datasets synth lml test1 test2 caseC]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from pivsep.correlation import (cross_piv, auto_piv, asym_auto_piv, multipass_piv,
                                multipass_auto_piv, window_grid)
from pivsep.separation import SepUNet, unet_split
from pivsep.postproc import uod_median
from pivsep.synth import SynthParams, make_sample, gt_window_displacement
import evaluate_real as R

WIN, STEP, MASK_R = 32, 16, 3
MODELS = {"zeroshot": "results/sep_unet_wide_bce.pt", "finetuned": "results/sep_unet_ft_real.pt"}
MAIN_MODEL = {"synth": "zeroshot", "lml": "zeroshot", "test1": "finetuned", "test2": "finetuned",
              "caseC": "finetuned"}
TEST2_EVAL = ("test2_3", "test2_4", "test2_5")   # pairs 0-2 were used for fine-tuning


def load_models():
    out = {}
    for name, path in MODELS.items():
        ck = torch.load(path, map_location="cpu")
        m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
        out[name] = m
    return out


def run_methods(models, I, ref, A=None, B=None):
    """ref = (cx, cy, u, v) on the final grid (true field or reference), used only for the
    oracle direction.  Returns {method: (u, v)}."""
    sref = (ref[0], ref[1], np.nan_to_num(ref[2]), np.nan_to_num(ref[3]))
    res = {}
    if A is not None:
        res["gt"] = multipass_piv(A, B)[2:]
    res["asym"] = multipass_auto_piv(I, "asym", mask_r=MASK_R)[2:]
    res["oracle"] = multipass_auto_piv(I, "oracle", mask_r=MASK_R, sign_ref=sref)[2:]
    res["asym_sp"] = asym_auto_piv(I, WIN, STEP, mask_r=MASK_R)[2:]
    ua, va = auto_piv(I, WIN, STEP, mask_r=MASK_R)[2:]
    flip = (ua * sref[2] + va * sref[3]) < 0
    res["oracle_sp"] = (np.where(flip, -ua, ua), np.where(flip, -va, va))
    for name, m in models.items():
        sA, sB = unet_split(m, I)
        res[f"unet:{name}"] = multipass_piv(sA, sB, mask_r=MASK_R)[2:]
        res[f"unet_sp:{name}"] = cross_piv(sA, sB, WIN, STEP, mask_r=MASK_R)[2:]
    for k in list(res):
        res[k + "+uod"] = uod_median(*res[k])[:2]
    return res


def errors(res, gu, gv, valid):
    return {k: dict(err=np.hypot(u - gu, v - gv)[valid], sign=((u * gu + v * gv) < 0)[valid])
            for k, (u, v) in res.items()}


def summarize(outs, n_img):
    t = {}
    for k in outs[0]:
        e = np.concatenate([o[k]["err"] for o in outs]); s = np.concatenate([o[k]["sign"] for o in outs])
        t[k] = dict(n=int(e.size), n_img=n_img, outlier=float(np.mean(e > 1)), sign_err=float(np.mean(s)),
                    median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))))
        if n_img > 1:      # 95 % bootstrap interval of the outlier rate, resampling whole images
            rng = np.random.default_rng(0)
            per = [(np.sum(o[k]["err"] > 1), o[k]["err"].size) for o in outs]
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, n_img, n_img)
                bs.append(sum(per[i][0] for i in idx) / max(sum(per[i][1] for i in idx), 1))
            t[k]["outlier_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return t


def report(name, r, t):
    keys = [k for k in t if k.endswith("+uod") or "_sp" not in k]
    print(f"{name} r={r} (n={next(iter(t.values()))['n']}):\n   " + "\n   ".join(
        f"{k:22s} out {t[k]['outlier']:.3f} sgn {t[k]['sign_err']:.3f} med {t[k]['median']:.3f} rms {t[k]['rms']:.3f}"
        for k in keys), flush=True)


# ------------------------------------------------------------------------------ datasets
def ds_synth(models, r, n_img=12):
    outs = []
    for i in range(n_img):
        s = make_sample(np.random.default_rng(1000 + i), SynthParams(r=r))
        cx, cy = window_grid(s["img"].shape, WIN, STEP)[2:]
        gu, gv = gt_window_displacement(s["flow"], cx, cy, WIN)
        valid = np.hypot(gu, gv) > 2
        res = run_methods(models, s["img"], (cx, cy, gu, gv), s["A"], s["B"])
        outs.append(errors(res, gu, gv, valid))
    return summarize(outs, n_img)


def ds_lml(models, r, cache, n_img=12, ppp=0.01):
    import evaluate_lml as E
    C = np.load(cache)
    outs = []
    for i in range(n_img):
        s = {k: C[f"translation|{r}|{ppp}|{i}|{k}"] for k in ("I", "A", "B", "P1", "P2")}
        I = E.to_model_scale(s["I"])
        cx, cy, gu, gv = E.gt_windows(s["P1"], s["P2"], I.shape)
        valid = np.isfinite(gu) & (np.hypot(gu, gv) > 2)
        res = run_methods(models, I, (cx, cy, gu, gv), s["A"].astype(np.float64), s["B"].astype(np.float64))
        outs.append(errors(res, gu, gv, valid))
    return summarize(outs, n_img)


def real_errors(models, r, A, B, inmask=None):
    cx, cy, ru, rv = multipass_piv(A, B)                     # reference: real separate frames
    _, _, bad = uod_median(ru, rv)
    valid = ~bad & (np.hypot(ru, rv) > 2)
    if inmask is not None:
        valid &= inmask
    I = R.make_double(A, B, r)
    res = run_methods(models, I, (cx, cy, ru, rv))
    return errors(res, ru, rv, valid)


def ds_real(models, r, A, B, inmask=None):
    return summarize([real_errors(models, r, A, B, inmask)], 1)


def ds_real_multi(models, r, frames):
    """Several pairs of one recording pooled (bootstrap over pairs)."""
    return summarize([real_errors(models, r, A, B) for A, B in frames], len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "lml", "test1", "test2", "caseC"])
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--lml-cache", default="data/lml/lml_samples.npz")
    ap.add_argument("--out", default="results/eval_paper.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    models = load_models()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["_meta"] = dict(main_model=MAIN_MODEL, win=WIN, step=STEP, mask_r=MASK_R,
                        passes="64/32, 32/16, 32/16", outlier_px=1.0, min_ref_disp_px=2.0)
    for ds in a.datasets:
        if ds == "lml" and not os.path.exists(a.lml_cache):
            import evaluate_lml as E
            os.makedirs(os.path.dirname(a.lml_cache), exist_ok=True)
            E.build_cache(a.lml_cache, "third_party/lml", [("translation", r, 0.01) for r in a.r], 12)
        if ds == "test1":
            P = R.pairs()["test1"]
            A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in P))
        if ds == "test2":
            P = R.pairs()
            frames2 = [R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in P[k])) for k in TEST2_EVAL]
        if ds == "caseC":
            if not os.path.isdir("data/pivchallenge_C"):
                print("caseC: data/pivchallenge_C not found, skipped", flush=True)
                continue
            import evaluate_pivc as PC
            frames, M = PC.load_case()
            A, B = frames["bgsub"]
            cx, cy = window_grid(A.shape, WIN, STEP)[2:]
            inm = PC.centre_mask(M, cx, cy)
        for r in a.r:
            t0 = time.time()
            if ds == "synth":
                t = ds_synth(models, r)
            elif ds == "lml":
                t = ds_lml(models, r, a.lml_cache)
            elif ds == "test1":
                t = ds_real(models, r, A, B)
            elif ds == "test2":
                t = ds_real_multi(models, r, frames2)
            else:
                t = ds_real(models, r, A, B, inm)
            out[f"{ds}/r{r}"] = t
            report(ds, r, t)
            print(f"   ({time.time() - t0:.0f}s)", flush=True)
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
