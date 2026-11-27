"""PINN (divergence-free stream-function MLP) post-processing in the unified pipeline, and
diagnostics that explain where it helps and where it does not.

Input is the proposed method exactly as in evaluate_paper.py: U-Net split (main model per
domain) -> multi-pass CC (64->32->32 px).  Fields compared on the same evaluation vectors:
  mp           multi-pass, no post-processing
  mp+uod       + UOD 3x3 with median replacement (adopted standard)
  mp+uod+pinn  + PINN fitted to the UOD field (all r)
  mp+pinn      PINN fitted to the raw field (r = 0.7 only; shows why UOD must come first)

Metrics on the evaluation vectors: outlier rate (> 1 px), direction-error rate, median and
RMS error, RMS vorticity error (interior windows, central differences on the grid).

Diagnostics (why PINN works or fails), per dataset:
  div_ratio   rms(div u) / rms(|grad u|_F) of the reference field.  0 for an exactly
              divergence-free planar flow; the PINN assumes 0.  For the synthetic sets it is
              given for the true field and for the cross-correlation of the true frames
              (measurement-noise level), so the real recordings can be compared with both.
  clusters    connected (8-neighbour) outlier clusters of mp+uod: share of outliers in
              clusters of >= 5 vectors and the largest cluster.  PINN is a smooth global fit:
              isolated errors are averaged away, large clusters are spread.
  fixed/new   outliers of mp+uod that PINN corrects (error <= 1 px afterwards) and correct
              vectors of mp+uod that PINN turns into outliers; the net effect of PINN is
              fixed - new, and "new" shows directly where the smooth fit hurts.

Case C: the PINN data loss uses only windows in the flow mask provided with the case (not
the evaluation vectors, which depend on the reference).

python scripts/evaluate_pinn.py --datasets synth --out results/eval_pinn_synth.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy import ndimage

from pivsep.correlation import multipass_piv, window_grid
from pivsep.separation import unet_split
from pivsep.postproc import uod_median, pinn_fit
from pivsep.synth import SynthParams, make_sample, gt_window_displacement
import evaluate_paper as EP
import evaluate_real as R

H = EP.STEP          # grid spacing [px]
STEPS = 800


def vort(u, v):
    return np.gradient(v, H, axis=1) - np.gradient(u, H, axis=0)


def div_ratio(u, v, valid):
    """rms(div) / rms(|grad u|_F) over windows whose 4 neighbours are valid."""
    g = [np.gradient(f, H, axis=a) for f in (u, v) for a in (1, 0)]   # du/dx, du/dy, dv/dx, dv/dy
    ok = valid.copy()
    ok[1:-1, 1:-1] &= valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    ok[[0, -1], :] = False; ok[:, [0, -1]] = False
    ok &= np.all([np.isfinite(x) for x in g], 0)
    div = g[0] + g[3]
    gn = np.sqrt(sum(x ** 2 for x in g))
    return dict(div_rms=float(np.sqrt(np.mean(div[ok] ** 2))), grad_rms=float(np.sqrt(np.mean(gn[ok] ** 2))),
                ratio=float(np.sqrt(np.mean(div[ok] ** 2)) / max(np.sqrt(np.mean(gn[ok] ** 2)), 1e-12)))


def clusters(out):
    lab, n = ndimage.label(out, structure=np.ones((3, 3)))
    if n == 0:
        return dict(n_out=0, share_ge5=0.0, largest=0)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(n_out=int(out.sum()), share_ge5=float(sizes[sizes >= 5].sum() / sizes.sum()), largest=int(sizes.max()))


def sample_metrics(fields, ref, valid):
    gu, gv = ref
    wv = vort(np.nan_to_num(gu), np.nan_to_num(gv))
    inner = valid.copy(); inner[[0, -1], :] = False; inner[:, [0, -1]] = False
    out = {}
    for k, (u, v) in fields.items():
        e = np.hypot(u - gu, v - gv)[valid]
        s = ((u * gu + v * gv) < 0)[valid]
        we = (vort(u, v) - wv)[inner]
        out[k] = dict(err=e, sign=s, verr=we)
    return out


def summarize(samples):
    t = {}
    for k in samples[0]:
        e = np.concatenate([s[k]["err"] for s in samples]); g = np.concatenate([s[k]["sign"] for s in samples])
        w = np.concatenate([s[k]["verr"] for s in samples])
        t[k] = dict(n=int(e.size), outlier=float(np.mean(e > 1)), sign_err=float(np.mean(g)),
                    median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))),
                    vort_rms=float(np.sqrt(np.mean(w ** 2))))
    return t


def run_one(model, I, ref_full, valid, r, fit_mask=None, seed=0):
    cx, cy, gu, gv = ref_full
    sA, sB = unet_split(model, I)
    u, v = multipass_piv(sA, sB, mask_r=EP.MASK_R)[2:]
    uu, vu = uod_median(u, v)[:2]
    fields = {"mp": (u, v), "mp+uod": (uu, vu)}
    fields["mp+uod+pinn"] = pinn_fit(cx, cy, uu, vu, steps=STEPS, seed=seed, fit_mask=fit_mask)
    if abs(r - 0.7) < 1e-9:
        fields["mp+pinn"] = pinn_fit(cx, cy, u, v, steps=STEPS, seed=seed, fit_mask=fit_mask)
    m = sample_metrics(fields, (gu, gv), valid)
    out_uod = valid & (np.hypot(uu - gu, vu - gv) > 1)
    pu, pv = fields["mp+uod+pinn"]
    out_pinn = valid & (np.hypot(pu - gu, pv - gv) > 1)
    c = clusters(out_uod)
    c.update(fixed=int((out_uod & ~out_pinn).sum()), new=int((~out_uod & out_pinn).sum()), n_valid=int(valid.sum()))
    return m, c


def merge_clusters(cl):
    n = sum(c["n_out"] for c in cl)
    return dict(n_out=n, share_ge5=float(sum(c["share_ge5"] * c["n_out"] for c in cl) / max(n, 1)),
                largest=max(c["largest"] for c in cl), fixed=sum(c["fixed"] for c in cl),
                new=sum(c["new"] for c in cl), n_valid=sum(c["n_valid"] for c in cl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "lml", "test1", "test2", "caseC"])
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lml-cache", default="data/lml/lml_samples.npz")
    ap.add_argument("--out", default="results/eval_pinn.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    models = EP.load_models()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["_meta"] = dict(main_model=EP.MAIN_MODEL, pinn_steps=STEPS, grid_px=H,
                        input="U-Net split -> multi-pass CC (same as evaluate_paper.py)")
    for ds in a.datasets:
        model = models[EP.MAIN_MODEL[ds]]
        diag = {}
        for r in a.r:
            t0 = time.time()
            samples, cls = [], []
            if ds == "synth":
                for i in range(12):
                    s = make_sample(np.random.default_rng(1000 + i), SynthParams(r=r))
                    cx, cy = window_grid(s["img"].shape, EP.WIN, EP.STEP)[2:]
                    gu, gv = gt_window_displacement(s["flow"], cx, cy, EP.WIN)
                    valid = np.hypot(gu, gv) > 2
                    m, c = run_one(model, s["img"], (cx, cy, gu, gv), valid, r, seed=i)
                    samples.append(m); cls.append(c)
                    if i == 0 and r == a.r[0]:
                        cu, cv = multipass_piv(s["A"], s["B"])[2:]
                        diag = dict(true=div_ratio(gu, gv, valid), cc_true_frames=div_ratio(cu, cv, valid))
            elif ds == "lml":
                import evaluate_lml as E
                C = np.load(a.lml_cache)
                for i in range(12):
                    s = {k: C[f"translation|{r}|0.01|{i}|{k}"] for k in ("I", "A", "B", "P1", "P2")}
                    I = E.to_model_scale(s["I"])
                    cx, cy, gu, gv = E.gt_windows(s["P1"], s["P2"], I.shape)
                    valid = np.isfinite(gu) & (np.hypot(gu, gv) > 2)
                    m, c = run_one(model, I, (cx, cy, gu, gv), valid, r, seed=i)
                    samples.append(m); cls.append(c)
                    if i == 0 and r == a.r[0]:
                        cu, cv = multipass_piv(s["A"].astype(np.float64), s["B"].astype(np.float64))[2:]
                        diag = dict(true=div_ratio(np.nan_to_num(gu), np.nan_to_num(gv), valid),
                                    cc_true_frames=div_ratio(cu, cv, valid))
            else:
                if ds == "caseC":
                    import evaluate_pivc as PC
                    frames, M = PC.load_case()
                    pairs = [frames["bgsub"]]
                else:
                    P = R.pairs()
                    keys = ["test1"] if ds == "test1" else list(EP.TEST2_EVAL)
                    pairs = [R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in P[k])) for k in keys]
                rat = []
                for i, (A, B) in enumerate(pairs):
                    cx, cy, ru, rv = multipass_piv(A, B)
                    valid = ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2)
                    fit_mask = None
                    if ds == "caseC":
                        fit_mask = PC.centre_mask(M, cx, cy)
                        valid &= fit_mask
                    m, c = run_one(model, R.make_double(A, B, r), (cx, cy, ru, rv), valid, r, fit_mask=fit_mask, seed=i)
                    samples.append(m); cls.append(c)
                    if r == a.r[0]:
                        rat.append(div_ratio(ru, rv, valid))
                if rat:
                    diag = dict(reference={k: float(np.mean([x[k] for x in rat])) for k in rat[0]})
            t = summarize(samples)
            t["_clusters_mp+uod"] = merge_clusters(cls)
            out[f"{ds}/r{r}"] = t
            if diag:
                out[f"{ds}/_div"] = diag
            print(f"{ds} r={r} ({time.time() - t0:.0f}s) n={t['mp']['n']}", flush=True)
            for k in ("mp", "mp+uod", "mp+uod+pinn", "mp+pinn"):
                if k in t:
                    x = t[k]
                    print(f"   {k:12s} out {x['outlier']:.3f} sgn {x['sign_err']:.3f} med {x['median']:.3f} "
                          f"rms {x['rms']:.3f} vort {x['vort_rms']:.4f}", flush=True)
            print(f"   clusters(mp+uod) {t['_clusters_mp+uod']}", flush=True)
            if diag:
                print(f"   div {diag}", flush=True)
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
