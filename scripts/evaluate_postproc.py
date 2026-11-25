"""Stage 2: mass-conservation (div u = 0) post-processing of PIV vector fields.

Input fields : U-Net split -> CC (proposed stage-1 output)  and  nonlinear auto-corr (classical)
Post-processing:
  raw       : no post-processing
  uod       : universal outlier detection + median replacement (standard)
  ls        : robust (Huber) least-squares stream-function fit   -> div u = 0
  uod+ls    : uod, then stream-function fit
  pinn      : div-free MLP stream function (Huber + vorticity smoothness)
  uod+pinn  : uod, then PINN
Metrics: vector error (median/rms/outlier), mean |div u|, vorticity RMS error, and
         dense (2x finer grid) RMS error: continuous fit vs cubic interpolation.

python scripts/evaluate_postproc.py --model results/sep_unet_wide_mse.pt
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator

from pivsep.synth import SynthParams, make_sample, gt_window_displacement
from pivsep.correlation import cross_piv, asym_auto_piv
from pivsep.separation import SepUNet, unet_split
from pivsep.postproc import uod_median, stream_ls, pinn_fit, divergence, vorticity

WIN, STEP = 32, 16
POST = ["raw", "uod", "ls", "uod+ls", "pinn", "uod+pinn"]
CONDS = {"r0.7": dict(r=0.7), "r0.9": dict(r=0.9), "noise0.05": dict(r=0.7, noise=0.05)}
_MODEL = None


def _init(path):
    global _MODEL
    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu")
    _MODEL = SepUNet(base=ck["base"]); _MODEL.load_state_dict(ck["state"]); _MODEL.eval()


def _cubic(cx, cy, u, v, px, py):
    out = []
    for f in (u, v):
        it = RegularGridInterpolator((cy[:, 0], cx[0]), f, method="cubic")
        out.append(it(np.stack([py.ravel(), px.ravel()], 1)).reshape(px.shape))
    return out


def job(args):
    cond, i = args
    rng = np.random.default_rng(1000 + i)
    s = make_sample(rng, SynthParams(**CONDS[cond]))
    cx, cy, *_ = cross_piv(s["A"], s["B"], WIN, STEP)
    gu, gv = gt_window_displacement(s["flow"], cx, cy, WIN)
    h = STEP
    # dense evaluation points: midpoints between window centres (interior)
    mx = (cx[:-1, :-1] + cx[1:, 1:]) / 2
    my = (cy[:-1, :-1] + cy[1:, 1:]) / 2
    dgu, dgv = gt_window_displacement(s["flow"], mx, my, WIN)
    inputs = {"unet": cross_piv(*unet_split(_MODEL, s["img"]), WIN, STEP)[2:],
              "asym": asym_auto_piv(s["img"], WIN, STEP)[2:]}
    wg = vorticity(gu, gv, h)
    rec = {}
    for name, (u, v) in inputs.items():
        uo, vo, _ = uod_median(u, v)
        fields, dense = {}, {}
        fields["raw"] = (u, v)
        fields["uod"] = (uo, vo)
        fields["ls"] = stream_ls(u, v, h)
        fields["uod+ls"] = stream_ls(uo, vo, h)
        for k, (a, b) in (("pinn", (u, v)), ("uod+pinn", (uo, vo))):
            fu, fv, du, dv = pinn_fit(cx, cy, a, b, steps=800, eval_xy=(mx, my), seed=i)
            fields[k] = (fu, fv); dense[k] = (du, dv)
        for k in ("raw", "uod", "ls", "uod+ls"):
            dense[k] = _cubic(cx, cy, *fields[k], mx, my)
        for k in POST:
            a, b = fields[k]
            e = np.hypot(a - gu, b - gv).ravel()
            ed = np.hypot(dense[k][0] - dgu, dense[k][1] - dgv).ravel()
            rec[(name, k)] = dict(err=e, dense=ed, div=np.abs(divergence(a, b, h)).ravel(),
                                  werr=(vorticity(a, b, h) - wg).ravel())
    rec[("gt", "gt")] = dict(err=np.zeros(1), dense=np.zeros(1),
                             div=np.abs(divergence(gu, gv, h)).ravel(), werr=np.zeros(1))
    return cond, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet_wide_mse.pt")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/eval_postproc.json")
    a = ap.parse_args()
    jobs = [(c, i) for c in CONDS for i in range(a.n_img)]
    with Pool(a.procs, initializer=_init, initargs=(a.model,)) as pool:
        results = pool.map(job, jobs)
    table = {}
    for cond in CONDS:
        recs = [r for c, r in results if c == cond]
        t = {}
        for key in recs[0]:
            cat = lambda f: np.concatenate([r[key][f] for r in recs])
            e, ed, dv, we = cat("err"), cat("dense"), cat("div"), cat("werr")
            t["/".join(key)] = dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e**2))),
                                    outlier=float(np.mean(e > 1)), dense_rms=float(np.sqrt(np.mean(ed**2))),
                                    div=float(dv.mean()), vort_rms=float(np.sqrt(np.mean(we**2))))
        table[cond] = t
        print(f"\n== {cond} ==")
        print(f"{'input/post':16s} {'median':>7s} {'rms':>7s} {'outl':>6s} {'dense':>7s} {'|div|':>7s} {'vortE':>7s}")
        for k, m in t.items():
            print(f"{k:16s} {m['median']:7.3f} {m['rms']:7.3f} {m['outlier']:6.3f} {m['dense_rms']:7.3f} "
                  f"{m['div']:7.4f} {m['vort_rms']:7.4f}")
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
