"""Independent synthetic validation with the LML image generator (S. Coudert, Lille):
PGlml (particles) -> DGlml (displacement, double exposure) -> SIGlml (rendering).
Build the tools first:  scripts/build_lml.sh

Everything about the images comes from the LML tools (particle size/brightness distribution,
Gaussian rendering with tail cut-off at grey level 35, additive sprites, uniform noise,
background ramp, 8-bit saturation, displacement fields).  Our only change: pulse coding, i.e.
the level of every exposure-2 particle is multiplied by r before rendering
(DGlml's own double exposure uses equal levels -> no direction information).

Differences from our own generator (pivsep/synth.py) that make this an out-of-distribution test:
  particle sigma 1-2 px (diameter 4-8 px, ours 2.8 +- 0.5), brightness uniform 200-255 with no
  light-sheet profile / out-of-plane loss, 8-bit quantisation + saturation, background ramp,
  displacement "plasma" field (random, not divergence-free, nearest-pixel lookup, no mean flow
  -> direction changes inside the field).

The LML tools seed their RNG from the clock, so generated samples are cached
(--cache, npz with images and particle lists) and evaluation always reads the cache.

Ground truth per window: mean displacement of the particles whose mid-point lies in the window,
weighted by particle intensity (level * sigma^2).  Evaluated where |d_gt| > 2 px.

python scripts/evaluate_lml.py --model zeroshot=results/sep_unet_wide_bce.pt finetuned=results/sep_unet_ft_real.pt
"""
import argparse, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from pivsep.correlation import cross_piv, auto_piv, asym_auto_piv, multipass_piv, window_grid
from pivsep.separation import SepUNet, unet_split
from pivsep.postproc import uod_median
import evaluate_real as R

WIN, STEP = 32, 16
GRAD_BINS = [0, 1, 2, 4, 1e9]


# ----------------------------------------------------------------------------- generation
def _run(cmd, inp=None):
    p = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-500:]}")
    return p.stdout


def _read_cimg(path):
    """CImg 1.3 binary: '1 float little_endian\\n<w> <h> <d> <v>\\n' + data (x fastest, v slowest)."""
    with open(path, "rb") as f:
        f.readline()
        w, h, d, v = map(int, f.readline().split())
        a = np.frombuffer(f.read(4 * w * h * d * v), "<f4").reshape(v, w * h * d)
    return a.T.copy()                                   # (n, 4): x, y, sigma, level


def _parse(txt):
    return np.array([list(map(float, l.split())) for l in txt.strip().splitlines()])


def _render(sig, P, size, tmp):
    out = os.path.join(tmp, "img.png")
    txt = "\n".join(f"{x:.4f} {y:.4f} {s:.4f} {l:.4f}" for x, y, s, l in P)
    _run([sig, "-i", "stdin", "-n", str(len(P)), "-W", str(size), "-H", str(size), "-o", out], txt)
    return np.array(Image.open(out)).astype(np.float32)


def generate(bin_dir, flow, r, rng, size=256, ppp=0.01, pad=10.5, d_range=(4.0, 8.0),
             plasma_amp=8.0, plasma_struct=64.0):
    PG, DG, SIG = (os.path.join(bin_dir, t, t) for t in ("PGlml", "DGlml", "SIGlml"))
    n = int(ppp * (size + 2 * pad) ** 2)
    L = size + 2 * pad                 # particles are generated in [0, L) and shifted by -pad afterwards:
    with tempfile.TemporaryDirectory() as tmp:   # DGlml plasma reads its map at (int)x,(int)y without
        pc = os.path.join(tmp, "p.cimg")         # bounds check -> segfault for negative coordinates
        _run([PG, "-n", str(n), "-x", "0", "-X", str(L), "-y", "0", "-Y", str(L), "-o", pc])
        assert _read_cimg(pc).shape == (n, 4)
        if flow == "translation":
            mag, ang = rng.uniform(*d_range), rng.uniform(0, 2 * np.pi)
            args = ["--function", "translation", "--dx", f"{mag * np.cos(ang):.4f}", "--dy", f"{mag * np.sin(ang):.4f}"]
        else:
            args = ["--function", "plasma", "--dx", str(plasma_amp), "--dy", str(plasma_struct),
                    "-W", str(int(L) + 8), "-H", str(int(L) + 8)]
        P = _parse(_run([DG, "-i", pc, "-d", "1", "-o", "stdout"] + args))
        P[:, :2] -= pad
        P1, P2 = P[:n].copy(), P[n:].copy()                # DGlml appends exposure 2 after exposure 1
        P2[:, 3] *= r                                      # pulse coding (the only change to LML output)
        I = _render(SIG, np.vstack([P1, P2]), size, tmp)   # one double exposure, one noise realisation
        A = _render(SIG, P1, size, tmp)                    # separately rendered exposures (upper bound only)
        B = _render(SIG, P2, size, tmp)
    return dict(I=I.astype(np.uint8), A=A.astype(np.uint8), B=B.astype(np.uint8),
                P1=P1.astype(np.float32), P2=P2.astype(np.float32))


def build_cache(path, bin_dir, conds, n_img, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for flow, r, ppp in conds:
        for i in range(n_img):
            s = generate(bin_dir, flow, r, rng, ppp=ppp)
            for k, v in s.items():
                data[f"{flow}|{r}|{ppp}|{i}|{k}"] = v
        print("generated", flow, r, ppp, flush=True)
    np.savez_compressed(path, **data)


# ----------------------------------------------------------------------------- evaluation
def gt_windows(P1, P2, shape):
    sy, sx, cx, cy = window_grid(shape, WIN, STEP)
    mid = (P1[:, :2] + P2[:, :2]) / 2
    d = P2[:, :2] - P1[:, :2]
    w = P1[:, 3] * P1[:, 2] ** 2
    gu = np.full(cx.shape, np.nan); gv = np.full(cx.shape, np.nan)
    for (i, j), y0 in np.ndenumerate(sy):
        x0 = sx[i, j]
        m = (mid[:, 0] >= x0) & (mid[:, 0] < x0 + WIN) & (mid[:, 1] >= y0) & (mid[:, 1] < y0 + WIN)
        if m.any():
            gu[i, j] = np.average(d[m, 0], weights=w[m]); gv[i, j] = np.average(d[m, 1], weights=w[m])
    return cx, cy, gu, gv


def to_model_scale(img8):
    a = R.preprocess(img8.astype(np.float64))
    return np.clip(a * (0.6 / np.percentile(a, 99.7)) + 0.02, 0, 1).astype(np.float32)


def evaluate_sample(models, s):
    I = to_model_scale(s["I"])
    A, B = s["A"].astype(np.float64), s["B"].astype(np.float64)
    cx, cy, gu, gv = gt_windows(s["P1"], s["P2"], I.shape)
    res = {"gt_cc": cross_piv(A, B, WIN, STEP)[2:], "gt_mp": multipass_piv(A, B)[2:]}
    ua, va = auto_piv(I, WIN, STEP)[2:]
    pick = (ua * gu + va * gv) < 0
    res["auto_or"] = (np.where(pick, -ua, ua), np.where(pick, -va, va))
    res["asym"] = asym_auto_piv(I, WIN, STEP)[2:]
    for name, m in models.items():
        sA, sB = unet_split(m, I)
        res[f"unet_ccm:{name}"] = cross_piv(sA, sB, WIN, STEP, mask_r=3)[2:]
        u, v = multipass_piv(sA, sB, mask_r=3)[2:]
        res[f"unet_mp:{name}"] = (u, v)
        res[f"unet_mp:{name}+uod"] = uod_median(u, v)[:2]
    valid = np.isfinite(gu) & (np.hypot(gu, gv) > 2.0)
    h = cx[0, 1] - cx[0, 0]
    G = WIN * np.sqrt(sum(np.gradient(np.nan_to_num(f), h, axis=ax) ** 2 for f in (gu, gv) for ax in (0, 1)))
    out = {"grad": G[valid]}
    for k, (u, v) in res.items():
        out[k] = dict(err=np.hypot(u - gu, v - gv)[valid], sign=((u * gu + v * gv) < 0)[valid])
    return out, (cx, cy, gu, gv, valid, res)


def summarize(outs):
    G = np.concatenate([o["grad"] for o in outs])
    t = {"grad_bins": {"edges": GRAD_BINS[:-1], "count": [int(((G >= a) & (G < b)).sum())
                                                          for a, b in zip(GRAD_BINS[:-1], GRAD_BINS[1:])]}}
    for k in [k for k in outs[0] if k != "grad"]:
        e = np.concatenate([o[k]["err"] for o in outs]); sg = np.concatenate([o[k]["sign"] for o in outs])
        t[k] = dict(n=int(e.size), median=float(np.median(e)), outlier=float(np.mean(e > 1)),
                    sign_err=float(np.mean(sg)),
                    outlier_by_grad=[float(np.mean(e[(G >= a) & (G < b)] > 1)) if ((G >= a) & (G < b)).any()
                                     else None for a, b in zip(GRAD_BINS[:-1], GRAD_BINS[1:])])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["zeroshot=results/sep_unet_wide_bce.pt",
                                                    "finetuned=results/sep_unet_ft_real.pt"])
    ap.add_argument("--lml-bin", default="third_party/lml")
    ap.add_argument("--cache", default="data/lml/lml_samples.npz")
    ap.add_argument("--n-img", type=int, default=12)
    ap.add_argument("--r", nargs="+", type=float, default=[0.9, 0.7, 0.5])
    ap.add_argument("--flows", nargs="+", default=["translation", "plasma"])
    ap.add_argument("--ppp", type=float, default=0.01, help="main density (~10 particles per 32x32 window)")
    ap.add_argument("--ppp-sweep", nargs="+", type=float, default=[0.005, 0.02], help="extra densities at r = 0.7")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--out", default="results/eval_lml.json")
    ap.add_argument("--dump", default="results/lml_example.npz")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    conds = [(f, r, a.ppp) for f in a.flows for r in a.r] + [(f, 0.7, p) for f in a.flows for p in a.ppp_sweep]
    if a.regenerate or not os.path.exists(a.cache):
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        build_cache(a.cache, a.lml_bin, conds, a.n_img)
    C = np.load(a.cache)
    models = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu")
        m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
        models[name] = m
    table, dump = {}, {}
    for flow, r, ppp in conds:
        outs = []
        for i in range(a.n_img):
            s = {k: C[f"{flow}|{r}|{ppp}|{i}|{k}"] for k in ("I", "A", "B", "P1", "P2")}
            o, extra = evaluate_sample(models, s)
            outs.append(o)
            if i == 0 and r == 0.7 and ppp == a.ppp:
                cx, cy, gu, gv, valid, res = extra
                dump.update({f"{flow}|I": s["I"], f"{flow}|A": s["A"], f"{flow}|cx": cx, f"{flow}|cy": cy,
                             f"{flow}|gu": gu, f"{flow}|gv": gv, f"{flow}|valid": valid})
                for k, (u, v) in res.items():
                    dump[f"{flow}|{k}|u"] = u; dump[f"{flow}|{k}|v"] = v
        t = summarize(outs)
        table[f"{flow}/r{r}/ppp{ppp}"] = t
        print(f"{flow} r={r} ppp={ppp} (n={t['asym']['n']}, grad bins {t['grad_bins']['count']}):\n   " + "\n   ".join(
            f"{k:24s} out {v['outlier']:.3f} sgn {v['sign_err']:.3f} med {v['median']:.3f}"
            for k, v in t.items() if k != "grad_bins"), flush=True)
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1)
    np.savez_compressed(a.dump, **{k: np.asarray(v) for k, v in dump.items()})
    print("saved", a.out, a.dump)


if __name__ == "__main__":
    main()
