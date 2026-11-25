"""Figures for stage 2 (div-free post-processing).
python scripts/make_figures_postproc.py --model results/sep_unet_wide_bce.pt
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pivsep.synth import SynthParams, make_sample, gt_window_displacement
from pivsep.correlation import cross_piv
from pivsep.separation import SepUNet, unet_split
from pivsep.postproc import pinn_fit, stream_ls, vorticity

POST = ["raw", "uod", "ls", "uod+ls", "pinn", "uod+pinn"]
PCOL = {"raw": "#9a9a9a", "uod": "#d0623a", "ls": "#7fb3e6", "uod+ls": "#3b7dd8",
        "pinn": "#b896e0", "uod+pinn": "#7b3fb5"}
COND_LABEL = {"r0.7": "r = 0.7", "r0.9": "r = 0.9", "noise0.05": "r = 0.7, noise 0.05"}


def fig_bars(ev, out):
    metrics = [("rms", "vector RMS error [px]"), ("outlier", "outlier rate (>1 px)"),
               ("vort_rms", "vorticity RMS error [1/Δt]")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    conds = list(ev)
    for r, inp in enumerate(["unet", "asym"]):
        for c, (m, yl) in enumerate(metrics):
            a = ax[r, c]
            w = 0.8 / len(POST)
            for i, p in enumerate(POST):
                vals = [ev[cd][f"{inp}/{p}"][m] for cd in conds]
                a.bar(np.arange(len(conds)) + (i - len(POST) / 2 + .5) * w, vals, w, color=PCOL[p], label=p)
            a.set_xticks(range(len(conds))); a.set_xticklabels([COND_LABEL[cd] for cd in conds])
            a.set_ylabel(yl); a.grid(alpha=.3, axis="y")
            a.set_title(("U-Net (BCE) split → CC" if inp == "unet" else "nonlinear auto-corr") + " input", fontsize=10)
            if m == "rms":
                a.set_yscale("log")
    ax[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_vort(model, out, seed=1003):
    rng = np.random.default_rng(seed)
    s = make_sample(rng, SynthParams(r=0.7, noise=0.05))
    cx, cy, u, v = cross_piv(*unet_split(model, s["img"]), 32, 16)
    gu, gv = gt_window_displacement(s["flow"], cx, cy, 32)
    h = 16
    fields = {"ground truth": (gu, gv), "U-Net → CC (raw)": (u, v),
              "+ stream-function LS": stream_ls(u, v, h),
              "+ div-free PINN": pinn_fit(cx, cy, u, v, steps=800)}
    wg = vorticity(gu, gv, h)
    lim = np.abs(wg).max() * 1.1
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    for a, (name, (fu, fv)) in zip(ax, fields.items()):
        w = vorticity(fu, fv, h)
        im = a.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper",
                      extent=[cx.min() - h / 2, cx.max() + h / 2, cy.max() + h / 2, cy.min() - h / 2])
        a.quiver(cx, cy, fu, -fv, scale=150, width=.004)
        err = np.sqrt(np.mean((w - wg) ** 2))
        a.set_title(name if name == "ground truth" else f"{name}\nvorticity RMS err {err:.4f}", fontsize=10)
        a.axis("off")
    fig.colorbar(im, ax=ax, fraction=.015, label="vorticity")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet_wide_bce.pt")
    ap.add_argument("--eval", default="results/eval_postproc.json")
    a = ap.parse_args()
    fig_bars(json.load(open(a.eval)), "results/fig_postproc_bars.png")
    ck = torch.load(a.model, map_location="cpu")
    m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    fig_vort(m, "results/fig_postproc_vorticity.png")
    print("figures written")


if __name__ == "__main__":
    main()
