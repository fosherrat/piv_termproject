"""Figures for the pulse-separation experiment.
python scripts/make_figures.py --model results/sep_unet.pt --eval results/eval.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pivsep.synth import SynthParams, make_sample
from pivsep.separation import SepUNet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run_image, METHODS

LABEL = {"gt_cc": "true A/B → CC (upper bound)", "auto": "auto-corr (sign by convention)",
         "auto_or": "auto-corr (oracle sign)", "asym": "nonlinear auto-corr  ΣI²(x)I(x+s)",
         "thr_cc": "threshold split → CC", "unet_cc": "U-Net split → CC (proposed)"}
COLOR = {"gt_cc": "#888888", "auto": "#c9a227", "auto_or": "#e0c96a", "asym": "#3b7dd8",
         "thr_cc": "#d0623a", "unet_cc": "#2a9d5c",
         "unet_cc:v1": "#8fd19e", "unet_cc:wide_mse": "#2a9d5c", "unet_cc:wide_bce": "#7b3fb5"}
LABEL.update({"unet_cc:v1": "U-Net v1 (narrow range, MSE)", "unet_cc:wide_mse": "U-Net wide range, MSE",
              "unet_cc:wide_bce": "U-Net wide range, BCE"})


def _methods(table):
    return [k for k in table if k not in ("grad_bins", "sep_wrong", "auto_or")]


def rgb(a, b, gain=1.6):
    """A -> red/magenta, B -> green/cyan composite."""
    a = np.clip(a * gain, 0, 1); b = np.clip(b * gain, 0, 1)
    return np.stack([a, b, 0.5 * (a + b)], -1)


def fig_example(model, out):
    rng = np.random.default_rng(1003)
    s = make_sample(rng, SynthParams(r=0.7))
    _, _, (cx, cy, gu, gv, res, tA, tB, uA, uB) = run_image(model, s)
    sl = (slice(80, 176), slice(80, 176))
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    ax[0].imshow(s["img"][sl], cmap="gray", vmin=0, vmax=0.8); ax[0].set_title("double exposure (r = 0.7)")
    ax[1].imshow(rgb(s["A"][sl], s["B"][sl])); ax[1].set_title("ground truth  (A red, B green)")
    ax[2].imshow(rgb(tA[sl], tB[sl])); ax[2].set_title("threshold split")
    ax[3].imshow(rgb(uA[sl], uB[sl])); ax[3].set_title("U-Net split")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, k in zip(ax, ["gt_cc", "auto", "asym", "unet_cc"]):
        u, v = (gu, gv) if k == "gt_cc" else res[k]
        e = np.hypot(u - gu, v - gv)
        a.imshow(s["img"], cmap="gray", vmin=0, vmax=1.2)
        q = a.quiver(cx, cy, u, -v, e, cmap="autumn_r", clim=(0, 2), scale=150, width=.004)
        a.set_title("ground truth" if k == "gt_cc" else f"{LABEL[k]}\noutlier {np.mean(e > 1):.0%}", fontsize=9)
        a.axis("off")
    fig.colorbar(q, ax=ax, fraction=.015, label="error [px]")
    fig.savefig(out.replace(".png", "_vectors.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_sweep(ev, out):
    sweeps = [("r", "intensity ratio r = I₂/I₁"), ("ppp", "particles per pixel"), ("noise", "noise σ")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, (key, xl) in enumerate(sweeps):
        xs = [float(x) for x in ev[key]]
        for k in _methods(next(iter(ev[key].values()))):
            ax[0, j].plot(xs, [ev[key][x][k]["outlier"] for x in ev[key]], "o-", c=COLOR[k], label=LABEL[k])
            if k != "gt_cc":
                ax[1, j].plot(xs, [ev[key][x][k]["sign_err"] for x in ev[key]], "o-", c=COLOR[k], label=LABEL[k])
        ax[0, j].set_ylabel(f"outlier rate (|err| > 1 px)"); ax[1, j].set_ylabel("wrong-direction rate")
        for a in ax[:, j]:
            a.set_xlabel(xl); a.grid(alpha=.3); a.set_ylim(-0.02, 1.02)
        if key == "r":
            for a in ax[:, j]: a.invert_xaxis()
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_grad(ev, out, cond=("r", "0.7")):
    """Outlier rate vs velocity difference across one interrogation window."""
    t = ev[cond[0]][cond[1]]
    edges = t["grad_bins"]["edges"]; cnt = t["grad_bins"]["count"]
    labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    labels = [f"{l} px\n(n={c})" for l, c in zip(labels, cnt)]
    ms = _methods(t)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(ms)
    for i, k in enumerate(ms):
        vals = [np.nan if v is None else v for v in t[k]["outlier_by_grad"]]
        ax.bar(np.arange(len(labels)) + (i - len(ms) / 2 + .5) * w, vals, w, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("velocity difference across one 32-px window, |∇d|·32"); ax.set_ylabel("outlier rate")
    ax.set_title(f"outlier rate vs local velocity gradient ({cond[0]} = {cond[1]})")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/sep_unet.pt")
    ap.add_argument("--eval", default="results/eval.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    model = SepUNet(base=ck["base"]); model.load_state_dict(ck["state"]); model.eval()
    fig_example(model, f"results/fig_example{a.tag}.png")
    if os.path.exists(a.eval):
        ev = json.load(open(a.eval))
        fig_sweep(ev, f"results/fig_sweep{a.tag}.png")
        if "grad_bins" in ev["r"]["0.7"]:
            fig_grad(ev, f"results/fig_grad{a.tag}.png")
    print("figures written to results/")


if __name__ == "__main__":
    main()
