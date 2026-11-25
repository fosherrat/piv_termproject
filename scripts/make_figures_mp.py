"""Figures: single-pass vs multi-pass + window deformation (synthetic and real).
python scripts/make_figures_mp.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import evaluate_real as R
from pivsep.separation import SepUNet

SYN = [("asym", "nonlinear auto-corr", "#3b7dd8", "-"),
       ("gt_cc", "true A/B, single-pass", "#888888", "--"),
       ("gt_mp", "true A/B, multi-pass", "#444444", "-"),
       ("unet_cc:wide_bce", "U-Net split, single-pass", "#b896e0", "--"),
       ("unet_mp:wide_bce", "U-Net split, multi-pass", "#7b3fb5", "-")]
REAL = [("asym", "nonlinear auto-corr", "#3b7dd8"),
        ("auto_or", "auto-corr, oracle sign", "#e0c96a"),
        ("unet_ccm:finetuned", "U-Net (fine-tuned), single-pass", "#b896e0"),
        ("unet_mp:finetuned", "U-Net (fine-tuned), multi-pass", "#7b3fb5"),
        ("unet_mp:finetuned+uod", "  + UOD", "#4b1d80")]


def fig_synth(ev, out):
    sweeps = [("r", "intensity ratio r"), ("ppp", "particles per pixel"), ("noise", "noise σ")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    for j, (key, xl) in enumerate(sweeps):
        xs = [float(x) for x in ev[key]]
        for k, lab, c, ls in SYN:
            ax[j].plot(xs, [ev[key][x][k]["outlier"] for x in ev[key]], ls, marker="o", c=c, label=lab)
        ax[j].set_xlabel(xl); ax[j].set_ylabel("outlier rate (>1 px)"); ax[j].grid(alpha=.3)
        ax[j].set_ylim(-0.01, 0.36)
        if key == "r":
            ax[j].invert_xaxis()
    t = ev["r"]["0.7"]
    edges = t["grad_bins"]["edges"]; cnt = t["grad_bins"]["count"]
    labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    labels = [f"{l}\n(n={n})" for l, n in zip(labels, cnt)]
    w = 0.8 / len(SYN)
    for i, (k, lab, c, _) in enumerate(SYN):
        ax[3].bar(np.arange(4) + (i - len(SYN) / 2 + .5) * w, t[k]["outlier_by_grad"], w, color=c, label=lab)
    ax[3].set_xticks(range(4)); ax[3].set_xticklabels(labels, fontsize=8)
    ax[3].set_xlabel("velocity difference across window [px] (r = 0.7)"); ax[3].set_ylabel("outlier rate")
    ax[3].grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=8)
    fig.suptitle("synthetic: single-pass vs multi-pass + window deformation")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_real_bars(ev, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
    for a, g in zip(ax, ["test1", "test2"]):
        conds = [k for k in ev if k.startswith(g)]
        w = 0.8 / len(REAL)
        for i, (k, lab, c) in enumerate(REAL):
            a.bar(np.arange(len(conds)) + (i - len(REAL) / 2 + .5) * w, [ev[cd][k]["outlier"] for cd in conds],
                  w, color=c, label=lab)
        a.set_xticks(range(len(conds))); a.set_xticklabels([cd.split("/")[1].replace("r", "r = ") for cd in conds])
        a.set_ylabel("outlier rate vs multi-pass reference"); a.grid(alpha=.3, axis="y")
        a.set_title({"test1": "test1 (experiment never used in training)",
                     "test2": "test2 pairs 3–5 (held-out instants)"}[g], fontsize=10)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_real_vectors(out, pair="test1", r=0.7):
    ck = torch.load("results/sep_unet_ft_real.pt", map_location="cpu")
    m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()[pair]))
    _, d = R.evaluate_pair({"finetuned": m}, A, B, r, multipass=True)
    I, cx, cy, (ru, rv), valid, res = d["I"], d["cx"], d["cy"], d["ref"], d["valid"], d["res"]
    show = [("reference: multi-pass CC of real A, B", (ru, rv)), ("nonlinear auto-corr", res["asym"]),
            ("fine-tuned U-Net, single-pass", res["unet_ccm:finetuned"]),
            ("fine-tuned U-Net, multi-pass", res["unet_mp:finetuned"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for a, (name, (u, v)) in zip(ax, show):
        e = np.hypot(u - ru, v - rv)
        a.imshow(I, cmap="gray", vmin=0, vmax=1.0)
        q = a.quiver(cx, cy, u, -v, np.where(valid, e, np.nan), cmap="autumn_r", clim=(0, 2), scale=120, width=.004)
        t = name if name.startswith("reference") else f"{name}\noutlier {np.mean(e[valid] > 1):.1%}"
        a.set_title(t, fontsize=9); a.axis("off")
        a.set_xlim(0, cx.max() + 16); a.set_ylim(cy.max() + 16, 0)
    fig.colorbar(q, ax=ax, fraction=.012, label="deviation from reference [px]")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_synth(json.load(open("results/eval_mp_synth.json")), "results/fig_mp_synth.png")
    fig_real_bars(json.load(open("results/eval_mp_real.json")), "results/fig_mp_real_bars.png")
    fig_real_vectors("results/fig_mp_real_vectors.png")
    print("figures written")
