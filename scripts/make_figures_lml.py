"""Figures for the LML-generator validation (run evaluate_lml.py first).
python scripts/make_figures_lml.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [("asym", "nonlinear auto-corr", "#3b7dd8"),
           ("auto_or", "auto-corr, oracle sign", "#e0c96a"),
           ("unet_ccm:zeroshot", "U-Net zero-shot, single-pass", "#c9b8e8"),
           ("unet_mp:zeroshot", "U-Net zero-shot, multi-pass", "#7b3fb5"),
           ("unet_mp:finetuned", "U-Net fine-tuned, multi-pass", "#4b1d80"),
           ("gt_mp", "true A/B, multi-pass (upper bound)", "#444444")]


def main():
    ev = json.load(open("results/eval_lml.json"))
    F = np.load("results/lml_example.npz")
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 4)
    # bars: r sweep for both flows, density sweep
    panels = [("translation", [f"translation/r{r}/ppp0.01" for r in (0.9, 0.7, 0.5)], ["r 0.9", "r 0.7", "r 0.5"]),
              ("plasma", [f"plasma/r{r}/ppp0.01" for r in (0.9, 0.7, 0.5)], ["r 0.9", "r 0.7", "r 0.5"]),
              ("density (r 0.7)", [f"{f}/r0.7/ppp{p}" for f in ("translation", "plasma") for p in (0.005, 0.01, 0.02)],
               [f"{f[:5]}.\n{p} ppp" for f in ("translation", "plasma") for p in (0.005, 0.01, 0.02)])]
    for j, (title, conds, labels) in enumerate(panels):
        a = fig.add_subplot(gs[0, j] if j < 2 else gs[0, 2:])
        w = 0.8 / len(METHODS)
        for i, (k, lab, c) in enumerate(METHODS):
            x = np.arange(len(conds)) + (i - len(METHODS) / 2 + .5) * w
            a.bar(x, [ev[cd][k]["outlier"] for cd in conds], w, color=c, label=lab)
            a.bar(x, [ev[cd][k]["sign_err"] for cd in conds], w, color="none", edgecolor="k", hatch="///", lw=0.4)
        a.set_xticks(range(len(conds))); a.set_xticklabels(labels, fontsize=8)
        a.set_title(title); a.set_ylim(0, 1.0); a.grid(alpha=.3, axis="y")
        if j == 0:
            a.set_ylabel("outlier rate (>1 px); hatched = direction error")
        if j == 2:
            a.legend(fontsize=8, loc="upper right")
    # example (plasma, r 0.7)
    f = "plasma"
    cx, cy, gu, gv, valid = (F[f"{f}|{k}"] for k in ("cx", "cy", "gu", "gv", "valid"))
    valid = valid.astype(bool)
    a = fig.add_subplot(gs[1, 0]); a.imshow(F[f"{f}|I"], cmap="gray")
    a.set_title("LML double exposure (plasma flow, r = 0.7)")
    a = fig.add_subplot(gs[1, 1]); a.imshow(F[f"{f}|I"][96:160, 96:160], cmap="gray", interpolation="nearest")
    a.set_title("zoom 64 x 64: particle pairs overlap")
    for j, (k, t) in enumerate([("asym", "nonlinear auto-corr"), ("unet_mp:zeroshot", "U-Net zero-shot, multi-pass")]):
        a = fig.add_subplot(gs[1, 2 + j])
        u, v = F[f"{f}|{k}|u"], F[f"{f}|{k}|v"]
        a.imshow(F[f"{f}|I"], cmap="gray", alpha=.35)
        a.quiver(cx, cy, gu, gv, color="#999999", angles="xy", scale_units="xy", scale=0.35, width=0.004)
        bad = (np.hypot(u - gu, v - gv) > 1) & valid
        ok = valid & ~bad
        for m, c in ((ok, "#2060c0"), (bad, "#e03030")):
            a.quiver(cx[m], cy[m], u[m], v[m], color=c, angles="xy", scale_units="xy", scale=0.35, width=0.004)
        a.set_title(f"{t}: red = outlier ({bad.sum() / valid.sum():.0%}), grey = truth", fontsize=9)
    for a in fig.axes[3:]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Independent synthetic test: LML generator (PGlml / DGlml / SIGlml, Lille) + pulse coding")
    fig.tight_layout(); fig.savefig("results/fig_lml.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
