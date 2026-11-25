"""Figures for the PIV Challenge case C validation (run evaluate_pivc.py first).
Also splits the error by reflection strength (mean of background image Cbg_1 in each window).
python scripts/make_figures_pivc.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from evaluate_pivc import DATA

METHODS = [("asym", "nonlinear auto-corr", "#3b7dd8"),
           ("auto_or", "auto-corr, oracle sign", "#e0c96a"),
           ("unet_ccm:zeroshot", "U-Net zero-shot, single-pass", "#c9b8e8"),
           ("unet_mp:zeroshot", "U-Net zero-shot, multi-pass", "#a07ad0"),
           ("unet_ccm:finetuned", "U-Net fine-tuned, single-pass", "#b896e0"),
           ("unet_mp:finetuned", "U-Net fine-tuned, multi-pass", "#7b3fb5"),
           ("unet_mp:finetuned+uod", "  + UOD", "#4b1d80")]


def reflection_level(cx, cy):
    """Mean background (reflection / flare) intensity in each 32x32 window, 0-255."""
    bg = tifffile.imread(os.path.join(DATA, "Cbg_1.tif")).astype(float)
    m = ndimage.uniform_filter(bg, 32)
    return m[np.round(cy).astype(int), np.round(cx).astype(int)]


def main():
    ev = json.load(open("results/eval_pivc.json"))
    F = np.load("results/pivc_fields.npz")
    cx, cy, ru, rv, valid = F["cx"], F["cy"], F["ru"], F["rv"], F["valid"].astype(bool)

    # ---- bars
    fig, ax = plt.subplots(1, 2, figsize=(17, 4.5), sharey=True)
    for a, var in zip(ax, ["bgsub", "raw"]):
        conds = [k for k in ev["results"] if k.startswith(var)]
        w = 0.8 / len(METHODS)
        for i, (k, lab, c) in enumerate(METHODS):
            x = np.arange(len(conds)) + (i - len(METHODS) / 2 + .5) * w
            a.bar(x, [ev["results"][cd][k]["outlier"] for cd in conds], w, color=c, label=lab)
            a.bar(x, [ev["results"][cd][k]["sign_err"] for cd in conds], w, color="none", edgecolor="k",
                  hatch="///", lw=0.4)
        a.set_xticks(range(len(conds))); a.set_xticklabels([cd.split("/")[1].replace("r", "r = ") for cd in conds])
        a.set_title({"bgsub": "background images subtracted", "raw": "no background images (reflection kept)"}[var])
        a.grid(alpha=.3, axis="y")
    ax[0].set_ylabel("outlier rate (>1 px); hatched = direction error")
    ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"PIV Challenge 2001 case C (impeller, strong wall reflection), n = {valid.sum()} vectors")
    fig.tight_layout(); fig.savefig("results/fig_pivc_bars.png", dpi=130); plt.close(fig)

    # ---- vector maps (r = 0.7, bgsub)
    show = [("reference: real A/B multi-pass CC", ru, rv),
            ("nonlinear auto-corr", F["bgsub|asym|u"], F["bgsub|asym|v"]),
            ("U-Net fine-tuned, multi-pass + UOD", F["bgsub|unet_mp:finetuned+uod|u"],
             F["bgsub|unet_mp:finetuned+uod|v"])]
    fig, ax = plt.subplots(2, 2, figsize=(16, 8.6))
    ax = ax.ravel()
    ax[0].imshow(F["I_raw"], cmap="gray", vmax=0.8)
    ax[0].set_title("pulse-coded double exposure I = A + 0.7 B (raw variant)")
    inm = F["inm"].astype(bool)
    for a, (t, u, v) in zip(ax[1:], show):
        a.imshow(F["A"], cmap="gray", vmax=0.8, alpha=0.6)
        err = np.hypot(u - ru, v - rv)
        bad = (err > 1) & valid
        ok = inm & ~bad
        a.quiver(cx[ok], cy[ok], u[ok], v[ok], color="#2060c0", angles="xy", scale_units="xy", scale=0.4,
                 width=0.0016)
        if t.startswith("reference"):
            a.set_title(f"{t}  (fluid vectors {inm.sum()}, evaluated {valid.sum()})")
        else:
            a.quiver(cx[bad], cy[bad], u[bad], v[bad], color="#e03030", angles="xy", scale_units="xy",
                     scale=0.4, width=0.0016)
            a.set_title(f"{t}: red = outlier ({bad.sum() / valid.sum():.1%})")
    for a in ax:
        a.set_xlim(0, 1000); a.set_ylim(512, 0); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig("results/fig_pivc_vectors.png", dpi=110); plt.close(fig)

    # ---- error vs reflection strength
    lev = reflection_level(cx, cy)
    bins = [0, 40, 80, 255]
    rows = {}
    for var in ["bgsub", "raw"]:
        for k, _, _ in METHODS:
            if f"{var}|{k}|u" not in F:
                continue
            e = np.hypot(F[f"{var}|{k}|u"] - ru, F[f"{var}|{k}|v"] - rv)
            rows[f"{var}|{k}"] = [float(np.mean(e[valid & (lev >= lo) & (lev < hi)] > 1))
                                  for lo, hi in zip(bins[:-1], bins[1:])]
    cnt = [int((valid & (lev >= lo) & (lev < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
    ev["by_reflection_r0.7"] = dict(bins=bins, count=cnt, outlier=rows)
    json.dump(ev, open("results/eval_pivc.json", "w"), indent=1)
    print("reflection bins", bins, "count", cnt)
    for k, v in rows.items():
        print(f"  {k:32s}", " ".join(f"{x:.3f}" for x in v))


if __name__ == "__main__":
    main()
