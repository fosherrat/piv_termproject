"""Figures for stage 3 (real recordings).
python scripts/make_figures_real.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import evaluate_real as R
from pivsep.separation import SepUNet, unet_split
from make_figures import rgb


def load_model(p):
    ck = torch.load(p, map_location="cpu")
    m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    return m


def main(pair="test1", r=0.7):
    zs, ft = load_model("results/sep_unet_wide_bce.pt"), load_model("results/sep_unet_ft_real.pt")
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()[pair]))
    _, d = R.evaluate_pair({"zeroshot": zs, "finetuned": ft}, A, B, r)
    I, cx, cy, (ru, rv), valid, res = d["I"], d["cx"], d["cy"], d["ref"], d["valid"], d["res"]

    # 1) separation crops
    sl = (slice(120, 220), slice(150, 250))
    zA, zB = unet_split(zs, I); fA, fB = unet_split(ft, I)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(I[sl], cmap="gray", vmin=0, vmax=0.8); ax[0].set_title(f"real {pair}: A + {r}·B")
    ax[1].imshow(rgb(A[sl], r * B[sl])); ax[1].set_title("true frames (A red, B green)")
    ax[2].imshow(rgb(zA[sl], zB[sl])); ax[2].set_title("zero-shot U-Net split")
    ax[3].imshow(rgb(fA[sl], fB[sl])); ax[3].set_title("fine-tuned U-Net split")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig("results/fig_real_separation.png", dpi=130); plt.close(fig)

    # 2) vector fields
    show = [("reference: CC of real A, B", (ru, rv)), ("nonlinear auto-corr", res["asym"]),
            ("fine-tuned U-Net → CC (centre-masked)", res["unet_ccm:finetuned"]),
            ("  + UOD + div-free PINN", res["unet_ccm:finetuned+pinn"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for a, (name, (u, v)) in zip(ax, show):
        e = np.hypot(u - ru, v - rv)
        a.imshow(I, cmap="gray", vmin=0, vmax=1.0)
        q = a.quiver(cx, cy, u, -v, np.where(valid, e, np.nan), cmap="autumn_r", clim=(0, 2), scale=120, width=.004)
        t = name if name.startswith("reference") else f"{name}\noutlier {np.mean(e[valid] > 1):.0%}"
        a.set_title(t, fontsize=9); a.axis("off")
        a.set_xlim(0, cx.max() + 16); a.set_ylim(cy.max() + 16, 0)
    fig.colorbar(q, ax=ax, fraction=.012, label="deviation from reference [px]")
    fig.savefig("results/fig_real_vectors.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
