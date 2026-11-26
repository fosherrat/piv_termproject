"""Case C vector maps with matched processing (paper/fig_casec.pdf):
reference, nonlinear auto-correlation and U-Net + CC, both multi-pass + UOD, r = 0.7.
Red: outlier (|error| > 1 px) among the evaluated vectors.

python scripts/make_figure_casec.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pivsep.correlation import multipass_piv, multipass_auto_piv
from pivsep.separation import unet_split
from pivsep.postproc import uod_median
import evaluate_real as R
import evaluate_pivc as PC
import evaluate_paper as EP


def main(r=0.7):
    torch.set_num_threads(os.cpu_count() or 4)
    m = EP.load_models()["finetuned"]
    frames, M = PC.load_case(); A, B = frames["bgsub"]
    cx, cy, ru, rv = multipass_piv(A, B)
    bad = uod_median(ru, rv)[2]
    valid = PC.centre_mask(M, cx, cy) & ~bad & (np.hypot(ru, rv) > 2)
    I = R.make_double(A, B, r)
    fields = {"nonlinear auto-correlation": uod_median(*multipass_auto_piv(I, "asym", mask_r=3)[2:])[:2],
              "U-Net + CC (ours)": uod_median(*multipass_piv(*unet_split(m, I), mask_r=3)[2:])[:2]}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 1.75), constrained_layout=True)
    show = np.where(M, I, np.nan)
    sub = (slice(None, None, 2), slice(None, None, 2))
    def quiv(a, u, v, col):
        a.quiver(cx[sub], cy[sub], u[sub], -v[sub], color=col[sub].reshape(-1, 4) if col.ndim == 3 else col,
                 angles="xy", scale_units="xy", scale=0.25, width=0.004, headwidth=3)
    for a in ax:
        a.imshow(show, cmap="gray", vmin=0, vmax=0.6); a.set_axis_off()
    blue = np.array(matplotlib.colors.to_rgba("C0")); red = np.array(matplotlib.colors.to_rgba("C3"))
    grey = np.array([0.6, 0.6, 0.6, 0.5])
    col = np.where(valid[..., None], blue, grey)
    quiv(ax[0], np.where(np.isnan(ru), 0, ru), np.where(np.isnan(rv), 0, rv), col)
    ax[0].set_title("reference (real $A,B$)", fontsize=8)
    for a, (name, (u, v)) in zip(ax[1:], fields.items()):
        out = valid & (np.hypot(u - ru, v - rv) > 1)
        c = np.where(out[..., None], red, np.where(valid[..., None], blue, grey))
        quiv(a, u, v, c)
        a.set_title(f"{name}: {100 * out.sum() / valid.sum():.1f}% outliers", fontsize=8)
    fig.savefig("paper/fig_casec.pdf"); fig.savefig("paper/fig_casec.png", dpi=200)
    print("saved paper/fig_casec.pdf")


if __name__ == "__main__":
    main()
