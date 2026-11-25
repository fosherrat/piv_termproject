"""Figures for stage 5 (masking + validation on real recordings).
python scripts/make_figures_validation.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import evaluate_validation as V
from pivsep.separation import SepUNet

STY = {"asym + mask + UOD5": ("nonlinear auto-corr + mask + UOD", "#3b7dd8", "s"),
       "auto (oracle sign) + mask + UOD5": ("auto-corr (oracle sign) + mask + UOD", "#c9a227", "D"),
       "unet mp raw": ("U-Net multi-pass, no validation", "#b896e0", "o"),
       "unet mp + mask": ("U-Net multi-pass + mask", "#8e62c9", "o"),
       "unet mp + mask + ppr + UOD5": ("U-Net multi-pass + mask + UOD", "#4b1d80", "o"),
       "unet mp + validation + fill": ("U-Net, validated + gaps filled", "#4b1d80", "^")}


def fig_tradeoff(ev, out):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, g in zip(ax, ["test2", "test1"]):
        for k, (lab, c, mk) in STY.items():
            pts = [(ev[f"{g}/r{r}"][k]["coverage"], ev[f"{g}/r{r}"][k]["outlier"], r) for r in (0.9, 0.7, 0.5)]
            a.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", c=c, label=lab, alpha=.9)
            for x, y, r in pts:
                a.annotate(f"r {r}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)
        a.set_xlabel("coverage (kept / trusted reference vectors)"); a.set_ylabel("outlier rate (>1 px)")
        a.set_xlim(0.4, 1.02); a.grid(alpha=.3)
        a.set_title({"test2": "test2 pairs 3-5 (held-out instants)", "test1": "test1 (unseen experiment)"}[g])
    ax[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("real recordings: outlier rate vs coverage (lower-right is better)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_map(out, name="test2_3", r=0.7):
    ev = json.load(open("results/eval_validation.json"))["selected"]
    ck = torch.load("results/sep_unet_ft_real.pt", map_location="cpu")
    m = SepUNet(base=ck["base"]); m.load_state_dict(ck["state"]); m.eval()
    d = V.prepare(m, name, r)
    u, v = d["unet"]; ru, rv = d["ref"]
    keep = V.validate(d, "unet", ev["rel_signal"], ev["ppr"], ev["uod5"])
    e = np.hypot(u - ru, v - rv)
    ext = [d["cx"].min(), d["cx"].max(), d["cy"].max(), d["cy"].min()]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    ax[0].imshow(d["I"], cmap="gray", vmax=1); ax[0].set_title(f"real {name}: A + {r}·B")
    im = ax[1].imshow(np.where(d["ref_ok"], e, np.nan), cmap="magma", vmin=0, vmax=3, extent=ext)
    ax[1].set_title(f"error before validation (outlier {np.mean(e[d['ref_ok']] > 1):.1%})")
    shown = np.where(d["ref_ok"] & keep, e, np.nan)
    ax[2].imshow(shown, cmap="magma", vmin=0, vmax=3, extent=ext)
    sel = d["ref_ok"] & keep
    ax[2].set_title(f"after mask + UOD 5x5 (outlier {np.mean(e[sel] > 1):.1%}, coverage {sel.sum() / d['ref_ok'].sum():.0%}; removed = white)")
    for a in ax: a.axis("off")
    fig.colorbar(im, ax=ax, fraction=.012, label="|U-Net - reference| [px]")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff(json.load(open("results/eval_validation.json")), "results/fig_validation_tradeoff.png")
    fig_map("results/fig_validation_map.png")
    print("figures written")
