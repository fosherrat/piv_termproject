"""Slide-ready result figures (one consistent style for every experiment).

Every figure is drawn at the size it occupies on the slide (content width 9.5 in), so text
renders at its final size.  Shared conventions:
  * one panel per dataset, always in the order of DS, panel title = dataset name
  * fixed colour per method (validated categorical palette): nonlinear AC orange,
    AC with true sign aqua, U-Net + CC (ours) blue, true A, B yellow
  * rates on a log axis with 0 drawn at the floor and labelled "0"; values printed on the
    bars of the proposed method only
  * one legend row above the panels
  * vector maps: 4 square panels (reference, nonlinear AC, AC true sign, ours) for every
    dataset, same arrow density; white = correct, red = outlier (> 1 px), grey = not evaluated

Inputs: results/eval_paper.json (summary figures) and a recomputation at r = 0.7 through
make_figures_paper.examples() (vector maps, error CDF).
Output: slides/fig/res_*.png

python scripts/make_figures_slides.py [--skip-recompute]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "slides/fig"
W = 9.5                      # slide content width [in]
DPI = 250
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
COL = {"asym": "#f47230", "oracle": "#00b894", "ours": "#257de0", "gt": "#f2a600",
       "zeroshot": "#6a4ce2"}
LAB = {"asym": "nonlinear AC", "oracle": "AC, true sign†", "ours": "U-Net + CC (ours)",
       "gt": "true A, B (upper bound)", "zeroshot": "U-Net zero-shot (synthetic only)"}
DS = [("synth", "Synthetic (ours)"), ("lml", "Synthetic (LML)"), ("test1", "Real: test1"),
      ("test2", "Real: test2 §"), ("caseC", "Real: case C")]
R = [0.9, 0.7, 0.5]
FLOOR = 5e-4
RED, WHITE, GREY = "#ec4647", "#ffffff", "#8c8c8c"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 10,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": "#9c9c9c", "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 10, "savefig.facecolor": "white"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=DPI)
    plt.close(fig)
    print("  ", name)


def key(d, ds, m):
    if m == "ours":
        return f"unet:{d['_meta']['main_model'][ds]}"
    if m == "zeroshot":
        return "unet:zeroshot"
    return m


def panels(h, n=len(DS), legend_rows=1, sharey=True, bottom=0.2):
    fig, ax = plt.subplots(1, n, figsize=(W, h), sharey=sharey)
    top = 1 - (0.30 * legend_rows + 0.32) / h
    fig.subplots_adjust(left=0.075, right=0.995, top=top, bottom=bottom, wspace=0.12 if sharey else 0.3)
    return fig, np.atleast_1d(ax)


def legend(fig, methods, h, marker=False):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[m], lw=2.5, marker="o", ms=6) if marker else Patch(color=COL[m]) for m in methods]
    fig.legend(hs, [LAB[m] for m in methods], loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.53, 1.0), handlelength=1.4, columnspacing=1.6)


def pct_axis(ax, hi=1.0):
    ax.set_yscale("log"); ax.set_ylim(FLOOR, hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v <= FLOOR * 1.01 else f"{100 * v:g}%"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.grid(axis="y"); ax.set_axisbelow(True)


def fmt_pct(v):
    p = 100 * v
    return "0" if p < 0.05 else (f"{p:.1f}" if p < 10 else f"{p:.0f}")


def r_ticks(ax, rows):
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r}" for r in rows])
    ax.tick_params(axis="x", length=0)


def r_label(fig):
    fig.text(0.535, 0.01, "pulse ratio r", ha="center", va="bottom", color=INK2, fontsize=10)


def bar_group(ax, d, ds, metric, suffix, methods, log=True, label_ours=True, clip=None):
    rows = [r for r in R if f"{ds}/r{r}" in d]
    ms = [m for m in methods if key(d, ds, m) + suffix in d[f"{ds}/r{rows[0]}"]]
    bw = 0.8 / len(ms)
    for j, m in enumerate(ms):
        x = np.arange(len(rows)) + (j - (len(ms) - 1) / 2) * bw
        y = np.array([d[f"{ds}/r{r}"][key(d, ds, m) + suffix][metric] for r in rows])
        yp = np.maximum(y, FLOOR) if log else (np.minimum(y, clip) if clip else y)
        ax.bar(x, yp, bw * 0.88, color=COL[m], linewidth=0, zorder=2)
        if clip is not None:
            for xi, yi in zip(x, y):
                if yi > clip:
                    ax.text(xi, clip * 0.96, f"{yi:.1f}", ha="center", va="top", fontsize=7.5, color="white",
                            rotation=90, fontweight="bold", zorder=3)
        if label_ours and m in ("ours", "zeroshot") or (label_ours == "all"):
            for xi, yi, ypi in zip(x, y, yp):
                txt = fmt_pct(yi) if log else f"{yi:.2f}"
                ty = ypi * 1.35 if log else ypi + (clip or ax.get_ylim()[1]) * 0.02
                ax.text(xi, ty, txt, ha="center", va="bottom", fontsize=8, color=INK, zorder=4, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
    r_ticks(ax, rows)


# ----------------------------------------------------------------------------- summary charts
def fig_outlier(d, suffix, name, ylabel):
    h = 2.9
    fig, ax = panels(h)
    for a, (ds, title) in zip(ax, DS):
        bar_group(a, d, ds, "outlier", suffix, ["asym", "oracle", "ours", "gt"])
        pct_axis(a); a.set_title(title)
    ax[0].set_ylabel(ylabel)
    legend(fig, ["asym", "oracle", "ours", "gt"], h)
    r_label(fig)
    save(fig, name)


def fig_direction(d, suffix=""):
    h = 2.9
    fig, ax = panels(h)
    for a, (ds, title) in zip(ax, DS):
        bar_group(a, d, ds, "sign_err", suffix, ["asym", "ours"])
        pct_axis(a); a.set_title(title)
    ax[0].set_ylabel("direction-error rate")
    legend(fig, ["asym", "ours"], h)
    r_label(fig)
    save(fig, "res_direction" + ("_uod" if suffix else "") + ".png")


def fig_median(d, suffix=""):
    h = 2.9
    fig, ax = panels(h, sharey=False)
    for a, (ds, title) in zip(ax, DS):
        top = 0.4 if ds == "synth" else 2.0
        a.set_ylim(0, top)
        bar_group(a, d, ds, "median", suffix, ["asym", "oracle", "ours", "gt"], log=False, clip=top)
        a.grid(axis="y"); a.set_axisbelow(True); a.set_title(title)
    ax[0].set_ylabel("median error [px]")
    legend(fig, ["asym", "oracle", "ours", "gt"], h)
    r_label(fig)
    save(fig, "res_median" + ("_uod" if suffix else "") + ".png")


def fig_process(d, r=0.7):
    h = 2.9
    stages = [("_sp", "SP"), ("_sp+uod", "SP\n+UOD"), ("", "MP"), ("+uod", "MP\n+UOD")]
    fig, ax = panels(h, bottom=0.2)
    for a, (ds, title) in zip(ax, DS):
        t = d[f"{ds}/r{r}"]
        for m in ["asym", "oracle", "ours"]:
            k = key(d, ds, m)
            names = [(k.replace("unet:", "unet_sp:") + s[3:]) if (s.startswith("_sp") and m == "ours") else (k + s)
                     for s, _ in stages]
            y = [max(t[n]["outlier"], FLOOR) for n in names]
            a.plot(range(4), y, "-o", color=COL[m], lw=2, ms=5, zorder=3)
            if m == "ours":
                for xi, n in enumerate(names):
                    lo = y[xi] < 4e-3
                    a.text(xi, y[xi] * 1.7 if lo else y[xi] / 1.6, fmt_pct(t[n]["outlier"]), ha="center",
                           va="bottom" if lo else "top", fontsize=8, color=INK, zorder=4, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        pct_axis(a); a.set_title(title)
        a.set_xticks(range(4)); a.set_xticklabels([s for _, s in stages]); a.set_xlim(-0.35, 3.35)
        a.tick_params(axis="x", length=0)
    ax[0].set_ylabel(f"outlier rate (r = {r})")
    legend(fig, ["asym", "oracle", "ours"], h, marker=True)
    save(fig, "res_process.png")


def fig_model(d):
    h = 2.9
    fig, ax = panels(h)
    for a, (ds, title) in zip(ax, DS):
        rows = [r for r in R if f"{ds}/r{r}" in d]
        for j, (m, k) in enumerate([("zeroshot", "unet:zeroshot+uod"), ("ours", "unet:finetuned+uod")]):
            x = np.arange(len(rows)) + (j - 0.5) * 0.36
            y = np.array([d[f"{ds}/r{r}"][k]["outlier"] for r in rows])
            a.bar(x, np.maximum(y, FLOOR), 0.32, color=COL[m], linewidth=0, zorder=2)
            for xi, yi in zip(x, y):
                a.text(xi, max(yi, FLOOR) * 1.35, fmt_pct(yi), ha="center", va="bottom", fontsize=7.5, color=INK,
                       zorder=4, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        r_ticks(a, rows)
        pct_axis(a); a.set_title(title)
    ax[0].set_ylabel("outlier rate (MP + UOD)")
    LAB_ = dict(LAB); LAB["ours"] = "U-Net fine-tuned on test2 pairs 0–2"
    legend(fig, ["zeroshot", "ours"], h)
    LAB.update(LAB_)
    r_label(fig)
    save(fig, "res_model.png")


# ----------------------------------------------------------------------------- recomputed figures
CROP = {"synth": (slice(None), slice(None)), "lml": (slice(None), slice(None)),
        "test1": (slice(0, 368), slice(64, 432)), "test2": (slice(None), slice(None)),
        "caseC": (slice(0, 512), slice(240, 752))}
ARROWS = 16                  # arrows across a panel


def fig_vectors(ds, s, uod=False):
    import matplotlib.colors as mc
    I, cx, cy, (ru, rv), valid = s["I"], s["cx"], s["cy"], s["ref"], s["valid"]
    ys, xs = CROP[ds]
    y0, y1 = (ys.start or 0), (ys.stop or I.shape[0]); x0, x1 = (xs.start or 0), (xs.stop or I.shape[1])
    img = I if s["mask"] is None else np.where(s["mask"][:I.shape[0], :I.shape[1]], I, np.nan)
    img = img[y0:y1, x0:x1]
    sel = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    rows = np.where(sel.any(1))[0]; cols = np.where(sel.any(0))[0]
    step = max(1, int(round(len(cols) / ARROWS)))
    sub = np.zeros_like(sel)
    sub[np.ix_(rows[::step], cols[::step])] = True
    sub &= sel
    spacing = (cx[0, 1] - cx[0, 0]) * step
    mag = np.nanmedian(np.hypot(ru, rv)[valid]); scale = mag / (0.8 * spacing)
    h = W / 4 + 0.32
    fig, ax = plt.subplots(1, 4, figsize=(W, h))
    fig.subplots_adjust(left=0.003, right=0.997, top=1 - 0.32 / h, bottom=0.01, wspace=0.03)
    vmax = 2.5 * np.nanpercentile(img, 99.5)
    rgba = {k: np.array(mc.to_rgba(c)) for k, c in (("r", RED), ("w", WHITE), ("g", GREY))}

    def quiv(a, u, v, out):
        col = np.where(out[..., None], rgba["r"], np.where(valid[..., None], rgba["w"], rgba["g"]))
        a.quiver(cx[sub] - x0, cy[sub] - y0, np.nan_to_num(u[sub]), -np.nan_to_num(v[sub]), color=col[sub],
                 angles="xy", scale_units="xy", scale=scale, width=0.008, headwidth=3.5, headlength=4,
                 headaxislength=3.6)
    for a in ax:
        a.imshow(img, cmap="gray", vmin=0, vmax=vmax, extent=(-0.5, x1 - x0 - 0.5, y1 - y0 - 0.5, -0.5))
        a.set_facecolor("#d9d9d9"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    quiv(ax[0], ru, rv, np.zeros_like(valid))
    ax[0].set_title("reference" + (" (true)" if ds in ("synth", "lml") else " (real A, B)"))
    for a, m in zip(ax[1:], ["asym", "oracle", "ours"]):
        u, v = s["fields"][m]
        if uod:
            from pivsep.postproc import uod_median
            u, v = uod_median(u, v)[:2]
        out = valid & (np.hypot(u - ru, v - rv) > 1)
        quiv(a, u, v, out)
        a.set_title(f"{LAB[m].replace(' (ours)', '')}: {100 * out.sum() / valid.sum():.1f}%",
                    color=COL[m] if m != "oracle" else "#008a6c")
    save(fig, f"res_vec_{ds}" + ("_uod" if uod else "") + ".png")


def fig_cdf(ex):
    h = 2.9
    fig, ax = panels(h, bottom=0.2)
    e_grid = np.logspace(-2, 1.5, 300)
    for a, (ds, title) in zip(ax, DS):
        if ds not in ex:
            continue
        for m in ["asym", "oracle", "ours"]:
            e = np.concatenate([np.hypot(s["fields"][m][0] - s["ref"][0], s["fields"][m][1] - s["ref"][1])[s["valid"]]
                                for s in ex[ds]])
            a.plot(e_grid, [np.mean(e <= g) for g in e_grid], color=COL[m], lw=2)
        a.axvline(1, color=INK2, lw=0.9, ls=":")
        a.set_xscale("log"); a.set_xlabel("|error| [px]"); a.set_title(title)
        a.grid(True); a.set_ylim(0, 1.02); a.set_xticks([0.01, 0.1, 1, 10]); a.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax[0].set_ylabel("fraction of vectors (r = 0.7)")
    legend(fig, ["asym", "oracle", "ours"], h, marker=False)
    save(fig, "res_cdf.png")


def fig_datasets():
    import evaluate_real as R_, evaluate_lml as E, evaluate_pivc as PC, evaluate_paper as EP
    from pivsep.synth import SynthParams, make_sample
    ims = {"synth": make_sample(np.random.default_rng(1000), SynthParams(r=0.7))["img"],
           "lml": E.to_model_scale(np.load("data/lml/lml_samples.npz")["translation|0.7|0.01|0|I"])}
    A, B = R_.normalize_pair(*(R_.crop(R_.preprocess(R_.load(f))) for f in R_.pairs()["test1"]))
    ims["test1"] = R_.make_double(A, B, 0.7)
    A, B = R_.normalize_pair(*(R_.crop(R_.preprocess(R_.load(f))) for f in R_.pairs()[EP.TEST2_EVAL[0]]))
    ims["test2"] = R_.make_double(A, B, 0.7)
    fr, M = PC.load_case(); A, B = fr["bgsub"]
    ims["caseC"] = np.where(M[:A.shape[0], :A.shape[1]], R_.make_double(A, B, 0.7), 0)
    gap = 0.15
    tw = (W - gap * 4) / 5
    h = tw + 0.32
    fig, ax = plt.subplots(1, 5, figsize=(W, h))
    fig.subplots_adjust(left=0, right=1, top=1 - 0.32 / h, bottom=0, wspace=gap / tw)
    for a, (ds, title) in zip(ax, DS):
        c = ims[ds][CROP[ds]]
        a.imshow(c, cmap="gray", vmin=0, vmax=np.percentile(c, 99.5)); a.set_axis_off(); a.set_title(title)
    save(fig, "res_datasets.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-recompute", action="store_true")
    ap.add_argument("--vectors-uod-only", action="store_true", help="recompute only the MP + UOD vector maps")
    a = ap.parse_args()
    style()
    d = json.load(open("results/eval_paper.json"))
    fig_outlier(d, "", "res_outlier.png", "outlier rate (multi-pass)")
    fig_outlier(d, "+uod", "res_outlier_uod.png", "outlier rate (MP + UOD)")
    fig_direction(d); fig_median(d); fig_process(d); fig_model(d)
    fig_direction(d, "+uod"); fig_median(d, "+uod")
    if a.vectors_uod_only:
        import make_figures_paper as MP
        ex = MP.examples()
        for ds, _ in DS:
            if ds in ex:
                fig_vectors(ds, ex[ds][0], uod=True)
    elif not a.skip_recompute:
        fig_datasets()
        import make_figures_paper as MP
        import torch
        ex = MP.examples()
        for ds, _ in DS:
            if ds in ex:
                fig_vectors(ds, ex[ds][0])
                fig_vectors(ds, ex[ds][0], uod=True)
        fig_cdf(ex)
    print("saved in", OUT)


if __name__ == "__main__":
    main()
