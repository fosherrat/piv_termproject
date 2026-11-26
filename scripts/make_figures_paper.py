"""Result figures for the paper.

From results/eval_paper.json:
  paper/fig_outliers.pdf   outlier rate per dataset and r (multi-pass; without / with UOD)
  paper/fig_dir_acc.pdf    direction-error rate and median error per dataset and r (multi-pass)
  paper/fig_process.pdf    processing ablation at r = 0.7 (SP, SP+UOD, MP, MP+UOD)
Recomputed at r = 0.7 (same pipeline as evaluate_paper.py):
  paper/fig_vec_<dataset>.pdf   vector maps: reference, nonlinear AC, AC (true sign), U-Net + CC
  paper/fig_cdf.pdf             cumulative distribution of the vector error

python scripts/make_figures_paper.py [--skip-recompute]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots in fixed order (validated: scripts of the dataviz guide), one colour per method
COL = {"ours": "#257de0", "asym": "#f47230", "oracle": "#00b894", "gt": "#f2a600"}
LAB = {"ours": "U-Net + CC (ours)", "asym": "nonlinear AC", "oracle": "AC, true sign$^\\dagger$",
       "gt": "true $A,B$ (upper bound)"}
DS = [("synth", "Synthetic (ours)"), ("lml", "Synthetic (LML)"), ("test1", "Real: test1"), ("caseC", "Real: case C")]
R = [0.9, 0.7, 0.5]
FLOOR = 5e-4      # zero rates are drawn at this level on log axes

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#666666", "axes.labelcolor": "#191919", "xtick.color": "#666666",
                     "ytick.color": "#666666", "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def key(d, ds, m):
    return f"unet:{d['_meta']['main_model'][ds]}" if m == "ours" else m


def bars(ax, d, ds, metric, suffix, methods, log=True, ci=False, clip=None):
    rows = [r for r in R if f"{ds}/r{r}" in d]
    ms = [m for m in methods if key(d, ds, m) + suffix in d[f"{ds}/r{rows[0]}"]]
    bw = 0.8 / len(ms)
    for j, m in enumerate(ms):
        x = np.arange(len(rows)) + (j - (len(ms) - 1) / 2) * bw
        t = [d[f"{ds}/r{r}"][key(d, ds, m) + suffix] for r in rows]
        y = np.array([v[metric] for v in t])
        y = np.maximum(y, FLOOR) if log else y
        ax.bar(x, y, bw * 0.92, color=COL[m], label=LAB[m], linewidth=0)
        if not log and clip is not None:
            for xi, yi in zip(x, y):
                if yi > clip:
                    ax.text(xi, clip * 0.97, f"{yi:.1f}", ha="center", va="top", fontsize=6, color="white",
                            rotation=90)
        if ci and "outlier_ci" in t[0]:
            lo = np.maximum([v["outlier_ci"][0] for v in t], FLOOR); hi = np.maximum([v["outlier_ci"][1] for v in t], FLOOR)
            ax.errorbar(x, y, [np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="none", ecolor="#191919",
                        elinewidth=0.6, capsize=1.5)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"$r={r}$" for r in rows])
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y"); ax.set_axisbelow(True)


def pct_axis(ax, lo=FLOOR, hi=1.0):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if v <= FLOOR * 1.01 else f"{100 * v:g}%"))


def fig_outliers(d):
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 4.0), sharey=True, constrained_layout=True)
    for i, (suf, lab) in enumerate([("", "multi-pass"), ("+uod", "multi-pass + UOD")]):
        for j, (ds, name) in enumerate(DS):
            bars(ax[i, j], d, ds, "outlier", suf, ["asym", "oracle", "ours", "gt"], ci=True)
            pct_axis(ax[i, j])
            if i == 0:
                ax[i, j].set_title(name, fontsize=8.5)
        ax[i, 0].set_ylabel(f"outlier rate\n({lab})")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.savefig("paper/fig_outliers.pdf", bbox_inches="tight"); fig.savefig("paper/fig_outliers.png", dpi=150, bbox_inches="tight")


def fig_dir_acc(d):
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 4.0), constrained_layout=True)
    for j, (ds, name) in enumerate(DS):
        bars(ax[0, j], d, ds, "sign_err", "", ["asym", "ours"])
        pct_axis(ax[0, j]); ax[0, j].set_title(name, fontsize=8.5)
        top = 2.0 if ds != "synth" else 0.4
        bars(ax[1, j], d, ds, "median", "", ["asym", "oracle", "ours", "gt"], log=False, clip=top)
        ax[1, j].set_ylim(0, top)
    ax[0, 0].set_ylabel("direction-error rate\n(multi-pass)"); ax[1, 0].set_ylabel("median error [px]\n(multi-pass)")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.savefig("paper/fig_dir_acc.pdf", bbox_inches="tight"); fig.savefig("paper/fig_dir_acc.png", dpi=150, bbox_inches="tight")


def fig_process(d, r=0.7):
    stages = [("_sp", "SP"), ("_sp+uod", "SP\n+UOD"), ("", "MP"), ("+uod", "MP\n+UOD")]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True, constrained_layout=True)
    for j, (ds, name) in enumerate(DS):
        t = d[f"{ds}/r{r}"]
        for m in ["asym", "oracle", "ours"]:
            k = key(d, ds, m)
            names = [(k.replace("unet:", "unet_sp:") + s[3:]) if (s.startswith("_sp") and m == "ours")
                     else (k + s) for s, _ in stages]
            y = [max(t[n]["outlier"], FLOOR) for n in names]
            ax[j].plot(range(4), y, "-o", color=COL[m], lw=2, ms=4, label=LAB[m])
        ax[j].set_yscale("log"); pct_axis(ax[j]); ax[j].grid(axis="y")
        ax[j].set_xticks(range(4)); ax[j].set_xticklabels([s for _, s in stages])
        ax[j].set_title(name, fontsize=8.5)
    ax[0].set_ylabel(f"outlier rate ($r={r}$)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.savefig("paper/fig_process.pdf", bbox_inches="tight"); fig.savefig("paper/fig_process.png", dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------ recomputed: vectors and CDF
def examples(r=0.7):
    """Yields (dataset, list of samples); each sample: dict(I, cx, cy, ref, valid, fields, mask)."""
    import torch
    import evaluate_paper as EP, evaluate_real as R
    from pivsep.synth import SynthParams, make_sample, gt_window_displacement
    from pivsep.correlation import window_grid, multipass_piv
    from pivsep.postproc import uod_median
    torch.set_num_threads(os.cpu_count() or 4)
    models = EP.load_models()
    meta = EP.MAIN_MODEL

    def run(ds, I, ref, valid, mask=None):
        res = EP.run_methods({meta[ds]: models[meta[ds]]}, I, ref)
        f = {"asym": res["asym"], "oracle": res["oracle"], "ours": res[f"unet:{meta[ds]}"]}
        return dict(I=I, cx=ref[0], cy=ref[1], ref=ref[2:], valid=valid, fields=f, mask=mask)

    out = {}
    S = []
    for i in range(12):
        s = make_sample(np.random.default_rng(1000 + i), SynthParams(r=r))
        cx, cy = window_grid(s["img"].shape, 32, 16)[2:]
        gu, gv = gt_window_displacement(s["flow"], cx, cy, 32)
        S.append(run("synth", s["img"], (cx, cy, gu, gv), np.hypot(gu, gv) > 2))
    out["synth"] = S
    import evaluate_lml as E
    C = np.load("data/lml/lml_samples.npz"); S = []
    for i in range(12):
        s = {k: C[f"translation|{r}|0.01|{i}|{k}"] for k in ("I", "P1", "P2")}
        I = E.to_model_scale(s["I"]); cx, cy, gu, gv = E.gt_windows(s["P1"], s["P2"], I.shape)
        S.append(run("lml", I, (cx, cy, gu, gv), np.isfinite(gu) & (np.hypot(gu, gv) > 2)))
    out["lml"] = S
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()["test1"]))
    cx, cy, ru, rv = multipass_piv(A, B)
    out["test1"] = [run("test1", R.make_double(A, B, r), (cx, cy, ru, rv),
                        ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2))]
    if os.path.isdir("data/pivchallenge_C"):
        import evaluate_pivc as PC
        frames, M = PC.load_case(); A, B = frames["bgsub"]
        cx, cy, ru, rv = multipass_piv(A, B)
        valid = PC.centre_mask(M, cx, cy) & ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2)
        out["caseC"] = [run("caseC", R.make_double(A, B, r), (cx, cy, ru, rv), valid, M)]
    return out


def fig_vectors(ds, name, s, r=0.7):
    I, cx, cy, (ru, rv), valid = s["I"], s["cx"], s["cy"], s["ref"], s["valid"]
    H, W = I.shape
    big = H * W > 300 * 300
    sub = (slice(None, None, 2), slice(None, None, 2)) if big else (slice(None), slice(None))
    wide = W / H > 1.5
    nr, nc = (2, 2) if wide else (1, 4)
    fig, ax = plt.subplots(nr, nc, figsize=(7.2, 7.2 / nc * H / W * nr + 0.3 * nr), constrained_layout=True)
    ax = ax.ravel()
    img = I if s["mask"] is None else np.where(s["mask"], I, np.nan)
    blue = np.array(matplotlib.colors.to_rgba(COL["ours"])); red = np.array(matplotlib.colors.to_rgba("#ec4647"))
    grey = np.array([0.55, 0.55, 0.55, 0.7]); white = np.array([1, 1, 1, 0.95])
    spacing = (cx[0, 1] - cx[0, 0]) * (2 if big else 1)
    mag = np.nanmedian(np.hypot(ru, rv)[valid]); scale = mag / (0.85 * spacing)
    def quiv(a, u, v, col):
        u = np.nan_to_num(u); v = np.nan_to_num(v)
        a.quiver(cx[sub], cy[sub], u[sub], -v[sub], color=col[sub].reshape(-1, 4), angles="xy",
                 scale_units="xy", scale=scale, width=0.0022 if wide else 0.006, headwidth=3.5, headlength=4)
    for a in ax:
        a.imshow(img, cmap="gray", vmin=0, vmax=2.5 * np.nanpercentile(img, 99.5)); a.set_axis_off()
    quiv(ax[0], ru, rv, np.where(valid[..., None], white, grey))
    ax[0].set_title("reference" + (" (true)" if ds in ("synth", "lml") else " (real $A,B$)"), fontsize=7.5)
    for a, m in zip(ax[1:], ["asym", "oracle", "ours"]):
        u, v = s["fields"][m]
        out = valid & (np.hypot(u - ru, v - rv) > 1)
        quiv(a, u, v, np.where(out[..., None], red, np.where(valid[..., None], white, grey)))
        a.set_title(f"{LAB[m].replace(' (ours)', '')}: {100 * out.sum() / valid.sum():.1f}%", fontsize=7.5)
    fig.savefig(f"paper/fig_vec_{ds}.pdf"); fig.savefig(f"paper/fig_vec_{ds}.png", dpi=150)
    plt.close(fig)


def fig_cdf(ex):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True, constrained_layout=True)
    e_grid = np.logspace(-2, 1.5, 300)
    for j, (ds, name) in enumerate(DS):
        if ds not in ex:
            continue
        for m in ["asym", "oracle", "ours"]:
            e = np.concatenate([np.hypot(s["fields"][m][0] - s["ref"][0], s["fields"][m][1] - s["ref"][1])[s["valid"]]
                                for s in ex[ds]])
            ax[j].plot(e_grid, [np.mean(e <= g) for g in e_grid], color=COL[m], lw=2, label=LAB[m])
        ax[j].axvline(1, color="#666666", lw=0.8, ls=":")
        ax[j].set_xscale("log"); ax[j].set_xlabel("$|\\epsilon|$ [px]"); ax[j].set_title(name, fontsize=8.5)
        ax[j].grid(True); ax[j].set_ylim(0, 1)
    ax[0].set_ylabel("fraction of vectors\nwith error $\\leq |\\epsilon|$ ($r=0.7$)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.savefig("paper/fig_cdf.pdf", bbox_inches="tight"); fig.savefig("paper/fig_cdf.png", dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-recompute", action="store_true")
    a = ap.parse_args()
    d = json.load(open("results/eval_paper.json"))
    fig_outliers(d); fig_dir_acc(d); fig_process(d)
    if not a.skip_recompute:
        ex = examples()
        for ds, name in DS:
            if ds in ex:
                fig_vectors(ds, name, ex[ds][0])
        fig_cdf(ex)
    print("saved figures in paper/")


if __name__ == "__main__":
    main()
