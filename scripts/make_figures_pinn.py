"""Slide figures for the PINN study, same style as make_figures_slides.py.

  res_pinn_outlier.png   outlier rate per dataset and r: MP, MP + UOD, MP + UOD + PINN
                         (and MP + PINN without UOD at r = 0.7, hatched)
  res_pinn_median.png    median error, same layout
  res_pinn_vort.png      RMS vorticity error, same layout
  res_pinn_diag.png      why: (a) outliers PINN fixes vs creates, (b) clustering of the outliers
                         it receives, (c) 2-D divergence of the reference field
  res_pinn_map_<ds>.png  r = 0.7 maps: reference over its 2-D divergence, MP + UOD, + PINN

Stage colours (validated categorical slots): MP blue (= "ours" in the other figures),
MP + UOD green, + PINN magenta; hatching = PINN without UOD.

python scripts/make_figures_pinn.py [--merge results/eval_pinn_*.json] [--skip-maps]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures_slides as S

SC = {"mp": "#257de0", "mp+uod": "#008a17", "mp+uod+pinn": "#ee7fac", "mp+pinn": "#ee7fac"}
SL = {"mp": "multi-pass (MP)", "mp+uod": "MP + UOD", "mp+uod+pinn": "MP + UOD + PINN",
      "mp+pinn": "MP + PINN (no UOD)"}
STAGES = ["mp", "mp+uod", "mp+uod+pinn"]


def legend(fig, keys):
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=SC[k], hatch="///" if k == "mp+pinn" else None, edgecolor="white" if k == "mp+pinn" else SC[k])
          for k in keys]
    fig.legend(hs, [SL[k] for k in keys], loc="upper center", ncol=len(keys), bbox_to_anchor=(0.53, 1.0),
               handlelength=1.4, columnspacing=1.6)


def stage_bars(ax, d, ds, metric, log=True, label=("mp+uod+pinn",), clip=None, fmt=None):
    rows = [r for r in S.R if f"{ds}/r{r}" in d]
    keys = STAGES + ["mp+pinn"]
    bw = 0.8 / len(keys)
    for j, k in enumerate(keys):
        x = np.arange(len(rows)) + (j - (len(keys) - 1) / 2) * bw
        y = np.array([d[f"{ds}/r{r}"][k][metric] if k in d[f"{ds}/r{r}"] else np.nan for r in rows])
        ok = np.isfinite(y)
        yp = np.maximum(y, S.FLOOR) if log else (np.minimum(y, clip) if clip else y)
        ax.bar(x[ok], yp[ok], bw * 0.88, color=SC[k], linewidth=0, zorder=2,
               hatch="///" if k == "mp+pinn" else None, edgecolor="white" if k == "mp+pinn" else None)
        if clip is not None:
            for xi, yi in zip(x[ok], y[ok]):
                if yi > clip:
                    ax.text(xi, clip * 0.96, (fmt or "{:.2f}").format(yi), ha="center", va="top", fontsize=7,
                            color="white", rotation=90, fontweight="bold", zorder=3)
        if k in label:
            for xi, yi, ypi in zip(x[ok], y[ok], yp[ok]):
                if clip is not None and yi > clip:
                    continue
                txt = S.fmt_pct(yi) if log else (fmt or "{:.2f}").format(yi)
                ty = ypi * 1.35 if log else ypi + ax.get_ylim()[1] * 0.02
                ax.text(xi, ty, txt, ha="center", va="bottom", fontsize=8, color=S.INK, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
    S.r_ticks(ax, rows)


def fig_stage(d, metric, name, ylabel, log=True, tops=None, fmt=None, scale=1.0):
    h = 2.9
    if scale != 1.0:
        d = {k: ({m: (dict(v, **{metric: v[metric] * scale}) if isinstance(v, dict) and metric in v else v)
                  for m, v in t.items()} if isinstance(t, dict) else t) for k, t in d.items()}
    fig, ax = S.panels(h, sharey=log)
    if not log:
        fig.subplots_adjust(wspace=0.32)
    for a, (ds, title) in zip(ax, S.DS):
        if log:
            S.pct_axis(a)
            stage_bars(a, d, ds, metric)
        else:
            top = tops[ds]
            a.set_ylim(0, top)
            stage_bars(a, d, ds, metric, log=False, clip=top, fmt=fmt)
            a.grid(axis="y"); a.set_axisbelow(True)
        a.set_title(title)
    ax[0].set_ylabel(ylabel)
    legend(fig, STAGES + ["mp+pinn"])
    S.r_label(fig)
    S.save(fig, name)


def fig_diag(d, r=0.7):
    h = 3.0
    fig, ax = plt.subplots(1, 3, figsize=(S.W, h))
    fig.subplots_adjust(left=0.065, right=0.995, top=1 - 0.42 / h, bottom=0.17, wspace=0.42)
    names = ["Synth.\n(ours)", "Synth.\n(LML)", "test1", "test2 §", "case C"]
    x = np.arange(len(S.DS))
    lab = dict(fontsize=8, color=S.INK, ha="center")
    # (a) fixed vs new outliers
    a = ax[0]
    for i, (ds, _) in enumerate(S.DS):
        c = d[f"{ds}/r{r}"]["_clusters_mp+uod"]
        fx, nw = 100 * c["fixed"] / c["n_valid"], 100 * c["new"] / c["n_valid"]
        a.bar(i, fx, 0.6, color=SC["mp+uod+pinn"], zorder=2)
        a.bar(i, -nw, 0.6, color="white", edgecolor=SC["mp+uod+pinn"], hatch="///", linewidth=1, zorder=2)
        a.text(i, fx + 0.3, f"+{fx:.1f}" if fx >= 0.05 else "0", va="bottom", **lab)
        a.text(i, -nw - 0.3, f"−{nw:.1f}" if nw >= 0.05 else "0", va="top", **lab)
    a.axhline(0, color=S.INK2, lw=0.8)
    lim = max(abs(np.array(a.get_ylim()))) * 1.35
    a.set_ylim(-lim, lim)
    a.set_ylabel("% of evaluated vectors")
    a.set_title("(a) outliers fixed (+) / created (−)", fontsize=9.5)
    # (b) clusters
    b = ax[1]
    for i, (ds, _) in enumerate(S.DS):
        c = d[f"{ds}/r{r}"]["_clusters_mp+uod"]
        b.bar(i, c["share_ge5"], 0.6, color=SC["mp+uod"], zorder=2)
        b.text(i, c["share_ge5"] + 0.02, f"{100 * c['share_ge5']:.0f}%", va="bottom", **lab)
    b.set_ylim(0, 1.15); b.set_yticks([0, 0.5, 1]); b.set_yticklabels(["0", "50%", "100%"])
    b.set_ylabel("outliers in clusters ≥ 5")
    b.set_title("(b) outliers in clusters (MP + UOD)", fontsize=9.5)
    # (c) absolute 2-D divergence of the reference (measured fields) vs the exact synthetic field
    c_ = ax[2]
    for i, (ds, _) in enumerate(S.DS):
        g = d.get(f"{ds}/_div", {})
        meas = g.get("cc_true_frames", g.get("reference"))
        if meas:
            c_.bar(i, meas["div_rms"], 0.6, color=S.COL["gt"], zorder=2)
            c_.text(i, meas["div_rms"] + 0.0006, f"{meas['div_rms']:.3f}", va="bottom", **lab)
        if "true" in g:
            c_.plot([i - 0.36, i + 0.36], [g["true"]["div_rms"]] * 2, color=S.INK, lw=2.2, zorder=3)
    c_.set_ylim(0, 0.034)
    c_.set_ylabel("rms(∇·u) [1/Δt]")
    c_.set_title("(c) 2-D divergence of reference", fontsize=9.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    c_.legend([Patch(color=S.COL["gt"]), Line2D([], [], color=S.INK, lw=2.2)], ["measured (CC of the frames)", "exact (synthetic)"],
              loc="upper left", fontsize=7.5, handlelength=1.2)
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(names, fontsize=8.5); a_.tick_params(axis="x", length=0)
        a_.grid(axis="y"); a_.set_axisbelow(True)
    S.save(fig, "res_pinn_diag.png")


# ------------------------------------------------------------------------------ maps
def maps():
    import torch
    import evaluate_paper as EP, evaluate_real as R, evaluate_pinn as P
    from pivsep.synth import SynthParams, make_sample, gt_window_displacement
    from pivsep.correlation import window_grid, multipass_piv
    from pivsep.separation import unet_split
    from pivsep.postproc import uod_median, pinn_fit
    torch.set_num_threads(os.cpu_count() or 4)
    models = EP.load_models()
    r = 0.7
    cases = {}
    s = make_sample(np.random.default_rng(1000), SynthParams(r=r))
    cx, cy = window_grid(s["img"].shape, EP.WIN, EP.STEP)[2:]
    gu, gv = gt_window_displacement(s["flow"], cx, cy, EP.WIN)
    cases["synth"] = (s["img"], (cx, cy, gu, gv), np.hypot(gu, gv) > 2, None, None)
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()[EP.TEST2_EVAL[0]]))
    cx, cy, ru, rv = multipass_piv(A, B)
    cases["test2"] = (R.make_double(A, B, r), (cx, cy, ru, rv), ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2), None, None)
    import evaluate_pivc as PC
    frames, M = PC.load_case(); A, B = frames["bgsub"]
    cx, cy, ru, rv = multipass_piv(A, B)
    fm = PC.centre_mask(M, cx, cy)
    cases["caseC"] = (R.make_double(A, B, r), (cx, cy, ru, rv), fm & ~uod_median(ru, rv)[2] & (np.hypot(ru, rv) > 2), fm, M)
    for ds, (I, ref, valid, fm, M) in cases.items():
        cx, cy, ru, rv = ref
        sA, sB = unet_split(models[EP.MAIN_MODEL[ds]], I)
        u, v = multipass_piv(sA, sB, mask_r=EP.MASK_R)[2:]
        uu, vu = uod_median(u, v)[:2]
        pu, pv = pinn_fit(cx, cy, uu, vu, steps=P.STEPS, fit_mask=fm)
        draw_map(ds, I, ref, valid, M, (uu, vu), (pu, pv))


def draw_map(ds, I, ref, valid, M, f_uod, f_pinn):
    import matplotlib.colors as mc
    cx, cy, ru, rv = ref
    ys, xs = S.CROP[ds]
    y0, y1 = (ys.start or 0), (ys.stop or I.shape[0]); x0, x1 = (xs.start or 0), (xs.stop or I.shape[1])
    img = I if M is None else np.where(M[:I.shape[0], :I.shape[1]], I, np.nan)
    img = img[y0:y1, x0:x1]
    sel = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    rows = np.where(sel.any(1))[0]; cols = np.where(sel.any(0))[0]
    step = max(1, int(round(len(cols) / S.ARROWS)))
    sub = np.zeros_like(sel); sub[np.ix_(rows[::step], cols[::step])] = True; sub &= sel
    spacing = (cx[0, 1] - cx[0, 0]) * step
    mag = np.nanmedian(np.hypot(ru, rv)[valid]); scale = mag / (0.8 * spacing)
    h = S.W / 4 + 0.32
    fig, ax = plt.subplots(1, 4, figsize=(S.W, h))
    fig.subplots_adjust(left=0.003, right=0.997, top=1 - 0.32 / h, bottom=0.01, wspace=0.03)
    rgba = {k: np.array(mc.to_rgba(c)) for k, c in (("r", S.RED), ("w", S.WHITE), ("g", S.GREY))}
    vmax = 2.5 * np.nanpercentile(img, 99.5)

    def quiv(a, u, v, out):
        col = np.where(out[..., None], rgba["r"], np.where(valid[..., None], rgba["w"], rgba["g"]))
        a.quiver(cx[sub] - x0, cy[sub] - y0, np.nan_to_num(u[sub]), -np.nan_to_num(v[sub]), color=col[sub],
                 angles="xy", scale_units="xy", scale=scale, width=0.008, headwidth=3.5, headlength=4, headaxislength=3.6)
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#d9d9d9")
        for sp in a.spines.values():
            sp.set_visible(False)
    for a in ax[:3]:
        a.imshow(img, cmap="gray", vmin=0, vmax=vmax, extent=(-0.5, x1 - x0 - 0.5, y1 - y0 - 0.5, -0.5))
    quiv(ax[0], ru, rv, np.zeros_like(valid))
    ax[0].set_title("reference" + (" (true)" if ds == "synth" else " (real A, B)"))
    for a, (k, (u, v)) in zip(ax[1:3], [("mp+uod", f_uod), ("mp+uod+pinn", f_pinn)]):
        out = valid & (np.hypot(u - ru, v - rv) > 1)
        quiv(a, u, v, out)
        a.set_title(f"{SL[k]}: {100 * out.sum() / valid.sum():.1f}%", color=SC[k] if k != "mp+uod+pinn" else "#c2417a")
    # 2-D divergence of the reference, normalised by rms |grad u|
    g = [np.gradient(np.nan_to_num(f), S_H, axis=ax_) for f in (ru, rv) for ax_ in (1, 0)]
    div = g[0] + g[3]; gn = np.sqrt(np.mean(sum(x ** 2 for x in g)[valid]))
    dn = np.where(valid, div / max(gn, 1e-12), np.nan)
    ext = (cx[0, 0] - S_H / 2 - x0, cx[0, -1] + S_H / 2 - x0, cy[-1, 0] + S_H / 2 - y0, cy[0, 0] - S_H / 2 - y0)
    im = ax[3].imshow(dn, cmap="RdBu_r", vmin=-1.5, vmax=1.5, extent=ext, interpolation="nearest")
    ax[3].set_xlim(-0.5, x1 - x0 - 0.5); ax[3].set_ylim(y1 - y0 - 0.5, -0.5)
    ax[3].set_title("reference ∇·u / rms|∇u|")
    cax = ax[3].inset_axes([0.06, 0.05, 0.4, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1.5, 0, 1.5])
    cb.ax.tick_params(labelsize=7.5, colors=S.INK, length=2, pad=1)
    cax.set_facecolor("white")
    S.save(fig, f"res_pinn_map_{ds}.png")


S_H = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/eval_pinn.json")
    ap.add_argument("--merge", nargs="+", default=None, help="partial result files to merge into --inp first")
    ap.add_argument("--skip-maps", action="store_true")
    a = ap.parse_args()
    S.style()
    if a.merge:
        d = {}
        for f in a.merge:
            for k, v in json.load(open(f)).items():
                d.setdefault(k, v)
        with open(a.inp, "w") as f:
            json.dump(d, f, indent=1)
    d = json.load(open(a.inp))
    fig_stage(d, "outlier", "res_pinn_outlier.png", "outlier rate")
    fig_stage(d, "median", "res_pinn_median.png", "median error [px]", log=False,
              tops={"synth": 0.4, "lml": 0.6, "test1": 0.6, "test2": 0.6, "caseC": 0.6})
    tops = {ds: 1.2e3 * max(d[f"{ds}/r{r}"][k]["vort_rms"] for r in S.R for k in ("mp+uod", "mp+uod+pinn"))
            for ds, _ in S.DS}
    fig_stage(d, "vort_rms", "res_pinn_vort.png", "RMS vorticity error [10⁻³/Δt]", log=False, tops=tops,
              fmt="{:.1f}", scale=1e3)
    fig_diag(d)
    if not a.skip_maps:
        maps()


if __name__ == "__main__":
    main()
