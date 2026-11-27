"""Slide-sized figures (larger fonts, one panel per file) for slides/piv_pulse_coding.pptx.

python scripts/make_slide_figures.py      -> slides/fig/*.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT = "slides/fig"
DPI = 220
BLUE, ORANGE, AQUA, YELLOW, INK, INK2 = "#257de0", "#f47230", "#00b894", "#f2a600", "#191919", "#666666"
NAVY = "#0F0F70"


def style():
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "legend.frameon": False, "grid.color": "#e6e6e6"})


def save(fig, name, transparent=False):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight", facecolor="none" if transparent else "white",
                transparent=transparent)
    plt.close(fig)


def equations():
    eqs = {
        "eq_model": r"$a_i=I_0 f(z_i),\quad b_i=r\,I_0 f(z_i+\delta),\quad f(z)=e^{-z^2/2\sigma^2}$",
        "eq_flip": r"$P_{\rm flip}(r,\delta)=\frac{1}{2}\max\left\{0,\;1-\frac{|\delta|}{2}-\frac{\sigma^2\ln(1/r)}{|\delta|}\right\}$",
        "eq_rstar": r"$r\;\leq\; r^\ast=\exp\!\left[-\,w\,(1-w/2)\,/\,\sigma^2\right]$",
        "eq_vote": r"$R_{\hat A\hat B}(+\mathbf{d})-R_{\hat A\hat B}(-\mathbf{d})\;\propto\;\sum_i a_i b_i\,(m_i^A-m_i^B)$",
        "eq_asym": r"$C(+\mathbf{d})-C(-\mathbf{d})\;\propto\;\sum_i a_i b_i\,(a_i-b_i)$",
        "eq_margin": r"$\mu_{\rm AC}=\frac{1-r}{1+r}\quad(0.18\ {\rm at}\ r=0.7)\qquad \mu_{\rm sep}\to 1$",
        "eq_hoeff": r"$P(\mathrm{wrong\ direction})\leq\exp\left[-\frac{1}{2}N_{\rm eff}(1-2\bar\varepsilon)^2\right]$",
        "eq_split": r"$\hat A=m\,I',\qquad \hat B=(1-m)\,I',\qquad \hat A+\hat B=I'$",
    }
    for k, e in eqs.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, e, fontsize=22, color=INK)
        save(fig, k + ".png", transparent=True)


def theory_panels():
    import make_figure_theory as T
    rr = np.linspace(0.45, 0.99, 28)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(rr, [T.bayes_single(r) for r in rr], color=INK, lw=2.5)
    ax.set_ylim(0, 0.5); ax.set_xlabel("pulse ratio $r$"); ax.set_ylabel("misassigned particles")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="y")
    ax.set_title("best possible single-particle classification", fontsize=12)
    save(fig, "s_bayes.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for w, ls in [(0.05, ":"), (0.1, "-"), (0.15, "--")]:
        ax.plot(rr, [T.flip(r, w) for r in rr], color=BLUE, ls=ls, lw=2.5, label=f"$w={w}$")
    rs = np.exp(-0.1 * (1 - 0.05) / 0.16)
    ax.axvline(rs, color=ORANGE, lw=1.5); ax.text(rs + 0.01, 0.44, f"$r^*={rs:.2f}$ ($w=0.1$)", color=ORANGE,
                                                  ha="left", fontsize=12)
    ax.set_xlabel("pulse ratio $r$"); ax.set_ylabel("reversed pairs"); ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(0.3, 0.62), title="out-of-plane $|\\delta|\\leq w$", fontsize=11, title_fontsize=11)
    save(fig, "s_flip.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    rp = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    e = np.array([T.window_error(r, 0.03) for r in rp])
    ax.semilogy(rp, np.maximum(e, 1e-5), "-", color=BLUE, lw=2.5, label="theory: ideal pair vote")
    d = json.load(open("results/eval_mp_synth.json"))["r"]
    rm = np.array([float(k) for k in d]); em = np.array([d[k]["unet_mp:wide_bce"]["sign_err"] for k in d])
    ax.semilogy(rm, np.maximum(em, 1e-5), "o", color=INK, ms=8, label="U-Net + CC (measured)")
    ax.set_ylim(1e-4, 1); ax.set_xlabel("pulse ratio $r$"); ax.set_ylabel("window direction error")
    ax.grid(axis="y"); ax.legend(loc="upper left", fontsize=11)
    ax.annotate("floor: 1 of 2700 vectors", (0.65, 3.7e-4), (0.55, 3e-3), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2))
    save(fig, "s_theory_b.png")


def unet_panels():
    import make_figure_unet as U
    import evaluate_paper as EP
    torch.set_num_threads(os.cpu_count() or 4)
    model = EP.load_models()["zeroshot"]
    style()                      # make_figure_unet sets small paper fonts on import

    fig, ax = plt.subplots(figsize=(6.6, 2.15)); U.draw_arch(ax)
    for t in ax.texts:                       # header notes: shorter, spread out for the slide size
        if t.get_position()[1] == 28.5:
            x = t.get_position()[0]
            if x < 30: t.set_text("encoder: conv-BN-ReLU ×2, pool"); t.set_position((20, 28.5))
            elif x < 60: t.set_text("decoder: up-conv + skip"); t.set_position((53, 28.5))
            else: t.set_text("intensity conserved"); t.set_position((90.5, 28.5))
    save(fig, "s_arch.png")

    s, I, Ah, Bh, m, cr = U.example(model)
    sc = lambda a: a[cr]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2), constrained_layout=True)
    pan = [(sc(I), "gray", "input $I$ ($r=0.7$)"), (sc(m), U.DIV, "mask $m$"),
           (U.comp(sc(Ah), sc(Bh)), None, "separated"), (U.comp(sc(s["A"]), sc(s["B"])), None, "truth")]
    for a, (im, cm, t) in zip(axs, pan):
        h = a.imshow(im, cmap=cm, vmin=0, vmax=(1 if cm is U.DIV else None) if cm is not None else None)
        a.set_axis_off(); a.set_title(t, fontsize=13)
        if cm is U.DIV:
            cb = fig.colorbar(h, ax=a, fraction=0.046, pad=0.02, ticks=[0, 1]); cb.ax.set_yticklabels(["B", "A"])
    save(fig, "s_sep_example.png")

    P, (gu, gv) = U.planes(s, I, Ah, Bh)
    titles = ["auto-correlation", "nonlinear AC", "CC of separated"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for a, p, t in zip(axs, P, titles):
        e = 16; yy, xx = np.mgrid[:2 * e, :2 * e] - e
        vmax = p[xx ** 2 + yy ** 2 > 9].max(); vmin = np.percentile(p, 5)
        a.imshow(np.clip(p, vmin, vmax), cmap="magma", extent=(-e - .5, e - .5, e - .5, -e - .5))
        a.plot([gu], [gv], "o", mfc="none", mec=BLUE, ms=16, mew=2.5)
        a.plot([-gu], [-gv], "o", mfc="none", mec="white", ms=16, mew=2)
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=20)
    save(fig, "s_planes.png")
    fig, a = plt.subplots(figsize=(3.6, 3.6))
    p = P[0]; e = 16; yy, xx = np.mgrid[:2 * e, :2 * e] - e
    a.imshow(np.clip(p, np.percentile(p, 5), p[xx ** 2 + yy ** 2 > 9].max()), cmap="magma",
             extent=(-e - .5, e - .5, e - .5, -e - .5))
    a.plot([gu], [gv], "o", mfc="none", mec="white", ms=16, mew=2); a.plot([-gu], [-gv], "o", mfc="none", mec="white", ms=16, mew=2)
    a.text(gu, gv - 4, "+d ?", color="white", ha="center", fontsize=13); a.text(-gu, -gv + 5.5, "−d ?", color="white", ha="center", fontsize=13)
    a.set_xticks([]); a.set_yticks([]); a.set_title("auto-correlation of one window", fontsize=12)
    save(fig, "s_ac_plane.png")

    rng = np.random.default_rng(0)
    rhos = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0]
    st = []
    for rho in rhos:
        v, w, _ = map(np.concatenate, zip(*[U.probe(model, rng, rho) for _ in range(3)]))
        st.append((np.average(v, weights=w), U.wq(v, w, 0.25), U.wq(v, w, 0.75)))
    st = np.array(st)
    vv, ww, ff = map(np.concatenate, zip(*[U.probe(model, rng, 0.7, 0.2, 1 / 0.7) for _ in range(4)]))
    fig, a = plt.subplots(figsize=(6.0, 4.0))
    a.axhline(0, color=INK2, lw=0.8); a.axvline(1, color=INK2, lw=0.8, ls=":")
    a.fill_between(rhos, st[:, 1], st[:, 2], color=BLUE, alpha=0.18, lw=0)
    a.plot(rhos, st[:, 0], "-o", color=BLUE, lw=2.5, ms=6, label="all pairs share $\\rho$")
    mu = np.average(vv[ff], weights=ww[ff])
    a.errorbar([1 / 0.7], [mu], yerr=[[mu - U.wq(vv[ff], ww[ff], .25)], [U.wq(vv[ff], ww[ff], .75) - mu]],
               fmt="D", color=ORANGE, ms=8, capsize=3, label="reversed pair among\nnormal neighbours")
    a.set_xscale("log"); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2]); a.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2"])
    a.set_xlabel(r"pair brightness ratio $\rho=b_i/a_i$"); a.set_ylabel(r"$\Delta m_i=m_i^A-m_i^B$")
    a.set_ylim(-1, 1); a.legend(loc="lower left", fontsize=11)
    save(fig, "s_probe.png")

    fig, a = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(-1, 1, 41)
    M = U.margins(model, 0.7)
    for k, c, lab in [("asym", ORANGE, "nonlinear AC"), ("sep", BLUE, "CC of separated frames")]:
        h, _ = np.histogram(np.clip(M[k], -1, 1), bins)
        a.step(bins[:-1], h / h.sum(), where="post", color=c, lw=2.5, label=lab)
    a.axvline(0, color=INK2, lw=0.8); a.axvline((1 - .7) / 1.7, color=ORANGE, lw=1, ls=":")
    a.text((1 - .7) / 1.7 + 0.03, 0.45, "$(1-r)/(1+r)$", color=ORANGE, fontsize=11)
    a.set_xlabel("direction margin per window ($r=0.7$)"); a.set_ylabel("fraction of windows")
    a.legend(loc="upper left", fontsize=11)
    save(fig, "s_margin.png")


def dataset_thumbs():
    import evaluate_real as R, evaluate_lml as E, evaluate_pivc as PC
    from pivsep.synth import SynthParams, make_sample
    ims = {"synth": make_sample(np.random.default_rng(1000), SynthParams(r=0.7))["img"],
           "lml": E.to_model_scale(np.load("data/lml/lml_samples.npz")["translation|0.7|0.01|0|I"])}
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()["test1"]))
    ims["test1"] = R.make_double(A, B, 0.7)
    A, B = R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in R.pairs()["test2_3"]))
    ims["test2"] = R.make_double(A, B, 0.7)
    fr, M = PC.load_case(); A, B = fr["bgsub"]; ims["caseC"] = np.where(M[:A.shape[0], :A.shape[1]], R.make_double(A, B, 0.7), 0)
    crops = {"synth": (slice(0, 256), slice(0, 256)), "lml": (slice(0, 256), slice(0, 256)),
             "test1": (slice(0, 368), slice(0, 368)), "test2": (slice(None), slice(None)), "caseC": (slice(0, 512), slice(240, 752))}
    for k, im in ims.items():
        fig, a = plt.subplots(figsize=(3, 3))
        c = im[crops[k]]
        a.imshow(c, cmap="gray", vmin=0, vmax=np.percentile(c, 99.5)); a.set_axis_off()
        save(fig, f"s_ds_{k}.png")


def main():
    os.makedirs(OUT, exist_ok=True); style()
    equations(); theory_panels(); dataset_thumbs(); unet_panels()
    print("saved", OUT)


if __name__ == "__main__":
    main()
