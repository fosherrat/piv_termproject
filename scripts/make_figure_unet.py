"""How the separation network works (paper/fig_unet.pdf).

(a) architecture: U-Net -> soft mask m -> A_hat = m I', B_hat = (1-m) I' -> multi-pass CC
(b) example (synthetic, r = 0.7): input, mask on particle pixels, separated frames vs truth
(c) correlation planes of one window: auto-correlation of I, nonlinear AC, CC of the separated frames
(d) probe: per-pair mask difference dm = m(first image) - m(second image) vs the pair brightness
    ratio rho = b/a (all pairs of an image share rho), and pairs whose order is reversed while the
    majority keeps rho = 0.7 (context)
(e) per-window direction margin (R(+d) - R(-d)) / (|R(+d)| + |R(-d)|) on synthetic images:
    separated frames vs nonlinear auto-correlation (Eqs. vote / asym of the paper)

python scripts/make_figure_unet.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
import torch

from pivsep.synth import SynthParams, make_sample, render, gt_window_displacement
from pivsep.separation import unet_split, normalize
from pivsep.correlation import window_grid, _windows, _corr
import evaluate_paper as EP

BLUE, ORANGE, GREY, INK, INK2 = "#257de0", "#f47230", "#e6e6e6", "#191919", "#666666"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False, "pdf.fonttype": 42})
DIV = LinearSegmentedColormap.from_list("ab", [ORANGE, GREY, BLUE])     # m = 0 -> B, m = 1 -> A


# ------------------------------------------------------------------ (a) architecture
def draw_arch(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.set_axis_off()
    def box(x, y, w, h, txt, fc="#f2f6fc", ec=BLUE, fs=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)
    def arr(x0, y0, x1, y1, style="-|>", ls="-", c=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=0.8,
                                     color=c, linestyle=ls))
    box(0, 11, 9, 8, "double\nexposure\n$I'$", fc="white", ec=INK2)
    ch, ys, hs = [24, 48, 96, 192], [20, 14, 8, 2], [8, 6.5, 5, 4]
    xe, xd = [12, 20, 28, 36], [None, 60, 52, 44]
    for k in range(4):
        box(xe[k], ys[k], 7, hs[k] * 0.7, f"{ch[k]}", fs=6)
        if k:
            arr(xe[k - 1] + 3.5, ys[k - 1], xe[k] + 3.5, ys[k] + hs[k] * 0.7, c=INK2)
    for k in (3, 2, 1):
        x0 = xd[k + 1] if k < 3 and xd[k + 1] else xe[3]
    for k in (2, 1, 0):
        xx = 44 + (2 - k) * 8
        box(xx, ys[k], 7, hs[k] * 0.7, f"{ch[k]}", fs=6)
        prev = (xe[3] + 3.5, ys[3] + hs[3] * 0.7) if k == 2 else (44 + (1 - k) * 8 + 3.5, ys[k + 1] + hs[k + 1] * 0.7)
        arr(prev[0], prev[1], xx + 3.5, ys[k], c=INK2)
        arr(xe[k] + 7.3, ys[k] + hs[k] * 0.35, xx - 0.3, ys[k] + hs[k] * 0.35, ls="--", c="#9c9c9c")
    arr(9.3, 15, 11.7, ys[0] + 2.8)
    box(69.5, ys[0], 8, hs[0] * 0.7, "mask $m$\n(sigmoid)", fc="#f2f6fc")
    arr(60 + 7.3, ys[0] + 2.8, 69.2, ys[0] + 2.8)
    box(84, 21, 13, 5, r"$\hat A = m\,I'$", fc="#eaf2fc", ec=BLUE)
    box(84, 12, 13, 5, r"$\hat B=(1-m)\,I'$", fc="#fdeee6", ec=ORANGE)
    arr(78, 23.5, 83.7, 23.5); arr(78, 21.5, 83.7, 14.5)
    box(84, 2, 13, 6, "multi-pass CC\n(deformation)", fc="white", ec=INK2)
    arr(90.5, 11.7, 90.5, 8.3)
    ax.text(26, 28.5, "encoder: 2 × (conv 3×3, BN, ReLU), max-pool", fontsize=6, color=INK2, ha="center")
    ax.text(52, 28.5, "decoder: up-conv + skip (dashed)", fontsize=6, color=INK2, ha="center")
    ax.text(88, 28.5, r"$\hat A+\hat B=I'$ (intensity conserved)", fontsize=6, color=INK2, ha="center")


# ------------------------------------------------------------------ (b, c) example
def example(model, r=0.7, crop=(slice(96, 176), slice(96, 176))):
    s = make_sample(np.random.default_rng(1000), SynthParams(r=r))
    I = s["img"]; Ah, Bh = unet_split(model, I); Ip = normalize(I)[0]
    m = np.where(Ip > 0.08, Ah / np.maximum(Ip, 1e-6), np.nan)
    return s, I, Ah, Bh, m, crop


def comp(a, b):
    """two-hue composite on black: a -> blue, b -> orange."""
    c = lambda h: np.array(matplotlib.colors.to_rgb(h))
    s = max(np.percentile(a + b, 99.5), 1e-6)
    return np.clip(a[..., None] / s * c(BLUE) + b[..., None] / s * c(ORANGE), 0, 1)


def planes(s, I, Ah, Bh, win=32):
    cx, cy = window_grid(I.shape, win, 16)[2:]
    gu, gv = gt_window_displacement(s["flow"], cx, cy, win)
    sy, sx = window_grid(I.shape, win, 16)[:2]
    k = (7, 7)
    wI = _windows(I, sy[k][None], sx[k][None], win)
    w0 = wI - wI.mean((1, 2), keepdims=True)
    ac = _corr(w0, w0)[0]
    wp = np.clip(wI - np.median(wI, axis=(1, 2), keepdims=True), 0, None)
    nl = _corr(wp ** 2, wp)[0]
    wa = _windows(Ah, sy[k][None], sx[k][None], win); wb = _windows(Bh, sy[k][None], sx[k][None], win)
    cc = _corr(wa - wa.mean(), wb - wb.mean())[0]
    return [ac, nl, cc], (gu[k], gv[k])


# ------------------------------------------------------------------ (d) probe
def probe(model, rng, rho, flip_frac=0.0, rho_flip=None, S=256, d=(5.0, 2.0)):
    pad = 12; n = int(0.03 * (S + 2 * pad) ** 2)
    x = rng.uniform(-pad, S + pad, n); y = rng.uniform(-pad, S + pad, n); z = rng.uniform(-1, 1, n)
    a = 0.6 * np.exp(-z ** 2 / (2 * 0.16)); flip = rng.random(n) < flip_frac
    b = a * np.where(flip, rho_flip or rho, rho); dp = np.clip(rng.normal(2.8, .5, n), 1.5, 5)
    I = np.clip(render((S, S), x, y, a, dp) + render((S, S), x + d[0], y + d[1], b, dp) + 0.02
                + rng.normal(0, 0.015, (S, S)), 0, 1).astype(np.float32)
    m = unet_split(model, I)[0] / np.maximum(normalize(I)[0], 1e-6)
    dm = ndimage.map_coordinates(m, [y, x], order=1) - ndimage.map_coordinates(m, [y + d[1], x + d[0]], order=1)
    ok = (x > 8) & (x < S - 16) & (y > 8) & (y < S - 16)
    return dm[ok], (a * b)[ok], flip[ok]


def wq(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]) / w.sum()
    return v[o][np.searchsorted(c, q)]


# ------------------------------------------------------------------ (e) window margins
def margins(model, r, n_img=12, win=32):
    out = {"sep": [], "asym": []}
    for i in range(n_img):
        s = make_sample(np.random.default_rng(1000 + i), SynthParams(r=r))
        I = s["img"]; Ah, Bh = unet_split(model, I)
        sy, sx, cx, cy = window_grid(I.shape, win, 16)
        gu, gv = gt_window_displacement(s["flow"], cx, cy, win)
        wI = _windows(I, sy, sx, win); wp = np.clip(wI - np.median(wI, axis=(1, 2), keepdims=True), 0, None)
        C = {"asym": _corr(wp ** 2, wp), "sep": _corr(_windows(Ah, sy, sx, win), _windows(Bh, sy, sx, win))}
        for k, P in C.items():
            for p, u, v in zip(P, gu.ravel(), gv.ravel()):
                if np.hypot(u, v) <= 2:
                    continue
                p = p - np.median(p)
                f = lambda a, b: ndimage.map_coordinates(p, [[win // 2 + b], [win // 2 + a]], order=1)[0]
                cp, cm = f(u, v), f(-u, -v)
                out[k].append((cp - cm) / (abs(cp) + abs(cm) + 1e-12))
    return {k: np.array(v) for k, v in out.items()}


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    model = EP.load_models()["zeroshot"]
    fig = plt.figure(figsize=(7.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.15, 1.0, 1.0, 1.25])
    ax = fig.add_subplot(gs[0, :]); draw_arch(ax); ax.set_title("(a) separation network", loc="left", fontsize=8.5)

    s, I, Ah, Bh, m, cr = example(model)
    sc = lambda a: a[cr]
    panels = [(sc(I), "gray", "input $I$ ($r=0.7$)"),
              (sc(m), DIV, "mask $m$ on particle pixels"),
              (comp(sc(Ah), sc(Bh)), None, r"separated: $\hat A$ blue, $\hat B$ orange"),
              (comp(sc(s["A"]), sc(s["B"])), None, "truth: $A$ blue, $B$ orange")]
    for j, (im, cm, t) in enumerate(panels):
        a = fig.add_subplot(gs[1, j])
        h = a.imshow(im, cmap=cm, vmin=0, vmax=(1 if cm is DIV else None) if cm is not None else None)
        a.set_axis_off(); a.set_title(("(b) " if j == 0 else "") + t, fontsize=7, loc="left" if j == 0 else "center")
        if cm is DIV:
            cb = fig.colorbar(h, ax=a, fraction=0.046, pad=0.02, ticks=[0, 0.5, 1]); cb.ax.tick_params(labelsize=6)
            cb.ax.set_yticklabels(["B", "0.5", "A"])

    P, (gu, gv) = planes(s, I, Ah, Bh)
    titles = ["auto-correlation of $I$", "nonlinear AC $\\sum I^2(x)I(x+s)$", r"CC of $\hat A,\hat B$"]
    for j, (p, t) in enumerate(zip(P, titles)):
        a = fig.add_subplot(gs[2, j])
        e = 16
        yy, xx = np.mgrid[:2 * e, :2 * e] - e
        vmax = p[xx ** 2 + yy ** 2 > 9].max(); vmin = np.percentile(p, 5)
        a.imshow(np.clip(p, vmin, vmax), cmap="magma", extent=(-e - .5, e - .5, e - .5, -e - .5))
        a.plot([gu], [gv], marker="o", mfc="none", mec=BLUE, ms=9, mew=1.2)
        a.plot([-gu], [-gv], marker="o", mfc="none", mec="white", ms=9, mew=1.0, ls="none")
        a.set_xticks([-10, 0, 10]); a.set_yticks([-10, 0, 10]); a.tick_params(labelsize=6)
        a.set_title(("(c) " if j == 0 else "") + t, fontsize=7, loc="left" if j == 0 else "center")
    a = fig.add_subplot(gs[2, 3]); a.set_axis_off()
    a.text(0, 0.5, "one $32\\times32$ window,\ncolour scale outside the\ncentre (self peak clipped)\nblue ring: $+\\mathbf{d}$ (true)\nwhite ring: $-\\mathbf{d}$\n\n"
                   "AC: equal peaks at $\\pm\\mathbf{d}$\nnonlinear AC: $+\\mathbf{d}$ higher\nby a factor $\\approx 1/r$\n"
                   "CC of separated frames:\nsingle peak at $+\\mathbf{d}$", fontsize=6.5, va="center", color=INK)

    rng = np.random.default_rng(0)
    rhos = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0]
    st = []
    for rho in rhos:
        v, w, _ = map(np.concatenate, zip(*[probe(model, rng, rho) for _ in range(3)]))
        st.append((np.average(v, weights=w), wq(v, w, 0.25), wq(v, w, 0.75)))
    st = np.array(st)
    a = fig.add_subplot(gs[3, :2])
    a.axhline(0, color=INK2, lw=0.6); a.axvline(1, color=INK2, lw=0.6, ls=":")
    a.fill_between(rhos, st[:, 1], st[:, 2], color=BLUE, alpha=0.18, lw=0)
    a.plot(rhos, st[:, 0], "-o", color=BLUE, lw=2, ms=4, label="all pairs share $\\rho$")
    vv, ww, ff = map(np.concatenate, zip(*[probe(model, rng, 0.7, 0.2, 1 / 0.7) for _ in range(4)]))
    a.errorbar([1 / 0.7], [np.average(vv[ff], weights=ww[ff])],
               yerr=[[np.average(vv[ff], weights=ww[ff]) - wq(vv[ff], ww[ff], 0.25)],
                     [wq(vv[ff], ww[ff], 0.75) - np.average(vv[ff], weights=ww[ff])]],
               fmt="D", color=ORANGE, ms=5, capsize=2, label="reversed pairs among\n80 % pairs with $\\rho=0.7$")
    a.set_xscale("log"); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2]); a.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2"])
    a.set_xlabel(r"pair brightness ratio $\rho=b_i/a_i$"); a.set_ylabel(r"$\Delta m_i=m_i^A-m_i^B$")
    a.set_ylim(-1, 1); a.legend(fontsize=6, loc="lower left")
    a.set_title("(d) per-pair decision (probe)", loc="left", fontsize=8.5)

    a = fig.add_subplot(gs[3, 2:])
    bins = np.linspace(-1, 1, 41)
    for r, ls in [(0.7, "-"), (0.9, "--")]:
        M = margins(model, r)
        for k, c, lab in [("asym", ORANGE, "nonlinear AC"), ("sep", BLUE, "CC of $\\hat A,\\hat B$")]:
            h, _ = np.histogram(np.clip(M[k], -1, 1), bins)
            a.step(bins[:-1], h / h.sum(), where="post", color=c, ls=ls, lw=1.6, label=f"{lab}, $r={r}$")
    a.axvline(0, color=INK2, lw=0.6)
    a.set_xlabel(r"window margin $(R(\mathbf{d})-R(-\mathbf{d}))/(|R(\mathbf{d})|+|R(-\mathbf{d})|)$")
    a.set_ylabel("fraction of windows"); a.legend(fontsize=6, loc="upper left")
    a.set_title("(e) direction margin per window", loc="left", fontsize=8.5)
    fig.savefig("paper/fig_unet.pdf"); fig.savefig("paper/fig_unet.png", dpi=150)
    print("saved paper/fig_unet.pdf")


if __name__ == "__main__":
    main()
