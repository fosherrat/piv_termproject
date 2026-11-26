"""Theory figure for the paper (paper/fig_theory.pdf).

(a) brightness-only limits under a Gaussian light sheet, f(z) = exp(-z^2 / 2 sigma^2), z ~ U(-1, 1):
    - Bayes error of classifying a single particle as pulse 1 / pulse 2 from its brightness
    - probability that a particle pair is mis-ordered by out-of-plane motion,
      P[r f(z + delta) / f(z) >= 1], delta ~ U(-w, w)   (closed form, Eq. (flip) in the paper)
(b) window-level direction error of an ideal pair classifier (each pair ordered by its true
    brightness ratio, window direction = sign of the intensity-weighted vote, N ~ Poisson(ppp * 32^2))
    against the measured direction error of U-Net + multi-pass CC on our synthetic images.

python scripts/make_figure_theory.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG, W = 0.4, 0.1


def bayes_single(r, n=2_000_000, rng=np.random.default_rng(0)):
    z = rng.uniform(-1, 1, n); la = -z ** 2 / (2 * SIG ** 2); lb = la + np.log(r)
    bins = np.linspace(lb.min(), 0, 4001)
    ha, _ = np.histogram(la, bins, density=True); hb, _ = np.histogram(lb, bins, density=True)
    return 0.5 * np.sum(np.minimum(ha, hb)) * (bins[1] - bins[0])


def flip_given_delta(r, d):
    d = np.abs(d) + 1e-12
    return 0.5 * np.clip(1 - d / 2 - SIG ** 2 * np.log(1 / r) / d, 0, 1)


def flip(r, w):
    d = np.linspace(0, w, 4001)[1:]
    return flip_given_delta(r, d).mean()


def window_error(r, ppp=0.03, w=W, nwin=200_000, rng=np.random.default_rng(1)):
    N = rng.poisson(ppp * 32 * 32, nwin); delta = rng.uniform(-w, w, nwin)
    idx = np.repeat(np.arange(nwin), N); z = rng.uniform(-1, 1, idx.size)
    a = np.exp(-z ** 2 / (2 * SIG ** 2)); b = r * np.exp(-(z + delta[idx]) ** 2 / (2 * SIG ** 2))
    S = np.bincount(idx, a * b * np.where(b < a, 1, -1), nwin)
    return np.mean(S < 0)


def main():
    rr = np.linspace(0.45, 0.99, 28)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    ax[0].plot(rr, [bayes_single(r) for r in rr], "k-", label="single particle (Bayes)")
    for w, ls in [(0.05, ":"), (0.1, "--"), (0.15, "-.")]:
        ax[0].plot(rr, [flip(r, w) for r in rr], "C0" + ls, label=f"pair mis-ordered, $w={w}$")
    ax[0].set_xlabel("pulse ratio $r$"); ax[0].set_ylabel("probability")
    ax[0].set_title("(a) brightness-only limits", fontsize=9); ax[0].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax[0].set_ylim(0, 0.5)

    rp = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    for ppp, c in [(0.03, "C0"), (0.01, "C1")]:
        e = np.array([window_error(r, ppp) for r in rp])
        ax[1].semilogy(rp, np.maximum(e, 1e-5), c + "-", label=f"ideal pair vote, {ppp * 1024:.0f} pairs/window")
    d = json.load(open("results/eval_mp_synth.json"))["r"]
    rm = np.array([float(k) for k in d]); em = np.array([d[k]["unet_mp:wide_bce"]["sign_err"] for k in d])
    ea = np.array([d[k]["asym"]["sign_err"] for k in d])
    ax[1].semilogy(rm, np.maximum(em, 1e-5), "ko", ms=4, label="U-Net + CC (measured)")
    ax[1].semilogy(rm, np.maximum(ea, 1e-5), "ks", mfc="none", ms=4, label="nonlin. AC, SP (measured)")
    ax[1].set_ylim(1e-4, 1); ax[1].set_xlabel("pulse ratio $r$"); ax[1].set_ylabel("window direction error")
    ax[1].set_title("(b) window level", fontsize=9)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    os.makedirs("paper", exist_ok=True)
    fig.savefig("paper/fig_theory.pdf"); fig.savefig("paper/fig_theory.png", dpi=150)
    print("saved paper/fig_theory.pdf")


if __name__ == "__main__":
    main()
