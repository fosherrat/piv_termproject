"""Synthetic PIV image generator with a known ground-truth flow.

Pixel convention: pixel (row=i, col=j) has its centre at (y=i, x=j).
Displacements are in pixels per pulse separation (dt = 1).

Two pulses are recorded:
  frame A : particles at x0, peak intensity I0(z) * 1
  frame B : particles at x0 + displacement, peak intensity I0(z') * r   (pulse coding, r < 1)
The double exposure image is A + B (+ background + noise, clipped to [0, 1]).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf


# --------------------------------------------------------------------------- flows
@dataclass
class RandomFlow:
    """Divergence-free 2-D flow: uniform stream + random Fourier stream function
    + optional Lamb-Oseen vortices.  All terms are exactly incompressible."""
    U: np.ndarray                     # mean displacement (2,)
    k: np.ndarray                     # (M, 2) wave vectors
    amp: np.ndarray                   # (M,) stream-function amplitudes
    phase: np.ndarray                 # (M,)
    vort: list = field(default_factory=list)   # [(xc, yc, gamma, rc)]

    def velocity(self, x, y):
        u = np.full_like(x, self.U[0], dtype=float)
        v = np.full_like(y, self.U[1], dtype=float)
        # psi = sum a cos(kx x + ky y + p);  u = dpsi/dy, v = -dpsi/dx
        arg = x[..., None] * self.k[:, 0] + y[..., None] * self.k[:, 1] + self.phase
        s = -self.amp * np.sin(arg)
        u += (s * self.k[:, 1]).sum(-1)
        v -= (s * self.k[:, 0]).sum(-1)
        for xc, yc, gam, rc in self.vort:
            dx, dy = x - xc, y - yc
            r2 = dx**2 + dy**2 + 1e-9
            ut = gam / (2 * np.pi * r2) * (1 - np.exp(-r2 / rc**2))
            u += -ut * dy
            v += ut * dx
        return u, v

    def displace(self, x, y, n_sub=8):
        """Lagrangian displacement over dt=1 using RK4."""
        h = 1.0 / n_sub
        px, py = x.astype(float).copy(), y.astype(float).copy()
        for _ in range(n_sub):
            k1 = self.velocity(px, py)
            k2 = self.velocity(px + .5 * h * k1[0], py + .5 * h * k1[1])
            k3 = self.velocity(px + .5 * h * k2[0], py + .5 * h * k2[1])
            k4 = self.velocity(px + h * k3[0], py + h * k3[1])
            px += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            py += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return px - x, py - y


def random_flow(rng, size, d_mean=(4.0, 8.0), d_pert=(0.0, 3.0), n_modes=6,
                n_vort=(0, 2)):
    """Random incompressible flow whose displacement magnitude is O(d_mean) px."""
    mag = rng.uniform(*d_mean)
    ang = rng.uniform(0, 2 * np.pi)
    U = mag * np.array([np.cos(ang), np.sin(ang)])
    wl = rng.uniform(48, 256, n_modes)            # wavelengths in px
    th = rng.uniform(0, 2 * np.pi, n_modes)
    kk = (2 * np.pi / wl)[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
    pert = rng.uniform(*d_pert)
    # velocity amplitude of mode ~ amp*|k| ; distribute pert among modes
    amp = pert / np.sqrt(n_modes) / np.linalg.norm(kk, axis=1) * rng.normal(1, .3, n_modes)
    flow = RandomFlow(U, kk, amp, rng.uniform(0, 2 * np.pi, n_modes))
    for _ in range(rng.integers(n_vort[0], n_vort[1] + 1)):
        rc = rng.uniform(12, 40)
        vmax = rng.uniform(0.5, 2.5) * rng.choice([-1, 1])          # peak swirl in px
        gam = vmax * 2 * np.pi * rc / 0.638                    # Lamb-Oseen max at r~1.12rc
        flow.vort.append((rng.uniform(0, size), rng.uniform(0, size), gam, rc))
    return flow


# --------------------------------------------------------------------------- rendering
def render(shape, x, y, peak, dp):
    """Render Gaussian particles (pixel-integrated with erf).  peak: (N,), dp: (N,)"""
    H, W = shape
    img = np.zeros(shape)
    sig = dp / 4.0                                   # I = I0 exp(-8 r^2 / d^2)
    R = 4
    offs = np.arange(-R, R + 1)
    cx, cy = np.round(x).astype(int), np.round(y).astype(int)
    px = cx[:, None] + offs                          # (N, 2R+1)
    py = cy[:, None] + offs
    s2 = (np.sqrt(2) * sig)[:, None]
    gx = .5 * (erf((px + .5 - x[:, None]) / s2) - erf((px - .5 - x[:, None]) / s2))
    gy = .5 * (erf((py + .5 - y[:, None]) / s2) - erf((py - .5 - y[:, None]) / s2))
    norm = (peak * 2 * np.pi * sig**2)[:, None, None]
    val = norm * gy[:, :, None] * gx[:, None, :]     # (N, row, col)
    rows = np.broadcast_to(py[:, :, None], val.shape)
    cols = np.broadcast_to(px[:, None, :], val.shape)
    m = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    np.add.at(img, (rows[m], cols[m]), val[m])
    return img


@dataclass
class SynthParams:
    size: int = 256
    ppp: float = 0.03            # particles per pixel (per exposure)
    dp_mean: float = 2.8         # particle image diameter [px]
    dp_std: float = 0.5
    sheet_sigma: float = 0.4     # Gaussian light-sheet profile, z in [-1, 1]
    w_max: float = 0.1           # out-of-plane motion (sheet units)
    r: float = 0.7               # pulse-2 / pulse-1 intensity ratio
    peak: float = 0.6            # pulse-1 peak intensity at sheet centre
    bg: float = 0.02
    noise: float = 0.015


def make_sample(rng, p: SynthParams, flow=None):
    """Returns dict with double-exposure image, clean A, clean B, flow."""
    S = p.size
    if flow is None:
        flow = random_flow(rng, S)
    pad = 12
    n = int(p.ppp * (S + 2 * pad) ** 2)
    x = rng.uniform(-pad, S + pad, n)
    y = rng.uniform(-pad, S + pad, n)
    z = rng.uniform(-1, 1, n)
    dp = np.clip(rng.normal(p.dp_mean, p.dp_std, n), 1.5, 5)
    dx, dy = flow.displace(x, y)
    z2 = z + rng.uniform(-p.w_max, p.w_max)
    prof = lambda zz: np.exp(-zz**2 / (2 * p.sheet_sigma**2))
    A = render((S, S), x, y, p.peak * prof(z), dp)
    B = render((S, S), x + dx, y + dy, p.peak * p.r * prof(z2), dp)
    img = A + B + p.bg + rng.normal(0, p.noise, (S, S))
    img = np.clip(img, 0, 1)
    return dict(img=img.astype(np.float32), A=A.astype(np.float32),
                B=B.astype(np.float32), flow=flow)


def gt_window_displacement(flow, centers_x, centers_y, win, n=6):
    """Mean Lagrangian displacement of points uniformly filling each window."""
    o = (np.arange(n) + .5) / n * win - win / 2
    ox, oy = np.meshgrid(o, o)
    X = centers_x[..., None] + ox.ravel()
    Y = centers_y[..., None] + oy.ravel()
    dx, dy = flow.displace(X, Y)
    return dx.mean(-1), dy.mean(-1)
