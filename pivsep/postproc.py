"""Post-processing of PIV vector fields using mass conservation (div u = 0).

Methods (all operate on the regular window grid; spacing h in px)
  uod_median   : universal outlier detection (normalized median test, Westerweel & Scarano 2005)
                 + median replacement                                    (standard PIV practice)
  stream_ls    : least-squares stream-function fit   u = dpsi/dy, v = -dpsi/dx
                 -> exactly divergence-free (discrete); Huber-IRLS makes it robust to outliers
  pinn         : MLP stream function psi(x, y) with Fourier features, fitted with a robust loss
                 + smoothness penalty |grad(vorticity)|^2 at collocation points.
                 div u = 0 is a hard constraint (u from autograd of psi); gives a continuous field.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import torch
import torch.nn as nn


# ------------------------------------------------------------------ diagnostics
def divergence(u, v, h):
    return np.gradient(u, h, axis=1) + np.gradient(v, h, axis=0)


def vorticity(u, v, h):
    return np.gradient(v, h, axis=1) - np.gradient(u, h, axis=0)


# ------------------------------------------------------------------ UOD + median
def uod_median(u, v, thr=2.0, eps=0.1, n_iter=2):
    u, v = u.copy(), v.copy()
    H, W = u.shape
    bad_all = np.zeros_like(u, bool)
    for _ in range(n_iter):
        bad = np.zeros_like(u, bool)
        for f in (u, v):
            P = np.pad(f, 1, mode="constant", constant_values=np.nan)
            nb = np.stack([P[i:i + H, j:j + W] for i in range(3) for j in range(3) if (i, j) != (1, 1)])
            med = np.nanmedian(nb, 0)
            res = np.nanmedian(np.abs(nb - med), 0)
            bad |= np.abs(f - med) / (res + eps) > thr
        if not bad.any():
            break
        for f in (u, v):
            f[bad] = np.nan
            for _ in range(H + W):                  # fill from valid neighbours, growing inward
                miss = np.isnan(f)
                if not miss.any():
                    break
                P = np.pad(f, 1, mode="constant", constant_values=np.nan)
                nb = np.stack([P[i:i + H, j:j + W] for i in range(3) for j in range(3) if (i, j) != (1, 1)])
                with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                    __import__("warnings").simplefilter("ignore")
                    fill = np.nanmedian(nb, 0)
                f[miss] = fill[miss]
        bad_all |= bad
    return u, v, bad_all


# ------------------------------------------------------------------ stream-function LS
def _diff_ops(H, W, h):
    """Central-difference (one-sided at edges) operators on a H x W node grid."""
    def d1(n):
        D = sp.lil_matrix((n, n))
        for i in range(n):
            if i == 0:
                D[i, 0], D[i, 1] = -1, 1
            elif i == n - 1:
                D[i, n - 2], D[i, n - 1] = -1, 1
            else:
                D[i, i - 1], D[i, i + 1] = -.5, .5
        return D.tocsr() / h
    Dx = sp.kron(sp.eye(H), d1(W))      # d/dx along columns
    Dy = sp.kron(d1(H), sp.eye(W))      # d/dy along rows
    return Dx.tocsr(), Dy.tocsr()


def stream_ls(u, v, h, robust=True, smooth=0.0, n_iter=8, delta=0.5):
    """min_psi  sum w_i |grad_perp psi - u|^2 (+ smooth |lap psi|^2), Huber IRLS weights."""
    H, W = u.shape
    Dx, Dy = _diff_ops(H, W, h)
    A = sp.vstack([Dy, -Dx]).tocsr()
    b = np.concatenate([u.ravel(), v.ravel()])
    w = np.ones(H * W)
    L = (Dx @ Dx + Dy @ Dy) if smooth > 0 else None
    for _ in range(n_iter if robust else 1):
        Wd = sp.diags(np.concatenate([w, w]))
        M = Wd @ A
        rhs = Wd @ b
        if L is not None:
            M = sp.vstack([M, np.sqrt(smooth) * h**2 * L])
            rhs = np.concatenate([rhs, np.zeros(H * W)])
        psi = lsqr(M, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
        r = (A @ psi - b).reshape(2, -1)
        rn = np.hypot(r[0], r[1])
        w = np.sqrt(np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-12)))   # sqrt of Huber weight
    uu, vv = (A @ psi).reshape(2, H, W)
    return uu, vv


# ------------------------------------------------------------------ PINN (hard div-free)
class StreamMLP(nn.Module):
    def __init__(self, n_ff=32, sigma=1.5, width=64, depth=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, n_ff, generator=g) * sigma)
        layers, d = [], 2 * n_ff + 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):                       # xy normalized to [-1, 1]
        z = 2 * np.pi * xy @ self.B
        return self.net(torch.cat([xy, torch.sin(z), torch.cos(z)], -1))


def pinn_fit(cx, cy, u, v, steps=1500, lr=3e-3, delta=0.5, seed=0, eval_xy=None,
             sigma=1.0, lam=1e-5, n_col=1024):
    """Fit psi(x,y) with robust (Huber) loss on vectors. u = dpsi/dy, v = -dpsi/dx.
    Returns fitted u, v on the input grid (and on eval_xy if given)."""
    torch.manual_seed(seed)
    c0 = np.array([cx.mean(), cy.mean()])
    sc = 0.5 * max(np.ptp(cx), np.ptp(cy))
    U = max(np.percentile(np.hypot(u, v), 95), 1e-6)     # robust velocity scale (outliers)
    xy = torch.tensor(np.stack([(cx.ravel() - c0[0]) / sc, (cy.ravel() - c0[1]) / sc], 1), dtype=torch.float32)
    tgt = torch.tensor(np.stack([u.ravel(), v.ravel()], 1) / U, dtype=torch.float32)
    model = StreamMLP(sigma=sigma, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    huber = nn.HuberLoss(delta=delta / U)

    def vel(p):
        p = p.clone().requires_grad_(True)
        psi = model(p)
        g = torch.autograd.grad(psi.sum(), p, create_graph=True)[0]
        return torch.stack([g[:, 1], -g[:, 0]], 1)     # (dpsi/dy, -dpsi/dx) in normalized units

    gcol = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        loss = huber(vel(xy), tgt)
        if lam > 0:
            # smoothness regularizer: penalize gradient of vorticity at random collocation points
            p = (torch.rand(n_col, 2, generator=gcol) * 2 - 1).requires_grad_(True)
            uv = vel(p)
            du = torch.autograd.grad(uv[:, 0].sum(), p, create_graph=True)[0]
            dv = torch.autograd.grad(uv[:, 1].sum(), p, create_graph=True)[0]
            w = dv[:, 0] - du[:, 1]
            dw = torch.autograd.grad(w.sum(), p, create_graph=True)[0]
            loss = loss + lam * (dw ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    def predict(px, py):
        p = torch.tensor(np.stack([(px.ravel() - c0[0]) / sc, (py.ravel() - c0[1]) / sc], 1), dtype=torch.float32)
        out = vel(p).detach().numpy() * U
        return out[:, 0].reshape(px.shape), out[:, 1].reshape(px.shape)

    res = predict(cx, cy)
    if eval_xy is not None:
        res = res + predict(*eval_xy)
    return res


def uod_flags(u, v, valid=None, k=2, thr=2.0, eps=0.1):
    """Universal outlier detection on a (2k+1)^2 neighbourhood, ignoring vectors already
    invalid (valid == False).  Returns a boolean array: True = outlier."""
    H, W = u.shape
    if valid is None:
        valid = np.ones_like(u, bool)
    bad = np.zeros_like(u, bool)
    offs = [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if (i, j) != (0, 0)]
    import warnings
    for f in (u, v):
        P = np.pad(np.where(valid, f, np.nan), k, mode="constant", constant_values=np.nan)
        nb = np.stack([P[k + i:k + i + H, k + j:k + j + W] for i, j in offs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(nb, 0)
            res = np.nanmedian(np.abs(nb - med), 0)
        r = np.abs(f - med) / (res + eps)
        bad |= np.nan_to_num(r, nan=np.inf) > thr
    return bad & valid


def fill_nan(u, v, valid, n_iter=200):
    """Replace invalid vectors by iterative neighbour averaging (for full-coverage output)."""
    from scipy import ndimage as nd
    out = []
    for f in (u, v):
        g = np.where(valid, f, 0.0); w = valid.astype(float)
        for _ in range(n_iter):
            num = nd.uniform_filter(g * w, 3); den = nd.uniform_filter(w, 3)
            new = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
            g = np.where(valid, f, new); w = np.where(valid, 1.0, (den > 0).astype(float))
        out.append(g)
    return out[0], out[1]
