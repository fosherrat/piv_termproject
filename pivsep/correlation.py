"""Window-based correlation PIV: cross-correlation, auto-correlation and an
asymmetric (nonlinear) auto-correlation for pulse-coded double exposures."""
import numpy as np


def window_grid(size, win=32, step=16):
    starts = np.arange(0, size - win + 1, step)
    sy, sx = np.meshgrid(starts, starts, indexing="ij")
    cx = sx + (win - 1) / 2
    cy = sy + (win - 1) / 2
    return sy, sx, cx, cy


def _windows(img, sy, sx, win):
    return np.stack([img[a:a + win, b:b + win] for a, b in zip(sy.ravel(), sx.ravel())])


def _corr(a, b):
    """c(s) = sum_x a(x) b(x+s), circular, zero shift at centre.  a,b: (N,w,w)"""
    Fa = np.fft.rfft2(a)
    Fb = np.fft.rfft2(b)
    c = np.fft.irfft2(np.conj(Fa) * Fb, s=a.shape[-2:])
    return np.fft.fftshift(c, axes=(-2, -1))


def _subpix(c, i, j):
    """3-point Gaussian fit around integer peak (i,j). returns (di, dj)."""
    w = c.shape[0]
    def fit(m, z, p):
        m, z, p = [np.log(max(v, 1e-12)) for v in (m, z, p)]
        den = 2 * (m - 2 * z + p)
        return 0.0 if den == 0 else (m - p) / den
    di = fit(c[i - 1, j], c[i, j], c[i + 1, j]) if 0 < i < w - 1 else 0.0
    dj = fit(c[i, j - 1], c[i, j], c[i, j + 1]) if 0 < j < w - 1 else 0.0
    return di, dj


def _peak(c, mask_r=0):
    w = c.shape[0]
    cc = c.copy()
    if mask_r > 0:
        yy, xx = np.mgrid[:w, :w] - w // 2
        cc[xx**2 + yy**2 <= mask_r**2] = -np.inf
    # avoid edges for subpixel
    cc[[0, -1], :] = -np.inf
    cc[:, [0, -1]] = -np.inf
    i, j = np.unravel_index(np.argmax(cc), cc.shape)
    cpos = c - c.min() + 1e-9                     # positive for log fit
    di, dj = _subpix(cpos, i, j)
    return (j + dj - w // 2), (i + di - w // 2), cc[i, j]


def cross_piv(A, B, win=32, step=16, mask_r=0):
    """Standard single-pass cross-correlation between frames A and B.
    mask_r > 0 excludes the zero-displacement region (used for frames separated from a
    double exposure: residual leakage of the same particle into both frames makes a
    spurious peak at s = 0, and double exposures require |d| > ~1.5 d_p anyway)."""
    sy, sx, cx, cy = window_grid(A.shape[0], win, step)
    wa = _windows(A, sy, sx, win); wb = _windows(B, sy, sx, win)
    wa = wa - wa.mean((1, 2), keepdims=True)
    wb = wb - wb.mean((1, 2), keepdims=True)
    C = _corr(wa, wb)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)


def auto_piv(I, win=32, step=16, mask_r=3):
    """Auto-correlation of a double exposure.  Returns one of the two symmetric
    peaks (+d or -d): direction is inherently ambiguous."""
    sy, sx, cx, cy = window_grid(I.shape[0], win, step)
    w = _windows(I, sy, sx, win)
    w = w - w.mean((1, 2), keepdims=True)
    C = _corr(w, w)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)


def asym_auto_piv(I, win=32, step=16, mask_r=3, power=2):
    """Nonlinear auto-correlation c(s) = sum I^p(x) I(x+s).
    With pulse coding (pulse 1 brighter, ratio r<1) the +d peak ~ r*I0^3 and the
    -d peak ~ r^2*I0^3 (p=2), so the larger peak gives the flow direction."""
    sy, sx, cx, cy = window_grid(I.shape[0], win, step)
    w = _windows(I, sy, sx, win)
    w = np.clip(w - np.median(w, axis=(1, 2), keepdims=True), 0, None)
    C = _corr(w**power, w)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)
