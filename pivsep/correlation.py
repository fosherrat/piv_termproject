"""Window-based correlation PIV: cross-correlation, auto-correlation and an
asymmetric (nonlinear) auto-correlation for pulse-coded double exposures."""
import numpy as np
from scipy import ndimage


def window_grid(shape, win=32, step=16):
    """Window grid for an image of shape (H, W) (an int means a square H = W image)."""
    H, W = (shape, shape) if np.isscalar(shape) else shape[:2]
    sy, sx = np.meshgrid(np.arange(0, H - win + 1, step), np.arange(0, W - win + 1, step), indexing="ij")
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
        d = 0.0 if den == 0 else (m - p) / den
        return d if np.isfinite(d) and abs(d) <= 1.0 else 0.0   # degenerate fit -> integer peak
    di = fit(c[i - 1, j], c[i, j], c[i + 1, j]) if 0 < i < w - 1 else 0.0
    dj = fit(c[i, j - 1], c[i, j], c[i, j + 1]) if 0 < j < w - 1 else 0.0
    return di, dj


def _peak(c, mask_r=0, mask_at=(0.0, 0.0), max_r=None):
    """Highest correlation peak.  mask_r > 0 excludes a disc of radius mask_r centred at
    displacement mask_at = (sx, sy); max_r limits the search to |s| <= max_r."""
    w = c.shape[0]
    cc = c.copy()
    yy, xx = np.mgrid[:w, :w] - w // 2
    if mask_r > 0:
        cc[(xx - mask_at[0])**2 + (yy - mask_at[1])**2 <= mask_r**2] = -np.inf
    if max_r is not None:
        cc[xx**2 + yy**2 > max_r**2] = -np.inf
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
    sy, sx, cx, cy = window_grid(A.shape, win, step)
    wa = _windows(A, sy, sx, win); wb = _windows(B, sy, sx, win)
    wa = wa - wa.mean((1, 2), keepdims=True)
    wb = wb - wb.mean((1, 2), keepdims=True)
    C = _corr(wa, wb)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)


def auto_piv(I, win=32, step=16, mask_r=3):
    """Auto-correlation of a double exposure.  Returns one of the two symmetric
    peaks (+d or -d): direction is inherently ambiguous."""
    sy, sx, cx, cy = window_grid(I.shape, win, step)
    w = _windows(I, sy, sx, win)
    w = w - w.mean((1, 2), keepdims=True)
    C = _corr(w, w)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)


def asym_auto_piv(I, win=32, step=16, mask_r=3, power=2):
    """Nonlinear auto-correlation c(s) = sum I^p(x) I(x+s).
    With pulse coding (pulse 1 brighter, ratio r<1) the +d peak ~ r*I0^3 and the
    -d peak ~ r^2*I0^3 (p=2), so the larger peak gives the flow direction."""
    sy, sx, cx, cy = window_grid(I.shape, win, step)
    w = _windows(I, sy, sx, win)
    w = np.clip(w - np.median(w, axis=(1, 2), keepdims=True), 0, None)
    C = _corr(w**power, w)
    out = np.array([_peak(c, mask_r)[:2] for c in C])
    return cx, cy, out[:, 0].reshape(cx.shape), out[:, 1].reshape(cx.shape)


# ----------------------------------------------------------------- multi-pass + deformation
def _sample(field, cx, cy, x, y):
    """Bilinear interpolation of a grid field (centres cx, cy) at points x, y."""
    step = cx[0, 1] - cx[0, 0]
    return ndimage.map_coordinates(field, [(y - cy[0, 0]) / step, (x - cx[0, 0]) / step],
                                   order=1, mode="nearest")


def multipass_piv(A, B, passes=((64, 32), (32, 16), (32, 16)), mask_r=0, validate=True,
                  smooth=1.0):
    """Multi-pass cross-correlation with symmetric window (image) deformation
    (Scarano 2002 style).

    pass k: the predictor field p(x) from pass k-1 is interpolated to every pixel and the
    images are deformed symmetrically, A'(x) = A(x - p/2), B'(x) = B(x + p/2) (cubic spline),
    so the residual displacement d - p is measured with a smaller window.
    Between passes the field is validated (UOD + median replacement) and lightly smoothed;
    after the first pass the residual search is limited to win/4.

    mask_r > 0 (separated double exposures): a particle leaking into BOTH separated frames
    produces a spurious peak at residual displacement -p (it is at 0 before deformation);
    a disc of radius mask_r around -p is excluded in every pass.
    The last pass uses the same grid as cross_piv(A, B, win=passes[-1][0], step=passes[-1][1]).
    """
    from .postproc import uod_median
    H, W = A.shape
    Y, X = np.mgrid[:H, :W].astype(float)
    prev = None
    for k, (win, step) in enumerate(passes):
        sy, sx, cx, cy = window_grid((H, W), win, step)
        if prev is None:
            pu = np.zeros(cx.shape); pv = np.zeros(cx.shape); Ad, Bd = A, B
        else:
            gx, gy, gu, gv = prev
            pu = _sample(gu, gx, gy, cx, cy); pv = _sample(gv, gx, gy, cx, cy)
            du = _sample(gu, gx, gy, X, Y); dv = _sample(gv, gx, gy, X, Y)
            Ad = ndimage.map_coordinates(A, [Y - dv / 2, X - du / 2], order=3, mode="nearest")
            Bd = ndimage.map_coordinates(B, [Y + dv / 2, X + du / 2], order=3, mode="nearest")
        wa = _windows(Ad, sy, sx, win); wb = _windows(Bd, sy, sx, win)
        wa = wa - wa.mean((1, 2), keepdims=True)
        wb = wb - wb.mean((1, 2), keepdims=True)
        C = _corr(wa, wb)
        # after the first pass only a residual is measured: limit it to win/4 (one-quarter rule),
        # otherwise a bad window can run away across passes
        max_r = None if k == 0 else win / 4
        res = np.array([_peak(c, mask_r, (-a, -b), max_r)[:2]
                        for c, a, b in zip(C, pu.ravel(), pv.ravel())])
        u = pu + res[:, 0].reshape(cx.shape)
        v = pv + res[:, 1].reshape(cx.shape)
        if k < len(passes) - 1:
            if validate:
                u, v, _ = uod_median(u, v)
            if smooth:
                u = ndimage.gaussian_filter(u, smooth, mode="nearest")
                v = ndimage.gaussian_filter(v, smooth, mode="nearest")
        prev = (cx, cy, u, v)
    return cx, cy, u, v


# ----------------------------------------------------------------- vector quality / validation
def window_quality(A, B, u, v, win=32, step=16, r=1.5, mask_r=3):
    """Per-vector quality from the (undeformed) cross-correlation of two separated frames,
    evaluated at the measured displacement d = (u, v):
      dir_ratio = R(+d) / R(-d)  - how clearly the frames are ordered (~1: direction ambiguous;
                                   specific to frames separated from a double exposure)
      ppr       = R(+d) / max R elsewhere (outside +-d and the centre mask) - primary-to-secondary
                  peak ratio, the standard PIV confidence measure
    R is baseline-shifted (R - median R) so the ratios are positive."""
    sy, sx, cx, cy = window_grid(A.shape, win, step)
    wa = _windows(A, sy, sx, win); wb = _windows(B, sy, sx, win)
    wa = wa - wa.mean((1, 2), keepdims=True)
    wb = wb - wb.mean((1, 2), keepdims=True)
    C = _corr(wa, wb)
    yy, xx = np.mgrid[:win, :win] - win // 2
    dr = np.zeros(cx.size); pr = np.zeros(cx.size)
    for k, (c, du, dv) in enumerate(zip(C, u.ravel(), v.ravel())):
        c = c - np.median(c)
        near_p = (xx - du) ** 2 + (yy - dv) ** 2 <= r ** 2
        near_m = (xx + du) ** 2 + (yy + dv) ** 2 <= r ** 2
        if not near_p.any() or not near_m.any():
            continue
        cp = max(c[near_p].max(), 1e-9); cm = max(c[near_m].max(), 1e-9)
        rest = ~((xx - du) ** 2 + (yy - dv) ** 2 <= 9) & ~((xx + du) ** 2 + (yy + dv) ** 2 <= 9) \
            & ~(xx ** 2 + yy ** 2 <= mask_r ** 2)
        co = max(c[rest].max(), 1e-9)
        dr[k] = cp / cm; pr[k] = cp / co
    return dr.reshape(cx.shape), pr.reshape(cx.shape)


def content_mask(I, win=32, step=16, sat=0.98, sat_frac=0.02, rel_signal=0.5):
    """Windows to exclude before any correlation (standard PIV masking, decided from the
    recorded image only): saturated objects (glare from the model, walls) and windows with
    too little particle signal (shadows, outside the light sheet).
    Signal = std of the high-passed image in the window, relative to the image median."""
    from scipy import ndimage as nd
    hp = I - nd.uniform_filter(I, 7)
    sy, sx, cx, cy = window_grid(I.shape, win, step)
    w = _windows(I, sy, sx, win); wh = _windows(hp, sy, sx, win)
    satf = (w >= sat).mean((1, 2))
    sig = wh.std((1, 2))
    keep = (satf <= sat_frac) & (sig >= rel_signal * np.median(sig))
    return keep.reshape(cx.shape)
