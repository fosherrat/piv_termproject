"""Separating a pulse-coded double exposure into frame A (pulse 1) and B (pulse 2)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


# --------------------------------------------------------------- classical baseline
def threshold_split(img, bg_quantile=50, k_noise=4.0):
    """Blob-wise intensity classification.
    1) segment particle blobs above (background + k*noise)
    2) each blob's peak intensity -> Otsu threshold over all blob peaks
    3) bright blobs -> A, dim blobs -> B (whole blob assigned)."""
    bg = np.percentile(img, bg_quantile)
    noise = 1.4826 * np.median(np.abs(img - bg))
    I = np.clip(img - bg, 0, None)
    lab, n = ndimage.label(I > k_noise * noise)
    if n < 2:
        return I, np.zeros_like(I)
    peaks = ndimage.maximum(I, lab, index=np.arange(1, n + 1))
    t = _otsu(peaks)
    bright = np.concatenate([[False], peaks > t])
    mA = bright[lab]
    return I * mA, I * (~mA & (lab > 0))


def _otsu(v, nbins=128):
    h, e = np.histogram(v, nbins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


# --------------------------------------------------------------- U-Net
def _block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class SepUNet(nn.Module):
    """Input: double exposure (1ch).  Output: soft mask m in [0,1];
    A_hat = m * I, B_hat = (1-m) * I  (intensity is conserved: A_hat + B_hat = I)."""

    def __init__(self, base=24, depth=3):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(1, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        h = x
        for i, e in enumerate(self.enc):
            h = e(h if i == 0 else F.max_pool2d(h, 2))
            skips.append(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            h = dec(torch.cat([up(h), s], 1))
        return torch.sigmoid(self.head(h))


def normalize(img):
    bg = np.median(img)
    I = np.clip(img - bg, 0, None)
    return I, bg


@torch.no_grad()
def unet_split(model, img):
    """Apply the trained model to a full image (fully convolutional)."""
    model.eval()
    I, _ = normalize(img)
    H, W = I.shape
    ph, pw = (-H) % 8, (-W) % 8
    x = torch.from_numpy(np.pad(I, ((0, ph), (0, pw)), mode="reflect")).float()[None, None]
    m = model(x)[0, 0, :H, :W].numpy()
    return m * I, (1 - m) * I
