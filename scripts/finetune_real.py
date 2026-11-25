"""Domain adaptation: fine-tune the separation U-Net on real recordings.

No manual labels are needed: real double-frame recordings give A and B, so a
pulse-coded double exposure I = A + r*B has known targets (A, r*B).
Training mixes real crops (training pairs only) with synthetic samples.

Split (no test pair is ever seen in training):
  train : test2 pairs 0, 1, 2
  test  : test2 pairs 3, 4, 5  (same set-up, other instants)
          test1                (different experiment, never seen)

python scripts/finetune_real.py --init results/sep_unet_wide_bce.pt --steps 1500
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from pivsep.separation import SepUNet, normalize
from train_sep import build, bce_loss, loss_fn, RANGE
import evaluate_real as R

TRAIN_PAIRS = ["test2_0", "test2_1", "test2_2"]


def real_frames(names):
    P = R.pairs()
    return [R.normalize_pair(*(R.crop(R.preprocess(R.load(f))) for f in P[n])) for n in names]


def real_batch(frames, rng, n, size):
    X = np.zeros((n, 1, size, size), np.float32); Y = np.zeros((n, 2, size, size), np.float32)
    for i in range(n):
        A, B = frames[rng.integers(len(frames))]
        y0 = rng.integers(0, A.shape[0] - size); x0 = rng.integers(0, A.shape[1] - size)
        a = A[y0:y0 + size, x0:x0 + size]; b = B[y0:y0 + size, x0:x0 + size]
        g = rng.uniform(0.6, 1.4)                       # intensity augmentation
        r = rng.uniform(0.45, 0.95)
        a, b = a * g, b * g * r
        I = np.clip(a + b + 0.02, 0, 1)
        X[i, 0], _ = normalize(I)
        Y[i, 0], Y[i, 1] = a, b
    return torch.from_numpy(X), torch.from_numpy(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="results/sep_unet_wide_bce.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--n-synth", type=int, default=3000)
    ap.add_argument("--out", default="results/sep_unet_ft_real.pt")
    a = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)

    ck = torch.load(a.init, map_location="cpu")
    model = SepUNet(base=ck["base"]); model.load_state_dict(ck["state"])
    RANGE.update(ppp_max=0.08, noise_max=0.05)
    Xs, Ys = build(a.n_synth, a.size, 11)
    frames = real_frames(TRAIN_PAIRS)
    Xv, Yv = real_batch(real_frames(["test2_3"])[:1], np.random.default_rng(99), 64, a.size)  # printed for monitoring only; no model selection
    print(f"synthetic {len(Xs)}, real train pairs {TRAIN_PAIRS}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    n_real = int(round(a.bs * a.real_frac))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        xr, yr = real_batch(frames, rng, n_real, a.size)
        idx = torch.from_numpy(rng.integers(0, len(Xs), a.bs - n_real))
        x = torch.cat([xr, Xs[idx]]); y = torch.cat([yr, Ys[idx]])
        k = int(rng.integers(4))
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
        if rng.random() < .5:
            x, y = x.flip(3), y.flip(3)
        loss = bce_loss(model(x), x, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), Xv, Yv).item()
            print(f"step {step:5d}  train {loss.item():.3e}  held-out real recon {vl:.3e}  {time.time() - t0:.0f}s",
                  flush=True)
    torch.save({"state": model.state_dict(), "base": ck["base"], "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
