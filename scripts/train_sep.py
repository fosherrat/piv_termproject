"""Train the A/B separation U-Net on synthetic pulse-coded double exposures.

python scripts/train_sep.py --n-train 4000 --steps 3000
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from pivsep.synth import SynthParams, make_sample
from pivsep.separation import SepUNet, normalize


RANGE = dict(ppp_max=0.05, noise_max=0.03)


def random_params(rng, size):
    return SynthParams(size=size,
                       r=rng.uniform(0.45, 0.95),
                       ppp=rng.uniform(0.015, RANGE["ppp_max"]),
                       noise=rng.uniform(0.005, RANGE["noise_max"]),
                       dp_mean=rng.uniform(2.2, 3.5),
                       w_max=rng.uniform(0.0, 0.15))


def build(n, size, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), np.float32)
    Y = np.zeros((n, 2, size, size), np.float32)
    for i in range(n):
        s = make_sample(rng, random_params(rng, size))
        X[i, 0], _ = normalize(s["img"])
        Y[i, 0], Y[i, 1] = s["A"], s["B"]
    return torch.from_numpy(X), torch.from_numpy(Y)


def loss_fn(m, x, y):
    """Reconstruction MSE of A_hat = m*I and B_hat = (1-m)*I (also the val metric)."""
    return F.mse_loss(m * x, y[:, :1]) + F.mse_loss((1 - m) * x, y[:, 1:])


def bce_loss(m, x, y):
    """Per-pixel classification: target mask A/(A+B), weighted by particle intensity
    so that background pixels do not count and dim particles still do."""
    s = y.sum(1, keepdim=True)
    t = (y[:, :1] / (s + 1e-6)).clamp(0, 1)
    w = (s / (s.amax((2, 3), keepdim=True) + 1e-6)).sqrt()
    return (F.binary_cross_entropy(m.clamp(1e-5, 1 - 1e-5), t, reduction="none") * w).sum() / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse")
    ap.add_argument("--ppp-max", type=float, default=0.05)
    ap.add_argument("--noise-max", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/sep_unet.pt")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.threads:
        torch.set_num_threads(a.threads)
    RANGE.update(ppp_max=a.ppp_max, noise_max=a.noise_max)
    print(vars(a), flush=True)

    t0 = time.time()
    Xtr, Ytr = build(a.n_train, a.size, 1)
    Xva, Yva = build(a.n_val, a.size, 2)
    print(f"data built in {time.time() - t0:.0f}s", flush=True)

    model = SepUNet(base=a.base)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        idx = torch.randint(0, a.n_train, (a.bs,), generator=g)
        x, y = Xtr[idx], Ytr[idx]
        # augmentation: random flips / 90deg rotations (flow is arbitrary anyway)
        k = int(torch.randint(0, 4, (1,), generator=g))
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
        if torch.rand(1, generator=g) < .5:
            x, y = x.flip(3), y.flip(3)
        m = model(x)
        loss = loss_fn(m, x, y) if a.loss == "mse" else bce_loss(m, x, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vl = np.mean([loss_fn(model(Xva[i:i + 50]), Xva[i:i + 50], Yva[i:i + 50]).item()
                              for i in range(0, a.n_val, 50)])
            print(f"step {step:5d}  train {loss.item():.3e}  val {vl:.3e}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "base": a.base, "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
