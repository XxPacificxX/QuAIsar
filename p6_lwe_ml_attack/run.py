"""Project 6 - Machine-learning secret recovery on toy LWE (SALSA-style).

LWE samples: b = <a, s> + e  (mod q), with a sparse binary secret s.
These parameters are deliberately tiny and insecure; real ML-KEM uses
n=256 per module, q=3329, and is far out of reach of this method.

Pipeline (after Wenger et al., "SALSA", NeurIPS 2022, simplified):
  1. Train a neural net to predict b from a (angles on the unit circle so the
     model sees modular arithmetic as rotation).
  2. Distinguisher: for each coordinate i, add a random K to a_i. If s_i = 0
     the model's prediction should not move; if s_i = 1 it should.
  3. Take the most sensitive coordinates as the secret, verify it
     statistically on fresh samples.
Baseline: exhaustive search over all weight-h secrets (grows as C(n, h)).

    python p6_lwe_ml_attack/run.py --n 16 24 --h 3
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import save_results, set_seed, table  # noqa: E402


def lwe_samples(m, s, q, sigma, rng):
    A = rng.integers(0, q, size=(m, len(s)))
    e = np.rint(rng.normal(0, sigma, size=m)).astype(np.int64)
    b = (A @ s + e) % q
    return A, b


def verify(A, b, s, q, sigma):
    """True if residuals b - <a,s> look like small noise (not uniform)."""
    r = (b - A @ s) % q
    r = np.where(r > q // 2, r - q, r)
    return np.mean(np.abs(r) <= 3 * sigma + 1) > 0.95


def to_angle_feats(A, q):
    th = 2 * np.pi * A / q
    return np.concatenate([np.cos(th), np.sin(th)], axis=-1).astype(np.float32)


def train_model(A, b, q, epochs, seed):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    n = A.shape[1]
    X = torch.tensor(to_angle_feats(A, q))
    thb = 2 * np.pi * b / q
    Y = torch.tensor(np.stack([np.cos(thb), np.sin(thb)], 1), dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(2 * n, 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, 2),
    )
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 512):
            idx = perm[i:i + 512]
            loss = ((model(X[idx]) - Y[idx]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  loss {tot / len(X):.4f}   (2.0 = predicting nothing)")
    model.eval()

    def predict(Aq):
        with torch.no_grad():
            return model(torch.tensor(to_angle_feats(Aq, q))).numpy()

    return predict


def distinguisher(predict, n, q, rng, trials=2000):
    """Sensitivity of the model's output to each coordinate."""
    A = rng.integers(0, q, size=(trials, n))
    base = predict(A)
    scores = np.zeros(n)
    for i in range(n):
        A2 = A.copy()
        A2[:, i] = (A2[:, i] + rng.integers(q // 4, 3 * q // 4, size=trials)) % q
        scores[i] = np.mean(np.linalg.norm(predict(A2) - base, axis=1))
    return scores


def brute_force(A, b, n, h, q, sigma):
    for idx in itertools.combinations(range(n), h):
        s = np.zeros(n, dtype=np.int64)
        s[list(idx)] = 1
        if verify(A, b, s, q, sigma):
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 24])
    ap.add_argument("--h", type=int, default=3, help="secret Hamming weight")
    ap.add_argument("--q", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    rows = []
    for n in a.n:
        s = np.zeros(n, dtype=np.int64)
        s[rng.choice(n, a.h, replace=False)] = 1
        A, b = lwe_samples(a.samples, s, a.q, a.sigma, rng)
        Av, bv = lwe_samples(2000, s, a.q, a.sigma, rng)
        print(f"\n== n={n} q={a.q} h={a.h} sigma={a.sigma}  secret support {np.flatnonzero(s).tolist()} ==")

        t0 = time.perf_counter()
        predict = train_model(A, b, a.q, a.epochs, a.seed)
        scores = distinguisher(predict, n, a.q, rng)
        guess = np.zeros(n, dtype=np.int64)
        guess[np.argsort(scores)[-a.h:]] = 1
        ml_time = time.perf_counter() - t0
        ml_ok = bool(verify(Av, bv, guess, a.q, a.sigma))
        order = np.argsort(scores)[::-1]
        print(f"    sensitivity ranking: {order[: a.h + 3].tolist()} ...  ML recovered: {ml_ok}")

        t0 = time.perf_counter()
        bf = brute_force(Av, bv, n, a.h, a.q, a.sigma)
        bf_time = time.perf_counter() - t0
        rows.append({
            "n": n, "search_space": math.comb(n, a.h),
            "ml_recovered": ml_ok, "ml_seconds": round(ml_time, 1),
            "bruteforce_recovered": bf is not None and bool((bf == s).all()),
            "bruteforce_seconds": round(bf_time, 2),
        })

    print("\n" + table(rows, list(rows[0])))
    print("Note: brute force is cheap at these toy sizes but scales as C(n,h); the ML attack's cost")
    print("is mostly training and grows far more slowly with n for fixed h. Try --n 32 48 --h 4.")
    print(f"\nSaved {save_results('p6_lwe_ml_attack', {'args': vars(a), 'results': rows})}")


if __name__ == "__main__":
    main()
