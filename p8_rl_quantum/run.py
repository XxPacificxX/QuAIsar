"""Project 8 - Reinforcement-learning decoder for the rotated surface code.

Environment: a distance-d rotated surface code under independent bit-flip
(X) noise with probability p (code-capacity model). The agent sees the
Z-stabilizer syndrome and flips one data qubit per step until the syndrome
is clear. Reward +1 if the final correction leaves no logical error, -1 if it
causes one (or the step budget runs out), small penalty per step.

Agent: Deep Q-Network (PyTorch). Baselines: no correction, and minimum-weight
perfect matching (PyMatching), the standard decoder.

    python p8_rl_quantum/run.py --d 3 --train-steps 40000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import results_dir, save_results, set_seed, table  # noqa: E402

import torch
from torch import nn


# ---------------------------------------------------------------- code
def rotated_surface_code(d):
    """Return (Hz, Hx, logical_Z) as 0/1 arrays over d*d data qubits."""
    q = lambda r, c: r * d + c
    Z, X = [], []
    for r in range(d - 1):
        for c in range(d - 1):
            face = [q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1)]
            (Z if (r + c) % 2 == 0 else X).append(face)
    for c in range(d - 1):
        if c % 2 == 0:
            X.append([q(0, c), q(0, c + 1)])  # top boundary
        else:
            X.append([q(d - 1, c), q(d - 1, c + 1)])  # bottom boundary
    for r in range(d - 1):
        if r % 2 == 1:
            Z.append([q(r, 0), q(r + 1, 0)])  # left boundary
        else:
            Z.append([q(r, d - 1), q(r + 1, d - 1)])  # right boundary

    def mat(rows):
        H = np.zeros((len(rows), d * d), dtype=np.uint8)
        for i, s in enumerate(rows):
            H[i, s] = 1
        return H

    Hz, Hx = mat(Z), mat(X)
    LZ = np.zeros(d * d, dtype=np.uint8)
    LZ[[q(0, c) for c in range(d)]] = 1  # top row
    assert Hz.shape[0] == Hx.shape[0] == (d * d - 1) // 2
    assert not ((Hz @ Hx.T) % 2).any(), "stabilizers must commute"
    assert not ((Hx @ LZ) % 2).any(), "logical Z must commute with X stabilizers"
    return Hz, Hx, LZ


class SurfaceCodeEnv:
    def __init__(self, d, p, max_steps=None, step_penalty=0.02, seed=0):
        self.Hz, _, self.LZ = rotated_surface_code(d)
        self.n = d * d
        self.p = p
        self.max_steps = max_steps or 2 * self.n
        self.step_penalty = step_penalty
        self.rng = np.random.default_rng(seed)

    def syndrome(self):
        return (self.Hz @ self.err) % 2

    def obs(self):
        # The agent sees the current syndrome AND its own flips so far. Without
        # the flips the problem is non-Markov: two different correction paths
        # can reach the same syndrome with different logical outcomes.
        return np.concatenate([self.syndrome(), self.flips]).astype(np.float32)

    def reset(self):
        while True:  # skip trivial episodes (empty syndrome) during training
            self.err = (self.rng.random(self.n) < self.p).astype(np.uint8)
            if self.syndrome().any():
                break
        self.flips = np.zeros(self.n, dtype=np.uint8)
        self.t = 0
        return self.obs()

    def step(self, a):
        self.err[a] ^= 1
        self.flips[a] ^= 1
        self.t += 1
        o = self.obs()
        if not self.syndrome().any():
            logical = (self.LZ @ self.err) % 2
            return o, (-1.0 if logical else 1.0), True
        if self.t >= self.max_steps or self.flips.all():
            return o, -1.0, True
        return o, -self.step_penalty, False


# ---------------------------------------------------------------- DQN
def masked(qvals, obs, n_syn):
    """Forbid re-flipping a qubit the agent already flipped (it would only undo it)."""
    return qvals.masked_fill(obs[..., n_syn:] > 0.5, -1e9)


def train_dqn(d, p, steps, seed):
    env = SurfaceCodeEnv(d, p, seed=seed)
    n_syn, n_act = env.Hz.shape[0], env.n
    n_obs = n_syn + n_act
    net = lambda: nn.Sequential(nn.Linear(n_obs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_act))
    q, tgt = net(), net()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), 5e-4)
    buf = deque(maxlen=50_000)
    gamma, batch = 0.95, 128
    s = env.reset()
    ep_rewards, ep_r = deque(maxlen=1000), 0.0
    for t in range(steps):
        eps = max(0.05, 1 - t / (0.4 * steps))
        if random.random() < eps:
            a = int(random.choice(np.flatnonzero(env.flips == 0)))
        else:
            with torch.no_grad():
                st = torch.tensor(s)
                a = int(masked(q(st), st, n_syn).argmax())
        s2, r, done = env.step(a)
        buf.append((s, a, r, s2, done))
        ep_r += r
        s = s2
        if done:
            ep_rewards.append(ep_r); ep_r = 0.0
            s = env.reset()
        if len(buf) >= 1000:
            S, A, R, S2, D = map(np.array, zip(*random.sample(buf, batch)))
            S, S2 = torch.tensor(S), torch.tensor(S2)
            with torch.no_grad():  # double DQN target with action masking
                a2 = masked(q(S2), S2, n_syn).argmax(1, keepdim=True)
                y = torch.tensor(R, dtype=torch.float32) + gamma * (1 - torch.tensor(D, dtype=torch.float32)) * tgt(S2).gather(1, a2).squeeze(1)
            pred = q(S).gather(1, torch.tensor(A).unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        if t % 1000 == 0:
            tgt.load_state_dict(q.state_dict())
        if t % 10000 == 0 and ep_rewards:
            print(f"    step {t:6d}  eps {eps:.2f}  mean episode reward {np.mean(ep_rewards):+.3f}")
    q.eval()
    return q


def rl_decode(q, Hz, syndrome, n, max_steps):
    s = syndrome.copy()
    corr = np.zeros(n, dtype=np.uint8)
    n_syn = Hz.shape[0]
    for _ in range(max_steps):
        if not s.any() or corr.all():
            break
        with torch.no_grad():
            st = torch.tensor(np.concatenate([s, corr]), dtype=torch.float32)
            a = int(masked(q(st), st, n_syn).argmax())
        corr[a] ^= 1
        s = (s + Hz[:, a]) % 2
    return corr


def logical_error_rate(decode, Hz, LZ, p, shots, rng):
    n = Hz.shape[1]
    fails = 0
    for _ in range(shots):
        err = (rng.random(n) < p).astype(np.uint8)
        syn = (Hz @ err) % 2
        res = err ^ decode(syn)
        if ((Hz @ res) % 2).any() or (LZ @ res) % 2:
            fails += 1
    return fails / shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--train-p", type=float, default=0.08)
    ap.add_argument("--train-steps", type=int, default=40_000)
    ap.add_argument("--p", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.11, 0.15])
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed + 1)

    Hz, _, LZ = rotated_surface_code(a.d)
    n = a.d * a.d
    print(f"Rotated surface code d={a.d}: {n} data qubits, {Hz.shape[0]} Z-checks. Training DQN at p={a.train_p} ...")
    q = train_dqn(a.d, a.train_p, a.train_steps, a.seed)

    import pymatching

    matching = pymatching.Matching(Hz)
    decoders = {
        "none": lambda s: np.zeros(n, dtype=np.uint8),
        "MWPM (PyMatching)": lambda s: matching.decode(s).astype(np.uint8),
        "DQN agent": lambda s: rl_decode(q, Hz, s, n, 2 * n),
    }
    rows = []
    for name, dec in decoders.items():
        row = {"decoder": name}
        for p in a.p:
            row[f"p={p}"] = round(logical_error_rate(dec, Hz, LZ, p, a.shots, rng), 4)
        rows.append(row)
    print("\nLogical error rate (lower is better; 'none' approximates an unprotected qubit):")
    print(table(rows, ["decoder"] + [f"p={p}" for p in a.p]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            ax.plot(a.p, [r[f"p={p}"] for p in a.p], marker="o", label=r["decoder"])
        ax.plot(a.p, a.p, "k--", alpha=0.4, label="physical error rate")
        ax.set_xlabel("physical bit-flip rate p"); ax.set_ylabel("logical error rate")
        ax.set_title(f"Surface code d={a.d}: RL vs matching"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(results_dir("p8_rl_quantum") / "logical_error.png", dpi=120)
    except ImportError:
        pass
    print(f"\nSaved {save_results('p8_rl_quantum', {'args': vars(a), 'results': rows})}")


if __name__ == "__main__":
    main()
