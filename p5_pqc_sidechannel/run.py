"""Project 5 - Profiled side-channel attack on an ML-KEM (Kyber)-style secret.

Simulates power traces of a device multiplying a secret polynomial
coefficient s (centered binomial, eta=2, so s in {-2..2}) by a known
ciphertext coefficient c mod q=3329. Leakage = Hamming weight (HW) of
intermediate values + Gaussian noise + random time jitter (misalignment).

Two ways to attack, each with two models (Gaussian templates, 1-D CNN):

  direct        classify s straight from the trace. Ignores c, so it cannot
                separate s=1 from s=2: both load as a value with HW 1.
  intermediate  predict the leaking intermediates, HW(s mod q) and
                HW(s*c mod q), then score every hypothesis s with the known
                c. This is how real profiled attacks work: c differs per
                trace, so HW(c) vs HW(2c mod q) separates 1 from 2.

Output: single-trace accuracy and how many attack traces are needed to
recover a coefficient, across noise/jitter levels.
Swap `simulate()` for real ChipWhisperer captures of liboqs to make it real.

    python p5_pqc_sidechannel/run.py --noise 0.5 1.5 --jitter 0 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import results_dir, save_results, set_seed, table  # noqa: E402

Q = 3329
SECRETS = np.array([-2, -1, 0, 1, 2])
CBD_P = np.array([1, 4, 6, 4, 1]) / 16  # centered binomial eta=2
LOG_PRIOR = np.log(CBD_P)
S_MOD = SECRETS % Q  # -1 -> 3328: negative coefficients have very different HW
HW_TABLE = np.array([bin(i).count("1") for i in range(Q)])
HWS_VALUES = np.unique(HW_TABLE[S_MOD])  # HW(s mod q) takes values {0, 1, 3, 10}
HWS_IDX = np.searchsorted(HWS_VALUES, HW_TABLE[S_MOD])  # class of each hypothesis
N_HWP = 13  # HW of a 12-bit product: 0..12
TRACE_LEN = 120
T0 = 40  # nominal leakage position


def simulate(n, rng, noise, jitter, secret=None):
    """Return traces (n, TRACE_LEN), secret class labels (n,), ciphertext coeffs c (n,)."""
    if secret is None:
        s = rng.choice(SECRETS, size=n, p=CBD_P)
    else:
        s = np.full(n, secret)
    c = rng.integers(0, Q, size=n)
    s_mod = s % Q
    prod = (s_mod * c) % Q
    traces = rng.normal(0, noise, size=(n, TRACE_LEN))
    traces += 0.3 * np.sin(np.arange(TRACE_LEN) / 6.0)  # deterministic background activity
    shift = rng.integers(-jitter, jitter + 1, size=n) if jitter else np.zeros(n, int)
    idx = np.arange(n)
    # load of s, then the product, then the reduced product written back
    for off, val in ((0, HW_TABLE[s_mod]), (6, HW_TABLE[prod]), (12, HW_TABLE[prod] * 0.6)):
        for k in range(3):  # each leak smeared over 3 samples
            traces[idx, T0 + off + k + shift] += val * (1.0 - 0.3 * k) * 0.5
    label = np.searchsorted(SECRETS, s)
    return traces.astype(np.float32), label, c


def intermediates(label, c):
    """Leakage classes the intermediate attack learns: HW(s mod q) class, HW(s*c mod q)."""
    return HWS_IDX[label], HW_TABLE[(S_MOD[label] * c) % Q]


def hypothesis_scores(ll_s, ll_p, c):
    """Per-trace log-likelihood of each secret hypothesis, using the known c.
    ll_s: (n, 4) over HW(s) classes, ll_p: (n, 13) over HW(product)."""
    hwp = HW_TABLE[(S_MOD[None, :] * c[:, None]) % Q]  # (n, 5): predicted HW per hypothesis
    return ll_s[:, HWS_IDX] + np.take_along_axis(ll_p, hwp, axis=1)


# ---------------------------------------------------------------- template attack
class Template:
    """Gaussian templates with a pooled covariance.

    With `values`, class means follow a linear leakage model mean = a + b*value
    (Schindler's stochastic model). That covers HW classes that are rare or
    absent in the profiling set (HW 11 of a product occurs ~0.06% of the time).
    """

    def __init__(self, n_classes, values=None, n_poi=8):
        self.K, self.values, self.n_poi = n_classes, values, n_poi

    def fit(self, X, y):
        if self.values is None:
            means = np.array([X[y == k].mean(0) for k in range(self.K)])
            score = means.var(0)  # sum-of-squared-differences POI selection
        else:
            v = self.values[y].astype(float)
            Xc, vc = X - X.mean(0), v - v.mean()
            score = np.abs(Xc.T @ vc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(vc) + 1e-12)
        self.poi = np.argsort(score)[-self.n_poi:]
        Xp = X[:, self.poi]
        if self.values is None:
            self.mu = np.array([Xp[y == k].mean(0) for k in range(self.K)])
        else:
            coef, *_ = np.linalg.lstsq(np.c_[np.ones_like(v), v], Xp, rcond=None)
            self.mu = np.c_[np.ones(self.K), self.values] @ coef
        resid = Xp - self.mu[y]
        cov = resid.T @ resid / (len(y) - self.K) + 1e-6 * np.eye(self.n_poi)
        self.icov = np.linalg.inv(cov)
        return self

    def log_lik(self, X):
        d = X[:, self.poi][:, None, :] - self.mu[None]
        ll = -0.5 * np.einsum("nki,ij,nkj->nk", d, self.icov, d)
        return ll - np.logaddexp.reduce(ll, axis=1, keepdims=True)


def template_attacks(X, y, c):
    direct = Template(len(SECRETS)).fit(X, y)
    hws, hwp = intermediates(y, c)
    t_s = Template(len(HWS_VALUES), values=HWS_VALUES).fit(X, hws)
    t_p = Template(N_HWP, values=np.arange(N_HWP)).fit(X, hwp)
    return {
        "direct": lambda Xa, ca: direct.log_lik(Xa),
        "intermediate": lambda Xa, ca: hypothesis_scores(t_s.log_lik(Xa), t_p.log_lik(Xa), ca),
    }


# ---------------------------------------------------------------- CNN attack
def cnn_attacks(X, y, c, epochs, seed):
    """One CNN trunk with three heads: s directly, HW(s mod q), HW(s*c mod q)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(2),
                nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(2),
                nn.Flatten(), nn.Linear(32 * (TRACE_LEN // 4), 128), nn.ReLU(), nn.Dropout(0.2),
            )
            self.direct = nn.Linear(128, len(SECRETS))
            self.hws = nn.Linear(128, len(HWS_VALUES))
            self.hwp = nn.Linear(128, N_HWP)

        def forward(self, x):
            h = self.trunk(x)
            return self.direct(h), self.hws(h), self.hwp(h)

    hws, hwp = intermediates(y, c)
    mean, std = X.mean(), X.std()
    Xt = torch.tensor((X - mean) / std).unsqueeze(1)
    Y = [torch.tensor(y), torch.tensor(hws), torch.tensor(hwp)]
    model = Net()
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    w = torch.tensor(1.0 / CBD_P, dtype=torch.float32)  # balance rare classes (+-2)
    ce_direct, ce = nn.CrossEntropyLoss(weight=w / w.sum()), nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        tot = 0.0
        for i in range(0, len(y), 256):
            b = perm[i:i + 256]
            # data augmentation: random shifts teach shift invariance
            xb = torch.roll(Xt[b], shifts=int(np.random.randint(-3, 4)), dims=2)
            o_d, o_s, o_p = model(xb)
            loss = ce_direct(o_d, Y[0][b]) + ce(o_s, Y[1][b]) + ce(o_p, Y[2][b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    CNN epoch {ep:3d}  loss {tot / len(y):.4f}")
    model.eval()

    # The softmax outputs are posteriors; divide out the class frequencies seen
    # in profiling to turn them into likelihoods (Laplace-smoothed).
    log_freq_s = np.log((np.bincount(hws, minlength=len(HWS_VALUES)) + 1) / (len(y) + len(HWS_VALUES)))
    log_freq_p = np.log((np.bincount(hwp, minlength=N_HWP) + 1) / (len(y) + N_HWP))

    def heads(Xa):
        with torch.no_grad():
            outs = model(torch.tensor((Xa - mean) / std).unsqueeze(1))
            return [torch.log_softmax(o, 1).numpy() for o in outs]

    return {
        # direct head was trained class-balanced, so it is already ~a likelihood
        "direct": lambda Xa, ca: heads(Xa)[0],
        "intermediate": lambda Xa, ca: hypothesis_scores(
            *(lambda h: (h[1] - log_freq_s, h[2] - log_freq_p))(heads(Xa)), ca),
    }


# ---------------------------------------------------------------- evaluation
def single_trace_accuracy(score_fn, X, y, c):
    return float(np.mean(np.argmax(score_fn(X, c) + LOG_PRIOR, 1) == y))


def traces_to_recover(score_fn, rng, noise, jitter, trials=100, max_traces=50):
    """Mean number of attack traces until the true coefficient ranks first and stays
    first (accumulating log-likelihoods across traces, the standard key-rank approach)."""
    needed = []
    for _ in range(trials):
        true = rng.choice(len(SECRETS), p=CBD_P)
        X, _, c = simulate(max_traces, rng, noise, jitter, secret=SECRETS[true])
        cum = np.cumsum(score_fn(X, c), axis=0) + LOG_PRIOR
        ok = np.argmax(cum, 1) == true
        n = max_traces + 1
        for i in range(max_traces - 1, -1, -1):
            if not ok[i]:
                break
            n = i + 1
        needed.append(n)
    needed = np.array(needed)
    return float(np.mean(needed)), float(np.mean(needed <= max_traces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiling", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.5])
    ap.add_argument("--jitter", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    rows = []
    for noise in a.noise:
        for jitter in a.jitter:
            print(f"\n== noise={noise} jitter=+-{jitter} samples ==")
            Xp, yp, cp = simulate(a.profiling, rng, noise, jitter)
            Xv, yv, cv = simulate(3000, rng, noise, jitter)
            attacks = {"Template": template_attacks(Xp, yp, cp),
                       "CNN": cnn_attacks(Xp, yp, cp, a.epochs, a.seed)}
            for model_name, fns in attacks.items():
                for mode, fn in fns.items():
                    n_mean, success = traces_to_recover(fn, rng, noise, jitter)
                    rows.append({
                        "noise": noise, "jitter": jitter, "attack": f"{model_name} {mode}",
                        "trace_acc": round(single_trace_accuracy(fn, Xv, yv, cv), 3),
                        "traces_to_recover": round(n_mean, 1),
                        "success<=50": round(success, 2),
                    })
    print("\n" + table(rows, ["noise", "jitter", "attack", "trace_acc", "traces_to_recover", "success<=50"]))
    print("trace_acc: secret guessed from ONE trace (majority-class guess = 0.375).")
    print("traces_to_recover: mean attack traces until the secret ranks first for good; 51 = failed.")
    print("'direct' ignores the ciphertext c; 'intermediate' uses c to score each hypothesis.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y, _ = simulate(2000, rng, 0.3, 0)
        fig, ax = plt.subplots(figsize=(8, 3))
        for k, s in enumerate(SECRETS):
            ax.plot(X[y == k].mean(0), label=f"s={s}")
        ax.set_title("Mean simulated trace per secret coefficient (noise 0.3, no jitter)")
        ax.set_xlabel("sample"); ax.legend(ncol=5, fontsize=8)
        fig.tight_layout(); fig.savefig(results_dir("p5_pqc_sidechannel") / "mean_traces.png", dpi=120)

        settings = sorted({(r["noise"], r["jitter"]) for r in rows})
        names = list(dict.fromkeys(r["attack"] for r in rows))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        width = 0.8 / len(names)
        for j, name in enumerate(names):
            vals = [next(r["traces_to_recover"] for r in rows if r["attack"] == name and (r["noise"], r["jitter"]) == st)
                    for st in settings]
            ax.bar(np.arange(len(settings)) + j * width, vals, width, label=name)
        ax.set_xticks(np.arange(len(settings)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([f"noise {n}\njitter ±{j}" for n, j in settings])
        ax.set_ylabel("traces to recover (51 = fail)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Direct vs intermediate-value attacks (lower is better)")
        fig.tight_layout(); fig.savefig(results_dir("p5_pqc_sidechannel") / "traces_to_recover.png", dpi=120)
    except ImportError:
        pass
    print(f"\nSaved {save_results('p5_pqc_sidechannel', {'args': vars(a), 'results': rows})}")


if __name__ == "__main__":
    main()
