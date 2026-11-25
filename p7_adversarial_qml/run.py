"""Project 7 - Adversarial robustness of quantum machine-learning classifiers.

Trains a variational quantum classifier (PennyLane, torch interface) and a
classical MLP with the same number of input features, then attacks both with
FGSM and PGD (L-infinity) and measures accuracy vs. perturbation size.
Finally retrains the VQC with adversarial training (PGD examples mixed in)
as a defense.

Data: sklearn handwritten digits, 3 vs 8 (a harder pair than 0 vs 1),
reduced by PCA to one feature per qubit. Swap in the p4 intrusion data to
study evasion attacks on a quantum IDS.

    python p7_adversarial_qml/run.py --qubits 6 --eps 0 0.05 0.1 0.2 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import results_dir, save_results, set_seed, table  # noqa: E402

import torch
from torch import nn


def load(n_feat, seed):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    d = load_digits()
    m = np.isin(d.target, [3, 8])
    X, y = d.data[m], (d.target[m] == 8).astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    pca = PCA(n_feat, random_state=seed).fit(Xtr)
    sc = MinMaxScaler((0, 1)).fit(pca.transform(Xtr))
    f = lambda A: torch.tensor(np.clip(sc.transform(pca.transform(A)), 0, 1), dtype=torch.float32)
    return f(Xtr), f(Xte), torch.tensor(ytr), torch.tensor(yte)


class VQC(nn.Module):
    """Inputs in [0,1] are encoded as RY(pi*x); data re-uploaded every layer."""

    def __init__(self, n_qubits, layers=3):
        super().__init__()
        import pennylane as qml

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            for l in range(layers):
                qml.AngleEmbedding(np.pi * x, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_qubits)
        self.w = nn.Parameter(0.1 * torch.randn(shape))
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # returns a logit for class 1
        out = self.circuit(x, self.w)  # x is (batch, features); PennyLane broadcasts
        return self.scale * out.to(torch.float32) + self.bias


def mlp(n):
    return nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))


def logits(model, x):
    return model(x).reshape(-1)


def pgd(model, x, y, eps, steps=10, alpha=None):
    """L-inf PGD (steps=1, alpha=eps gives FGSM). Inputs stay in [0,1]."""
    if eps == 0:
        return x
    alpha = alpha or 2.5 * eps / steps
    lossf = nn.BCEWithLogitsLoss()
    xa = x.clone().detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = lossf(logits(model, xa), y.float())
        (g,) = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.clamp(torch.min(torch.max(xa, x - eps), x + eps), 0, 1)
    return xa.detach()


def train(model, X, y, epochs, lr, adv_eps=0.0, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        tot = 0.0
        for i in range(0, len(y), 32):
            b = perm[i:i + 32]
            xb, yb = X[b], y[b]
            if adv_eps:  # adversarial training: half clean, half PGD
                xb = torch.cat([xb, pgd(model, xb, yb, adv_eps, steps=5)])
                yb = torch.cat([yb, yb])
            loss = lossf(logits(model, xb), yb.float())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    {tag} epoch {ep:3d}  loss {tot / len(y):.4f}")
    return model


def acc(model, X, y):
    with torch.no_grad():
        return float(((logits(model, X) > 0).long() == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eps", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--adv-eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    Xtr, Xte, ytr, yte = load(a.qubits, a.seed)
    print(f"Digits 3 vs 8, {a.qubits} PCA features, train={len(ytr)} test={len(yte)}")

    models = {
        "MLP": train(mlp(a.qubits), Xtr, ytr, a.epochs * 3, 1e-2, tag="MLP"),
        "VQC": train(VQC(a.qubits), Xtr, ytr, a.epochs, 5e-2, tag="VQC"),
    }
    print("Adversarial training of the VQC (defense) ...")
    models["VQC + adv. training"] = train(VQC(a.qubits), Xtr, ytr, a.epochs, 5e-2,
                                          adv_eps=a.adv_eps, tag="VQC-AT")

    rows = []
    for name, m in models.items():
        for attack, steps in (("FGSM", 1), ("PGD-10", 10)):
            row = {"model": name, "attack": attack}
            for eps in a.eps:
                xa = pgd(m, Xte, yte, eps, steps=steps, alpha=eps if steps == 1 else None)
                row[f"eps={eps}"] = round(acc(m, xa, yte), 3)
            rows.append(row)
    print("\nTest accuracy under attack (inputs scaled to [0,1], eps is L-inf):")
    print(table(rows, ["model", "attack"] + [f"eps={e}" for e in a.eps]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            if r["attack"] == "PGD-10":
                ax.plot(a.eps, [r[f"eps={e}"] for e in a.eps], marker="o", label=r["model"])
        ax.set_xlabel("PGD epsilon (L-inf)"); ax.set_ylabel("test accuracy")
        ax.set_title("Robustness: quantum vs classical"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(results_dir("p7_adversarial_qml") / "robustness.png", dpi=120)
    except ImportError:
        pass
    print(f"\nSaved {save_results('p7_adversarial_qml', {'args': vars(a), 'results': rows})}")


if __name__ == "__main__":
    main()
