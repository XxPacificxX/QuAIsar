"""Project 7 study - Do angle-encoded quantum classifiers only *look* robust?

A gradient-masking evaluation (after Athalye et al., ICML 2018, "Obfuscated
gradients give a false sense of security") of a variational quantum
classifier (VQC), against two classical controls:

  MLP      plain multilayer perceptron on the same PCA features
  SinMLP   MLP on sin/cos(k*pi*x) features, k=1..L. This gives a classical
           model the same periodic input dependence (Fourier spectrum) that
           L-layer angle re-uploading gives the VQC. If SinMLP shows the same
           symptoms, the effect comes from the encoding, not from anything
           quantum.

Attacks (L-inf, inputs in [0, 1], margin loss):
  FGSM          single gradient-sign step of size eps
  FGSM-LS       FGSM direction, worst case over step sizes in (0, eps]
                (separates "the gradient is uninformative" from "one full
                step overshoots a curved / periodic loss")
  PGD-20        20 steps, 1 start
  PGD-50x5      50 steps, 5 random restarts (strongest white-box)
  SPSA          gradient-free: finite-difference gradient estimates
  Transfer      PGD-50x5 examples crafted on the MLP, evaluated on each model

Masking checklist reported per model:
  * single-step much weaker than iterative          (FGSM - PGD gap)
  * line search recovers most of that gap           (curvature / periodicity)
  * gradient-free attack beats white-box            (gradients misleading)
  * transfer attack beats white-box                 (gradients misleading)
  * unbounded attack (eps = 1) fails to reach ~0%   (something is broken)
  * fraction of test points whose loss is non-monotone along the FGSM ray

    python p7_adversarial_qml/masking_study.py --qubits 4 6 8 --seeds 5
    python p7_adversarial_qml/masking_study.py --smoke      # quick check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import results_dir, set_seed, table  # noqa: E402

PROJECT = "p7_masking_study"
LAYERS = 3  # re-uploading layers; also the SinMLP's highest frequency


# ---------------------------------------------------------------- data
def load(n_feat, seed, pair=(3, 8)):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    d = load_digits()
    m = np.isin(d.target, pair)
    X, y = d.data[m], (d.target[m] == pair[1]).astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    pca = PCA(n_feat, random_state=seed).fit(Xtr)
    sc = MinMaxScaler((0, 1)).fit(pca.transform(Xtr))
    f = lambda A: torch.tensor(np.clip(sc.transform(pca.transform(A)), 0, 1), dtype=torch.float32)
    return f(Xtr), f(Xte), torch.tensor(ytr), torch.tensor(yte)


# ---------------------------------------------------------------- models (all return one logit)
class VQC(nn.Module):
    """Angle encoding RY(pi*x), re-uploaded before each strongly entangling layer."""

    def __init__(self, n_qubits, layers=LAYERS, scale=1.0):
        super().__init__()
        import pennylane as qml

        dev = qml.device("default.qubit", wires=n_qubits)
        angle = scale * np.pi  # encoding frequency: RY(scale * pi * x)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            for l in range(layers):
                qml.AngleEmbedding(angle * x, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_qubits)
        self.w = nn.Parameter(0.1 * torch.randn(shape))
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.scale * self.circuit(x, self.w).to(torch.float32) + self.bias


class MLP(nn.Module):
    def __init__(self, n, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


class SinMLP(nn.Module):
    """Classical control with the VQC's periodic input dependence: features
    sin/cos(k*pi*x) for k = 1..LAYERS, then an MLP."""

    def __init__(self, n, hidden=32, kmax=LAYERS, scale=1.0):
        super().__init__()
        self.register_buffer("k", torch.arange(1, kmax + 1, dtype=torch.float32))
        self.scale = scale
        self.net = MLP(2 * kmax * n, hidden)

    def forward(self, x):
        z = self.scale * np.pi * x.unsqueeze(-1) * self.k  # (batch, n, kmax)
        return self.net(torch.cat([torch.sin(z), torch.cos(z)], -1).flatten(1))


def logit(model, x):
    return model(x).reshape(-1)


def train(model, X, y, epochs, lr, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            loss = lossf(logit(model, X[b]), y[b].float())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------- attacks
def margin(model, x, y):
    """Positive = correctly classified. Attacks minimise it."""
    return (2 * y.float() - 1) * logit(model, x)


def grad_sign(model, x, y):
    x = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(margin(model, x, y).sum(), x)
    return -g.sign()  # direction that lowers the margin


def project(xa, x, eps):
    return torch.clamp(torch.min(torch.max(xa, x - eps), x + eps), 0, 1)


def fgsm(model, x, y, eps):
    return project(x + eps * grad_sign(model, x, y), x, eps)


def fgsm_ls(model, x, y, eps, n_steps=20):
    """FGSM direction, but keep the worst point along the ray for t in (0, 1]."""
    d = grad_sign(model, x, y)
    best_x, best_m = x.clone(), torch.full((len(y),), float("inf"))
    with torch.no_grad():
        for t in torch.linspace(1.0 / n_steps, 1.0, n_steps):
            xa = project(x + t * eps * d, x, eps)
            m = margin(model, xa, y)
            better = m < best_m
            best_x[better], best_m[better] = xa[better], m[better]
    return best_x


def nonmonotone_fraction(model, x, y, eps, n_steps=20):
    """Share of points whose margin along the FGSM ray reaches its minimum before t = 1,
    i.e. a full FGSM step overshoots."""
    d = grad_sign(model, x, y)
    with torch.no_grad():
        ms = torch.stack([margin(model, project(x + t * eps * d, x, eps), y)
                          for t in torch.linspace(1.0 / n_steps, 1.0, n_steps)])
    return float((ms.argmin(0) < n_steps - 1).float().mean())


def pgd(model, x, y, eps, steps, restarts=1):
    alpha = 2.5 * eps / steps
    best_x = x.clone()
    with torch.no_grad():
        best_m = margin(model, x, y)
    for r in range(restarts):
        xa = x.clone() if r == 0 else project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
        for _ in range(steps):
            xa = project(xa + alpha * grad_sign(model, xa, y), x, eps)
        with torch.no_grad():
            m = margin(model, xa, y)
        better = m < best_m
        best_x[better], best_m[better] = xa[better], m[better]
    return best_x


def spsa(model, x, y, eps, iters=40, samples=32, delta=0.01):
    """Gradient-free L-inf attack: SPSA gradient estimate + signed step (Uesato et al. 2018)."""
    n, dim = x.shape
    alpha = eps / 8
    xa = x.clone()
    with torch.no_grad():
        best_x, best_m = x.clone(), margin(model, x, y)
        for _ in range(iters):
            v = torch.randint(0, 2, (samples, n, dim)).float() * 2 - 1
            pts = torch.cat([xa + delta * v, xa - delta * v]).reshape(-1, dim)
            yy = y.repeat(2 * samples)
            m = margin(model, pts.clamp(0, 1), yy).reshape(2, samples, n)
            g = ((m[0] - m[1]).unsqueeze(-1) * v).mean(0) / (2 * delta)
            xa = project(xa - alpha * g.sign(), x, eps)
            cur = margin(model, xa, y)
            better = cur < best_m
            best_x[better], best_m[better] = xa[better], cur[better]
    return best_x


def accuracy(model, x, y):
    with torch.no_grad():
        return float((margin(model, x, y) > 0).float().mean())


def correct(model, x, y):
    with torch.no_grad():
        return (margin(model, x, y) > 0)


# ---------------------------------------------------------------- one (qubits, seed) cell
def run_cell(n, seed, eps_list, cfg):
    set_seed(seed)
    Xtr, Xte, ytr, yte = load(n, seed)
    if cfg["test_limit"]:
        Xte, yte = Xte[: cfg["test_limit"]], yte[: cfg["test_limit"]]
    t0 = time.perf_counter()
    models = {
        "VQC": train(VQC(n), Xtr, ytr, cfg["vqc_epochs"], 5e-2),
        "SinMLP": train(SinMLP(n), Xtr, ytr, cfg["mlp_epochs"], 1e-2),
        "MLP": train(MLP(n), Xtr, ytr, cfg["mlp_epochs"], 1e-2),
    }
    t_train = time.perf_counter() - t0

    out = {}
    transfer_src = {eps: pgd(models["MLP"], Xte, yte, eps, 50, cfg["restarts"]) for eps in eps_list}
    for name, m in models.items():
        res = {"clean": accuracy(m, Xte, yte), "attacks": {}, "nonmonotone": {}, "worst_case": {}}
        for eps in eps_list:
            adv = {
                "FGSM": fgsm(m, Xte, yte, eps),
                "FGSM-LS": fgsm_ls(m, Xte, yte, eps),
                "PGD-20": pgd(m, Xte, yte, eps, 20),
                "PGD-50x5": pgd(m, Xte, yte, eps, 50, cfg["restarts"]),
                "SPSA": spsa(m, Xte, yte, eps, cfg["spsa_iters"], cfg["spsa_samples"]),
                "Transfer(MLP)": transfer_src[eps],
            }
            ok = torch.ones(len(yte), dtype=torch.bool) & correct(m, Xte, yte)
            for attack, xa in adv.items():
                c = correct(m, xa, yte)
                res["attacks"].setdefault(attack, {})[str(eps)] = float(c.float().mean())
                ok &= c
            res["worst_case"][str(eps)] = float(ok.float().mean())  # per-point minimum over all attacks
            res["nonmonotone"][str(eps)] = nonmonotone_fraction(m, Xte, yte, eps)
        res["unbounded_pgd"] = accuracy(m, pgd(m, Xte, yte, 1.0, 100), yte)
        out[name] = res
    out["_seconds"] = {"train": round(t_train, 1), "total": round(time.perf_counter() - t0, 1)}
    return out


# ---------------------------------------------------------------- aggregate and report
ATTACKS = ["FGSM", "FGSM-LS", "PGD-20", "PGD-50x5", "SPSA", "Transfer(MLP)"]
MODELS = ["VQC", "SinMLP", "MLP"]


def summarise(cells, qubits, eps_list):
    """cells[(n, seed)] -> per-model results. Returns rows of mean +- std."""
    def ms(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    rows, checks = [], []
    for n in qubits:
        seeds = [c for (nn_, _), c in cells.items() if nn_ == n]
        for model in MODELS:
            for eps in eps_list:
                e = str(eps)
                row = {"qubits": n, "model": model, "eps": eps,
                       "clean": ms([c[model]["clean"] for c in seeds])}
                for a in ATTACKS:
                    row[a] = ms([c[model]["attacks"][a][e] for c in seeds])
                row["worst"] = ms([c[model]["worst_case"][e] for c in seeds])
                rows.append(row)

                fg = np.array([c[model]["attacks"]["FGSM"][e] for c in seeds])
                ls = np.array([c[model]["attacks"]["FGSM-LS"][e] for c in seeds])
                pg = np.array([c[model]["attacks"]["PGD-50x5"][e] for c in seeds])
                sp = np.array([c[model]["attacks"]["SPSA"][e] for c in seeds])
                tr = np.array([c[model]["attacks"]["Transfer(MLP)"][e] for c in seeds])
                checks.append({
                    "qubits": n, "model": model, "eps": eps,
                    "FGSM-PGD gap": f"{np.mean(fg - pg):+.3f}",
                    "gap closed by LS": (f"{np.mean(fg - ls) / np.mean(fg - pg):.0%}"
                                         if np.mean(fg - pg) > 0.02 else "-"),
                    "SPSA beats PGD": f"{np.mean(sp < pg - 1e-9):.0%} of seeds",
                    "transfer beats PGD": f"{np.mean(tr < pg - 1e-9):.0%} of seeds",
                    "non-monotone pts": f"{np.mean([c[model]['nonmonotone'][e] for c in seeds]):.0%}",
                })
        for model in MODELS:
            checks.append({"qubits": n, "model": model, "eps": "1.0 (unbounded)",
                           "FGSM-PGD gap": f"acc {np.mean([c[model]['unbounded_pgd'] for c in seeds]):.3f}",
                           "gap closed by LS": "", "SPSA beats PGD": "", "transfer beats PGD": "",
                           "non-monotone pts": ""})
    return rows, checks


def plot(cells, qubits, eps_list, out_dir, name="masking_study.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MODELS), len(qubits), figsize=(4 * len(qubits), 3.2 * len(MODELS)),
                             sharey=True, squeeze=False)
    styles = {"FGSM": ("#ef4444", "o"), "FGSM-LS": ("#f59e0b", "s"), "PGD-50x5": ("#111827", "D"),
              "SPSA": ("#8b5cf6", "^"), "Transfer(MLP)": ("#06b6d4", "v")}
    xs = [0] + list(eps_list)
    for j, n in enumerate(qubits):
        seeds = [c for (nn_, _), c in cells.items() if nn_ == n]
        for i, model in enumerate(MODELS):
            ax = axes[i][j]
            clean = np.mean([c[model]["clean"] for c in seeds])
            for a, (col, mk) in styles.items():
                vals = np.array([[c[model]["attacks"][a][str(e)] for e in eps_list] for c in seeds])
                mean, sd = vals.mean(0), vals.std(0)
                ax.errorbar(xs, [clean, *mean], yerr=[0, *sd], color=col, marker=mk, ms=4, lw=1.4,
                            capsize=2, label=a)
            ax.set_title(f"{model}, {n} features" + (" (qubits)" if model == "VQC" else ""), fontsize=10)
            ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
            if j == 0: ax.set_ylabel("robust accuracy")
            if i == len(MODELS) - 1: ax.set_xlabel("epsilon (L-inf)")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Single-step vs iterative vs gradient-free attacks (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=130)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the ± in tables on Windows consoles
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--vqc-epochs", type=int, default=20)
    ap.add_argument("--mlp-epochs", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--spsa-iters", type=int, default=40)
    ap.add_argument("--spsa-samples", type=int, default=32)
    ap.add_argument("--test-limit", type=int, default=0, help="use only the first N test points (0 = all)")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to check the pipeline")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--aggregate", action="store_true",
                    help="merge every raw_*.json in the results folder (e.g. from parallel runs) and report")
    a = ap.parse_args()
    if a.aggregate:
        out_dir = results_dir(PROJECT)
        cells, eps = {}, None
        for f in sorted(out_dir.glob("raw_*.json")):
            d = json.loads(f.read_text())
            eps = d["args"]["eps"]
            for k, v in d["cells"].items():
                n, s = map(int, k.split("_"))
                cells[(n, s)] = v
        qubits = sorted({n for n, _ in cells})
        rows, checks = summarise(cells, qubits, eps)
        print(f"{len(cells)} (qubits, seed) cells from {out_dir}\n")
        print(table(rows, ["qubits", "model", "eps", "clean"] + ATTACKS + ["worst"]))
        print("\nGradient-masking checklist:")
        print(table(checks, list(checks[0])))
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "checks": checks, "eps": eps}, indent=1))
        plot(cells, qubits, eps, out_dir)
        print(f"\nSaved summary.json and masking_study.png to {out_dir}")
        return
    if a.smoke:
        a.qubits, a.seeds, a.eps = [4], 1, [0.1, 0.3]
        a.vqc_epochs, a.mlp_epochs, a.restarts, a.spsa_iters, a.spsa_samples, a.test_limit = 5, 20, 2, 10, 8, 40
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = {k: getattr(a, k) for k in ("vqc_epochs", "mlp_epochs", "restarts", "spsa_iters", "spsa_samples", "test_limit")}

    out_dir = results_dir(PROJECT + ("_smoke" if a.smoke else ""))
    tag = "q" + "-".join(map(str, a.qubits))
    raw_path = out_dir / f"raw_{tag}.json"
    cells = {}
    for n in a.qubits:
        for seed in range(a.seeds):
            t0 = time.perf_counter()
            c = run_cell(n, seed, a.eps, cfg)
            cells[(n, seed)] = c
            vq = c["VQC"]
            e = str(a.eps[-1])
            print(f"qubits={n} seed={seed}  clean={vq['clean']:.3f}  eps={e}: FGSM={vq['attacks']['FGSM'][e]:.3f} "
                  f"PGD={vq['attacks']['PGD-50x5'][e]:.3f} SPSA={vq['attacks']['SPSA'][e]:.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            raw_path.write_text(json.dumps({"args": vars(a), "cells": {f"{k[0]}_{k[1]}": v for k, v in cells.items()}},
                                           indent=1))
    rows, checks = summarise(cells, a.qubits, a.eps)
    print("\nRobust accuracy, mean ± std over seeds:")
    print(table(rows, ["qubits", "model", "eps", "clean"] + ATTACKS + ["worst"]))
    print("\nGradient-masking checklist:")
    print(table(checks, list(checks[0])))
    (out_dir / f"summary_{tag}.json").write_text(json.dumps({"args": vars(a), "rows": rows, "checks": checks}, indent=1))
    plot(cells, a.qubits, a.eps, out_dir, f"masking_study_{tag}.png")
    print(f"\nSaved raw results, summary and plot to {out_dir}")


if __name__ == "__main__":
    main()
