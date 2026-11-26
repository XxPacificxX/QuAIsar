"""Mechanism experiments for the p7 gradient-masking study.

1. Encoding-frequency sweep. The VQC encodes RY(s*pi*x) and the SinMLP uses
   sin/cos(k*s*pi*x). Raising s makes the loss oscillate faster inside the
   same eps-ball. If periodic encoding causes the FGSM/PGD gap, the gap and the
   share of "overshooting" points should grow with s for BOTH models, and stay
   near zero for the plain MLP.
2. Ray profiles. The margin along the FGSM direction x + t*eps*sign(grad),
   t in [0, 1.5]: shows directly whether one full step overshoots the minimum.

    python p7_adversarial_qml/mechanism.py               # full (about 20-30 min on a laptop CPU)
    python p7_adversarial_qml/mechanism.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from masking_study import (MLP, VQC, SinMLP, accuracy, fgsm, fgsm_ls, grad_sign, load, margin,
                           nonmonotone_fraction, pgd, train)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from common.utils import results_dir, set_seed  # noqa: E402

OUT = results_dir("p7_masking_study")


def attack_suite(model, X, y, eps_list, restarts):
    out = {}
    for eps in eps_list:
        e = str(eps)
        out[e] = {
            "FGSM": accuracy(model, fgsm(model, X, y, eps), y),
            "FGSM-LS": accuracy(model, fgsm_ls(model, X, y, eps), y),
            "PGD-50x5": accuracy(model, pgd(model, X, y, eps, 50, restarts), y),
            "nonmonotone": nonmonotone_fraction(model, X, y, eps),
        }
    return out


def sweep(scales, seeds, n, eps_list, cfg):
    cells = []
    for seed in range(seeds):
        set_seed(seed)
        Xtr, Xte, ytr, yte = load(n, seed)
        mlp = train(MLP(n), Xtr, ytr, cfg["mlp_epochs"], 1e-2)
        cells.append({"seed": seed, "model": "MLP", "scale": None, "clean": accuracy(mlp, Xte, yte),
                      "eps": attack_suite(mlp, Xte, yte, eps_list, cfg["restarts"])})
        for s in scales:
            t0 = time.perf_counter()
            for name, model in (("VQC", VQC(n, scale=s)), ("SinMLP", SinMLP(n, scale=s))):
                set_seed(seed)
                epochs, lr = (cfg["vqc_epochs"], 5e-2) if name == "VQC" else (cfg["mlp_epochs"], 1e-2)
                model = train(model, Xtr, ytr, epochs, lr)
                cells.append({"seed": seed, "model": name, "scale": s, "clean": accuracy(model, Xte, yte),
                              "eps": attack_suite(model, Xte, yte, eps_list, cfg["restarts"])})
            v = cells[-2]["eps"][str(eps_list[-1])]
            print(f"seed={seed} scale={s}: VQC clean={cells[-2]['clean']:.3f} FGSM={v['FGSM']:.3f} "
                  f"PGD={v['PGD-50x5']:.3f} nonmono={v['nonmonotone']:.2f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    return cells


def ray_profiles(n, eps, cfg, t_max=1.5, steps=61):
    """Normalised margin m(t)/m(0) along the FGSM ray for correctly classified test points."""
    set_seed(0)
    Xtr, Xte, ytr, yte = load(n, 0)
    models = {
        "VQC": train(VQC(n), Xtr, ytr, cfg["vqc_epochs"], 5e-2),
        "SinMLP": train(SinMLP(n), Xtr, ytr, cfg["mlp_epochs"], 1e-2),
        "MLP": train(MLP(n), Xtr, ytr, cfg["mlp_epochs"], 1e-2),
    }
    ts = torch.linspace(0, t_max, steps)
    out = {"t": ts.tolist(), "eps": eps}
    for name, m in models.items():
        with torch.no_grad():
            ok = margin(m, Xte, yte) > 0
        X, y = Xte[ok], yte[ok]
        d = grad_sign(m, X, y)
        with torch.no_grad():
            m0 = margin(m, X, y)
            curves = torch.stack([margin(m, (X + t * eps * d).clamp(0, 1), y) for t in ts]) / m0
        q = torch.quantile(curves, torch.tensor([0.25, 0.5, 0.75]), dim=1)
        within = curves[ts <= 1.0 + 1e-6]
        out[name] = {"q25": q[0].tolist(), "median": q[1].tolist(), "q75": q[2].tolist(),
                     "overshoot_share": float((within.argmin(0) < within.shape[0] - 1).float().mean()),
                     "examples": curves[:, :6].T.tolist()}
    return out


def plots(cells, scales, eps_list, rays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = str(eps_list[-1])
    colors = {"VQC": "#8b5cf6", "SinMLP": "#06b6d4", "MLP": "#6b7280"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name in ("VQC", "SinMLP"):
        gap = [[c["eps"][e]["FGSM"] - c["eps"][e]["PGD-50x5"] for c in cells if c["model"] == name and c["scale"] == s]
               for s in scales]
        nm = [[c["eps"][e]["nonmonotone"] for c in cells if c["model"] == name and c["scale"] == s] for s in scales]
        cl = [[c["clean"] for c in cells if c["model"] == name and c["scale"] == s] for s in scales]
        for ax, vals in zip(axes, (gap, nm, cl)):
            ax.errorbar(scales, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], marker="o",
                        capsize=3, color=colors[name], label=name)
    mlp = [c for c in cells if c["model"] == "MLP"]
    for ax, val in zip(axes, (np.mean([c["eps"][e]["FGSM"] - c["eps"][e]["PGD-50x5"] for c in mlp]),
                              np.mean([c["eps"][e]["nonmonotone"] for c in mlp]),
                              np.mean([c["clean"] for c in mlp]))):
        ax.axhline(val, color=colors["MLP"], ls="--", label="MLP (no periodic encoding)")
    for ax, title in zip(axes, (f"FGSM − PGD accuracy gap (ε={e})", f"Share of points where FGSM overshoots (ε={e})",
                                "Clean accuracy")):
        ax.set_title(title, fontsize=10); ax.set_xlabel("encoding frequency scale s"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mechanism_sweep.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    t = np.array(rays["t"])
    for ax, name in zip(axes, ("VQC", "SinMLP", "MLP")):
        r = rays[name]
        for ex in r["examples"]:
            ax.plot(t, ex, color=colors[name], alpha=0.25, lw=0.8)
        ax.fill_between(t, r["q25"], r["q75"], color=colors[name], alpha=0.2)
        ax.plot(t, r["median"], color=colors[name], lw=2, label="median")
        ax.axvline(1.0, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{name}: FGSM overshoots for {r['overshoot_share']:.0%} of points", fontsize=10)
        ax.set_xlabel("step size t (in units of ε)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("margin / clean margin")
    fig.suptitle(f"Margin along the FGSM direction (ε={rays['eps']}); the full FGSM step is t = 1", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "ray_profiles.png", dpi=150); plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--vqc-epochs", type=int, default=20)
    ap.add_argument("--mlp-epochs", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.scales, a.seeds, a.eps, a.vqc_epochs, a.mlp_epochs, a.restarts = [0.5, 1.0], 1, [0.3], 4, 10, 1
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = {"vqc_epochs": a.vqc_epochs, "mlp_epochs": a.mlp_epochs, "restarts": a.restarts}

    cells = sweep(a.scales, a.seeds, a.qubits, a.eps, cfg)
    rays = ray_profiles(a.qubits, a.eps[-1], cfg)
    name = "mechanism_smoke.json" if a.smoke else "mechanism.json"
    (OUT / name).write_text(json.dumps({"args": vars(a), "sweep": cells, "rays": rays}, indent=1))
    if not a.smoke:
        plots(cells, a.scales, a.eps, rays)
    for nm in ("VQC", "SinMLP", "MLP"):
        print(f"ray profile {nm}: FGSM overshoots for {rays[nm]['overshoot_share']:.0%} of points")
    print(f"Saved {OUT / name}")


if __name__ == "__main__":
    main()
