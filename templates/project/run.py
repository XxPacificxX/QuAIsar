"""{{NAME}} - {{QUESTION}}

Created from templates/project. Replace `run_once` with your experiment and
keep its return shape (metric name -> number); the multi-seed loop, summary
table and results file come for free.

    python {{NAME}}/run.py --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import save_results, set_seed, table  # noqa: E402

PROJECT = "{{NAME}}"


def run_once(args, seed: int) -> dict[str, float]:
    """One complete experiment for one seed. Return {metric: value}.

    The placeholder trains a classical baseline on synthetic data. Swap in
    your quantum model, attack or decoder; return every number you want
    averaged across seeds.
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=args.samples, n_features=8, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return {"accuracy": float(model.score(Xte, yte))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3, help="repeat the experiment with this many seeds")
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()

    runs = []
    for seed in range(a.seeds):
        set_seed(seed)
        metrics = run_once(a, seed)
        runs.append({"seed": seed, **metrics})
        print(f"  seed {seed}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    names = [k for k in runs[0] if k != "seed"]
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs]))} for k in names}
    rows = [{"metric": k, "mean": round(v["mean"], 4), "std": round(v["std"], 4)} for k, v in summary.items()]
    print("\n" + table(rows, ["metric", "mean", "std"]))
    print(f"\nSaved {save_results(PROJECT, {'args': vars(a), 'runs': runs, 'summary': summary})}")


if __name__ == "__main__":
    main()
