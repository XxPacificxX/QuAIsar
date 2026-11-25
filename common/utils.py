"""Shared helpers: seeding, results output, timing."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def save_results(project: str, data: dict) -> Path:
    out = RESULTS / project
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(data, indent=2, default=float))
    return path


def results_dir(project: str) -> Path:
    out = RESULTS / project
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - t0, 3)


def table(rows: list[dict], cols: list[str]) -> str:
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in cols) for r in rows)
    return f"{line}\n{sep}\n{body}"
