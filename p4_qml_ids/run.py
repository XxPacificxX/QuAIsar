"""Project 4 - Hybrid quantum-classical intrusion detection.

Compares classical classifiers against a quantum-kernel SVM and a variational
quantum classifier (VQC) on network-flow data.

Data: put UNSW-NB15 `UNSW_NB15_training-set.csv` (has a `label` column) in
p4_qml_ids/data/. Without it the script falls back to a synthetic, flow-like
dataset and says so in the output.

    python p4_qml_ids/run.py --qubits 4 --train 200 --test 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import save_results, set_seed, table, timer  # noqa: E402

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

DATA = Path(__file__).parent / "data" / "UNSW_NB15_training-set.csv"


def load_data(seed: int):
    if DATA.exists():
        import pandas as pd

        df = pd.read_csv(DATA)
        y = df["label"].to_numpy().astype(int)
        X = df.drop(columns=["label", "id", "attack_cat"], errors="ignore")
        X = X.select_dtypes("number").to_numpy(dtype=float)
        X = np.log1p(np.clip(X, 0, None))  # heavy-tailed flow counters
        return X, y, "UNSW-NB15"
    X, y = make_classification(
        n_samples=4000, n_features=30, n_informative=8, n_redundant=10,
        weights=[0.6, 0.4], flip_y=0.03, class_sep=0.9, random_state=seed,
    )
    return X, y, "synthetic (drop UNSW-NB15 CSV into p4_qml_ids/data/ for real data)"


def preprocess(X, y, n_qubits, n_train, n_test, seed):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=n_train, test_size=n_test, stratify=y, random_state=seed
    )
    pipe_scale = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n_qubits, random_state=seed).fit(pipe_scale.transform(Xtr))
    ang = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(pipe_scale.transform(Xtr)))
    f = lambda A: np.clip(ang.transform(pca.transform(pipe_scale.transform(A))), 0, np.pi)
    return f(Xtr), f(Xte), ytr, yte


# ---------------------------------------------------------------- quantum models
def quantum_kernel(Xtr, Xte, n_qubits, reps=2):
    """Fidelity kernel k(x,y)=|<phi(x)|phi(y)>|^2 with an IQP-style feature map.

    Each sample's statevector is simulated once, so the Gram matrix is a matrix
    product instead of N^2 circuit runs (exact simulation, no shot noise).
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def feature_state(x):
        for _ in range(reps):
            for w in range(n_qubits):
                qml.Hadamard(w)
                qml.RZ(x[w], w)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.IsingZZ((np.pi - x[i]) * (np.pi - x[j]) / np.pi, wires=[i, j])
        return qml.state()

    Str = np.array([feature_state(x) for x in Xtr])
    Ste = np.array([feature_state(x) for x in Xte])
    K_train = np.abs(Str.conj() @ Str.T) ** 2
    K_test = np.abs(Ste.conj() @ Str.T) ** 2
    return K_train, K_test


def train_vqc(Xtr, ytr, Xte, n_qubits, layers=3, epochs=40, lr=0.1, seed=0):
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_qubits)
    w = pnp.array(rng.normal(0, 0.1, size=shape), requires_grad=True)
    b = pnp.array(0.0, requires_grad=True)
    yt = pnp.array(2 * ytr - 1, requires_grad=False)  # {-1, +1}
    Xb = pnp.array(Xtr, requires_grad=False)  # (batch, features): broadcast over samples

    def loss(w, b):
        pred = circuit(w, Xb) + b
        return pnp.mean((pred - yt) ** 2)

    opt = qml.AdamOptimizer(lr)
    for ep in range(epochs):
        (w, b), cost = opt.step_and_cost(loss, w, b)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    VQC epoch {ep:3d}  loss {float(cost):.4f}")
    scores = np.asarray(circuit(w, pnp.array(Xte))) + float(b)
    return (scores > 0).astype(int)


# ---------------------------------------------------------------- main
def metrics(name, y, p, t):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y, p), 3),
        "f1": round(f1_score(y, p), 3),
        "attack_recall": round(recall_score(y, p), 3),
        "seconds": t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)

    X, y, source = load_data(a.seed)
    print(f"Data: {source}  ({X.shape[0]} rows, {X.shape[1]} features)")
    Xtr, Xte, ytr, yte = preprocess(X, y, a.qubits, a.train, a.test, a.seed)
    print(f"Compressed to {a.qubits} features (one per qubit); train={len(ytr)} test={len(yte)}\n")

    rows, t = [], {}
    classical = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(200, random_state=a.seed),
        "SVM (RBF)": SVC(kernel="rbf", gamma="scale"),
    }
    try:
        from xgboost import XGBClassifier

        classical["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, verbosity=0)
    except ImportError:
        pass
    for name, m in classical.items():
        with timer(t, name):
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
        rows.append(metrics(name, yte, p, t[name]))

    print("Quantum kernel SVM ...")
    with timer(t, "qk"):
        Ktr, Kte = quantum_kernel(Xtr, Xte, a.qubits)
        p = SVC(kernel="precomputed").fit(Ktr, ytr).predict(Kte)
    rows.append(metrics("Quantum-kernel SVM", yte, p, t["qk"]))

    print("Variational quantum classifier ...")
    with timer(t, "vqc"):
        p = train_vqc(Xtr, ytr, Xte, a.qubits, epochs=a.epochs, seed=a.seed)
    rows.append(metrics("VQC", yte, p, t["vqc"]))

    print("\n" + table(rows, ["model", "accuracy", "f1", "attack_recall", "seconds"]))
    path = save_results("p4_qml_ids", {"data": source, "args": vars(a), "results": rows})
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
