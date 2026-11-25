# Qortex

**Q**uantum + c**ortex**: a quantum × cyber × AI research lab.

Five research starters where quantum computing, security, and machine learning meet.
Every project runs end to end on a CPU laptop in minutes and writes its results to `results/<project>/`.

| # | Folder | Question | Quantum | Cyber | AI |
|---|--------|----------|---------|-------|----|
| 4 | `p4_qml_ids` | Do quantum classifiers help intrusion detection? | quantum kernel SVM, VQC | network intrusion data | classical baselines |
| 5 | `p5_pqc_sidechannel` | Can deep learning extract ML-KEM secrets from power traces? | post-quantum crypto target | side-channel attack | 1-D CNN vs templates |
| 6 | `p6_lwe_ml_attack` | Can a neural net recover an LWE secret? | lattice crypto (PQC basis) | cryptanalysis | SALSA-style model + distinguisher |
| 7 | `p7_adversarial_qml` | Are quantum classifiers robust to adversarial inputs? | VQC | evasion attacks | FGSM / PGD, adversarial training |
| 8 | `p8_rl_quantum` | Can RL learn to decode quantum errors? | surface code QEC | protecting quantum info | Deep Q-Network vs MWPM |

## Setup (Windows)

Use a short venv path (Windows path-length and App Control quirks) and call tools via `python -m`:

```powershell
python -m venv C:\Users\pacif\.venvs\qca
C:\Users\pacif\.venvs\qca\Scripts\python.exe -m pip install numpy scipy scikit-learn pandas matplotlib xgboost gymnasium pennylane pymatching
C:\Users\pacif\.venvs\qca\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run any project from this folder:

```powershell
C:\Users\pacif\.venvs\qca\Scripts\python.exe p4_qml_ids\run.py
```

Every script takes `--help`.

## Where each project goes next

**4 · Quantum IDS** — Download UNSW-NB15 (`UNSW_NB15_training-set.csv`) into `p4_qml_ids/data/`; the script picks it up automatically. Then vary `--qubits` and `--train` and plot accuracy and runtime curves. Try projected quantum kernels, and run on IBM hardware via `qiskit` to measure how much hardware noise costs. The honest finding is usually "no quantum advantage yet", and quantifying *why* is the contribution.

**5 · PQC side channels** — The simulator leaks Hamming weights of `s mod q` and `s·c mod q`. The script runs two attack styles with both templates and a CNN. *Direct* classifies `s` from the trace alone and cannot separate s=1 from s=2, which both load with HW 1. *Intermediate* predicts the HW of each intermediate value, then scores every hypothesis `s` against the known ciphertext coefficient `c`. That is how real profiled attacks work. Next steps: capture real traces of liboqs ML-KEM on a ChipWhisperer or use public post-quantum trace sets; add first-order masking (split `s` into two random shares) and show both attacks fail until they combine the two leak points; and attack all 256 coefficients with one shared model.

**6 · ML vs LWE** — Push `--n` and `--h` up until the attack breaks, and chart where. Add BKZ lattice reduction (`fpylll`, easiest under WSL) as the real classical baseline. Then follow SALSA Picante/Verde: preprocess samples with lattice reduction before training, and use transformers.

**7 · Adversarial QML** — Compare against an MLP with matched parameter count, test whether the "concentration" of quantum models makes them more or less robust as qubits grow, and attack the p4 IDS models (evasion against a quantum IDS).

**8 · RL decoding** — Increase `--d 5`, add Y/Z noise (depolarizing) using both Hz and Hx, add measurement errors (repeated syndrome rounds), or replace the DQN with a GNN policy. A second track is RL for QKD: an agent that tunes BB84 basis bias and error-correction parameters under drifting channel noise.

## Caveats

- All quantum circuits run on exact simulators (PennyLane `default.qubit`); no hardware noise or shot noise unless you add it.
- Datasets in 4 (fallback) and 5 are synthetic; label any reported results accordingly.
- Project 6 parameters are intentionally toy-sized and insecure. Nothing here threatens real ML-KEM.
