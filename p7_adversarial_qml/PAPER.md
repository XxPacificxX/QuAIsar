# Working paper: Do angle-encoded quantum classifiers only *look* robust?

Status: **full draft built from complete results** (15 runs per model + a 25-run frequency sweep). The manuscript is [paper/QuAIsar-p7-paper.pdf](../paper/QuAIsar-p7-paper.pdf), with LaTeX source in `paper/latex/`. Rebuild it with `python paper/build_paper.py --pdf`. These notes are the original planning document.

Before submission: add author names for arXiv:2505.16714 and arXiv:2511.14989 in `paper/build_paper.py` (REFS), and have the author review every claim.

## Claim under test

Reports of adversarial robustness in quantum classifiers rest mostly on gradient-based attacks (FGSM, PGD). If part of that robustness is *gradient masking* (attacks fail because gradients are misleading, not because the model is robust), the apparent advantage should:

1. vanish under strong iterative attacks with restarts,
2. be matched or beaten by gradient-free and transfer attacks, and
3. appear equally in a **classical** model that shares the quantum model's periodic input encoding.

Point 3 is the new test. Angle encoding with *L* re-uploading layers makes the model a truncated Fourier series in each input with frequencies up to *L* (Schuld, Sweke & Meyer, PRA 2021). A periodic, curved loss surface lets a single full gradient-sign step overshoot. That makes FGSM look weak without the model being robust. A classical "SinMLP" with sin/cos(kπx) features, k = 1..L, has the same spectrum and nothing quantum.

## Related work (to expand)

- **West et al., 2022** ([arXiv:2211.12681](https://arxiv.org/abs/2211.12681)): amplitude-encoded VQCs (10–12 qubits). FGSM, PGD, AutoAttack and transfer attacks. Reports "one-way" resilience: classical adversarial examples transfer poorly to VQCs. Cites Athalye et al. but runs no masking checklist.
- **SoK: Critical evaluation of QML for adversarial robustness, 2025** ([arXiv:2511.14989](https://arxiv.org/abs/2511.14989)): angle and amplitude encoding, FGSM and PGD only, ε ≤ 0.15. Finds quantum MLPs *more* vulnerable to gradient-based evasion. No masking checks, no gradient-free or transfer attacks.
- **Experimental robustness benchmarking on superconducting hardware, 2025** ([arXiv:2505.16714](https://arxiv.org/abs/2505.16714)): attributes higher measured robustness to noise-induced gradient attenuation, a mechanism close to gradient masking.
- **Gong et al., 2022** ([arXiv:2212.02531](https://arxiv.org/abs/2212.02531)): randomized encodings cause barren plateaus that hide gradients, proposed as a defense.
- **Athalye, Carlini & Wagner, ICML 2018**: obfuscated gradients and the evaluation checklist used here.
- **Uesato et al., ICML 2018**: SPSA gradient-free attack.

**Gap:** no prior work found that applies the masking checklist to angle-encoded VQCs, or separates encoding-driven effects from quantum ones with a classical spectral control.

## Method

`masking_study.py` (this folder). Digits 3 vs 8, PCA to *n* ∈ {4, 6, 8} features (one per qubit), inputs in [0, 1]. Five seeds; each seed changes the train/test split and the initialisation.

| Model | Description |
|---|---|
| VQC | RY(πx) angle encoding re-uploaded before each of 3 strongly entangling layers; ⟨Z₀⟩ → logit |
| SinMLP | sin/cos(kπx), k = 1..3 → MLP(32, 32): same input spectrum, classical |
| MLP | MLP(32, 32) on raw features |

Attacks (L∞, margin loss): FGSM; FGSM with line search along the sign direction; PGD-20; PGD-50 with 5 restarts; SPSA (gradient-free); transfer from the MLP; unbounded PGD (ε = 1) as a sanity check.

## Results

*Pending the full run.*

## Threats to validity

- Exact state-vector simulation: no shot noise or hardware noise, both of which may add their own masking.
- Small datasets (digits, PCA-reduced) and small models.
- SPSA settings bound how strong the gradient-free attack is; a negative SPSA result is weaker evidence than a positive one.
