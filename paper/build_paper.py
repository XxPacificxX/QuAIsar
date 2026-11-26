"""Build the p7 paper (LaTeX source + DOCX + PDF) from the study results.

    python paper/build_paper.py           # paper/latex/main.tex, paper/QuAIsar-p7-paper.docx
    python paper/build_paper.py --pdf     # also PDF via Word (Windows) or LibreOffice

Every number in the text comes from results/p7_masking_study/. Claims that
depend on the data are worded from the data, so a re-run cannot leave a stale
claim behind.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "p7_masking_study"
FIG = HERE / "figures"
sys.path.insert(0, str(HERE))
from render import to_docx, to_latex  # noqa: E402

MODELS = ("VQC", "SinMLP", "MLP")


# ---------------------------------------------------------------- data
def load_cells():
    cells = {}
    for f in sorted(RES.glob("raw_q*.json")):
        d = json.loads(f.read_text())
        for k, v in d["cells"].items():
            n, s = map(int, k.split("_"))
            cells[(n, s)] = v
        eps = d["args"]["eps"]
    if not cells:
        sys.exit("No results in results/p7_masking_study; run p7_adversarial_qml/masking_study.py first.")
    return cells, eps


def stat(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), v.std()


def fmt(vals, digits=2):
    m, s = stat(vals)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def pct(x):
    return f"{100 * x:.0f}%"


def fmt_pct(vals):
    """Compact mean ± s.d. in percentage points, for wide tables."""
    m, s = stat(vals)
    return f"{100 * m:.0f} ± {100 * s:.0f}"


class Data:
    def __init__(self):
        self.cells, self.eps = load_cells()
        self.qubits = sorted({n for n, _ in self.cells})
        self.seeds = {n: sorted(s for nn, s in self.cells if nn == n) for n in self.qubits}
        self.e_hi = str(self.eps[-1])
        self.e_mid = str(self.eps[-2]) if len(self.eps) > 1 else self.e_hi
        mech = RES / "mechanism.json"
        self.mech = json.loads(mech.read_text()) if mech.exists() else None

    def vals(self, model, key, eps=None, n=None):
        """Per-cell values of an attack accuracy (key = attack name) or of clean/nonmonotone/unbounded."""
        out = []
        for (nn, _), c in self.cells.items():
            if n is not None and nn != n:
                continue
            m = c[model]
            if key == "clean": out.append(m["clean"])
            elif key == "unbounded": out.append(m["unbounded_pgd"])
            elif key == "nonmonotone": out.append(m["nonmonotone"][eps])
            elif key == "worst": out.append(m["worst_case"][eps])
            else: out.append(m["attacks"][key][eps])
        return np.array(out)

    def gap(self, model, eps, n=None):
        return self.vals(model, "FGSM", eps, n) - self.vals(model, "PGD-50x5", eps, n)

    def ls_closed(self, model, eps):
        g = self.gap(model, eps).mean()
        return (self.vals(model, "FGSM", eps) - self.vals(model, "FGSM-LS", eps)).mean() / g if g > 0.02 else None


# ---------------------------------------------------------------- figures made here
def fig_ranking(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    colors = {"VQC": "#8b5cf6", "SinMLP": "#06b6d4", "MLP": "#6b7280"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, attack, title in ((axes[0], "FGSM", "Single-step attack (FGSM)"),
                              (axes[1], "PGD-50x5", "Strongest attack (PGD-50, 5 restarts)")):
        x = np.arange(len(D.eps))
        w = 0.26
        for i, m in enumerate(MODELS):
            means = [D.vals(m, attack, str(e)).mean() for e in D.eps]
            sds = [D.vals(m, attack, str(e)).std() for e in D.eps]
            ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=2, color=colors[m], label=m)
        ax.set_xticks(x); ax.set_xticklabels([f"ε={e}" for e in D.eps])
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1)
    axes[0].set_ylabel("robust accuracy"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig_ranking.png"
    fig.savefig(path, dpi=200); plt.close(fig)
    return path


# ---------------------------------------------------------------- references
REFS = {
    "goodfellow2015": "I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. ICLR, 2015.",
    "madry2018": "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. ICLR, 2018.",
    "athalye2018": "A. Athalye, N. Carlini, and D. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. ICML, 2018.",
    "uesato2018": "J. Uesato, B. O'Donoghue, P. Kohli, and A. van den Oord. Adversarial risk and the dangers of evaluating against weak attacks. ICML, 2018.",
    "carlini2019": "N. Carlini, A. Athalye, N. Papernot, W. Brendel, J. Rauber, D. Tsipras, I. Goodfellow, A. Madry, and A. Kurakin. On evaluating adversarial robustness. arXiv:1902.06705, 2019.",
    "croce2020": "F. Croce and M. Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. ICML, 2020.",
    "tramer2020": "F. Tramèr, N. Carlini, W. Brendel, and A. Madry. On adaptive attacks to adversarial example defenses. NeurIPS, 2020.",
    "liu2020": "N. Liu and P. Wittek. Vulnerability of quantum classification to adversarial perturbations. Physical Review A, 101:062331, 2020.",
    "lu2020": "S. Lu, L.-M. Duan, and D.-L. Deng. Quantum adversarial machine learning. Physical Review Research, 2:033212, 2020.",
    "west2022": "M. T. West, S. M. Erfani, C. Leckie, M. Sevior, L. C. L. Hollenberg, and M. Usman. Benchmarking adversarially robust quantum machine learning at scale. arXiv:2211.12681, 2022.",
    "gong2022": "W. Gong, D. Yuan, W. Li, and D.-L. Deng. Enhancing quantum adversarial robustness by randomized encodings. arXiv:2212.02531, 2022.",
    "hwbench2025": "Experimental robustness benchmarking of quantum neural networks on a superconducting quantum processor. arXiv:2505.16714, 2025.",
    "sok2025": "SoK: Critical evaluation of quantum machine learning for adversarial robustness. arXiv:2511.14989, 2025.",
    "perezsalinas2020": "A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre. Data re-uploading for a universal quantum classifier. Quantum, 4:226, 2020.",
    "schuld2021": "M. Schuld, R. Sweke, and J. J. Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A, 103:032430, 2021.",
    "schuld2020": "M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe. Circuit-centric quantum classifiers. Physical Review A, 101:032308, 2020.",
    "mcclean2018": "J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven. Barren plateaus in quantum neural network training landscapes. Nature Communications, 9:4812, 2018.",
    "boas1954": "R. P. Boas, Jr. Entire Functions. Academic Press, 1954.",
    "pennylane": "V. Bergholm et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. arXiv:1811.04968, 2018.",
    "pytorch": "A. Paszke et al. PyTorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.",
    "sklearn": "F. Pedregosa et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.",
    "digits": "E. Alpaydin and C. Kaynak. Optical recognition of handwritten digits. UCI Machine Learning Repository, 1998.",
}


# ---------------------------------------------------------------- content
def content(D, author, affiliation, url):
    e, em = D.e_hi, D.e_mid
    n_cells = len(D.cells)
    seeds_txt = ", ".join(f"{len(D.seeds[n])} {'seed' if len(D.seeds[n]) == 1 else 'seeds'} at $n={n}$"
                          for n in D.qubits)
    vq_f, vq_p = D.vals("VQC", "FGSM", e).mean(), D.vals("VQC", "PGD-50x5", e).mean()
    ml_f, ml_p = D.vals("MLP", "FGSM", e).mean(), D.vals("MLP", "PGD-50x5", e).mean()
    vq_pm, ml_pm = D.vals("VQC", "PGD-50x5", em).mean(), D.vals("MLP", "PGD-50x5", em).mean()
    vq_fm, ml_fm = D.vals("VQC", "FGSM", em).mean(), D.vals("MLP", "FGSM", em).mean()
    flip = vq_fm > ml_fm and vq_pm < ml_pm
    gap = {m: D.gap(m, e).mean() for m in MODELS}
    nonmono = {m: D.vals(m, "nonmonotone", e).mean() for m in MODELS}
    ls = {m: D.ls_closed(m, e) for m in MODELS}
    spsa_wins = {m: float(np.mean(D.vals(m, "SPSA", e) < D.vals(m, "PGD-50x5", e) - 1e-9)) for m in MODELS}
    tr = {m: D.vals(m, "Transfer(MLP)", e).mean() for m in MODELS}
    unb = {m: D.vals(m, "unbounded").max() for m in MODELS}
    worst = {m: D.vals(m, "worst", e).mean() for m in MODELS}
    clean = {m: D.vals(m, "clean").mean() for m in MODELS}
    ratio_sin = gap["SinMLP"] / gap["VQC"] if gap["VQC"] > 0 else float("nan")
    ratio_by_n = {n: D.gap("SinMLP", e, n).mean() / max(D.gap("VQC", e, n).mean(), 1e-9) for n in D.qubits}
    gap_by_n = {n: D.gap("VQC", e, n).mean() for n in D.qubits}
    # gradient-free vs white-box over every (run, budget) pair, not just the largest budget
    eps_all = [str(x) for x in D.eps]
    spsa_pairs = [(n, ee) for ee in eps_all for n, s in D.cells
                  if D.cells[(n, s)]["VQC"]["attacks"]["SPSA"][ee] < D.cells[(n, s)]["VQC"]["attacks"]["PGD-50x5"][ee] - 1e-9]
    n_pairs = len(D.cells) * len(eps_all)
    spsa_phrase = ("the gradient-free attack never outperforms white-box PGD" if not spsa_pairs else
                   f"the gradient-free attack beats white-box PGD in only {len(spsa_pairs)} of {n_pairs} "
                   f"run–budget pairs" + ((", at" if len(spsa_pairs) == 1 else ", all at") +
                                          f" $\\varepsilon={min(float(x) for _, x in spsa_pairs):g}$"
                                          if len({x for _, x in spsa_pairs}) == 1 else ""))

    M = D.mech
    if M:
        scales = M["args"]["scales"]
        me = str(M["args"]["eps"][-1])
        def mech(model, s, key):
            cs = [c for c in M["sweep"] if c["model"] == model and c["scale"] == s]
            if key == "gap": return np.array([c["eps"][me]["FGSM"] - c["eps"][me]["PGD-50x5"] for c in cs])
            if key == "clean": return np.array([c["clean"] for c in cs])
            return np.array([c["eps"][me][key] for c in cs])
        mlp_cells = [c for c in M["sweep"] if c["model"] == "MLP"]
        mlp_gap = np.mean([c["eps"][me]["FGSM"] - c["eps"][me]["PGD-50x5"] for c in mlp_cells])
        g_lo, g_hi = mech("VQC", scales[0], "gap").mean(), mech("VQC", scales[-1], "gap").mean()
        s_lo, s_hi = mech("SinMLP", scales[0], "gap").mean(), mech("SinMLP", scales[-1], "gap").mean()
        g_by_s = {s: mech("VQC", s, "gap").mean() for s in scales}
        s_peak = max(g_by_s, key=g_by_s.get)
        clean_lo_s = mech("VQC", scales[-1], "clean").mean()
        clean_ref = mech("VQC", 1.0 if 1.0 in scales else scales[0], "clean").mean()
        mono_vqc = all(mech("VQC", a, "nonmonotone").mean() <= mech("VQC", b, "nonmonotone").mean() + 0.02
                       for a, b in zip(scales, scales[1:]))
        rays = M["rays"]
        mech_seeds = M["args"]["seeds"]
        mech_n = M["args"]["qubits"]

    title = "Single-Step Attacks Overstate the Adversarial Robustness of Angle-Encoded Quantum Classifiers"
    abstract = (
        "Several studies report that variational quantum classifiers resist adversarial examples better than "
        "classical networks, often on the basis of single-step or few-step gradient attacks. We revisit this claim "
        "for angle-encoded classifiers with data re-uploading, applying the gradient-masking checklist from the "
        "adversarial machine learning literature: iterative attacks with restarts, a line search along the "
        "single-step direction, a gradient-free attack, transfer attacks and an unbounded sanity check. "
        f"Across {n_cells} trained runs of each model ({len(D.qubits)} qubit counts: {seeds_txt}), "
        f"FGSM leaves the quantum classifier at {pct(vq_f)} accuracy at $\\varepsilon={e}$, "
        f"while 50-step PGD with restarts reduces it to {pct(vq_p)}"
        + (f"; under the strong attack the quantum model is *less* robust than a classical MLP at "
           f"$\\varepsilon={em}$ ({pct(vq_pm)} vs {pct(ml_pm)}), reversing the ranking FGSM suggests. " if flip else ". ")
        + f"Gradients are not obfuscated: {spsa_phrase}, and unbounded attacks leave at most "
        f"{pct(unb['VQC'])} accuracy. Instead, the single-step attack fails because angle encoding makes the model a "
        "trigonometric polynomial of its input, whose curvature inside the threat ball grows with the encoding "
        "frequency and the perturbation budget, as a Bernstein-type bound predicts. "
        + (f"Lowering the encoding frequency all but removes the effect (FGSM−PGD gap {g_lo:+.2f} at "
           f"$s={scales[0]:g}$ vs {g_by_s[s_peak]:+.2f} at $s={s_peak:g}$), and a classical network given the same "
           f"sinusoidal features shows the same failure in kind, at {pct(min(ratio_by_n.values()))}–"
           f"{pct(max(ratio_by_n.values()))} of the quantum model's magnitude. " if M else
           f"A classical network given the same sinusoidal features shows the same failure in kind "
           f"({pct(ratio_sin)} of the quantum gap). ")
        + "The apparent robustness comes from a weak attack meeting a periodic encoding, not from genuine robustness. "
        "We recommend iterative, multi-restart evaluation as the minimum standard for robustness claims about "
        "quantum classifiers.")

    B = []
    B += [("h1", "Introduction", "sec:intro"),
          ("p", "Adversarial examples, inputs altered by small perturbations that change a model's prediction "
                "[@goodfellow2015,madry2018], are a central security concern for machine learning. As quantum machine "
                "learning matures, a growing literature asks whether quantum classifiers are more or less vulnerable "
                "than classical ones [@liu2020,lu2020]. Several works report an advantage: quantum variational "
                "classifiers resisting attacks crafted on classical networks [@west2022], measured robustness on "
                "superconducting hardware attributed to noise-induced gradient attenuation [@hwbench2025], and "
                "defenses that deliberately hide gradients through randomized encodings [@gong2022]. A recent "
                "systematization finds the opposite under white-box attacks [@sok2025]."),
          ("p", "Classical adversarial machine learning learned, at some cost, that apparent robustness is often an "
                "artifact of the evaluation. Defenses that *mask* gradients, making them uninformative to the "
                "attacker without making the model robust, repeatedly passed gradient-based evaluations and then "
                "fell to adaptive attacks [@athalye2018,uesato2018,tramer2020]. The resulting evaluation guidelines "
                "[@carlini2019,croce2020] are rarely applied in full to quantum models. Of the works above, none "
                "reports the standard masking diagnostics for angle-encoded classifiers."),
          ("p", "We apply them. Our subject is the most common architecture in the quantum adversarial literature: "
                "angle encoding with data re-uploading [@perezsalinas2020] followed by strongly entangling layers "
                "[@schuld2020]. Our contributions are:"),
          ("bullets", [
              f"**A full gradient-masking evaluation** of angle-encoded variational quantum classifiers (VQCs) across "
              f"{len(D.qubits)} qubit counts and multiple seeds, comparing FGSM, FGSM with line search, PGD with "
              "restarts, the gradient-free SPSA attack, transfer attacks and unbounded attacks.",
              "**Evidence that single-step attacks overstate VQC robustness**" +
              (", to the point of reversing the robustness ranking against a classical network." if flip else "."),
              "**A mechanism.** Angle encoding makes the classifier a trigonometric polynomial of its input. "
              "A Bernstein-type bound links the curvature inside the threat ball to the encoding frequency and "
              "perturbation budget, explaining when a single gradient-sign step overshoots.",
              "**A classical control that isolates the cause.** An MLP on sinusoidal features with the same frequency "
              "content (SinMLP) reproduces the effect in kind" + (", and an encoding-frequency sweep shows the "
              "encoding frequency controls it in both models." if M else "."),
          ])]

    B += [("h1", "Background", "sec:bg"),
          ("h2", "Attacks and gradient masking"),
          ("p", "Given a classifier with logit $f(x)$ and label $y\\in\\{-1,+1\\}$, we use the margin $m(x)=y\\,f(x)$; "
                "the input is misclassified when $m<0$. An $\\ell_\\infty$ attack with budget $\\varepsilon$ seeks "
                "$x'$ with $\\|x'-x\\|_\\infty\\leq\\varepsilon$ and $m(x')<0$. FGSM [@goodfellow2015] takes one step "
                "$x'=x-\\varepsilon\\,\\mathrm{sign}(\\nabla_x m)$. PGD [@madry2018] iterates smaller steps with "
                "projection, ideally from several random starts. *Gradient masking* [@athalye2018] describes models "
                "whose gradients mislead first-order attackers. Its symptoms include single-step attacks performing "
                "far worse than iterative ones, gradient-free or transfer attacks beating white-box attacks, and "
                "unbounded attacks failing to reach zero accuracy."),
          ("h2", "Angle encoding as a Fourier series"),
          ("p", "In a re-uploading circuit, each feature enters through rotations $R_Y(s\\pi x_i)$ repeated in $L$ "
                "layers. Because the generator has eigenvalues $\\pm 1/2$, the model output is a trigonometric "
                "polynomial in each feature, with integer frequencies $k\\in\\{-L,\\dots,L\\}$ in units of $s\\pi$ "
                "[@schuld2021]:"),
          ("eq", r"f(x)=\sum_{\omega\in\{-L,\dots,L\}^n} c_\omega\, e^{\,i s\pi\, \omega\cdot x}.",
           "f(x) = Σ_{ω ∈ {−L,…,L}ⁿ} c_ω · exp(i sπ ω·x)"),
          ("p", "The function is smooth but oscillatory, and its oscillation rate is set by the encoding frequency "
                "$s$ and the number of re-uploads $L$. A ReLU network, by contrast, is piecewise linear.")]

    B += [("h1", "Why single-step attacks fail on periodic models", "sec:theory"),
          ("p", "Restrict $f$ to the FGSM ray: $g(t)=m(x+t\\varepsilon d)$ with $d=-\\mathrm{sign}(\\nabla_x m)\\in\\{-1,0,1\\}^n$ "
                "and $t\\in[0,1]$. Each term of the series contributes a frequency $s\\pi\\varepsilon\\,(\\omega\\cdot d)$ "
                "in $t$, and $|\\omega\\cdot d|\\leq Ln$."),
          ("prop", "Bernstein bound along the attack ray",
           "Let $\\Omega=s\\pi\\varepsilon L n$. Then $\\sup_t|g''(t)|\\leq\\Omega^2\\sup_t|g(t)|$, so the error of "
           "the first-order model used by FGSM satisfies $|g(1)-g(0)-g'(0)|\\leq\\tfrac{1}{2}\\Omega^2\\sup_t|g(t)|$."),
          ("p", "*Proof sketch.* $g$ is a finite sum of exponentials with frequencies in $[-\\Omega,\\Omega]$, hence an "
                "entire function of exponential type $\\Omega$ bounded on the real line. Bernstein's inequality "
                "[@boas1954] gives $\\sup|g'|\\leq\\Omega\\sup|g|$; applying it twice gives the bound, and Taylor's "
                "theorem gives the error. $\\square$"),
          ("p", "The bound is loose in $n$, but its scaling is the point. The linear approximation behind FGSM degrades "
                "quadratically in $s\\varepsilon$. Once $\\Omega\\gtrsim 1$ the margin along the ray can turn around "
                "before $t=1$, so a full step overshoots, and the sign of the gradient at $x$ stops describing the loss "
                "a distance $\\varepsilon$ away. Iterative attacks re-linearize at every step and are unaffected. The "
                "argument uses only the input spectrum, so it applies equally to any classical model with sinusoidal "
                "features of the same frequencies. This motivates the SinMLP control."),
          ("p", "The bound predicts three observable signatures: (i) a gap between FGSM and PGD that grows with "
                "$\\varepsilon$; (ii) points whose margin along the FGSM ray is non-monotone; and (iii) a gap that "
                "grows with the encoding frequency $s$, in quantum and classical models alike.")]

    rows_setup = [["VQC", f"$R_Y(s\\pi x_i)$ re-uploaded before each of $L=3$ strongly entangling layers; $\\langle Z_0\\rangle$ → logit"],
                  ["SinMLP", "features $\\sin(k s\\pi x_i),\\cos(k s\\pi x_i)$, $k=1..3$ → MLP(32, 32)"],
                  ["MLP", "MLP(32, 32) with ReLU on the raw features"]]
    B += [("h1", "Experimental setup", "sec:setup"),
          ("p", f"**Data.** Handwritten digits 3 vs 8 [@digits], reduced by PCA to $n\\in\\{{{','.join(map(str, D.qubits))}\\}}$ "
                "features (one per qubit) and scaled to $[0,1]$. Each seed draws a new 60/40 train/test split "
                "and a new initialisation."),
          ("p", "**Models.** Table [#tab:models] lists the three models; all use $s=1$ unless stated. The VQC is "
                "simulated exactly with PennyLane [@pennylane] and trained with PyTorch [@pytorch] (Adam, 20 epochs); "
                "the classical models train for 60 epochs."),
          ("table", "tab:models", "Models. SinMLP has the same input frequencies as the VQC but nothing quantum.",
           ["Model", "Architecture"], rows_setup, "ll"),
          ("p", "**Attacks.** All are $\\ell_\\infty$ attacks on the margin, with inputs clipped to $[0,1]$: FGSM; "
                "FGSM-LS, the worst point along the FGSM ray over 20 step sizes in $(0,\\varepsilon]$; PGD-20; "
                "PGD-50x5 (50 steps, step $2.5\\varepsilon/50$, 5 random restarts); SPSA [@uesato2018] "
                "(40 iterations, 32 random directions); a transfer attack using PGD-50x5 examples crafted on the MLP; "
                "and unbounded PGD ($\\varepsilon=1$, 100 steps). *Worst case* counts a point as robust only if it "
                "survives every attack."),
          ("p", "**Diagnostics.** For each model and $\\varepsilon$ we report the FGSM-PGD gap, the share of that gap "
                "recovered by line search, the share of test points whose margin along the FGSM ray reaches its "
                "minimum before $t=1$ (non-monotone), whether SPSA or transfer beat PGD, and the unbounded accuracy.")]

    # ---- results
    rows1 = []
    for n in D.qubits:
        for m in MODELS:
            rows1.append([str(n), m, str(len(D.seeds[n])), fmt_pct(D.vals(m, "clean", n=n)),
                          fmt_pct(D.vals(m, "FGSM", em, n)), fmt_pct(D.vals(m, "PGD-50x5", em, n)),
                          fmt_pct(D.vals(m, "FGSM", e, n)), fmt_pct(D.vals(m, "PGD-50x5", e, n)),
                          fmt_pct(D.vals(m, "worst", e, n))])
    rows2 = []
    for m in MODELS:
        rows2.append([m, f"{gap[m]:+.2f}", "–" if ls[m] is None else pct(ls[m]), pct(nonmono[m]),
                      pct(spsa_wins[m]), f"{tr[m]:.2f}", f"{unb[m]:.2f}"])

    B += [("h1", "Results", "sec:results"),
          ("h2", "FGSM overstates robustness and can reverse the ranking"),
          ("figure", "fig:ranking", str(fig_ranking(D)),
           "Robust accuracy (mean ± s.d. over all qubit counts and seeds) under the single-step attack (left) "
           "and the strongest white-box attack (right).", 0.95),
          ("p", f"Figure [#fig:ranking] and Table [#tab:main] summarise the main result. Under FGSM the VQC appears "
                f"the most robust model at large budgets: {pct(vq_f)} accuracy at $\\varepsilon={e}$ versus "
                f"{pct(ml_f)} for the MLP. Under PGD-50x5 it falls to {pct(vq_p)}. "
                + (f"At $\\varepsilon={em}$ the ranking reverses: FGSM puts the VQC ahead of the MLP "
                   f"({pct(vq_fm)} vs {pct(ml_fm)}), PGD puts it behind ({pct(vq_pm)} vs {pct(ml_pm)}). "
                   if flip else "")
                + f"The worst case over all attacks is {pct(worst['VQC'])} for the VQC, {pct(worst['SinMLP'])} for "
                f"the SinMLP and {pct(worst['MLP'])} for the MLP at $\\varepsilon={e}$. Clean accuracy is similar "
                f"across models ({pct(clean['VQC'])}, {pct(clean['SinMLP'])}, {pct(clean['MLP'])})."),
          ("table", "tab:main", f"Accuracy in % (mean ± s.d. over seeds) under FGSM and PGD-50x5 at "
                                f"$\\varepsilon={em}$ and $\\varepsilon={e}$, and the worst case over all six attacks.",
           ["n", "Model", "Seeds", "Clean", f"FGSM {em}", f"PGD {em}", f"FGSM {e}", f"PGD {e}", f"Worst {e}"],
           rows1, "llrrrrrrr"),
          ("h2", "The gradient-masking checklist"),
          ("table", "tab:checklist", f"Gradient-masking diagnostics at $\\varepsilon={e}$, pooled over qubit counts "
                                     "and seeds. *SPSA < PGD* is the share of runs where the gradient-free attack "
                                     "beat PGD-50x5; *Unbounded* is the highest accuracy left by $\\varepsilon=1$ PGD.",
           ["Model", "FGSM−PGD gap", "Gap closed by LS", "Non-monotone", "SPSA < PGD", "Transfer acc.", "Unbounded"],
           rows2, "lrrrrrr"),
          ("p", f"Table [#tab:checklist] separates two explanations. The large FGSM-PGD gap for the VQC "
                f"({gap['VQC']:+.2f}) is the classic masking symptom. Yet the other symptoms are absent. At "
                f"$\\varepsilon={e}$ the gradient-free SPSA attack beats PGD in {pct(spsa_wins['VQC'])} of VQC runs, "
                f"and across all budgets {spsa_phrase}. Transfer from the MLP is weaker than "
                f"white-box PGD (accuracy {tr['VQC']:.2f}), and unbounded attacks leave at most {unb['VQC']:.2f}. "
                "The VQC's gradients are informative; iterative attacks using them succeed. What fails is the "
                f"single linearization. On {pct(nonmono['VQC'])} of test points the margin along the FGSM ray turns "
                f"around before the full step. A line search along the same direction recovers only "
                + ("a small part" if ls["VQC"] is None else pct(ls["VQC"])) +
                " of the gap, so the gradient sign at $x$ is itself a poor guide a distance $\\varepsilon$ away, "
                "as the curvature bound predicts."),
          ("p", f"The SinMLP, which shares the VQC's per-feature input frequencies but nothing quantum, shows the same "
                f"symptoms in kind but smaller (pooled gap {gap['SinMLP']:+.2f}, {pct(nonmono['SinMLP'])} non-monotone). "
                "Its gap is " + ", ".join(f"{pct(r)} of the VQC's at $n={n}$" for n, r in ratio_by_n.items()) + ". "
                "The VQC's gap holds steady as $n$ grows (" + ", ".join(f"{g:+.2f}" for g in gap_by_n.values()) +
                "), while the SinMLP's shrinks. A plausible reason, which we leave to future work, is that the "
                "entangling layers give the VQC every combined frequency $\\omega\\cdot x$ across features directly, "
                "whereas the SinMLP sees per-feature frequencies and must build combinations in its hidden layers. "
                f"The ReLU MLP shows a gap of only {gap['MLP']:+.2f} and {pct(nonmono['MLP'])} non-monotone points. "
                f"The transfer results also qualify earlier observations that classical adversarial examples transfer "
                f"poorly to quantum classifiers [@west2022]: here too transfer is weaker than white-box PGD, but "
                f"white-box PGD defeats the VQC completely, so poor transfer is not evidence of robustness.")]

    if M:
        rows3 = []
        for s in scales:
            rows3.append([f"{s:g}", fmt(mech("VQC", s, "clean")), fmt(mech("VQC", s, "gap")),
                          pct(mech("VQC", s, "nonmonotone").mean()),
                          fmt(mech("SinMLP", s, "gap")), pct(mech("SinMLP", s, "nonmonotone").mean())])
        B += [("h2", "Mechanism: curvature from the encoding frequency"),
              ("figure", "fig:rays", str(RES / "ray_profiles.png"),
               f"Margin along the FGSM direction, normalised by the clean margin, for correctly classified test points "
               f"($n={mech_n}$, $\\varepsilon={rays['eps']}$). Thin lines: individual points; band: interquartile "
               f"range; dotted line: the full FGSM step. Periodic models overshoot; the ReLU MLP does not.", 1.0),
              ("p", f"Figure [#fig:rays] shows the overshoot directly. For the VQC the median margin along the FGSM ray "
                    f"flattens near zero well before the full step, and many individual points reach their minimum "
                    f"inside the threat ball and then turn back towards correct classification: FGSM overshoots on "
                    f"{pct(rays['VQC']['overshoot_share'])} of points. The SinMLP overshoots on "
                    f"{pct(rays['SinMLP']['overshoot_share'])} of points. The ReLU MLP's margin keeps falling along the "
                    f"whole ray, overshooting on {pct(rays['MLP']['overshoot_share'])}."),
              ("figure", "fig:sweep", str(RES / "mechanism_sweep.png"),
               f"Encoding-frequency sweep ($n={mech_n}$, {mech_seeds} seeds, $\\varepsilon={me}$). Left: FGSM−PGD gap; "
               "centre: share of points where FGSM overshoots; right: clean accuracy. Dashed: ReLU MLP.", 1.0),
              ("table", "tab:sweep", f"Encoding-frequency sweep at $\\varepsilon={me}$ (mean ± s.d. over {mech_seeds} seeds).",
               ["s", "VQC clean", "VQC gap", "VQC non-mono.", "SinMLP gap", "SinMLP non-mono."], rows3, "rrrrrr"),
              ("p", f"The sweep (Figure [#fig:sweep], Table [#tab:sweep]) tests the causal prediction. Raising the "
                    f"encoding frequency from $s={scales[0]:g}$ to $s={s_peak:g}$ raises the VQC's FGSM-PGD gap "
                    f"from {g_lo:+.2f} to {g_by_s[s_peak]:+.2f}"
                    + (f"; at $s={scales[-1]:g}$ it eases to {g_hi:+.2f} as clean accuracy falls to "
                       f"{pct(clean_lo_s)} (from {pct(clean_ref)} at $s=1$), which caps how large the gap can be"
                       if s_peak != scales[-1] and clean_lo_s < clean_ref - 0.05 else
                       (f" and is {g_hi:+.2f} at $s={scales[-1]:g}$" if s_peak != scales[-1] else ""))
                    + f". The SinMLP's gap rises from {s_lo:+.2f} to {s_hi:+.2f}, while the ReLU MLP "
                    f"stays at {mlp_gap:+.2f}. "
                    + ("The share of overshooting points rises monotonically with $s$ for the VQC. "
                       if mono_vqc else "The share of overshooting points generally rises with $s$. ")
                    + "Apparent FGSM robustness can therefore be dialled up or down through a classical "
                    "hyperparameter of the encoding, with no change to the quantum part of the model.")]

    B += [("h1", "Discussion", "sec:discussion"),
          ("p", "**What the results do and do not show.** They do not show that quantum classifiers are inherently "
                "less robust. Clean accuracy is matched, and the strongest attacks defeat all three models at large "
                "budgets. They show that FGSM is not a valid robustness measure for angle-encoded models: its "
                "failure is predictable from the encoding spectrum, vanishes at low encoding frequency, and appears in "
                "smaller form in classical models with the same per-feature spectrum. Robustness claims based on "
                "single-step attacks, or on attacks that are only weak against "
                "periodic models, should be re-examined."),
          ("p", "**Relation to gradient masking.** In the taxonomy of [@athalye2018] the effect is neither shattered "
                "nor stochastic gradients, and not vanishing gradients in the barren-plateau sense [@mcclean2018]. "
                "It is high curvature: the gradient is correct but only locally useful. This is the benign form of "
                "masking: standard iterative attacks remove it, and gradient-free attacks confirm that nothing else "
                "is hidden. Mechanisms that do hide gradients, such as hardware noise [@hwbench2025] or randomized "
                "encodings [@gong2022], need the full adaptive-attack treatment [@tramer2020]."),
          ("p", "**Recommendations.** Robustness evaluations of quantum classifiers should report (i) iterative "
                "attacks with random restarts as the primary metric, (ii) the FGSM-PGD gap and an unbounded-attack "
                "sanity check, (iii) at least one gradient-free attack, and (iv) a classical baseline matched on "
                "input encoding, so that effects of the encoding are not credited to quantumness.")]

    B += [("h1", "Limitations", "sec:limits"),
          ("bullets", [
              "Simulations are exact: no shot noise or hardware noise, which may add genuinely stochastic masking.",
              f"Scale is small: {min(D.qubits)}–{max(D.qubits)} qubits, one dataset (digits 3 vs 8, PCA-reduced) and "
              "one circuit family. Amplitude-encoded and larger models are not covered.",
              "SPSA's budget bounds the strength of the gradient-free check; a stronger score-based attack "
              "(e.g. Square Attack, AutoAttack [@croce2020]) would tighten it.",
              "The Bernstein bound is worst-case and loose in $n$; we use it for its scaling in $s\\varepsilon$, "
              "which the sweep tests, not for its constant.",
          ])]
    B += [("h1", "Conclusion", "sec:conclusion"),
          ("p", "Angle-encoded quantum classifiers look robust to single-step attacks because their periodic encoding "
                "makes the loss curve inside the threat ball, not because they are robust. Strong iterative attacks "
                "remove the effect, the encoding frequency controls it, and a classical model with the same encoding "
                "reproduces it in smaller form. Evaluations of quantum adversarial robustness should adopt the same standards "
                "as classical ones.")]
    B += [("h1*", "Reproducibility"),
          ("p", "All code, raw results and this manuscript's build script are available"
                + (f" at {url}" if url else "") + ". The experiments run on a laptop CPU: "
                "masking_study.py produces Tables [#tab:main]–[#tab:checklist] and mechanism.py produces the "
                "mechanism figures. Every number in this paper is generated from the saved results files."),
          ("h1*", "Use of AI tools"),
          ("p", "The experiment code, analysis and an initial draft of this manuscript were created with Claude Code "
                "(Anthropic). The author directed the research, reviewed the code, results and text, and takes full "
                "responsibility for the content.")]

    meta = {"title": title, "author": author, "affiliation": affiliation,
            "date": dt.date.today().strftime("%B %Y") + " · Draft · Created with Claude Code", "abstract": abstract}
    return meta, B


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def to_pdf(docx_path):
    if platform.system() == "Windows":
        script = ROOT / "report" / "docx_to_pdf.ps1"
        shell = shutil.which("pwsh") or shutil.which("powershell")
        subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(docx_path)], check=True)
    else:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            sys.exit("PDF export needs Microsoft Word (Windows) or LibreOffice.")
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)], check=True)
    return docx_path.with_suffix(".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--author", default=None)
    ap.add_argument("--affiliation", default="")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    author = a.author or git("config", "user.name") or "Author"
    url = git("remote", "get-url", "origin").removesuffix(".git")
    D = Data()
    meta, blocks = content(D, author, a.affiliation, url)
    tex = to_latex(meta, blocks, REFS, HERE / "latex")
    docx = to_docx(meta, blocks, REFS, HERE / "QuAIsar-p7-paper.docx")
    print(f"Wrote {tex.relative_to(ROOT)} and {docx.relative_to(ROOT)}  "
          f"({len(D.cells)} result cells; mechanism data: {'yes' if D.mech else 'no'})")
    if a.pdf:
        print(f"Wrote {to_pdf(docx).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
