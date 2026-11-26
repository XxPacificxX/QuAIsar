"""Build a detailed LinkedIn document (PDF, 4:5 portrait slides) from results/.

The short carousel (make_carousel.py) gives one takeaway per project. This deep
dive explains each project's question, method, result and caveats, with four
slides on the adversarial-robustness study behind the paper.

    python report/make_deep_dive.py     # report/QuAIsar-linkedin-deep-dive.pdf
"""
from __future__ import annotations

import datetime as dt
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_report import load_p7_study  # noqa: E402
from make_carousel import (CYAN, DPI, GOLD, GREY, INK, MUTED, NAVY, PANEL, PINK, RES, ROOT, VIOLET,  # noqa: E402
                           body, chart_axes, draw_mark, load, plt, repo_url, slide, title)

OUT = HERE / "QuAIsar-linkedin-deep-dive.pdf"


def author():
    try:
        return subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def bullets(fig, items, y, size=13, width=50, gap=0.029, color=INK, marker_color=CYAN):
    for item in items:
        lines = textwrap.wrap(item, width)
        fig.text(0.08, y, "•", fontsize=size + 2, color=marker_color, va="top", fontweight="bold")
        for line in lines:
            fig.text(0.115, y, line, fontsize=size, color=color, va="top")
            y -= gap
        y -= gap * 0.45
    return y


def label(fig, text, y, color=CYAN):
    fig.text(0.08, y, text.upper(), fontsize=11, color=color, fontweight="bold", va="top")
    return y - 0.035


def legend_ink(ax, **kw):
    leg = ax.legend(frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK)


def build():
    p4, p5, p6, p8 = (load(p) for p in ("p4_qml_ids", "p5_pqc_sidechannel", "p6_lwe_ml_attack", "p8_rl_quantum"))
    S = load_p7_study()
    if S is None:
        sys.exit("Run p7_adversarial_qml/masking_study.py first (the deep dive uses the multi-seed study).")
    mech_path = RES / "p7_masking_study" / "mechanism.json"
    mech = json.loads(mech_path.read_text()) if mech_path.exists() else None
    url = repo_url()
    who = author()
    pages = []
    N = 19

    def new(tag=None):
        return slide(len(pages) + 1, N, tag)

    # 1 cover --------------------------------------------------------------
    fig = new()
    draw_mark(fig.add_axes([0.30, 0.56, 0.40, 0.32]))
    fig.text(0.5, 0.49, "QuAIsar", fontsize=48, fontweight="bold", ha="center")
    fig.text(0.5, 0.445, "DEEP DIVE", fontsize=14, ha="center", color=CYAN, fontweight="bold")
    fig.text(0.5, 0.385, "Five experiments at the intersection of", fontsize=15, ha="center")
    fig.text(0.5, 0.355, "quantum computing, cybersecurity and AI", fontsize=15, ha="center")
    if who:
        fig.text(0.5, 0.26, who, fontsize=14, ha="center", fontweight="bold")
    fig.text(0.5, 0.23, dt.date.today().strftime("%B %Y"), fontsize=12, ha="center", color=MUTED)
    fig.text(0.5, 0.16, "Created with Claude Code", fontsize=11, ha="center", color=MUTED, style="italic")
    pages.append(fig)

    # 2 context -------------------------------------------------------------
    fig = new("Context")
    y = title(fig, "Why these three fields belong together")
    bullets(fig, [
        "Large quantum computers would break today's public-key cryptography, so the world is moving to "
        "post-quantum schemes such as ML-KEM.",
        "New cryptography brings new attack surfaces: power and timing side channels, implementation bugs, and "
        "machine-learning-powered cryptanalysis.",
        "Quantum machine learning is being proposed for security tasks, yet its own security, such as robustness "
        "to adversarial inputs, is barely tested.",
        "AI is now both the attacker's and the defender's strongest tool.",
    ], y - 0.04, size=14, width=46, gap=0.032)
    body(fig, "QuAIsar asks five concrete, testable questions where these fields meet.", 0.2, size=14, color=CYAN)
    pages.append(fig)

    # 3 approach ----------------------------------------------------------------
    fig = new("Approach")
    y = title(fig, "How I worked")
    bullets(fig, [
        "One question per experiment, always against a classical baseline.",
        "Small enough to run on a laptop CPU in minutes, so anyone can reproduce it.",
        "Every number in every chart is read from saved results files; nothing is typed in by hand.",
        "Limits stated next to each result, not buried at the end.",
        "Tools: PennyLane (quantum simulation), PyTorch, scikit-learn, XGBoost and PyMatching.",
    ], y - 0.04, size=14, width=46, gap=0.032)
    pages.append(fig)

    # 4-5 p4 ----------------------------------------------------------------------
    r4 = p4["results"]
    a4 = p4["args"]
    qk = next(r for r in r4 if r["model"] == "Quantum-kernel SVM")
    cl = [r for r in r4 if r["model"] not in ("Quantum-kernel SVM", "VQC")]
    slow = qk["seconds"] / max(min(r["seconds"] for r in cl), 1e-3)
    fig = new("01 · Quantum intrusion detection")
    y = title(fig, "Do quantum classifiers detect attacks better?")
    y = label(fig, "Why it matters", y - 0.03)
    y = body(fig, "Quantum machine learning is often pitched for security analytics. Before building on it, it "
                  "should beat what we already have.", y, size=13.5, width=50, gap=0.029)
    y = label(fig, "Method", y - 0.02)
    bullets(fig, [
        f"Network-flow features compressed to {a4['qubits']} numbers, one per qubit.",
        "Two quantum models: a quantum-kernel SVM and a variational quantum classifier (VQC).",
        "Four classical baselines: logistic regression, random forest, SVM and XGBoost, on the same features.",
        f"{a4['train']} training flows, {a4['test']} test flows.",
    ], y, size=13, width=50)
    pages.append(fig)

    fig = new("01 · Result")
    y = title(fig, "Same accuracy, far slower")
    ax = chart_axes(fig, [0.12, 0.42, 0.80, 0.30])
    short = {"LogisticRegression": "LogReg", "RandomForest": "Forest", "SVM (RBF)": "SVM", "XGBoost": "XGBoost",
             "Quantum-kernel SVM": "Q-kernel", "VQC": "VQC"}
    cols = [VIOLET if r["model"] == "VQC" else CYAN if r["model"].startswith("Quantum") else GREY for r in r4]
    x = np.arange(len(r4))
    ax.bar(x, [r["accuracy"] for r in r4], color=cols, width=0.62)
    for xi, r in zip(x, r4):
        ax.text(xi, r["accuracy"] + 0.02, f"{r['accuracy']:.0%}", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([short.get(r["model"], r["model"]) for r in r4], fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("test accuracy")
    spread = 100 * (max(r["accuracy"] for r in r4) - min(r["accuracy"] for r in r4))
    y = body(fig, f"All six models land within {spread:.0f} percentage points of each other, while the "
                  f"quantum kernel runs about {slow:,.0f}× slower than the fastest classical model.", 0.36, size=13.5,
             width=50, gap=0.029)
    body(fig, "Caveat: synthetic data and only 4 qubits. The honest finding is \"no advantage yet\", and measuring "
              "why is the real contribution.", y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    # 6-7 p5 ----------------------------------------------------------------------
    r5 = p5["results"]
    fig = new("02 · Side-channel attack on ML-KEM")
    y = title(fig, "Can AI read secret keys from power use?")
    y = label(fig, "Why it matters", y - 0.03)
    y = body(fig, "ML-KEM is the new NIST post-quantum standard. Even unbreakable maths can leak through a "
                  "device's power consumption while it computes.", y, size=13.5, width=50, gap=0.029)
    y = label(fig, "Method", y - 0.02)
    bullets(fig, [
        "Simulated power traces of a device multiplying a secret key coefficient by a ciphertext coefficient "
        "(mod 3329), with noise and timing jitter.",
        "Two attackers: classic Gaussian templates and a 1-D convolutional neural network.",
        "Two strategies: read the secret directly, or predict the leaking intermediate values and test every "
        "possible secret against the known ciphertext.",
    ], y, size=13, width=50)
    pages.append(fig)

    fig = new("02 · Result")
    settings = sorted({(r["noise"], r["jitter"]) for r in r5})
    attacks = list(dict.fromkeys(r["attack"] for r in r5))
    get5 = lambda n, j, a: next(r for r in r5 if r["noise"] == n and r["jitter"] == j and r["attack"] == a)
    pairs = [(get5(n, j, f"{m} direct"), get5(n, j, f"{m} intermediate")) for n, j in settings for m in ("Template", "CNN")]
    d, i = max(pairs, key=lambda p: p[0]["traces_to_recover"] / p[1]["traces_to_recover"])
    y = title(fig, f"Using the ciphertext: up to {d['traces_to_recover'] / i['traces_to_recover']:.0f}× fewer traces")
    ax = chart_axes(fig, [0.12, 0.40, 0.80, 0.30])
    colors = {"Template direct": GREY, "Template intermediate": CYAN, "CNN direct": "#475569", "CNN intermediate": PINK}
    w = 0.2
    for k, a in enumerate(attacks):
        vals = [get5(n, j, a)["traces_to_recover"] for n, j in settings]
        ax.bar(np.arange(len(settings)) + (k - 1.5) * w, vals, w, color=colors.get(a, GREY), label=a)
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels([f"noise {n}\njitter ±{j}" for n, j in settings], fontsize=9.5)
    ax.set_ylabel("traces needed (lower is better)")
    legend_ink(ax, fontsize=8.5, loc="upper left", ncol=2)
    y = body(fig, "Knowing the ciphertext helps every attack in every setting. When traces are misaligned, the "
                  "neural network keeps working where templates fail.", 0.335, size=13.5, width=50, gap=0.029)
    body(fig, "Caveat: simulated leakage with no countermeasures. Real ML-KEM chips use masking; that is the next "
              "test.", y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    # 8-9 p6 ----------------------------------------------------------------------
    r6 = p6["results"]
    a6 = p6["args"]
    fig = new("03 · Neural net vs. lattice crypto")
    y = title(fig, "Can a neural network crack a lattice secret?")
    y = label(fig, "Why it matters", y - 0.03)
    y = body(fig, "Learning With Errors (LWE) is the maths behind ML-KEM. Machine-learning attacks on it are a new "
                  "research direction worth watching.", y, size=13.5, width=50, gap=0.029)
    y = label(fig, "Method", y - 0.02)
    bullets(fig, [
        f"Deliberately tiny LWE: modulus {a6['q']}, a secret of 0s and 1s with only {a6['h']} ones.",
        "A network learns to predict each noisy equation's answer from its inputs.",
        "A sensitivity test then reveals which secret positions the network relies on.",
        "Baseline: try every possible secret (brute force).",
    ], y, size=13, width=50)
    pages.append(fig)

    fig = new("03 · Result")
    y = title(fig, f"Secret recovered at {sum(r['ml_recovered'] for r in r6)} of {len(r6)} sizes")
    yy = y - 0.05
    for r in r6:
        fig.patches.append(__import__("matplotlib").patches.FancyBboxPatch(
            (0.08, yy - 0.075), 0.84, 0.085, boxstyle="round,pad=0.006,rounding_size=0.012",
            transform=fig.transFigure, fc=PANEL, ec="none"))
        fig.text(0.11, yy - 0.032, f"n = {r['n']}", fontsize=17, color=GOLD, fontweight="bold", va="center")
        fig.text(0.30, yy - 0.020, f"{r['search_space']:,} possible secrets", fontsize=12, color=MUTED, va="center")
        fig.text(0.30, yy - 0.050, f"Neural net: {'recovered' if r['ml_recovered'] else 'failed'} in {r['ml_seconds']:.0f} s"
                 f"   ·   Brute force: {r['bruteforce_seconds']:.2f} s", fontsize=12, color=INK, va="center")
        yy -= 0.105
    y = body(fig, ("The network found the exact secret every time. " if all(r["ml_recovered"] for r in r6) else
                   "The network found the secret in some cases. ") +
             "But at these sizes brute force is far faster, so this shows the method works, not that it is useful yet.",
             yy - 0.02, size=13.5, width=50, gap=0.029)
    body(fig, "Next: grow the problem until one method breaks. Real ML-KEM is far beyond this approach.",
         y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    # 10-13 p7 --------------------------------------------------------------------
    e, em = S["e_hi"], S["e_mid"]
    a = S["acc"]
    fig = new("04 · Adversarial robustness")
    y = title(fig, "Are quantum classifiers harder to fool?")
    y = label(fig, "Why it matters", y - 0.03)
    y = body(fig, "Several papers report that quantum classifiers resist adversarial inputs better than classical "
                  "networks. Classical ML learned the hard way that weak attacks can create a false sense of "
                  "security.", y, size=13.5, width=50, gap=0.029)
    y = label(fig, "Method", y - 0.02)
    bullets(fig, [
        f"{S['runs']} trained models per type: {S['seeds']} seeds at each of "
        f"{', '.join(map(str, S['qubits']))} qubits.",
        "Quantum classifier vs a plain neural network, plus a classical control with the same sine/cosine "
        "input encoding as the quantum model.",
        "Six attacks, from one-step FGSM to 50-step PGD with restarts, a gradient-free attack and transfer.",
    ], y, size=13, width=50)
    pages.append(fig)

    fig = new("04 · Result")
    y = title(fig, f"{a('VQC', 'FGSM', e):.0%} under a weak attack, {a('VQC', 'PGD-50x5', e):.0%} under a strong one")
    ax = chart_axes(fig, [0.12, 0.40, 0.80, 0.29])
    models = (("VQC", VIOLET, "Quantum (VQC)"), ("SinMLP", CYAN, "Classical, same encoding"), ("MLP", GREY, "Classical MLP"))
    w = 0.26
    for k, (m, c, lab) in enumerate(models):
        vals = [a(m, "FGSM", e), a(m, "PGD-50x5", e)]
        xs = np.arange(2) + (k - 1) * w
        ax.bar(xs, vals, w, color=c, label=lab)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 0.015, f"{v:.0%}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(range(2)); ax.set_xticklabels(["Weak attack (FGSM)", "Strong attack (PGD)"], fontsize=11)
    ax.set_ylim(0, 0.75); ax.set_ylabel(f"accuracy under attack (ε={e})")
    legend_ink(ax, fontsize=9, loc="upper right")
    flip = a("VQC", "FGSM", em) > a("MLP", "FGSM", em) and a("VQC", "PGD-50x5", em) < a("MLP", "PGD-50x5", em)
    y = body(fig, "The weak attack makes the quantum model look like the most robust of the three. The strong "
                  "attack defeats it completely.", 0.33, size=13.5, width=50, gap=0.029)
    if flip:
        body(fig, f"At a smaller budget (ε={em}) the ranking flips: under the strong attack the quantum model keeps "
                  f"{a('VQC', 'PGD-50x5', em):.0%} accuracy vs {a('MLP', 'PGD-50x5', em):.0%} for the plain network.",
             y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    fig = new("04 · Why it happens")
    y = title(fig, "One big step overshoots")
    if mech:
        rays = mech["rays"]
        t = np.array(rays["t"])
        ax = chart_axes(fig, [0.14, 0.50, 0.78, 0.26])
        for m, c, lab in (("MLP", GREY, "Classical MLP"), ("SinMLP", CYAN, "Classical, same encoding"),
                          ("VQC", VIOLET, "Quantum (VQC)")):
            r = rays[m]
            ax.fill_between(t, r["q25"], r["q75"], color=c, alpha=0.18)
            ax.plot(t, r["median"], color=c, lw=2.6, label=f"{lab}: overshoots on {r['overshoot_share']:.0%}")
        ax.axvline(1.0, color=INK, ls=":", lw=1.2)
        ax.text(1.02, ax.get_ylim()[1] * 0.9, "full FGSM\nstep", color=INK, fontsize=9, va="top")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("how far the attack pushes (in units of ε)")
        ax.set_ylabel("model confidence\n(relative to clean)")
        legend_ink(ax, fontsize=8.5, loc="lower left")
    y = body(fig, "Angle encoding turns the quantum model into a wave-like (trigonometric) function of its input. "
                  "The chart follows each model's confidence as the attack pushes further (lines: median; bands: "
                  "middle half of inputs). The quantum model's confidence flattens out early and, for many inputs, "
                  "turns back, so a single full-size step lands past the weak spot.", 0.43, size=13,
             width=52, gap=0.028)
    body(fig, f"The gradients are not hidden: the gradient-free attack beat the strong attack in only "
              f"{S['spsa_pairs']} of {S['pairs']} cases, and unlimited attacks leave at most "
              f"{S['unbounded_max']:.0%} accuracy.", y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    fig = new("04 · Testing the cause")
    y = title(fig, "Lower the frequency, lose the illusion")
    if mech:
        me = str(mech["args"]["eps"][-1])
        scales = mech["args"]["scales"]
        ax = chart_axes(fig, [0.14, 0.40, 0.78, 0.29])
        for m, c, lab in (("VQC", VIOLET, "Quantum (VQC)"), ("SinMLP", CYAN, "Classical, same encoding")):
            g = [[cc["eps"][me]["FGSM"] - cc["eps"][me]["PGD-50x5"] for cc in mech["sweep"]
                  if cc["model"] == m and cc["scale"] == s] for s in scales]
            ax.errorbar(scales, [np.mean(v) for v in g], yerr=[np.std(v) for v in g], color=c, marker="o",
                        lw=2.2, capsize=3, label=lab)
        mlp = np.mean([cc["eps"][me]["FGSM"] - cc["eps"][me]["PGD-50x5"] for cc in mech["sweep"] if cc["model"] == "MLP"])
        ax.axhline(mlp, color=GREY, ls="--", lw=1.6, label="Classical MLP")
        ax.set_xlabel("encoding frequency"); ax.set_ylabel("weak-attack illusion\n(FGSM − PGD accuracy)")
        legend_ink(ax, fontsize=9, loc="upper left")
    ratios = list(S["ratio_by_n"].values())
    y = body(fig, "At low encoding frequency the illusion nearly vanishes in the quantum model too. The classical "
                  f"model with the same encoding shows a smaller version ({min(ratios):.0%}–{max(ratios):.0%} of the "
                  "quantum effect). The cause is the encoding meeting a weak attack, not genuine robustness.", 0.33,
             size=13, width=56, gap=0.027)
    body(fig, "Written up as a paper: \"Single-Step Attacks Overstate the Adversarial Robustness of Angle-Encoded "
              "Quantum Classifiers\".", y - 0.008, size=12, color=CYAN, width=60, gap=0.025)
    pages.append(fig)

    # 14-15 p8 --------------------------------------------------------------------
    a8 = p8["args"]
    fig = new("05 · RL for quantum error correction")
    y = title(fig, "Can an AI agent fix quantum errors?")
    y = label(fig, "Why it matters", y - 0.03)
    y = body(fig, "Quantum computers are fragile. Error correction is what makes them useful, and it needs "
                  "decoders that are fast and accurate.", y, size=13.5, width=50, gap=0.029)
    y = label(fig, "Method", y - 0.02)
    bullets(fig, [
        f"A distance-{a8['d']} surface code (9 qubits protecting 1) under random bit-flip errors.",
        "A deep Q-network sees the error syndrome and flips one qubit at a time until it clears.",
        "Rewarded only if the final correction leaves the logical qubit intact.",
        "Compared with the standard decoder, minimum-weight perfect matching (PyMatching).",
    ], y, size=13, width=50)
    pages.append(fig)

    r8 = p8["results"]; ps = a8["p"]
    mw = next(r for r in r8 if r["decoder"].startswith("MWPM"))
    dq = next(r for r in r8 if r["decoder"].startswith("DQN"))
    no = next(r for r in r8 if r["decoder"] == "none")
    fig = new("05 · Result")
    y = title(fig, "It matched the standard decoder")
    ax = chart_axes(fig, [0.14, 0.42, 0.78, 0.28])
    ax.plot(ps, [no[f"p={p}"] for p in ps], color=GREY, marker="o", lw=2, label="No correction")
    ax.plot(ps, [mw[f"p={p}"] for p in ps], color=CYAN, marker="s", lw=2.6, label="Matching decoder")
    ax.plot(ps, [dq[f"p={p}"] for p in ps], color=PINK, marker="^", lw=2, ls="--", label="RL agent (DQN)")
    ax.set_xlabel("physical error rate"); ax.set_ylabel("logical error rate")
    legend_ink(ax, fontsize=9)
    pt = a8["train_p"]
    y = body(fig, f"The lesson was in the failure. A first version saw only the error syndrome and made about three "
                  f"times as many logical errors. Once it could also see its own corrections, it matched the matching "
                  f"decoder ({dq[f'p={pt}']:.1%} vs {mw[f'p={pt}']:.1%} at p = {pt}).", 0.36, size=13, width=52, gap=0.028)
    body(fig, "Caveat: a distance-3 code has only 16 syndromes, so a lookup table is already optimal. "
              "Distance 5 with realistic noise is the real test.", y - 0.01, size=12, color=MUTED, width=56, gap=0.026)
    pages.append(fig)

    # 16 lessons ----------------------------------------------------------------------
    fig = new("Lessons")
    y = title(fig, "Five things I learned")
    y -= 0.03
    for k, (head, text) in enumerate((
            ("Weak tests create fake wins.", "The quantum model looked robust only until the attack got stronger."),
            ("Know what the attacker knows.", "Using the public ciphertext cut the power traces needed by up to "
                                              f"{d['traces_to_recover'] / i['traces_to_recover']:.0f}×."),
            ("Give agents the right view.", "The RL decoder only worked once it could see its own actions."),
            ("Quantum isn't automatically better.", "Same accuracy as classical models, far slower, at today's sizes."),
            ("Toy results need a scaling story.", "Solving small cases proves a method works, not that it matters.")), 1):
        fig.text(0.08, y, f"{k}", fontsize=24, color=CYAN, fontweight="bold", va="top")
        fig.text(0.15, y - 0.004, head, fontsize=14.5, color=INK, fontweight="bold", va="top")
        y = body_at(fig, text, 0.15, y - 0.040, size=12.5, width=48) - 0.022
    pages.append(fig)

    # 17 limits -------------------------------------------------------------------------
    fig = new("Honest limits")
    y = title(fig, "What to keep in mind")
    y -= 0.02
    for head, text in (("Simulated.", "Exact quantum simulators, simulated power traces, synthetic network data in places."),
                       ("Small.", f"{min(S['qubits'])}–{max(S['qubits'])} qubits, toy LWE sizes, a distance-3 code."),
                       ("Early.", "The adversarial study uses many runs; the other four projects are single runs so far."),
                       ("Open.", "All code, results and reports are public, so anyone can check or extend them.")):
        fig.text(0.08, y, head, fontsize=16, color=CYAN, fontweight="bold", va="top")
        y = body(fig, text, y - 0.045, size=13.5, width=50, gap=0.029) - 0.03
    pages.append(fig)

    # 18 next ---------------------------------------------------------------------------
    fig = new("What's next")
    y = title(fig, "Where this goes next")
    bullets(fig, [
        "Submit the adversarial-robustness paper as an arXiv preprint.",
        "Re-run the other four projects over several seeds.",
        "Intrusion detection on the real UNSW-NB15 dataset.",
        "Attack a masked ML-KEM implementation, then real hardware traces.",
        "Distance-5 error correction with realistic noise.",
        "Find where neural LWE attacks overtake brute force and lattice reduction.",
    ], y - 0.04, size=14, width=46, gap=0.032)
    pages.append(fig)

    # 19 CTA ----------------------------------------------------------------------------
    fig = new()
    draw_mark(fig.add_axes([0.36, 0.62, 0.28, 0.224]))
    fig.text(0.5, 0.55, "Code, data, report and paper", fontsize=20, fontweight="bold", ha="center")
    if url:
        fig.text(0.5, 0.50, url, fontsize=15, ha="center", color=CYAN)
    fig.text(0.5, 0.42, "Questions, critiques and collaborators welcome.", fontsize=14, ha="center")
    fig.text(0.5, 0.38, "Comment or message me.", fontsize=14, ha="center", color=MUTED)
    fig.text(0.5, 0.26, "Created with Claude Code", fontsize=12, ha="center", color=MUTED, style="italic")
    pages.append(fig)

    assert len(pages) == N, f"slide count {len(pages)} != {N}"
    images = []
    for f in pages:
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=DPI, facecolor=NAVY)
        plt.close(f)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    images[0].save(OUT, save_all=True, append_images=images[1:], resolution=DPI)
    return OUT


def body_at(fig, text, x, y, size=13, width=48, color=MUTED, gap=0.027):
    for line in textwrap.wrap(text, width):
        fig.text(x, y, line, fontsize=size, color=color, va="top")
        y -= gap
    return y


if __name__ == "__main__":
    print(f"Wrote {build().relative_to(ROOT)}")
