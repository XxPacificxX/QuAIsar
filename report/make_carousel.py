"""Build a LinkedIn carousel (PDF, 1080x1350 per slide) from results/.

LinkedIn shows an uploaded PDF document as swipeable slides. Every number is
read from the results files, so re-running a project and this script keeps
the carousel in sync.

    python report/make_carousel.py      # report/QuAIsar-linkedin-carousel.pdf
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "QuAIsar-linkedin-carousel.pdf"
sys.path.insert(0, str(ROOT / "assets"))
from make_logo import FONT, draw_mark  # noqa: E402

NAVY, PANEL, INK, MUTED = "#0B1030", "#141A45", "#F1F5F9", "#94A3B8"
CYAN, VIOLET, PINK, GOLD, GREY = "#22D3EE", "#8B5CF6", "#F472B6", "#FDE68A", "#64748B"
W, H = 7.2, 9.0  # inches: LinkedIn's 4:5 portrait (1080 x 1350 at 150 dpi)
DPI = 200        # rendered at 1440 x 1800 so text stays sharp on high-density screens
plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#334155"})


def load(name):
    return json.loads((RES / name / "results.json").read_text())


def repo_url():
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return url.removesuffix(".git").replace("https://", "")


# ---------------------------------------------------------------- slide scaffolding
def slide(page, total, tag=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    # thin gradient bar at the top
    bar = fig.add_axes([0, 0.985, 1, 0.015]); bar.axis("off")
    bar.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
               cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", [CYAN, VIOLET, PINK]))
    if tag:
        fig.text(0.08, 0.925, tag.upper(), fontsize=11, color=CYAN, fontweight="bold")
    mark = fig.add_axes([0.075, 0.025, 0.045, 0.036]); draw_mark(mark)
    fig.text(0.13, 0.037, "QuAIsar", fontsize=10, color=INK, fontweight="bold", va="center")
    fig.text(0.13, 0.019, "Created with Claude Code", fontsize=7.5, color=MUTED, va="center", style="italic")
    fig.text(0.92, 0.037, f"{page} / {total}", fontsize=10, color=MUTED, ha="right", va="center")
    return fig


def title(fig, text, y=0.86, size=27):
    lines = textwrap.wrap(text, 24)
    for i, line in enumerate(lines):
        fig.text(0.08, y - i * 0.052, line, fontsize=size, fontweight="bold", color=INK, va="top")
    return y - len(lines) * 0.052


def body(fig, text, y, size=14, width=46, color=INK, gap=0.031):
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width) or [""]:
            fig.text(0.08, y, line, fontsize=size, color=color, va="top")
            y -= gap
        y -= gap * 0.35
    return y


def big_number(fig, value, label, y, color=CYAN):
    fig.text(0.08, y, value, fontsize=64, fontweight="bold", color=color, va="top")
    fig.text(0.08, y - 0.095, label, fontsize=14, color=MUTED, va="top")


def chart_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#334155", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return ax


def bars(ax, labels, values, colors, fmt="{:.0%}", ymax=1.0):
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.62)
    for xi, v in zip(x, values):
        ax.text(xi, v + ymax * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=12, color=INK,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, ymax * 1.15)


# ---------------------------------------------------------------- data
def p7_numbers():
    """Pooled VQC/MLP accuracy under FGSM and PGD-50x5 from the multi-seed masking study,
    falling back to the single-seed demo results."""
    study = sorted((RES / "p7_masking_study").glob("raw_q*.json"))
    if study:
        cells = [c for f in study for c in json.loads(f.read_text())["cells"].values()]
        eps = json.loads(study[0].read_text())["args"]["eps"]
        e = str(eps[-1])
        get = lambda m, a: float(np.mean([c[m]["attacks"][a][e] for c in cells]))
        return {"eps": eps[-1], "runs": len(cells), "vqc_f": get("VQC", "FGSM"), "vqc_p": get("VQC", "PGD-50x5"),
                "mlp_f": get("MLP", "FGSM"), "mlp_p": get("MLP", "PGD-50x5"), "multi": True}
    r = load("p7_adversarial_qml")["results"]
    eps = load("p7_adversarial_qml")["args"]["eps"][-1]
    pick = lambda m, a: next(x for x in r if x["model"] == m and x["attack"] == a)[f"eps={eps}"]
    return {"eps": eps, "runs": 1, "vqc_f": pick("VQC", "FGSM"), "vqc_p": pick("VQC", "PGD-10"),
            "mlp_f": pick("MLP", "FGSM"), "mlp_p": pick("MLP", "PGD-10"), "multi": False}


# ---------------------------------------------------------------- slides
def build():
    p4, p5, p6, p8 = (load(p) for p in ("p4_qml_ids", "p5_pqc_sidechannel", "p6_lwe_ml_attack", "p8_rl_quantum"))
    p7 = p7_numbers()
    url = repo_url()
    N = 9
    pages = []

    # 1 cover
    fig = slide(1, N)
    draw_mark(fig.add_axes([0.28, 0.50, 0.44, 0.352]))
    fig.text(0.5, 0.43, "QuAIsar", fontsize=50, fontweight="bold", ha="center", color=INK)
    fig.text(0.5, 0.375, "5 experiments where quantum computing,", fontsize=16, ha="center", color=INK)
    fig.text(0.5, 0.345, "cybersecurity and AI meet", fontsize=16, ha="center", color=INK)
    fig.text(0.5, 0.25, "What I learned building them  →", fontsize=15, ha="center", color=CYAN, fontweight="bold")
    fig.text(0.5, 0.20, "Created with Claude Code", fontsize=11, ha="center", color=MUTED, style="italic")
    pages.append(fig)

    # 2 the idea
    fig = slide(2, N, "The idea")
    y = title(fig, "One question per experiment. Laptop-sized. Fully reproducible.")
    items = [("01", "Quantum classifiers for intrusion detection"),
             ("02", "AI side-channel attack on post-quantum crypto"),
             ("03", "Neural network vs. a lattice-crypto secret"),
             ("04", "Are quantum classifiers robust to adversarial attacks?"),
             ("05", "Reinforcement learning that corrects quantum errors")]
    y -= 0.05
    for num, text in items:
        fig.patches.append(FancyBboxPatch((0.08, y - 0.058), 0.84, 0.066, boxstyle="round,pad=0.006,rounding_size=0.012",
                                          transform=fig.transFigure, fc=PANEL, ec="none"))
        fig.text(0.11, y - 0.024, num, fontsize=17, color=CYAN, fontweight="bold", va="center")
        fig.text(0.21, y - 0.024, text, fontsize=13, color=INK, va="center")
        y -= 0.084
    body(fig, "Every script runs on a CPU in minutes and saves its results. The reports are generated "
              "from those files.", y - 0.005, size=12.5, color=MUTED, width=54, gap=0.027)
    pages.append(fig)

    # 3 p4
    r4 = p4["results"]
    qk = next(r for r in r4 if r["model"] == "Quantum-kernel SVM")
    cl = [r for r in r4 if r["model"] not in ("Quantum-kernel SVM", "VQC")]
    best = max(cl, key=lambda r: r["accuracy"])
    slow = qk["seconds"] / max(min(r["seconds"] for r in cl), 1e-3)
    fig = slide(3, N, "01 · Quantum intrusion detection")
    y = title(fig, "Quantum classifiers: same accuracy, far slower")
    big_number(fig, f"{slow:,.0f}×", "slower than the fastest classical model, for the same accuracy", y - 0.03)
    ax = chart_axes(fig, [0.12, 0.2, 0.78, 0.3])
    show = [best, next(r for r in r4 if r["model"] == "VQC"), qk]
    bars(ax, [best["model"].replace("Regression", "Reg."), "Quantum VQC", "Quantum kernel"],
         [r["accuracy"] for r in show], [GREY, VIOLET, CYAN])
    ax.set_ylabel("test accuracy")
    fig.text(0.08, 0.135, "At 4 qubits there's no quantum advantage to find (synthetic data).", fontsize=11.5, color=MUTED)
    pages.append(fig)

    # 4 p5
    r5 = p5["results"]
    pairs = [(d, next(i for i in r5 if i["noise"] == d["noise"] and i["jitter"] == d["jitter"]
                      and i["attack"] == d["attack"].replace("direct", "intermediate")))
             for d in r5 if d["attack"].endswith("direct")]
    d, i = max(pairs, key=lambda p: p[0]["traces_to_recover"] / p[1]["traces_to_recover"])
    fig = slide(4, N, "02 · Side-channel attack on ML-KEM")
    y = title(fig, "Using the ciphertext makes the attack far stronger")
    big_number(fig, f"{d['traces_to_recover'] / i['traces_to_recover']:.0f}× fewer",
               "power traces needed to recover a secret key coefficient", y - 0.03, color=PINK)
    ax = chart_axes(fig, [0.12, 0.2, 0.78, 0.3])
    bars(ax, ["Direct attack", "Intermediate-value attack"], [d["traces_to_recover"], i["traces_to_recover"]],
         [GREY, PINK], fmt="{:.1f}", ymax=max(d["traces_to_recover"], 1))
    ax.set_ylabel("traces needed (lower is better)")
    fig.text(0.08, 0.135, f"{d['attack'].split()[0]} model, noise σ={d['noise']}, jitter ±{d['jitter']}. Simulated traces.",
             fontsize=11.5, color=MUTED)
    pages.append(fig)

    # 5 p6
    r6 = p6["results"]
    fig = slide(5, N, "03 · Neural net vs. lattice crypto")
    y = title(fig, "A neural network recovered the LWE secret")
    big_number(fig, f"{sum(r['ml_recovered'] for r in r6)} / {len(r6)}",
               "toy problem sizes solved (n = " + ", ".join(str(r["n"]) for r in r6) + ")", y - 0.03, color=GOLD)
    y = body(fig, "The model learns to predict the noisy dot product, then a sensitivity test reveals which "
                  "secret coordinates matter.", y - 0.2, size=14)
    body(fig, "Honest caveat: at these toy sizes brute force is still faster, and real ML-KEM is far out of reach. "
              "The interesting question is where the two curves cross.", y - 0.02, size=13, color=MUTED)
    pages.append(fig)

    # 6 p7 (headline)
    fig = slide(6, N, "04 · Adversarial robustness")
    y = title(fig, "Quantum classifiers only look robust")
    ax = chart_axes(fig, [0.12, 0.42, 0.78, 0.29])
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [p7["vqc_f"], p7["vqc_p"]], w, color=VIOLET, label="Quantum (VQC)")
    ax.bar(x + w / 2, [p7["mlp_f"], p7["mlp_p"]], w, color=GREY, label="Classical (MLP)")
    for xi, v in zip(list(x - w / 2) + list(x + w / 2), [p7["vqc_f"], p7["vqc_p"], p7["mlp_f"], p7["mlp_p"]]):
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["Weak attack (FGSM)", "Strong attack (PGD)"], fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_ylabel(f"accuracy under attack (ε={p7['eps']})")
    leg = ax.legend(fontsize=10, frameon=False, loc="upper right")
    for t in leg.get_texts(): t.set_color(INK)
    y = body(fig, "A weak attack makes the quantum model look far more robust. A strong attack breaks it completely.",
             0.365, size=13.5, width=50)
    body(fig, "Why: angle encoding turns the model into a wave-like function, so a single attack step overshoots. "
              "Lower the encoding frequency and the effect nearly vanishes; a classical network with the same "
              "sine/cosine features shows a smaller version of it."
              + (f" ({p7['runs']} runs per model.)" if p7["multi"] else ""), y - 0.01, size=12, color=MUTED,
         width=58, gap=0.026)
    pages.append(fig)

    # 7 p8
    r8 = p8["results"]
    ps = p8["args"]["p"]
    mw = next(r for r in r8 if r["decoder"].startswith("MWPM"))
    dq = next(r for r in r8 if r["decoder"].startswith("DQN"))
    no = next(r for r in r8 if r["decoder"] == "none")
    fig = slide(7, N, "05 · RL for quantum error correction")
    y = title(fig, "An RL agent matched the standard decoder")
    ax = chart_axes(fig, [0.14, 0.33, 0.76, 0.36])
    ax.plot(ps, [no[f"p={p}"] for p in ps], color=GREY, marker="o", lw=2, label="No correction")
    ax.plot(ps, [mw[f"p={p}"] for p in ps], color=CYAN, marker="s", lw=2.5, label="Matching decoder")
    ax.plot(ps, [dq[f"p={p}"] for p in ps], color=PINK, marker="^", lw=2, ls="--", label="RL agent (DQN)")
    ax.set_xlabel("physical error rate"); ax.set_ylabel("logical error rate")
    leg = ax.legend(fontsize=10, frameon=False)
    for t in leg.get_texts(): t.set_color(INK)
    body(fig, "The lesson: the first version lost badly. It only worked once the agent could see its own "
              "corrections, not just the error syndrome.", 0.27, size=13)
    fig.text(0.08, 0.135, f"Distance-{p8['args']['d']} surface code, bit-flip noise.", fontsize=11.5, color=MUTED)
    pages.append(fig)

    # 8 caveats / lessons
    fig = slide(8, N, "What I'd tell you before you trust this")
    y = title(fig, "Honest limits")
    for head, text in (("Simulated", "Exact quantum simulators, simulated power traces, and synthetic data in places."),
                       ("Small", "4–8 qubits, toy LWE sizes, a distance-3 code."),
                       ("Early", "Most results are single-seed. The adversarial study uses multiple seeds."),
                       ("Open", "All code and results are public, so anyone can check or extend them.")):
        fig.text(0.08, y - 0.02, head, fontsize=17, color=CYAN, fontweight="bold", va="top")
        y = body(fig, text, y - 0.065, size=13.5) - 0.025
    pages.append(fig)

    # 9 CTA
    fig = slide(9, N)
    draw_mark(fig.add_axes([0.36, 0.60, 0.28, 0.224]))
    fig.text(0.5, 0.52, "Code, results and full report", fontsize=20, fontweight="bold", ha="center")
    if url:
        fig.text(0.5, 0.465, url, fontsize=15, ha="center", color=CYAN)
    fig.text(0.5, 0.39, "A paper on the adversarial-robustness result", fontsize=14, ha="center", color=INK)
    fig.text(0.5, 0.36, "is on the way.", fontsize=14, ha="center", color=INK)
    fig.text(0.5, 0.27, "Feedback and collaborators welcome.", fontsize=14, ha="center", color=MUTED)
    fig.text(0.5, 0.17, "Created with Claude Code", fontsize=12, ha="center", color=MUTED, style="italic")
    pages.append(fig)

    # Render each slide to PNG and assemble the PDF with Pillow. (matplotlib's own PDF
    # writer needs fontTools' compiled modules, which some locked-down Windows PCs block;
    # LinkedIn shows document pages as images anyway.)
    import io
    from PIL import Image

    images = []
    for f in pages:
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=DPI, facecolor=NAVY)
        plt.close(f)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    images[0].save(OUT, save_all=True, append_images=images[1:], resolution=DPI)
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build().relative_to(ROOT)}")
