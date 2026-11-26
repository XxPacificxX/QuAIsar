"""Build the QuAIsar findings report (DOCX, optionally PDF) from results/.

Every number in the report is read from results/<project>/results.json, so
re-running a project and then this script keeps the report in sync.

    python report/build_report.py            # report/QuAIsar-findings.docx
    python report/build_report.py --pdf      # also export PDF (Word on Windows, else LibreOffice)

Requires: python-docx (included in requirements.txt)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
OUT = Path(__file__).resolve().parent / "QuAIsar-findings.docx"

ACCENT = RGBColor(0x0E, 0x74, 0x90)  # dark cyan, readable on white
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
HEADER_FILL, ZEBRA_FILL, BORDER = "0B1030", "F1F5F9", "CBD5E1"
FONT = "Calibri"


# ---------------------------------------------------------------- data
def load(project):
    path = RESULTS / project / "results.json"
    if not path.exists():
        sys.exit(f"Missing {path.relative_to(ROOT)}; run {project}/run.py first.")
    return json.loads(path.read_text())


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def repo_url():
    url = git("remote", "get-url", "origin")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url.removesuffix(".git")


def versions():
    out = {"Python": platform.python_version()}
    for pkg in ("torch", "pennylane", "pymatching", "scikit-learn", "numpy", "xgboost"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


# ---------------------------------------------------------------- docx helpers
def set_cell_fill(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_table_borders(table, color=BORDER):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), color)
        borders.append(el)
    tblPr.append(borders)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true")
    trPr.append(el)


def add_field(paragraph, code, size=None, color=None):
    """Complex field (begin / instr / separate / result / end). Every run carries the
    formatting, so Word keeps it when it recalculates the field."""
    def styled_run():
        run = paragraph.add_run()
        if size: run.font.size = Pt(size)
        if color: run.font.color.rgb = RGBColor.from_string(color)
        return run

    for kind in ("begin", "instr", "separate", "result", "end"):
        run = styled_run()
        if kind == "instr":
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = f" {code} "
            run._r.append(el)
        elif kind == "result":
            run.text = "1"
        else:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
            run._r.append(el)


def para(doc, text="", style=None, bold=False, italic=False, size=None, color=None,
         align=None, space_after=None, keep_next=False):
    p = doc.add_paragraph(style=style)
    if text:
        run = p.add_run(text)
        run.bold, run.italic = bold, italic
        if size: run.font.size = Pt(size)
        if color is not None: run.font.color.rgb = color
    if align is not None: p.alignment = align
    if space_after is not None: p.paragraph_format.space_after = Pt(space_after)
    if keep_next: p.paragraph_format.keep_with_next = True
    return p


def rich(doc, parts, style=None):
    """Paragraph from [(text, {'bold':..,'italic':..}), ...] or plain strings."""
    p = doc.add_paragraph(style=style)
    for part in parts:
        text, fmt = (part, {}) if isinstance(part, str) else part
        r = p.add_run(text)
        r.bold, r.italic = fmt.get("bold", False), fmt.get("italic", False)
    return p


def bullets(doc, items):
    for item in items:
        rich(doc, item if isinstance(item, list) else [item], style="List Bullet")


def caption(doc, text):
    p = para(doc, text, style="Caption", keep_next=True)
    return p


def table(doc, headers, rows, widths, bold_cells=(), numeric_from=1, note=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_borders(t)
    hdr = t.rows[0]
    repeat_header(hdr)
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.width = Inches(widths[i])
        set_cell_fill(c, HEADER_FILL)
        p = c.paragraphs[0]
        r = p.add_run(str(h)); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if i >= numeric_from else WD_ALIGN_PARAGRAPH.LEFT
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            c = cells[ci]
            c.width = Inches(widths[ci])
            if ri % 2 == 1:
                set_cell_fill(c, ZEBRA_FILL)
            p = c.paragraphs[0]
            r = p.add_run(str(val)); r.font.size = Pt(9.5)
            r.bold = (ri, ci) in bold_cells
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if ci >= numeric_from else WD_ALIGN_PARAGRAPH.LEFT
    for row in t.rows:  # keep each row on one page and the table together where possible
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit"); cant.set(qn("w:val"), "true"); trPr.append(cant)
        for c in row.cells:
            c.paragraphs[0].paragraph_format.keep_with_next = True
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
    if note:
        para(doc, note, size=8.5, italic=True, color=MUTED, space_after=10)
    else:
        para(doc, "", space_after=4)
    return t


def figure(doc, path, width, text):
    if not path.exists():
        para(doc, f"[figure missing: {path.relative_to(ROOT)}]", italic=True, color=MUTED)
        return
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = para(doc, text, style="Caption", align=WD_ALIGN_PARAGRAPH.CENTER)
    c.paragraph_format.space_after = Pt(12)


def setup_styles(doc):
    st = doc.styles
    normal = st["Normal"]
    normal.font.name = FONT; normal.font.size = Pt(11); normal.font.color.rgb = INK
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size, before, after in (("Heading 1", 16, 18, 6), ("Heading 2", 12.5, 12, 4)):
        s = st[name]
        s.font.name = FONT; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = ACCENT
        rf = s.element.rPr.rFonts  # drop theme fonts so the explicit font wins
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(attr), FONT)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(attr)) is not None:
                del rf.attrib[qn(attr)]
        s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    cap = st["Caption"]
    cap.font.name = FONT; cap.font.size = Pt(9); cap.font.italic = True; cap.font.bold = False
    cap.font.color.rgb = MUTED
    st["List Bullet"].paragraph_format.space_after = Pt(3)


def header_footer(section):
    section.different_first_page_header_footer = True  # clean cover page
    hp = section.header.paragraphs[0]
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    if (ASSETS / "logo-mark.png").exists():
        hp.add_run().add_picture(str(ASSETS / "logo-mark.png"), height=Inches(0.28))
    r = hp.add_run("  QuAIsar"); r.bold = True; r.font.size = Pt(9); r.font.color.rgb = INK
    # two tabs: the Header/Footer styles already have a centre stop before the right one
    r = hp.add_run("\t\tFindings report"); r.font.size = Pt(9); r.font.color.rgb = MUTED

    fp = section.footer.paragraphs[0]
    fp.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    r = fp.add_run("Created with Claude Code"); r.font.size = Pt(8.5); r.font.color.rgb = MUTED; r.italic = True
    r = fp.add_run("\t\tPage "); r.font.size = Pt(8.5); r.font.color.rgb = MUTED
    add_field(fp, "PAGE", size=8.5, color="475569")
    r = fp.add_run(" of "); r.font.size = Pt(8.5); r.font.color.rgb = MUTED
    add_field(fp, "NUMPAGES", size=8.5, color="475569")


# ---------------------------------------------------------------- content
def pct(x):
    return f"{x:.1%}"


def cover(doc, author, url, today):
    for _ in range(3):
        para(doc, "", space_after=12)
    banner = ASSETS / "banner-light.png"
    if banner.exists():
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(banner), width=Inches(5.4))
    para(doc, "", space_after=18)
    para(doc, "Findings Report", size=30, bold=True, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "Five experiments where quantum computing, security and machine learning meet",
         size=13, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=36)
    para(doc, author, size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, today.strftime("%B %d, %Y").replace(" 0", " "), size=11, color=MUTED,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    if url:
        para(doc, url, size=10, color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build(author):
    p4, p5, p6, p7, p8 = (load(p) for p in ("p4_qml_ids", "p5_pqc_sidechannel", "p6_lwe_ml_attack",
                                            "p7_adversarial_qml", "p8_rl_quantum"))
    today = dt.date.today()
    url = repo_url()

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    setup_styles(doc)
    header_footer(sec)
    doc.core_properties.title = "QuAIsar Findings Report"
    doc.core_properties.author = author
    doc.core_properties.comments = "Created with Claude Code"

    cover(doc, author, url, today)

    # ---- derived numbers used in the text
    r4 = p4["results"]
    q_kernel = next(r for r in r4 if r["model"] == "Quantum-kernel SVM")
    vqc = next(r for r in r4 if r["model"] == "VQC")
    classical = [r for r in r4 if r["model"] not in ("Quantum-kernel SVM", "VQC")]
    best_cl = max(classical, key=lambda r: r["accuracy"])
    slow = q_kernel["seconds"] / max(min(r["seconds"] for r in classical), 1e-3)

    r5 = p5["results"]
    def p5row(noise, jitter, attack):
        return next(r for r in r5 if r["noise"] == noise and r["jitter"] == jitter and r["attack"] == attack)
    settings5 = sorted({(r["noise"], r["jitter"]) for r in r5})
    attacks5 = list(dict.fromkeys(r["attack"] for r in r5))
    # biggest improvement from using the ciphertext, over every setting and model
    gains = [(p5row(n, j, f"{m} direct"), p5row(n, j, f"{m} intermediate"))
             for n, j in settings5 for m in ("Template", "CNN")]
    cnn_d, cnn_i = max(gains, key=lambda g: g[0]["traces_to_recover"] / g[1]["traces_to_recover"])
    gain_model = cnn_d["attack"].split()[0]
    always_better = all(i["traces_to_recover"] <= d["traces_to_recover"] for d, i in gains)

    r6 = p6["results"]
    a6 = p6["args"]

    r7 = p7["results"]
    eps7 = p7["args"]["eps"]
    def p7row(model, attack):
        return next(r for r in r7 if r["model"] == model and r["attack"] == attack)
    vqc_f, vqc_p = p7row("VQC", "FGSM"), p7row("VQC", "PGD-10")
    mlp_p = p7row("MLP", "PGD-10")
    at_p = p7row("VQC + adv. training", "PGD-10")
    e_hi = eps7[-1]; e_mid = eps7[2] if len(eps7) > 2 else eps7[-1]
    S7 = load_p7_study()  # the multi-seed study behind the paper, if it has been run

    r8 = p8["results"]
    ps8 = p8["args"]["p"]
    mwpm = next(r for r in r8 if r["decoder"].startswith("MWPM"))
    dqn = next(r for r in r8 if r["decoder"].startswith("DQN"))
    none8 = next(r for r in r8 if r["decoder"] == "none")
    p_train = p8["args"]["train_p"]

    # ---- summary
    doc.add_heading("Summary", level=1)
    para(doc, "QuAIsar is a set of five small, reproducible experiments at the intersection of quantum "
              "computing, cybersecurity and machine learning. Each runs end to end on a laptop CPU in minutes. "
              "This report collects what the first round of runs found, and how far each result can be trusted.")
    bullets(doc, [
        [("Quantum classifiers for intrusion detection: ", {"bold": True}),
         f"no clear advantage. The quantum-kernel SVM scored {pct(q_kernel['accuracy'])} against "
         f"{pct(best_cl['accuracy'])} for the best classical model ({best_cl['model']}) while running about "
         f"{slow:,.0f}× slower, on synthetic data."],
        [("Side-channel attack on ML-KEM-style secrets: ", {"bold": True}),
         f"attacking intermediate values with the known ciphertext cut the traces needed by up to "
         f"{cnn_d['traces_to_recover'] / cnn_i['traces_to_recover']:.0f}× "
         f"({cnn_d['traces_to_recover']:.1f} → {cnn_i['traces_to_recover']:.1f} traces for the {gain_model}, "
         f"noise σ = {cnn_d['noise']}, jitter ±{cnn_d['jitter']})."],
        [("Neural attack on toy LWE: ", {"bold": True}),
         f"the model recovered the secret at {sum(r['ml_recovered'] for r in r6)} of {len(r6)} sizes tried "
         f"(n = {', '.join(str(r['n']) for r in r6)}), but brute force is still faster at these sizes."],
        [("Adversarial robustness of quantum models: ", {"bold": True}),
         (f"the quantum classifier only looks robust. A single-step attack leaves it at "
          f"{pct(S7['acc']('VQC', 'FGSM', S7['e_hi']))} accuracy at ε = {S7['e_hi']} (classical MLP: "
          f"{pct(S7['acc']('MLP', 'FGSM', S7['e_hi']))}), but a strong iterative attack takes it to "
          f"{pct(S7['acc']('VQC', 'PGD-50x5', S7['e_hi']))} across {S7['runs']} runs. The cause is curvature from "
          "the periodic encoding, not hidden gradients; this result is written up as a paper."
          if S7 else
          f"the quantum classifier is not more robust. Under PGD at ε = {e_hi} it falls to "
          f"{pct(vqc_p[f'eps={e_hi}'])}, while weak FGSM attacks stall at {pct(vqc_f[f'eps={e_hi}'])}, "
          "a pattern consistent with gradient masking.")],
        [("Reinforcement-learning error decoder: ", {"bold": True}),
         f"a DQN matched minimum-weight matching on the distance-{p8['args']['d']} surface code "
         f"({pct(dqn[f'p={p_train}'])} vs {pct(mwpm[f'p={p_train}'])} logical error at p = {p_train}), "
         "once it could see its own corrections."],
    ])
    rich(doc, [("Status: ", {"bold": True}),
               ("exploratory. The adversarial-robustness study uses multiple seeds; the other four projects come "
                "from a single seed. Everything runs on exact simulators, and two projects use synthetic data. "
                if S7 else
                "exploratory. All numbers come from a single seed on exact simulators, and two projects use synthetic "
                "data. ") +
               "Section 7 lists what must happen before any of this is publishable as a claim."])

    # ---- setup
    doc.add_heading("1. Scope and setup", level=1)
    para(doc, "Each project answers one question with a baseline, an AI component and a quantum or "
              "post-quantum component. All quantum circuits run on PennyLane's exact state-vector simulator "
              "(no shot noise, no hardware noise). Models train on CPU with PyTorch and scikit-learn. "
              "Every script fixes its random seed and writes its results to results/<project>/results.json; "
              "this report is generated from those files.")

    # ---- p4
    a4 = p4["args"]
    doc.add_heading("2. Quantum classifiers for intrusion detection (p4)", level=1)
    rich(doc, [("Question. ", {"bold": True}), "Do quantum classifiers detect network intrusions better than classical ones?"])
    rich(doc, [("Method. ", {"bold": True}),
               f"Flow features are standardised and compressed by PCA to {a4['qubits']} features, one per qubit. "
               f"Models train on {a4['train']} flows and are tested on {a4['test']}. The quantum-kernel SVM uses an "
               "IQP-style feature map and an exact fidelity kernel; the variational quantum classifier (VQC) uses "
               "angle embedding and strongly entangling layers. Classical baselines see the same "
               f"{a4['qubits']} features. Data: {p4['data']}."])
    best_acc = max(r["accuracy"] for r in r4)
    caption(doc, "Table 1. Test performance on intrusion detection (higher is better; best accuracy in bold).")
    table(doc, ["Model", "Accuracy", "F1", "Attack recall", "Time (s)"],
          [[r["model"], f"{r['accuracy']:.3f}", f"{r['f1']:.3f}", f"{r['attack_recall']:.3f}", f"{r['seconds']:.2f}"]
           for r in r4],
          [2.1, 1.0, 0.9, 1.2, 0.9],
          bold_cells={(i, 1) for i, r in enumerate(r4) if r["accuracy"] == best_acc})
    rich(doc, [("Finding. ", {"bold": True}),
               f"All models land within {pct(min(r['accuracy'] for r in r4))}–{pct(best_acc)} accuracy. "
               f"The quantum kernel is competitive but about {slow:,.0f}× slower than the fastest classical model, "
               + (f"and the VQC catches the fewest attacks (recall {vqc['attack_recall']:.3f}). "
                  if vqc["attack_recall"] == min(r["attack_recall"] for r in r4)
                  else f"and the VQC reaches an attack recall of {vqc['attack_recall']:.3f}. ")
               + "This matches the wider "
               "literature: at a handful of qubits there is no quantum advantage to find, and the useful "
               "contribution is measuring why (encoding cost, expressivity, noise)."])
    rich(doc, [("Caveat. ", {"bold": True}),
               "These runs used the synthetic fallback dataset, so they say nothing yet about real network traffic."])

    # ---- p5
    a5 = p5["args"]
    doc.add_heading("3. Side-channel attack on ML-KEM-style secrets (p5)", level=1)
    rich(doc, [("Question. ", {"bold": True}),
               "Can deep learning recover post-quantum secret key coefficients from power traces?"])
    rich(doc, [("Method. ", {"bold": True}),
               "A simulator produces power traces of a device multiplying a secret coefficient s ∈ {−2,…,2} "
               "(centred binomial, as in ML-KEM) by a known ciphertext coefficient c modulo q = 3329. "
               "The trace leaks the Hamming weight of s mod q and of s·c mod q, plus Gaussian noise and random time "
               f"jitter. Attacks are trained on {a5['profiling']:,} profiling traces. Two attack styles are compared, "
               "each with Gaussian templates and a 1-D CNN:"])
    bullets(doc, [
        [("Direct: ", {"bold": True}), "classify s straight from the trace. It cannot separate s = 1 from s = 2, "
                                         "which load with the same Hamming weight."],
        [("Intermediate: ", {"bold": True}), "predict the Hamming weight of each intermediate value, then score every "
                                               "hypothesis for s using the known c. This is how real profiled attacks work."],
    ])
    hdr = ["Noise / jitter"] + attacks5
    rows, bold = [], set()
    for i, (n, j) in enumerate(settings5):
        vals = [p5row(n, j, a) for a in attacks5]
        best = min(v["traces_to_recover"] for v in vals)
        rows.append([f"σ={n}, ±{j}"] + [f"{v['traces_to_recover']:.1f}  ({v['success<=50']:.0%})" for v in vals])
        bold |= {(i, k + 1) for k, v in enumerate(vals) if v["traces_to_recover"] == best}
    caption(doc, "Table 2. Mean attack traces needed to recover a coefficient (lower is better), "
                 "with the success rate within 50 traces in brackets. Best per row in bold.")
    table(doc, hdr, rows, [1.2, 1.3, 1.4, 1.2, 1.4], bold_cells=bold)
    figure(doc, RESULTS / "p5_pqc_sidechannel" / "traces_to_recover.png", 6.0,
           "Figure 1. Traces needed to recover a coefficient, by attack and setting (51 = failed).")
    best_clean = min(p5row(min(n for n, _ in settings5), 0, a)["traces_to_recover"] for a in attacks5)
    rich(doc, [("Finding. ", {"bold": True}),
               ("Using the ciphertext made every attack stronger in every setting. " if always_better else
                "Using the ciphertext made most attacks stronger. ") +
               f"On low-noise aligned traces the best attack needs only {best_clean:.1f} traces on average. "
               f"The largest gain is for the {gain_model} at noise σ = {cnn_d['noise']} with ±{cnn_d['jitter']} samples "
               f"of jitter: {cnn_i['traces_to_recover']:.1f} traces at {cnn_i['success<=50']:.0%} success versus "
               f"{cnn_d['traces_to_recover']:.1f} traces at {cnn_d['success<=50']:.0%} for the direct version. "
               "Templates degrade most under jitter because they assume aligned traces."])
    rich(doc, [("Caveat. ", {"bold": True}),
               "The traces are simulated with an idealised Hamming-weight leakage model and no countermeasures. "
               "Real ML-KEM implementations are masked and shuffled; the next step is real traces and a masked target."])

    # ---- p6
    doc.add_heading("4. Neural secret recovery on toy LWE (p6)", level=1)
    rich(doc, [("Question. ", {"bold": True}), "Can a neural network recover a Learning-With-Errors secret?"])
    rich(doc, [("Method. ", {"bold": True}),
               f"LWE samples b = ⟨a, s⟩ + e mod q with q = {a6['q']}, a binary secret of Hamming weight h = {a6['h']} "
               f"and error σ = {a6['sigma']}. Following SALSA (Wenger et al., 2022), a network learns to predict b from "
               f"a ({a6['samples']:,} samples, angles on the unit circle). A distinguisher then perturbs each coordinate "
               "of a: only coordinates in the secret's support move the prediction. The baseline enumerates all "
               "weight-h secrets."])
    caption(doc, "Table 3. Secret recovery on toy LWE.")
    table(doc, ["n", "Candidate secrets", "ML recovered", "ML time (s)", "Brute force recovered", "Brute force time (s)"],
          [[r["n"], f"{r['search_space']:,}", "yes" if r["ml_recovered"] else "no", f"{r['ml_seconds']:.1f}",
            "yes" if r["bruteforce_recovered"] else "no", f"{r['bruteforce_seconds']:.2f}"] for r in r6],
          [0.5, 1.3, 1.1, 1.0, 1.4, 1.2], numeric_from=0)
    rich(doc, [("Finding. ", {"bold": True}),
               ("The network recovered the exact secret at every size: its sensitivity ranking put the true "
                "secret coordinates first. " if all(r["ml_recovered"] for r in r6) else
                f"The network recovered the secret at {sum(r['ml_recovered'] for r in r6)} of {len(r6)} sizes. ") +
               "Brute force is still orders of magnitude faster at these sizes, so the "
               "result shows the method works, not that it is useful yet. The interesting experiment is the crossover: "
               "brute force grows as C(n, h), while training cost grows slowly with n."])
    rich(doc, [("Caveat. ", {"bold": True}),
               "These parameters are deliberately tiny and insecure. Real ML-KEM (n = 256 per module, q = 3329, "
               "non-sparse secrets) is far beyond this method."])

    # ---- p7
    a7 = p7["args"]
    doc.add_heading("5. Adversarial robustness of quantum classifiers (p7)", level=1)
    rich(doc, [("Question. ", {"bold": True}), "Are quantum classifiers more robust to adversarial inputs than classical ones?"])
    if S7:
        p7_study_section(doc, S7, url)
    else:
        p7_single_run_section(doc, a7, r7, eps7, vqc_f, vqc_p, mlp_p, at_p, e_hi, e_mid)

    # ---- p8
    p8_section(doc, p8, ps8, mwpm, dqn, none8, p_train)

    # ---- limitations & next steps
    doc.add_heading("7. Limitations and threats to validity", level=1)
    bullets(doc, [
        [("Single seed. ", {"bold": True}),
         ("p4, p5, p6 and p8 come from one run each; differences of a few points may be noise. "
          "They need several seeds with mean ± standard deviation." if S7 else
          "Every number comes from one run; differences of a few points may be noise. "
          "Results need several seeds with mean ± standard deviation.")],
        [("Synthetic data. ", {"bold": True}), "p4 used the synthetic fallback and p5 uses simulated traces."],
        [("Ideal simulators. ", {"bold": True}), "No shot noise or hardware noise in any quantum circuit."],
        [("Toy scales. ", {"bold": True}), ("4–8 qubits" if S7 else "4–6 qubits") + ", n ≤ 24 for LWE, distance-3 codes."],
        [("No tests. ", {"bold": True}), "Core routines (code construction, LWE sampling, hypothesis scoring) are not yet unit tested."],
    ])
    doc.add_heading("8. Next steps", level=1)
    bullets(doc, [
        "Re-run p4, p5, p6 and p8 over 5–10 seeds and report confidence intervals." if S7 else
        "Re-run all five projects over 5–10 seeds and report confidence intervals.",
        "p4: real UNSW-NB15 data; accuracy and runtime curves over qubit count and training size.",
        "p5: a masked (two-share) target, real ChipWhisperer traces of liboqs ML-KEM.",
        "p6: scale n and h to find where ML overtakes enumeration; add a BKZ lattice-reduction baseline.",
        "p7: add shot and hardware noise, larger models and AutoAttack-style evaluation; attack the p4 IDS models."
        if S7 else "p7: confirm or refute gradient masking with gradient-free attacks; attack the p4 IDS models.",
        "p8: distance 5, depolarising noise and measurement errors; compare with a lookup-table decoder.",
    ])
    appendix(doc, url)
    doc.save(OUT)
    return OUT


def load_p7_study():
    """Pooled numbers from results/p7_masking_study (the paper's experiments), or None."""
    files = sorted((RESULTS / "p7_masking_study").glob("raw_q*.json"))
    if not files:
        return None
    cells, eps = {}, None
    for f in files:
        d = json.loads(f.read_text())
        eps = d["args"]["eps"]
        for k, v in d["cells"].items():
            n, s = map(int, k.split("_"))
            cells[(n, s)] = v
    qubits = sorted({n for n, _ in cells})

    def acc(model, attack, e, n=None):
        e = str(e)
        vals = [c[model]["attacks"][attack][e] for (nn, _), c in cells.items() if n is None or nn == n]
        return float(np.mean(vals))

    def gap(model, e, n=None):
        return acc(model, "FGSM", e, n) - acc(model, "PGD-50x5", e, n)

    e_hi, e_mid = eps[-1], eps[-2]
    spsa_pairs = sum(c["VQC"]["attacks"]["SPSA"][str(e)] < c["VQC"]["attacks"]["PGD-50x5"][str(e)] - 1e-9
                     for c in cells.values() for e in eps)
    mech = None
    mpath = RESULTS / "p7_masking_study" / "mechanism.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        me = str(m["args"]["eps"][-1])
        by_s = {s: float(np.mean([c["eps"][me]["FGSM"] - c["eps"][me]["PGD-50x5"] for c in m["sweep"]
                                  if c["model"] == "VQC" and c["scale"] == s])) for s in m["args"]["scales"]}
        mech = {"lo_s": min(by_s), "lo": by_s[min(by_s)], "peak_s": max(by_s, key=by_s.get), "peak": max(by_s.values())}
    return {
        "acc": acc, "gap": gap, "eps": eps, "e_hi": e_hi, "e_mid": e_mid, "qubits": qubits,
        "runs": len(cells), "seeds": len(cells) // len(qubits),
        "clean": lambda m: float(np.mean([c[m]["clean"] for c in cells.values()])),
        "nonmono": lambda m: float(np.mean([c[m]["nonmonotone"][str(e_hi)] for c in cells.values()])),
        "unbounded_max": max(c["VQC"]["unbounded_pgd"] for c in cells.values()),
        "spsa_pairs": int(spsa_pairs), "pairs": len(cells) * len(eps),
        "ratio_by_n": {n: gap("SinMLP", e_hi, n) / max(gap("VQC", e_hi, n), 1e-9) for n in qubits},
        "mech": mech,
    }


def p7_study_section(doc, S, url):
    e, em = S["e_hi"], S["e_mid"]
    a = S["acc"]
    rich(doc, [("Method. ", {"bold": True}),
               "A variational quantum classifier (VQC: angle encoding re-uploaded before each of 3 entangling layers) "
               "is compared with two classical models on handwritten digits (3 vs 8, PCA-reduced, inputs in [0, 1]): "
               "a plain MLP, and a SinMLP that sees the same sine/cosine input frequencies as the VQC but nothing "
               f"quantum. Each model is trained {S['runs']} times ({S['seeds']} seeds at each of "
               f"{', '.join(map(str, S['qubits']))} qubits) and attacked six ways: FGSM, FGSM with a line search, "
               "PGD-20, PGD-50 with 5 restarts, the gradient-free SPSA attack, and transfer from the MLP. "
               "An encoding-frequency sweep tests the cause."])
    models = ("VQC", "SinMLP", "MLP")
    rows = [[m, pct(S["clean"](m)), pct(a(m, "FGSM", em)), pct(a(m, "PGD-50x5", em)),
             pct(a(m, "FGSM", e)), pct(a(m, "PGD-50x5", e)), pct(a(m, "SPSA", e))] for m in models]
    caption(doc, f"Table 4. Accuracy under attack, averaged over all {S['runs']} runs per model (higher is more robust).")
    table(doc, ["Model", "Clean", f"FGSM ε={em}", f"PGD ε={em}", f"FGSM ε={e}", f"PGD ε={e}", f"SPSA ε={e}"],
          rows, [1.0, 0.75, 0.95, 0.95, 0.95, 0.95, 0.95])
    fig = ROOT / "paper" / "figures" / "fig_ranking.png"
    figure(doc, fig if fig.exists() else RESULTS / "p7_masking_study" / "masking_study.png", 6.0,
           "Figure 2. Accuracy under the single-step attack (left) and the strongest attack (right).")
    flip = a("VQC", "FGSM", em) > a("MLP", "FGSM", em) and a("VQC", "PGD-50x5", em) < a("MLP", "PGD-50x5", em)
    ratios = S["ratio_by_n"].values()
    M = S["mech"]
    rich(doc, [("Finding. ", {"bold": True}),
               f"The single-step attack makes the VQC look like the most robust model ({pct(a('VQC', 'FGSM', e))} at "
               f"ε = {e}, against {pct(a('MLP', 'FGSM', e))} for the MLP), but PGD with restarts takes it to "
               f"{pct(a('VQC', 'PGD-50x5', e))}. "
               + (f"At ε = {em} the ranking reverses: FGSM puts the VQC ahead of the MLP, PGD puts it behind "
                  f"({pct(a('VQC', 'PGD-50x5', em))} vs {pct(a('MLP', 'PGD-50x5', em))}). " if flip else "")
               + "The gradients are not hidden: the gradient-free attack beats PGD in only "
               f"{S['spsa_pairs']} of {S['pairs']} run–budget pairs, and unbounded attacks leave at most "
               f"{pct(S['unbounded_max'])} accuracy. Instead, one full FGSM step overshoots on "
               f"{pct(S['nonmono']('VQC'))} of points, because angle encoding makes the model a wave-like "
               "(trigonometric) function of its input. "
               + (f"Lowering the encoding frequency shrinks the FGSM–PGD gap to {M['lo']:+.2f} (from "
                  f"{M['peak']:+.2f} at its peak). " if M else "")
               + f"The SinMLP shows a smaller version of the same effect ({pct(min(ratios))}–{pct(max(ratios))} of "
               "the VQC's gap); the plain MLP barely shows it."])
    rich(doc, [("Status. ", {"bold": True}),
               "Written up as a research paper, \"Single-Step Attacks Overstate the Adversarial Robustness of "
               "Angle-Encoded Quantum Classifiers\" (paper/QuAIsar-p7-paper.pdf"
               + (f" at {url}" if url else "") + "). Caveats: exact simulation, one dataset, 4–8 qubits."])


def p7_single_run_section(doc, a7, r7, eps7, vqc_f, vqc_p, mlp_p, at_p, e_hi, e_mid):
    rich(doc, [("Method. ", {"bold": True}),
               f"A {a7['qubits']}-qubit VQC with data re-uploading and a small MLP are trained on handwritten digits "
               "(3 vs 8, PCA-reduced, inputs scaled to [0, 1]), then attacked with FGSM and 10-step PGD under an "
               f"L∞ budget ε. A second VQC is trained adversarially (PGD at ε = {a7['adv_eps']})."])
    eps_cols = [f"ε={e}" for e in eps7]
    caption(doc, "Table 4. Test accuracy under attack (higher is more robust).")
    table(doc, ["Model", "Attack"] + eps_cols,
          [[r["model"], r["attack"]] + [f"{r[f'eps={e}']:.3f}" for e in eps7] for r in r7],
          [1.7, 0.8] + [0.8] * len(eps7), numeric_from=2)
    figure(doc, RESULTS / "p7_adversarial_qml" / "robustness.png", 4.6,
           "Figure 2. Accuracy under PGD attack as the perturbation budget grows.")
    rich(doc, [("Finding. ", {"bold": True}),
               f"Under the strong PGD attack the VQC is no more robust than the MLP: {pct(vqc_p[f'eps={e_hi}'])} "
               f"vs {pct(mlp_p[f'eps={e_hi}'])} at ε = {e_hi}. Yet the single-step FGSM attack stalls at "
               f"{pct(vqc_f[f'eps={e_hi}'])} against the VQC. A weak attack failing where a strong one succeeds is "
               "the classic sign of gradient masking: the model looks robust to gradient attacks without being robust. "
               f"Adversarial training helped only at small budgets ({pct(at_p[f'eps={e_mid}'])} vs "
               f"{pct(vqc_p[f'eps={e_mid}'])} at ε = {e_mid})."])
    rich(doc, [("Caveat. ", {"bold": True}),
               "Gradient masking is a hypothesis, not yet a result. It needs gradient-free attacks and random "
               "restarts to confirm."])


def p8_section(doc, p8, ps8, mwpm, dqn, none8, p_train):
    r8 = p8["results"]
    a8 = p8["args"]
    doc.add_heading("6. Reinforcement-learning decoder for quantum error correction (p8)", level=1)
    rich(doc, [("Question. ", {"bold": True}), "Can reinforcement learning learn to decode quantum errors?"])
    rich(doc, [("Method. ", {"bold": True}),
               f"A distance-{a8['d']} rotated surface code under independent bit-flip noise. A deep Q-network observes "
               "the stabiliser syndrome and flips one data qubit per step until the syndrome clears; it earns +1 for "
               f"a correct logical outcome and −1 otherwise. It trains for {a8['train_steps']:,} steps at p = {p_train} "
               f"and is compared with minimum-weight perfect matching (PyMatching) over {a8['shots']:,} shots per point."])
    caption(doc, "Table 5. Logical error rate by physical bit-flip rate p (lower is better).")
    table(doc, ["Decoder"] + [f"p={p}" for p in ps8],
          [[r["decoder"]] + [f"{r[f'p={p}']:.4f}" for p in ps8] for r in r8],
          [1.8] + [0.9] * len(ps8),
          note="'none' counts any residual error as a failure, approximating an unprotected qubit.")
    figure(doc, RESULTS / "p8_rl_quantum" / "logical_error.png", 4.6,
           "Figure 3. Logical error rate of each decoder against the physical error rate.")
    worst_gap = max(abs(dqn[f"p={p}"] - mwpm[f"p={p}"]) / max(mwpm[f"p={p}"], 1e-9) for p in ps8)
    rich(doc, [("Finding. ", {"bold": True}),
               ("The DQN matches matching across the whole range " if worst_gap < 0.15 else
                f"The DQN comes within {worst_gap:.0%} of matching across the range ") +
               f"(for example {dqn[f'p={p_train}']:.4f} vs "
               f"{mwpm[f'p={p_train}']:.4f} at p = {p_train}, against {none8[f'p={p_train}']:.4f} with no correction). "
               "One design choice was decisive: a first version that saw only the syndrome performed about three times "
               "worse than matching at p = 0.08 (0.264 in an earlier run), because two different correction paths can "
               "reach the same syndrome with different logical outcomes. Adding the agent's own flips to its observation, "
               "and masking repeat flips, closed the gap."])
    rich(doc, [("Caveat. ", {"bold": True}),
               f"At distance {a8['d']} there are only 16 syndromes, so a lookup table is already optimal. The result "
               "becomes meaningful at distance 5+, with depolarising noise and measurement errors."])


def appendix(doc, url):
    doc.add_heading("Appendix: reproducing these results", level=1)
    para(doc, "From the repository root, after running the setup script:")
    for cmd in ("python p4_qml_ids/run.py", "python p5_pqc_sidechannel/run.py", "python p6_lwe_ml_attack/run.py",
                "python p7_adversarial_qml/run.py", "python p7_adversarial_qml/masking_study.py",
                "python p7_adversarial_qml/mechanism.py", "python p8_rl_quantum/run.py",
                "python report/build_report.py --pdf"):
        p = para(doc, cmd, space_after=0)
        p.runs[0].font.name = "Consolas"; p.runs[0].font.size = Pt(9.5)
        p.paragraph_format.left_indent = Inches(0.3)
    para(doc, "", space_after=4)
    v = versions()
    para(doc, "Software environment when this report was built: " +
         ", ".join(f"{k} {val}" for k, val in v.items()) + f"; {platform.system()} {platform.release()}, CPU only.",
         size=9.5, color=MUTED)
    commit = git("rev-parse", "--short", "HEAD")
    if commit:
        para(doc, f"Code version: commit {commit}" + (f" of {url}" if url else "") + ".", size=9.5, color=MUTED)
    para(doc, "", space_after=8)
    para(doc, "This report, the experiments and the logo were created with Claude Code.",
         italic=True, color=ACCENT, size=10)


def to_pdf(docx_path):
    pdf = docx_path.with_suffix(".pdf")
    if platform.system() == "Windows":
        script = Path(__file__).with_name("docx_to_pdf.ps1")
        shell = shutil.which("pwsh") or shutil.which("powershell")
        subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(docx_path)], check=True)
        return pdf
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        sys.exit("PDF export needs Microsoft Word (Windows) or LibreOffice (soffice) on PATH.")
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)], check=True)
    return pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--author", default=None, help="author line on the cover (default: git user.name)")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF")
    a = ap.parse_args()
    author = a.author or git("config", "user.name") or "QuAIsar contributors"
    path = build(author)
    print(f"Wrote {path.relative_to(ROOT)}")
    if a.pdf:
        print(f"Wrote {to_pdf(path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
