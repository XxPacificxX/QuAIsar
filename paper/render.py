"""Render one list of content blocks to LaTeX and to DOCX.

Inline markup inside text:
    $...$        math (LaTeX). In DOCX it is converted to Unicode.
    **bold**     *italic*
    [@a,b]       citation(s) -> \\cite{a,b} / [1, 2]
    [#fig:x]     cross-reference -> \\ref{fig:x} / 3

Blocks (tuples):
    ("h1", text[, label])  ("h2", text)  ("p", text)  ("bullets", [text, ...])
    ("eq", latex, unicode_text)
    ("prop", title, text)                       numbered proposition
    ("table", label, caption, headers, rows, align)   align like "lrr"
    ("figure", label, path, caption, width)     width = fraction of text width
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

TOKEN = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[@[\w:,\-]+\]|\[#[\w:\-]+\])")

# ---------------------------------------------------------------- numbering
def number_things(blocks):
    """Figure/table/proposition/section numbers by order of appearance."""
    nums, counters = {}, {"fig": 0, "tab": 0, "sec": 0, "prop": 0}
    for b in blocks:
        if b[0] == "figure":
            counters["fig"] += 1; nums[b[1]] = counters["fig"]
        elif b[0] == "table":
            counters["tab"] += 1; nums[b[1]] = counters["tab"]
        elif b[0] == "h1":
            counters["sec"] += 1
            if len(b) > 2: nums[b[2]] = counters["sec"]
    return nums


def cite_order(blocks, refs):
    order = []
    def scan(text):
        for m in re.finditer(r"\[@([\w:,\-]+)\]", text):
            for k in m.group(1).split(","):
                if k not in refs:
                    raise KeyError(f"unknown citation key {k!r}")
                if k not in order:
                    order.append(k)
    for b in blocks:
        for part in b[1:]:
            if isinstance(part, str):
                scan(part)
            elif isinstance(part, list):
                for item in part:
                    if isinstance(item, str): scan(item)
                    elif isinstance(item, list):
                        for cell in item: scan(str(cell))
    return order


# ---------------------------------------------------------------- LaTeX
UNICODE_TEX = {"×": r"$\times$", "±": r"$\pm$", "ε": r"$\varepsilon$", "−": r"$-$", "≈": r"$\approx$",
               "≤": r"$\leq$", "≥": r"$\geq$", "π": r"$\pi$", "→": r"$\rightarrow$", "∞": r"$\infty$",
               "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\dots{}",
               "⟨": r"$\langle$", "⟩": r"$\rangle$", "Ω": r"$\Omega$", "ω": r"$\omega$", "∈": r"$\in$",
               "²": r"$^2$", "₀": r"$_0$", "°": r"$^\circ$", "·": r"$\cdot$", "%": r"\%"}
TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "$": r"\$"}


def tex_text(s):
    out = []
    for ch in s:
        out.append(TEX_ESC.get(ch) or UNICODE_TEX.get(ch) or ch)
    return "".join(out)


def tex_inline(s):
    parts = []
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("$") and tok.endswith("$"):
            parts.append(tok)
        elif tok.startswith("**"):
            parts.append(r"\textbf{" + tex_inline(tok[2:-2]) + "}")
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            parts.append(r"\emph{" + tex_inline(tok[1:-1]) + "}")
        elif tok.startswith("[@"):
            parts.append(r"~\cite{" + tok[2:-1] + "}")
        elif tok.startswith("[#"):
            parts.append(r"\ref{" + tok[2:-1] + "}")
        else:
            parts.append(tex_text(tok))
    return "".join(parts).replace(" ~\\cite", "~\\cite")


def to_latex(meta, blocks, refs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    order = cite_order(blocks, refs)
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
         r"\usepackage[margin=1in]{geometry}", r"\usepackage{amsmath,amssymb,amsthm}",
         r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage[hidelinks]{hyperref}",
         r"\usepackage{microtype}", r"\newtheorem{proposition}{Proposition}",
         r"\title{" + tex_inline(meta["title"]) + "}",
         r"\author{" + tex_inline(meta["author"]) + (r"\\ \small " + tex_inline(meta["affiliation"]) if meta.get("affiliation") else "") + "}",
         r"\date{" + tex_inline(meta["date"]) + "}",
         r"\begin{document}", r"\maketitle",
         r"\begin{abstract}", tex_inline(meta["abstract"]), r"\end{abstract}", ""]
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            L.append(r"\section{" + tex_inline(b[1]) + "}" + (r"\label{" + b[2] + "}" if len(b) > 2 else ""))
        elif kind == "h1*":
            L.append(r"\section*{" + tex_inline(b[1]) + "}")
        elif kind == "h2":
            L.append(r"\subsection{" + tex_inline(b[1]) + "}")
        elif kind == "p":
            L.append(tex_inline(b[1]) + "\n")
        elif kind == "bullets":
            L.append(r"\begin{itemize}")
            L += [r"\item " + tex_inline(i) for i in b[1]]
            L.append(r"\end{itemize}")
        elif kind == "eq":
            L += [r"\begin{equation}", b[1], r"\end{equation}"]
        elif kind == "prop":
            L += [r"\begin{proposition}[" + tex_inline(b[1]) + "]", tex_inline(b[2]), r"\end{proposition}"]
        elif kind == "table":
            _, label, cap, headers, rows, align = b
            L += [r"\begin{table}[t]", r"\centering", r"\caption{" + tex_inline(cap) + "}", r"\label{" + label + "}",
                  r"\small", r"\begin{tabular}{" + align + "}", r"\toprule",
                  " & ".join(r"\textbf{" + tex_inline(h) + "}" for h in headers) + r" \\", r"\midrule"]
            L += [" & ".join(tex_inline(str(c)) for c in row) + r" \\" for row in rows]
            L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        elif kind == "figure":
            _, label, path, cap, width = b
            shutil.copy(path, out_dir / "figures" / Path(path).name)
            L += [r"\begin{figure}[t]", r"\centering",
                  r"\includegraphics[width=" + f"{width:.2f}" + r"\linewidth]{figures/" + Path(path).name + "}",
                  r"\caption{" + tex_inline(cap) + "}", r"\label{" + label + "}", r"\end{figure}", ""]
        else:
            raise ValueError(kind)
    L.append(r"\begin{thebibliography}{99}")
    for k in order:
        L.append(r"\bibitem{" + k + "} " + tex_text(refs[k]))
    L += [r"\end{thebibliography}", r"\end{document}"]
    path = out_dir / "main.tex"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------- DOCX
MATH_UNI = [  # longest commands first
    ("varepsilon", "ε"), ("epsilon", "ε"), ("lesssim", "≲"), ("gtrsim", "≳"), ("langle", "⟨"), ("rangle", "⟩"),
    ("infty", "∞"), ("Omega", "Ω"), ("omega", "ω"), ("approx", "≈"), ("times", "×"), ("nabla", "∇"),
    ("theta", "θ"), ("sigma", "σ"), ("alpha", "α"), ("delta", "δ"), ("cdot", "·"), ("leq", "≤"), ("geq", "≥"),
    ("sum", "Σ"), ("sup", "sup"), ("max", "max"), ("min", "min"), ("sin", "sin"), ("cos", "cos"), ("pm", "±"),
    ("pi", "π"), ("le", "≤"), ("ge", "≥"), ("in", "∈"), ("ell", "ℓ"), ("mid", "|"), ("lVert", "‖"), ("rVert", "‖"),
    ("to", "→"), ("quad", "  "), ("ldots", "…"), ("dots", "…"), ("square", "□"), ("mathrm", ""), ("operatorname", ""), ("text", ""), ("mathbf", ""),
    ("left", ""), ("right", ""), ("big", ""), ("Big", ""),
]
SUP = str.maketrans("0123456789+-n()iT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ⁽⁾ⁱᵀ")
SUB_CHARS = dict(zip("0123456789+-()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
SUB = str.maketrans(SUB_CHARS)


def math_to_unicode(m):
    s = m
    s = re.sub(r"\\t?frac\{1\}\{2\}|\\tfrac12|\\frac12", "½", s)
    s = re.sub(r"\\t?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = s.replace(r"\{", "\x01").replace(r"\}", "\x02")  # literal braces survive brace stripping
    for cmd, uni in MATH_UNI:
        s = re.sub(r"\\" + cmd + r"(?![a-zA-Z])", uni, s)
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    s = s.replace(r"\|", "‖")

    def sup(mm):
        body = mm.group(1) or mm.group(2)
        conv = body.translate(SUP)
        return conv if all(c not in body for c in "abcdefghjklmopqrsuvwxyz") else "^" + body
    def sub(mm):
        body = mm.group(1) or mm.group(2)
        if all(c in SUB_CHARS for c in body):
            return body.translate(SUB)       # e.g. Z_0 -> Z₀, sup_t -> supₜ
        return body if len(body) == 1 else "_(" + body + ")"
    s = re.sub(r"\^\{([^}]*)\}|\^(\S)", sup, s)
    s = re.sub(r"_\{([^}]*)\}|_(\S)", sub, s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return s.replace("\x01", "{").replace("\x02", "}").replace("⟨ ", "⟨")


def docx_runs(paragraph, s, nums, cites, size=None, bold=False, italic=False):
    from docx.shared import Pt

    for tok in TOKEN.split(s):
        if not tok:
            continue
        b, i, text = bold, italic, tok
        if tok.startswith("$") and tok.endswith("$"):
            text = math_to_unicode(tok[1:-1])
            r = paragraph.add_run(text); r.font.name = "Cambria Math"
            r.bold = bold
            if size: r.font.size = Pt(size)
            continue
        if tok.startswith("**"):
            docx_runs(paragraph, tok[2:-2], nums, cites, size, True, italic); continue
        if tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            docx_runs(paragraph, tok[1:-1], nums, cites, size, bold, True); continue
        if tok.startswith("[@"):
            text = " [" + ", ".join(str(cites[k]) for k in tok[2:-1].split(",")) + "]"
        elif tok.startswith("[#"):
            text = str(nums[tok[2:-1]])
        r = paragraph.add_run(text); r.bold, r.italic = b, i
        if size: r.font.size = Pt(size)


def to_docx(meta, blocks, refs, path: Path):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    nums = number_things(blocks)
    order = cite_order(blocks, refs)
    cites = {k: i + 1 for i, k in enumerate(order)}
    BODY = "Cambria"

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    st = doc.styles
    st["Normal"].font.name = BODY; st["Normal"].font.size = Pt(10.5)
    st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), BODY)
    st["Normal"].paragraph_format.space_after = Pt(5)
    st["Normal"].paragraph_format.line_spacing = 1.1
    for name, size in (("Heading 1", 13), ("Heading 2", 11)):
        h = st[name]
        h.font.name = BODY; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = RGBColor(0, 0, 0)
        rf = h.element.rPr.rFonts
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), BODY)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
        h.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        h.paragraph_format.space_after = Pt(4); h.paragraph_format.keep_with_next = True
    st["List Bullet"].paragraph_format.space_after = Pt(2)

    def centered(text, size, bold=False, italic=False, after=4):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(after)
        docx_runs(p, text, nums, cites, size, bold, italic)
        return p

    centered(meta["title"], 16, bold=True, after=10)
    centered(meta["author"], 11.5, after=1)
    if meta.get("affiliation"): centered(meta["affiliation"], 10, italic=True, after=1)
    centered(meta["date"], 10, after=14)
    centered("**Abstract**", 10.5, after=2)
    p = doc.add_paragraph(); p.paragraph_format.left_indent = p.paragraph_format.right_indent = Inches(0.5)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    docx_runs(p, meta["abstract"], nums, cites, 10)

    sec_n = sub_n = prop_n = 0
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            sec_n += 1; sub_n = 0
            h = doc.add_heading(level=1); docx_runs(h, f"{sec_n}  {b[1]}", nums, cites)
        elif kind == "h1*":
            h = doc.add_heading(level=1); docx_runs(h, b[1], nums, cites)
        elif kind == "h2":
            sub_n += 1
            h = doc.add_heading(level=2); docx_runs(h, f"{sec_n}.{sub_n}  {b[1]}", nums, cites)
        elif kind == "p":
            p = doc.add_paragraph()
            # justify prose, but not paragraphs with long URLs (they stretch word gaps)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if "http" in b[1] else WD_ALIGN_PARAGRAPH.JUSTIFY
            docx_runs(p, b[1], nums, cites)
        elif kind == "bullets":
            for item in b[1]:
                docx_runs(doc.add_paragraph(style="List Bullet"), item, nums, cites)
        elif kind == "eq":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(b[2]); r.font.name = "Cambria Math"
        elif kind == "prop":
            prop_n += 1
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.25)
            docx_runs(p, f"**Proposition {prop_n} ({b[1]}).** ", nums, cites)
            docx_runs(p, b[2], nums, cites, italic=True)
        elif kind == "table":
            _, label, cap, headers, rows, align = b
            c = doc.add_paragraph(); c.paragraph_format.keep_with_next = True
            docx_runs(c, f"**Table {nums[label]}.** {cap}", nums, cites, 9.5)
            t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
            tblPr = t._tbl.tblPr
            borders = OxmlElement("w:tblBorders")
            for edge, sz in (("top", "8"), ("bottom", "8"), ("insideH", "0")):
                el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "single" if sz != "0" else "nil")
                el.set(qn("w:sz"), sz); el.set(qn("w:color"), "000000"); borders.append(el)
            tblPr.append(borders)
            def fill(cells, values, bold):
                for j, (cell, v) in enumerate(zip(cells, values)):
                    pp = cell.paragraphs[0]
                    pp.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER}.get(
                        align[j], WD_ALIGN_PARAGRAPH.RIGHT)
                    pp.paragraph_format.space_after = Pt(0); pp.paragraph_format.keep_with_next = True
                    docx_runs(pp, str(v), nums, cites, 9, bold)
            fill(t.rows[0].cells, headers, True)
            hb = OxmlElement("w:tcBorders"); bot = OxmlElement("w:bottom")
            bot.set(qn("w:val"), "single"); bot.set(qn("w:sz"), "4"); bot.set(qn("w:color"), "000000"); hb.append(bot)
            for cell in t.rows[0].cells:
                cell._tc.get_or_add_tcPr().append(__import__("copy").deepcopy(hb))
            for row in rows:
                fill(t.add_row().cells, row, False)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif kind == "figure":
            _, label, img, cap, width = b
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(img), width=Inches(6.5 * width))
            c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            c.paragraph_format.space_after = Pt(10)
            docx_runs(c, f"**Figure {nums[label]}.** {cap}", nums, cites, 9.5)
        else:
            raise ValueError(kind)

    h = doc.add_heading(level=1); h.add_run("References")
    for k in order:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3); p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"[{cites[k]}]  {refs[k]}"); r.font.size = Pt(9.5)
    doc.core_properties.title = meta["title"].replace("*", "")
    doc.core_properties.author = meta["author"]
    doc.save(path)
    return path
