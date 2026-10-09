"""Render the engine's methods document to PDF from a JSON spec.

There is no LaTeX in this container, so equations are typeset with matplotlib's
mathtext -- a real TeX math subset -- rendered to tight PNGs at high DPI and
placed as flowables in a reportlab document. Prose, headings and tables flow
through Platypus. The practical consequence is that every equation string must
stay inside mathtext's grammar: no \\text, no \\mathrm, no \\left/\\right, no
alignment environments. A long derivation is several equations, not one block.

Input is a JSON file:

    {"title": ..., "subtitle": ..., "sections": [
        {"title": ..., "modules": [...], "overview": ...,
         "equations": [{"label":..., "latex":..., "explain":..., "code_ref":...}],
         "conventions": ..., "status": ..., "validation": ...}]}

Usage:
    python scripts/build_methods_pdf.py spec.json out.pdf
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

#: Equations are rendered at this DPI and then scaled down in the document, so
#: they stay sharp when the PDF is zoomed. 300 is the point where the file size
#: stops being trivial and the gain stops being visible.
_DPI = 300
_EQ_FONTSIZE = 13
#: Points per rendered pixel. 72/_DPI reproduces the nominal size; a little
#: under that keeps a long equation inside the text column.
_SCALE = 72.0 / _DPI * 0.92

_INK = colors.HexColor("#16181d")
_MUTED = colors.HexColor("#5b6472")
_RULE = colors.HexColor("#c8cedb")
_ACCENT = colors.HexColor("#1f4e79")


def _styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="Times-Roman", textColor=_INK)
    out = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                fontSize=23, leading=27, textColor=_INK,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontName="Times-Italic",
                                   fontSize=11.5, leading=15, textColor=_MUTED,
                                   alignment=1, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold",
                             fontSize=15.5, leading=19, textColor=_ACCENT,
                             spaceBefore=16, spaceAfter=5),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold",
                             fontSize=11.5, leading=14, textColor=_INK,
                             spaceBefore=11, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=10,
                               leading=14.2, alignment=TA_JUSTIFY, **base),
        "eqlabel": ParagraphStyle("el", parent=ss["Normal"], fontName="Times-Bold",
                                  fontSize=9.4, leading=12, textColor=_ACCENT,
                                  spaceBefore=7, spaceAfter=1),
        "explain": ParagraphStyle("ex", parent=ss["Normal"], fontSize=9.3,
                                  leading=12.6, alignment=TA_JUSTIFY,
                                  leftIndent=9, **base),
        "ref": ParagraphStyle("r", parent=ss["Normal"], fontName="Courier",
                              fontSize=7.8, leading=10, textColor=_MUTED,
                              leftIndent=9, spaceAfter=3),
        "mods": ParagraphStyle("m", parent=ss["Normal"], fontName="Courier",
                               fontSize=8.1, leading=11, textColor=_MUTED,
                               spaceAfter=6),
        "note": ParagraphStyle("n", parent=ss["Normal"], fontSize=9.4,
                               leading=13, alignment=TA_JUSTIFY,
                               leftIndent=9, borderPadding=0, **base),
    }
    return out


#: mathtext has no manual sizing commands, so \big( is a parse error where ( is
#: fine. Stripping them is lossless here: mathtext already sizes delimiters to
#: their content. \left and \right it does understand, so they are left alone.
_SIZING = ("\\bigg", "\\Bigg", "\\big", "\\Big", "\\bigl", "\\bigr",
           "\\Bigl", "\\Bigr", "\\biggl", "\\biggr", "\\;", "\\,", "\\!")


def _sanitise(latex: str) -> str:
    """Make a LaTeX string safe for matplotlib's mathtext subset."""
    out = str(latex)
    for tok in _SIZING:
        out = out.replace(tok.replace("\\\\", "\\"), " " if tok.endswith((";", ",")) else "")
    # \qquad / \quad are spacing commands mathtext does accept, but a bare
    # \mathcal or \mathbb may not resolve for every character; fall back to
    # plain italics rather than failing the whole equation.
    out = out.replace("\\mathcal{M}", "M").replace("\\mathbb{R}", "R")
    return out


def _render_equation(latex: str, out_png: Path) -> tuple[int, int] | None:
    """Typeset one equation to a tight PNG. Returns (w, h) px, or None on failure."""
    latex = _sanitise(latex)
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0, 0, f"${latex}$", fontsize=_EQ_FONTSIZE, color="#16181d")
        fig.savefig(out_png, dpi=_DPI, transparent=True,
                    bbox_inches="tight", pad_inches=0.03)
    except Exception as exc:                      # mathtext parse error
        plt.close(fig)
        print(f"    [eq] FAILED to render: {latex[:70]}\n          {exc}")
        return None
    plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(out_png) as im:
        return im.size


def _p(text: str) -> str:
    """Escape prose for reportlab's mini-HTML, keeping intentional <i>/<b>."""
    safe = html.escape(str(text or ""), quote=False)
    for tag in ("i", "b", "sub", "super", "br/"):
        safe = safe.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        safe = safe.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return safe


def build(spec: dict, out_pdf: Path, workdir: Path) -> None:
    S = _styles()
    workdir.mkdir(parents=True, exist_ok=True)
    story: list = []

    story.append(Spacer(1, 22 * mm))
    story.append(Paragraph(_p(spec.get("title", "Methods")), S["title"]))
    if spec.get("subtitle"):
        story.append(Paragraph(_p(spec["subtitle"]), S["subtitle"]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="55%", thickness=0.7, color=_RULE,
                            hAlign="CENTER"))
    story.append(Spacer(1, 6 * mm))
    if spec.get("preamble"):
        for para in spec["preamble"].split("\n\n"):
            story.append(Paragraph(_p(para), S["body"]))
            story.append(Spacer(1, 2.2 * mm))

    # Contents
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Contents", S["h2"]))
    for i, sec in enumerate(spec.get("sections", []), 1):
        story.append(Paragraph(f"{i}.&nbsp;&nbsp;{_p(sec.get('title',''))}",
                               S["mods"]))
    story.append(PageBreak())

    n_eq = n_fail = 0
    for i, sec in enumerate(spec.get("sections", []), 1):
        story.append(Paragraph(f"{i}.&nbsp; {_p(sec.get('title',''))}", S["h1"]))
        mods = sec.get("modules") or []
        if mods:
            story.append(Paragraph(" &middot; ".join(_p(m) for m in mods), S["mods"]))
        if sec.get("overview"):
            for para in str(sec["overview"]).split("\n\n"):
                story.append(Paragraph(_p(para), S["body"]))
                story.append(Spacer(1, 1.8 * mm))

        eqs = sec.get("equations") or []
        if eqs:
            story.append(Paragraph("Formulation", S["h2"]))
        for j, eq in enumerate(eqs, 1):
            n_eq += 1
            png = workdir / f"eq_{i:02d}_{j:02d}.png"
            size = _render_equation(eq.get("latex", ""), png)
            block = [Paragraph(f"({i}.{j})&nbsp; {_p(eq.get('label',''))}",
                               S["eqlabel"])]
            if size:
                w, h = size
                block.append(Spacer(1, 1.0 * mm))
                block.append(Image(str(png), width=w * _SCALE, height=h * _SCALE,
                                   hAlign="CENTER"))
                block.append(Spacer(1, 1.4 * mm))
            else:
                n_fail += 1
                block.append(Paragraph(
                    f"<font face='Courier' size='8'>{_p(eq.get('latex',''))}</font>",
                    S["ref"]))
            if eq.get("explain"):
                block.append(Paragraph(_p(eq["explain"]), S["explain"]))
            if eq.get("code_ref"):
                block.append(Paragraph(_p(eq["code_ref"]), S["ref"]))
            story.append(KeepTogether(block))

        for key, head in (("conventions", "Conventions, units and sign"),
                          ("status", "Implementation status"),
                          ("validation", "Validation")):
            if sec.get(key):
                story.append(Paragraph(head, S["h2"]))
                for para in str(sec[key]).split("\n\n"):
                    story.append(Paragraph(_p(para), S["note"]))
                    story.append(Spacer(1, 1.4 * mm))

        if i < len(spec.get("sections", [])):
            story.append(Spacer(1, 3 * mm))
            story.append(HRFlowable(width="100%", thickness=0.4, color=_RULE))

    def _footer(canv, doc):
        canv.saveState()
        canv.setFont("Times-Roman", 8)
        canv.setFillColor(_MUTED)
        canv.drawCentredString(A4[0] / 2.0, 12 * mm, str(doc.page))
        if doc.page > 1 and spec.get("running_head"):
            canv.drawCentredString(A4[0] / 2.0, A4[1] - 12 * mm,
                                   spec["running_head"])
        canv.restoreState()

    SimpleDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=spec.get("title", "Methods"), author="Quantize",
    ).build(story, onFirstPage=_footer, onLaterPages=_footer)

    print(f"  {n_eq} equations, {n_fail} failed to typeset")
    print(f"  wrote {out_pdf}  ({out_pdf.stat().st_size/1024:.0f} KB)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    build(spec, out, out.parent / "_eq")


if __name__ == "__main__":
    main()
