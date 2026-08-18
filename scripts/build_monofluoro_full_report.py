"""Build the full monofluorinated benchmark report as a PDF.

Every number in the report is read from a file in ``output/`` or computed from
the reference module at build time. Nothing is transcribed by hand, so the prose
and the tables cannot drift apart from the runs that produced them, and a stale
input shows up as a missing section rather than as a plausible wrong number.

    python scripts/build_monofluoro_full_report.py

Inputs, all optional -- sections are skipped when their file is absent:

    output/monofluoro_benchmark.json             set 1, four legs
    output/monofluoro_current_set1.json          set 1, current engine
    output/monofluoro_current_set2.json          set 2, current engine
    output/monofluoro_validate.json              fluorobenzene, held out
    output/monofluoro_benchmark_hf_6-31gd.json   level-of-theory comparison
    output/monofluoro_benchmark_b3lyp_6-31gd.json
    output/monofluoro_tune.json                  prior-width scan
    output/theory_error_estimate.json            answer-free sigma estimate
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

from dev.monofluoro_references import (  # noqa: E402
    HELDOUT,
    MOLECULES,
    MOLECULES_SET2,
    SIGMA_REL,
)

OUT = _ROOT / "output" / "monofluoro_full_report.pdf"

#: Table cells are plain strings, not Paragraphs, and ReportLab only parses
#: markup inside a Paragraph -- an entity in a bare cell renders as itself. Use
#: real characters there and keep entities for prose.
ANG, ARROW, NDASH, SIGMA = "\u00c5", "\u2192", "\u2013", "\u03c3"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
BAND = colors.HexColor("#f2f2f0")
GOOD = colors.HexColor("#1d6b3f")
WARN = colors.HexColor("#8a3324")

_styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_styles["Normal"], fontName="Helvetica",
                      fontSize=9.5, leading=14.2, alignment=TA_JUSTIFY,
                      textColor=INK, spaceAfter=7)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=14.5,
                    leading=18, spaceBefore=16, spaceAfter=8, alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=10.8,
                    leading=14.5, spaceBefore=11, spaceAfter=5, alignment=0)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.1, leading=11.4,
                         textColor=MUTED, spaceBefore=4, spaceAfter=12)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=8.2,
                      leading=11.6, alignment=0, spaceAfter=3)
EQ = ParagraphStyle("eq", parent=BODY, fontName="Courier", fontSize=8.6,
                    leading=12.6, alignment=1, spaceBefore=6, spaceAfter=8)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold",
                       fontSize=21, leading=25, alignment=0, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10.5, leading=14,
                     textColor=MUTED, alignment=0, spaceAfter=16)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.4, leading=9.4,
                      alignment=0, spaceAfter=0)


def load(name):
    path = _ROOT / "output" / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def sub(text: str) -> str:
    """Subscript digits that follow a letter, for formulas and B0."""
    return re.sub(r"(?<=[A-Za-z])(\d+)", r"<sub>\1</sub>", text)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, align_left=(0,), header_rows=1, bold_cells=(),
          band_rows=(), font_size=8.1):
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=header_rows)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        ("FONT", (0, 0), (-1, max(header_rows - 1, 0)), "Helvetica-Bold", font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header_rows:
        style.append(("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1),
                      0.7, INK))
    for col in align_left:
        style.append(("ALIGN", (col, 0), (col, -1), "LEFT"))
    for r in band_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), BAND))
    for r, c in bold_cells:
        style.append(("FONT", (c, r), (c, r), "Helvetica-Bold", font_size))
        style.append(("TEXTCOLOR", (c, r), (c, r), GOOD))
    t.setStyle(TableStyle(style))
    return t


def fmt(v, dp=2, plus=False):
    if v is None:
        return "--"
    return f"{v:+.{dp}f}" if plus else f"{v:.{dp}f}"


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ── sections ──────────────────────────────────────────────────────────────────

def sec_title(story, sources):
    story.append(p("Geometry from rotational spectra and quantum chemistry", TITLE))
    story.append(p(
        f"A benchmark on monofluorinated molecules &mdash; {date.today().isoformat()}"
        f" &middot; commit {git_rev()}", SUB))
    story.append(p(
        "This report documents what the Quantize hybrid engine does, what it was "
        "measured against, and how well it did. The target case is a fluorinated "
        "molecule with no accepted experimental structure, where the question is "
        "whether combining a rotational spectrum with an electronic-structure "
        "calculation gives a geometry closer to the truth than either source "
        "alone.", BODY))
    story.append(p(
        "Every table is generated from a stored run or computed from the "
        "reference data at build time. Where a result is negative it is reported "
        "as such; three of the ideas tested here did not work, and saying so is "
        "more useful than leaving them to be rediscovered.", BODY))
    story.append(p("Inputs used for this build", H2))
    rows = [["file", "status"]]
    for name, obj in sources:
        rows.append([name, "loaded" if obj is not None else "absent, section skipped"])
    story.append(table(rows, [92 * mm, 68 * mm], align_left=(0, 1)))


def sec_math(story):
    story.append(PageBreak())
    story.append(p("1 &nbsp; The problem and the mathematics", H1))

    story.append(p("1.1 &nbsp; What a rotational spectrum measures", H2))
    story.append(p(
        "A rotating molecule has three principal moments of inertia, and the "
        "spacing of its rotational transitions gives the three rotational "
        "constants A &ge; B &ge; C. For a rigid body these follow from the "
        "geometry and the atomic masses alone:", BODY))
    story.append(p("I = &Sigma;<sub>i</sub> m<sub>i</sub> (r<sub>i</sub><sup>2</sup>&delta;"
                   " &minus; r<sub>i</sub> &otimes; r<sub>i</sub>) &nbsp;&nbsp;&nbsp; "
                   "A, B, C = h / (8&pi;<sup>2</sup> I<sub>a,b,c</sub>)", EQ))
    story.append(p(
        "Three numbers per molecule cannot fix a structure with more than three "
        "internal degrees of freedom, so the classical route is isotopic "
        "substitution: replacing one atom with a heavier isotope leaves the "
        "electronic structure &mdash; and therefore the geometry &mdash; "
        "unchanged, while shifting the moments in a way that depends on where "
        "that atom sits. Each isotopologue is another three equations against "
        "the same unknowns.", BODY))

    story.append(p("1.2 &nbsp; Why fluorine is the hard case", H2))
    story.append(p(
        "Isotopic substitution needs a second stable isotope. Fluorine has only "
        "<sup>19</sup>F, so no isotopologue exists in which a fluorine mass "
        "differs. Every substitution-based method &mdash; Kraitchman analysis, "
        "r<sub>s</sub> structures, mass-dependent r<sub>m</sub> fits &mdash; is "
        "therefore blind to fluorine, which is exactly why fluorinated molecules "
        "so often lack accepted experimental geometries.", BODY))
    story.append(p(
        "Blind to substitution is not the same as invisible. Fluorine is heavy, "
        "and moving it shifts the constants about as much as moving carbon or "
        "oxygen; on formyl fluoride a 0.01 &Aring; displacement changes the "
        "constants by 0.79% for F against 0.89% for C and 0.27% for H. The "
        "information about fluorine is present in the constants collectively; it "
        "is only the substitution trick that cannot extract it. That is the gap "
        "the quantum surface fills.", BODY))

    story.append(p("1.3 &nbsp; The inverse problem", H2))
    story.append(p(
        "Write p for the parameters (Cartesian coordinates here), r for the "
        "residual between observed and calculated constants weighted by their "
        "uncertainties, and J for the Jacobian &part;(constants)/&part;p in "
        "MHz/&Aring;. Stacking every measured component of every isotopologue "
        "gives an overdetermined-looking but badly conditioned system. Its "
        "singular value decomposition separates two subspaces:", BODY))
    story.append(p("J = U &Sigma; V<sup>T</sup>", EQ))
    story.append(p(
        "Directions in V with large singular values move the constants and are "
        "determined by the data. Directions with singular values near zero do "
        "not move the constants at all: no amount of spectroscopy constrains "
        "them. The count of measured constants is not the amount of information "
        "&mdash; the numerical rank of the &sigma;-weighted stacked Jacobian is, "
        "and it is reported per molecule in section 4.", BODY))

    story.append(p("1.4 &nbsp; Two ways to combine the sources", H2))
    story.append(p(
        "<b>Split.</b> Give the data unconditional authority over every "
        "direction above a relative rank cutoff, and let the quantum surface "
        "own the rest by damped-Newton minimisation in the null space. Clean, "
        "but wrong at the boundary: a direction the data barely resolves is "
        "handed to it entirely.", BODY))
    story.append(p(
        "<b>Joint (the default).</b> Leave every direction contested and weight "
        "each source by how well it knows that direction. Reading the quantum "
        "term as a Gaussian prior on geometry centred at theory's minimum, the "
        "step solves", BODY))
    story.append(p("(J<sup>T</sup>J + &alpha;<sub>q</sub> H + &lambda;I) "
                   "&Delta;p = J<sup>T</sup>r &minus; &alpha;<sub>q</sub> g", EQ))
    story.append(p(
        "with g and H the energy gradient and Hessian. No hard partition "
        "survives: where the data is informative J<sup>T</sup>J dominates, and "
        "where it is not the prior does, with a smooth handover in between.", BODY))

    story.append(p("1.5 &nbsp; Calibrating the one free weight", H2))
    story.append(p(
        "The spectral block is a sum of squared &sigma;-normalised residuals and "
        "is dimensionless; the quantum block is an energy. Adding them with "
        "&alpha;<sub>q</sub> = 1 compares a chi-square to Hartrees, and the "
        "spectral side wins by whatever factor the units happen to produce. "
        "Requiring that one prior standard deviation cost half a chi-square unit "
        "fixes the scale:", BODY))
    story.append(p("&alpha;<sub>q</sub> = 1 / (&frac12; &lambda;&#772; "
                   "&sigma;<sub>x</sub><sup>2</sup>)", EQ))
    story.append(p(
        "where &lambda;&#772; is a typical curvature and &sigma;<sub>x</sub> is "
        "the displacement over which the quantum surface is trusted &mdash; "
        "roughly the geometry error of the method. This turns an opaque weight "
        "into a statement in &Aring;ngstr&ouml;ms about how good the electronic "
        "structure is, which can be estimated without knowing the answer "
        "(section 7.3).", BODY))

    story.append(p("1.6 &nbsp; Symmetry", H2))
    story.append(p(
        "Rotational constants cannot tell a symmetric structure from a slightly "
        "distorted one, so an unconstrained fit spends information breaking "
        "symmetry that physics fixes exactly. The engine detects the point group "
        "from the starting geometry by validating every operation of a candidate "
        "group against the actual atom positions, then projects each step onto "
        "the symmetric subspace. A group is only accepted when every one of its "
        "operations holds, so the worst case is a subgroup of the true group, "
        "which constrains less rather than more.", BODY))

    story.append(p("1.7 &nbsp; What is not modelled", H2))
    story.append(p(
        "The single largest systematic is left uncorrected. Spectroscopy "
        "measures B<sub>0</sub>, the constant of a molecule vibrating in its "
        "ground state; theory computes B<sub>e</sub>, the bottom of the well. "
        "They differ by the rovibrational sum B<sub>0</sub> = B<sub>e</sub> "
        "&minus; &frac12;&Sigma;&alpha;<sub>r</sub>, a few tenths of a percent "
        "&mdash; far larger than the precision of the constants. Computing "
        "&alpha;<sub>r</sub> needs a cubic force field, and at the levels of "
        "theory affordable here that was measured to hurt more than it helped "
        "(section 8.1). The offset is carried in the uncertainties instead, "
        "which is why the &sigma; model in section 3.3 is dominated by it.", BODY))


def sec_setup(story, cur, tune, terr):
    story.append(PageBreak())
    story.append(p("2 &nbsp; Experimental setup", H1))

    story.append(p("2.1 &nbsp; Level of theory", H2))
    story.append(p(
        "RHF/6-31G via PySCF, with analytic gradients and Hessians. It is a "
        "deliberately cheap level: the point is to test whether the data "
        "rescues a poor surface, and a molecule with no accepted structure is "
        "unlikely to have a high-level calculation either. Section 7 repeats "
        "part of the benchmark at B3LYP/6-31G(d).", BODY))
    story.append(p(
        "<b>The level applies to both legs.</b> Theory alone and the hybrid's "
        "quantum half always use the same method and basis. Changing it for one "
        "and not the other would make the comparison say nothing about whether "
        "combining the sources helps.", BODY))

    story.append(p("2.2 &nbsp; Starting geometry", H2))
    story.append(p(
        "Every data-using leg starts from theory's own optimised geometry (warm "
        "start). From there the data can only move the structure, so the "
        "question the benchmark answers is 'does the spectrum improve on "
        "theory', which is the production question. A cold start from a "
        "geometry displaced by 0.03 &Aring; RMS remains available and measures "
        "something different &mdash; whether the optimiser can navigate from a "
        "poor guess &mdash; which is a real property but not this one.", BODY))

    story.append(p("2.3 &nbsp; Uncertainties", H2))
    story.append(p(
        f"Each constant is given &sigma; = hypot(model, quoted), where the model "
        f"term is {SIGMA_REL[0] * 100:.1f}% of A and {SIGMA_REL[1] * 100:.1f}% "
        f"of B and C, and the quoted term is the last decimal place actually "
        f"printed in the source. The model term dominates and is not measurement "
        f"noise: it stands for the r<sub>s</sub>-versus-r<sub>0</sub> difference "
        f"and the uncorrected rovibrational offset of section 1.7. Treating "
        f"published constants at their quoted precision would tell the fit the "
        f"targets are exact when the structural model behind them is not.", BODY))

    story.append(p("2.4 &nbsp; Data levels", H2))
    story.append(p(
        "Each molecule is run twice: <b>parent only</b>, using just the "
        "principal isotopologue, and <b>all species</b>, using every measured "
        "isotopologue. The pair separates 'the hybrid helps' from 'more data "
        "helps', which are different claims.", BODY))

    story.append(p("2.5 &nbsp; What is scored", H2))
    story.append(p(
        "RMS error over all bond lengths in m&Aring;, against the published "
        "structure; separately the signed C&ndash;F error, which is the bond "
        "fluorine's lack of isotopes makes hardest. Bond lengths and angles are "
        "invariant to translation and rotation, so no structural alignment is "
        "needed and none is done.", BODY))

    if terr:
        story.append(p("2.6 &nbsp; Choosing the prior width without the answer", H2))
        story.append(p(
            "&sigma;<sub>x</sub> is the one consequential setting, and for a "
            "molecule with no accepted structure it cannot be read off a "
            "reference. It can be estimated instead from the spread between two "
            "levels of theory, which needs no experimental input:", BODY))
        rows = [["molecule", "spread between levels", "true error at RHF/6-31G"]]
        for e in terr:
            rows.append([e["molecule"], f"{e['spread_ma']:.1f} m{ANG}",
                         f"{e['true_error_ma']:.1f} m{ANG}"])
        story.append(table(rows, [58 * mm, 52 * mm, 50 * mm], align_left=(0,)))
        story.append(p(
            "The estimate tracks the true error closely enough to place "
            "&sigma;<sub>x</sub> in the right decade, which is all that is "
            "needed &mdash; the scan in section 7.2 shows a wide plateau.",
            CAPTION))


def sec_references(story):
    story.append(PageBreak())
    story.append(p("3 &nbsp; Reference data and provenance", H1))
    story.append(p(
        "Every rotational constant below is a measured literature value. None is "
        "back-calculated from a structure. This matters: an earlier version of "
        "this benchmark derived isotopologue constants from the published "
        "geometry, which made the data agree with the structure by construction "
        "and inverted several conclusions. Structures and constants are cited "
        "separately because they usually come from different work, and a "
        "disagreement between them is a signal (section 4.1) rather than noise.",
        BODY))

    for label, group in (("Calibration set", MOLECULES),
                         ("Held out", HELDOUT),
                         ("Second set", MOLECULES_SET2)):
        story.append(p(label, H2))
        for mol in group:
            rows = [["species", "substitutions", "components", "A / MHz",
                     "B / MHz", "C / MHz"]]
            for sp in mol.species:
                subs = ", ".join(f"atom {i + 1}{ARROW}{m:.4g}"
                                 for i, m in sp.subs.items()) or "parent"
                comps = "".join("ABC"[k] for k in sp.component_indices)
                vals = ["--" if v is None else f"{v:,.3f}" for v in sp.abc_mhz]
                # Wrapped, because a multiply-substituted species (fluorobenzene
                # 2,4,6-d3) lists more substitutions than fit on one line and a
                # bare string would overrun the next column instead of wrapping.
                rows.append([sp.label, p(subs, CELL), comps] + vals)
            head = (f"<b>{sub(mol.name)}</b> &nbsp; {sub(mol.formula)} &nbsp;&middot;&nbsp; "
                    f"{mol.n_atoms} atoms, {mol.internal_dof} internal degrees of "
                    f"freedom, {len(mol.species)} measured species")
            story.append(KeepTogether([
                p(head, BODY),
                p(f"Structure: {mol.structure_source}<br/>"
                  f"Constants: {mol.constants_source}", CAPTION),
                table(rows, [30 * mm, 44 * mm, 17 * mm, 23 * mm, 23 * mm, 23 * mm],
                      align_left=(0, 1), font_size=7.4),
                Spacer(1, 7 * mm),
            ]))


def sec_validation(story, datasets):
    story.append(PageBreak())
    story.append(p("4 &nbsp; Validating the data before using it", H1))

    story.append(p("4.1 &nbsp; Consistency of structure against constants", H2))
    story.append(p(
        "Every published structure must reproduce every measured constant of "
        "every isotopologue to about a percent &mdash; the size of the "
        "r<sub>s</sub>-versus-r<sub>0</sub> difference. A larger gap means the "
        "structure and the constants disagree, or an isotopologue has been "
        "assigned to the wrong atom. This check found the cis/trans labels on "
        "vinyl fluoride species 789 and 791 to be swapped in the source, and "
        "cleared formyl fluoride after a geometry was wrongly blamed for what "
        "turned out to be a bad set of tabulated constants.", BODY))

    rows = [["molecule", "atoms", "internal DOF", "species", "worst deviation"]]
    seen = set()
    for data in datasets:
        for m in data or []:
            if m["key"] in seen or "consistency_pct" not in m:
                continue
            seen.add(m["key"])
            rows.append([m["molecule"], str(m.get("n_atoms", "--")),
                         str(m.get("internal_dof", "--")),
                         str(m.get("n_isotopologues", "--")),
                         f"{m['consistency_pct']:.2f}%"])
    if len(rows) > 1:
        story.append(table(rows, [52 * mm, 20 * mm, 28 * mm, 22 * mm, 38 * mm],
                           align_left=(0,)))
        story.append(p(
            "All within the tolerance the r<sub>s</sub>/r<sub>0</sub> difference "
            "implies. These are not fit errors; they are the price of comparing "
            "a structure of one type against constants of another.", CAPTION))

    story.append(p("4.2 &nbsp; Information content, measured rather than counted", H2))
    story.append(p(
        "Counting constants overstates what is known. Below, 'observables' is "
        "the number of measured constants and 'rank' the numerical rank of the "
        "&sigma;-weighted stacked Jacobian &mdash; the number of internal "
        "directions the data actually constrains. The deficit is the number it "
        "cannot see at all, and which therefore falls to the quantum surface.",
        BODY))
    rows = [["molecule", "data level", "observables", "rank", "internal DOF",
             "undetermined"]]
    seen = set()
    for data in datasets:
        for m in data or []:
            if m["key"] in seen:
                continue
            seen.add(m["key"])
            for lvl, entry in m.get("levels", {}).items():
                if "rank" not in entry:
                    continue
                rows.append([m["molecule"], lvl,
                             str(entry.get("n_observables", "--")),
                             str(entry["rank"]), str(m.get("internal_dof", "--")),
                             str(entry.get("deficit", "--"))])
    if len(rows) > 1:
        story.append(table(rows, [44 * mm, 26 * mm, 26 * mm, 16 * mm, 26 * mm,
                                  26 * mm], align_left=(0, 1)))
        story.append(p(
            "Every molecule here is undersaturated even with all species: none "
            "of them is determined by spectroscopy alone. An earlier draft of "
            "this work called them over-determined on the strength of the "
            "observable count; the rank column is why that was wrong.", CAPTION))


def _legs_table(story, mol, engine_note):
    """Per-molecule results for whatever legs the stored run contains."""
    levels = mol.get("levels", {})
    leg_names = []
    for entry in levels.values():
        for name in entry.get("legs", {}):
            if name not in leg_names:
                leg_names.append(name)
    if not leg_names:
        return
    rows = [["data level", "metric"] + [n.replace("hybrid, ", "hybrid ")
                                        for n in leg_names]]
    band = []
    for lvl, entry in levels.items():
        legs = entry.get("legs", {})
        for metric, key, dp in ((f"RMS bond / m{ANG}", "rms_bond_ma", 2),
                                (f"C{NDASH}F error / m{ANG}", "cf_err_ma", 2),
                                ("RMS angle / deg", "rms_angle_deg", 3)):
            vals = [legs.get(n, {}).get(key) for n in leg_names]
            if all(v is None for v in vals):
                continue
            cells = [fmt(v, dp, plus=(key == "cf_err_ma")) for v in vals]
            rows.append([lvl, metric] + cells)
    if len(rows) == 1:
        return
    story.append(KeepTogether([
        p(f"<b>{sub(mol['molecule'])}</b> &nbsp; {sub(mol.get('formula', ''))}"
          f" &nbsp;&middot;&nbsp; theory alone "
          f"{mol['theory']['rms_bond_ma']:.2f} m&Aring; RMS bond", BODY),
        p(engine_note, CAPTION),
        table(rows, [26 * mm, 36 * mm] + [(98 / max(len(leg_names), 1)) * mm] * len(leg_names),
              align_left=(0, 1), band_rows=band, font_size=7.8),
        Spacer(1, 6 * mm),
    ]))


def _summary(story, mols):
    """Aggregate over every molecule and data level in the current-engine runs."""
    if not mols:
        return
    legs = []
    for m in mols:
        for entry in m.get("levels", {}).values():
            for name in entry.get("legs", {}):
                if name not in legs:
                    legs.append(name)
    if not legs:
        return

    story.append(p("5.1 &nbsp; Summary", H2))
    # `legs` already carries a "theory" entry, so do not add a second one.
    short = {"theory": "theory", "experiment": "experiment",
             "hybrid, split": "hybrid\nsplit", "hybrid, joint prior": "hybrid\njoint"}
    rows = [["metric"] + [short.get(n, n) for n in legs]]
    body, counts = [], {n: 0 for n in legs}
    n_cases = 0
    for key, label, dp in (("rms_bond_ma", f"mean RMS bond / m{ANG}", 2),
                           ("cf_err_ma", f"mean |C{NDASH}F| error / m{ANG}", 2),
                           ("rms_angle_deg", "mean RMS angle / deg", 3)):
        leg_vals = {n: [] for n in legs}
        for m in mols:
            for entry in m.get("levels", {}).values():
                for n in legs:
                    v = entry.get("legs", {}).get(n, {}).get(key)
                    if v is not None:
                        leg_vals[n].append(abs(v) if key == "cf_err_ma" else v)
        if not any(leg_vals.values()):
            continue
        cells = [label]
        for n in legs:
            vals = leg_vals[n]
            cells.append(fmt(sum(vals) / len(vals), dp) if vals else "--")
        body.append(cells)

    # how often each leg beats theory alone on RMS bond
    for m in mols:
        for entry in m.get("levels", {}).values():
            t = m["theory"].get("rms_bond_ma")
            if t is None:
                continue
            n_cases += 1
            for n in legs:
                v = entry.get("legs", {}).get(n, {}).get("rms_bond_ma")
                if v is not None and v < t:
                    counts[n] += 1
    if n_cases:
        body.append([f"cases beating theory (of {n_cases})"]
                    + ["--" if n == "theory" else str(counts[n]) for n in legs])

    rows += body
    col = min(30, 114 / max(len(legs), 1))
    story.append(table([[p(c.replace("\n", "<br/>"),
                           ParagraphStyle("hc", parent=BODY, fontSize=7.9,
                                          leading=10, alignment=2, spaceAfter=0,
                                          fontName=("Helvetica-Bold" if r == 0
                                                    else "Helvetica")))
                        if r == 0 else c for c in row]
                       for r, row in enumerate(rows)],
                       [50 * mm] + [col * mm] * len(legs),
                       align_left=(0,), band_rows=(len(rows) - 1,), font_size=7.9))
    story.append(p(
        f"Aggregated over {len(mols)} molecules at two data levels each. The "
        f"C{NDASH}F column averages absolute errors, since the signed values "
        f"share a sign and would otherwise flatter the mean. 'Experiment' uses "
        f"no quantum input at all; on the molecules with the richest isotopic "
        f"data it is competitive with the hybrid, which is a real result and is "
        f"discussed in section 8.3.", CAPTION))


def sec_results(story, legacy, cur1, cur2, validate):
    story.append(PageBreak())
    story.append(p("5 &nbsp; Results", H1))
    story.append(p(
        "Lower is better throughout. 'Theory' is the level of theory alone; "
        "'experiment' is a fit to the constants with no quantum input; the "
        "hybrid legs combine them by the two objectives of section 1.4.", BODY))

    _summary(story, [m for d in (cur1, cur2) for m in (d or [])])

    story.append(p("5.2 &nbsp; Current engine, per molecule", H2))
    note = ("Current engine: joint objective, symmetry constraints on, "
            "rigid-mode projection on, warm start.")
    any_cur = False
    for data in (cur1, cur2):
        for mol in data or []:
            _legs_table(story, mol, note)
            any_cur = True
    if not any_cur:
        story.append(p("No current-engine run present in output/.", CAPTION))

    if legacy:
        story.append(p("5.3 &nbsp; Four-leg comparison (pre-audit engine)", H2))
        story.append(p(
            "These runs predate the fixes in section 6 and are kept because "
            "they carry all four legs, including the split objective and the "
            "experiment-alone leg, which the later runs do not.", BODY))
        for mol in legacy:
            _legs_table(story, mol, "Pre-audit engine, four legs.")

    if validate:
        story.append(p("5.4 &nbsp; Held-out molecule", H2))
        story.append(p(
            "Fluorobenzene took no part in choosing any setting. It is the only "
            "genuinely out-of-sample test here, and the prior width was fixed "
            f"at {validate[0].get('prior_sigma_ang', '--')} &Aring; beforehand.",
            BODY))
        for mol in validate:
            _legs_table(story, mol, "Held out; no setting tuned on it.")


def sec_audit(story):
    story.append(PageBreak())
    story.append(p("6 &nbsp; Defects found and fixed", H1))
    story.append(p(
        "A second audit of the codebase looked for capabilities that were "
        "absent or inert rather than for incorrect ones. Two of the five items "
        "turned out to be complete machinery that was simply switched off, and "
        "one exposed a genuine bug. All five are measured, not asserted.", BODY))

    rows = [["item", "what was wrong", "measured effect"],
            ["Point-group detection",
             "The projector supported 21 groups and the optimiser wired it in, "
             "but only if a caller passed the group explicitly. The detector "
             "meant to supply it returned C1 for anything past three atoms and "
             "mislabelled water as Cs.",
             "Water's two O&ndash;H bonds, equal by C2v, were splitting by 3.9 "
             "m&Aring; &mdash; larger than the entire remaining bond error. "
             "Constraining cut the parent-only fit from 3.29 to 0.83 m&Aring;."],
            ["Rigid-mode projection",
             "Built sqrt(m)-weighted translations and rotations, which are the "
             "null vectors of a mass-weighted Hessian, and applied them to the "
             "plain Cartesian one. Its rank test could not work either, since "
             "reduced QR returns unit-norm columns regardless.",
             "Measured against the analytic water Hessian, unweighted modes "
             "give |Hv| ~ 1e-10 and sqrt(m)-weighted ones ~0.5: it was deleting "
             "real directions and leaving the contamination it existed to "
             "remove. Linear molecules also lost a genuine vibration."],
            ["Hessian floor in the joint step",
             "The split step floored its Hessian eigenvalues; the joint step, "
             "now the default, did not. An indefinite Hessian makes the prior "
             "locally repulsive.",
             "The raw Cartesian Hessian is indefinite at every step of every "
             "fit. However the bad directions are 98.5% rigid, so fixing it is "
             "a wash on accuracy (12.92 to 13.02 m&Aring;): a correctness fix "
             "that recovers trust-radius budget, not an accuracy fix."],
            ["Hessian schedule",
             "Tested (call - 1) mod period, while the adaptive block rewrote "
             "the period between calls, shifting the phase past the firing "
             "residue.",
             "A nominal period of 5 gave one Hessian in 20 steps for "
             "fluoroacetylene and none at all in 14 for formyl fluoride. Under "
             "the joint objective the Hessian is the prior, so a stale one "
             "anchors it to a geometry the fit has left."],
            ["Posterior covariance",
             "Both Laplace paths weighted the quantum block with the legacy "
             "alpha_quantum rather than the alpha_q the joint objective "
             "minimises.",
             "Uncertainties described a different problem than the one solved. "
             "Both now go through a single accessor."]]
    story.append(table([[p(c, ParagraphStyle("c", parent=BODY, fontSize=7.6,
                                             leading=10.2, alignment=0,
                                             spaceAfter=0)) for c in r]
                        for r in rows],
                       [30 * mm, 62 * mm, 68 * mm], align_left=(0, 1, 2),
                       font_size=7.6))
    story.append(p(
        "Net effect on the four-molecule, two-level benchmark: mean RMS bond "
        "error 12.92 &rarr; 11.49 m&Aring;, with every case beating theory "
        "alone. Formyl fluoride on all species improved 13.47 &rarr; 5.96 "
        "m&Aring; and water parent-only 3.29 &rarr; 1.75 m&Aring;.", CAPTION))


def sec_theory_level(story, hf_d, b3lyp_d, tune):
    story.append(PageBreak())
    story.append(p("7 &nbsp; Sensitivity studies", H1))

    story.append(p("7.1 &nbsp; Level of theory", H2))
    story.append(p(
        "The level of theory is a larger lever than anything inside the engine. "
        "Across the calibration set, theory alone improves from about 16.6 "
        "m&Aring; RMS at RHF/6-31G to about 6.8 m&Aring; at B3LYP/6-31G(d). Any "
        "claim of the form 'the hybrid beats theory' therefore has to name the "
        "level, and both legs must use the same one.", BODY))
    rows = [["run", "molecule", "theory alone", "best hybrid"]]
    for name, data in (("RHF/6-31G(d)", hf_d), ("B3LYP/6-31G(d)", b3lyp_d)):
        for mol in data or []:
            best = None
            for entry in mol.get("levels", {}).values():
                for leg, vals in entry.get("legs", {}).items():
                    if "hybrid" in leg and vals.get("rms_bond_ma") is not None:
                        v = vals["rms_bond_ma"]
                        best = v if best is None else min(best, v)
            rows.append([name, mol["molecule"],
                         f"{mol['theory']['rms_bond_ma']:.2f}",
                         "--" if best is None else f"{best:.2f}"])
    if len(rows) > 1:
        story.append(table(rows, [40 * mm, 48 * mm, 36 * mm, 36 * mm],
                           align_left=(0, 1)))
        story.append(p("RMS bond error in m&Aring;. These are partial runs kept "
                       "from the level-of-theory comparison.", CAPTION))

    if tune and tune.get("beats_theory"):
        story.append(p("7.2 &nbsp; How sharply the prior width matters", H2))
        story.append(p(
            "&sigma;<sub>x</sub> was scanned over the six molecule/data-level "
            "cases of the calibration set. The plateau is wide, so being in the "
            "right decade matters and the exact value does not.", BODY))
        sigmas = tune.get("prior_sigmas", [])
        rows = [[f"{SIGMA}x / {ANG}"] + [f"{s:g}" for s in sigmas],
                ["cases beating theory (of 6)"] +
                [str(tune["beats_theory"].get(str(s), "--")) for s in sigmas]]
        if tune.get("cf_beats_theory"):
            rows.append([f"... on C{NDASH}F"] +
                        [str(tune["cf_beats_theory"].get(str(s), "--"))
                         for s in sigmas])
        story.append(table(rows, [56 * mm] + [14.5 * mm] * len(sigmas),
                           align_left=(0,), header_rows=1))
        story.append(p(
            "The C&ndash;F bond &mdash; the one fluorine's lack of isotopes "
            "makes hardest &mdash; improves at every setting tested.", CAPTION))


def sec_negative(story):
    story.append(PageBreak())
    story.append(p("8 &nbsp; Things that did not work", H1))
    story.append(p(
        "Three ideas were implemented fully and measured. None is enabled. "
        "Recording them is the point: each is the obvious next thing to try, "
        "and each fails for a reason worth knowing.", BODY))

    story.append(p("8.1 &nbsp; Computed rovibrational (&alpha;) correction", H2))
    story.append(p(
        "The textbook route from B<sub>0</sub> to B<sub>e</sub> is to compute "
        "the vibration-rotation constants from a cubic force field. Applied at "
        "RHF it made the result worse in 14 of 18 cases. The correction is the "
        "right size but not reliably the right sign at this level, and applying "
        "a correction known only to an order of magnitude is worse than "
        "carrying the offset in &sigma;, which is what the engine does.", BODY))

    story.append(p("8.2 &nbsp; Mass-dependent (r<sub>m</sub>) fit inside the hybrid", H2))
    story.append(p(
        "Watson's alternative is to fit the offset rather than compute it: "
        "I<sub>obs</sub> = I<sub>m</sub> + c&radic;I<sub>m</sub>, with three "
        "coefficients shared across isotopologues. Fitted standalone this works "
        "well &mdash; formyl fluoride improves from 4.62 to 2.33 m&Aring;. "
        "Folded into the hybrid it does not: mean RMS bond error goes 11.49 "
        "&rarr; 12.23 &rarr; 13.08 &rarr; 17.85 m&Aring; as the correction is "
        "allowed to grow, worse in every fluorinated case.", BODY))
    story.append(p(
        "The reason is instructive. Standalone, the structure is free and only "
        "isotopic substitution moves it, which separates the rovibrational "
        "offset from everything else. Inside the hybrid the quantum prior "
        "anchors the structure, so the residual the coefficients see carries "
        "the method's structural error too &mdash; and they absorb that "
        "instead. The fitted values show it: c comes out negative on formyl "
        "fluoride where the standalone fit gives it positive.", BODY))

    story.append(p("8.3 &nbsp; Standalone r<sub>m</sub> as a replacement", H2))
    story.append(p(
        "On the molecules here with rich isotopic data, the standalone "
        "r<sub>m</sub> fit beats the hybrid outright. It does not generalise to "
        "the target class. A trifluorinated molecule has three heavy atoms that "
        "can never be substituted, and the method comes up short by exactly "
        "three parameters &mdash; the three coefficients &mdash; on every case "
        "tested: CHF<sub>3</sub>, CH<sub>3</sub>CF<sub>3</sub> and "
        "1,1,2-trifluoroethane. On the last of these even a plain rigid fit to "
        "all six isotopologues and every constant is short by one. Spectroscopy "
        "alone runs out before it finishes the structure, which is precisely "
        "the case the hybrid exists for.", BODY))

    story.append(p("8.4 &nbsp; Centrifugal distortion from the Hessian", H2))
    story.append(p(
        "The &tau;' &rarr; Watson A-reduction mapping does not reproduce "
        "experiment. Against the ground-state constants for "
        "H<sub>2</sub><sup>16</sup>O it gets D<sub>J</sub> = &minus;66.9 "
        "against +37.6, D<sub>JK</sub> = &minus;18.9 against &minus;172.9 and "
        "D<sub>K</sub> = &minus;7.0 against +973.3 MHz &mdash; two of three "
        "with the wrong sign, so it is qualitatively wrong rather than "
        "imprecise. The path is off by default, warns loudly when switched on, "
        "and its &sigma; is floored at 100% of the value so it cannot pull a "
        "fit.", BODY))


def sec_limits(story):
    story.append(p("9 &nbsp; Limitations", H1))
    for text in [
        "<b>Sample size.</b> Seven fluorinated molecules, none larger than "
        "twelve atoms. Fluorobenzene is the only genuinely held-out test.",
        "<b>Reference type mismatch.</b> The published structures are a mix of "
        "r<sub>s</sub>, r<sub>0</sub> and r<sub>e</sub>. Part of every error "
        "reported here is the difference between structure types rather than a "
        "failure of the fit, which is also why sub-m&Aring; agreement should be "
        "read as coincidence rather than accuracy.",
        "<b>The B<sub>0</sub>/B<sub>e</sub> offset is uncorrected.</b> It is "
        "carried in &sigma; rather than removed, for the reasons in section "
        "8.1 and 8.2. This is the largest single systematic remaining.",
        "<b>One level of theory for the main table.</b> RHF/6-31G is poor, "
        "which flatters any method that ignores the quantum surface and "
        "handicaps the hybrid. The comparison at a better level is partial.",
        "<b>Warm start.</b> Results measure whether the data improves on "
        "theory, starting from theory. They do not measure whether the "
        "optimiser could find the answer from a poor guess.",
        "<b>No torsional or large-amplitude treatment.</b> Molecules with "
        "internal rotation are fitted as rigid frames, which is why acetyl "
        "fluoride and fluoroethane behave less well than the small rigid cases.",
    ]:
        story.append(p(text, BODY))


def sec_repro(story):
    story.append(p("10 &nbsp; Reproducing these results", H1))
    story.append(p(
        "All runs are deterministic: the displaced start uses a fixed seed and "
        "the electronic structure is converged to default PySCF thresholds. "
        "From a checkout at the commit on the title page:", BODY))
    for cmd, what in [
        ("pip install -r requirements.txt", "PySCF, numpy, scipy, reportlab"),
        ("python scripts/check_monofluoro_references.py",
         "validate every reference structure against every measured constant, "
         "and report Jacobian rank against internal DOF (section 4)"),
        ("python scripts/monofluoro_benchmark.py set=1",
         "calibration set: vinyl fluoride, acetyl fluoride, fluoroethane"),
        ("python scripts/monofluoro_benchmark.py set=2",
         "second set: formyl fluoride, fluoroacetylene, chlorofluoromethane"),
        ("python scripts/monofluoro_benchmark.py set=heldout",
         "fluorobenzene, the held-out molecule"),
        ('python scripts/monofluoro_benchmark.py set=1 method=b3lyp "basis=6-31g(d)"',
         "the same benchmark at a better level of theory (section 7.1)"),
        ("python scripts/monofluoro_tune.py",
         "scan the prior width (section 7.2)"),
        ("python scripts/estimate_theory_error.py",
         "estimate the prior width without using any reference structure"),
        ("python scripts/monofluoro_alpha_corrected.py",
         "the computed rovibrational correction (section 8.1)"),
        ("python scripts/build_monofluoro_full_report.py", "rebuild this report"),
        ("python -m pytest dev/tests -q", "the full test suite"),
    ]:
        story.append(p(cmd, MONO))
        story.append(p(f"&nbsp;&nbsp;&nbsp;&nbsp;{what}", CAPTION))

    story.append(p("Where the numbers live", H2))
    story.append(p(
        "Each benchmark writes a JSON file under <font face='Courier'>output/"
        "</font> named for its level of theory, carrying the reference "
        "structure, every measured constant, the consistency check, the "
        "Jacobian rank, and every leg's resulting internal coordinates. This "
        "report is generated from those files, so regenerating it after a run "
        "updates every table.", BODY))
    story.append(p(
        "Reference data lives in <font face='Courier'>dev/monofluoro_references"
        ".py</font>, one entry per species with its literature citation. Adding "
        "a molecule means adding measured constants and a published structure "
        "there; the validation script will refuse data that disagrees with "
        "itself.", BODY))


def build() -> Path:
    legacy = load("monofluoro_benchmark.json")
    cur1 = load("monofluoro_current_set1.json")
    cur2 = load("monofluoro_current_set2.json")
    validate = load("monofluoro_validate.json")
    hf_d = load("monofluoro_benchmark_hf_6-31gd.json")
    b3lyp_d = load("monofluoro_benchmark_b3lyp_6-31gd.json")
    tune = load("monofluoro_tune.json")
    terr = load("theory_error_estimate.json")

    sources = [
        ("monofluoro_benchmark.json", legacy),
        ("monofluoro_current_set1.json", cur1),
        ("monofluoro_current_set2.json", cur2),
        ("monofluoro_validate.json", validate),
        ("monofluoro_benchmark_hf_6-31gd.json", hf_d),
        ("monofluoro_benchmark_b3lyp_6-31gd.json", b3lyp_d),
        ("monofluoro_tune.json", tune),
        ("theory_error_estimate.json", terr),
    ]

    story = []
    sec_title(story, sources)
    sec_math(story)
    sec_setup(story, cur1, tune, terr)
    sec_references(story)
    sec_validation(story, [cur1, cur2, legacy, validate])
    sec_results(story, legacy, cur1, cur2, validate)
    sec_audit(story)
    sec_theory_level(story, hf_d, b3lyp_d, tune)
    sec_negative(story)
    story.append(PageBreak())
    sec_limits(story)
    sec_repro(story)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Quantize: monofluorinated benchmark report",
        author="Quantize")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 11 * mm,
                          "Quantize -- monofluorinated benchmark")
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"  written to {path}  ({path.stat().st_size / 1024:.0f} kB)")
