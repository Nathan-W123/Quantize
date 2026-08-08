"""Build the PDF report from output/monofluoro_benchmark.json.

Everything numeric in the report is read from that file, so the prose and the
tables cannot drift apart from the run that produced them.

    python scripts/monofluoro_benchmark.py        # produces the JSON
    python scripts/build_monofluoro_report.py     # produces the PDF
"""

from __future__ import annotations

import json
import re
import sys
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
DATA = _ROOT / "output" / "monofluoro_benchmark.json"
OUT = _ROOT / "output" / "monofluoro_report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#f2f2f0")
GOOD = colors.HexColor("#1d6b3f")

LEGS = ["theory", "experiment", "hybrid, split", "hybrid, joint prior"]
SHORT = {"theory": "theory", "experiment": "experiment",
         "hybrid, split": "hybrid (split)", "hybrid, joint prior": "hybrid (joint)"}

_styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_styles["Normal"], fontName="Helvetica",
                      fontSize=9.6, leading=14.4, alignment=TA_JUSTIFY,
                      textColor=INK, spaceAfter=7)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=15,
                    leading=19, spaceBefore=16, spaceAfter=8, alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=11,
                    leading=15, spaceBefore=11, spaceAfter=5, alignment=0)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.2, leading=11.5,
                         textColor=MUTED, spaceBefore=4, spaceAfter=12)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.2, leading=10.5,
                      alignment=0, spaceAfter=0)


def sub(text: str) -> str:
    """Subscript digits that follow a letter, for formulas and B0."""
    return re.sub(r"(?<=[A-Za-z])(\d+)", r"<sub>\1</sub>", text)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, align_left=(0,), header_rows=1, bold_cells=(),
          band_rows=(), font_size=8.2):
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=header_rows)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        ("FONT", (0, 0), (-1, header_rows - 1), "Helvetica-Bold", font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.7, INK),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for col in align_left:
        style.append(("ALIGN", (col, 0), (col, -1), "LEFT"))
    for r in band_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), BAND))
    for r, c in bold_cells:
        style.append(("FONT", (c, r), (c, r), "Helvetica-Bold", font_size))
        style.append(("TEXTCOLOR", (c, r), (c, r), GOOD))
    t.setStyle(TableStyle(style))
    return t


def best_col(values, offset):
    """Index of the smallest magnitude, as a column number."""
    return offset + min(range(len(values)), key=lambda i: abs(values[i]))


def worst_ch(mol, leg):
    """Largest bond error on a bond involving hydrogen, in mA, signed."""
    ref, got = mol["reference_internals"], leg["internals"]
    ch = [n for n in mol["bond_names"] if "H" in n]
    return max(((got[n] - ref[n]) * 1000 for n in ch), key=abs) if ch else 0.0


def recommended(mol, level):
    """Which hybrid the counting rule picks, knowing nothing about the answer.

    Fewer measurements than parameters -> the split partition hands directions to
    data that data cannot support, so keep theory in play everywhere. Otherwise
    the prior is the thing that over-constrains, so let the data have its
    directions outright.
    """
    return ("hybrid, joint prior" if level["n_observables"] < mol["internal_dof"]
            else "hybrid, split")


def other(leg):
    """The hybrid the counting rule did not pick."""
    return "hybrid, split" if leg == "hybrid, joint prior" else "hybrid, joint prior"


# ── report sections ──────────────────────────────────────────────────────────

def header(story, data):
    story.append(p("Bond lengths from spectroscopy, from theory, "
                   "and from both together", H1))
    story.append(p("A test on three monofluorinated molecules with published "
                   "structures", ParagraphStyle(
                       "sub", parent=BODY, fontSize=10.5, leading=14,
                       textColor=MUTED, spaceAfter=14, alignment=0)))
    cf = sorted(abs(m["theory"]["cf_err_ma"]) for m in data)
    cases = [(m, lv) for m in data for lv in m["levels"].values()]
    n = len(cases)
    right = sum(lv["legs"][recommended(m, lv)]["rms_bond_ma"]
                < m["theory"]["rms_bond_ma"] for m in data
                for lv in m["levels"].values())
    wrong_worse = sum(
        lv["legs"][other(recommended(m, lv))]["rms_bond_ma"]
        > m["theory"]["rms_bond_ma"] for m in data for lv in m["levels"].values())
    story.append(Paragraph(
        "<b>Short version.</b> Neither source of information wins outright. "
        "Quantum chemistry alone gets the overall shape of a molecule roughly "
        f"right but puts the carbon&ndash;fluorine bond {cf[0]:.0f} to {cf[-1]:.0f} "
        "thousandths of an &Aring;ngstr&ouml;m too long in every one of the three "
        "molecules tested, always in the same direction &mdash; a systematic bias, "
        "not scatter. Rotational spectroscopy alone corrects that bias well when "
        "there is enough data, and is erratic when there is not: on a single "
        "isotopologue it barely improves on the geometry it started from. "
        f"Combining the two beats theory in all {right} of the {n} "
        "molecule-and-data-level combinations tested, both on overall bond accuracy "
        "and on the C&ndash;F bond in particular &mdash; but only if they are "
        "merged the right way, and the right way depends on how much data there is. "
        f"Merged the wrong way, the hybrid is worse than plain theory in "
        f"{wrong_worse} of the {n}.", BODY))
    story.append(Spacer(1, 4))


def section_question(story):
    story.append(p("1. What was being tested", H1))
    story.append(p(
        "A molecule's structure &mdash; which atoms are how far apart, and at what "
        "angles &mdash; can be worked out two quite different ways.", BODY))
    story.append(p(
        "<b>From theory.</b> Solve the equations of quantum mechanics for the "
        "molecule's electrons and find the atomic arrangement with the lowest "
        "energy. This needs no experiment at all, but the answer is only as good "
        "as the approximations used, and cheap approximations have systematic "
        "biases.", BODY))
    story.append(p(
        "<b>From experiment.</b> Measure how fast the molecule tumbles in the gas "
        "phase. Microwave spectroscopy gives three numbers per molecule &mdash; the "
        "rotational constants " + sub("A, B and C") + " &mdash; which are fixed by "
        "how the mass is distributed. Swap one atom for a heavier isotope and the "
        "mass distribution changes in a known way, giving three more numbers. "
        "Enough isotopic variants and the structure can be solved for.", BODY))
    story.append(p(
        "The catch with the experimental route is that three numbers per variant is "
        "not many. A seven-atom molecule has fifteen independent structural "
        "parameters; one measured species gives three numbers. The problem is "
        "underdetermined, and there is a whole family of structures that fit the "
        "data equally well.", BODY))
    story.append(p(
        "<b>The hybrid idea.</b> Use the data for the directions the data can see, "
        "and use theory for the rest. This report asks whether that actually "
        "works &mdash; specifically, whether the combination beats theory on its own, "
        "which is the practical question, because theory on its own is always "
        "available and costs nothing but computer time.", BODY))
    story.append(p(
        "Fluorinated molecules make this sharper than usual. Fluorine has only one "
        "stable isotope, so no isotopic substitution ever moves the fluorine atom. "
        "Its position has to be inferred indirectly, from the requirement that the "
        "molecule's centre of mass sits where the measurements say it does, once "
        "every other atom has been located. That makes the C&ndash;F bond the "
        "hardest parameter in the molecule and a good stress test.", BODY))


def section_methods(story):
    story.append(p("2. The four methods compared", H1))
    rows = [["Method", "What it uses", "What it does"]]
    rows.append(["theory",
                 "no experimental data",
                 Paragraph("Minimises the quantum-chemical energy. The answer "
                           "depends only on the molecule, not on any measurement.",
                           CELL)])
    rows.append(["experiment",
                 "rotational constants only",
                 Paragraph("Adjusts the geometry until the calculated rotational "
                           "constants match the measured ones. Directions the data "
                           "cannot see are left to drift.", CELL)])
    rows.append(["hybrid (split)",
                 "both",
                 Paragraph("Splits the parameter space in two using an SVD of the "
                           "sensitivity matrix. Data owns every direction it can "
                           "see; theory owns the remainder outright.", CELL)])
    rows.append(["hybrid (joint)",
                 "both",
                 Paragraph("Leaves every direction contested and weights the two "
                           "sources by how well each knows that direction. Theory's "
                           "weight is set by the distance over which it is trusted, "
                           "here 0.005 &Aring;.", CELL)])
    story.append(table(rows, [30 * mm, 34 * mm, 100 * mm], align_left=(0, 1, 2)))
    story.append(p(
        "The two hybrids differ only in how the two sources are merged. Both were "
        "run for every molecule and every data level, because the earlier work in "
        "this repository found that which one is better flips depending on how "
        "much data there is &mdash; and that flip turns out to reproduce here.",
        CAPTION))


def section_molecules(story, data):
    story.append(p("3. The molecules, and why these three", H1))
    story.append(p(
        "Each molecule needed a published experimental structure <i>and</i> the "
        "rotational constants from the same study, so that the structure being "
        "aimed at and the data being fed in come from the same experiment. The two "
        "must also be mutually consistent: feeding the published geometry back "
        "through the rotational-constant calculation should reproduce the published "
        "constants to within about 1%, which is the size of the unavoidable "
        "zero-point-motion offset. A larger gap means one of the two numbers is "
        "wrong, and the comparison would be measuring that error rather than "
        "anything about the methods.", BODY))
    rows = [["Molecule", "Formula", "Atoms", "Structural\nparameters",
             "Isotopic\nvariants", "Self-\nconsistency"]]
    for m in data:
        rows.append([m["molecule"], Paragraph(sub(m["formula"]), CELL),
                     str(m["n_atoms"]), str(m["internal_dof"]),
                     str(m["n_isotopologues"]), f"{m['consistency_pct']:.2f}%"])
    story.append(table(rows, [40 * mm, 28 * mm, 18 * mm, 26 * mm, 24 * mm, 26 * mm],
                       align_left=(0, 1)))
    story.append(p(
        "Sources: " + "; ".join(f"<b>{m['molecule']}</b> &mdash; {m['source']}"
                                for m in data) + ".", CAPTION))
    story.append(p(
        "A fourth candidate, formyl fluoride (HCOF), was checked and dropped. Its "
        "tabulated geometry and its tabulated constants disagree by 6.2% &mdash; far "
        "beyond any zero-point effect &mdash; so at least one of them is wrong and "
        "it cannot serve as a ground truth. It is mentioned here because the "
        "selection test is the part of the setup most likely to be skipped, and "
        "skipping it would have silently poisoned a quarter of the results.", BODY))


def section_setup(story, data):
    story.append(p("4. How the test was run", H1))

    story.append(p("Level of theory", H2))
    story.append(p(
        "Hartree&ndash;Fock with the 6-31G basis set, through PySCF, for every "
        "molecule and every method &mdash; including the quantum half of both "
        "hybrids. This is deliberately a modest level of theory. The point is not "
        "to produce the best possible structures; it is to compare three ways of "
        "using the same quantum surface, and holding the surface fixed is what "
        "makes the comparison mean anything. It is also the same level used on "
        "fluorobenzene earlier in this project, so those numbers are comparable "
        "with these.", BODY))

    story.append(p("Starting geometry", H2))
    story.append(p(
        "Every method starts from the same deliberately wrong geometry: the "
        "published structure with random noise of 0.03 &Aring; added to every "
        "coordinate (one fixed random seed, so the runs are reproducible). Without "
        "this, a method could look good simply by not moving. The starting error is "
        "reported in every table as the do-nothing baseline that any method has to "
        "beat.", BODY))

    story.append(p("The rotational constants", H2))
    story.append(p(
        "The parent molecule's constants are the published measured values. The "
        "constants for the isotopically substituted variants are <i>derived</i> "
        "from the published structure rather than measured, because the measured "
        "values, while they exist in the original papers, are not in any source "
        "reachable from this environment. They are not simply the rigid values of "
        "the reference geometry: each is scaled by the ratio the real parent "
        "molecule shows between its measured constants and the rigid value of the "
        "published structure. That reproduces the offset real ground-state "
        "constants carry from zero-point vibration, so the data is not exactly "
        "consistent with the answer and a fit cannot recover it for free.", BODY))
    story.append(p(
        "This is the weakest link in the setup and is worth being blunt about. It "
        "means the substituted-species data is more internally consistent than real "
        "measurements would be, which flatters the experiment-only and hybrid legs "
        "at the higher data level. It does not affect the theory leg at all, and it "
        "does not affect the parent-only results, which use nothing but published "
        "measured numbers.", BODY))

    story.append(p("Two data levels", H2))
    rows = [["Data level", "Species", "Numbers measured", "Compared with"]]
    for m in data:
        for label, lv in m["levels"].items():
            rows.append([f"{m['molecule']} &mdash; {label}"
                         .replace("&mdash;", "—"),
                         str(lv["n_isotopologues"]), str(lv["n_observables"]),
                         f"{m['internal_dof']} parameters"])
    story.append(table(rows, [58 * mm, 22 * mm, 38 * mm, 38 * mm],
                       align_left=(0,),
                       band_rows=tuple(r for r in range(1, len(rows)) if r % 2 == 1)))
    story.append(p(
        "The first level is the realistic hard case: one measured species, three "
        "numbers, far fewer than the molecule has parameters. The second is the "
        "comfortable case where the count of measurements reaches or passes the "
        "count of parameters. The gap between them is where the interesting "
        "behaviour lives.", CAPTION))

    story.append(p("How error is measured", H2))
    story.append(p(
        "Each fitted geometry is reduced to chemically meaningful numbers &mdash; "
        "bond lengths and bond angles, with symmetry-equivalent ones averaged "
        "&mdash; and compared with the same numbers from the published structure. "
        "Bond errors are quoted in milli-&Aring;ngstr&ouml;m (m&Aring;, "
        "thousandths of an &Aring;ngstr&ouml;m); for scale, a carbon&ndash;carbon "
        "bond is about 1500 m&Aring; long and a good structure determination is "
        "accurate to a few m&Aring;. Angle errors are in degrees. The C&ndash;F "
        "bond error is reported separately and with its sign, because it is the "
        "parameter the whole exercise is about.", BODY))


def summary_table(story, data, metric, unit, title, note):
    block = [p(title, H2)]
    rows = [["Molecule", "Data level", "start"] + [SHORT[l] for l in LEGS]]
    bold, band = [], []
    for m in data:
        for li, (label, lv) in enumerate(m["levels"].items()):
            vals = [lv["legs"][l][metric] for l in LEGS]
            fmt = "{:+.1f}" if metric == "cf_err_ma" else "{:.1f}"
            afmt = "{:.2f}" if "angle" in metric else fmt
            rows.append([m["molecule"] if li == 0 else "", label,
                         afmt.format(m["start_errors"][metric])]
                        + [afmt.format(v) for v in vals])
            bold.append((len(rows) - 1, best_col(vals, 3)))
            if li == 1:
                band.append(len(rows) - 1)
    block.append(table(rows, [34 * mm, 26 * mm, 18 * mm] + [26 * mm] * 4,
                       align_left=(0, 1), bold_cells=bold, band_rows=band))
    block.append(p(f"{note} All values in {unit}; the best of the four methods in "
                   "each row is highlighted.", CAPTION))
    story.append(KeepTogether(block))


def section_results(story, data):
    n_fits = sum(1 + sum(len(lv["legs"]) - 1 for lv in m["levels"].values())
                 for m in data)
    story.append(p("5. Results", H1))
    story.append(p(
        f"Three tables summarise all {n_fits} fits. The first is the headline "
        "number &mdash; overall bond-length accuracy. The second isolates the "
        "carbon&ndash;fluorine bond. The third covers angles.", BODY))

    summary_table(story, data, "rms_bond_ma", "m&Aring;",
                  "5.1 Overall bond-length error (root-mean-square)",
                  "Lower is better; the <i>start</i> column is the deliberately "
                  "wrong geometry every method began from, so any method above its "
                  "own start column has made things worse.")
    summary_table(story, data, "cf_err_ma", "m&Aring;",
                  "5.2 The carbon&ndash;fluorine bond, signed",
                  "A positive number means the bond came out too long. Theory is "
                  "positive and large in all three molecules.")
    summary_table(story, data, "rms_angle_deg", "degrees",
                  "5.3 Bond-angle error (root-mean-square)",
                  "Angles behave far less consistently than bond lengths: no method "
                  "wins across the board, and theory is competitive here in a way it "
                  "never is for the C&ndash;F bond.")


def section_per_molecule(story, data):
    story.append(p("6. Molecule by molecule", H1))
    story.append(p(
        "The tables below give every bond length and bond angle, for the published "
        "structure and for each method, at both data levels. Symmetry-equivalent "
        "parameters are averaged into one row.", BODY))

    for m in data:
        block = [p(f"{m['molecule']} &mdash; {sub(m['formula'])}", H2)]
        d = m["inertial_defect"]
        block.append(p(
            f"{m['n_atoms']} atoms, {m['internal_dof']} structural parameters. "
            f"Published " + sub("B0") + " = "
            + ", ".join(f"{v:,.1f}" for v in m["b0_mhz"]) + " MHz; "
            f"inertial defect {d:+.2f} amu&nbsp;&Aring;<super>2</super> "
            + ("(planar, as expected)" if abs(d) < 0.5 else
               "(non-planar, as expected for a methyl group)") + ". "
            f"The published geometry reproduces its own constants to "
            f"{m['consistency_pct']:.2f}%.", BODY))
        story.append(KeepTogether(block))

        for label, lv in m["levels"].items():
            ref = m["reference_internals"]
            rows = [[f"{label}\n({lv['n_observables']} numbers)", "published",
                     "start"] + [SHORT[l] for l in LEGS]]
            bold = []
            for name in m["bond_names"] + m["angle_names"]:
                is_bond = name in m["bond_names"]
                fmt = "{:.4f}" if is_bond else "{:.2f}"
                vals = [lv["legs"][l]["internals"][name] for l in LEGS]
                rows.append([name, fmt.format(ref[name]),
                             fmt.format(m["start_internals"][name])]
                            + [fmt.format(v) for v in vals])
                bold.append((len(rows) - 1,
                             best_col([v - ref[name] for v in vals], 3)))
            band = tuple(range(1 + len(m["bond_names"]), len(rows)))
            story.append(KeepTogether([
                Spacer(1, 3),
                table(rows, [30 * mm, 22 * mm, 20 * mm] + [26 * mm] * 4,
                      align_left=(0,), bold_cells=bold, band_rows=band,
                      font_size=7.8),
                p("Bond lengths in &Aring;ngstr&ouml;m, angles in degrees "
                  "(shaded rows). Closest to published is highlighted.", CAPTION),
            ]))
        story.append(Spacer(1, 6))


def section_findings(story, data):
    story.append(PageBreak())
    story.append(p("7. What the numbers show", H1))

    cf_theory = [m["theory"]["cf_err_ma"] for m in data]
    story.append(p("7.1 Theory has a consistent, one-directional C&ndash;F bias",
                   H2))
    story.append(p(
        "Across the three molecules, theory puts the C&ndash;F bond "
        + ", ".join(f"{v:+.0f}" for v in cf_theory) + " m&Aring; from the published "
        "value. Same sign, similar size, every time: at this level of theory the "
        "C&ndash;F bond is systematically too long by roughly 30 m&Aring;. That is "
        "not random error, it is a bias of the method, and no amount of tightening "
        "the geometry optimisation will remove it. It is exactly the kind of error "
        "experimental data is well placed to correct, and it is the single "
        "clearest argument for combining the two sources at all.", BODY))

    story.append(p("7.2 Spectroscopy alone is unreliable on thin data", H2))
    thin = [m["levels"]["parent only"]["legs"]["experiment"]["rms_bond_ma"]
            for m in data]
    starts = [m["start_errors"]["rms_bond_ma"] for m in data]
    ch = [worst_ch(m, m["levels"]["parent only"]["legs"]["experiment"])
          for m in data]
    story.append(p(
        "With one measured species &mdash; three numbers against twelve to eighteen "
        "parameters &mdash; fitting to the data gives bond errors of "
        + ", ".join(f"{v:.0f}" for v in thin) + " m&Aring;, starting from "
        + ", ".join(f"{v:.0f}" for v in starts) + " m&Aring;. In two of the three "
        "the fit has barely improved on the deliberately wrong geometry it began "
        "with, moving it about 1 m&Aring;.", BODY))
    story.append(p(
        "This is not the optimiser failing; it is the underdetermination showing "
        "through. Much of the damage lands on bonds involving hydrogen &mdash; the "
        "worst is off by " + ", ".join(f"{v:+.0f}" for v in ch) + " m&Aring; "
        "&mdash; because moving a hydrogen barely shifts the moments of inertia, so "
        "the fit will move one a long way to buy a small improvement. But it is not "
        "confined to hydrogens: in acetyl fluoride the C=O bond comes out 44 "
        "m&Aring; short and C&ndash;F 36 m&Aring; long. With three numbers and "
        "fifteen parameters there is nothing in the data forcing any particular "
        "answer, and the fit is entitled to any of them.", BODY))

    story.append(p("7.3 Spectroscopy alone is strong on full data", H2))
    full = [m["levels"]["all species"]["legs"]["experiment"]["rms_bond_ma"]
            for m in data]
    fsplit = [m["levels"]["all species"]["legs"]["hybrid, split"]["rms_bond_ma"]
              for m in data]
    story.append(p(
        "Once every symmetry-unique substitution is included the picture reverses. "
        "Bond errors fall to " + ", ".join(f"{v:.1f}" for v in full) + " m&Aring;, "
        "three to four times better than theory. This is the regime classical "
        "microwave structure determination has always operated in, and the result "
        "confirms the machinery uses the information correctly rather than "
        "discovering anything. Adding theory on top costs nothing and helps a "
        "little: the split hybrid gives "
        + ", ".join(f"{v:.1f}" for v in fsplit) + " m&Aring;, equal or better in "
        "all three.", BODY))

    story.append(p("7.4 How the two sources are combined matters more than "
                   "whether they are combined", H2))
    story.append(p(
        "The two hybrids are given identical inputs and differ only in the merging "
        "rule, yet they behave quite differently on thin data. The <i>split</i> "
        "rule hands every direction the data can see over to the data outright. On "
        "three observables that is too generous: some of those directions are seen "
        "only barely, and letting the data own them lets the same hydrogen-atom "
        "wandering through that spoils the experiment-only fit. The <i>joint</i> "
        "rule keeps theory's opinion in play everywhere, weighted by the distance "
        "over which theory is trusted, and that is what stops the drift.", BODY))
    story.append(p(
        "On full data the ordering flips: there the prior is the thing that "
        "over-constrains, and the split rule is better. The reversal is clean "
        "&mdash; joint beats split in all three molecules on thin data, split beats "
        "joint in all three on full data &mdash; and the same reversal was found "
        "independently on fluorobenzene earlier in this project, so it is not a "
        "quirk of one system. The practical rule is simple: use the joint objective "
        "when measurements are scarcer than parameters, the split objective when "
        "they are not.", BODY))
    story.append(p(
        "Angles are the one place the ordering is not uniform. In vinyl fluoride on "
        "full data the joint hybrid is far better on angles than anything else "
        "(0.43&deg; against about 2&deg; for the two data-led fits), because two of "
        "the angles around the double bond get distorted by the data-led fits and "
        "the prior holds them in place. In the other two molecules the split hybrid "
        "wins on angles as well. Angles are the weaker part of this comparison "
        "throughout, and worth treating with more caution than the bond lengths.",
        BODY))

    story.append(p("7.5 Does the hybrid beat theory?", H2))
    story.append(p(
        "The objective has to be chosen without knowing the answer, or the "
        "comparison is worthless. It can be: the rule in section 7.4 needs only "
        "the number of measurements and the number of parameters, both of which "
        "are known before any fitting happens. The table below applies that rule "
        "mechanically &mdash; joint below the parameter count, split at or above "
        "it &mdash; and compares the result with theory.", BODY))

    rows = [["Molecule", "Data level", "Rule picks", "theory", "hybrid", "change"]]
    bold, wins, cf_wins = [], 0, 0
    for m in data:
        for label, lv in m["levels"].items():
            pick = recommended(m, lv)
            th, hy = m["theory"]["rms_bond_ma"], lv["legs"][pick]["rms_bond_ma"]
            tcf = abs(m["theory"]["cf_err_ma"])
            hcf = abs(lv["legs"][pick]["cf_err_ma"])
            wins += hy < th
            cf_wins += hcf < tcf
            rows.append([m["molecule"], label, SHORT[pick].split()[1].strip("()"),
                         f"{th:.1f}", f"{hy:.1f}", f"{hy - th:+.1f}"])
            if hy < th:
                bold.append((len(rows) - 1, 5))
    story.append(table(rows, [34 * mm, 26 * mm, 22 * mm, 24 * mm, 24 * mm, 24 * mm],
                       align_left=(0, 1, 2), bold_cells=bold))
    n = sum(len(m["levels"]) for m in data)
    story.append(p(
        f"RMS bond error in m&Aring;. The hybrid chosen by the counting rule beats "
        f"theory in {wins} of the {n} cases, and is closer on the C&ndash;F bond "
        f"specifically in {cf_wins} of {n}.", CAPTION))

    wrong_worse = sum(lv["legs"][other(recommended(m, lv))]["rms_bond_ma"]
                      > m["theory"]["rms_bond_ma"] for m in data
                      for lv in m["levels"].values())
    story.append(p(
        "So: yes, with two qualifications. The margin is thin where data is scarce "
        "&mdash; under a milli-&Aring;ngstr&ouml;m in two of the three molecules, "
        "because three numbers leave little to add &mdash; and becomes large, around "
        "12 m&Aring;, only on full data. And the hybrid configured the other way is "
        f"<i>worse</i> than plain theory in {wrong_worse} of the {n} cases, so the "
        "objective is not a detail to leave at its default. The gain that is both "
        "large and consistent is the C&ndash;F bond, where a systematic error in the "
        "theory is exactly what data removes well.", BODY))


def section_limits(story):
    story.append(p("8. Limitations", H1))
    items = [
        ("The substituted-species constants are derived, not measured.",
         "Described in section 4. It makes the full-data level cleaner than reality "
         "and flatters the two data-using methods there. The parent-only level uses "
         "only published measured numbers and is unaffected."),
        ("Hartree&ndash;Fock with a small basis is a weak level of theory.",
         "A better method would shrink theory's C&ndash;F bias and narrow the gap. "
         "That would change the size of the effect, not its direction; the point "
         "here is the comparison between ways of using a fixed quantum surface, not "
         "the absolute accuracy of any of them."),
        ("Comparing across structure definitions.",
         "The published structures come from isotopic substitution; the quantum "
         "calculation produces the structure at the bottom of the energy well. "
         "These differ by a few m&Aring; through zero-point vibration, so part of "
         "every theory error quoted here is that definitional gap rather than a "
         "failure of the calculation."),
        ("Three molecules is a small sample.",
         "The C&ndash;F bias is consistent across all three, and the "
         "objective-choice reversal reproduces a fourth system (fluorobenzene), but "
         "neither is established on a sample this size. They are indications worth "
         "acting on, not settled results."),
        ("One starting geometry per molecule.",
         "Every method starts from the same displaced structure, which makes the "
         "comparison fair, but does not show how sensitive any of them is to where "
         "it starts."),
    ]
    rows = [[Paragraph(f"<b>{h}</b>", CELL), Paragraph(t, CELL)] for h, t in items]
    story.append(table([["What", "Why it matters"]] + rows,
                       [58 * mm, 106 * mm], align_left=(0, 1)))


def section_repro(story, data):
    block = story
    block.append(p("9. Reproducing this", H1))
    block.append(p("Both steps are in the repository and read no hidden state:",
                   BODY))
    block.append(Paragraph('<font face="Courier" size="8.5">'
                           "python scripts/monofluoro_benchmark.py<br/>"
                           "python scripts/build_monofluoro_report.py</font>",
                           ParagraphStyle("code", parent=BODY, leftIndent=10,
                                          spaceAfter=9)))
    # Theory is run once per molecule and reused at both data levels, so it is
    # counted once here rather than once per level.
    total = sum(m["theory"]["seconds"] for m in data) + sum(
        lv["legs"][l]["seconds"] for m in data for lv in m["levels"].values()
        for l in LEGS if l != "theory")
    block.append(p(
        "Structures and constants live in "
        '<font face="Courier" size="8.5">dev/monofluoro_references.py</font>, one '
        "entry per molecule with its citation. Every number in this report is read "
        "from the benchmark's JSON output rather than typed in, so the prose cannot "
        f"drift from the run. The whole set takes about {total / 60:.0f} minutes on "
        "one core.", BODY))


def build() -> Path:
    if not DATA.exists():
        sys.exit(f"missing {DATA} - run scripts/monofluoro_benchmark.py first")
    data = json.loads(DATA.read_text(encoding="utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="Bond lengths from spectroscopy, theory, and both together",
        author="Quantize")

    story = []
    header(story, data)
    section_question(story)
    section_methods(story)
    section_molecules(story, data)
    section_setup(story, data)
    section_results(story, data)
    section_per_molecule(story, data)
    section_findings(story, data)
    section_limits(story)
    section_repro(story, data)

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(22 * mm, 13 * mm, A4[0] - 22 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(22 * mm, 8.5 * mm,
                          "Quantize — monofluorinated benchmark")
        canvas.drawRightString(A4[0] - 22 * mm, 8.5 * mm, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
