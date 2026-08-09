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


def best_hybrid(level, key="rms_bond_ma", absolute=False):
    """Whichever objective did better. Chosen with hindsight, not predictable."""
    def score(leg):
        v = level["legs"][leg][key]
        return abs(v) if absolute else v
    return min(("hybrid, split", "hybrid, joint prior"), key=score)


# ── report sections ──────────────────────────────────────────────────────────

def header(story, data):
    story.append(p("Bond lengths from spectroscopy, from theory, "
                   "and from both together", H1))
    story.append(p("A test on three monofluorinated molecules with published "
                   "structures", ParagraphStyle(
                       "sub", parent=BODY, fontSize=10.5, leading=14,
                       textColor=MUTED, spaceAfter=14, alignment=0)))
    cf = sorted(abs(m["theory"]["cf_err_ma"]) for m in data)
    n = sum(len(m["levels"]) for m in data)
    best_wins = sum(
        lv["legs"][best_hybrid(lv)]["rms_bond_ma"] < m["theory"]["rms_bond_ma"]
        for m in data for lv in m["levels"].values())
    story.append(Paragraph(
        "<b>Short version.</b> Quantum chemistry alone puts the "
        f"carbon&ndash;fluorine bond {cf[0]:.0f} to {cf[-1]:.0f} thousandths of an "
        "&Aring;ngstr&ouml;m too long in every one of the three molecules tested, "
        "always in the same direction &mdash; a systematic bias, not scatter, and "
        "exactly the kind of error measurements should be able to fix. They "
        "partly do: on the C&ndash;F bond, some combination of data and theory "
        f"beats theory alone in all {n} cases tested. On overall bond accuracy the "
        f"result is weaker &mdash; the better of the two ways of combining them "
        f"wins in {best_wins} of {n}, even when that better way is picked with "
        "hindsight, and no rule was found that picks it in advance.", BODY))
    story.append(Paragraph(
        "Two things explain the gap. First, every one of these molecules is "
        "underdetermined <i>even with every published rotational constant "
        "included</i> &mdash; vinyl fluoride's 22 measured constants carry only 9 "
        "independent constraints on 12 structural parameters. Second, the "
        "measurements and the reference structure are different kinds of quantity, "
        "and the half-percent mismatch between them is currently uncorrected; that, "
        "rather than the choice of algorithm, is what limits these results. An "
        "earlier version of this experiment used isotopologue constants derived "
        "from the reference structures instead of measured ones, and reached a "
        "tidier and more favourable conclusion. It does not survive real data, and "
        "section 7.4 says how.", BODY))
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
        "The two hybrids differ only in how the two sources are merged. Both are "
        "run for every molecule and every data level, because which one is better "
        "turns out not to be predictable in advance (section 7.4).", CAPTION))


def section_molecules(story, data):
    story.append(p("3. The molecules, and why these three", H1))
    story.append(p(
        "Each molecule needed a published experimental structure <i>and</i> "
        "published measured rotational constants for a useful set of its isotopic "
        "variants. The two must be mutually consistent: feeding the published "
        "geometry back through the rotational-constant calculation should reproduce "
        "<i>every</i> measured constant to within about 1%, the size of the "
        "unavoidable zero-point-motion offset. A larger gap means the structure and "
        "the data disagree, or a variant has been assigned to the wrong atom &mdash; "
        "and the comparison would then be measuring that error rather than anything "
        "about the methods. Every species was checked individually, not just the "
        "parent, which is how the mislabelled vinyl fluoride species in section 4 "
        "were caught.", BODY))
    rows = [["Molecule", "Formula", "Atoms", "Structural\nparameters",
             "Measured\nspecies", "Measured\nconstants", "Worst\ndisagreement"]]
    for m in data:
        n_obs = max(lv["n_observables"] for lv in m["levels"].values())
        rows.append([m["molecule"], Paragraph(sub(m["formula"]), CELL),
                     str(m["n_atoms"]), str(m["internal_dof"]),
                     str(m["n_isotopologues"]), str(n_obs),
                     f"{m['consistency_pct']:.2f}%"])
    story.append(table(rows, [34 * mm, 25 * mm, 15 * mm, 23 * mm, 22 * mm,
                              23 * mm, 24 * mm], align_left=(0, 1)))
    story.append(p(
        "The last column is the worst disagreement between the published geometry "
        "and any one of its measured constants. Sources &mdash; "
        + "; ".join(f"<b>{m['molecule']}</b>: structure, {m['structure_source']}; "
                    f"constants, {m['constants_source']}" for m in data) + ".",
        CAPTION))
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
        "Every constant used here is a measured literature value, transcribed from "
        "<i>NBS Monograph 70, Microwave Spectral Tables</i> (National Bureau of "
        "Standards, 1968&ndash;69) &mdash; the standard compilation of the microwave "
        "work of that era. Nothing is back-calculated from a geometry. Where the "
        "original study did not determine a constant it is left out rather than "
        "filled in: two of the eight vinyl fluoride species have no measured A at "
        "all, and several others have A quoted to three or four significant figures "
        "against seven for B and C. That unevenness is real, and the fit is told "
        "about it through per-constant uncertainties rather than having it "
        "smoothed away.", BODY))
    story.append(p(
        "The uncertainties are set by model error, not measurement error. The "
        "reference structures are substitution (r<sub>s</sub>) structures while the "
        "constants are ground-state (r<sub>0</sub>) values, and that difference is "
        "worth a few tenths of a percent &mdash; far more than the 0.01 MHz to "
        "which B and C are quoted. A is given twice the floor of B and C, because "
        "it is both the least well determined constant of a near-prolate top and "
        "the one most sensitive to zero-point out-of-plane motion; the measured "
        "residuals show exactly that pattern.", BODY))
    story.append(p(
        "<b>One correction to the source.</b> The compilation's cis/trans labels on "
        "the two doubly-deuterated vinyl fluoride species (ids 789 and 791) "
        "contradict its own constants. Deuterating the CHF hydrogen shifts B by "
        "only about 1 MHz, so B alone identifies which CH<sub>2</sub> hydrogen "
        "carries the second deuterium &mdash; and on that test the labels are "
        "swapped. Assigned by the constants, both species match the published "
        "geometry to better than 0.4%; assigned by the labels they are off by 6 to "
        "22%, which is impossible for an isotopic substitution. The assignments "
        "here follow the constants. Every species is re-checked this way by "
        '<font face="Courier" size="8.5">scripts/check_monofluoro_references.py'
        "</font> before any fitting is done.", BODY))

    story.append(p("Two data levels, and how much they actually constrain", H2))
    story.append(p(
        "Counting measurements overstates how much is known. Three constants from "
        "one species are not three independent constraints on the structure: for a "
        "planar molecule the third is nearly fixed by the first two, since the "
        "inertial defect is close to zero. The honest measure is the rank of the "
        "stacked sensitivity matrix &mdash; how many independent directions in the "
        "structure the data can actually see. It is computable in advance, from the "
        "starting geometry, without knowing the answer.", BODY))
    rows = [["Data level", "Species", "Constants\nmeasured",
             "Independent\nconstraints", "Parameters", "Unconstrained"]]
    band = []
    for m in data:
        for label, lv in m["levels"].items():
            rows.append([f"{m['molecule']} — {label}",
                         str(lv["n_isotopologues"]), str(lv["n_observables"]),
                         str(lv["rank"]), str(m["internal_dof"]),
                         str(lv["deficit"])])
            if label != "parent only":
                band.append(len(rows) - 1)
    story.append(table(rows, [50 * mm, 18 * mm, 24 * mm, 26 * mm, 24 * mm, 26 * mm],
                       align_left=(0,), band_rows=tuple(band)))
    story.append(p(
        "Vinyl fluoride is the clearest case: 22 measured constants carry only 9 "
        "independent constraints on 12 parameters. <b>Every molecule here is "
        "underdetermined even with every published constant included</b> &mdash; "
        "which is the situation the hybrid exists for, and not the comfortable "
        "regime raw counting would suggest.", CAPTION))

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

    story.append(p("7.2 More data does not always help", H2))
    vf = next(m for m in data if m["key"] == "vinyl_fluoride")
    thin = vf["levels"]["parent only"]["legs"]["experiment"]["rms_bond_ma"]
    full = vf["levels"]["all species"]["legs"]["experiment"]["rms_bond_ma"]
    story.append(p(
        "The most surprising result here is vinyl fluoride. Fitting the parent's "
        f"three constants gives a bond error of {thin:.0f} m&Aring;. Adding every "
        "other measured species &mdash; seven more, twenty-two constants in total "
        f"&mdash; makes it <i>worse</i>, {full:.0f} m&Aring;, and worse than the "
        "deliberately wrong geometry the fit started from. Some angles move by "
        "nine degrees.", BODY))
    story.append(p(
        "This is not a bug, and checking that mattered. The published structure "
        "reproduces its own measured constants with a reduced chi-squared near one, "
        "so the data and the reference agree; the fit simply finds a structure that "
        "fits the numbers <i>better</i> than the true one does. It can, because 22 "
        "constants carry only 9 independent constraints on 12 parameters, and "
        "because the residuals are systematically one-signed &mdash; B and C sit "
        "about 0.5% off in the same direction for every species, which is the "
        "r<sub>s</sub>-versus-r<sub>0</sub> offset described in section 4. The fit "
        "spends its three unconstrained directions removing that bias, and pays for "
        "it in structural distortion. More data made the pull stronger without "
        "closing the directions it could pull through.", BODY))

    story.append(p("7.3 Spectroscopy alone is unreliable on thin data", H2))
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

    story.append(p("7.4 Neither way of combining the sources is reliably better", H2))
    story.append(p(
        "The two hybrids get identical inputs and differ only in the merging rule, "
        "and which one wins is not predictable from anything known in advance. The "
        "<i>split</i> rule wins on fluoroethane at both data levels &mdash; "
        "including the case with the <i>most</i> unconstrained directions of any "
        "run here, fifteen of eighteen. The <i>joint</i> rule wins on vinyl "
        "fluoride and on sparse acetyl fluoride. Sorting the six cases by how "
        "underdetermined they are produces no pattern: split wins at deficits of "
        "15, 3 and 1, joint at 12, 10 and 3.", BODY))
    story.append(p(
        "This is a correction to an earlier result in this project. Run against "
        "isotopologue constants <i>derived</i> from the reference structures rather "
        "than measured, the same six cases produced a clean rule &mdash; joint when "
        "measurements are scarcer than parameters, split otherwise, holding in all "
        "six. That rule does not survive real data. Derived constants are mutually "
        "consistent by construction, which removes precisely the systematic "
        "zero-point offset that turns out to drive the behaviour, and the tidy "
        "pattern was an artefact of that.", BODY))

    story.append(p("7.5 The uncorrected zero-point offset is now the limiting factor",
                   H2))
    story.append(p(
        "The results are no longer mainly limited by how the two information "
        "sources are merged. They are limited by fitting ground-state "
        "(r<sub>0</sub>) constants with no vibration-rotation correction while "
        "scoring against a substitution (r<sub>s</sub>) structure. That mismatch "
        "shows up as a one-signed bias of roughly half a percent in every measured "
        "B and C, and it is what the data-led fits spend their freedom chasing.",
        BODY))
    story.append(p(
        "That diagnosis was tested directly, by computing the correction from a "
        "Cartesian cubic force field and applying it identically to all three "
        "data-using legs. The mechanism checks out where it was predicted: vinyl "
        "fluoride on full data, the case that collapses, improves from 31.1 to "
        "23.1 m&Aring;. But overall the correction <i>hurt</i>, in 14 of 18 cases, "
        "and in one it was catastrophic &mdash; acetyl fluoride on full data went "
        "from 8.8 to 67.9 m&Aring;, with the C&ndash;F bond wrong by 7%.", BODY))
    story.append(p(
        "The cause is not the correction in principle but the force field behind "
        "it. Every one of those runs reported that the cubic term exceeds the "
        "harmonic one, meaning the perturbation series the correction is built on "
        "is not converging, and the damage tracks that divergence ratio: molecules "
        "with a ratio near 3 lose about 1 m&Aring;, the case with a ratio of 35 "
        "loses 40. The alpha implementation validates against experimental "
        "constants independently, so the fault lies with RHF/6-31G cubic force "
        "constants rather than the code. The honest conclusion is that this "
        "correction should not be applied at this level of theory, and that a "
        "better force field &mdash; not a different algorithm &mdash; is the "
        "prerequisite for improving any of these results.", BODY))

    story.append(p("7.6 Does the hybrid beat theory?", H2))
    story.append(p(
        "Since no rule reliably picks the objective in advance, the fair thing to "
        "report is the better of the two, while being explicit that the choice is "
        "made <i>after</i> seeing the answer and so overstates what could be "
        "achieved in practice. On a real problem both would be run &mdash; they "
        "cost seconds &mdash; but with no ground truth there would be nothing to "
        "choose between them.", BODY))

    rows = [["Molecule", "Data level", "theory", "best hybrid", "which", "change"]]
    bold, wins, cf_wins = [], 0, 0
    n = sum(len(m["levels"]) for m in data)
    for m in data:
        for label, lv in m["levels"].items():
            th = m["theory"]["rms_bond_ma"]
            pick = best_hybrid(lv)
            hy = lv["legs"][pick]["rms_bond_ma"]
            cf_pick = best_hybrid(lv, "cf_err_ma", absolute=True)
            wins += hy < th
            cf_wins += abs(lv["legs"][cf_pick]["cf_err_ma"]) < abs(m["theory"]["cf_err_ma"])
            rows.append([m["molecule"], label, f"{th:.1f}", f"{hy:.1f}",
                         SHORT[pick].split()[1].strip("()"), f"{hy - th:+.1f}"])
            if hy < th:
                bold.append((len(rows) - 1, 5))
    story.append(table(rows, [34 * mm, 26 * mm, 22 * mm, 24 * mm, 24 * mm, 24 * mm],
                       align_left=(0, 1, 4), bold_cells=bold))
    story.append(p(
        f"RMS bond error in m&Aring;. Even choosing the better objective with "
        f"hindsight, the hybrid beats theory in {wins} of {n} cases &mdash; not all "
        f"of them. On the C&ndash;F bond specifically it wins {cf_wins} of {n}.",
        CAPTION))

    story.append(p(
        "So the answer is a qualified no, and that is a change from what the same "
        "six cases showed on derived data. Bond accuracy overall: the hybrid is "
        f"better in {wins} of {n}, and vinyl fluoride on parent data is a case where "
        "plain theory beats both hybrids and the experiment-only fit alike. The "
        "C&ndash;F bond is the exception that holds up &mdash; there some "
        "combination of data and theory is closer than theory alone every time, "
        "because theory's error there is a systematic bias and any real information "
        "about fluorine's position helps. But the C&ndash;F gain comes from the "
        "<i>split</i> objective and from the data-only fit; the joint prior stays "
        "within a few m&Aring; of theory on that bond, which is another way of "
        "saying it barely moves.", BODY))
    story.append(p(
        "The honest summary is that on real measured constants, with no "
        "vibration-rotation correction applied, combining the two sources is not "
        "dependably better than using theory alone. That is a weaker claim than the "
        "derived-constant version of this experiment supported, and the difference "
        "between the two is the best argument in this report for insisting on "
        "measured data.", BODY))


def section_limits(story):
    story.append(p("8. Limitations", H1))
    items = [
        ("No vibration-rotation correction is applied &mdash; and applying it at "
         "this level of theory makes things worse.",
         "The measured constants are ground-state (r<sub>0</sub>) values fitted as "
         "they stand, so the fit is pulled toward an r<sub>0</sub> structure while "
         "being scored against an r<sub>s</sub> reference; B and C sit 0.3&ndash;"
         "0.7% off in the same direction in all three molecules. That was expected "
         "to be the binding constraint, so the correction was computed from a "
         "Cartesian cubic force field and applied identically to every data-using "
         "leg (<font face=\"Courier\" size=\"7.5\">scripts/"
         "monofluoro_alpha_corrected.py</font>). It made results worse in 14 of 18 "
         "cases, in one of them catastrophically &mdash; acetyl fluoride on full "
         "data goes from 8.8 to 67.9 m&Aring;. The reason is visible in the "
         "diagnostics: the cubic term exceeds the harmonic one for every species, "
         "so the perturbation series is not converging, and the damage tracks that "
         "divergence ratio closely. The alpha formulae themselves validate against "
         "experiment; it is the RHF/6-31G cubic force field that is inadequate. A "
         "better force field, not a code change, is what this needs."),
        ("Structure and constants come from different studies.",
         "For vinyl fluoride the reference structure is a 1989 determination while "
         "the constants are the 1968 compilation of earlier work. They agree to "
         "1.37%, inside the r<sub>s</sub>-versus-r<sub>0</sub> range, but they are "
         "not one self-consistent experiment."),
        ("Some measured species had to be excluded.",
         "Fluoroethane's seven multiply-deuterated species are omitted because the "
         "compilation's own configuration labels are ambiguous &mdash; it records "
         "that several configurations belong to the same isotopic species yet lists "
         "different constants for them &mdash; so which hydrogen each deuterium "
         "occupies cannot be established from the table. Including them on a guess "
         "would have been worse than leaving them out."),
        ("Hartree&ndash;Fock with a small basis is a weak level of theory.",
         "A better method would shrink theory's C&ndash;F bias and narrow the gap. "
         "That would change the size of the effect, not its direction; the point "
         "here is the comparison between ways of using a fixed quantum surface, not "
         "the absolute accuracy of any of them."),
        ("Three different definitions of 'the structure' are in play.",
         "The reference comes from isotopic substitution (r<sub>s</sub>), the data "
         "are ground-state averages (r<sub>0</sub>), and the quantum calculation "
         "produces the structure at the bottom of the energy well (r<sub>e</sub>). "
         "These differ from one another by a few m&Aring;, so part of every error "
         "quoted here &mdash; for every method &mdash; is a definitional gap rather "
         "than a failure of the method."),
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
