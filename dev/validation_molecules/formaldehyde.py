"""Validation-molecule entry: formaldehyde, H2CO (C2v).

Standalone (numpy only); mirrors the fields of ReferenceMolecule / Isotopologue
in dev/monofluoro_references.py so it can be pasted in. Every number is
transcribed from the downloaded sources; see formaldehyde.numbers.json next to this file
for the exact text line of each one, and dev/tests/test_validation_references.py for the consistency checks.

Atom order: C0, O1, H2, H3. C at the origin, O on +z, the H pair in the
xz-plane mirrored in x (C2 axis = z).
"""
from __future__ import annotations

import numpy as np

# ── Masses (u) ──────────────────────────────────────────────────────────────
# Same values as dev/monofluoro_references.py (M_H, M_D, M_C12, M_C13, M_O16,
# M_O18 exist there and are copied verbatim).
M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
M_O16, M_O18 = 15.9949146196, 17.9991610
#: NOT in the module -- added here. NIST "Atomic Weights and Isotopic
#: Compositions" (physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=O,
#: read 2026-11-26): "Relative Atomic Mass = 16.99913175650(69)".
#: (Same page gives 18O = 17.99915961286(76); the module's M_O18 = 17.9991610
#: is 1.4e-6 u higher, an older table value -- irrelevant here, left as is.)
M_O17 = 16.99913175650

ELEMS = ["C", "O", "H", "H"]
PARENT_MASSES = (M_C12, M_O16, M_H, M_H)

# ── Equilibrium structures (Angstrom, deg; (value, 1-sigma)) ────────────────
#: PRIMARY. r_e^SE, Penocchio, Mendolicchio, Tasinato & Barone, Can. J. Chem.
#: 94, 1065 (2016), Table 3, "formaldehyde", column rSE_e B2PLYP/VTZ. Table
#: footnote: uncertainties are rounded UP to 1e-4 A / 1e-2 deg when smaller,
#: so (1) here means <= 0.1 mA.
H2CO_RE_SE = {"C=O": (1.2047, 0.0001), "C-H": (1.1008, 0.0001), "H-C-O": (121.61, 0.01)}
H2CO_RE_SE_SOURCE = (
    "r_e^SE, Penocchio, Mendolicchio, Tasinato & Barone, Can. J. Chem. 94, 1065 "
    "(2016), doi 10.1139/cjc-2016-0282, Table 3 (formaldehyde, rSE_e B2PLYP/VTZ "
    "column): r(C=O) 1.2047(1), r(C-H) 1.1008(1) A, a(H-C-O) 121.61(1) deg; "
    "read from the Europe PMC author manuscript (ricerca.sns.it emss-73951.pdf)")
#: Same table, rSE_e B3LYP/SNSD column (second set of vibrational corrections).
H2CO_RE_SE_B3LYP = {"C=O": (1.2050, 0.0001), "C-H": (1.1005, 0.0001), "H-C-O": (121.61, 0.01)}

#: ALTERNATIVE, purely experimental r_e: Lohilahti, PhD thesis, Univ. Oulu
#: (2006), as quoted in Morgan et al., JCTC 2018, Table 3, column
#: "Experiment" (footnote e). The source gives the H-C-H angle, not H-C-O.
H2CO_RE_EXP = {"C=O": (1.20461, 0.00019), "C-H": (1.10046, 0.00016), "H-C-H": (116.722, 0.093)}
H2CO_RE_EXP_SOURCE = (
    "experimental r_e, J. Lohilahti, PhD thesis (Univ. Oulu, 2006), as quoted in "
    "Table 3 ('Experiment' column, footnote e) of Morgan et al., J. Chem. Theory "
    "Comput. 2018 (OSTI 1468297): r_e(CO) 1.20461(19), r_e(CH) 1.10046(16) A, "
    "theta_e(HCH) 116.722(93) deg")

#: Stated uncertainty in mA (largest bond sigma), for geometry_re_uncertainty_ma.
H2CO_RE_SE_UNCERTAINTY_MA = 0.1      # both bonds (1) in 1e-4 A
H2CO_RE_EXP_UNCERTAINTY_MA = 0.19    # C=O (19) in 1e-5 A; C-H is 0.16


def h2co_geometry(r_co, r_ch, a_hco_deg=None, a_hch_deg=None):
    """Planar C2v H2C=O from bond lengths and ONE angle (H-C-O or H-C-H).

    C0 at the origin, O1 on +z, H2/H3 in the xz-plane at +-x. For a planar
    C2v molecule H-C-O = 180 - (H-C-H)/2 exactly, which is how the H-C-H form
    is handled.
    """
    if (a_hco_deg is None) == (a_hch_deg is None):
        raise ValueError("give exactly one of a_hco_deg, a_hch_deg")
    a = np.radians(a_hco_deg if a_hco_deg is not None else 180.0 - 0.5 * a_hch_deg)
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r_co],
        [r_ch * s, 0.0, r_ch * c],
        [-r_ch * s, 0.0, r_ch * c],
    ])


GEOMETRY_RE_SE = h2co_geometry(H2CO_RE_SE["C=O"][0], H2CO_RE_SE["C-H"][0],
                               a_hco_deg=H2CO_RE_SE["H-C-O"][0])
GEOMETRY_RE_EXP = h2co_geometry(H2CO_RE_EXP["C=O"][0], H2CO_RE_EXP["C-H"][0],
                                a_hch_deg=H2CO_RE_EXP["H-C-H"][0])

# ── Ground-state constants ──────────────────────────────────────────────────
# Muller & Lewen 2017 (Table 1) publish A-(B+C)/2, (B+C)/2, (B-C)/4, not A, B, C.
# Transcribed as printed (value, 1-sigma, MHz):
ML2017_TABLE1 = {
    "H2C16O": {"A-(B+C)/2": (245551.4495, 0.0040), "(B+C)/2": (36419.11528, 0.00025),
               "(B-C)/4": (1207.4358721, 0.0000033)},
    "H2C18O": {"A-(B+C)/2": (247253.578, 0.054), "(B+C)/2": (34707.84108, 0.00025),
               "(B-C)/4": (1097.2174152, 0.0000059)},
    "H2C17O": {"A-(B+C)/2": (246452.397, 0.095), "(B+C)/2": (35513.40370, 0.00032),
               "(B-C)/4": (1148.454801, 0.000090)},
}


def abc_from_combinations(x, y, z, sx, sy, sz):
    """A = x + y, B = y + 2z, C = y - 2z, with x = A-(B+C)/2, y = (B+C)/2,
    z = (B-C)/4. Sigmas in quadrature, treating x, y, z as UNCORRELATED: the
    papers print no correlation matrix, so the covariance cannot be included.
    sigma_A = hypot(sx, sy); sigma_B = sigma_C = hypot(sy, 2 sz). The
    worst-case (fully correlated) linear sums sx + sy and sy + 2 sz can be up
    to sqrt(2) larger; the transcription notes list both for every species."""
    return ((x + y, y + 2 * z, y - 2 * z),
            (float(np.hypot(sx, sy)), float(np.hypot(sy, 2 * sz)), float(np.hypot(sy, 2 * sz))))


def _iso(label, subs, abc, decimals, sigma, source, note=""):
    return {"label": label, "subs": subs, "abc_mhz": abc, "decimals": decimals,
            "measured_sigma_mhz": sigma, "source": source, "note": note}


#: Literal values; the transcription check verified each against abc_from_combinations(ML2017_TABLE1)
#: and against numbers.json. For the three Muller & Lewen species the A, B, C are
#: exact sums of printed numbers, so `decimals` is the smaller of the inputs'
#: printed decimals (there is no printed A, B, C to count digits on).
SPECIES = [
    _iso("H2C16O parent", {}, (281970.56478, 38833.9870242, 34004.2435358), (4, 5, 5),
         (0.00401, 0.000250, 0.000250),
         "Muller & Lewen, J. Mol. Spectrosc. 331, 28 (2017), arXiv:1610.02174v2, Table 1, "
         "column H2C16O, converted from A-(B+C)/2, (B+C)/2, (B-C)/4",
         "Cross-check: Bruenken et al. 2003 A0/B0/C0 quoted by Morgan et al. 2018 Table 3, "
         "281970.5578(61) / 38833.98715(31) / 34004.24349(31): differences +0.0070 / "
         "-0.00013 / +0.00005 MHz, within the combined sigmas (0.0073 / 0.0004 / 0.0004)."),
    _iso("H2C18O", {1: M_O18}, (281961.41908, 36902.2759104, 32513.4062496), (3, 5, 5),
         (0.0540, 0.000250, 0.000250),
         "Muller & Lewen 2017, Table 1, column H2C18O, converted"),
    _iso("H2C17O", {1: M_O17}, (281965.80070, 37810.313302, 33216.494098), (3, 5, 5),
         (0.0950, 0.000367, 0.000367),
         "Muller & Lewen 2017, Table 1, column H2C17O, converted"),
    _iso("H2(13)CO", {0: M_C13}, (281993.04, 37809.107, 33215.941), (2, 3, 3),
         (0.00310, 0.000213, 0.000207),
         "A, B, C: CDMS documentation page, tag 031503 (version 1, May 2000). Sigma: CDMS "
         "spectroscopy-data fit file daten/H2CO/h2c-13-o/13.fit (SPFIT, dated May 04 2000), "
         "final iteration: A 281993.03965(310), B 37809.106966(213), C 33215.941417(207)",
         "The fit file's values round to the page's values, and its date matches the "
         "entry's, so the sigmas are taken to belong to this entry."),
    _iso("HDCO", {2: M_D}, (198118.36, 34909.107, 29562.871), (2, 3, 3), None,
         "CDMS documentation page, tag 031501 (version 2, Jan. 2016)",
         "Page gives no uncertainty. The only open fit file (daten/H2CO/hdco/HDCO.fit) is dated "
         "Sep 2008, before this entry version added the Zakharenko et al. 2015 THz data; its "
         "converted A, B, C happen to round to the page's values, but it is a different fit, so "
         "its sigmas (A 0.0042, B = C 0.00076 MHz, numbers.json) are NOT adopted."),
    _iso("D2CO", {2: M_D, 3: M_D}, (141653.60, 32282.506, 26186.347), (2, 3, 3), None,
         "CDMS documentation page, tag 032502 (version 3, Jan. 2016)",
         "Page gives no uncertainty. The open fit file (daten/H2CO/d2co/D2CO.fit, Sep 2008) "
         "gives A = 141653.593, which does NOT round to the page's 141653.60, confirming it is "
         "an older fit; its sigmas (A 0.0019, B = C 0.00040 MHz) are NOT adopted."),
]

CONSTANTS_SOURCE = (
    "H2C16O/H2C18O/H2C17O: Muller & Lewen, J. Mol. Spectrosc. 331, 28 (2017), arXiv:1610.02174, "
    "Table 1 (S reduction, I^r), converted to A, B, C; H2(13)CO, HDCO, D2CO: CDMS catalog "
    "documentation pages e031503 / e031501 / e032502 (S reduction fits); H2(13)CO sigmas "
    "from the CDMS fit file 13.fit")

ENTRY = {
    "key": "formaldehyde",
    "name": "Formaldehyde",
    "formula": "H2CO",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE_SE,
    "masses": np.array(PARENT_MASSES),
    "species": SPECIES,
    "structure_source": H2CO_RE_SE_SOURCE,
    "constants_source": CONSTANTS_SOURCE,
    "structure_type": "r_e",
    "geometry_re_uncertainty_ma": H2CO_RE_SE_UNCERTAINTY_MA,
    "bonds": {"C=O": [(0, 1)], "C-H": [(0, 2), (0, 3)]},
    "angles": {"H-C-O": [(2, 0, 1), (3, 0, 1)], "H-C-H": [(2, 0, 3)]},
    # second, independent equilibrium reference, kept for comparison
    "alt_geometry_re": GEOMETRY_RE_EXP,
    "alt_geometry_re_source": H2CO_RE_EXP_SOURCE,
    "alt_geometry_re_uncertainty_ma": H2CO_RE_EXP_UNCERTAINTY_MA,
}
