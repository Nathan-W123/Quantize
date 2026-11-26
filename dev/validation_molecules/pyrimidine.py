"""Validation-molecule entry: pyrimidine, m-C4H4N2 (C2v, planar, 10 atoms). Standalone; needs only numpy.

Every number below is transcribed from a downloaded source; pyrimidine.numbers.json
(same folder) gives each one with its table/row/column and the exact text line.
An independent re-read against its own download found no discrepancy
(2026-11-26); dev/tests/test_validation_references.py checks the geometry and
the constants' consistency.

Source (structure AND constants)
--------------------------------
Heim, Esselman, Wood, McMahon et al., "Molecular structure determination:
Equilibrium structure of pyrimidine (m-C4H4N2) from rotational spectroscopy
(reSE) and high-level ab initio calculation (re) agree within the uncertainty
of experimental measurement", J. Chem. Phys. 152, 104303 (2020), doi
10.1063/1.5144914. PDF from https://www.osti.gov/servlets/purl/1800415 (13
pages; it is the typeset AIP version, not a preprint layout).

Structure (primary): Table IV, column "r_e^SE CCSD(T)" (footnote c:
    "Semi-experimental structure with corrections for vibration-rotation
    interaction [CCSD(T)/ANO1] and electron mass [CCSD(T)/ANO1]. Uncertainty
    (2σ) from xrefit structure determination."). 16 isotopologues, all three
    moments each, equal weights (text). Uncertainties are 2σ AS PRINTED.
Structure (comparison):
    Table IV "r_e^SE Ref. 31" = Penocchio, Piccardo & Barone, J. Chem. Theory
    Comput. 11, 4689 (2015), B2PLYP/cc-pVTZ corrections, "with all C-H bond
    lengths fixed" (footnote a; the three C-H rows carry footnote d "Structural
    parameter fixed at this value", so they have NO uncertainty -> None).
    Footnote a: "Uncertainty, in parentheses, as reported in the study" -- the
    CONFIDENCE LEVEL IS NOT STATED in this paper. C(2)...C(5) and C(5)-C(4)-H
    carry footnote e ("Value and uncertainty calculated from reported
    geometrical parameters"), i.e. Heim et al. converted them.
    Table IV "r_e^SE DFT" = this paper's own B3LYP/6-311+G(2d,p)-corrected
    structure (footnote b, 2σ). NOTE: the paper's own DFT column is B3LYP, not
    B2PLYP; the B2PLYP-corrected column is the Ref. 31 one. Both are kept.
Ground-state constants:
    Table II (normal + 4 heavy-atom species) and Table III (11 [2H] species),
    "S-reduced Hamiltonian, IIIr representation". A0, B0, C0 exactly as
    printed. The CONFIDENCE LEVEL of the parenthesised uncertainties is NOT
    STATED anywhere in the paper; they are stored in measured_sigma_mhz as
    printed, and SIGMA_LEVEL_CONSTANTS records that.
    Typesetting: Table II C0 of [5-13C] is printed "3049.30 507 (19)" (odd digit
    grouping, same in -layout and -raw) = 3049.30507, 5 decimals, sigma 0.00019.

Atom order (the paper's ring numbering, N1 C2 N3 C4 C5 C6; H named by carbon):
    0 N1, 1 C2, 2 N3, 3 C4, 4 C5, 5 C6, 6 H2, 7 H4, 8 H5, 9 H6.
    C2, C5, H2, H5 lie on the C2 axis (z here). N1/N3, C4/C6, H4/H6 are the
    mirror-equivalent pairs, so [4-13C] = 13C at C4 (index 3; C6 is
    equivalent) and [1-15N] = 15N at N1 (index 0). The paper confirms which
    sites are off-axis: "Isotopic substitution in the [4-13 C] and [1-15 N]
    isotopologues lies off the principal axis coordinates of the normal
    isotopologue" and mu_a = 0 only for [2-13C], [5-13C] (Table II).
"""

import numpy as np

# ── Masses (u) ───────────────────────────────────────────────────────────────
# Copied from dev/monofluoro_references.py (module constants):
M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
M_N14 = 14.0030740048
# NEW (absent from the module). NIST "Atomic Weights and Isotopic
# Compositions", physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=N,
# Mass Number = 15: "Relative Atomic Mass = 15.00010889888(64)" (read
# 2026-11-26). The same page gives 14.00307400443(20) for 14N (module:
# 14.0030740048, 3.7e-10 u higher) and the H page 1.00782503223(9) for 1H
# (module: 1.00782503207, 1.6e-10 u lower); both differences are negligible.
M_N15 = 15.00010889888

ELEMS = ["N", "C", "N", "C", "C", "C", "H", "H", "H", "H"]
ATOM_LABELS = ["N1", "C2", "N3", "C4", "C5", "C6", "H2", "H4", "H5", "H6"]
PARENT_MASSES = [M_N14, M_C12, M_N14, M_C12, M_C12, M_C12, M_H, M_H, M_H, M_H]

# ── Equilibrium structures (Heim et al. 2020, Table IV) ──────────────────────
# Keys follow the Table IV rows (the nine independent C2v parameters as the
# paper chooses them). "C2...C5" is the cross-ring C(2)-C(5) distance, NOT a
# bond; N1-C6 (= N3-C4) is not published and is derived by the builder.
RE_KEYS = ("C2-H", "C2...C5", "C5-H", "C2-N1", "C4-C5", "C4-H", "N1-C2-H", "C4-C5-H", "C5-C4-H")

#: Table IV "r_e^SE CCSD(T)": (value, uncertainty AS PRINTED, 2σ). Å / deg.
PYRIMIDINE_RE_SE = {
    "C2-H": (1.08165, 0.00039),      # R C(2)–H        1.081 65 (39)
    "C2...C5": (2.65395, 0.00052),   # R C(2)–C(5)     2.653 95 (52)
    "C5-H": (1.07929, 0.00036),      # R C(5)–H        1.079 29 (36)
    "C2-N1": (1.33333, 0.00038),     # R C(2)–N(1)     1.333 33 (38)
    "C4-C5": (1.38675, 0.00045),     # R C(4)–C(5)     1.386 75 (45)
    "C4-H": (1.08224, 0.00030),      # R C(4)–H        1.082 24 (30)
    "N1-C2-H": (116.332, 0.024),     # θ N(1)–C(2)–H   116.332 (24)
    "C4-C5-H": (121.665, 0.022),     # θ C(4)–C(5)–H   121.665 (22)
    "C5-C4-H": (121.261, 0.038),     # θ C(5)–C(4)–H   121.261 (38)
}
PYRIMIDINE_RE_SE_SIGMA_LEVEL = "2sigma"   # Table IV footnote c, abstract, Fig. 5 caption
#: The same, halved to 1σ (arithmetic on the printed 2σ, nothing else).
PYRIMIDINE_RE_SE_1SIGMA = {k: (v, u / 2.0) for k, (v, u) in PYRIMIDINE_RE_SE.items()}
PYRIMIDINE_RE_SE_SOURCE = (
    "r_e^SE, Heim, Esselman, Wood, McMahon et al., J. Chem. Phys. 152, 104303 (2020), "
    "doi 10.1063/1.5144914, Table IV 'r_e^SE CCSD(T)' (16 isotopologues, CCSD(T)/ANO1 "
    "vibration-rotation and electron-mass corrections): C2-N1 1.33333(38), C4-C5 1.38675(45), "
    "C2...C5 2.65395(52), C2-H 1.08165(39), C4-H 1.08224(30), C5-H 1.07929(36) A, "
    "N1-C2-H 116.332(24), C4-C5-H 121.665(22), C5-C4-H 121.261(38) deg, 2-sigma")

#: Table IV "r_e^SE DFT" (this paper, B3LYP/6-311+G(2d,p) corrections), 2σ as printed.
PYRIMIDINE_RE_SE_B3LYP = {
    "C2-H": (1.08105, 0.00061), "C2...C5": (2.65457, 0.00081), "C5-H": (1.07875, 0.00057),
    "C2-N1": (1.33373, 0.00059), "C4-C5": (1.38716, 0.00070), "C4-H": (1.08164, 0.00048),
    "N1-C2-H": (116.331, 0.038), "C4-C5-H": (121.661, 0.035), "C5-C4-H": (121.259, 0.059),
}
PYRIMIDINE_RE_SE_B3LYP_SIGMA_LEVEL = "2sigma"   # Table IV footnote b

#: Table IV "r_e^SE Ref. 31" (Penocchio et al. 2015, B2PLYP/cc-pVTZ corrections), as
#: printed. C-H values were FIXED in that fit (footnote d): uncertainty None.
PYRIMIDINE_RE_SE_B2PLYP_REF31 = {
    "C2-H": (1.0820, None), "C2...C5": (2.6534, 0.0003), "C5-H": (1.0794, None),
    "C2-N1": (1.3333, 0.0003), "C4-C5": (1.3868, 0.0002), "C4-H": (1.0825, None),
    "N1-C2-H": (116.32, 0.01), "C4-C5-H": (121.65, 0.01), "C5-C4-H": (121.21, 0.01),
}
PYRIMIDINE_RE_SE_B2PLYP_REF31_SIGMA_LEVEL = None   # "as reported in the study"; level not stated
PYRIMIDINE_RE_SE_B2PLYP_REF31_SOURCE = (
    "r_e^SE, Penocchio, Piccardo & Barone, J. Chem. Theory Comput. 11, 4689 (2015) "
    "(B2PLYP/cc-pVTZ corrections, all C-H fixed), as printed in Heim et al. 2020 Table IV "
    "column 'r_e^SE Ref. 31'; C2...C5 and C5-C4-H converted by Heim et al. (footnote e); "
    "uncertainty level not stated")


def _rot_xz(v, ang_rad):
    """Rotate a vector lying in the xz plane by ang about +y (x->z positive)."""
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    return np.array([c * v[0] - s * v[2], 0.0, s * v[0] + c * v[2]])


def pyrimidine_geometry_re(r_c2h, r_c2c5, r_c5h, r_c2n1, r_c4c5, r_c4h,
                           a_n1c2h_deg, a_c4c5h_deg, a_c5c4h_deg):
    """Planar C2v pyrimidine in this file's atom order
    (N1, C2, N3, C4, C5, C6, H2, H4, H5, H6) from the nine Table IV parameters.

    C2 at the origin, C5 on +z (the C2 axis), ring in the xz plane (y = 0),
    N1 and C6 on the +x side, N3 and C4 their mirror images (x -> -x).
      H2 = -r(C2-H) z;  N1 makes angle N1-C2-H with C2->H2.
      H5 = C5 + r(C5-H) z;  C6 makes angle C4-C5-H (= C6-C5-H) with C5->H5.
      H6 on C6 makes angle C5-C6-H (= C5-C4-H) with C6->C5, on the side away
      from N1 (outside the ring).
    N1-C6 follows from the rest; it is not a published parameter.
    """
    a2, a5, a4 = np.radians([a_n1c2h_deg, a_c4c5h_deg, a_c5c4h_deg])
    c2 = np.zeros(3)
    h2 = np.array([0.0, 0.0, -r_c2h])
    n1 = r_c2n1 * np.array([np.sin(a2), 0.0, -np.cos(a2)])
    c5 = np.array([0.0, 0.0, r_c2c5])
    h5 = np.array([0.0, 0.0, r_c2c5 + r_c5h])
    c6 = c5 + r_c4c5 * np.array([np.sin(a5), 0.0, np.cos(a5)])
    e = (c5 - c6) / np.linalg.norm(c5 - c6)          # C6 -> C5
    f = (n1 - c6) / np.linalg.norm(n1 - c6)          # C6 -> N1
    toward_f = np.sign(e[0] * f[2] - e[2] * f[0])    # +1 if rotating e by + moves it toward f
    h6 = c6 + r_c4h * _rot_xz(e, -toward_f * a4)     # rotate away from N1: outside the ring
    m = np.array([-1.0, 1.0, 1.0])
    n3, c4, h4 = n1 * m, c6 * m, h6 * m
    return np.array([n1, c2, n3, c4, c5, c6, h2, h4, h5, h6])


def geometry_from(params):
    return pyrimidine_geometry_re(*(params[k][0] for k in RE_KEYS))


GEOMETRY_RE = geometry_from(PYRIMIDINE_RE_SE)
GEOMETRY_RE_B3LYP = geometry_from(PYRIMIDINE_RE_SE_B3LYP)
GEOMETRY_RE_B2PLYP_REF31 = geometry_from(PYRIMIDINE_RE_SE_B2PLYP_REF31)

# ── Ground-state constants, Tables II and III (S reduction, IIIr) ─────────────
SIGMA_LEVEL_CONSTANTS = "not stated by the source (printed parenthesised uncertainties)"
CONSTANTS_REDUCTION = "Watson S reduction, IIIr representation"

#: label (the paper's column heading), subs {atom index: mass}, (A0, B0, C0) MHz,
#: decimals as printed, printed uncertainties (MHz) in the same order.
SPECIES = [
    # Table II
    {"label": "m-C4H4N2", "subs": {},
         "abc_mhz": (6276.827750, 6067.165761, 3084.44922), "decimals": (6, 6, 5),
         "measured_sigma_mhz": (0.000079, 0.000078, 0.00010)},
    {"label": "[2-13C]", "subs": {1: M_C13},
         "abc_mhz": (6152.67855, 6067.53926, 3054.25159), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00026, 0.00021, 0.00018)},
    {"label": "[4-13C]", "subs": {3: M_C13},
         "abc_mhz": (6256.09720, 5957.21866, 3050.84569), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00019, 0.00019, 0.00023)},
    {"label": "[5-13C]", "subs": {4: M_C13},
         "abc_mhz": (6132.81727, 6067.35963, 3049.30507), "decimals": (5, 5, 5),   # C0 printed "3049.30 507 (19)"
         "measured_sigma_mhz": (0.00029, 0.00025, 0.00019)},
    {"label": "[1-15N]", "subs": {0: M_N15},
         "abc_mhz": (6253.95822, 5954.15235, 3049.53046), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00038, 0.00038, 0.00051)},
    # Table III, first block
    {"label": "[2-2H]", "subs": {6: M_D},
         "abc_mhz": (6066.8537, 5871.7522, 2983.30009), "decimals": (4, 4, 5),
         "measured_sigma_mhz": (0.0027, 0.0036, 0.00051)},
    {"label": "[4-2H]", "subs": {7: M_D},
         "abc_mhz": (6242.39729, 5694.50625, 2977.37753), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00015, 0.00014, 0.00020)},
    {"label": "[5-2H]", "subs": {8: M_D},
         "abc_mhz": (6066.96171, 5840.14871, 2975.12051), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00023, 0.00021, 0.00030)},
    {"label": "[2,4-2H]", "subs": {6: M_D, 7: M_D},
         "abc_mhz": (5934.8059, 5604.1316, 2881.89348), "decimals": (4, 4, 5),
         "measured_sigma_mhz": (0.0020, 0.0021, 0.00036)},
    {"label": "[2,5-2H]", "subs": {6: M_D, 8: M_D},
         "abc_mhz": (6066.6645, 5479.2542, 2878.52975), "decimals": (4, 4, 5),
         "measured_sigma_mhz": (0.0089, 0.0085, 0.00027)},
    # Table III, second block
    {"label": "[4,5-2H]", "subs": {7: M_D, 8: M_D},
         "abc_mhz": (5913.95177, 5602.39335, 2876.48887), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00019, 0.00019, 0.00027)},
    {"label": "[4,6-2H]", "subs": {7: M_D, 9: M_D},
         "abc_mhz": (6082.41737, 5457.75988, 2876.11603), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00033, 0.00035, 0.00014)},
    {"label": "[2,4,5-2H]", "subs": {6: M_D, 7: M_D, 8: M_D},
         "abc_mhz": (5809.6599, 5351.0853, 2785.07418), "decimals": (4, 4, 5),
         "measured_sigma_mhz": (0.0011, 0.0015, 0.00025)},
    {"label": "[2,4,6-2H]", "subs": {6: M_D, 7: M_D, 9: M_D},
         "abc_mhz": (5692.64799, 5457.51365, 2785.90867), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00096, 0.00094, 0.00052)},
    {"label": "[4,5,6-2H]", "subs": {7: M_D, 8: M_D, 9: M_D},
         "abc_mhz": (5680.01344, 5457.63849, 2782.89077), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00022, 0.00021, 0.00010)},
    {"label": "[2,4,5,6-2H]", "subs": {6: M_D, 7: M_D, 8: M_D, 9: M_D},
         "abc_mhz": (5457.39248, 5330.69699, 2696.31571), "decimals": (5, 5, 5),
         "measured_sigma_mhz": (0.00051, 0.00055, 0.00031)},
]

ENTRY = {
    "key": "pyrimidine",
    "name": "Pyrimidine",
    "formula": "m-C4H4N2",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE,
    "masses": np.array(PARENT_MASSES),
    "species": SPECIES,
    "structure_source": PYRIMIDINE_RE_SE_SOURCE,
    "structure_type": "r_e",
    # Largest BOND uncertainty as the source quotes it (2σ, C4-C5 0.45 mA).
    # The cross-ring C2...C5 distance parameter has 0.52 mA (2σ); the derived
    # N1-C6 bond has no published uncertainty (needs the fit's covariance).
    # Halved to 1σ: 0.225 / 0.26 mA. RE_REFERENCE_MAX_UNCERTAINTY_MA is 0.5,
    # so 0.45 is "ok"; counting C2...C5 at 2σ it would be "too_coarse": a
    # convention decision for the caller.
    "geometry_re_uncertainty_ma": 0.45,
    "constants_source": ("Heim et al., J. Chem. Phys. 152, 104303 (2020), Tables II and III: "
                      "A0, B0, C0, S reduction, IIIr; uncertainty level not stated"),
    "alt_geometry_re": GEOMETRY_RE_B2PLYP_REF31,
    "alt_geometry_re_source": PYRIMIDINE_RE_SE_B2PLYP_REF31_SOURCE,
    "bonds": {
        "C2-N1": [(1, 0), (1, 2)],
        "N1-C6": [(0, 5), (2, 3)],          # derived, not published
        "C4-C5": [(3, 4), (5, 4)],
        "C2-H": [(1, 6)],
        "C4-H": [(3, 7), (5, 9)],
        "C5-H": [(4, 8)],
        # "C2...C5" (the cross-ring C(2)-C(5) distance, a Table IV parameter)
        # is not a bond: listed here it would be scored as one and pooled as a
        # C-C bond offset. dev/tests/test_validation_references.py checks it
        # on the geometry instead.
    },
    "angles": {
        "N1-C2-H": [(0, 1, 6), (2, 1, 6)],
        "C4-C5-H": [(3, 4, 8), (5, 4, 8)],
        "C5-C4-H": [(4, 3, 7), (4, 5, 9)],
        "N1-C2-N3": [(0, 1, 2)],
        "C4-C5-C6": [(3, 4, 5)],
        "C2-N1-C6": [(1, 0, 5), (1, 2, 3)],
        "N1-C6-C5": [(0, 5, 4), (2, 3, 4)],
        "N3-C4-H": [(2, 3, 7), (0, 5, 9)],
    },
}
