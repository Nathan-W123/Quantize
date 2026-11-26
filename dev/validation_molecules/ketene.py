"""Validation-molecule entry: ketene, H2C=C=O (C2v). Standalone; needs only numpy.

Every number below is transcribed from a downloaded source; ketene.numbers.json (same
folder) gives each one with its table/row/column and the exact text line, and
dev/tests/test_validation_references.py checks the geometry and the constants' consistency; an independent re-read of every number against the sources found no discrepancy (2026-11-26).

Sources
-------
Structure (primary, r_e^SE):
    Smith, Esselman, Wood, Stanton, Woods & McMahon, J. Chem. Phys. 158,
    244304 (2023), doi 10.1063/5.0154770, Table IV, column "re SE this work"
    (16 isotopologues, 32 independent moments, CCSD(T)/cc-pCVTZ
    vibration-rotation and electron-mass corrections). Uncertainties are 2σ
    as printed (abstract: "The 2σ uncertainties of the re SE parameters are
    <=0.0007 Å and 0.014° ..."; the Fig. 7 caption says the same; Table IV's
    footnote a belongs to its East and Guarnieri columns). The table's "re SE
    recommended" column (rounded) is also kept.
Structure (comparison, older r_e):
    East, Allen & Klippenstein, J. Chem. Phys. 102, 8506 (1995), Table I,
    row "r_e (I; ls-ABC)" (the structure that paper recommends). Its
    parenthesised figures are standard errors (1σ; Table I footnote a). It
    publishes the H-C-H angle, 121.781(12) deg; Smith et al. Table IV
    re-prints the same structure as C-C-H = 119.110 with every uncertainty
    doubled to 2σ (1.160 30(58), 1.312 12(60), 1.075 76(14), 119.110(12)).
Ground-state constants:
    Smith et al. 2023, Table II (+ continuation), "Spectroscopic constants for
    ketene isotopologues, ground vibrational state (S-reduced Hamiltonian, I^r
    representation)". A0, B0, C0 exactly as printed, S reduction. The normal
    species uses the "Current work" column (the paper also prints Guarnieri et
    al. 2003 and a CCSD(T) column; neither is used). [1,2-13C] is NOT a
    new measurement: the paper reprints Guarnieri's constants for it.
    The CONFIDENCE LEVEL of Table II's parenthesised uncertainties is NOT
    STATED anywhere in the paper; they are stored in measured_sigma_mhz as
    printed, and SIGMA_LEVEL_TABLE_II records that the level is unstated.

Atom order (Smith's numbering: C1 = carbonyl carbon, C2 = CH2 carbon):
    0 C1, 1 C2, 2 O, 3 H, 4 H.  [2-2H] = one H on C2 replaced by D (index 3).
"""

import numpy as np

# ── Masses (u) ───────────────────────────────────────────────────────────────
# Copied from dev/monofluoro_references.py (module constants):
M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
M_O16, M_O18 = 15.9949146196, 17.9991610
# NEW (absent from the module). NIST "Atomic Weights and Isotopic
# Compositions", physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=O,
# Mass Number = 17: "Relative Atomic Mass = 16.99913175650(69)" (read
# 2026-11-26). Note the same page gives 17.99915961286(76) for 18O, while the
# module's M_O18 is 17.9991610 (an older value, 1.4e-6 u higher).
M_O17 = 16.99913175650

ELEMS = ["C", "C", "O", "H", "H"]
ATOM_LABELS = ["C1 (carbonyl)", "C2 (methylene)", "O", "H", "H"]
PARENT_MASSES = [M_C12, M_C12, M_O16, M_H, M_H]

# ── Equilibrium structures ───────────────────────────────────────────────────
#: Smith et al. 2023 Table IV "re SE this work": (value, uncertainty AS PRINTED, 2σ).
KETENE_RE_SE = {
    "C1=C2": (1.31218, 0.00069),     # rC–C (Å)  1.312 18 (69)
    "C2-H": (1.07593, 0.00016),      # rC–H (Å)  1.075 93 (16)
    "C1=O": (1.16064, 0.00066),      # rC–O (Å)  1.160 64 (66)
    "C1=C2-H": (119.086, 0.014),     # θC–C–H (deg) 119.086 (14)
}
KETENE_RE_SE_SIGMA_LEVEL = "2sigma"   # as stated by the source
#: The same, halved to 1σ (arithmetic on the printed 2σ, nothing else).
KETENE_RE_SE_1SIGMA = {k: (v, u / 2.0) for k, (v, u) in KETENE_RE_SE.items()}
#: Table IV "re SE recommended" (bold in the paper), 2σ as printed.
KETENE_RE_SE_RECOMMENDED = {
    "C1=C2": (1.3122, 0.0007), "C2-H": (1.0759, 0.0002),
    "C1=O": (1.1606, 0.0007), "C1=C2-H": (119.086, 0.014),
}
KETENE_RE_SE_SOURCE = (
    "r_e^SE, Smith, Esselman, Wood, Stanton, Woods & McMahon, J. Chem. Phys. 158, "
    "244304 (2023), doi 10.1063/5.0154770, Table IV 're SE this work' (16 "
    "isotopologues, CCSD(T)/cc-pCVTZ corrections): r(C-C) = 1.31218(69), "
    "r(C-H) = 1.07593(16), r(C-O) = 1.16064(66) A, C-C-H = 119.086(14) deg, 2-sigma")

#: East, Allen & Klippenstein 1995, Table I, r_e (I; ls-ABC): (value, 1σ standard error).
KETENE_RE_EAST1995 = {
    "C1=O": (1.16030, 0.00029),      # 1.160 30~29!
    "C1=C2": (1.31212, 0.00030),     # 1.312 12~30!
    "C2-H": (1.07576, 0.00007),      # 1.075 76 ~7!
    "H-C2-H": (121.781, 0.012),      # 121.781~12!
}
KETENE_RE_EAST1995_SIGMA_LEVEL = "1sigma (standard errors, Table I footnote a)"
KETENE_RE_EAST1995_SOURCE = (
    "r_e (I; ls-ABC), East, Allen & Klippenstein, J. Chem. Phys. 102, 8506 (1995), "
    "Table I: r(C=O) = 1.16030(29), r(C=C) = 1.31212(30), r(C-H) = 1.07576(7) A, "
    "H-C-H = 121.781(12) deg, standard errors; reprinted as 2-sigma in Smith et "
    "al. 2023 Table IV (C-C-H 119.110(12))")


def ketene_geometry_re(r_c1c2, r_c2h, r_c1o, a_c1c2h_deg):
    """Planar C2v H2C=C=O in this file's atom order (C1, C2, O, H, H).

    C1 at the origin, O on +z, C2 on -z (the a axis); the H atoms in the xz
    plane, each making angle C1=C2-H with the C2->C1 direction.
    """
    t = np.radians(a_c1c2h_deg)
    c1 = np.zeros(3)
    o = np.array([0.0, 0.0, r_c1o])
    c2 = np.array([0.0, 0.0, -r_c1c2])
    h3 = c2 + r_c2h * np.array([np.sin(t), 0.0, np.cos(t)])
    h4 = c2 + r_c2h * np.array([-np.sin(t), 0.0, np.cos(t)])
    return np.array([c1, c2, o, h3, h4])


def ketene_geometry_re_from_hch(r_c1o, r_c1c2, r_c2h, a_hch_deg):
    """Same geometry from an H-C-H angle (East et al.'s parameter): in a
    planar C2v CH2 group, C=C-H = 180 - HCH/2 exactly."""
    return ketene_geometry_re(r_c1c2, r_c2h, r_c1o, 180.0 - a_hch_deg / 2.0)


GEOMETRY_RE = ketene_geometry_re(*(KETENE_RE_SE[k][0] for k in ("C1=C2", "C2-H", "C1=O", "C1=C2-H")))
GEOMETRY_RE_EAST1995 = ketene_geometry_re_from_hch(
    *(KETENE_RE_EAST1995[k][0] for k in ("C1=O", "C1=C2", "C2-H", "H-C2-H")))

# ── Ground-state constants, Smith et al. 2023 Table II (S reduction, I^r) ──────
SIGMA_LEVEL_TABLE_II = "not stated by the source (printed parenthesised uncertainties)"
CONSTANTS_REDUCTION = "Watson S reduction, I^r representation"

_D2 = {3: M_D, 4: M_D}
#: label, subs {atom index: mass}, (A0, B0, C0) MHz, decimals as printed,
#: printed uncertainties (MHz) in the same order.
SPECIES = [
    {"label": "H2C=C=O (normal, 'Current work')", "subs": {},
         "abc_mhz": (282121.6, 10293.31894, 9915.90304), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (1.8, 0.00024, 0.00024)},
    {"label": "[1-13C]", "subs": {0: M_C13},
         "abc_mhz": (282104.2, 10293.62959, 9916.21087), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (2.5, 0.00044, 0.00044)},
    {"label": "[2-13C]", "subs": {1: M_C13},
         "abc_mhz": (282108.1, 9960.97788, 9607.13942), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (2.6, 0.00037, 0.00035)},
    {"label": "[18O]", "subs": {2: M_O18},
         "abc_mhz": (282142.0, 9761.23701, 9421.12472), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (4.0, 0.00038, 0.00037)},
    {"label": "[17O]", "subs": {2: M_O17},
         "abc_mhz": (282175.0, 10013.4722, 9655.9096), "decimals": (0, 4, 4),
         "measured_sigma_mhz": (13.0, 0.0014, 0.0013)},
    {"label": "[1,2-13C] (Guarnieri's constants, reprinted)", "subs": {0: M_C13, 1: M_C13},
         "abc_mhz": (282031.0, 9960.8650, 9607.0550), "decimals": (0, 4, 4),
         "measured_sigma_mhz": (22.0, 0.0025, 0.0027)},
    {"label": "[2-2H]", "subs": {3: M_D},
         "abc_mhz": (194292.2, 9647.06533, 9174.64351), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (1.3, 0.00021, 0.00020)},
    {"label": "[2-2H, 1-13C]", "subs": {3: M_D, 0: M_C13},
         "abc_mhz": (194256.1, 9646.68707, 9174.26089), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (2.3, 0.00065, 0.00064)},
    {"label": "[2-2H, 2-13C]", "subs": {3: M_D, 1: M_C13},
         "abc_mhz": (193984.2, 9373.43143, 8926.17792), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (2.6, 0.00063, 0.00060)},
    {"label": "[2-2H, 18O]", "subs": {3: M_D, 2: M_O18},
         "abc_mhz": (194243.2, 9145.12930, 8719.33331), "decimals": (1, 5, 5),
         "measured_sigma_mhz": (3.1, 0.00076, 0.00072)},
    {"label": "[2-2H, 17O]", "subs": {3: M_D, 2: M_O17},
         "abc_mhz": (194260.6, 9383.1715, 8935.5431), "decimals": (1, 4, 4),
         "measured_sigma_mhz": (2.7, 0.0013, 0.0013)},
    {"label": "[2,2-2H]", "subs": dict(_D2),
         "abc_mhz": (141490.38, 9120.83067, 8552.69981), "decimals": (2, 5, 5),
         "measured_sigma_mhz": (0.28, 0.00017, 0.00016)},
    {"label": "[2,2-2H, 1-13C]", "subs": {**_D2, 0: M_C13},
         "abc_mhz": (141484.52, 9119.42658, 8551.48444), "decimals": (2, 5, 5),
         "measured_sigma_mhz": (0.67, 0.00065, 0.00071)},
    {"label": "[2,2-2H, 2-13C]", "subs": {**_D2, 1: M_C13},
         "abc_mhz": (141483.16, 8890.47351, 8349.84131), "decimals": (2, 5, 5),
         "measured_sigma_mhz": (0.65, 0.00067, 0.00061)},
    {"label": "[2,2-2H, 18O]", "subs": {**_D2, 2: M_O18},
         "abc_mhz": (141489.40, 8641.83891, 8130.05311), "decimals": (2, 5, 5),
         "measured_sigma_mhz": (0.85, 0.00056, 0.00054)},
    {"label": "[2,2-2H, 17O]", "subs": {**_D2, 2: M_O17},
         "abc_mhz": (141490.0, 8869.0621, 8330.8819), "decimals": (1, 4, 4),
         "measured_sigma_mhz": (1.0, 0.0017, 0.0013)},
]

#: Not used (for reference): Table II's literature column for the normal
#: species, "Guarnieri et al. 2003" (footnote a: "Constants as reported in Ref. 60").
NORMAL_GUARNIERI2003 = {"abc_mhz": (282032.0, 10293.31963, 9915.90393), "decimals": (0, 5, 5),
                            "measured_sigma_mhz": (22.0, 0.00081, 0.00080)}

ENTRY = {
    "key": "ketene",
    "name": "Ketene",
    "formula": "H2C=C=O",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE,
    "masses": np.array(PARENT_MASSES),
    "species": SPECIES,
    "structure_source": KETENE_RE_SE_SOURCE,
    "structure_type": "r_e",
    # Largest bond uncertainty as the source quotes it (2σ, rC–C). Halved to
    # 1σ it is 0.345 mA. RE_REFERENCE_MAX_UNCERTAINTY_MA is 0.5, so the 2σ
    # figure is "too_coarse" and the 1σ figure "ok": a convention decision.
    # (METHYL_CHLORIDE stores its source's 95% figure as quoted.)
    "geometry_re_uncertainty_ma": 0.69,
    "constants_source": ("Smith et al., J. Chem. Phys. 158, 244304 (2023), Table II: "
                      "A0, B0, C0, S reduction, I^r; [1,2-13C] as reprinted from "
                      "Guarnieri; uncertainty level not stated"),
    "alt_geometry_re": GEOMETRY_RE_EAST1995,
    "alt_geometry_re_source": KETENE_RE_EAST1995_SOURCE,
    "bonds": {"C1=C2": [(0, 1)], "C1=O": [(0, 2)], "C2-H": [(1, 3), (1, 4)]},
    "angles": {"C1=C2-H": [(0, 1, 3), (0, 1, 4)], "H-C2-H": [(3, 1, 4)],
            "O=C1=C2": [(2, 0, 1)]},
}
