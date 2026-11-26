"""Validation-molecule entry: thioformaldehyde, H2CS (C2v).

Standalone (numpy only); mirrors the fields of ReferenceMolecule / Isotopologue
in dev/monofluoro_references.py so it can be pasted in. Every number is
transcribed from the downloaded sources; see thioformaldehyde.numbers.json next to this file
for the exact text line of each one, and dev/tests/test_validation_references.py for the consistency checks.

Atom order: C0, S1, H2, H3. C at the origin, S on +z, the H pair in the
xz-plane mirrored in x (C2 axis = z).
"""
from __future__ import annotations

import numpy as np

# ── Masses (u) ──────────────────────────────────────────────────────────────
# M_H, M_D, M_C12, M_C13 copied verbatim from dev/monofluoro_references.py.
M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
#: NOT in the module -- added here. NIST "Atomic Weights and Isotopic
#: Compositions" (physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=S,
#: read 2026-11-26), "Relative Atomic Mass": 32S 31.9720711744(14),
#: 33S 32.9714589098(15), 34S 33.967867004(47), 36S 35.96708071(20).
#: (dev/tests/test_kraitchman.py already uses the same 32S and 34S values.)
M_S32 = 31.9720711744
M_S33 = 32.9714589098
M_S34 = 33.967867004
M_S36 = 35.96708071

ELEMS = ["C", "S", "H", "H"]
PARENT_MASSES = (M_C12, M_S32, M_H, M_H)

# ── Equilibrium structures ──────────────────────────────────────────────────
# Muller et al., A&A 621, A143 (2019), arXiv:1812.01554v2, Table 7. The paper
# prints bond lengths in pm; (value, 1-sigma) below are AS PRINTED (pm, deg).
H2CS_RE_SE_PM = {   # row "reSE (CCSD(T))" -- PRIMARY
    "C=S": (160.909, 0.001), "C-H": (108.531, 0.002), "H-C-S": (121.758, 0.002)}
H2CS_RE_SE_B3LYP_PM = {   # row "reSE (B3LYP)"
    "C=S": (160.975, 0.002), "C-H": (108.526, 0.006), "H-C-S": (121.706, 0.005)}
H2CS_RE_SE_MP2_PM = {     # row "reSE (MP2)"
    "C=S": (160.934, 0.006), "C-H": (108.556, 0.015), "H-C-S": (121.759, 0.013)}


def pm_to_angstrom(d):
    """Unit conversion only: lengths / 100, angles unchanged."""
    return {k: ((v / 100.0, s / 100.0) if k != "H-C-S" else (v, s)) for k, (v, s) in d.items()}


H2CS_RE_SE = pm_to_angstrom(H2CS_RE_SE_PM)        # 1.60909(1), 1.08531(2) A, 121.758(2) deg
H2CS_RE_SE_SOURCE = (
    "r_e^SE, Muller, Maeda, Thorwirth, Lewen, Schlemmer, Medvedev, Winnewisser, De Lucia & "
    "Herbst, A&A 621, A143 (2019), arXiv:1812.01554v2, Table 7, row 'reSE (CCSD(T))': "
    "r(CS) 160.909(1) pm, r(CH) 108.531(2) pm, angle(HCS) 121.758(2) deg (8 isotopologues, "
    "CCSD(T) first-order vibrational + electronic + centrifugal corrections, Table 6)")
#: Stated uncertainty in mA: C-H 0.002 pm = 0.02 mA (C=S 0.01 mA).
H2CS_RE_SE_UNCERTAINTY_MA = 0.02


def h2cs_geometry(r_cs, r_ch, a_hcs_deg):
    """Planar C2v H2C=S: C0 at the origin, S1 on +z, H2/H3 in the xz-plane at +-x."""
    a = np.radians(a_hcs_deg)
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, r_cs],
        [r_ch * s, 0.0, r_ch * c],
        [-r_ch * s, 0.0, r_ch * c],
    ])


GEOMETRY_RE_SE = h2cs_geometry(H2CS_RE_SE["C=S"][0], H2CS_RE_SE["C-H"][0], H2CS_RE_SE["H-C-S"][0])

# ── Ground-state constants ──────────────────────────────────────────────────
# Tables 1 and 3 print A-(B+C)/2, (B+C)/2, (B-C)/4 with sigmas (MHz):
MU2019_COMBINATIONS = {   # (value, 1-sigma)
    "H2CS":       {"A-(B+C)/2": (274437.5932, 0.0115), "(B+C)/2": (17175.745955, 0.000196),
                   "(B-C)/4": (261.6240523, 0.0000165)},                       # Table 1
    "H2C33S":     {"A-(B+C)/2": (274588.054, 0.306), "(B+C)/2": (17024.740821, 0.000110),
                   "(B-C)/4": (257.050936, 0.000039)},                         # Table 1
    "H2C34S":     {"A-(B+C)/2": (274729.12, 0.34), "(B+C)/2": (16882.911552, 0.000112),
                   "(B-C)/4": (252.793027, 0.000073)},                         # Table 1
    "H2C36S":     {"A-(B+C)/2": (274987.91, 0.94), "(B+C)/2": (16621.73726, 0.00035),
                   "(B-C)/4": (245.04552, 0.00043)},                           # Table 1
    "H2(13)CS":   {"A-(B+C)/2": (275113.82, 0.35), "(B+C)/2": (16514.989247, 0.000141),
                   "(B-C)/4": (241.898578, 0.000082)},                         # Table 3
    "H2(13)C34S": {"A-(B+C)/2": (275418.9, 16.1), "(B+C)/2": (16219.21918, 0.00122),
                   "(B-C)/4": (233.32414, 0.00081)},                           # Table 3
    "HDCS":       {"A-(B+C)/2": (187214.14, 0.44), "(B+C)/2": (15501.179704, 0.000242),
                   "(B-C)/4": (304.81583, 0.00044)},                           # Table 3
    "D2CS":       {"A-(B+C)/2": (132198.92, 0.26), "(B+C)/2": (14200.0562, 0.0059),
                   "(B-C)/4": (352.105440, 0.000153)},                         # Table 3
}


def abc_from_combinations(x, y, z, sx, sy, sz):
    """A = x + y, B = y + 2z, C = y - 2z, with x = A-(B+C)/2, y = (B+C)/2,
    z = (B-C)/4. Sigmas in quadrature, treating x, y, z as UNCORRELATED: the
    paper prints no correlation matrix, so the covariance cannot be included.
    sigma_A = hypot(sx, sy); sigma_B = sigma_C = hypot(sy, 2 sz). The
    worst-case (fully correlated) linear sums sx + sy and sy + 2 sz can be up
    to sqrt(2) larger; the transcription notes list both for every species."""
    return ((x + y, y + 2 * z, y - 2 * z),
            (float(np.hypot(sx, sy)), float(np.hypot(sy, 2 * sz)), float(np.hypot(sy, 2 * sz))))


def _iso(label, subs, abc, decimals, sigma, note=""):
    return {"label": label, "subs": subs, "abc_mhz": abc, "decimals": decimals,
            "measured_sigma_mhz": sigma,
            "source": ("Muller et al., A&A 621, A143 (2019), Table 6 column B_i,0 (values, "
                       "decimals as printed); sigma propagated from Table "
                       + ("1" if label in ("H2CS", "H2C33S", "H2C34S", "H2C36S") else "3")),
            "note": note}


#: abc_mhz = Table 6 "B_i,0" exactly as printed; sigma = abc_from_combinations
#: on Tables 1/3 (3 significant figures). The transcription check verified both.
SPECIES = [
    _iso("H2CS", {}, (291613.34, 17698.994, 16652.498), (2, 3, 3), (0.0115, 0.000199, 0.000199)),
    _iso("H2C33S", {1: M_S33}, (291612.9, 17538.843, 16510.639), (1, 3, 3), (0.306, 0.000135, 0.000135),
         "Table 6 prints A0 = 291612.9 but Table 1 gives A = 274588.054 + 17024.740821 = 291612.795, "
         "which rounds to 291612.8: a 0.105 MHz (0.34 sigma) internal inconsistency in the paper."),
    _iso("H2C34S", {1: M_S34}, (291612.0, 17388.498, 16377.325), (1, 3, 3), (0.340, 0.000184, 0.000184)),
    _iso("H2C36S", {1: M_S36}, (291609.7, 17111.828, 16131.646), (1, 3, 3), (0.940, 0.000928, 0.000928),
         "Table 6 prints A0 = 291609.7 but Table 1 gives 291609.647 (rounds to 291609.6): "
         "0.053 MHz (0.06 sigma) internal inconsistency."),
    _iso("H2(13)CS", {0: M_C13}, (291628.8, 16998.786, 16031.192), (1, 3, 3), (0.350, 0.000216, 0.000216)),
    _iso("H2(13)C34S", {0: M_C13, 1: M_S34}, (291638., 16685.867, 15752.571), (0, 3, 3),
         (16.1, 0.00203, 0.00203)),
    _iso("HDCS", {2: M_D}, (202715.3, 16110.811, 14891.548), (1, 3, 3), (0.440, 0.000913, 0.000913)),
    _iso("D2CS", {2: M_D, 3: M_D}, (146399.0, 14904.267, 13495.845), (1, 3, 3), (0.260, 0.00591, 0.00591)),
]

#: Semi-experimental EQUILIBRIUM constants, Table 6 column "B_i,e (CCSD(T))" (MHz),
#: = B0 + CCSD(T) vibrational + electronic + centrifugal corrections. Not
#: measured constants; kept because the r_e^SE above was fitted to them, so the
#: rigid-rotor constants of GEOMETRY_RE_SE must reproduce them closely.
MU2019_BE_CCSDT = {
    "H2CS": (294414.850, 17773.682, 16761.817), "H2C33S": (294415.07, 17612.739, 16618.596),
    "H2C34S": (294414.91, 17461.653, 16484.009), "H2C36S": (294413.75, 17183.630, 16235.996),
    "H2(13)CS": (294415.62, 17068.807, 16133.462), "H2(13)C34S": (294426., 16754.300, 15852.580),
    "HDCS": (204370.79, 16175.679, 14988.163), "D2CS": (147572.97, 14961.330, 13582.139),
}

CONSTANTS_SOURCE = (
    "Muller et al., A&A 621, A143 (2019), arXiv:1812.01554v2: B0 from Table 6 (column B_i,0); "
    "1-sigma propagated (uncorrelated) from the A-(B+C)/2, (B+C)/2, (B-C)/4 sigmas of Tables 1 "
    "and 3 (S reduction, I^r)")

ENTRY = {
    "key": "thioformaldehyde",
    "name": "Thioformaldehyde",
    "formula": "H2CS",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE_SE,
    "masses": np.array(PARENT_MASSES),
    "species": SPECIES,
    "structure_source": H2CS_RE_SE_SOURCE,
    "constants_source": CONSTANTS_SOURCE,
    "structure_type": "r_e",
    "geometry_re_uncertainty_ma": H2CS_RE_SE_UNCERTAINTY_MA,
    "bonds": {"C=S": [(0, 1)], "C-H": [(0, 2), (0, 3)]},
    "angles": {"H-C-S": [(2, 0, 1), (3, 0, 1)], "H-C-H": [(2, 0, 3)]},
}
