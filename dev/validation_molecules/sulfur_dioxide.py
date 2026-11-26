"""Validation-molecule entry: sulfur dioxide, SO2 (C2v). Standalone; needs only numpy.

Every number below is transcribed from a downloaded source; sulfur_dioxide.numbers.json (same
folder) gives each one with its table/row/column and the exact text line, and
dev/tests/test_validation_references.py checks the geometry and the constants' consistency; an independent re-read of every number against the sources found no discrepancy (2026-11-26).

Sources
-------
Structure (r_e):
    r_e(SO) = 143.078(2) pm, angle(OSO) = 119.330(3) deg, attributed to
    Flaud & Lafferty, J. Mol. Spectrosc. 161, 396 (1993) [ref. 89], as quoted
    in Demaison, Mol. Phys. 105, 3109 (2007), Section 5.3 "Accuracy" -- in the
    running text, NOT in one of the review's tables. The review does not state
    the confidence level of (2) / (3). Flaud & Lafferty itself was not read.
Ground-state constants:
    NIST Triatomic Spectral Database, SO2 tables (F.J. Lovas, J. Phys. Chem.
    Ref. Data 7, 1445 (1978) = NIST ref. 78007):
      * 32S16O2, 33S16O2 (table 1) and 34S16O2 (table 4, (0,0,0) column):
        Watson's DETERMINABLE parameters A'', B'', C'' with uncertainties that
        the footnote states are one standard deviation. Used here as printed.
        The same tables also print "Derived Parameters (assuming planarity
        conditions)" A', B', C'; those are kept in DERIVED_PLANAR_78007 only.
      * all other species (table 5, "Rotational and quartic distortion
        constants ... by the Kielson [sic] and Wilson method"): A', B', C' for
        the (0,0,0) state with NO uncertainties (footnote a: "No uncertainties
        are given in the references cited."), so measured_sigma_mhz is None.
        Table 5 also lists 32S16O2, 33S16O2 and 34S16O2 from older references;
        those rows are kept in KW_TABLE_DUPLICATES only.
    The constant type therefore differs between the three sigma'd species
    (Watson determinable) and the rest (Kivelson-Wilson); for 32S16O2 the two
    differ by 0.04, 0.05 and 0.15 MHz in A, B, C (60778.5553 vs 60778.514,
    10317.91220 vs 10317.965, 8799.651056 vs 8799.804).

Atom order: 0 S, 1 O, 2 O. A singly substituted O is index 2 (the two O are
equivalent in the parent, so the choice is arbitrary); in the 17O18O species
17O is index 1 and 18O index 2.
"""

import numpy as np

# ── Masses (u) ───────────────────────────────────────────────────────────────
# Copied from dev/monofluoro_references.py (module constants):
M_O16, M_O18 = 15.9949146196, 17.9991610
# NEW (absent from the module). NIST "Atomic Weights and Isotopic
# Compositions", physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=S
# (and ele=O), read 2026-11-26, "Relative Atomic Mass" rows:
M_S32 = 31.9720711744     # 31.9720711744(14)
M_S33 = 32.9714589098     # 32.9714589098(15)
M_S34 = 33.967867004      # 33.967867004(47)
M_S36 = 35.96708071       # 35.96708071(20)
M_O17 = 16.99913175650    # 16.99913175650(69)
# The same O page gives 17.99915961286(76) for 18O; the module's M_O18
# (17.9991610) is an older value, 1.4e-6 u higher. Kept for consistency.

ELEMS = ["S", "O", "O"]
PARENT_MASSES = [M_S32, M_O16, M_O16]

# ── Equilibrium structure ────────────────────────────────────────────────────
#: (value, uncertainty as printed; confidence level not stated by the source)
SO2_RE = {
    "S-O": (1.43078, 0.00002),     # 143.078(2) pm, converted pm -> Angstrom (/100)
    "O-S-O": (119.330, 0.003),     # 119.330(3) deg
}
SO2_RE_PRINTED = {"S-O_pm": (143.078, 0.002), "O-S-O_deg": (119.330, 0.003)}
SO2_RE_SIGMA_LEVEL = "not stated by the source"
SO2_RE_SOURCE = (
    "r_e, Flaud & Lafferty, J. Mol. Spectrosc. 161, 396 (1993), as quoted in "
    "Demaison, Mol. Phys. 105, 3109 (2007), Sec. 5.3 (text): r_e(SO) = "
    "143.078(2) pm, angle(OSO) = 119.330(3) deg; uncertainty level not stated")


def so2_geometry_re(r_so, a_oso_deg):
    """C2v XY2: S at the origin, C2 axis along +y, molecule in the xy plane
    (the same construction as monofluoro_references._bent_xy)."""
    half = np.radians(a_oso_deg) / 2.0
    return np.array([
        [0.0, 0.0, 0.0],
        [r_so * np.sin(half), r_so * np.cos(half), 0.0],
        [-r_so * np.sin(half), r_so * np.cos(half), 0.0],
    ])


GEOMETRY_RE = so2_geometry_re(SO2_RE["S-O"][0], SO2_RE["O-S-O"][0])

# ── Ground-state constants (MHz) ─────────────────────────────────────────────
#: label, subs, (A, B, C), decimals as printed, 1σ (MHz) or None, constant type, NIST ref.
_W = "Watson determinable A'', B'', C''"
_KW = "Kivelson-Wilson A', B', C'"
SPECIES = [
    {"label": "32S16O2", "subs": {}, "abc_mhz": (60778.5553, 10317.91220, 8799.651056),
         "decimals": (4, 5, 6), "measured_sigma_mhz": (0.0056, 0.00106, 0.000927),
         "constant_type": _W, "ref": "78007"},
    {"label": "33S16O2", "subs": {0: M_S33}, "abc_mhz": (59856.4540, 10318.1328, 8780.08934),
         "decimals": (4, 4, 5), "measured_sigma_mhz": (0.0681, 0.0121, 0.00116),
         "constant_type": _W, "ref": "78007"},
    {"label": "34S16O2", "subs": {0: M_S34}, "abc_mhz": (58991.1970, 10318.3582, 8761.25810),
         "decimals": (4, 4, 5), "measured_sigma_mhz": (0.0150, 0.0028, 0.00246),
         "constant_type": _W, "ref": "78007"},
    {"label": "36S16O2", "subs": {0: M_S36}, "abc_mhz": (57399.507, 10318.7849, 8725.4710),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "64014"},
    {"label": "32S16O18O", "subs": {2: M_O18}, "abc_mhz": (59101.191, 9724.523, 8331.647),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65016"},
    {"label": "32S18O2", "subs": {1: M_O18, 2: M_O18}, "abc_mhz": (57384.526, 9170.351, 7889.6007),
         "decimals": (3, 3, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "63010"},
    {"label": "34S16O18O", "subs": {0: M_S34, 2: M_O18}, "abc_mhz": (57314.690, 9724.228, 8294.697),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65016"},
    {"label": "34S18O2", "subs": {0: M_S34, 1: M_O18, 2: M_O18},
         "abc_mhz": (55597.855, 9170.815, 7854.9773),
         "decimals": (3, 3, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "63017"},
    {"label": "33S16O18O", "subs": {0: M_S33, 2: M_O18}, "abc_mhz": (58179.596, 9724.495, 8312.748),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68017"},
    {"label": "33S18O2", "subs": {0: M_S33, 1: M_O18, 2: M_O18},
         "abc_mhz": (56463.093, 9170.713, 7871.928),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68017"},
    {"label": "32S16O17O", "subs": {2: M_O17}, "abc_mhz": (59883.544, 10008.076, 8555.1956),
         "decimals": (3, 3, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65017"},
    {"label": "32S17O2", "subs": {1: M_O17, 2: M_O17}, "abc_mhz": (58977.398, 9709.1769, 8317.7300),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68037"},
    {"label": "34S16O17O", "subs": {0: M_S34, 2: M_O17}, "abc_mhz": (58096.578, 10008.3346, 8517.6794),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65017"},
    {"label": "34S17O2", "subs": {0: M_S34, 1: M_O17, 2: M_O17},
         "abc_mhz": (57190.236, 9709.6284, 8281.2770),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68037"},
    {"label": "33S16O17O", "subs": {0: M_S33, 2: M_O17}, "abc_mhz": (58961.521, 10008.284, 8535.969),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68017"},
    {"label": "33S17O2", "subs": {0: M_S33, 1: M_O17, 2: M_O17},
         "abc_mhz": (58055.949, 9709.489, 8299.264),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68017"},
    {"label": "32S17O18O", "subs": {1: M_O17, 2: M_O18}, "abc_mhz": (58184.900, 9435.2693, 8100.7437),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65017"},
    {"label": "33S17O18O", "subs": {0: M_S33, 1: M_O17, 2: M_O18},
         "abc_mhz": (57263.881, 9435.627, 8082.606),
         "decimals": (3, 3, 3), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "68017"},
    {"label": "34S17O18O", "subs": {0: M_S34, 1: M_O17, 2: M_O18},
         "abc_mhz": (56398.745, 9435.6558, 8065.1794),
         "decimals": (3, 4, 4), "measured_sigma_mhz": None, "constant_type": _KW, "ref": "65017"},
]

#: Same tables, "Derived Parameters (assuming planarity conditions)", 1σ.
DERIVED_PLANAR_78007 = {
    "32S16O2": {"abc_mhz": (60778.5452, 10317.9625, 8799.8050), "decimals": (4, 4, 4),
                    "measured_sigma_mhz": (0.0056, 0.0010, 0.0009)},
    "33S16O2": {"abc_mhz": (59856.444, 10318.1809, 8780.2420), "decimals": (3, 4, 4),
                    "measured_sigma_mhz": (0.068, 0.0117, 0.0121)},
    "34S16O2": {"abc_mhz": (58991.187, 10318.4033, 8761.4064), "decimals": (3, 4, 4),
                    "measured_sigma_mhz": (0.015, 0.0027, 0.0024)},
}
#: Table 5 (Kivelson-Wilson) rows for the three sigma'd species, older refs, no σ.
KW_TABLE_DUPLICATES = {
    "32S16O2": {"abc_mhz": (60778.514, 10317.965, 8799.804), "decimals": (3, 3, 3), "ref": "68036"},
    "34S16O2": {"abc_mhz": (58991.174, 10318.399, 8761.394), "decimals": (3, 3, 3), "ref": "65015"},
    "33S16O2": {"abc_mhz": (59856.387, 10318.283, 8780.124), "decimals": (3, 3, 3), "ref": "68017"},
}

ENTRY = {
    "key": "sulfur_dioxide",
    "name": "Sulfur dioxide",
    "formula": "SO2",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE,
    "masses": np.array(PARENT_MASSES),
    "species": SPECIES,
    "structure_source": SO2_RE_SOURCE,
    "structure_type": "r_e",
    # 143.078(2) pm -> 0.02 mA on the bond, as quoted (level unstated).
    "geometry_re_uncertainty_ma": 0.02,
    "constants_source": ("NIST Triatomic Spectral Database, SO2 (Lovas, J. Phys. Chem. Ref. "
                      "Data 7, 1445 (1978)): Watson determinable A'', B'', C'' with 1-sigma "
                      "for 32S/33S/34S16O2; Kivelson-Wilson A', B', C' without uncertainties "
                      "for the other 16 ground-state species"),
    "bonds": {"S-O": [(0, 1), (0, 2)]},
    "angles": {"O-S-O": [(1, 0, 2)]},
}
