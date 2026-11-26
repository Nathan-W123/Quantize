"""Validation-molecule entry: thiophene, c-C4H4S (C2v, 9 atoms). Standalone; needs only numpy.

Every number is transcribed from a downloaded source; thiophene.numbers.json (same folder)
gives each one with its table / row / column and the exact text line. An independent re-read
against its own downloads found no discrepancy (2026-11-26); dev/tests/test_validation_references.py
checks the geometry and the constants' consistency.

WHICH PAPERS: the 2025 one for the structure, the 2021 one for the SCORED constants
-----------------------------------------------------------------------------------
The 2025 study (Esselman, Atwood, Nguyen, Stanton, Woods & McMahon; Orr et al. is the 2021
paper it updates) gives the structure. Its Table II constants print no uncertainties, so
ENTRY scores the 2021 S-reduced constants with their printed uncertainties, the 24 species
the 2021 structure used (decisions D-158, after D-155); the 2025 set stays in SPECIES.

Structure (primary, r_e^SE):
    Esselman et al., J. Chem. Phys. 163, 144301 (2025), doi 10.1063/5.0292964, Table IV,
    column "re SE" with N isotopologues = 46 (the recommended structure, Fig. 6(a)):
    CCSD(T)/cc-pCVTZ vibration-rotation and electron-mass corrections. Uncertainties are 2σ
    (Table IV footnote c: "Uncertainty reported is 2σ."). RC3-C4, θC3-C2-H, θC4-C3-H and
    θC2-C3-C4 are printed in ITALICS = "dependent parameter[s] ... with propagated
    statistical uncertainty" (footnote b); the other eight are the independent ones (they are
    also bold, but footnote a's bold only marks the best values for comparison) -- read from
    the PDF's fonts (pdftohtml -xml), see thiophene.numbers.json. The builder below
    takes the eight independent ones; the dependent four are kept for the check only.
    Bonds are printed to 4 decimals (1e-4 A), so the rounding alone is +-0.05 mA.
Structure (comparison): Orr et al., J. Chem. Phys. 154, 244310 (2021), doi
    10.1063/5.0055267, Table IV "re SE" (24 isotopologues, 2σ, 5 decimals; it prints
    1/2 θC2-S1-C5 as the independent angle and θC2-S1-C5, RC3-C4 as dependent, footnote g).
    And the computed "CCSD(T) BTE" r_e from 2025 Table IV (Orr 2021's best theoretical
    estimate, cc-pCV5Z + corrections) -- COMPUTED, no uncertainty, not a reference.
Ground-state constants (primary): 2025 Table II, "Average determinable rotational constants
    (B0'' ave)" for all 49 isotopologues the paper lists. These are NOT S-reduced A0/B0/C0:
    each is the average of the A- and S-reduction constants after conversion to the
    centrifugal-distortion-free "determinable" constants (2025 Eqs. (1)-(6)); they are what
    the r_e^SE fit used. Table II prints NO uncertainty for them (footnote a: values rounded
    "to either the last significant digit determined by least-squares fitting ... using
    two-digit statistical uncertainties or to the last digit of agreement between the
    B0''(A) - B0''(S) values"), so measured_sigma_mhz is None for every species and only the
    quoted-precision floor applies. The printed |B0''(A) - B0''(S)| values are kept, as
    printed, under abs_A_minus_S_mhz_as_printed: a diagnostic, NOT an uncertainty.
    Three species are marked red (footnote b) = excluded from the 46-isotopologue fit:
    [2-2H, 3-13C], [3-2H, 33S], [2,3-2H, 5-13C] (used_in_re_SE=False). They have A, B and C,
    so they are listed, flagged.
    The S-reduced constants of the other 48 isotopologues are only in the paper's
    supplementary material (not obtained). For the PARENT, 2025 Table I prints S-reduced
    A0, B0, C0 with parenthesised uncertainties (level not stated): PARENT_2025_TABLE_I_S.
Ground-state constants (secondary, with uncertainties): 2021 Table II, S reduction, I^r,
    26 isotopologues (24 used in the 2021 structure; [2,3,4,5-2H, 34S] and [3-2H, 33S]
    shaded = not used). Parenthesised uncertainties in units of the last digit; their
    CONFIDENCE LEVEL IS NOT STATED in the paper. Kept as SPECIES_2021_S, with the
    quartic constants needed to turn them into determinable constants.

Atom order (the papers' numbering, 2025 Fig. 1 / 2021 Table VI):
    0 S1, 1 C2, 2 C3, 3 C4, 4 C5, 5 H2 (on C2), 6 H3 (on C3), 7 H4 (on C4), 8 H5 (on C5).
    So [n-13C] -> index n-1 and [n-2H] -> index n+3; [2,3-2H, 5-13C] = {5: D, 6: D, 4: 13C}.

Doubts
    * [3,4-2H, 34S] (4 microwave lines): its Table II constants give an inertial defect of
      0.063325 u A^2, but Table III prints 0.06314 for it -- 14x what rounding allows; the
      column position was re-checked in the PDF, so the inconsistency is inside the paper
      (equivalent to 0.021 MHz in A, or 0.008 in B, or 0.003 in C). Kept as printed.
    * 2025 Table II gives 6909 lines for the parent, Tables I and III 6923; Table I's S-reduced
      parent constants map to B0'' values 15-41 Hz from Table II's (1-2.5x Table I's sigmas).
    * 2025 abstract says precisions <=0.0002 A and <=0.02 deg; Table IV's 2σ column prints up
      to 0.0003 A (RC3-C4, dependent) and 0.024 deg (θS1-C2-H, independent). Stored as printed.
    * [3,4-2H] moved by +0.145 MHz in A and -0.059 MHz in B between the 2021 (203 lines) and
      2025 (213 lines) fits -- 9 and 7.5 of the 2021 sigmas; both sets pass their own
      inertial-defect checks, so it is a refit, not a slip.

The r_e^SE vs r_e discrepancy, one line: Orr 2021 found the CCSD(T)/cc-pCV5Z-based best
theoretical r_e outside the r_e^SE 2σ bars for the C2/H2 parameters (RC2-C3, RC2-H,
θS1-C2-H) and blamed C2/H2 lying near the b principal axis; the 2025 study, with 46
isotopologues and b-type lines from axis-rotating species (|a| of C2/C5 0.054 -> 0.15-0.2 A),
reproduces the 2021 r_e^SE and concludes the remaining disagreement (5 of 8 parameters: RS-C2,
RC2-C3, RC2-H, θC2-S-C5, θS-C2-H) lies in the computed r_e, not in r_e^SE.
"""

import numpy as np

# ── Masses (u) ───────────────────────────────────────────────────────────────
# Copied from dev/monofluoro_references.py (module constants):
M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
# NEW (absent from the module). NIST "Atomic Weights and Isotopic Compositions",
# physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=S&ascii=ascii2&isotype=all
# (read 2026-11-26):
#   Mass Number = 32  "Relative Atomic Mass = 31.9720711744(14)"
#   Mass Number = 33  "Relative Atomic Mass = 32.9714589098(15)"
#   Mass Number = 34  "Relative Atomic Mass = 33.967867004(47)"
# (The same NIST H page gives 1.00782503223(9) for 1H; the module's M_H is 1.6e-10 u lower,
# an older value, kept here as the module has it. 2H and 13C agree with NIST to all digits.)
M_S32 = 31.9720711744
M_S33 = 32.9714589098
M_S34 = 33.967867004

ELEMS = ["S", "C", "C", "C", "C", "H", "H", "H", "H"]
ATOM_LABELS = ["S1", "C2", "C3", "C4", "C5", "H2", "H3", "H4", "H5"]
PARENT_MASSES = [M_S32, M_C12, M_C12, M_C12, M_C12, M_H, M_H, M_H, M_H]

# ── Equilibrium structures ───────────────────────────────────────────────────
#: 2025 Table IV, "re SE", N isotopologues = 46: (value, uncertainty AS PRINTED, 2σ).
#: The eight INDEPENDENT parameters (not italic in the table; footnote b).
THIOPHENE_RE_SE = {
    "S1-C2": (1.7104, 0.0001),        # RS1–C2 (Å)       1.7104 (1)
    "C2-C3": (1.3658, 0.0002),        # RC2–C3 (Å)       1.3658 (2)
    "C2-H": (1.0770, 0.0001),         # RC2–H (Å)        1.0770 (1)
    "C3-H": (1.0786, 0.0001),         # RC3–H (Å)        1.0786 (1)
    "S1-C2-C3": (111.600, 0.011),     # θS1–C2–C3 (○)  111.600 (11)
    "S1-C2-H": (120.087, 0.024),      # θS1–C2–H (○)   120.087 (24)
    "C2-C3-H": (123.413, 0.015),      # θC2–C3–H (○)   123.413 (15)
    "C2-S1-C5": (92.056, 0.011),      # θC2–S1–C5 (○)   92.056 (11)
}
#: Same column, the four DEPENDENT parameters (italic in the table), 2σ as printed.
THIOPHENE_RE_SE_DEPENDENT = {
    "C3-C4": (1.4222, 0.0003),        # RC3–C4 (Å)       1.4222 (3)
    "C3-C2-H": (128.313, 0.026),      # θC3–C2–H (○)   128.313 (26)
    "C4-C3-H": (124.215, 0.009),      # θC4–C3–H (○)   124.215 (9)
    "C2-C3-C4": (112.372, 0.018),     # θC2–C3–C4 (○)  112.372 (18)
}
THIOPHENE_RE_SE_SIGMA_LEVEL = "2sigma"     # Table IV footnote c, as stated by the source
#: The same, halved to 1σ (arithmetic on the printed 2σ, nothing else).
THIOPHENE_RE_SE_1SIGMA = {k: (v, u / 2.0) for k, (v, u) in THIOPHENE_RE_SE.items()}
THIOPHENE_RE_SE_SOURCE = (
    "r_e^SE, Esselman, Atwood, Nguyen, Stanton, Woods & McMahon, J. Chem. Phys. 163, 144301 "
    "(2025), doi 10.1063/5.0292964, Table IV 're SE' (46 isotopologues, CCSD(T)/cc-pCVTZ "
    "vibration-rotation and electron-mass corrections): r(S1-C2) = 1.7104(1), r(C2-C3) = "
    "1.3658(2), r(C2-H) = 1.0770(1), r(C3-H) = 1.0786(1) A, S1-C2-C3 = 111.600(11), "
    "S1-C2-H = 120.087(24), C2-C3-H = 123.413(15), C2-S1-C5 = 92.056(11) deg, 2-sigma")

#: 2021 Table IV "re SE" (24 isotopologues), 2σ as printed. Independent set as that table
#: prints it: half of C2-S1-C5 (the full angle and RC3-C4 are footnote g, dependent).
THIOPHENE_RE_SE_2021 = {
    "S1-C2": (1.71049, 0.00018),      # 1.710 49 (18)
    "C2-C3": (1.36564, 0.00031),      # 1.365 64 (31)
    "C2-H": (1.07714, 0.00017),       # 1.077 14 (17)
    "C3-H": (1.07856, 0.00014),       # 1.078 56 (14)
    "half C2-S1-C5": (46.024, 0.007), # 46.024 (7)
    "S1-C2-C3": (111.608, 0.016),     # 111.608 (16)
    "S1-C2-H": (120.065, 0.028),      # 120.065 (28)
    "C2-C3-H": (123.414, 0.023),      # 123.414 (23)
}
THIOPHENE_RE_SE_2021_DEPENDENT = {"C3-C4": (1.4224, 0.0010), "C2-S1-C5": (92.047, 0.015)}
THIOPHENE_RE_SE_2021_SOURCE = (
    "r_e^SE, Orr et al., J. Chem. Phys. 154, 244310 (2021), doi 10.1063/5.0055267, Table IV "
    "(24 isotopologues, CCSD(T)/cc-pCVTZ corrections), 2-sigma")

#: COMPUTED best theoretical estimate r_e (2025 Table IV column "CCSD(T) BTE", from Orr 2021:
#: CCSD(T)/cc-pCV5Z + basis, correlation, relativistic and DBOC corrections). No uncertainty.
#: Kept only to show the r_e^SE - r_e gap the two papers discuss; NOT a reference.
THIOPHENE_RE_BTE = {
    "S1-C2": 1.7101, "C2-C3": 1.3666, "C2-H": 1.0764, "C3-H": 1.0787,
    "S1-C2-C3": 111.610, "S1-C2-H": 120.241, "C2-C3-H": 123.411, "C2-S1-C5": 92.072,
}
THIOPHENE_RE_BTE_DEPENDENT = {"C3-C4": 1.4223, "C3-C2-H": 128.149, "C4-C3-H": 124.235, "C2-C3-C4": 112.354}

PARAM_ORDER = ("S1-C2", "C2-C3", "C2-H", "C3-H", "S1-C2-C3", "S1-C2-H", "C2-C3-H", "C2-S1-C5")


def thiophene_geometry_re(r_s1c2, r_c2c3, r_c2h, r_c3h, a_s1c2c3, a_s1c2h, a_c2c3h, a_c2s1c5):
    """Planar C2v thiophene in this file's atom order (S1, C2, C3, C4, C5, H2, H3, H4, H5)
    from the eight independent parameters of 2025 Table IV (Å, degrees).

    S1 at the origin, the C2 axis along x with the ring on the -x side (the a axis of the
    papers' Table VI frame), the molecule in the xy plane; C2, C3 (and their H) on +y,
    C5, C4 on -y. C3 is placed from C2 by r(C2-C3) and the angle S1-C2-C3 (rotating the
    C2->S1 direction towards the ring); H2 by the angle S1-C2-H on the outside; H3 from C3
    by the angle C2-C3-H. C4/C5/H4/H5 are the mirror images, so r(C3-C4) and the other
    three angles follow (they are the source's dependent parameters).
    """
    def rot(v, deg):
        t = np.radians(deg)
        return np.array([np.cos(t) * v[0] - np.sin(t) * v[1], np.sin(t) * v[0] + np.cos(t) * v[1]])

    half = np.radians(a_c2s1c5 / 2.0)
    s1 = np.zeros(2)
    c2 = r_s1c2 * np.array([-np.cos(half), np.sin(half)])
    u = (s1 - c2) / np.linalg.norm(s1 - c2)          # C2 -> S1
    c3 = c2 + r_c2c3 * rot(u, -a_s1c2c3)              # into the ring
    h2 = c2 + r_c2h * rot(u, +a_s1c2h)                # outside the ring
    v = (c2 - c3) / np.linalg.norm(c2 - c3)          # C3 -> C2
    h3 = c3 + r_c3h * rot(v, +a_c2c3h)

    def m(p):
        return np.array([p[0], -p[1]])

    xy = [s1, c2, c3, m(c3), m(c2), h2, h3, m(h3), m(h2)]
    return np.array([[p[0], p[1], 0.0] for p in xy])


GEOMETRY_RE = thiophene_geometry_re(*(THIOPHENE_RE_SE[k][0] for k in PARAM_ORDER))
GEOMETRY_RE_2021 = thiophene_geometry_re(
    *(THIOPHENE_RE_SE_2021[k][0] for k in PARAM_ORDER[:-1]), 2.0 * THIOPHENE_RE_SE_2021["half C2-S1-C5"][0])
GEOMETRY_BTE = thiophene_geometry_re(*(THIOPHENE_RE_BTE[k] for k in PARAM_ORDER))

# ── Ground-state constants ───────────────────────────────────────────────────
CONSTANTS_KIND = ("determinable constants B0'' (centrifugal-distortion-free), average of the "
                  "A- and S-reduction (I^r) values, 2025 Table II")
SIGMA_LEVEL_TABLE_II_2025 = "no uncertainties printed (measured_sigma_mhz = None)"
SIGMA_LEVEL_TABLE_II_2021 = "not stated by the source (printed parenthesised uncertainties)"

#: label, subs {atom index: mass}, (A0'', B0'', C0'') MHz as printed, decimals as printed,
#: measured_sigma_mhz None (not printed), used_in_re_SE (False = red in Table II),
#: N lines, |X0''(A) - X0''(S)| for X = A, B, C as printed (MHz; diagnostic, not a sigma).
SPECIES = [
    {"label": 'Thiophene', "subs": {},
         "abc_mhz": (8041.596617, 5418.263644, 3235.783055), "decimals": (6, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 6909, "abs_A_minus_S_mhz_as_printed": ('0.000 013', '0.000 011', '0.000 015')},
    {"label": '[34S]', "subs": {0: M_S34},
         "abc_mhz": (8041.71643, 5274.18554, 3183.84539), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 2136, "abs_A_minus_S_mhz_as_printed": ('0.000 12', '0.000 079', '0.000 003')},
    {"label": '[2-13C]', "subs": {1: M_C13},
         "abc_mhz": (7852.77625, 5418.406363, 3204.808691), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1663, "abs_A_minus_S_mhz_as_printed": ('0.000 31', '0.000 072', '0.000 014')},
    {"label": '[3-13C]', "subs": {2: M_C13},
         "abc_mhz": (7981.28134, 5319.306824, 3190.621115), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1669, "abs_A_minus_S_mhz_as_printed": ('0.000 20', '0.000 084', '0.000 028')},
    {"label": '[33S]', "subs": {0: M_S33},
         "abc_mhz": (8041.6578, 5344.29988, 3209.260358), "decimals": (4, 5, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 536, "abs_A_minus_S_mhz_as_printed": ('0.001 4', '0.000 25', '0.000 003')},
    {"label": '[2-2H]', "subs": {5: M_D},
         "abc_mhz": (7437.198, 5413.722581, 3131.817145), "decimals": (4, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1360, "abs_A_minus_S_mhz_as_printed": ('0.000 1', '0.000 059', '0.000 011')},
    {"label": '[2-2H, 34S]', "subs": {5: M_D, 0: M_S34},
         "abc_mhz": (7436.7851, 5270.744923, 3083.343123), "decimals": (4, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 241, "abs_A_minus_S_mhz_as_printed": ('0.000 2', '0.000 031', '0.000 021')},
    {"label": '[2-2H, 2-13C]', "subs": {5: M_D, 1: M_C13},
         "abc_mhz": (7282.1504, 5413.95545, 3104.05399), "decimals": (4, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000 1', '0.000 01', '0.000 00')},
    {"label": '[2-2H, 3-13C]', "subs": {5: M_D, 2: M_C13},
         "abc_mhz": (7385.37, 5316.47799, 3090.08018), "decimals": (2, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": False, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.000 24', '0.000 24')},
    {"label": '[2-2H, 4-13C]', "subs": {5: M_D, 3: M_C13},
         "abc_mhz": (7386.522, 5311.7909, 3088.6175), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 0', '0.000 1')},
    {"label": '[2-2H, 5-13C]', "subs": {5: M_D, 4: M_C13},
         "abc_mhz": (7268.88, 5413.6219, 3101.5123), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 1')},
    {"label": '[3-2H]', "subs": {6: M_D},
         "abc_mhz": (7855.9984, 5138.23084, 3105.224718), "decimals": (5, 5, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1894, "abs_A_minus_S_mhz_as_printed": ('0.000 02', '0.000 01', '0.000 006')},
    {"label": '[3-2H, 34S]', "subs": {6: M_D, 0: M_S34},
         "abc_mhz": (7854.54195, 5003.23039, 3055.167806), "decimals": (5, 5, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 667, "abs_A_minus_S_mhz_as_printed": ('0.000 05', '0.000 01', '0.000 006')},
    {"label": '[3-2H, 2-13C]', "subs": {6: M_D, 1: M_C13},
         "abc_mhz": (7681.75, 5137.726, 3077.43306), "decimals": (3, 3, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 133, "abs_A_minus_S_mhz_as_printed": ('0.002', '0.001', '0.000 01')},
    {"label": '[3-2H, 3-13C]', "subs": {6: M_D, 2: M_C13},
         "abc_mhz": (7816.981, 5045.95, 3065.28942), "decimals": (3, 2, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 112, "abs_A_minus_S_mhz_as_printed": ('0.006', '0.00', '0.000 03')},
    {"label": '[3-2H, 4-13C]', "subs": {6: M_D, 3: M_C13},
         "abc_mhz": (7775.957, 5060.89, 3064.44297), "decimals": (3, 2, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 113, "abs_A_minus_S_mhz_as_printed": ('0.002', '0.00', '0.000 01')},
    {"label": '[3-2H, 5-13C]', "subs": {6: M_D, 4: M_C13},
         "abc_mhz": (7671.879, 5138.1415, 3075.99416), "decimals": (3, 4, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 147, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 1', '0.000 00')},
    {"label": '[3-2H, 33S]', "subs": {6: M_D, 0: M_S33},
         "abc_mhz": (7854.4, 5069.32, 3079.66483), "decimals": (1, 2, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": False, "n_lines": 57, "abs_A_minus_S_mhz_as_printed": ('1.2', '0.50', '0.000 33')},
    {"label": '[2,3-2H]', "subs": {5: M_D, 6: M_D},
         "abc_mhz": (7277.0016, 5137.62852, 3010.35181), "decimals": (5, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1201, "abs_A_minus_S_mhz_as_printed": ('0.000 94', '0.000 31', '0.000 36')},
    {"label": '[2,3-2H, 34S]', "subs": {5: M_D, 6: M_D, 0: M_S34},
         "abc_mhz": (7276.614, 5002.8001, 2963.4633), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 1')},
    {"label": '[2,3-2H, 5-13C]', "subs": {5: M_D, 6: M_D, 4: M_C13},
         "abc_mhz": (7112.06, 5137.394, 2981.752), "decimals": (2, 3, 3), "measured_sigma_mhz": None,
         "used_in_re_SE": False, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.000', '0.000')},
    {"label": '[2,4-2H]', "subs": {5: M_D, 7: M_D},
         "abc_mhz": (7288.38376, 5124.658844, 3007.832771), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1186, "abs_A_minus_S_mhz_as_printed": ('0.000 21', '0.000 065', '0.000 023')},
    {"label": '[2,4-2H, 34S]', "subs": {5: M_D, 7: M_D, 0: M_S34},
         "abc_mhz": (7284.13, 4992.4457, 2961.0899), "decimals": (2, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.000 0', '0.000 0')},
    {"label": '[2,4-2H, 2-13C]', "subs": {5: M_D, 7: M_D, 1: M_C13},
         "abc_mhz": (7137.399, 5124.0055, 2981.5794), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 0')},
    {"label": '[2,4-2H, 4-13C]', "subs": {5: M_D, 7: M_D, 3: M_C13},
         "abc_mhz": (7257.82, 5029.063, 2969.556), "decimals": (2, 3, 3), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.001', '0.000')},
    {"label": '[2,5-2H]', "subs": {5: M_D, 8: M_D},
         "abc_mhz": (6903.42764, 5410.6994, 3032.132301), "decimals": (5, 5, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1653, "abs_A_minus_S_mhz_as_printed": ('0.000 09', '0.000 30', '0.000 093')},
    {"label": '[2,5-2H, 34S]', "subs": {5: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6903.53651, 5268.34411, 2986.90827), "decimals": (5, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 423, "abs_A_minus_S_mhz_as_printed": ('0.000 01', '0.000 00', '0.000 00')},
    {"label": '[2,5-2H, 2-13C]', "subs": {5: M_D, 8: M_D, 1: M_C13},
         "abc_mhz": (6763.9252, 5410.8, 3004.93117), "decimals": (4, 1, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 165, "abs_A_minus_S_mhz_as_printed": ('0.000 0', '0.0', '0.000 01')},
    {"label": '[2,5-2H, 3-13C]', "subs": {5: M_D, 8: M_D, 2: M_C13},
         "abc_mhz": (6860.1601, 5310.3978, 2992.1604), "decimals": (4, 4, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 178, "abs_A_minus_S_mhz_as_printed": ('0.000 1', '0.000 1', '0.000 01')},
    {"label": '[3,4-2H]', "subs": {6: M_D, 7: M_D},
         "abc_mhz": (7616.703, 4914.62173, 2986.05587), "decimals": (4, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 213, "abs_A_minus_S_mhz_as_printed": ('0.000 4', '0.000 08', '0.000 00')},
    {"label": '[3,4-2H, 34S]', "subs": {6: M_D, 7: M_D, 0: M_S34},
         "abc_mhz": (7616.757, 4785.0468, 2937.7277), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 0')},
    {"label": '[3,4-2H, 2-13C]', "subs": {6: M_D, 7: M_D, 1: M_C13},
         "abc_mhz": (7446.968, 4914.9136, 2959.6668), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 0')},
    {"label": '[3,4-2H, 3-13C]', "subs": {6: M_D, 7: M_D, 2: M_C13},
         "abc_mhz": (7561.86, 4839.5138, 2949.8191), "decimals": (2, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 4, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.000 0', '0.000 0')},
    {"label": '[2,3,4-2H]', "subs": {5: M_D, 6: M_D, 7: M_D},
         "abc_mhz": (7072.2762, 4909.47012, 2896.80283), "decimals": (4, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1056, "abs_A_minus_S_mhz_as_printed": ('0.000 1', '0.000 04', '0.000 06')},
    {"label": '[2,3,4-2H, 34S]', "subs": {5: M_D, 6: M_D, 7: M_D, 0: M_S34},
         "abc_mhz": (7071.868, 4781.0158, 2851.5234), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 6, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 0', '0.000 0')},
    {"label": '[2,3,4-2H, 2-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 1: M_C13},
         "abc_mhz": (6932.03, 4909.6691, 2873.0397), "decimals": (2, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.00', '0.000 0', '0.000 0')},
    {"label": '[2,3,4-2H, 3-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 2: M_C13},
         "abc_mhz": (7025.233, 4835.6718, 2863.1489), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 0', '0.000 0')},
    {"label": '[2,3,4-2H, 4-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 3: M_C13},
         "abc_mhz": (7025.652, 4832.0627, 2861.943), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 1', '0.000 1')},
    {"label": '[2,3,4-2H, 5-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 4: M_C13},
         "abc_mhz": (6919.665, 4909.685, 2870.9162), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 0', '0.000 1')},
    {"label": '[2,3,5-2H]', "subs": {5: M_D, 6: M_D, 8: M_D},
         "abc_mhz": (6775.85029, 5124.45005, 2916.733219), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1678, "abs_A_minus_S_mhz_as_printed": ('0.000 05', '0.000 003', '0.000 002')},
    {"label": '[2,3,5-2H, 34S]', "subs": {5: M_D, 6: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6773.39798, 4992.25393, 2872.97115), "decimals": (5, 5, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 465, "abs_A_minus_S_mhz_as_printed": ('0.000 01', '0.000 00', '0.000 01')},
    {"label": '[2,3,5-2H, 2-13C]', "subs": {5: M_D, 6: M_D, 8: M_D, 1: M_C13},
         "abc_mhz": (6647.28, 5123.0487, 2892.2035), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 0')},
    {"label": '[2,3,5-2H, 3-13C]', "subs": {5: M_D, 6: M_D, 8: M_D, 2: M_C13},
         "abc_mhz": (6752.175, 5028.7918, 2881.18001), "decimals": (3, 4, 5), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 1', '0.000 01')},
    {"label": '[2,3,5-2H, 4-13C]', "subs": {5: M_D, 6: M_D, 8: M_D, 3: M_C13},
         "abc_mhz": (6710.866, 5049.7571, 2880.4451), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 0')},
    {"label": '[2,3,5-2H, 5-13C]', "subs": {5: M_D, 6: M_D, 8: M_D, 4: M_C13},
         "abc_mhz": (6639.519, 5123.7918, 2890.9588), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 5, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 0', '0.000 1')},
    {"label": '[2,3,4,5-2H]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D},
         "abc_mhz": (6587.73178, 4905.809308, 2810.899256), "decimals": (5, 6, 6), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 1359, "abs_A_minus_S_mhz_as_printed": ('0.000 05', '0.000 005', '0.000 005')},
    {"label": '[2,3,4,5-2H, 34S]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6587.818, 4778.0638, 2768.491), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 39, "abs_A_minus_S_mhz_as_printed": ('0.001', '0.000 0', '0.000 0')},
    {"label": '[2,3,4,5-2H, 2-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D, 1: M_C13},
         "abc_mhz": (6460.569, 4906.0497, 2787.5383), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 6, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 024', '0.000 050')},
    {"label": '[2,3,4,5-2H, 3-13C]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D, 2: M_C13},
         "abc_mhz": (6547.442, 4829.7357, 2778.5072), "decimals": (3, 4, 4), "measured_sigma_mhz": None,
         "used_in_re_SE": True, "n_lines": 6, "abs_A_minus_S_mhz_as_printed": ('0.000', '0.000 1', '0.000 0')},
]

#: 2021 Table II, S reduction, I^r: (A0, B0, C0) MHz as printed, decimals, the printed
#: parenthesised uncertainties converted to MHz (level NOT stated), whether the 2021 structure
#: used the species (shaded columns = not used), and DJ, DJK, d1, d2 in kHz as printed
#: (values in square brackets in the paper were held fixed at CCSD(T) predictions).
SPECIES_2021_S = [
    {"label": 'Thiophene', "subs": {},
         "abc_mhz": (8041.595658, 5418.262629, 3235.780781), "decimals": (6, 6, 6), "measured_sigma_mhz": (4.4e-05, 3.6e-05, 3.9e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.703074, 0.535835, -0.3116416, -0.0673942)},
    {"label": '[2-13C]', "subs": {1: M_C13},
         "abc_mhz": (7852.77554, 5418.40562, 3204.80655), "decimals": (5, 5, 6), "measured_sigma_mhz": (0.00031, 0.0001, 8.7e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.699151, 0.50644, -0.313237, -0.068193)},
    {"label": '[3-13C]', "subs": {2: M_C13},
         "abc_mhz": (7981.28038, 5319.305933, 3190.618898), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00034, 9.8e-05, 8.8e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.675952, 0.55134, -0.298436, -0.0648635)},
    {"label": '[34S]', "subs": {0: M_S34},
         "abc_mhz": (8041.71507, 5274.184579, 3183.843198), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.0002, 6.9e-05, 6.7e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.680947, 0.51005, -0.297537, -0.0629687)},
    {"label": '[33S]', "subs": {0: M_S33},
         "abc_mhz": (8041.6515, 5344.29761, 3209.25834), "decimals": (4, 5, 5), "measured_sigma_mhz": (0.002, 0.00059, 0.00013),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.69162, 0.52162, -0.304257, -0.065028)},
    {"label": '[2-2H]', "subs": {5: M_D},
         "abc_mhz": (7437.19729, 5413.721571, 3131.815069), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.0002, 8.4e-05, 8.3e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.681599, 0.44481, -0.313542, -0.0697451)},
    {"label": '[3-2H]', "subs": {6: M_D},
         "abc_mhz": (7855.99758, 5138.229854, 3105.222733), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00011, 6.3e-05, 7.2e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.606648, 0.56738, -0.265109, -0.0582064)},
    {"label": '[2,3-2H]', "subs": {5: M_D, 6: M_D},
         "abc_mhz": (7277.0013, 5137.627501, 3010.350104), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00027, 9.7e-05, 8.7e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.592435, 0.45748, -0.268999, -0.06022)},
    {"label": '[2,4-2H]', "subs": {5: M_D, 7: M_D},
         "abc_mhz": (7288.38317, 5124.657587, 3007.831223), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00021, 7.7e-05, 8.3e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.578844, 0.538407, -0.262829, -0.0607188)},
    {"label": '[2,5-2H]', "subs": {5: M_D, 8: M_D},
         "abc_mhz": (6903.42658, 5410.698544, 3032.130273), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00012, 6.8e-05, 7e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.666346, 0.32619, -0.316194, -0.0709025)},
    {"label": '[3,4-2H]', "subs": {6: M_D, 7: M_D},
         "abc_mhz": (7616.557, 4914.6794, 2986.0546), "decimals": (3, 4, 5), "measured_sigma_mhz": (0.016, 0.0078, 0.00018),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.53795, 0.531, -0.23302, -0.050945)},
    {"label": '[2,3,4-2H]', "subs": {5: M_D, 6: M_D, 7: M_D},
         "abc_mhz": (7072.27605, 4909.46919, 2896.80116), "decimals": (5, 5, 6), "measured_sigma_mhz": (0.00033, 0.0001, 9.4e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.520247, 0.46951, -0.234458, -0.0532931)},
    {"label": '[2,3,5-2H]', "subs": {5: M_D, 6: M_D, 8: M_D},
         "abc_mhz": (6775.84957, 5124.449058, 2916.73161), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.00012, 6.4e-05, 6.6e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.567721, 0.41849, -0.26663, -0.0621192)},
    {"label": '[2,3,4,5-2H]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D},
         "abc_mhz": (6587.73179, 4905.808107, 2810.897843), "decimals": (5, 6, 6), "measured_sigma_mhz": (0.0002, 8.5e-05, 7.7e-05),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.507404, 0.37771, -0.236303, -0.0545074)},
    {"label": '[2,5-2H, 34S]', "subs": {5: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6903.535, 5268.34308, 2986.9067), "decimals": (4, 5, 5), "measured_sigma_mhz": (0.0011, 0.00037, 0.0003),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.64538, 0.31768, -0.302494, -0.066729)},
    {"label": '[2,3,5-2H, 34S]', "subs": {5: M_D, 6: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6773.39956, 4992.2539, 2872.96911), "decimals": (5, 5, 5), "measured_sigma_mhz": (0.00092, 0.00032, 0.00023),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.55289, 0.3762, -0.255757, -0.057659)},
    {"label": '[2,3,4,5-2H, 34S]', "subs": {5: M_D, 6: M_D, 7: M_D, 8: M_D, 0: M_S34},
         "abc_mhz": (6588.36, 4777.72, 2768.4921), "decimals": (2, 2, 4), "measured_sigma_mhz": (0.32, 0.18, 0.0018),
         "used_in_re_SE_2021": False, "quartic_DJ_DJK_d1_d2_khz": (0.487, 0.348, -0.2225, -0.0503)},
    {"label": '[2,5-2H, 2-13C]', "subs": {5: M_D, 8: M_D, 1: M_C13},
         "abc_mhz": (6763.947, 5410.781, 3004.92939), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.025, 0.017, 0.00026),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.66744, 0.271, -0.31845, -0.07065)},
    {"label": '[2,5-2H, 3-13C]', "subs": {5: M_D, 8: M_D, 2: M_C13},
         "abc_mhz": (6860.081, 5310.447, 2992.15823), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.023, 0.015, 0.00026),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.64197, 0.329, -0.30329, -0.06776)},
    {"label": '[2-2H, 34S]', "subs": {5: M_D, 0: M_S34},
         "abc_mhz": (7436.863, 5270.7, 3083.34137), "decimals": (3, 4, 5), "measured_sigma_mhz": (0.016, 0.0092, 0.00019),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.66389, 0.404, -0.30128, -0.0652)},
    {"label": '[3-2H, 2-13C]', "subs": {6: M_D, 1: M_C13},
         "abc_mhz": (7681.747, 5137.725, 3077.43102), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.05, 0.024, 0.00028),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.60549, 0.52, -0.26733, -0.05859)},
    {"label": '[3-2H, 3-13C]', "subs": {6: M_D, 2: M_C13},
         "abc_mhz": (7816.981, 5045.949, 3065.2874), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.061, 0.027, 0.00028),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.58584, 0.563, -0.25428, -0.05559)},
    {"label": '[3-2H, 4-13C]', "subs": {6: M_D, 3: M_C13},
         "abc_mhz": (7775.956, 5060.889, 3064.44094), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.056, 0.025, 0.00027),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.58884, 0.552, -0.25668, -0.05613)},
    {"label": '[3-2H, 5-13C]', "subs": {6: M_D, 4: M_C13},
         "abc_mhz": (7671.947, 5138.103, 3075.99325), "decimals": (3, 3, 5), "measured_sigma_mhz": (0.041, 0.019, 0.00024),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.60206, 0.534, -0.26554, -0.05844)},
    {"label": '[3-2H, 34S]', "subs": {6: M_D, 0: M_S34},
         "abc_mhz": (7854.54126, 5003.22949, 3055.16594), "decimals": (5, 5, 5), "measured_sigma_mhz": (0.00065, 0.00016, 0.00014),
         "used_in_re_SE_2021": True, "quartic_DJ_DJK_d1_d2_khz": (0.588513, 0.53756, -0.253387, -0.054331)},
    {"label": '[3-2H, 33S]', "subs": {6: M_D, 0: M_S33},
         "abc_mhz": (7854.84, 5069.12, 3079.66292), "decimals": (2, 2, 5), "measured_sigma_mhz": (0.47, 0.2, 0.00044),
         "used_in_re_SE_2021": False, "quartic_DJ_DJK_d1_d2_khz": (0.5958, 0.534, -0.2571, -0.05493)},
]

#: 2025 Table I, normal isotopologue, "S reduction, Ir representation / Experimental": A0, B0, C0
#: in MHz with the printed parenthesised uncertainty (level NOT stated); quartic constants in kHz.
#: Not used in SPECIES (whose parent row is the Table II B0'' average); kept for reference and
#: for the check that Eqs. (4)-(6) take these to the Table II values.
PARENT_2025_TABLE_I_S = {
    "abc_mhz": (8041.595574, 5418.262567, 3235.780744), "decimals": (6, 6, 6),
    "measured_sigma_mhz": (0.000030, 0.000017, 0.000016),
    "DJ_DJK_d1_d2_khz": (0.7030613, 0.535798, -0.3116357, -0.06739408),
    "inertial_defect_uA2": 0.065692,
}

# ── Draft entry, in the shape of dev/monofluoro_references.py ─────────────────

ENTRY = {
    "key": "thiophene",
    "name": "Thiophene",
    "formula": "c-C4H4S",
    "elems": ELEMS,
    "geometry": GEOMETRY_RE,
    "masses": np.array(PARENT_MASSES),
    # Isotopologue(**kw) for kw in species; the parent is first. The 2021 S-reduced
    # constants with printed uncertainties, the 24 species the 2021 structure used (D-158);
    # the 2025 determinable constants, which print none, are SPECIES above.
    "species": [{"label": s["label"], "subs": dict(s["subs"]), "abc_mhz": s["abc_mhz"],
              "decimals": s["decimals"], "measured_sigma_mhz": s["measured_sigma_mhz"]}
             for s in SPECIES_2021_S if s["used_in_re_SE_2021"]],
    "structure_source": THIOPHENE_RE_SE_SOURCE,
    "structure_type": "r_e",
    "geometry_re_source": THIOPHENE_RE_SE_SOURCE,
    # Largest bond uncertainty as the source quotes it (2σ): 0.3 mA on RC3-C4, a dependent
    # parameter; the largest INDEPENDENT bond figure is 0.2 mA (RC2-C3). Halved to 1σ:
    # 0.15 / 0.1 mA. All are inside RE_REFERENCE_MAX_UNCERTAINTY_MA = 0.5 -> "ok" either way.
    "geometry_re_uncertainty_ma": 0.3,
    "constants_source": ("Orr et al., J. Chem. Phys. 154, 244310 (2021), Table II: A0, B0, C0, S "
                      "reduction, I^r, the 24 isotopologues used in its structure (two shaded "
                      "columns left out); uncertainty level not stated"),
    "bonds": {
        "S1-C2": [(0, 1), (0, 4)],
        "C2-C3": [(1, 2), (4, 3)],
        "C3-C4": [(2, 3)],
        "C2-H": [(1, 5), (4, 8)],
        "C3-H": [(2, 6), (3, 7)],
    },
    "angles": {
        "C2-S1-C5": [(1, 0, 4)],
        "S1-C2-C3": [(0, 1, 2), (0, 4, 3)],
        "C2-C3-C4": [(1, 2, 3), (4, 3, 2)],
        "S1-C2-H": [(0, 1, 5), (0, 4, 8)],
        "C3-C2-H": [(2, 1, 5), (3, 4, 8)],
        "C2-C3-H": [(1, 2, 6), (4, 3, 7)],
        "C4-C3-H": [(3, 2, 6), (2, 3, 7)],
    },
    "alt_geometry_re": GEOMETRY_RE_2021,
    "alt_geometry_re_source": THIOPHENE_RE_SE_2021_SOURCE,
    "computed_geometry_re_bte": GEOMETRY_BTE,
}
