"""Published structures and MEASURED rotational constants for monofluorinated molecules.

Every rotational constant in this file is a measured literature value transcribed
from NBS Monograph 70, *Microwave Spectral Tables* (National Bureau of Standards,
1968-69) -- the standard compilation of pre-1968 microwave work:

  * Vol. IV, "Polyatomic Molecules Without Internal Rotation", p. 183, entry 780
    (fluoroethene / vinyl fluoride);
  * Vol. III, "Polyatomic Molecules With Internal Rotation", p. 119, entry 390
    (acetyl fluoride) and p. 158, entry 550 (fluoroethane / ethyl fluoride).

Nothing here is back-calculated from a geometry. Where the compilation reports no
value for a constant -- A is frequently undetermined for a near-prolate top
measured only through low-J transitions -- the constant is recorded as None and
simply is not used, rather than being filled in.

Each geometry is a published substitution (r_s) structure, paired with these
constants and checked for mutual consistency: the geometry must reproduce every
measured constant to roughly 1%, the size of the r_s-versus-r_0 difference.
`scripts/check_monofluoro_references.py` runs that check on every species.

Formyl fluoride was initially rejected on that test, its CCCBDB geometry and
constants disagreeing by 6.2%. Against the measured constants transcribed here
the same structure agrees to 0.54%, so the fault lay with the CCCBDB constants
rather than the geometry, and it appears in the second set below.

Three sets are defined. MOLECULES (vinyl fluoride, acetyl fluoride,
fluoroethane) is what the hybrid's prior width was calibrated on. HELDOUT
(fluorobenzene) and MOLECULES_SET2 (formyl fluoride, fluoroacetylene,
chlorofluoromethane) took no part in that choice and exist to test whether the
calibration generalises.

Two transcription notes
-----------------------
* For vinyl fluoride, the compilation's cis/trans labels on the two
  doubly-deuterated species (ids 789 and 791) do not match their own constants.
  Deuterating the CHF hydrogen changes B by only ~1 MHz, so the B value alone
  identifies which CH2 hydrogen carries the second deuterium; on that test 789 is
  the trans species and 791 the cis one, each fitting the published geometry to
  <=0.34% while the labelled assignment is off by 6-22%. The assignments below
  follow the constants, not the labels.
* For fluoroethane, only species 551-556 are used. The compilation also lists
  seven multiply-deuterated species (557-563) whose configuration labels are
  ambiguous -- its own footnote records that several "configurations belong to
  the same isotopic molecular species" yet carry different constants -- so which
  hydrogen each deuterium occupies cannot be pinned down from the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
M_O16, M_O18 = 15.9949146196, 17.9991610
M_F = 18.99840322
M_CL35, M_CL37 = 34.96885268, 36.96590259
M_N14 = 14.0030740048

#: Relative uncertainty assigned to each of A, B, C.
#:
#: These are dominated by model error, not measurement error. The structures are
#: r_s and the constants are ground-state (r_0), and that difference is worth a
#: few tenths of a percent -- far more than the 0.01 MHz to which B and C are
#: quoted. A carries twice the floor because it is both the least well determined
#: constant of a near-prolate top and the one most sensitive to zero-point
#: out-of-plane motion; the residuals in the consistency check show exactly that
#: pattern.
SIGMA_REL = (0.010, 0.005, 0.005)


def _quoted_step(value: float, decimals: int) -> float:
    """Half the last quoted digit, as a floor on the measurement uncertainty."""
    return 0.5 * 10.0 ** (-decimals)


@dataclass
class Isotopologue:
    """One measured species: which atoms were substituted, and what was observed."""

    label: str
    #: atom index -> substituted mass. Empty for the parent.
    subs: dict
    #: measured (A, B, C) in MHz; None for a constant the study did not determine.
    abc_mhz: tuple
    #: decimals quoted for each constant, used to floor the uncertainty.
    decimals: tuple = (2, 2, 2)
    #: Per-species relative uncertainty, overriding SIGMA_REL. Needed where the
    #: rigid-rotor model error is larger than the usual r_s-vs-r_0 gap.
    sigma_rel: tuple | None = None

    def masses(self, parent: np.ndarray) -> np.ndarray:
        m = parent.copy()
        for i, mass in self.subs.items():
            m[i] = mass
        return m

    @property
    def component_indices(self) -> list[int]:
        return [k for k, v in enumerate(self.abc_mhz) if v is not None]

    def observed(self) -> list[float]:
        return [v for v in self.abc_mhz if v is not None]

    def sigmas(self) -> list[float]:
        out = []
        for k in self.component_indices:
            v = float(self.abc_mhz[k])
            rel = self.sigma_rel or SIGMA_REL
            model = rel[k] * abs(v)
            meas = _quoted_step(v, self.decimals[k])
            out.append(float(np.hypot(model, meas)))
        return out


@dataclass
class ReferenceMolecule:
    """A published structure with the measured constants of its isotopologues."""

    key: str
    name: str
    formula: str
    elems: list[str]
    geometry: np.ndarray               # (N, 3) Angstrom, published r_s structure
    masses: np.ndarray                 # (N,) amu, parent
    species: list[Isotopologue]
    structure_source: str
    constants_source: str
    bonds: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.elems)

    @property
    def internal_dof(self) -> int:
        return 3 * self.n_atoms - 6

    @property
    def b0_mhz(self) -> np.ndarray:
        """Parent A, B, C. The parent is always the first species."""
        return np.array(self.species[0].abc_mhz, dtype=float)

    @property
    def n_observables(self) -> int:
        return sum(len(s.component_indices) for s in self.species)

    def internal_coordinates(self, coords) -> dict[str, float]:
        """Symmetry-averaged bond lengths (A) and angles (deg)."""
        x = np.asarray(coords, dtype=float)
        out: dict[str, float] = {}
        for name, pairs in self.bonds.items():
            out[name] = float(np.mean([np.linalg.norm(x[j] - x[i]) for i, j in pairs]))
        for name, triples in self.angles.items():
            vals = []
            for i, j, k in triples:
                u, v = x[i] - x[j], x[k] - x[j]
                cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                vals.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            out[name] = float(np.mean(vals))
        return out


# ── Vinyl fluoride, CH2=CHF ──────────────────────────────────────────────────
# Structure: Hayashi & Inagusa, J. Mol. Spectrosc. 138, 135 (1989), via CCCBDB.
# Constants: NBS Monograph 70 Vol. IV p.183, entry 780 (refs 804, 875, 948).
# Atom order: C1 (bears F and H4), C2 (bears H5, H6), F3, H4, H5, H6.
# H5 is cis to F across the double bond, H6 is trans (verified geometrically).

VINYL_FLUORIDE = ReferenceMolecule(
    key="vinyl_fluoride",
    name="Vinyl fluoride",
    formula="CH2=CHF",
    elems=["C", "C", "F", "H", "H", "H"],
    geometry=np.array([
        [0.0000,  0.4476, 0.0000],
        [1.1877, -0.1487, 0.0000],
        [-1.1356, -0.2769, 0.0000],
        [-0.2349,  1.5038, 0.0000],
        [1.2321, -1.2348, 0.0000],
        [2.0966,  0.4290, 0.0000],
    ]),
    masses=np.array([M_C12, M_C12, M_F, M_H, M_H, M_H]),
    species=[
        Isotopologue("781 parent", {}, (64582.7, 10636.79, 9118.19), (1, 2, 2)),
        Isotopologue("782 13C (CH2)", {1: M_C13}, (None, 10295.26, 8859.05),
                     (0, 2, 2)),
        Isotopologue("783 13C (CHF)", {0: M_C13}, (None, 10635.02, 9082.78),
                     (0, 2, 2)),
        Isotopologue("784 D (CHF)", {3: M_D}, (48960.0, 10635.60, 8753.27),
                     (-1, 2, 2)),
        Isotopologue("787 D cis", {4: M_D}, (53400.0, 10278.20, 8610.48),
                     (-2, 2, 2)),
        Isotopologue("788 D trans", {5: M_D}, (62440.0, 9668.14, 8384.03),
                     (-1, 2, 2)),
        # Labelled c-/t- in the compilation; assigned from the constants instead.
        Isotopologue("789 D trans + D (CHF)", {5: M_D, 3: M_D},
                     (49250.0, 9667.07, 8077.02), (-1, 2, 2)),
        Isotopologue("791 D cis + D (CHF)", {4: M_D, 3: M_D},
                     (42700.0, 10274.57, 8272.36), (-2, 2, 2)),
    ],
    structure_source="Hayashi & Inagusa, J. Mol. Spectrosc. 138, 135 (1989); CCCBDB",
    constants_source="NBS Monograph 70 Vol. IV p.183 (entry 780)",
    bonds={
        "C1=C2":  [(0, 1)],
        "C1-F":   [(0, 2)],
        "C1-H4":  [(0, 3)],
        "C2-H5":  [(1, 4)],
        "C2-H6":  [(1, 5)],
    },
    angles={
        "F-C1=C2":  [(2, 0, 1)],
        "H4-C1=C2": [(3, 0, 1)],
        "F-C1-H4":  [(2, 0, 3)],
        "C1=C2-H5": [(0, 1, 4)],
        "C1=C2-H6": [(0, 1, 5)],
        "H5-C2-H6": [(4, 1, 5)],
    },
)


# ── Acetyl fluoride, CH3COF ──────────────────────────────────────────────────
# Structure: Pierce & Krisher, J. Chem. Phys. 31, 875 (1959), via CCCBDB.
# Constants: NBS Monograph 70 Vol. III p.119, entry 390 (ref 263).
# Atom order: C1 (carbonyl), C2 (methyl), O3, F4, H5 (in plane), H6/H7 (pair).
# The compilation writes the methyl carbon first, so C13H3C12O16F19 is 13C on the
# methyl (index 1) and C12H3C13O16F19 is 13C on the carbonyl (index 0); the
# constants confirm it, since the carbonyl carbon sits near the centre of mass
# and substituting it leaves B essentially unchanged.

ACETYL_FLUORIDE = ReferenceMolecule(
    key="acetyl_fluoride",
    name="Acetyl fluoride",
    formula="CH3COF",
    elems=["C", "C", "O", "F", "H", "H", "H"],
    geometry=np.array([
        [0.0000,  0.1904,  0.0000],
        [0.9884, -0.9419,  0.0000],
        [0.2158,  1.3515,  0.0000],
        [-1.2600, -0.2887,  0.0000],
        [2.0023, -0.5584,  0.0000],
        [0.8401, -1.5729,  0.8688],
        [0.8401, -1.5729, -0.8688],
    ]),
    masses=np.array([M_C12, M_C12, M_O16, M_F, M_H, M_H, M_H]),
    species=[
        Isotopologue("391 parent", {}, (11039.28, 9685.65, 5322.05)),
        Isotopologue("392 CD3", {4: M_D, 5: M_D, 6: M_D},
                     (10319.46, 7986.06, 4763.02)),
        Isotopologue("393 13C methyl", {1: M_C13}, (11038.83, 9309.72, 5206.44)),
        Isotopologue("394 18O", {2: M_O18}, (10522.74, 9470.63, 5136.20)),
        Isotopologue("395 D2 out of plane", {5: M_D, 6: M_D},
                     (10586.16, 8420.42, 4971.86)),
        Isotopologue("396 D in plane + D out", {4: M_D, 5: M_D},
                     (10591.19, 8467.89, 4908.64)),
        Isotopologue("397 D in plane", {4: M_D}, (10919.20, 8971.39, 5075.09)),
        Isotopologue("398 D out of plane", {5: M_D}, (10805.96, 9016.47, 5134.54)),
        Isotopologue("399 13C carbonyl", {0: M_C13}, (11034.56, 9686.60, 5321.32)),
        Isotopologue("401 CD3 + 13C carbonyl",
                     {0: M_C13, 4: M_D, 5: M_D, 6: M_D},
                     (10315.05, 7985.33, 4761.80)),
    ],
    structure_source="Pierce & Krisher, J. Chem. Phys. 31, 875 (1959); CCCBDB",
    constants_source="NBS Monograph 70 Vol. III p.119 (entry 390)",
    bonds={
        "C1-C2":  [(0, 1)],
        "C1=O":   [(0, 2)],
        "C1-F":   [(0, 3)],
        "C2-H5":  [(1, 4)],
        "C2-H6":  [(1, 5), (1, 6)],
    },
    angles={
        "O=C1-F":   [(2, 0, 3)],
        "O=C1-C2":  [(2, 0, 1)],
        "F-C1-C2":  [(3, 0, 1)],
        "C1-C2-H5": [(0, 1, 4)],
        "C1-C2-H6": [(0, 1, 5), (0, 1, 6)],
    },
)


# ── Fluoroethane, CH3CH2F ────────────────────────────────────────────────────
# Structure: Kraitchman & Dailey (1955), via CCCBDB.
# Constants: NBS Monograph 70 Vol. III p.158, entry 550 (refs 108, 274).
# Atom order: C1 (bears F), C2 (methyl), F3, H4/H5 on C1, H6 (anti, in the
# symmetry plane) and H7/H8 on C2.
# As above the compilation writes the methyl carbon first, so C12H3C13H2F19 is
# 13C on the CH2F carbon (index 0); its B is unchanged from the parent because
# that carbon lies almost on the centre of mass.

FLUOROETHANE = ReferenceMolecule(
    key="fluoroethane",
    name="Fluoroethane",
    formula="CH3CH2F",
    elems=["C", "C", "F", "H", "H", "H", "H", "H"],
    geometry=np.array([
        [0.0000,  0.5577,  0.0000],
        [1.1298, -0.4366,  0.0000],
        [-1.2233, -0.1190,  0.0000],
        [-0.0067,  1.1950,  0.8904],
        [-0.0067,  1.1950, -0.8904],
        [2.0845,  0.0915,  0.0000],
        [1.0799, -1.0684,  0.8868],
        [1.0799, -1.0684, -0.8868],
    ]),
    masses=np.array([M_C12, M_C12, M_F] + [M_H] * 5),
    species=[
        Isotopologue("551 parent", {}, (36070.30, 9364.54, 8199.74)),
        Isotopologue("552 D on CH2F", {3: M_D}, (31140.2, 9252.63, 7995.51),
                     (1, 2, 2)),
        Isotopologue("553 D methyl gauche", {6: M_D}, (32601.8, 8953.99, 7866.31),
                     (1, 2, 2)),
        Isotopologue("554 D methyl anti", {5: M_D}, (35693.3, 8623.86, 7611.59),
                     (1, 2, 2)),
        Isotopologue("555 13C on CH2F", {0: M_C13}, (35250.1, 9365.66, 8157.15),
                     (1, 2, 2)),
        Isotopologue("556 13C methyl", {1: M_C13}, (35915.4, 9089.05, 7980.16),
                     (1, 2, 2)),
    ],
    structure_source="Kraitchman & Dailey (1955); CCCBDB",
    constants_source="NBS Monograph 70 Vol. III p.158 (entry 550)",
    bonds={
        "C1-C2":  [(0, 1)],
        "C1-F":   [(0, 2)],
        "C1-H4":  [(0, 3), (0, 4)],
        "C2-H6":  [(1, 5)],
        "C2-H7":  [(1, 6), (1, 7)],
    },
    angles={
        "F-C1-C2":  [(2, 0, 1)],
        "F-C1-H4":  [(2, 0, 3), (2, 0, 4)],
        "C2-C1-H4": [(1, 0, 3), (1, 0, 4)],
        "H4-C1-H5": [(3, 0, 4)],
        "C1-C2-H6": [(0, 1, 5)],
        "C1-C2-H7": [(0, 1, 6), (0, 1, 7)],
    },
)


# ── Fluorobenzene, C6H5F — HELD OUT ──────────────────────────────────────────
# Structure: Bak, Christensen, Hansen-Nygaard & Tannenbaum, J. Chem. Phys. 26,
# 134 (1957), Cartesians via NIST CCCBDB casno 462066.
# Constants: NBS Monograph 70 Vol. IV p.281, entry 1280 (ref 737).
#
# This molecule is deliberately excluded from MOLECULES. The hybrid's prior
# width was calibrated on the three above, so testing it here -- on a molecule
# that took no part in that choice -- is the only way to tell whether the
# setting generalises or was fitted to the calibration set.
#
# It is also a harder case than any of them: four measured species give 12
# constants against 30 internal degrees of freedom, and the substitutions are
# all deuterium, so no carbon is ever located directly.
#
# Atom order: F, C1 ipso, C2/C6 ortho, C3/C5 meta, C4 para, H on 2,6,3,5,4.

FLUOROBENZENE = ReferenceMolecule(
    key="fluorobenzene",
    name="Fluorobenzene",
    formula="C6H5F",
    elems=["F", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"],
    geometry=np.array([
        [0.0000,  0.0000, -2.2030],
        [0.0000,  0.0000, -0.8490],
        [0.0000, -1.2170, -0.1930],
        [0.0000,  1.2170, -0.1930],
        [0.0000, -1.2080,  1.2020],
        [0.0000,  1.2080,  1.2020],
        [0.0000,  0.0000,  1.9030],
        [0.0000, -2.1370, -0.7610],
        [0.0000,  2.1370, -0.7610],
        [0.0000, -2.1470,  1.7430],
        [0.0000,  2.1470,  1.7430],
        [0.0000,  0.0000,  2.9830],
    ]),
    masses=np.array([M_F] + [M_C12] * 6 + [M_H] * 5),
    species=[
        Isotopologue("1281 parent", {}, (5663.54, 2570.64, 1767.94)),
        # 3d: one meta hydrogen, which breaks C2v -- the compilation lists it
        # as Cs, confirming a single substitution off the symmetry axis.
        Isotopologue("1282 3d (meta)", {9: M_D}, (5394.27, 2529.99, 1722.07)),
        # 4d: the para hydrogen lies on the symmetry axis, so C2v survives.
        Isotopologue("1283 4d (para)", {11: M_D}, (5663.64, 2459.72, 1714.75)),
        Isotopologue("1284 2,4,6-d3", {7: M_D, 8: M_D, 11: M_D},
                     (5134.71, 2445.03, 1656.19)),
    ],
    structure_source=("Bak, Christensen, Hansen-Nygaard & Tannenbaum, "
                      "J. Chem. Phys. 26, 134 (1957); CCCBDB"),
    constants_source="NBS Monograph 70 Vol. IV p.281 (entry 1280)",
    bonds={
        "C1-F":        [(1, 0)],
        "C1-C2":       [(1, 2), (1, 3)],
        "C2-C3":       [(2, 4), (3, 5)],
        "C3-C4":       [(4, 6), (5, 6)],
        "C2-H2":       [(2, 7), (3, 8)],
        "C3-H3":       [(4, 9), (5, 10)],
        "C4-H4":       [(6, 11)],
    },
    angles={
        "F-C1-C2":     [(0, 1, 2), (0, 1, 3)],
        "C2-C1-C6":    [(2, 1, 3)],
        "C1-C2-C3":    [(1, 2, 4), (1, 3, 5)],
        "C2-C3-C4":    [(2, 4, 6), (3, 5, 6)],
        "C3-C4-C5":    [(4, 6, 5)],
        "C1-C2-H2":    [(1, 2, 7), (1, 3, 8)],
        "C2-C3-H3":    [(2, 4, 9), (3, 5, 10)],
        "C3-C4-H4":    [(4, 6, 11), (5, 6, 11)],
    },
)


#: Molecules the hybrid's prior width was calibrated on.
MOLECULES = [VINYL_FLUORIDE, ACETYL_FLUORIDE, FLUOROETHANE]

#: Molecules held out of that calibration, for validation only.
HELDOUT = [FLUOROBENZENE]


# ═══════════════════════════════════════════════════════════════════════════
# Second set: three further monofluorinated molecules, chosen after the first
# three to test the calibrated hybrid on data it played no part in tuning.
#
# All constants are measured values from NBS Monograph 70. The set was picked
# to span the regimes that matter rather than to be uniform:
#
#   formyl fluoride     over-determined  (12 constants, 6 parameters)
#   fluoroacetylene     data nearly exact (6 B values, 3 bond lengths)
#   chlorofluoromethane undersaturated   (6 constants, 9 parameters)
#
# Every structure below reproduces every one of its measured constants to
# within the r_s-versus-r_0 range, checked by
# scripts/check_monofluoro_references.py.
# ═══════════════════════════════════════════════════════════════════════════

def _fcho_geometry(rco=1.181, rcf=1.338, rch=1.095, a_ocf=122.8, a_och=127.3):
    C = np.zeros(3)
    O = np.array([0.0, rco, 0.0])
    t, u = np.radians(a_ocf), np.radians(a_och)
    F = rcf * np.array([np.sin(t), np.cos(t), 0.0])
    H = rch * np.array([-np.sin(u), np.cos(u), 0.0])
    return np.array([C, O, F, H])


# ── Formyl fluoride, FCHO ────────────────────────────────────────────────────
# Structure: Le Blanc, Laurie & Gwinn, J. Chem. Phys. 33, 598 (1960).
# Constants: NBS Monograph 70 Vol. IV p.62, entry 270.
# Rejected from the first set on a CCCBDB geometry/constants mismatch of 6.2%;
# against the measured constants here the same structure agrees to 0.54%, so it
# was the CCCBDB constants that were wrong, not the geometry.
FORMYL_FLUORIDE = ReferenceMolecule(
    key="formyl_fluoride",
    name="Formyl fluoride",
    formula="FCHO",
    elems=["C", "O", "F", "H"],
    geometry=_fcho_geometry(),
    masses=np.array([M_C12, M_O16, M_F, M_H]),
    species=[
        Isotopologue("271 parent", {}, (91153.57, 11760.37, 10396.79)),
        Isotopologue("272 D", {3: M_D}, (65096.59, 11761.74, 9941.71)),
        Isotopologue("273 13C", {0: M_C13}, (88505.1, 11755.2, 10357.3), (1, 1, 1)),
        Isotopologue("274 18O", {1: M_O18}, (89769.5, 11102.9, 9863.4), (1, 1, 1)),
    ],
    structure_source="Le Blanc, Laurie & Gwinn, J. Chem. Phys. 33, 598 (1960)",
    constants_source="NBS Monograph 70 Vol. IV p.62 (entry 270)",
    bonds={"C=O": [(0, 1)], "C-F": [(0, 2)], "C-H": [(0, 3)]},
    angles={"O=C-F": [(1, 0, 2)], "O=C-H": [(1, 0, 3)], "F-C-H": [(2, 0, 3)]},
)


# ── Fluoroacetylene, HC≡CF ───────────────────────────────────────────────────
# Structure: Tyler & Sheridan, Trans. Faraday Soc. 59, 2661 (1963).
# Constants: NBS Monograph 70 Vol. IV p.131, entry 620 -- B only, as a linear
# molecule has no A and B equals C.
# The published r_s structure was derived from these same six constants, so it
# reproduces them to 0.06%. Spectroscopy alone should therefore do very well
# here; the question this molecule asks is whether the hybrid stays out of the
# way when the data is already excellent.
FLUOROACETYLENE = ReferenceMolecule(
    key="fluoroacetylene",
    name="Fluoroacetylene",
    formula="HC2F",
    elems=["H", "C", "C", "F"],
    geometry=np.column_stack([
        np.zeros(4), np.zeros(4), np.cumsum([0.0, 1.0553, 1.1980, 1.2790])]),
    masses=np.array([M_H, M_C12, M_C12, M_F]),
    species=[
        Isotopologue("621 parent", {}, (None, 9706.22, None)),
        Isotopologue("622 13C (CF)", {2: M_C13}, (None, 9700.71, None)),
        Isotopologue("623 13C (CH)", {1: M_C13}, (None, 9373.95, None)),
        Isotopologue("624 D", {0: M_D}, (None, 8736.09, None)),
        Isotopologue("625 D + 13C (CF)", {0: M_D, 2: M_C13}, (None, 8733.94, None)),
        Isotopologue("626 D + 13C (CH)", {0: M_D, 1: M_C13}, (None, 8486.33, None)),
    ],
    structure_source="Tyler & Sheridan, Trans. Faraday Soc. 59, 2661 (1963)",
    constants_source="NBS Monograph 70 Vol. IV p.131 (entry 620)",
    bonds={"C-H": [(0, 1)], "C#C": [(1, 2)], "C-F": [(2, 3)]},
    # Linear: the single angle is 180 deg and carries no structural information,
    # but errors() needs a non-empty angle set.
    angles={"H-C#C": [(0, 1, 2)]},
)


def _ch2clf_geometry(rF=1.359, rCl=1.759, rH=1.087,
                     aFCl=110.2, aClH=108.0, aFH=109.0):
    aFCl, aClH, aFH = np.radians([aFCl, aClH, aFH])
    C = np.zeros(3)
    Cl = np.array([0.0, 0.0, rCl])
    f = np.array([np.sin(aFCl), 0.0, np.cos(aFCl)])
    uz = np.cos(aClH)
    ux = (np.cos(aFH) - f[2] * uz) / f[0]
    uy = np.sqrt(max(0.0, 1 - ux ** 2 - uz ** 2))
    return np.array([C, Cl, rF * f,
                     rH * np.array([ux, uy, uz]), rH * np.array([ux, -uy, uz])])


# ── Chlorofluoromethane, CH2ClF ──────────────────────────────────────────────
# Structure: Müller, J. Am. Chem. Soc. 75, 860 (1953) / Landolt-Bornstein.
# Constants: NBS Monograph 70 Vol. IV p.77, entry 330.
# The two species come from chlorine's own natural isotopes, so no synthesis
# was needed -- and equally, no other substitution is available. Six constants
# against nine parameters makes this the undersaturated case of the set.
CHLOROFLUOROMETHANE = ReferenceMolecule(
    key="chlorofluoromethane",
    name="Chlorofluoromethane",
    formula="CH2ClF",
    elems=["C", "Cl", "F", "H", "H"],
    geometry=_ch2clf_geometry(),
    masses=np.array([M_C12, M_CL35, M_F, M_H, M_H]),
    species=[
        Isotopologue("331 Cl-35", {}, (41810.1, 5715.7, 5194.6), (1, 1, 1)),
        Isotopologue("332 Cl-37", {1: M_CL37}, (41738.2, 5580.5, 5081.6), (1, 1, 1)),
    ],
    structure_source="Muller, J. Am. Chem. Soc. 75, 860 (1953)",
    constants_source="NBS Monograph 70 Vol. IV p.77 (entry 330)",
    bonds={"C-Cl": [(0, 1)], "C-F": [(0, 2)], "C-H": [(0, 3), (0, 4)]},
    angles={"F-C-Cl": [(2, 0, 1)], "Cl-C-H": [(1, 0, 3), (1, 0, 4)],
            "F-C-H": [(2, 0, 3), (2, 0, 4)], "H-C-H": [(3, 0, 4)]},
)


#: The second, independent set.
MOLECULES_SET2 = [FORMYL_FLUORIDE, FLUOROACETYLENE, CHLOROFLUOROMETHANE]


# ── Water, H2O — a deliberate stress test ────────────────────────────────────
# Structure: r_e = 0.9578 A, 104.48 deg, the accepted equilibrium geometry.
# Constants: NIST CCCBDB experimental values for H2O and D2O.
#
# Water is the hardest case here and it is included to show where the approach
# breaks down rather than where it works. Every fluorinated molecule above
# reproduces its own measured constants to 0.06-1.4%, with the residuals all
# one-signed -- the r_s-versus-r_0 offset. Water manages only 2.4%, and its
# residuals change sign across A, B and C (-1.8%, +0.5%, +2.5%).
#
# That is not a bad reference structure, it is the rigid-rotor model failing.
# Water is light and floppy, so zero-point averaging is both large and strongly
# anisotropic, and no single rigid geometry can reproduce all three constants.
# CCCBDB's own effective geometries for H2O and D2O differ (0.958 A / 104.478
# deg against 0.956 A / 105.200 deg), which is the same effect seen directly.
#
# sigma is therefore set to 2.5% rather than the usual 0.5-1%, to match the
# model error actually present. Weighting these constants as though they were
# good to 0.5% would ask the fit to reproduce mutually inconsistent numbers.
WATER = ReferenceMolecule(
    key="water",
    name="Water",
    formula="H2O",
    elems=["O", "H", "H"],
    geometry=np.array([
        [0.0000000,  0.0000000, 0.0],
        [0.757430,  0.5865400, 0.0],
        [-0.757430,  0.5865400, 0.0],
    ]),
    masses=np.array([M_O16, M_H, M_H]),
    species=[
        Isotopologue("H2-16O", {},
                     (835712.5, 435059.0, 278357.3), (1, 1, 1), (0.025,)*3),
        Isotopologue("D2-16O", {1: M_D, 2: M_D},
                     (462278.3, 218038.5, 145258.5), (1, 1, 1), (0.025,)*3),
    ],
    structure_source="Accepted equilibrium structure r_e = 0.9578 A, 104.48 deg",
    constants_source="NIST CCCBDB experimental rotational constants (H2O, D2O)",
    bonds={"O-H": [(0, 1), (0, 2)]},
    angles={"H-O-H": [(1, 0, 2)]},
)

#: Water on its own. Not monofluorinated -- included as a stress test.
WATER_SET = [WATER]


def _methyl(base, axis, r, ang_deg, phi0_deg):
    """Three hydrogens on `base`, splayed about `axis` at `ang_deg`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    p1 = np.array([0.0, 0.0, 1.0])
    p1 = p1 - (p1 @ a) * a
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(a, p1)
    t = np.radians(ang_deg)
    return [base + r * (np.cos(t) * a + np.sin(t) *
                        (np.cos(np.radians(phi0_deg + 120 * k)) * p1 +
                         np.sin(np.radians(phi0_deg + 120 * k)) * p2))
            for k in range(3)]


def _acetone_geometry(rCO=1.214, rCC=1.520, rCH=1.103,
                      aCCO=122.0, aHCC=110.5, phi0=30.0):
    C1 = np.zeros(3)
    O = np.array([0.0, rCO, 0.0])
    t = np.radians(aCCO)
    C2 = rCC * np.array([np.sin(t), np.cos(t), 0.0])
    C3 = rCC * np.array([-np.sin(t), np.cos(t), 0.0])
    return np.array([C1, O, C2, C3]
                    + _methyl(C2, C1 - C2, rCH, aHCC, phi0)
                    + _methyl(C3, C1 - C3, rCH, aHCC, -phi0))


# ── Acetone, (CH3)2CO ────────────────────────────────────────────────────────
# Structure: NIST CCCBDB, from Kuchitsu's compilation of free polyatomic
# structures (1998).
# Constants: NBS Monograph 70 Vol. III p.203, entry 750 (ref 271).
#
# Two measured species -- all-H and all-D -- give six constants against 24
# internal degrees of freedom, so this is the most undersaturated molecule in
# the whole set. Only two of ten atoms are ever substituted, and both methyls
# move together, so the carbonyl carbon and oxygen are never located directly.
#
# Two caveats worth stating. The reference is very likely a thermal-average
# (r_g) structure rather than r_s: its C-H of 1.103 A is long, and every
# residual against the measured constants comes out negative, meaning the
# reference is larger than the ground state -- the opposite sign to every r_s
# structure above. And acetone has two coupled methyl rotors, so the measured
# constants are torsionally averaged effective values; the compilation lists
# separate constants for the A and B torsional states, which differ from these
# by around 20 MHz in A.
#
# The methyl torsional phase turns out not to matter: rotating both methyls
# through 0, 30, 60 and 90 degrees changes the worst deviation by less than
# 0.01%, because the three hydrogens are symmetric about the rotor axis.
ACETONE = ReferenceMolecule(
    key="acetone",
    name="Acetone",
    formula="C3H6O",
    elems=["C", "O", "C", "C"] + ["H"] * 6,
    geometry=_acetone_geometry(),
    masses=np.array([M_C12, M_O16, M_C12, M_C12] + [M_H] * 6),
    species=[
        Isotopologue("751 (CH3)2CO", {}, (10165.20, 8515.27, 4910.15)),
        Isotopologue("752 (CD3)2CO", {i: M_D for i in range(4, 10)},
                     (8469.40, 6419.60, 4011.28)),
    ],
    structure_source="NIST CCCBDB, Kuchitsu compilation (1998)",
    constants_source="NBS Monograph 70 Vol. III p.203 (entry 750)",
    bonds={
        "C1=O":  [(0, 1)],
        "C1-C2": [(0, 2), (0, 3)],
        "C2-H":  [(2, 4), (2, 5), (2, 6), (3, 7), (3, 8), (3, 9)],
    },
    angles={
        "C2-C1-C3": [(2, 0, 3)],
        "O=C1-C2":  [(1, 0, 2), (1, 0, 3)],
        "C1-C2-H":  [(0, 2, 4), (0, 2, 5), (0, 2, 6),
                     (0, 3, 7), (0, 3, 8), (0, 3, 9)],
        "H-C2-H":   [(4, 2, 5), (4, 2, 6), (5, 2, 6),
                     (7, 3, 8), (7, 3, 9), (8, 3, 9)],
    },
)

#: Acetone on its own. Not monofluorinated -- a large, heavily undersaturated
#: test with two coupled internal rotors.
ACETONE_SET = [ACETONE]


# ═══════════════════════════════════════════════════════════════════════════
# Third set: three molecules chosen to probe different failure modes rather
# than different chemistry. None is fluorinated; all are small or symmetric so
# they run quickly.
#
#   ozone            no hydrogens at all, and a molecule single-reference
#                    quantum chemistry is known to describe badly
#   ethylene oxide   a strained three-membered ring
#   isocyanic acid   quasi-linear, where one measured constant is unusable
# ═══════════════════════════════════════════════════════════════════════════

def _ozone_geometry(r=1.2717, angle=116.78):
    a = np.radians(angle) / 2
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.sin(a), r * np.cos(a), 0.0],
                     [-r * np.sin(a), r * np.cos(a), 0.0]])


# ── Ozone, O3 ────────────────────────────────────────────────────────────────
# Structure: r(O-O) = 1.2717 A, 116.78 deg (standard r_s values).
# Constants: NBS Monograph 70 Vol. IV p.402, entry 1840.
#
# Fifteen measured constants for what is really two parameters, so the spectral
# side is as well determined as it ever gets here. What makes ozone interesting
# is the other side: its wavefunction has strong multireference character, and
# single-determinant methods describe it poorly. If the hybrid is going to earn
# its keep by rescuing bad theory, this is where it should show.
#
# It also has no hydrogens, which removes the failure mode that dominated every
# earlier molecule -- light atoms that barely shift the moments of inertia and
# so absorb all the fit's error.
#
# Atom order: central O, then the two terminal O.
OZONE = ReferenceMolecule(
    key="ozone",
    name="Ozone",
    formula="O3",
    elems=["O", "O", "O"],
    geometry=_ozone_geometry(),
    masses=np.array([M_O16] * 3),
    species=[
        Isotopologue("1841 all-16O", {}, (106536.1, 13349.12, 11834.45), (1, 2, 2)),
        Isotopologue("1842 18O terminal", {1: M_O18}, (104569.4, 12590.4, 11214.6),
                     (1, 1, 1)),
        Isotopologue("1843 18O central", {0: M_O18}, (98645.96, 13352.51, 11731.78),
                     (2, 2, 2)),
        Isotopologue("1844 18O central+terminal", {0: M_O18, 1: M_O18},
                     (96676.8, 12591.4, 11115.6), (1, 1, 1)),
        Isotopologue("1845 all-18O", {0: M_O18, 1: M_O18, 2: M_O18},
                     (94768.2, 11886.5, 10536.9), (1, 1, 1)),
    ],
    structure_source="r_s structure, r(O-O) = 1.2717 A, angle 116.78 deg",
    constants_source="NBS Monograph 70 Vol. IV p.402 (entry 1840)",
    bonds={"O-O": [(0, 1), (0, 2)]},
    angles={"O-O-O": [(1, 0, 2)]},
)


def _oxirane_geometry(cc=1.459, co=1.425, ch=1.084,
                      a_hcc=119.078, a_hco=114.704):
    C1 = np.array([-cc / 2, 0.0, 0.0])
    C2 = np.array([cc / 2, 0.0, 0.0])
    O = np.array([0.0, np.sqrt(co ** 2 - (cc / 2) ** 2), 0.0])
    out = [C1, C2, O]
    for C, other in ((C1, C2), (C2, C1)):
        a = (other - C) / np.linalg.norm(other - C)
        b = (O - C) / np.linalg.norm(O - C)
        ux = np.cos(np.radians(a_hcc)) * np.sign(a[0])
        uy = (np.cos(np.radians(a_hco)) - b[0] * ux) / b[1]
        uz = np.sqrt(max(0.0, 1 - ux ** 2 - uy ** 2))
        out += [C + ch * np.array([ux, uy, s * uz]) for s in (1, -1)]
    return np.array(out)


# ── Ethylene oxide (oxirane), C2H4O ──────────────────────────────────────────
# Structure: NIST CCCBDB, Kuchitsu compilation (1995).
# Constants: NBS Monograph 70 Vol. IV p.211, entry 840.
#
# A strained three-membered ring, where the bonding is unlike any of the open
# molecules above. Fifteen measured constants against fifteen degrees of
# freedom, so it is nominally exactly determined.
#
# The CH2 groups are tilted rather than symmetric about the ring bisector:
# H-C-C is 119.08 deg while H-C-O is 114.70 deg. Building them symmetrically
# instead costs a factor of two in the consistency check (2.3% against 1.5%),
# and swapping the two angles costs a factor of four (5.8%), so the tilt is
# resolved by the constants themselves and not merely asserted.
#
# Atom order: C1, C2, O, then the two hydrogens on C1 and the two on C2.
ETHYLENE_OXIDE = ReferenceMolecule(
    key="ethylene_oxide",
    name="Ethylene oxide",
    formula="C2H4O",
    elems=["C", "C", "O", "H", "H", "H", "H"],
    geometry=_oxirane_geometry(),
    masses=np.array([M_C12, M_C12, M_O16] + [M_H] * 4),
    species=[
        Isotopologue("841 parent", {}, (25483.7, 22120.9, 14098.0), (1, 1, 1)),
        Isotopologue("842 13C", {0: M_C13}, (25291.2, 21597.4, 13825.2), (1, 1, 1)),
        Isotopologue("843 d4", {3: M_D, 4: M_D, 5: M_D, 6: M_D},
                     (20399.0, 15457.0, 11544.0), (0, 0, 0)),
        Isotopologue("844 trans-d2", {3: M_D, 6: M_D},
                     (22945.1, 18198.6, 12585.5), (1, 1, 1)),
        Isotopologue("845 cis-d2", {3: M_D, 5: M_D},
                     (22700.3, 18318.5, 12650.3), (1, 1, 1)),
    ],
    structure_source="NIST CCCBDB, Kuchitsu compilation (1995)",
    constants_source="NBS Monograph 70 Vol. IV p.211 (entry 840)",
    bonds={
        "C-C": [(0, 1)],
        "C-O": [(0, 2), (1, 2)],
        "C-H": [(0, 3), (0, 4), (1, 5), (1, 6)],
    },
    angles={
        "C-O-C": [(0, 2, 1)],
        "O-C-C": [(2, 0, 1), (2, 1, 0)],
        "H-C-H": [(3, 0, 4), (5, 1, 6)],
        "H-C-C": [(3, 0, 1), (4, 0, 1), (5, 1, 0), (6, 1, 0)],
        "H-C-O": [(3, 0, 2), (4, 0, 2), (5, 1, 2), (6, 1, 2)],
    },
)


def _hnco_geometry(rNH=1.0026, rNC=1.2140, rCO=1.1662,
                   a_hnc=123.9, a_nco=172.6):
    N = np.zeros(3)
    C = np.array([rNC, 0.0, 0.0])
    t = np.radians(a_hnc)
    H = rNH * np.array([np.cos(t), np.sin(t), 0.0])
    b = np.radians(180.0 - a_nco)
    O = C + rCO * np.array([np.cos(b), -np.sin(b), 0.0])
    return np.array([N, H, C, O])


# ── Isocyanic acid, HNCO ─────────────────────────────────────────────────────
# Structure: r_s values, r(NH) 1.0026, r(NC) 1.2140, r(CO) 1.1662 A;
# HNC 123.9 deg, NCO 172.6 deg.
# Constants: NBS Monograph 70 Vol. IV p.69, entry 300.
#
# Quasi-linear: the NCO angle is only 7 degrees from straight, so the molecule
# executes a large-amplitude bend and its A constant is enormous (956 GHz) and
# effectively meaningless as a rigid-rotor quantity. The published structure
# reproduces the measured B and C to 0.26% but misses A by 11%, which is the
# large-amplitude motion showing rather than a bad structure.
#
# A is therefore excluded, leaving four constants for six degrees of freedom.
# That is the point of including it: real datasets contain constants that
# should not be fitted, and the machinery has to be told which.
#
# The direction of the NCO bend is fixed by the constants: bending the oxygen
# away from the hydrogen fits A to 11%, bending it toward gives 32%.
#
# Atom order: N, H, C, O.
ISOCYANIC_ACID = ReferenceMolecule(
    key="isocyanic_acid",
    name="Isocyanic acid",
    formula="HNCO",
    elems=["N", "H", "C", "O"],
    geometry=_hnco_geometry(),
    masses=np.array([M_N14, M_H, M_C12, M_O16]),
    species=[
        Isotopologue("301 HNCO", {}, (None, 11071.02, 10910.58)),
        Isotopologue("303 DNCO", {1: M_D}, (None, 10313.61, 10079.67)),
    ],
    structure_source="r_s structure (Jones, Shoolery, Shulman & Yost 1950)",
    constants_source="NBS Monograph 70 Vol. IV p.69 (entry 300)",
    bonds={"N-H": [(0, 1)], "N-C": [(0, 2)], "C-O": [(2, 3)]},
    angles={"H-N-C": [(1, 0, 2)], "N-C-O": [(0, 2, 3)]},
)

#: Third set: different failure modes rather than different chemistry.
MOLECULES_SET3 = [OZONE, ETHYLENE_OXIDE, ISOCYANIC_ACID]
