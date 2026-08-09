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

Formyl fluoride (HCOF) was considered and rejected on that test: its CCCBDB
geometry and constants disagree by 6.2%, far beyond any zero-point effect.

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
            model = SIGMA_REL[k] * abs(v)
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
