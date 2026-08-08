"""Published experimental structures for a set of monofluorinated molecules.

Each entry pairs a literature geometry with the rotational constants from the
same source, and every pair has been checked for mutual consistency: the
geometry must reproduce its own constants to within about 1%, which is the size
of the r_s-versus-r_0 difference. Anything worse means one of the two is wrong.

Formyl fluoride (HCOF) was considered and rejected on exactly that test: the
CCCBDB geometry and constants disagree by 6.2%, far beyond any zero-point
effect, so one of them cannot be right.

All data from NIST CCCBDB, with the primary references recorded per molecule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CM_TO_MHZ = 29979.2458

M_H, M_D = 1.00782503207, 2.01410177812
M_C12, M_C13 = 12.0, 13.00335483507
M_O16, M_O18 = 15.9949146196, 17.9991610
M_F = 18.99840322


@dataclass
class ReferenceMolecule:
    """A published structure with its own rotational constants."""

    key: str
    name: str
    formula: str
    elems: list[str]
    geometry: np.ndarray               # (N, 3) Angstrom, the published structure
    masses: np.ndarray                 # (N,) amu, parent isotopologue
    b0_mhz: np.ndarray                 # (3,) observed A, B, C
    source: str
    #: label -> (atom index, substituted mass). Symmetry-equivalent atoms are
    #: represented once, since substituting either gives the same constants.
    substitutions: dict
    bonds: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.elems)

    @property
    def internal_dof(self) -> int:
        return 3 * self.n_atoms - 6

    def isotopologue_masses(self) -> list[tuple[str, np.ndarray]]:
        out = [("parent", self.masses.copy())]
        for label, (index, mass) in self.substitutions.items():
            masses = self.masses.copy()
            masses[index] = mass
            out.append((label, masses))
        return out

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
# Hayashi & Inagusa, J. Mol. Spectrosc. 138, 135 (1989)
# Atom order: C1 (bears F and H4), C2 (bears H5, H6), F3, H4, H5, H6

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
    b0_mhz=np.array([2.15431, 0.35481, 0.30415]) * _CM_TO_MHZ,
    source="Hayashi & Inagusa, J. Mol. Spectrosc. 138, 135 (1989); NIST CCCBDB",
    substitutions={
        "1-13C": (0, M_C13), "2-13C": (1, M_C13),
        "4-D": (3, M_D), "5-D": (4, M_D), "6-D": (5, M_D),
    },
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
# Pierce & Krisher, J. Chem. Phys. 31, 875 (1959)
# Atom order: C1 (carbonyl), C2 (methyl), O3, F4, H5 (in plane), H6/H7 (pair)

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
    b0_mhz=np.array([0.36823, 0.32308, 0.17752]) * _CM_TO_MHZ,
    source="Pierce & Krisher, J. Chem. Phys. 31, 875 (1959); NIST CCCBDB",
    substitutions={
        "1-13C": (0, M_C13), "2-13C": (1, M_C13), "18O": (2, M_O18),
        "5-D (in plane)": (4, M_D), "6-D (out of plane)": (5, M_D),
    },
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
# Kraitchman & Dailey (1955); NIST CCCBDB
# Atom order: C1 (bears F), C2 (methyl), F3, H4/H5 on C1, H6 (anti) and
# H7/H8 on C2

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
    b0_mhz=np.array([1.20318, 0.31237, 0.27351]) * _CM_TO_MHZ,
    source="Kraitchman & Dailey (1955); NIST CCCBDB",
    substitutions={
        "1-13C": (0, M_C13), "2-13C": (1, M_C13),
        "4-D (CHF)": (3, M_D), "6-D (anti)": (5, M_D), "7-D (methyl)": (6, M_D),
    },
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


MOLECULES = [VINYL_FLUORIDE, ACETYL_FLUORIDE, FLUOROETHANE]
