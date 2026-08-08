"""Published experimental structures and rotational constants, for benchmarking.

Every entry is a literature value with its source recorded. These are ground
truth for the comparison scripts; nothing here is fitted or synthesised.
"""

from __future__ import annotations

import numpy as np

_CM_TO_MHZ = 29979.2458


# ── Fluorobenzene, C6H5F ─────────────────────────────────────────────────────
#
# Structure: microwave substitution (r_s) determination.
#   Bak, B.; Christensen, D.; Hansen-Nygaard, L.; Tannenbaum, E.
#   "Microwave Determination of the Structure of Fluorobenzene",
#   J. Chem. Phys. 26, 134 (1957).
#   Cartesians as tabulated by NIST CCCBDB (casno 462066).
#
# Ground-state rotational constants, same source via CCCBDB:
#   A = 0.18892, B = 0.08575, C = 0.05897 cm-1.
#
# Consistency: the geometry reproduces those constants to 0.19-0.23%, which is
# the expected r_s versus r_0 difference, and the constants give an inertial
# defect of +0.046 amu*A^2 -- small and positive, as a planar molecule requires.

FLUOROBENZENE_ELEMS = ["F", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"]

#: Atom order: F, C1 ipso, C2/C6 ortho, C3/C5 meta, C4 para, then H on 2,6,3,5,4.
FLUOROBENZENE_GEOM = np.array([
    [0.0000,  0.0000, -2.2030],   #  0  F
    [0.0000,  0.0000, -0.8490],   #  1  C1  ipso
    [0.0000, -1.2170, -0.1930],   #  2  C2  ortho
    [0.0000,  1.2170, -0.1930],   #  3  C6  ortho
    [0.0000, -1.2080,  1.2020],   #  4  C3  meta
    [0.0000,  1.2080,  1.2020],   #  5  C5  meta
    [0.0000,  0.0000,  1.9030],   #  6  C4  para
    [0.0000, -2.1370, -0.7610],   #  7  H2  ortho
    [0.0000,  2.1370, -0.7610],   #  8  H6  ortho
    [0.0000, -2.1470,  1.7430],   #  9  H3  meta
    [0.0000,  2.1470,  1.7430],   # 10  H5  meta
    [0.0000,  0.0000,  2.9830],   # 11  H4  para
])

FLUOROBENZENE_MASSES = np.array(
    [18.99840322] + [12.0] * 6 + [1.00782503207] * 5
)

#: Observed ground-state constants (MHz).
FLUOROBENZENE_B0_MHZ = np.array([0.18892, 0.08575, 0.05897]) * _CM_TO_MHZ

FLUOROBENZENE_SOURCE = (
    "Bak, Christensen, Hansen-Nygaard & Tannenbaum, J. Chem. Phys. 26, 134 "
    "(1957); Cartesians and constants via NIST CCCBDB casno 462066"
)

#: Symmetry-unique internal coordinates. Equivalent pairs are listed together
#: and averaged by :func:`internal_coordinates`.
FLUOROBENZENE_BONDS = {
    "C1-F   (C-F)":          [(1, 0)],
    "C1-C2  (ipso-ortho)":   [(1, 2), (1, 3)],
    "C2-C3  (ortho-meta)":   [(2, 4), (3, 5)],
    "C3-C4  (meta-para)":    [(4, 6), (5, 6)],
    "C2-H2  (ortho C-H)":    [(2, 7), (3, 8)],
    "C3-H3  (meta C-H)":     [(4, 9), (5, 10)],
    "C4-H4  (para C-H)":     [(6, 11)],
}

FLUOROBENZENE_ANGLES = {
    "F-C1-C2":               [(0, 1, 2), (0, 1, 3)],
    "C2-C1-C6 (ipso)":       [(2, 1, 3)],
    "C1-C2-C3 (ortho)":      [(1, 2, 4), (1, 3, 5)],
    "C2-C3-C4 (meta)":       [(2, 4, 6), (3, 5, 6)],
    "C3-C4-C5 (para)":       [(4, 6, 5)],
    "C1-C2-H2":              [(1, 2, 7), (1, 3, 8)],
    "C2-C3-H3":              [(2, 4, 9), (3, 5, 10)],
    "C3-C4-H4":              [(4, 6, 11), (5, 6, 11)],
}


def internal_coordinates(coords, bonds=None, angles=None) -> dict[str, float]:
    """Symmetry-averaged bond lengths (A) and angles (deg)."""
    bonds = FLUOROBENZENE_BONDS if bonds is None else bonds
    angles = FLUOROBENZENE_ANGLES if angles is None else angles
    x = np.asarray(coords, dtype=float)
    out: dict[str, float] = {}
    for name, pairs in bonds.items():
        out[name] = float(np.mean([np.linalg.norm(x[j] - x[i]) for i, j in pairs]))
    for name, triples in angles.items():
        vals = []
        for i, j, k in triples:
            u, v = x[i] - x[j], x[k] - x[j]
            cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
            vals.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        out[name] = float(np.mean(vals))
    return out
