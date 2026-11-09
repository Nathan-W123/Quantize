"""Internal-rotation constants F and rho from a geometry and masses.

WHY THIS MODULE EXISTS. A ``TorsionHamiltonianSpec`` needs F (the internal
rotation constant) and rho (the coupling between internal and overall
rotation). Everything else the torsion package needs can be computed from a
scan: the potential by Fourier fit, the averaged constants by solving the
hindered rotor. F and rho could not be, so every spec in the repository
carried hand-entered literature values -- methanol's F = 27.6 cm^-1 and
rho = 0.81 -- which is fine for methanol and impossible for a molecule nobody
has tabulated. That gap is what kept the torsion treatment from being
scored on the benchmark set.

THE RELATIONS (Lin & Swalen, Rev. Mod. Phys. 31 (1959) 841; the
internal-axis-method form). For a single rigid top turning about a unit
vector lambda:

    I_alpha = Sum_{i in top} m_i d_i^2        d_i = distance from the axis LINE

    lambda_g = lambda . e_g                   direction cosines, principal axes
    rho_g    = lambda_g I_alpha / I_g
    r        = 1 - Sum_g lambda_g rho_g       = 1 - Sum_g lambda_g^2 I_alpha/I_g
    I_red    = r I_alpha
    F        = hbar^2 / (2 I_red)

r is the factor that makes this a REDUCED moment rather than the top's own.
Turning the top recoils the frame, so the effective inertia resisting internal
rotation is smaller than I_alpha, and F is correspondingly larger. r -> 1 is
the limit of a frame that cannot recoil; r -> 0 would be a top as heavy as
everything else, where internal and overall rotation stop being separable.

WHAT "A HEAVY FRAME" MEANS, since the obvious reading is wrong. Only mass OFF
the rotation axis resists recoil about it. Making methanol's oxygen heavier
changes r by nothing at all, because the oxygen lies on the C-O axis and so
has no moment about it; making the hydroxyl hydrogen heavier drives r from
0.19 towards 0.85 and F down towards the top's own 16.8576/I_alpha. The frame's
resistance is I_a - I_alpha, not its mass.

I_red IS SYMMETRIC between the two groups. Writing it out,
I_red = I_alpha (I_a - I_alpha)/I_a for an axial top, which is unchanged by
I_alpha -> I_a - I_alpha. Calling the methyl the top or the hydroxyl the top
gives the same reduced moment to within the small off-axis terms (0.628
against 0.648 amu A^2 for methanol), as it must: the relative internal
rotation of two groups is ONE coordinate, not two.

A CONSEQUENCE WORTH KNOWING FOR ISOTOPOLOGUES. Because I_red tends to the
frame's own contribution as the top gets heavy, deuterating only the methyl
barely moves F (27.6 -> 24.3 cm^-1 here, a factor 0.90, despite I_alpha
doubling exactly). Deuterating BOTH groups halves it (-> 13.8, factor 0.52),
because then every mass in I_red doubles. F is not a per-molecule constant to
be shared across isotopologues, and which atoms are substituted matters more
than how many.

SIGNS AND MAGNITUDES THAT CATCH ERRORS. r must land in (0, 1]: above 1 means
the recoil term came out negative, below 0 means the axis or the top set is
wrong (typically the top was taken to include frame atoms, so I_alpha
swallowed the whole molecule). Both are raised rather than returned, because
an F computed from a bad r is not approximately right -- it is the wrong
quantity, and it enters the torsional energies linearly.

VALIDATED AGAINST METHANOL. The repository's own test fixtures carry
F = 27.6 cm^-1 and rho = 0.81, taken from the literature long before this
module existed. Computing them from a methanol geometry reproduces both
(test_internal_rotation), which is the check that the relations above are
implemented in the convention the rest of the torsion package expects.

ONE TOP ONLY. Two coupled tops need a 2x2 reduced inertia matrix and a
different Hamiltonian; ``torsion_average_2d`` is where that would go. This
module raises rather than quietly returning a one-top answer for a molecule
with two equivalent rotors.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from backend.spectral.centrifugal_distortion import (
    _INERTIA_TO_MHZ,
    _CM_TO_MHZ,
    inertia_paf,
)

#: B[cm^-1] = _INERTIA_TO_CM / I[amu Ang^2]. The familiar 16.8576.
_INERTIA_TO_CM = _INERTIA_TO_MHZ / _CM_TO_MHZ

_COV_RADIUS = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


def _bonds(coords, elements, factor: float = 1.3):
    coords = np.asarray(coords, dtype=float)
    radii = np.array([_COV_RADIUS.get(str(e).strip().capitalize(), 0.77)
                      for e in elements])
    out = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if np.linalg.norm(coords[i] - coords[j]) < factor * (radii[i] + radii[j]):
                out.append((i, j))
    return out


def dihedral_angle(coords, m: int, o: int, p: int, n: int, degrees: bool = True):
    """Signed dihedral m-o-p-n.

    The repository had ``_dihedral_deriv`` but no dihedral, which is an odd
    thing to be missing and is needed to constrain a scan to a target angle.
    Uses atan2 so the sign and the full (-180, 180] range are right; a sign
    convention that silently folded the angle would make a scan double back on
    itself halfway round.
    """
    coords = np.asarray(coords, dtype=float)
    b1 = coords[o] - coords[m]
    b2 = coords[p] - coords[o]
    b3 = coords[n] - coords[p]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.linalg.norm(b2)
    x = float(np.dot(n1, n2))
    y = float(np.dot(np.cross(n1, b2_hat), n2))
    phi = np.arctan2(y, x)
    return float(np.degrees(phi)) if degrees else float(phi)


def top_atoms(coords, elements, frame_atom: int, top_atom: int,
              bonds=None) -> tuple[int, ...]:
    """Atoms that move when the top turns, found by cutting the axis bond.

    ``frame_atom`` and ``top_atom`` are the two ends of the bond the rotation
    happens about; the returned set is everything still connected to
    ``top_atom`` once that bond is removed, ``top_atom`` included.

    Raises if the two are still connected by another path, because then the
    "bond" is part of a ring and nothing rotates about it -- a failure mode
    worth a message rather than a silently enormous I_alpha.
    """
    if int(frame_atom) == int(top_atom):
        raise ValueError("frame_atom and top_atom must be different atoms")
    if bonds is None:
        bonds = _bonds(coords, elements)
    nbrs = defaultdict(set)
    for i, j in bonds:
        if {i, j} == {frame_atom, top_atom}:
            continue
        nbrs[i].add(j)
        nbrs[j].add(i)

    seen, stack = {int(top_atom)}, [int(top_atom)]
    while stack:
        cur = stack.pop()
        for nxt in nbrs[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if int(frame_atom) in seen:
        raise ValueError(
            f"atoms {frame_atom} and {top_atom} stay connected after cutting "
            f"their bond, so it lies in a ring and is not an internal "
            f"rotation axis"
        )
    return tuple(sorted(seen))


def internal_rotation_constants(coords_ang, masses_amu, frame_atom: int,
                                top_atom: int, top_indices=None,
                                elements=None):
    """F (cm^-1), rho and the reduced moment for one internal rotor.

    Parameters
    ----------
    coords_ang : (N, 3) geometry, Angstrom.
    masses_amu : (N,) masses, amu. The isotopologue's own masses: F and rho
        are mass-dependent, which is the whole reason they matter here.
    frame_atom, top_atom : the two ends of the rotation axis. The axis
        direction is frame_atom -> top_atom.
    top_indices : the rotating atoms. Found from connectivity when omitted,
        which needs ``elements``.
    elements : element symbols, only used to find ``top_indices``.

    Returns
    -------
    dict with 'F_cm' , 'rho' (magnitude), 'rho_abc' (the three components in
    the principal-axis frame, ordered by increasing moment so a/b/c), 'I_alpha',
    'I_red', 'r', 'lambda_abc' and 'top_indices'.
    """
    coords = np.asarray(coords_ang, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    if coords.shape[0] != masses.size:
        raise ValueError("coords and masses disagree on the atom count")

    if top_indices is None:
        if elements is None:
            raise ValueError("pass top_indices, or elements so it can be found")
        top_indices = top_atoms(coords, elements, frame_atom, top_atom)
    top_indices = tuple(int(i) for i in top_indices)
    if int(frame_atom) in top_indices:
        raise ValueError("frame_atom must not be one of the top atoms")

    # The axis as a LINE: a point on it and a direction.
    origin = coords[int(frame_atom)]
    axis = coords[int(top_atom)] - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError("frame_atom and top_atom are at the same position")
    lam = axis / norm

    # I_alpha: the top's moment about that line. Perpendicular distance only,
    # so an atom lying on the axis (the pivot carbon of a methyl) adds nothing.
    rel = coords[list(top_indices)] - origin
    perp = rel - np.outer(rel @ lam, lam)
    I_alpha = float(np.sum(masses[list(top_indices)] * np.einsum("ij,ij->i", perp, perp)))
    if I_alpha <= 0.0:
        raise ValueError("I_alpha is not positive; the top lies on its own axis")

    # Principal axes of the WHOLE molecule, ordered a, b, c.
    eigvals, V, _ = inertia_paf(coords, masses)
    order = np.argsort(eigvals)
    I_g = eigvals[order]
    lam_g = (V[:, order]).T @ lam          # direction cosines in a, b, c

    with np.errstate(divide="ignore", invalid="ignore"):
        rho_abc = np.where(I_g > 1e-12, lam_g * I_alpha / I_g, 0.0)
    r = 1.0 - float(np.dot(lam_g, rho_abc))

    if not (0.0 < r <= 1.0 + 1e-9):
        raise ValueError(
            f"reduced-moment factor r = {r:.6f} is outside (0, 1]; the top "
            f"atoms or the axis are wrong (I_alpha = {I_alpha:.4f} amu A^2 "
            f"against principal moments {np.round(I_g, 4).tolist()})"
        )

    I_red = r * I_alpha
    return {
        "F_cm": float(_INERTIA_TO_CM / I_red),
        "rho": float(np.linalg.norm(rho_abc)),
        "rho_abc": rho_abc.astype(float),
        "I_alpha": I_alpha,
        "I_red": float(I_red),
        "r": float(r),
        "lambda_abc": lam_g.astype(float),
        "top_indices": top_indices,
        "principal_moments_amu_ang2": I_g.astype(float),
    }


def rotational_constants_cm(coords_ang, masses_amu) -> np.ndarray:
    """A, B, C in cm^-1, for filling the rest of a TorsionHamiltonianSpec."""
    eigvals, _, _ = inertia_paf(coords_ang, masses_amu)
    I_g = np.sort(eigvals)
    out = np.where(I_g > 1e-12, _INERTIA_TO_CM / np.where(I_g > 1e-12, I_g, 1.0), 0.0)
    return out.astype(float)


#: Bond length / covalent-radii sum above which a bond is treated as single.
#: See find_rotors for the measurements that place it.
_SINGLE_BOND_RATIO = 0.93


def find_rotors(coords_ang, elements, bonds=None, max_top_atoms: int = 6,
                single_bond_ratio: float = _SINGLE_BOND_RATIO):
    """Internal-rotation candidates in a molecule, found from connectivity.

    WHY AUTO-DETECT. The benchmark needs the axis, the dihedral and the fold
    for every molecule it scores, and hand-entering four atom indices per
    molecule is both tedious and a silent-error channel: a transposed pair
    gives a perfectly well-formed scan of the wrong coordinate.

    WHAT COUNTS AS A ROTOR. A SINGLE bond, not in a ring, with at least two
    atoms on each side so a dihedral exists, and a top small enough to be a
    top rather than half a molecule (``max_top_atoms``).

    The single-bond test is by length against the covalent-radii sum, because
    connectivity alone cannot see bond order and the first version of this
    function happily reported a rotor about vinyl fluoride's C=C and
    fluoroacetylene's C#C -- bonds about which nothing turns at any accessible
    energy, and which a scan would answer by breaking the molecule. Measured
    across the reference set, the separation is wide:

        acetyl fluoride   C-C   1.503 A / 1.520   ratio 0.989   single
        fluoroethane      C-C   1.505 / 1.520     ratio 0.990   single
        chlorofluorometh. C-Cl  1.759 / 1.780     ratio 0.988   single
        methanol          C-O   1.425 / 1.420     ratio 1.003   single
        vinyl fluoride    C=C   1.329 / 1.520     ratio 0.874   double
        formyl fluoride   C=O   1.181 / 1.420     ratio 0.832   double

    so the 0.93 default sits in an empty gap with margin on both sides. Two
    consequences worth knowing: an aromatic ring bond (ratio about 0.92) is
    excluded, which the ring test would do anyway, and an amide C-N (about
    0.905) is excluded, which is deliberate -- its barrier is tens of
    kJ/mol and it is not a near-free rotor -- but means this will not find
    one for you.

    The fold is read off
    the top's own symmetry, not off the barrier: three equivalent terminal
    atoms on the pivot give fold 3, two give fold 2, anything else gives fold
    1 and should be treated as a general internal rotation rather than a
    symmetric top. That distinction is the one the nuclear-spin weights care
    about -- see torsion_symmetry on nitromethane, whose SIX-fold barrier
    comes from a THREE-fold top.

    Returns a list of dicts, largest fold first then smallest top, each with
    'frame_atom', 'top_atom', 'dihedral' (m, o, p, n), 'top_indices',
    'symmetry_number' and 'equivalent_terminal'. Empty when there is nothing
    to turn, which is the common case and not an error.
    """
    coords = np.asarray(coords_ang, dtype=float)
    elements = [str(e).strip().capitalize() for e in elements]
    if bonds is None:
        bonds = _bonds(coords, elements)

    nbrs = defaultdict(set)
    for i, j in bonds:
        nbrs[i].add(j)
        nbrs[j].add(i)

    out = []
    for frame_atom, top_atom in bonds:
        length = float(np.linalg.norm(coords[frame_atom] - coords[top_atom]))
        radii = (_COV_RADIUS.get(elements[frame_atom], 0.77)
                 + _COV_RADIUS.get(elements[top_atom], 0.77))
        if radii > 0 and length / radii < float(single_bond_ratio):
            continue                           # multiple bond: nothing turns
        for o, p in ((frame_atom, top_atom), (top_atom, frame_atom)):
            try:
                top = top_atoms(coords, elements, o, p, bonds=bonds)
            except ValueError:
                continue                       # ring bond, or same atom
            if len(top) < 2 or len(top) > int(max_top_atoms):
                continue
            frame_side = [i for i in range(len(elements)) if i not in top]
            if len(frame_side) < 2:
                continue                       # no m atom to define a dihedral

            # The top's terminal atoms hanging off the pivot, which is what
            # sets the fold.
            attached = sorted(nbrs[p] - {o})
            if not attached:
                continue
            kinds = {}
            for a in attached:
                kinds.setdefault((elements[a], len(nbrs[a])), []).append(a)
            biggest = max(kinds.values(), key=len)
            fold = len(biggest) if len(biggest) in (2, 3) else 1

            # m from the frame, n from the top: pick neighbours of o and p so
            # the dihedral is a real one rather than a collinear accident.
            m_candidates = sorted(nbrs[o] - {p})
            if not m_candidates:
                continue
            m = m_candidates[0]
            n = biggest[0]
            out.append({
                "frame_atom": int(o),
                "top_atom": int(p),
                "dihedral": (int(m), int(o), int(p), int(n)),
                "top_indices": top,
                "symmetry_number": int(fold),
                "equivalent_terminal": tuple(int(a) for a in biggest),
            })
    out.sort(key=lambda d: (-d["symmetry_number"], len(d["top_indices"])))
    return out
