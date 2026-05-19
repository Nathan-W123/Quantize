"""
Geometric computation of RAM-lite torsion parameters F and rho.

Provides functions to derive the internal-rotation kinetic constant F
and the coupling parameter rho from a molecular geometry, along with a
finite-difference Jacobian of torsion level energies with respect to
Cartesian coordinates.

Physical formulae
-----------------
F [cm^-1] = h / (8 pi^2 c I_alpha) = 16.857629206 / I_alpha [amu Ang^2]

rho = I_alpha / I_total
where I_total is the full molecular moment of inertia about the same axis.

This is a one-axis approximation.  It does not account for rho-axis
re-orientation or the full inertia-tensor treatment used in global RAM.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

from backend.torsion_hamiltonian import TorsionHamiltonianSpec
from backend.torsion_uncertainty import torsion_level_observables

# F [cm^-1] = _F_CM1_PER_AMU_A2 / I_alpha [amu Ang^2]
_F_CM1_PER_AMU_A2: float = 16.857629206


def _axis_unit_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Unit vector from p1 to p2."""
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Axis atoms are coincident; cannot define rotation axis direction.")
    return v / n


def _perp_sq(coords: np.ndarray, origin: np.ndarray, hat: np.ndarray) -> np.ndarray:
    """Squared perpendicular distance of each atom (row in coords) from the axis."""
    d = coords - origin[None, :]
    axial = (d @ hat)[:, None] * hat[None, :]
    perp = d - axial
    return np.einsum("ij,ij->i", perp, perp)


def compute_F_rho_from_geometry(
    coords_ang: np.ndarray,
    masses_amu: np.ndarray,
    top_indices: Sequence[int],
    axis_atom_indices: tuple[int, int],
) -> tuple[float, float]:
    """
    Compute RAM-lite F [cm^-1] and rho from molecular geometry.

    Parameters
    ----------
    coords_ang : (N, 3) Cartesian coordinates in Angstroms
    masses_amu : (N,) atomic masses in amu
    top_indices : atom indices of the rotating internal top (e.g. methyl H atoms)
    axis_atom_indices : two atom indices defining the internal rotation axis

    Returns
    -------
    F : float  torsional kinetic constant [cm^-1]
    rho : float  coupling parameter I_alpha / I_total (dimensionless, 0 < rho < 1)
    """
    coords = np.asarray(coords_ang, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).ravel()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_ang must be shape (N, 3).")
    N = coords.shape[0]
    if masses.size != N:
        raise ValueError("masses_amu must have the same length as coords_ang rows.")
    if len(axis_atom_indices) != 2:
        raise ValueError("axis_atom_indices must be a 2-element sequence.")
    i1, i2 = int(axis_atom_indices[0]), int(axis_atom_indices[1])
    if i1 == i2:
        raise ValueError("axis_atom_indices must name two distinct atoms.")
    if not (0 <= i1 < N and 0 <= i2 < N):
        raise IndexError(f"axis_atom_indices {axis_atom_indices} out of range for {N} atoms.")
    top_list = [int(t) for t in top_indices]
    if not top_list:
        raise ValueError("top_indices must be non-empty.")
    for t in top_list:
        if not (0 <= t < N):
            raise IndexError(f"top_index {t} out of range for {N} atoms.")

    hat = _axis_unit_vector(coords[i1], coords[i2])
    r2 = _perp_sq(coords, coords[i1], hat)

    top_mask = np.zeros(N, dtype=bool)
    for t in top_list:
        top_mask[t] = True

    I_alpha = float(np.dot(masses[top_mask], r2[top_mask]))
    I_total = float(np.dot(masses, r2))

    if I_alpha <= 0.0:
        raise ValueError("Top inertia I_alpha <= 0; check top_indices and geometry.")
    if I_total <= 0.0:
        raise ValueError("Total inertia I_total <= 0; check geometry and masses.")

    return _F_CM1_PER_AMU_A2 / I_alpha, I_alpha / I_total


def update_spec_F_rho(
    spec: TorsionHamiltonianSpec,
    coords_ang: np.ndarray,
    masses_amu: np.ndarray,
    top_indices: Sequence[int],
    axis_atom_indices: tuple[int, int],
) -> TorsionHamiltonianSpec:
    """Return a deep copy of spec with F and rho recomputed from the given geometry."""
    F, rho = compute_F_rho_from_geometry(coords_ang, masses_amu, top_indices, axis_atom_indices)
    new_spec = deepcopy(spec)
    new_spec.F = F
    new_spec.rho = rho
    return new_spec


def torsion_geometry_jacobian(
    spec: TorsionHamiltonianSpec,
    coords_ang: np.ndarray,
    masses_amu: np.ndarray,
    top_indices: Sequence[int],
    axis_atom_indices: tuple[int, int],
    level_requests: list[dict],
    *,
    dx_ang: float = 1e-4,
) -> np.ndarray:
    """
    Finite-difference Jacobian of torsion level energies w.r.t. Cartesian coordinates.

    Each column corresponds to perturbing one Cartesian degree of freedom (x, y, z
    for each atom in order), updating F and rho from the perturbed geometry, and
    recomputing the level energies.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec  (used as template; F/rho overridden at each step)
    coords_ang : (N, 3) reference geometry in Angstroms
    masses_amu : (N,) atomic masses
    top_indices, axis_atom_indices : same as compute_F_rho_from_geometry
    level_requests : list of {'J': int, 'K': int, 'level_index': int}
    dx_ang : finite-difference step in Angstroms (default 1e-4)

    Returns
    -------
    J_out : (n_levels, 3*N) float array — columns ordered [x0,y0,z0, x1,y1,z1, ...]
    """
    coords = np.asarray(coords_ang, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    N = coords.shape[0]
    n_lev = len(level_requests)
    J_out = np.zeros((n_lev, 3 * N), dtype=float)

    def _eval(c: np.ndarray) -> np.ndarray:
        try:
            s = update_spec_F_rho(spec, c, masses, top_indices, axis_atom_indices)
            return torsion_level_observables(s, level_requests)
        except Exception:
            return np.full(n_lev, np.nan)

    ref = _eval(coords)
    for i in range(N):
        for j in range(3):
            col = 3 * i + j
            cp = coords.copy()
            cm = coords.copy()
            cp[i, j] += dx_ang
            cm[i, j] -= dx_ang
            fp = _eval(cp)
            fm = _eval(cm)
            valid = np.isfinite(fp) & np.isfinite(fm)
            J_out[valid, col] = (fp[valid] - fm[valid]) / (2.0 * dx_ang)
    return J_out
