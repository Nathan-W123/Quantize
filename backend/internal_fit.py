"""
Internal-coordinate data structures and transformations.

Phase 1 — InternalCoordinate, InternalCoordinateSet:
    q(x) and B = dq/dx layer on top of the existing Wilson B-matrix utilities.

Phase 2 — Back-transformation:
    internal_step_to_cartesian_step(B, dq, damping)  →  dx
    apply_internal_step(x0, q_target, coord_set)      →  x_new, error

Phase 3 — Spectral Jacobian conversion:
    spectral_jacobian_q(Jx, Bplus)  →  Jq = Jx @ B+

Units convention (enforced throughout this module):
    Bond lengths      : Angstroms
    Angles            : radians
    Dihedrals         : radians
This matches the Wilson B-matrix rows (∂q/∂x in Å or rad per Å).
Use values_deg() for human-readable display.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from backend.quantum import (
    _detect_bonds,
    _detect_angles,
    _detect_dihedrals,
    _bond_deriv,
    _angle_deriv,
    _dihedral_deriv,
)


# ── Phase 1: data structures ──────────────────────────────────────────────────

@dataclass
class InternalCoordinate:
    """Descriptor for a single primitive internal coordinate."""
    kind: str           # "bond", "angle", or "dihedral"
    atoms: tuple        # atom indices (0-based)
    name: str           # e.g. "bond 1-2", "angle 1-2-3"
    active: bool = True
    prior_value: Optional[float] = None   # natural units (Å or rad); used in Phase 6
    prior_sigma: Optional[float] = None   # natural units; used in Phase 6


def _coord_value(coords: np.ndarray, ic: InternalCoordinate) -> float:
    """Evaluate ic at coords. Returns Å for bonds, radians for angles/dihedrals."""
    xyz = np.asarray(coords, dtype=float)
    if ic.kind == "bond":
        i, j = ic.atoms
        return float(np.linalg.norm(xyz[i] - xyz[j]))
    elif ic.kind == "angle":
        i, o, k = ic.atoms
        u = xyz[i] - xyz[o]
        v = xyz[k] - xyz[o]
        c = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        return float(np.arccos(np.clip(c, -1.0, 1.0)))
    elif ic.kind == "dihedral":
        i, j, k, l = ic.atoms
        b0 = xyz[i] - xyz[j]
        b1 = xyz[k] - xyz[j]
        b2 = xyz[l] - xyz[k]
        b1n = b1 / max(np.linalg.norm(b1), 1e-12)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        return float(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))
    return np.nan


def _coord_B_row(coords: np.ndarray, ic: InternalCoordinate) -> np.ndarray:
    """Wilson B-matrix row for ic, shape (3N,)."""
    xyz = np.asarray(coords, dtype=float)
    if ic.kind == "bond":
        i, j = ic.atoms
        return _bond_deriv(xyz, i, j).ravel()
    elif ic.kind == "angle":
        i, o, k = ic.atoms
        return _angle_deriv(xyz, i, o, k).ravel()
    elif ic.kind == "dihedral":
        i, j, k, l = ic.atoms
        return _dihedral_deriv(xyz, i, j, k, l).ravel()
    return np.zeros(3 * len(xyz))


class InternalCoordinateSet:
    """
    Manages a set of primitive internal coordinates for a molecule.

    Connectivity is detected once at construction from the initial geometry
    and kept fixed during optimization (step sizes are small enough that
    bond topology does not change).

    Parameters
    ----------
    coords : (N, 3) array   Initial geometry in Å (used for connectivity only).
    elems  : list[str]      Element symbols.
    use_dihedrals : bool    Include dihedral angles (default False — start conservative).
    """

    def __init__(self, coords, elems, use_dihedrals: bool = False):
        coords = np.asarray(coords, dtype=float)
        self.elems = list(elems)
        self.use_dihedrals = bool(use_dihedrals)

        bonds = _detect_bonds(coords, self.elems)
        angles = _detect_angles(bonds)
        dihedrals = _detect_dihedrals(bonds) if use_dihedrals else []

        self.coordinates: List[InternalCoordinate] = []
        for i, j in bonds:
            self.coordinates.append(InternalCoordinate(
                kind="bond", atoms=(i, j), name=f"bond {i+1}-{j+1}",
            ))
        for i, o, k in angles:
            self.coordinates.append(InternalCoordinate(
                kind="angle", atoms=(i, o, k), name=f"angle {i+1}-{o+1}-{k+1}",
            ))
        for i, j, k, l in dihedrals:
            self.coordinates.append(InternalCoordinate(
                kind="dihedral", atoms=(i, j, k, l),
                name=f"dihedral {i+1}-{j+1}-{k+1}-{l+1}",
            ))

    # ── Coordinate values ─────────────────────────────────────────────────────

    def values(self, coords) -> np.ndarray:
        """All coordinate values in natural units (Å, rad, rad), shape (n_int,)."""
        return np.array([_coord_value(coords, ic) for ic in self.coordinates], dtype=float)

    def values_deg(self, coords) -> np.ndarray:
        """Same as values() but angles/dihedrals in degrees (for display)."""
        v = self.values(coords)
        for i, ic in enumerate(self.coordinates):
            if ic.kind in ("angle", "dihedral"):
                v[i] = np.degrees(v[i])
        return v

    def active_values(self, coords) -> np.ndarray:
        """Values for active coordinates only, shape (n_active,)."""
        return np.array(
            [_coord_value(coords, ic) for ic in self.coordinates if ic.active],
            dtype=float,
        )

    # ── B-matrix ──────────────────────────────────────────────────────────────

    def B_matrix(self, coords) -> np.ndarray:
        """Wilson B-matrix for all coordinates, shape (n_int, 3N)."""
        coords = np.asarray(coords, dtype=float)
        n3 = 3 * len(self.elems)
        if not self.coordinates:
            return np.zeros((0, n3), dtype=float)
        return np.array([_coord_B_row(coords, ic) for ic in self.coordinates], dtype=float)

    def active_B_matrix(self, coords) -> np.ndarray:
        """B-matrix rows for active coordinates only, shape (n_active, 3N)."""
        coords = np.asarray(coords, dtype=float)
        n3 = 3 * len(self.elems)
        rows = [_coord_B_row(coords, ic) for ic in self.coordinates if ic.active]
        return np.array(rows, dtype=float) if rows else np.zeros((0, n3), dtype=float)

    # ── Masks and metadata ────────────────────────────────────────────────────

    def active_mask(self) -> np.ndarray:
        """Boolean array (n_int,) — True where coordinate is active."""
        return np.array([ic.active for ic in self.coordinates], dtype=bool)

    def active_coords(self) -> List[InternalCoordinate]:
        return [ic for ic in self.coordinates if ic.active]

    def names(self) -> List[str]:
        return [ic.name for ic in self.coordinates]

    def active_names(self) -> List[str]:
        return [ic.name for ic in self.coordinates if ic.active]

    @property
    def n_int(self) -> int:
        return len(self.coordinates)

    @property
    def n_active(self) -> int:
        return sum(1 for ic in self.coordinates if ic.active)

    # ── Pseudo-inverse ────────────────────────────────────────────────────────

    @staticmethod
    def damped_pseudoinverse(B: np.ndarray, damping: float = 1e-6) -> np.ndarray:
        """
        B+ = B^T (B B^T + λI)^{-1}   shape (3N, n_active).

        Minimises ||dx||² subject to B dx ≈ dq.
        The damping λ prevents blow-up when B is rank-deficient.
        """
        BBt = B @ B.T
        n = BBt.shape[0]
        return B.T @ np.linalg.solve(BBt + damping * np.eye(n), np.eye(n))


# ── Phase 2: back-transformation ─────────────────────────────────────────────

def internal_step_to_cartesian_step(
    B: np.ndarray,
    dq: np.ndarray,
    damping: float = 1e-6,
) -> np.ndarray:
    """
    Minimum-norm Cartesian step that achieves the requested internal step.

    dx = B^T (B B^T + λI)^{-1} dq

    Parameters
    ----------
    B       : (n_active, 3N)  Active Wilson B-matrix at current geometry.
    dq      : (n_active,)     Requested internal-coordinate step (Å, rad).
    damping : float           Tikhonov regularisation λ.

    Returns
    -------
    dx : (3N,)  Cartesian step in Å.
    """
    BBt = B @ B.T
    n = BBt.shape[0]
    return B.T @ np.linalg.solve(BBt + damping * np.eye(n), dq)


def _wrap_dihedral_diff(dq: np.ndarray, active_coords: List[InternalCoordinate]) -> np.ndarray:
    """Wrap dihedral differences to (−π, π] so a 179°→−179° move is not treated as 358°."""
    dq = dq.copy()
    for i, ic in enumerate(active_coords):
        if ic.kind == "dihedral":
            dq[i] = (dq[i] + np.pi) % (2.0 * np.pi) - np.pi
    return dq


def apply_internal_step(
    x0,
    q_target: np.ndarray,
    coord_set: InternalCoordinateSet,
    max_micro: int = 20,
    tol: float = 1e-7,
    damping: float = 1e-6,
):
    """
    Find x_new such that coord_set.active_values(x_new) ≈ q_target.

    Uses micro-iterations with damped least-squares back-transformation.

    Parameters
    ----------
    x0        : (N, 3)        Starting Cartesian geometry (Å).
    q_target  : (n_active,)   Target internal-coordinate values (Å/rad).
    coord_set : InternalCoordinateSet
    max_micro : int           Maximum micro-iteration steps.
    tol       : float         Convergence tolerance on residual norm.
    damping   : float         Damping for B+ computation.

    Returns
    -------
    x_new             : (N, 3) ndarray  Converged Cartesian geometry.
    backtransform_err : float           Residual norm |q_target − q(x_new)|.
    """
    x = np.asarray(x0, dtype=float).copy()
    active = coord_set.active_coords()

    for _ in range(max_micro):
        q_curr = coord_set.active_values(x)
        dq = _wrap_dihedral_diff(q_target - q_curr, active)
        if np.linalg.norm(dq) < tol:
            break
        B = coord_set.active_B_matrix(x)
        dx = internal_step_to_cartesian_step(B, dq, damping)
        x = x + dx.reshape(-1, 3)

    q_final = coord_set.active_values(x)
    err = float(np.linalg.norm(_wrap_dihedral_diff(q_target - q_final, active)))
    return x, err


# ── Phase 3: spectral Jacobian conversion ────────────────────────────────────

def spectral_jacobian_q(Jx: np.ndarray, Bplus: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian spectral Jacobian to internal-coordinate Jacobian.

    Jq = Jx @ B+

    Parameters
    ----------
    Jx    : (m, 3N)      Stacked spectral Jacobian (m observables, 3N Cartesian DOF).
    Bplus : (3N, n_act)  Damped pseudo-inverse of the active Wilson B-matrix.

    Returns
    -------
    Jq : (m, n_act)  Spectral Jacobian in internal-coordinate space.
    """
    return Jx @ Bplus


def quantum_terms_q(
    gradient: np.ndarray,
    hessian: np.ndarray,
    Bplus: np.ndarray,
):
    """
    Transform quantum gradient and Hessian from Cartesian to internal-coordinate space.

    gq = B+^T gx
    Hq = B+^T Hx B+

    Note: Hq omits second-derivative coordinate terms (Pulay forces). This is
    an approximation; treat the quantum term as a prior and validate against
    Cartesian mode.

    Parameters
    ----------
    gradient : (3N,)      Energy gradient in Hartree/Å.
    hessian  : (3N, 3N)   Energy Hessian in Hartree/Å².
    Bplus    : (3N, n_act) Damped pseudo-inverse of active Wilson B-matrix.

    Returns
    -------
    gq : (n_act,)        Internal-coordinate gradient.
    Hq : (n_act, n_act)  Internal-coordinate Hessian.
    """
    gq = Bplus.T @ gradient
    Hq = Bplus.T @ hessian @ Bplus
    return gq, Hq


# ── Phase 6: native q-space internal priors ───────────────────────────────────

def build_internal_priors(
    coord_set: InternalCoordinateSet,
    coords,
    sigma_bond: float = 0.05,
    sigma_angle_deg: float = 3.0,
    sigma_dihedral_deg: float = 15.0,
    prior_values: Optional[np.ndarray] = None,
):
    """
    Build a native internal-coordinate prior block for the SVD system.

    Prior residuals:  r_prior = (q0 − q) / sigma
    Prior Jacobian:   J_prior = diag(1 / sigma)   (identity in q-space)

    This is exact in q-space and avoids the approximation of multiplying a
    Cartesian prior by B+.  Append the returned (J_prior, r_prior) directly
    to the internal Jacobian and residual before the SVD step.

    Parameters
    ----------
    coord_set   : InternalCoordinateSet
    coords      : (N, 3)  Current geometry (used to compute q if prior_values is None).
    sigma_bond  : float   Prior width for bonds [Å].
    sigma_angle_deg : float   Prior width for valence angles [deg] (converted to rad).
    sigma_dihedral_deg : float  Prior width for dihedrals [deg] (converted to rad).
    prior_values : (n_active,) or None
        Target q values in natural units (Å, rad).  If None, uses the initial
        geometry passed to coord_set (current coords).

    Returns
    -------
    J_prior : (n_active, n_active)  Diagonal prior Jacobian (diag(1/sigma)).
    r_prior : (n_active,)           Prior residuals (q0 − q_curr) / sigma.
    sigma   : (n_active,)           Prior widths in natural units.
    """
    active = coord_set.active_coords()
    n = len(active)
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)

    sigma_angle_rad = float(sigma_angle_deg) * np.pi / 180.0
    sigma_dihedral_rad = float(sigma_dihedral_deg) * np.pi / 180.0

    sigma = np.array([
        sigma_bond if ic.kind == "bond" else
        sigma_angle_rad if ic.kind == "angle" else
        sigma_dihedral_rad
        for ic in active
    ], dtype=float)
    sigma = np.maximum(sigma, 1e-12)

    if prior_values is None:
        q0 = coord_set.active_values(coords)
    else:
        q0 = np.asarray(prior_values, dtype=float)

    q_curr = coord_set.active_values(coords)
    diff = q0 - q_curr
    diff = _wrap_dihedral_diff(diff, active)   # wrap dihedrals to (−π, π]

    r_prior = diff / sigma
    J_prior = np.diag(1.0 / sigma)
    return J_prior, r_prior, sigma
