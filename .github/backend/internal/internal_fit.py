"""
Internal-coordinate data structures and transformations.

Phase 1 â€” InternalCoordinate, InternalCoordinateSet:
    q(x) and B = dq/dx layer on top of the existing Wilson B-matrix utilities.

Phase 2 â€” Back-transformation:
    internal_step_to_cartesian_step(B, dq, damping)  â†’  dx
    apply_internal_step(x0, q_target, coord_set)      â†’  x_new, error

Phase 3 â€” Spectral Jacobian conversion:
    spectral_jacobian_q(Jx, Bplus)  â†’  Jq = Jx @ B+

Units convention (enforced throughout this module):
    Bond lengths      : Angstroms
    Angles            : radians
    Dihedrals         : radians
This matches the Wilson B-matrix rows (âˆ‚q/âˆ‚x in Ã… or rad per Ã…).
Use values_deg() for human-readable display.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any

from backend.quantum import (
    _detect_bonds,
    _detect_angles,
    _detect_dihedrals,
    _bond_deriv,
    _angle_deriv,
    _dihedral_deriv,
)
from backend.internal.internal_prior_library import resolve_default_prior
from backend.priors.prior_models import PriorRecord, normalize_user_prior_atoms
from backend.priors.prior_weighting import (
    adaptive_sigma_from_identifiability,
    identifiability_scores_from_jacobian,
)


# â”€â”€ Phase 1: data structures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class InternalCoordinate:
    """Descriptor for a single primitive internal coordinate."""
    kind: str           # "bond", "angle", or "dihedral"
    atoms: tuple        # atom indices (0-based)
    name: str           # e.g. "bond 1-2", "angle 1-2-3"
    active: bool = True
    prior_value: Optional[float] = None   # natural units (Ã… or rad); used in Phase 6
    prior_sigma: Optional[float] = None   # natural units; used in Phase 6


def _coord_value(coords: np.ndarray, ic: InternalCoordinate) -> float:
    """Evaluate ic at coords. Returns Ã… for bonds, radians for angles/dihedrals."""
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
    coords : (N, 3) array   Initial geometry in Ã… (used for connectivity only).
    elems  : list[str]      Element symbols.
    use_dihedrals : bool    Include dihedral angles (default False â€” start conservative).
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

    # â”€â”€ Coordinate values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def values(self, coords) -> np.ndarray:
        """All coordinate values in natural units (Ã…, rad, rad), shape (n_int,)."""
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

    # â”€â”€ B-matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Masks and metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def active_mask(self) -> np.ndarray:
        """Boolean array (n_int,) â€” True where coordinate is active."""
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

    # â”€â”€ Pseudo-inverse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def damped_pseudoinverse(B: np.ndarray, damping: float = 1e-6) -> np.ndarray:
        """
        B+ = B^T (B B^T + Î»I)^{-1}   shape (3N, n_active).

        Minimises ||dx||Â² subject to B dx â‰ˆ dq.
        The damping Î» prevents blow-up when B is rank-deficient.
        """
        BBt = B @ B.T
        n = BBt.shape[0]
        return B.T @ np.linalg.solve(BBt + damping * np.eye(n), np.eye(n))

    def b_rank_diagnostics(self, coords) -> dict:
        """
        SVD-based condition number and rank of the active Wilson B-matrix.

        Returns
        -------
        dict with keys:
          n_coords  : number of active internal coordinates
          n_dof     : number of Cartesian DOF (3N)
          rank      : numerical rank (singular values > 1e-10 Ã— Ïƒ_max)
          kappa_B   : condition number Ïƒ_max/Ïƒ_min_nonzero, or None if rank < 2
        """
        B = self.active_B_matrix(coords)
        if B.size == 0 or B.shape[0] == 0:
            return {"n_coords": 0, "n_dof": int(3 * len(self.elems)), "rank": 0, "kappa_B": None}
        sv = np.linalg.svd(B, compute_uv=False)
        sv_pos = sv[sv > 1e-10 * max(float(sv[0]), 1e-30)]
        rank = int(len(sv_pos))
        kappa = float(sv[0] / sv_pos[-1]) if rank > 1 else 1.0
        return {
            "n_coords": int(B.shape[0]),
            "n_dof": int(B.shape[1]),
            "rank": rank,
            "kappa_B": kappa,
        }


# â”€â”€ Phase 2: back-transformation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def internal_step_to_cartesian_step(
    B: np.ndarray,
    dq: np.ndarray,
    damping: float = 1e-6,
) -> np.ndarray:
    """
    Minimum-norm Cartesian step that achieves the requested internal step.

    dx = B^T (B B^T + Î»I)^{-1} dq

    Parameters
    ----------
    B       : (n_active, 3N)  Active Wilson B-matrix at current geometry.
    dq      : (n_active,)     Requested internal-coordinate step (Ã…, rad).
    damping : float           Tikhonov regularisation Î».

    Returns
    -------
    dx : (3N,)  Cartesian step in Ã….
    """
    BBt = B @ B.T
    n = BBt.shape[0]
    return B.T @ np.linalg.solve(BBt + damping * np.eye(n), dq)


def _wrap_dihedral_diff(dq: np.ndarray, active_coords: List[InternalCoordinate]) -> np.ndarray:
    """Wrap dihedral differences to (âˆ’Ï€, Ï€] so a 179Â°â†’âˆ’179Â° move is not treated as 358Â°."""
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
    Find x_new such that coord_set.active_values(x_new) â‰ˆ q_target.

    Uses micro-iterations with damped least-squares back-transformation.

    Parameters
    ----------
    x0        : (N, 3)        Starting Cartesian geometry (Ã…).
    q_target  : (n_active,)   Target internal-coordinate values (Ã…/rad).
    coord_set : InternalCoordinateSet
    max_micro : int           Maximum micro-iteration steps.
    tol       : float         Convergence tolerance on residual norm.
    damping   : float         Damping for B+ computation.

    Returns
    -------
    x_new             : (N, 3) ndarray  Converged Cartesian geometry.
    backtransform_err : float           Residual norm |q_target âˆ’ q(x_new)|.
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


# â”€â”€ Phase 3: spectral Jacobian conversion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    gradient : (3N,)      Energy gradient in Hartree/Ã….
    hessian  : (3N, 3N)   Energy Hessian in Hartree/Ã…Â².
    Bplus    : (3N, n_act) Damped pseudo-inverse of active Wilson B-matrix.

    Returns
    -------
    gq : (n_act,)        Internal-coordinate gradient.
    Hq : (n_act, n_act)  Internal-coordinate Hessian.
    """
    gq = Bplus.T @ gradient
    Hq = Bplus.T @ hessian @ Bplus
    return gq, Hq


# â”€â”€ Phase 6: native q-space internal priors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_internal_priors(
    coord_set: InternalCoordinateSet,
    coords,
    sigma_bond: float = 0.05,
    sigma_angle_deg: float = 3.0,
    sigma_dihedral_deg: float = 15.0,
    prior_values: Optional[np.ndarray] = None,
    prior_mode: str = "soft",
    prior_specs: Optional[list[dict[str, Any]]] = None,
    elems: Optional[list[str]] = None,
    freeze_sigma_floor: float = 1e-6,
    spectral_jacobian_q: Optional[np.ndarray] = None,
    adaptive_config: Optional[dict[str, Any]] = None,
    sv_rel_threshold: float = 1e-3,
    sigma_scale: float = 1.0,
    return_metadata: bool = False,
):
    """
    Build a native internal-coordinate prior block for the SVD system.

    Prior residuals:  r_prior = (q0 âˆ’ q) / sigma
    Prior Jacobian:   J_prior = diag(1 / sigma)   (identity in q-space)

    This is exact in q-space and avoids the approximation of multiplying a
    Cartesian prior by B+.  Append the returned (J_prior, r_prior) directly
    to the internal Jacobian and residual before the SVD step.

    Parameters
    ----------
    coord_set   : InternalCoordinateSet
    coords      : (N, 3)  Current geometry (used to compute q if prior_values is None).
    sigma_bond  : float   Prior width for bonds [Ã…].
    sigma_angle_deg : float   Prior width for valence angles [deg] (converted to rad).
    sigma_dihedral_deg : float  Prior width for dihedrals [deg] (converted to rad).
    prior_values : (n_active,) or None
        Target q values in natural units (Ã…, rad).  If None, uses the initial
        geometry passed to coord_set (current coords).

    Parameters (new)
    ----------------
    prior_mode : {"off", "soft", "adaptive", "hard_freeze"}
        off: disables prior block (returns empty matrices).
        soft/adaptive: Gaussian prior in q-space (adaptive uses provided sigmas for now).
        hard_freeze: same formulation with tiny sigma floor for user-marked coordinates.
    prior_specs : list[dict], optional
        User-specified priors. Each item should include:
        type/kind, atoms, target, sigma, optional units/source/confidence/notes, optional mode.
    elems : list[str], optional
        Element symbols for chemistry default prior matching.
    freeze_sigma_floor : float
        Small sigma used for hard-freeze coordinates.

    Returns
    -------
    J_prior : (n_active, n_active)  Diagonal prior Jacobian (diag(1/sigma)).
    r_prior : (n_active,)           Prior residuals (q0 âˆ’ q_curr) / sigma.
    sigma   : (n_active,)           Prior widths in natural units.
    meta    : list[dict], optional  Prior provenance metadata per coordinate (if return_metadata).
    """
    mode = str(prior_mode or "soft").strip().lower()
    if mode not in {"off", "soft", "adaptive", "hard_freeze"}:
        raise ValueError(f"Unknown internal prior mode: {prior_mode!r}")

    active = coord_set.active_coords()
    n = len(active)
    if n == 0 or mode == "off":
        out = (np.zeros((0, 0)), np.zeros(0), np.zeros(0))
        return out + ([],) if return_metadata else out

    sigma_angle_rad = float(sigma_angle_deg) * np.pi / 180.0
    sigma_dihedral_rad = float(sigma_dihedral_deg) * np.pi / 180.0

    sigma_base = np.array([
        sigma_bond if ic.kind == "bond" else
        sigma_angle_rad if ic.kind == "angle" else
        sigma_dihedral_rad
        for ic in active
    ], dtype=float)
    sigma = np.maximum(sigma_base.copy(), 1e-12)

    if prior_values is None:
        q0 = coord_set.active_values(coords)
    else:
        q0 = np.asarray(prior_values, dtype=float)
    if q0.size != n:
        raise ValueError("prior_values length must match number of active internal coordinates.")

    q_curr = coord_set.active_values(coords)

    # Build metadata/provenance table seeded from current defaults.
    meta: list[dict[str, Any]] = []
    for i, ic in enumerate(active):
        meta.append({
            "name": ic.name,
            "kind": ic.kind,
            "atoms": tuple(int(a) for a in ic.atoms),
            "target_value": float(q0[i]),
            "sigma": float(sigma[i]),
            "units": "angstrom" if ic.kind == "bond" else "radian",
            "source": "geometry_initial",
            "confidence": None,
            "notes": None,
            "mode": mode,
            "bond_order": None,
            "atom_types": None,
        })

    # Apply chemistry defaults where available.
    if elems is not None:
        for i, ic in enumerate(active):
            dflt = resolve_default_prior(ic, elems, q_curr[i])
            if dflt is None:
                continue
            q0[i] = float(dflt.target_value)
            sigma[i] = float(max(dflt.sigma, 1e-12))
            meta[i].update({
                "target_value": float(dflt.target_value),
                "sigma": float(max(dflt.sigma, 1e-12)),
                "units": dflt.units,
                "source": dflt.source,
                "confidence": dflt.confidence,
                "notes": dflt.notes,
                "bond_order": dflt.bond_order,
                "atom_types": dflt.atom_types,
            })

    # Apply user-provided priors last (highest precedence).
    by_key = {(ic.kind, tuple(ic.atoms)): i for i, ic in enumerate(active)}
    for raw in (prior_specs or []):
        kind = str(raw.get("type", raw.get("kind", ""))).strip().lower()
        atoms_raw = raw.get("atoms", [])
        if not kind or not isinstance(atoms_raw, (list, tuple)):
            continue
        atoms = normalize_user_prior_atoms(atoms_raw, len(coord_set.elems))
        idx = by_key.get((kind, tuple(atoms)))
        if idx is None:
            continue
        units = str(raw.get("units", "angstrom" if kind == "bond" else "radian")).strip().lower()
        target = float(raw.get("target", raw.get("target_value", q0[idx])))
        sig = float(raw.get("sigma", sigma[idx]))
        row_mode = str(raw.get("mode", mode)).strip().lower()
        if kind != "bond" and units in {"degree", "degrees", "deg"}:
            target = float(np.deg2rad(target))
            sig = float(np.deg2rad(sig))
            units = "radian"
        if kind == "bond" and units in {"angstrom", "a", "Ã¥"}:
            units = "angstrom"
        if row_mode == "hard_freeze":
            sig = min(abs(sig), float(freeze_sigma_floor))
        q0[idx] = float(target)
        sigma[idx] = float(max(abs(sig), 1e-12))
        meta[idx].update({
            "target_value": float(target),
            "sigma": float(max(abs(sig), 1e-12)),
            "units": units,
            "source": str(raw.get("source", "user_supplied")),
            "confidence": raw.get("confidence"),
            "notes": raw.get("notes"),
            "mode": row_mode,
        })

    if mode == "hard_freeze":
        sigma = np.minimum(sigma, float(freeze_sigma_floor))

    # Adaptive weighting: weaken priors for data-visible coordinates.
    if mode == "adaptive":
        ad = adaptive_config or {}
        gamma = float(ad.get("identifiability_gamma", 2.0))
        min_scale = float(ad.get("min_sigma_scale", 0.5))
        max_scale = float(ad.get("max_sigma_scale", 3.0))
        if spectral_jacobian_q is not None:
            scores, _, _ = identifiability_scores_from_jacobian(
                spectral_jacobian_q,
                sv_rel_threshold=sv_rel_threshold,
            )
            if scores.shape == sigma.shape:
                sigma, scale = adaptive_sigma_from_identifiability(
                    sigma,
                    scores,
                    gamma=gamma,
                    min_sigma_scale=min_scale,
                    max_sigma_scale=max_scale,
                )
                for i in range(n):
                    meta[i]["identifiability_score"] = float(scores[i])
                    meta[i]["sigma_scale"] = float(scale[i])
                    meta[i]["adaptive_gamma"] = float(gamma)
            else:
                for i in range(n):
                    meta[i]["identifiability_score"] = None
                    meta[i]["sigma_scale"] = 1.0
                    meta[i]["adaptive_note"] = "shape_mismatch"
        else:
            for i in range(n):
                meta[i]["identifiability_score"] = None
                meta[i]["sigma_scale"] = 1.0
                meta[i]["adaptive_note"] = "missing_spectral_jacobian_q"
    sigma = sigma * max(float(sigma_scale), 1e-6)
    for i in range(n):
        meta[i]["global_sigma_scale"] = float(max(float(sigma_scale), 1e-6))
    sigma = np.maximum(sigma, 1e-12)

    diff = q0 - q_curr
    diff = _wrap_dihedral_diff(diff, active)   # wrap dihedrals to (âˆ’Ï€, Ï€]

    r_prior = diff / sigma
    J_prior = np.diag(1.0 / sigma)
    out = (J_prior, r_prior, sigma)
    return out + (meta,) if return_metadata else out
