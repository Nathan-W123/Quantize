"""
Uncertainty utilities for torsion Hamiltonian parameters.

This module is intentionally independent of external QC dependencies and operates
on torsion model outputs only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from backend.torsion_hamiltonian import TorsionHamiltonianSpec, solve_ram_lite_levels


@dataclass(frozen=True)
class TorsionParameter:
    """Parameterized entry used for packing/unpacking torsion fit vectors."""

    name: str
    path: tuple[str, ...]
    step_abs: float = 1e-6
    step_rel: float = 1e-4


def default_torsion_parameters(
    spec: TorsionHamiltonianSpec,
    *,
    include_completeness: bool = False,
    potential_vcos: Optional[list[int]] = None,
    potential_vsin: Optional[list[int]] = None,
) -> list[TorsionParameter]:
    """
    Build default torsion parameter list for uncertainty propagation.

    Always includes ``F`` and ``rho``. Potential harmonics are optional; if not
    provided, keys present in ``spec.potential`` are used. Completeness terms are
    included only when requested.
    """
    params: list[TorsionParameter] = [
        TorsionParameter("F", ("F",), step_abs=1e-6, step_rel=1e-4),
        TorsionParameter("rho", ("rho",), step_abs=1e-7, step_rel=1e-4),
    ]

    vcos_keys = sorted({int(k) for k in (potential_vcos or list(spec.potential.vcos.keys())) if int(k) > 0})
    vsin_keys = sorted({int(k) for k in (potential_vsin or list(spec.potential.vsin.keys())) if int(k) > 0})

    for k in vcos_keys:
        params.append(TorsionParameter(f"Vcos_{k}", ("potential", "vcos", str(k)), step_abs=1e-6, step_rel=1e-4))
    for k in vsin_keys:
        params.append(TorsionParameter(f"Vsin_{k}", ("potential", "vsin", str(k)), step_abs=1e-6, step_rel=1e-4))

    if include_completeness:
        params.extend(
            [
                TorsionParameter("F4", ("F4",), step_abs=1e-8, step_rel=1e-4),
                TorsionParameter("F6", ("F6",), step_abs=1e-10, step_rel=1e-4),
                TorsionParameter("c_mk", ("c_mk",), step_abs=1e-7, step_rel=1e-4),
                TorsionParameter("c_k2", ("c_k2",), step_abs=1e-7, step_rel=1e-4),
            ]
        )
    return params


def _spec_to_nested_dict(spec: TorsionHamiltonianSpec) -> dict:
    return {
        "F": float(spec.F),
        "rho": float(spec.rho),
        "F4": float(spec.F4),
        "F6": float(spec.F6),
        "c_mk": float(spec.c_mk),
        "c_k2": float(spec.c_k2),
        "A": float(spec.A),
        "B": float(spec.B),
        "C": float(spec.C),
        "n_basis": int(spec.n_basis),
        "units": str(spec.units),
        "potential": {
            "v0": float(spec.potential.v0),
            "vcos": {str(int(k)): float(v) for k, v in spec.potential.vcos.items()},
            "vsin": {str(int(k)): float(v) for k, v in spec.potential.vsin.items()},
            "units": str(spec.potential.units),
        },
    }


def _nested_get(d: dict, path: tuple[str, ...], default: float = 0.0) -> float:
    cur = d
    for p in path[:-1]:
        cur = cur[p]
    return float(cur.get(path[-1], default)) if isinstance(cur, dict) else float(cur[path[-1]])


def _nested_set(d: dict, path: tuple[str, ...], value: float) -> None:
    cur = d
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = float(value)


def pack_torsion_parameters(spec: TorsionHamiltonianSpec, params: list[TorsionParameter]) -> np.ndarray:
    base = _spec_to_nested_dict(spec)
    return np.asarray([_nested_get(base, p.path) for p in params], dtype=float)


def unpack_torsion_parameters(
    template_spec: TorsionHamiltonianSpec,
    params: list[TorsionParameter],
    values: np.ndarray,
) -> TorsionHamiltonianSpec:
    d = _spec_to_nested_dict(template_spec)
    vec = np.asarray(values, dtype=float).ravel()
    if vec.size != len(params):
        raise ValueError("Parameter vector size does not match parameter schema.")
    for p, v in zip(params, vec):
        _nested_set(d, p.path, float(v))

    updated = deepcopy(template_spec)
    updated.F = float(d["F"])
    updated.rho = float(d["rho"])
    updated.F4 = float(d["F4"])
    updated.F6 = float(d["F6"])
    updated.c_mk = float(d["c_mk"])
    updated.c_k2 = float(d["c_k2"])
    updated.A = float(d["A"])
    updated.B = float(d["B"])
    updated.C = float(d["C"])
    updated.n_basis = int(d["n_basis"])
    updated.units = str(d["units"])
    updated.potential.v0 = float(d["potential"]["v0"])
    updated.potential.vcos = {int(k): float(v) for k, v in d["potential"]["vcos"].items()}
    updated.potential.vsin = {int(k): float(v) for k, v in d["potential"]["vsin"].items()}
    updated.potential.units = str(d["potential"]["units"])
    return updated


def finite_difference_jacobian(
    observable_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    step_abs: float | np.ndarray = 1e-6,
    step_rel: float | np.ndarray = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobian for arbitrary vector observable."""
    x = np.asarray(x0, dtype=float).ravel()
    y0 = np.asarray(observable_fn(x), dtype=float).ravel()
    m = y0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)

    abs_arr = np.broadcast_to(np.asarray(step_abs, dtype=float), (n,))
    rel_arr = np.broadcast_to(np.asarray(step_rel, dtype=float), (n,))

    for j in range(n):
        h = max(float(abs_arr[j]), float(rel_arr[j]) * max(1.0, abs(float(x[j]))))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        yp = np.asarray(observable_fn(xp), dtype=float).ravel()
        ym = np.asarray(observable_fn(xm), dtype=float).ravel()
        if yp.size != m or ym.size != m:
            raise ValueError("Observable function changed output size across perturbations.")
        J[:, j] = (yp - ym) / (2.0 * h)
    return J, y0


def torsion_level_observables(
    spec: TorsionHamiltonianSpec,
    requests: list[dict],
) -> np.ndarray:
    """
    Return vector of torsion level energies for requested (J, K, level_index) rows.
    """
    out = np.zeros(len(requests), dtype=float)
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, req in enumerate(requests):
        J = int(req.get("J", 0))
        K = int(req.get("K", 0))
        li = int(req["level_index"])
        grouped.setdefault((J, K), []).append((i, li))

    for (J, K), pairs in grouped.items():
        need = max(li for _, li in pairs) + 1
        levels = solve_ram_lite_levels(spec, J=J, K=K, n_levels=need)["energies_cm-1"]
        for i, li in pairs:
            out[i] = float(levels[li])
    return out


def torsion_level_jacobian(
    spec: TorsionHamiltonianSpec,
    requests: list[dict],
    params: list[TorsionParameter],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobian of requested torsion levels wrt selected parameters.
    """
    p0 = pack_torsion_parameters(spec, params)
    step_abs = np.asarray([p.step_abs for p in params], dtype=float)
    step_rel = np.asarray([p.step_rel for p in params], dtype=float)

    def _obs(pvec: np.ndarray) -> np.ndarray:
        s = unpack_torsion_parameters(spec, params, pvec)
        return torsion_level_observables(s, requests)

    J, y0 = finite_difference_jacobian(_obs, p0, step_abs=step_abs, step_rel=step_rel)
    return J, y0, p0


def covariance_from_normal_matrix(
    J: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    damping: float = 1e-8,
    regularization_diag: Optional[np.ndarray] = None,
    rank_tol: float = 1e-10,
) -> dict:
    """
    Estimate covariance from weighted normal matrix with safeguards.

    Computes N = J^T W J + damping*I + diag(regularization_diag).
    """
    A = np.asarray(J, dtype=float)
    if A.ndim != 2:
        raise ValueError("J must be 2D.")
    m, n = A.shape
    warnings: list[str] = []

    if weights is None:
        WJ = A
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != m:
            raise ValueError("weights size must equal number of Jacobian rows.")
        if np.any(w < 0.0):
            warnings.append("Negative weights detected; covariance may be invalid.")
        WJ = np.sqrt(np.maximum(w, 0.0))[:, None] * A

    N = WJ.T @ WJ
    if regularization_diag is not None:
        rdiag = np.asarray(regularization_diag, dtype=float).ravel()
        if rdiag.size != n:
            raise ValueError("regularization_diag size must match number of parameters.")
        N = N + np.diag(np.maximum(rdiag, 0.0))
    if damping > 0.0:
        N = N + float(damping) * np.eye(n)

    N = 0.5 * (N + N.T)
    evals, evecs = np.linalg.eigh(N)
    eval_max = float(np.max(evals)) if evals.size else 0.0
    thresh = max(float(rank_tol) * max(eval_max, 1.0), 1e-16)
    rank = int(np.sum(evals > thresh))
    if rank < n:
        warnings.append(
            f"Normal matrix is rank-deficient or ill-conditioned (rank={rank}/{n}). "
            "Reported covariance uses pseudo-inverse in deficient directions."
        )

    inv_evals = np.zeros_like(evals)
    mask = evals > thresh
    inv_evals[mask] = 1.0 / evals[mask]
    cov = (evecs * inv_evals) @ evecs.T
    cov = 0.5 * (cov + cov.T)

    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "covariance": cov,
        "std_err": std_err,
        "normal_matrix": N,
        "eigenvalues": evals,
        "rank": rank,
        "warnings": warnings,
    }


def covariance_from_matched_level_residuals(
    spec: TorsionHamiltonianSpec,
    matched_rows: list[dict],
    params: list[TorsionParameter],
    *,
    damping: float = 1e-8,
    rank_tol: float = 1e-10,
    default_sigma_cm1: float = 1.0,
) -> dict:
    """
    Convenience helper for matched torsion level residual models.

    ``matched_rows`` expects dictionaries with:
      - ``J``, ``K``, ``level_index``
      - optional ``sigma_cm-1`` uncertainty per row
    """
    requests = [{"J": r["J"], "K": r["K"], "level_index": r["level_index"]} for r in matched_rows]
    J, y_pred, p0 = torsion_level_jacobian(spec, requests, params)

    sigma = np.asarray([float(r.get("sigma_cm-1", default_sigma_cm1)) for r in matched_rows], dtype=float)
    sigma = np.maximum(sigma, 1e-12)
    weights = 1.0 / (sigma**2)

    cov_out = covariance_from_normal_matrix(J, weights=weights, damping=damping, rank_tol=rank_tol)
    cov_out.update({
        "jacobian": J,
        "predicted": y_pred,
        "param0": p0,
        "param_names": [p.name for p in params],
    })
    return cov_out
