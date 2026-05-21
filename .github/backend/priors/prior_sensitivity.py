from __future__ import annotations

from typing import Any

import numpy as np

from backend.internal.internal_fit import build_internal_priors


def classify_prior_dominance(
    j_spectral_q: np.ndarray,
    sigma_prior: np.ndarray | None,
    names: list[str],
) -> dict[str, str]:
    """
    Label each coordinate as data-constrained / prior-dominated / mixed / unidentifiable.
    """
    j = np.asarray(j_spectral_q, dtype=float)
    n_q = j.shape[1] if j.ndim == 2 else 0
    if n_q == 0:
        return {}
    info_data = np.sum(j * j, axis=0)
    if sigma_prior is None:
        sigma = np.full(n_q, np.inf, dtype=float)
    else:
        sigma = np.asarray(sigma_prior, dtype=float)
        if sigma.shape[0] != n_q:
            sigma = np.resize(sigma, n_q)
    info_prior = np.where(np.isfinite(sigma), 1.0 / np.maximum(sigma, 1e-12) ** 2, 0.0)

    out: dict[str, str] = {}
    for i, nm in enumerate(names):
        idat = float(info_data[i])
        ipri = float(info_prior[i])
        total = idat + ipri
        if total <= 1e-16:
            out[nm] = "unidentifiable"
            continue
        frac_data = idat / total
        if frac_data >= 0.8:
            out[nm] = "data-constrained"
        elif frac_data <= 0.2:
            out[nm] = "prior-dominated"
        else:
            out[nm] = "mixed information"
    return out


def _local_map_q(
    q_curr: np.ndarray,
    q0: np.ndarray,
    sigma: np.ndarray,
    j_spectral_q: np.ndarray,
    residual_w: np.ndarray,
    lambda_reg: float = 1e-6,
) -> np.ndarray:
    """
    Local linearized MAP estimate in q-space around current geometry.
    """
    j = np.asarray(j_spectral_q, dtype=float)
    r = np.asarray(residual_w, dtype=float)
    q = np.asarray(q_curr, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    n = q.shape[0]

    a = j.T @ j + np.diag(1.0 / (s * s)) + float(lambda_reg) * np.eye(n)
    b = j.T @ r + (q0 - q) / (s * s)
    try:
        dq = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        dq = np.linalg.pinv(a) @ b
    return q + dq


def prior_sensitivity_analysis(
    coord_set,
    coords: np.ndarray,
    j_spectral_q: np.ndarray,
    residual_w: np.ndarray,
    *,
    sigma_bond: float,
    sigma_angle_deg: float,
    sigma_dihedral_deg: float,
    prior_mode: str,
    prior_specs: list[dict[str, Any]] | None,
    elems: list[str],
    freeze_sigma_floor: float,
    spectral_jacobian_q: np.ndarray | None,
    adaptive_config: dict[str, Any] | None,
    sv_rel_threshold: float,
    lambda_reg: float = 1e-6,
    scales: tuple[float, float, float] = (0.5, 1.0, 2.0),
) -> list[dict[str, Any]]:
    """
    Evaluate q-space local prior sensitivity by scaling prior sigma.
    """
    q_curr = coord_set.active_values(coords)
    names = coord_set.active_names()
    q_by_scale: dict[float, np.ndarray] = {}
    for s in scales:
        _, _, sigma_s, meta = build_internal_priors(
            coord_set,
            coords,
            sigma_bond=sigma_bond,
            sigma_angle_deg=sigma_angle_deg,
            sigma_dihedral_deg=sigma_dihedral_deg,
            prior_values=None,
            prior_mode=prior_mode,
            prior_specs=prior_specs,
            elems=elems,
            freeze_sigma_floor=freeze_sigma_floor,
            spectral_jacobian_q=spectral_jacobian_q,
            adaptive_config=adaptive_config,
            sv_rel_threshold=sv_rel_threshold,
            sigma_scale=float(s),
            return_metadata=True,
        )
        q0 = np.array([float(m["target_value"]) for m in meta], dtype=float)
        q_by_scale[float(s)] = _local_map_q(
            q_curr=q_curr,
            q0=q0,
            sigma=sigma_s,
            j_spectral_q=j_spectral_q,
            residual_w=residual_w,
            lambda_reg=lambda_reg,
        )

    rows: list[dict[str, Any]] = []
    for i, nm in enumerate(names):
        vals = np.array([q_by_scale[s][i] for s in scales], dtype=float)
        ic = coord_set.active_coords()[i]
        if ic.kind == "bond":
            vals_u = vals
            unit = "A"
            d = float(np.max(vals_u) - np.min(vals_u))
            mild = 0.005
            strong = 0.02
        else:
            vals_u = np.degrees(vals)
            unit = "deg"
            d = float(np.max(vals_u) - np.min(vals_u))
            mild = 0.5
            strong = 2.0
        if d < mild:
            label = "stable"
        elif d < strong:
            label = "mildly prior-sensitive"
        else:
            label = "strongly prior-sensitive"
        rows.append(
            {
                "name": nm,
                "delta": d,
                "unit": unit,
                "sensitivity_label": label,
                "q_0p5x": float(vals_u[0]),
                "q_1p0x": float(vals_u[1]),
                "q_2p0x": float(vals_u[2]),
            }
        )
    return rows

