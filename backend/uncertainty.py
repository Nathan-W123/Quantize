"""
Uncertainty quantification for internal-coordinate structure refinement.

Computes parameter covariance and confidence intervals from the internal-coordinate
Jacobian and the prior regularisation.

Main entry point:
    compute_uncertainty(Jq, weights, sigma_prior, lambda_reg)
    → covariance matrix, standard errors, 95% confidence intervals
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_uncertainty(
    Jq: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma_prior: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
):
    """
    Parameter covariance and confidence intervals for internal coordinates.

    Implements:
        Cq = (Jq^T W Jq + λ Σ_prior^{-1})^{-1}

    Parameters
    ----------
    Jq         : (m, n_q)   Weighted internal-coordinate Jacobian (spectral rows only).
                            Each row is already divided by its observational sigma.
    weights    : (m,) or None
                            Per-observation weights.  If None, uniform weight 1.
    sigma_prior : (n_q,) or None
                            Prior standard deviations in natural units (Å, rad).
                            If None, no prior regularisation is applied beyond lambda_reg.
    lambda_reg : float      Additional Tikhonov regularisation added to the diagonal.
                            Prevents blow-up for unidentifiable parameters.

    Returns
    -------
    cov        : (n_q, n_q)   Posterior covariance matrix.
    std_err    : (n_q,)       Standard errors = sqrt(diag(cov)).
    ci_95      : (n_q, 2)     95% confidence intervals [q - 1.96*se, q + 1.96*se].
                              The caller must add these to the q values to get intervals.
    """
    Jq = np.asarray(Jq, dtype=float)
    m, n_q = Jq.shape

    if weights is not None:
        W = np.asarray(weights, dtype=float)
        JtWJ = Jq.T @ (W[:, None] * Jq)
    else:
        JtWJ = Jq.T @ Jq

    reg = lambda_reg * np.eye(n_q)
    if sigma_prior is not None:
        sp = np.asarray(sigma_prior, dtype=float)
        sp = np.maximum(sp, 1e-12)
        reg += np.diag(1.0 / sp ** 2)

    A = JtWJ + reg
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)

    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    half_width = 1.96 * std_err
    ci_95 = np.column_stack([-half_width, half_width])
    return cov, std_err, ci_95


def uncertainty_table(
    coord_set,
    coords,
    Jq: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma_prior: Optional[np.ndarray] = None,
    lambda_reg: float = 1e-6,
) -> list[dict]:
    """
    Build a human-readable uncertainty table for all active internal coordinates.

    Parameters
    ----------
    coord_set  : InternalCoordinateSet
    coords     : (N, 3)  Current geometry.
    Jq         : (m, n_q)  Internal Jacobian (already weighted).
    weights    : (m,) or None
    sigma_prior : (n_q,) or None
    lambda_reg : float

    Returns
    -------
    rows : list of dict with keys:
        name, value, value_unit, std_err, std_err_unit, ci_lo, ci_hi, ci_unit
    """
    from backend.internal_fit import InternalCoordinateSet  # avoid circular at module level

    cov, std_err, ci_95 = compute_uncertainty(Jq, weights, sigma_prior, lambda_reg)
    q_vals = coord_set.active_values(coords)
    active = coord_set.active_coords()

    rows = []
    for i, (ic, q, se, ci) in enumerate(zip(active, q_vals, std_err, ci_95)):
        if ic.kind == "bond":
            val = q
            val_u = "Å"
            se_u = "Å"
        else:
            val = np.degrees(q)
            se = np.degrees(se)
            ci = np.degrees(ci)
            val_u = "deg"
            se_u = "deg"
        rows.append({
            "name": ic.name,
            "value": float(val),
            "value_unit": val_u,
            "std_err": float(se),
            "std_err_unit": se_u,
            "ci_lo": float(val + ci[0]),
            "ci_hi": float(val + ci[1]),
            "ci_unit": val_u,
        })
    return rows


def print_uncertainty_table(rows: list[dict]) -> None:
    """Pretty-print the uncertainty table returned by uncertainty_table()."""
    header = f"{'Coordinate':<28}  {'Value':>10}  {'±1σ':>8}  {'95% CI':>20}  {'Unit'}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        ci_str = f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]"
        print(
            f"{r['name']:<28}  {r['value']:>10.6f}  {r['std_err']:>8.6f}  {ci_str:>20}  {r['value_unit']}"
        )
