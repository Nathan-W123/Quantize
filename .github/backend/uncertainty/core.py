"""
Uncertainty quantification for internal-coordinate structure refinement.

Computes parameter covariance and confidence intervals from the internal-coordinate
Jacobian and the prior regularisation.

Main entry point:
    compute_uncertainty(Jq, weights, sigma_prior, lambda_reg)
    â†’ covariance matrix, standard errors, 95% confidence intervals
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def compute_uncertainty(
    Jq: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma_prior: Optional[np.ndarray] = None,
    lambda_reg: float = 0.0,
    residual_w: Optional[np.ndarray] = None,
    chi2_inflate: bool = True,
):
    """
    Parameter covariance and confidence intervals for internal coordinates.

    Implements:
        Cq = (Jq^T W Jq + Î» Î£_prior^{-1})^{-1}

    When residual_w is supplied and chi2_inflate=True the covariance is
    inflated by the reduced chiÂ² (sÂ² = Ï‡Â²/dof) whenever sÂ² > 1, so that
    confidence intervals reflect actual fit quality rather than nominal Ïƒ.
    This prevents overoptimistic CIs when residuals exceed their stated Ïƒ.

    Parameters
    ----------
    Jq         : (m, n_q)   Weighted internal-coordinate Jacobian (spectral rows only).
                            Each row is already divided by its observational sigma.
    weights    : (m,) or None
                            Per-observation weights.  If None, uniform weight 1.
    sigma_prior : (n_q,) or None
                            Prior standard deviations in natural units (Ã…, rad).
                            If None, no prior regularisation is applied beyond lambda_reg.
    lambda_reg : float      Additional Tikhonov regularisation added to the diagonal.
                            Prevents blow-up for unidentifiable parameters.
    residual_w : (m,) or None
                            Pre-weighted residuals (observed âˆ’ calculated, each divided
                            by its observational sigma).  Required for chiÂ²-inflation.
                            If None, no chiÂ²-inflation is applied.
    chi2_inflate : bool     If True (default) and residual_w is not None, inflate the
                            covariance by sÂ² = Ï‡Â²/dof when sÂ² > 1.

    Returns
    -------
    cov        : (n_q, n_q)   Posterior covariance matrix (chiÂ²-inflated if applicable).
    std_err    : (n_q,)       Standard errors = sqrt(diag(cov)).
    ci_95      : (n_q, 2)     95% confidence intervals [q - 1.96*se, q + 1.96*se].
                              The caller must add these to the q values to get intervals.
    chi2_scale : float        Inflation factor applied (1.0 if none applied).
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

    # ChiÂ²-scaled uncertainty inflation.
    # s^2 = chi2_weighted / dof; rescale the covariance to match the residuals.
    #
    # Two-sided. Inflating only when chi2 > 1 assumes the stated sigmas can be
    # optimistic but never conservative, and a sigma that is too large is just
    # as wrong: it makes the fit under-use its data and quote intervals wider
    # than the residuals justify. Measured on this benchmark, chi2/dof came out
    # near 0.05, so the one-sided test never fired on the failure that was
    # actually present.
    chi2_scale = 1.0
    if chi2_inflate and residual_w is not None:
        rw = np.asarray(residual_w, dtype=float)
        dof = max(1, int(rw.size) - n_q)
        chi2_red = float(np.dot(rw, rw)) / dof
        if np.isfinite(chi2_red) and chi2_red > 0.0:
            chi2_scale = chi2_red
            cov = cov * chi2_scale

    std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    half_width = 1.96 * std_err
    ci_95 = np.column_stack([-half_width, half_width])
    return cov, std_err, ci_95, chi2_scale


def uncertainty_table(
    coord_set,
    coords,
    Jq: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma_prior: Optional[np.ndarray] = None,
    lambda_reg: float = 1e-6,
    dominance_labels: Optional[dict[str, str]] = None,
    sensitivity_rows: Optional[list[dict]] = None,
    residual_w: Optional[np.ndarray] = None,
    chi2_inflate: bool = True,
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
    residual_w : (m,) or None
                 Pre-weighted residuals; enables chiÂ²-inflation when supplied.
    chi2_inflate : bool  Passed to compute_uncertainty.

    Returns
    -------
    rows : list of dict with keys:
        name, value, value_unit, std_err, std_err_unit, ci_lo, ci_hi, ci_unit,
        chi2_scale (float â€” inflation factor, 1.0 when no inflation applied)
    """
    from backend.internal.internal_fit import InternalCoordinateSet  # avoid circular at module level

    cov, std_err, ci_95, chi2_scale = compute_uncertainty(
        Jq, weights, sigma_prior, lambda_reg,
        residual_w=residual_w, chi2_inflate=chi2_inflate,
    )
    q_vals = coord_set.active_values(coords)
    active = coord_set.active_coords()

    rows = []
    sens_map = {str(r.get("name")): r for r in (sensitivity_rows or [])}
    for i, (ic, q, se, ci) in enumerate(zip(active, q_vals, std_err, ci_95)):
        if ic.kind == "bond":
            val = q
            val_u = "Ã…"
            se_u = "Ã…"
        else:
            val = np.degrees(q)
            se = np.degrees(se)
            ci = np.degrees(ci)
            val_u = "deg"
            se_u = "deg"
        row = {
            "name": ic.name,
            "value": float(val),
            "value_unit": val_u,
            "std_err": float(se),
            "std_err_unit": se_u,
            "ci_lo": float(val + ci[0]),
            "ci_hi": float(val + ci[1]),
            "ci_unit": val_u,
            "prior_dominance": (dominance_labels or {}).get(ic.name, ""),
            "prior_sensitivity": sens_map.get(ic.name, {}).get("sensitivity_label", ""),
            "prior_delta": float(sens_map.get(ic.name, {}).get("delta", np.nan))
            if ic.name in sens_map
            else np.nan,
            "prior_delta_unit": sens_map.get(ic.name, {}).get("unit", ""),
            "chi2_scale": float(chi2_scale),
        }
        rows.append(row)
    if chi2_scale > 1.0:
        print(
            f"  [Uncertainty] ChiÂ²-inflation applied: sÂ²={chi2_scale:.3f} "
            f"(residuals exceed stated Ïƒ; CIs inflated by Ã—{chi2_scale**0.5:.3f})"
        )
    return rows


def print_uncertainty_table(rows: list[dict]) -> None:
    """Pretty-print the uncertainty table returned by uncertainty_table()."""
    header = f"{'Coordinate':<28}  {'Value':>10}  {'Â±1Ïƒ':>8}  {'95% CI':>20}  {'Unit'}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        ci_str = f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]"
        dom = r.get("prior_dominance", "")
        sens = r.get("prior_sensitivity", "")
        extra = f"  {dom}"
        if sens:
            extra += f" | {sens}"
        print(f"{r['name']:<28}  {r['value']:>10.6f}  {r['std_err']:>8.6f}  {ci_str:>20}  {r['value_unit']}{extra}")
