"""
Iterative least-squares fitting of RAM-lite torsion Hamiltonian parameters.

Implements a damped Gauss-Newton loop using the finite-difference Jacobian
infrastructure from ``torsion_uncertainty``.  Does not require scipy.

Exported functions
------------------
fit_torsion_to_levels(spec, observed_rows, params, ...)
    Fit RAM-lite parameters to observed torsion-level energies (cm^-1).

fit_torsion_to_transitions(spec, observed_transitions, params, ...)
    Fit RAM-lite parameters to observed transition frequencies (cm^-1).

select_fit_params(spec, param_names)
    Build a TorsionParameter list from human-readable names.

Parameter name convention
-------------------------
  F, rho, v0, F4, F6, c_mk, c_k2
  Vcos_<n>   e.g. Vcos_3, Vcos_6
  Vsin_<n>   e.g. Vsin_3
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.torsion_average import (
    TorsionScan,
    average_torsion_scan_quantum,
    average_torsion_scan_quantum_thermal,
)
from backend.torsion_hamiltonian import TorsionHamiltonianSpec, solve_ram_lite_levels
from backend.torsion_uncertainty import (
    TorsionParameter,
    default_torsion_parameters,
    finite_difference_jacobian,
    pack_torsion_parameters,
    torsion_level_jacobian,
    torsion_level_observables,
    unpack_torsion_parameters,
)

_MHZ_PER_CM1 = 29979.2458


@dataclass
class TorsionRotationalTarget:
    """
    Observed effective rotational constant for joint torsion + rotation fitting.

    component    : 'A', 'B', or 'C'
    obs_cm1      : observed B0 value in cm^-1
    sigma_cm1    : measurement uncertainty in cm^-1
    isotopologue : label for reporting (not used in fitting logic)
    """
    component: str
    obs_cm1: float
    sigma_cm1: float = 0.05
    isotopologue: str = ""




def select_fit_params(
    spec: TorsionHamiltonianSpec,
    param_names: list[str],
) -> list[TorsionParameter]:
    """Build a TorsionParameter list from human-readable parameter names.

    Recognised names (case-insensitive):
      F, rho, v0, F4, F6, c_mk, c_k2, Vcos_<n>, Vsin_<n>

    Unrecognised names raise ValueError.
    """
    _SCALAR_MAP = {
        "f": TorsionParameter("F", ("F",), step_abs=1e-6, step_rel=1e-4),
        "rho": TorsionParameter("rho", ("rho",), step_abs=1e-7, step_rel=1e-4),
        "v0": TorsionParameter("v0", ("potential", "v0"), step_abs=1e-6, step_rel=1e-4),
        "f4": TorsionParameter("F4", ("F4",), step_abs=1e-8, step_rel=1e-4),
        "f6": TorsionParameter("F6", ("F6",), step_abs=1e-10, step_rel=1e-4),
        "c_mk": TorsionParameter("c_mk", ("c_mk",), step_abs=1e-7, step_rel=1e-4),
        "c_k2": TorsionParameter("c_k2", ("c_k2",), step_abs=1e-7, step_rel=1e-4),
        # Watson A-reduction quartic centrifugal distortion (cm-1)
        "dj": TorsionParameter("DJ", ("DJ",), step_abs=1e-9, step_rel=1e-4),
        "djk": TorsionParameter("DJK", ("DJK",), step_abs=1e-9, step_rel=1e-4),
        "dk": TorsionParameter("DK", ("DK",), step_abs=1e-9, step_rel=1e-4),
        "d1": TorsionParameter("d1", ("d1",), step_abs=1e-10, step_rel=1e-4),
        "d2": TorsionParameter("d2", ("d2",), step_abs=1e-11, step_rel=1e-4),
    }
    result: list[TorsionParameter] = []
    for raw in param_names:
        name = str(raw).strip()
        lower = name.lower()
        if lower in _SCALAR_MAP:
            result.append(_SCALAR_MAP[lower])
        elif lower.startswith("vcos_"):
            n_str = lower[5:]
            try:
                n = int(n_str)
            except ValueError:
                raise ValueError(f"Invalid Vcos parameter name '{name}'; expected 'Vcos_<int>'.")
            if n <= 0:
                raise ValueError(f"Vcos harmonic order must be positive, got '{name}'.")
            result.append(
                TorsionParameter(f"Vcos_{n}", ("potential", "vcos", str(n)), step_abs=1e-6, step_rel=1e-4)
            )
        elif lower.startswith("vsin_"):
            n_str = lower[5:]
            try:
                n = int(n_str)
            except ValueError:
                raise ValueError(f"Invalid Vsin parameter name '{name}'; expected 'Vsin_<int>'.")
            if n <= 0:
                raise ValueError(f"Vsin harmonic order must be positive, got '{name}'.")
            result.append(
                TorsionParameter(f"Vsin_{n}", ("potential", "vsin", str(n)), step_abs=1e-6, step_rel=1e-4)
            )
        else:
            raise ValueError(
                f"Unknown parameter name '{name}'. Valid: F, rho, v0, F4, F6, c_mk, c_k2, "
                f"DJ, DJK, DK, d1, d2, Vcos_<n>, Vsin_<n>."
            )
    return result


def _torsion_level_requests_from_rows(rows: list[dict]) -> list[dict]:
    """Convert user-supplied target rows to level requests for the Jacobian."""
    out = []
    for r in rows:
        out.append({"J": int(r.get("J", 0)), "K": int(r.get("K", 0)),
                    "level_index": int(r["level_index"])})
    return out


def _obs_energies_from_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(r["energy_cm-1"]) for r in rows], dtype=float)


def _obs_sigmas_from_rows(rows: list[dict], default: float = 1.0) -> np.ndarray:
    return np.asarray([float(r.get("sigma_cm-1", default)) for r in rows], dtype=float)


def _gauss_newton_step(
    J: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve (J^T W J + λI) δp = J^T W r for the parameter update."""
    WJ = np.sqrt(weights)[:, None] * J
    Wr = np.sqrt(weights) * residuals
    N = WJ.T @ WJ + damping * np.eye(J.shape[1])
    g = WJ.T @ Wr
    try:
        delta = np.linalg.solve(N, g)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(N, g, rcond=None)[0]
    return delta


def _normalise_bounds(
    params: list[TorsionParameter],
    bounds: Optional[dict | list | tuple],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(params), -np.inf, dtype=float)
    upper = np.full(len(params), np.inf, dtype=float)
    if not bounds:
        return lower, upper

    by_name = {p.name: i for i, p in enumerate(params)}
    if isinstance(bounds, dict):
        items = bounds.items()
    else:
        items = []
        for i, pair in enumerate(bounds):
            if i < len(params):
                items.append((params[i].name, pair))

    for raw_name, raw_pair in items:
        name = str(raw_name)
        if name not in by_name:
            continue
        if isinstance(raw_pair, dict):
            lo = raw_pair.get("min", raw_pair.get("lower", raw_pair.get("lo")))
            hi = raw_pair.get("max", raw_pair.get("upper", raw_pair.get("hi")))
        else:
            vals = list(raw_pair)
            lo = vals[0] if len(vals) > 0 else None
            hi = vals[1] if len(vals) > 1 else None
        idx = by_name[name]
        if lo is not None:
            lower[idx] = float(lo)
        if hi is not None:
            upper[idx] = float(hi)
        if lower[idx] > upper[idx]:
            raise ValueError(f"Invalid bounds for {name}: lower bound exceeds upper bound.")
    return lower, upper


def _normalise_priors(
    params: list[TorsionParameter],
    priors: Optional[dict],
) -> list[tuple[int, float, float]]:
    if not priors:
        return []
    by_name = {p.name: i for i, p in enumerate(params)}
    out: list[tuple[int, float, float]] = []
    for raw_name, raw_prior in priors.items():
        name = str(raw_name)
        if name not in by_name:
            continue
        if isinstance(raw_prior, dict):
            val = raw_prior.get("value", raw_prior.get("mean", raw_prior.get("target")))
            sig = raw_prior.get("sigma", raw_prior.get("std", raw_prior.get("std_err")))
        else:
            vals = list(raw_prior)
            val = vals[0] if len(vals) > 0 else None
            sig = vals[1] if len(vals) > 1 else None
        if val is None or sig is None:
            raise ValueError(f"Prior for {name} must provide value and sigma.")
        sigma = max(float(sig), 1e-12)
        out.append((by_name[name], float(val), sigma))
    return out


def _augment_with_priors(
    J_mat: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    p: np.ndarray,
    priors_norm: list[tuple[int, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not priors_norm:
        return J_mat, residuals, weights
    prior_J = np.zeros((len(priors_norm), p.size), dtype=float)
    prior_res = np.zeros(len(priors_norm), dtype=float)
    prior_weights = np.zeros(len(priors_norm), dtype=float)
    for row_idx, (param_idx, target, sigma) in enumerate(priors_norm):
        prior_J[row_idx, param_idx] = 1.0
        prior_res[row_idx] = target - float(p[param_idx])
        prior_weights[row_idx] = 1.0 / (sigma ** 2)
    return (
        np.vstack([J_mat, prior_J]),
        np.concatenate([residuals, prior_res]),
        np.concatenate([weights, prior_weights]),
    )


def _uncertainty_from_jacobian(
    J_mat: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    damping: float,
    n_obs: int,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_params == 0:
        empty = np.zeros((0, 0), dtype=float)
        return np.array([], dtype=float), empty, empty
    WJ = np.sqrt(weights)[:, None] * J_mat
    normal = WJ.T @ WJ + float(damping) * np.eye(n_params)
    try:
        inv_normal = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        inv_normal = np.linalg.pinv(normal)
    dof = max(int(n_obs) - int(n_params), 1)
    chi2 = float(np.sum(weights[:n_obs] * residuals[:n_obs] ** 2))
    scale = chi2 / dof
    covariance = inv_normal * scale
    std_err = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denom = np.outer(std_err, std_err)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(denom > 0.0, covariance / denom, 0.0)
    np.fill_diagonal(correlation, 1.0)
    return std_err, covariance, correlation


def fit_torsion_to_levels(
    spec: TorsionHamiltonianSpec,
    observed_rows: list[dict],
    params: Optional[list[TorsionParameter]] = None,
    *,
    max_iter: int = 50,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    damping: float = 1e-6,
    default_sigma_cm1: float = 1.0,
    bounds: Optional[dict | list | tuple] = None,
    priors: Optional[dict] = None,
) -> dict:
    """
    Fit RAM-lite torsion parameters to observed level energies by damped Gauss-Newton.

    Parameters
    ----------
    spec : starting TorsionHamiltonianSpec
    observed_rows : list of dicts with 'J', 'K', 'level_index', 'energy_cm-1';
                   optional 'sigma_cm-1' per row
    params : parameters to fit; defaults to F, rho, and all Vcos/Vsin harmonics
             in spec.potential
    max_iter : maximum Gauss-Newton iterations
    xtol : convergence tolerance on max |Δp / p_scale|
    ftol : convergence tolerance on RMS residual change
    damping : Levenberg-Marquardt damping added to normal matrix
    default_sigma_cm1 : default per-level uncertainty used when 'sigma_cm-1' absent

    Returns
    -------
    dict with:
      fitted_spec       : TorsionHamiltonianSpec — best-fit parameters
      param_names       : list[str]
      param_values      : np.ndarray — final parameter values
      param_values_init : np.ndarray — initial parameter values
      rms_cm-1          : float — final RMS residual
      rms_cm-1_init     : float — initial RMS residual
      n_iter            : int
      converged         : bool
      residuals_cm-1    : np.ndarray — per-level residuals at convergence
      warnings          : list[str]
    """
    if not observed_rows:
        return {
            "fitted_spec": deepcopy(spec),
            "param_names": [],
            "param_values": np.array([], dtype=float),
            "param_values_init": np.array([], dtype=float),
            "rms_cm-1": float("inf"),
            "rms_cm-1_init": float("inf"),
            "n_iter": 0,
            "converged": False,
            "residuals_cm-1": np.array([], dtype=float),
            "warnings": ["No observed rows provided; nothing to fit."],
        }

    if params is None:
        params = default_torsion_parameters(spec, include_completeness=False)

    requests = _torsion_level_requests_from_rows(observed_rows)
    obs = _obs_energies_from_rows(observed_rows)
    sigma = np.maximum(_obs_sigmas_from_rows(observed_rows, default_sigma_cm1), 1e-12)
    weights = 1.0 / (sigma ** 2)

    p0 = pack_torsion_parameters(spec, params)
    lower, upper = _normalise_bounds(params, bounds)
    priors_norm = _normalise_priors(params, priors)
    p0 = np.clip(p0, lower, upper)
    current_spec = deepcopy(spec)

    def _obs_fn(pvec: np.ndarray) -> np.ndarray:
        s = unpack_torsion_parameters(spec, params, pvec)
        return torsion_level_observables(s, requests)

    pred0 = _obs_fn(p0)
    res0 = obs - pred0
    rms0 = float(np.sqrt(np.mean(weights * res0 ** 2) / np.mean(weights)))
    rms_init = rms0

    p = p0.copy()
    rms_prev = rms0
    converged = False
    warnings: list[str] = []
    n_iter = 0

    step_abs = np.asarray([pp.step_abs for pp in params], dtype=float)
    step_rel = np.asarray([pp.step_rel for pp in params], dtype=float)

    for iteration in range(int(max_iter)):
        n_iter = iteration + 1
        J_mat, pred = finite_difference_jacobian(
            _obs_fn, p, step_abs=step_abs, step_rel=step_rel
        )
        res = obs - pred
        rms = float(np.sqrt(np.mean(weights * res ** 2) / np.mean(weights)))

        J_aug, res_aug, weights_aug = _augment_with_priors(J_mat, res, weights, p, priors_norm)
        delta = _gauss_newton_step(J_aug, res_aug, weights_aug, damping)
        p_scale = np.maximum(np.abs(p), 1.0)
        xtol_check = float(np.max(np.abs(delta) / p_scale))

        p = np.clip(p + delta, lower, upper)
        rms_change = abs(rms - rms_prev)
        rms_prev = rms

        if xtol_check < float(xtol) and rms_change < float(ftol):
            converged = True
            break

    # Final evaluation
    current_spec = unpack_torsion_parameters(spec, params, p)
    pred_final = _obs_fn(p)
    res_final = obs - pred_final
    rms_final = float(np.sqrt(np.mean(weights * res_final ** 2) / np.mean(weights)))
    J_final, _pred_check = finite_difference_jacobian(
        _obs_fn, p, step_abs=step_abs, step_rel=step_rel
    )
    J_unc, res_unc, weights_unc = _augment_with_priors(
        J_final, res_final, weights, p, priors_norm
    )
    std_err, covariance, correlation = _uncertainty_from_jacobian(
        J_unc, res_unc, weights_unc, damping, len(obs), len(params)
    )

    if not converged:
        warnings.append(
            f"Gauss-Newton did not converge after {max_iter} iterations "
            f"(final RMS={rms_final:.6f} cm^-1)."
        )

    return {
        "fitted_spec": current_spec,
        "param_names": [pp.name for pp in params],
        "param_values": p,
        "param_values_init": p0,
        "rms_cm-1": rms_final,
        "rms_cm-1_init": rms_init,
        "n_iter": n_iter,
        "converged": converged,
        "residuals_cm-1": res_final,
        "std_err": std_err,
        "covariance": covariance,
        "correlation": correlation,
        "bounds_lower": lower,
        "bounds_upper": upper,
        "priors_used": [
            {"parameter": params[i].name, "value": value, "sigma": sigma}
            for i, value, sigma in priors_norm
        ],
        "warnings": warnings,
    }


def fit_torsion_to_transitions(
    spec: TorsionHamiltonianSpec,
    observed_transitions: list[dict],
    params: Optional[list[TorsionParameter]] = None,
    *,
    max_iter: int = 50,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    damping: float = 1e-6,
    default_sigma_cm1: float = 0.1,
    bounds: Optional[dict | list | tuple] = None,
    priors: Optional[dict] = None,
) -> dict:
    """
    Fit RAM-lite torsion parameters to observed transition frequencies.

    observed_transitions : list of dicts with 'J_lo', 'K_lo', 'level_lo',
                           'J_hi', 'K_hi', 'level_hi',
                           and either 'freq_cm-1' or 'freq_mhz';
                           optional 'sigma_cm-1' per transition

    Returns the same dict shape as fit_torsion_to_levels but with
    residuals in cm^-1 over transitions rather than levels.
    """
    if not observed_transitions:
        return {
            "fitted_spec": deepcopy(spec),
            "param_names": [],
            "param_values": np.array([], dtype=float),
            "param_values_init": np.array([], dtype=float),
            "rms_cm-1": float("inf"),
            "rms_cm-1_init": float("inf"),
            "n_iter": 0,
            "converged": False,
            "residuals_cm-1": np.array([], dtype=float),
            "warnings": ["No observed transitions provided; nothing to fit."],
        }

    if params is None:
        params = default_torsion_parameters(spec, include_completeness=False)

    # Build level requests spanning both lo and hi states.
    lo_requests = [
        {"J": int(t["J_lo"]), "K": int(t["K_lo"]), "level_index": int(t["level_lo"])}
        for t in observed_transitions
    ]
    hi_requests = [
        {"J": int(t["J_hi"]), "K": int(t["K_hi"]), "level_index": int(t["level_hi"])}
        for t in observed_transitions
    ]
    all_requests = lo_requests + hi_requests
    n_trans = len(observed_transitions)

    # Resolve observed frequencies.
    obs_freq: list[float] = []
    for t in observed_transitions:
        f_cm1 = t.get("freq_cm-1")
        if f_cm1 is None:
            f_mhz = t.get("freq_mhz")
            if f_mhz is None:
                raise ValueError("Each transition must have 'freq_cm-1' or 'freq_mhz'.")
            f_cm1 = float(f_mhz) / _MHZ_PER_CM1
        obs_freq.append(float(f_cm1))
    obs = np.asarray(obs_freq, dtype=float)
    sigma = np.asarray(
        [float(t.get("sigma_cm-1", default_sigma_cm1)) for t in observed_transitions], dtype=float
    )
    sigma = np.maximum(sigma, 1e-12)
    weights = 1.0 / (sigma ** 2)

    p0 = pack_torsion_parameters(spec, params)
    lower, upper = _normalise_bounds(params, bounds)
    priors_norm = _normalise_priors(params, priors)
    p0 = np.clip(p0, lower, upper)

    def _obs_fn(pvec: np.ndarray) -> np.ndarray:
        s = unpack_torsion_parameters(spec, params, pvec)
        levels = torsion_level_observables(s, all_requests)
        lo_e = levels[:n_trans]
        hi_e = levels[n_trans:]
        return hi_e - lo_e

    pred0 = _obs_fn(p0)
    res0 = obs - pred0
    rms0 = float(np.sqrt(np.mean(weights * res0 ** 2) / np.mean(weights)))

    p = p0.copy()
    rms_prev = rms0
    converged = False
    warnings: list[str] = []
    n_iter = 0

    step_abs = np.asarray([pp.step_abs for pp in params], dtype=float)
    step_rel = np.asarray([pp.step_rel for pp in params], dtype=float)

    for iteration in range(int(max_iter)):
        n_iter = iteration + 1
        J_mat, pred = finite_difference_jacobian(
            _obs_fn, p, step_abs=step_abs, step_rel=step_rel
        )
        res = obs - pred
        rms = float(np.sqrt(np.mean(weights * res ** 2) / np.mean(weights)))
        J_aug, res_aug, weights_aug = _augment_with_priors(J_mat, res, weights, p, priors_norm)
        delta = _gauss_newton_step(J_aug, res_aug, weights_aug, damping)
        p_scale = np.maximum(np.abs(p), 1.0)
        xtol_check = float(np.max(np.abs(delta) / p_scale))
        p = np.clip(p + delta, lower, upper)
        rms_change = abs(rms - rms_prev)
        rms_prev = rms
        if xtol_check < float(xtol) and rms_change < float(ftol):
            converged = True
            break

    fitted_spec = unpack_torsion_parameters(spec, params, p)
    pred_final = _obs_fn(p)
    res_final = obs - pred_final
    rms_final = float(np.sqrt(np.mean(weights * res_final ** 2) / np.mean(weights)))
    J_final, _pred_check = finite_difference_jacobian(
        _obs_fn, p, step_abs=step_abs, step_rel=step_rel
    )
    J_unc, res_unc, weights_unc = _augment_with_priors(
        J_final, res_final, weights, p, priors_norm
    )
    std_err, covariance, correlation = _uncertainty_from_jacobian(
        J_unc, res_unc, weights_unc, damping, len(obs), len(params)
    )

    if not converged:
        warnings.append(
            f"Gauss-Newton did not converge after {max_iter} iterations "
            f"(final RMS={rms_final:.6f} cm^-1)."
        )

    return {
        "fitted_spec": fitted_spec,
        "param_names": [pp.name for pp in params],
        "param_values": p,
        "param_values_init": p0,
        "rms_cm-1": rms_final,
        "rms_cm-1_init": rms0,
        "n_iter": n_iter,
        "converged": converged,
        "residuals_cm-1": res_final,
        "std_err": std_err,
        "covariance": covariance,
        "correlation": correlation,
        "bounds_lower": lower,
        "bounds_upper": upper,
        "priors_used": [
            {"parameter": params[i].name, "value": value, "sigma": sigma}
            for i, value, sigma in priors_norm
        ],
        "warnings": warnings,
    }


def fit_torsion_joint(
    spec,
    level_rows: list[dict],
    rotational_targets: list["TorsionRotationalTarget"],
    scan: "TorsionScan",
    elements,
    masses=None,
    params: list | None = None,
    *,
    sigma_level_cm1: float = 0.1,
    sigma_rot_cm1: float = 0.05,
    temperature_K: float = 298.15,
    use_thermal: bool = False,
    max_states: int = 6,
    max_iter: int = 50,
    damping: float = 1e-4,
    step_abs: float = 1e-6,
    step_rel: float = 1e-4,
    xtol: float = 1e-8,
    ftol: float = 1e-10,
    bounds: Optional[dict | list | tuple] = None,
    priors: Optional[dict] = None,
) -> dict:
    """
    Joint Gauss-Newton fit of torsion parameters to both level energies and
    torsion-averaged rotational constants.

    Fits F, rho, and/or Vcos_n simultaneously against:
      - observed torsional level energies / transition frequencies (``level_rows``)
      - observed ground-state rotational constants A/B/C (``rotational_targets``)

    The torsional Hamiltonian spec is used directly for both level prediction and
    quantum probability-density averaging, keeping both data streams in sync.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    level_rows : list of dicts with 'obs_cm1' and optionally 'sigma_cm1', 'type',
        'J', 'symmetry' — same format as fit_torsion_to_levels / fit_torsion_to_transitions
    rotational_targets : list of TorsionRotationalTarget
    scan : TorsionScan — torsion grid used for quantum averaging
    elements : sequence of element symbols
    masses : optional explicit masses (defaults to built-in element masses)
    sigma_level_cm1 : default uncertainty on torsional levels [cm^-1]
    sigma_rot_cm1 : default uncertainty on rotational constants [cm^-1]
    temperature_K : temperature for quantum-thermal averaging (if use_thermal=True)
    use_thermal : if True use quantum_thermal averaging; otherwise ground-state quantum
    max_states : max torsional states for thermal averaging
    max_iter, damping, step_abs, step_rel, xtol, ftol : Gauss-Newton controls
    priors : optional list of prior dicts (same format as fit_torsion_to_levels)

    Returns
    -------
    dict with all standard fitting keys plus:
      rms_level_cm-1 : RMS on torsional levels only
      rms_rot_cm-1 : RMS on rotational constants only
      n_level_obs : number of level observations
      n_rot_obs : number of rotational observations
    """
    from backend.torsion_average import average_torsion_scan_quantum_thermal

    warnings: list[str] = []

    if params is None:
        params = select_fit_params(spec)
    if not params:
        warnings.append("No free parameters selected for joint fitting.")

    # ── Build level observations ────────────────────────────────────────────
    _component_map = {"A": 0, "B": 1, "C": 2}
    level_requests = _torsion_level_requests_from_rows(level_rows) if level_rows else []
    level_obs = _obs_energies_from_rows(level_rows) if level_rows else np.array([], dtype=float)
    level_sigma = np.maximum(
        _obs_sigmas_from_rows(level_rows, sigma_level_cm1) if level_rows else np.array([], dtype=float),
        1e-12,
    )
    level_weights = 1.0 / level_sigma ** 2 if level_sigma.size > 0 else np.array([], dtype=float)

    # ── Build rotational observations ───────────────────────────────────────
    rot_comp_indices = []
    rot_obs = []
    rot_sigma = []
    for rt in rotational_targets:
        ci = _component_map.get(str(rt.component).upper())
        if ci is None:
            raise ValueError(
                f"TorsionRotationalTarget.component must be 'A', 'B', or 'C'; got {rt.component!r}"
            )
        rot_comp_indices.append(ci)
        rot_obs.append(float(rt.obs_cm1))
        s = float(rt.sigma_cm1) if float(rt.sigma_cm1) > 0 else sigma_rot_cm1
        rot_sigma.append(s)
    rot_obs = np.asarray(rot_obs, dtype=float)
    rot_sigma = np.asarray(rot_sigma, dtype=float)
    rot_weights = 1.0 / rot_sigma ** 2 if rot_sigma.size > 0 else np.array([], dtype=float)

    # ── Pack initial parameters ─────────────────────────────────────────────
    p0 = pack_torsion_parameters(spec, params)
    lower, upper = _normalise_bounds(params, bounds)
    priors_norm = _normalise_priors(params, priors)

    n_level = len(level_requests)
    n_rot = len(rot_obs)

    def _obs_fn(pvec: np.ndarray):
        s_cur = unpack_torsion_parameters(spec, params, pvec)
        level_pred = np.array([], dtype=float)
        if n_level > 0:
            level_pred = torsion_level_observables(s_cur, level_requests)
        rot_pred = np.array([], dtype=float)
        if n_rot > 0:
            if use_thermal:
                avg_out = average_torsion_scan_quantum_thermal(
                    elements,
                    scan,
                    s_cur,
                    masses=masses,
                    temperature_K=float(temperature_K),
                    max_states=int(max_states),
                )
            else:
                from backend.torsion_average import average_torsion_scan_quantum
                avg_out = average_torsion_scan_quantum(
                    elements,
                    scan,
                    s_cur,
                    masses=masses,
                    state_index=0,
                )
            avg_abc = np.asarray(avg_out["averaged_constants"], dtype=float)
            rot_pred = avg_abc[rot_comp_indices]
        return np.concatenate([level_pred, rot_pred])

    obs_full = np.concatenate([level_obs, rot_obs])
    weights_full = np.concatenate([level_weights, rot_weights]) if n_rot > 0 else level_weights

    p = p0.copy()
    J0, pred0 = finite_difference_jacobian(_obs_fn, p, step_abs=step_abs, step_rel=step_rel)
    res0 = obs_full - pred0
    rms0 = float(np.sqrt(np.mean(weights_full * res0 ** 2) / np.mean(weights_full)))

    converged = False
    n_iter = 0
    rms_final = rms0

    for n_iter in range(1, int(max_iter) + 1):
        J_aug, res_aug, w_aug = _augment_with_priors(J0, res0, weights_full, p, priors_norm)
        dp = _gauss_newton_step(J_aug, res_aug, w_aug, damping=damping)
        p_new = np.clip(p + dp, lower, upper)
        J_new, pred_new = finite_difference_jacobian(
            _obs_fn, p_new, step_abs=step_abs, step_rel=step_rel
        )
        res_new = obs_full - pred_new
        rms_new = float(np.sqrt(np.mean(weights_full * res_new ** 2) / np.mean(weights_full)))

        xtol_check = float(np.max(np.abs(p_new - p) / (np.abs(p) + 1e-12)))
        rms_change = abs(rms_new - rms_final)
        p, J0, res0, rms_final = p_new, J_new, res_new, rms_new

        if xtol_check < float(xtol) and rms_change < float(ftol):
            converged = True
            break

    fitted_spec = unpack_torsion_parameters(spec, params, p)
    pred_final = _obs_fn(p)

    # ── Split residuals for per-stream RMS ─────────────────────────────────
    res_level = (obs_full[:n_level] - pred_final[:n_level]) if n_level > 0 else np.array([], dtype=float)
    res_rot = (obs_full[n_level:] - pred_final[n_level:]) if n_rot > 0 else np.array([], dtype=float)
    rms_level = float(np.sqrt(np.mean(res_level ** 2))) if res_level.size > 0 else float("nan")
    rms_rot = float(np.sqrt(np.mean(res_rot ** 2))) if res_rot.size > 0 else float("nan")

    res_final_full = obs_full - pred_final
    rms_final = float(np.sqrt(np.mean(weights_full * res_final_full ** 2) / np.mean(weights_full)))

    J_final, _ = finite_difference_jacobian(_obs_fn, p, step_abs=step_abs, step_rel=step_rel)
    J_unc, res_unc, weights_unc = _augment_with_priors(
        J_final, res_final_full, weights_full, p, priors_norm
    )
    std_err, covariance, correlation = _uncertainty_from_jacobian(
        J_unc, res_unc, weights_unc, damping, len(obs_full), len(params)
    )

    if not converged:
        warnings.append(
            f"Joint Gauss-Newton did not converge after {max_iter} iterations "
            f"(final RMS={rms_final:.6f} cm^-1)."
        )

    return {
        "fitted_spec": fitted_spec,
        "param_names": [pp.name for pp in params],
        "param_values": p,
        "param_values_init": p0,
        "rms_cm-1": rms_final,
        "rms_cm-1_init": rms0,
        "rms_level_cm-1": rms_level,
        "rms_rot_cm-1": rms_rot,
        "n_level_obs": n_level,
        "n_rot_obs": n_rot,
        "n_iter": n_iter,
        "converged": converged,
        "residuals_cm-1": res_final_full,
        "std_err": std_err,
        "covariance": covariance,
        "correlation": correlation,
        "bounds_lower": lower,
        "bounds_upper": upper,
        "priors_used": [
            {"parameter": params[i].name, "value": value, "sigma": sigma}
            for i, value, sigma in priors_norm
        ],
        "warnings": warnings,
    }
