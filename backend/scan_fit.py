"""
Torsion scan Fourier potential fitting and coverage diagnostics.

Potential convention (TorsionFourierPotential / direct Fourier form):
    V(phi) = v0 + sum_n [ vcos_n * cos(n*phi) + vsin_n * sin(n*phi) ]

hindered_rotor.py uses a different RAM-style (1-cos) convention:
    V(phi) = sum_n (V_n / 2) * (1 - cos(n*phi))
    mapping: v0 = sum_n V_n/2, vcos_n = -V_n/2.

scan_fit.py always works in the direct Fourier form to produce coefficients
that can be passed directly to TorsionFourierPotential.

Functions
---------
energies_to_cm1         : convert energies from any supported unit to cm^-1
validate_scan_coverage  : check phi coverage, spacing, duplicates
fit_fourier_potential   : linear least-squares Fourier fit to scan energies
scan_to_torsion_potential: convenience wrapper with symmetry filtering
scan_fit_diagnostics    : residuals and quality metrics at scan points
export_scan_fit_csv     : write phi / energy / V_fitted / residual CSV
ingest_scan_csv         : read torsion scan from a CSV file
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from backend.torsion_hamiltonian import TorsionFourierPotential

# ── Unit conversions ──────────────────────────────────────────────────────────
_HARTREE_TO_CM1 = 219474.6313705
_KCAL_PER_MOL_TO_CM1 = 349.7550874793
_KJ_PER_MOL_TO_CM1 = 83.5934722514


def energies_to_cm1(energies, unit: str) -> np.ndarray:
    """Convert scan energies to cm^-1.

    Supported units: cm-1, hartree (ha, eh), kcal/mol (kcal), kj/mol (kj).
    """
    u = str(unit).strip().lower()
    e = np.asarray(energies, dtype=float)
    if u in {"cm-1", "cm_1", "cm1"}:
        return e
    if u in {"hartree", "ha", "eh"}:
        return e * _HARTREE_TO_CM1
    if u in {"kcal/mol", "kcal"}:
        return e * _KCAL_PER_MOL_TO_CM1
    if u in {"kj/mol", "kj"}:
        return e * _KJ_PER_MOL_TO_CM1
    raise ValueError(
        f"Unknown energy unit '{unit}'. Allowed: cm-1, hartree, kcal/mol, kj/mol."
    )


# ── Scan coverage validation ──────────────────────────────────────────────────

def validate_scan_coverage(
    phi_rad: np.ndarray,
    energies: Optional[np.ndarray] = None,
    *,
    period_rad: float = 2.0 * np.pi,
    min_points: int = 5,
    max_gap_frac: float = 0.25,
    endpoint_tol_rad: float = 0.05,
) -> dict:
    """
    Check torsion scan coverage, spacing, periodicity, and duplicate endpoints.

    Parameters
    ----------
    phi_rad : array of torsion angles in radians
    energies : optional array of energies (same length as phi_rad); checked for
               length match and finiteness if provided
    period_rad : expected period of the torsion coordinate (default 2*pi)
    min_points : minimum number of scan points required
    max_gap_frac : maximum allowed gap as a fraction of the period
    endpoint_tol_rad : tolerance for detecting duplicate endpoint (rad)

    Returns
    -------
    dict with keys:
      ok                   : bool — True if no fatal errors
      warnings             : list[str]
      errors               : list[str]
      n_points             : int
      coverage_frac        : float — direct span / period
      max_gap_frac         : float — largest gap / period (includes wrap-around)
      has_duplicate_endpoint : bool
      period_rad           : float
    """
    phi = np.sort(np.asarray(phi_rad, dtype=float).ravel())
    n = int(phi.size)
    errors: list[str] = []
    warnings: list[str] = []

    if n < min_points:
        errors.append(
            f"Only {n} scan points (minimum {min_points} required for a reliable fit)."
        )

    period = float(period_rad)
    if period <= 0.0:
        errors.append("period_rad must be positive.")
        return {"ok": False, "warnings": warnings, "errors": errors,
                "n_points": n, "coverage_frac": 0.0, "max_gap_frac": 1.0,
                "has_duplicate_endpoint": False, "period_rad": period}

    span = float(phi[-1] - phi[0]) if n > 0 else 0.0
    coverage_frac = span / period

    if n > 0 and coverage_frac < 0.5:
        warnings.append(
            f"Scan covers only {100*coverage_frac:.1f}% of the period "
            f"({span:.3f} rad of {period:.3f} rad); consider extending the scan."
        )

    # Gaps including wrap-around gap
    if n > 1:
        interior_gaps = np.diff(phi)
        wrap_gap = period - span
        all_gaps = np.append(interior_gaps, wrap_gap)
    else:
        all_gaps = np.array([period], dtype=float)

    max_gap = float(np.max(all_gaps)) if all_gaps.size > 0 else 0.0
    max_gap_f = max_gap / period

    if max_gap_f > max_gap_frac:
        warnings.append(
            f"Largest scan gap is {100*max_gap_f:.1f}% of period "
            f"(threshold {100*max_gap_frac:.1f}%); consider denser sampling in that region."
        )

    # Duplicate endpoint: first and last phi are ~period apart
    has_dup = False
    if n >= 2:
        # After sorting, phi[0] + period ≈ phi[-1] means phi[-1] is a duplicate of phi[0]
        residual = abs(float(phi[-1] - phi[0]) - period)
        if residual < endpoint_tol_rad:
            has_dup = True
            warnings.append(
                f"Scan appears to have a duplicate endpoint "
                f"(phi[0]={phi[0]:.4f} rad, phi[-1]={phi[-1]:.4f} rad, "
                f"expected period={period:.4f} rad); "
                "remove the duplicate before fitting to avoid biasing the constant term."
            )

    # Energy checks
    if energies is not None:
        e = np.asarray(energies, dtype=float).ravel()
        if e.size != n:
            errors.append(
                f"Energies length ({e.size}) does not match phi length ({n})."
            )
        elif not np.all(np.isfinite(e)):
            errors.append("Energies contain non-finite values (NaN or Inf).")

    # Negative gaps (non-monotonic after sort — shouldn't happen but check)
    if n > 1 and np.any(interior_gaps < 0):
        errors.append("Internal error: negative gaps after sorting phi.")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "warnings": warnings,
        "errors": errors,
        "n_points": n,
        "coverage_frac": float(coverage_frac),
        "max_gap_frac": float(max_gap_f),
        "has_duplicate_endpoint": has_dup,
        "period_rad": period,
    }


# ── Fourier fitting ───────────────────────────────────────────────────────────

def fit_fourier_potential(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    *,
    n_harmonics: int = 6,
    include_harmonics: Optional[list[int]] = None,
    cosine_only: bool = False,
    zero_at_minimum: bool = True,
) -> dict:
    """
    Fit Fourier potential coefficients from torsion scan energies.

    Fits V(phi) = v0 + sum_n [vcos_n*cos(n*phi) + vsin_n*sin(n*phi)]
    by linear least squares.

    Parameters
    ----------
    phi_rad : torsion angles in radians
    energies_cm1 : potential energies in cm^-1 (relative; absolute values work too
                   since v0 absorbs the offset)
    n_harmonics : maximum harmonic order (1..n_harmonics), used when
                  include_harmonics is None
    include_harmonics : explicit list of harmonic orders to include; overrides
                        n_harmonics when provided
    cosine_only : if True, only cosine terms are fitted (omits sin columns)
    zero_at_minimum : if True, shift v0 so that min(V) = 0 on a fine grid

    Returns
    -------
    dict with keys:
      v0, vcos (dict), vsin (dict)  : fitted coefficients
      residuals_cm1                 : array of fit residuals at scan points
      rms_cm1                       : RMS of residuals
      fit_matrix_rank               : rank of the design matrix
      n_params                      : number of fitted parameters
      n_points                      : number of scan points
      harmonics                     : list of harmonic orders used
      cosine_only                   : bool
      fit_ok                        : bool — True if rank >= min(n_params, n_points)
      warnings                      : list[str]
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    n = phi.size

    if n != e.size:
        raise ValueError(
            f"phi_rad ({n}) and energies_cm1 ({e.size}) must have the same length."
        )
    if n < 2:
        raise ValueError("At least 2 scan points are required for Fourier fitting.")

    # Select harmonics
    if include_harmonics is not None:
        harmonics = sorted(int(h) for h in include_harmonics if int(h) > 0)
    else:
        harmonics = list(range(1, int(n_harmonics) + 1))

    # Build design matrix: [1, cos(h*phi), [sin(h*phi)], ...]
    cols = [np.ones(n, dtype=float)]
    for h in harmonics:
        cols.append(np.cos(h * phi))
        if not cosine_only:
            cols.append(np.sin(h * phi))

    A = np.column_stack(cols)
    n_params = A.shape[1]
    warnings: list[str] = []

    if n < n_params:
        warnings.append(
            f"Underdetermined fit: {n} scan points < {n_params} parameters. "
            "Reduce n_harmonics or supply more scan points."
        )

    # Least-squares fit
    coeffs, _residuals_sq, rank, _sv = np.linalg.lstsq(A, e, rcond=None)
    rank = int(rank)

    if rank < n_params:
        warnings.append(
            f"Design matrix rank ({rank}) < n_params ({n_params}); "
            "the fitted potential is underdetermined — consider fewer harmonics."
        )

    # Unpack coefficients
    v0 = float(coeffs[0])
    vcos: dict[int, float] = {}
    vsin: dict[int, float] = {}
    idx = 1
    for h in harmonics:
        vcos[h] = float(coeffs[idx])
        idx += 1
        if not cosine_only:
            vsin[h] = float(coeffs[idx])
            idx += 1

    # Residuals at scan points
    V_at_scan = A @ coeffs
    residuals_cm1 = e - V_at_scan
    rms = float(np.sqrt(np.mean(residuals_cm1 ** 2)))

    if rms > 100.0:
        warnings.append(
            f"Large RMS residual ({rms:.1f} cm^-1); "
            "check that energies are relative (zero at minimum) and in cm^-1."
        )

    # Shift v0 so that min(V) = 0 on a fine grid
    if zero_at_minimum:
        phi_fine = np.linspace(float(phi.min()), float(phi.max()), max(1801, 5 * n))
        V_fine = np.full(phi_fine.size, v0, dtype=float)
        for h in harmonics:
            V_fine += vcos.get(h, 0.0) * np.cos(h * phi_fine)
            V_fine += vsin.get(h, 0.0) * np.sin(h * phi_fine)
        v0 -= float(np.min(V_fine))

    return {
        "v0": v0,
        "vcos": vcos,
        "vsin": vsin,
        "residuals_cm1": residuals_cm1,
        "rms_cm1": rms,
        "fit_matrix_rank": rank,
        "n_params": n_params,
        "n_points": n,
        "harmonics": harmonics,
        "cosine_only": cosine_only,
        "fit_ok": rank >= min(n_params, n),
        "warnings": warnings,
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def scan_to_torsion_potential(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    *,
    n_harmonics: int = 6,
    symmetry_number: int = 1,
    cosine_only: Optional[bool] = None,
    zero_at_minimum: bool = True,
) -> tuple[TorsionFourierPotential, dict]:
    """
    Fit a TorsionFourierPotential from scan energies with symmetry filtering.

    For symmetry_number > 1 (e.g. 3 for methanol CH3), only harmonics that
    are multiples of symmetry_number are included (e.g. n=3, 6, 9, ...).
    cosine_only defaults to True when symmetry_number > 1 (Cs + fold symmetry).

    Parameters
    ----------
    phi_rad : torsion angles in radians
    energies_cm1 : potential energies in cm^-1
    n_harmonics : number of symmetry-allowed harmonic groups to include
                  (e.g. n_harmonics=3 with symmetry_number=3 → orders 3, 6, 9)
    symmetry_number : fold symmetry of the rotor (1 = no symmetry constraint)
    cosine_only : restrict to cosine terms; defaults to True when symmetry_number > 1
    zero_at_minimum : shift v0 so that min(V) = 0

    Returns
    -------
    (TorsionFourierPotential, fit_result_dict)
    """
    if cosine_only is None:
        cosine_only = symmetry_number > 1

    if symmetry_number > 1:
        include_harmonics = [n * symmetry_number for n in range(1, n_harmonics + 1)]
    else:
        include_harmonics = list(range(1, n_harmonics + 1))

    result = fit_fourier_potential(
        phi_rad,
        energies_cm1,
        include_harmonics=include_harmonics,
        cosine_only=cosine_only,
        zero_at_minimum=zero_at_minimum,
    )

    pot = TorsionFourierPotential(
        v0=result["v0"],
        vcos=result["vcos"],
        vsin=result["vsin"],
        units="cm-1",
    )
    return pot, result


# ── Diagnostics ───────────────────────────────────────────────────────────────

def scan_fit_diagnostics(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    v0: float,
    vcos: dict[int, float],
    vsin: dict[int, float],
) -> dict:
    """
    Evaluate fitted potential at scan points and compute quality metrics.

    Returns
    -------
    dict with:
      V_fitted_cm1        : fitted V(phi) at each scan point
      residuals_cm1       : energies - V_fitted
      rms_cm1             : RMS residual
      max_abs_residual_cm1: maximum absolute residual
      n_points            : number of scan points
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()

    V = np.full(phi.size, float(v0), dtype=float)
    for k, amp in vcos.items():
        V += float(amp) * np.cos(int(k) * phi)
    for k, amp in vsin.items():
        V += float(amp) * np.sin(int(k) * phi)

    residuals = e - V
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "V_fitted_cm1": V,
        "residuals_cm1": residuals,
        "rms_cm1": rms,
        "max_abs_residual_cm1": float(np.max(np.abs(residuals))),
        "n_points": int(phi.size),
    }


# ── Export ────────────────────────────────────────────────────────────────────

def export_scan_fit_csv(
    path: Path,
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    v0: float,
    vcos: dict[int, float],
    vsin: dict[int, float],
) -> None:
    """Write scan + fitted potential to CSV.

    Columns: phi_deg, energy_cm1, V_fitted_cm1, residual_cm1
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    diag = scan_fit_diagnostics(phi, e, v0, vcos, vsin)
    V_fit = diag["V_fitted_cm1"]
    res = diag["residuals_cm1"]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(path), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phi_deg", "energy_cm1", "V_fitted_cm1", "residual_cm1"])
        for phi_i, e_i, v_i, r_i in zip(
            phi * 180.0 / np.pi, e, V_fit, res
        ):
            writer.writerow([
                f"{float(phi_i):.4f}",
                f"{float(e_i):.6f}",
                f"{float(v_i):.6f}",
                f"{float(r_i):.6f}",
            ])


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_scan_csv(
    path: Path,
    *,
    phi_col: str = "phi_deg",
    energy_col: str = "energy_cm1",
    angle_unit: str = "degrees",
    energy_unit: str = "cm-1",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a torsion scan CSV file.

    Returns (phi_rad, energies_cm1).

    The CSV must have a header row. Column names are configurable via
    phi_col and energy_col. Angle and energy units are converted automatically.
    """
    phi_list: list[float] = []
    energy_list: list[float] = []

    with open(Path(path), newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if phi_col not in row:
                raise ValueError(
                    f"Column '{phi_col}' not found in {path}. "
                    f"Available: {list(row.keys())}"
                )
            if energy_col not in row:
                raise ValueError(
                    f"Column '{energy_col}' not found in {path}. "
                    f"Available: {list(row.keys())}"
                )
            phi_list.append(float(row[phi_col]))
            energy_list.append(float(row[energy_col]))

    phi = np.array(phi_list, dtype=float)
    energies = np.array(energy_list, dtype=float)

    au = str(angle_unit).strip().lower()
    if au in {"deg", "degrees", "degree"}:
        phi = phi * np.pi / 180.0
    elif au in {"rad", "radians", "radian"}:
        pass
    else:
        raise ValueError(f"Unknown angle unit '{angle_unit}'. Use 'degrees' or 'radians'.")

    energies = energies_to_cm1(energies, energy_unit)
    return phi, energies
