"""
Integration of LAM (torsional) corrections with the rovibrational correction pipeline.

Defines how torsion-averaged constants combine with the existing alpha-correction
framework, prevents double-counting of low-frequency torsional modes, propagates
torsion uncertainty into reported rotational constants, and classifies the
provenance of the final reported constants.

Correction hierarchy
--------------------
1. Rigid equilibrium constants (Be)
2. + Standard rovibrational (alpha) corrections for non-torsional modes → B_rovib
3a. Torsion path: + torsional averaging correction (from torsion_average.py) → B_eff
3b. Standard path: + torsional alpha from VPT2 included in step 2 → B0

If step 3a is used, the torsional-mode alpha contributions must be excluded from
step 2 to prevent double-counting.

Reported constant modes
-----------------------
  'rigid'           : Be only; no alpha or torsion correction applied
  'rovib_corrected' : standard alpha correction, no explicit torsion averaging
  'torsion_averaged': rovib (non-torsional modes) + torsion-averaged correction
  'globally_fit'    : parameters fitted directly to observed transitions/levels
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ── Constant-source classification ──────────────────────────────────────────

_VALID_SOURCES = frozenset({"rigid", "rovib_corrected", "torsion_averaged", "globally_fit"})


def classify_constant_source(cfg: dict) -> str:
    """
    Determine the reported-constant mode from a runner config dict.

    Checks for: torsion_hamiltonian.fitting.enabled → 'globally_fit'
                torsion_hamiltonian.scan_average.enabled → 'torsion_averaged'
                torsion_hamiltonian.enabled → 'rovib_corrected'
                otherwise → 'rigid'

    Returns one of: 'rigid', 'rovib_corrected', 'torsion_averaged', 'globally_fit'.
    """
    if not isinstance(cfg, dict):
        return "rigid"
    th = cfg.get("torsion_hamiltonian") or {}
    if not isinstance(th, dict) or not th.get("enabled", False):
        return "rigid"

    fitting = th.get("fitting") or {}
    if isinstance(fitting, dict) and fitting.get("enabled", False):
        return "globally_fit"

    avg = th.get("scan_average") or {}
    if isinstance(avg, dict) and avg.get("enabled", False):
        return "torsion_averaged"

    return "rovib_corrected"


# ── Double-counting prevention ───────────────────────────────────────────────

def remove_torsional_alpha_contributions(
    alpha_abc: np.ndarray,
    torsional_mode_alphas_abc: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Subtract torsional-mode alpha contributions from the total alpha vector.

    Use before applying rovib correction when the torsional mode is handled
    via explicit torsion averaging, to prevent double-counting its zero-point
    motion contribution.

    Parameters
    ----------
    alpha_abc : (N,) array — total alpha vector [A, B, C, ...] in cm^-1
    torsional_mode_alphas_abc : sequence of (N,) arrays — one per torsional mode to subtract

    Returns
    -------
    alpha_corrected : (3,) array — alpha[0:3] with torsional contributions removed
    """
    alpha = np.asarray(alpha_abc, dtype=float).ravel()
    if alpha.size < 3:
        raise ValueError("alpha_abc must have at least 3 elements (A, B, C).")
    result = alpha[:3].copy()
    for mode_alpha in torsional_mode_alphas_abc:
        ma = np.asarray(mode_alpha, dtype=float).ravel()
        if ma.size < 3:
            raise ValueError("Each entry in torsional_mode_alphas_abc must have at least 3 elements.")
        result -= ma[:3]
    return result


# ── LAM uncertainty contribution ─────────────────────────────────────────────

def lam_uncertainty_contribution(
    torsion_rms_cm1: float,
    n_torsion_levels: int,
    *,
    sigma_averaged_cm1: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
) -> float:
    """
    Estimate the uncertainty in B constants from torsional averaging.

    When ``sigma_averaged_cm1`` is provided (propagated from grid-point uncertainties
    via ``propagate_averaging_uncertainty``), it is used directly and the heuristic
    is bypassed.  The reported value is the maximum component (A, B, or C), which is
    conservative and gives a single scalar for the LAM report.

    Fallback heuristic (when sigma_averaged_cm1 is None):
      sigma_B_lam ≈ scale_factor * torsion_rms_cm1 / sqrt(n_torsion_levels)

    Parameters
    ----------
    torsion_rms_cm1 : RMS of torsion level/transition residuals [cm^-1]
    n_torsion_levels : number of levels used
    sigma_averaged_cm1 : (3,) propagated uncertainty on averaged [A,B,C] [cm^-1].
        When supplied, replaces the heuristic.
    scale_factor : optional tuning factor for the heuristic (default 1.0)

    Returns
    -------
    float : estimated sigma_B [cm^-1]
    """
    if sigma_averaged_cm1 is not None:
        sa = np.asarray(sigma_averaged_cm1, dtype=float).ravel()
        if sa.size >= 1:
            return float(np.max(sa))
    if float(torsion_rms_cm1) < 0.0:
        raise ValueError("torsion_rms_cm1 must be non-negative.")
    if int(n_torsion_levels) < 1:
        raise ValueError("n_torsion_levels must be >= 1.")
    return float(scale_factor) * float(torsion_rms_cm1) / float(np.sqrt(int(n_torsion_levels)))


# ── Correction combination ───────────────────────────────────────────────────

def combine_lam_corrections(
    B_rigid_cm1: np.ndarray,
    alpha_nontorsional_cm1: Optional[np.ndarray] = None,
    torsion_correction_cm1: Optional[np.ndarray] = None,
    *,
    source: str = "torsion_averaged",
) -> dict:
    """
    Build the combined effective rotational constant report.

    Parameters
    ----------
    B_rigid_cm1 : (3,) rigid equilibrium [A, B, C] in cm^-1
    alpha_nontorsional_cm1 : (3,) half-alpha shift from non-torsional modes only
        (i.e. torsional contributions already removed via remove_torsional_alpha_contributions)
    torsion_correction_cm1 : (3,) additional shift from torsion averaging
        (B_torsion_avg - B_rigid)
    source : classification string

    Returns
    -------
    dict with:
      B_rigid_cm-1, B_rovib_cm-1, B_effective_cm-1 : (3,) float arrays
      source : str
      corrections_applied : list[str]
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(_VALID_SOURCES)}, got {source!r}.")

    B = np.asarray(B_rigid_cm1, dtype=float).ravel()[:3]
    corrections: list[str] = []

    B_rovib = B.copy()
    if alpha_nontorsional_cm1 is not None:
        a = np.asarray(alpha_nontorsional_cm1, dtype=float).ravel()[:3]
        B_rovib = B + a
        corrections.append("rovib_nontorsional")

    B_eff = B_rovib.copy()
    if torsion_correction_cm1 is not None:
        tc = np.asarray(torsion_correction_cm1, dtype=float).ravel()[:3]
        B_eff = B_rovib + tc
        corrections.append("torsion_averaging")

    return {
        "B_rigid_cm-1": B,
        "B_rovib_cm-1": B_rovib,
        "B_effective_cm-1": B_eff,
        "source": source,
        "corrections_applied": corrections,
    }


def lam_correction_report(
    B_rigid_cm1: np.ndarray,
    *,
    B_torsion_avg_cm1: Optional[np.ndarray] = None,
    sigma_torsion_avg_cm1: Optional[np.ndarray] = None,
    alpha_full_cm1: Optional[np.ndarray] = None,
    torsional_mode_alphas_cm1: Optional[list[np.ndarray]] = None,
    torsion_rms_cm1: float = 0.0,
    n_torsion_levels: int = 1,
    source: str = "torsion_averaged",
) -> dict:
    """
    Full LAM correction report with automatic double-counting prevention.

    If B_torsion_avg_cm1 is given, torsional-mode alphas are subtracted from
    alpha_full before the rovib step, then the torsion-averaged correction is added.

    Parameters
    ----------
    B_rigid_cm1 : (3,) rigid equilibrium constants
    B_torsion_avg_cm1 : (3,) torsion-averaged constants from torsion_average.py
    sigma_torsion_avg_cm1 : (3,) propagated uncertainty on averaged constants [cm^-1].
        When provided, replaces the heuristic in lam_uncertainty_contribution.
        Comes from ``propagate_averaging_uncertainty()["sigma_total"]``.
    alpha_full_cm1 : (3,) full alpha vector (all modes, including torsion)
    torsional_mode_alphas_cm1 : list of (3,) alpha contributions for torsional modes
        to subtract before applying rovib correction
    torsion_rms_cm1 : RMS of torsion-level residuals [cm^-1]
    n_torsion_levels : number of levels used
    source : constant-source classification

    Returns
    -------
    dict from combine_lam_corrections plus:
      lam_uncertainty_cm-1 : float
      torsion_rms_cm-1 : float
      sigma_B_torsion_avg_cm-1 : list[float] | None  — per-constant propagated sigma
    """
    B = np.asarray(B_rigid_cm1, dtype=float).ravel()[:3]

    alpha_nt: Optional[np.ndarray] = None
    if alpha_full_cm1 is not None:
        a = np.asarray(alpha_full_cm1, dtype=float).ravel()[:3]
        if torsional_mode_alphas_cm1:
            a = remove_torsional_alpha_contributions(a, torsional_mode_alphas_cm1)
        alpha_nt = a * 0.5  # convert to half-alpha (B_rovib = B_e + 0.5*sum_r alpha_r)

    torsion_corr: Optional[np.ndarray] = None
    if B_torsion_avg_cm1 is not None:
        B_avg = np.asarray(B_torsion_avg_cm1, dtype=float).ravel()[:3]
        torsion_corr = B_avg - B

    report = combine_lam_corrections(B, alpha_nt, torsion_corr, source=source)
    report["lam_uncertainty_cm-1"] = lam_uncertainty_contribution(
        float(torsion_rms_cm1),
        max(int(n_torsion_levels), 1),
        sigma_averaged_cm1=sigma_torsion_avg_cm1,
    )
    report["torsion_rms_cm-1"] = float(torsion_rms_cm1)
    if sigma_torsion_avg_cm1 is not None:
        report["sigma_B_torsion_avg_cm-1"] = (
            np.asarray(sigma_torsion_avg_cm1, dtype=float).ravel()[:3].tolist()
        )
    else:
        report["sigma_B_torsion_avg_cm-1"] = None
    return report


def format_lam_report_for_summary(report: dict) -> dict:
    """
    Convert a lam_correction_report dict into JSON-serializable scalars.

    Converts numpy arrays to plain Python lists for embedding in summary JSON.
    """
    out: dict = {}
    for k, v in report.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
