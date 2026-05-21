"""
Harmonic vibration-rotation alpha constants from the Cartesian Hessian.

Computes the isotopologue-specific harmonic alpha sum Σ_r α_r^K (K = A, B, C)
from:
  - The Cartesian Hessian in Hartree/Bohr²
  - The equilibrium geometry in Angstroms
  - The atomic masses in amu (per isotopologue)

Two contributions are included (Watson 1968):
  α_r^K(centrifugal): from ∂²B_K/∂Q_r² at equilibrium (finite difference)
  α_r^K(Coriolis):    from angular-momentum coupling between modes r and s
                       via ζ_{rs}^ε coupling constants and the Coriolis formula

Watson sign convention:  B_0^K = B_e^K − ½ Σ_r α_r^K

References:
  Watson, J. K. G.  Mol. Phys. 15 (1968) 479.
  Papoušek, D.; Aliev, M. R.  Molecular Vibrational-Rotational Spectra (1982).
"""

from __future__ import annotations

import numpy as np

from backend.spectral.centrifugal_distortion import (
    _BOHR_TO_ANG,
    _CM_TO_MHZ,
    _MHZ_TO_CM,
    _ZPE_AMP,
    bk_mode_derivatives as _bk_mode_derivatives,
    inertia_paf as _inertia_paf,
    normal_modes as _normal_modes,
    rotational_constants_mhz as _rotational_constants,
)

# Re-export for callers that imported helpers from this module.
_rotational_constants = _rotational_constants
_inertia_paf = _inertia_paf
_normal_modes = _normal_modes
_bk_mode_derivatives = _bk_mode_derivatives


def compute_harmonic_alpha(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    masses_amu,
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.02,
):
    """
    Compute summed harmonic alpha Σ_r α_r^K for each rotational component K.

    Watson convention:  B_0^K = B_e^K − ½ Σ_r α_r^K

    Parameters
    ----------
    hess_bohr     : (3N, 3N) Cartesian Hessian in Hartree/Bohr²
    coords_ang    : (N, 3)   equilibrium geometry in Angstroms
    masses_amu    : (N,)     atomic masses in amu (isotopologue-specific)
    min_freq_cm   : modes below this threshold are excluded
    fd_delta      : finite-difference step [Å/√amu] for centrifugal term
    sigma_fraction: fractional uncertainty on each returned alpha value

    Returns
    -------
    alpha_sum : dict  {K: float}   Σ_r α_r^K in MHz  (K in {'A','B','C'})
    B_e       : dict  {K: float}   B_e^K in MHz
    sigma     : dict  {K: float}   uncertainty on Σ_r α_r^K in MHz
    """
    N = len(masses_amu)
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)

    I_eig, V_paf, _ = _inertia_paf(coords, masses)
    B_e_mhz = _rotational_constants(coords, masses)
    B_e_cm = B_e_mhz * _MHZ_TO_CM

    omega_cm, L_mw = _normal_modes(hess_bohr, masses)
    real_mask = omega_cm >= min_freq_cm
    omega_cm = omega_cm[real_mask]
    L_mw = L_mw[:, real_mask]
    n_vib = len(omega_cm)

    labels = ["A", "B", "C"]
    if n_vib == 0:
        return (
            {k: 0.0 for k in labels},
            {k: float(B_e_mhz[i]) for i, k in enumerate(labels)},
            {k: 0.0 for k in labels},
            {"near_degen_skips": 0},
        )

    L_reshaped = L_mw.reshape(N, 3, n_vib)
    L_paf = np.einsum("aJr,JK->aKr", L_reshaped, V_paf)

    _ax_pairs = [(1, 2), (2, 0), (0, 1)]
    zeta = np.zeros((3, n_vib, n_vib))
    for K, (a1, a2) in enumerate(_ax_pairs):
        zeta[K] = (
            np.einsum("ar,as->rs", L_paf[:, a1, :], L_paf[:, a2, :])
            - np.einsum("ar,as->rs", L_paf[:, a2, :], L_paf[:, a1, :])
        )

    B0_ref = _rotational_constants(coords, masses)
    _, d2B_mhz_all = _bk_mode_derivatives(coords, masses, L_mw, omega_cm, fd_delta, B0_ref)
    alpha_cent = np.zeros((3, n_vib))
    for r in range(n_vib):
        zpe_amp = _ZPE_AMP / omega_cm[r]
        alpha_cent[:, r] = -d2B_mhz_all[:, r] * zpe_amp

    alpha_cor_cm = np.zeros((3, n_vib))
    near_degen_skips = 0

    for r in range(n_vib):
        wr2 = omega_cm[r] ** 2
        for K in range(3):
            cor = 0.0
            for J in range(3):
                if J == K:
                    continue
                eps = 3 - K - J
                for s in range(n_vib):
                    if s == r:
                        continue
                    ws2 = omega_cm[s] ** 2
                    denom = ws2 - wr2
                    if abs(denom) < 0.01:
                        near_degen_skips += 1
                        continue
                    cor += B_e_cm[J] * zeta[eps, r, s] ** 2 * omega_cm[s] / denom
            alpha_cor_cm[K, r] = -4.0 * B_e_cm[K] * cor

    alpha_cor = alpha_cor_cm * _CM_TO_MHZ
    alpha_total = alpha_cent + alpha_cor
    alpha_sum = alpha_total.sum(axis=1)

    sigma_vals = {
        k: max(abs(float(alpha_sum[i])) * sigma_fraction, 1.0)
        for i, k in enumerate(labels)
    }

    return (
        {k: float(alpha_sum[i]) for i, k in enumerate(labels)},
        {k: float(B_e_mhz[i]) for i, k in enumerate(labels)},
        sigma_vals,
        {"near_degen_skips": near_degen_skips},
    )


def build_correction_table_from_hessian(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    isotopologues: list[dict],
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.02,
) -> tuple[dict, dict]:
    """
    Build a correction_table dict (compatible with parse_correction_table)
    from the harmonic alpha computed from the Hessian.
    """
    table: dict = {}
    total_near_degen_skips = 0
    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        masses = list(iso.get("masses", []))
        if not masses:
            continue
        alpha_sum, _, sigma, res_info = compute_harmonic_alpha(
            hess_bohr,
            coords_ang,
            masses,
            min_freq_cm=min_freq_cm,
            fd_delta=fd_delta,
            sigma_fraction=sigma_fraction,
        )
        total_near_degen_skips += res_info.get("near_degen_skips", 0)
        table[name] = {
            comp: {
                "alpha_sum_mhz": alpha_sum[comp],
                "sigma_mhz": sigma[comp],
                "source": "harmonic_hessian",
                "method": "harmonic_VR",
                "notes": "Harmonic alpha from Hessian (centrifugal+Coriolis)",
            }
            for comp in ("A", "B", "C")
        }
    return table, {"total_near_degen_skips": total_near_degen_skips}
