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
from scipy import constants as _sc


# ── Physical constants ────────────────────────────────────────────────────────

_BOHR_TO_ANG = _sc.physical_constants["Bohr radius"][0] * 1e10   # 0.529177…
_ANG_TO_BOHR = 1.0 / _BOHR_TO_ANG
_HARTREE_J   = _sc.physical_constants["Hartree energy"][0]        # 4.3597…e-18 J
_AMU_SI      = _sc.atomic_mass                                     # 1.6605…e-27 kg
_H_SI        = _sc.h                                               # 6.6261…e-34 J·s
_C_CM        = _sc.c * 100                                         # cm/s
_ANG_M       = 1e-10                                               # m/Å
_BOHR_M      = _BOHR_TO_ANG * _ANG_M                              # m/Bohr

# Eigenvalue [Hartree/(amu·Bohr²)] → wavenumber [cm⁻¹]
#   ω [rad/s] = sqrt(λ × E_h / (amu × Bohr_m²))   then ÷ (2π c_cm)
_EIGVAL_TO_CM = np.sqrt(_HARTREE_J / (_AMU_SI * _BOHR_M**2)) / (2 * np.pi * _C_CM)

# Zero-point amplitude ⟨Q_r²⟩₀ = ZPE_AMP / ω̃_r   [Å²/amu  per  cm⁻¹]
#   ⟨Q_r²⟩ = h/(8π² c ω̃ × amu × Å²)
_ZPE_AMP = _H_SI / (8 * np.pi**2 * _C_CM * _AMU_SI * _ANG_M**2)

# Inertia [amu·Å²] → rotational constant [MHz]
_INERTIA_TO_MHZ = _H_SI / (8 * np.pi**2 * _AMU_SI * _ANG_M**2) * 1e-6

# Wavenumber [cm⁻¹] → MHz
_CM_TO_MHZ = _C_CM * 1e-6    # 29979.2458 MHz / cm⁻¹
_MHZ_TO_CM = 1.0 / _CM_TO_MHZ


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rotational_constants(coords_ang, masses_amu):
    """
    Principal rotational constants A ≥ B ≥ C in MHz.

    Returns shape (3,) sorted largest → smallest (smallest moment first).
    """
    masses  = np.asarray(masses_amu, dtype=float)
    coords  = np.asarray(coords_ang, dtype=float)
    com     = (masses[:, None] * coords).sum(0) / masses.sum()
    r       = coords - com
    r2      = np.einsum("ia,ia->i", r, r)
    I       = np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)
    eigvals = np.sort(np.linalg.eigvalsh(I))          # ascending: I_A ≤ I_B ≤ I_C
    eigvals = np.where(eigvals > 1e-10, eigvals, np.inf)
    return _INERTIA_TO_MHZ / eigvals                   # A ≥ B ≥ C


def _inertia_paf(coords_ang, masses_amu):
    """
    Return (eigvals, V_paf, coords_paf):
      eigvals   : (3,) principal moments in amu·Å², sorted I_A ≤ I_B ≤ I_C
      V_paf     : (3,3) rotation matrix — columns are PAF axes X,Y,Z = A,B,C
      coords_paf: (N,3) COM-centered coords in PAF
    """
    masses  = np.asarray(masses_amu, dtype=float)
    coords  = np.asarray(coords_ang, dtype=float)
    com     = (masses[:, None] * coords).sum(0) / masses.sum()
    r       = coords - com
    r2      = np.einsum("ia,ia->i", r, r)
    I       = np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)
    eigvals, V = np.linalg.eigh(I)                    # ascending: smallest I first
    coords_paf = r @ V
    return eigvals, V, coords_paf


def _normal_modes(hess_bohr, masses_amu, n_rigid=6, min_eigval=1e-6):
    """
    Mass-weighted normal modes from the Cartesian Hessian.

    Parameters
    ----------
    hess_bohr  : (3N, 3N) Hessian in Hartree/Bohr²
    masses_amu : (N,) atomic masses in amu
    n_rigid    : number of zero/near-zero modes to skip (6 for nonlinear, 5 for linear)
    min_eigval : positive eigenvalue threshold [Hartree/(amu·Bohr²)]

    Returns
    -------
    omega_cm : (n_vib,) vibrational frequencies in cm⁻¹ (positive real modes only)
    L_mw     : (3N, n_vib) normalised mass-weighted eigenvectors, sorted by frequency
    """
    masses       = np.asarray(masses_amu, dtype=float)
    M_inv_sqrt   = np.repeat(1.0 / np.sqrt(masses), 3)
    F_mw         = hess_bohr * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
    eigenvalues, L = np.linalg.eigh(F_mw)             # ascending order
    # Sort then skip translations/rotations
    sort_idx    = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    L           = L[:, sort_idx]
    vib_evals   = eigenvalues[n_rigid:]
    L_vib       = L[:, n_rigid:]
    # Keep only positive (real vibrational) modes
    pos_mask    = vib_evals > min_eigval
    vib_evals   = vib_evals[pos_mask]
    L_vib       = L_vib[:, pos_mask]
    omega_cm    = _EIGVAL_TO_CM * np.sqrt(vib_evals)
    return omega_cm, L_vib


# ── Core computation ──────────────────────────────────────────────────────────

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
    N      = len(masses_amu)
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)

    # ── Step 1: Principal axis frame (PAF) ────────────────────────────────────
    I_eig, V_paf, _ = _inertia_paf(coords, masses)
    # B_e in MHz: index 0 = A (largest constant), 2 = C (smallest)
    B_e_mhz = np.where(I_eig > 1e-10, _INERTIA_TO_MHZ / I_eig, 0.0)
    B_e_cm  = B_e_mhz * _MHZ_TO_CM

    # ── Step 2: Normal modes ──────────────────────────────────────────────────
    omega_cm, L_mw = _normal_modes(hess_bohr, masses)
    # Filter below min_freq_cm
    real_mask = omega_cm >= min_freq_cm
    omega_cm  = omega_cm[real_mask]
    L_mw      = L_mw[:, real_mask]
    n_vib     = len(omega_cm)

    labels = ["A", "B", "C"]
    if n_vib == 0:
        return (
            {k: 0.0 for k in labels},
            {k: float(B_e_mhz[i]) for i, k in enumerate(labels)},
            {k: 0.0 for k in labels},
            {"near_degen_skips": 0},
        )

    omega_mhz = omega_cm * _CM_TO_MHZ

    # ── Step 3: Rotate normal modes into PAF ──────────────────────────────────
    # L_mw : (3N, n_vib) — eigenvectors in original Cartesian frame
    # V_paf: (3, 3)       — columns are PAF axis directions
    L_reshaped = L_mw.reshape(N, 3, n_vib)           # (N, xyz_orig, n_vib)
    L_paf      = np.einsum("aJr,JK->aKr", L_reshaped, V_paf)  # (N, K_paf, n_vib)

    # ── Step 4: Coriolis coupling constants ζ_{rs}^K ─────────────────────────
    # K=0 (A, x-PAF): (α,β) = (1,2)  →  ζ_{rs}^A = Σ_a L_paf[a,1,r]·L_paf[a,2,s] - swap
    # K=1 (B, y-PAF): (α,β) = (2,0)
    # K=2 (C, z-PAF): (α,β) = (0,1)
    # Convention: ζ[K,r,s] = −ζ[K,s,r]
    _ax_pairs = [(1, 2), (2, 0), (0, 1)]
    zeta = np.zeros((3, n_vib, n_vib))
    for K, (a1, a2) in enumerate(_ax_pairs):
        zeta[K] = (
            np.einsum("ar,as->rs", L_paf[:, a1, :], L_paf[:, a2, :])
            - np.einsum("ar,as->rs", L_paf[:, a2, :], L_paf[:, a1, :])
        )

    # ── Step 5: Centrifugal contribution (finite difference) ─────────────────
    # α_r^K(cent) = −(∂²B_K/∂Q_r²) × ⟨Q_r²⟩₀
    # ⟨Q_r²⟩₀ = _ZPE_AMP / ω̃_r  [Å²/amu]
    # ∂²B_K/∂Q_r² = [B(+δ)+B(−δ)−2B(0)] / δ²  [MHz/(Å²/amu)]
    B0_ref   = _rotational_constants(coords, masses)  # MHz, shape (3,)
    alpha_cent = np.zeros((3, n_vib))

    for r in range(n_vib):
        # Cartesian displacement in Å/√amu
        d_r       = (L_mw[:, r].reshape(N, 3) / np.sqrt(masses[:, None])) * _BOHR_TO_ANG
        c_plus    = coords + fd_delta * d_r
        c_minus   = coords - fd_delta * d_r
        B_plus    = _rotational_constants(c_plus,  masses)
        B_minus   = _rotational_constants(c_minus, masses)
        d2B_mhz   = (B_plus + B_minus - 2.0 * B0_ref) / (fd_delta ** 2)
        zpe_amp   = _ZPE_AMP / omega_cm[r]          # Å²/amu
        alpha_cent[:, r] = -d2B_mhz * zpe_amp

    # ── Step 6: Coriolis contribution ─────────────────────────────────────────
    # α_r^K(Cor) [cm⁻¹] = −4 B_e^K Σ_{J≠K} B_e^J Σ_{s≠r} ζ[ε,r,s]² ω_s/(ω_s²−ω_r²)
    # where ε = 3−K−J  (the third axis not K, not J)
    # All B_e and ω in cm⁻¹; then convert result to MHz.
    alpha_cor_cm = np.zeros((3, n_vib))
    near_degen_skips = 0  # count (r,s) pairs skipped due to |ω_s²−ω_r²| < threshold

    for r in range(n_vib):
        wr2 = omega_cm[r] ** 2
        for K in range(3):
            cor = 0.0
            for J in range(3):
                if J == K:
                    continue
                eps = 3 - K - J           # third axis (0,1, or 2)
                for s in range(n_vib):
                    if s == r:
                        continue
                    ws2 = omega_cm[s] ** 2
                    denom = ws2 - wr2
                    if abs(denom) < 0.01:  # near-degenerate modes → skip
                        near_degen_skips += 1
                        continue
                    cor += B_e_cm[J] * zeta[eps, r, s] ** 2 * omega_cm[s] / denom
            alpha_cor_cm[K, r] = -4.0 * B_e_cm[K] * cor

    alpha_cor = alpha_cor_cm * _CM_TO_MHZ  # convert to MHz

    # ── Step 7: Sum over modes ────────────────────────────────────────────────
    alpha_total = alpha_cent + alpha_cor         # (3, n_vib) MHz
    alpha_sum   = alpha_total.sum(axis=1)        # (3,) MHz

    sigma_vals = {k: max(abs(float(alpha_sum[i])) * sigma_fraction, 1.0)
                  for i, k in enumerate(labels)}

    return (
        {k: float(alpha_sum[i]) for i, k in enumerate(labels)},
        {k: float(B_e_mhz[i])  for i, k in enumerate(labels)},
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

    Returns
    -------
    table           : dict keyed by isotopologue name → component → spec
    resonance_info  : {"total_near_degen_skips": int} aggregated across isotopologues
    """
    table: dict = {}
    total_near_degen_skips = 0
    for iso in isotopologues:
        name   = str(iso.get("name", "iso"))
        masses = list(iso.get("masses", []))
        if not masses:
            continue
        alpha_sum, _, sigma, res_info = compute_harmonic_alpha(
            hess_bohr, coords_ang, masses,
            min_freq_cm=min_freq_cm,
            fd_delta=fd_delta,
            sigma_fraction=sigma_fraction,
        )
        total_near_degen_skips += res_info.get("near_degen_skips", 0)
        table[name] = {
            comp: {
                "alpha_sum_mhz": alpha_sum[comp],
                "sigma_mhz":     sigma[comp],
                "source":        "harmonic_hessian",
                "method":        "harmonic_VR",
                "notes":         "Harmonic alpha from Hessian (centrifugal+Coriolis)",
            }
            for comp in ("A", "B", "C")
        }
    return table, {"total_near_degen_skips": total_near_degen_skips}
