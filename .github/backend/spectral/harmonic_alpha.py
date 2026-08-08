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
    _EIGVAL_TO_CM,
    _MHZ_TO_CM,
    _ZPE_AMP,
    bk_mode_derivatives as _bk_mode_derivatives,
    inertia_paf as _inertia_paf,
    normal_modes as _normal_modes,
    rigid_mode_count as _rigid_mode_count,
    rotational_constants_mhz as _rotational_constants,
)

# Re-export for callers that imported helpers from this module.
_rotational_constants = _rotational_constants
_inertia_paf = _inertia_paf
_normal_modes = _normal_modes
_bk_mode_derivatives = _bk_mode_derivatives

# |ω_r² − ω_s²| below this (cm⁻²) is treated as a Coriolis resonance and skipped
# rather than divided through, which would blow up the denominator.
_DEGENERACY_TOL_CM2 = 1.0


def compute_harmonic_alpha(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    masses_amu,
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.02,
    hessian_fn=None,
    fd_delta_cubic: float = 0.01,
    cubic_cart=None,
):
    """
    Compute summed alpha Σ_r α_r^K for each rotational component K.

    Watson convention:  B_0^K = B_e^K − ½ Σ_r α_r^K

    Parameters
    ----------
    hess_bohr     : (3N, 3N) Cartesian Hessian in Hartree/Bohr²
    coords_ang    : (N, 3)   equilibrium geometry in Angstroms
    masses_amu    : (N,)     atomic masses in amu (isotopologue-specific)
    min_freq_cm   : modes below this threshold are excluded
    fd_delta      : finite-difference step [Å·√amu] for the centrifugal term
    sigma_fraction: fractional uncertainty on each returned alpha value.
                    Ignored (floored at 100%) when the anharmonic term is
                    unavailable, because the omitted term is not a small
                    correction — see ``_anharmonic_alpha``.
    hessian_fn    : optional callable ``coords_ang -> (3N,3N) Hessian`` used to
                    build the Cartesian cubic force field by finite difference.
                    Without it (and without ``cubic_cart``) only the harmonic
                    and Coriolis terms are returned, which is a systematically
                    biased estimate of α. Costs 6N Hessian evaluations.
    fd_delta_cubic: Cartesian finite-difference step [Å] for the cubic term
    cubic_cart    : optional precomputed ∂³V/∂x_i∂x_j∂x_k from
                    :func:`cartesian_cubic_force_field`. Mass-independent, so
                    pass it in to avoid recomputing per isotopologue.

    Returns
    -------
    alpha_sum : dict  {K: float}   Σ_r α_r^K in MHz  (K in {'A','B','C'})
    B_e       : dict  {K: float}   B_e^K in MHz
    sigma     : dict  {K: float}   uncertainty on Σ_r α_r^K in MHz
    info      : dict               per-term breakdown and diagnostics
    """
    N = len(masses_amu)
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)

    I_eig, V_paf, _ = _inertia_paf(coords, masses)
    B_e_mhz = _rotational_constants(coords, masses)
    B_e_cm = B_e_mhz * _MHZ_TO_CM

    omega_cm, L_mw = _normal_modes(
        hess_bohr, masses, n_rigid=_rigid_mode_count(coords, masses)
    )
    real_mask = omega_cm >= min_freq_cm
    omega_cm = omega_cm[real_mask]
    L_mw = L_mw[:, real_mask]
    n_vib = len(omega_cm)
    # Exact inverse of omega_cm = _EIGVAL_TO_CM * sqrt(eigenvalue).
    vib_evals = (omega_cm / _EIGVAL_TO_CM) ** 2

    labels = ["A", "B", "C"]
    if n_vib == 0:
        # No correction is computable. Report it as unknown rather than as a
        # confident zero: a zero with sigma 0 tells the fit that B_0 == B_e.
        return (
            {k: 0.0 for k in labels},
            {k: float(B_e_mhz[i]) for i, k in enumerate(labels)},
            {k: float("inf") for k in labels},
            {
                "near_degen_skips": 0,
                "anharmonic_status": "no_modes",
                "warning": (
                    f"no vibrational modes above {min_freq_cm} cm-1; "
                    "alpha is undetermined, not zero"
                ),
            },
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
    dB1_mhz_all, d2B_mhz_all = _bk_mode_derivatives(
        coords, masses, L_mw, omega_cm, fd_delta, B0_ref
    )
    zpe_amp = _ZPE_AMP / omega_cm                      # ⟨Q_r²⟩ at v=0, Å²·amu

    # ── Term 1: harmonic / "centrifugal" ⟨Q_r²⟩ ∂²B/∂Q_r² ────────────────────
    # B_v = B_e + ½ (∂²B/∂Q_r²)⟨Q_r²⟩_v with ⟨Q_r²⟩_v = (2v+1)·zpe_amp, so
    # α_r ≡ -∂B_v/∂v_r = -(∂²B/∂Q_r²)·zpe_amp. For a diatomic this reduces
    # exactly to the Dunham result -6B_e²/ω_e.
    alpha_cent = -d2B_mhz_all * zpe_amp[None, :]

    # ── Term 2: Coriolis ────────────────────────────────────────────────────
    # Mills (1972); Papoušek & Aliev (1982):
    #   α_r^{ξ,Cor} = -(2 B_ξ²/ω_r) Σ_{s≠r} (ζ^{(ξ)}_{rs})² (3ω_r²+ω_s²)/(ω_r²-ω_s²)
    # The ζ superscript is the SAME principal axis as the rotational constant
    # being corrected: rotation about ξ is what couples the two modes.
    alpha_cor_cm = np.zeros((3, n_vib))
    near_degen_skips = 0
    for r in range(n_vib):
        wr2 = omega_cm[r] ** 2
        for K in range(3):
            cor = 0.0
            for s in range(n_vib):
                if s == r:
                    continue
                ws2 = omega_cm[s] ** 2
                denom = wr2 - ws2
                if abs(denom) < _DEGENERACY_TOL_CM2:
                    near_degen_skips += 1
                    continue
                cor += zeta[K, r, s] ** 2 * (3.0 * wr2 + ws2) / denom
            alpha_cor_cm[K, r] = -2.0 * B_e_cm[K] ** 2 / omega_cm[r] * cor
    alpha_cor = alpha_cor_cm * _CM_TO_MHZ

    # ── Term 3: anharmonic (cubic force constants) ──────────────────────────
    alpha_anh = np.zeros((3, n_vib))
    anh_status = "not_requested"
    if cubic_cart is None and hessian_fn is not None:
        try:
            cubic_cart = cartesian_cubic_force_field(hessian_fn, coords, fd_delta_cubic)
        except Exception as exc:                      # noqa: BLE001 - degrade, don't abort
            anh_status = f"cubic_ff_failed: {type(exc).__name__}: {exc}"
    if cubic_cart is not None:
        try:
            alpha_anh = _anharmonic_alpha(
                np.asarray(cubic_cart, dtype=float), masses, L_mw,
                vib_evals, dB1_mhz_all, zpe_amp,
            )
            anh_status = "cubic_fd"
        except Exception as exc:                      # noqa: BLE001 - degrade, don't abort
            anh_status = f"failed: {type(exc).__name__}: {exc}"

    alpha_total = alpha_cent + alpha_cor + alpha_anh
    alpha_sum = alpha_total.sum(axis=1)

    # Without the anharmonic term the residual model error is large and
    # systematic (for CO it is 2.7x the harmonic term, opposite sign), so the
    # reported uncertainty must not pretend to be a few percent.
    eff_sigma_fraction = (
        sigma_fraction if anh_status == "cubic_fd" else max(sigma_fraction, 1.0)
    )
    sigma_vals = {
        k: max(abs(float(alpha_sum[i])) * eff_sigma_fraction, 1.0)
        for i, k in enumerate(labels)
    }

    return (
        {k: float(alpha_sum[i]) for i, k in enumerate(labels)},
        {k: float(B_e_mhz[i]) for i, k in enumerate(labels)},
        sigma_vals,
        {
            "near_degen_skips": near_degen_skips,
            "anharmonic_status": anh_status,
            "alpha_centrifugal_mhz": {
                k: float(alpha_cent.sum(axis=1)[i]) for i, k in enumerate(labels)
            },
            "alpha_coriolis_mhz": {
                k: float(alpha_cor.sum(axis=1)[i]) for i, k in enumerate(labels)
            },
            "alpha_anharmonic_mhz": {
                k: float(alpha_anh.sum(axis=1)[i]) for i, k in enumerate(labels)
            },
            "frequencies_cm": omega_cm.tolist(),
        },
    )


def cartesian_cubic_force_field(hessian_fn, coords_ang, fd_delta_ang: float = 0.01):
    """Third derivatives ∂³V/∂x_i∂x_j∂x_k by central differences of the Hessian.

    Returned in the Hessian's own units per Å. The tensor depends only on the
    electronic potential, not on nuclear masses, so one evaluation serves every
    isotopologue — which is why this is factored out of the per-isotopologue
    alpha calculation.

    Costs 6N Hessian evaluations.
    """
    coords = np.asarray(coords_ang, dtype=float)
    n_cart = coords.size
    cubic = np.zeros((n_cart, n_cart, n_cart))
    for k in range(n_cart):
        step = np.zeros(n_cart)
        step[k] = fd_delta_ang
        h_plus = np.asarray(hessian_fn(coords + step.reshape(coords.shape)), dtype=float)
        h_minus = np.asarray(hessian_fn(coords - step.reshape(coords.shape)), dtype=float)
        cubic[:, :, k] = (h_plus - h_minus) / (2.0 * fd_delta_ang)
    # Symmetrise over the first two indices; V is smooth so ∂³V is fully symmetric
    # and averaging cancels first-order finite-difference asymmetry.
    return 0.5 * (cubic + np.swapaxes(cubic, 0, 1))


def _anharmonic_alpha(cubic_cart, masses, L_mw, vib_evals, dB1_mhz, zpe_amp):
    """Anharmonic contribution to α from semi-diagonal cubic force constants.

    Cubic anharmonicity displaces the vibrational average of each normal
    coordinate away from equilibrium.  To first order in perturbation theory,

        ⟨Q_s⟩_v = -(1 / 2λ_s) Σ_r φ_rrs ⟨Q_r²⟩_v

    and feeding that through B(Q) to first order gives

        α_r^{anh,K} = ⟨Q_r²⟩₀ · Σ_s (∂B_K/∂Q_s) · φ_rrs / λ_s.

    φ_rrs = Σ_ijk (∂³V/∂x_i∂x_j∂x_k) D_ir D_jr D_ks, where D = L/√m is the
    Cartesian displacement per unit normal coordinate. φ and λ come out in the
    same units, so their ratio needs no conversion constant.

    For a diatomic this reduces to -6 a₁ B_e²/ω_e, which together with the
    harmonic term reproduces the Dunham result α_e = -(6B_e²/ω_e)(1 + a₁).
    """
    N = len(masses)
    masses = np.asarray(masses, dtype=float)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    D = L_mw * inv_sqrt_m[:, None]                    # (3N, n_vib)

    lam = np.asarray(vib_evals, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("non-positive vibrational eigenvalue in anharmonic alpha")

    # phi[r, s] = semi-diagonal cubic constant φ_rrs
    phi = np.einsum("ijk,ir,jr,ks->rs", cubic_cart, D, D, D, optimize=True)

    # α_anh[K, r] = zpe_amp[r] * Σ_s dB1[K, s] * φ_rrs / λ_s
    return zpe_amp[None, :] * np.einsum(
        "Ks,rs,s->Kr", dB1_mhz, phi, 1.0 / lam, optimize=True
    )


def build_correction_table_from_hessian(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    isotopologues: list[dict],
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.02,
    hessian_fn=None,
    fd_delta_cubic: float = 0.01,
) -> tuple[dict, dict]:
    """
    Build a correction_table dict (compatible with parse_correction_table)
    from the alpha computed from the Hessian.

    When ``hessian_fn`` is supplied the cubic (anharmonic) term is included.
    The semi-diagonal cubic force constants depend only on the electronic PES,
    not on nuclear masses, so they are computed once and reused across every
    isotopologue rather than once per isotopologue.
    """
    table: dict = {}
    total_near_degen_skips = 0
    method = "VPT2_semidiag" if hessian_fn is not None else "harmonic_VR"
    notes = (
        "alpha from Hessian (harmonic + Coriolis + cubic anharmonic)"
        if hessian_fn is not None
        else "alpha from Hessian (harmonic + Coriolis only; anharmonic term omitted)"
    )
    statuses: list[str] = []
    cubic_cart = None
    if hessian_fn is not None:
        cubic_cart = cartesian_cubic_force_field(hessian_fn, coords_ang, fd_delta_cubic)
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
            cubic_cart=cubic_cart,
        )
        total_near_degen_skips += res_info.get("near_degen_skips", 0)
        statuses.append(str(res_info.get("anharmonic_status", "unknown")))
        table[name] = {
            comp: {
                "alpha_sum_mhz": alpha_sum[comp],
                "sigma_mhz": sigma[comp],
                "source": "harmonic_hessian",
                "method": method,
                "notes": notes,
            }
            for comp in ("A", "B", "C")
        }
    return table, {
        "total_near_degen_skips": total_near_degen_skips,
        "anharmonic_statuses": statuses,
    }
