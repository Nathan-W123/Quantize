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

# Assumed size of the cubic term, relative to the harmonic one, when it has not
# been computed. See the sigma block in compute_harmonic_alpha.
_OMITTED_CUBIC_SCALE = 3.0

# Half-range of the linear-molecule intra-pair (l-resonance) coefficient, in
# units of the -2B_e^2/omega pair term. Pinned by the one-structure consistency
# scan on fluoroacetylene's six measured B values at B3LYP/6-31G(d): c = 0
# gives r(C-F) = 1.27641 A, ~2.5 mA below r_s -- exactly where an equilibrium
# bond should sit -- while c = +1 lands 3 mA above it; the one-structure RMS
# (0.039 / 0.024 / 0.067 MHz at c = 0 / +0.5 / +1) cannot separate |c| <= 0.5,
# so c defaults to 0 and this half-range carries the ignorance as sigma.
_LINEAR_PAIR_HALF_RANGE = 0.5

_NONCONVERGENT_POLICIES = {"warn", "inflate", "drop"}


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
    mode_derivs=None,
    nm_sigma_fraction: float = 0.15,
    linear_pair_coeff=None,
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
    mode_derivs   : optional output of :func:`normal_mode_hessian_derivatives`.
                    Preferred over the Cartesian tensor: rotation-free
                    displacements with per-mode ZPE-scaled steps, which is what
                    keeps the cubic term meaningful on a noisy DFT surface.
                    Computed once with the reference masses and reused across
                    isotopologues via the rigid-completed mode decomposition.
    nm_sigma_fraction : fractional uncertainty applied to the correction when
                    the normal-mode cubic term is present and its own noise
                    diagnostic is clean -- the B3LYP-literature scale for VPT2
                    alphas, in place of the 100%-of-cubic floor that the
                    unvalidated Cartesian path earns.

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
    # Degenerate-bend pairs of a linear molecule. The generic Coriolis sum
    # skips the intra-pair term (its denominator is zero by symmetry), but for
    # a linear molecule that term is not negligible noise -- it is the
    # l-resonance contribution to alpha, of order 2 B_e^2/omega per bend
    # component, which at fluoroacetylene's scale is tens of MHz. It enters
    # with a closed-form coefficient in the linear-molecule VPT2 literature;
    # here the coefficient is a parameter, pinned empirically by requiring the
    # corrected constants of all measured isotopologues to be reproducible by
    # ONE rigid structure (see test/scan provenance at the call sites), because
    # an unverified textbook coefficient is exactly the kind of thing this
    # module has been bitten by.
    is_lin = _rigid_mode_count(coords, masses) == 5
    pair_partner = {}
    if is_lin:
        for i in range(n_vib):
            for j in range(i + 1, n_vib):
                if abs(omega_cm[i] - omega_cm[j]) < 2.0:
                    pair_partner[i] = j
                    pair_partner[j] = i
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
            if (linear_pair_coeff is not None and r in pair_partner
                    and K in (1, 2)):
                # perpendicular components only: for a linear molecule the
                # A-like eigenvalue is not a rotational constant at all.
                alpha_cor_cm[K, r] += (float(linear_pair_coeff)
                                       * -2.0 * B_e_cm[K] ** 2 / omega_cm[r])
    alpha_cor = alpha_cor_cm * _CM_TO_MHZ

    # ── Term 3: anharmonic (cubic force constants) ──────────────────────────
    alpha_anh = np.zeros((3, n_vib))
    alpha_anh_err = None
    anh_status = "not_requested"
    nm_noise_ratio = None
    if mode_derivs is not None:
        try:
            phi, phi_err = phi_semidiagonal_for_masses(
                mode_derivs, hess_bohr, masses, L_mw)
            alpha_anh = _anharmonic_alpha(
                None, masses, L_mw, vib_evals, dB1_mhz_all, zpe_amp, phi=phi,
            )
            # The finite-difference error of each phi_rrs, pushed through the
            # SAME contraction as alpha itself. |dB1| keeps every contribution
            # additive: errors from shared displaced Hessians are correlated,
            # so signed cancellation cannot be counted on.
            alpha_anh_err = zpe_amp[None, :] * np.einsum(
                "Ks,rs,s->Kr", np.abs(dB1_mhz_all), phi_err,
                1.0 / vib_evals, optimize=True,
            )
            anh_status = "cubic_nm"
            nm_noise_ratio = float(mode_derivs.get("noise_ratio", float("nan")))
        except Exception as exc:                      # noqa: BLE001 - degrade, don't abort
            anh_status = f"failed: {type(exc).__name__}: {exc}"
    else:
        if cubic_cart is None and hessian_fn is not None:
            try:
                cubic_cart = cartesian_cubic_force_field(hessian_fn, coords, fd_delta_cubic)
            except Exception as exc:                  # noqa: BLE001 - degrade, don't abort
                anh_status = f"cubic_ff_failed: {type(exc).__name__}: {exc}"
        if cubic_cart is not None:
            try:
                alpha_anh = _anharmonic_alpha(
                    np.asarray(cubic_cart, dtype=float), masses, L_mw,
                    vib_evals, dB1_mhz_all, zpe_amp,
                )
                anh_status = "cubic_fd"
            except Exception as exc:                  # noqa: BLE001 - degrade, don't abort
                anh_status = f"failed: {type(exc).__name__}: {exc}"

    alpha_total = alpha_cent + alpha_cor + alpha_anh
    alpha_sum = alpha_total.sum(axis=1)

    # Uncertainty on a truncated perturbation series is estimated by the size of
    # the last term retained -- here the cubic contribution. A flat fraction of
    # the total is not defensible: for H2O the cubic term is 4.9x the harmonic
    # one for B and 10.6x for C, so the series is barely converging there and a
    # 2% error bar understates the truth by more than an order of magnitude.
    # Getting this wrong is not cosmetic; the fit weights each target by 1/sigma²,
    # so an overconfident sigma lets a biased target drag the structure.
    anh_sum = alpha_anh.sum(axis=1)
    harm_sum = (alpha_cent + alpha_cor).sum(axis=1)

    # Convergence of the perturbation series, per component. alpha is a
    # truncated expansion; when the cubic term exceeds the harmonic one the
    # series is not converging and the correction cannot be trusted, however
    # small its formal uncertainty looks.
    #
    # This matters more than it appears. A biased target is not the same as an
    # uncertain one: when the spectral block is exactly- or under-determined the
    # fit drives its residual to zero and the weights drop out entirely, so no
    # sigma model can protect against a bad correction. Knowing which components
    # to distrust -- or drop -- is the only defence.
    anh_err_sum = (np.abs(alpha_anh_err).sum(axis=1)
                   if alpha_anh_err is not None else None)

    anh_ratio = {}
    nonconvergent = []
    for i, k in enumerate(labels):
        denom = abs(float(harm_sum[i]))
        ratio = float("inf") if denom == 0.0 else abs(float(anh_sum[i])) / denom
        anh_ratio[k] = ratio
        if anh_status == "cubic_nm":
            # With a validated cubic source, cubic > harmonic is normal VPT2
            # physics (water's B: 4.9x), not divergence. What condemns a
            # component here is its own propagated finite-difference error
            # exceeding the term it belongs to -- a per-component verdict, so
            # a linear molecule's quartic-dominated bend rows (whose asymmetry
            # once condemned every component through a global scalar) only
            # taint the components they actually feed -- or a ratio so large
            # no series survives it.
            if (anh_err_sum is not None
                    and float(anh_err_sum[i]) > max(abs(float(anh_sum[i])), 1.0)) \
                    or ratio > 20.0:
                nonconvergent.append(k)
        elif ratio > 1.0:
            nonconvergent.append(k)

    if anh_status == "cubic_nm":
        sigma_vals = {
            k: max(
                abs(float(alpha_sum[i])) * nm_sigma_fraction,
                float(anh_err_sum[i]),
                1.0,
            )
            for i, k in enumerate(labels)
        }
    elif anh_status == "cubic_fd":
        sigma_vals = {
            k: max(
                abs(float(alpha_sum[i])) * sigma_fraction,
                abs(float(anh_sum[i])),
                1.0,
            )
            for i, k in enumerate(labels)
        }
    else:
        # The cubic term was not computed, so the size of the dominant
        # contribution is unknown. 100% is not conservative enough: measured
        # against the reference molecules the cubic term is 2.6x the harmonic one
        # for CO and 5.6x / 10.7x for H2O's B and C, so an uncertainty equal to
        # the harmonic term alone would understate the error. _OMITTED_CUBIC_SCALE
        # is an empirical floor, not a derived bound -- computing the cubic term
        # is strongly preferred to relying on it.
        sigma_vals = {
            k: max(
                abs(float(alpha_sum[i])) * max(sigma_fraction, _OMITTED_CUBIC_SCALE),
                1.0,
            )
            for i, k in enumerate(labels)
        }

    if pair_partner and anh_status in ("cubic_nm", "cubic_fd"):
        # A linear molecule's degenerate bends carry an intra-pair
        # (l-resonance) alpha term of the form c * (-2 B_e^2/omega) per bend
        # component that the generic asymmetric-top sum skips (zero
        # denominator). Its coefficient is applied above when supplied; what
        # cannot be removed is the ignorance about c itself, pinned empirically
        # to a half-range of _LINEAR_PAIR_HALF_RANGE by the one-structure
        # consistency scan on fluoroacetylene's six measured isotopologues.
        # That ignorance is a systematic on the summed alpha, so it widens
        # sigma in quadrature for the two real rotational constants.
        for K in (1, 2):
            band_cm = _LINEAR_PAIR_HALF_RANGE * sum(
                2.0 * B_e_cm[K] ** 2 / omega_cm[r] for r in pair_partner)
            k = labels[K]
            sigma_vals[k] = float(np.hypot(sigma_vals[k], band_cm * _CM_TO_MHZ))

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
            "anharmonic_ratio": anh_ratio,
            "nm_noise_ratio": nm_noise_ratio,
            "alpha_anharmonic_err_mhz": (
                {k: float(anh_err_sum[i]) for i, k in enumerate(labels)}
                if anh_err_sum is not None else None
            ),
            "is_linear": bool(is_lin),
            "nonconvergent_components": nonconvergent,
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


def normal_mode_hessian_derivatives(
    hessian_fn,
    coords_ang,
    hess_bohr,
    masses_ref,
    min_freq_cm: float = 50.0,
    step_scale: float = 0.175,
    step_max_q: float = 0.05,
    step_min_q: float = 5e-3,
):
    """dH/dQ_r along each reference-mass normal mode, by central differences.

    This replaces Cartesian-axis displacement as the source of the cubic force
    field, for two measured reasons. A Cartesian step is part rigid rotation,
    and the Hessian changes under rotation even on an exact surface, so every
    displaced Hessian carries a large signal that is frame, not physics; normal
    modes are orthogonal to the rigid subspace at the reference geometry, so
    the same step size buys pure vibrational signal. And a fixed Cartesian step
    is the wrong size for every mode at once -- far into anharmonic territory
    for a soft torsion, down in the convergence noise for a C-H stretch. Here
    each step is a fixed fraction of that mode's zero-point amplitude, which
    equalises signal-to-noise across modes. On the same B3LYP surface the
    Cartesian scheme produced cubic/harmonic ratios of 233-990 (numerical
    garbage, correctly refused by the convergence gate); this scheme is what
    the VPT2 literature uses.

    The default step_scale comes from a measured step-halving study on the
    B3LYP/6-31G(d) formyl fluoride surface: the permutation asymmetry fell
    0.779 -> 0.191 -> 0.094 across scales 0.35 / 0.175 / 0.0875 -- the clean
    delta-squared signature of truncation, not noise -- while the alpha values
    themselves moved 0.1-0.3%. 0.175 keeps the truncation bound under ~0.2
    with steps still comfortably above the surface-noise floor.

    Steps are in the same Q units as everything else here (Angstrom sqrt(amu)),
    clamped to [step_min_q, step_max_q] so soft modes do not wander out of the
    cubic regime and stiff modes stay above the Hessian's own noise floor.

    Costs 2(3N-6) Hessian evaluations, against 6N for the Cartesian scheme.

    Returns a dict with the derivative stack ``dH`` (n_vib, 3N, 3N), the
    Cartesian displacement per unit Q ``D_ref`` (3N, n_vib), the reference
    frequencies, the steps taken, and a permutation-asymmetry noise measure:
    phi3[r, s, t] = D_s^T (dH/dQ_r) D_t equals the true third derivative
    contracted three ways, which is symmetric in (r, s); the antisymmetric part
    is pure finite-difference error, so it is a free, per-run error estimate
    rather than an assumption.
    """
    coords = np.asarray(coords_ang, dtype=float)
    masses = np.asarray(masses_ref, dtype=float)
    omega_cm, L_mw = _normal_modes(
        hess_bohr, masses, n_rigid=_rigid_mode_count(coords, masses)
    )
    keep = omega_cm >= min_freq_cm
    omega_cm, L_mw = omega_cm[keep], L_mw[:, keep]
    n_vib = len(omega_cm)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    D_ref = L_mw * inv_sqrt_m[:, None]                 # (3N, n_vib), dx per unit Q

    zpe_q = np.sqrt(_ZPE_AMP / omega_cm)               # zero-point amplitude, A*sqrt(amu)
    steps = np.clip(step_scale * zpe_q, step_min_q, step_max_q)

    n_cart = coords.size
    dH = np.zeros((n_vib, n_cart, n_cart))
    for r in range(n_vib):
        dx = (steps[r] * D_ref[:, r]).reshape(coords.shape)
        h_p = np.asarray(hessian_fn(coords + dx), dtype=float)
        h_m = np.asarray(hessian_fn(coords - dx), dtype=float)
        dH[r] = (h_p - h_m) / (2.0 * steps[r])

    phi3 = np.einsum("rij,is,jt->rst", dH, D_ref, D_ref, optimize=True)
    asym = float(np.max(np.abs(phi3 - np.swapaxes(phi3, 0, 1))))
    scale = float(np.median(np.abs(phi3[np.abs(phi3) > 0]))) if np.any(phi3) else 0.0
    return {
        "dH": dH,
        "D_ref": D_ref,
        "omega_cm_ref": omega_cm,
        "steps_q": steps,
        "coords_ang": coords,
        "masses_ref": masses,
        "noise_asym": asym,
        "noise_scale": scale,
        "noise_ratio": (asym / scale) if scale > 0 else float("inf"),
    }


def _rigid_directions_with_derivatives(hess_bohr, coords_ang):
    """Unit rigid-body directions and the analytic dH along each.

    Translations leave the Hessian exactly unchanged. An infinitesimal rotation
    by angle theta about axis g conjugates it, H -> R H R^T blockwise, so the
    derivative along the *unit-normalised* rotation displacement field w = G x
    is [G, H] / |w|, with G = I_N (x) g. This is what lets an isotopologue's
    normal mode -- which is never exactly inside the reference isotopologue's
    vibrational span, because mass-weighting differs -- be completed exactly
    instead of truncated.
    """
    coords = np.asarray(coords_ang, dtype=float)
    n = coords.shape[0]
    H = np.asarray(hess_bohr, dtype=float)
    dirs, derivs = [], []
    for k in range(3):                                  # translations
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        v = t.ravel()
        dirs.append(v / np.linalg.norm(v))
        derivs.append(np.zeros_like(H))
    centered = coords - coords.mean(axis=0)
    for k in range(3):                                  # rotations
        g = np.zeros((3, 3))
        a, b = [(1, 2), (2, 0), (0, 1)][k]
        g[a, b], g[b, a] = -1.0, 1.0
        w = (centered @ g.T).ravel()
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-10:
            continue
        G = np.kron(np.eye(n), g)
        dirs.append(w / nrm)
        derivs.append((G @ H - H @ G) / nrm)
    return np.column_stack(dirs), derivs


def phi_semidiagonal_for_masses(mode_derivs, hess_bohr, masses_iso, L_mw_iso):
    """Semi-diagonal cubic constants phi_rrs in an isotopologue's own modes.

    The measured derivative field covers the reference isotopologue's
    vibrational directions. Another isotopologue's mode is a linear combination
    of those plus a small rigid component (mass-weighting moves the vibrational
    subspace), and the rigid part's contribution is supplied analytically by
    the rotation commutator rather than dropped -- dropping it is a silent
    error of exactly the kind that produced the Cartesian scheme's garbage.

    Returns ``(phi, phi_err)``. Every semi-diagonal constant is measured twice
    -- phi_rrs is D_r (dH/dQ_r) D_s from the r-displacement and
    D_r (dH/dQ_s) D_r from the s-displacement, equal in exact arithmetic -- so
    the elementwise disagreement ``phi_err`` is a free, per-constant bound on
    the finite-difference error of exactly the numbers the alpha formula
    consumes. That locality matters: one bad displacement direction (a linear
    molecule's quartic-dominated bend, say) contaminates its own entries, not
    a global scalar that condemns the whole tensor.
    """
    D_ref = mode_derivs["D_ref"]
    dH = mode_derivs["dH"]
    coords = mode_derivs["coords_ang"]
    masses = np.asarray(masses_iso, dtype=float)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    D_iso = L_mw_iso * inv_sqrt_m[:, None]              # (3N, n_vib_iso)

    R_dirs, R_derivs = _rigid_directions_with_derivatives(hess_bohr, coords)
    basis = np.column_stack([D_ref, R_dirs])            # (3N, n_vib_ref + n_rigid)
    coef, *_ = np.linalg.lstsq(basis, D_iso, rcond=None)
    n_ref = D_ref.shape[1]

    n_iso = D_iso.shape[1]
    phi = np.zeros((n_iso, n_iso))
    phi_alt = np.zeros((n_iso, n_iso))
    for r in range(n_iso):
        dH_r = np.einsum("k,kij->ij", coef[:n_ref, r], dH, optimize=True)
        for j, dHrig in enumerate(R_derivs):
            c = coef[n_ref + j, r]
            if c != 0.0:
                dH_r = dH_r + c * dHrig
        phi[r, :] = D_iso[:, r] @ dH_r @ D_iso
        # Second estimate of phi_ssr for every s, from THIS displacement:
        # D_s (dH/dQ_r) D_s.
        phi_alt[:, r] = np.einsum("is,ij,js->s", D_iso, dH_r, D_iso,
                                  optimize=True)
    err = np.abs(phi - phi_alt)
    if n_iso > 1:
        # The diagonal phi_rrr has no second estimate -- both contractions use
        # the same displaced Hessians -- so stand in the demonstrated error
        # scale of that mode's own row and column.
        off = err.copy()
        np.fill_diagonal(off, 0.0)
        diag_proxy = np.maximum(off.max(axis=0), off.max(axis=1))
        err[np.diag_indices(n_iso)] = diag_proxy
    return phi, err


def _anharmonic_alpha(cubic_cart, masses, L_mw, vib_evals, dB1_mhz, zpe_amp,
                      phi=None):
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
    lam = np.asarray(vib_evals, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("non-positive vibrational eigenvalue in anharmonic alpha")

    if phi is None:
        masses = np.asarray(masses, dtype=float)
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
        D = L_mw * inv_sqrt_m[:, None]                # (3N, n_vib)
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
    nonconvergent_policy: str = "warn",
    cubic_scheme: str = "cartesian",
    linear_pair_coeff=None,
) -> tuple[dict, dict]:
    """
    Build a correction_table dict (compatible with parse_correction_table)
    from the alpha computed from the Hessian.

    When ``hessian_fn`` is supplied the cubic (anharmonic) term is included.
    The semi-diagonal cubic force constants depend only on the electronic PES,
    not on nuclear masses, so they are computed once and reused across every
    isotopologue rather than once per isotopologue.

    ``nonconvergent_policy`` decides what happens to a component whose cubic
    term exceeds its harmonic one, i.e. whose perturbation series is not
    converging:

      ``"warn"``    keep it and annotate the notes (default; preserves
                    historical behaviour)
      ``"inflate"`` additionally scale its sigma by the divergence ratio, on the
                    geometric assumption that the first omitted term is larger
                    than the last retained one by roughly that factor
      ``"drop"``    omit the component from the table entirely, so the fit falls
                    back to whatever the uncorrected data or the quantum prior
                    provides

    ``"inflate"`` only bites when the spectral block is over-determined; where it
    is exactly determined the fit reproduces its targets regardless of weight and
    only ``"drop"`` changes the outcome.
    """
    policy = str(nonconvergent_policy or "warn").strip().lower()
    if policy not in _NONCONVERGENT_POLICIES:
        raise ValueError(
            f"Unknown nonconvergent_policy '{nonconvergent_policy}'. "
            f"Valid: {sorted(_NONCONVERGENT_POLICIES)}"
        )
    ref_masses_lin = list(isotopologues[0].get("masses", [])) if isotopologues else []
    if hessian_fn is not None and ref_masses_lin and linear_pair_coeff is None and \
            _rigid_mode_count(np.asarray(coords_ang, float), ref_masses_lin) == 5:
        # Linear molecule: the degenerate bends carry an intra-pair
        # (l-resonance) alpha term the asymmetric-top Coriolis sum must skip.
        # Its coefficient defaults to the empirically pinned c = 0 -- the
        # one-structure scan over fluoroacetylene's six measured B values put
        # the r_e-plausible answer there (see _LINEAR_PAIR_HALF_RANGE) -- and
        # compute_harmonic_alpha widens sigma by the half-range of that scan,
        # so the remaining coefficient ignorance is carried as uncertainty
        # rather than either guessed at or used as a reason to withhold the
        # entire cubic correction. (An earlier withhold-everything gate here
        # turned out to be treating the symptom: the real 39-mA failure was a
        # sigma explosion from a global noise scalar, fixed in the per-entry
        # phi error propagation.)
        linear_pair_coeff = 0.0
    table: dict = {}
    total_near_degen_skips = 0
    method = "VPT2_semidiag" if hessian_fn is not None else "harmonic_VR"
    notes = (
        "alpha from Hessian (harmonic + Coriolis + cubic anharmonic)"
        if hessian_fn is not None
        else "alpha from Hessian (harmonic + Coriolis only; anharmonic term omitted)"
    )
    statuses: list[str] = []
    nonconvergent: dict = {}
    dropped: dict = {}
    cubic_cart = None
    mode_derivs = None
    if hessian_fn is not None:
        if str(cubic_scheme).lower() == "normal_mode" and isotopologues:
            ref_masses = list(isotopologues[0].get("masses", []))
            if ref_masses:
                mode_derivs = normal_mode_hessian_derivatives(
                    hessian_fn, coords_ang, hess_bohr, ref_masses,
                    min_freq_cm=min_freq_cm,
                )
        if mode_derivs is None:
            cubic_cart = cartesian_cubic_force_field(
                hessian_fn, coords_ang, fd_delta_cubic)
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
            mode_derivs=mode_derivs,
            linear_pair_coeff=linear_pair_coeff,
        )
        total_near_degen_skips += res_info.get("near_degen_skips", 0)
        statuses.append(str(res_info.get("anharmonic_status", "unknown")))
        ratios = res_info.get("anharmonic_ratio", {})
        bad = set(res_info.get("nonconvergent_components", []))
        for comp in bad:
            nonconvergent.setdefault(name, {})[comp] = float(ratios.get(comp, float("nan")))
        entries: dict = {}
        for comp in ("A", "B", "C"):
            if res_info.get("is_linear") and comp == "A":
                # A linear molecule has no A constant; the first inertia
                # eigenvalue is ~0 and the "alpha" computed from it is
                # meaningless at any sigma. No species should be fitting an
                # A component here, so write nothing rather than nonsense.
                continue
            ratio = float(ratios.get(comp, float("nan")))
            if comp in bad and policy == "drop":
                dropped.setdefault(name, []).append(comp)
                continue
            # parse_correction_table defines sigma_mhz as the uncertainty on the
            # correction itself, and the correction is delta = 0.5*alpha_sum
            # (vpt2_delta_b), so the alpha uncertainty is halved to match.
            sigma_mhz = 0.5 * sigma[comp]
            note = notes
            if comp in bad:
                note = (
                    f"{notes}; WARNING cubic/harmonic = {ratio:.1f} -- "
                    "perturbation series not converging, correction unreliable"
                )
                if policy == "inflate" and np.isfinite(ratio):
                    sigma_mhz *= max(1.0, ratio)
                    note += "; sigma inflated by the divergence ratio"
            entries[comp] = {
                "alpha_sum_mhz": alpha_sum[comp],
                "sigma_mhz": sigma_mhz,
                "source": "harmonic_hessian",
                "method": method,
                "notes": note,
            }
        table[name] = entries
    return table, {
        "total_near_degen_skips": total_near_degen_skips,
        "anharmonic_statuses": statuses,
        "nonconvergent": nonconvergent,
        "nonconvergent_policy": policy,
        "dropped_components": dropped,
    }
