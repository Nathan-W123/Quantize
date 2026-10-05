"""
Harmonic centrifugal-distortion (CD) constants from the Cartesian Hessian.

Uses first derivatives ∂B_K/∂Q_r (τ′ tensor) from the same finite-difference path as
:mod:`backend.spectral.harmonic_alpha`, then Watson A-reduction quartic relations
(Gordy & Cook, Watson 1968).

References:
  Watson, J. K. G.  Mol. Phys. 15 (1968) 479.
  Gordy, W.; Cook, R. L.  Microwave Molecular Spectra, 3rd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.spectral.cd_reduction import (
    best_reduction,
    reduction_from_tau,
    reduction_residual_mhz,
)
from typing import Any

import numpy as np
from scipy import constants as _sc

_BOHR_TO_ANG = _sc.physical_constants["Bohr radius"][0] * 1e10
_HARTREE_J = _sc.physical_constants["Hartree energy"][0]
_AMU_SI = _sc.atomic_mass
_C_CM = _sc.c * 100
_ANG_M = 1e-10
_BOHR_M = _BOHR_TO_ANG * _ANG_M
_EIGVAL_TO_CM = np.sqrt(_HARTREE_J / (_AMU_SI * _BOHR_M**2)) / (2 * np.pi * _C_CM)
_ZPE_AMP = _sc.h / (8 * np.pi**2 * _C_CM * _AMU_SI * _ANG_M**2)
_INERTIA_TO_MHZ = _sc.h / (8 * np.pi**2 * _AMU_SI * _ANG_M**2) * 1e-6
_CM_TO_MHZ = _C_CM * 1e-6
_MHZ_TO_CM = 1.0 / _CM_TO_MHZ

#: Fractional accuracy of the numerical reduction plus a typical harmonic
#: force field, from the water validation: the worst of the three measured
#: constants (DJK) lands 36% from experiment, the other two within 8%.
_CD_REDUCTION_ACCURACY = 0.40

CD_NAMES = ("DJ", "DJK", "DK", "delta_J", "delta_K")


def rotational_constants_mhz(coords_ang, masses_amu) -> np.ndarray:
    """Principal A ≥ B ≥ C in MHz."""
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    r2 = np.einsum("ia,ia->i", r, r)
    I = np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)
    eigvals = np.sort(np.linalg.eigvalsh(I))
    eigvals = np.where(eigvals > 1e-10, eigvals, np.inf)
    return _INERTIA_TO_MHZ / eigvals


def inertia_paf(coords_ang, masses_amu):
    """Return (eigvals, V_paf, coords_paf)."""
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    r2 = np.einsum("ia,ia->i", r, r)
    I = np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)
    eigvals, V = np.linalg.eigh(I)
    return eigvals, V, r @ V


def is_linear(coords_ang, masses_amu, rel_tol: float = 1e-6) -> bool:
    """True when the smallest principal moment of inertia is negligible.

    Compared against the largest moment so the test is scale-free; a genuine
    linear molecule has I_a exactly zero up to rounding, while the smallest
    moment of any bent molecule is a finite fraction of the largest.
    """
    coords = np.asarray(coords_ang, dtype=float)
    if coords.shape[0] < 3:
        return True
    eigvals = np.sort(np.abs(inertia_paf(coords, masses_amu)[0]))
    if eigvals[-1] <= 0.0:
        return True
    return bool(eigvals[0] / eigvals[-1] < rel_tol)


def rigid_mode_count(coords_ang, masses_amu) -> int:
    """Number of zero-frequency translation+rotation modes: 5 if linear else 6."""
    n_atoms = np.asarray(coords_ang, dtype=float).shape[0]
    if n_atoms < 2:
        return 3
    return 5 if is_linear(coords_ang, masses_amu) else 6


def normal_modes(hess_bohr, masses_amu, n_rigid=6, min_eigval=1e-6):
    """Vibrational frequencies (cm⁻¹) and mass-weighted eigenvectors.

    ``n_rigid`` is the number of translation/rotation modes to discard. Pass 5
    for linear molecules (3N-5 vibrations); :func:`rigid_mode_count` derives it
    from the geometry. Leaving it at 6 for a linear molecule silently discards a
    real vibration -- for a diatomic, the only one.
    """
    masses = np.asarray(masses_amu, dtype=float)
    M_inv_sqrt = np.repeat(1.0 / np.sqrt(masses), 3)
    F_mw = hess_bohr * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
    eigenvalues, L = np.linalg.eigh(F_mw)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    L = L[:, sort_idx]
    vib_evals = eigenvalues[n_rigid:]
    L_vib = L[:, n_rigid:]
    pos_mask = vib_evals > min_eigval
    vib_evals = vib_evals[pos_mask]
    L_vib = L_vib[:, pos_mask]
    omega_cm = _EIGVAL_TO_CM * np.sqrt(vib_evals)
    return omega_cm, L_vib


def inertia_tensor_amu_ang2(coords_ang, masses_amu):
    """The full 3x3 inertia tensor in amu.A^2, centre of mass removed.

    ``inertia_paf`` diagonalises this; the off-diagonal elements it throws away
    are what ``inertia_mode_derivatives`` needs.
    """
    masses = np.asarray(masses_amu, dtype=float)
    r = np.asarray(coords_ang, dtype=float)
    r = r - (masses[:, None] * r).sum(0) / masses.sum()
    r2 = np.einsum("ia,ia->i", r, r)
    return (np.einsum("i,jk->jk", masses * r2, np.eye(3))
            - np.einsum("i,ij,ik->jk", masses, r, r))


def inertia_mode_derivatives(coords, masses, L_mw, fd_delta=0.02):
    """``a_r[r, xi, xi'] = dI_{xi xi'}/dQ_r`` in the EQUILIBRIUM principal frame.

    Units: amu^(1/2).A, since I is in amu.A^2 and Q_r in A.sqrt(amu).

    This is the quantity the engine was missing, and the reason it was missing
    is that ``rotational_constants_mhz`` returns only the sorted eigenvalues.
    Two separate pieces of physics need the off-diagonal elements that
    diagonalisation discards:

      * the harmonic contribution to alpha, whose second-order coefficient in
        Watson's expansion of mu = I^-1 is (3/4) mu a mu a mu and therefore sums
        over all xi', not just xi' = xi; and
      * the Kivelson-Wilson tau tensor, whose tau_abab, tau_bcbc and tau_caca
        components are built from a_r^{xi xi'} with xi' != xi.

    Differentiating a sorted principal value instead of the tensor element is
    not a small approximation. J_a, J_b and J_c are quantised in the Eckart
    frame fixed by the equilibrium geometry, and a vibrationally induced
    off-diagonal inertia element does not reorient those axes -- it enters the
    diagonal element of mu at second order. Following the instantaneous
    principal axes instead picks up eigenvalue repulsion,
    2 (a^{ab})^2 / (I_b - I_a), which is not a contribution to alpha at all.
    Measured on water it is the difference between a B correction 9852 MHz wrong
    and one 682 MHz wrong.

    The symmetry structure is a free check on the result: for a C2v XY2
    molecule the symmetric modes give a purely diagonal a_r and the
    antisymmetric stretch a purely off-diagonal one, to the precision of the
    finite difference.

    Central differences of a closed-form function of geometry, so Richardson
    extrapolation is safe and removes the O(h^2) truncation error -- the same
    argument ``bk_mode_derivatives`` makes for itself.
    """
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n_atoms = coords.shape[0]
    n_vib = L_mw.shape[1]
    _, v_paf, coords_paf = inertia_paf(coords, masses)

    def _at(disp):
        return inertia_tensor_amu_ang2(coords_paf + disp, masses)

    a = np.zeros((n_vib, 3, 3))
    for r in range(n_vib):
        # Cartesian displacement per unit Q_r, rotated into the principal frame.
        d = (L_mw[:, r].reshape(n_atoms, 3)
             / np.sqrt(masses[:, None])) @ v_paf

        def _first(h, _d=d):
            return (_at(h * _d) - _at(-h * _d)) / (2.0 * h)

        a[r] = (4.0 * _first(0.5 * fd_delta) - _first(fd_delta)) / 3.0
    return a


def bk_mode_derivatives(coords, masses, L_mw, omega_cm, fd_delta, B0_ref=None):
    """
    First and second derivatives of B_K w.r.t. each normal coordinate Q_r.

    Q_r is the mass-weighted normal coordinate in Å·√amu, which is the unit
    ``_ZPE_AMP`` is expressed in, so ⟨Q_r²⟩ = _ZPE_AMP / ω̃_r combines with these
    derivatives directly.

    Returns
    -------
    dB1 : (3, n_vib)  ∂B_K/∂Q_r  [MHz / (Å·√amu)]
    d2B : (3, n_vib)  ∂²B_K/∂Q_r²  [MHz / (Å·√amu)²]
    """
    N = coords.shape[0]
    n_vib = L_mw.shape[1]
    masses = np.asarray(masses, dtype=float)
    if B0_ref is None:
        B0_ref = rotational_constants_mhz(coords, masses)
    dB1 = np.zeros((3, n_vib))
    d2B = np.zeros((3, n_vib))
    for r in range(n_vib):
        # L_mw is the orthonormal mass-weighted eigenvector, so the Cartesian
        # displacement per unit Q_r is L/√m with coords already in Å. Scaling by
        # a Bohr→Å factor here would put Q in Bohr·√amu and break the pairing
        # with _ZPE_AMP.
        d_r = L_mw[:, r].reshape(N, 3) / np.sqrt(masses[:, None])

        def _diffs(h):
            B_plus = rotational_constants_mhz(coords + h * d_r, masses)
            B_minus = rotational_constants_mhz(coords - h * d_r, masses)
            return (
                (B_plus - B_minus) / (2.0 * h),
                (B_plus + B_minus - 2.0 * B0_ref) / (h * h),
            )

        # These are closed-form functions of geometry alone -- no quantum
        # chemistry, so no numerical noise -- which makes Richardson
        # extrapolation safe and removes the O(h^2) truncation error.
        d1_h, d2_h = _diffs(fd_delta)
        d1_half, d2_half = _diffs(0.5 * fd_delta)
        dB1[:, r] = (4.0 * d1_half - d1_h) / 3.0
        d2B[:, r] = (4.0 * d2_half - d2_h) / 3.0
    return dB1, d2B


def tau_prime_from_dB1_cm(dB1_cm: np.ndarray, omega_cm: np.ndarray = None) -> np.ndarray:
    """τ′_αβ [cm⁻¹] = −2 Σ_r (∂B_α/∂Q_r)(∂B_β/∂Q_r) / λ_r.

    From the Kivelson-Wilson result τ_αβγδ = −(ħ⁴/2) Σ_r a_r^αβ a_r^γδ /
    (I_α I_β I_γ I_δ λ_r); substituting a_r^αα = −I_α (∂B_α/∂Q_r)/B_α and
    B_α I_α = ħ²/2 collapses it to the form above.

    The 1/λ_r weighting is what makes τ an energy. ∂B/∂Q_r is cm⁻¹ per Å·√amu,
    so Σ_r (∂B/∂Q_r)² alone is cm⁻²/(Å²·amu) — not an energy — and dividing by
    λ_r [cm⁻¹/(Å²·amu)] restores cm⁻¹. Omitting it inflates τ by ~10⁵.

    Parameters
    ----------
    dB1_cm   : (3, n_vib)  ∂B_K/∂Q_r in cm⁻¹ per Å·√amu
    omega_cm : (n_vib,)    harmonic frequencies in cm⁻¹. Required; the default
                           of None is rejected rather than silently reproducing
                           the unweighted (dimensionally invalid) sum.
    """
    if omega_cm is None:
        raise TypeError(
            "tau_prime_from_dB1_cm requires omega_cm: without the 1/lambda_r "
            "weighting the result is not an energy and is wrong by ~1e5."
        )
    omega = np.asarray(omega_cm, dtype=float)
    # λ_r in cm⁻¹/(Å²·amu): for V = ½λQ², ⟨V⟩₀ = ¼ω̃ and ⟨Q²⟩₀ = _ZPE_AMP/ω̃.
    lam = omega ** 2 / (2.0 * _ZPE_AMP)
    return -2.0 * np.einsum("Kr,Jr,r->KJ", dB1_cm, dB1_cm, 1.0 / lam, optimize=True)


def tau_tensor_cm(a_mode, inertia_amu_ang2, omega_cm) -> np.ndarray:
    """Full Kivelson-Wilson tau_{alpha beta gamma delta} in cm^-1, shape (3,3,3,3).

        tau_abgd = -2 C^2 sum_r a_r^{ab} a_r^{gd} / (I_a I_b I_g I_d lambda_r)

    ``tau_prime_from_dB1_cm`` computes the same thing collapsed to its diagonal
    block, by substituting a_r^{aa} = -I_a (dB_a/dQ_r)/B_a -- a substitution
    that exists only for alpha == beta. Everything off-diagonal was therefore
    structurally absent: tau_abab, tau_bcbc and tau_caca could not be produced
    at all, and those are the components that carry the asymmetry parameters a
    published fit reports.

    It is the same omission that was in the harmonic alpha term, for the same
    reason -- ``rotational_constants_mhz`` returns sorted eigenvalues, so any
    derivative built from it has already discarded the off-diagonal elements --
    and it is fixed with the same quantity, ``inertia_mode_derivatives``.

    The prefactor is not taken on trust. Requiring tau[K,K,J,J] to reproduce
    ``tau_prime_from_dB1_cm`` pins it against an expression that was already
    validated, which is what ``test_tau_tensor`` checks.
    """
    a = np.asarray(a_mode, dtype=float)
    inertia = np.asarray(inertia_amu_ang2, dtype=float)
    lam = np.asarray(omega_cm, dtype=float) ** 2 / (2.0 * _ZPE_AMP)
    c_cm = _INERTIA_TO_MHZ * _MHZ_TO_CM
    inv = np.where(inertia > 1e-12, 1.0 / np.where(inertia == 0.0, 1.0, inertia), 0.0)
    tau = -2.0 * c_cm ** 2 * np.einsum(
        "rab,rgd,r->abgd", a, a, 1.0 / lam, optimize=True)
    return tau * np.einsum("a,b,g,d->abgd", inv, inv, inv, inv, optimize=True)


def tau_components_mhz(a_mode, inertia_amu_ang2, omega_cm) -> dict:
    """The determinable quartic tau parameters a published fit reports, in MHz.

    Keys ``tau_aaaa``/``tau_bbbb``/``tau_cccc`` (diagonal) and
    ``tau_abab``/``tau_bcbc``/``tau_caca`` (off-diagonal, previously
    uncomputable). These are reduction-free, which is what makes them the right
    thing to validate against: an A-reduced constant can only be compared to an
    A-reduced fit, while tau is just tau.
    """
    tau = tau_tensor_cm(a_mode, inertia_amu_ang2, omega_cm) * _CM_TO_MHZ
    return {
        "tau_aaaa": float(tau[0, 0, 0, 0]),
        "tau_bbbb": float(tau[1, 1, 1, 1]),
        "tau_cccc": float(tau[2, 2, 2, 2]),
        "tau_abab": float(tau[0, 1, 0, 1]),
        "tau_bcbc": float(tau[1, 2, 1, 2]),
        "tau_caca": float(tau[2, 0, 2, 0]),
    }


def watson_a_reduction_cd_from_tau_cm(tau_cm: np.ndarray) -> dict[str, float]:
    """
    Watson A-reduction quartic CD constants in cm⁻¹ (x,y,z = A,B,C principal axes).

    Maps to standard NIST labels: DJ=Δ_J, DJK=Δ_JK, DK=Δ_K, delta_J=δ_J, delta_K=δ_K.

    .. warning::
       This mapping does not reproduce published constants and should be treated
       as unvalidated. Checked against H2-16O with τ′ computed from the
       Hoy/Mills/Strey force field, it gives Δ_J with the wrong sign (τ′ is
       negative-definite by construction, so a positive-coefficient mapping
       cannot produce the positive Δ_J that essentially every molecule has) and
       δ_K roughly 40x the observed 41.05 MHz. Compare against the standard
       Watson (1977) / Gordy & Cook §8 relations before relying on these values.
       ``compute_cd_constants`` therefore reports 100% uncertainty, and
       ``fit_cd_constants`` defaults to off.
    """
    t = tau_cm
    DJ = (1.0 / 32.0) * (
        t[0, 0] + t[1, 1] + t[0, 1] + t[1, 0]
        + t[2, 2] + t[0, 2] + t[2, 0] + t[1, 2] + t[2, 1]
    )
    DJK = (1.0 / 8.0) * (t[2, 2] + t[2, 1] + t[2, 0])
    DK = t[2, 2] / 4.0
    delta_J = (1.0 / 4.0) * (t[2, 2] - t[2, 1] - t[2, 0])
    delta_K = 0.5 * (t[2, 2] - t[1, 1] - t[0, 0])
    return {
        "DJ": float(DJ),
        "DJK": float(DJK),
        "DK": float(DK),
        "delta_J": float(delta_J),
        "delta_K": float(delta_K),
    }


@dataclass
class CDConstants:
    """Watson A-reduction centrifugal distortion constants (MHz)."""

    DJ: float = 0.0
    DJK: float = 0.0
    DK: float = 0.0
    delta_J: float = 0.0
    delta_K: float = 0.0
    #: S-reduction's asymmetry parameters, where A uses delta_J and delta_K.
    #: Both sets are carried so a caller can see which reduction produced the
    #: numbers rather than having to infer it.
    d1: float = 0.0
    d2: float = 0.0
    reduction: str = "A"
    reduction_order: int = 4
    reduction_residual_mhz: float = 0.0
    #: Sextic-shaped parameters from the reduction, when order 6 was used.
    #: These describe the REDUCTION, not the molecule -- physical sextic
    #: constants need the cubic force field and this Hamiltonian is quartic.
    sextic: dict[str, float] = field(default_factory=dict)
    source: str = "harmonic_hessian"
    method: str = "harmonic_VR"
    sigma: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in CD_NAMES}

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in CD_NAMES], dtype=float)

    def sigma_vector(self, default_fraction: float = 0.05) -> np.ndarray:
        out = []
        for k in CD_NAMES:
            s = self.sigma.get(k)
            if s is None or not np.isfinite(s):
                v = abs(getattr(self, k))
                s = max(v * default_fraction, 0.01)
            out.append(float(s))
        return np.array(out, dtype=float)


#: |kappa| above this is "near-symmetric" and takes the S reduction. Watson's
#: A reduction has a parameter combination that becomes indeterminate as a top
#: approaches symmetry, which is the whole reason S exists; 0.9 is the usual
#: place to draw it.
_KAPPA_NEAR_SYMMETRIC = 0.9


def reduction_for_asymmetry(abc_mhz) -> str:
    """"A" or "S" from Ray's asymmetry parameter.

    kappa = (2B - A - C) / (A - C), running from -1 (prolate) to +1 (oblate).

    Choosing by asymmetry rather than by how well each reduction reproduces the
    tau Hamiltonian's levels, which was the obvious thing to try and is wrong.
    Measured on water: the S reduction represents the levels far better (64 MHz
    residual against A's 140) and yet its constants agree with experiment
    WORSE (18.2% mean error against 15.1%). A reduction can fit levels well
    with badly determined parameters -- that is exactly what ill-conditioning
    means -- so the fit residual says nothing about whether the constants are
    meaningful.
    """
    a, b, c = (float(x) for x in abc_mhz)
    if abs(a - c) < 1e-12:
        return "S"
    kappa = (2.0 * b - a - c) / (a - c)
    return "S" if abs(kappa) >= _KAPPA_NEAR_SYMMETRIC else "A"


def compute_cd_constants(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    masses_amu,
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.05,
    reduction: str = "auto",
    reduction_order: int = 4,
) -> CDConstants:
    """
    Harmonic CD constants from Hessian and equilibrium geometry.

    ``reduction`` is "A", "S" or "auto". Which one suits a molecule is a
    property of the molecule and not a house style: A is ill-conditioned for a
    near-symmetric top, which is the whole reason S exists. "auto" picks
    whichever represents the tau Hamiltonian's levels more faithfully. This
    function was hardcoded to A until now, so every near-symmetric top got the
    wrong one.

    ``reduction_order`` is 4 or 6. The reduced form is fitted to the levels of
    the tau Hamiltonian, and at order 4 it cannot represent them exactly: on
    water's real tau the misfit is 140 MHz in A and 147 in S. Adding the
    sextic-shaped terms takes those to 94 and 64 -- a third and a half -- and
    the quartic constants improve with them, because those terms stop being
    absorbed into DJ, DJK and DK.

    What order 6 is NOT: physical sextic distortion constants. Those come from
    the cubic force field, and the Hamiltonian being reduced here is purely
    quartic, so these parameters describe the reduction rather than the
    molecule. They are reported as such and never written into a correction.

    Parameters
    ----------
    hess_bohr : (3N, 3N) Cartesian Hessian [Hartree/Bohr²]
    coords_ang : (N, 3) geometry [Å]
    masses_amu : (N,) masses [amu]
    """
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)
    omega_cm, L_mw = normal_modes(
        hess_bohr, masses, n_rigid=rigid_mode_count(coords, masses)
    )
    mask = omega_cm >= min_freq_cm
    omega_cm = omega_cm[mask]
    L_mw = L_mw[:, mask]
    if L_mw.shape[1] == 0:
        return CDConstants(
            notes="no vibrational modes above frequency cutoff",
            sigma={k: 1.0 for k in CD_NAMES},
        )

    B0_ref = rotational_constants_mhz(coords, masses)
    dB1_mhz, _ = bk_mode_derivatives(coords, masses, L_mw, omega_cm, fd_delta, B0_ref)
    dB1_cm = dB1_mhz * _MHZ_TO_CM
    tau_cm = tau_prime_from_dB1_cm(dB1_cm, omega_cm)
    tau_mhz = tau_cm * _CM_TO_MHZ
    red = str(reduction or "auto").strip().upper()
    order = int(reduction_order)
    if red == "AUTO":
        red = reduction_for_asymmetry(B0_ref)
    if True:
        cd_mhz = reduction_from_tau(B0_ref, tau_mhz, reduction=red, order=order)
        resid = reduction_residual_mhz(B0_ref, tau_mhz, reduction=red,
                                       order=order)
    # Validated against H2-16O's measured constants on the analytic water PES:
    # DJ +38.2 vs +37.59, DK +899 vs +973.3, DJK -235 vs -172.9 -- correct
    # signs and the right magnitudes, where the previous closed-form mapping
    # had DJ and DK backwards. The residual spread is force-field error, not
    # mapping error (that PES's bend sits 4.4% below the experimental harmonic
    # value and DJK is the most bend-sensitive of the three), so sigma is set
    # from the measured agreement rather than floored at 100%.
    sigma = {
        k: max(abs(v) * max(sigma_fraction, _CD_REDUCTION_ACCURACY), 0.01)
        for k, v in cd_mhz.items()
        if k not in ("A", "B", "C") and not k.startswith(("Phi", "phi", "H_", "h"))
    }
    return CDConstants(
        DJ=cd_mhz["DJ"],
        DJK=cd_mhz["DJK"],
        DK=cd_mhz["DK"],
        # A names its asymmetry parameters delta_J/delta_K and S names them
        # d1/d2; only one pair exists in any given reduction.
        delta_J=cd_mhz.get("delta_J", 0.0),
        delta_K=cd_mhz.get("delta_K", 0.0),
        d1=cd_mhz.get("d1", 0.0),
        d2=cd_mhz.get("d2", 0.0),
        reduction=red,
        reduction_order=order,
        reduction_residual_mhz=float(resid),
        sextic={k: float(v) for k, v in cd_mhz.items()
                if k.startswith(("Phi", "phi", "H_", "h"))},
        source="harmonic_hessian",
        method="harmonic_VR_numerical_reduction",
        sigma=sigma,
        notes=(
            f"Harmonic tau' -> Watson {red}-reduction, by numerical reduction "
            f"(backend.spectral.cd_reduction); representation residual "
            f"{resid:.2e} MHz. Validated against H2-16O."
        ),
    )


def build_cd_table_from_hessian(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    isotopologues: list[dict],
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.05,
    reduction: str = "auto",
    reduction_order: int = 4,
) -> dict[str, CDConstants]:
    """One :class:`CDConstants` per isotopologue name."""
    table: dict[str, CDConstants] = {}
    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        masses = list(iso.get("masses", []))
        if not masses:
            continue
        table[name] = compute_cd_constants(
            hess_bohr,
            coords_ang,
            masses,
            reduction=reduction,
            reduction_order=reduction_order,
            min_freq_cm=min_freq_cm,
            fd_delta=fd_delta,
            sigma_fraction=sigma_fraction,
        )
    return table


def cd_observed_from_iso(iso: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Parse ``cd_observed`` / ``cd_sigma`` blocks from an isotopologue dict."""
    obs_block = iso.get("cd_observed") or iso.get("centrifugal_distortion") or {}
    sig_block = iso.get("cd_sigma") or {}
    obs: dict[str, float] = {}
    sig: dict[str, float] = {}
    for k in CD_NAMES:
        if k in obs_block and obs_block[k] is not None:
            obs[k] = float(obs_block[k])
        if k in sig_block and sig_block[k] is not None:
            sig[k] = float(sig_block[k])
    return obs, sig
