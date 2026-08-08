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


def compute_cd_constants(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    masses_amu,
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.05,
) -> CDConstants:
    """
    Harmonic CD constants from Hessian and equilibrium geometry.

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
    cd_cm = watson_a_reduction_cd_from_tau_cm(tau_cm)
    cd_mhz = {k: v * _CM_TO_MHZ for k, v in cd_cm.items()}
    # The tau -> A-reduction mapping below is not validated against experiment
    # (see watson_a_reduction_cd_from_tau_cm), so the per-constant sigma is
    # floored at 100% rather than the caller's nominal fraction.
    sigma = {
        k: max(abs(cd_mhz[k]) * max(sigma_fraction, 1.0), 0.01) for k in CD_NAMES
    }
    return CDConstants(
        DJ=cd_mhz["DJ"],
        DJK=cd_mhz["DJK"],
        DK=cd_mhz["DK"],
        delta_J=cd_mhz["delta_J"],
        delta_K=cd_mhz["delta_K"],
        source="harmonic_hessian",
        method="harmonic_VR",
        sigma=sigma,
        notes=(
            "Harmonic τ′ → Watson A-reduction CD. "
            "A-reduction mapping UNVALIDATED — treat as order-of-magnitude."
        ),
    )


def build_cd_table_from_hessian(
    hess_bohr: np.ndarray,
    coords_ang: np.ndarray,
    isotopologues: list[dict],
    min_freq_cm: float = 50.0,
    fd_delta: float = 0.05,
    sigma_fraction: float = 0.05,
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
