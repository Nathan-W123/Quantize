"""Analytic reference molecules with published experimental constants.

These build exact Cartesian Hessians (and, where anharmonic, Hessian callables)
from closed-form potentials, so the alpha / B_e machinery can be validated
against experiment without needing Psi4 or ORCA installed.

Experimental values:
  CO   -- Huber, K.P.; Herzberg, G. "Constants of Diatomic Molecules" (1979)
  H2O  -- equilibrium geometry: Csaszar et al., ground-state constants: JPL/HITRAN
  Force field: Hoy, A.R.; Mills, I.M.; Strey, G. Mol. Phys. 24 (1972) 1265
"""

from __future__ import annotations

import numpy as np
from scipy import constants as _sc

_HARTREE_J = _sc.physical_constants["Hartree energy"][0]
_BOHR_A = _sc.physical_constants["Bohr radius"][0] * 1e10
_C_CM = _sc.c * 100
_AMU = _sc.atomic_mass
_AJ = 1e-18                      # 1 aJ = 1 mdyn*Angstrom


# ── Carbon monoxide ──────────────────────────────────────────────────────────

CO_MASSES = np.array([12.0, 15.9949146196])
CO_R_E = 1.1283230               # Angstrom
CO_OMEGA_E = 2169.81280          # cm-1
CO_OMEGA_E_X_E = 13.28831        # cm-1
CO_B_E = 1.93128087              # cm-1
CO_ALPHA_E = 0.01750441          # cm-1
CO_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, CO_R_E]])


def co_pekeris_alpha() -> float:
    """Exact Dunham/Pekeris alpha_e for a Morse oscillator, in cm-1.

    alpha_e = (6 Be^2 / we) (sqrt(we xe / Be) - 1)
    """
    return (6.0 * CO_B_E**2 / CO_OMEGA_E) * (np.sqrt(CO_OMEGA_E_X_E / CO_B_E) - 1.0)


def co_harmonic_alpha() -> float:
    """Harmonic-only alpha for a diatomic: -6 Be^2 / we, in cm-1."""
    return -6.0 * CO_B_E**2 / CO_OMEGA_E


def _co_morse_params():
    mu = CO_MASSES[0] * CO_MASSES[1] / CO_MASSES.sum()
    de_j = (CO_OMEGA_E**2 / (4.0 * CO_OMEGA_E_X_E)) * _sc.h * _sc.c * 100
    k_si = mu * _AMU * (2 * np.pi * _C_CM * CO_OMEGA_E) ** 2
    a_inv_ang = np.sqrt(k_si / (2 * de_j)) * 1e-10
    return de_j, a_inv_ang


def co_hessian(coords_ang) -> np.ndarray:
    """Exact Morse Cartesian Hessian for CO in Hartree/Bohr^2."""
    de_j, a = _co_morse_params()
    x = np.asarray(coords_ang, dtype=float)
    d = x[1] - x[0]
    r = float(np.linalg.norm(d))
    u = d / r
    e = np.exp(-a * (r - CO_R_E))
    d2v = 2 * de_j * a**2 * (2 * e**2 - e)          # J/Angstrom^2
    dv = 2 * de_j * a * (e - e**2)                  # J/Angstrom
    blk = d2v * np.outer(u, u) + (dv / r) * (np.eye(3) - np.outer(u, u))
    hess = np.zeros((6, 6))
    hess[:3, :3] = hess[3:, 3:] = blk
    hess[:3, 3:] = hess[3:, :3] = -blk
    return hess / _HARTREE_J * _BOHR_A**2


# ── Water ────────────────────────────────────────────────────────────────────

H2O_MASSES = np.array([15.9949146196, 1.0078250319, 1.0078250319])
H2O_ELEMS = ["O", "H", "H"]
H2O_R_E = 0.95784                # Angstrom
H2O_THETA_E = np.radians(104.508)
# H2-16O ground-state rotational constants, MHz
H2O_B0_MHZ = np.array([835840.29, 435351.72, 278138.70])
# Harmonic valence force field, mdyn/A and mdyn*A/rad^2
_F_R, _F_RR, _F_TH, _F_RTH = 8.454, -0.101, 0.697, 0.228
# Local-mode OH Morse parameters
_OH_WE, _OH_WEXE = 3885.0, 82.0


def h2o_coords(r_ang: float = H2O_R_E, theta_rad: float = H2O_THETA_E) -> np.ndarray:
    half = theta_rad / 2.0
    return np.array([
        [0.0, 0.0, 0.0],
        [0.0, r_ang * np.sin(half), r_ang * np.cos(half)],
        [0.0, -r_ang * np.sin(half), r_ang * np.cos(half)],
    ])


def _h2o_internals(x):
    r1 = float(np.linalg.norm(x[1] - x[0]))
    r2 = float(np.linalg.norm(x[2] - x[0]))
    u1 = (x[1] - x[0]) / r1
    u2 = (x[2] - x[0]) / r2
    return r1, r2, float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))


def _h2o_energy(x, anharmonic: bool):
    r1, r2, th = _h2o_internals(x)
    d1, d2, dt = r1 - H2O_R_E, r2 - H2O_R_E, th - H2O_THETA_E
    if anharmonic:
        de = _OH_WE**2 / (4 * _OH_WEXE) * _sc.h * _sc.c * 100 / _AJ    # mdyn*A
        a = np.sqrt(_F_R / (2 * de))
        stretch = de * ((1 - np.exp(-a * d1)) ** 2 + (1 - np.exp(-a * d2)) ** 2)
    else:
        stretch = 0.5 * _F_R * (d1**2 + d2**2)
    return (stretch + _F_RR * d1 * d2 + 0.5 * _F_TH * H2O_R_E**2 * dt**2
            + _F_RTH * H2O_R_E * (d1 + d2) * dt)


def h2o_hessian(coords_ang, anharmonic: bool = True) -> np.ndarray:
    """Cartesian Hessian of the water valence force field in Hartree/Bohr^2."""
    x = np.asarray(coords_ang, dtype=float)
    step, n = 1e-4, x.size
    flat = x.ravel().copy()
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            vals = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = flat.copy()
                v[i] += si * step
                v[j] += sj * step
                vals.append(_h2o_energy(v.reshape(-1, 3), anharmonic))
            hess[i, j] = hess[j, i] = (
                vals[0] - vals[1] - vals[2] + vals[3]
            ) / (4 * step * step)
    return hess * _AJ / _HARTREE_J * _BOHR_A**2
