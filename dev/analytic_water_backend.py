"""Registers an analytic water backend so the full CLI path can run offline.

Importing this module registers ``analytic_water`` with the backend registry,
which lets a config name it in ``quantum.backend`` and exercise the real
``python -m cli run`` pipeline -- config validation, run directory, internal
coordinates, autoconfig, multistart, corrections, reporting -- on a machine with
no Psi4 or ORCA.

The surface is the Hoy/Mills/Strey valence force field with Morse OH stretches
from dev/tests/reference_molecules.py, whose minimum sits at the accepted
equilibrium geometry. It is not a substitute for a real electronic-structure
calculation; its bend frequency is ~4% low and it carries no bend anharmonicity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import constants as _sc

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import reference_molecules as _rm  # noqa: E402
from backend.base_backend import GradientResult, HessianResult, QuantumBackend  # noqa: E402
from backend.registry import register_backend  # noqa: E402

_HARTREE_J = _sc.physical_constants["Hartree energy"][0]
_BOHR_A = _sc.physical_constants["Bohr radius"][0] * 1e10
_AJ = 1e-18


# Displaced minimum for the "realistic method" surface: a hybrid-DFT-sized
# geometry error for water, so the quantum prior carries a bias the way a real
# electronic-structure calculation does.
R_DETUNED, THETA_DETUNED = 0.96870, 103.700


def _energy_hartree(coords_ang) -> float:
    return _rm._h2o_energy(np.asarray(coords_ang, dtype=float), True) * _AJ / _HARTREE_J


def _gradient_bohr(coords_ang, step: float = 1e-5) -> np.ndarray:
    flat = np.asarray(coords_ang, dtype=float).ravel().copy()
    grad = np.zeros(flat.size)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (
            _energy_hartree(plus.reshape(-1, 3))
            - _energy_hartree(minus.reshape(-1, 3))
        ) / (2 * step)
    return grad * _BOHR_A


@register_backend
class AnalyticWaterBackend(QuantumBackend):
    """Analytic water PES presented through the QuantumBackend interface."""

    name = "analytic_water"
    supports_parallel = True

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])
        if [e for e in self.elems] not in ([], ["O", "H", "H"]):
            raise ValueError(
                "analytic_water only models H2O with atoms ordered O, H, H; "
                f"got {self.elems}"
            )

    def run_hessian(self, coords_ang) -> HessianResult:
        return HessianResult(
            energy=_energy_hartree(coords_ang),
            gradient_bohr=_gradient_bohr(coords_ang),
            hessian_bohr=_rm.h2o_hessian(coords_ang),
        )

    def run_gradient(self, coords_ang) -> GradientResult:
        return GradientResult(
            energy=_energy_hartree(coords_ang),
            gradient_bohr=_gradient_bohr(coords_ang),
        )


# ── Detuned variant ──────────────────────────────────────────────────────────

def _detuned_energy_hartree(coords_ang) -> float:
    """Same force field, minimum displaced to (R_DETUNED, THETA_DETUNED)."""
    x = np.asarray(coords_ang, dtype=float)
    r1 = float(np.linalg.norm(x[1] - x[0]))
    r2 = float(np.linalg.norm(x[2] - x[0]))
    u1 = (x[1] - x[0]) / r1
    u2 = (x[2] - x[0]) / r2
    theta = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
    d1, d2 = r1 - R_DETUNED, r2 - R_DETUNED
    dt = theta - np.radians(THETA_DETUNED)
    de = _rm._OH_WE ** 2 / (4 * _rm._OH_WEXE) * _sc.h * _sc.c * 100 / _AJ
    a = np.sqrt(_rm._F_R / (2 * de))
    stretch = de * ((1 - np.exp(-a * d1)) ** 2 + (1 - np.exp(-a * d2)) ** 2)
    energy_aj = (
        stretch
        + _rm._F_RR * d1 * d2
        + 0.5 * _rm._F_TH * R_DETUNED ** 2 * dt ** 2
        + _rm._F_RTH * R_DETUNED * (d1 + d2) * dt
    )
    return energy_aj * _AJ / _HARTREE_J


def _fd_gradient(fn, coords_ang, step=1e-5):
    flat = np.asarray(coords_ang, dtype=float).ravel().copy()
    grad = np.zeros(flat.size)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (fn(plus.reshape(-1, 3)) - fn(minus.reshape(-1, 3))) / (2 * step)
    return grad * _BOHR_A


def _fd_hessian(fn, coords_ang, step=1e-4):
    flat = np.asarray(coords_ang, dtype=float).ravel().copy()
    n = flat.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            vals = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = flat.copy()
                v[i] += si * step
                v[j] += sj * step
                vals.append(fn(v.reshape(-1, 3)))
            hess[i, j] = hess[j, i] = (
                vals[0] - vals[1] - vals[2] + vals[3]
            ) / (4 * step * step)
    return hess * _BOHR_A ** 2


@register_backend
class DetunedWaterBackend(QuantumBackend):
    """Water surface with a realistic method error in its predicted geometry."""

    name = "analytic_water_detuned"
    supports_parallel = True

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])

    def run_hessian(self, coords_ang) -> HessianResult:
        return HessianResult(
            energy=_detuned_energy_hartree(coords_ang),
            gradient_bohr=_fd_gradient(_detuned_energy_hartree, coords_ang),
            hessian_bohr=_fd_hessian(_detuned_energy_hartree, coords_ang),
        )

    def run_gradient(self, coords_ang) -> GradientResult:
        return GradientResult(
            energy=_detuned_energy_hartree(coords_ang),
            gradient_bohr=_fd_gradient(_detuned_energy_hartree, coords_ang),
        )
