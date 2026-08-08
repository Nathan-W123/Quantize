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
