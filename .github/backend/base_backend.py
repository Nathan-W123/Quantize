"""
Abstract base class and shared return types for quantum chemistry backends.

Adding a new backend
--------------------
1. Subclass QuantumBackend in a new file, e.g. ``backend/gaussian_backend.py``.
2. Set ``name = "gaussian"`` (the string key used in configs).
3. Decorate with ``@register_backend`` from ``backend.registry``.
4. Implement ``run_hessian`` and ``run_gradient``; optionally override
   ``run_rovib`` and ``run_cheap_opt`` (both default to returning ``None``).
5. Add one import line in ``.github/backend/__init__.py`` to trigger registration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from backend.quantum import ANG_TO_BOHR


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class GradientResult:
    """Energy and gradient returned by a gradient-only calculation."""
    energy: float
    gradient_bohr: np.ndarray   # Hartree / Bohr, shape (3*N,)


@dataclass
class HessianResult:
    """Energy, gradient, and Hessian returned by a frequency calculation."""
    energy: float
    gradient_bohr: np.ndarray   # Hartree / Bohr, shape (3*N,)
    hessian_bohr: np.ndarray    # Hartree / Bohr², shape (3*N, 3*N)


@dataclass
class RovibResult:
    """Per-isotopologue VPT2 rovibrational corrections.

    ``isotopologue_corrections`` is a list of dicts (one per isotopologue) with
    the same keys that ``MolecularOptimizer._run_rovib`` currently writes into
    ``self.spectral.isotopologues``:
        alpha_constants, delta_vib_constants, delta_elec_constants,
        delta_bob_constants, sigma_correction_constants
    """
    isotopologue_corrections: list[dict] = field(default_factory=list)


# ── Unified quantum state (replaces QuantumEngine / Psi4State in optimizer) ──

class QuantumState:
    """Unit-converted view over a HessianResult or an in-place gradient update."""

    def __init__(
        self,
        energy: float,
        gradient_bohr: np.ndarray,
        hessian_bohr: np.ndarray,
    ) -> None:
        self.energy = float(energy)
        self._gradient_bohr = np.asarray(gradient_bohr, dtype=float).ravel()
        self._hessian_bohr = np.asarray(hessian_bohr, dtype=float)

    @property
    def gradient(self) -> np.ndarray:
        """Gradient in Hartree / Angstrom."""
        return self._gradient_bohr * ANG_TO_BOHR

    @property
    def hessian(self) -> np.ndarray:
        """Hessian in Hartree / Angstrom²."""
        return self._hessian_bohr * ANG_TO_BOHR ** 2


# ── Abstract backend interface ────────────────────────────────────────────────

class QuantumBackend(ABC):
    """Interface every quantum chemistry backend must implement."""

    name: ClassVar[str]
    supports_parallel: ClassVar[bool] = True

    @abstractmethod
    def run_hessian(self, coords_ang: np.ndarray) -> HessianResult:
        """Full frequency job: return energy, gradient, and Hessian."""

    @abstractmethod
    def run_gradient(self, coords_ang: np.ndarray) -> GradientResult:
        """Cheap gradient-only job: return energy and gradient."""

    def run_rovib(
        self,
        coords_ang: np.ndarray,
        isotopologues: list[dict] | None = None,
    ) -> RovibResult | None:
        """Optional: anharmonic VPT2 rovibrational corrections.

        ``isotopologues`` is a list of dicts (one per isotopologue) containing
        at minimum: ``name``, ``masses``, ``component_indices``,
        ``alpha_constants``, and optionally ``rovib_table``.

        Returns ``None`` if the backend does not support rovib calculations.
        """
        return None

    def run_cheap_opt(self, coords_ang: np.ndarray) -> np.ndarray | None:
        """Optional: fast geometry pre-optimisation.

        Returns updated coordinates (Angstrom) or ``None`` if unsupported.
        """
        return None
