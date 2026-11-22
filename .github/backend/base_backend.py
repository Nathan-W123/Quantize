"""
Abstract base class and shared return types for quantum chemistry backends.

Adding a new backend
--------------------
1. Subclass QuantumBackend in a new file, e.g. ``backend/gaussian_backend.py``.
2. Set ``name = "gaussian"`` (the string key used in configs).
3. Decorate with ``@register_backend`` from ``backend.registry``.
4. Implement ``run_hessian`` and ``run_gradient``; optionally override
   ``run_rovib`` (defaults to returning ``None``).
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


# ── The translational sum rule ────────────────────────────────────────────────

def impose_translational_sum_rule(hessian_bohr):
    """The Cartesian Hessian with the translational (acoustic) sum rule imposed.

    Moving the whole molecule changes nothing, so every row of an exact
    Hessian sums to zero over the atoms: sum_j H[i a, j b] = 0. This
    subtracts each atom's row-sum block from its diagonal block and
    symmetrises: a symmetric on-site error is removed exactly, an invariant
    Hessian is left as it is, and a second pass changes nothing measurable
    (~1e-18 Ha/bohr^2 on a real one).

    Why it is needed (decisions D-55): pyscf's analytic RKS Hessian (2.15,
    CPU) has no grid-weight response, and at its default grid every
    B3LYP/6-31G(d) Hessian in the benchmark breaks the rule -- row sums of
    2e-4 (water) to 1.9e-2 Ha/bohr^2 (chlorofluoromethane); at the theory
    geometries each molecule's largest rigid-mode eigenvalue is 23-117
    cm^-1. Projecting the rigid modes out removes their
    leak into the vibrations but not the error inside them: imposing the rule
    took chlorofluoromethane's vibrations from up to 5 cm^-1 off a level-7
    grid to 0.1, and the harmonic alpha of three molecules from 0.4-0.8 MHz
    off a converged grid to 0.006-0.03. RHF Hessians satisfy it to ~1e-9, so
    there it does nothing. It fixes translations only; the rotational
    residue a gradient leaves off a stationary point is the rigid-mode
    projector's (harmonic_alpha._project_rigid_mw).

    Anything that is not a square 3N x 3N matrix is returned unchanged.
    """
    h = np.asarray(hessian_bohr, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 3 or h.shape[0] == 0:
        return h
    n3 = h.shape[0]
    n = n3 // 3
    hr = (0.5 * (h + h.T)).reshape(n, 3, n, 3).copy()
    sums = hr.sum(2)
    for i in range(n):
        hr[i, :, i, :] -= sums[i]
    hs = hr.reshape(n3, n3)
    return 0.5 * (hs + hs.T)


# ── Unified quantum state (replaces QuantumEngine / Psi4State in optimizer) ──

class QuantumState:
    """Unit-converted view over a HessianResult or an in-place gradient update.

    The Hessian is stored with the translational sum rule imposed
    (impose_translational_sum_rule): every Hessian the optimizer holds as
    its quantum state -- the quantum prior's curvature, the reported widths,
    the corrections built from it -- enters here. Hessians that go straight
    to the alpha module (the correction-level table, cubic displacements)
    get it there.
    """

    def __init__(
        self,
        energy: float,
        gradient_bohr: np.ndarray,
        hessian_bohr: np.ndarray,
    ) -> None:
        self.energy = float(energy)
        self._gradient_bohr = np.asarray(gradient_bohr, dtype=float).ravel()
        self._hessian_bohr = impose_translational_sum_rule(hessian_bohr)

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
    #: Whether the calculation runs inside this Python process (pyscf, Psi4)
    #: rather than in a program it starts (ORCA). An in-process backend's
    #: threads live in this process, so a worker forked from it can inherit
    #: their locks and hang (registry.fork_hang_warning).
    runs_in_process: ClassVar[bool] = True

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
