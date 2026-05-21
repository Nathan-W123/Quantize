"""
Psi4 quantum chemistry backend.

Thin wrapper around the existing :class:`~backend.Psi4.Psi4Engine` that
adapts its return values to the standard :class:`~backend.base_backend.HessianResult`
and :class:`~backend.base_backend.GradientResult` types.
"""

from __future__ import annotations

import numpy as np

from backend.base_backend import GradientResult, HessianResult, QuantumBackend
from backend.registry import register_backend


@register_backend
class Psi4Backend(QuantumBackend):
    """Psi4-based in-memory quantum chemistry computations."""

    name = "psi4"
    supports_parallel = True

    def __init__(
        self,
        elems,
        method,
        basis,
        charge,
        multiplicity,
        memory="2 GB",
        num_threads=1,
        output_file=None,
    ):
        from backend.psi4.Psi4 import Psi4Engine  # pylint: disable=import-outside-toplevel
        self._engine = Psi4Engine(
            elems=elems,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            memory=memory,
            num_threads=num_threads,
            output_file=output_file,
        )

    def run_hessian(self, coords_ang: np.ndarray) -> HessianResult:
        print("  [Psi4] Running Hessian calculation (gradient + Hessian)...")
        state = self._engine.run_hessian(coords_ang)
        print(f"  [Psi4] Done.  Energy = {state.energy:.10f} Hartree")
        return HessianResult(
            energy=state.energy,
            gradient_bohr=state._gradient_bohr.copy(),
            hessian_bohr=state._hessian_bohr.copy(),
        )

    def run_gradient(self, coords_ang: np.ndarray) -> GradientResult:
        print("  [Psi4] Running gradient update...")
        energy, grad_bohr = self._engine.run_gradient(coords_ang)
        print(f"  [Psi4] Done.  Energy = {energy:.10f} Hartree")
        return GradientResult(
            energy=float(energy),
            gradient_bohr=np.asarray(grad_bohr, dtype=float).ravel(),
        )
    # run_rovib â†’ inherits None default (Psi4 does not support VPT2 rovib)
    # run_cheap_opt â†’ inherits None default
