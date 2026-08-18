"""A real quantum-chemistry backend, via PySCF.

Registers ``pyscf_hf`` so configs can use genuine electronic structure -- SCF
energies with analytic gradients and Hessians -- rather than the analytic model
surfaces in dev/analytic_water_backend.py. PySCF is pip-installable, so this
works where Psi4 and ORCA are not available.

Method and basis come from the config's ``quantum.method`` / ``quantum.basis``
via the standard backend kwargs, defaulting to RHF/STO-3G. Small bases are
deliberately allowed: they carry large, well-understood geometry errors, which
is what you want when testing whether spectroscopy can correct a biased prior.

Cost scales sharply with basis. On fluorobenzene (12 atoms): STO-3G is about
0.3 s per SCF and 26 s per analytic Hessian; 6-31G is 1 s and 68 s.
"""

from __future__ import annotations

import warnings

import numpy as np

from backend.base_backend import GradientResult, HessianResult, QuantumBackend
from backend.registry import register_backend

warnings.filterwarnings("ignore", module="pyscf")


@register_backend
class PySCFBackend(QuantumBackend):
    """RHF (or DFT) energies, gradients and Hessians from PySCF."""

    name = "pyscf_hf"
    supports_parallel = False           # PySCF already threads internally

    def __init__(self, elems=None, method: str = "hf", basis: str = "sto-3g",
                 charge: int = 0, multiplicity: int = 1, **_kwargs):
        self.elems = list(elems or [])
        self.method = str(method or "hf").strip().lower()
        self.basis = str(basis or "sto-3g").strip() or "sto-3g"
        self.charge = int(charge)
        self.spin = int(multiplicity) - 1
        self._cache: dict[bytes, object] = {}

    # ── internals ────────────────────────────────────────────────────────────

    def _mean_field(self, coords_ang):
        from pyscf import dft, gto, scf

        coords = np.asarray(coords_ang, dtype=float)
        key = coords.round(10).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        mol = gto.M(
            atom=[(e, tuple(xyz)) for e, xyz in zip(self.elems, coords)],
            basis=self.basis, charge=self.charge, spin=self.spin,
            unit="Angstrom", verbose=0,
        )
        if self.method in ("hf", "rhf", "scf"):
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = self.method
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"PySCF SCF did not converge ({self.method}/{self.basis})")
        # One geometry at a time; the optimizer revisits the current point for
        # gradient and Hessian, and holding every past geometry would grow
        # without bound over a long run.
        self._cache = {key: mf}
        return mf

    # ── QuantumBackend interface ─────────────────────────────────────────────

    def run_gradient(self, coords_ang) -> GradientResult:
        mf = self._mean_field(coords_ang)
        grad = mf.nuc_grad_method().kernel()          # Hartree/Bohr, (N, 3)
        return GradientResult(
            energy=float(mf.e_tot),
            gradient_bohr=np.asarray(grad, dtype=float).ravel(),
        )

    def run_hessian(self, coords_ang) -> HessianResult:
        mf = self._mean_field(coords_ang)
        grad = mf.nuc_grad_method().kernel()
        hess = mf.Hessian().kernel()                  # (N, N, 3, 3) Hartree/Bohr^2
        n = len(self.elems)
        hess_flat = np.asarray(hess, dtype=float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        return HessianResult(
            energy=float(mf.e_tot),
            gradient_bohr=np.asarray(grad, dtype=float).ravel(),
            hessian_bohr=0.5 * (hess_flat + hess_flat.T),
        )

    # ── Convenience for the "theory alone" leg ───────────────────────────────

    def optimise(self, coords_ang, max_steps: int = 200, gtol: float = 2e-5):
        """Relax the geometry on the PES alone, with no experimental input.

        L-BFGS on Cartesians using the analytic gradient. PySCF's own driver
        needs geomeTRIC, which has no wheel here, and for a rigid aromatic this
        converges perfectly well without one.
        """
        from scipy.optimize import minimize

        from backend.quantum import ANG_TO_BOHR

        shape = np.asarray(coords_ang, dtype=float).shape

        def fun(flat):
            coords = flat.reshape(shape)
            mf = self._mean_field(coords)
            grad_bohr = np.asarray(mf.nuc_grad_method().kernel(), dtype=float).ravel()
            # Hartree/Bohr -> Hartree/Angstrom
            return float(mf.e_tot), grad_bohr * ANG_TO_BOHR

        sol = minimize(
            fun, np.asarray(coords_ang, dtype=float).ravel(), jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_steps, "gtol": gtol, "ftol": 1e-14},
        )
        return sol.x.reshape(shape)
