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

Correlated methods
------------------
``method`` also accepts "mp2", "ccsd" and "ccsd(t)". Energies and gradients are
analytic; PySCF has no analytic Hessian for any of them, so the Hessian is a
central finite difference of the analytic gradient, costing 6N gradient
evaluations. That is what decides whether a correlated force field is
reachable, and the answer depends sharply on the molecule. Measured here, for
one cubic force field under the normal-mode scheme (2 x n_vib Hessians, each
6N gradients):

    water / cc-pVTZ              58 bf    MP2  1.3 s/grad   ->  MP2  <0.1 h
                                          CCSD  20 s/grad   ->  CCSD  0.6 h
    formyl fluoride / cc-pVTZ   104 bf    MP2 10.7 s/grad   ->  MP2   0.9 h
                                          CCSD  72 s/grad   ->  CCSD  5.7 h

MP2/cc-pVTZ is therefore affordable through about five atoms, CCSD only for
three, and CCSD(T)/cc-pVTZ -- the level quantize.py names as its ORCA default
-- is out of reach on this hardware for anything but a triatomic. The published
error budget for the semi-experimental method puts an MP2/cc-pVTZ cubic force
field at about 0.21% error in the alpha sum against CCSD(T)/cc-pVTZ's 0.09%, so
MP2 is a real rung on the ladder rather than a consolation prize.

The gradient is analytic in every case, so ``optimise`` is cheap at all four
methods even where the Hessian is not.
"""

from __future__ import annotations

import warnings

import numpy as np

from backend.base_backend import GradientResult, HessianResult, QuantumBackend
from backend.registry import register_backend

warnings.filterwarnings("ignore", module="pyscf")


#: Methods built on an RHF reference instead of being a DFT functional.
_CORRELATED = ("mp2", "ccsd", "ccsd(t)", "ccsd_t")

#: Correlated methods have analytic gradients in PySCF but no analytic Hessian,
#: so run_hessian falls back to central differences of the gradient. The step
#: is in Bohr and is the usual compromise: large enough that the gradient's own
#: convergence noise does not dominate the difference, small enough that the
#: cubic term stays negligible.
_FD_STEP_BOHR = 0.01

#: SCF and coupled-cluster convergence used whenever a gradient is about to be
#: finite-differenced. PySCF's defaults (conv_tol 1e-9, and conv_tol_grad its
#: square root, ~3e-5) are set for energies, not for differencing: a gradient
#: good to 3e-5 Hartree/Bohr, differenced over 2 x 0.01 Bohr, leaves ~1.5e-3
#: Hartree/Bohr^2 of noise in the Hessian. Against water's analytic HF Hessian
#: that showed up as 5% on the whole matrix and 9% on the bend -- the softest
#: mode, where the absolute noise is the largest fraction of the eigenvalue.
#: Tightening by three orders fixes it; see test_pyscf_correlated_backend.py,
#: which pins the agreement.
_TIGHT_SCF_CONV = 1e-12
_TIGHT_SCF_CONV_GRAD = 1e-9
_TIGHT_CC_CONV = 1e-9


@register_backend
class PySCFBackend(QuantumBackend):
    """RHF (or DFT) energies, gradients and Hessians from PySCF."""

    name = "pyscf_hf"
    supports_parallel = False           # PySCF already threads internally

    def __init__(self, elems=None, method: str = "hf", basis: str = "sto-3g",
                 charge: int = 0, multiplicity: int = 1, **_kwargs):
        self.elems = list(elems or [])
        self.method = str(method or "hf").strip().lower()
        #: Correlated methods are built on an RHF reference rather than being
        #: an exchange-correlation functional, so they branch before the DFT
        #: path. Anything not named here is still handed to dft.RKS as an xc
        #: string, which is what lets arbitrary functionals through untouched.
        self.correlated = self.method in _CORRELATED
        #: Tighten SCF/CC convergence when a gradient will be differenced.
        #: Always on for correlated methods, since their Hessian is always a
        #: finite difference; switched on temporarily by _fd_hessian otherwise.
        self._tight = self.correlated
        self.basis = str(basis or "sto-3g").strip() or "sto-3g"
        self.charge = int(charge)
        self.spin = int(multiplicity) - 1
        self._cache: dict[bytes, object] = {}
        #: Converged density from the last geometry, reused as the SCF guess
        #: at the next one. Kept outside _cache because the cache holds a
        #: single geometry while this is deliberately carried across them.
        self._last_dm = None

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
        if self.method in ("hf", "rhf", "scf") or self.correlated:
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = self.method
        # Larger bases need more cycles than the default, and an optimiser
        # visits distorted geometries where the default DIIS path stalls.
        mf.max_cycle = max(int(getattr(mf, "max_cycle", 50)), 200)
        if self._tight:
            mf.conv_tol = _TIGHT_SCF_CONV
            mf.conv_tol_grad = _TIGHT_SCF_CONV_GRAD
        if self._last_dm is not None:
            # Chain the density from the previous geometry. Successive points
            # in an optimisation or a finite-difference Hessian are close, so
            # the converged density is a far better guess than the default
            # superposition of atomic densities.
            try:
                mf.kernel(dm0=self._last_dm)
            except Exception:  # noqa: BLE001 - fall back to the plain guess
                mf.kernel()
        else:
            mf.kernel()
        if not mf.converged:
            # Second-order (Newton) SCF. Slower per iteration and needs a
            # starting density, but it converges cases where DIIS oscillates,
            # which is what a triple-zeta basis at a displaced geometry does.
            # Measured: b3lyp/cc-pvtz on water aborted the whole benchmark here
            # before this fallback existed.
            try:
                mf = mf.newton()
                mf.max_cycle = 100
                mf.kernel()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"PySCF SCF did not converge ({self.method}/{self.basis}); "
                    f"second-order fallback also failed: {exc}"
                ) from exc
        if not mf.converged:
            raise RuntimeError(
                f"PySCF SCF did not converge ({self.method}/{self.basis}) "
                "even with the second-order solver"
            )
        self._last_dm = mf.make_rdm1()
        # One geometry at a time; the optimizer revisits the current point for
        # gradient and Hessian, and holding every past geometry would grow
        # without bound over a long run.
        self._cache = {key: mf}
        return mf

    def _correlated(self, mf):
        """The post-SCF object for a correlated method, converged.

        Returns (object, total energy, gradient callable).

        CCSD(T) has no object of its own in PySCF: the (T) correction is an
        energy added to a converged CCSD. The trap is that the CCSD object's
        own ``nuc_grad_method`` then returns the CCSD gradient, so folding the
        triples into the energy and taking derivatives from that object gives
        CCSD(T) energies with CCSD forces -- frequencies identical to CCSD to
        every printed digit, under a label that says otherwise. Measured on
        water, the true CCSD(T) gradient differs from the CCSD one by 5.2e-05
        Hartree/Bohr, which is small but is exactly the quantity the
        correction chain is sensitive to. The gradient is therefore taken from
        pyscf.grad.ccsd_t rather than from the CCSD object.
        """
        from pyscf import cc, mp

        if self.method == "mp2":
            obj = mp.MP2(mf)
            obj.kernel()
            return obj, float(obj.e_tot), obj.nuc_grad_method().kernel

        obj = cc.CCSD(mf)
        if self._tight:
            obj.conv_tol = _TIGHT_CC_CONV
            obj.conv_tol_normt = _TIGHT_CC_CONV
        obj.kernel()
        if not obj.converged:
            raise RuntimeError(
                f"PySCF CCSD did not converge ({self.method}/{self.basis})"
            )
        if self.method in ("ccsd(t)", "ccsd_t"):
            from pyscf.grad import ccsd_t as ccsd_t_grad
            return (obj, float(obj.e_tot + obj.ccsd_t()),
                    ccsd_t_grad.Gradients(obj).kernel)
        return obj, float(obj.e_tot), obj.nuc_grad_method().kernel

    def _energy_gradient(self, coords_ang):
        """(energy, flat gradient in Hartree/Bohr) at one geometry.

        One path for every method, so the optimiser and the finite-difference
        Hessian below do not each need to know whether they are looking at a
        mean field or a correlated wavefunction.
        """
        mf = self._mean_field(coords_ang)
        if not self.correlated:
            grad = mf.nuc_grad_method().kernel()
            return float(mf.e_tot), np.asarray(grad, dtype=float).ravel()
        _obj, e_tot, grad_fn = self._correlated(mf)
        return e_tot, np.asarray(grad_fn(), dtype=float).ravel()

    # ── QuantumBackend interface ─────────────────────────────────────────────

    def run_gradient(self, coords_ang) -> GradientResult:
        energy, grad = self._energy_gradient(coords_ang)
        return GradientResult(energy=energy, gradient_bohr=grad)

    def run_hessian(self, coords_ang) -> HessianResult:
        if self.correlated:
            return self._fd_hessian(coords_ang)
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

    def _fd_hessian(self, coords_ang) -> HessianResult:
        """Central finite difference of the analytic gradient.

        6N gradient evaluations, which for a correlated method is the whole
        cost of the thing -- see this module's docstring for where that stops
        being affordable.

        The displacement is applied in Angstrom because that is the backend's
        coordinate unit, while the gradient comes back per Bohr, so the
        difference quotient carries one ANG_TO_BOHR to land in Hartree/Bohr^2.
        The result is symmetrised: the two halves differ only by
        finite-difference noise, and averaging them is the cheapest way to stop
        that noise reaching the normal-mode analysis as an asymmetry it will
        otherwise interpret as an imaginary frequency.
        """
        from backend.quantum import ANG_TO_BOHR

        coords = np.asarray(coords_ang, dtype=float)
        n3 = coords.size
        step_ang = _FD_STEP_BOHR / ANG_TO_BOHR

        # Every gradient in this sweep is about to be differenced, including
        # the reference one, so the cached mean field from a looser setting is
        # not reusable. Clear it rather than silently differencing a mixture.
        was_tight = self._tight
        self._tight = True
        if not was_tight:
            self._cache = {}
            self._last_dm = None
        try:
            return self._fd_hessian_inner(coords, n3, step_ang)
        finally:
            self._tight = was_tight
            if not was_tight:
                self._cache = {}
                self._last_dm = None

    def _fd_hessian_inner(self, coords, n3, step_ang) -> HessianResult:
        from backend.quantum import ANG_TO_BOHR

        energy, grad0 = self._energy_gradient(coords)
        hess = np.zeros((n3, n3))
        flat = coords.ravel().copy()
        for i in range(n3):
            plus = flat.copy();  plus[i] += step_ang
            minus = flat.copy(); minus[i] -= step_ang
            _, gp = self._energy_gradient(plus.reshape(coords.shape))
            _, gm = self._energy_gradient(minus.reshape(coords.shape))
            hess[i] = (gp - gm) / (2.0 * step_ang) / ANG_TO_BOHR

        # The displaced geometries evicted the reference point from the cache,
        # so restore it: the caller usually asks for the gradient next.
        self._mean_field(coords)
        return HessianResult(
            energy=energy,
            gradient_bohr=grad0,
            hessian_bohr=0.5 * (hess + hess.T),
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
            energy, grad_bohr = self._energy_gradient(flat.reshape(shape))
            # Hartree/Bohr -> Hartree/Angstrom
            return energy, grad_bohr * ANG_TO_BOHR

        sol = minimize(
            fun, np.asarray(coords_ang, dtype=float).ravel(), jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_steps, "gtol": gtol, "ftol": 1e-14},
        )
        return sol.x.reshape(shape)
