"""Spectroscopy-restrained geometry optimisation inside geomeTRIC.

The engine in :mod:`backend.quantize` carries its own optimiser: a
sigma-weighted joint step with an explicit rigid-mode projector, written
because the spectral residual has to enter the same linear solve as the
quantum prior. That works, but it means this project maintains optimisation
machinery that geomeTRIC already does better -- in particular TRIC
coordinates, which represent collective translations and rotations as explicit
degrees of freedom via quaternion exponential maps. Those six directions are
exactly the ones that caused the rigid-mode projector bugs here.

This module takes the other route. Rather than reimplementing coordinates and
trust radii, it expresses the whole hybrid objective as a single *energy*, and
hands that to geomeTRIC:

    E_total(x) = E_QM(x) + w * chi2_spectral(x)

    w = 0.5 * lambda_bar * sigma_x^2        [Hartree per chi-square unit]

The weight is not a tuning knob. It is the same calibration the in-house
optimiser derives for alpha_q, solved the other way round: alpha_q converts
Hartrees into chi-square units, so its reciprocal converts a chi-square back
into Hartrees. lambda_bar is the mean positive curvature of the quantum
Hessian and sigma_x the displacement over which that surface is trusted, so
one prior standard deviation of geometry costs half a chi-square unit in both
formulations. A geometry that minimises this energy minimises the joint
objective.

What this buys, beyond not maintaining an optimiser: geomeTRIC drives real
quantum chemistry packages, so the same restrained optimisation runs against
any of its supported backends rather than only the PySCF path wired up here.

The spectral gradient is finite-differenced, which is cheap and exact enough:
rotational constants are closed-form functions of geometry and masses with no
quantum chemistry in them, so the differencing costs nothing and carries no
convergence noise.
"""

from __future__ import annotations

import numpy as np

from backend.spectral.centrifugal_distortion import rotational_constants_mhz

#: Bohr per Angstrom. geomeTRIC speaks Bohr; the spectral machinery here and
#: the reference structures are in Angstrom.
_BOHR_PER_ANG = 1.8897261254578281


def spectral_chi2_and_gradient(coords_ang, targets):
    """Weighted spectral residual sum and its Cartesian gradient.

    ``targets`` is a sequence of dicts with ``masses``, ``component`` (0/1/2
    for A/B/C), ``value`` in MHz and ``sigma`` in MHz -- the same corrected
    targets the rest of the engine fits.

    Returns ``(chi2, grad_ang)`` with the gradient in 1/Angstrom, shaped
    (3*N,).

    The derivative is analytic, via the engine's existing
    d(lambda)/dx = v^T (dI/dx) v Jacobian. Finite differences were tried first
    and are a trap here: rotational constants run to hundreds of GHz and
    d(B)/dx to ~1e6 MHz/Angstrom, so a tightly weighted residual produces a
    chi-square steep enough that third-order asymmetry leaks into a central
    difference. Measured on water with a 100 MHz sigma, the differenced
    gradient read 40 /Angstrom at the exact minimum, where the true value is
    zero -- a spurious force that would have pulled a converged geometry off
    its own target.
    """
    from backend.spectral.spectral import _jacobian_full_analytic

    coords = np.asarray(coords_ang, dtype=float)
    chi2 = 0.0
    grad = np.zeros(coords.size, dtype=float)
    for t in targets:
        masses = np.asarray(t["masses"], dtype=float)
        comp = int(t["component"])
        sigma = float(t["sigma"])
        calc = rotational_constants_mhz(coords, masses)
        resid = (float(calc[comp]) - float(t["value"])) / sigma
        chi2 += resid * resid
        jac = _jacobian_full_analytic(coords, masses, 1e-3)
        grad += 2.0 * resid * np.asarray(jac[comp], dtype=float) / sigma
    return chi2, grad


def calibrated_spectral_weight(hessian_hartree_per_ang2, sigma_x_ang):
    """Hartrees per chi-square unit, from the quantum surface's own curvature.

    Mirror image of ``SubspaceOptimizer._calibrated_alpha_q``: there the
    quantum energy is converted into chi-square units so it can be added to a
    spectral residual, here a spectral residual is converted into Hartrees so
    it can be added to an energy. Both place one prior standard deviation of
    geometry at half a chi-square unit, so the two formulations have the same
    minimum.
    """
    from backend.spectral.SVD import _CURVATURE_FLOOR_REL

    h = np.asarray(hessian_hartree_per_ang2, dtype=float)
    evals = np.linalg.eigvalsh(0.5 * (h + h.T))
    # Same rigid-mode cut as the optimiser, imported rather than repeated: the
    # two must agree or the bridge stops being the reciprocal of alpha_q.
    positive = evals[evals > _CURVATURE_FLOOR_REL * max(
        float(np.max(np.abs(evals))), 1.0)]
    if positive.size == 0:
        return 1.0
    lam_bar = float(np.mean(positive))
    return 0.5 * lam_bar * float(sigma_x_ang) ** 2


def make_engine(elems, targets, quantum_backend, weight_hartree):
    """A geomeTRIC Engine whose energy is E_QM + w * chi2_spectral.

    Built lazily so that importing this module does not require geomeTRIC to
    be installed; only calling this function does.
    """
    from geometric.engine import Engine
    from geometric.molecule import Molecule

    class SpectralRestrainedEngine(Engine):
        """Reports the joint objective to geomeTRIC as if it were an energy."""

        def __init__(self):
            mol = Molecule()
            mol.elem = list(elems)
            mol.xyzs = [np.zeros((len(elems), 3))]
            super().__init__(mol)
            self.n_calls = 0
            self.last_breakdown = None

        def calc_new(self, coords, dirname):
            self.n_calls += 1
            xyz_ang = np.asarray(coords, dtype=float).reshape(-1, 3) / _BOHR_PER_ANG

            res = quantum_backend.run_gradient(xyz_ang)
            e_qm = float(res.energy)
            g_qm_bohr = np.asarray(res.gradient_bohr, dtype=float).ravel()

            chi2, g_chi2_ang = spectral_chi2_and_gradient(xyz_ang, targets)
            # d/dx_bohr = d/dx_ang / (bohr per ang)
            g_chi2_bohr = g_chi2_ang / _BOHR_PER_ANG

            self.last_breakdown = {"e_qm": e_qm, "chi2": chi2,
                                   "e_spectral": weight_hartree * chi2}
            return {"energy": e_qm + weight_hartree * chi2,
                    "gradient": g_qm_bohr + weight_hartree * g_chi2_bohr}

    return SpectralRestrainedEngine()


def optimise_with_tric(elems, coords_ang, targets, quantum_backend,
                       weight_hartree, dirname, coordsys="tric",
                       maxiter=100, converge_set=None):
    """Run geomeTRIC on the joint objective; return the optimised geometry.

    ``coordsys`` is passed through to geomeTRIC: "tric" builds the
    translation-rotation-internal system, "cart" plain Cartesians, which is
    useful for isolating whether a difference came from the coordinates or
    from the restraint.
    """
    from geometric.internal import (
        CartesianCoordinates,
        DelocalizedInternalCoordinates,
    )
    from geometric.molecule import Molecule
    from geometric.optimize import Optimizer
    from geometric.params import OptParams

    coords = np.asarray(coords_ang, dtype=float)
    mol = Molecule()
    mol.elem = list(elems)
    mol.xyzs = [coords.copy()]

    engine = make_engine(elems, targets, quantum_backend, weight_hartree)
    x0_bohr = coords.ravel() * _BOHR_PER_ANG

    if str(coordsys).lower() == "cart":
        ic = CartesianCoordinates(mol)
    else:
        # connect=False, addcart=False is the TRIC construction: primitive
        # internals for bonded fragments plus explicit translation and
        # rotation coordinates for each fragment.
        ic = DelocalizedInternalCoordinates(mol, build=True, connect=False,
                                            addcart=False)

    kwargs = {"maxiter": int(maxiter)}
    if converge_set:
        kwargs["converge"] = list(converge_set)
    params = OptParams(**kwargs)

    opt = Optimizer(x0_bohr, mol, ic, engine, str(dirname), params,
                    print_info=False)
    # optimizeGeometry returns the trajectory as a Molecule; the authoritative
    # final coordinates are the optimiser's own state, in Bohr.
    opt.optimizeGeometry()
    return (np.asarray(opt.X, dtype=float).reshape(-1, 3)
            / _BOHR_PER_ANG), engine
