"""Correlated methods in the PySCF backend, and the convergence they need.

PySCF has analytic gradients for MP2, CCSD and CCSD(T) but no analytic Hessian
for any of them, so the backend differences the gradient. That is only sound if
the gradient is converged far past the default, and the first version of this
code was not: against water's analytic HF Hessian the finite-difference route
was 5% out overall and 9% out on the bend, because PySCF's default
conv_tol_grad (~3e-5) leaves ~1.5e-3 Hartree/Bohr^2 of noise once differenced
over 2 x 0.01 Bohr. The softest mode carries the largest fraction of that.

The decisive test is the first one below: run the SAME method through BOTH
routes. HF has an analytic Hessian, so differencing its analytic gradient must
reproduce it. Any disagreement is the finite-difference machinery, not physics.
"""

from __future__ import annotations

import numpy as np
import pytest

import dev.pyscf_backend  # noqa: F401  (registers "pyscf_hf")
from backend.registry import get_backend
from backend.spectral.centrifugal_distortion import normal_modes, rigid_mode_count

WATER = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.7575, 0.5868],
                  [0.0, -0.7575, 0.5868]])
MASSES = np.array([15.9949146196, 1.00782503207, 1.00782503207])
ELEMS = ["O", "H", "H"]

#: Water's experimental harmonic frequencies (cm-1). The bend is what matters
#: here: alpha carries 1/omega through both the zero-point amplitude and the
#: Coriolis sum, so a stiff bend is a systematic bias in the correction.
EXPERIMENTAL_BEND_CM = 1649.0


def _freqs(hess):
    omega, _ = normal_modes(hess, MASSES, n_rigid=rigid_mode_count(WATER, MASSES))
    return np.sort(omega)[-3:]


def test_finite_difference_hessian_reproduces_the_analytic_one():
    """The route is verified at a level where both routes exist."""
    backend = get_backend("pyscf_hf")(elems=ELEMS, method="hf", basis="6-31g")
    analytic = backend.run_hessian(WATER).hessian_bohr
    differenced = backend._fd_hessian(WATER).hessian_bohr

    rel = np.abs(differenced - analytic).max() / np.abs(analytic).max()
    assert rel < 1e-4, f"finite-difference Hessian is {rel:.2e} from analytic"
    assert np.abs(_freqs(differenced) - _freqs(analytic)).max() < 1.0


def test_correlated_methods_are_recognised_not_treated_as_functionals():
    """Without this branch 'mp2' would be handed to dft.RKS as an xc string."""
    for method in ("mp2", "ccsd", "ccsd(t)"):
        backend = get_backend("pyscf_hf")(elems=ELEMS, method=method, basis="sto-3g")
        assert backend.correlated, f"{method} not recognised as correlated"
        assert backend._tight, "a differenced gradient needs tight convergence"
    assert not get_backend("pyscf_hf")(elems=ELEMS, method="b3lyp").correlated


def test_correlation_removes_the_bend_stiffness_that_biases_alpha():
    """The reason to want a correlated force field at all.

    RHF is ~11% stiff on water's bend against the experimental harmonic value,
    and that error propagates into every correction through 1/omega. MP2 at the
    same basis removes essentially all of it. This is the measurement that says
    the correlated path is worth its cost, so it is pinned rather than
    described.
    """
    hf = get_backend("pyscf_hf")(elems=ELEMS, method="hf", basis="6-31g")
    mp2 = get_backend("pyscf_hf")(elems=ELEMS, method="mp2", basis="6-31g")

    hf_bend = _freqs(hf.run_hessian(WATER).hessian_bohr)[0]
    mp2_bend = _freqs(mp2.run_hessian(WATER).hessian_bohr)[0]

    hf_err = abs(hf_bend - EXPERIMENTAL_BEND_CM) / EXPERIMENTAL_BEND_CM
    mp2_err = abs(mp2_bend - EXPERIMENTAL_BEND_CM) / EXPERIMENTAL_BEND_CM

    assert hf_err > 0.08, f"RHF bend error was {hf_err:.1%}, expected ~11%"
    assert mp2_err < 0.03, f"MP2 bend error was {mp2_err:.1%}, expected ~1%"
    assert mp2_err < hf_err / 3.0


def test_correlated_energies_are_ordered_as_physics_requires():
    """E(HF) > E(MP2) and CCSD below HF: a cheap guard on the wiring."""
    energies = {}
    for method in ("hf", "mp2", "ccsd"):
        backend = get_backend("pyscf_hf")(elems=ELEMS, method=method, basis="sto-3g")
        energies[method] = backend.run_gradient(WATER).energy
    assert energies["mp2"] < energies["hf"]
    assert energies["ccsd"] < energies["hf"]


def test_gradient_is_analytic_for_correlated_methods():
    """Cheap enough that optimise() works at every method, which is the point:
    only the Hessian is differenced."""
    backend = get_backend("pyscf_hf")(elems=ELEMS, method="mp2", basis="sto-3g")
    result = backend.run_gradient(WATER)
    assert result.gradient_bohr.shape == (9,)
    assert np.isfinite(result.gradient_bohr).all()
    assert np.linalg.norm(result.gradient_bohr) > 0.0


def test_ccsd_t_uses_its_own_gradient_not_ccsd_s():
    """The trap this guards against produced a silently mislabelled method.

    PySCF has no CCSD(T) object: the triples are an energy added to a
    converged CCSD. Taking derivatives from that CCSD object therefore returns
    the CCSD gradient, so an earlier version of this backend reported CCSD(T)
    energies with CCSD forces -- water's frequencies came out identical to
    CCSD to every printed digit while the label said CCSD(T). The gradient now
    comes from pyscf.grad.ccsd_t, and on water the two differ by 5.2e-05
    Hartree/Bohr.

    Small, but it is the quantity the whole correction chain is built on, and
    a benchmark that cannot tell two methods apart is worse than one that
    omits the method.
    """
    ccsd = get_backend("pyscf_hf")(elems=ELEMS, method="ccsd", basis="sto-3g")
    ccsd_t = get_backend("pyscf_hf")(elems=ELEMS, method="ccsd(t)", basis="sto-3g")

    g_ccsd = ccsd.run_gradient(WATER)
    g_ccsd_t = ccsd_t.run_gradient(WATER)

    assert g_ccsd_t.energy < g_ccsd.energy, "triples must lower the energy"
    difference = np.abs(g_ccsd_t.gradient_bohr - g_ccsd.gradient_bohr).max()
    assert difference > 1e-8, (
        "CCSD(T) gradient is identical to CCSD -- the triples gradient is not "
        "being used"
    )
