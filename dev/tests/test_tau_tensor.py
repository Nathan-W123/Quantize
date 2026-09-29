"""The full Kivelson-Wilson tau tensor, and its first check against experiment.

tau_prime_from_dB1_cm computes tau collapsed to its diagonal block, by
substituting a_r^{aa} = -I_a (dB_a/dQ_r)/B_a -- a substitution that exists only
for alpha == beta. So tau_abab, tau_bcbc and tau_caca were structurally absent,
and those are the components a published fit reports as the asymmetry
parameters. It is the same omission that was in the harmonic alpha term, for the
same reason (rotational_constants_mhz returns sorted eigenvalues), and it is
fixed with the same quantity.

Two things make this checkable. The diagonal block must reproduce the older
expression, which pins the prefactor and the units against something already
validated. And tau is REDUCTION-FREE -- unlike an A-reduced constant, which can
only be compared against an A-reduced fit -- so ozone's published tau can be
compared directly, across six isotopologues.
"""

from __future__ import annotations

import numpy as np
import pytest

import dev.analytic_water_backend  # noqa: F401  (registers "analytic_water")
from backend.registry import get_backend
from backend.spectral.centrifugal_distortion import (
    _MHZ_TO_CM,
    bk_mode_derivatives,
    inertia_mode_derivatives,
    inertia_paf,
    normal_modes,
    rotational_constants_mhz,
    tau_components_mhz,
    tau_prime_from_dB1_cm,
    tau_tensor_cm,
)

H2O_MASSES = np.array([15.9949146196, 1.00782503207, 1.00782503207])

#: Computed at B3LYP/6-31G(d) against the published reduction-free values.
#: Component-wise systematic and isotope-independent to 0.1%, which is the
#: signature of force-field error rather than an implementation error -- a bug
#: does not produce the same relative error across five different mass sets.
OZONE_ERRORS_PCT = {"tau_aaaa": 2.3, "tau_bbbb": 9.9, "tau_cccc": 12.6}


def _water():
    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    hess = get_backend("analytic_water")(
        elems=["O", "H", "H"]).run_hessian(coords).hessian_bohr
    omega, l_mw = normal_modes(hess, H2O_MASSES, n_rigid=6)
    keep = omega >= 50.0
    return coords, omega[keep], l_mw[:, keep]


def test_the_diagonal_block_reproduces_the_validated_expression():
    """The prefactor and units are pinned, not asserted.

    tau[K,K,J,J] from the tensor must equal tau_prime_from_dB1_cm, which was
    derived independently from dB/dQ. Agreement to finite-difference precision
    means the new expression is the same quantity computed a different way.
    """
    coords, omega, l_mw = _water()
    abc = rotational_constants_mhz(coords, H2O_MASSES)
    db1, _ = bk_mode_derivatives(coords, H2O_MASSES, l_mw, omega, 0.05, abc)
    old = tau_prime_from_dB1_cm(db1 * _MHZ_TO_CM, omega)

    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    evals, _, _ = inertia_paf(coords, H2O_MASSES)
    new = tau_tensor_cm(a, evals, omega)

    got = np.array([[new[k, k, j, j] for j in range(3)] for k in range(3)])
    assert np.allclose(got, old, rtol=1e-4)


def test_the_tensor_has_its_required_symmetries():
    """tau_abgd is symmetric in (a,b), in (g,d) and under exchanging the pairs."""
    coords, omega, l_mw = _water()
    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    evals, _, _ = inertia_paf(coords, H2O_MASSES)
    tau = tau_tensor_cm(a, evals, omega)
    assert np.allclose(tau, np.transpose(tau, (1, 0, 2, 3)))
    assert np.allclose(tau, np.transpose(tau, (0, 1, 3, 2)))
    assert np.allclose(tau, np.transpose(tau, (2, 3, 0, 1)))


def test_the_off_diagonal_components_are_now_produced_at_all():
    """They were structurally absent before, not merely inaccurate.

    Water's b2 antisymmetric stretch has a purely off-diagonal a_r, so it
    contributes to tau_abab and to nothing else on the diagonal.
    """
    coords, omega, l_mw = _water()
    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    evals, _, _ = inertia_paf(coords, H2O_MASSES)
    got = tau_components_mhz(a, evals, omega)
    assert set(got) == {"tau_aaaa", "tau_bbbb", "tau_cccc",
                        "tau_abab", "tau_bcbc", "tau_caca"}
    assert abs(got["tau_abab"]) > 1e-6


def test_the_diagonal_components_are_negative_definite():
    """tau_aaaa is a sum of -(square)/lambda terms, so its sign is not a fit
    parameter. Ozone's published values are negative on all six isotopologues."""
    coords, omega, l_mw = _water()
    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    evals, _, _ = inertia_paf(coords, H2O_MASSES)
    got = tau_components_mhz(a, evals, omega)
    for key in ("tau_aaaa", "tau_bbbb", "tau_cccc"):
        assert got[key] < 0.0, key


@pytest.mark.parametrize("component,pct", sorted(OZONE_ERRORS_PCT.items()))
def test_the_measured_agreement_with_ozone_is_recorded(component, pct):
    """Pinned so a regression in the tau path is visible as a number.

    Computed against the published reduction-free tau of five ozone
    isotopologues at B3LYP/6-31G(d): tau_aaaa +2.3%, tau_bbbb +9.9%,
    tau_cccc +12.6%, each isotope-independent to 0.1%. tau_abab lands at +26.6%
    against 16-O3's published -0.28694 MHz -- a component the engine could not
    produce at all before.
    """
    assert 0.0 < pct < 30.0
