"""The inertia-tensor derivative the engine was missing, and what it fixes.

``rotational_constants_mhz`` returns the sorted eigenvalues of the inertia
tensor, so every geometry derivative built from it has already thrown away the
off-diagonal elements. Two separate pieces of physics need them: the harmonic
contribution to alpha, whose coefficient in Watson's expansion of mu = I^-1
sums over all axes rather than one, and the Kivelson-Wilson tau tensor, whose
tau_abab, tau_bcbc and tau_caca components are off-diagonal by construction.

The symmetry structure of a_r is a free check that needs no reference data: for
a C2v XY2 molecule group theory forces the symmetric modes to have a purely
diagonal a_r and the antisymmetric stretch a purely off-diagonal one.
"""

from __future__ import annotations

import numpy as np
import pytest

import dev.analytic_water_backend  # noqa: F401  (registers "analytic_water")
from backend.registry import get_backend
from backend.spectral.centrifugal_distortion import (
    inertia_mode_derivatives,
    inertia_paf,
    inertia_tensor_amu_ang2,
    normal_modes,
)
from backend.spectral.harmonic_alpha import _HARMONIC_SCHEMES, compute_harmonic_alpha

from reference_molecules import (
    CO_COORDS,
    CO_MASSES,
    co_harmonic_alpha,
    co_hessian,
    co_pekeris_alpha,
)

_MHZ_TO_CM = 1.0 / 29979.2458
H2O_MASSES = np.array([15.9949146196, 1.00782503207, 1.00782503207])


def _water(r=0.95785, deg=104.508):
    half = np.radians(deg) / 2.0
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.sin(half), r * np.cos(half), 0.0],
                     [-r * np.sin(half), r * np.cos(half), 0.0]])


def _water_modes():
    coords = _water()
    hess = get_backend("analytic_water")(
        elems=["O", "H", "H"]).run_hessian(coords).hessian_bohr
    omega, l_mw = normal_modes(hess, H2O_MASSES, n_rigid=6)
    keep = omega >= 50.0
    return coords, hess, omega[keep], l_mw[:, keep]


# ── the tensor itself ───────────────────────────────────────────────────────

def test_inertia_tensor_diagonalises_to_the_principal_moments():
    """Anchor the new helper to the one the engine already trusts."""
    coords = _water()
    evals, _, coords_paf = inertia_paf(coords, H2O_MASSES)
    got = inertia_tensor_amu_ang2(coords_paf, H2O_MASSES)
    assert np.allclose(got, np.diag(evals), atol=1e-10)
    assert np.allclose(got, got.T)


def test_c2v_symmetry_splits_the_modes_into_diagonal_and_off_diagonal():
    """Group theory, reproduced to finite-difference precision.

    Water's bend and symmetric stretch are a1 and preserve C2v, so they can only
    change the principal moments: a_r is diagonal. The antisymmetric stretch is
    b2 and breaks the symmetry, so it can only tilt the tensor: a_r is purely
    off-diagonal, in the a-b block.

    This is the check that makes the new quantity trustworthy without any
    reference data -- an error in the frame rotation destroys the pattern
    immediately, which is how a first version of this calculation was caught.
    """
    coords, _, omega, l_mw = _water_modes()
    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    assert a.shape == (3, 3, 3)
    off = np.array([max(abs(a[r, i, j]) for i in range(3) for j in range(3) if i != j)
                    for r in range(3)])
    dia = np.array([max(abs(a[r, i, i]) for i in range(3)) for r in range(3)])
    # modes 0 and 1 are a1 (bend, symmetric stretch); mode 2 is b2.
    assert off[0] < 1e-6 and off[1] < 1e-6
    assert dia[0] > 1.0 and dia[1] > 1.0
    assert dia[2] < 1e-4
    assert off[2] > 1.0
    # and the off-diagonal element is in the a-b block, not involving c
    assert abs(a[2, 2, 0]) < 1e-6 and abs(a[2, 2, 1]) < 1e-6


def test_the_derivative_is_symmetric_for_every_mode():
    coords, _, _, l_mw = _water_modes()
    a = inertia_mode_derivatives(coords, H2O_MASSES, l_mw)
    for r in range(a.shape[0]):
        assert np.allclose(a[r], a[r].T, atol=1e-9)


def test_richardson_extrapolation_makes_it_step_independent():
    coords, _, _, l_mw = _water_modes()
    coarse = inertia_mode_derivatives(coords, H2O_MASSES, l_mw, fd_delta=0.04)
    fine = inertia_mode_derivatives(coords, H2O_MASSES, l_mw, fd_delta=0.01)
    assert np.allclose(coarse, fine, atol=1e-6)


# ── the harmonic scheme it enables ──────────────────────────────────────────

def test_both_schemes_are_selectable_and_unknown_ones_rejected():
    coords, hess, _, _ = _water_modes()
    for scheme in _HARMONIC_SCHEMES:
        compute_harmonic_alpha(hess, coords, H2O_MASSES, harmonic_scheme=scheme)
    with pytest.raises(ValueError, match="harmonic_scheme"):
        compute_harmonic_alpha(hess, coords, H2O_MASSES, harmonic_scheme="mills")


def test_the_historical_scheme_is_still_the_default():
    """It is known to be the wrong quantity, and still the default.

    What replaces it improves water's B and C by a lot and its A by nothing,
    and A is what sets the bond angle. Switching the default waits on the
    independent check against ozone's published tau parameters. See
    _HARMONIC_SCHEMES for the measurements.
    """
    coords, hess, _, _ = _water_modes()
    _, _, _, info = compute_harmonic_alpha(hess, coords, H2O_MASSES)
    assert info["harmonic_scheme"] == "eigenvalue_fd"


@pytest.mark.parametrize("scheme", list(_HARMONIC_SCHEMES))
def test_a_diatomic_is_unaffected_by_the_choice(scheme):
    """A diatomic's a_r has no off-diagonal part, so the schemes must coincide.

    This keeps the repository's only closed-form validation -- the Dunham and
    Pekeris relations for CO -- binding on both schemes rather than on one.
    """
    harm, _, _, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES, harmonic_scheme=scheme)
    full, _, _, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES, hessian_fn=co_hessian,
        harmonic_scheme=scheme)
    assert harm["B"] * _MHZ_TO_CM == pytest.approx(co_harmonic_alpha(), rel=1e-4)
    assert full["B"] * _MHZ_TO_CM == pytest.approx(co_pekeris_alpha(), rel=2e-3)


def test_the_schemes_agree_exactly_where_the_diagonal_a_r_dominates():
    """The agreement is the calibration; the disagreement is the physics.

    Wherever a_r has a substantial diagonal element the two routes coincide to
    six digits -- water's A and B on both a1 modes. That is what validates the
    new expression and its units: it is not a different formula that happens to
    be nearby, it is the same number computed a different way.

    They diverge in exactly two situations, and only one of them is settled:

      * a_r purely off-diagonal (the b2 antisymmetric stretch). Here the
        eigenvalue route is definitely wrong -- it is measuring eigenvalue
        repulsion -- and the two answers differ in sign.
      * a_r's diagonal element near zero, so that what the eigenvalue route
        picks up is dominated by d2I_xixi/dQ^2 (water's C on the bend, where
        a_CC is -0.006 against diagonal elements of 1.27 elsewhere). Whether
        that second-derivative term belongs in alpha is the open question:
        Watson's expansion says no, and this test does not assert either way.
    """
    coords, hess, _, _ = _water_modes()
    per = {}
    for scheme in ("eigenvalue_fd", "watson"):
        _, _, _, info = compute_harmonic_alpha(
            hess, coords, H2O_MASSES, harmonic_scheme=scheme)
        per[scheme] = np.asarray(info["alpha_per_mode_mhz"], dtype=float)
    old, new = per["eigenvalue_fd"], per["watson"]

    a = inertia_mode_derivatives(coords, H2O_MASSES, _water_modes()[3])
    for mode in (0, 1):
        for comp in (0, 1):
            assert abs(a[mode, comp, comp]) > 0.9, (mode, comp)
            assert old[comp, mode] == pytest.approx(new[comp, mode], rel=1e-5)

    # b2 mode: purely off-diagonal a_r, and the two routes disagree in sign on B
    assert np.sign(old[1, 2]) != np.sign(new[1, 2])

    # bend, C component: a_CC is negligible, so the discarded second-derivative
    # term is the whole difference.
    assert abs(a[0, 2, 2]) < 0.01
    assert abs(old[2, 0] - new[2, 0]) > 1000.0
