"""One quantum-prior weight, used by everything that reports on the fit.

`SubspaceOptimizer.effective_quantum_weight` says it plainly: "anything that
reports on the fit -- a covariance, a posterior, a log-probability -- has to
weight the two halves the same way the step did. Reading alpha_quantum
directly gives the legacy heuristic, which is not what the joint objective
minimises whenever quantum_prior_sigma_ang is set (the default)."

`laplace_approximation` followed that. `log_probability` and
`evaluate_posterior_state` did not -- they read alpha_quantum. On water the
two differ by 1811x (1.0 against 1810.58), so the MCMC sampled, and the
posterior report described, an objective nobody minimised.

Note what this does NOT change: `run()` never calls log_probability, so no
fitted geometry and no benchmark number moves. The blast radius is the MCMC
target and the reported log_posterior.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.posterior import evaluate_posterior_state  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(**kw):
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O, "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


@pytest.fixture(scope="module")
def fit():
    return _fit()


@pytest.fixture(scope="module")
def fit_cartesian():
    """Cartesian mode with priors on, so that branch is actually exercised."""
    return _fit(coordinate_mode="cartesian", use_internal_priors=True,
                prior_weight=1.0)


def _laplace_weight(opt):
    J, _ = opt.spectral.stacked(opt.coords)
    _g, H = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    return float(opt.optimizer.effective_quantum_weight(
        J, int(np.linalg.matrix_rank(J)), hessian=H))


def test_the_gap_being_closed_is_real(fit):
    """Guard the premise: if the two agreed, these tests would prove nothing."""
    assert abs(_laplace_weight(fit) / fit.optimizer.alpha_quantum) > 100.0


def test_weight_matches_what_the_laplace_uses(fit):
    assert fit.quantum_prior_weight() == pytest.approx(
        _laplace_weight(fit), rel=1e-12)


def test_weight_is_not_the_legacy_alpha(fit):
    assert fit.quantum_prior_weight() != pytest.approx(
        fit.optimizer.alpha_quantum, rel=1e-6)


def test_weight_does_not_move_with_geometry(fit):
    """A position-dependent weight makes log_probability a moving target.

    Under the legacy dynamic path the weight depends on rank(J), which
    depends on the geometry. A Metropolis chain against a target that
    changes shape as it walks samples nothing in particular, so the weight
    is computed once and held.
    """
    w0 = fit.quantum_prior_weight()
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert fit.quantum_prior_weight(
            coords=fit.coords + rng.normal(0, 0.05, fit.coords.shape)) == w0


def test_refresh_recomputes(fit):
    w = fit.quantum_prior_weight()
    assert fit.quantum_prior_weight(refresh=True) == pytest.approx(w, rel=1e-12)


def test_posterior_state_uses_the_same_weight(fit):
    """The third copy of this term, now reading from the same place."""
    st = evaluate_posterior_state(fit, fit.coords, use_proxy=True)
    expected = (-0.5 * st.chi_sq_prior
                - fit.quantum_prior_weight() * st.energy_term)
    assert st.log_prior == pytest.approx(expected, rel=1e-12)
    # And it is NOT the legacy value, which is what it used to be.
    legacy = -0.5 * st.chi_sq_prior - fit.optimizer.alpha_quantum * st.energy_term
    assert st.log_prior != pytest.approx(legacy, rel=1e-6)


def test_log_probability_decomposes_with_that_weight(fit):
    x = fit.coords
    _J, rw = fit.spectral.stacked(x, want_jacobian=False)
    chi = float(np.sum(np.asarray(rw) ** 2))
    chi_prior = 0.0
    if fit.internal_prior is not None and fit.coordinate_mode == "cartesian":
        _n, rp = fit.internal_prior.stacked(x, want_jacobian=False)
        chi_prior = fit.prior_weight * float(np.sum(np.asarray(rp) ** 2))
    expected = (-0.5 * (chi + chi_prior)
                - fit.quantum_prior_weight() * fit.quantum_prior_energy(x))
    assert float(fit.log_probability(x)) == pytest.approx(expected, rel=1e-12)


def test_translation_is_exactly_free(fit):
    """A rigid translation changes no constant and no energy."""
    lp0 = float(fit.log_probability(fit.coords))
    moved = fit.coords + np.array([0.37, -0.11, 0.05])
    assert float(fit.log_probability(moved)) == pytest.approx(lp0, abs=1e-6)


def test_projected_reference_terms_are_cached_and_correct(fit):
    g_direct, H_direct = fit._project_quantum_terms(
        fit.quantum.gradient, fit.quantum.hessian)
    g, H = fit._projected_reference_terms()
    assert np.array_equal(g, g_direct) and np.array_equal(H, H_direct)
    # Same objects back on a second call: the SVD is not repeated.
    g2, H2 = fit._projected_reference_terms()
    assert g2 is g and H2 is H


def test_cache_invalidates_when_the_geometry_moves(fit):
    """The projector depends on self.coords, which moves during a fit."""
    g1, _H1 = fit._projected_reference_terms()
    saved = fit.coords.copy()
    try:
        fit.coords = saved + 0.05
        g2, _H2 = fit._projected_reference_terms()
        assert not np.array_equal(g1, g2)
    finally:
        fit.coords = saved
    g3, _H3 = fit._projected_reference_terms()
    assert np.array_equal(g1, g3)


def test_internal_prior_residuals_match_without_the_jacobian(fit_cartesian):
    """Exercised with a real internal prior, not a fixture that has none."""
    opt = fit_cartesian
    if opt.internal_prior is None:
        pytest.skip("no internal prior on this fit")
    rng = np.random.default_rng(99)
    for _ in range(25):
        x = opt.coords + rng.normal(0.0, 0.02, opt.coords.shape)
        J, r_full = opt.internal_prior.stacked(x)
        J_none, r_cheap = opt.internal_prior.stacked(x, want_jacobian=False)
        assert J is not None and J_none is None
        assert np.array_equal(r_full, r_cheap)


def test_internal_prior_branch_is_really_taken(fit_cartesian):
    """Otherwise the test above is vacuous."""
    assert fit_cartesian.coordinate_mode == "cartesian"
    assert fit_cartesian.internal_prior is not None
