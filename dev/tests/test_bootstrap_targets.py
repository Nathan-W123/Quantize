"""The bootstrap checks the reported covariance instead of describing a third one.

Measured on the engine before this change (water config through the CLI path,
48 draws; formyl fluoride HF/6-31G, 48 draws), three defects compounded:

* Draws came from each constant's quoted sigma independently, while the fit
  weights by the effective sigma and the reported covariance propagates the
  model error correlated across isotopologues. The cloud estimated neither.
* A Hessian refresh inside a refit rebuilt the vibrational corrections from
  the UNPERTURBED raw constants and wrote them over the draw, so the sample
  refitted the base data (51% of draws on the water config, per the item-3
  verifier).
* The gate refused samples whose final residual was over twice the base
  fit's -- which perturbed data is by construction: 25 of 48 refused on water,
  39 of 48 on formyl fluoride, nearly all at their own optimum.

The cloud came out 1.86x narrower than its own target on water and 1.5-2x on
formyl fluoride. Now "error-model" draws the constants from
constant_error_covariance (the matrix the reported covariance propagates),
corrections are frozen for the refit, and a sample is gated on reaching its
own optimum. "rto" also draws the quantum prior's centre, so its cloud is the
reported posterior. Measured after: water, 1000 draws, cloud/analytic 0.99
(bond) and 1.05 (angle) for error-model, 0.97 and 1.03 for rto; formyl
fluoride, 400 draws, error-model 0.96-1.04 on all six coordinates; every draw
usable.
"""

from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.bootstrap import (  # noqa: E402
    BootstrapEngine, _freeze_corrections, _run_bootstrap_worker)
from backend.uncertainty_v2.models import BootstrapConfig  # noqa: E402
from backend.uncertainty_v2.reporting import format_bootstrap_summary  # noqa: E402
from backend.uncertainty_v2.resampling import (  # noqa: E402
    draw_quantum_prior_shift, resample_error_model)

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"
MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = np.array([835840.29, 435351.72, 278138.70])
B0_D2O = np.array([462324.0, 218038.0, 145258.0])


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
                     [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0]])


@pytest.fixture(scope="module")
def config_fit():
    """The water config the CLI runs, fitted (corrections from the Hessian on)."""
    from runner.run_from_config import build_optimizer_from_config
    from runner.usability import load_config
    cfg = load_config(CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
        opt.run()
    return opt


@pytest.fixture(scope="module")
def fixture_fit():
    """Two species with a 2.4% model error: the correlated systematic is live."""
    isos = [{"name": n, "masses": m, "obs_constants": list(b), "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.024 * b)}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O), ("D2-16O", MASSES_D2O, B0_D2O))]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 max_iter=3, quantum_prior_sigma_ang=0.02,
                                 auto_systematic_sigma=False)
        opt.run()
    return opt


# ── The draws ────────────────────────────────────────────────────────────────

def test_the_error_covariance_is_what_the_reported_covariance_propagates(fixture_fit):
    """constant_error_covariance and _systematic_covariance are one error
    model: A (D Sigma D - I) A^T is the systematic term exactly."""
    opt = fixture_fit
    x = opt.coords
    cov, _ = opt.laplace_approximation(coords=x, check_gauss_newton=False)
    cov, _b, _n = opt.physical_covariance(cov, coords=x)
    J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
    Sigma = opt.constant_error_covariance(coords=x)
    D = np.diag(scale)
    A = cov @ J[:scale.size].T
    expect = A @ (D @ Sigma @ D - np.eye(scale.size)) @ A.T
    got = opt._systematic_covariance(J, cov, coords=x, row_scale=scale)
    np.testing.assert_allclose(got, expect, rtol=1e-8, atol=1e-12 * np.abs(got).max())


def test_the_error_model_draws_have_that_covariance(fixture_fit):
    opt = fixture_fit
    Sigma = opt.constant_error_covariance()
    rng = np.random.default_rng(3)
    base = np.concatenate([np.asarray(i["obs_constants"], float) for i in opt.spectral.isotopologues])
    deltas = np.array([np.concatenate([np.asarray(i["obs_constants"], float) for i in
                                       resample_error_model(opt.spectral.isotopologues, Sigma, rng)]) - base
                       for _ in range(20000)])
    emp = np.cov(deltas.T)
    assert np.linalg.norm(emp - Sigma) / np.linalg.norm(Sigma) < 0.03
    # The model error is shared: H2O's A and D2O's A move together (rho 0.99).
    corr = emp[0, 3] / np.sqrt(emp[0, 0] * emp[3, 3])
    assert corr == pytest.approx(0.99, abs=0.01)


def test_the_rto_shift_has_the_prior_s_covariance(config_fit):
    """xi = H eta with eta ~ N(0, (alpha H)^-1): Cov(xi) = H / alpha on the
    physical directions."""
    opt = config_fit
    rng = np.random.default_rng(5)
    xi = np.array([draw_quantum_prior_shift(opt, rng) for _ in range(20000)])
    _g, Hp = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    Hs = opt.optimizer._spd_hessian(Hp)
    alpha = opt.quantum_prior_weight(refresh=True)
    B = opt.physical_basis()
    expect = B @ (B.T @ Hs @ B) @ B.T / alpha
    emp = np.cov(xi.T)
    assert np.linalg.norm(emp - expect) / np.linalg.norm(expect) < 0.03


# ── The refit keeps its draw ─────────────────────────────────────────────────

def _forced_refresh_copy(opt):
    """A base whose refits refresh the Hessian at every quantum update and
    update at every step -- the schedule under which a correction rebuild
    fires as soon as a refit moves."""
    base = deepcopy(opt)
    base.orca_update_thresh = 0.0
    base.hess_recalc_every = 1
    base.adaptive_hess_schedule = False
    return base


def test_a_draw_survives_a_hessian_refresh_inside_the_refit(config_fit, tmp_path):
    base = _forced_refresh_copy(config_fit)
    Sigma = base.constant_error_covariance()
    pert = resample_error_model(base.spectral.isotopologues, Sigma, np.random.default_rng(9))
    with contextlib.redirect_stdout(io.StringIO()):
        res = _run_bootstrap_worker((base, pert, str(tmp_path / "s"), 1, 9, None))
    assert res.success and res.draw_kept is True


def test_without_the_freeze_the_refresh_erases_the_draw(config_fit, tmp_path):
    """The control: the same refit with the corrections live is fitted to the
    unperturbed constants -- so the test above can fail."""
    base = _forced_refresh_copy(config_fit)
    assert base._harmonic_from_hessian, "premise: this config rebuilds corrections"
    Sigma = base.constant_error_covariance()
    pert = resample_error_model(base.spectral.isotopologues, Sigma, np.random.default_rng(9))
    opt = deepcopy(base)
    opt.spectral.isotopologues = pert
    drawn = [np.asarray(i["obs_constants"], float).copy() for i in pert]
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    kept = all(np.allclose(np.asarray(i["obs_constants"], float), d)
               for i, d in zip(opt.spectral.isotopologues, drawn))
    assert not kept
    frozen = deepcopy(base)
    _freeze_corrections(frozen)
    assert not frozen._harmonic_from_hessian and not frozen._anharmonic_from_hessian


# ── The weight belongs to the current quantum reference ──────────────────────

def test_the_weight_follows_the_quantum_reference(config_fit):
    """quantum_prior_weight was cached once and never cleared when a quantum
    update moved the reference, so a refit was described with the weight of
    a Hessian it had left (2097 against its own steps' 1848)."""
    opt = deepcopy(config_fit)
    a = opt.quantum_prior_weight()
    assert opt._quantum_prior_weight_cache == a
    opt.coords = opt.coords + np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.02, 0.0]])
    with contextlib.redirect_stdout(io.StringIO()):
        opt._run_hessian()
    assert opt._quantum_prior_weight_cache is None
    J, _ = opt.spectral.stacked(opt.coords)
    _g, Hp = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    expect = opt.optimizer.effective_quantum_weight(J, int(np.linalg.matrix_rank(J)), hessian=Hp)
    assert opt.quantum_prior_weight() == pytest.approx(expect, rel=1e-12)
    assert opt.quantum_prior_weight() != pytest.approx(a, rel=1e-6)


# ── The clouds against their analytic targets ────────────────────────────────

def _cloud(opt, strategy, n, seed, tmp_path):
    cfg = BootstrapConfig(n_samples=n, max_workers=2, seed=seed, strategy=strategy,
                          persist_samples=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return BootstrapEngine(opt, cfg).run(base_workdir=str(tmp_path))


@pytest.mark.parametrize("strategy", ["error-model", "rto"])
def test_the_cloud_reproduces_its_analytic_target(config_fit, strategy, tmp_path):
    """200 draws: the Monte Carlo error on a width is 5%, so 15% is three of
    them. Measured with these seeds: error-model 1.01 / 1.07, rto 0.96 / 1.07;
    with 1000 draws 0.99 / 1.05 and 0.97 / 1.03."""
    s = _cloud(config_fit, strategy, 200, 11, tmp_path)
    assert s.usable_samples == 200
    a = s.analytic
    for b, t in zip(a["bootstrap_std"], a["target_std"]):
        assert b / t == pytest.approx(1.0, abs=0.15)
    if strategy == "rto":
        assert a["target_std"] == pytest.approx(a["posterior_std"])
    else:
        assert all(t < p for t, p in zip(a["target_std"], a["posterior_std"]))


def test_rto_refuses_what_it_cannot_sample(fixture_fit, tmp_path):
    opt = deepcopy(fixture_fit)
    opt._prior_target_coords = np.asarray(opt.coords, dtype=float)
    with pytest.raises(ValueError, match="prior_target_coords"):
        BootstrapEngine(opt, BootstrapConfig(n_samples=2, strategy="rto")).run(str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        with_prior = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(),
            isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                            "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}],
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=2,
            quantum_prior_sigma_ang=0.02, ic_prior_weight=1.0)
        with_prior.run()
    with pytest.raises(ValueError, match="internal-coordinate prior"):
        BootstrapEngine(with_prior, BootstrapConfig(n_samples=2, strategy="rto")).run(str(tmp_path))


def test_the_summary_says_what_it_estimates(config_fit, tmp_path):
    import json
    s = _cloud(config_fit, "error-model", 6, 3, tmp_path)
    txt = format_bootstrap_summary(s)
    assert "estimates: data and model error alone" in txt
    assert "posterior" in txt and "analytic" in txt
    json.dumps(s.to_dict())                      # crosses to the report as JSON
    assert s.to_dict()["analytic"]["names"]
