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
refits are held on the base's corrections and quantum model, and a sample is
gated on reaching its own optimum. "rto" also draws the quantum prior's
centre, so its cloud is the reported posterior.

Measured after (seed 7, 2 workers; paired = cloud sd over the sd of the
linear prediction of the same draws, which removes the draws' own Monte
Carlo error): water config, 1000 draws, error-model paired 1.000 / 1.004
(bond / angle), rto 1.003 / 1.004; formyl fluoride HF/6-31G, 400 draws,
error-model 0.98-1.02 and rto 0.98-1.00 on all six coordinates, the
shortfall on the two O=C-X angles being the data side's own nonlinearity.
Every draw usable. Unpaired cloud/analytic also carries the draws' own Monte
Carlo error (2.2% at 1000 draws, 3.5% at 400): formyl fluoride rto at seed 7
read 0.92-1.01, the linear prediction of those same draws 0.94-1.02.
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
    BootstrapEngine, _freeze_corrections, _refit_model, _run_bootstrap_worker, _same_model)
from backend.uncertainty_v2.models import BootstrapConfig  # noqa: E402
from backend.uncertainty_v2.reporting import format_bootstrap_summary  # noqa: E402
from backend.uncertainty_v2.resampling import (  # noqa: E402
    draw_quantum_prior_shift, resample_error_model, symmetric_sqrt)

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


def test_a_draw_is_the_symmetric_root_applied_to_the_seed():
    """An eigenvector factor V sqrt(w) is not unique -- any sign per vector,
    any rotation inside a degenerate block -- so the same seed could give a
    different draw on another BLAS. The symmetric root is unique: here the
    degenerate block is the identity, whatever eigh returns for it."""
    R = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
    Sigma = R @ np.diag([1.0, 1.0, 4.0]) @ R.T
    np.testing.assert_allclose(symmetric_sqrt(Sigma), R @ np.diag([1.0, 1.0, 2.0]) @ R.T,
                               atol=1e-12)
    isos = [{"name": "x", "obs_constants": [0.0, 0.0, 0.0]}]
    z = np.random.default_rng(4).standard_normal(3)
    got = resample_error_model(isos, Sigma, np.random.default_rng(4))[0]["obs_constants"]
    np.testing.assert_allclose(got, R @ np.diag([1.0, 1.0, 2.0]) @ R.T @ z, atol=1e-12)
    # A perturbation at round-off level moves the draw at round-off level.
    E = 1e-13 * np.random.default_rng(1).standard_normal((3, 3))
    moved = resample_error_model(isos, Sigma + E + E.T, np.random.default_rng(4))[0]["obs_constants"]
    np.testing.assert_allclose(moved, got, atol=1e-10)


def test_the_rto_shift_does_not_depend_on_the_basis_of_the_subspace(config_fit, monkeypatch):
    """physical_basis may return any orthonormal basis of the physical
    directions; the shift for a seed must be the same for all of them."""
    opt = deepcopy(config_fit)
    B = opt.physical_basis()
    Q = np.linalg.qr(np.random.default_rng(2).standard_normal((B.shape[1], B.shape[1])))[0]
    a = draw_quantum_prior_shift(opt, np.random.default_rng(8))
    monkeypatch.setattr(opt, "physical_basis", lambda coords=None, n=None: B @ Q)
    b = draw_quantum_prior_shift(opt, np.random.default_rng(8))
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12 * np.abs(a).max())


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


def test_a_refit_holds_the_base_quantum_model(config_fit, tmp_path):
    """Even on a schedule that refreshes the quantum point at every step,
    the worker's refit makes no quantum update: it fits the quadratic model
    the reported covariance, the analytic target and the rto shift were
    built from."""
    base = _forced_refresh_copy(config_fit)
    calls = []
    base._update_orca = lambda: calls.append(1)
    Sigma = base.constant_error_covariance()
    pert = resample_error_model(base.spectral.isotopologues, Sigma, np.random.default_rng(9))
    with contextlib.redirect_stdout(io.StringIO()):
        res = _run_bootstrap_worker((base, pert, str(tmp_path / "s"), 1, 9, None))
    assert res.success and res.usable and res.moved_ang > 1e-4
    assert calls == []
    assert res.draw_kept is True


def test_a_draw_survives_a_hessian_refresh_inside_the_refit(config_fit):
    """The correction freeze on its own: with quantum updates forced back on
    after it, Hessian refreshes happen and the drawn constants survive them."""
    opt = deepcopy(_forced_refresh_copy(config_fit))
    _freeze_corrections(opt)
    opt.orca_update_thresh = 0.0
    refreshes = []
    _orig = opt._run_hessian
    opt._run_hessian = lambda: (refreshes.append(1), _orig())[1]
    Sigma = opt.constant_error_covariance()
    pert = resample_error_model(opt.spectral.isotopologues, Sigma, np.random.default_rng(9))
    opt.spectral.isotopologues = pert
    drawn = [np.asarray(i["obs_constants"], float).copy() for i in pert]
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert refreshes, "premise: the refit refreshed the Hessian"
    assert all(np.allclose(np.asarray(i["obs_constants"], float), d)
               for i, d in zip(opt.spectral.isotopologues, drawn))


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


def test_the_kept_check_sees_the_targets_and_the_weighting(fixture_fit):
    """draw_kept used to compare the drawn constants only. A mass-dependent
    refit rewrites the B0->Be targets on top of them, and a rescale the
    sigmas, and neither showed."""
    opt = deepcopy(fixture_fit)
    ref = _refit_model(opt)
    assert _same_model(ref, _refit_model(opt))
    iso = opt.spectral.isotopologues[0]
    iso["delta_total_constants"] = np.asarray(
        iso.get("delta_total_constants", np.zeros(3)), float) + np.array([0.0, 5.0, 0.0])
    assert not _same_model(ref, _refit_model(opt))
    opt = deepcopy(fixture_fit)
    opt.spectral.scale_sigma(2.0)
    assert not _same_model(ref, _refit_model(opt))


def test_a_refit_keeps_the_base_s_targets_and_weighting(tmp_path):
    """The rm1 model refitted its B0->Be offsets at every refit iteration,
    which the reported covariance and the certificate both hold fixed; a
    chi-square rescale re-weighted the refit under the error model."""
    isos = [{"name": n, "masses": m, "obs_constants": list(b), "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.024 * b)}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O), ("D2-16O", MASSES_D2O, B0_D2O))]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 max_iter=60, quantum_prior_sigma_ang=0.02,
                                 auto_systematic_sigma=False,
                                 mass_dependent_model="rm1", chi2_rescale=True)
        opt.run()
    frozen = deepcopy(opt)
    _freeze_corrections(frozen)
    assert frozen._mass_dependent_model is None and frozen.chi2_rescale is False
    calls = []
    _orig = type(opt.spectral).fit_mass_dependent_correction
    opt.spectral.fit_mass_dependent_correction = (
        lambda *a, **k: (calls.append(1), _orig(opt.spectral, *a, **k))[1])
    pert = resample_error_model(opt.spectral.isotopologues, opt.constant_error_covariance(),
                                np.random.default_rng(4))
    with contextlib.redirect_stdout(io.StringIO()):
        res = _run_bootstrap_worker((opt, pert, str(tmp_path / "s"), 1, 4, None))
    assert res.success and res.draw_kept is True and calls == []
    assert res.mode_distance is not None and res.mode_distance < 0.02


def test_a_refit_that_runs_out_of_iterations_at_its_optimum_is_a_draw(config_fit, tmp_path):
    """With max_iter 8 every refit ends on max_iter (the plateau stop cannot
    fire before iteration 11); they used to be refused, all of them, even at
    their optimum."""
    base = deepcopy(config_fit)
    base.max_iter = 8
    s = _cloud(base, "error-model", 6, 3, tmp_path)
    assert all(r.termination == "max_iter" for r in s.sample_results), "premise"
    assert s.usable_samples == 6
    assert all(r.mode_distance < 0.02 for r in s.sample_results)


def test_the_gate_is_stated_in_the_cloud_s_own_widths(config_fit):
    """distance_to_mode is in posterior widths; the error-model cloud is the
    posterior less its prior part, so the limit and the polish threshold
    shrink by the smallest cloud-to-posterior width ratio (0.83 here, 0.075
    on formyl fluoride)."""
    em = BootstrapEngine(config_fit, BootstrapConfig(n_samples=2, strategy="error-model"))
    a = em.analytic_reference()
    ratio = min(t / p for t, p in zip(a["target_std"], a["posterior_std"]))
    assert ratio < 0.9, "premise: the prior holds part of this posterior"
    limit, polish = em._gate_limits()
    assert limit == pytest.approx(0.25 * ratio)
    assert polish == pytest.approx(0.02 * ratio)
    rto = BootstrapEngine(config_fit, BootstrapConfig(n_samples=2, strategy="rto"))
    assert rto._gate_limits() == (0.25, 0.02)


def _one_constant_split():
    iso = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": [278138.70],
            "sigma_constants": [0.2], "component_indices": [2],
            "sigma_systematic_constants": [0.024 * 278138.70]}]
    x0 = _coords() + np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=x0, isotopologues=iso,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 max_iter=200, quantum_prior_sigma_ang=0.02,
                                 auto_systematic_sigma=False, objective_mode="split",
                                 symmetry="C2v")
        opt.run()
    return opt


@pytest.mark.parametrize("strategy", ["error-model", "rto", "isotopologue-gaussian"])
def test_the_bootstrap_refuses_the_split_objective(strategy, tmp_path):
    """The split posterior leaves out the step's range-null coupling, so a
    split cloud estimates something its analytic column does not describe:
    error-model on this one-constant water gave the angle 0.04x its target."""
    opt = _one_constant_split()
    with pytest.raises(ValueError, match="split objective"):
        BootstrapEngine(opt, BootstrapConfig(n_samples=2, strategy=strategy)).run(str(tmp_path))
    assert BootstrapEngine.strategy_problem(opt, strategy, fitted=False)
    # The split report says so too.
    post = opt.internal_coordinate_posterior()
    assert any("split objective" in w for w in post["warnings"])
    # A worker called directly still refreshes under split (the hold is
    # for the joint objective's quadratic only).
    held = deepcopy(opt)
    _freeze_corrections(held)
    assert held.orca_update_thresh == opt.orca_update_thresh


def test_rto_refuses_the_split_objective(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(),
            isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                            "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}],
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=2,
            quantum_prior_sigma_ang=0.02, objective_mode="split")
        opt.run()
    with pytest.raises(ValueError, match="split objective"):
        BootstrapEngine(opt, BootstrapConfig(n_samples=2, strategy="rto")).run(str(tmp_path))


def test_the_projected_reference_follows_a_quantum_update(config_fit):
    """_projected_reference_terms was cached on the coordinates alone, so a
    quantum update at unchanged coordinates paired the new reference point
    with the old gradient and Hessian."""
    opt = deepcopy(config_fit)
    before = opt.distance_to_mode()
    with contextlib.redirect_stdout(io.StringIO()):
        opt._run_hessian()                       # a new reference, same coordinates
    after = opt.distance_to_mode()
    opt._proj_ref_terms_cache = None
    opt._quantum_prior_weight_cache = None
    assert after == pytest.approx(opt.distance_to_mode(), rel=1e-12)
    assert after != pytest.approx(before, rel=1e-3), "premise: the reference moved"


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


def _paired(opt, summary, strategy):
    """Each usable sample against the linear prediction of its own draw,
    rebuilt from the seed it carries: C (J^T D e + alpha xi). Returns the
    cloud's sd over the prediction's, and their correlation, per internal
    coordinate. Pairing removes the draws' own Monte Carlo error, which is
    what lets a 1.5% tolerance mean something at 200 draws."""
    from backend.internal.internal_fit import InternalCoordinateSet
    x = np.asarray(opt.coords, float)
    cov, _ = opt.laplace_approximation(coords=x, check_gauss_newton=False)
    cov, _b, _n = opt.physical_covariance(cov, coords=x)
    J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
    A = cov @ J[:scale.size].T
    Sigma = opt.constant_error_covariance()
    isos = opt.spectral.isotopologues
    b0 = np.concatenate([np.asarray(i["obs_constants"], float) for i in isos])
    cs = InternalCoordinateSet(x, opt.elems, opt._ic_use_dihedrals)
    Bq = np.asarray(cs.active_B_matrix(x), float)
    q0 = np.asarray(cs.active_values(x), float)
    lin, got = [], []
    for r in summary.usable_results:
        rng = np.random.default_rng(r.seed)
        e = np.concatenate([np.asarray(i["obs_constants"], float)
                            for i in resample_error_model(isos, Sigma, rng)]) - b0
        d = A @ (scale * e)
        if strategy == "rto":
            d = d + opt.quantum_prior_weight(refresh=True) * cov @ draw_quantum_prior_shift(opt, rng)
        lin.append(Bq @ d)
        got.append(np.asarray(cs.active_values(np.asarray(r.coords, float))) - q0)
    lin, got = np.array(lin), np.array(got)
    return (got.std(axis=0, ddof=1) / lin.std(axis=0, ddof=1),
            np.array([np.corrcoef(lin[:, k], got[:, k])[0, 1] for k in range(lin.shape[1])]))


@pytest.mark.parametrize("strategy", ["error-model", "rto"])
def test_the_cloud_reproduces_its_analytic_target(config_fit, strategy, tmp_path):
    """200 draws: the Monte Carlo error on a width is 5%, so 15% is three of
    them. Paired with the linear prediction of the same draws the refits
    agree to 1.5%: measured (seed 11) error-model 1.001 / 1.006 and rto
    1.003 / 1.008. With the refits refreshing the quantum point, as they did
    before they were held on the base's model, the same check read 0.997 /
    1.025 and 0.982 / 1.014."""
    s = _cloud(config_fit, strategy, 200, 11, tmp_path)
    assert s.usable_samples == 200
    a = s.analytic
    for b, t in zip(a["bootstrap_std"], a["target_std"]):
        assert b / t == pytest.approx(1.0, abs=0.15)
    if strategy == "rto":
        assert a["target_std"] == pytest.approx(a["posterior_std"])
    else:
        assert all(t < p for t, p in zip(a["target_std"], a["posterior_std"]))
    ratio, corr = _paired(config_fit, s, strategy)
    assert np.all(np.abs(ratio - 1.0) < 0.015), ratio
    assert np.all(corr > 0.999), corr


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


def test_the_analytic_column_says_when_the_geometry_is_a_saddle():
    """H2O + D2O with C1 forced: the mode is a saddle along the antisymmetric
    stretch, so internal_coordinate_posterior refuses -- and the bootstrap's
    analytic and posterior columns, formed from the same Laplace, must say so
    rather than print a width the engine does not report."""
    isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": list(B0_H2O),
             "sigma_constants": [0.2] * 3},
            {"name": "D2-16O", "masses": MASSES_D2O, "obs_constants": list(B0_D2O),
             "sigma_constants": [0.2] * 3}]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 max_iter=40, quantum_prior_sigma_ang=0.02, symmetry="C1")
        opt.run()
        post = opt.internal_coordinate_posterior()
    assert "error" in post, "premise: a stationary saddle"
    a = BootstrapEngine(opt, BootstrapConfig(n_samples=2)).analytic_reference()
    assert a["no_posterior"] and "saddle" in a["no_posterior"]


def test_the_fit_loop_does_not_finite_difference_a_jacobian_it_discards(config_fit):
    """The loop read only the residuals of stacked_unweighted, twice per
    iteration, while it finite-differenced the unweighted Jacobian each
    time: 27% of a water refit and 35% of a formyl fluoride one."""
    opt = deepcopy(config_fit)
    calls = []
    _orig = type(opt.spectral).stacked_unweighted

    def counting(self, coords, want_jacobian=True):
        calls.append(want_jacobian)
        return _orig(self, coords, want_jacobian=want_jacobian)

    opt.spectral.stacked_unweighted = counting.__get__(opt.spectral)
    opt.coords = opt.coords + np.array([[0.0, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.005, 0.0]])
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert calls and not any(calls)
