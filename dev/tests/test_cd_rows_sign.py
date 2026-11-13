"""Centrifugal-distortion rows: their sign, the default path, the bootstrap.

Measured before (water on the analytic surface, H2O + D2O, observed
distortion constants 5% off the model's own prediction, 2% sigma, CD
fitting on):

* With the default sanitizer the cd_observed blocks were dropped, so the
  fit had no distortion rows at all and fit_cd_constants did nothing --
  silently, since the "no cd_observed" warning reads the raw isotopologues.
* With the sanitizer off the rows were stacked as predicted minus observed
  against J = d(predicted)/dx, so -J^T r was the gradient of -1/2 |r_cd|^2
  (cosine -1.0000 against finite differences): the fit was pushed away from
  the observed constants (weighted residual 7.25 -> 7.71), stopped 1.29
  widths from its own optimum, and 35 of 40 error-model refits were refused.
* The error-model and rto draws left the observed distortion constants
  alone while the analytic target counted their precision.

After: rows on the default path, cosine +1.0000, the fit 7e-5 widths from
its optimum (residual 7.25 -> 7.07), 40 of 40 refits usable.
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

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.spectral.centrifugal_distortion import CD_NAMES, compute_cd_constants  # noqa: E402
from backend.uncertainty_v2.bootstrap import BootstrapEngine  # noqa: E402
from backend.uncertainty_v2.models import BootstrapConfig  # noqa: E402
from backend.uncertainty_v2.resampling import resample_cd_observed  # noqa: E402

M_O, M_H, M_D = 15.9949146196, 1.00782503207, 2.0141017778
SPEC = (("H2-16O", [M_O, M_H, M_H], [835840.29, 435351.72, 278138.70]),
        ("D2-16O", [M_O, M_D, M_D], [462324.0, 218038.0, 145258.0]))


def _x0(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([[0.0, 0.0, 0.0], [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
                     [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0]])


def _isos(cd=None):
    out = []
    for name, m, b in SPEC:
        d = {"name": name, "masses": m, "obs_constants": list(b), "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.024 * np.asarray(b))}
        if cd:
            d["cd_observed"] = cd[name]
            d["cd_sigma"] = {k: 0.02 * abs(v) for k, v in cd[name].items()}
        out.append(d)
    return out


def _make(cd, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return MolecularOptimizer(
            elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(cd),
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=60,
            auto_systematic_sigma=False, fit_cd_constants=bool(cd),
            cd_weight=1.0 if cd else 0.0, **kw)


@pytest.fixture(scope="module")
def observed_cd():
    base = _make(None)
    with contextlib.redirect_stdout(io.StringIO()):
        base.run()
    hb = np.asarray(base.quantum._hessian_bohr, float)
    cd = {}
    for name, m, _b in SPEC:
        pred = compute_cd_constants(hb, base.coords, m)
        cd[name] = {k: 1.05 * float(v) for k, v in pred.as_dict().items()
                    if k in ("DJ", "DJK", "DK", "delta_J", "delta_K")}
    return cd


@pytest.fixture(scope="module")
def cd_fit(observed_cd):
    opt = _make(observed_cd)                      # the default sanitizer
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


def _n_rot(opt):
    return sum(len(i["obs_constants"]) for i in opt.spectral.isotopologues)


def test_the_default_path_keeps_the_observed_distortion_constants(cd_fit):
    J, _r = cd_fit.spectral.stacked(cd_fit.coords)
    assert J.shape[0] - _n_rot(cd_fit) == 6, "DJ, DJK, DK on each of two isotopologues"
    # delta_J and delta_K are observed here but not fitted (no validation;
    # water's predicted delta_K has the wrong sign).
    assert "delta_K" in cd_fit.spectral.isotopologues[0]["cd_observed"]


def test_the_distortion_rows_point_the_gradient_downhill(cd_fit):
    """-J^T r must be the gradient of 1/2 |r|^2 on the distortion rows as on
    the rotational ones."""
    sp, x, n = cd_fit.spectral, np.asarray(cd_fit.coords, float), _n_rot(cd_fit)
    J, r = sp.stacked(x)
    g = -(J[n:].T @ r[n:])

    def half(xx):
        _, q = sp.stacked(xx.reshape(-1, 3), want_jacobian=False)
        return 0.5 * float(np.sum(q[n:] ** 2))

    fd = np.zeros(x.size)
    for k in range(x.size):
        a, b = x.ravel().copy(), x.ravel().copy()
        a[k] += 1e-5
        b[k] -= 1e-5
        fd[k] = (half(a) - half(b)) / 2e-5
    cos = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd)))
    assert cos > 0.999


def test_a_fit_with_distortion_rows_reaches_its_own_optimum(cd_fit):
    """And the rows pull toward the observed constants: the fit that has them
    ends with a smaller distortion residual than the same fit without them
    (with the old sign it ended with a larger one)."""
    n = _n_rot(cd_fit)
    assert cd_fit.distance_to_mode() < 0.01
    plain = _make(None)
    with contextlib.redirect_stdout(io.StringIO()):
        plain.run()
    sp = cd_fit.spectral
    _J, r_with = sp.stacked(cd_fit.coords, want_jacobian=False)
    _J, r_without = sp.stacked(plain.coords, want_jacobian=False)
    assert np.linalg.norm(r_with[n:]) < np.linalg.norm(r_without[n:])


def test_the_distortion_draws_carry_the_rows_weights(cd_fit):
    """Each drawn constant moves its weighted residual by a unit normal."""
    sp = cd_fit.spectral
    sig = sp.cd_row_sigmas(cd_fit.coords)
    assert all(sig), "premise: both isotopologues have rows"
    rng = np.random.default_rng(1)
    z = []
    for _ in range(4000):
        p = resample_cd_observed(sp.isotopologues, sig, rng)
        z.append([(p[i]["cd_observed"][k] - sp.isotopologues[i]["cd_observed"][k]) / sig[i][k]
                  for i in range(len(p)) for k in CD_NAMES if k in sig[i]])
    z = np.array(z)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=0.05)
    # and the weight is what stacked applies: sqrt(cd_weight) / sigma, with
    # sigma the given 2% and the model's 40% in quadrature
    keys0 = [k for k in CD_NAMES if k in sig[0]]
    obs0 = sp.isotopologues[0]["cd_observed"]
    sig_cd = np.array([max(np.hypot(0.02 * abs(obs0[k]), 0.40 * abs(obs0[k])), 0.01) for k in keys0])
    np.testing.assert_allclose([sig[0][k] for k in keys0], sig_cd / np.sqrt(sp.cd_weight))


def test_an_error_model_bootstrap_with_distortion_rows_is_usable(cd_fit, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        s = BootstrapEngine(cd_fit, BootstrapConfig(n_samples=12, max_workers=2, seed=7,
                                                    strategy="error-model",
                                                    persist_samples=False)).run(str(tmp_path))
    assert s.usable_samples == 12
    assert all(r.mode_distance < 0.01 for r in s.sample_results)


# ── Round 2 ──────────────────────────────────────────────────────────────────

def _isos_no_sigma(cd):
    out = _isos(None)
    for d in out:
        d["cd_observed"] = cd[d["name"]]
    return out


def test_harmonic_cd_corrections_do_not_freeze_the_rows(observed_cd):
    """harmonic_cd_from_hessian attached a fixed prediction that the rows
    then used, so their Jacobian was zero (|J_cd| 0.000 against 974): the
    rows constrained nothing and added a constant to the residual."""
    fits = {}
    for harm in (False, True):
        opt = _make(observed_cd, harmonic_cd_from_hessian=harm)
        with contextlib.redirect_stdout(io.StringIO()):
            opt.run()
        J, _r = opt.spectral.stacked(opt.coords)
        fits[harm] = (np.asarray(opt.coords, float), np.linalg.norm(J[_n_rot(opt):]))
    assert fits[True][1] > 0.5 * fits[False][1] > 0.0
    np.testing.assert_allclose(fits[True][0], fits[False][0], atol=1e-9)


def test_the_default_cd_sigma_is_a_constant_of_the_objective(observed_cd):
    """Without a cd_sigma block the weight was 40% of the PREDICTED value,
    which moves with the geometry while its derivative is not in J: at
    cd_weight 1e4 and a 20% offset the fit stopped 4.2 widths from its own
    optimum. Now 40% of the observed value, fixed when the engine is built."""
    cd = {n: {k: 1.20 / 1.05 * v for k, v in block.items()} for n, block in observed_cd.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos_no_sigma(cd),
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=60,
            auto_systematic_sigma=False, fit_cd_constants=True, cd_weight=1e4)
        opt.run()
    iso = opt.spectral.isotopologues[0]
    rows = opt.spectral.cd_row_sigmas(opt.coords)[0]
    for k, v in rows.items():
        assert iso["cd_model_sigma"][k] == pytest.approx(max(0.40 * abs(iso["cd_observed"][k]), 0.01))
        assert v == pytest.approx(iso["cd_model_sigma"][k] / np.sqrt(1e4))
    assert opt.distance_to_mode() < 0.01


def test_the_cd_draws_carry_their_share_of_the_target(observed_cd, tmp_path):
    """Where the distortion rows matter (cd_weight 4e6 with the model's 40%
    in their sigma: a material share of the target variance here), the
    error-model draws' linear prediction has the target's covariance only
    with the distortion constants drawn too."""
    from backend.internal.internal_fit import InternalCoordinateSet
    from backend.uncertainty_v2.resampling import resample_error_model
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(observed_cd),
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=60,
            auto_systematic_sigma=False, fit_cd_constants=True, cd_weight=4e6)
        opt.run()
    x = np.asarray(opt.coords, float)
    cov, _ = opt.laplace_approximation(coords=x, check_gauss_newton=False)
    cov, _b, _n = opt.physical_covariance(cov, coords=x)
    J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
    n = scale.size
    _hs, h_prior, _p = opt._posterior_blocks(x)
    c_sys = opt._systematic_covariance(J, cov, coords=x, row_scale=scale)
    target = cov + c_sys - cov @ h_prior @ cov
    cs = InternalCoordinateSet(x, opt.elems, opt._ic_use_dihedrals)
    Bq = np.asarray(cs.active_B_matrix(x), float)
    Sigma = opt.constant_error_covariance()
    sig = opt.spectral.cd_row_sigmas(x)
    isos = opt.spectral.isotopologues
    b0 = np.concatenate([np.asarray(i["obs_constants"], float) for i in isos])
    rng = np.random.default_rng(2)
    with_cd, without = [], []
    for _ in range(4000):
        p = resample_error_model(isos, Sigma, rng)
        p = resample_cd_observed(p, sig, rng)
        e = np.concatenate([np.asarray(i["obs_constants"], float) for i in p]) - b0
        z = np.concatenate([[(p[i]["cd_observed"][k] - isos[i]["cd_observed"][k]) / sig[i][k]
                             for k in CD_NAMES if k in sig[i]] for i in range(len(p))])
        d_rot = cov @ (J[:n].T @ (scale * e))
        with_cd.append(Bq @ (d_rot + cov @ (J[n:].T @ z)))
        without.append(Bq @ d_rot)
    t = np.sqrt(np.diag(Bq @ target @ Bq.T))
    ratio = np.array(with_cd).std(axis=0) / t
    short = np.array(without).std(axis=0) / t
    assert np.all(np.abs(ratio - 1.0) < 0.04), ratio
    assert np.any(short < 0.92), "premise: the distortion rows' share is material"


def test_a_config_can_supply_observed_distortion_constants(observed_cd):
    """run_generic._build_isotopologue copied obs/sigma/alpha/delta only, so
    no config could carry cd_observed."""
    from runner.run_from_config import build_optimizer_from_config
    from runner.usability import load_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    iso = cfg["isotopologues"][0]
    name = iso.get("name", "H2-16O")
    iso["cd_observed"] = dict(observed_cd.get(name, observed_cd["H2-16O"]))
    cfg.setdefault("rovibrational_corrections", {})
    cfg["rovibrational_corrections"]["fit_cd_constants"] = True
    cfg["rovibrational_corrections"]["cd_weight"] = 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
        opt.run()
    assert opt.spectral.isotopologues[0].get("cd_observed")
    J, _r = opt.spectral.stacked(opt.coords)
    assert J.shape[0] > _n_rot(opt)


# ── Round 3 ──────────────────────────────────────────────────────────────────

EXP_H2O = {"DJ": 37.59, "DJK": -172.9, "DK": 973.3, "delta_J": 15.26, "delta_K": 41.05}


def _water_cd(cd_sigma=None, cd_weight=1.0, **kw):
    isos = _isos(None)
    isos[0]["cd_observed"] = dict(EXP_H2O)
    if cd_sigma is not None:
        isos[0]["cd_sigma"] = cd_sigma
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(), isotopologues=isos,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 max_iter=80, auto_systematic_sigma=False,
                                 fit_cd_constants=True, cd_weight=cd_weight, **kw)
        opt.run()
    return opt


def test_a_given_cd_sigma_is_the_measurement_part():
    """A literature-precision sigma (1%) used alone treated the harmonic
    model as exact: pulls up to 31 on DJK, the geometry moved 0.4-0.6
    widths. The model's 40% now goes in quadrature."""
    opt = _water_cd({k: 0.01 * abs(v) for k, v in EXP_H2O.items()})
    sig = opt.spectral.cd_row_sigmas(opt.coords)[0]
    for k, v in sig.items():
        assert v == pytest.approx(np.hypot(0.01, 0.40) * abs(EXP_H2O[k]), rel=1e-12)
    keys, obs, pred, sg = opt.spectral._cd_rows(opt.spectral.isotopologues[0], opt.coords)
    assert np.all(np.abs((obs - pred) / sg) < 2.0), "pulls at the model's accuracy"


def test_a_partial_cd_sigma_block_gives_the_rest_the_model_error():
    """A key missing from a given block used to get 5% of its value, 8x
    tighter than an absent block's 40%."""
    opt = _water_cd({"DJ": 0.01 * EXP_H2O["DJ"]})
    sig = opt.spectral.cd_row_sigmas(opt.coords)[0]
    assert sig["DJK"] == pytest.approx(0.40 * abs(EXP_H2O["DJK"]), rel=1e-12)
    assert sig["DK"] == pytest.approx(0.40 * abs(EXP_H2O["DK"]), rel=1e-12)


def test_unvalidated_distortion_constants_are_named_and_not_fitted(capsys):
    from backend.spectral.spectral import SpectralEngine
    isos = _isos(None)
    isos[0]["cd_observed"] = dict(EXP_H2O)
    SpectralEngine(isos, cd_weight=1.0, fit_cd_constants=True)
    out = capsys.readouterr().out
    assert "delta_J, delta_K not fitted" in out


def test_a_supplied_fixed_prediction_is_said_to_be_unused(capsys):
    isos = _isos(None)
    isos[0]["cd_observed"] = dict(EXP_H2O)
    from backend.spectral.centrifugal_distortion import CDConstants
    isos[0]["pred_cd"] = CDConstants(DJ=38.0, DJK=-200.0, DK=900.0)
    MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(), isotopologues=isos,
                       quantum_backend="analytic_water", use_autoconfig=False, max_iter=2,
                       fit_cd_constants=True, cd_weight=1.0)
    assert "pred_cd on H2-16O is not used for fitting" in capsys.readouterr().out



def test_the_posterior_lists_the_distortion_rows_it_counts(cd_fit):
    """The rows drive the fit and its chi-square; the report could not show
    them because the posterior did not carry them."""
    post = cd_fit.internal_coordinate_posterior()
    rows = post["distortion_rows"]
    assert post["n_distortion_rows"] == len(rows) == 6
    assert post["n_rotational_rows"] + post["n_distortion_rows"] == post["n_observations"]
    r, sg, labels = cd_fit.spectral.cd_residuals_mhz(cd_fit.coords)
    np.testing.assert_allclose([x["residual_mhz"] for x in rows], r, rtol=0, atol=1e-12)
    w = np.sqrt(cd_fit.spectral.cd_weight)
    np.testing.assert_allclose([x["z"] for x in rows], r / (sg / w), rtol=1e-12)


def test_the_report_counts_the_rows_behind_its_chi_square():
    from runner.usability import generate_uncertainty_report_section
    result = {"uncertainty_rows": [{"name": "bond 1-2", "value": 0.96, "std_err": 0.006,
                                    "ci_lo": 0.948, "ci_hi": 0.972}],
              "uncertainty_meta": {"chi2_per_observation": 0.05, "n_observations": 6,
                                   "n_rotational_rows": 3, "n_distortion_rows": 3,
                                   "blocks": ["spectral data"]}}
    txt = generate_uncertainty_report_section(result)
    assert "over 6 rows (3 rotational + 3 distortion)" in txt
    result["uncertainty_meta"]["n_distortion_rows"] = 0
    assert "rotational +" not in generate_uncertainty_report_section(result)


def test_the_cli_prints_the_distortion_rows(observed_cd, tmp_path, monkeypatch):
    import yaml
    import cli
    from runner.usability import load_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    cfg["isotopologues"][0]["cd_observed"] = dict(observed_cd["H2-16O"])
    cfg.setdefault("rovibrational_corrections", {})
    cfg["rovibrational_corrections"].update({"fit_cd_constants": True, "cd_weight": 1.0})
    path = tmp_path / "water_cd.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(path), "--mode", "bootstrap", "--samples", "4",
                         "--workers", "1", "--seed", "3", "--no-bootstrap-persist",
                         "--no-laplace-persist"])
    assert code == 0, err.getvalue()[-2000:]
    assert "Centrifugal-distortion rows (fitted)" in out.getvalue()
    assert "rotational-constant residuals" in out.getvalue()
