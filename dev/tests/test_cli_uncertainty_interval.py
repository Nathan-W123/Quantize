"""The CLI's interval describes a sample of the posterior, and only that.

Measured before (the water config through the CLI, 40 error-model draws,
seed 3): the "95% Credible Interval" table was the error-model cloud's --
5.36 mA on the bond and 1.04 deg on the angle, against the reported
posterior's 6.52 mA and 1.16 deg -- directly under a summary saying that
cloud is the posterior less its prior part (on formyl fluoride up to 13x
narrower). In --mode both the bootstrap draws were appended to the MCMC
chains. The default engine was 'legacy', which ran the whole base fit and
then exited; and rto on a spectral-only config died with a traceback after
the base fit.

After: the interval comes from the MCMC chains when they pass their checks,
from an rto bootstrap when one ran, and otherwise from draws of the reported
posterior itself (6.5 mA / 1.13 deg on the same run); clouds are never
pooled.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
import cli  # noqa: E402

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


def _run(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(CONFIG)] + list(args))
    return code, out.getvalue(), err.getvalue()


def _bundle(tmp_path):
    hits = list(tmp_path.rglob("uncertainty_run_summary.json"))
    assert len(hits) == 1, hits
    return json.loads(hits[0].read_text(encoding="utf-8"))


_QUIET = ["--workers", "1", "--seed", "3", "--no-bootstrap-persist", "--no-laplace-persist"]


def test_the_default_bootstrap_reports_the_posterior_not_its_cloud(tmp_path, monkeypatch):
    code, out, err = _run(["--mode", "bootstrap", "--samples", "6"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert b["engine"] == "v2", "v2 is the default"
    assert b["bootstrap"]["config"]["strategy"] == "error-model"
    assert "reported posterior" in b["cloud"]["source"]
    assert b["cloud"]["n_samples"] == cli._ANALYTIC_DRAWS
    a = b["bootstrap"]["analytic"]
    std = {s["name"]: s["std"] for s in b["cloud"]["coordinate_stats"]}
    for name, post in zip(a["names"], a["posterior_std"]):
        # 2000 draws: the sd's Monte Carlo error is 1.6%.
        assert std[name] == pytest.approx(post, rel=0.06)
    # The premise that makes the check bite: on the bond the cloud estimates
    # something 16% narrower than the posterior.
    assert min(t / p for t, p in zip(a["target_std"], a["posterior_std"])) < 0.9


def test_an_rto_bootstrap_is_the_interval(tmp_path, monkeypatch):
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto",
                           "--samples", "6"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "rto refits" in b["cloud"]["source"]
    assert b["cloud"]["n_samples"] == b["bootstrap"]["usable_samples"] == 6


def test_two_clouds_are_never_pooled():
    mcmc = [np.full((3, 3), 1.0)] * 5
    boot = [np.full((3, 3), 2.0)] * 3
    cloud, source = cli._interval_cloud("both", mcmc, boot, None, 0)
    assert len(cloud) == 5 and "MCMC" in source
    cloud, source = cli._interval_cloud("both", [], boot, None, 0)
    assert len(cloud) == 3 and "rto" in source
    cloud, source = cli._interval_cloud("mcmc", [], boot, None, 0)
    assert cloud == [] and source is None, "an unusable chain is not replaced"


def test_the_removed_engine_stops_before_the_fit(tmp_path, monkeypatch):
    import runner.run_from_config as R
    monkeypatch.setattr(R, "build_optimizer_from_config",
                        lambda *a, **k: pytest.fail("the optimizer was built"))
    code, out, err = _run(["--uncertainty-engine", "legacy", "--mode", "bootstrap"],
                          tmp_path, monkeypatch)
    assert code == 2 and "has been removed" in err


def test_rto_on_a_spectral_only_config_stops_before_the_fit(tmp_path, monkeypatch):
    import runner.run_from_config as R
    real = R.build_optimizer_from_config

    def build(cfg, *a, **k):
        opt = real(cfg, *a, **k)
        opt.spectral_only = True
        opt.run = lambda: pytest.fail("the base fit ran")
        return opt

    monkeypatch.setattr(R, "build_optimizer_from_config", build)
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto",
                           "--samples", "2"], tmp_path, monkeypatch)
    assert code == 2
    assert "needs a quantum prior" in err and "Traceback" not in err


def test_the_interval_table_uses_the_same_sd_as_the_bootstrap_summary():
    """ddof 1 in both: the same cloud printed 3.39 and 3.17 mA at 8 draws."""
    from backend.uncertainty.analysis import analyze_coordinate_distribution
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]])
    cloud = base[None] + 0.003 * rng.standard_normal((8, 3, 3))
    stats, _corr, _names = analyze_coordinate_distribution(cloud, ["O", "H", "H"])
    from backend.internal.internal_fit import InternalCoordinateSet
    cs = InternalCoordinateSet(cloud[0], ["O", "H", "H"])
    q = np.array([cs.active_values(c) for c in cloud])
    assert stats[0]["std"] == pytest.approx(float(np.std(q[:, 0], ddof=1)), rel=1e-12)


def test_the_predictive_check_s_residuals_are_the_full_call_s():
    """want_jacobian=False skips the finite-differenced Jacobian (3.3 s of
    the check's 3.4 over 2000 formyl fluoride draws); the residuals must be
    bit-identical to the full call's."""
    from backend.spectral.spectral import SpectralEngine
    masses = [15.9949146196, 1.00782503207, 1.00782503207]
    eng = SpectralEngine([{"name": "w", "masses": masses,
                           "obs_constants": [835840.29, 435351.72, 278138.70],
                           "sigma_constants": [0.2, 0.2, 0.2]}])
    x = np.array([[0.0, 0.0, 0.0], [0.7572, 0.5865, 0.0], [-0.7572, 0.5865, 0.0]])
    J, r = eng.stacked_unweighted(x)
    none, r_fast = eng.stacked_unweighted(x, want_jacobian=False)
    assert none is None and J.shape == (3, 9)
    np.testing.assert_array_equal(r_fast, r)
