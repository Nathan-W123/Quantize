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
    monkeypatch.setattr(cli, "_MIN_RTO_DRAWS", 6)       # 200 by default; see below
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto",
                           "--samples", "6"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "rto refits" in b["cloud"]["source"]
    assert b["cloud"]["n_samples"] == b["bootstrap"]["usable_samples"] == 6


def test_a_small_rto_cloud_is_a_check_not_the_interval(tmp_path, monkeypatch):
    """Percentiles from 20 refits are mostly the extremes: the water config
    read 0.70 / 0.79 of the posterior's widths with a bond flagged bimodal."""
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto",
                           "--samples", "6"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "reported posterior" in b["cloud"]["source"]
    assert any("fewer than the 200" in w for w in b["warnings"])


class _Fitted:
    """What _interval_cloud reads from an optimizer before it draws."""

    def __init__(self, saddle=None):
        self.coords = np.zeros((3, 3))
        self._gn_saddle = None if saddle is None else (self.coords.tobytes(), saddle)


def test_two_clouds_are_never_pooled(monkeypatch):
    monkeypatch.setattr(cli, "_MIN_RTO_DRAWS", 3)
    mcmc = [np.full((3, 3), 1.0)] * 5
    boot = [np.full((3, 3), 2.0)] * 3
    cloud, source = cli._interval_cloud("both", mcmc, boot, _Fitted(), 0)
    assert len(cloud) == 5 and "MCMC" in source
    cloud, source = cli._interval_cloud("both", [], boot, _Fitted(), 0)
    assert len(cloud) == 3 and "rto" in source
    cloud, source = cli._interval_cloud("mcmc", [], boot, _Fitted(), 0)
    assert cloud == [] and source is None, "an unusable chain is not replaced"


def test_rto_refits_at_a_saddle_are_not_an_interval(monkeypatch):
    """At a stationary saddle the refits fall into other basins (water,
    C1-forced H2O + D2O: bimodal bonds, sd 3.7x the invalid Laplace's)."""
    monkeypatch.setattr(cli, "_MIN_RTO_DRAWS", 3)
    boot = [np.full((3, 3), 2.0)] * 3
    with pytest.raises(ValueError, match="not quoted as an interval"):
        cli._interval_cloud("bootstrap", [], boot, _Fitted(saddle="a saddle"), 0)


def test_a_refused_interval_still_writes_the_bundle(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise ValueError("no reported posterior to draw from: test")
    monkeypatch.setattr(cli, "_interval_cloud", refuse)
    code, out, err = _run(["--mode", "bootstrap", "--samples", "4"] + _QUIET,
                          tmp_path, monkeypatch)
    assert code == 1
    assert "no reported posterior" in err and "no reported posterior" not in out
    b = _bundle(tmp_path)
    assert b["cloud_refused"].startswith("no reported posterior")
    assert b["bootstrap"]["usable_samples"] == 4, "the bootstrap block is kept"


def test_the_laplace_ridge_reaches_the_interval_and_the_analytic_column(tmp_path, monkeypatch):
    """--laplace-regularization used to move the printed Laplace summary
    and nothing else, so one run quoted two different 'Laplace' answers."""
    widths = {}
    for reg in ("1e-8", "1e4"):
        run = tmp_path / reg
        run.mkdir()
        code, _o, err = _run(["--mode", "bootstrap", "--samples", "4",
                              "--laplace-regularization", reg] + _QUIET, run, monkeypatch)
        assert code == 0, err[-2000:]
        b = _bundle(run)
        widths[reg] = (b["bootstrap"]["analytic"]["posterior_std"],
                       {s["name"]: s["std"] for s in b["cloud"]["coordinate_stats"]})
    post_lo, cloud_lo = widths["1e-8"]
    post_hi, cloud_hi = widths["1e4"]
    assert post_hi[0] < 0.9 * post_lo[0], "premise: the ridge narrows the Laplace"
    assert cloud_hi["bond 1-2"] == pytest.approx(post_hi[0], rel=0.06)


def test_the_summary_says_when_there_is_no_reported_posterior():
    from backend.uncertainty_v2.models import BootstrapConfig, BootstrapRunSummary
    from backend.uncertainty_v2.reporting import format_bootstrap_summary
    s = BootstrapRunSummary(config=BootstrapConfig(n_samples=1), total_samples=0,
                            successful_samples=0, failed_samples=0, mean_rms_mhz=None)
    s.analytic = {"target": "posterior (reported covariance)", "names": ["bond 1-2"],
                  "kinds": ["bond"], "units": ["angstrom"], "target_std": [0.01],
                  "posterior_std": [0.01], "bootstrap_std": None,
                  "no_posterior": "no Laplace width exists at this geometry -- saddle"}
    txt = format_bootstrap_summary(s)
    assert "NO REPORTED POSTERIOR" in txt
    assert "The reported uncertainty is the posterior column" not in txt


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
    b = _bundle(tmp_path)
    assert b["exit_status"] == 2 and "needs a quantum prior" in b["bootstrap_refused"]


def test_a_refusal_before_the_fit_replaces_the_last_run_s_bundle(tmp_path, monkeypatch):
    """A pre-fit refusal wrote no bundle, so the previous run's stayed in
    place, exit_status 0 and all."""
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0 and _bundle(tmp_path)["exit_status"] == 0
    import runner.run_from_config as R
    real = R.build_optimizer_from_config

    def build(cfg, *a, **k):
        opt = real(cfg, *a, **k)
        opt.run = lambda: pytest.fail("the base fit ran")
        return opt

    monkeypatch.setattr(R, "build_optimizer_from_config", build)
    import backend.uncertainty_v2.bootstrap as BS
    monkeypatch.setattr(BS.BootstrapEngine, "strategy_problem",
                        staticmethod(lambda opt, strategy, fitted=True: "refused before the fit"))
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 2
    b = _bundle(tmp_path)
    assert b["exit_status"] == 2 and b["bootstrap_refused"] == "refused before the fit"


def test_refused_chains_in_mode_both_are_a_partial_result(tmp_path, monkeypatch):
    """--mode both with the chains refused fell back to the reported
    posterior and exited 0, as if both halves had run."""
    code, out, err = _run(["--mode", "both", "--samples", "4", "--mcmc-steps", "60",
                           "--burn-in", "10", "--chains", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 3
    b = _bundle(tmp_path)
    assert b["partial"] is True and b["exit_status"] == 3
    assert [r for r in b["partial_reasons"] if r.startswith("the MCMC chains were refused")]
    assert not any(w.startswith("partial result") for w in b["warnings"]), "recorded once"
    assert "reported posterior" in b["cloud"]["source"]


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


def test_refused_mcmc_chains_go_to_stderr_and_keep_the_bundle(tmp_path, monkeypatch):
    """--mode mcmc with unusable chains printed its refusal on stdout and
    wrote no bundle, so the MCMC block (R-hat, ESS) was lost."""
    code, out, err = _run(["--mode", "mcmc", "--seed", "7", "--mcmc-steps", "60",
                           "--burn-in", "10", "--chains", "2"], tmp_path, monkeypatch)
    assert code == 1
    assert "every MCMC sample was refused" in err and "every MCMC sample was refused" not in out
    assert "R-hat" in err or "ESS" in err, "the reason, not only the verdict, is on stderr"
    b = _bundle(tmp_path)
    assert b["cloud_refused"].startswith("every MCMC sample was refused")
    assert "mcmc" in b


def test_a_sampler_interval_says_it_does_not_carry_the_ridge(tmp_path, monkeypatch):
    """With a ridge set, an rto (or MCMC) interval is not the posterior
    column the summary calls the reported uncertainty; say so."""
    monkeypatch.setattr(cli, "_MIN_RTO_DRAWS", 6)
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto", "--samples", "6",
                           "--laplace-regularization", "1e4"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "rto refits" in b["cloud"]["source"]
    assert any("do not carry it" in w for w in b["warnings"])


def test_mode_both_still_runs_the_chains_when_the_bootstrap_is_refused(tmp_path, monkeypatch):
    """A strategy refused after the fit (a split fit its posterior does not
    describe, say) returned before the MCMC block, so --mode both lost its
    chains too. The refusal is about the bootstrap's check column; the chains
    sample the posterior and still run, and the exit status still says part
    of the request was refused."""
    from backend.uncertainty_v2 import BootstrapEngine

    def refuse(self):
        raise ValueError("refused for this test")

    monkeypatch.setattr(BootstrapEngine, "_check_strategy", refuse)
    monkeypatch.setattr(BootstrapEngine, "run", lambda *a, **k: pytest.fail("the bootstrap ran"))
    code, out, err = _run(["--mode", "both", "--samples", "2", "--mcmc-steps", "60",
                           "--burn-in", "10", "--chains", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 3, "partial: an interval exists, part of the request was refused"
    assert "refused for this test" in err
    b = _bundle(tmp_path)
    assert b["bootstrap_refused"] == "refused for this test"
    assert "mcmc" in b and "bootstrap" not in b
    assert b["partial"] is True and b["exit_status"] == 3
    assert any("refused after the base fit" in w for w in b["partial_reasons"])


def test_the_ridge_note_needs_a_ridge_that_moves_the_widths(tmp_path, monkeypatch):
    """The note fired on any ridge but the default 1e-8, 0 included, which
    moves no width (about 1e-8 relative on water)."""
    monkeypatch.setattr(cli, "_MIN_RTO_DRAWS", 6)
    code, out, err = _run(["--mode", "bootstrap", "--bootstrap-strategy", "rto", "--samples", "6",
                           "--laplace-regularization", "0"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "rto refits" in b["cloud"]["source"]
    assert not any("do not carry it" in w for w in b["warnings"])


def test_an_interval_from_a_doubtful_split_posterior_says_so(tmp_path, monkeypatch):
    """After a split refusal --mode both quoted the chains' 95% interval with
    no caveat, though the Laplace summary above it said those chains need
    not describe the split fit. The caveat now sits at the interval, in the
    bundle's warnings and on its cloud block."""
    from backend.quantize import MolecularOptimizer
    monkeypatch.setattr(MolecularOptimizer, "split_inconsistency",
                        lambda self: "the split step couples the data's range and null space")
    code, out, err = _run(["--mode", "bootstrap", "--samples", "4"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0, err[-2000:]
    b = _bundle(tmp_path)
    assert "reported posterior" in b["cloud"]["source"]
    assert "need not be the split fit's spread" in b["cloud"]["caveat"]
    assert "this run's bootstrap of the fit found no departure" in b["cloud"]["caveat"]
    assert any("need not be the split fit's spread" in w for w in b["warnings"])
    assert "[note] split objective" in out


def test_one_chain_is_refused_for_its_count_not_as_unconverged(tmp_path, monkeypatch):
    """With --chains 1 there is no R-hat at all; the line said every sample
    was 'refused as unconverged' and advised raising --mcmc-steps."""
    code, out, err = _run(["--mode", "mcmc", "--seed", "7", "--mcmc-steps", "200",
                           "--burn-in", "50", "--chains", "1"], tmp_path, monkeypatch)
    assert code == 1
    assert "no MCMC convergence diagnostics could be computed" in err
    assert "unconverged" not in err and "--chains 2" in err
    assert _bundle(tmp_path)["exit_status"] == 1


def test_the_autocorrelation_plot_is_quiet_on_a_short_stuck_chain(tmp_path):
    """corrcoef of lagged segments divided by a zero segment variance: the
    plot printed numpy RuntimeWarnings on stderr for short refused chains."""
    import warnings
    pytest.importorskip("matplotlib")
    from scripts.uncertainty_plots import plot_autocorrelations
    base = np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]])
    chain = np.repeat(base[None], 12, axis=0)
    chain[6:, 1, 0] += 1e-3                       # two flat runs: rejected proposals
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_autocorrelations(chain, ["O", "H", "H"], str(tmp_path))
    assert (tmp_path / "autocorrelations.png").is_file()



def test_the_uncertainty_command_recomputes_the_hessian_at_the_answer(tmp_path, monkeypatch):
    """Every width the command reports reads the quantum Hessian, which the
    fit computes on its own schedule. The command recomputes it at the
    answer first; --no-hessian-refresh skips that and the bootstrap summary
    says how far away the fit's Hessian was."""
    skip_dir, run_dir = tmp_path / "skip", tmp_path / "refresh"
    skip_dir.mkdir()
    run_dir.mkdir()
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2", "--no-hessian-refresh"] + _QUIET,
                          skip_dir, monkeypatch)
    assert code == 0, err[-2000:]
    b0 = _bundle(skip_dir)
    assert b0["hessian_refresh"] == "skipped (--no-hessian-refresh)"
    age = 1e3 * b0["bootstrap"]["analytic"]["hessian_distance_ang"]
    assert age > 0.1, "premise: the fit's Hessian is not at its answer"
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, run_dir, monkeypatch)
    assert code == 0, err[-2000:]
    b1 = _bundle(run_dir)
    assert b1["hessian_refresh"]["age_before_ma"] == pytest.approx(age, rel=1e-6)
    assert b1["hessian_refresh"]["passes"] >= 1
    assert 1e3 * b1["bootstrap"]["analytic"]["hessian_distance_ang"] < 0.1 * age
    assert "quantum Hessian recomputed at the fit's final geometry" in out
    assert b1["hessian_refresh"]["settled"] is True



@pytest.mark.parametrize("args,key", [(["--samples", "0"], "usage_error"),
                                      (["--mode", "mcmc", "--burn-in", "5", "--mcmc-steps", "5"],
                                       "usage_error")])
def test_a_usage_error_replaces_the_last_run_s_bundle(args, key, tmp_path, monkeypatch):
    """The stale bundle was removed only after the optimizer was built, so a
    usage error left the previous run's (exit_status 0) in place."""
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0 and _bundle(tmp_path)["exit_status"] == 0
    code, out, err = _run(args, tmp_path, monkeypatch)
    assert code == 2
    b = _bundle(tmp_path)
    assert b["exit_status"] == 2 and key in b


def test_a_config_error_is_a_config_error(tmp_path, monkeypatch):
    """quantize uncertainty printed a raw traceback for a bad cd_model_sigma
    (and left the last bundle); quantize run printed only 'Run failed'."""
    import yaml
    from runner.usability import load_config
    cfg = load_config(CONFIG)
    cfg["isotopologues"][0]["cd_observed"] = {"DJ": 37.59}
    cfg["isotopologues"][0]["cd_model_sigma"] = {"DJ": "abc"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(path), "--mode", "bootstrap", "--samples", "2"])
    assert code == 2 and "Traceback" not in err.getvalue()
    assert "cd_model_sigma.DJ" in err.getvalue()
    assert _bundle(tmp_path)["config_error"].startswith("config:")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["run", str(path)])
    assert code == 2 and "cd_model_sigma.DJ" in err.getvalue()


def test_the_config_workdir_is_the_optimizer_s():
    from runner.run_from_config import build_optimizer_from_config
    from runner.run_generic import config_workdir
    from runner.usability import load_config
    cfg = load_config(CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
    assert config_workdir(cfg) == opt.workdir


def test_a_failure_inside_a_fit_is_not_a_config_error(tmp_path, monkeypatch):
    """The build runs the config's multistart, so the `except ValueError`
    around it reported numpy's LinAlgError (a ValueError) from a start's fit
    as a config error with exit 2. Only the translation's errors are."""
    import backend.multistart as M

    def broken(*a, **k):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(M, "run_multistart", broken)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(np.linalg.LinAlgError):
            cli.main(["uncertainty", str(CONFIG), "--mode", "bootstrap", "--samples", "2"])
    assert not list(tmp_path.rglob("uncertainty_run_summary.json")), "no config_error bundle"


def test_the_removed_engine_replaces_the_last_run_s_bundle(tmp_path, monkeypatch):
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0 and _bundle(tmp_path)["exit_status"] == 0
    code, out, err = _run(["--uncertainty-engine", "legacy"], tmp_path, monkeypatch)
    b = _bundle(tmp_path)
    assert code == 2 and b["exit_status"] == 2 and "has been removed" in b["usage_error"]


@pytest.mark.parametrize("edit,said", [
    (lambda c: c.__setitem__("elements", ["O", "H", "Xx"]), "elements[2]"),
    (lambda c: c.__setitem__("rng_seed", "x"), "rng_seed"),
    (lambda c: c.__setitem__("symmetry", "Q9"), "symmetry"),
    (lambda c: c.__setitem__("coordinate_mode", "polar"), "coordinate_mode"),
    (lambda c: c.__setitem__("internal_coordinates", {"damping": "x"}), "internal_coordinates.damping"),
    (lambda c: c.setdefault("rovibrational_corrections", {}).__setitem__(
        "harmonic_sigma_fraction", [1]), "harmonic_sigma_fraction"),
    (lambda c: c["isotopologues"][0].__setitem__("cd_observed", {"DJ": [1]}), "cd_observed.DJ"),
])
def test_a_bad_field_is_named_by_validation(edit, said, tmp_path):
    """These passed `quantize validate` and then failed in the builder with
    Python's own text ("'Xx' is not in list", "could not convert string to
    float: 'x'") or a traceback (TypeError)."""
    import yaml
    from runner.usability import ConfigError, load_config, validate_config
    cfg = load_config(CONFIG)
    edit(cfg)
    with pytest.raises(ConfigError, match=said.replace(".", r"\.").replace("[", r"\[")):
        validate_config(cfg)


def test_a_config_that_cannot_be_read_or_normalized_is_a_config_error(tmp_path):
    """A non-UTF-8 file raised UnicodeDecodeError, and a block of the wrong
    shape AttributeError inside the normalizer: tracebacks, and the last
    run's bundle left in place. `quantum: null` reads as absent, as the
    validator already read it, instead of failing the build."""
    from runner.usability import ConfigError, load_config, validate_config
    bad = tmp_path / "latin.yaml"
    bad.write_bytes(b"name: \xe9au\n")
    with pytest.raises(ConfigError, match="Could not read"):
        load_config(bad)
    shape = tmp_path / "shape.yaml"
    shape.write_text(CONFIG.read_text(encoding="utf-8") + "\ninternal_coordinates: abc\n",
                     encoding="utf-8")
    with pytest.raises(ConfigError):
        validate_config(load_config(shape))
    from runner.run_generic import build_run_inputs
    cfg = load_config(CONFIG)
    cfg["quantum"] = None
    validate_config(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        inputs = build_run_inputs(cfg, verbose=False)
    assert inputs["optimizer_kwargs"]["quantum_backend"] == "orca", "the default, as when absent"


def test_a_fit_that_stops_at_max_iter_is_warned_about(tmp_path, monkeypatch):
    import yaml
    from runner.usability import load_config
    cfg = load_config(CONFIG)
    cfg.setdefault("optimizer", {})["max_iter"] = 2
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(path), "--mode", "bootstrap", "--samples", "2"] + _QUIET)
    b = _bundle(tmp_path)
    assert b["hessian_refresh"]["termination"] == "max_iter"
    assert any("stopped at max_iter" in w for w in b["warnings"])
    assert "recomputed at the fit's final geometry" in out.getvalue()


def test_the_summaries_are_strict_json(tmp_path):
    """A constant coordinate's kurtosis (and a ratio against a zero analytic
    width) is NaN, and json.dump wrote it as a bare NaN, which a strict JSON
    parser rejects. Every uncertainty file now writes null for it."""
    from backend.uncertainty_v2.bootstrap import BootstrapEngine
    from backend.uncertainty_v2.models import BootstrapConfig, BootstrapRunSummary, json_ready

    def strict(path):
        def no_constant(name):
            raise ValueError(f"non-JSON constant {name}")
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=no_constant)

    assert json_ready({"k": [1.0, float("nan"), np.float64("inf")], "a": np.array([np.nan])}) == \
        {"k": [1.0, None, None], "a": [None]}
    s = BootstrapRunSummary(config=BootstrapConfig(n_samples=4), total_samples=4,
                            successful_samples=4, failed_samples=0, mean_rms_mhz=None)
    s.analytic = {"check": {"kurtosis": [float("nan"), 3.1], "ratio": [np.float64("nan"), 1.0]}}
    BootstrapEngine._persist_summary(s, str(tmp_path))
    got = strict(tmp_path / "bootstrap_summary.json")
    assert got["analytic"]["check"]["kurtosis"] == [None, 3.1]
    out = cli._write_bundle(str(tmp_path), {"bootstrap": {"analytic": s.analytic}, "x": float("nan")})
    got = strict(out)
    assert got["x"] is None and got["bootstrap"]["analytic"]["check"]["ratio"] == [None, 1.0]


@pytest.mark.parametrize("edit,said", [
    (lambda c: c.setdefault("optimizer", {}).__setitem__("trust_radius", "x"), "optimizer.trust_radius"),
    (lambda c: c.setdefault("optimizer", {}).__setitem__("max_iter", "abc"), "optimizer.max_iter"),
    (lambda c: c.setdefault("optimizer", {}).__setitem__("max_iter", None), "optimizer.max_iter"),
    (lambda c: c.setdefault("optimizer", {}).__setitem__("objective_mode", "bogus"), "objective_mode"),
    (lambda c: c.__setitem__("rng_seed", -1), "rng_seed"),
    (lambda c: c.setdefault("rovibrational_corrections", {}).__setitem__("cd_weight", None), "cd_weight"),
    (lambda c: c.__setitem__("rovib_recalc_every", "x"), "rovib_recalc_every"),
])
def test_values_the_constructor_would_refuse_are_named_by_validation(edit, said):
    """Round 9: with the build catching only the translation's errors, a bad
    optimizer: value (passed to the constructor unconverted) or
    coordinate_mode: null failed inside the multistart as a traceback; an
    unknown objective_mode ran as joint; a stated null for cd_weight was
    float(None); rng_seed -1 failed in numpy."""
    from runner.usability import ConfigError, load_config, validate_config
    cfg = load_config(CONFIG)
    edit(cfg)
    with pytest.raises(ConfigError, match=said):
        validate_config(cfg)


def test_what_the_builder_reads_validates_and_is_read_as_meant():
    """Quoted integers ran before the integer checks (rng_seed "17") and must
    still; an empty cd_observed list is absent; coordinate_mode null is the
    default; optimizer overrides arrive as numbers; point-group names in any
    case."""
    from backend.symmetry import _normalize_pg
    from runner.run_generic import build_run_inputs
    from runner.usability import load_config, validate_config
    cfg = load_config(CONFIG)
    cfg["rng_seed"] = "17"
    cfg["internal_coordinates"] = {"microiterations": "20"}
    cfg.setdefault("rovibrational_corrections", {})["rovib_recalc_every"] = "2"
    cfg["isotopologues"][0]["cd_observed"] = []
    cfg["coordinate_mode"] = None
    cfg["optimizer"] = {"trust_radius": "0.05", "max_iter": "40", "objective_mode": "Joint"}
    validate_config(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        inputs = build_run_inputs(cfg, verbose=False)
    kw = inputs["optimizer_kwargs"]
    assert kw["coordinate_mode"] == "internal"
    assert kw["trust_radius"] == 0.05 and kw["max_iter"] == 40 and kw["objective_mode"] == "joint"
    assert [_normalize_pg(p) for p in ("cs", "c2v", "D2H", "c1")] == ["Cs", "C2v", "D2h", "C1"]


def test_quantize_run_reports_a_translation_error_as_a_config_error(tmp_path, monkeypatch):
    import runner.run_generic as RG

    def bad(*a, **k):
        raise TypeError("float() argument must be a string or a real number, not 'NoneType'")

    monkeypatch.setattr(RG, "build_run_inputs", bad)
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["run", str(CONFIG), "--no-run-dir"])
    assert code == 2 and "Config error" in err.getvalue() and "Traceback" not in err.getvalue()


def test_a_config_that_fails_normalization_replaces_the_last_bundle(tmp_path, monkeypatch):
    """It parses, so its name -- and its workdir -- are known; the last run's
    exit-0 bundle stayed in place."""
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    assert code == 0 and _bundle(tmp_path)["exit_status"] == 0
    bad = tmp_path / "shape.yaml"
    bad.write_text(CONFIG.read_text(encoding="utf-8") + "\ninternal_coordinates: abc\n",
                   encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(bad), "--mode", "bootstrap", "--samples", "2"])
    b = _bundle(tmp_path)
    assert code == 2 and b["exit_status"] == 2 and b["config_error"].startswith("config:")


def test_an_unsettled_refresh_is_warned_about_on_stderr(tmp_path, monkeypatch):
    from backend.quantize import MolecularOptimizer
    real = MolecularOptimizer.refresh_hessian_at_answer
    monkeypatch.setattr(MolecularOptimizer, "refresh_hessian_at_answer",
                        lambda self, max_passes=3: real(self, max_passes=1))
    code, out, err = _run(["--mode", "bootstrap", "--samples", "2"] + _QUIET, tmp_path, monkeypatch)
    b = _bundle(tmp_path)
    assert b["hessian_refresh"]["settled"] is False, "premise: one pass leaves it moving"
    assert "still moving" in err, "it went to the bundle and stdout only"
    assert any("still moving" in w for w in b["warnings"])
