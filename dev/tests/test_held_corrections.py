"""Hessian-built vibrational corrections are built once, at the fit level's own
minimum, and held (D-182, backlog N37).

Before, the fit's own corrections were rebuilt from every Hessian about
wherever the fit had reached, off the surface's stationary point. Measured
through quantize run (HF/6-31G, three starts): SO2's bond 2.37 -> 0.70 mA from
r_e, formaldehyde's C-H -7.49 -> -6.90 mA, water's -1.55 -> 0.46 mA, in 84-88%
less time. These tests use the analytic water surface the end-to-end config
runs on.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
import backend.spectral.harmonic_alpha as ha  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from runner.run_from_config import build_optimizer_from_config  # noqa: E402
from runner.run_generic import main as run_generic_main  # noqa: E402
from runner.usability import (  # noqa: E402
    ConfigError, load_config, prepare_run_directory, validate_config)

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


def _builds(monkeypatch):
    """Count the correction-table builds."""
    calls = []
    real = ha.build_correction_table_from_hessian

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(ha, "build_correction_table_from_hessian", counted)
    return calls


def _cfg(**rc):
    cfg = load_config(CONFIG)
    cfg["rovibrational_corrections"] = {**cfg["rovibrational_corrections"], **rc}
    return cfg


def _fit(cfg, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        opt = build_optimizer_from_config(cfg, verbose=False, **kw)
        opt.run()
    return opt, out.getvalue()


def _targets(opt):
    return np.array([np.asarray(i["obs_constants"], dtype=float)
                     for i in opt.spectral.isotopologues])


def test_the_table_is_built_once_at_the_minimum_and_held(monkeypatch):
    """One build for the run, at a point where the surface is stationary, and
    the Hessian refresh at the answer leaves it and the targets exactly as
    they were."""
    calls = _builds(monkeypatch)
    opt, _ = _fit(_cfg(), multistart=False)
    assert opt._corr_at_fit_level and opt._corr_backend is opt._backend
    targets = _targets(opt)
    with contextlib.redirect_stdout(io.StringIO()):
        info = opt.refresh_hessian_at_answer()
    assert len(calls) == 1, f"{len(calls)} builds"
    assert info["passes"] == 1 and info["settled"]
    np.testing.assert_array_equal(_targets(opt), targets)
    geom = np.asarray(opt._corr_table[0], dtype=float)
    g = np.asarray(opt._backend.run_gradient(geom).gradient_bohr, dtype=float)
    assert np.max(np.abs(g)) < 1e-4, "the table sits at the surface's stationary point"
    assert opt.held_corrections_note() is None


def test_the_opt_out_rebuilds_the_table_at_each_hessian(monkeypatch):
    """corrections_at_theory_minimum: false restores the old behaviour: no
    held table, a build per Hessian, and a refresh that moves the targets."""
    calls = _builds(monkeypatch)
    opt, _ = _fit(_cfg(corrections_at_theory_minimum=False), multistart=False)
    assert opt._corr_backend is None and not opt._corr_at_fit_level
    n = len(calls)
    targets = _targets(opt)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.refresh_hessian_at_answer()
    assert n >= 1 and len(calls) > n
    assert not np.array_equal(_targets(opt), targets)


def test_the_starts_share_one_table(monkeypatch, tmp_path):
    """quantize run builds the table once for every start, not once per
    start, and every start fits the same targets."""
    import runner.run_generic as rg
    real = rg.run_multistart
    # In-process, so the count sees every start's builds (BALANCED runs two
    # workers on a backend that supports them).
    monkeypatch.setattr(rg, "run_multistart",
                        lambda *a, **k: real(*a, **{**k, "max_workers": 1}))
    calls = _builds(monkeypatch)
    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    cfg["output"] = {"root": str(tmp_path / "runs"), "artifacts": True}
    validate_config(cfg)
    prepare_run_directory(cfg, CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_generic_main(cfg, refresh_hessian=True)
    runs = [r for r in result["results"] if r.get("spectral_isotopologues_snapshot")]
    assert len(runs) >= 2, "premise: more than one start"
    assert len(calls) == 1, f"{len(calls)} builds for {len(runs)} starts"
    t0 = np.asarray(runs[0]["spectral_isotopologues_snapshot"][0]["obs_constants"], dtype=float)
    for r in runs[1:]:
        np.testing.assert_array_equal(
            np.asarray(r["spectral_isotopologues_snapshot"][0]["obs_constants"], dtype=float), t0)


def test_the_uncertainty_builder_reuses_the_table(monkeypatch):
    """build_optimizer_from_config (quantize uncertainty) builds it once for
    its refreshed seeding starts and the optimizer it returns."""
    calls = _builds(monkeypatch)
    opt, _ = _fit(_cfg(), multistart=True, refresh_hessian=True)
    assert len(calls) == 1, f"{len(calls)} builds"
    assert opt._corr_at_fit_level


def test_a_saddle_falls_back_to_the_fit_s_hessian(monkeypatch):
    """Where the stationary point reached is not a minimum, the corrections
    follow the fit's Hessian as before, and the run says so."""
    monkeypatch.setattr(MolecularOptimizer, "_stationary_point_is_saddle",
                        lambda self, geom, hess: True)
    calls = _builds(monkeypatch)
    opt, out = _fit(_cfg(), multistart=False)
    assert "not a minimum" in out
    assert opt._corr_backend is None and not opt._corr_at_fit_level
    assert len(calls) >= 1
    raw = np.array([835840.29, 435351.72, 278138.70])
    assert np.all(np.abs(_targets(opt)[0] - raw) > 100.0), "still corrected"


def test_the_saddle_check_reads_the_curvature():
    """The minimum of the analytic surface is a minimum; linear water, a
    stationary point by symmetry, has a negative bend."""
    opt, _ = _fit(_cfg(), multistart=False)
    geom = np.asarray(opt._corr_table[0], dtype=float)
    be = opt._backend
    assert not opt._stationary_point_is_saddle(geom, be.run_hessian(geom).hessian_bohr)
    r = float(np.linalg.norm(geom[1] - geom[0]))
    linear = np.array([[0.0, 0.0, 0.0], [0.0, r, 0.0], [0.0, -r, 0.0]])
    assert opt._stationary_point_is_saddle(linear, be.run_hessian(linear).hessian_bohr)


def test_an_answer_far_from_the_table_is_flagged():
    """More than HELD_CORRECTIONS_FAR_ANG between the answer and where its
    held corrections were evaluated is a warning in the posterior."""
    opt, _ = _fit(_cfg(), multistart=False)
    geom, tbl, info = opt._corr_table
    far = np.asarray(geom, dtype=float).copy()
    far[1:] *= 1.4                                   # bonds 0.38 A longer
    opt._corr_table = (far, tbl, info)
    note = opt.held_corrections_note()
    assert note is not None and "corrections_at_theory_minimum" in note
    with contextlib.redirect_stdout(io.StringIO()):
        post = opt.internal_coordinate_posterior()
    assert any("corrections were evaluated" in w for w in post["warnings"])


def test_the_key_must_be_a_boolean():
    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    cfg["rovibrational_corrections"]["corrections_at_theory_minimum"] = "no"
    with pytest.raises(ConfigError, match="corrections_at_theory_minimum"):
        validate_config(cfg)
    cfg["rovibrational_corrections"]["corrections_at_theory_minimum"] = False
    validate_config(cfg)
