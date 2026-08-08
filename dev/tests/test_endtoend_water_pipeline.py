"""End-to-end coverage of the real CLI pipeline, offline.

Everything else exercises library functions directly. This drives the same path
`python -m cli run` takes -- config validation, run directory, internal
coordinates, autoconfig, multistart, the B_0 -> B_e correction chain, reporting
-- using the analytic water backend so no Psi4 or ORCA is needed.
"""

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

import dev.analytic_water_backend  # noqa: F401  (registers "analytic_water")
from runner.run_generic import main as run_generic_main
from runner.usability import (
    ConfigError,
    prepare_run_directory,
    valid_backends,
    validate_config,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"

R_E, THETA_E = 0.95784, 104.508


def _structure(coords):
    x = np.asarray(coords, dtype=float)
    r1 = np.linalg.norm(x[1] - x[0])
    r2 = np.linalg.norm(x[2] - x[0])
    u1 = (x[1] - x[0]) / r1
    u2 = (x[2] - x[0]) / r2
    theta = np.degrees(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
    return 0.5 * (r1 + r2), theta


def _run(tmp_path, harmonic, anharmonic):
    cfg = copy.deepcopy(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    cfg["rovibrational_corrections"]["harmonic_from_hessian"] = harmonic
    cfg["rovibrational_corrections"]["anharmonic_from_hessian"] = anharmonic
    cfg["output"] = {"root": str(tmp_path / "runs"), "artifacts": True}
    validate_config(cfg)
    prepare_run_directory(cfg, _CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_generic_main(cfg)


# ── Backend registration ─────────────────────────────────────────────────────

def test_valid_backends_follows_the_registry():
    """A hardcoded list defeated @register_backend: a newly registered backend
    was rejected by config validation before the runner ever saw the name."""
    allowed = valid_backends()
    assert {"orca", "psi4", "none"} <= allowed
    assert "analytic_water" in allowed


def test_unknown_backend_is_still_rejected():
    cfg = copy.deepcopy(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    cfg["quantum"]["backend"] = "not_a_backend"
    with pytest.raises(ConfigError, match="quantum.backend"):
        validate_config(cfg)


# ── The pipeline itself ──────────────────────────────────────────────────────

def test_pipeline_runs_and_writes_its_artifacts(tmp_path):
    result = _run(tmp_path, True, True)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "exports" / "final_geometry.csv").is_file()
    assert (run_dir / "exports" / "residuals.csv").is_file()


def test_corrections_reach_the_pipeline(tmp_path):
    """The spectral targets must stop being the raw B_0 once corrections run."""
    result = _run(tmp_path, True, True)
    snap = result["best"]["spectral_isotopologues_snapshot"][0]
    targets = np.asarray(snap["obs_constants"], dtype=float)
    raw = np.array([835840.29, 435351.72, 278138.70])
    assert not np.allclose(targets, raw)


def test_full_chain_improves_the_bond_length_end_to_end(tmp_path):
    """Fitting raw B_0 conflates structure with zero-point motion; correcting to
    B_e should pull the recovered bond length toward r_e."""
    r_raw, _ = _structure(_run(tmp_path / "a", False, False)["best"]["coords"])
    r_full, _ = _structure(_run(tmp_path / "b", True, True)["best"]["coords"])
    assert abs(r_full - R_E) < 0.5 * abs(r_raw - R_E)
    assert abs(r_full - R_E) < 0.010


def test_hybrid_beats_both_theory_and_experiment_alone():
    """The point of the package: combining a biased quantum surface with
    vibrationally-averaged constants should beat either on its own.

    Bond length only. With one isotopologue the angle stays prior-dominated --
    the run reports a standard error over a degree on it -- so it is not a
    parameter either leg determines.
    """
    import scripts.theory_vs_experiment_vs_hybrid as tve

    r_theory, _ = tve.theory_alone()
    (r_expt, _), _ = tve._pipeline("none", False, False)
    (r_hybrid, _), _ = tve._pipeline("analytic_water_detuned", True, True)

    err_theory = abs(r_theory - R_E)
    err_expt = abs(r_expt - R_E)
    err_hybrid = abs(r_hybrid - R_E)

    assert err_hybrid < err_theory, (
        f"hybrid {err_hybrid * 1000:.1f} mA vs theory {err_theory * 1000:.1f} mA"
    )
    assert err_hybrid < err_expt, (
        f"hybrid {err_hybrid * 1000:.1f} mA vs experiment {err_expt * 1000:.1f} mA"
    )
    # Experiment alone returns an r_0-like structure, longer than r_e.
    assert r_expt > R_E


def test_run_reports_low_confidence_for_a_single_isotopologue(tmp_path):
    """One isotopologue cannot pin water's structure, and the run should say so
    rather than presenting the numbers as settled."""
    result = _run(tmp_path, True, True)
    score = float((result.get("score") or {}).get("score", 100.0))
    assert score < 50.0
