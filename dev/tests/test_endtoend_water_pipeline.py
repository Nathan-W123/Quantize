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
    B_e should pull the recovered bond length toward r_e.

    Measured: 2.99 mA raw, 1.65 mA with the full chain (0.55x). The ratio was
    0.34x while the rule library silently replaced this config's stated prior
    widths (0.12 A / 5 deg) with its own 0.015 / 2; honouring them costs the
    bond 0.6 mA and buys the angle 0.95 deg (+1.83 -> +0.88 deg).
    """
    r_raw, _ = _structure(_run(tmp_path / "a", False, False)["best"]["coords"])
    r_full, _ = _structure(_run(tmp_path / "b", True, True)["best"]["coords"])
    assert abs(r_full - R_E) < 0.65 * abs(r_raw - R_E)
    assert abs(r_full - R_E) < 0.010


def test_hybrid_beats_both_theory_and_experiment_alone():
    """The point of the package: combining a biased quantum surface with
    vibrationally-averaged constants should beat either on its own.

    With the joint objective -- the default -- the hybrid beats both on the
    pair taken together, each error as a fraction of theory's own (measured:
    hybrid 0.74, experiment 1.23, theory 2.0), and beats theory on each
    parameter. With one isotopologue the bond and angle pull against each
    other (see the test below): the joint fit buys the angle (0.34 deg
    against experiment's 0.77) with some of the bond (3.5 mA against 3.0).
    Handing the bond to the data outright is what the split objective does,
    and there the bond alone beats both (0.4 mA).

    This used to assert the bond-only win against the config default, which
    was split until config runs moved to the joint objective.
    """
    import scripts.theory_vs_experiment_vs_hybrid as tve

    r_theory, th_theory = tve.theory_alone()

    def frac(r, th):        # 2.0 is theory alone, by construction
        return (abs(r - R_E) / abs(r_theory - R_E)
                + abs(th - THETA_E) / abs(th_theory - THETA_E))

    (r_expt, th_expt), _ = tve._pipeline("none", False, False)
    (r_joint, th_joint), _ = tve._pipeline("analytic_water_detuned", True, True)
    (r_split, _th_split), _ = tve._pipeline("analytic_water_detuned", True, True,
                                           objective_mode="split")

    assert frac(r_joint, th_joint) < frac(r_expt, th_expt) < frac(r_theory, th_theory)
    assert abs(r_joint - R_E) < abs(r_theory - R_E)
    assert abs(th_joint - THETA_E) < abs(th_theory - THETA_E)
    assert abs(th_joint - THETA_E) < abs(th_expt - THETA_E)

    err_split = abs(r_split - R_E)
    assert err_split < abs(r_theory - R_E), (
        f"split hybrid {err_split * 1000:.1f} mA vs theory {abs(r_theory - R_E) * 1000:.1f} mA")
    assert err_split < abs(r_expt - R_E), (
        f"split hybrid {err_split * 1000:.1f} mA vs experiment {abs(r_expt - R_E) * 1000:.1f} mA")
    # Experiment alone returns an r_0-like structure, longer than r_e.
    assert r_expt > R_E


def test_calibrated_quantum_prior_improves_the_weakly_determined_angle():
    """The split objective gives the data unconditional authority over every
    direction above a *relative* rank cutoff, including ones it barely resolves.
    Water's bond angle is such a direction, and it comes out worse than either
    theory or experiment alone. The joint objective with a calibrated prior
    leaves the direction contested and recovers it.

    The bond length is a trade, not a win, and it is asserted as such below.
    Fixing the rigid-mode projector (see test_rigid_mode_projector.py) sharpened
    the angle from 0.43 to 0.02 degrees and cost the bond length, which moved
    from 0.7 mA short of r_e to 6.6 mA long. On this configuration -- one
    isotopologue on a deliberately detuned surface -- the angle is
    prior-dominated and the two parameters pull against each other. With real
    multi-isotopologue data the same fixes improve the bond length too: water's
    parent-only RHF fit goes from 3.29 to 1.75 mA RMS.

    WHY SIGMA IS 0.010 HERE (IT WAS 0.005, THEN 0.003). sigma_x is the
    displacement over which theory is trusted, and alpha_q = 1 / (1/2
    lambda_bar sigma_x^2) turns it into a weight through a typical curvature.
    The earlier tunings were made with a second prior in the objective: this
    config states internal-prior widths but no weight, and config runs used
    to default that weight to 1, so the rule-library prior (0.12 A / 5 deg)
    pulled alongside the quantum one and sigma_x had to be tightened to
    0.003 to compensate. With the library prior off while the quantum prior
    is on (D-09, now on the config path too), swept again (2.0 = theory
    alone):

        sigma_x   bond (mA)   angle (deg)   combined
        0.0030    +10.32      -0.654        1.756
        0.0050     +9.51      -0.465        1.448
        0.0075     +8.24      -0.237        1.049
        0.0100     +6.91      -0.049        0.695
        0.0200     +3.46      +0.336        0.733   (the level-table default)
        split      +0.44      +0.615        0.801

    The best width now sits at the detuned surface's own error (theory alone
    is 10.9 mA long), which is what sigma_x is meant to be. The trade is
    still monotone -- a tighter prior buys the angle with the bond -- and the
    combined minimum is near where the angle error crosses zero, so the
    assertions are inequalities with margin rather than pins to that edge.
    """
    import scripts.tune_quantum_prior as tune

    r_split, th_split = tune.run(objective_mode="split")
    r_joint, th_joint = tune.run(objective_mode="joint", quantum_prior_sigma_ang=0.010)

    # The weakly determined direction, which is what the calibrated prior is for
    # (0.049 against the split objective's 0.615 deg).
    assert abs(th_joint - THETA_E) < abs(th_split - THETA_E)

    # The bond beats theory alone (6.91 against 10.9 mA).
    assert abs(r_joint - R_E) * 1000 < tune.THEORY_DR

    # So does the angle, outright: 0.06x theory's own error. The ceiling keeps
    # it from drifting back toward the split objective's 0.76x.
    angle_vs_theory = abs(th_joint - THETA_E) / abs(tune.THEORY_DTHETA)
    assert angle_vs_theory < 0.5, (
        f"the angle is now {angle_vs_theory:.3f}x theory's own error, past the "
        f"0.5 ceiling; it measured 0.061x"
    )

    # And the pair together beats theory alone and the split objective. 2.0 is
    # theory; this measured 0.695 against split's 0.801.
    combined = tune._combined(r_joint, th_joint)
    assert combined < min(1.0, tune._combined(r_split, th_split)), (
        f"combined error {combined:.3f} against 0.695 when measured, "
        f"{tune._combined(r_split, th_split):.3f} for split and 2.0 for theory alone"
    )

    # And the trade is recorded rather than hidden: against the split objective
    # the calibrated prior buys the angle at the bond length's expense here.
    assert abs(r_joint - R_E) > abs(r_split - R_E), (
        "the bond-length trade has reversed; if the prior now wins on both, this "
        "test and the docstring above are out of date"
    )


def test_absolute_singular_value_floor_returns_directions_to_theory():
    """sv_min_abs is the blunt alternative: it drops a weakly resolved direction
    outright rather than leaving it contested.

    Both parameters end up on theory's side of the reference, which is what
    "returns directions to theory" means and is the behaviour worth pinning.
    What it does *not* do is land them there accurately -- it overshoots the
    angle past theory's own error -- so it is not a substitute for the
    calibrated prior, which is the point of the comparison in the last
    assertion.

    This test previously asserted the floor recovered the angle to within a
    quarter of the split objective's error. That held only because the
    rigid-mode projector was removing the wrong subspace: it built its
    translation and rotation modes with sqrt(m) weighting, which are the null
    vectors of a *mass-weighted* Hessian, and applied them to the plain
    Cartesian Hessian whose null vectors are unweighted. Measured against the
    analytic water Hessian, the unweighted translations give |H v| ~ 1e-10
    while the sqrt(m)-weighted ones give ~0.5 -- so the projector was deleting
    real, energy-changing directions and leaving the rigid contamination it was
    meant to remove. With that fixed the angle is no longer rescued by the
    floor, and the numbers here record the corrected behaviour.
    """
    import scripts.tune_quantum_prior as tune

    # A split-objective mechanism: under joint the floor only moves the
    # range/null bookkeeping, not the step.
    r_floor, th_floor = tune.run(objective_mode="split", sv_min_abs=1e5)
    r_joint, th_joint = tune.run(objective_mode="joint",
                                 quantum_prior_sigma_ang=0.010)

    # Theory alone errs long on the bond and small on the angle; the floor hands
    # both directions back to it, so both errors must take theory's sign.
    assert np.sign(r_floor - R_E) == np.sign(tune.THEORY_DR)
    assert np.sign(th_floor - THETA_E) == np.sign(tune.THEORY_DTHETA)

    # But handing a direction back wholesale is worse than leaving it contested:
    # the calibrated prior keeps the data's influence in proportion and lands
    # the angle closer (0.049 against 0.795 deg; sigma_x as in the test above).
    assert abs(th_joint - THETA_E) < abs(th_floor - THETA_E), (
        f"calibrated prior {abs(th_joint - THETA_E):.3f} deg should beat the "
        f"blunt floor {abs(th_floor - THETA_E):.3f} deg"
    )


def test_run_reports_low_confidence_for_a_single_isotopologue(tmp_path):
    """One isotopologue cannot pin water's structure, and the run should say so
    rather than presenting the numbers as settled.

    "Says so" is the report's own verdict band: below 60 it prints "geometry
    regularized by quantum prior; low spectral confidence". Measured 52.0
    (43.0 before the config's stated prior widths were honoured).
    """
    result = _run(tmp_path, True, True)
    score = float((result.get("score") or {}).get("score", 100.0))
    assert score < 60.0
