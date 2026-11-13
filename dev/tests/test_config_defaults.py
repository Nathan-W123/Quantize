"""Config runs fit the objective the engine and the benchmark fit.

run_generic.build_run_inputs shipped the legacy solver set-up: the split
objective, quantum_prior_sigma_ang=None and alpha_quantum 0.278 -- a quantum
weight 1e-4 to 1e-6 of the data's curvature, so the prior was inert and
switching objective_mode alone moved geometries by at most 0.0064 mA -- no
symmetry unless the config named one, and the rule-library internal prior on
at weight 1. MolecularOptimizer and the benchmark use joint, a width
calibrated to the level of theory, alpha_quantum 1 and the detected point
group, and (since D-09) no library prior while the quantum prior is on.

Now a config gets the same unless it says otherwise, and what it says wins.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from runner.run_generic import build_run_inputs  # noqa: E402
from runner.usability import load_config  # noqa: E402

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"
LEGACY_ALPHA = 0.2778639378704326


@pytest.fixture(scope="module")
def base():
    cfg = load_config(CONFIG)       # the path the CLI takes
    # The defaults are what is under test: strip what this config states.
    cfg.pop("optimizer", None)
    cfg.pop("symmetry", None)
    return cfg


def _kw(cfg, **edits):
    cfg = copy.deepcopy(cfg)
    for key, value in edits.items():
        cfg[key] = value
    return build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]


def _built(cfg, **edits):
    from runner.run_from_config import build_optimizer_from_config
    cfg = copy.deepcopy(cfg)
    for key, value in edits.items():
        cfg[key] = value
    return build_optimizer_from_config(cfg, multistart=False, verbose=False)


def test_a_config_gets_the_joint_objective_with_a_calibrated_width(base):
    kw = _kw(base)
    assert kw["objective_mode"] == "joint"
    assert kw["alpha_quantum"] == 1.0
    # Absent rather than None: None is the legacy weight, absence the
    # level-table calibration.
    assert "quantum_prior_sigma_ang" not in kw
    import backend.quantize as Q
    from backend.spectral.prior_widths import level_for_backend, prior_sigma_for_level
    opt = _built(base)
    lvl = level_for_backend(opt.quantum_backend, opt.orca_method, opt.orca_basis)
    expect, _cal = prior_sigma_for_level(*lvl, default=Q.DEFAULT_QUANTUM_PRIOR_SIGMA_ANG)
    assert opt.optimizer.quantum_prior_sigma_ang == pytest.approx(expect)
    assert opt.optimizer.objective_mode == "joint"


def test_a_stated_width_is_used_as_stated(base):
    opt = _built(base, optimizer={"quantum_prior_sigma_ang": 0.007})
    assert opt.optimizer.quantum_prior_sigma_ang == pytest.approx(0.007)
    assert opt.optimizer.alpha_quantum == pytest.approx(1.0)


def test_an_explicit_split_keeps_the_legacy_weight(base):
    """The calibrated alpha (~1e3) would otherwise multiply split's
    null-space Newton step."""
    kw = _kw(base, optimizer={"objective_mode": "split"})
    assert kw["objective_mode"] == "split"
    assert kw["quantum_prior_sigma_ang"] is None
    assert kw["alpha_quantum"] == pytest.approx(LEGACY_ALPHA)
    opt = _built(base, optimizer={"objective_mode": "split"})
    assert opt.optimizer.quantum_prior_sigma_ang is None


def test_split_with_a_stated_width_is_calibrated(base):
    kw = _kw(base, optimizer={"objective_mode": "split", "quantum_prior_sigma_ang": 0.01})
    assert kw["quantum_prior_sigma_ang"] == pytest.approx(0.01)
    assert kw["alpha_quantum"] == 1.0


def test_no_symmetry_key_means_the_detected_group(base):
    assert _kw(base)["symmetry"] == "auto"
    opt = _built(base)
    assert opt.symmetry is not None and opt.symmetry.point_group == "C2v"


def test_an_explicit_null_symmetry_is_unconstrained(base):
    kw = _kw(base, symmetry=None)
    assert kw["symmetry"] is None
    assert _built(base, symmetry=None).symmetry is None


def test_a_named_group_is_used(base):
    assert _kw(base, symmetry="C2v")["symmetry"] == "C2v"


def test_the_rule_library_prior_is_off_unless_the_config_states_a_weight(base):
    kw = _kw(base)
    assert kw["ic_prior_weight"] is None
    assert _built(base)._ic_prior_weight == 0.0          # the quantum prior is on
    stated = copy.deepcopy(base)
    stated.setdefault("internal_coordinates", {})["prior_weight"] = 0.5
    assert _kw(stated)["ic_prior_weight"] == pytest.approx(0.5)
    assert _built(stated)._ic_prior_weight == pytest.approx(0.5)


@pytest.mark.parametrize("mode,expect", [("joint", False), ("split", True)])
def test_enforce_quantum_descent_is_a_split_gate(base, mode, expect):
    """STRICT asks for it; the joint step has its own ratio test."""
    kw = _kw(base, preset="STRICT", optimizer={"objective_mode": mode})
    assert kw["enforce_quantum_descent"] is expect


def test_joint_runs_converge_on_the_engine_s_scale(base):
    """BALANCED's conv_freq of 1000 MHz made the plateau stop and the
    adaptive Hessian schedule treat any fit as flat (isocyanic acid stopped
    15 mA from its optimum). Joint config runs use the engine's default; an
    explicit split keeps the preset's; a stated value wins."""
    import backend.quantize as Q
    import inspect
    engine_default = inspect.signature(Q.MolecularOptimizer).parameters["conv_freq"].default
    assert "conv_freq" not in _kw(base)
    assert _built(base).conv_freq == engine_default
    assert _kw(base, optimizer={"objective_mode": "split"})["conv_freq"] == pytest.approx(1000.0)
    assert _kw(base, optimizer={"conv_freq": 5.0})["conv_freq"] == pytest.approx(5.0)


def test_loading_a_config_does_not_state_what_it_did_not_state():
    """normalize_config used to fill in prior_weight 1 and widths 0.04 A /
    2 deg, so every CLI run looked as if it had asked for the library prior
    and the engine's rules for unstated values never applied."""
    cfg = load_config(CONFIG)
    ic = cfg["internal_coordinates"]
    assert "prior_weight" not in ic
    assert ic["prior_sigma_bond"] == pytest.approx(0.12)      # stated in the file
    kw = build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]
    assert kw["ic_prior_weight"] is None
    bare = copy.deepcopy(cfg)
    for k in ("prior_sigma_bond", "prior_sigma_angle_deg"):
        bare["internal_coordinates"].pop(k)
    from runner.config_schema import normalize_config
    again = normalize_config(bare)["internal_coordinates"]
    assert "prior_sigma_bond" not in again and "prior_sigma_angle_deg" not in again


def test_the_start_anchor_is_off_while_the_quantum_prior_is_on(base):
    """BALANCED/STRICT anchor each fit to its own start. With a quantum prior
    that made restarts and multistart seeds fit different objectives (4/9
    restart-stationary on 9 molecules, cartesian mode; 8/9 without). It stays
    for spectral-only runs, where nothing else holds the unseen directions."""
    assert _kw(base)["use_internal_priors"] is False
    assert _built(base).internal_prior is None
    spectral = copy.deepcopy(base)
    spectral["quantum"]["backend"] = "none"
    assert build_run_inputs(spectral, verbose=False)["optimizer_kwargs"]["use_internal_priors"] is True


def test_no_shipped_config_states_a_library_prior_weight():
    """A stated weight is honoured, so a template prior_weight: 1.0 kept the
    rule-library prior on in every example and in every config copied from
    one -- measured worse on average (mean bond 12.88 vs 9.51 mA, angle 2.34
    vs 0.97 deg on 9 molecules) and 23% slower."""
    stated = []
    for path in sorted((_ROOT / "configs").glob("*.yaml")):
        ic = load_config(path).get("internal_coordinates") or {}
        if "prior_weight" in ic:
            stated.append(path.name)
    assert not stated, f"shipped configs stating prior_weight: {stated}"
