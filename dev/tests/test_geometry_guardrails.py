"""Geometry guardrails: a trial step that breaks a bond or a contact is refused.

Off by default in the API, on in the config runner (runner/run_generic.py),
its presets (runner/run_settings.py) and molecule_runners/run_water.py. A
trial geometry is invalid when a bonded pair (bonds detected at the start)
is shorter than guardrail_bond_scale_min or longer than
guardrail_bond_scale_max times its covalent-radius sum, or a non-bonded pair
is closer than guardrail_clash_scale times it, on more than
guardrail_max_violations counts. An invalid trial is refused while
accept_requires_geometry_valid holds, and the damping rises by
guardrail_lambda_boost and the trust radius shrinks by guardrail_trust_shrink.

On the nine benchmark fits (HF/6-31G, joint) they refuse no step and change
no answer (T-66), so nothing else exercised them (backlog N35).

Analytic water, no quantum chemistry.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

import backend.quantize as Q
from test_cd_rows_sign import _isos, _x0

import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)

_OH = 0.66 + 0.31        # covalent-radius sum the check uses for O-H (A)
_HH = 0.31 + 0.31


def _optimizer(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Q.MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(None),
                                    quantum_backend="analytic_water", use_autoconfig=False,
                                    coordinate_mode="cartesian", **kw)


def _crushed():
    x = _x0()
    x[1] = x[1] / np.linalg.norm(x[1]) * 0.60          # O-H 0.60 A: ratio 0.62 < 0.65
    return x


def _stretched():
    x = _x0()
    x[1] = x[1] / np.linalg.norm(x[1]) * 1.50          # O-H 1.50 A: ratio 1.55 > 1.45
    return x


def _clash():
    return _x0(deg=20.0)                                # H...H 0.333 A < 0.6 x 0.62 = 0.372 A


def test_the_premises_of_the_three_geometries():
    x = _crushed()
    assert np.linalg.norm(x[1]) / _OH < 0.65 and 0.65 < np.linalg.norm(x[2]) / _OH < 1.45
    x = _stretched()
    assert np.linalg.norm(x[1]) / _OH > 1.45
    x = _clash()
    assert np.linalg.norm(x[1] - x[2]) < 0.60 * _HH
    assert all(0.65 < np.linalg.norm(x[i]) / _OH < 1.45 for i in (1, 2))


def test_they_are_off_by_default_and_then_refuse_nothing():
    opt = _optimizer()
    assert opt.enable_geometry_guardrails is False and opt._guardrail_bonds == []
    for x in (_crushed(), _stretched(), _clash()):
        valid, stats = opt._geometry_validity(x)
        assert valid is True and stats["violations"] == 0


def test_a_sound_geometry_passes_and_only_the_bonds_are_bonded():
    opt = _optimizer(enable_geometry_guardrails=True)
    assert sorted(tuple(sorted(b)) for b in opt._guardrail_bonds) == [(0, 1), (0, 2)]
    valid, stats = opt._geometry_validity(_x0())
    assert valid is True and stats["violations"] == 0
    assert 0.65 < stats["bond_ratio_min"] <= stats["bond_ratio_max"] < 1.45
    assert stats["clash_ratio_min"] > 1.0


@pytest.mark.parametrize("make, field", [(_crushed, "bond_ratio_min"), (_stretched, "bond_ratio_max"),
                                         (_clash, "clash_ratio_min")])
def test_each_rule_refuses_its_geometry(make, field):
    opt = _optimizer(enable_geometry_guardrails=True)
    valid, stats = opt._geometry_validity(make())
    assert valid is False and stats["violations"] == 1, stats
    limit = {"bond_ratio_min": 0.65, "bond_ratio_max": 1.45, "clash_ratio_min": 1.0}[field]
    assert (stats[field] < limit) if field != "bond_ratio_max" else (stats[field] > limit), stats


def test_the_scales_and_the_allowance_are_honoured():
    # Each threshold moved past its geometry passes it.
    assert _optimizer(enable_geometry_guardrails=True, guardrail_bond_scale_min=0.6)._geometry_validity(_crushed())[0]
    assert _optimizer(enable_geometry_guardrails=True, guardrail_bond_scale_max=1.6)._geometry_validity(_stretched())[0]
    assert _optimizer(enable_geometry_guardrails=True, guardrail_clash_scale=0.5)._geometry_validity(_clash())[0]
    # One violation allowed passes one, not two.
    allow = _optimizer(enable_geometry_guardrails=True, guardrail_max_violations=1)
    assert allow._geometry_validity(_crushed())[0] is True
    both = _stretched()
    both[2] = both[2] / np.linalg.norm(both[2]) * 0.60
    valid, stats = allow._geometry_validity(both)
    assert valid is False and stats["violations"] == 2


def _fit_with_a_refused_first_trial(monkeypatch, **kw):
    """Run three iterations with the first trial reported invalid; return the
    history and (lambda_damp, trust_radius) as adapt_lambda first sees them,
    which is after the guardrail's own boost and shrink."""
    opt = _optimizer(enable_geometry_guardrails=True, max_iter=3, **kw)
    calls = {"n": 0}
    real = opt._geometry_validity

    def first_invalid(trial):
        calls["n"] += 1
        valid, stats = real(trial)
        if calls["n"] == 1:
            return False, dict(stats, violations=1)
        return valid, stats

    seen = []
    real_adapt = opt.optimizer.adapt_lambda

    def spy(accepted, *a, **k):
        seen.append((float(opt.optimizer.lambda_damp), float(opt.optimizer.trust_radius)))
        return real_adapt(accepted, *a, **k)

    monkeypatch.setattr(opt, "_geometry_validity", first_invalid)
    monkeypatch.setattr(opt.optimizer, "adapt_lambda", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt.history, seen


def test_the_fit_refuses_an_invalid_trial_and_tightens_its_step(monkeypatch):
    # Premise: unforced, the first trial is a sound, accepted step.
    opt = _optimizer(enable_geometry_guardrails=True, max_iter=1)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert opt.history[0]["accepted"] is True and opt.history[0]["geometry_valid"] is True
    hist, seen = _fit_with_a_refused_first_trial(monkeypatch)
    assert hist[0]["geometry_valid"] is False and hist[0]["accepted"] is False
    assert hist[0]["guardrail_violations"] == 1
    # Against the same run with factors of 1, the boost and the shrink are the
    # configured ones (2.0 and 0.8 by default).
    _hist1, seen1 = _fit_with_a_refused_first_trial(monkeypatch, guardrail_lambda_boost=1.0,
                                                    guardrail_trust_shrink=1.0)
    assert seen[0][0] == pytest.approx(2.0 * seen1[0][0])
    assert seen[0][1] == pytest.approx(0.8 * seen1[0][1])
    _hist3, seen3 = _fit_with_a_refused_first_trial(monkeypatch, guardrail_lambda_boost=3.0,
                                                    guardrail_trust_shrink=0.5)
    assert seen3[0][0] == pytest.approx(3.0 * seen1[0][0])
    assert seen3[0][1] == pytest.approx(0.5 * seen1[0][1])


def test_an_invalid_trial_is_accepted_when_validity_is_not_required(monkeypatch):
    hist, _seen = _fit_with_a_refused_first_trial(monkeypatch, accept_requires_geometry_valid=False)
    assert hist[0]["geometry_valid"] is False and hist[0]["accepted"] is True
