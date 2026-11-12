"""The internal-coordinate prior: when it is on, and whose widths it uses.

Two decisions, both stated by the user on 2026-11-12:

1. The objective is spectra + quantum chemistry. The internal-coordinate
   prior is a third term -- rule-book centres and widths -- so with a quantum
   prior active it is off by default. Spectral-only fits, which have nothing
   else to hold the directions the data cannot see, keep it on. An explicit
   ic_prior_weight is used as given.

   Measured with it on: vinyl fluoride's H-C-H pulled toward 109.5 deg against
   120.1 published, formyl fluoride's O-C-H toward 120 against 127.3, and
   formyl fluoride's angle error 2.03 deg against 1.08 with it off.

2. Widths given are widths used. The rule library replaced the width of
   every coordinate it had a rule for, so a configured 0.12 Angstrom / 5 deg
   became 0.015 / 2 without a word. A width set explicitly now wins for its
   kind of coordinate; left unset, the library's applies where it has a rule.
   The library's centres apply either way.
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.internal.internal_fit import (  # noqa: E402
    InternalCoordinateSet, build_internal_priors)
from backend.internal.internal_prior_library import resolve_default_prior  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]
_CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _opt(**kw):
    base = dict(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O, "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02)
    base.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return MolecularOptimizer(**base)


# ── 1. When it is on ─────────────────────────────────────────────────────────

def test_off_by_default_when_a_quantum_prior_is_on():
    opt = _opt()
    assert opt.coordinate_mode == "internal"
    assert opt._ic_prior_weight == 0.0
    assert "quantum prior" in opt._ic_prior_weight_reason


def test_on_by_default_for_a_spectral_only_fit():
    opt = _opt(spectral_only=True)
    assert opt._ic_prior_weight == 1.0


def test_an_explicit_weight_is_used_as_given():
    assert _opt(ic_prior_weight=0.4)._ic_prior_weight == pytest.approx(0.4)
    assert _opt(ic_prior_weight=0.0, spectral_only=True)._ic_prior_weight == 0.0


def test_the_banner_says_why():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(),
            isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                            "obs_constants": B0_H2O,
                            "sigma_constants": [0.2] * 3}],
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=3, quantum_prior_sigma_ang=0.02)
    assert "prior_weight=0.00 (off: the quantum prior is on)" in buf.getvalue()


def test_with_it_off_the_step_carries_no_prior_rows(monkeypatch):
    """Off means off: the joint step test sees the spectral rows only."""
    sizes = []
    orig = MolecularOptimizer._joint_step_ratio

    def spy(self, r0, r1, *a, **k):
        sizes.append(np.asarray(r0).size)
        return orig(self, r0, r1, *a, **k)

    monkeypatch.setattr(MolecularOptimizer, "_joint_step_ratio", spy)
    opt = _opt()
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert sizes and set(sizes) == {3}


# ── 2. Whose widths ──────────────────────────────────────────────────────────

def _water_set():
    x = _coords()
    return InternalCoordinateSet(x, ["O", "H", "H"], False), x


def _library_widths():
    """The rule library's own target and width for each active coordinate."""
    cs, x = _water_set()
    q = cs.active_values(x)
    out = []
    for ic, qi in zip(cs.active_coords(), q):
        rule = resolve_default_prior(ic, ["O", "H", "H"], qi)
        out.append(None if rule is None else (rule.target_value, rule.sigma))
    return out


def test_the_library_has_rules_for_water_so_the_test_means_something():
    rules = _library_widths()
    assert all(r is not None for r in rules), rules


def test_unset_widths_take_the_library_s():
    cs, x = _water_set()
    _J, _r, sigma = build_internal_priors(
        cs, x, sigma_bond=0.12, sigma_angle_deg=5.0, elems=["O", "H", "H"])
    lib = _library_widths()
    assert np.allclose(sigma, [s for _t, s in lib])


def test_configured_widths_win_and_library_centres_still_apply():
    cs, x = _water_set()
    J, r, sigma = build_internal_priors(
        cs, x, sigma_bond=0.12, sigma_angle_deg=5.0, elems=["O", "H", "H"],
        configured_kinds={"bond", "angle"})
    kinds = [ic.kind for ic in cs.active_coords()]
    expect = [0.12 if k == "bond" else np.radians(5.0) for k in kinds]
    assert np.allclose(sigma, expect)
    # The centre is still the library's: r = (q0 - q)/sigma with q0 its target.
    q = cs.active_values(x)
    targets = np.array([t for t, _s in _library_widths()])
    assert np.allclose(r, (targets - q) / np.asarray(expect))


def test_the_optimizer_marks_only_what_was_given():
    assert _opt()._ic_prior_configured_kinds == set()
    assert _opt(ic_prior_sigma_bond=0.1)._ic_prior_configured_kinds == {"bond"}
    both = _opt(ic_prior_sigma_bond=0.1, ic_prior_sigma_angle_deg=4.0)
    assert both._ic_prior_configured_kinds == {"bond", "angle"}
    assert both._ic_prior_sigma_bond == pytest.approx(0.1)
    assert both._ic_prior_sigma_angle_deg == pytest.approx(4.0)


def test_the_config_path_passes_widths_only_when_stated():
    from runner.run_generic import build_run_inputs
    from runner.usability import load_config

    cfg = load_config(_CONFIG)
    stated = build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]
    assert stated["ic_prior_sigma_bond"] == pytest.approx(0.12)
    assert stated["ic_prior_sigma_angle_deg"] == pytest.approx(5.0)

    silent_cfg = copy.deepcopy(cfg)
    for key in ("prior_sigma_bond", "prior_sigma_angle_deg"):
        silent_cfg["internal_coordinates"].pop(key, None)
    silent = build_run_inputs(silent_cfg, verbose=False)["optimizer_kwargs"]
    assert silent["ic_prior_sigma_bond"] is None
    assert silent["ic_prior_sigma_angle_deg"] is None
