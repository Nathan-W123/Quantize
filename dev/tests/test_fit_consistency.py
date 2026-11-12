"""The fit converges to the optimum of the objective it states.

Five defects kept it from doing so, each measured on real molecules (pyscf
HF/6-31G, published constants, internal coordinates) before it was fixed:

  * the quantum gradient was the one from the last quantum update, used as if
    it were current until the fit drifted 5 mA away;
  * in joint mode a step was accepted on the spectral residual alone, not on
    the joint objective the step minimises;
  * the default internal-coordinate path compared residual vectors of
    different lengths, because the q-space prior rows were added to the trial
    side only when an InternalPriorEngine existed;
  * the calibrated quantum weight came from the q-space Hessian, which mixes
    Angstrom and radian units, and autoconfig retuned it mid-fit;
  * a second pass on the same optimizer inherited the trust radius the first
    pass ended on.

Measured together, before -> after, fresh restart from the fit's answer:

    formyl fluoride, class defaults      restart moved 15.5  -> 0.04 mA
    vinyl fluoride, class defaults       restart moved 14.8  -> 0.46 mA
    vinyl fluoride, autoconfig off       80 iterations, not converged, 24 QM
                                         calls, 22.1 s -> converged in 12, 3
                                         QM calls, 5.2 s

A second pass on the same optimizer, which keeps the same Hessian and so the
same objective, moves 0.0 mA in every case measured.

These tests pin each mechanism on the analytic water surface, where they run
in milliseconds; each fails on the code before the fix.
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

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


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
                        "obs_constants": B0_H2O,
                        "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02)
    base.update(kw)
    return MolecularOptimizer(**base)


def _run(opt):
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


# ── Row parity on the default path ───────────────────────────────────────────

def test_the_default_internal_path_compares_like_rows(monkeypatch):
    """Class defaults: internal mode, ic_prior_weight=1, no InternalPriorEngine.

    The step carries the q-space prior rows; the trial side used to get them
    only when self.internal_prior existed. With the sides of unequal length
    this fixture accepted 0 of its 3 steps and never left its start.
    """
    opt = _opt()
    assert opt.coordinate_mode == "internal" and opt.internal_prior is None
    sizes = []
    orig = MolecularOptimizer._joint_step_ratio

    def spy(self, r0, r1, *a, **k):
        sizes.append((np.asarray(r0).size, np.asarray(r1).size))
        return orig(self, r0, r1, *a, **k)

    monkeypatch.setattr(MolecularOptimizer, "_joint_step_ratio", spy)
    _run(opt)
    assert sizes and all(a == b for a, b in sizes), sizes
    assert sizes[0][0] > 3                 # 3 spectral rows plus the prior's
    assert sum(1 for e in opt.history if e.get("accepted")) >= 1


# ── The quantum gradient between quantum updates ─────────────────────────────

def test_the_gradient_is_re_expanded_to_first_order():
    """g(x) ~ g(x_ref) + H (x - x_ref): error second order in the displacement,
    where the stored gradient's is first order."""
    opt = _opt()
    _run(opt)
    x_ref = np.asarray(opt._orca_ref_coords, dtype=float)
    g_ref = np.asarray(opt.quantum.gradient, dtype=float)
    H = np.asarray(opt.quantum.hessian, dtype=float)
    rng = np.random.default_rng(3)
    delta = rng.standard_normal(x_ref.shape)
    delta *= 1e-3 / np.sqrt(np.mean(delta ** 2))          # 1 mA RMS
    opt.coords = x_ref + delta
    g_true = _true_gradient(opt)
    stale = float(np.linalg.norm(g_ref - g_true))
    expanded = float(np.linalg.norm(
        opt._expanded_quantum_gradient(g_ref, H) - g_true))
    assert stale > 0.0
    assert expanded < 0.05 * stale, (expanded, stale)


def _true_gradient(opt):
    """The backend's gradient at opt.coords, in the stored gradient's units,
    leaving the optimizer's quantum state and reference geometry as found."""
    saved = (opt.quantum.energy, opt.quantum._gradient_bohr.copy(),
             np.asarray(opt._orca_ref_coords, dtype=float).copy())
    try:
        opt._run_gradient()
        return np.asarray(opt.quantum.gradient, dtype=float).copy()
    finally:
        opt.quantum.energy, opt.quantum._gradient_bohr, opt._orca_ref_coords = saved


# ── The joint step test ──────────────────────────────────────────────────────

def _one_parameter_case(opt, r0, dp, g=5.0, h=10.0, alpha=1.0):
    J = np.array([[1.0]])
    r0 = np.array([r0])
    dp = np.array([dp])
    r1 = r0 - J @ dp                       # the residual model is exact here
    return opt._joint_step_ratio(r0, r1, J, dp, np.array([g]),
                                 np.array([[h]]), alpha, opt.coords)


def test_a_step_that_lowers_the_joint_objective_is_accepted_though_r_rises():
    """At the joint optimum J^T r = alpha g, so the descent direction on the
    low-residual side raises |r|. The residual test refused every such step.

    r 0.5 -> 0.7, and the quantum term falls by 0.8: Phi falls by 0.68.
    """
    opt = _opt()
    accept, rho = _one_parameter_case(opt, r0=0.5, dp=-0.2)
    assert accept is True
    assert rho == pytest.approx(1.0)


def test_a_step_that_raises_the_joint_objective_is_refused_though_r_falls():
    """And the converse, which the residual test accepted: r 0.5 -> 0.3 while
    the quantum term rises by 1.2, so Phi rises by 1.12."""
    opt = _opt()
    accept, _rho = _one_parameter_case(opt, r0=0.5, dp=0.2)
    assert accept is False


def test_split_objective_keeps_the_residual_test(monkeypatch):
    """Split has no single objective; the ratio test is joint-mode only."""
    called = []
    monkeypatch.setattr(MolecularOptimizer, "_joint_step_ratio",
                        lambda self, *a, **k: called.append(1) or (True, 1.0))
    _run(_opt(objective_mode="split"))
    assert called == []


# ── The weight the quantum prior carries ─────────────────────────────────────

def test_internal_mode_uses_the_cartesian_weight():
    """The step's alpha_q is the one quantum_prior_weight() reports -- the
    Cartesian definition -- not the q-space one, which on this fit was
    1.5-2.1x larger and is discontinuous in planarity on real molecules."""
    opt = _run(_opt(max_iter=10))
    step_alpha = float(opt.history[-1]["alpha_q_eff"])
    reported = float(opt.quantum_prior_weight(refresh=True))
    assert step_alpha == pytest.approx(reported, rel=1e-3)


def test_autoconfig_leaves_a_calibrated_weight_alone():
    """alpha_quantum multiplies the calibrated weight, so retuning it mid-fit
    changes the objective under the fit. On this fixture autoconfig set it to
    0.531; on formyl fluoride alpha_q fell 3438 -> 2216 within one fit."""
    opt = _run(_opt(use_autoconfig=True, max_iter=10))
    seen = {round(float(e["alpha_quantum"]), 12) for e in opt.history}
    assert seen == {1.0}, seen


# ── Every pass starts from the base trust region ─────────────────────────────

def test_a_second_pass_starts_from_the_base_trust_radius(monkeypatch):
    opt = _opt(max_iter=40)
    _run(opt)
    base = float(opt._base_trust_radius)
    assert float(opt.optimizer.trust_radius) < base   # the first pass shrank it
    first = []
    orig = type(opt.optimizer).step

    def spy(self, *a, **k):
        if not first:
            first.append(float(self.trust_radius))
        return orig(self, *a, **k)

    monkeypatch.setattr(type(opt.optimizer), "step", spy)
    _run(opt)
    assert first == [base]
