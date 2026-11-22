"""Internal mode: a step the back-transform cannot realise is retaken in Cartesians.

A valence angle has its maximum at 180 deg, so near a linear chain an
internal step can ask for what no geometry has; the back-transform realises
something else, that step raises the objective and is refused, and it comes
back unchanged (it lies inside the trust radius) until the plateau rule
stops the fit. Fluoroacetylene (HF/6-31G) stopped so 0.008 posterior widths
short and its bootstrap kept 27 of 50 refits. ic_cartesian_fallback (opt-in)
retakes such a step in Cartesians on the same objective and steps in
Cartesians for the rest of the pass (decisions D-105, D-106).

The fits are an H-C-C-F chain on a valence force field, no quantum chemistry.
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

import backend.quantize as Q  # noqa: E402
import backend.spectral.spectral as SP  # noqa: E402
from backend.base_backend import GradientResult, HessianResult, QuantumBackend  # noqa: E402
from backend.registry import register_backend  # noqa: E402

BOHR = 0.529177210903
R0 = (1.06, 1.20, 1.28)        # H-C, C-C, C-F (theory)
KB = 0.05
def ang(a, b, c):
    u, v = a - b, c - b
    return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
def E(x):
    x = np.asarray(x, float).reshape(4, 3)
    e = sum(0.5 * k * (np.linalg.norm(x[i + 1] - x[i]) - r) ** 2 for i, (k, r) in enumerate(zip((1.4, 4.0, 2.0), R0)))
    e += KB * (1 + np.cos(ang(x[0], x[1], x[2]))) + KB * (1 + np.cos(ang(x[1], x[2], x[3])))
    return e
def G(x, h=1e-5):
    f = np.asarray(x, float).ravel(); g = np.zeros(12)
    for i in range(12):
        p, m = f.copy(), f.copy(); p[i] += h; m[i] -= h; g[i] = (E(p) - E(m)) / (2 * h)
    return g
def H(x, h=1e-4):
    f = np.asarray(x, float).ravel(); Hm = np.zeros((12, 12))
    for i in range(12):
        p, m = f.copy(), f.copy(); p[i] += h; m[i] -= h; Hm[:, i] = (G(p) - G(m)) / (2 * h)
    return 0.5 * (Hm + Hm.T)
@register_backend
class _LinearTetratomic(QuantumBackend):
    name = "test_linear_tetratomic"
    def __init__(self, elems=None, **_): pass
    def run_hessian(self, x): return HessianResult(energy=E(x), gradient_bohr=G(x) * BOHR, hessian_bohr=H(x) * BOHR ** 2)
    def run_gradient(self, x): return GradientResult(energy=E(x), gradient_bohr=G(x) * BOHR)
TRUE = np.array([[0, 0, -1.062], [0, 0, 0], [0, 0, 1.198], [0, 0, 2.477]], float)
M = dict(H=1.00782503207, D=2.01410177785, C=12.0, C13=13.00335483507, F=18.99840316)
isos = []
for nm, m in (("p", "HCCF"), ("d", "DCCF"), ("c1", "H3CF"), ("c2", "HC3F")):
    ms = [M["C13"] if ch == "3" else M[ch] for ch in m]
    isos.append(dict(name=nm, masses=ms, obs_constants=[float(SP._rotational_constants(TRUE, np.array(ms))[1])], sigma_constants=[0.5], alpha_constants=[0.0], component_indices=[1]))


def _fit(seed, fallback):
    start = TRUE + np.random.default_rng(seed).normal(0, 0.03, TRUE.shape)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=["H", "C", "C", "F"], coords=start, isotopologues=isos,
                                   quantum_backend="test_linear_tetratomic", use_autoconfig=False, max_iter=80,
                                   hess_recalc_every=10, coordinate_mode="internal", ic_cartesian_fallback=fallback)
        x = np.asarray(opt.run(), float)
    return opt, x


def test_a_stalled_near_linear_fit_reaches_its_optimum():
    stuck, _ = _fit(2, False)
    # Premise: without it the fit stops short with refused steps.
    # Measured: 4.1e-2 widths, 5 refused steps.
    assert stuck.distance_to_mode() > 1e-2
    assert sum(not r["accepted"] for r in stuck.history) >= 3
    opt, x = _fit(2, True)
    # Measured: 5.8e-8 widths, 11 Cartesian steps.
    assert opt._ic_fallback_steps > 0
    assert opt.distance_to_mode() < 1e-5


@pytest.mark.parametrize("seed", [1, 3])
def test_a_fit_that_never_stalls_is_unchanged(seed):
    a, xa = _fit(seed, False)
    b, xb = _fit(seed, True)
    assert b._ic_fallback_steps == 0
    np.testing.assert_array_equal(xa, xb)


def test_it_is_off_by_default():
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=["H", "C", "C", "F"], coords=TRUE, isotopologues=isos,
                                   quantum_backend="test_linear_tetratomic", use_autoconfig=False,
                                   coordinate_mode="internal")
    assert opt._ic_cartesian_fallback is False


@pytest.mark.parametrize("value, ok", [(True, True), (False, True), ("yes", False), (1, False)])
def test_the_config_takes_a_boolean(value, ok):
    from runner.usability import ConfigError, _validate_run_controls
    cfg = {"internal_coordinates": {"cartesian_fallback": value}}
    if ok:
        _validate_run_controls(cfg, ["H", "C", "C", "F"])
    else:
        with pytest.raises(ConfigError, match="cartesian_fallback"):
            _validate_run_controls(cfg, ["H", "C", "C", "F"])


def test_after_the_first_fallback_the_pass_steps_in_cartesians():
    # Alternating let the internal steps undo the fallback's (acetyl
    # fluoride, HF/6-31G: 0.40 widths from its optimum against 0.19 without).
    opt, _ = _fit(2, True)
    h = opt.history
    first = next(i for i, r in enumerate(h) if r.get("cartesian_fallback"))
    later = [r for r in h[first:] if r["accepted"]]
    assert later and all(r["cartesian_fallback"] for r in later)


def test_a_refused_step_the_back_transform_realised_is_not_retaken():
    # A converged fit's last, zero-length steps are refused with a round-off
    # back-transform error; they passed a purely relative test and switched
    # converged fits to Cartesians (formyl fluoride, chlorofluoromethane).
    import test_ic_dihedrals_auto as P
    opt, _ = P._optimizer(dihedrals=True, ic_cartesian_fallback=True)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    refused = [r for r in opt.history if not r["accepted"]]
    assert refused, "premise: the fit refuses steps"
    assert max(r["backtransform_error"] for r in refused) < 1e-6, "premise: realised to round-off"
    assert opt._ic_fallback_steps == 0
