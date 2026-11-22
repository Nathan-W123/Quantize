"""A split fit that stalls short of its own fixed point says so.

The split step fits the data on the range of its stacked Jacobian and lets
the quantum surface alone decide the null space, so its fixed point has no
energy gradient left there. Its rank is cut at sv_min_abs (1/sigma_x in the
Python API), which can put directions the data still weigh on the energy's
side; the acceptance test still scores the data there, refuses the energy's
moves, and the fit stops with that gradient unspent. split_stall measures it
on what the step itself last solved with (its own null space, in q-space in
internal mode, after any pull toward a prior target), so a fit at its own
fixed point is not called stalled because its posterior reads another null
space (T-48, T-49; decisions D-115, D-117). The fits here are analytic water.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

from test_bootstrap_targets import _coords, _split_iso, _split_water, B0_D2O, B0_H2O, MASSES_D2O, MASSES_H2O  # noqa: E402

_STALL = "short of the minimum of its null-space energy"
_OFF = "posterior widths from the mode"


@pytest.mark.parametrize("mode", ["internal", "cartesian"])
def test_a_split_fit_that_stalls_says_so(mode):
    # One constant at sigma_x 0.5 mA: the cut puts every direction on the
    # energy's side, the step's moves are refused, and the fit stops.
    opt = _split_water((2,), sigma=0.0005, coordinate_mode=mode)
    assert not any(h["accepted"] for h in opt.history[-2:]), "premise: its last steps are refused"
    assert opt.split_coupling() <= opt.SPLIT_COUPLING_TOL, "premise: the coupling cannot see it"
    widths, atom, n_neg = opt.split_stall()
    # Measured: 30.88 widths, 10.03 mA (largest atom), in both modes.
    assert widths == pytest.approx(30.88, rel=0.01) and atom == pytest.approx(0.01003, rel=0.01), (widths, atom)
    assert n_neg == 0
    note = opt.split_inconsistency()
    assert note and _STALL in note and "step schedule" in note
    assert any(_STALL in w for w in opt.internal_coordinate_posterior()["warnings"])


@pytest.mark.parametrize("mode", ["internal", "cartesian"])
def test_a_coupled_split_fit_that_stalls_names_the_stall_not_the_coupling(mode):
    opt = _split_water((1, 2), sigma=0.0005, coordinate_mode=mode)
    assert opt.split_coupling() > 0.5, "premise: a coupling it would otherwise report"
    assert opt.split_stall()[0] > 1.0             # measured 3.0 / 3.5
    note = opt.split_inconsistency()
    assert _STALL in note and "null response" not in note


@pytest.mark.parametrize("mode", ["internal", "cartesian"])
def test_the_newton_step_spends_the_step_s_own_null_gradient(mode):
    opt = _split_water((2,), sigma=0.0005, coordinate_mode=mode)
    rec = opt._last_split_step
    dx, n_neg, dq = opt._split_step_newton()
    Vn = rec["Vt"][rec["rank"]:].T
    w = np.linalg.eigvalsh(Vn.T @ rec["H"] @ Vn)
    assert n_neg == 0 and w.min() > -1e-12 * abs(w).max(), "premise: no negative curvature"
    before = np.linalg.norm(Vn.T @ rec["g"])
    after = np.linalg.norm(Vn.T @ (rec["g"] + rec["H"] @ dq))
    assert before > 1e-4 and after < 1e-8 * before
    if mode == "cartesian":
        np.testing.assert_allclose(dx, dq)
    else:
        np.testing.assert_allclose(dx, rec["Bplus"] @ dq)


def test_the_widths_are_counted_as_distance_to_mode_counts_them():
    # The largest component of the step along the posterior precision's
    # eigendirections, each in its own width.
    opt = _split_water((1, 2), sigma=0.0005, coordinate_mode="cartesian")
    x = np.asarray(opt.coords, dtype=float)
    dx, _n, _dq = opt._split_step_newton()
    _z, basis, w, V = opt._mode_decomposition(x)
    expected = float(np.max(np.abs(V.T @ (basis.T @ dx)) * np.sqrt(w)))
    assert opt.split_stall()[0] == pytest.approx(expected, rel=1e-6)


def test_a_direction_without_a_minimum_is_counted_not_measured():
    # Negative curvature along a null direction: the quadratic model has no
    # minimum there, so the step leaves it out (an eigenvalue floor would set
    # its length instead: 1.5-2.2 A on acetyl fluoride and isocyanic acid).
    opt = _split_water((2,), sigma=0.0005, coordinate_mode="cartesian")
    rec = opt._last_split_step
    Vn = rec["Vt"][rec["rank"]:].T
    Hn = Vn.T @ rec["H"] @ Vn
    w, U = np.linalg.eigh(Hn)
    k = int(np.argmax(np.abs(U.T @ (Vn.T @ rec["g"]))))
    v = Vn @ U[:, k]                        # the null direction the gradient pushes along most
    assert w[k] > 0 and abs(v @ rec["g"]) > 1e-6, "premise: a measured direction with a gradient"
    before = opt._split_step_newton()
    rec["H"] = rec["H"] - 2.0 * w[k] * np.outer(v, v)
    dx, n_neg, dq = opt._split_step_newton()
    assert before[1] == 0 and n_neg == 1
    assert abs(v @ dq) < 1e-12 * max(1.0, np.linalg.norm(dq))
    # The other directions are measured as before.
    np.testing.assert_allclose(dq, before[2] - (v @ before[2]) * v, atol=1e-12)


def test_a_fit_at_its_own_fixed_point_keeps_the_off_mode_note():
    # A and B at sigma_x 0.5 mA, internal: the step has spent its own null
    # gradient (0.028 widths) while its posterior, reading a Cartesian null
    # space 0.27 (sine) away, puts the mode 3.6 widths off (T-49 verifier).
    opt = _split_water((0, 1), sigma=0.0005)
    assert opt.coordinate_mode == "internal"
    assert opt.split_stall()[0] < opt.SPLIT_STALL_LIMIT
    note = opt.split_inconsistency()
    assert _OFF in note and _STALL not in note


@pytest.mark.parametrize("case", ["internal prior, one constant", "two isotopologues, C only"])
def test_where_step_and_posterior_read_different_null_spaces_the_off_mode_note_stays(case):
    # Both are also short of their own null-space minimum (0.35 and 1.3
    # widths), but they stand 2.2 and 3.8 widths off the posterior's mode
    # because the step decides its null space differently.
    if case.startswith("internal"):
        opt = _split_water((2,), coordinate_mode="internal", ic_prior_weight=1.0)
    else:
        opt = _split_water(isotopologues=[_split_iso("H2-16O", MASSES_H2O, B0_H2O, (2,)),
                                          _split_iso("D2-16O", MASSES_D2O, B0_D2O, (2,))], symmetry="C1")
    assert opt.split_stall()[0] > opt.SPLIT_STALL_LIMIT, "premise: a stall the note would report"
    assert opt._split_null_mismatch() > opt.SPLIT_NULL_AGREE
    note = opt.split_inconsistency()
    assert _OFF in note and _STALL not in note


def test_a_fit_stopped_at_max_iter_is_said_unconverged_not_stalled():
    opt = _split_water((2,), sigma=0.0005, max_iter=6)
    assert opt.termination["reason"] == "max_iter"
    assert opt.split_stall()[0] > opt.SPLIT_STALL_LIMIT, "premise: it would read as stalled"
    note = opt.split_inconsistency()
    assert "max_iter" in note and _STALL not in note


def test_the_note_fires_above_the_limit_and_not_below(monkeypatch):
    opt = _split_water((2,), sigma=0.0005)
    assert opt.SPLIT_STALL_LIMIT == 0.25
    for widths, fires in ((0.3, True), (0.2, False)):
        monkeypatch.setattr(type(opt), "split_stall", lambda self, w=widths: (w, 1e-3, 0))
        note = opt.split_inconsistency() or ""
        assert (_STALL in note) == fires, (widths, note)


def test_the_step_s_gradient_is_the_pull_toward_a_prior_target():
    from backend.quantize import _kabsch_align
    target = _coords(r=0.98785, deg=107.508)
    opt = _split_water((2,), coordinate_mode="cartesian", prior_target_coords=target)
    rec = opt._last_split_step
    _g, H = opt._projected_reference_terms()
    pull = np.asarray(H) @ (rec["coords"] - _kabsch_align(np.asarray(target), rec["coords"])).ravel()
    assert np.linalg.norm(pull) > 1e-4, "premise: the target pulls"
    np.testing.assert_allclose(rec["g"], pull, rtol=1e-8, atol=1e-12 * np.linalg.norm(pull))


def test_without_a_null_space_or_after_an_accepted_step_there_is_no_stall():
    full = _split_water((1, 2))                       # the data reach every direction
    assert full.split_stall() in (None, (0.0, 0.0, 0))
    assert full.split_inconsistency() is None
    one = _split_water((2,))                          # its last step was accepted
    assert one.history[-1]["accepted"] and one.split_stall() is None
    assert "null response" in one.split_inconsistency()


def test_off_the_split_objective_there_is_no_stall():
    import contextlib
    import io
    import dev.analytic_water_backend  # noqa: F401
    from backend.quantize import MolecularOptimizer
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(),
                                 isotopologues=[_split_iso("H2-16O", MASSES_H2O, B0_H2O, (2,))],
                                 quantum_backend="analytic_water", use_autoconfig=False, max_iter=50,
                                 auto_systematic_sigma=False, objective_mode="joint")
        opt.run()
    assert opt.split_stall() is None and getattr(opt, "_last_split_step", None) is None
