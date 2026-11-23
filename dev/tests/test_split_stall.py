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
    # A weak but positive curvature is still a minimum, measured.
    rec["H"] = rec["H"] + (1.0 + 1e-3) * w[k] * np.outer(v, v)    # -w[k] -> 1e-3 w[k]
    wk = float(v @ rec["H"] @ v)
    assert 0 < wk < 1e-2 * w[k]
    dx, n_neg, dq = opt._split_step_newton()
    assert n_neg == 0
    assert v @ dq == pytest.approx(-(v @ rec["g"]) / wk, rel=1e-6)


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


def test_with_the_data_reaching_every_direction_the_stall_reads_zero():
    full = _split_water((1, 2))
    assert full.split_stall()[0] < 1e-12
    assert full.split_inconsistency() is None


def test_a_fit_that_ends_on_an_accepted_step_is_measured_from_its_last_record():
    one = _split_water((2,))
    assert one.history[-1]["accepted"], "premise: the last step was accepted"
    assert not np.array_equal(one._last_split_step["coords"], one.coords), "premise: a step behind"
    widths, _atom, _n = one.split_stall()
    assert widths < one.SPLIT_STALL_LIMIT            # measured 0.0185
    assert "null response" in one.split_inconsistency()


@pytest.mark.parametrize("mode", ["cartesian", "internal"])
def test_the_record_is_carried_over_the_last_step(mode):
    # A fit that ends on an accepted step is a step from its record: the
    # model's gradient is carried there exactly, so the null-space minimum
    # it points to does not move (acetyl fluoride, Cartesian, ends so after
    # eleven refusals, 4.0 widths short; T-50 verifier).
    opt = _split_water((2,), sigma=0.0005, coordinate_mode=mode)
    x0 = np.asarray(opt.coords, dtype=float).copy()
    dx0 = opt._split_step_newton()[0]
    opt.coords = x0 + (1e-3 * dx0 / np.linalg.norm(dx0)).reshape(-1, 3)
    dx1 = opt._split_step_newton()[0]
    assert np.linalg.norm(dx1) < np.linalg.norm(dx0), "premise: the gradient was carried"
    np.testing.assert_allclose(np.asarray(opt.coords).ravel() + dx1, x0.ravel() + dx0,
                               atol=(1e-12 if mode == "cartesian" else 2e-6) * max(1.0, np.linalg.norm(dx0)) + 1e-9)


def _hdo():
    from backend.spectral.centrifugal_distortion import rotational_constants_mhz
    masses = [15.9949146196, 1.00782503207, 2.0141017778]
    return masses, rotational_constants_mhz(_coords(0.9600, 104.3), masses)


@pytest.mark.parametrize("case", ["H2O C + HDO A", "HDO B only"])
def test_under_symmetry_one_null_space_reads_as_one(case):
    # An HDO constant sees a symmetry-forbidden direction, so the null
    # vectors under C2v mix allowed and forbidden parts. Cut by two rules
    # the step's and the posterior's (equal to 2e-16) read 1.0 apart, and a
    # stall was reported as a disagreement or a coupling (T-50 verifier).
    m_hdo, b_hdo = _hdo()
    if case.startswith("H2O"):
        isos = [_split_iso("H2-16O", MASSES_H2O, B0_H2O, (2,)), _split_iso("HD-16O", m_hdo, b_hdo, (0,))]
        sigma = 0.0005
    else:
        isos, sigma = [_split_iso("HD-16O", m_hdo, b_hdo, (1,))], 0.02
    opt = _split_water(isotopologues=isos, symmetry="auto", sigma=sigma, coordinate_mode="cartesian")
    assert str(opt.point_group).upper().startswith("C2V"), "premise: the symmetry is imposed"
    assert opt._split_null_mismatch() < 1e-10
    assert opt.split_stall()[0] > opt.SPLIT_STALL_LIMIT     # measured 29.8 and 0.456
    assert _STALL in opt.split_inconsistency()


def test_the_null_spaces_agree_below_the_limit_and_not_above(monkeypatch):
    opt = _split_water((2,), sigma=0.0005)
    assert opt.SPLIT_NULL_AGREE == 0.1
    for mismatch, stall_note in ((0.05, True), (0.3, False)):
        monkeypatch.setattr(type(opt), "_split_null_mismatch", lambda self, m=mismatch: m)
        note = opt.split_inconsistency()
        assert (_STALL in note) == stall_note, (mismatch, note)


def test_a_prior_row_counts_in_the_recorded_rank():
    # Internal prior, one constant: one spectral row, but the step's stack
    # carries the q-space prior rows, and its rank is the stack's.
    opt = _split_water((2,), coordinate_mode="internal", ic_prior_weight=1.0)
    J, _r = opt.spectral.stacked(opt.coords)
    assert J.shape[0] == 1, "premise: a single spectral row"
    assert opt._last_split_step["rank"] == 2


def test_off_its_posterior_s_mode_with_one_null_space_the_note_names_no_cause():
    # Same null space, no stall: the note says the posterior is not centred
    # on the fit and names no cause it has not measured (here the target's
    # pull along the data's directions moves the mode, not the coupling;
    # T-53 verifier).
    opt = _split_water((2,), coordinate_mode="cartesian", sigma=0.005,
                       prior_target_coords=_coords(r=0.98785, deg=107.508))
    assert opt._split_null_mismatch() < opt.SPLIT_NULL_AGREE, "premise: one null space"
    assert opt.split_stall()[0] < opt.SPLIT_STALL_LIMIT, "premise: no stall"
    note = opt.split_off_mode()
    assert note and "same null space" in note and "not centred on this fit" in note
    assert "disagree" not in note and "coupling" not in note


@pytest.mark.parametrize("mode", ["cartesian", "internal"])
def test_the_carry_adds_the_model_s_change_over_any_step(mode):
    # Range and null parts alike: g + H d, d the step in the step's own
    # parameters (the change in the active internal coordinates in internal
    # mode, not a map through B+).
    opt = _split_water((1, 2), sigma=0.0005, coordinate_mode=mode)
    rec = opt._last_split_step
    x0 = np.asarray(rec["coords"], dtype=float)
    # A step along the record's range and along its null-space Newton step.
    u_r = rec["Vt"][0]
    u_n = opt._split_step_newton()[2]
    u = u_r / np.linalg.norm(u_r) + u_n / np.linalg.norm(u_n)
    step = 1e-3 * (u if rec["Bplus"] is None else rec["Bplus"] @ u)
    delta = step.reshape(-1, 3)
    opt.coords = x0 + delta
    if mode == "cartesian":
        d = delta.ravel()
    else:
        cs = rec["coord_set"]
        d = np.asarray(cs.active_values(opt.coords)) - np.asarray(cs.active_values(x0))
    carried = opt._split_record()["g"]
    np.testing.assert_allclose(carried, rec["g"] + rec["H"] @ d, rtol=1e-12, atol=1e-15)
    Vn = rec["Vt"][rec["rank"]:].T
    Vr = rec["Vt"][:rec["rank"]].T
    assert np.linalg.norm(Vr.T @ (rec["H"] @ d)) > 1e-6, "premise: the step has a range part"
    assert np.linalg.norm(Vn.T @ (rec["H"] @ d)) > 1e-6, "premise: and a null part"


def test_a_dihedral_carried_across_180_degrees_is_wrapped():
    # A trans dihedral stepping from +179.9 to -179.9 deg moves 0.2 deg, not
    # -359.8: carried unwrapped, the gradient would jump by 2 pi H.
    opt = _split_water((2,), sigma=0.0005)
    rec = dict(opt._last_split_step)
    x0 = np.asarray(rec["coords"], dtype=float)
    x1 = x0 + 1e-4

    class _Dihedral:
        kind = "dihedral"

    class _Set:
        def active_coords(self):
            return [_Dihedral()]

        def active_values(self, x):
            near = np.allclose(np.asarray(x, dtype=float), x0)
            return np.array([np.radians(179.9) if near else np.radians(-179.9)])

    rec.update(coord_set=_Set(), Bplus=np.ones((x0.size, 1)), g=np.zeros(1), H=np.eye(1),
               Vt=np.eye(1), rank=0)
    opt._last_split_step = rec
    opt.coords = x1
    g = opt._split_record()["g"]
    np.testing.assert_allclose(g, [np.radians(0.2)], rtol=1e-9)


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
