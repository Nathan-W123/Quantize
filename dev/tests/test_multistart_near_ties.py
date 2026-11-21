"""Multistart starts that end near-tied with the selected one but far from it.

The ranking picks the start with the lowest energy, but measured through
`cli run` (T-23: five molecules, ten runs, 30 starts) every misranked pick was
between starts within 1.4e-6 Ha of each other at their answers -- below what
any energy these fits carry resolves -- and such near-tied starts sat up to
20.6 mA apart along flat directions (a methyl torsion, a CH2 twist). The reported geometry
is then one of several the fit does not decide between. The engine now says
so (decisions D-70): each start records the fit's own model energy at its
answer, and near_tied_starts lists the others within NEAR_TIE_ENERGY_HA and
farther than NEAR_TIE_DISTANCE_ANG. On the T-23 data that flags 5 of the 10
runs (8.6-20.6 mA), and none of formyl fluoride's or chlorofluoromethane's;
checked against true SCF energies, every start listed is near-tied.
"""

import contextlib
import io

import numpy as np

import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)
from backend.multistart import (NEAR_TIE_DISTANCE_ANG, NEAR_TIE_ENERGY_HA,
                                near_tie_note, near_tied_starts)

_X = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.4, 1.1, 0.0], [0.3, -0.5, 0.9]])


def _rotated(x, deg):
    t = np.radians(deg)
    r = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    return x @ r.T + np.array([0.3, -0.2, 1.0])


def _res(idx, x, e, freq=1.0):
    return {"idx": idx, "coords": np.asarray(x, dtype=float), "energy_at_answer_model": e,
            "freq_rms": freq}


def test_only_a_near_tied_far_start_inside_the_gate_is_listed():
    best = _res(1, _X, -100.0)
    far = _X.copy(); far[3] += [0.0, 0.0, 0.010]          # one atom 10 mA away
    near = _X.copy(); near[3] += [0.0, 0.0, 0.001]        # 1 mA: under the distance
    results = [
        best,
        _res(2, _rotated(far, 37.0), -100.0 + 2e-5),       # tied, far (rotated, moved)
        _res(3, _rotated(near, -12.0), -100.0 - 1e-5),     # tied, near
        _res(4, far, -100.0 + 3 * NEAR_TIE_ENERGY_HA),     # far, not tied
        _res(5, far, -100.0, freq=10.0),                   # tied, far, outside the gate
    ]
    got = near_tied_starts(results, best)
    assert [d["idx"] for d in got] == [2]
    # Alignment removes the rigid rotation and shift: the rotated, moved copy
    # is exactly as far as the unrotated one (the 10 mA one-atom move, spread
    # by the RMS alignment to 5.7 mA at the largest atom).
    plain = near_tied_starts([best, _res(2, far, -100.0 + 2e-5)], best)
    assert abs(got[0]["max_atom_distance_ang"] - plain[0]["max_atom_distance_ang"]) < 1e-12
    assert got[0]["max_atom_distance_ang"] > NEAR_TIE_DISTANCE_ANG
    assert abs(got[0]["energy_difference_ha"] - 2e-5) < 1e-12


def test_a_rigid_copy_is_not_far():
    best = _res(1, _X, -100.0)
    results = [best, _res(2, _rotated(_X, 80.0), -100.0)]
    assert near_tied_starts(results, best) == []


def test_without_a_model_energy_there_is_nothing_to_say():
    best = _res(1, _X, None)
    far = _X.copy(); far[0] += 0.05
    assert near_tied_starts([best, _res(2, far, -100.0)], best) == []


def test_the_note_names_the_starts_and_distances():
    best = _res(1, _X, -100.0)
    far = _X.copy(); far[2] += [0.02, 0.0, 0.0]
    near = near_tied_starts([best, _res(2, far, -100.0 + 1e-5)], best)
    note = near_tie_note(near)
    mA = f"{1000.0 * near[0]['max_atom_distance_ang']:.1f} mA"
    assert note is not None and "start 2" in note and mA in note and "+1.0e-05 Eh" in note
    assert near_tie_note([]) is None
    assert NEAR_TIE_DISTANCE_ANG == 2e-3


def test_a_real_start_records_its_model_energy_at_the_answer(tmp_path):
    from backend.multistart import run_multistart
    masses = [15.99491461956, 1.00782503207, 1.00782503207]
    b0 = [835840.29, 435351.72, 278138.70]
    th = np.radians(104.5)
    x0 = np.array([[0.0, 0.0, 0.0], [0.96 * np.sin(th / 2), 0.96 * np.cos(th / 2), 0.0],
                   [-0.96 * np.sin(th / 2), 0.96 * np.cos(th / 2), 0.0]])
    isos = [{"name": "H2-16O", "masses": masses, "obs_constants": b0, "sigma_constants": [0.2] * 3}]
    kwargs = dict(quantum_backend="analytic_water", use_autoconfig=False, max_iter=5,
                  quantum_prior_sigma_ang=0.02, base_workdir=str(tmp_path), job_name="nt")
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_multistart(starts=[x0], elems=["O", "H", "H"], isotopologues=isos,
                             optimizer_kwargs=kwargs, max_workers=1, job_name="nt")
    e = res[0]["energy_at_answer_model"]
    assert e is not None and np.isfinite(e)


def test_the_report_line_comes_from_the_stored_near_ties():
    from runner.usability import _near_tie_note
    far = _X.copy(); far[1] += [0.0, 0.015, 0.0]
    best = _res(1, _X, -100.0)
    best["multistart_near_ties"] = near_tied_starts([best, _res(3, far, -100.0)], best)
    assert _near_tie_note(best).startswith("near-tied starts: start 3")
    best["multistart_near_ties"] = []
    assert _near_tie_note(best) is None
    assert _near_tie_note({}) is None

