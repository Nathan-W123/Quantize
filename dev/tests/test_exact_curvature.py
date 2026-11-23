"""Exact curvature where the Gauss-Newton check flags it (D-145).

Along a probed soft direction where the exact curvature is well above
Gauss-Newton's (finite difference stable, same sign, over 50% apart), the
joint step adds the difference to its matrix after the floor and the damping,
and the reported widths, the distance to the mode and the bootstrap gate use
the exact value. The check itself keeps comparing against Gauss-Newton.

Measured on a hydrogen-bonded water dimer (HF/6-31G, T-58 to T-70):
Gauss-Newton kept 0.0077 of an exact 252 along the H-bond torsion, so the
reported torsion widths were 181 and 11 times too wide and the gate refused
refits that stood within 0.019 exact widths of their optimum. Those runs
need quantum chemistry; here the rows are planted on analytic water.
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
from backend.internal.internal_fit import InternalCoordinateSet
from test_cd_rows_sign import _isos, _x0

import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)


def _optimizer(**kw):
    kw.setdefault("coordinate_mode", "cartesian")
    with contextlib.redirect_stdout(io.StringIO()):
        return Q.MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(None),
                                    quantum_backend="analytic_water", use_autoconfig=False, **kw)


def _row(v, c_gn=1.0, c_ex=5.0, index=0, **over):
    row = {"index": index, "direction": np.asarray(v, dtype=float), "fd_stable": True,
           "sign_flipped": False, "relative_discrepancy": abs(c_ex - c_gn) / max(abs(c_gn), 1e-300),
           "curvature_gauss_newton": c_gn, "curvature_exact": c_ex, "omitted_term": c_ex - c_gn}
    row.update(over)
    return row


@pytest.mark.parametrize("over, report, step", [
    ({}, True, True),
    ({"fd_stable": False}, False, False),
    ({"sign_flipped": True}, False, False),
    ({"relative_discrepancy": 0.4}, False, False),
    # The exact curvature the smaller: the report uses it, the step does not.
    ({"curvature_exact": 0.4, "relative_discrepancy": 0.6}, True, False),
    ({"curvature_exact": -0.6, "relative_discrepancy": 1.6}, False, False),
])
def test_which_probed_directions_take_the_exact_curvature(over, report, step):
    row = _row(np.ones(9), **over)
    assert Q.MolecularOptimizer._exact_curvature_applies(row) is report
    assert Q.MolecularOptimizer._exact_curvature_applies(row, step=True) is step


def test_the_joint_step_adds_the_term_after_the_floor_and_the_damping():
    opt = _optimizer().optimizer
    rng = np.random.default_rng(0)
    J, r = rng.normal(size=(6, 4)), rng.normal(size=6)
    g, M = rng.normal(size=4), rng.normal(size=(4, 4))
    H = M @ M.T - 2.0 * np.eye(4)                 # indefinite: the floor acts
    v = np.array([1.0, 2.0, 0.0, -1.0]) / np.sqrt(6.0)
    K = 7.0 * np.outer(v, v)
    alpha = 3.0
    prior = alpha * opt._spd_hessian(H)
    A = J.T @ J + prior + opt.lambda_damp * (np.linalg.norm(prior) / 4) * np.eye(4)
    rhs = J.T @ r - alpha * g
    assert np.allclose(opt._joint_step(J, r, g, H, alpha), np.linalg.solve(A, rhs))
    assert np.allclose(opt._joint_step(J, r, g, H, alpha, curvature_correction=K),
                       np.linalg.solve(A + K, rhs))
    # A term of the wrong shape is not applied.
    assert np.allclose(opt._joint_step(J, r, g, H, alpha, curvature_correction=np.eye(3)),
                       np.linalg.solve(A, rhs))


def test_the_cartesian_step_term_is_the_curvature_difference_along_the_direction(monkeypatch):
    opt = _optimizer()
    v = np.zeros(9)
    v[3], v[6] = 1.0, -1.0
    v /= np.linalg.norm(v)
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [
        _row(v, 1.0, 5.0), _row(np.eye(9)[0], 1.0, 5.0, fd_stable=False)])
    K = opt._step_curvature_correction(opt.coords)
    assert np.allclose(K, 4.0 * np.outer(v, v))
    assert opt._exact_curvature_steps == 1
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [_row(v, 5.0, 1.0)])
    assert opt._step_curvature_correction(opt.coords) is None
    assert opt._exact_curvature_steps == 1


def test_the_internal_step_term_is_its_q_space_image_where_the_step_reaches_it(monkeypatch):
    opt = _optimizer(coordinate_mode="internal")
    x = opt.coords
    cs = InternalCoordinateSet(x, opt.elems, opt._ic_use_dihedrals)
    Bq = np.asarray(cs.active_B_matrix(x), dtype=float)
    Bp = InternalCoordinateSet.damped_pseudoinverse(Bq, opt._ic_damping)
    reached = Bp @ (Bq @ np.array([0, 0, 0, 1.0, 0.5, 0, 0, 0, 0]))
    reached /= np.linalg.norm(reached)
    assert reached @ Bp @ (Bq @ reached) > 0.99                 # premise: realised
    # A rotation about z: no bond or angle moves, so the step cannot reach it.
    rot = np.cross([0.0, 0.0, 1.0], x).ravel()
    rot /= np.linalg.norm(rot)
    assert abs(rot @ Bp @ (Bq @ rot)) < 1e-6                     # premise: not realised
    # Mostly a rotation, partly reached: realised 0.36 < 0.5, with an image
    # B+^T v that is not small, so the reach rule is what leaves it out.
    part = 0.6 * reached + 0.8 * rot
    part /= np.linalg.norm(part)
    assert 0.1 < part @ Bp @ (Bq @ part) < 0.5 and np.linalg.norm(Bp.T @ part) > 0.1   # premise
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [_row(reached, 1.0, 5.0),
                                                                          _row(rot, 1.0, 5.0, index=1),
                                                                          _row(part, 1.0, 5.0, index=2)])
    K = opt._step_curvature_correction(x, Bq=Bq, Bplus=Bp)
    u = Bp.T @ reached
    assert K.shape == (Bq.shape[0], Bq.shape[0])
    assert np.allclose(K, 4.0 * np.outer(u, u))


@pytest.mark.parametrize("kw", [{"exact_curvature_where_flagged": False}, {"spectral_only": True},
                                {"objective_mode": "split"}])
def test_no_step_term_without_the_option_a_quantum_half_or_the_joint_step(monkeypatch, kw):
    opt = _optimizer(**kw)
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [_row(np.eye(9)[3])])
    assert opt._step_curvature_correction(opt.coords) is None


@pytest.mark.parametrize("mode", ["cartesian", "internal"])
def test_the_fit_hands_the_term_to_its_step(monkeypatch, mode):
    seen = []
    for on in (True, False):
        opt = _optimizer(coordinate_mode=mode, max_iter=1, exact_curvature_where_flagged=on)
        v = opt.physical_basis()[:, 0]
        monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, _v=v, **k: [_row(_v)])
        real = opt.optimizer._joint_step

        def spy(*a, _real=real, _on=on, **k):
            seen.append((_on, k.get("curvature_correction"), np.shape(a[0])[1]))
            return _real(*a, **k)

        monkeypatch.setattr(opt.optimizer, "_joint_step", spy)
        with contextlib.redirect_stdout(io.StringIO()):
            opt.run()
    assert seen[0][0] is True and seen[0][1] is not None
    # In the step's own parameters (q-space in internal mode), or the step
    # would drop it on shape.
    assert np.shape(seen[0][1]) == (seen[0][2], seen[0][2])
    assert seen[-1][0] is False and seen[-1][1] is None


def _planted(opt, factor=100.0, k=0):
    """The analytic fit's own probed rows, with direction k's exact curvature
    set to `factor` times Gauss-Newton's."""
    rows = opt.gauss_newton_discrepancy(coords=opt.coords)
    out = []
    for r in rows:
        r = dict(r)
        if r["index"] == k:
            r.update(curvature_exact=factor * r["curvature_gauss_newton"], fd_stable=True,
                     sign_flipped=False, relative_discrepancy=abs(factor - 1.0))
        else:
            r.update(curvature_exact=r["curvature_gauss_newton"], relative_discrepancy=0.0)
        out.append(r)
    return out


def test_the_widths_and_the_distance_use_the_exact_curvature_and_the_check_does_not(monkeypatch):
    opt = _optimizer()
    rows = _planted(opt)
    v, c_gn = rows[0]["direction"], rows[0]["curvature_gauss_newton"]
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, **k: 1e-3 * np.sqrt(c_gn) * v)
    cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert float(v @ cov @ v) == pytest.approx(1.0 / (100.0 * c_gn + 1e-8), rel=1e-6)
    other = rows[1]["direction"]
    opt._exact_curvature = False
    cov_gn, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert float(v @ cov_gn @ v) == pytest.approx(1.0 / (c_gn + 1e-8), rel=1e-6)
    assert float(other @ cov_gn @ other) == pytest.approx(float(other @ cov @ other), rel=1e-9)
    d_gn = opt.distance_to_mode()
    opt._exact_curvature = True
    assert d_gn == pytest.approx(1e-3, rel=1e-3)
    assert opt.distance_to_mode() == pytest.approx(1e-3 / 10.0, rel=1e-3)
    assert opt.distance_to_mode(exact_curvature=False) == pytest.approx(d_gn, rel=1e-12)
    # The prior blocks, which the error-model cloud subtracts, are untouched.
    hs1, hp1, parts1 = opt._posterior_blocks()
    hs0, hp0, parts0 = opt._posterior_blocks(exact_curvature=False)
    assert np.array_equal(hp1, hp0) and parts1.keys() == parts0.keys()
    dH = hs1 - hs0
    assert np.allclose(dH, 99.0 * c_gn * np.outer(v, v), atol=1e-9 * np.abs(hs0).max())


def test_an_overstated_curvature_widens_the_width_but_not_the_step(monkeypatch):
    # The water dimer from one start (internal, seed 4, T-71): 64 by
    # Gauss-Newton against an exact 31 along a direction carrying 18% of an
    # angle's variance.
    opt = _optimizer()
    rows = _planted(opt, factor=0.25)
    v, c_gn = rows[0]["direction"], rows[0]["curvature_gauss_newton"]
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, **k: np.zeros(opt.coords.size))
    cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert float(v @ cov @ v) == pytest.approx(1.0 / (0.25 * c_gn + 1e-8), rel=1e-6)
    assert opt._step_curvature_correction(opt.coords) is None
    msgs = opt.gauss_newton_warnings()
    assert len(msgs) == 1 and "overstates the curvature" in msgs[0], msgs
    assert "the fit's step keeps Gauss-Newton's" in msgs[0]
    assert "Gauss-Newton's width is 2 times narrower" in msgs[0], msgs[0]


def test_far_from_the_mode_the_widths_stay_gauss_newton_s(monkeypatch):
    opt = _optimizer()
    rows = _planted(opt)
    v, c_gn = rows[0]["direction"], rows[0]["curvature_gauss_newton"]
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    # 50 Gauss-Newton widths out, 5 exact ones: past the 2-width limit either way.
    monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, **k: 50.0 * np.sqrt(c_gn) * v)
    corr = opt._exact_curvature_correction(opt.coords)
    assert corr["applied"] is False and corr["distance"] == pytest.approx(5.0, rel=1e-3)
    assert opt._applied_curvature(opt.coords) is None
    cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert float(v @ cov @ v) == pytest.approx(1.0 / (c_gn + 1e-8), rel=1e-6)
    assert opt.distance_to_mode() == pytest.approx(50.0, rel=1e-3)        # the larger reading
    msgs = opt.gauss_newton_warnings()
    assert len(msgs) == 1 and "widths from the mode" in msgs[0] and "are kept" in msgs[0], msgs


def test_the_bootstrap_gate_reads_the_corrected_distance(monkeypatch):
    import backend.uncertainty_v2.bootstrap as B
    opt = _optimizer()
    rows = _planted(opt)
    v, c_gn = rows[0]["direction"], rows[0]["curvature_gauss_newton"]
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    # 0.5 Gauss-Newton widths, over the 0.25 limit; 0.05 by the exact curvature.
    monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, **k: 0.5 * np.sqrt(c_gn) * v)
    assert B._distance(opt, opt.coords) == pytest.approx(0.05, rel=1e-3)
    opt._exact_curvature = False
    assert B._distance(opt, opt.coords) == pytest.approx(0.5, rel=1e-3)


# --- T-72 review follow-ups ----------------------------------------------------

def _plant_correction(monkeypatch, opt, factor):
    """A correction on the softest Gauss-Newton direction, at the mode."""
    rows = _planted(opt, factor=factor)
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, **k: np.zeros(opt.coords.size))
    return rows[0]["direction"], rows[0]["curvature_gauss_newton"]


@pytest.mark.parametrize("strategy", ["error-model", "rto"])
def test_the_bootstrap_target_leaves_out_the_curvature_no_refit_spreads_by(monkeypatch, strategy):
    # A refit moves by cov J^T dr, so the data cloud is cov J^T J cov: the
    # exact-curvature term K is in the reported cov but not in any cloud.
    from backend.internal.internal_fit import InternalCoordinateSet
    from backend.uncertainty_v2 import BootstrapEngine
    from backend.uncertainty_v2.models import BootstrapConfig
    from backend.uncertainty_v2.posterior import reported_covariance
    opt = _optimizer()
    _plant_correction(monkeypatch, opt, factor=100.0)
    x = opt.coords
    hs, hp, _ = opt._posterior_blocks(x, exact_curvature=False)
    K = opt._applied_curvature(x, (hs, hp))
    assert K is not None and np.abs(K).max() > 0                       # premise
    c_total, cov = reported_covariance(opt, x)
    want = c_total - cov @ ((hp + K) if strategy == "error-model" else K) @ cov
    cs = InternalCoordinateSet(x, opt.elems, opt._ic_use_dihedrals)
    Bq = np.asarray(cs.active_B_matrix(x), dtype=float)
    conv = np.array([1.0 if k == "bond" else np.degrees(1.0) for k in (c.kind for c in cs.active_coords())])
    with contextlib.redirect_stdout(io.StringIO()):
        ref = BootstrapEngine(opt, BootstrapConfig(n_samples=2, strategy=strategy)).analytic_reference()
    assert np.allclose(ref["target_std"], np.sqrt(np.clip(np.diag(Bq @ want @ Bq.T), 0, None)) * conv, rtol=1e-9)


def test_the_distance_cannot_read_nearer_when_the_exact_curvature_softens(monkeypatch):
    # Exact curvature 1% of Gauss-Newton's: the exact reading is ten times
    # the Gauss-Newton one. Beyond 2 exact widths the widths stay
    # Gauss-Newton's but the distance is the larger reading, so it rises
    # monotonically with the gradient (T-72 review: 0.21 / 2.1 read 0.21).
    opt = _optimizer()
    rows = _planted(opt, factor=0.01)
    v, c_gn = rows[0]["direction"], rows[0]["curvature_gauss_newton"]
    monkeypatch.setattr(opt, "gauss_newton_discrepancy", lambda *a, **k: [dict(r) for r in rows])
    seen = []
    for z_gn in (0.15, 0.19, 0.21, 0.5):
        monkeypatch.setattr(opt, "_posterior_gradient", lambda *a, _z=z_gn, **k: _z * np.sqrt(c_gn) * v)
        seen.append(opt.distance_to_mode())
    assert seen == sorted(seen), seen
    assert seen[2] == pytest.approx(2.1, rel=1e-3) and seen[3] == pytest.approx(5.0, rel=1e-3)
    cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert float(v @ cov @ v) == pytest.approx(1.0 / (c_gn + 1e-8), rel=1e-6)   # widths: Gauss-Newton's


@pytest.mark.parametrize("kw, says", [
    ({}, "as does the fit's step"),
    ({"objective_mode": "split"}, "the split objective's step keeps Gauss-Newton's"),
    ({"spectral_only": True}, "a spectral-only fit's step"),
])
def test_the_note_says_whether_the_step_takes_the_exact_curvature(monkeypatch, kw, says):
    opt = _optimizer(**kw)
    _plant_correction(monkeypatch, opt, factor=100.0)
    msgs = opt.gauss_newton_warnings()
    assert len(msgs) == 1 and says in msgs[0], msgs


def test_a_direction_the_exact_curvature_widens_is_reported_by_its_reported_share(monkeypatch):
    # Raise the relevance bar above the direction's Gauss-Newton share: once
    # the exact curvature widens it, its share of the reported variance clears
    # the bar, and the note is given (it was judged on Gauss-Newton's alone).
    opt = _optimizer()
    _plant_correction(monkeypatch, opt, factor=1e-4)
    monkeypatch.setattr(type(opt), "_GN_RELEVANT_SHARE", 0.999)
    msgs = opt.gauss_newton_warnings()
    assert len(msgs) == 1 and "overstates the curvature" in msgs[0], msgs
    assert "by Gauss-Newton)" in msgs[0]


def test_the_check_compares_against_gauss_newton_even_with_a_correction_in_force(monkeypatch):
    # The probe's Gauss-Newton curvatures are the Gauss-Newton block's own
    # eigenvalues, also while a correction is in force for the widths.
    opt = _optimizer()
    x = opt.coords
    hs, hp, _ = opt._posterior_blocks(x, exact_curvature=False)
    basis = opt.physical_basis(coords=x, n=hs.shape[0])
    w = np.linalg.eigvalsh(basis.T @ (hs + hp) @ basis)
    v = basis @ np.linalg.eigh(basis.T @ (hs + hp) @ basis)[1][:, 0]
    planted = {"correction": 1e6 * np.outer(v, v), "rows": [], "distance": 0.0, "applied": True}
    monkeypatch.setattr(opt, "_exact_curvature_correction", lambda *a, **k: planted)
    assert not np.allclose(opt._posterior_blocks(x)[0], hs)                  # premise: in force
    rows = opt.gauss_newton_discrepancy(coords=x)
    assert [r["curvature_gauss_newton"] for r in rows] == pytest.approx(list(w[:len(rows)]), rel=1e-9)
