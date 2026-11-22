"""The harmonic distortion prediction does not turn with the molecule.

A Cartesian Hessian belongs to the orientation it was computed in. The
distortion rows predicted DJ, DJK and DK from the fit's Hessian at the trial
geometry in whatever orientation the trial geometry had, so the predicted
constants changed when the molecule merely turned -- formyl fluoride at
HF/6-31G: up to 0.5% for a 0.01 rad rotation, while A, B and C moved 1e-15.
A fit with distortion rows then turned the molecule to fit them, a direction
nothing else resists, and a straight-line step along a rotation also
stretches it, so steps over ~2 mA were refused and the fits crawled (T-32,
T-35). The engine now keeps the geometry the Hessian was computed at and
turns each trial geometry onto it before predicting (decisions D-88).
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

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.spectral.centrifugal_distortion import CD_NAMES, compute_cd_constants  # noqa: E402
from backend.spectral.spectral import CD_FIT_KEYS  # noqa: E402
from test_cd_rows_sign import _water_cd, _x0  # noqa: E402

_IDX = [CD_NAMES.index(k) for k in CD_FIT_KEYS]


def _rot(axis, th):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def _turned(x, axis, th, shift=(0.3, -0.2, 0.1)):
    c = x.mean(axis=0)
    return (x - c) @ _rot(axis, th).T + c + np.asarray(shift)


@pytest.fixture(scope="module")
def fit():
    opt = _water_cd()
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


def _pred(sp, iso, x):
    return sp._predict_cd_for_iso(iso, x).vector()[_IDX]


def test_the_fit_hands_over_the_hessians_geometry(fit):
    sp = fit.spectral
    assert sp._hess_coords_for_cd is not None
    np.testing.assert_array_equal(sp._hess_coords_for_cd, fit._hessian_ref_coords)


def test_the_prediction_does_not_turn_with_the_molecule(fit):
    sp, iso = fit.spectral, fit.spectral.isotopologues[0]
    x = np.asarray(fit.coords, float)
    p0 = _pred(sp, iso, x)
    for th in (0.05, 0.5, 2.0):
        for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]):
            got = _pred(sp, iso, _turned(x, ax, th))
            np.testing.assert_allclose(got, p0, rtol=1e-8, err_msg=f"{th} rad about {ax}")
    # Premise: without the Hessian's geometry the same turn does move it.
    keep = sp._hess_coords_for_cd
    try:
        sp._hess_coords_for_cd = None
        moved = max(np.abs(_pred(sp, iso, _turned(x, [0, 0, 1], 0.05)) / p0 - 1).max(),
                    np.abs(_pred(sp, iso, _turned(x, [1, 2, 3], 0.05)) / p0 - 1).max())
    finally:
        sp._hess_coords_for_cd = keep
    assert moved > 1e-4, moved


def _rotation_pull(sp, iso, x):
    c = x.mean(axis=0)
    sp._cd_jacobian_cache.clear()
    J = sp._cd_jacobian(x, iso)
    out = []
    for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        d = np.cross(ax, x - c).ravel()
        out.append(np.abs(J @ (d / np.linalg.norm(d))).max() / np.abs(J).max())
    return max(out)


def test_the_rows_do_not_pull_along_a_rigid_rotation(fit):
    # At a turned geometry: on this planar fixture the pull along a rotation
    # vanishes to first order at the Hessian's own orientation (1e-12 with or
    # without the frame), so the defect shows only once the molecule has
    # turned -- as it does during a fit.
    sp, iso = fit.spectral, fit.spectral.isotopologues[0]
    y = _turned(np.asarray(fit.coords, float), [1, 2, 3], 0.3)
    # Measured: 2.6e-6 (finite-difference noise) against 3.8e-2 without the
    # frame; 0.1 and 1 rad turns: 1e-6 / 3e-6 against 1.3e-2 / 1.2e-1.
    assert _rotation_pull(sp, iso, y) <= 2e-5
    keep = sp._hess_coords_for_cd
    try:
        sp._hess_coords_for_cd = None
        assert _rotation_pull(sp, iso, y) > 1e-3         # premise
    finally:
        sp._hess_coords_for_cd = keep
        sp._cd_jacobian_cache.clear()


def test_at_the_hessians_own_geometry_nothing_changes(fit):
    sp, iso = fit.spectral, fit.spectral.isotopologues[0]
    ref = np.asarray(sp._hess_coords_for_cd, float)
    want = compute_cd_constants(sp._hess_bohr_for_cd, ref, iso["masses"],
                                reduction=iso["cd_reduction"], with_residual=False,
                                min_freq_cm=sp._cd_min_freq_cm, fd_delta=sp._cd_fd_delta,
                                sigma_fraction=sp._cd_sigma_fraction).vector()[_IDX]
    np.testing.assert_allclose(_pred(sp, iso, ref), want, rtol=1e-12)


def test_without_a_stated_geometry_the_trial_frame_is_used(fit):
    sp, iso = fit.spectral, fit.spectral.isotopologues[0]
    keep = (sp._hess_bohr_for_cd, sp._hess_coords_for_cd)
    try:
        sp.set_hessian_for_cd(keep[0])
        assert sp._hess_coords_for_cd is None
        y = _turned(np.asarray(fit.coords, float), [0, 0, 1], 0.3)
        want = compute_cd_constants(keep[0], y, iso["masses"], reduction=iso["cd_reduction"],
                                    with_residual=False, min_freq_cm=sp._cd_min_freq_cm,
                                    fd_delta=sp._cd_fd_delta,
                                    sigma_fraction=sp._cd_sigma_fraction).vector()[_IDX]
        np.testing.assert_array_equal(_pred(sp, iso, y), want)
    finally:
        sp.set_hessian_for_cd(keep[0], coords=keep[1])


def test_the_report_table_does_not_turn_either(fit):
    # _apply_harmonic_cd_corrections builds the reported harmonic table at the
    # fit's geometry: with the molecule turned (the Hessian unchanged) it
    # must report the same constants.
    import copy
    opt = copy.deepcopy(fit)
    def table():
        with contextlib.redirect_stdout(io.StringIO()):
            opt._apply_harmonic_cd_corrections()
        iso = opt.spectral.isotopologues[0]
        pred = iso.get("pred_cd") or iso["rovib_correction"].pred_cd
        return pred.vector()[_IDX]
    t0 = table()
    opt.coords = _turned(np.asarray(opt.coords, float), [1, 2, 3], 0.4)
    np.testing.assert_allclose(table(), t0, rtol=1e-8)


def test_every_hand_off_carries_the_geometry(fit):
    # The fit's refresh path is covered above; the gradient-only update and
    # the harmonic-table path hand the Hessian over too.
    import copy
    opt = copy.deepcopy(fit)
    opt.spectral._hess_coords_for_cd = None
    opt.hess_recalc_every = 10 ** 6
    opt._since_hessian = 0
    with contextlib.redirect_stdout(io.StringIO()):
        opt._update_orca()                          # gradient only
    assert opt._since_hessian == 1, "premise: no Hessian was computed"
    np.testing.assert_array_equal(opt.spectral._hess_coords_for_cd, opt._hessian_ref_coords)
    harm = _water_cd(harmonic_cd_from_hessian=True)
    with contextlib.redirect_stdout(io.StringIO()):
        harm.run()
    assert harm._harmonic_cd_from_hessian, "premise"
    assert harm.spectral._hess_coords_for_cd is not None
    np.testing.assert_array_equal(harm.spectral._hess_coords_for_cd, harm._hessian_ref_coords)


def test_with_a_target_prior_the_hessian_belongs_to_the_aligned_target():
    # With a target-centred prior the Hessian is computed at (or rotated into)
    # the target aligned onto the fit, not at the fit's own geometry, and the
    # rows must be handed that geometry. A target in another frame and with
    # another shape makes the two differ.
    target = _turned(_x0(r=0.975, deg=103.0), [1, 2, 3], 1.1)
    opt = _water_cd(prior_target_coords=target, coordinate_mode="cartesian")
    assert opt._target_prior_active(), "premise: the target prior is on"
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    ref = np.asarray(opt._hessian_ref_coords, float)
    assert np.abs(ref - np.asarray(opt.coords, float)).max() > 1e-3, "premise"
    np.testing.assert_array_equal(opt.spectral._hess_coords_for_cd, ref)
    sp, iso = opt.spectral, opt.spectral.isotopologues[0]
    x = np.asarray(opt.coords, float)
    np.testing.assert_allclose(_pred(sp, iso, _turned(x, [0, 1, 1], 0.7)), _pred(sp, iso, x), rtol=1e-8)

