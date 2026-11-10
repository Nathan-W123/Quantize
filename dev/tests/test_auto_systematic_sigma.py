"""Auto-derived systematic sigmas: available, verified, and off by default.

The B0-versus-Be model gap is the dominant error term in this engine, and
until now a user who quoted honest measurement precision got intervals that
omitted it, with only a printed warning to say so. Deriving it was tried
before and reverted because the propagation produced bond widths of 47
Angstrom and an angle width of 10975 degrees -- which turned out to be the
sigma-units mismatch fixed in test_systematic_sigma_units, not a problem with
deriving it.

With that fixed the derived term is sensible (0.0147 Angstrom, 1.70 degrees
on water) and it repairs a real under-coverage: the old angle interval of
0.0273 degrees was NINE TIMES too narrow to reach the 0.246-degree angle
error the engine itself makes.

It is still OFF by default, and these tests pin why. See
test_the_interval_is_anti_monotonic_in_sigma.
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

import dev.analytic_water_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = [835840.29, 435351.72, 278138.70]
B0_D2O = [462324.0, 218038.0, 145258.0]
R_OH = 0.95785
#: Water angle error the engine itself makes, measured against the accepted
#: structure earlier in development: -0.246 deg at base, +0.182 corrected.
KNOWN_ANGLE_ERROR_DEG = 0.246


def _coords(r=R_OH, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(auto, sigma_mhz=0.2, **kw):
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [sigma_mhz] * 3}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),
                            ("D2-16O", MASSES_D2O, B0_D2O))]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, auto_systematic_sigma=auto, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
        unc = opt.geometry_uncertainty()
    return opt, unc


def _widths(unc):
    return (max(v for k, v in unc.items() if k.count("-") == 1),
            max(v for k, v in unc.items() if k.count("-") == 2))


def test_off_by_default():
    """A default that changes every reported interval needs a decision."""
    opt, _unc = _fit(auto=False)
    assert opt.auto_systematic_sigma is False
    isos = [{"name": "H2-16O", "masses": MASSES_H2O,
             "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}]
    with contextlib.redirect_stdout(io.StringIO()):
        default = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=3, quantum_prior_sigma_ang=0.02)
    assert default.auto_systematic_sigma is False
    assert default.spectral.isotopologues[0].get(
        "sigma_systematic_constants") is None


def test_opt_in_derives_a_systematic_without_touching_the_quoted_sigma():
    opt, _unc = _fit(auto=True)
    iso = opt.spectral.isotopologues[0]
    sysg = np.asarray(iso["sigma_systematic_constants"], dtype=float)
    raw = np.asarray(iso["sigma_constants"], dtype=float)
    assert np.all(sysg > 0.0)
    # The user's measurement precision is left exactly as quoted.
    assert np.allclose(raw, 0.2)
    # And the derived gap is a sizeable fraction of each constant, not a
    # rescaling of the quoted sigma.
    frac = sysg / np.asarray(B0_H2O, dtype=float)
    assert np.all(frac > 0.005) and np.all(frac < 0.20), frac


def test_derived_widths_are_physically_sensible():
    """47 Angstrom and 10975 degrees caused the original revert."""
    _opt, unc = _fit(auto=True)
    bond, angle = _widths(unc)
    assert bond < R_OH, f"bond width {bond:.4f} Ang exceeds the bond"
    assert angle < 180.0, f"angle width {angle:.1f} deg exceeds a half turn"
    assert bond > 0.0 and angle > 0.0


def test_it_repairs_the_angle_under_coverage():
    """The point of the change: reach the error the engine actually makes."""
    _o0, unc_off = _fit(auto=False)
    _o1, unc_on = _fit(auto=True)
    _b_off, a_off = _widths(unc_off)
    _b_on, a_on = _widths(unc_on)
    assert a_off < KNOWN_ANGLE_ERROR_DEG, (
        "premise broken: the old interval already covered the known error")
    assert a_on > KNOWN_ANGLE_ERROR_DEG, (
        f"derived interval {a_on:.3f} deg still does not reach the "
        f"{KNOWN_ANGLE_ERROR_DEG} deg error the engine makes")


def test_the_bond_is_barely_affected():
    """A coherent shift in the constants moves the weakly-determined
    coordinate. The bond was already conservative and should stay put."""
    _o0, unc_off = _fit(auto=False)
    _o1, unc_on = _fit(auto=True)
    b_off, a_off = _widths(unc_off)
    b_on, a_on = _widths(unc_on)
    assert (b_on / b_off) < 2.0
    assert (a_on / a_off) > 10.0


def test_fitted_geometry_does_not_move():
    """This is a sigma on the REPORT. If it moved the answer it would be a
    different change entirely, and would need benchmark re-validation."""
    o_off, _ = _fit(auto=False)
    o_on, _ = _fit(auto=True)
    assert np.array_equal(o_off.coords, o_on.coords)


def test_explicit_systematic_is_not_overwritten():
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.01 * np.asarray(b))}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),)]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=3, quantum_prior_sigma_ang=0.02,
            auto_systematic_sigma=True)
    got = np.asarray(opt.spectral.isotopologues[0]["sigma_systematic_constants"])
    assert np.allclose(got, 0.01 * np.asarray(B0_H2O))


def test_the_interval_is_anti_monotonic_in_sigma():
    """WHY THIS IS NOT THE DEFAULT, pinned so it cannot be forgotten.

    With sigma_floor_rel off, so the quoted sigma reaches the weighting,
    admitting 1000x more measurement uncertainty NARROWS the reported bond
    width: 0.0455 -> 0.0180 Angstrom. The mechanism is real -- tighter data
    is weighted more heavily, so a coherent model shift moves the optimum
    further -- but an uncertainty that shrinks when you admit more error
    cannot be published.

    The fix is to let the systematic weight the fit as well as the report, so
    the two sigmas cannot disagree by 1e5. When that lands, this test should
    start failing, and that is the signal to make the derivation default.
    """
    _o_t, unc_tight = _fit(auto=True, sigma_mhz=2.0, sigma_floor_rel=0.0)
    _o_l, unc_loose = _fit(auto=True, sigma_mhz=2000.0, sigma_floor_rel=0.0)
    b_tight, _a1 = _widths(unc_tight)
    b_loose, _a2 = _widths(unc_loose)
    assert b_loose < b_tight, (
        "the anti-monotonicity is gone -- if the systematic now weights the "
        "fit, make auto_systematic_sigma default True and delete this test")


def test_with_the_floor_on_it_is_monotonic():
    """The default floor masks the problem, which is exactly why relying on
    it would be the wrong reason to ship this on."""
    widths = []
    for sig in (2.0, 20.0, 200.0, 2000.0):
        _o, unc = _fit(auto=True, sigma_mhz=sig)
        widths.append(_widths(unc)[0])
    for lo, hi in zip(widths, widths[1:]):
        assert hi >= lo - 1e-9, widths
