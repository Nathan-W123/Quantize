"""Weighting the fit with the model error: available, off, and here is why.

F14 is the inconsistency the note in `MolecularOptimizer.__init__` flagged
from the start -- "weighting data at 1 kHz while the model is wrong at 100
MHz is itself inconsistent". `systematic_in_weighting` resolves it by
flooring the weighting sigma at the measured per-species model gap, which is
what `sigma_floor_rel` approximates with a flat fraction.

It does fix the statistics. It also costs accuracy, and accuracy is the point
of the engine, so it is OFF.

Measured on water against the accepted structure, from five starting
geometries:

  start            weighting OFF          weighting ON
                   dr (mA)   dang         dr (mA)   dang
  accepted           2.490  0.6488          2.360  1.5908
  +3% scaled         1.413  0.6883          2.933  1.5093
  -3% scaled         0.992  0.6105          3.028  1.6281
  r+0.05 ang-5       2.367  0.6659          4.291  1.2766
  r-0.05 ang+5       2.472  0.6789          3.906  1.6067

Worse on the angle in 5 of 5 starts, on the bond in 4 of 5, and more
start-dependent (bond spread 2.4-4.3 mA against 1.0-2.5).

MECHANISM. Spectral data alone, no prior, gives dr +1.435 mA and dang
+0.690 -- essentially the weighting-OFF answer. So this fit is
data-determined, and the data is good. Flooring sigma at the model gap
down-weights it by about 1e5 and hands the answer to the quantum prior,
which is less accurate here.

WHAT THAT MEANS, and it is worth stating plainly: this engine's accuracy
depends on the inconsistency. It gets good geometries precisely by trusting
the constants far beyond what the model error justifies. Hence the awkward
position its uncertainties are in -- the point estimate is better than the
model deserves, so an honest interval is wider than the error actually made.
Resolving that is a research question about the B0-versus-Be correction, not
a weighting switch.
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
R_ACCEPTED = 0.95785
ANGLE_ACCEPTED = 104.508


def _geom(r, deg):
    t = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(t / 2), r * np.cos(t / 2), 0.0],
        [-r * np.sin(t / 2), r * np.cos(t / 2), 0.0],
    ])


def _fit(start, weigh, **kw):
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [0.2] * 3}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),
                            ("D2-16O", MASSES_D2O, B0_D2O))]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=start, isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=40,
        quantum_prior_sigma_ang=0.02, auto_systematic_sigma=True,
        systematic_in_weighting=weigh, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    r = float(np.linalg.norm(opt.coords[1] - opt.coords[0]))
    v1 = (opt.coords[1] - opt.coords[0]) / r
    v2 = (opt.coords[2] - opt.coords[0]) / np.linalg.norm(opt.coords[2] - opt.coords[0])
    ang = float(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))
    return opt, r, ang


STARTS = [
    _geom(R_ACCEPTED, ANGLE_ACCEPTED),
    _geom(R_ACCEPTED, ANGLE_ACCEPTED) * 1.03,
    _geom(R_ACCEPTED, ANGLE_ACCEPTED) * 0.97,
    _geom(R_ACCEPTED + 0.05, ANGLE_ACCEPTED - 5),
    _geom(R_ACCEPTED - 0.05, ANGLE_ACCEPTED + 5),
]


def test_off_by_default():
    isos = [{"name": "H2-16O", "masses": MASSES_H2O,
             "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_geom(R_ACCEPTED, ANGLE_ACCEPTED),
            isotopologues=isos, quantum_backend="analytic_water",
            use_autoconfig=False, max_iter=3, quantum_prior_sigma_ang=0.02)
    assert opt.systematic_in_weighting is False
    assert opt.spectral.systematic_in_weighting is False


def test_it_does_reach_the_weighting_when_on():
    """Otherwise the rest of this file measures nothing."""
    opt_on, _r, _a = _fit(STARTS[1], True)
    iso = opt_on.spectral.isotopologues[0]
    eff = np.asarray(opt_on.spectral._effective_sigma(
        iso["sigma_constants"], iso=iso), dtype=float)
    sysg = np.asarray(iso["sigma_systematic_constants"], dtype=float)
    assert np.allclose(eff, np.abs(sysg)), (eff, sysg)


def test_it_does_fix_the_statistics():
    """Residuals come back into the range where Gauss-Newton is valid."""
    opt_off, _r0, _a0 = _fit(STARTS[1], False)
    opt_on, _r1, _a1 = _fit(STARTS[1], True)

    def rms(opt):
        _J, rw = opt.spectral.stacked(opt.coords, want_jacobian=False)
        rw = np.asarray(rw, dtype=float)
        return float(np.sqrt(np.sum(rw ** 2) / rw.size))

    assert rms(opt_off) > 5.0, "premise broken: residuals were already small"
    assert rms(opt_on) < 2.0, f"residuals still {rms(opt_on):.2f} sigma"


def test_the_reported_width_stops_tracking_the_quoted_sigma():
    """A consequence worth knowing: with the model gap as the floor, better
    measurements no longer narrow the interval. Honest -- the model error
    dominates -- but it means the engine cannot reward better data until the
    B0-versus-Be correction is applied."""
    widths = []
    for sigma in (2.0, 2000.0):
        isos = [{"name": n, "masses": m, "obs_constants": list(b),
                 "sigma_constants": [sigma] * 3}
                for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),
                                ("D2-16O", MASSES_D2O, B0_D2O))]
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=STARTS[1], isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=40, quantum_prior_sigma_ang=0.02,
            auto_systematic_sigma=True, systematic_in_weighting=True,
            sigma_floor_rel=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            opt.run()
            unc = opt.geometry_uncertainty()
        widths.append(max(v for k, v in unc.items() if k.count("-") == 1))
    assert widths[0] == pytest.approx(widths[1], rel=1e-9), widths


def test_but_it_costs_accuracy_so_it_stays_off():
    """THE REASON THIS IS NOT ON. Pinned, so it is re-measured not re-argued.

    If a change to the weighting scheme makes this test fail, the trade has
    moved and the default should be reconsidered -- that is the point of
    asserting it rather than only writing it down.
    """
    worse_angle = 0
    worse_bond = 0
    for start in STARTS:
        _o0, r0, a0 = _fit(start, False)
        _o1, r1, a1 = _fit(start, True)
        if abs(a1 - ANGLE_ACCEPTED) > abs(a0 - ANGLE_ACCEPTED):
            worse_angle += 1
        if abs(r1 - R_ACCEPTED) > abs(r0 - R_ACCEPTED):
            worse_bond += 1
    assert worse_angle == len(STARTS), (
        f"weighting the fit with the model error is no longer worse on the "
        f"angle ({worse_angle}/{len(STARTS)}) -- re-evaluate the default")
    assert worse_bond >= len(STARTS) - 1, worse_bond


def test_the_fit_is_data_determined_which_is_why_down_weighting_hurts():
    """The mechanism, pinned: the prior is not where the good answer is."""
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [0.2] * 3}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),
                            ("D2-16O", MASSES_D2O, B0_D2O))]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=STARTS[1], isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=40,
        quantum_prior_sigma_ang=0.02, spectral_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    r_data = float(np.linalg.norm(opt.coords[1] - opt.coords[0]))
    _o, r_off, _a = _fit(STARTS[1], False)
    # Data alone and the full fit agree, so the data is carrying the answer.
    assert abs(r_data - r_off) < 1e-4, (r_data, r_off)
