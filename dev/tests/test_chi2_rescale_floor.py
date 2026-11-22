"""chi2 rescaling stops when the model-error floor makes it a no-op.

A downward scale cannot go below each row's model-error part (see
SpectralEngine.scale_sigma). Once every row is there, scaling again changes
nothing, and refitting the same objective is a no-op -- which run() used to do,
and announce as "refitting", up to chi2_rescale_max_passes times. These pin
the count of refits in both regimes, so the early stop cannot swallow a real
tightening.
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
from backend.quantize import MolecularOptimizer  # noqa: E402
from test_bootstrap_targets import (  # noqa: E402
    B0_D2O, B0_H2O, MASSES_D2O, MASSES_H2O, _coords)


def _isos(sigma, systematic=None):
    out = []
    for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O), ("D2-16O", MASSES_D2O, B0_D2O)):
        iso = {"name": n, "masses": m, "obs_constants": list(b), "sigma_constants": [sigma] * 3}
        if systematic is not None:
            iso["sigma_systematic_constants"] = [systematic] * 3
            iso["sigma_constants_include_systematic"] = True
        out.append(iso)
    return out


def _run(isos):
    opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                             quantum_backend="analytic_water", use_autoconfig=False,
                             max_iter=60, chi2_rescale=True, chi2_rescale_max_passes=3)
    calls = {"n": 0}
    inner = opt._optimise_once

    def counted():
        calls["n"] += 1
        return inner()
    opt._optimise_once = counted
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt.run()
    return opt, calls["n"], buf.getvalue()


def test_a_floored_rescale_stops_after_the_pass_that_changed_something():
    # Mostly model error: hypot(1e4, 0.2) MHz, large enough that chi2/nu is
    # well below 1/threshold (this fixture gives 202 at 100 MHz, so ~0.02 here)
    # and the rescale is downward. The first pass strips the small measurement
    # part (a real, if tiny, change, so it refits once); the second finds every
    # row at its floor and stops.
    total = float(np.hypot(1e4, 0.2))
    opt, n, log = _run(_isos(total, systematic=1e4))
    assert "tightening" in log and "widening" not in log, log   # premise
    assert n == 2, (n, log)                       # the fit, plus one refit
    assert log.count("not refitting") == 1
    assert log.count("and refitting") == 1
    for iso in opt.spectral.isotopologues:        # and the sigmas sit at the floor
        np.testing.assert_allclose(np.asarray(iso["sigma_constants"], float), 1e4, rtol=1e-12)


def test_a_real_tightening_still_refits_every_pass():
    # Measurement only, same size: no floor, so the pass tightens (chi2/nu
    # 0.158 -> about 2.5x here) and must refit -- the early stop must not fire,
    # and the sigmas must actually come down.
    opt, n, log = _run(_isos(1e4))
    assert "tightening" in log and "widening" not in log, log   # premise
    assert "not refitting" not in log
    assert log.count("and refitting") >= 1 and n == 1 + log.count("and refitting"), (n, log)
    for iso in opt.spectral.isotopologues:
        assert np.all(np.asarray(iso["sigma_constants"], float) < 0.5e4)


# -- The checks are kept (N19, T-29) -----------------------------------------
#
# run() keeps one record per rescale check, so a fit near a switch point (a
# reduced chi-square near 4 or 1/4) can be seen without instrumenting it. The
# record's dof is the one reduced_chi_square() computed; the dof gate reads the
# same number instead of computing the hat-matrix trace a second time.

def _checks(opt):
    return [(c["outcome"], c["chi2_nu"], c["dof"]) for c in opt.chi2_rescale_checks]


def test_each_check_is_kept_with_what_it_did():
    # Measured: floored 0.1579 -> refit (factor 0.397) -> 0.1579 at floor;
    # measurement-only 0.1579 -> refit -> 1.0426 in band.
    floored, _, _ = _run(_isos(float(np.hypot(1e4, 0.2)), systematic=1e4))
    assert [c[0] for c in _checks(floored)] == ["refit", "at floor"]
    assert floored.chi2_rescale_checks[0]["factor"] == pytest.approx(np.sqrt(0.1579), rel=1e-3)
    tight, _, _ = _run(_isos(1e4))
    assert [c[0] for c in _checks(tight)] == ["refit", "in band"]
    # The last check was made at the answer: recomputing there gives it back.
    last = tight.chi2_rescale_checks[-1]
    assert last["chi2_nu"] == tight.reduced_chi_square()
    assert last["dof"] == tight._chi2_dof()


def test_an_in_band_fit_records_one_check_and_an_off_rescale_none():
    one, n, _ = _run(_isos(1e4)[:1])          # 0.4322 on 1.27 dof: in band
    assert n == 1 and [c[0] for c in _checks(one)] == ["in band"]
    assert 0.25 <= one.chi2_rescale_checks[0]["chi2_nu"] <= 4.0
    one.chi2_rescale = False                  # a later run keeps only its own checks
    with contextlib.redirect_stdout(io.StringIO()):
        one.run()
    assert one.chi2_rescale_checks == []
    off = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=_isos(1e4),
                             quantum_backend="analytic_water", use_autoconfig=False,
                             max_iter=60, chi2_rescale=False)
    with contextlib.redirect_stdout(io.StringIO()):
        off.run()
    assert off.chi2_rescale_checks == []


def test_too_few_dof_is_recorded_and_still_stops_the_rescale(monkeypatch):
    # The gate now reads the dof reduced_chi_square() kept. Force it below
    # CHI2_RESCALE_MIN_DOF (1) at a chi-square far out of band -- 5 MHz on
    # this fixture is 202 at dof 4, so ~900 at 0.9 -- and the fit must stop
    # without a refit, the record saying why.
    import backend.quantize as Q
    monkeypatch.setattr(Q.MolecularOptimizer, "_chi2_dof", lambda self: 0.9)
    opt, n, log = _run(_isos(5.0))
    first = opt.chi2_rescale_checks[0]
    assert first["chi2_nu"] > 4.0, first                  # premise: out of band
    assert n == 1 and "refitting" not in log
    assert _checks(opt) == [("too few dof", first["chi2_nu"], 0.9)]


def test_a_replaced_reduced_chi_square_still_gets_its_dof(monkeypatch):
    # A wrapper that does not set the kept dof (a test double, an instrument)
    # must not break the gate: run() computes the dof itself then.
    opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=_isos(1e4)[:1],
                             quantum_backend="analytic_water", use_autoconfig=False,
                             max_iter=60, chi2_rescale=True)
    monkeypatch.setattr(opt, "reduced_chi_square", lambda: 0.4322)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert opt.chi2_rescale_checks[0]["dof"] == pytest.approx(opt._chi2_dof())
