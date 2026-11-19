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
