"""Centrifugal-distortion rows: their sign, the default path, the bootstrap.

Measured before (water on the analytic surface, H2O + D2O, observed
distortion constants 5% off the model's own prediction, 2% sigma, CD
fitting on):

* With the default sanitizer the cd_observed blocks were dropped, so the
  fit had no distortion rows at all and fit_cd_constants did nothing --
  silently, since the "no cd_observed" warning reads the raw isotopologues.
* With the sanitizer off the rows were stacked as predicted minus observed
  against J = d(predicted)/dx, so -J^T r was the gradient of -1/2 |r_cd|^2
  (cosine -1.0000 against finite differences): the fit was pushed away from
  the observed constants (weighted residual 7.25 -> 7.71), stopped 1.29
  widths from its own optimum, and 35 of 40 error-model refits were refused.
* The error-model and rto draws left the observed distortion constants
  alone while the analytic target counted their precision.

After: rows on the default path, cosine +1.0000, the fit 7e-5 widths from
its optimum (residual 7.25 -> 7.07), 40 of 40 refits usable.
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

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.spectral.centrifugal_distortion import CD_NAMES, compute_cd_constants  # noqa: E402
from backend.uncertainty_v2.bootstrap import BootstrapEngine  # noqa: E402
from backend.uncertainty_v2.models import BootstrapConfig  # noqa: E402
from backend.uncertainty_v2.resampling import resample_cd_observed  # noqa: E402

M_O, M_H, M_D = 15.9949146196, 1.00782503207, 2.0141017778
SPEC = (("H2-16O", [M_O, M_H, M_H], [835840.29, 435351.72, 278138.70]),
        ("D2-16O", [M_O, M_D, M_D], [462324.0, 218038.0, 145258.0]))


def _x0(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([[0.0, 0.0, 0.0], [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
                     [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0]])


def _isos(cd=None):
    out = []
    for name, m, b in SPEC:
        d = {"name": name, "masses": m, "obs_constants": list(b), "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.024 * np.asarray(b))}
        if cd:
            d["cd_observed"] = cd[name]
            d["cd_sigma"] = {k: 0.02 * abs(v) for k, v in cd[name].items()}
        out.append(d)
    return out


def _make(cd, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return MolecularOptimizer(
            elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(cd),
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=60,
            auto_systematic_sigma=False, fit_cd_constants=bool(cd),
            cd_weight=1.0 if cd else 0.0, **kw)


@pytest.fixture(scope="module")
def observed_cd():
    base = _make(None)
    with contextlib.redirect_stdout(io.StringIO()):
        base.run()
    hb = np.asarray(base.quantum._hessian_bohr, float)
    cd = {}
    for name, m, _b in SPEC:
        pred = compute_cd_constants(hb, base.coords, m)
        cd[name] = {k: 1.05 * float(v) for k, v in pred.as_dict().items()
                    if k in ("DJ", "DJK", "DK", "delta_J", "delta_K")}
    return cd


@pytest.fixture(scope="module")
def cd_fit(observed_cd):
    opt = _make(observed_cd)                      # the default sanitizer
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


def _n_rot(opt):
    return sum(len(i["obs_constants"]) for i in opt.spectral.isotopologues)


def test_the_default_path_keeps_the_observed_distortion_constants(cd_fit):
    J, _r = cd_fit.spectral.stacked(cd_fit.coords)
    assert J.shape[0] - _n_rot(cd_fit) == 10, "five constants on each of two isotopologues"


def test_the_distortion_rows_point_the_gradient_downhill(cd_fit):
    """-J^T r must be the gradient of 1/2 |r|^2 on the distortion rows as on
    the rotational ones."""
    sp, x, n = cd_fit.spectral, np.asarray(cd_fit.coords, float), _n_rot(cd_fit)
    J, r = sp.stacked(x)
    g = -(J[n:].T @ r[n:])

    def half(xx):
        _, q = sp.stacked(xx.reshape(-1, 3), want_jacobian=False)
        return 0.5 * float(np.sum(q[n:] ** 2))

    fd = np.zeros(x.size)
    for k in range(x.size):
        a, b = x.ravel().copy(), x.ravel().copy()
        a[k] += 1e-5
        b[k] -= 1e-5
        fd[k] = (half(a) - half(b)) / 2e-5
    cos = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd)))
    assert cos > 0.999


def test_a_fit_with_distortion_rows_reaches_its_own_optimum(cd_fit):
    n = _n_rot(cd_fit)
    _J, r_end = cd_fit.spectral.stacked(cd_fit.coords, want_jacobian=False)
    _J, r_start = cd_fit.spectral.stacked(_x0(), want_jacobian=False)
    assert cd_fit.distance_to_mode() < 0.01
    assert np.linalg.norm(r_end[n:]) < np.linalg.norm(r_start[n:])


def test_the_distortion_draws_carry_the_rows_weights(cd_fit):
    """Each drawn constant moves its weighted residual by a unit normal."""
    sp = cd_fit.spectral
    sig = sp.cd_row_sigmas(cd_fit.coords)
    assert all(sig), "premise: both isotopologues have rows"
    rng = np.random.default_rng(1)
    z = []
    for _ in range(4000):
        p = resample_cd_observed(sp.isotopologues, sig, rng)
        z.append([(p[i]["cd_observed"][k] - sp.isotopologues[i]["cd_observed"][k]) / sig[i][k]
                  for i in range(len(p)) for k in CD_NAMES if k in sig[i]])
    z = np.array(z)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=0.05)
    # and the weight is what stacked applies: sqrt(cd_weight) / sigma_cd
    keys0 = [k for k in CD_NAMES if k in sig[0]]
    sig_cd = np.array([max(0.02 * abs(sp.isotopologues[0]["cd_observed"][k]), 0.01) for k in keys0])
    np.testing.assert_allclose([sig[0][k] for k in keys0], sig_cd / np.sqrt(sp.cd_weight))


def test_an_error_model_bootstrap_with_distortion_rows_is_usable(cd_fit, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        s = BootstrapEngine(cd_fit, BootstrapConfig(n_samples=12, max_workers=2, seed=7,
                                                    strategy="error-model",
                                                    persist_samples=False)).run(str(tmp_path))
    assert s.usable_samples == 12
    assert all(r.mode_distance < 0.01 for r in s.sample_results)
