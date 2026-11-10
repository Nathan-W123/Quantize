"""`stacked(want_jacobian=False)` must return the same residuals, faster.

The Jacobian is finite-differenced at roughly 6N rotational-constant
evaluations per isotopologue; the residuals cost one. A caller that only
needs residuals -- gauss_newton_discrepancy probes chi-squared curvature at
displaced geometries -- pays about 6N times too much by taking the default.

The risk in adding the flag is that the residual path drifts from the full
one, since the weighting it shares (effective sigma, component weights,
robust weights, the centrifugal-distortion block) is exactly what must not be
reimplemented. These tests pin bit-identity rather than closeness.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
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


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


@pytest.fixture(scope="module", params=[1, 2], ids=["one_species", "two_species"])
def fit(request):
    isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
             "sigma_constants": [0.2] * 3}]
    if request.param > 1:
        isos.append({"name": "D2-16O", "masses": MASSES_D2O,
                     "obs_constants": B0_D2O, "sigma_constants": [0.2] * 3})
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


def test_residuals_are_bit_identical_at_the_optimum(fit):
    _J, r_full = fit.spectral.stacked(fit.coords)
    J_none, r_cheap = fit.spectral.stacked(fit.coords, want_jacobian=False)
    assert J_none is None
    assert np.array_equal(r_full, r_cheap)


def test_residuals_are_bit_identical_away_from_it(fit):
    """Displaced geometries are the ones the curvature probe actually uses."""
    rng = np.random.default_rng(12345)
    for _ in range(25):
        x = fit.coords + rng.normal(0.0, 0.02, fit.coords.shape)
        _J, r_full = fit.spectral.stacked(x)
        _n, r_cheap = fit.spectral.stacked(x, want_jacobian=False)
        assert np.array_equal(r_full, r_cheap), "residual paths have drifted"


def test_default_still_returns_a_jacobian(fit):
    """Every existing caller takes the default and must be unaffected."""
    J, r = fit.spectral.stacked(fit.coords)
    assert J is not None
    assert J.shape[0] == r.shape[0]
    assert J.shape[1] == fit.coords.size


def test_residual_only_is_actually_cheaper(fit):
    """The flag exists for speed; if it is not faster it has no purpose."""
    def timeit(kw, n=60):
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fit.spectral.stacked(fit.coords, **kw)
            best = min(best, time.perf_counter() - t0)
        return best
    with_j = timeit({})
    without = timeit({"want_jacobian": False})
    # Measured ~6x on water. Assert a weak 2x so this does not become a
    # flaky timing test on a loaded machine, while still failing if the flag
    # silently stops skipping the Jacobian.
    assert without < with_j / 2.0, (
        f"residual-only {without:.6f}s vs full {with_j:.6f}s: not skipping "
        f"the Jacobian")


def test_the_curvature_probe_gets_the_same_answer_either_way(fit):
    """Switching the probe to the cheap path must not move its numbers."""
    rows = fit.gauss_newton_discrepancy()
    assert rows

    # Recompute the softest direction's exact curvature using the FULL path.
    J, _ = fit.spectral.stacked(fit.coords)
    H_spec = J.T @ J
    H_quant = np.zeros_like(H_spec)
    if fit.quantum is not None:
        _g, H_q = fit._project_quantum_terms(fit.quantum.gradient, fit.quantum.hessian)
        a_eff = fit.optimizer.effective_quantum_weight(
            J, int(np.linalg.matrix_rank(J)), hessian=H_q)
        H_quant = a_eff * fit.optimizer._spd_hessian(H_q)
    H_gn = H_spec + H_quant
    n = H_gn.shape[0]
    cov = np.linalg.inv(H_gn + 1e-8 * np.eye(n))
    _cp, basis, _g2 = fit.physical_covariance(cov)
    blk = basis.T @ H_gn @ basis
    w, V = np.linalg.eigh((blk + blk.T) / 2)
    v = basis @ V[:, 0]
    v /= np.linalg.norm(v)

    x0 = fit.coords.ravel()
    step = rows[0]["step"]

    def half_chi2_full(x):
        _j, r = fit.spectral.stacked(x.reshape(-1, 3))     # full path
        return 0.5 * float(np.sum(np.asarray(r) ** 2))

    f0 = half_chi2_full(x0)
    c_spec = (half_chi2_full(x0 + step * v) - 2 * f0
              + half_chi2_full(x0 - step * v)) / (step * step)
    expected = c_spec + float(v @ H_quant @ v)
    assert expected == pytest.approx(rows[0]["curvature_exact"], rel=1e-9)
