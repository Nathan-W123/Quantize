"""The Laplace covariance is inverted over the physical directions only.

Inverting the full 3N matrix and projecting afterwards gives the same matrix
in exact arithmetic and a different one in floating point. The rigid-body
directions carry only the quantum block's eigenvalue floor, so their
variances sit about sixteen orders above the physical ones, and the
inverse's round-off swamps every physical entry. On water with the relative
sigma floor off, that refused every sigma at or below 2 MHz under C2v --
where the physical block's condition number is 16.4 -- and under C1 returned
widths silently wrong by 67% at 2 MHz and an angle width of exactly 0.0 at
0.02 MHz.

These tests drive the cases that broke. They fail on the 3N inverse.
"""

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

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(sigma_mhz, symmetry="auto", **kw):
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O,
                        "sigma_constants": [sigma_mhz] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, sigma_floor_rel=0.0,
        symmetry=symmetry, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


def _widths(opt):
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.geometry_uncertainty()


def test_tight_sigma_is_answered_and_scales_linearly():
    """Under C2v both physical directions are fixed by the data, so the
    widths must scale exactly with the quoted sigma -- down to 0.02 MHz.

    Measured: O1-H2 1.333229e-05, -06, -07, -08 Angstrom at 20, 2, 0.2 and
    0.02 MHz, each matching an exact rational inverse to 1e-15. The 3N
    inverse refused all but the first.
    """
    sigmas = [20.0, 2.0, 0.2, 0.02]
    out = [_widths(_fit(s)) for s in sigmas]
    for s, w in zip(sigmas, out):
        assert w is not None, f"refused at {s} MHz"
    for name in ("O1-H2", "H2-O1-H3"):
        for a, b in zip(out, out[1:]):
            assert a[name] / b[name] == pytest.approx(10.0, rel=1e-6), name


def test_c1_angle_matches_c2v_where_the_3n_inverse_got_it_wrong():
    """The angle is symmetric, so freeing the antisymmetric stretch must not
    change its width at a symmetric optimum.

    At 2 MHz the 3N inverse reported the C1 angle 67% too narrow while the
    3N lo/hi guard passed, because the largest variance -- the prior-only
    antisymmetric stretch -- sat in its denominator. Both are 1.3716e-4 deg.
    """
    c2v = _widths(_fit(2.0))
    c1 = _widths(_fit(2.0, symmetry="C1"))
    assert c1 is not None and c2v is not None
    assert c1["H2-O1-H3"] == pytest.approx(c2v["H2-O1-H3"], rel=1e-6)


def test_c1_at_the_tightest_sigma_reports_a_width_not_zero():
    """0.02 MHz under C1: the 3N inverse clamped the angle variance to 0.0,
    which reads as perfect precision. The exact width is 1.649e-6 deg."""
    w = _widths(_fit(0.02, symmetry="C1"))
    assert w is not None
    assert np.isfinite(w["H2-O1-H3"]) and w["H2-O1-H3"] > 1e-7


def test_covariance_comes_back_projected():
    """Nothing along a rigid-body or symmetry-forbidden direction, so the
    projection downstream has nothing left to remove."""
    opt = _fit(0.2)
    cov, std = opt.laplace_approximation()
    assert cov is not None and std is not None
    scale = float(np.linalg.norm(cov))
    once, basis, n_removed = opt.physical_covariance(cov)
    assert n_removed == 7               # 6 rigid-body + 1 symmetry-forbidden
    assert float(np.linalg.norm(once - cov)) / scale < 1e-14
    removed = np.eye(cov.shape[0]) - basis @ basis.T
    assert float(np.linalg.norm(removed @ cov @ removed)) / scale < 1e-14
    # and the reported standard errors are physical-scale numbers, not the
    # 112-Angstrom widths the floor used to give the rigid directions
    assert float(np.max(std)) < 1.0


def test_rank_is_judged_on_the_physical_block_without_the_ridge():
    """One constant and no quantum prior: one physical direction is blind.

    The ridge alone makes that block look full rank (rank 1 becomes 2), so
    the warning has to be taken from the unridged block. Taken on the 3N
    matrix instead it reported "rank 3 of 9" on every healthy fit, which
    said nothing about the fit at all.
    """
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O[:1], "sigma_constants": [0.2],
                        "component_indices": [0]}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        spectral_only=True, quantum_prior_sigma_ang=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    assert cov is not None
    assert "over the 2 physical directions the Hessian is rank 1" in buf.getvalue()

    # And a healthy hybrid fit says nothing about rank.
    healthy = _fit(0.2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        healthy.laplace_approximation(check_gauss_newton=False)
    assert "rank" not in buf.getvalue()


def test_gauss_newton_probe_width_is_exact():
    """The probe's width along each direction is 1/sqrt(curvature + 1e-8),
    not a round-off-limited entry of a 3N inverse (4.965e-3 against the exact
    5.645e-9 at 0.02 MHz, which set the finite-difference step 1e5 times too
    large)."""
    opt = _fit(0.02, symmetry="C1")
    rows = opt.gauss_newton_discrepancy()
    assert rows
    for row in rows:
        expect = 1.0 / np.sqrt(row["curvature_gauss_newton"] + 1e-8)
        assert row["sigma_along"] == pytest.approx(expect, rel=1e-12)


def test_physical_basis_is_cached_and_read_only():
    opt = _fit(0.2)
    b1 = opt.physical_basis()
    assert opt.physical_basis() is b1                # same point: cached
    with pytest.raises(ValueError):
        b1[0, 0] = 1.0                               # cannot corrupt the cache
    moved = opt.physical_basis(coords=opt.coords + 1e-3)
    assert moved is not b1                           # new point: recomputed
    assert opt.physical_basis() is not moved
