"""Analytic vs finite-difference rotational-constant Jacobian (spectral.py)."""

import numpy as np

from backend.spectral import (
    SpectralEngine,
    _jacobian_full,
    _jacobian_full_analytic,
)


def test_analytic_matches_finite_difference_random_geometries():
    """Asymmetric geometries avoid FD cancellation noise on near-zero ∂A/∂x entries."""
    rng = np.random.default_rng(2)
    delta = 1e-3
    for n in (3, 4, 6):
        masses = rng.uniform(1.0, 40.0, size=n)
        coords = 0.4 * rng.standard_normal((n, 3))
        Ja = _jacobian_full_analytic(coords, masses, delta)
        Jf = _jacobian_full(coords, masses, delta)
        scale = max(float(np.max(np.abs(Jf))), 1.0)
        np.testing.assert_allclose(Ja, Jf, rtol=1e-4, atol=1e-3 * scale)


def test_spectral_engine_analytic_flag_off_uses_fd():
    masses = np.array([12.0, 16.0])
    coords = np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]], dtype=float)
    iso = [{"name": "test", "masses": masses, "obs_constants": np.array([1.0, 2.0, 3.0])}]
    se_fd = SpectralEngine(iso, delta=1e-3, analytic_jacobian=False)
    se_an = SpectralEngine(iso, delta=1e-3, analytic_jacobian=True)
    Jfd = se_fd.jacobian(coords, masses, None)
    Jan = se_an.jacobian(coords, masses, None)
    np.testing.assert_allclose(Jan, Jfd, rtol=5e-5, atol=1e-6)
