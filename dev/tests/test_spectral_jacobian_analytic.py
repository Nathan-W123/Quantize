"""Analytic vs finite-difference rotational-constant Jacobian (spectral.py)."""

import numpy as np

from backend.spectral.spectral import (
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


def _loop_reference(coords, masses):
    """The five-deep loop the closed form replaced, kept as the algebra's
    reference: dI/dx_{j,a} built term by term, then v^T dI v per axis."""
    from backend.spectral.spectral import _INERTIA_TO_MHZ, _inertia_tensor
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(coords)
    evals, evecs = np.linalg.eigh(_inertia_tensor(coords, masses))
    mfrac = masses / masses.sum()
    r = coords - masses @ coords / masses.sum()
    out = np.zeros((3, 3 * n))
    for j in range(n):
        for a in range(3):
            dI = np.zeros((3, 3))
            for i in range(n):
                ci = (1.0 if i == j else 0.0) - mfrac[j]
                for p in range(3):
                    for q in range(3):
                        dI[p, q] += masses[i] * ci * (
                            (2.0 * r[i, a] if p == q else 0.0)
                            - (r[i, q] if p == a else 0.0) - (r[i, p] if q == a else 0.0))
            for s in range(3):
                out[s, 3 * j + a] = -_INERTIA_TO_MHZ * float(evecs[:, s] @ dI @ evecs[:, s]) / evals[s] ** 2
    return out


def test_the_closed_form_is_the_loop_s_algebra():
    """The analytic Jacobian was a Python loop (80-85% of a fit's engine
    time); its closed form must be the same algebra, not an approximation:
    measured 2.4e-14 relative over 1400 random geometries of 3-20 atoms."""
    rng = np.random.default_rng(20261117)
    worst = 0.0
    for n in (3, 5, 9, 14):
        for _ in range(25):
            x = rng.normal(size=(n, 3)) * 1.3
            m = rng.uniform(1.0, 40.0, size=n)
            a = _loop_reference(x, m)
            b = _jacobian_full_analytic(x, m, 1e-3)
            worst = max(worst, float(np.abs(a - b).max() / np.abs(a).max()))
    assert worst < 1e-12, worst
