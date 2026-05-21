"""Tests for harmonic centrifugal distortion from Hessian."""

from __future__ import annotations

import numpy as np

from backend.spectral.centrifugal_distortion import (
    CD_NAMES,
    bk_mode_derivatives,
    build_cd_table_from_hessian,
    compute_cd_constants,
    tau_prime_from_dB1_cm,
    watson_a_reduction_cd_from_tau_cm,
)
from backend.spectral.harmonic_alpha import compute_harmonic_alpha, _bk_mode_derivatives


def _water_coords():
    return np.array(
        [
            [0.0, 0.0, 0.1174],
            [0.0, 0.7572, -0.4696],
            [0.0, -0.7572, -0.4696],
        ],
        dtype=float,
    )


def _water_masses():
    return np.array([15.99491461956, 1.00782503207, 1.00782503207], dtype=float)


def _random_spd_hess(n_atoms: int, scale: float = 0.01) -> np.ndarray:
    n3 = 3 * n_atoms
    r = np.random.default_rng(0).standard_normal((n3, n3))
    h = r @ r.T * scale
    return (h + h.T) / 2.0


def test_bk_mode_derivatives_shared_with_harmonic_alpha():
    coords = _water_coords()
    masses = _water_masses()
    hess = _random_spd_hess(3)
    from backend.spectral.centrifugal_distortion import normal_modes

    omega, L = normal_modes(hess, masses)
    d1a, d2a = bk_mode_derivatives(coords, masses, L, omega, 0.05)
    d1b, d2b = _bk_mode_derivatives(coords, masses, L, omega, 0.05)
    assert d1a.shape == d1b.shape
    assert np.allclose(d1a, d1b)
    assert np.allclose(d2a, d2b)


def test_compute_cd_constants_finite():
    coords = _water_coords()
    masses = _water_masses()
    hess = _random_spd_hess(3, scale=0.02)
    cd = compute_cd_constants(hess, coords, masses)
    for k in CD_NAMES:
        assert np.isfinite(getattr(cd, k))
    assert cd.source == "harmonic_hessian"


def test_build_cd_table_from_hessian():
    coords = _water_coords()
    masses = _water_masses()
    hess = _random_spd_hess(3, scale=0.02)
    isos = [{"name": "H2-16O", "masses": masses.tolist()}]
    table = build_cd_table_from_hessian(hess, coords, isos)
    assert "H2-16O" in table
    assert np.isfinite(table["H2-16O"].DJ)


def test_harmonic_alpha_still_runs():
    coords = _water_coords()
    masses = _water_masses()
    hess = _random_spd_hess(3, scale=0.02)
    alpha_sum, B_e, sigma, info = compute_harmonic_alpha(hess, coords, masses)
    assert set(alpha_sum.keys()) == {"A", "B", "C"}
    assert all(np.isfinite(list(B_e.values())))


def test_spectral_engine_cd_stacked_optional():
    from backend.spectral.spectral import SpectralEngine

    coords = _water_coords()
    masses = _water_masses()
    hess = _random_spd_hess(3, scale=0.02)
    iso = {
        "name": "H2-16O",
        "masses": masses.tolist(),
        "obs_constants": [800000.0, 430000.0, 270000.0],
        "sigma_constants": [100.0, 100.0, 100.0],
        "component_indices": [0, 1, 2],
        "cd_observed": {"DJ": 30.0, "DJK": -100.0, "DK": 500.0},
    }
    se = SpectralEngine(
        [iso],
        cd_weight=1.0,
        fit_cd_constants=True,
        hess_bohr_for_cd=hess,
    )
    J, r = se.stacked(coords)
    assert J.shape[0] >= 3 + 3  # B rows + at least 3 CD rows
    assert r.size == J.shape[0]


def test_watson_cd_from_tau_symmetric():
    tau = np.array([[-1.0, 0.2, 0.1], [0.2, -0.5, 0.0], [0.1, 0.0, -2.0]])
    cd = watson_a_reduction_cd_from_tau_cm(tau)
    assert set(cd.keys()) == set(CD_NAMES)
    assert np.isfinite(cd["DJ"])
