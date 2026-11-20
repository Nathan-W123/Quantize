"""The harmonic alpha of a linear molecule must not depend on a hair's bend.

compute_harmonic_alpha decides linearity once (rigid_mode_count, i.e.
is_linear: I_min / I_max < 1e-6). Its Watson harmonic term used to keep the
axial inverse moment 1/I_a whenever I_a > 1e-8 amu A^2, an absolute cut, so a
geometry bent far enough to pass that cut but still linear by the module's own
test was given a linear molecule's modes and a harmonic term divided by its
tiny I_a. On fluoroacetylene (HF/6-31G) alpha_B stepped 41.26 -> 33.69 MHz as
I_a crossed 1e-8 on a clean bend, and reached -7.5e6 MHz at the off-stationary
point of the benchmark's tracking slope (decisions D-50).

The surface here is an analytic HCN-like valence force field, so the test needs
no electronic structure. Each geometry gets its own Hessian, as the benchmark's
hessian_fn does.
"""

import numpy as np
import pytest

from backend.spectral.centrifugal_distortion import inertia_paf, is_linear
from backend.spectral.harmonic_alpha import compute_harmonic_alpha

_AJ_TO_HARTREE = 1.0 / 4.3597447222071       # 1 aJ in Hartree
_BOHR_A = 0.529177210903

MASSES = np.array([1.00782503, 12.0, 14.0030740])
R_CH, R_CN = 1.065, 1.153                    # A
K_CH, K_CN, K_BEND = 6.2, 18.7, 0.25          # aJ / A^2, aJ / A^2, aJ


def _energy_aj(x):
    p = x.reshape(3, 3)
    u, v = p[0] - p[1], p[2] - p[1]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    # 1 + cos(theta) is (pi - theta)^2 / 2 near linear and smooth through it.
    bend = 1.0 + float(np.dot(u, v) / (ru * rv))
    return 0.5 * K_CH * (ru - R_CH) ** 2 + 0.5 * K_CN * (rv - R_CN) ** 2 + K_BEND * bend


def _hessian_hartree_bohr2(coords, h=1e-3):
    x0 = np.asarray(coords, dtype=float).ravel()
    n = x0.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e = np.zeros(n); e[i] += h
            f = np.zeros(n); f[j] += h
            val = (_energy_aj(x0 + e + f) - _energy_aj(x0 + e - f)
                   - _energy_aj(x0 - e + f) + _energy_aj(x0 - e - f)) / (4 * h * h)
            H[i, j] = H[j, i] = val
    return H * _AJ_TO_HARTREE * _BOHR_A ** 2


def _bent(d_ang):
    """H moved off the molecular axis by d_ang; C and N on it."""
    return np.array([[d_ang, 0.0, -R_CH], [0.0, 0.0, 0.0], [0.0, 0.0, R_CN]])


def _i_axial(coords):
    return float(np.sort(np.abs(inertia_paf(coords, MASSES)[0]))[0])


def _d_for(i_target):
    # I_a grows as d^2 for small d; one Newton-style rescale from a probe is exact enough.
    d0 = 1e-4
    return d0 * np.sqrt(i_target / _i_axial(_bent(d0)))


def _alpha_b(coords):
    alpha, _, _, info = compute_harmonic_alpha(
        _hessian_hartree_bohr2(coords), coords, MASSES)
    return alpha["B"], info


def test_alpha_is_continuous_across_the_old_axial_cut():
    below, above = _bent(_d_for(0.5e-8)), _bent(_d_for(2e-8))
    # The premise: one geometry on each side of the old 1e-8 cut, both linear
    # by the module's own rule, so neither should see 1/I_a.
    assert _i_axial(below) < 1e-8 < _i_axial(above)
    assert is_linear(below, MASSES) and is_linear(above, MASSES)

    a_lin, info_lin = _alpha_b(_bent(0.0))
    a_below, info_below = _alpha_b(below)
    a_above, info_above = _alpha_b(above)
    assert info_lin["is_linear"] and info_below["is_linear"] and info_above["is_linear"]
    assert info_lin["n_vib"] == info_below["n_vib"] == info_above["n_vib"] == 4

    # Measured 2026-11-20 (alpha_B -88.768 MHz linear): with the fix the three
    # agree to 2.0e-5 MHz (the bend moves alpha at second order in d ~ 1e-4
    # A). Before it the geometry above the cut read -202.016, 113 MHz off,
    # all in the harmonic term. 1e-3 MHz leaves 50x headroom on the first
    # and fails the second by 1e5x.
    assert a_below == pytest.approx(a_lin, abs=1e-3)
    assert a_above == pytest.approx(a_lin, abs=1e-3)
