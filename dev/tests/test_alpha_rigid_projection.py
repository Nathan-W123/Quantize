"""Harmonic alpha away from a stationary point must not see the molecule rotate.

compute_harmonic_alpha runs at geometries that are not minima of the surface:
the benchmark's offsets configurations build their table at the offset-shifted
geometry, and tracking evaluates it at fitted geometries. There the gradient
makes the raw Hessian's rotational block non-zero, rotation leaks into the
vibrational eigenvectors, and the Watson inertia-derivative term reads the
leaked rotation as an off-diagonal inertia derivative divided by the small
moment. On isocyanic acid (HF/6-31G) at its offset geometry that moved alpha_B
by ~100 MHz (decisions D-52). The engine now projects translations and
rotations out of the Hessian first (_project_rigid_mw).

The reference here is the same surface tilted by its own gradient in valence
coordinates, E'(x) = E(x) - g . q(x): it is stationary at the geometry and
rotation-invariant, so its Hessian has no rigid contamination to begin with.
It is not identical to the projected Hessian -- off a stationary point the
vibrational part of the gradient term depends on the coordinates it is written
in (the bend here differs by 4 cm^-1) -- so the projected alpha agrees with it
to about 1 MHz, against the 23.5 MHz the unprojected one is off.

The surface is an analytic bent H-N-C valence force field (I_a 0.57 amu A^2),
so the test needs no electronic structure.
"""

import numpy as np
import pytest

from backend.spectral import harmonic_alpha as HA
from backend.spectral.centrifugal_distortion import is_linear

_AJ_TO_HARTREE = 1.0 / 4.3597447222071       # 1 aJ in Hartree
_BOHR_A = 0.529177210903

MASSES = np.array([1.00782503, 14.0030740, 12.0])   # H, N, C
Q_EQ = np.array([1.00, 1.17, np.deg2rad(125.0)])     # r_NH, r_NC (A), H-N-C (rad)
K = np.array([6.5, 16.0, 0.5])                       # aJ/A^2, aJ/A^2, aJ/rad^2


def _valence(x):
    p = np.asarray(x, dtype=float).reshape(3, 3)
    u, v = p[0] - p[1], p[2] - p[1]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    return np.array([ru, rv, np.arccos(np.clip(u @ v / (ru * rv), -1.0, 1.0))])


def _energy_aj(x, tilt=None):
    d = _valence(x) - Q_EQ
    e = 0.5 * float(np.sum(K * d * d))
    return e if tilt is None else e - float(tilt @ _valence(x))


def _hessian_hartree_bohr2(coords, tilt=None, h=1e-3):
    x0 = np.asarray(coords, dtype=float).ravel()
    n = x0.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e = np.zeros(n); e[i] = h
            f = np.zeros(n); f[j] = h
            H[i, j] = H[j, i] = (
                _energy_aj(x0 + e + f, tilt) - _energy_aj(x0 + e - f, tilt)
                - _energy_aj(x0 - e + f, tilt) + _energy_aj(x0 - e - f, tilt)) / (4 * h * h)
    return H * _AJ_TO_HARTREE * _BOHR_A ** 2


def _geometry(r_nh, r_nc, theta):
    return np.array([r_nh * np.array([np.cos(theta), 0.0, np.sin(theta)]),
                     np.zeros(3), np.array([r_nc, 0.0, 0.0])])


def _alpha_b(hess, coords):
    alpha, _, _, _ = HA.compute_harmonic_alpha(hess, coords, MASSES)
    return alpha["B"]


def test_alpha_off_a_stationary_point_is_rotation_free(monkeypatch):
    # N-C stretched 0.015 A off its minimum: a gradient of 0.24 aJ/A, the size
    # isocyanic acid's offset geometry has (0.27 aJ/A).
    x0 = _geometry(Q_EQ[0], Q_EQ[1] + 0.015, Q_EQ[2])
    g = K * (_valence(x0) - Q_EQ)
    assert not is_linear(x0, MASSES)
    assert np.abs(g).max() > 0.2

    hess = _hessian_hartree_bohr2(x0)
    reference = _alpha_b(_hessian_hartree_bohr2(x0, tilt=g), x0)
    projected = _alpha_b(hess, x0)
    # The premise: without the projection this geometry is contaminated.
    with monkeypatch.context() as mp:
        mp.setattr(HA, "_project_rigid_mw", lambda h, c, m: np.asarray(h, dtype=float))
        unprojected = _alpha_b(hess, x0)

    # Measured 2026-11-20 (alpha_B ~ -546 MHz): unprojected 23.5 MHz from the
    # reference, projected 0.92. 2 MHz is 2.2x headroom on the projected value
    # and fails the unprojected one by 12x; the premise bound (15) sits 1.6x
    # under its measured 23.5.
    assert abs(unprojected - reference) > 15.0
    assert projected == pytest.approx(reference, abs=2.0)


def test_projection_is_inert_at_a_stationary_point(monkeypatch):
    x_eq = _geometry(*Q_EQ)
    hess = _hessian_hartree_bohr2(x_eq)
    projected = _alpha_b(hess, x_eq)
    with monkeypatch.context() as mp:
        mp.setattr(HA, "_project_rigid_mw", lambda h, c, m: np.asarray(h, dtype=float))
        unprojected = _alpha_b(hess, x_eq)
    # Measured 2.2e-3 MHz (of -571): only the rigid modes' finite-difference
    # residue is removed. 0.02 leaves 9x headroom.
    assert projected == pytest.approx(unprojected, abs=0.02)
