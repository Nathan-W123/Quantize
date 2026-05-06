from __future__ import annotations

import numpy as np

from backend.internal_fit import (
    InternalCoordinateSet,
    apply_internal_step,
    spectral_jacobian_q,
)
from backend.spectral import SpectralEngine


def _water_coords():
    # Bent-water-like geometry in Angstrom.
    return np.array(
        [
            [0.0000, 0.0000, 0.1174],   # O
            [0.0000, 0.7572, -0.4696],  # H
            [0.0000, -0.7572, -0.4696], # H
        ],
        dtype=float,
    )


def _water_iso():
    return {
        "name": "H2-16O",
        "masses": [15.99491461956, 1.00782503207, 1.00782503207],
        "obs_constants": [835840.3, 435351.7, 278138.7],
        "component_indices": [0, 1, 2],
        "sigma_constants": [0.2, 0.2, 0.2],
        "alpha_constants": [0.0, 0.0, 0.0],
    }


def test_internal_backtransform_hits_target():
    coords = _water_coords()
    coord_set = InternalCoordinateSet(coords, ["O", "H", "H"], use_dihedrals=False)
    q0 = coord_set.active_values(coords)
    # Small feasible perturbation: stretch first internal coordinate.
    q_target = q0.copy()
    q_target[0] += 1e-3

    x_new, err = apply_internal_step(
        coords, q_target, coord_set, max_micro=40, tol=1e-10, damping=1e-8
    )
    q_new = coord_set.active_values(x_new)

    assert np.all(np.isfinite(q_new))
    assert err < 1e-6
    assert abs(q_new[0] - q_target[0]) < 1e-5


def test_internal_jacobian_bridge_matches_fd():
    coords = _water_coords()
    iso = _water_iso()

    engine = SpectralEngine([iso], analytic_jacobian=True)
    idx = np.asarray(iso["component_indices"], dtype=int)
    Jx = engine.jacobian(coords, np.asarray(iso["masses"], dtype=float), idx)

    coord_set = InternalCoordinateSet(coords, ["O", "H", "H"], use_dihedrals=False)
    B = coord_set.active_B_matrix(coords)
    Bplus = InternalCoordinateSet.damped_pseudoinverse(B, damping=1e-8)
    Jq = spectral_jacobian_q(Jx, Bplus)

    # Compare Jq columns against direct finite differences in q-space.
    h = 1e-5
    q0 = coord_set.active_values(coords)
    fd_cols = []
    for j in range(coord_set.n_active):
        qp = q0.copy()
        qm = q0.copy()
        qp[j] += h
        qm[j] -= h
        xp, _ = apply_internal_step(coords, qp, coord_set, max_micro=40, damping=1e-8)
        xm, _ = apply_internal_step(coords, qm, coord_set, max_micro=40, damping=1e-8)
        yp = engine.rotational_constants(xp, np.asarray(iso["masses"], dtype=float))[idx]
        ym = engine.rotational_constants(xm, np.asarray(iso["masses"], dtype=float))[idx]
        fd_cols.append((yp - ym) / (2.0 * h))
    Jq_fd = np.column_stack(fd_cols)

    assert Jq.shape == Jq_fd.shape
    assert np.all(np.isfinite(Jq))
    assert np.all(np.isfinite(Jq_fd))
    # Finite-difference and transformed Jacobians should agree to practical tolerance.
    assert np.allclose(Jq, Jq_fd, rtol=1e-1, atol=5.0)

