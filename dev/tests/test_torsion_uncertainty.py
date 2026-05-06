from __future__ import annotations

import numpy as np

from backend.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec
from backend.torsion_uncertainty import (
    TorsionParameter,
    covariance_from_matched_level_residuals,
    covariance_from_normal_matrix,
    default_torsion_parameters,
    torsion_level_jacobian,
)


def _base_spec() -> TorsionHamiltonianSpec:
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 18.0}, units="cm-1")
    return TorsionHamiltonianSpec(F=4.5, rho=0.2, A=1.1, B=0.45, potential=pot, n_basis=6)


def test_torsion_level_jacobian_shape_and_finiteness():
    spec = _base_spec()
    params = default_torsion_parameters(spec, include_completeness=False, potential_vcos=[3])
    requests = [
        {"J": 0, "K": 0, "level_index": 0},
        {"J": 0, "K": 0, "level_index": 1},
        {"J": 1, "K": 0, "level_index": 0},
    ]

    J, y0, p0 = torsion_level_jacobian(spec, requests, params)

    assert J.shape == (len(requests), len(params))
    assert y0.shape == (len(requests),)
    assert p0.shape == (len(params),)
    assert np.all(np.isfinite(J))
    assert np.all(np.isfinite(y0))


def test_covariance_from_normal_matrix_is_symmetric_psdish():
    J = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.0, -1.0],
            [1.5, -0.5, 0.3],
            [0.1, 0.7, 0.8],
        ],
        dtype=float,
    )
    out = covariance_from_normal_matrix(J, damping=1e-8)
    cov = out["covariance"]

    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-12, rtol=0.0)
    eig = np.linalg.eigvalsh(cov)
    assert np.min(eig) > -1e-10
    assert out["rank"] == 3
    assert out["warnings"] == []


def test_singular_case_emits_rank_warning_and_finite_std_err():
    J = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=float,
    )
    out = covariance_from_normal_matrix(J, damping=0.0, rank_tol=1e-12)

    assert out["rank"] < 2
    assert any("rank-deficient" in w for w in out["warnings"])
    assert np.all(np.isfinite(out["std_err"]))


def test_matched_level_helper_returns_consistent_shapes():
    spec = _base_spec()
    params = [
        TorsionParameter("F", ("F",), step_abs=1e-6, step_rel=1e-4),
        TorsionParameter("rho", ("rho",), step_abs=1e-7, step_rel=1e-4),
        TorsionParameter("Vcos_3", ("potential", "vcos", "3"), step_abs=1e-6, step_rel=1e-4),
    ]
    matched = [
        {"J": 0, "K": 0, "level_index": 0, "sigma_cm-1": 0.05},
        {"J": 0, "K": 0, "level_index": 1, "sigma_cm-1": 0.05},
        {"J": 1, "K": 0, "level_index": 0, "sigma_cm-1": 0.08},
    ]

    out = covariance_from_matched_level_residuals(spec, matched, params, damping=1e-8)

    assert out["jacobian"].shape == (len(matched), len(params))
    assert out["predicted"].shape == (len(matched),)
    assert out["covariance"].shape == (len(params), len(params))
    assert out["std_err"].shape == (len(params),)
    assert out["param_names"] == ["F", "rho", "Vcos_3"]
