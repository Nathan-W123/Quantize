from __future__ import annotations

import numpy as np

from backend.internal_fit import InternalCoordinateSet, build_internal_priors
from backend.prior_sensitivity import classify_prior_dominance, prior_sensitivity_analysis


def _tiny_molecule():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.10, 0.0, 0.0],
            [0.0, 1.10, 0.0],
        ],
        dtype=float,
    )
    elems = ["C", "H", "H"]
    return coords, elems


def test_classify_prior_dominance_outputs_labels():
    coords, elems = _tiny_molecule()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    n = cs.n_active
    j = np.zeros((4, n), dtype=float)
    j[:, 0] = 1.0
    sigma = np.ones(n, dtype=float) * 0.1
    labels = classify_prior_dominance(j, sigma, cs.active_names())
    assert set(labels.keys()) == set(cs.active_names())
    assert all(v in {"data-constrained", "prior-dominated", "mixed information", "unidentifiable"} for v in labels.values())


def test_prior_sensitivity_analysis_returns_expected_fields():
    coords, elems = _tiny_molecule()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    j = np.random.default_rng(0).normal(size=(6, cs.n_active))
    residual = np.zeros(6, dtype=float)
    rows = prior_sensitivity_analysis(
        cs,
        coords,
        j,
        residual,
        sigma_bond=0.04,
        sigma_angle_deg=2.0,
        sigma_dihedral_deg=15.0,
        prior_mode="soft",
        prior_specs=[],
        elems=elems,
        freeze_sigma_floor=1e-6,
        spectral_jacobian_q=j,
        adaptive_config={"identifiability_gamma": 2.0},
        sv_rel_threshold=1e-3,
        lambda_reg=1e-6,
    )
    assert len(rows) == cs.n_active
    for r in rows:
        assert "name" in r and "delta" in r and "sensitivity_label" in r
        assert r["sensitivity_label"] in {"stable", "mildly prior-sensitive", "strongly prior-sensitive"}

