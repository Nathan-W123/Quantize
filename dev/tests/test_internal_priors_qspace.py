from __future__ import annotations

import numpy as np

from backend.internal.internal_fit import InternalCoordinateSet, build_internal_priors
from backend.priors.prior_weighting import (
    adaptive_sigma_from_identifiability,
    identifiability_scores_from_jacobian,
)


def _methane_like():
    # Simple tetrahedral CH4-ish coordinates in Angstrom.
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],   # C
            [0.6291, 0.6291, 0.6291],   # H
            [-0.6291, -0.6291, 0.6291], # H
            [-0.6291, 0.6291, -0.6291], # H
            [0.6291, -0.6291, -0.6291], # H
        ],
        dtype=float,
    )
    elems = ["C", "H", "H", "H", "H"]
    return coords, elems


def test_internal_priors_mode_off_returns_empty_block():
    coords, elems = _methane_like()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    jp, rp, sig = build_internal_priors(cs, coords, prior_mode="off", elems=elems)
    assert jp.shape == (0, 0)
    assert rp.size == 0
    assert sig.size == 0


def test_internal_priors_default_library_applies_ch_bond_targets():
    coords, elems = _methane_like()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    _, _, sig, meta = build_internal_priors(cs, coords, elems=elems, return_metadata=True)
    bond_rows = [m for m in meta if m["kind"] == "bond"]
    assert bond_rows
    # C-H defaults from internal prior library should populate provenance and sigma.
    assert any(m["source"] == "default_library" for m in bond_rows)
    assert all(float(m["sigma"]) > 0.0 for m in bond_rows)
    assert sig.shape[0] == cs.n_active


def test_internal_priors_user_prior_degree_conversion_and_override():
    coords, elems = _methane_like()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    # Choose first angle row from active coordinates.
    angle_ic = next(ic for ic in cs.active_coords() if ic.kind == "angle")
    atoms_1based = [a + 1 for a in angle_ic.atoms]
    jp, rp, sig, meta = build_internal_priors(
        cs,
        coords,
        elems=elems,
        prior_specs=[
            {
                "type": "angle",
                "atoms": atoms_1based,
                "target": 109.5,
                "sigma": 0.5,
                "units": "degrees",
                "source": "user_supplied",
            }
        ],
        return_metadata=True,
    )
    assert jp.shape[0] == cs.n_active
    m = next(m for m in meta if m["kind"] == "angle" and tuple(m["atoms"]) == tuple(angle_ic.atoms))
    assert m["source"] == "user_supplied"
    assert np.isclose(float(m["target_value"]), np.deg2rad(109.5))
    assert np.isclose(float(m["sigma"]), np.deg2rad(0.5))
    # Residual and sigma vectors align with active coordinates.
    assert rp.shape[0] == cs.n_active
    assert sig.shape[0] == cs.n_active


def test_internal_priors_hard_freeze_sets_tiny_sigma():
    coords, elems = _methane_like()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    bond_ic = next(ic for ic in cs.active_coords() if ic.kind == "bond")
    atoms_1based = [a + 1 for a in bond_ic.atoms]
    _, _, sig, meta = build_internal_priors(
        cs,
        coords,
        elems=elems,
        prior_mode="hard_freeze",
        freeze_sigma_floor=1e-7,
        prior_specs=[
            {
                "type": "bond",
                "atoms": atoms_1based,
                "target": 1.09,
                "sigma": 0.02,
                "units": "angstrom",
                "mode": "hard_freeze",
            }
        ],
        return_metadata=True,
    )
    idx = next(i for i, m in enumerate(meta) if m["kind"] == "bond" and tuple(m["atoms"]) == tuple(bond_ic.atoms))
    assert sig[idx] <= 1e-7 + 1e-12


def test_adaptive_sigma_scaling_is_monotonic_and_clipped():
    base = np.array([1.0, 1.0, 1.0], dtype=float)
    scores = np.array([0.0, 0.5, 1.0], dtype=float)
    sigma, scale = adaptive_sigma_from_identifiability(
        base,
        scores,
        gamma=2.0,
        min_sigma_scale=0.5,
        max_sigma_scale=2.0,
    )
    assert np.all(scale >= 0.5)
    assert np.all(scale <= 2.0)
    assert sigma[0] <= sigma[1] <= sigma[2]
    assert np.isclose(scale[-1], 2.0)  # clipped from 3.0 to 2.0


def test_internal_priors_adaptive_uses_spectral_identifiability():
    coords, elems = _methane_like()
    cs = InternalCoordinateSet(coords, elems, use_dihedrals=False)
    n = cs.n_active
    # Build a Jacobian that strongly identifies coordinate 0, weakly others.
    j = np.zeros((4, n), dtype=float)
    j[:, 0] = np.array([1.0, 0.9, 1.1, 1.0])
    j[:, 1] = np.array([0.01, 0.0, 0.01, 0.0])
    if n > 2:
        j[:, 2] = np.array([0.02, 0.0, 0.0, 0.01])

    _, _, sig_soft = build_internal_priors(cs, coords, elems=elems, prior_mode="soft")
    _, _, sig_adapt, meta = build_internal_priors(
        cs,
        coords,
        elems=elems,
        prior_mode="adaptive",
        spectral_jacobian_q=j,
        adaptive_config={"identifiability_gamma": 2.0, "min_sigma_scale": 0.5, "max_sigma_scale": 3.0},
        return_metadata=True,
    )
    assert sig_soft.shape == sig_adapt.shape
    assert np.all(sig_adapt >= sig_soft)
    scores, _, _ = identifiability_scores_from_jacobian(j)
    assert scores.shape[0] == n
    scales = np.array([float(m.get("sigma_scale", 1.0)) for m in meta], dtype=float)
    assert np.all(scales >= 1.0)
    assert np.any(scales > 1.0)
