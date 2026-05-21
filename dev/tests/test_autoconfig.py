"""Unit tests for AutoConfig heuristic bases and runtime tuning."""

import numpy as np
import pytest

from backend.autoconfig import AutoConfigEngine
from backend.autoconfig_bases import ProblemShape, count_spectral_rows, infer_optimizer_bases


def _ocs_isotopologues():
    return [
        {"component_indices": [1]},
        {"component_indices": [1]},
    ]


def _benzene_like_isotopologues():
    return [{"component_indices": [0, 1, 2]}] * 4


def test_count_spectral_rows():
    assert count_spectral_rows(_ocs_isotopologues()) == 2
    assert count_spectral_rows(_benzene_like_isotopologues()) == 12


def test_infer_optimizer_bases_underdetermined_vs_constrained():
    base = dict(
        trust_radius=0.005,
        null_trust_radius=0.0025,
        lambda_damp=1.6e-4,
        sv_threshold=1.4e-5,
        alpha_quantum=0.28,
    )
    ocs = ProblemShape(n_atoms=3, n_params=9, n_jacobian_rows=2)
    big = ProblemShape(n_atoms=12, n_params=36, n_jacobian_rows=12, has_internal_priors=True, prior_weight=1.0)
    ocs_b = infer_optimizer_bases(ocs, **base)
    big_b = infer_optimizer_bases(big, **base)
    assert big_b["trust_radius"] <= ocs_b["trust_radius"]
    assert big_b["sv_threshold"] >= ocs_b["sv_threshold"]
    assert big_b["lambda_damp"] >= ocs_b["lambda_damp"]


def test_autoconfig_sv_gap_increases_sv_threshold():
    engine = AutoConfigEngine(
        n_params=9,
        base_trust_radius=0.005,
        base_null_trust_radius=0.0025,
        base_lambda_damp=1e-4,
        base_prior_weight=1.0,
        base_sigma_floor_mhz=0.02,
        base_max_spectral_weight=100.0,
        base_torsion_a_weight=1.0,
        base_sv_threshold=1e-5,
        base_alpha_quantum=0.3,
        tune_sv_threshold=True,
        tune_alpha_quantum=False,
        smoothing=1.0,
    )
    # Unstable rank boundary: s[0]/s[1] small gap at rank=1
    s_unstable = np.array([10.0, 9.5, 0.01, 0.001])
    c1 = engine.suggest(
        rank=1,
        singular_values=s_unstable,
        residual_mhz=np.array([1.0]),
        sigma_scale_mhz=0.1,
        torsion_a_residuals=np.array([]),
        torsion_bc_residuals=np.array([]),
        reject_streak=0,
        has_internal_priors=False,
    )
    # Stable gap at rank=1: 10 / 0.01 = 1000
    engine.state = None
    s_stable = np.array([10.0, 0.01, 0.001])
    c2 = engine.suggest(
        rank=1,
        singular_values=s_stable,
        residual_mhz=np.array([1.0]),
        sigma_scale_mhz=0.1,
        torsion_a_residuals=np.array([]),
        torsion_bc_residuals=np.array([]),
        reject_streak=0,
        has_internal_priors=False,
    )
    assert c1["sv_threshold"] > c2["sv_threshold"]
    assert engine._sv_gap_at_rank(s_unstable, 1) == pytest.approx(10.0 / 9.5, rel=1e-6)


def test_autoconfig_alpha_lower_when_null_space_large():
    engine = AutoConfigEngine(
        n_params=30,
        base_trust_radius=0.005,
        base_null_trust_radius=0.0025,
        base_lambda_damp=1e-4,
        base_prior_weight=1.0,
        base_sigma_floor_mhz=0.02,
        base_max_spectral_weight=100.0,
        base_torsion_a_weight=1.0,
        base_sv_threshold=1e-5,
        base_alpha_quantum=1.0,
        tune_sv_threshold=False,
        tune_alpha_quantum=True,
        smoothing=1.0,
    )
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01])
    low_rank = engine.suggest(
        rank=2,
        singular_values=s,
        residual_mhz=np.array([1.0, 2.0]),
        sigma_scale_mhz=0.1,
        torsion_a_residuals=np.array([]),
        torsion_bc_residuals=np.array([]),
        reject_streak=0,
        has_internal_priors=False,
    )
    engine.state = None
    high_rank = engine.suggest(
        rank=8,
        singular_values=s,
        residual_mhz=np.array([1.0, 2.0]),
        sigma_scale_mhz=0.1,
        torsion_a_residuals=np.array([]),
        torsion_bc_residuals=np.array([]),
        reject_streak=0,
        has_internal_priors=False,
    )
    assert low_rank["alpha_quantum"] < high_rank["alpha_quantum"]
