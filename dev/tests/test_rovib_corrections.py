"""Tests for backend.rovib_corrections.resolve_alpha_components."""

from __future__ import annotations

import numpy as np
import pytest

from backend.correction_models import RovibCorrection
from backend.rovib_corrections import (
    build_correction_from_iso,
    effective_sigma_constants,
    resolve_alpha_components,
)


def test_hybrid_auto_prefers_user_then_orca():
    existing = np.zeros(3)
    idx = [0, 1, 2]
    parsed = np.array([10.0, 20.0, 30.0])
    user = {"A": 1.0, "B": np.nan, "C": 3.0}  # B should fall back to ORCA
    resolved, corr = resolve_alpha_components(
        existing, idx, parsed, user, mode="hybrid_auto",
        isotopologue_name="parent",
    )
    assert resolved[0] == pytest.approx(1.0)
    assert resolved[1] == pytest.approx(20.0)
    assert resolved[2] == pytest.approx(3.0)
    assert isinstance(corr, RovibCorrection)
    assert corr.source in {"hybrid", "user", "orca"}


def test_user_only_skips_orca():
    existing = np.zeros(3)
    idx = [0, 1, 2]
    parsed = np.array([10.0, 20.0, 30.0])
    user = {"A": 1.0, "B": np.nan, "C": np.nan}
    resolved, _ = resolve_alpha_components(
        existing, idx, parsed, user, mode="user_only"
    )
    assert resolved[0] == pytest.approx(1.0)
    # B & C must remain at existing because user_only never falls back to ORCA.
    assert resolved[1] == pytest.approx(0.0)
    assert resolved[2] == pytest.approx(0.0)


def test_orca_only_skips_user():
    existing = np.zeros(3)
    idx = [0, 1, 2]
    parsed = np.array([10.0, 20.0, 30.0])
    user = {"A": 1.0, "B": 2.0, "C": 3.0}
    resolved, _ = resolve_alpha_components(
        existing, idx, parsed, user, mode="orca_only"
    )
    assert np.allclose(resolved, [10.0, 20.0, 30.0])


def test_nan_fallback_keeps_existing():
    existing = np.array([7.0, 8.0, 9.0])
    idx = [0, 1, 2]
    resolved, _ = resolve_alpha_components(
        existing, idx, parsed_alpha_abc=None, user_alpha_abc=None,
        mode="hybrid_auto",
    )
    assert np.allclose(resolved, [7.0, 8.0, 9.0])


def test_strict_user_raises_when_missing():
    existing = np.zeros(3)
    with pytest.raises(ValueError, match="strict_user"):
        resolve_alpha_components(
            existing, [0, 1, 2], parsed_alpha_abc=[1.0, 2.0, 3.0],
            user_alpha_abc=None, mode="strict_user",
        )


def test_strict_backend_raises_when_missing():
    existing = np.zeros(3)
    with pytest.raises(ValueError, match="strict_backend"):
        resolve_alpha_components(
            existing, [0, 1, 2], parsed_alpha_abc=None,
            user_alpha_abc={"A": 1.0, "B": 2.0, "C": 3.0},
            mode="strict_backend",
        )


def test_b_c_only_subset():
    """Only B and C are fitted; resolution should index the right components."""
    existing = np.zeros(2)
    idx = [1, 2]  # B, C
    parsed = np.array([10.0, 20.0, 30.0])
    user = {"A": 1.0, "B": 2.0, "C": 3.0}
    resolved, _ = resolve_alpha_components(
        existing, idx, parsed, user, mode="hybrid_auto"
    )
    # Should pick user's B (=2) and user's C (=3).
    assert resolved[0] == pytest.approx(2.0)
    assert resolved[1] == pytest.approx(3.0)


def test_manual_delta_does_not_overwrite_alpha():
    existing = np.array([5.0, 6.0, 7.0])
    parsed = np.array([10.0, 20.0, 30.0])
    user = {"A": 1.0, "B": 2.0, "C": 3.0}
    resolved, _ = resolve_alpha_components(
        existing, [0, 1, 2], parsed, user, mode="manual_delta"
    )
    # Delta-mode means alpha array is left alone.
    assert np.allclose(resolved, existing)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown rovib mode"):
        resolve_alpha_components(
            np.zeros(3), [0, 1, 2], None, None, mode="banana_split"
        )


def test_build_correction_from_iso_alpha_path():
    iso = {
        "name": "parent",
        "component_indices": [0, 1, 2],
        "alpha_constants": [10.0, 20.0, 30.0],
    }
    corr = build_correction_from_iso(iso, method="VPT2", basis="cc-pVTZ")
    assert corr.alpha_A == pytest.approx(10.0)
    assert corr.alpha_B == pytest.approx(20.0)
    assert corr.alpha_C == pytest.approx(30.0)
    # delta_vib falls back to 0.5 * alpha.
    dv = corr.delta_vib_vector()
    assert dv[0] == pytest.approx(5.0)
    assert dv[1] == pytest.approx(10.0)
    assert dv[2] == pytest.approx(15.0)


def test_build_correction_from_iso_delta_path():
    iso = {
        "name": "parent",
        "component_indices": [0, 1, 2],
        "delta_vib_constants": [4.0, 8.0, 12.0],
        "delta_elec_constants": [0.1, 0.2, 0.3],
        "sigma_correction_constants": [0.01, 0.02, 0.03],
    }
    corr = build_correction_from_iso(iso)
    assert corr.delta_vib_A == pytest.approx(4.0)
    total = corr.delta_total_vector()
    assert total[0] == pytest.approx(4.1)
    assert corr.sigma_delta_A == pytest.approx(0.01)


def test_effective_sigma_combines_in_quadrature():
    iso = {
        "component_indices": [0, 1, 2],
        "sigma_constants": [3.0, 4.0, 0.0],
        "sigma_correction_constants": [4.0, 3.0, 0.0],
    }
    eff = effective_sigma_constants(iso)
    assert eff[0] == pytest.approx(5.0)  # 3-4-5
    assert eff[1] == pytest.approx(5.0)
    assert eff[2] == pytest.approx(0.0)
