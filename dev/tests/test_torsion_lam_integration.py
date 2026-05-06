"""Tests for backend/torsion_lam_integration.py (Phase 10: LAM correction integration)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_lam_integration import (
    classify_constant_source,
    combine_lam_corrections,
    format_lam_report_for_summary,
    lam_correction_report,
    lam_uncertainty_contribution,
    remove_torsional_alpha_contributions,
)


# ── classify_constant_source ──────────────────────────────────────────────────

class TestClassifyConstantSource:
    def test_no_config_is_rigid(self):
        assert classify_constant_source({}) == "rigid"

    def test_torsion_disabled_is_rigid(self):
        cfg = {"torsion_hamiltonian": {"enabled": False}}
        assert classify_constant_source(cfg) == "rigid"

    def test_torsion_enabled_no_fit_no_avg_is_rovib(self):
        cfg = {"torsion_hamiltonian": {"enabled": True}}
        assert classify_constant_source(cfg) == "rovib_corrected"

    def test_fitting_enabled_is_globally_fit(self):
        cfg = {"torsion_hamiltonian": {"enabled": True, "fitting": {"enabled": True}}}
        assert classify_constant_source(cfg) == "globally_fit"

    def test_scan_average_enabled_is_torsion_averaged(self):
        cfg = {"torsion_hamiltonian": {"enabled": True, "scan_average": {"enabled": True}}}
        assert classify_constant_source(cfg) == "torsion_averaged"

    def test_fitting_takes_priority_over_scan_average(self):
        cfg = {
            "torsion_hamiltonian": {
                "enabled": True,
                "fitting": {"enabled": True},
                "scan_average": {"enabled": True},
            }
        }
        assert classify_constant_source(cfg) == "globally_fit"

    def test_non_dict_returns_rigid(self):
        assert classify_constant_source(None) == "rigid"
        assert classify_constant_source("bad") == "rigid"


# ── remove_torsional_alpha_contributions ─────────────────────────────────────

class TestRemoveTorsionalAlphaContributions:
    def test_single_mode_subtracted(self):
        alpha = np.array([0.010, 0.005, 0.003])
        mode_alpha = np.array([0.002, 0.001, 0.0005])
        result = remove_torsional_alpha_contributions(alpha, [mode_alpha])
        np.testing.assert_allclose(result, alpha - mode_alpha, atol=1e-12)

    def test_two_modes_subtracted(self):
        alpha = np.array([0.010, 0.005, 0.003])
        m1 = np.array([0.002, 0.001, 0.0005])
        m2 = np.array([0.001, 0.0005, 0.0002])
        result = remove_torsional_alpha_contributions(alpha, [m1, m2])
        np.testing.assert_allclose(result, alpha - m1 - m2, atol=1e-12)

    def test_no_modes_unchanged(self):
        alpha = np.array([0.010, 0.005, 0.003])
        result = remove_torsional_alpha_contributions(alpha, [])
        np.testing.assert_allclose(result, alpha[:3])

    def test_too_short_alpha_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            remove_torsional_alpha_contributions(np.array([0.1, 0.2]), [])

    def test_too_short_mode_alpha_raises(self):
        alpha = np.array([0.1, 0.05, 0.03])
        with pytest.raises(ValueError, match="at least 3"):
            remove_torsional_alpha_contributions(alpha, [np.array([0.01, 0.005])])

    def test_only_first_3_components_used(self):
        alpha = np.array([0.010, 0.005, 0.003, 0.999])
        mode = np.array([0.002, 0.001, 0.0005, 0.999])
        result = remove_torsional_alpha_contributions(alpha, [mode])
        assert result.size == 3
        np.testing.assert_allclose(result, alpha[:3] - mode[:3], atol=1e-12)


# ── lam_uncertainty_contribution ─────────────────────────────────────────────

class TestLamUncertaintyContribution:
    def test_zero_rms_gives_zero_uncertainty(self):
        assert lam_uncertainty_contribution(0.0, 5) == 0.0

    def test_scales_with_rms(self):
        u1 = lam_uncertainty_contribution(1.0, 4)
        u2 = lam_uncertainty_contribution(2.0, 4)
        assert abs(u2 - 2 * u1) < 1e-12

    def test_scales_with_n_levels(self):
        u1 = lam_uncertainty_contribution(1.0, 1)
        u4 = lam_uncertainty_contribution(1.0, 4)
        assert abs(u4 - u1 / 2.0) < 1e-12

    def test_scale_factor(self):
        u1 = lam_uncertainty_contribution(1.0, 4, scale_factor=1.0)
        u2 = lam_uncertainty_contribution(1.0, 4, scale_factor=2.0)
        assert abs(u2 - 2 * u1) < 1e-12

    def test_negative_rms_raises(self):
        with pytest.raises(ValueError):
            lam_uncertainty_contribution(-0.1, 4)

    def test_zero_n_levels_raises(self):
        with pytest.raises(ValueError):
            lam_uncertainty_contribution(1.0, 0)


# ── combine_lam_corrections ───────────────────────────────────────────────────

class TestCombineLamCorrections:
    def test_no_corrections_leaves_B_unchanged(self):
        B = np.array([4.0, 0.8, 0.75])
        result = combine_lam_corrections(B)
        np.testing.assert_allclose(result["B_rigid_cm-1"], B)
        np.testing.assert_allclose(result["B_rovib_cm-1"], B)
        np.testing.assert_allclose(result["B_effective_cm-1"], B)
        assert result["corrections_applied"] == []

    def test_rovib_correction_applied(self):
        B = np.array([4.0, 0.8, 0.75])
        alpha_nt = np.array([0.01, 0.005, 0.003])
        result = combine_lam_corrections(B, alpha_nontorsional_cm1=alpha_nt)
        np.testing.assert_allclose(result["B_rovib_cm-1"], B + alpha_nt)
        assert "rovib_nontorsional" in result["corrections_applied"]

    def test_torsion_correction_applied(self):
        B = np.array([4.0, 0.8, 0.75])
        tc = np.array([0.02, 0.01, 0.005])
        result = combine_lam_corrections(B, torsion_correction_cm1=tc)
        np.testing.assert_allclose(result["B_effective_cm-1"], B + tc)
        assert "torsion_averaging" in result["corrections_applied"]

    def test_both_corrections_additive(self):
        B = np.array([4.0, 0.8, 0.75])
        alpha_nt = np.array([0.01, 0.005, 0.003])
        tc = np.array([0.02, 0.01, 0.005])
        result = combine_lam_corrections(B, alpha_nontorsional_cm1=alpha_nt, torsion_correction_cm1=tc)
        np.testing.assert_allclose(result["B_effective_cm-1"], B + alpha_nt + tc)
        assert len(result["corrections_applied"]) == 2

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            combine_lam_corrections(np.array([1.0, 1.0, 1.0]), source="bad_source")

    def test_source_stored(self):
        B = np.array([4.0, 0.8, 0.75])
        result = combine_lam_corrections(B, source="globally_fit")
        assert result["source"] == "globally_fit"


# ── lam_correction_report ─────────────────────────────────────────────────────

class TestLamCorrectionReport:
    def test_minimal_call(self):
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B)
        assert "B_rigid_cm-1" in report
        assert "lam_uncertainty_cm-1" in report
        assert "torsion_rms_cm-1" in report

    def test_with_torsion_avg_subtracts_from_rigid(self):
        B = np.array([4.0, 0.8, 0.75])
        B_avg = np.array([4.01, 0.81, 0.76])
        report = lam_correction_report(B, B_torsion_avg_cm1=B_avg)
        expected_eff = B + (B_avg - B)
        np.testing.assert_allclose(report["B_effective_cm-1"], expected_eff, atol=1e-12)

    def test_double_counting_prevention(self):
        """Subtracting torsional alpha from full alpha before rovib step."""
        B = np.array([4.0, 0.8, 0.75])
        alpha_full = np.array([0.020, 0.010, 0.005])
        alpha_tors = np.array([0.005, 0.002, 0.001])
        report = lam_correction_report(
            B,
            alpha_full_cm1=alpha_full,
            torsional_mode_alphas_cm1=[alpha_tors],
        )
        # Expected: B_rovib = B + (alpha_full - alpha_tors) * 0.5
        expected_alpha_nt = (alpha_full - alpha_tors) * 0.5
        np.testing.assert_allclose(report["B_rovib_cm-1"], B + expected_alpha_nt, atol=1e-12)

    def test_uncertainty_propagated(self):
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B, torsion_rms_cm1=0.1, n_torsion_levels=4)
        expected_unc = 0.1 / np.sqrt(4)
        assert abs(report["lam_uncertainty_cm-1"] - expected_unc) < 1e-12

    def test_all_required_keys(self):
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B, torsion_rms_cm1=0.05, n_torsion_levels=6)
        for k in ("B_rigid_cm-1", "B_rovib_cm-1", "B_effective_cm-1",
                  "source", "corrections_applied",
                  "lam_uncertainty_cm-1", "torsion_rms_cm-1"):
            assert k in report


# ── format_lam_report_for_summary ─────────────────────────────────────────────

class TestFormatLamReportForSummary:
    def test_arrays_become_lists(self):
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B)
        formatted = format_lam_report_for_summary(report)
        assert isinstance(formatted["B_rigid_cm-1"], list)
        assert isinstance(formatted["B_effective_cm-1"], list)

    def test_scalars_unchanged(self):
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B, torsion_rms_cm1=0.05)
        formatted = format_lam_report_for_summary(report)
        assert isinstance(formatted["torsion_rms_cm-1"], float)
        assert isinstance(formatted["source"], str)

    def test_json_serializable(self):
        import json
        B = np.array([4.0, 0.8, 0.75])
        report = lam_correction_report(B, torsion_rms_cm1=0.1, n_torsion_levels=3)
        formatted = format_lam_report_for_summary(report)
        json.dumps(formatted)  # should not raise
