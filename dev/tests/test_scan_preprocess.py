"""Tests for backend/scan_preprocess.py (Phase 3: scan preprocessing)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.scan_preprocess import (
    deduplicate_endpoint,
    extend_by_symmetry,
    preprocess_scan,
    sort_scan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _methanol_segment(n: int = 13, barrier_cm1: float = 373.5, period: float = 2 * np.pi / 3):
    """One symmetry segment of a V3 potential."""
    phi = np.linspace(0.0, period, n, endpoint=True)
    e = 0.5 * barrier_cm1 * (1.0 - np.cos(3.0 * phi))
    return phi, e


# ── sort_scan ─────────────────────────────────────────────────────────────────

class TestSortScan:
    def test_already_sorted_unchanged(self):
        phi = np.array([0.0, 1.0, 2.0])
        e = np.array([0.0, 1.0, 0.5])
        phi2, e2 = sort_scan(phi, e)
        np.testing.assert_array_equal(phi2, phi)
        np.testing.assert_array_equal(e2, e)

    def test_reversed_order(self):
        phi = np.array([3.0, 2.0, 1.0, 0.0])
        e = np.array([10.0, 5.0, 2.0, 0.0])
        phi2, e2 = sort_scan(phi, e)
        np.testing.assert_array_equal(phi2, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(e2, [0.0, 2.0, 5.0, 10.0])

    def test_unsorted_random(self):
        rng = np.random.default_rng(0)
        phi = rng.uniform(0, 2 * np.pi, 20)
        e = rng.uniform(0, 100, 20)
        phi2, e2 = sort_scan(phi, e)
        assert np.all(np.diff(phi2) >= 0)
        assert phi2.size == 20

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            sort_scan(np.array([0.0, 1.0]), np.array([0.0]))

    def test_single_point(self):
        phi2, e2 = sort_scan(np.array([1.5]), np.array([3.0]))
        assert phi2[0] == 1.5

    def test_does_not_mutate_input(self):
        phi = np.array([2.0, 0.0, 1.0])
        e = np.array([1.0, 0.0, 0.5])
        phi_orig = phi.copy()
        sort_scan(phi, e)
        np.testing.assert_array_equal(phi, phi_orig)


# ── deduplicate_endpoint ──────────────────────────────────────────────────────

class TestDeduplicateEndpoint:
    def test_detects_exact_period_endpoint(self):
        phi = np.array([0.0, np.pi / 3, 2 * np.pi / 3, 2 * np.pi])
        e = np.array([0.0, 100.0, 100.0, 0.0])
        phi2, e2, removed = deduplicate_endpoint(phi, e, period_rad=2 * np.pi, tol_rad=0.05)
        assert removed is True
        assert phi2.size == 3
        assert phi2[-1] == pytest.approx(2 * np.pi / 3)

    def test_no_duplicate_when_not_periodic(self):
        phi = np.linspace(0.0, 5.5, 10)  # less than full period
        e = np.zeros(10)
        phi2, e2, removed = deduplicate_endpoint(phi, e, period_rad=2 * np.pi)
        assert removed is False
        assert phi2.size == 10

    def test_near_duplicate_within_tol(self):
        # phi[-1] = phi[0] + period - 0.01 → within default tol 0.05
        phi = np.array([0.0, 1.0, 2.0, 3.0, 6.27])  # 6.27 ≈ 2π - 0.013
        e = np.zeros(5)
        _, _, removed = deduplicate_endpoint(phi, e, period_rad=2 * np.pi, tol_rad=0.05)
        assert removed is True

    def test_boundary_outside_tol(self):
        phi = np.array([0.0, 1.0, 2.0, 3.0, 6.0])  # 6.0 far from 2π
        e = np.zeros(5)
        _, _, removed = deduplicate_endpoint(phi, e, period_rad=2 * np.pi, tol_rad=0.05)
        assert removed is False

    def test_single_point_no_removal(self):
        phi, e, removed = deduplicate_endpoint(np.array([0.0]), np.array([1.0]))
        assert removed is False
        assert phi.size == 1

    def test_threefold_period(self):
        # Methanol-like 3-fold scan: endpoint at 0 + 2π/3
        period = 2 * np.pi / 3
        phi = np.linspace(0.0, period, 13, endpoint=True)
        e = np.zeros(13)
        _, _, removed = deduplicate_endpoint(phi, e, period_rad=period, tol_rad=0.01)
        assert removed is True


# ── extend_by_symmetry ───────────────────────────────────────────────────────

class TestExtendBySymmetry:
    def test_n1_is_identity(self):
        phi, e = _methanol_segment(7)
        phi2, e2 = extend_by_symmetry(phi, e, 1)
        np.testing.assert_array_equal(phi2, phi)
        np.testing.assert_array_equal(e2, e)

    def test_n3_triples_points(self):
        # 5-point segment → 3 copies → ~15 points (minus 3 duplicates at boundaries)
        phi, e = _methanol_segment(5, period=2 * np.pi / 3)
        phi2, e2 = extend_by_symmetry(phi, e, 3, period_rad=2 * np.pi)
        # Should cover ~0 to 2π
        assert phi2[-1] > np.pi  # at least past half period
        assert phi2.size > phi.size

    def test_extended_is_sorted(self):
        phi, e = _methanol_segment(9, period=2 * np.pi / 3)
        phi2, e2 = extend_by_symmetry(phi, e, 3, period_rad=2 * np.pi)
        assert np.all(np.diff(phi2) > 0), "Extended scan must be strictly sorted"

    def test_energy_periodicity(self):
        # For exact V3 potential, extension should give same energy values
        phi_seg, e_seg = _methanol_segment(7, period=2 * np.pi / 3)
        phi_full, e_full = extend_by_symmetry(phi_seg, e_seg, 3, period_rad=2 * np.pi)
        # First 7 points of extended scan should match original (same phi)
        np.testing.assert_allclose(phi_full[:7], phi_seg, atol=1e-12)
        np.testing.assert_allclose(e_full[:7], e_seg, atol=1e-12)

    def test_invalid_symmetry_number_raises(self):
        with pytest.raises(ValueError, match="symmetry_number must be >= 1"):
            extend_by_symmetry(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            extend_by_symmetry(np.array([0.0, 1.0]), np.array([0.0]), 2)


# ── preprocess_scan ───────────────────────────────────────────────────────────

class TestPreprocessScan:
    def test_default_sorts_and_deduplicates(self):
        # Unsorted + duplicate endpoint
        phi = np.array([2 * np.pi, 0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        e = np.array([0.0, 0.0, 100.0, 200.0, 100.0])
        phi2, e2, info = preprocess_scan(phi, e)
        # Should be sorted
        assert np.all(np.diff(phi2) > 0)
        # duplicate endpoint 2π ≈ 0 + period → removed
        assert info["deduplicated"] is True
        assert info["sorted"] is True
        assert info["n_points_in"] == 5
        assert info["n_points_out"] == 4

    def test_no_preprocessing_flags(self):
        phi = np.array([2.0, 0.0, 1.0])
        e = np.array([1.0, 0.0, 0.5])
        phi2, e2, info = preprocess_scan(
            phi, e, do_sort=False, do_deduplicate=False, do_extend_by_symmetry=False
        )
        np.testing.assert_array_equal(phi2, phi)
        assert info["sorted"] is False
        assert info["deduplicated"] is False
        assert info["extended_by_symmetry"] is False

    def test_extend_by_symmetry_triples(self):
        phi_seg, e_seg = _methanol_segment(5, period=2 * np.pi / 3)
        phi2, e2, info = preprocess_scan(
            phi_seg, e_seg,
            symmetry_number=3,
            period_rad=2 * np.pi,
            do_deduplicate=True,
            do_extend_by_symmetry=True,
        )
        assert info["extended_by_symmetry"] is True
        assert phi2.size > phi_seg.size

    def test_warning_on_duplicate_removal(self):
        phi = np.array([0.0, 1.0, 2.0, 3.0, 2 * np.pi])
        e = np.zeros(5)
        _, _, info = preprocess_scan(phi, e, do_deduplicate=True)
        assert info["deduplicated"] is True
        assert any("endpoint" in w.lower() for w in info["warnings"])

    def test_info_keys_present(self):
        phi, e = _methanol_segment(5)
        _, _, info = preprocess_scan(phi, e)
        for k in ("sorted", "deduplicated", "extended_by_symmetry",
                  "symmetry_number", "n_points_in", "n_points_out", "warnings"):
            assert k in info

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            preprocess_scan(np.array([0.0, 1.0]), np.array([0.0]))

    def test_no_mutation_of_input(self):
        phi = np.array([2.0, 0.0, 1.0])
        e = np.array([2.0, 0.0, 1.0])
        phi_orig = phi.copy()
        e_orig = e.copy()
        preprocess_scan(phi, e)
        np.testing.assert_array_equal(phi, phi_orig)
        np.testing.assert_array_equal(e, e_orig)
