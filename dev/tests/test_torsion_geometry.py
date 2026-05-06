"""Tests for backend/torsion_geometry.py (Phase 7: geometry coupling)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_geometry import (
    compute_F_rho_from_geometry,
    torsion_geometry_jacobian,
    update_spec_F_rho,
)
from backend.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _methanol_like_geometry():
    """Simplified methanol-like geometry: C-C axis with 3 H atoms in top.

    O is placed off the C-C axis so I_total > I_alpha and rho < 1.
    """
    # Rotation axis: C1 (axis frame end) to C2 (axis top end), along x.
    # O is bonded to C1 but sits above the axis (non-zero y offset).
    coords = np.array([
        [1.430,  1.200, 0.000],   # 0: O (frame, off-axis — perpendicular distance 1.2 Ang)
        [1.430,  0.000, 0.000],   # 1: C1 (axis, frame end)
        [2.890,  0.000, 0.000],   # 2: C2 (axis, top end)
        [3.310,  1.027, 0.000],   # 3: H1 (top)
        [3.310, -0.514, 0.890],   # 4: H2 (top)
        [3.310, -0.514,-0.890],   # 5: H3 (top)
    ])
    masses = np.array([15.999, 12.011, 12.011, 1.008, 1.008, 1.008])
    top_indices = [3, 4, 5]
    axis_indices = (1, 2)
    return coords, masses, top_indices, axis_indices


def _make_spec():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: -100.0}, units="cm-1")
    return TorsionHamiltonianSpec(F=27.6, rho=0.81, A=4.0, B=0.8, C=0.75,
                                  potential=pot, n_basis=8, units="cm-1")


# ── compute_F_rho_from_geometry ───────────────────────────────────────────────

class TestComputeFRhoFromGeometry:
    def test_returns_positive_F(self):
        coords, masses, top, axis = _methanol_like_geometry()
        F, rho = compute_F_rho_from_geometry(coords, masses, top, axis)
        assert F > 0.0

    def test_rho_between_zero_and_one(self):
        coords, masses, top, axis = _methanol_like_geometry()
        F, rho = compute_F_rho_from_geometry(coords, masses, top, axis)
        assert 0.0 < rho < 1.0

    def test_larger_top_gives_smaller_F(self):
        """Heavier/larger top → larger I_alpha → smaller F."""
        coords, masses, top, axis = _methanol_like_geometry()
        F_light, _ = compute_F_rho_from_geometry(coords, masses, top, axis)
        heavy_masses = masses.copy()
        heavy_masses[[3, 4, 5]] *= 4.0  # heavier H
        F_heavy, _ = compute_F_rho_from_geometry(coords, heavy_masses, top, axis)
        assert F_heavy < F_light

    def test_symmetric_H_positions_give_equal_contributions(self):
        """Three H atoms at equal distance from axis should give equal I contributions."""
        # Build a perfect C3 top: 3 H equidistant from z-axis
        r = 1.0  # Angstrom perpendicular to z
        coords = np.array([
            [0.0, 0.0, 0.0],   # 0: axis atom 1
            [0.0, 0.0, 2.0],   # 1: axis atom 2
            [r,   0.0, 1.0],   # 2: H1
            [r * np.cos(2*np.pi/3), r * np.sin(2*np.pi/3), 1.0],  # H2
            [r * np.cos(4*np.pi/3), r * np.sin(4*np.pi/3), 1.0],  # H3
        ])
        masses = np.array([12.0, 12.0, 1.008, 1.008, 1.008])
        top = [2, 3, 4]
        axis = (0, 1)
        F, rho = compute_F_rho_from_geometry(coords, masses, top, axis)
        # Expected I_alpha = 3 * 1.008 * r^2
        expected_I = 3.0 * 1.008 * r**2
        expected_F = 16.857629206 / expected_I
        assert abs(F - expected_F) < 1e-6

    def test_bad_coords_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_F_rho_from_geometry(
                np.ones((5, 2)), np.ones(5), [2, 3, 4], (0, 1)
            )

    def test_mass_length_mismatch_raises(self):
        coords, masses, top, axis = _methanol_like_geometry()
        with pytest.raises(ValueError, match="length"):
            compute_F_rho_from_geometry(coords, masses[:3], top, axis)

    def test_coincident_axis_atoms_raises(self):
        coords, masses, top, _ = _methanol_like_geometry()
        with pytest.raises(ValueError):
            compute_F_rho_from_geometry(coords, masses, top, (1, 1))

    def test_empty_top_raises(self):
        coords, masses, _, axis = _methanol_like_geometry()
        with pytest.raises(ValueError, match="non-empty"):
            compute_F_rho_from_geometry(coords, masses, [], axis)

    def test_out_of_range_top_index_raises(self):
        coords, masses, top, axis = _methanol_like_geometry()
        with pytest.raises(IndexError):
            compute_F_rho_from_geometry(coords, masses, [99], axis)

    def test_formula_F_units(self):
        """F = 16.857629206 / I_alpha, verify unit constant for known geometry."""
        # Single H atom at distance 1 Ang from axis
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 1.0]])
        masses = np.array([12.0, 12.0, 1.008])
        F, _ = compute_F_rho_from_geometry(coords, masses, [2], (0, 1))
        expected = 16.857629206 / (1.008 * 1.0**2)
        assert abs(F - expected) < 1e-6


# ── update_spec_F_rho ─────────────────────────────────────────────────────────

class TestUpdateSpecFRho:
    def test_returns_new_spec_with_updated_F_rho(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        new_spec = update_spec_F_rho(spec, coords, masses, top, axis)
        F_geom, rho_geom = compute_F_rho_from_geometry(coords, masses, top, axis)
        assert abs(new_spec.F - F_geom) < 1e-10
        assert abs(new_spec.rho - rho_geom) < 1e-10

    def test_does_not_mutate_original(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        orig_F = spec.F
        orig_rho = spec.rho
        update_spec_F_rho(spec, coords, masses, top, axis)
        assert spec.F == orig_F
        assert spec.rho == orig_rho

    def test_other_spec_fields_preserved(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        new_spec = update_spec_F_rho(spec, coords, masses, top, axis)
        assert new_spec.n_basis == spec.n_basis
        assert new_spec.A == spec.A
        assert new_spec.potential.vcos == spec.potential.vcos


# ── torsion_geometry_jacobian ─────────────────────────────────────────────────

class TestTorsionGeometryJacobian:
    def _requests(self, n=3):
        return [{"J": 0, "K": 0, "level_index": i} for i in range(n)]

    def test_output_shape(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        reqs = self._requests(3)
        J = torsion_geometry_jacobian(spec, coords, masses, top, axis, reqs)
        assert J.shape == (3, 3 * coords.shape[0])

    def test_jacobian_finite(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        reqs = self._requests(2)
        J = torsion_geometry_jacobian(spec, coords, masses, top, axis, reqs)
        assert np.all(np.isfinite(J))

    def test_frame_atoms_have_near_zero_columns(self):
        """Moving axis-defining frame atoms should also affect F/rho but less so."""
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        reqs = self._requests(2)
        J = torsion_geometry_jacobian(spec, coords, masses, top, axis, reqs)
        # Top atom columns (atoms 3,4,5 → columns 9-17) should be non-zero
        top_cols = np.concatenate([np.arange(3*t, 3*t+3) for t in top])
        assert np.any(np.abs(J[:, top_cols]) > 1e-10)

    def test_empty_requests(self):
        coords, masses, top, axis = _methanol_like_geometry()
        spec = _make_spec()
        J = torsion_geometry_jacobian(spec, coords, masses, top, axis, [])
        assert J.shape == (0, 3 * coords.shape[0])
