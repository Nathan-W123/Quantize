"""Tests for backend/torsion_symmetry.py (Phase 9: symmetry and tunneling)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_symmetry import (
    c3_symmetry_block_energies,
    predict_tunneling_splitting,
    symmetry_purity_table,
    symmetry_selection_rules,
    tunneling_splitting_to_csv_rows,
    wang_transformation_c3,
)
from backend.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec


def _make_c3_spec(F=27.6, rho=0.0, V3=-186.8, n_basis=10):
    """Spec with a pure C3 potential (only Vcos_3 harmonic)."""
    pot = TorsionFourierPotential(v0=186.1, vcos={3: V3}, units="cm-1")
    return TorsionHamiltonianSpec(F=F, rho=rho, A=4.25, B=0.823, C=0.793,
                                  potential=pot, n_basis=n_basis, units="cm-1")


# ── wang_transformation_c3 ────────────────────────────────────────────────────

class TestWangTransformationC3:
    def test_unitary(self):
        from backend.torsion_hamiltonian import basis_m_values
        m = basis_m_values(5)
        W = wang_transformation_c3(m)
        assert W.shape == (m.size, m.size)
        # Permutation matrix: W @ W^H = I
        np.testing.assert_allclose(W @ W.conj().T, np.eye(m.size), atol=1e-12)

    def test_groups_by_residue(self):
        from backend.torsion_hamiltonian import basis_m_values
        m = basis_m_values(4)  # m = -4..4
        W = wang_transformation_c3(m)
        # Rows mapped to old indices should group A (mod3=0), E1 (mod3=1), E2 (mod3=2)
        n_A = np.sum(np.mod(m, 3) == 0)
        n_E1 = np.sum(np.mod(m, 3) == 1)
        # First n_A rows should map to m%3==0 columns
        for new_i in range(n_A):
            old_i = int(np.argmax(np.abs(W[new_i])))
            assert int(m[old_i]) % 3 == 0

    def test_block_diagonalizes_c3_hamiltonian(self):
        """Applying Wang transform to a pure-C3 Hamiltonian gives block-diagonal result."""
        from backend.torsion_hamiltonian import basis_m_values, build_ram_lite_hamiltonian
        spec = _make_c3_spec(rho=0.0)
        H, m_vals, _ = build_ram_lite_hamiltonian(spec, J=0, K=0)
        W = wang_transformation_c3(m_vals)
        H_block = W @ H @ W.conj().T

        n_m = m_vals.size
        n_A = np.sum(np.mod(m_vals, 3) == 0)
        n_E1 = np.sum(np.mod(m_vals, 3) == 1)

        # Off-diagonal blocks between A and E1 should be ~0
        A_block_rows = slice(0, n_A)
        E1_block_rows = slice(n_A, n_A + n_E1)
        np.testing.assert_allclose(
            np.abs(H_block[A_block_rows, E1_block_rows]), 0.0, atol=1e-10
        )


# ── c3_symmetry_block_energies ────────────────────────────────────────────────

class TestC3SymmetryBlockEnergies:
    def test_returns_required_keys(self):
        spec = _make_c3_spec()
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=4)
        for k in ("A", "E1", "E2", "warnings"):
            assert k in out

    def test_energies_sorted(self):
        spec = _make_c3_spec()
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=5)
        for label in ("A", "E1", "E2"):
            e = out[label]["energies_cm-1"]
            if len(e) > 1:
                assert np.all(np.diff(e) >= -1e-10), f"{label} block not sorted"

    def test_truncation(self):
        spec = _make_c3_spec()
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=3)
        for label in ("A", "E1", "E2"):
            e = out[label]["energies_cm-1"]
            assert len(e) <= 3

    def test_e1_e2_degenerate_for_symmetric_potential(self):
        """Pure C3 potential → E1 and E2 ground state energies should match."""
        spec = _make_c3_spec(rho=0.0)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=3)
        E1 = out["E1"]["energies_cm-1"]
        E2 = out["E2"]["energies_cm-1"]
        if len(E1) > 0 and len(E2) > 0:
            assert abs(E1[0] - E2[0]) < 0.1  # nearly degenerate for symmetric potential

    def test_a_block_lowest_energy_below_E_for_high_barrier(self):
        """For a high barrier, A species should be lower in energy (standard result)."""
        spec = _make_c3_spec(V3=-500.0)  # large barrier
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=1)
        E_A = out["A"]["energies_cm-1"]
        E_E = out["E1"]["energies_cm-1"]
        if len(E_A) > 0 and len(E_E) > 0:
            assert E_A[0] <= E_E[0]


# ── predict_tunneling_splitting ───────────────────────────────────────────────

class TestPredictTunnelingSplitting:
    def test_returns_list_of_dicts(self):
        spec = _make_c3_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=3)
        assert isinstance(rows, list)
        assert len(rows) == 3

    def test_required_keys(self):
        spec = _make_c3_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=2)
        for row in rows:
            for k in ("vt", "J", "K", "E_A_cm-1", "E_E_cm-1", "splitting_cm-1", "splitting_MHz"):
                assert k in row

    def test_splitting_consistent_mhz(self):
        _MHZ_PER_CM1 = 29979.2458
        spec = _make_c3_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=2)
        for row in rows:
            expected_mhz = row["splitting_cm-1"] * _MHZ_PER_CM1
            assert abs(row["splitting_MHz"] - expected_mhz) < 1e-6

    def test_vt_index_sequential(self):
        spec = _make_c3_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=4)
        for i, row in enumerate(rows):
            assert row["vt"] == i

    def test_splitting_nonzero_below_barrier(self):
        """Tunneling below the barrier should give a finite (non-zero) A/E splitting."""
        spec = _make_c3_spec(V3=-186.8)
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
        assert len(rows) > 0
        split = abs(rows[0]["splitting_cm-1"])
        assert split > 1e-6


# ── symmetry_selection_rules ──────────────────────────────────────────────────

class TestSymmetrySelectionRules:
    def test_A_A_allowed(self):
        result = symmetry_selection_rules("A", "A", rotor_fold=3)
        assert result["allowed"] is True

    def test_E_E_allowed(self):
        result = symmetry_selection_rules("E", "E", rotor_fold=3)
        assert result["allowed"] is True

    def test_E1_E2_allowed(self):
        result = symmetry_selection_rules("E1", "E2", rotor_fold=3)
        assert result["allowed"] is True

    def test_A_E_forbidden(self):
        result = symmetry_selection_rules("A", "E", rotor_fold=3)
        assert result["allowed"] is False

    def test_E_A_forbidden(self):
        result = symmetry_selection_rules("E1", "A", rotor_fold=3)
        assert result["allowed"] is False

    def test_unknown_rotor_fold(self):
        result = symmetry_selection_rules("A", "A", rotor_fold=2)
        assert result["allowed"] is None

    def test_case_insensitive(self):
        result = symmetry_selection_rules("a", "e", rotor_fold=3)
        assert result["allowed"] is False


# ── symmetry_purity_table ─────────────────────────────────────────────────────

class TestSymmetryPurityTable:
    def test_length_matches_n_levels(self):
        spec = _make_c3_spec()
        rows = symmetry_purity_table(spec, J=0, K=0, n_levels=5)
        assert len(rows) == 5

    def test_required_keys(self):
        spec = _make_c3_spec()
        rows = symmetry_purity_table(spec, J=0, K=0, n_levels=3)
        for row in rows:
            for k in ("level_index", "energy_cm-1", "symmetry_label", "symmetry_sublabel", "purity"):
                assert k in row

    def test_purity_between_0_and_1(self):
        spec = _make_c3_spec()
        rows = symmetry_purity_table(spec, J=0, K=0, n_levels=4)
        for row in rows:
            assert 0.0 <= row["purity"] <= 1.0 + 1e-10

    def test_high_barrier_gives_high_purity(self):
        """Very high barrier → eigenstates nearly pure A or E."""
        spec = _make_c3_spec(V3=-1000.0)
        rows = symmetry_purity_table(spec, J=0, K=0, n_levels=3)
        for row in rows:
            assert row["purity"] > 0.9


# ── tunneling_splitting_to_csv_rows ──────────────────────────────────────────

class TestTunnelingSplittingToCsvRows:
    def test_formats_floats_as_strings(self):
        spec = _make_c3_spec()
        splits = predict_tunneling_splitting(spec, J=0, K=0, n_levels=2)
        csv_rows = tunneling_splitting_to_csv_rows(splits)
        for row in csv_rows:
            assert isinstance(row["splitting_cm-1"], str)
            assert isinstance(row["splitting_MHz"], str)
            assert "." in row["splitting_cm-1"]
