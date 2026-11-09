"""Tests for backend/torsion_symmetry.py (Phase 9: symmetry and tunneling)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion.torsion_symmetry import (
    _spin_function_counts,
    c3_symmetry_block_energies,
    nuclear_spin_weights,
    predict_tunneling_splitting,
    species_label,
    symmetry_purity_table,
    symmetry_selection_rules,
    tunneling_splitting_to_csv_rows,
    wang_transformation_c3,
)
from backend.torsion.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec


def _make_c3_spec(F=27.6, rho=0.0, V3=-186.8, n_basis=10):
    """Spec with a pure C3 potential (only Vcos_3 harmonic)."""
    pot = TorsionFourierPotential(v0=186.1, vcos={3: V3}, units="cm-1")
    return TorsionHamiltonianSpec(F=F, rho=rho, A=4.25, B=0.823, C=0.793,
                                  potential=pot, n_basis=n_basis, units="cm-1")



# ── Nuclear-spin weights, derived rather than tabulated ──────────────────────

class TestNuclearSpinWeightDerivation:
    """The two anchors the module's derivation comment claims.

    The character projection is only trustworthy if it reproduces values that
    were established independently of it. These are those values.
    """

    def test_reproduces_the_c2_weights_this_module_already_had(self):
        # ANCHOR 1. The old hard-coded C2 entry was A:B = 1:3, and it was
        # right: two protons give one singlet (symmetric under exchange, so it
        # pairs with the antisymmetric rotational species) and three triplet
        # states. The derivation has to land on exactly that.
        assert nuclear_spin_weights(2, spin=0.5) == {"A": 1, "B": 3}

    def test_reproduces_published_methanol_c3_weights(self):
        # ANCHOR 2. Three protons: an I=3/2 quartet (4 states, totally
        # symmetric) and two I=1/2 doublets (2 each, one per E component).
        # A:E = 4:2 per component is the methanol convention; the old C3
        # table said 1:2, inverting which modification is heavier.
        w = nuclear_spin_weights(3, spin=0.5)
        assert w["A"] == 4
        assert w["E1"] == 2 and w["E2"] == 2
        assert w["E"] == 2

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_counts_exhaust_the_spin_space(self, n):
        # Every product spin state belongs to exactly one species, so the
        # counts must sum to (2I+1)^n. This is the invariant that would catch
        # a sign or normalisation slip in the projection.
        counts = _spin_function_counts(n, spin=0.5)
        assert sum(counts.values()) == 2 ** n
        assert all(c >= 0 for c in counts.values())

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_conjugate_species_have_equal_counts(self, n):
        # a_k and a_{n-k} are complex conjugates of a real quantity, so the
        # two halves of a degenerate E pair must carry the same weight.
        counts = _spin_function_counts(n, spin=0.5)
        for k in range(1, n):
            assert counts[k] == counts[(-k) % n]

    @pytest.mark.parametrize("n,spin", [(3, 1.0), (2, 1.0), (3, 1.5)])
    def test_higher_spin_also_exhausts_the_space(self, n, spin):
        # Deuterated tops (I=1) and heavier nuclei go through the same formula.
        d = int(round(2 * spin + 1))
        assert sum(_spin_function_counts(n, spin=spin).values()) == d ** n

    def test_weights_cover_every_species_of_the_fold(self):
        for n in (2, 3, 4, 6):
            w = nuclear_spin_weights(n, spin=0.5)
            for k in range(n):
                assert species_label(k, n) in w

    def test_fermion_statistics_pick_the_right_total_species(self):
        # An n-cycle is even iff n is odd, so the total wavefunction carries
        # k=0 for odd n and k=n/2 for even n. A consequence: for odd n the
        # A torsional species takes the totally symmetric spin count, while
        # for even n it is B that does.
        c3 = _spin_function_counts(3, spin=0.5)
        assert nuclear_spin_weights(3, spin=0.5)["A"] == c3[0]
        c2 = _spin_function_counts(2, spin=0.5)
        assert nuclear_spin_weights(2, spin=0.5)["B"] == c2[0]

    def test_species_label_names_b_only_for_even_folds(self):
        assert species_label(0, 3) == "A"
        assert species_label(1, 2) == "B"
        assert species_label(3, 6) == "B"
        assert species_label(1, 3) == "E1"
        assert species_label(2, 3) == "E2"
        # Odd folds have no k = n/2, so no B.
        assert "B" not in {species_label(k, 5) for k in range(5)}

    def test_rejects_spinless_nuclei(self):
        with pytest.raises(ValueError):
            _spin_function_counts(3, spin=0.0)


# ── wang_transformation_c3 ────────────────────────────────────────────────────

class TestWangTransformationC3:
    def test_unitary(self):
        from backend.torsion.torsion_hamiltonian import basis_m_values
        m = basis_m_values(5)
        W = wang_transformation_c3(m)
        assert W.shape == (m.size, m.size)
        # Permutation matrix: W @ W^H = I
        np.testing.assert_allclose(W @ W.conj().T, np.eye(m.size), atol=1e-12)

    def test_groups_by_residue(self):
        from backend.torsion.torsion_hamiltonian import basis_m_values
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
        from backend.torsion.torsion_hamiltonian import basis_m_values, build_ram_lite_hamiltonian
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

    def test_rotor_fold_below_two_is_undefined(self):
        # A one-fold "rotor" has no torsional symmetry species to speak of.
        result = symmetry_selection_rules("A", "A", rotor_fold=1)
        assert result["allowed"] is None

    def test_unknown_label_for_fold(self):
        # A C2 rotor has no E species: k = n/2 is the non-degenerate B.
        result = symmetry_selection_rules("E1", "A", rotor_fold=2)
        assert result["allowed"] is None

    def test_arbitrary_fold_is_handled(self):
        # The k = +/- k' rule needs no per-fold table, so an uncommon fold
        # such as C5 works: A <-> A is allowed, A <-> E1 is not.
        assert symmetry_selection_rules("A", "A", rotor_fold=5)["allowed"] is True
        assert symmetry_selection_rules("A", "E1", rotor_fold=5)["allowed"] is False

    def test_degenerate_pair_allowed_but_distinct_pairs_not(self):
        # C6: E1 and E5 are complex conjugates (one species), E1 and E2 are not.
        assert symmetry_selection_rules("E1", "E5", rotor_fold=6)["allowed"] is True
        assert symmetry_selection_rules("E1", "E2", rotor_fold=6)["allowed"] is False

    def test_case_insensitive(self):
        result = symmetry_selection_rules("a", "e", rotor_fold=3)
        assert result["allowed"] is False

    def test_c2_A_A_allowed(self):
        result = symmetry_selection_rules("A", "A", rotor_fold=2)
        assert result["allowed"] is True

    def test_c2_B_B_allowed(self):
        result = symmetry_selection_rules("B", "B", rotor_fold=2)
        assert result["allowed"] is True

    def test_c2_A_B_forbidden(self):
        result = symmetry_selection_rules("A", "B", rotor_fold=2)
        assert result["allowed"] is False
        result = symmetry_selection_rules("B", "A", rotor_fold=2)
        assert result["allowed"] is False

    def test_c2_unknown_label(self):
        result = symmetry_selection_rules("E1", "A", rotor_fold=2)
        assert result["allowed"] is None


# ── C2 symmetry mode (2-fold rotors) ─────────────────────────────────────────

def _make_c2_spec(V2=600.0, n_basis=20):
    """Spec with a pure C2 potential (only Vcos_2 harmonic)."""
    pot = TorsionFourierPotential(v0=float(V2), vcos={2: -float(V2)}, units="cm-1")
    return TorsionHamiltonianSpec(F=10.0, rho=0.0, potential=pot, n_basis=n_basis, units="cm-1")


class TestC2SymmetryMode:
    def test_c2_labels_are_A_or_B(self):
        from backend.torsion.torsion_hamiltonian import solve_ram_lite_levels
        res = solve_ram_lite_levels(_make_c2_spec(), J=0, K=0, symmetry_mode="c2", n_levels=6)
        assert set(res["symmetry_labels"]) <= {"A", "B"}

    def test_c2_blocks_A_and_B(self):
        from backend.torsion.torsion_hamiltonian import solve_ram_lite_levels
        res = solve_ram_lite_levels(
            _make_c2_spec(), J=0, K=0, symmetry_mode="c2", return_blocks=True
        )
        assert set(res["symmetry_blocks"].keys()) == {"A", "B"}

    def test_c2_ground_pair_nearly_degenerate_below_barrier(self):
        """Deep below a high V2 barrier the A/B tunneling pair is nearly degenerate."""
        from backend.torsion.torsion_hamiltonian import solve_ram_lite_levels
        res = solve_ram_lite_levels(
            _make_c2_spec(V2=2000.0), J=0, K=0, symmetry_mode="c2", n_levels=2
        )
        e = res["energies_cm-1"]
        assert abs(e[1] - e[0]) < 1e-3
        assert set(res["symmetry_labels"][:2]) == {"A", "B"}

    def test_c2_line_list_enforces_selection_rules(self):
        from backend.torsion.torsion_intensities import compute_torsion_line_list
        lines = compute_torsion_line_list(
            _make_c2_spec(), J_values=[0, 1], K_values=[0],
            n_levels=4, symmetry_mode="c2", rotor_fold=2,
        )
        assert lines
        cross = [r for r in lines if r["symmetry_lo"] != r["symmetry_hi"]]
        assert cross
        for row in cross:
            assert row["allowed"] is False
            assert row["relative_intensity"] == 0.0

    def test_c2_mismatched_potential_warns(self):
        from backend.torsion.torsion_hamiltonian import solve_ram_lite_levels
        pot = TorsionFourierPotential(v0=100.0, vcos={3: -100.0}, units="cm-1")
        spec = TorsionHamiltonianSpec(F=10.0, n_basis=15, potential=pot, units="cm-1")
        res = solve_ram_lite_levels(spec, J=0, K=0, symmetry_mode="c2")
        assert any("non-2-fold" in w for w in res["warnings"])


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
