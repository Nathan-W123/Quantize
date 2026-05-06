"""Tests for backend/torsion_rot_hamiltonian.py (Phase 8: full torsion-rotation Hamiltonian)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_rot_hamiltonian import (
    _jk_asym_coupling,
    build_full_torsion_rotation_hamiltonian,
    compare_ram_lite_vs_full,
    solve_full_torsion_rotation_levels,
)
from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)


def _make_spec(F=27.6, rho=0.81, A=4.25, B=0.823, C=0.793, Vcos3=-186.8, n_basis=10):
    pot = TorsionFourierPotential(v0=186.1, vcos={3: Vcos3}, units="cm-1")
    return TorsionHamiltonianSpec(
        F=F, rho=rho, A=A, B=B, C=C, potential=pot, n_basis=n_basis, units="cm-1"
    )


def _symmetric_spec(F=5.0, rho=0.0, A=1.0, B=1.0, C=1.0, n_basis=5):
    """Symmetric-top spec: B == C so asym terms vanish."""
    pot = TorsionFourierPotential(units="cm-1")
    return TorsionHamiltonianSpec(F=F, rho=rho, A=A, B=B, C=C, potential=pot, n_basis=n_basis)


# ── _jk_asym_coupling ─────────────────────────────────────────────────────────

class TestJKAsymCoupling:
    def test_zero_for_k_at_boundary(self):
        # <J=1, K=2|...|J=1, K=0> → K+2=2 > J=1, should give 0
        assert _jk_asym_coupling(1, 0) == pytest.approx(0.0)

    def test_positive_for_valid_K(self):
        assert _jk_asym_coupling(2, 0) > 0.0

    def test_J0_always_zero(self):
        assert _jk_asym_coupling(0, 0) == pytest.approx(0.0)

    def test_formula_J2_K0(self):
        # J=2, K=0: sqrt([6-0][6-2]) = sqrt(6*4) = sqrt(24)
        expected = float(np.sqrt(6 * 4))
        assert _jk_asym_coupling(2, 0) == pytest.approx(expected, rel=1e-10)

    def test_negative_K_handled(self):
        # Coupling at K=-1: sqrt([J(J+1)-(-1)(0)] * [J(J+1)-0*1])
        # For J=2, K=-2: t1 = 6-(-2)(-1)=6-2=4, t2=6-(-1)(0)=6>0 → sqrt(24)
        val = _jk_asym_coupling(2, -2)
        assert val >= 0.0


# ── build_full_torsion_rotation_hamiltonian ───────────────────────────────────

class TestBuildFullHamiltonian:
    def test_hermitian(self):
        spec = _make_spec()
        H, K_vals, m_vals, _ = build_full_torsion_rotation_hamiltonian(spec, J=1)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_correct_dimension(self):
        spec = _make_spec(n_basis=6)
        J = 2
        H, K_vals, m_vals, _ = build_full_torsion_rotation_hamiltonian(spec, J=J)
        expected_dim = (2 * J + 1) * (2 * 6 + 1)
        assert H.shape == (expected_dim, expected_dim)
        assert K_vals.size == 2 * J + 1
        assert m_vals.size == 2 * 6 + 1

    def test_J0_matches_ram_lite(self):
        """For J=0 there is only K=0; full solution must match RAM-lite."""
        spec = _symmetric_spec()
        H_full, _, m_vals, _ = build_full_torsion_rotation_hamiltonian(spec, J=0)
        e_full = np.sort(np.linalg.eigvalsh(H_full).real)

        rl = solve_ram_lite_levels(spec, J=0, K=0)
        e_rl = np.sort(rl["energies_cm-1"])
        n = min(len(e_full), len(e_rl), 5)
        np.testing.assert_allclose(e_full[:n], e_rl[:n], atol=1e-8)

    def test_symmetric_top_no_asym_coupling(self):
        """B==C → asymmetric coupling factor = 0; off-diagonal K blocks should be zero."""
        spec = _symmetric_spec(B=1.0, C=1.0)
        H, K_vals, m_vals, _ = build_full_torsion_rotation_hamiltonian(spec, J=2)
        n_m = m_vals.size
        n_K = K_vals.size
        # Check K+2 off-diagonal blocks are zero
        for ki in range(n_K - 2):
            rs = ki * n_m
            re = rs + n_m
            cs = (ki + 2) * n_m
            ce = cs + n_m
            assert np.allclose(H[rs:re, cs:ce], 0.0, atol=1e-12), \
                f"K block {ki} to {ki+2} should be zero for symmetric top"

    def test_real_eigenvalues(self):
        spec = _make_spec()
        H, _, _, _ = build_full_torsion_rotation_hamiltonian(spec, J=1)
        e = np.linalg.eigvalsh(H)
        assert np.all(np.isfinite(e))
        assert np.allclose(e.imag, 0.0, atol=1e-10)


# ── solve_full_torsion_rotation_levels ────────────────────────────────────────

class TestSolveFullTorsionRotation:
    def test_sorted_energies(self):
        spec = _make_spec()
        out = solve_full_torsion_rotation_levels(spec, J=1, n_levels=10)
        e = out["energies_cm-1"]
        assert np.all(np.diff(e) >= -1e-10), "Energies must be sorted ascending"

    def test_result_keys(self):
        spec = _make_spec()
        out = solve_full_torsion_rotation_levels(spec, J=1, n_levels=5)
        for k in ("J", "energies_cm-1", "Ka_labels", "Kc_labels",
                  "eigenvectors", "K_vals", "m_vals", "dim", "warnings"):
            assert k in out

    def test_n_levels_truncation(self):
        spec = _make_spec()
        out = solve_full_torsion_rotation_levels(spec, J=2, n_levels=8)
        assert len(out["energies_cm-1"]) == 8
        assert len(out["Ka_labels"]) == 8

    def test_Ka_in_valid_range(self):
        spec = _make_spec()
        J = 2
        out = solve_full_torsion_rotation_levels(spec, J=J, n_levels=10)
        Ka = out["Ka_labels"]
        assert np.all(Ka >= 0)
        assert np.all(Ka <= J)

    def test_J0_single_level_ground_state(self):
        """For J=0, full solution should be identical to RAM-lite K=0 ground state."""
        spec = _symmetric_spec()
        out_full = solve_full_torsion_rotation_levels(spec, J=0, n_levels=1)
        rl = solve_ram_lite_levels(spec, J=0, K=0, n_levels=1)
        assert abs(out_full["energies_cm-1"][0] - rl["energies_cm-1"][0]) < 1e-8

    def test_asym_coupling_changes_energies(self):
        """Asymmetric top (B >> C) should give meaningfully different energies than symmetric case."""
        J = 2
        # Use a large B-C difference so the coupling effect is clearly observable
        spec_asym = _make_spec(B=1.5, C=0.5)   # B-C = 1.0 → strong asymmetry
        spec_sym  = _make_spec(B=1.0, C=1.0)   # B-C = 0 → no coupling
        e_asym = solve_full_torsion_rotation_levels(spec_asym, J, n_levels=10)["energies_cm-1"]
        e_sym  = solve_full_torsion_rotation_levels(spec_sym,  J, n_levels=10)["energies_cm-1"]
        # With B-C=1.0 vs 0, the spectra should differ by more than 0.1 cm^-1
        assert np.max(np.abs(e_asym - e_sym)) > 0.1


# ── compare_ram_lite_vs_full ──────────────────────────────────────────────────

class TestCompareRamLiteVsFull:
    def test_returns_required_keys(self):
        spec = _make_spec()
        result = compare_ram_lite_vs_full(spec, J=1, K=0, n_levels=4)
        for k in ("ram_lite_energies_cm-1", "full_energies_cm-1",
                  "max_diff_cm-1", "rms_diff_cm-1", "n_compared"):
            assert k in result

    def test_symmetric_top_J0_exact_match(self):
        """For J=0 the only block is K=0, so RAM-lite and full must agree exactly."""
        spec = _symmetric_spec()
        result = compare_ram_lite_vs_full(spec, J=0, K=0, n_levels=4)
        assert result["rms_diff_cm-1"] < 1e-8

    def test_diff_finite(self):
        spec = _make_spec()
        result = compare_ram_lite_vs_full(spec, J=1, K=0, n_levels=4)
        assert np.isfinite(result["max_diff_cm-1"])
        assert np.isfinite(result["rms_diff_cm-1"])
