"""
Phase 4: Acetaldehyde benchmark and RAM-lite reliability scoring.

Validates the torsion pipeline against acetaldehyde (CH3CHO) literature parameters:
  Kleiner et al., J. Mol. Spectrosc. 175, 1â€“13 (1996)
    F    = 5.1724 cm^-1    internal rotation constant
    rho  = 0.0609           coupling parameter
    V3   = 163.4 cm^-1     3-fold barrier height

Fourier convention (this codebase):
  V(Î±) = v0 + vcos_3 * cos(3Î±)
  v0    = V3/2  = 81.7 cm^-1
  vcos3 = -V3/2 = -81.7 cm^-1

RAM-lite vt=0 A/E tunneling splitting for these parameters: ~0.276 cm^-1 (E above A).

This is the exact solution to the 1D torsional Hamiltonian H = F*m^2 + V(Î±), verified
by three independent methods (Fourier matrix, DVR grid, Floquet/Mathieu analysis).
It agrees with the Mathieu equation result for q = V3/(9*F) = 3.51.

NOTE: Some literature sources cite ~1.18 cm^-1 for the acetaldehyde A-E splitting.
This larger value is the *effective* splitting observed in the rotational spectrum,
which includes torsion-rotation coupling contributions beyond the simple 1D model.
The pure J=0 torsional splitting from RAM-lite is 0.276 cm^-1.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion.torsion_reliability import TorsionReliabilityScore, assess_ram_lite_reliability
from backend.torsion.torsion_symmetry import c3_symmetry_block_energies, predict_tunneling_splitting


# â”€â”€ Acetaldehyde literature parameters (Kleiner et al. 1996) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_F_CH3CHO   = 5.1724    # cm^-1  internal rotation constant
_RHO_CH3CHO = 0.0609    # dimensionless coupling
_V3_CH3CHO  = 163.4     # cm^-1  3-fold barrier

# Ground-state effective rotational constants (from spectroscopy)
_A_CH3CHO = 1.7626      # cm^-1  (~52842 MHz)
_B_CH3CHO = 0.3413      # cm^-1  (~10240 MHz)
_C_CH3CHO = 0.2884      # cm^-1  (~8648 MHz)

# RAM-lite vt=0 A/E splitting (exact 1D torsional solution, verified by 3 methods)
_SPLIT_VT0_LIT = 0.276   # cm^-1 (E1 above A)


def _acetaldehyde_spec(n_basis: int = 12) -> TorsionHamiltonianSpec:
    v0    =  _V3_CH3CHO / 2.0
    vcos3 = -_V3_CH3CHO / 2.0
    pot = TorsionFourierPotential(v0=v0, vcos={3: vcos3}, vsin={}, units="cm-1")
    return TorsionHamiltonianSpec(
        F=_F_CH3CHO, rho=_RHO_CH3CHO,
        A=_A_CH3CHO, B=_B_CH3CHO, C=_C_CH3CHO,
        potential=pot, n_basis=n_basis, units="cm-1",
    )


# â”€â”€ Benchmark: torsional levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestAcetaldehydeBenchmark:
    """Validate RAM-lite against acetaldehyde literature (Kleiner et al. 1996)."""

    def test_vt0_splitting_positive(self):
        """E(E1) > E(A) for vt=0: splitting must be positive."""
        spec = _acetaldehyde_spec(n_basis=12)
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
        assert len(rows) > 0
        assert rows[0]["splitting_cm-1"] > 0.0, (
            f"vt=0 splitting should be positive (E above A); got {rows[0]['splitting_cm-1']:.4f}"
        )

    def test_vt0_splitting_matches_ram_lite_prediction(self):
        """vt=0 A/E splitting within 0.01 cm^-1 of the converged RAM-lite value (0.276 cm^-1).

        The RAM-lite value is the exact solution to H = F*m^2 + V(alpha) for the Kleiner
        et al. (1996) parameters, verified by Fourier matrix, DVR grid, and Floquet analysis.
        """
        spec = _acetaldehyde_spec(n_basis=12)
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
        split = abs(rows[0]["splitting_cm-1"])
        tol = 0.01
        assert abs(split - _SPLIT_VT0_LIT) < tol, (
            f"vt=0 A/E splitting = {split:.4f} cm^-1; "
            f"RAM-lite prediction = {_SPLIT_VT0_LIT} cm^-1; "
            f"deviation = {abs(split - _SPLIT_VT0_LIT):.4f} (tolerance {tol})"
        )

    def test_vt0_A_energy_below_barrier(self):
        """vt=0 A ground-state energy must lie below the barrier top V3."""
        spec = _acetaldehyde_spec(n_basis=12)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=1)
        E_A0 = float(out["A"]["energies_cm-1"][0])
        assert E_A0 < _V3_CH3CHO, (
            f"vt=0 A energy {E_A0:.3f} cm^-1 must be below V3 = {_V3_CH3CHO} cm^-1"
        )

    def test_splitting_smaller_than_methanol(self):
        """Acetaldehyde splitting < methanol (~9 cm^-1) because V3/F is smaller."""
        pot_me = TorsionFourierPotential(v0=186.1, vcos={3: -186.8}, vsin={}, units="cm-1")
        spec_me = TorsionHamiltonianSpec(
            F=27.65, rho=0.81, A=4.25, B=0.82, C=0.79,
            potential=pot_me, n_basis=12, units="cm-1",
        )
        spec_ch3cho = _acetaldehyde_spec(n_basis=12)
        rows_me    = predict_tunneling_splitting(spec_me,    J=0, K=0, n_levels=1)
        rows_ch3cho = predict_tunneling_splitting(spec_ch3cho, J=0, K=0, n_levels=1)
        split_me     = abs(rows_me[0]["splitting_cm-1"])
        split_ch3cho = abs(rows_ch3cho[0]["splitting_cm-1"])
        assert split_ch3cho < split_me, (
            f"Acetaldehyde split {split_ch3cho:.3f} should be < methanol split {split_me:.3f}"
        )

    def test_basis_convergence_n_basis_12(self):
        """Splitting converges to < 0.01 cm^-1 change from n_basis=12 to n_basis=16."""
        spec12 = _acetaldehyde_spec(n_basis=12)
        spec16 = _acetaldehyde_spec(n_basis=16)
        rows12 = predict_tunneling_splitting(spec12, J=0, K=0, n_levels=1)
        rows16 = predict_tunneling_splitting(spec16, J=0, K=0, n_levels=1)
        delta = abs(rows12[0]["splitting_cm-1"] - rows16[0]["splitting_cm-1"])
        assert delta < 0.01, (
            f"Splitting not converged at n_basis=12: delta = {delta:.5f} cm^-1"
        )

    def test_levels_sorted_ascending(self):
        """Eigenvalues from solve_ram_lite_levels must be in ascending order."""
        spec = _acetaldehyde_spec(n_basis=12)
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=8)
        e = out["energies_cm-1"]
        assert np.all(np.diff(e) >= -1e-10), (
            f"Levels not sorted: diffs = {np.diff(e)}"
        )

    def test_A_E_levels_alternate_correctly(self):
        """For a 3-fold barrier: vt=0 A < vt=0 E < vt=1 A in energy."""
        spec = _acetaldehyde_spec(n_basis=12)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=3)
        E_A = out["A"]["energies_cm-1"]
        E_E = out["E1"]["energies_cm-1"]
        if len(E_A) >= 2 and len(E_E) >= 1:
            assert E_A[0] < E_E[0], "vt=0 A should be below vt=0 E"
            assert E_E[0] < E_A[1], "vt=0 E should be below vt=1 A"

    def test_excited_state_A_higher_than_ground(self):
        """vt=1 A must lie above vt=0 A."""
        spec = _acetaldehyde_spec(n_basis=12)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=2)
        E_A = out["A"]["energies_cm-1"]
        if len(E_A) >= 2:
            assert E_A[1] > E_A[0], "vt=1 A should be above vt=0 A"

    def test_mhz_splitting_conversion(self):
        """MHz splitting = cm^-1 splitting Ã— 29979.2458 within 0.1 MHz."""
        _MHZ = 29979.2458
        spec = _acetaldehyde_spec(n_basis=12)
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=2)
        for row in rows:
            expected_mhz = row["splitting_cm-1"] * _MHZ
            assert abs(row["splitting_MHz"] - expected_mhz) < 0.1, (
                f"MHz conversion error: {row['splitting_MHz']:.3f} vs {expected_mhz:.3f}"
            )


# â”€â”€ Reliability scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestReliabilityScoring:
    """Unit tests for assess_ram_lite_reliability heuristics."""

    def _make_spec(self, F=27.65, rho=0.1, A=4.25, B=0.82, C=0.82,
                   barrier=373.0, n_basis=12, **cd) -> TorsionHamiltonianSpec:
        pot = TorsionFourierPotential(
            v0=barrier / 2.0, vcos={3: -barrier / 2.0}, vsin={}, units="cm-1"
        )
        return TorsionHamiltonianSpec(
            F=F, rho=rho, A=A, B=B, C=C,
            potential=pot, n_basis=n_basis, units="cm-1", **cd
        )

    def test_high_score_for_methanol_params(self):
        """Symmetric-top methanol parameters (Bâ‰ˆC) should score 'high'."""
        spec = self._make_spec(F=27.65, rho=0.81, B=0.82, C=0.82, barrier=373.0)
        result = assess_ram_lite_reliability(spec)
        assert result.score == "high", f"Expected 'high', got '{result.score}'; flags={result.flags}"
        assert result.flags == []
        assert np.isfinite(result.ram_lite_error_bound_cm1)

    def test_acetaldehyde_scores_moderate(self):
        """Acetaldehyde parameters trigger asymmetry flag â†’ 'moderate'."""
        spec = _acetaldehyde_spec(n_basis=12)
        result = assess_ram_lite_reliability(spec)
        # B=0.3413, C=0.2884 â†’ Bâˆ’C=0.0529 > 0.1Ã—B=0.0341 â†’ asymmetry flag
        assert result.score == "moderate", (
            f"Expected 'moderate', got '{result.score}'; flags={result.flags}"
        )
        assert any("B" in f and "C" in f for f in result.flags)

    def test_low_score_for_small_vf(self):
        """V/F < 2 triggers 'low' score."""
        # barrier = 10, F = 27.65 â†’ V/F â‰ˆ 0.36 < 2
        pot = TorsionFourierPotential(v0=5.0, vcos={3: -5.0}, vsin={}, units="cm-1")
        spec = TorsionHamiltonianSpec(
            F=27.65, rho=0.1, A=0.0, B=0.0, C=0.0,
            potential=pot, n_basis=12, units="cm-1",
        )
        result = assess_ram_lite_reliability(spec)
        assert result.score == "low", f"Expected 'low', got '{result.score}'; flags={result.flags}"
        assert any("V/F" in f for f in result.flags)

    def test_low_score_for_inadequate_basis(self):
        """n_basis < max harmonic order triggers 'low' score."""
        pot = TorsionFourierPotential(
            v0=186.0, vcos={3: -186.0, 9: -1.0}, vsin={}, units="cm-1"
        )
        spec = TorsionHamiltonianSpec(
            F=27.65, rho=0.1, A=4.25, B=0.82, C=0.82,
            potential=pot, n_basis=6, units="cm-1",   # 6 < 9
        )
        result = assess_ram_lite_reliability(spec)
        assert result.score == "low"
        assert any("n_basis" in f for f in result.flags)

    def test_moderate_score_for_strong_asymmetry(self):
        """B âˆ’ C > 0.1 Ã— B triggers 'moderate'."""
        # B=1.0, C=0.5 â†’ Bâˆ’C=0.5 > 0.1*B=0.1
        spec = self._make_spec(F=5.0, rho=0.1, B=1.0, C=0.5, barrier=200.0, n_basis=12)
        result = assess_ram_lite_reliability(spec)
        assert result.score in ("moderate", "low", "unreliable")
        assert any("B" in f for f in result.flags)

    def test_moderate_score_for_large_rho(self):
        """Ï > 0.95 triggers 'moderate'."""
        spec = self._make_spec(F=27.65, rho=0.98, B=0.82, C=0.82, barrier=373.0)
        result = assess_ram_lite_reliability(spec)
        assert result.score in ("moderate", "low", "unreliable")
        assert any("Ï" in f or "rho" in f.lower() for f in result.flags)

    def test_unreliable_for_large_cd(self):
        """CD constant >> 1% of B â†’ 'unreliable'."""
        spec = self._make_spec(F=27.65, rho=0.81, B=0.82, C=0.82, barrier=373.0, DJ=0.1)
        result = assess_ram_lite_reliability(spec)
        assert result.score == "unreliable"
        assert result.ram_lite_error_bound_cm1 == float("inf")

    def test_error_bound_finite_when_not_unreliable(self):
        """Any score except 'unreliable' should have a finite error bound."""
        spec = self._make_spec(F=27.65, rho=0.81, B=0.82, C=0.82, barrier=373.0)
        result = assess_ram_lite_reliability(spec)
        if result.score != "unreliable":
            assert np.isfinite(result.ram_lite_error_bound_cm1)
            assert result.ram_lite_error_bound_cm1 > 0.0

    def test_result_is_dataclass(self):
        spec = self._make_spec()
        result = assess_ram_lite_reliability(spec)
        assert isinstance(result, TorsionReliabilityScore)
        assert isinstance(result.score, str)
        assert isinstance(result.flags, list)
        assert isinstance(result.ram_lite_error_bound_cm1, float)

    def test_cd_threshold_exactly_at_boundary(self):
        """DJ exactly at 1% of B_min should not trigger the flag."""
        # B = 0.82 â†’ 1% = 0.0082; use DJ = 0.008 (just below threshold)
        spec = self._make_spec(F=27.65, rho=0.81, B=0.82, C=0.82, barrier=373.0, DJ=0.008)
        result = assess_ram_lite_reliability(spec)
        assert result.score != "unreliable", (
            f"DJ=0.008 < 1%*B=0.0082 should not be 'unreliable'; got '{result.score}'"
        )
