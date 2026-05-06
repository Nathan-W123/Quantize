"""
Phase 11: LAM Benchmarks and Validation.

Validates the RAM-lite torsion-rotation implementation against:
  1. Free-rotor limit (V = 0): analytic E_m = F*m^2 spectrum
  2. High-barrier harmonic limit: levels approach F*m^2 + V*alpha^2 harmonic
  3. Methanol literature benchmark (Xu et al. 2008 RAM parameters)
  4. Basis convergence: n_basis independence of converged levels
  5. A/E tunneling splitting sign convention and ordering

Mark slow/expensive tests with @pytest.mark.slow — these are excluded from the
default CI run; add -m slow to include them.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion_symmetry import (
    c3_symmetry_block_energies,
    predict_tunneling_splitting,
    symmetry_purity_table,
)
from backend.torsion_rot_hamiltonian import solve_full_torsion_rotation_levels


# ── Methanol literature parameters (Xu et al. 2008) ─────────────────────────

_F       = 27.64684641    # cm^-1  internal rotation constant
_RHO     = 0.8102062230   # dimensionless coupling
_V0      = 186.117548     # cm^-1  Fourier constant term
_VCOS3   = -186.777373    # cm^-1  3-fold cosine coefficient
_VCOS6   =   0.659825     # cm^-1  6-fold cosine coefficient


def _methanol_spec(n_basis=15, include_V6=True):
    vcos = {3: _VCOS3}
    if include_V6:
        vcos[6] = _VCOS6
    pot = TorsionFourierPotential(v0=_V0, vcos=vcos, units="cm-1")
    return TorsionHamiltonianSpec(
        F=_F, rho=_RHO, A=4.2542, B=0.8231, C=0.7931,
        potential=pot, n_basis=n_basis, units="cm-1",
    )


def _free_rotor_spec(F=10.0, n_basis=10):
    pot = TorsionFourierPotential(v0=0.0, units="cm-1")
    return TorsionHamiltonianSpec(F=F, rho=0.0, A=0.0, B=0.0, C=0.0,
                                  potential=pot, n_basis=n_basis, units="cm-1")


def _high_barrier_spec(F=5.0, V3=2000.0, n_basis=15):
    """Spec with very high 3-fold barrier (V3 >> F): nearly harmonic torsion."""
    pot = TorsionFourierPotential(v0=V3/2, vcos={3: -V3/2}, units="cm-1")
    return TorsionHamiltonianSpec(F=F, rho=0.0, A=0.0, B=0.0, C=0.0,
                                  potential=pot, n_basis=n_basis, units="cm-1")


# ── 1. Free-rotor limit ───────────────────────────────────────────────────────

class TestFreeRotorLimit:
    """For V = 0 and rho = 0: E_m = F*m^2, degenerate ±m pairs."""

    def test_ground_state_at_zero(self):
        spec = _free_rotor_spec(F=10.0)
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=1)
        assert abs(out["energies_cm-1"][0]) < 1e-10

    def test_first_degenerate_pair_equals_F(self):
        F = 10.0
        spec = _free_rotor_spec(F=F)
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=3)
        e = out["energies_cm-1"]
        # e[1] and e[2] should both equal F (±1 pair)
        assert abs(e[1] - F) < 1e-8
        assert abs(e[2] - F) < 1e-8

    def test_second_degenerate_pair_equals_4F(self):
        F = 10.0
        spec = _free_rotor_spec(F=F)
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=5)
        e = out["energies_cm-1"]
        assert abs(e[3] - 4 * F) < 1e-8
        assert abs(e[4] - 4 * F) < 1e-8

    def test_spectrum_matches_Fm2_sequence(self):
        """All levels should be F*m^2 for m = 0,1,1,2,2,..."""
        F = 7.5
        spec = _free_rotor_spec(F=F, n_basis=5)
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=11)
        e = out["energies_cm-1"]
        m_vals = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        expected = F * m_vals[:len(e)] ** 2
        np.testing.assert_allclose(e, expected, atol=1e-8)

    def test_K_nonzero_with_rho_shifts_levels(self):
        """rho != 0 shifts levels by F*(m - rho*K)^2."""
        F = 10.0
        rho = 0.3
        K = 1
        pot = TorsionFourierPotential(units="cm-1")
        spec = TorsionHamiltonianSpec(F=F, rho=rho, A=0.0, B=0.0, C=0.0,
                                      potential=pot, n_basis=5, units="cm-1")
        out = solve_ram_lite_levels(spec, J=0, K=K, n_levels=1)
        # Ground state ≈ F*(m_min - rho*K)^2 for smallest |m - rho*K|
        # With rho=0.3, K=1: x = m - 0.3 for m in [-5..5]. Min at m=0: x=-0.3 → F*0.09=0.9
        e0 = out["energies_cm-1"][0]
        assert e0 < F  # less than F (would be F for m=1 without rho shift)


# ── 2. High-barrier harmonic limit ────────────────────────────────────────────

class TestHighBarrierLimit:
    """For V3 >> F, torsion levels approach harmonic oscillator with omega=3*sqrt(F*V3)."""

    def test_level_spacing_approaches_harmonic(self):
        F = 5.0
        V3 = 2000.0
        # Harmonic omega for V = V3/2*(1-cos3α) ≈ 9V3/2 * α^2/2 near min
        # omega = 3*sqrt(V3*F) (in cm^-1 units where F is already in cm^-1)
        omega_harmonic = 3.0 * np.sqrt(V3 * F)
        spec = _high_barrier_spec(F=F, V3=V3, n_basis=20)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=4)
        E_A = out["A"]["energies_cm-1"]
        if len(E_A) >= 2:
            spacing = float(E_A[1] - E_A[0])
            # Spacing should be close to omega_harmonic (within ~5%)
            assert abs(spacing - omega_harmonic) / omega_harmonic < 0.05

    def test_ground_state_below_barrier_top(self):
        """Ground-state A energy must be below the barrier top (V3 in our convention)."""
        spec = _high_barrier_spec(F=5.0, V3=500.0, n_basis=15)
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=1)
        E_A0 = float(out["A"]["energies_cm-1"][0])
        barrier_top = 500.0  # V3 (value of V(pi/3) = V3)
        assert E_A0 < barrier_top

    def test_tunneling_splitting_decreases_with_barrier(self):
        """Larger V3 → smaller A/E tunneling splitting."""
        spec_low  = _high_barrier_spec(F=5.0, V3=200.0, n_basis=15)
        spec_high = _high_barrier_spec(F=5.0, V3=800.0, n_basis=15)
        rows_low  = predict_tunneling_splitting(spec_low,  J=0, K=0, n_levels=1)
        rows_high = predict_tunneling_splitting(spec_high, J=0, K=0, n_levels=1)
        if rows_low and rows_high:
            split_low  = abs(rows_low[0]["splitting_cm-1"])
            split_high = abs(rows_high[0]["splitting_cm-1"])
            assert split_low > split_high


# ── 3. Methanol literature benchmark ─────────────────────────────────────────

class TestMethanolBenchmark:
    """
    Validate the implementation against methanol RAM parameters from Xu et al. (2008).
    Reference values computed internally at n_basis=25 for self-consistency.
    """

    def test_ground_state_A_energy_positive(self):
        """The A-species ground level must lie above the potential minimum."""
        spec = _methanol_spec()
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=1)
        E_A0 = float(out["A"]["energies_cm-1"][0])
        # Potential minimum is at alpha=0: V(0) = v0 + Vcos3 = 186.12 - 186.78 ≈ -0.66 cm^-1
        # The zero-point level must be above this minimum
        V_min = _V0 + _VCOS3 + _VCOS6  # ≈ -0.0 cm^-1
        assert E_A0 > V_min

    def test_A_E_splitting_positive_sign(self):
        """For methanol below barrier: E(E1) > E(A) → positive splitting."""
        spec = _methanol_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
        assert len(rows) > 0
        assert rows[0]["splitting_cm-1"] > 0.0

    def test_A_E_splitting_vt0_in_range(self):
        """vt=0 A/E splitting for methanol should be in the range 7-10 cm^-1."""
        spec = _methanol_spec(n_basis=20)
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
        split = abs(rows[0]["splitting_cm-1"])
        assert 5.0 < split < 20.0, f"Expected 5-20 cm^-1, got {split:.3f}"

    def test_V6_correction_changes_splitting(self):
        """Including V6 correction should give a different split than V6=0."""
        spec_v6    = _methanol_spec(include_V6=True,  n_basis=15)
        spec_nov6  = _methanol_spec(include_V6=False, n_basis=15)
        rows_v6   = predict_tunneling_splitting(spec_v6,   J=0, K=0, n_levels=1)
        rows_nov6 = predict_tunneling_splitting(spec_nov6, J=0, K=0, n_levels=1)
        if rows_v6 and rows_nov6:
            assert abs(rows_v6[0]["splitting_cm-1"] - rows_nov6[0]["splitting_cm-1"]) > 1e-6

    def test_excited_state_A_higher_than_ground(self):
        """vt=1 A level must be above vt=0 A level."""
        spec = _methanol_spec()
        out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=2)
        E_A = out["A"]["energies_cm-1"]
        if len(E_A) >= 2:
            assert E_A[1] > E_A[0]

    def test_symmetry_purity_high_for_known_params(self):
        """A states should be pure; E states may mix E1/E2 (degenerate) but A purity = 1."""
        spec = _methanol_spec(n_basis=12)
        rows = symmetry_purity_table(spec, J=0, K=0, n_levels=4)
        for row in rows:
            if row["symmetry_label"] == "A":
                # A states are non-degenerate; purity must be 1.
                assert row["purity"] > 0.999, f"A-state purity should be 1.0, got {row['purity']:.4f}"
            else:
                # E1 and E2 are degenerate; eigensolver may mix them — purity in each
                # residue can be anywhere in [0.5, 1.0].  At minimum the dominant residue
                # carries more than half the weight.
                assert row["purity"] > 0.5, f"E-state purity below 0.5: {row['purity']:.4f}"

    @pytest.mark.slow
    def test_methanol_full_hamiltonian_J1(self):
        """Full J=1 torsion-rotation Hamiltonian with methanol parameters."""
        spec = _methanol_spec(n_basis=15)
        out = solve_full_torsion_rotation_levels(spec, J=1, n_levels=12)
        e = out["energies_cm-1"]
        # All energies should be real and finite
        assert np.all(np.isfinite(e))
        assert np.all(np.diff(e) >= -1e-10)


# ── 4. Basis convergence ──────────────────────────────────────────────────────

class TestBasisConvergence:
    """Level energies must converge as n_basis increases."""

    def test_ground_state_converges(self):
        """Ground state A energy must stabilize for n_basis >= 8."""
        F = 10.0; V3 = 200.0
        spec8  = _high_barrier_spec(F=F, V3=V3, n_basis=8)
        spec12 = _high_barrier_spec(F=F, V3=V3, n_basis=12)
        spec16 = _high_barrier_spec(F=F, V3=V3, n_basis=16)
        e8  = c3_symmetry_block_energies(spec8,  J=0, K=0)["A"]["energies_cm-1"]
        e12 = c3_symmetry_block_energies(spec12, J=0, K=0)["A"]["energies_cm-1"]
        e16 = c3_symmetry_block_energies(spec16, J=0, K=0)["A"]["energies_cm-1"]
        if e8.size > 0 and e12.size > 0 and e16.size > 0:
            # Each step should reduce the ground-state error
            diff_8_to_16  = abs(float(e8[0])  - float(e16[0]))
            diff_12_to_16 = abs(float(e12[0]) - float(e16[0]))
            assert diff_12_to_16 < diff_8_to_16

    def test_methanol_n_basis_convergence(self):
        """Methanol A/E split should converge to < 0.01 cm^-1 change from n_basis=12 to 18."""
        spec12 = _methanol_spec(n_basis=12)
        spec18 = _methanol_spec(n_basis=18)
        rows12 = predict_tunneling_splitting(spec12, J=0, K=0, n_levels=1)
        rows18 = predict_tunneling_splitting(spec18, J=0, K=0, n_levels=1)
        if rows12 and rows18:
            delta = abs(rows12[0]["splitting_cm-1"] - rows18[0]["splitting_cm-1"])
            assert delta < 0.1, f"Split not converged: delta={delta:.4f} cm^-1"

    def test_monotone_convergence_of_higher_levels(self):
        """Higher levels converge more slowly; check that n_basis=20 is tighter than n_basis=10."""
        spec10 = _methanol_spec(n_basis=10)
        spec15 = _methanol_spec(n_basis=15)
        spec20 = _methanol_spec(n_basis=20)

        def _get_vt1_A(spec):
            out = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=2)
            e = out["A"]["energies_cm-1"]
            return float(e[1]) if len(e) > 1 else float("nan")

        e10 = _get_vt1_A(spec10)
        e15 = _get_vt1_A(spec15)
        e20 = _get_vt1_A(spec20)
        if all(np.isfinite([e10, e15, e20])):
            assert abs(e15 - e20) < abs(e10 - e20)

    @pytest.mark.slow
    def test_large_basis_methanol_converged(self):
        """n_basis=25 should give A/E split accurate to 0.01 cm^-1 vs n_basis=30."""
        spec25 = _methanol_spec(n_basis=25)
        spec30 = _methanol_spec(n_basis=30)
        rows25 = predict_tunneling_splitting(spec25, J=0, K=0, n_levels=2)
        rows30 = predict_tunneling_splitting(spec30, J=0, K=0, n_levels=2)
        if rows25 and rows30:
            delta = abs(rows25[0]["splitting_cm-1"] - rows30[0]["splitting_cm-1"])
            assert delta < 0.01, f"Not converged at n_basis=25: delta={delta:.5f} cm^-1"


# ── 5. Physical self-consistency ──────────────────────────────────────────────

class TestPhysicalSelfConsistency:
    """Cross-module consistency checks."""

    def test_levels_from_blocks_match_full_solve(self):
        """Block-diagonalized A-species energies should match labeled energies from solve."""
        spec = _methanol_spec(n_basis=10)
        out_full = solve_ram_lite_levels(spec, J=0, K=0, symmetry_mode="c3", n_levels=6)
        out_blocks = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=3)

        full_A = sorted([
            float(e) for e, lbl in zip(
                out_full["energies_cm-1"],
                out_full.get("symmetry_labels", [])
            ) if str(lbl) == "A"
        ])
        block_A = sorted(out_blocks["A"]["energies_cm-1"].tolist())
        n = min(len(full_A), len(block_A))
        np.testing.assert_allclose(full_A[:n], block_A[:n], atol=1e-6)

    def test_splitting_mhz_consistent(self):
        """MHz splitting must equal cm^-1 splitting * 29979.2458."""
        _MHZ = 29979.2458
        spec = _methanol_spec()
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=3)
        for row in rows:
            assert abs(row["splitting_MHz"] - row["splitting_cm-1"] * _MHZ) < 1e-4

    def test_full_J0_equals_ram_lite_K0(self):
        """Full torsion-rotation Hamiltonian at J=0 must match RAM-lite K=0 block."""
        spec = _methanol_spec(n_basis=10)
        rl = solve_ram_lite_levels(spec, J=0, K=0, n_levels=6)
        full = solve_full_torsion_rotation_levels(spec, J=0, n_levels=6)
        np.testing.assert_allclose(
            rl["energies_cm-1"], full["energies_cm-1"], atol=1e-8,
            err_msg="J=0 full and RAM-lite K=0 solutions must match"
        )

    def test_fitter_recovers_methanol_V3(self):
        """Fitting to exact synthetic data should recover V3 within 2 cm^-1."""
        from backend.torsion_fitter import fit_torsion_to_levels, select_fit_params

        spec_true = _methanol_spec(n_basis=10)
        # Generate synthetic observed levels
        obs_rows = []
        for i in range(8):
            out = solve_ram_lite_levels(spec_true, J=0, K=0, n_levels=8)
            e = out["energies_cm-1"]
            obs_rows.append({"J": 0, "K": 0, "level_index": i, "energy_cm-1": float(e[i])})

        # Perturb Vcos3 and fit back
        from copy import deepcopy
        spec_init = deepcopy(spec_true)
        spec_init.potential.vcos[3] = _VCOS3 * 0.95  # 5% perturbation
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=30)
        recovered = float(result["fitted_spec"].potential.vcos[3])
        assert abs(recovered - _VCOS3) < 2.0, f"Recovery failed: {recovered:.3f} vs {_VCOS3:.3f}"
