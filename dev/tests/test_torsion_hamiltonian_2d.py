"""
Phase 6: 2D two-rotor torsion Hamiltonian tests.

Validates:
  - Free-rotor eigenvalues (analytic: F1*m1² + F2*m2²)
  - Hermiticity of H for arbitrary potentials
  - Separable V(α1,α2) = V1(α1) + V2(α2), F12=0 → 2D eigenvalues equal all
    pairwise sums of independent 1D eigenvalues (verified against solve_ram_lite_levels)
  - Basis size = (2*n_basis_1+1) * (2*n_basis_2+1)
  - Eigenvalues returned in ascending order
  - F12 coupling shifts eigenvalues relative to the uncoupled case
  - Probability density normalization and non-negativity on a rectangular grid
  - 2D scan averaging: constant scan, output keys, weight normalization,
    sigma_averaged non-negativity, empty scan error
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_average_2d import (
    TorsionGridPoint2D,
    TorsionScan2D,
    average_torsion_scan_2d_quantum,
)
from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion_hamiltonian_2d import (
    TorsionFourierPotential2D,
    TorsionHamiltonianSpec2D,
    build_2d_torsion_hamiltonian,
    solve_2d_torsion_levels,
    torsion_probability_density_2d,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _free_rotor_2d(F1: float = 1.0, F2: float = 2.0, n1: int = 4, n2: int = 4) -> TorsionHamiltonianSpec2D:
    return TorsionHamiltonianSpec2D(
        F1=F1, F2=F2, rho1=0.0, rho2=0.0, F12=0.0,
        potential=TorsionFourierPotential2D(v0=0.0, vcos={}),
        n_basis_1=n1, n_basis_2=n2,
    )


def _separable_2d(
    F1: float = 5.0, V3_1: float = 100.0,
    F2: float = 3.0, V3_2: float = 80.0,
    n: int = 6,
) -> TorsionHamiltonianSpec2D:
    return TorsionHamiltonianSpec2D(
        F1=F1, F2=F2, rho1=0.0, rho2=0.0, F12=0.0,
        potential=TorsionFourierPotential2D(
            v0=V3_1 / 2.0 + V3_2 / 2.0,
            vcos={(3, 0): -V3_1 / 2.0, (0, 3): -V3_2 / 2.0},
        ),
        n_basis_1=n, n_basis_2=n,
    )


def _1d_spec(F: float, V3: float, n: int) -> TorsionHamiltonianSpec:
    return TorsionHamiltonianSpec(
        F=F, rho=0.0, A=0.0, B=0.0, C=0.0,
        potential=TorsionFourierPotential(
            v0=V3 / 2.0, vcos={3: -V3 / 2.0}, vsin={}, units="cm-1"
        ),
        n_basis=n, units="cm-1",
    )


# ── Hamiltonian construction ──────────────────────────────────────────────────

class TestBuild2DHamiltonian:
    def test_basis_size(self):
        """H must have shape ((2n1+1)*(2n2+1),) * 2."""
        n1, n2 = 3, 5
        H = build_2d_torsion_hamiltonian(_free_rotor_2d(n1=n1, n2=n2))
        assert H.shape == ((2 * n1 + 1) * (2 * n2 + 1),) * 2

    def test_hermitian_free_rotor(self):
        """Free-rotor H must be real symmetric."""
        H = build_2d_torsion_hamiltonian(_free_rotor_2d())
        assert np.allclose(H, H.T, atol=1e-12)

    def test_hermitian_separable_potential(self):
        """H with separable 3-fold potential must be real symmetric."""
        H = build_2d_torsion_hamiltonian(_separable_2d())
        assert np.allclose(H, H.T, atol=1e-12)

    def test_hermitian_coupling_potential(self):
        """H with cross-term cos(3α1 + 3α2) must be real symmetric."""
        spec = TorsionHamiltonianSpec2D(
            F1=5.0, F2=3.0, rho1=0.0, rho2=0.0, F12=0.0,
            potential=TorsionFourierPotential2D(
                v0=50.0, vcos={(3, 0): -40.0, (0, 3): -30.0, (3, 3): -10.0},
            ),
            n_basis_1=5, n_basis_2=5,
        )
        H = build_2d_torsion_hamiltonian(spec)
        assert np.allclose(H, H.T, atol=1e-12)

    def test_free_rotor_is_diagonal(self):
        """V=0 Hamiltonian must be purely diagonal (no off-diagonal elements)."""
        H = build_2d_torsion_hamiltonian(_free_rotor_2d(n1=3, n2=3))
        off = H - np.diag(np.diag(H))
        assert np.allclose(off, 0.0, atol=1e-12)

    def test_free_rotor_eigenvalues_analytic(self):
        """V=0: eigenvalues must equal {F1*m1² + F2*m2²} for all (m1,m2) in basis."""
        F1, F2 = 1.0, 2.0
        n1, n2 = 4, 4
        spec = _free_rotor_2d(F1=F1, F2=F2, n1=n1, n2=n2)
        E_num = np.sort(solve_2d_torsion_levels(spec)["energies_cm-1"])

        m1 = np.arange(-n1, n1 + 1)
        m2 = np.arange(-n2, n2 + 1)
        M1, M2 = np.meshgrid(m1, m2, indexing="ij")
        E_analytic = np.sort((F1 * M1 ** 2 + F2 * M2 ** 2).ravel())

        assert np.allclose(E_num, E_analytic, atol=1e-8)

    def test_v0_offset_shifts_all_eigenvalues(self):
        """Adding a constant v0 must shift every eigenvalue by v0."""
        spec0 = _free_rotor_2d(n1=3, n2=3)
        spec_v0 = TorsionHamiltonianSpec2D(
            F1=1.0, F2=2.0, rho1=0.0, rho2=0.0, F12=0.0,
            potential=TorsionFourierPotential2D(v0=10.0, vcos={}),
            n_basis_1=3, n_basis_2=3,
        )
        E0 = solve_2d_torsion_levels(spec0)["energies_cm-1"]
        Ev = solve_2d_torsion_levels(spec_v0)["energies_cm-1"]
        assert np.allclose(Ev - E0, 10.0, atol=1e-10)

    def test_f12_coupling_breaks_degeneracy(self):
        """F12≠0 must shift at least one eigenvalue compared with F12=0."""
        base = dict(F1=5.0, F2=3.0, rho1=0.0, rho2=0.0,
                    potential=TorsionFourierPotential2D(v0=0.0, vcos={}),
                    n_basis_1=5, n_basis_2=5)
        spec0 = TorsionHamiltonianSpec2D(**base, F12=0.0)
        spec12 = TorsionHamiltonianSpec2D(**base, F12=2.0)
        E0 = solve_2d_torsion_levels(spec0)["energies_cm-1"]
        E12 = solve_2d_torsion_levels(spec12)["energies_cm-1"]
        assert not np.allclose(E0, E12), "F12≠0 must shift some eigenvalues"

    def test_zero_coefficient_term_has_no_effect(self):
        """A vcos term with coefficient 0.0 must not change H."""
        spec_bare = _free_rotor_2d(n1=4, n2=4)
        spec_zero = TorsionHamiltonianSpec2D(
            F1=1.0, F2=2.0, rho1=0.0, rho2=0.0, F12=0.0,
            potential=TorsionFourierPotential2D(v0=0.0, vcos={(3, 0): 0.0, (0, 3): 0.0}),
            n_basis_1=4, n_basis_2=4,
        )
        H_bare = build_2d_torsion_hamiltonian(spec_bare)
        H_zero = build_2d_torsion_hamiltonian(spec_zero)
        assert np.allclose(H_bare, H_zero, atol=1e-12)


# ── Diagonalization ───────────────────────────────────────────────────────────

class TestSolve2DTorsionLevels:
    def test_eigenvalues_ascending(self):
        out = solve_2d_torsion_levels(_separable_2d(), n_levels=20)
        E = out["energies_cm-1"]
        assert np.all(np.diff(E) >= -1e-10)

    def test_n_levels_truncation(self):
        out = solve_2d_torsion_levels(_free_rotor_2d(n1=5, n2=5), n_levels=10)
        assert len(out["energies_cm-1"]) == 10

    def test_n_levels_none_returns_all(self):
        n1, n2 = 3, 3
        spec = _free_rotor_2d(n1=n1, n2=n2)
        out = solve_2d_torsion_levels(spec, n_levels=None)
        assert len(out["energies_cm-1"]) == (2 * n1 + 1) * (2 * n2 + 1)

    def test_output_keys(self):
        out = solve_2d_torsion_levels(_free_rotor_2d())
        for key in ("energies_cm-1", "eigenvectors", "n_basis_1", "n_basis_2", "J", "K"):
            assert key in out

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors from eigh must be column-orthonormal."""
        out = solve_2d_torsion_levels(_separable_2d(n=4), n_levels=10)
        V = out["eigenvectors"]
        assert np.allclose(V.T @ V, np.eye(10), atol=1e-10)

    def test_separable_eigenvalues_match_pairwise_sums(self):
        """V(α1,α2) = V1(α1) + V2(α2) with F12=0: 2D eigenvalues must equal
        all pairwise sums E1_a + E2_b of the independent 1D eigenvalues."""
        F1, V3_1 = 5.0, 100.0
        F2, V3_2 = 3.0, 80.0
        n = 5

        E1 = solve_ram_lite_levels(_1d_spec(F1, V3_1, n), J=0, K=0)["energies_cm-1"]
        E2 = solve_ram_lite_levels(_1d_spec(F2, V3_2, n), J=0, K=0)["energies_cm-1"]

        spec2d = _separable_2d(F1=F1, V3_1=V3_1, F2=F2, V3_2=V3_2, n=n)
        E_2d = np.sort(solve_2d_torsion_levels(spec2d)["energies_cm-1"])

        M1, M2 = np.meshgrid(E1, E2, indexing="ij")
        E_expected = np.sort((M1 + M2).ravel())

        assert np.allclose(E_2d, E_expected, atol=1e-6), (
            f"Max deviation from separable limit: "
            f"{np.max(np.abs(E_2d - E_expected)):.2e} cm^-1"
        )

    def test_separable_ground_state_below_barrier(self):
        """Ground-state energy must lie below the sum of the two barrier heights."""
        spec = _separable_2d(V3_1=100.0, V3_2=80.0)
        E0 = solve_2d_torsion_levels(spec, n_levels=1)["energies_cm-1"][0]
        # The potential v0 = V3_1/2 + V3_2/2 sets the zero; vt=0 < v0 + V3/2 (top of barrier)
        assert E0 < 100.0 + 80.0


# ── Probability density ───────────────────────────────────────────────────────

class TestProbabilityDensity2D:
    def test_output_shape(self):
        spec = _free_rotor_2d(n1=3, n2=4)
        eigenvec = solve_2d_torsion_levels(spec)["eigenvectors"][:, 0]
        G1, G2 = 30, 40
        a1 = np.linspace(0, 2 * np.pi, G1, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, G2, endpoint=False)
        density = torsion_probability_density_2d(eigenvec, 3, 4, a1, a2)
        assert density.shape == (G1, G2)

    def test_non_negative(self):
        spec = _free_rotor_2d(n1=4, n2=4)
        eigenvec = solve_2d_torsion_levels(spec)["eigenvectors"][:, 0]
        a = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        density = torsion_probability_density_2d(eigenvec, 4, 4, a, a)
        assert np.all(density >= -1e-14)

    def test_normalization_ground_state(self):
        """∫∫ |ψ₀(α1,α2)|² dα1 dα2 ≈ 1 on a dense uniform grid."""
        spec = _separable_2d(n=5)
        eigenvec = solve_2d_torsion_levels(spec)["eigenvectors"][:, 0]
        G = 120
        a = np.linspace(0, 2 * np.pi, G, endpoint=False)
        density = torsion_probability_density_2d(eigenvec, 5, 5, a, a)
        dalpha = 2.0 * np.pi / G
        norm = np.sum(density) * dalpha ** 2
        assert np.isclose(norm, 1.0, atol=0.01), f"Norm = {norm:.6f}"

    def test_normalization_excited_state(self):
        """Normalization holds for the first excited state too."""
        spec = _separable_2d(n=5)
        eigenvec = solve_2d_torsion_levels(spec)["eigenvectors"][:, 1]
        G = 120
        a = np.linspace(0, 2 * np.pi, G, endpoint=False)
        density = torsion_probability_density_2d(eigenvec, 5, 5, a, a)
        dalpha = 2.0 * np.pi / G
        norm = np.sum(density) * dalpha ** 2
        assert np.isclose(norm, 1.0, atol=0.01), f"Norm (excited) = {norm:.6f}"

    def test_free_rotor_ground_state_is_uniform(self):
        """Free-rotor ground state (m1=0, m2=0) → uniform density = 1/(4π²)."""
        spec = _free_rotor_2d(n1=6, n2=6)
        out = solve_2d_torsion_levels(spec)
        # Ground state has eigenvalue 0 and is the e_{m1=0,m2=0} basis vector
        eigenvec = out["eigenvectors"][:, 0]
        G = 60
        a = np.linspace(0, 2 * np.pi, G, endpoint=False)
        density = torsion_probability_density_2d(eigenvec, 6, 6, a, a)
        expected = 1.0 / (4.0 * np.pi ** 2)
        assert np.allclose(density, expected, rtol=1e-6)


# ── 2D scan averaging ─────────────────────────────────────────────────────────

class TestTorsionAverage2D:
    def _uniform_scan(self, n_phi: int = 6, abc=None) -> TorsionScan2D:
        if abc is None:
            abc = np.array([4.0, 0.9, 0.8])
        phi_vals = np.linspace(0, 360, n_phi, endpoint=False)
        gps = []
        for p1 in phi_vals:
            for p2 in phi_vals:
                gps.append(TorsionGridPoint2D(
                    phi1=p1, phi2=p2,
                    rotational_constants=abc.copy(),
                ))
        return TorsionScan2D(name="uniform", grid_points=gps, angle_unit="degrees")

    def test_output_keys(self):
        scan = self._uniform_scan()
        spec = _free_rotor_2d(n1=3, n2=3)
        out = average_torsion_scan_2d_quantum(scan, spec)
        for key in (
            "averaged_constants", "weights", "phi1_radians", "phi2_radians",
            "state_index", "sigma_averaged", "method", "warnings",
        ):
            assert key in out, f"Missing key: {key}"

    def test_method_label(self):
        scan = self._uniform_scan()
        spec = _free_rotor_2d(n1=3, n2=3)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert out["method"] == "quantum_2d_rotor"

    def test_constant_scan_returns_constant(self):
        """All grid points have the same A/B/C → average equals that value."""
        abc = np.array([4.0, 0.9, 0.8])
        scan = self._uniform_scan(abc=abc)
        spec = _free_rotor_2d(n1=5, n2=5)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert np.allclose(out["averaged_constants"], abc, atol=1e-4)

    def test_weights_sum_to_one(self):
        scan = self._uniform_scan(n_phi=5)
        spec = _separable_2d(n=4)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert np.isclose(np.sum(out["weights"]), 1.0, atol=1e-10)

    def test_sigma_averaged_non_negative(self):
        scan = self._uniform_scan(n_phi=4)
        spec = _free_rotor_2d(n1=3, n2=3)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert np.all(out["sigma_averaged"] >= 0.0)

    def test_sigma_zero_for_constant_scan(self):
        """Constant A/B/C across all points → zero representational scatter."""
        abc = np.array([4.0, 0.9, 0.8])
        scan = self._uniform_scan(abc=abc, n_phi=6)
        spec = _free_rotor_2d(n1=4, n2=4)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert np.allclose(out["sigma_averaged"], 0.0, atol=1e-10)

    def test_phi_arrays_in_radians(self):
        scan = self._uniform_scan(n_phi=4)
        spec = _free_rotor_2d(n1=3, n2=3)
        out = average_torsion_scan_2d_quantum(scan, spec)
        assert np.all(out["phi1_radians"] >= 0.0)
        assert np.all(out["phi1_radians"] < 2.0 * np.pi + 1e-10)

    def test_state_index_stored(self):
        scan = self._uniform_scan(n_phi=3)
        spec = _free_rotor_2d(n1=3, n2=3)
        out = average_torsion_scan_2d_quantum(scan, spec, state_index=2)
        assert out["state_index"] == 2

    def test_empty_scan_raises(self):
        scan = TorsionScan2D(name="empty", grid_points=[], angle_unit="degrees")
        spec = _free_rotor_2d()
        with pytest.raises(ValueError, match="no grid points"):
            average_torsion_scan_2d_quantum(scan, spec)

    def test_invalid_state_index_raises(self):
        scan = self._uniform_scan(n_phi=2)
        spec = _free_rotor_2d(n1=1, n2=1)   # N = 3*3 = 9 states
        with pytest.raises(ValueError, match="state_index"):
            average_torsion_scan_2d_quantum(scan, spec, state_index=9999)

    def test_radians_angle_unit(self):
        """Scan given in radians must produce the same result as degrees."""
        abc = np.array([4.0, 0.9, 0.8])
        phi_deg = np.linspace(0, 360, 4, endpoint=False)
        phi_rad = np.deg2rad(phi_deg)
        gps_deg = [TorsionGridPoint2D(phi1=p1, phi2=p2, rotational_constants=abc.copy())
                   for p1 in phi_deg for p2 in phi_deg]
        gps_rad = [TorsionGridPoint2D(phi1=p1, phi2=p2, rotational_constants=abc.copy())
                   for p1 in phi_rad for p2 in phi_rad]
        scan_deg = TorsionScan2D(name="d", grid_points=gps_deg, angle_unit="degrees")
        scan_rad = TorsionScan2D(name="r", grid_points=gps_rad, angle_unit="radians")
        spec = _free_rotor_2d(n1=3, n2=3)
        out_deg = average_torsion_scan_2d_quantum(scan_deg, spec)
        out_rad = average_torsion_scan_2d_quantum(scan_rad, spec)
        assert np.allclose(out_deg["averaged_constants"], out_rad["averaged_constants"], atol=1e-10)
        assert np.allclose(out_deg["weights"], out_rad["weights"], atol=1e-10)
