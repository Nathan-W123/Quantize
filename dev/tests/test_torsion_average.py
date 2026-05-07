from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_average import (
    ConformerTorsionSpec,
    TorsionGridPoint,
    TorsionScan,
    average_rotational_constants_with_weights,
    average_torsion_scan_boltzmann,
    average_torsion_scan_quantum,
    average_torsion_scan_quantum_thermal,
    conformer_torsion_average,
    get_grid_sigma_abc,
    propagate_averaging_uncertainty,
    torsional_motion_correction,
)
from backend.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec


def _hr_to_spec(F: float, V3: float, basis_size: int = 21) -> TorsionHamiltonianSpec:
    """Convert hindered-rotor parameters to TorsionHamiltonianSpec for averaging."""
    n_basis = (basis_size - 1) // 2
    return TorsionHamiltonianSpec(
        F=F, rho=0.0, A=0.0, B=0.0, C=0.0,
        potential=TorsionFourierPotential(
            v0=V3 / 2.0, vcos={3: -V3 / 2.0}, vsin={}, units="cm-1"
        ),
        n_basis=n_basis, units="cm-1",
    )


def _water_like_geoms():
    g0 = np.array(
        [[0.0, 0.0, 0.1174], [0.0, 0.7572, -0.4696], [0.0, -0.7572, -0.4696]],
        dtype=float,
    )
    g1 = g0.copy()
    g1[1, 0] += 0.02
    g2 = g0.copy()
    g2[2, 0] -= 0.02
    return g0, g1, g2


def test_weighted_average_rotational_constants():
    C = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    w = np.array([0.25, 0.75])
    out = average_rotational_constants_with_weights(C, w)
    assert np.allclose(out, [2.5, 3.5, 4.5])


def test_boltzmann_torsion_average_prefers_low_energy():
    g0, g1, g2 = _water_like_geoms()
    scan = TorsionScan(
        name="t",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, energy=0.0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=120.0, geometry=g1, energy=0.5, rotational_constants=np.array([20.0, 16.0, 12.0])),
            TorsionGridPoint(phi=240.0, geometry=g2, energy=0.5, rotational_constants=np.array([20.0, 16.0, 12.0])),
        ],
        angle_unit="degrees",
        energy_unit="kcal/mol",
        periodic=True,
    )
    out = average_torsion_scan_boltzmann(["O", "H", "H"], scan, temperature_K=100.0)
    assert out["weights"][0] > out["weights"][1]


def test_quantum_torsion_average_returns_expected_keys():
    g0, g1, g2 = _water_like_geoms()
    scan = TorsionScan(
        name="q",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=120.0, geometry=g1, rotational_constants=np.array([11.0, 8.5, 6.5])),
            TorsionGridPoint(phi=240.0, geometry=g2, rotational_constants=np.array([11.0, 8.5, 6.5])),
            TorsionGridPoint(phi=360.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=480.0, geometry=g1, rotational_constants=np.array([11.0, 8.5, 6.5])),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    spec = _hr_to_spec(F=5.0, V3=100.0, basis_size=21)
    out = average_torsion_scan_quantum(["O", "H", "H"], scan, spec, state_index=0)
    for k in (
        "averaged_constants",
        "grid_constants",
        "weights",
        "phi_radians",
        "state_index",
        "torsional_energies_cm1",
        "warnings",
        "method",
    ):
        assert k in out
    assert out["method"] == "quantum_hindered_rotor"


def test_torsional_motion_correction_zero_for_identical_grid():
    g0, _, _ = _water_like_geoms()
    scan = TorsionScan(
        name="id",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0),
            TorsionGridPoint(phi=120.0, geometry=g0),
            TorsionGridPoint(phi=240.0, geometry=g0),
            TorsionGridPoint(phi=360.0, geometry=g0),
            TorsionGridPoint(phi=480.0, geometry=g0),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    spec = _hr_to_spec(F=5.0, V3=50.0, basis_size=11)
    out = torsional_motion_correction(["O", "H", "H"], g0, scan, spec=spec, mode="quantum")
    assert np.allclose(out["delta_constants"], 0.0, atol=1e-8)


def test_missing_boltzmann_energies_raises():
    g0, g1, _ = _water_like_geoms()
    scan = TorsionScan(
        name="b",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, energy=0.0),
            TorsionGridPoint(phi=120.0, geometry=g1, energy=None),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    with pytest.raises(ValueError):
        average_torsion_scan_boltzmann(["O", "H", "H"], scan)


def test_small_grid_warning():
    g0, g1, g2 = _water_like_geoms()
    scan = TorsionScan(
        name="small",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=120.0, geometry=g1, rotational_constants=np.array([11.0, 8.5, 6.5])),
            TorsionGridPoint(phi=240.0, geometry=g2, rotational_constants=np.array([11.0, 8.5, 6.5])),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    spec = _hr_to_spec(F=5.0, V3=80.0, basis_size=11)
    out = average_torsion_scan_quantum(["O", "H", "H"], scan, spec)
    joined = " | ".join(out["warnings"]).lower()
    assert "fewer than 5" in joined


def test_sigma_propagation_zero_sigma_is_representational_only():
    """When sigma_abc=0 everywhere, sigma_total equals sigma_representational."""
    C = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    w = np.array([1.0, 1.0, 1.0]) / 3.0
    avg = np.sum(w[:, None] * C, axis=0)
    sigma_grid = np.zeros_like(C)
    unc = propagate_averaging_uncertainty(C, w, avg, sigma_grid)
    assert np.allclose(unc["sigma_statistical"], 0.0)
    assert np.allclose(unc["sigma_total"], unc["sigma_representational"])
    assert np.all(unc["sigma_representational"] >= 0.0)


def test_sigma_propagation_grows_with_measurement_uncertainty():
    """Adding nonzero sigma_grid increases sigma_total relative to no-sigma case."""
    C = np.array([[10.0, 8.0, 6.0], [11.0, 8.5, 6.5], [10.5, 8.2, 6.2]])
    w = np.array([1.0, 1.0, 1.0]) / 3.0
    avg = np.sum(w[:, None] * C, axis=0)
    sigma_grid_zero = np.zeros_like(C)
    sigma_grid_nonzero = np.full_like(C, 0.1)
    unc_zero = propagate_averaging_uncertainty(C, w, avg, sigma_grid_zero)
    unc_nonzero = propagate_averaging_uncertainty(C, w, avg, sigma_grid_nonzero)
    assert np.all(unc_nonzero["sigma_total"] >= unc_zero["sigma_total"])
    assert np.all(unc_nonzero["sigma_statistical"] > 0.0)


def test_sigma_propagation_none_sigma_gives_zero_statistical():
    """When sigma_grid=None, sigma_statistical is zero."""
    C = np.array([[10.0, 8.0, 6.0], [12.0, 9.0, 7.0]])
    w = np.array([0.5, 0.5])
    avg = np.sum(w[:, None] * C, axis=0)
    unc = propagate_averaging_uncertainty(C, w, avg, None)
    assert np.allclose(unc["sigma_statistical"], 0.0)
    assert np.allclose(unc["sigma_total"], unc["sigma_representational"])


def test_sigma_propagation_constant_scan_zero_representational():
    """When all grid points are identical, representational scatter is zero."""
    const_abc = np.array([10.0, 8.0, 6.0])
    C = np.tile(const_abc, (5, 1))
    w = np.ones(5) / 5.0
    avg = np.sum(w[:, None] * C, axis=0)
    unc = propagate_averaging_uncertainty(C, w, avg, None)
    assert np.allclose(unc["sigma_representational"], 0.0, atol=1e-12)
    assert np.allclose(unc["sigma_total"], 0.0, atol=1e-12)


def test_get_grid_sigma_abc_returns_none_when_not_set():
    g0, g1, _ = _water_like_geoms()
    scan = TorsionScan(
        name="s",
        atoms=None,
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0),
            TorsionGridPoint(phi=120.0, geometry=g1),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    assert get_grid_sigma_abc(scan) is None


def test_get_grid_sigma_abc_returns_array_when_set():
    g0, g1, _ = _water_like_geoms()
    scan = TorsionScan(
        name="s",
        atoms=None,
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, sigma_abc=np.array([0.01, 0.005, 0.004])),
            TorsionGridPoint(phi=120.0, geometry=g1, sigma_abc=np.array([0.01, 0.005, 0.004])),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    result = get_grid_sigma_abc(scan)
    assert result is not None
    assert result.shape == (2, 3)
    assert np.allclose(result, 0.01, atol=0.01)


def test_averaging_functions_include_sigma_averaged_key():
    """All three averaging functions must return sigma_averaged."""
    g0, g1, g2 = _water_like_geoms()
    base_gps = [
        TorsionGridPoint(phi=0.0, geometry=g0, energy=0.0, rotational_constants=np.array([10.0, 8.0, 6.0])),
        TorsionGridPoint(phi=120.0, geometry=g1, energy=0.2, rotational_constants=np.array([10.5, 8.3, 6.3])),
        TorsionGridPoint(phi=240.0, geometry=g2, energy=0.2, rotational_constants=np.array([10.5, 8.3, 6.3])),
        TorsionGridPoint(phi=360.0, geometry=g0, energy=0.0, rotational_constants=np.array([10.0, 8.0, 6.0])),
        TorsionGridPoint(phi=480.0, geometry=g1, energy=0.2, rotational_constants=np.array([10.5, 8.3, 6.3])),
    ]
    scan = TorsionScan(name="t", atoms=None, grid_points=base_gps,
                       angle_unit="degrees", energy_unit="kcal/mol", periodic=True)
    spec = _hr_to_spec(F=5.0, V3=80.0, basis_size=21)

    out_q = average_torsion_scan_quantum(["O", "H", "H"], scan, spec)
    out_qt = average_torsion_scan_quantum_thermal(["O", "H", "H"], scan, spec, temperature_K=250.0)
    out_b = average_torsion_scan_boltzmann(["O", "H", "H"], scan)

    for out, name in [(out_q, "quantum"), (out_qt, "quantum_thermal"), (out_b, "boltzmann")]:
        assert "sigma_averaged" in out, f"{name} missing sigma_averaged"
        assert "uncertainty_breakdown" in out, f"{name} missing uncertainty_breakdown"
        assert out["sigma_averaged"].shape == (3,), f"{name} sigma_averaged wrong shape"
        assert np.all(out["sigma_averaged"] >= 0.0), f"{name} sigma_averaged has negative values"


def test_averaging_sigma_grows_when_grid_sigmas_provided():
    """Providing sigma_abc on grid points increases sigma_averaged vs no-sigma."""
    g0, g1, g2 = _water_like_geoms()

    def make_scan(with_sigma):
        gps = []
        for phi, g, abc in [
            (0.0, g0, [10.0, 8.0, 6.0]),
            (120.0, g1, [10.5, 8.3, 6.3]),
            (240.0, g2, [10.5, 8.3, 6.3]),
            (360.0, g0, [10.0, 8.0, 6.0]),
            (480.0, g1, [10.5, 8.3, 6.3]),
        ]:
            sigma = np.array([0.05, 0.03, 0.02]) if with_sigma else None
            gps.append(TorsionGridPoint(
                phi=phi, geometry=g,
                rotational_constants=np.array(abc),
                sigma_abc=sigma,
            ))
        return TorsionScan(name="t", atoms=None, grid_points=gps,
                           angle_unit="degrees", periodic=True)

    spec = _hr_to_spec(F=5.0, V3=80.0, basis_size=21)
    out_no_sig = average_torsion_scan_quantum(["O", "H", "H"], make_scan(False), spec)
    out_with_sig = average_torsion_scan_quantum(["O", "H", "H"], make_scan(True), spec)
    assert np.all(out_with_sig["sigma_averaged"] >= out_no_sig["sigma_averaged"])


def test_lam_uncertainty_uses_sigma_when_provided():
    from backend.torsion_lam_integration import lam_uncertainty_contribution
    sigma = np.array([0.05, 0.03, 0.02])
    result = lam_uncertainty_contribution(10.0, 1, sigma_averaged_cm1=sigma)
    assert np.isclose(result, 0.05)  # max of [0.05, 0.03, 0.02]


def test_lam_uncertainty_falls_back_to_heuristic_when_no_sigma():
    from backend.torsion_lam_integration import lam_uncertainty_contribution
    result = lam_uncertainty_contribution(2.0, 4)
    assert np.isclose(result, 2.0 / np.sqrt(4))


def test_lam_report_includes_sigma_B_key():
    from backend.torsion_lam_integration import lam_correction_report
    B_rigid = np.array([4.25, 0.823, 0.793])
    B_avg = np.array([4.23, 0.820, 0.790])
    sigma_avg = np.array([0.02, 0.01, 0.008])
    report = lam_correction_report(
        B_rigid,
        B_torsion_avg_cm1=B_avg,
        sigma_torsion_avg_cm1=sigma_avg,
        torsion_rms_cm1=0.5,
        n_torsion_levels=4,
        source="torsion_averaged",
    )
    assert "sigma_B_torsion_avg_cm-1" in report
    assert np.isclose(report["lam_uncertainty_cm-1"], 0.02)  # max(sigma_avg)
    assert report["sigma_B_torsion_avg_cm-1"] is not None


def test_lam_report_sigma_B_none_when_not_provided():
    from backend.torsion_lam_integration import lam_correction_report
    B_rigid = np.array([4.25, 0.823, 0.793])
    report = lam_correction_report(
        B_rigid, torsion_rms_cm1=1.0, n_torsion_levels=4, source="torsion_averaged"
    )
    assert report["sigma_B_torsion_avg_cm-1"] is None
    assert np.isclose(report["lam_uncertainty_cm-1"], 1.0 / np.sqrt(4))


def test_quantum_thermal_average_returns_populations():
    g0, g1, g2 = _water_like_geoms()
    scan = TorsionScan(
        name="therm",
        atoms=(0, 1, 2, 2),
        grid_points=[
            TorsionGridPoint(phi=0.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=120.0, geometry=g1, rotational_constants=np.array([11.0, 8.5, 6.5])),
            TorsionGridPoint(phi=240.0, geometry=g2, rotational_constants=np.array([11.0, 8.5, 6.5])),
            TorsionGridPoint(phi=360.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
            TorsionGridPoint(phi=480.0, geometry=g1, rotational_constants=np.array([11.0, 8.5, 6.5])),
        ],
        angle_unit="degrees",
        periodic=True,
    )
    spec = _hr_to_spec(F=5.0, V3=60.0, basis_size=21)
    out = average_torsion_scan_quantum_thermal(["O", "H", "H"], scan, spec, temperature_K=250.0, max_states=4)
    assert out["method"] == "quantum_hindered_rotor_thermal"
    assert out["state_populations"].size == 4
    assert np.isclose(np.sum(out["state_populations"]), 1.0, atol=1e-12)


# ── Phase 5: conformer Boltzmann averaging ────────────────────────────────────

def _make_conformer(abc, energy_cm1):
    """Helper: rigid-body ConformerTorsionSpec with given constants and energy."""
    return ConformerTorsionSpec(
        rotational_constants_cm1=np.asarray(abc, dtype=float),
        relative_energy_cm1=float(energy_cm1),
    )


class TestConformerTorsionAverage:
    """Phase 5 conformer-plus-torsion Boltzmann weighting tests."""

    def test_equal_energy_gives_equal_weights(self):
        """Two conformers at the same energy → 50/50 Boltzmann weight."""
        c1 = _make_conformer([4.0, 0.9, 0.8], 0.0)
        c2 = _make_conformer([4.2, 0.95, 0.85], 0.0)
        out = conformer_torsion_average([c1, c2], temperature_K=298.15)
        w = out["conformer_weights"]
        assert np.isclose(w[0], 0.5, atol=1e-10)
        assert np.isclose(w[1], 0.5, atol=1e-10)

    def test_equal_energy_averaged_constants_are_midpoint(self):
        """50/50 Boltzmann → averaged constants = arithmetic mean of the two sets."""
        abc1 = np.array([4.0, 0.9, 0.8])
        abc2 = np.array([4.2, 0.95, 0.85])
        c1 = _make_conformer(abc1, 0.0)
        c2 = _make_conformer(abc2, 0.0)
        out = conformer_torsion_average([c1, c2], temperature_K=298.15)
        expected = (abc1 + abc2) / 2.0
        assert np.allclose(out["averaged_constants"], expected, atol=1e-10)

    def test_high_energy_conformer_weight_vanishes(self):
        """Conformer at >> kT above minimum contributes negligible weight."""
        # kT at 300 K ≈ 208.5 cm^-1; 5000 cm^-1 >> kT
        c1 = _make_conformer([4.0, 0.9, 0.8], 0.0)
        c2 = _make_conformer([5.0, 1.2, 1.1], 5000.0)
        out = conformer_torsion_average([c1, c2], temperature_K=300.0)
        w = out["conformer_weights"]
        assert w[1] < 1e-6, f"High-energy conformer weight should vanish; got {w[1]:.2e}"

    def test_high_energy_conformer_result_matches_low_energy(self):
        """When one conformer dominates, result ≈ that conformer's constants."""
        abc1 = np.array([4.0, 0.9, 0.8])
        c1 = _make_conformer(abc1, 0.0)
        c2 = _make_conformer([5.0, 1.2, 1.1], 5000.0)
        out = conformer_torsion_average([c1, c2], temperature_K=300.0)
        assert np.allclose(out["averaged_constants"], abc1, atol=1e-4)

    def test_single_conformer_weight_is_one(self):
        """Single conformer → weight = 1, result = its constants, scatter = 0."""
        abc = np.array([4.0, 0.9, 0.8])
        out = conformer_torsion_average([_make_conformer(abc, 0.0)], temperature_K=300.0)
        assert np.isclose(out["conformer_weights"][0], 1.0)
        assert np.allclose(out["averaged_constants"], abc, atol=1e-12)
        assert np.allclose(out["uncertainty_breakdown"]["sigma_conformer_scatter"], 0.0, atol=1e-12)

    def test_scatter_uncertainty_grows_with_spread(self):
        """Larger spread between conformer constants → larger sigma_conformer_scatter."""
        c_narrow1 = _make_conformer([4.0, 0.9, 0.8], 0.0)
        c_narrow2 = _make_conformer([4.01, 0.901, 0.801], 0.0)
        c_wide1 = _make_conformer([4.0, 0.9, 0.8], 0.0)
        c_wide2 = _make_conformer([5.0, 1.5, 1.3], 0.0)
        out_narrow = conformer_torsion_average([c_narrow1, c_narrow2], temperature_K=298.15)
        out_wide = conformer_torsion_average([c_wide1, c_wide2], temperature_K=298.15)
        scatter_narrow = out_narrow["uncertainty_breakdown"]["sigma_conformer_scatter"]
        scatter_wide = out_wide["uncertainty_breakdown"]["sigma_conformer_scatter"]
        assert np.all(scatter_wide > scatter_narrow)

    def test_output_contains_required_keys(self):
        """conformer_torsion_average output must include all documented keys."""
        out = conformer_torsion_average(
            [_make_conformer([4.0, 0.9, 0.8], 0.0),
             _make_conformer([4.2, 0.95, 0.85], 200.0)],
            temperature_K=298.15,
        )
        for key in (
            "conformer_constants", "conformer_weights", "conformer_energies_cm1",
            "averaged_constants", "sigma_averaged", "uncertainty_breakdown",
            "temperature_K", "n_conformers", "warnings", "method",
        ):
            assert key in out, f"Missing key: {key}"
        for subkey in ("sigma_torsion", "sigma_conformer_scatter", "sigma_total"):
            assert subkey in out["uncertainty_breakdown"], f"Missing uncertainty_breakdown.{subkey}"

    def test_method_label(self):
        """method key must be the expected string."""
        out = conformer_torsion_average([_make_conformer([4.0, 0.9, 0.8], 0.0)], temperature_K=298.15)
        assert out["method"] == "conformer_boltzmann_torsion_average"

    def test_n_conformers_matches_input(self):
        """n_conformers equals the number of ConformerTorsionSpec objects supplied."""
        conformers = [_make_conformer([4.0, 0.9, 0.8], float(e)) for e in [0, 100, 300]]
        out = conformer_torsion_average(conformers, temperature_K=298.15)
        assert out["n_conformers"] == 3
        assert out["conformer_constants"].shape == (3, 3)
        assert out["conformer_weights"].shape == (3,)

    def test_weights_sum_to_one(self):
        """Boltzmann weights must always sum to exactly 1."""
        conformers = [_make_conformer([4.0 + 0.1 * i, 0.9, 0.8], float(i) * 50) for i in range(5)]
        out = conformer_torsion_average(conformers, temperature_K=250.0)
        assert np.isclose(np.sum(out["conformer_weights"]), 1.0, atol=1e-12)

    def test_temperature_stored_in_output(self):
        """temperature_K in output matches input."""
        T = 350.0
        out = conformer_torsion_average([_make_conformer([4.0, 0.9, 0.8], 0.0)], temperature_K=T)
        assert np.isclose(out["temperature_K"], T)

    def test_sigma_total_matches_quadrature(self):
        """sigma_total = sqrt(sigma_torsion² + sigma_scatter²)."""
        out = conformer_torsion_average(
            [_make_conformer([4.0, 0.9, 0.8], 0.0),
             _make_conformer([4.5, 1.0, 0.9], 100.0)],
            temperature_K=298.15,
        )
        bd = out["uncertainty_breakdown"]
        expected = np.sqrt(bd["sigma_torsion"] ** 2 + bd["sigma_conformer_scatter"] ** 2)
        assert np.allclose(bd["sigma_total"], expected, atol=1e-12)

    def test_sigma_averaged_equals_sigma_total(self):
        """sigma_averaged at the top level equals uncertainty_breakdown.sigma_total."""
        out = conformer_torsion_average(
            [_make_conformer([4.0, 0.9, 0.8], 0.0),
             _make_conformer([4.2, 0.95, 0.85], 150.0)],
            temperature_K=298.15,
        )
        assert np.allclose(out["sigma_averaged"], out["uncertainty_breakdown"]["sigma_total"])

    def test_empty_conformer_list_raises(self):
        """Empty conformer list must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            conformer_torsion_average([], temperature_K=298.15)

    def test_nonpositive_temperature_raises(self):
        """temperature_K ≤ 0 must raise ValueError."""
        c = _make_conformer([4.0, 0.9, 0.8], 0.0)
        with pytest.raises(ValueError, match="temperature_K"):
            conformer_torsion_average([c], temperature_K=0.0)
        with pytest.raises(ValueError, match="temperature_K"):
            conformer_torsion_average([c], temperature_K=-100.0)

    def test_wrong_length_constants_raises(self):
        """rotational_constants_cm1 with wrong length must raise ValueError."""
        bad = ConformerTorsionSpec(
            rotational_constants_cm1=np.array([4.0, 0.9]),  # only 2 elements
            relative_energy_cm1=0.0,
        )
        with pytest.raises(ValueError):
            conformer_torsion_average([bad], temperature_K=298.15)

    def test_relative_energies_shifted_so_min_is_zero(self):
        """Energies in output are shifted so the minimum is 0, regardless of input values."""
        c1 = _make_conformer([4.0, 0.9, 0.8], 500.0)
        c2 = _make_conformer([4.2, 0.95, 0.85], 800.0)
        out = conformer_torsion_average([c1, c2], temperature_K=298.15)
        assert np.isclose(np.min(out["conformer_energies_cm1"]), 0.0)

    def test_with_torsion_scan_integration(self):
        """Conformer with torsion_scan+hamiltonian uses torsion-averaged constants."""
        g0, g1, g2 = _water_like_geoms()
        scan = TorsionScan(
            name="s",
            atoms=(0, 1, 2, 2),
            grid_points=[
                TorsionGridPoint(phi=0.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
                TorsionGridPoint(phi=120.0, geometry=g1, rotational_constants=np.array([10.5, 8.3, 6.3])),
                TorsionGridPoint(phi=240.0, geometry=g2, rotational_constants=np.array([10.5, 8.3, 6.3])),
                TorsionGridPoint(phi=360.0, geometry=g0, rotational_constants=np.array([10.0, 8.0, 6.0])),
                TorsionGridPoint(phi=480.0, geometry=g1, rotational_constants=np.array([10.5, 8.3, 6.3])),
            ],
            angle_unit="degrees",
            periodic=True,
        )
        spec = _hr_to_spec(F=5.0, V3=80.0, basis_size=21)
        conf_with_scan = ConformerTorsionSpec(
            rotational_constants_cm1=np.array([10.0, 8.0, 6.0]),
            relative_energy_cm1=0.0,
            torsion_scan=scan,
            torsion_hamiltonian=spec,
        )
        conf_rigid = _make_conformer([10.0, 8.0, 6.0], 1000.0)
        out = conformer_torsion_average(
            [conf_with_scan, conf_rigid],
            elements=["O", "H", "H"],
            temperature_K=300.0,
        )
        # Low-energy conformer dominates; result should be finite and physically reasonable
        assert np.all(np.isfinite(out["averaged_constants"]))
        assert np.all(out["averaged_constants"] > 0.0)
