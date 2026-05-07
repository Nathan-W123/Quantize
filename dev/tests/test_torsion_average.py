from __future__ import annotations

import numpy as np
import pytest

from backend.torsion_average import (
    TorsionGridPoint,
    TorsionScan,
    average_rotational_constants_with_weights,
    average_torsion_scan_boltzmann,
    average_torsion_scan_quantum,
    average_torsion_scan_quantum_thermal,
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
