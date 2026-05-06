from __future__ import annotations

import numpy as np
import pytest

from backend.hindered_rotor import HinderedRotorModel
from backend.torsion_average import (
    TorsionGridPoint,
    TorsionScan,
    average_rotational_constants_with_weights,
    average_torsion_scan_boltzmann,
    average_torsion_scan_quantum,
    average_torsion_scan_quantum_thermal,
    torsional_motion_correction,
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
    model = HinderedRotorModel(
        name="m",
        symmetry_number=3,
        rotational_constant_F=5.0,
        rotational_constant_unit="cm-1",
        fourier_terms={3: 100.0},
        potential_energy_unit="cm-1",
        basis_size=21,
    )
    out = average_torsion_scan_quantum(["O", "H", "H"], scan, model, state_index=0)
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
    model = HinderedRotorModel(name="m", rotational_constant_F=5.0, fourier_terms={3: 50.0}, basis_size=11)
    out = torsional_motion_correction(["O", "H", "H"], g0, scan, model=model, mode="quantum")
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
    model = HinderedRotorModel(name="m", rotational_constant_F=5.0, fourier_terms={3: 80.0}, basis_size=11)
    out = average_torsion_scan_quantum(["O", "H", "H"], scan, model)
    joined = " | ".join(out["warnings"]).lower()
    assert "fewer than 5" in joined


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
    model = HinderedRotorModel(name="m", rotational_constant_F=5.0, fourier_terms={3: 60.0}, basis_size=21)
    out = average_torsion_scan_quantum_thermal(["O", "H", "H"], scan, model, temperature_K=250.0, max_states=4)
    assert out["method"] == "quantum_hindered_rotor_thermal"
    assert out["state_populations"].size == 4
    assert np.isclose(np.sum(out["state_populations"]), 1.0, atol=1e-12)
