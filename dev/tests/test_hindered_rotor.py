from __future__ import annotations

import numpy as np
import pytest

from backend.hindered_rotor import (
    HinderedRotorModel,
    basis_convergence_report,
    boltzmann_torsion_weights,
    build_fourier_potential,
    build_hindered_rotor_hamiltonian,
    convert_energy_to_cm1,
    model_diagnostics,
    solve_hindered_rotor,
    torsional_probability_on_grid,
)


def test_energy_conversion_to_cm1():
    cm = convert_energy_to_cm1(1.0, "cm-1")
    kcal = convert_energy_to_cm1(1.0, "kcal/mol")
    kj = convert_energy_to_cm1(1.0, "kj/mol")
    eh = convert_energy_to_cm1(1.0, "hartree")
    assert cm > 0.0
    assert kcal > cm
    assert kj > cm
    assert eh > kcal


def test_build_fourier_potential_zero_terms():
    phi = np.linspace(0.0, 2.0 * np.pi, 16)
    V = build_fourier_potential(phi, {}, unit="cm-1")
    assert np.allclose(V, 0.0)


def test_build_fourier_potential_v3_periodicity():
    phi = np.array([0.1, 0.7, 1.8])
    V = build_fourier_potential(phi, {3: 100.0}, unit="cm-1")
    V_shift = build_fourier_potential(phi + 2.0 * np.pi / 3.0, {3: 100.0}, unit="cm-1")
    assert np.allclose(V, V_shift, atol=1e-10)


def test_hamiltonian_shape_and_symmetry():
    model = HinderedRotorModel(
        name="t",
        rotational_constant_F=1.0,
        basis_size=5,
        fourier_terms={3: 50.0},
    )
    H, m = build_hindered_rotor_hamiltonian(model)
    assert H.shape == (5, 5)
    assert m.shape == (5,)
    assert np.allclose(H, H.T)


def test_hamiltonian_rejects_even_basis():
    model = HinderedRotorModel(name="t", rotational_constant_F=1.0, basis_size=10)
    with pytest.raises(ValueError):
        build_hindered_rotor_hamiltonian(model)


def test_free_rotor_energies():
    model = HinderedRotorModel(name="free", rotational_constant_F=1.0, basis_size=9, fourier_terms={})
    out = solve_hindered_rotor(model)
    e = out["energies_cm1"]
    assert np.isclose(e[0], 0.0, atol=1e-10)
    assert np.isclose(e[1], 1.0, atol=1e-10)
    assert np.isclose(e[2], 1.0, atol=1e-10)
    assert np.isclose(e[3], 4.0, atol=1e-10)


def test_torsional_probability_weights_sum_to_one():
    model = HinderedRotorModel(name="free", rotational_constant_F=1.0, basis_size=9)
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    out = torsional_probability_on_grid(model, phi, state_index=0)
    w = out["weights"]
    assert np.isclose(np.sum(w), 1.0, atol=1e-12)
    assert np.all(w >= 0.0)


def test_boltzmann_weights_prefers_low_energy():
    w = boltzmann_torsion_weights([0.0, 0.01], temperature_K=298.15, energy_unit="hartree")
    assert w[0] > w[1]


def test_model_diagnostics_warns_on_symmetry_mismatch():
    model = HinderedRotorModel(
        name="diag",
        symmetry_number=3,
        rotational_constant_F=1.0,
        fourier_terms={2: 10.0, 3: 50.0},
        basis_size=9,
    )
    warns = model_diagnostics(model)
    assert any("inconsistent" in w.lower() for w in warns)


def test_basis_convergence_report_runs():
    model = HinderedRotorModel(name="conv", rotational_constant_F=2.0, fourier_terms={3: 20.0}, basis_size=11)
    rep = basis_convergence_report(model, basis_sizes=[7, 11, 15], state_index=0)
    assert rep["basis_sizes"].size >= 2
    assert rep["energies_cm1"].size >= 2
