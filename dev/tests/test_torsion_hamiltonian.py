from __future__ import annotations

import numpy as np

from backend.torsion_hamiltonian import (
    TorsionEffectiveConstantFourier,
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    build_ram_lite_hamiltonian,
    evaluate_effective_torsion_constant_on_grid,
    solve_ram_lite_levels,
    torsion_probability_density,
    motion_average_constants_on_grid,
    torsion_objective_from_levels,
    assign_levels_by_keys,
)


def test_hamiltonian_is_hermitian():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 10.0}, units="cm-1")
    spec = TorsionHamiltonianSpec(F=5.0, rho=0.2, A=1.0, B=0.4, potential=pot, n_basis=5)
    H, _, _ = build_ram_lite_hamiltonian(spec, J=3, K=1)
    assert np.allclose(H, H.conj().T)


def test_free_rotor_spectrum_has_m2_trend():
    pot = TorsionFourierPotential(v0=0.0, units="cm-1")
    spec = TorsionHamiltonianSpec(F=2.0, rho=0.0, A=0.0, B=0.0, potential=pot, n_basis=4)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=5)
    e = out["energies_cm-1"]
    # Lowest levels should be 0, 2, 2, 8, 8 for F=2 and m^2 ladder (degenerate +/-m)
    assert np.isclose(e[0], 0.0, atol=1e-10)
    assert np.isclose(e[1], 2.0, atol=1e-8)
    assert np.isclose(e[2], 2.0, atol=1e-8)


def test_barrier_changes_ground_state_energy():
    pot0 = TorsionFourierPotential(v0=0.0, units="cm-1")
    pot1 = TorsionFourierPotential(v0=0.0, vcos={3: 30.0}, units="cm-1")
    spec0 = TorsionHamiltonianSpec(F=3.0, potential=pot0, n_basis=6)
    spec1 = TorsionHamiltonianSpec(F=3.0, potential=pot1, n_basis=6)
    e0 = solve_ram_lite_levels(spec0, J=0, K=0, n_levels=1)["energies_cm-1"][0]
    e1 = solve_ram_lite_levels(spec1, J=0, K=0, n_levels=1)["energies_cm-1"][0]
    assert not np.isclose(e0, e1)


def test_rho_coupling_changes_k_blocks():
    pot = TorsionFourierPotential(v0=0.0, units="cm-1")
    spec = TorsionHamiltonianSpec(F=4.0, rho=0.3, A=0.0, B=0.0, potential=pot, n_basis=5)
    e_k0 = solve_ram_lite_levels(spec, J=1, K=0, n_levels=1)["energies_cm-1"][0]
    e_k1 = solve_ram_lite_levels(spec, J=1, K=1, n_levels=1)["energies_cm-1"][0]
    assert not np.isclose(e_k0, e_k1)


def test_completeness_terms_default_preserve_legacy_behavior():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 8.0}, units="cm-1")
    spec_legacy = TorsionHamiltonianSpec(F=4.0, rho=0.3, A=1.2, B=0.5, potential=pot, n_basis=5)
    spec_with_explicit_zeros = TorsionHamiltonianSpec(
        F=4.0,
        rho=0.3,
        F4=0.0,
        F6=0.0,
        c_mk=0.0,
        c_k2=0.0,
        A=1.2,
        B=0.5,
        potential=pot,
        n_basis=5,
    )

    out_legacy = solve_ram_lite_levels(spec_legacy, J=2, K=1, n_levels=6)
    out_zeroed = solve_ram_lite_levels(spec_with_explicit_zeros, J=2, K=1, n_levels=6)

    assert np.allclose(out_legacy["energies_cm-1"], out_zeroed["energies_cm-1"], atol=1e-12, rtol=0.0)


def test_completeness_terms_enabled_change_spectrum():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 8.0}, units="cm-1")
    base = TorsionHamiltonianSpec(F=4.0, rho=0.25, A=1.0, B=0.5, potential=pot, n_basis=6)
    enhanced = TorsionHamiltonianSpec(
        F=4.0,
        rho=0.25,
        F4=0.03,
        F6=0.002,
        c_mk=0.15,
        c_k2=0.4,
        A=1.0,
        B=0.5,
        potential=pot,
        n_basis=6,
    )

    e_base = solve_ram_lite_levels(base, J=3, K=2, n_levels=4)["energies_cm-1"]
    out_enhanced = solve_ram_lite_levels(enhanced, J=3, K=2, n_levels=4)
    e_enhanced = out_enhanced["energies_cm-1"]

    assert not np.allclose(e_base, e_enhanced)
    assert any("completeness terms" in w.lower() for w in out_enhanced["warnings"])


def test_probability_density_normalizes():
    pot = TorsionFourierPotential(v0=0.0, units="cm-1")
    spec = TorsionHamiltonianSpec(F=2.0, potential=pot, n_basis=3)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=1)
    v = out["eigenvectors"][:, 0]
    m = out["m_values"]
    alpha = np.linspace(-np.pi, np.pi, 181)
    p = torsion_probability_density(v, alpha, m)
    assert np.isclose(np.sum(p), 1.0, atol=1e-10)
    assert np.all(p >= 0.0)


def test_motion_average_constants_on_grid():
    B = np.array(
        [
            [100.0, 50.0, 30.0],
            [110.0, 55.0, 33.0],
        ]
    )
    p = np.array([1.0, 3.0])
    avg = motion_average_constants_on_grid(B, p)
    assert np.allclose(avg, [107.5, 53.75, 32.25])


def test_torsion_objective_from_levels_rms():
    pred = [
        {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 10.0},
        {"J": 0, "K": 0, "level_index": 1, "energy_cm-1": 20.0},
    ]
    targ = [
        {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 12.0},
        {"J": 0, "K": 0, "level_index": 1, "energy_cm-1": 18.0},
    ]
    out = torsion_objective_from_levels(pred, targ)
    assert len(out["rows"]) == 2
    # residuals: +2, -2 => RMS = 2
    assert np.isclose(out["rms_cm-1"], 2.0)


def test_assign_levels_by_keys_cross_block_label_sublabel_priority():
    pred = [
        {"J": 0, "K": 0, "level_index": 0, "label": "A", "sublabel": "A", "energy_cm-1": 10.0},
        {"J": 0, "K": 0, "level_index": 0, "label": "E", "sublabel": "E1", "energy_cm-1": 20.0},
        {"J": 0, "K": 0, "level_index": 0, "label": "E", "sublabel": "E2", "energy_cm-1": 21.0},
    ]
    targ = [
        {"J": 0, "K": 0, "index": 0, "label": "E", "sublabel": "E2", "energy_cm-1": 22.0},
        {"J": 0, "K": 0, "index": 0, "label": "A", "sublabel": "A", "energy_cm-1": 11.0},
        {"J": 0, "K": 0, "index": 0, "label": "E", "sublabel": "E1", "energy_cm-1": 19.0},
    ]
    out = assign_levels_by_keys(pred, targ)
    assert len(out["matches"]) == 3
    assert out["unmatched_predicted"] == []
    assert out["unmatched_target"] == []
    # Residuals: +1, +1, -1 => RMS = 1
    assert np.isclose(out["rms_cm-1"], 1.0)


def test_assign_levels_by_keys_ambiguous_collision_warns_and_pairs_order():
    pred = [
        {"J": 1, "K": 0, "level_index": 0, "energy_cm-1": 10.0},
        {"J": 1, "K": 0, "level_index": 0, "energy_cm-1": 11.0},
    ]
    targ = [
        {"J": 1, "K": 0, "level_index": 0, "energy_cm-1": 12.0},
        {"J": 1, "K": 0, "level_index": 0, "energy_cm-1": 13.0},
    ]
    out = assign_levels_by_keys(pred, targ, key_priority=[("J", "K", "level_index")])
    assert len(out["matches"]) == 2
    assert any("ambiguous collision" in w.lower() for w in out["warnings"])
    assert np.isclose(out["matches"][0]["residual_cm-1"], 2.0)
    assert np.isclose(out["matches"][1]["residual_cm-1"], 2.0)


def test_assign_levels_by_keys_missing_target_and_objective_backward_compatible():
    pred = [
        {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 10.0},
    ]
    targ = [
        {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 11.0},
        {"J": 0, "K": 0, "level_index": 1, "energy_cm-1": 20.0},
    ]
    out = torsion_objective_from_levels(pred, targ)
    assert "rows" in out
    assert "rms_cm-1" in out
    assert "warnings" in out
    assert "unmatched_target" in out
    assert len(out["rows"]) == 1
    assert out["unmatched_target"] == [1]
    assert any("missing predicted level" in w.lower() for w in out["warnings"])


def test_c3_symmetry_labels_and_blocks_for_symmetric_potential():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 24.0}, units="cm-1")
    spec = TorsionHamiltonianSpec(F=2.0, rho=0.0, A=0.0, B=0.0, potential=pot, n_basis=6)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=6, symmetry_mode="c3", return_blocks=True)

    assert "symmetry_labels" in out
    assert "symmetry_sublabels" in out
    assert "symmetry_purity" in out
    assert "symmetry_blocks" in out

    labels = list(out["symmetry_labels"])
    assert set(labels).issubset({"A", "E"})
    assert "A" in labels
    assert "E" in labels
    assert np.all(out["symmetry_purity"] > 0.95)
    assert set(out["symmetry_blocks"].keys()) == {"A", "E1", "E2"}


def test_c3_symmetric_case_shows_low_level_e_pair_near_degeneracy():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 18.0}, units="cm-1")
    spec = TorsionHamiltonianSpec(F=2.0, rho=0.0, A=0.0, B=0.0, potential=pot, n_basis=7)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=8, symmetry_mode="c3", return_blocks=True)

    e1 = out["symmetry_blocks"]["E1"]["energies_cm-1"][0]
    e2 = out["symmetry_blocks"]["E2"]["energies_cm-1"][0]
    assert np.isclose(e1, e2, atol=1e-8)


def test_c3_mode_warns_if_potential_has_non_threefold_terms():
    pot = TorsionFourierPotential(v0=0.0, vcos={2: 5.0, 3: 10.0}, units="cm-1")
    spec = TorsionHamiltonianSpec(F=3.0, potential=pot, n_basis=5)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=3, symmetry_mode="c3")
    joined = " | ".join(out["warnings"]).lower()
    assert "non-3-fold harmonics" in joined


def test_effective_Falpha_constant_matches_scalar_F_hamiltonian():
    pot = TorsionFourierPotential(v0=0.0, vcos={3: 5.0}, units="cm-1")
    scalar_spec = TorsionHamiltonianSpec(
        F=4.0,
        rho=0.25,
        A=1.2,
        B=0.45,
        potential=pot,
        n_basis=5,
    )
    falpha_spec = TorsionHamiltonianSpec(
        F=4.0,
        rho=0.25,
        A=1.2,
        B=0.45,
        potential=pot,
        n_basis=5,
        F_alpha=TorsionEffectiveConstantFourier(f0=4.0),
    )
    H_scalar, _, _ = build_ram_lite_hamiltonian(scalar_spec, J=2, K=1)
    H_falpha, _, _ = build_ram_lite_hamiltonian(falpha_spec, J=2, K=1)
    assert np.allclose(H_scalar, H_falpha, atol=1e-12, rtol=0.0)


def test_effective_Falpha_nonconstant_changes_spectrum():
    pot = TorsionFourierPotential(v0=0.0, units="cm-1")
    base_spec = TorsionHamiltonianSpec(F=3.0, rho=0.2, potential=pot, n_basis=6)
    mod_spec = TorsionHamiltonianSpec(
        F=3.0,
        rho=0.2,
        potential=pot,
        n_basis=6,
        F_alpha=TorsionEffectiveConstantFourier(f0=3.0, fcos={1: 0.9}),
    )
    e_base = solve_ram_lite_levels(base_spec, J=1, K=1, n_levels=4)["energies_cm-1"]
    e_mod = solve_ram_lite_levels(mod_spec, J=1, K=1, n_levels=4)["energies_cm-1"]
    assert np.max(np.abs(e_mod - e_base)) > 1e-6


def test_effective_Falpha_grid_eval_and_warning_for_nonpositive_region():
    falpha = TorsionEffectiveConstantFourier(f0=1.0, fcos={1: 1.2})
    alpha = np.array([0.0, np.pi])
    vals = evaluate_effective_torsion_constant_on_grid(falpha, alpha)
    assert np.isclose(vals[0], 2.2)
    assert np.isclose(vals[1], -0.2)

    spec = TorsionHamiltonianSpec(
        F=1.0,
        potential=TorsionFourierPotential(v0=0.0, units="cm-1"),
        n_basis=4,
        F_alpha=falpha,
    )
    _, _, warnings = build_ram_lite_hamiltonian(spec, J=0, K=0)
    assert any("non-positive" in w for w in warnings)
