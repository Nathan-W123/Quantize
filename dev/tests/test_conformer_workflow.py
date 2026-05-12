from __future__ import annotations

import numpy as np

from backend.conformer_generation import build_conformer_ensemble
from backend.conformer_mixture import ConformerMixture
from backend.spectral import SpectralEngine


def _chain_coords():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [3.08, 0.2, 0.0],
            [4.20, 1.0, 1.0],
        ],
        dtype=float,
    )


def test_conformer_mixture_coords_use_reference_geometry():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conf = ref.copy()
    conf[2, 2] += 0.4
    mix = ConformerMixture(
        ref,
        conformer_defs=[{"name": "lifted", "coords": conf, "weight": 1.0}],
        weight_mode="fixed",
    )
    coords = mix.conformer_coords(ref)
    assert np.allclose(coords[0], conf)


def test_conformer_mixture_boltzmann_honours_energy_units():
    ref = np.zeros((2, 3), dtype=float)
    mix = ConformerMixture(
        ref,
        conformer_defs=[
            {"name": "low", "offset": np.zeros_like(ref), "energy": 0.0, "energy_unit": "cm-1"},
            {"name": "high", "offset": np.zeros_like(ref), "energy": 500.0, "energy_unit": "cm-1"},
        ],
        weight_mode="boltzmann",
        temperature_k=300.0,
    )
    w = mix.weights()
    assert np.isclose(np.sum(w), 1.0, atol=1e-12)
    assert w[0] > w[1]


def test_build_conformer_ensemble_generates_multiple_distinct_conformers():
    coords = _chain_coords()
    out = build_conformer_ensemble(
        coords,
        ["C", "C", "C", "C"],
        [(0, 1), (1, 2), (2, 3)],
        {
            "enabled": True,
            "weight_mode": "boltzmann",
            "generation": {
                "enabled": True,
                "angle_grid_deg": [60.0, 180.0, 300.0],
                "max_rotatable_bonds": 1,
                "optimize": False,
                "include_input_geometry": True,
                "prune_rmsd_ang": 1.0e-3,
                "prune_constants_mhz": 1.0e-3,
            },
        },
    )
    summary = out["summary"]
    assert out["enabled"] is True
    assert summary["generation"]["rotatable_bonds"]
    assert summary["n_conformers"] >= 2
    assert summary["generated_count"] >= 1


def test_spectral_engine_conformer_mixture_matches_weighted_average():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.1, 1.1, 0.0],
            [3.2, 1.6, 0.9],
        ],
        dtype=float,
    )
    conf = ref.copy()
    conf[3] += np.array([0.2, -0.4, 0.6], dtype=float)
    iso = {
        "name": "toy",
        "masses": [12.0, 12.0, 12.0, 12.0],
        "obs_constants": [0.0, 0.0, 0.0],
        "sigma_constants": [1.0, 1.0, 1.0],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    }
    engine = SpectralEngine(
        [iso],
        conformer_defs=[
            {"name": "a", "coords": ref, "weight": 0.25},
            {"name": "b", "coords": conf, "weight": 0.75},
        ],
        conformer_reference_coords=ref,
        conformer_weight_mode="fixed",
    )
    calc_ref = engine.rotational_constants(ref, np.asarray(iso["masses"], dtype=float))
    calc_conf = engine.rotational_constants(conf, np.asarray(iso["masses"], dtype=float))
    expected = 0.25 * calc_ref + 0.75 * calc_conf
    _, residual = engine.stacked_unweighted(ref)
    np.testing.assert_allclose(-residual, expected, rtol=1e-10, atol=1e-8)
