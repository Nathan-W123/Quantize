from __future__ import annotations

import numpy as np

from runner.run_generic import (
    _predict_torsion_levels_for_coords,
    _run_torsion_phase2_exports,
    _torsion_transition_objective_from_levels,
)


def _water_coords():
    return np.array(
        [
            [0.0, 0.0, 0.1174],
            [0.0, 0.7572, -0.4696],
            [0.0, -0.7572, -0.4696],
        ],
        dtype=float,
    )


def test_torsion_phase2_writes_exports(tmp_path):
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.1,
            "n_basis": 5,
            "n_levels": 4,
            "J_values": [0, 1],
            "K_values": [0],
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
        },
    }
    isotopologues = [
        {
            "name": "H2-16O",
            "masses": [15.99491461956, 1.00782503207, 1.00782503207],
        }
    ]
    best = {"coords": _water_coords()}

    out = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=["O", "H", "H"],
        best=best,
        isotopologues=isotopologues,
        run_dir=tmp_path,
    )
    assert out["torsion_levels_csv"].is_file()
    assert out["torsion_summary_json"].is_file()
    text = out["torsion_summary_json"].read_text(encoding="utf-8")
    assert "ram_lite" in text


def test_predict_torsion_levels_for_coords_returns_rows():
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.1,
            "n_basis": 5,
            "n_levels": 3,
            "J_values": [0],
            "K_values": [0],
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
        },
    }
    isotopologues = [{"name": "H2-16O", "masses": [15.99491461956, 1.00782503207, 1.00782503207]}]
    rows, warnings = _predict_torsion_levels_for_coords(
        cfg, ["O", "H", "H"], _water_coords(), isotopologues
    )
    assert len(rows) == 3
    assert all("energy_cm-1" in r for r in rows)
    assert isinstance(warnings, list)


def test_torsion_transition_objective_from_levels_deterministic():
    predicted_rows = [
        {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 1.0},
        {"J": 1, "K": 0, "level_index": 1, "energy_cm-1": 4.0},
    ]
    transitions = [
        {"J_lo": 0, "K_lo": 0, "level_lo": 0, "J_hi": 1, "K_hi": 0, "level_hi": 1, "freq_cm-1": 2.5}
    ]
    out = _torsion_transition_objective_from_levels(predicted_rows, transitions)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["predicted_cm-1"] == 3.0
    assert out["rows"][0]["residual_cm-1"] == -0.5
    assert out["rms_cm-1"] == 0.5


def test_torsion_phase2_writes_transition_objective_csv(tmp_path):
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.1,
            "n_basis": 5,
            "n_levels": 4,
            "J_values": [0, 1],
            "K_values": [0],
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
            "transitions": [
                {"J_lo": 0, "K_lo": 0, "level_lo": 0, "J_hi": 1, "K_hi": 0, "level_hi": 1, "freq_cm-1": 1.0}
            ],
        },
    }
    isotopologues = [
        {
            "name": "H2-16O",
            "masses": [15.99491461956, 1.00782503207, 1.00782503207],
        }
    ]
    best = {"coords": _water_coords()}
    out = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=["O", "H", "H"],
        best=best,
        isotopologues=isotopologues,
        run_dir=tmp_path,
    )
    assert out["torsion_transition_objective_csv"] is not None
    assert out["torsion_transition_objective_csv"].is_file()


def test_torsion_phase2_writes_uncertainty_csv(tmp_path):
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.1,
            "n_basis": 7,
            "n_levels": 4,
            "J_values": [0],
            "K_values": [0],
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
            "targets": [
                {"J": 0, "K": 0, "level_index": 0, "energy_cm-1": 0.1},
                {"J": 0, "K": 0, "level_index": 1, "energy_cm-1": 0.2},
            ],
            "uncertainty": {"enabled": True, "damping": 1e-8},
        },
    }
    isotopologues = [
        {"name": "H2-16O", "masses": [15.99491461956, 1.00782503207, 1.00782503207]}
    ]
    best = {"coords": _water_coords()}
    out = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=["O", "H", "H"],
        best=best,
        isotopologues=isotopologues,
        run_dir=tmp_path,
    )
    assert out["torsion_parameter_uncertainty_csv"] is not None
    assert out["torsion_parameter_uncertainty_csv"].is_file()


def test_torsion_phase2_writes_symmetry_blocks_csv(tmp_path):
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.0,
            "n_basis": 5,
            "n_levels": 4,
            "J_values": [0],
            "K_values": [0],
            "symmetry_mode": "c3",
            "export_symmetry_blocks": True,
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
        },
    }
    isotopologues = [{"name": "H2-16O", "masses": [15.99491461956, 1.00782503207, 1.00782503207]}]
    out = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=["O", "H", "H"],
        best={"coords": _water_coords()},
        isotopologues=isotopologues,
        run_dir=tmp_path,
    )
    assert out["torsion_symmetry_blocks_csv"] is not None
    assert out["torsion_symmetry_blocks_csv"].is_file()


def test_torsion_phase2_writes_scan_average_csv(tmp_path):
    cfg = {
        "name": "water_torsion_test",
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 5.0,
            "rho": 0.1,
            "n_basis": 5,
            "n_levels": 3,
            "J_values": [0],
            "K_values": [0],
            "potential": {"v0": 0.0, "vcos": {"3": 20.0}},
            "scan": {
                "mode": "boltzmann",
                "angle_unit": "degrees",
                "energy_unit": "cm-1",
                "grid_points": [
                    {"phi": 0.0, "energy": 0.0, "rotational_constants": [10.0, 8.0, 6.0]},
                    {"phi": 120.0, "energy": 1.0, "rotational_constants": [11.0, 8.5, 6.5]},
                    {"phi": 240.0, "energy": 1.0, "rotational_constants": [11.0, 8.5, 6.5]},
                    {"phi": 360.0, "energy": 0.0, "rotational_constants": [10.0, 8.0, 6.0]},
                    {"phi": 480.0, "energy": 1.0, "rotational_constants": [11.0, 8.5, 6.5]},
                ],
            },
        },
    }
    isotopologues = [{"name": "H2-16O", "masses": [15.99491461956, 1.00782503207, 1.00782503207]}]
    out = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=["O", "H", "H"],
        best={"coords": _water_coords()},
        isotopologues=isotopologues,
        run_dir=tmp_path,
    )
    assert out["torsion_scan_average_csv"] is not None
    assert out["torsion_scan_average_csv"].is_file()
