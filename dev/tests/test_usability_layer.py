from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from runner.usability import (
    ConfigError,
    load_config,
    prepare_run_directory,
    validate_config,
    write_outputs,
)


def minimal_water_config():
    return {
        "name": "water_test",
        "elements": ["O", "H", "H"],
        "geometry": {
            "method": "coords",
            "coords_angstrom": [
                [0.0, 0.0, 0.1174],
                [0.0, 0.7572, -0.4696],
                [0.0, -0.7572, -0.4696],
            ],
        },
        "isotopologues": [
            {
                "name": "H2-16O",
                "masses": [15.99491461956, 1.00782503207, 1.00782503207],
                "components": ["A", "B", "C"],
                "obs_b0_mhz": [835840.3, 435351.7, 278138.7],
                "alpha_mhz": [-43390.0, 10560.0, 6240.0],
                "sigma_mhz": [0.2, 0.2, 0.2],
            }
        ],
        "quantum": {"backend": "none"},
        "preset": "FAST_DEBUG",
    }


def test_load_config_accepts_json(tmp_path):
    path = tmp_path / "water.json"
    path.write_text(json.dumps(minimal_water_config()), encoding="utf-8")

    cfg = load_config(path)

    assert cfg["name"] == "water_test"
    validate_config(cfg)


def test_validate_config_reports_bad_sigma():
    cfg = minimal_water_config()
    cfg["isotopologues"][0]["sigma_mhz"][1] = 0.0

    with pytest.raises(ConfigError, match="sigma_mhz\\[1\\].*positive"):
        validate_config(cfg)


def test_prepare_run_directory_copies_input(tmp_path):
    cfg = minimal_water_config()
    cfg["output"] = {"root": str(tmp_path / "runs")}
    input_path = tmp_path / "input.yaml"
    input_path.write_text("name: water_test\n", encoding="utf-8")

    run_dir = prepare_run_directory(cfg, input_path)

    assert run_dir.is_dir()
    assert (run_dir / "plots").is_dir()
    assert (run_dir / "exports").is_dir()
    assert (run_dir / "input.yaml").is_file()
    assert Path(cfg["_run_dir"]) == run_dir


def test_write_outputs_creates_report_csvs_and_plots(tmp_path):
    coords = np.array(
        [
            [0.0, 0.0, 0.1174],
            [0.0, 0.7572, -0.4696],
            [0.0, -0.7572, -0.4696],
        ],
        dtype=float,
    )
    result = {
        "name": "water_test",
        "run_dir": str(tmp_path),
        "elems": ["O", "H", "H"],
        "best": {
            "idx": 1,
            "coords": coords,
            "freq_rms": 1.23,
            "energy": -76.0,
            "history": [
                {"iteration": 1, "freq_rms": 5.0, "step_norm": 1e-2},
                {"iteration": 2, "freq_rms": 1.23, "step_norm": 1e-4},
            ],
            "spectral_isotopologues_snapshot": [
                {
                    "name": "H2-16O",
                    "masses": [15.99491461956, 1.00782503207, 1.00782503207],
                    "obs_constants": [835840.3, 435351.7, 278138.7],
                    "sigma_constants": [0.2, 0.2, 0.2],
                    "alpha_constants": [-43390.0, 10560.0, 6240.0],
                    "component_indices": [0, 1, 2],
                }
            ],
        },
        "score": {"score": 75.0, "constrained_rank": 3, "internal_dof": 3},
    }

    artifacts = write_outputs(result)

    assert artifacts["report_md"].is_file()
    assert artifacts["geometry_csv"].is_file()
    assert artifacts["residuals_csv"].is_file()
    with artifacts["residuals_csv"].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["component"] for row in rows] == ["A", "B", "C"]
    assert (tmp_path / "plots" / "residuals.png").is_file()
    assert (tmp_path / "plots" / "singular_values.png").is_file()
    assert (tmp_path / "plots" / "convergence.png").is_file()
