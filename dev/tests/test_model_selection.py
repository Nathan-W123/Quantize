from __future__ import annotations

from backend.model_selection import (
    ModeCalibration,
    calibrate_mode_thresholds,
    recommend_coordinate_mode,
    recommend_torsion_model,
)


def test_recommend_coordinate_mode_prefers_lower_rms_when_scores_similar():
    summary = {
        "cartesian": {"freq_rms_mhz": 2.0, "success_score": 70.0},
        "internal": {"freq_rms_mhz": 1.0, "success_score": 69.0},
    }
    rec = recommend_coordinate_mode(summary, calibration=ModeCalibration(low_delta=0.1, moderate_delta=0.5))
    assert rec.recommended_mode == "internal"
    assert rec.confidence in {"low", "moderate", "high"}
    assert rec.delta > 0.0


def test_recommend_coordinate_mode_can_prefer_cartesian():
    summary = {
        "cartesian": {"freq_rms_mhz": 1.0, "success_score": 85.0},
        "internal": {"freq_rms_mhz": 3.0, "success_score": 60.0},
    }
    rec = recommend_coordinate_mode(summary)
    assert rec.recommended_mode == "cartesian"


def test_calibrate_mode_thresholds_from_history_rows():
    rows = []
    for i in range(8):
        rows.append(
            {
                "config": f"cfg{i}",
                "mode": "cartesian",
                "freq_rms_mhz": 2.0 + i,
                "success_score": 50.0,
            }
        )
        rows.append(
            {
                "config": f"cfg{i}",
                "mode": "internal",
                "freq_rms_mhz": 1.0 + 0.5 * i,
                "success_score": 55.0,
            }
        )
    cal = calibrate_mode_thresholds(rows)
    assert cal.source == "history"
    assert cal.n_pairs >= 5
    assert cal.moderate_delta > cal.low_delta


def test_recommend_torsion_model_not_applicable_when_disabled():
    cfg = {"torsion_hamiltonian": {"enabled": False}}
    rec = recommend_torsion_model(cfg)
    assert rec.recommended_model == "not_applicable"
    assert rec.reliability_score is None


def test_recommend_torsion_model_ram_lite_or_caution_for_reasonable_spec():
    cfg = {
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 27.0,
            "rho": 0.81,
            "A": 4.25,
            "B": 0.823,
            "C": 0.793,
            "n_basis": 10,
            "potential": {"v0": 186.0, "vcos": {3: -186.0}, "vsin": {}},
        }
    }
    rec = recommend_torsion_model(cfg)
    assert rec.recommended_model in {"ram_lite", "ram_lite_with_caution"}
    assert rec.reliability_score in {"high", "moderate", "low", "unreliable"}
