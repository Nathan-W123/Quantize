from __future__ import annotations

import numpy as np
import pytest

from backend.large_amplitude import (
    LargeAmplitudeGridPoint,
    LargeAmplitudeModel,
    normalize_weights,
    boltzmann_weights,
    average_rotational_constants,
    effective_motion_correction,
)


def _water_geom(r_oh=0.958, angle_deg=104.5):
    ang = np.radians(angle_deg / 2.0)
    x = r_oh * np.sin(ang)
    z = r_oh * np.cos(ang)
    return np.array(
        [
            [0.0, 0.0, 0.0],     # O
            [x, 0.0, z],         # H
            [-x, 0.0, z],        # H
        ],
        dtype=float,
    )


def test_normalize_weights_basic():
    w = normalize_weights([1.0, 1.0, 2.0])
    assert np.allclose(w, [0.25, 0.25, 0.5])


def test_normalize_weights_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_weights([1.0, -1.0, 2.0])


def test_boltzmann_weights_prefers_low_energy():
    w = boltzmann_weights([0.0, 0.02], temperature_K=298.15, units="hartree")
    assert w[0] > w[1]


def test_boltzmann_weights_shift_invariant():
    w1 = boltzmann_weights([0.0, 0.005, 0.01], temperature_K=200.0, units="hartree")
    w2 = boltzmann_weights([1.0, 1.005, 1.01], temperature_K=200.0, units="hartree")
    assert np.allclose(w1, w2)


def test_average_rotational_constants_user_weights():
    g1 = _water_geom(r_oh=0.95, angle_deg=104.5)
    g2 = _water_geom(r_oh=0.97, angle_deg=104.5)
    model = LargeAmplitudeModel(
        name="water_stretch",
        coordinate_type="custom",
        weighting="user",
        grid_points=[
            LargeAmplitudeGridPoint(0.0, g1, weight=1.0),
            LargeAmplitudeGridPoint(1.0, g2, weight=3.0),
        ],
    )
    out = average_rotational_constants(["O", "H", "H"], model)
    assert out["grid_constants"].shape == (2, 3)
    # closer to second point due to larger weight
    assert np.all(np.isfinite(out["averaged_constants"]))
    assert np.allclose(out["weights"], [0.25, 0.75])


def test_effective_motion_correction_zero_for_identical_grid():
    g = _water_geom()
    model = LargeAmplitudeModel(
        name="identical_grid",
        coordinate_type="torsion",
        weighting="user",
        grid_points=[
            LargeAmplitudeGridPoint(-1.0, g, weight=1.0),
            LargeAmplitudeGridPoint(0.0, g, weight=1.0),
            LargeAmplitudeGridPoint(1.0, g, weight=1.0),
        ],
    )
    out = effective_motion_correction(["O", "H", "H"], g, model)
    assert np.allclose(out["delta"], 0.0, atol=1e-9)


def test_missing_user_weights_raises_or_warns():
    g1 = _water_geom(r_oh=0.95)
    g2 = _water_geom(r_oh=0.97)
    model = LargeAmplitudeModel(
        name="bad_user_weights",
        coordinate_type="custom",
        weighting="user",
        grid_points=[
            LargeAmplitudeGridPoint(0.0, g1, weight=1.0),
            LargeAmplitudeGridPoint(1.0, g2, weight=None),
        ],
    )
    with pytest.raises(ValueError, match="weights are missing"):
        average_rotational_constants(["O", "H", "H"], model)


def test_floppy_model_adds_warning():
    g1 = _water_geom(r_oh=0.95)
    g2 = _water_geom(r_oh=0.97)
    g3 = _water_geom(r_oh=0.96)
    model = LargeAmplitudeModel(
        name="floppy_test",
        coordinate_type="floppy",
        weighting="user",
        grid_points=[
            LargeAmplitudeGridPoint(0.0, g1, weight=1.0),
            LargeAmplitudeGridPoint(1.0, g2, weight=1.0),
            LargeAmplitudeGridPoint(2.0, g3, weight=1.0),
        ],
    )
    out = average_rotational_constants(["O", "H", "H"], model)
    assert any("model-dependent" in w for w in out["warnings"])

