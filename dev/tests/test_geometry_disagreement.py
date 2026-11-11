"""Two levels disagreeing on the SHAPE, detected without a reference.

HF/6-31G puts isocyanic acid's H-N-C angle 30.34 deg from the published value
while B3LYP/6-31G(d) is 1.14 deg away. Nothing in the engine noticed. The
check has to work without a published structure, because the engine exists
for molecules that have none and a reference is exactly what is missing when
this happens -- but the two levels disagree by 29.19 deg on that coordinate,
which is visible from the two geometries alone.

Measured over the 10-molecule reference set, HF/6-31G against B3LYP/6-31G(d):

  angles, n=61   median 0.437   p90 1.364   p99 13.70   max 29.19 deg
      29.194  isocyanic acid    H-N-C
       3.377  isocyanic acid    N-C-O
       1.984  chlorofluoromethane H-C-H
       1.831  vinyl fluoride    H4-C1=C2
       1.786  ozone             O-O-O

  bonds, n=52    median 0.01439  p90 0.02982  p99 0.03891  max 0.04285 Ang

The angle threshold sits in a genuinely empty gap: nothing at all lies
between 3.4 and 29.2 deg. The bond threshold does NOT, because no bond
failure was observed -- its distribution peaks at 3x its median, which is
just two levels differing systematically -- so that one is a sanity bound
rather than a calibrated one, and these tests say so.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "ffd", str(_ROOT / "scripts" / "force_field_diagnostics.py"))
FFD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FFD)

#: Measured cross-level angle disagreements, worst first, degrees.
MEASURED_ANGLES = [29.194, 3.377, 1.984, 1.831, 1.786]
MEASURED_ANGLE_MEDIAN = 0.437
#: Measured cross-level bond disagreements, Angstrom.
MEASURED_BOND_MAX = 0.04285
MEASURED_BOND_MEDIAN = 0.01439


def _bent(angle_deg, r=1.1):
    t = np.radians(angle_deg)
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.cos(t), r * np.sin(t), 0.0],
                     [r, 0.0, 0.0],
                     [2.3, 0.0, 0.0]])


ELEMS = ["N", "H", "C", "O"]
BONDS = {"N-H": [(0, 1)], "N-C": [(0, 2)], "C-O": [(2, 3)]}
ANGLES = {"H-N-C": [(1, 0, 2)]}


def test_the_angle_threshold_sits_in_an_empty_gap():
    """Between the real failure and the worst non-failure there is nothing."""
    worst_real, next_worst = MEASURED_ANGLES[0], MEASURED_ANGLES[1]
    thr = FFD.GEOMETRY_ANGLE_DISAGREEMENT_WARN_DEG
    assert next_worst < thr < worst_real, (
        f"{thr} deg no longer separates the failure ({worst_real}) from the "
        f"worst ordinary disagreement ({next_worst})")
    # And the gap is wide, not marginal.
    assert worst_real / next_worst > 5.0


def test_it_flags_the_isocyanic_acid_scale_of_disagreement():
    a, b = _bent(124.0), _bent(124.0 - 29.19)
    r = FFD.geometry_disagreement(a, b, ELEMS, bonds=BONDS, angles=ANGLES,
                                  label_a="HF/6-31G", label_b="B3LYP/6-31G(d)")
    assert r["flagged"] is True
    assert r["max_angle_dev"] == pytest.approx(29.19, abs=0.05)
    assert r["worst_angle"] == "H-N-C(1,0,2)"
    assert "HF/6-31G" in r["reason"] and "different minimum" in r["reason"]


def test_it_stays_quiet_at_the_worst_ordinary_disagreement():
    """3.38 deg is the worst real non-failure; it must not fire."""
    a, b = _bent(124.0), _bent(124.0 - MEASURED_ANGLES[1])
    r = FFD.geometry_disagreement(a, b, ELEMS, bonds=BONDS, angles=ANGLES)
    assert r["flagged"] is False
    assert r["flagged_angles"] == []
    assert r["max_angle_dev"] == pytest.approx(MEASURED_ANGLES[1], abs=0.05)


def test_identical_geometries_disagree_about_nothing():
    a = _bent(124.0)
    r = FFD.geometry_disagreement(a, a.copy(), ELEMS, bonds=BONDS, angles=ANGLES)
    assert r["flagged"] is False
    assert r["max_angle_dev"] == pytest.approx(0.0, abs=1e-9)
    assert r["max_bond_dev"] == pytest.approx(0.0, abs=1e-12)


def test_the_bond_threshold_is_above_everything_ever_observed():
    """It is a sanity bound, not calibrated against a failure."""
    thr = FFD.GEOMETRY_BOND_DISAGREEMENT_WARN_ANG
    assert thr > 3.0 * MEASURED_BOND_MAX
    # The observed bond distribution is smooth, which is why there is nothing
    # to calibrate against: its max is only about 3x its median.
    assert MEASURED_BOND_MAX / MEASURED_BOND_MEDIAN < 4.0


def test_a_gross_bond_difference_does_fire():
    a = _bent(124.0)
    b = a.copy()
    b[3] = [2.3 + 0.4, 0.0, 0.0]          # C-O out by 0.4 Ang
    r = FFD.geometry_disagreement(a, b, ELEMS, bonds=BONDS, angles=ANGLES)
    assert r["flagged"] is True
    assert any(name.startswith("C-O") for name, _d in r["flagged_bonds"])


def test_it_degrades_instead_of_raising():
    """A diagnostic must not take down the run it qualifies."""
    a = _bent(124.0)
    for bad_a, bad_b in ((a, a[:3]), (a, "nonsense"), (a[:2], a[:2])):
        r = FFD.geometry_disagreement(bad_a, bad_b, ELEMS,
                                      bonds=BONDS, angles=ANGLES)
        assert r["flagged"] is False
        assert r["reason"], "a refusal must say why"


def test_no_coordinates_given_is_reported_not_passed():
    a = _bent(124.0)
    r = FFD.geometry_disagreement(a, a.copy(), ELEMS)
    assert r["flagged"] is False
    assert "no bonds or angles" in r["reason"]


def test_it_needs_no_reference_structure():
    """The whole point: the signature takes two geometries and nothing else."""
    import inspect
    params = inspect.signature(FFD.geometry_disagreement).parameters
    assert "coords_a" in params and "coords_b" in params
    assert not any("ref" in p for p in params), (
        "geometry_disagreement must not require a reference structure")
