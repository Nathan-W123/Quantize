"""The water reference, checked against the sources it cites.

Every statement that the engine's rovibrational correction is wrong by N sigma
is measured against a "true" correction built as B_e(r_e) - B_0(observed). Both
inputs are published numbers, so both can be checked -- and if either is wrong,
the verdict on the correction is really a verdict on the reference.

These pin:

  * the stored coordinates against the r_e parameters the module documents,
  * the stored B_0 against the values NIST CCCBDB quotes,
  * the resulting true correction against the spread over every published
    equilibrium structure, so a benchmark can never rest on one choice of
    reference, and
  * the planarity of the corrected constants, which is a check on the
    correction needing no reference structure at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.centrifugal_distortion import (
    _INERTIA_TO_MHZ,
    rotational_constants_mhz,
)
from dev.monofluoro_references import (
    M_D,
    M_H,
    M_O16,
    WATER,
    WATER_R_E,
    WATER_THETA_E_DEG,
    _bent_xy,
)

_CM = 29979.2458  # MHz per cm-1

#: NIST CCCBDB's experimental rotational constants for H2-16O, in cm-1,
#: attributed there to 1966Herzberg.
_CCCBDB_H2O_CM = (27.877, 14.512, 9.285)

#: Published equilibrium structures, (r_e in Angstrom, angle in degrees).
#: Csaszar 2005 is the spectroscopic (nonadiabatic) r_e; Hoy & Bunker 1979 is
#: what CCCBDB lists; the last entry is the older effective structure still
#: quoted in textbooks, carried here to show a wrong reference cannot rescue
#: the correction.
_STRUCTURES = {
    "csaszar_2005": (0.95777, 104.48),
    "hoy_bunker_1979": (0.9578, 104.4776),
    "textbook_r0_era": (0.9572, 104.52),
}

_MASSES = {"H2-16O": np.array([M_O16, M_H, M_H]),
           "D2-16O": np.array([M_O16, M_D, M_D])}


def _observed(label: str) -> np.ndarray:
    for sp in WATER.species:
        if sp.label == label:
            return np.asarray(sp.abc_mhz, dtype=float)
    raise KeyError(label)


def _true_correction(label: str, r_ang: float, theta_deg: float) -> np.ndarray:
    """B_e - B_0 in MHz: what a correct rovibrational correction must supply."""
    return (rotational_constants_mhz(_bent_xy(r_ang, theta_deg), _MASSES[label])
            - _observed(label))


def _defect(abc_mhz: np.ndarray) -> float:
    i = _INERTIA_TO_MHZ / np.asarray(abc_mhz, dtype=float)
    return float(i[2] - i[0] - i[1])


# ── The reference against its sources ────────────────────────────────────────

def test_stored_geometry_matches_the_documented_r_e():
    """Coordinates and citation must not drift apart; they did once, by 0.2 mA."""
    g = np.asarray(WATER.geometry, dtype=float)
    r = np.linalg.norm(g[1] - g[0])
    v1, v2 = g[1] - g[0], g[2] - g[0]
    theta = np.degrees(np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    assert r == pytest.approx(WATER_R_E, abs=1e-6)
    assert theta == pytest.approx(WATER_THETA_E_DEG, abs=1e-6)
    assert np.linalg.norm(g[2] - g[0]) == pytest.approx(r, abs=1e-9)


def test_documented_r_e_is_a_published_equilibrium_structure():
    r_pub, th_pub = _STRUCTURES["csaszar_2005"]
    assert WATER_R_E == pytest.approx(r_pub, abs=5e-5)
    assert WATER_THETA_E_DEG == pytest.approx(th_pub, abs=5e-3)


def test_stored_constants_match_cccbdb():
    obs = _observed("H2-16O")
    want = np.array(_CCCBDB_H2O_CM) * _CM
    assert np.all(np.abs(obs - want) / want < 1e-4)


def test_observed_constants_are_ground_state_not_equilibrium():
    """B_0 carries water's known ~0.05 amu A^2 zero-point inertial defect.

    A reference that had quietly stored equilibrium constants would make the
    correction look unnecessary rather than wrong.
    """
    assert _defect(_observed("H2-16O")) == pytest.approx(0.049, abs=0.005)
    assert _defect(_observed("D2-16O")) == pytest.approx(0.068, abs=0.005)


# ── The true correction, and how much it depends on the reference ────────────

def test_true_correction_is_insensitive_to_which_published_r_e_is_used():
    """Across every modern determination the truth moves by <300 MHz.

    The correction error being diagnosed is ~10 000 MHz on B, so the diagnosis
    cannot be an artefact of the structure chosen.
    """
    for label in _MASSES:
        stack = np.array([_true_correction(label, r, th)
                          for key, (r, th) in _STRUCTURES.items()
                          if key != "textbook_r0_era"])
        assert np.all(stack.max(axis=0) - stack.min(axis=0) < 300.0)


def test_true_correction_has_the_expected_sign_pattern():
    """A falls, B and C rise, on both isotopologues.

    Water's bending mode drives a large negative alpha on A, so B_0 > B_e there
    while the two heavier axes go the other way. A correction chain that got
    this backwards would still fit, against a self-consistent wrong reference.
    """
    for label in _MASSES:
        d = _true_correction(label, *_STRUCTURES["csaszar_2005"])
        assert d[0] < -4000.0
        assert d[1] > 500.0
        assert d[2] > 2000.0


def test_true_correction_makes_the_constants_planar():
    """By construction, but it is the property the engine's correction must share."""
    for label in _MASSES:
        corrected = _observed(label) + _true_correction(
            label, *_STRUCTURES["csaszar_2005"])
        assert _defect(corrected) == pytest.approx(0.0, abs=1e-9)


def test_a_wrong_reference_would_not_rescue_the_b_correction():
    """Even the discarded textbook structure leaves the B truth near +2500 MHz.

    The engine computes ~+12 000 MHz for H2-16O, so no published structure puts
    the two within 3x of each other.
    """
    worst = max(_true_correction(lab, *_STRUCTURES["textbook_r0_era"])[1]
                for lab in _MASSES)
    assert worst < 3000.0
