"""Cheap ORCA-relaxed seeds (skipped when ORCA is missing).

The alias parity cell below stays graph-only — it does **not** call ORCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import assert_finite_coords, initial_guess


@pytest.mark.orca
def test_initial_guess_water_after_cheap_opt(water_cheap_relaxed_guess):
    coords, elems = water_cheap_relaxed_guess
    assert elems == ["O", "H", "H"]
    assert_finite_coords(coords, elems)


@pytest.mark.orca
def test_initial_guess_ocs_after_cheap_opt(ocs_cheap_relaxed_guess):
    coords, elems = ocs_cheap_relaxed_guess
    assert elems == ["O", "C", "S"]
    assert_finite_coords(coords, elems)
    assert coords.shape == (3, 3)


def test_alias_matches_direct_import():
    """Coarse molecular_input only (bond graph)—no QM step."""
    from backend.geometryguess import guess_geometry_molecular_input

    a = initial_guess(None, elems=["N", "H"], bonds=[(0, 1)], center=False)
    b = guess_geometry_molecular_input(None, elems=["N", "H"], bonds=[(0, 1)], center=False)
    assert np.allclose(a[0], b[0])
    assert a[1] == b[1]


@pytest.mark.orca
def test_rotational_constants_finite_water(water_cheap_relaxed_guess):
    from backend.spectral import _rotational_constants

    coords, elems = water_cheap_relaxed_guess
    masses = np.array([15.99491, 1.007825, 1.007825])
    abc = _rotational_constants(coords, masses)
    assert np.all(np.isfinite(abc))
    assert np.all(abc > 0)
