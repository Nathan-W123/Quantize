"""SDF parsing tests (no HTTP)."""

from __future__ import annotations

import numpy as np

from backend.pubchem_geometry import parse_sdf_v2000_first_mol

_MINIMAL_SDF = """931
  test

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$"""


def test_parse_sdf_v2000_water_like():
    elems, coords = parse_sdf_v2000_first_mol(_MINIMAL_SDF)
    assert elems == ["O", "H", "H"]
    assert coords.shape == (3, 3)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(coords[1], [1.0, 0.0, 0.0])
