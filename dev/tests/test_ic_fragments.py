"""Internal coordinates join a disconnected bond graph.

Bonds come from the covalent-radii rule (distance < 1.2 x the sum of the
radii: O-H 1.164 A). Water started at r = 1.25 A had no bond and so no angle:
its internal coordinate set was empty, every step zero, and the fit stopped
where it began, 820 posterior widths from its optimum, silently (T-44). The
internal set now joins fragments through their closest pair of atoms
(decisions D-108); a connected molecule's set is unchanged.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.internal.internal_fit import InternalCoordinateSet  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.quantum import _detect_bonds, _join_fragments  # noqa: E402
from test_cd_rows_sign import _isos, _x0  # noqa: E402


def test_a_stretched_molecule_keeps_its_bonds_and_angle():
    x = _x0(r=1.25, deg=80.0)
    assert _detect_bonds(x, ["O", "H", "H"]) == [], "premise: the radii rule finds no bond"
    names = InternalCoordinateSet(x, ["O", "H", "H"], False).active_names()
    assert names == ["bond 1-2", "bond 1-3", "angle 2-1-3"], names


def test_a_connected_molecule_is_unchanged():
    x = _x0()
    bonds = _detect_bonds(x, ["O", "H", "H"])
    assert bonds and _join_fragments(x, bonds) == sorted(bonds)


def test_two_molecules_are_joined_by_their_closest_atoms():
    a = _x0()
    b = _x0() + np.array([0.0, 0.0, 3.0])          # a second water 3 A away
    x = np.vstack([a, b])
    bonds = _join_fragments(x, _detect_bonds(x, ["O", "H", "H"] * 2))
    joining = [p for p in bonds if (p[0] < 3) != (p[1] < 3)]
    assert len(joining) == 1
    i, j = joining[0]
    d = np.linalg.norm(x[:, None, :3] - x[None, :, :3], axis=2)
    cross = d[:3, 3:]
    assert np.isclose(d[i, j], cross.min())


@pytest.mark.parametrize("start", [(1.25, 80.0), (1.4, 120.0)])
def test_a_far_start_fit_reaches_its_optimum(start):
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(r=start[0], deg=start[1]),
                                 isotopologues=_isos(None), quantum_backend="analytic_water",
                                 use_autoconfig=False, max_iter=80, coordinate_mode="internal")
        opt.run()
    # Measured: 6.5e-6 and 1.8e-7 widths (820 and 1090 before).
    assert opt.distance_to_mode() < 1e-3
    assert sum(r["accepted"] for r in opt.history) > 2
