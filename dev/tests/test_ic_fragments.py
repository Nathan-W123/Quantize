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
    # The joining pairs are named contacts (they get no rule-book prior).
    assert names == ["contact 1-2", "contact 1-3", "angle 2-1-3"], names


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


# ── Joining coordinates are contacts, not bonds (T-45 verifier) ─────────────

def _dimer():
    """A hydrogen-bonded water dimer: O1-H2-H3 and O4-H5-H6, H5 pointing at O1."""
    a = _x0()
    o4 = np.array([0.0, -2.91, 0.0])
    h5 = o4 + np.array([0.0, 0.957, 0.0])
    th = np.radians(104.5)
    h6 = o4 + 0.957 * np.array([0.0, np.cos(th), np.sin(th)])
    return np.vstack([a, o4, h5, h6])


def test_the_joining_pair_is_named_a_contact():
    x = _dimer()
    cs = InternalCoordinateSet(x, ["O", "H", "H", "O", "H", "H"], False)
    contacts = [ic for ic in cs.active_coords() if ic.contact]
    assert [ic.name for ic in contacts if ic.kind == "bond"] == ["contact 1-5"]
    assert all(("1-5" in ic.name or "5-1" in ic.name or ic.name.endswith("-5") or "-5-" in ic.name
                or ic.name.startswith("contact")) for ic in contacts)
    assert not any(ic.contact for ic in cs.active_coords() if ic.name in ("bond 1-2", "bond 4-5"))


def test_a_contact_gets_no_rule_book_prior():
    from backend.internal.internal_fit import build_internal_priors
    from backend.internal.internal_prior_library import resolve_default_prior
    x = _dimer()
    elems = ["O", "H", "H", "O", "H", "H"]
    cs = InternalCoordinateSet(x, elems, False)
    k = cs.active_names().index("contact 1-5")
    ic = cs.active_coords()[k]
    # Premise: the covalent rules have an O-H rule that would apply to it.
    assert resolve_default_prior(ic, elems, 1.95) is not None
    out = build_internal_priors(cs, x, elems=elems, return_metadata=True)
    meta = out[-1]
    assert meta[k]["source"] == "geometry_initial"
    assert abs(meta[k]["target_value"] - np.linalg.norm(x[0] - x[4])) < 1e-12
    for i, c in enumerate(cs.active_coords()):
        if c.contact:
            assert meta[i]["source"] == "geometry_initial", c.name


def test_a_spectral_only_dimer_keeps_its_hydrogen_bond():
    import copy
    import backend.spectral.spectral as SP
    truth = _dimer()
    elems = ["O", "H", "H", "O", "H", "H"]
    M_O, M_H, M_D = 15.9949146196, 1.00782503207, 2.0141017778
    isos = []
    for name, m in (("p", [M_O, M_H, M_H, M_O, M_H, M_H]), ("a", [M_O, M_D, M_D, M_O, M_H, M_H]),
                    ("b", [M_O, M_H, M_H, M_O, M_D, M_D]), ("ab", [M_O, M_D, M_D, M_O, M_D, M_D]),
                    ("h5", [M_O, M_H, M_H, M_O, M_D, M_H])):
        abc = SP._rotational_constants(truth, np.array(m))
        isos.append(dict(name=name, masses=m, obs_constants=[float(v) for v in abc],
                         sigma_constants=[0.05] * 3, component_indices=[0, 1, 2]))
    start = truth + np.random.default_rng(4).normal(0, 0.02, truth.shape)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=elems, coords=start, isotopologues=copy.deepcopy(isos),
                                 quantum_backend="none", spectral_only=True, use_autoconfig=False,
                                 max_iter=200, coordinate_mode="internal")
        x = np.asarray(opt.run(), float)
    # Measured: 1.9491 A against 1.9530 (with the covalent prior: 1.459).
    assert abs(np.linalg.norm(x[0] - x[4]) - np.linalg.norm(truth[0] - truth[4])) < 0.02
