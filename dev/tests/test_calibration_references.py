"""Calibration-only references: measured for the bond offsets, never scored.

Leave-one-out cannot calibrate a bond class seen in one molecule, so
chlorofluoromethane's C-Cl, isocyanic acid's C-N, water's O-H and ozone's O-O
went uncorrected. dev.monofluoro_references.CALIBRATION_ONLY adds equilibrium
structures that give those classes a second member. These pin the published
values each geometry was built from, that each class actually gains a member,
and that the benchmark's offsets pool takes them only when asked
(calibration=extended).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.spectral.bond_offsets import (  # noqa: E402
    bond_classes_of,
    leave_one_out_offsets,
)
from dev.monofluoro_references import (  # noqa: E402
    CALIBRATION_ONLY,
    CHLOROFLUOROMETHANE,
    HYDROGEN_CYANIDE,
    HYDROGEN_PEROXIDE,
    ISOCYANIC_ACID,
    METHYL_CHLORIDE,
    OZONE,
    WATER,
)


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


# The published values, typed from the sources independently of the builders.
PUBLISHED = {
    "methyl_chloride": {"C-Cl": 1.777716, "C-H": 1.083484, "Cl-C-H": 108.3723},
    "hydrogen_cyanide": {"C-H": 1.06501, "C-N": 1.15324},
    "hydrogen_peroxide": {"O-O": 1.4523, "O-H": 0.9618, "O-O-H": 99.78},
}


@pytest.mark.parametrize("mol", CALIBRATION_ONLY, ids=lambda m: m.key)
def test_each_geometry_reproduces_its_published_structure(mol):
    got = mol.internal_coordinates(mol.geometry)
    for name, value in PUBLISHED[mol.key].items():
        assert got[name] == pytest.approx(value, abs=1e-9), name
    # Equivalent bonds really are equivalent, not just averaged to the value.
    x = mol.geometry
    for name, pairs in mol.bonds.items():
        lengths = [np.linalg.norm(x[j] - x[i]) for i, j in pairs]
        assert np.ptp(lengths) < 1e-12, name


def test_the_shapes_the_names_do_not_pin():
    x = METHYL_CHLORIDE.geometry          # C3v: all three H-C-H equal
    hch = [np.degrees(np.arccos(np.dot(x[i] - x[0], x[j] - x[0])
                                / np.linalg.norm(x[i] - x[0]) / np.linalg.norm(x[j] - x[0])))
           for i, j in ((2, 3), (2, 4), (3, 4))]
    assert np.ptp(hch) < 1e-9
    x = HYDROGEN_CYANIDE.geometry          # linear
    assert np.linalg.norm(np.cross(x[0] - x[1], x[2] - x[1])) < 1e-12
    x = HYDROGEN_PEROXIDE.geometry         # H-O-O-H dihedral 113.5 deg
    assert abs(_dihedral(x[2], x[0], x[1], x[3])) == pytest.approx(113.5, abs=1e-9)


@pytest.mark.parametrize("mol", CALIBRATION_ONLY, ids=lambda m: m.key)
def test_each_is_an_equilibrium_reference_inside_the_bar_and_never_scored(mol):
    assert mol.structure_type == "r_e"
    assert mol.equilibrium_reference_quality() == "ok"
    assert mol.species == []          # nothing to fit, so nothing to score


def test_every_singleton_class_but_n_h_gains_a_member():
    # Synthetic offsets: only which classes exist matters here.
    scored = [CHLOROFLUOROMETHANE, ISOCYANIC_ACID, WATER, OZONE]
    per = {m.key: {c: [0.01] for c in bond_classes_of(m).values()}
           for m in scored + list(CALIBRATION_ONLY)}
    covered = {
        "chlorofluoromethane": "C-Cl",
        "isocyanic_acid": "C-N",
        "water": "H-O",
        "ozone": "O-O",
    }
    for key, cls in covered.items():
        without = {k: v for k, v in per.items() if k in {m.key for m in scored}}
        assert cls not in leave_one_out_offsets(without, key), key
        assert cls in leave_one_out_offsets(per, key), key
    assert "H-N" not in leave_one_out_offsets(per, "isocyanic_acid")


def _bench(*argv):
    import importlib.util as iu
    spec = iu.spec_from_file_location("accbench_cal", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    saved = sys.argv
    try:
        sys.argv = ["accbench_cal", *argv]
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_the_pool_is_chosen_by_key_not_by_what_the_cache_holds():
    b = _bench()
    theory = {"water": {"offsets": {"H-O": [0.01, 0.01]}},
              "hydrogen_peroxide": {"offsets": {"O-O": [0.02], "H-O": [0.03, 0.03]}}}
    assert set(b.offsets_pool(theory, ["water"])) == {"water"}
    assert set(b.offsets_pool(theory, ["water", "hydrogen_peroxide"])) == {"water", "hydrogen_peroxide"}


def test_extended_calibration_is_opt_in_and_validated():
    assert _bench().EXTENDED_CALIBRATION is False
    assert _bench("calibration=extended").EXTENDED_CALIBRATION is True
    with pytest.raises(SystemExit):
        _bench("calibration=everything")
