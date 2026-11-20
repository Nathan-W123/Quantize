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
import json
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
import dev.monofluoro_references as REF  # noqa: E402
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
    # Every bond and every angle individually, not their symmetry average.
    x = mol.geometry
    for name, pairs in mol.bonds.items():
        for i, j in pairs:
            assert np.linalg.norm(x[j] - x[i]) == pytest.approx(PUBLISHED[mol.key][name], abs=1e-9), name
    for name, triples in mol.angles.items():
        for i, j, k in triples:
            u, v = x[i] - x[j], x[k] - x[j]
            ang = np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
            want = PUBLISHED[mol.key].get(name, got[name])
            assert ang == pytest.approx(want, abs=1e-9), name


def test_the_shapes_the_names_do_not_pin():
    x = METHYL_CHLORIDE.geometry          # C3v: all three H-C-H equal
    hch = [np.degrees(np.arccos(np.dot(x[i] - x[0], x[j] - x[0])
                                / np.linalg.norm(x[i] - x[0]) / np.linalg.norm(x[j] - x[0])))
           for i, j in ((2, 3), (2, 4), (3, 4))]
    assert np.ptp(hch) < 1e-9
    x = HYDROGEN_CYANIDE.geometry          # linear, H and N on opposite sides of C
    u, v = x[0] - x[1], x[2] - x[1]
    assert np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v) == pytest.approx(-1.0, abs=1e-12)
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
    assert b.offsets_pool(theory, ["water"]) == {"water": {"H-O": [0.01, 0.01]}}
    both = b.offsets_pool(theory, ["water", "hydrogen_peroxide"])
    assert both == {"water": {"H-O": [0.01, 0.01]},
                    "hydrogen_peroxide": {"O-O": [0.02], "H-O": [0.03, 0.03]}}


def test_a_calibration_entry_on_another_conformer_is_left_out():
    # HF/6-31G's H2O2 minimum is trans-planar (H-O-O-H 179.8 deg): not the
    # r_e conformer, so its offsets are not a bond-length bias. HF/6-31G's
    # CH3Cl is 86 mA long in C-Cl but the same shape, and stays.
    b = _bench()
    theory = {
        "hydrogen_peroxide": {"coords": REF._h2o2_geometry(1.4623, 0.9544, 101.18, 179.8).tolist()},
        "methyl_chloride": {"coords": REF._ch3cl_geometry(1.8633, 1.0736, 107.0).tolist()},
        "hydrogen_cyanide": {"coords": HYDROGEN_CYANIDE.geometry.tolist()},
    }
    with contextlib.redirect_stdout(io.StringIO()) as out:
        kept = b.usable_calibration(list(CALIBRATION_ONLY), theory)
    assert [m.key for m in kept] == ["methyl_chloride", "hydrogen_cyanide"]
    assert "hydrogen_peroxide left out" in out.getvalue()
    assert b.aligned_rmsd(theory["methyl_chloride"]["coords"], METHYL_CHLORIDE.geometry) < 0.05


class _NoQuantum:
    """Any quantum call means main() started fitting a molecule."""
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        raise AssertionError(f"main() fitted a molecule (backend.{name})")


def _run_main(tmp_path, monkeypatch, *argv):
    """main() with every scored molecule already on file in the results file
    this run should use, the theory cache faked, and fitting forbidden.
    Returns the keys main() built its offsets pool from."""
    b = _bench(*argv)
    pool = list(b.MOLECULES) + list(b.MOLECULES_SET2) + [b.WATER_SET[0], b.OZONE, b.ISOCYANIC_ACID]
    fake = {m.key: {"coords": np.asarray(m.geometry, float).tolist(),
                    "offsets": {c: [0.001] for c in bond_classes_of(m).values()}}
            for m in pool + list(CALIBRATION_ONLY)}
    # As HF/6-31G has it: H2O2 trans-planar, so the guard must drop it.
    fake["hydrogen_peroxide"]["coords"] = REF._h2o2_geometry(1.4623, 0.9544, 101.18, 179.8).tolist()
    seen = {}
    real_pool = b.offsets_pool

    def spy(theory, keys):
        seen["keys"] = list(keys)
        return real_pool(theory, keys)
    monkeypatch.setattr(b, "_ROOT", tmp_path)
    monkeypatch.setattr(b, "code_vintage", lambda: "test")
    monkeypatch.setattr(b, "theory_geometries", lambda mols: {m.key: fake[m.key] for m in mols})
    monkeypatch.setattr(b, "offsets_pool", spy)
    monkeypatch.setattr(b, "get_backend", lambda name: _NoQuantum)
    monkeypatch.setattr(b, "RUN_DIAGNOSTICS", False)
    configs = [c for c in ["base", "offsets", "elec", "lam", "all"] if c in b.CONFIGS]
    name = "accuracy_upgrades_hf_6-31g" + ("_calext" if "calibration=extended" in argv else "") + ".json"
    (tmp_path / "output").mkdir(parents=True)
    (tmp_path / "output" / name).write_text(json.dumps(
        {m.key: {c: {"code_vintage": "test"} for c in configs} for m in pool}))
    with contextlib.redirect_stdout(io.StringIO()):
        b.main()
    return seen["keys"], [m.key for m in pool]


def test_main_pools_and_files_by_the_flag(tmp_path, monkeypatch):
    # Default: the scored molecules only, results in the default file (a
    # wrong file would hold no rows and main() would start fitting).
    keys, scored = _run_main(tmp_path / "d", monkeypatch)
    assert keys == scored
    # Extended: the calibration entries of the right conformer join the pool
    # (trans-planar H2O2 does not), are never fitted (they have no rows
    # anywhere), and the rows live in the _calext file.
    keys, scored = _run_main(tmp_path / "e", monkeypatch, "calibration=extended")
    assert keys == scored + ["methyl_chloride", "hydrogen_cyanide"]


def test_extended_calibration_is_opt_in_and_validated():
    assert _bench().EXTENDED_CALIBRATION is False
    assert _bench("calibration=extended").EXTENDED_CALIBRATION is True
    with pytest.raises(SystemExit):
        _bench("calibration=everything")
