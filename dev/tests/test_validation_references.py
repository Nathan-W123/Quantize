"""The validation set (dev/validation_references.py, D-154).

These molecules score the engine on data no default or calibration here has
seen, so a transcription slip in them would read as an engine error. The tests
check what can be checked without the sources: the geometry reproduces the
published parameters, each species' rigid-rotor constants at that geometry sit
a fraction of a percent from its measured B0 (a wrong substitution site moves
them by percents), the masses agree with the benchmark module, the reference
bar is read at 1-sigma, and the set never enters the offsets pool.
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

from backend.spectral.centrifugal_distortion import rotational_constants_mhz

import dev.monofluoro_references as REF
import dev.validation_references as VAL

#: The published parameters, typed from the sources' own lines (the entry
#: modules quote them): Angstrom and degrees.
PUBLISHED = {
    # Penocchio et al. 2016, Table 3, rSE_e B2PLYP/VTZ.
    "formaldehyde": {"C=O": 1.2047, "C-H": 1.1008, "H-C-O": 121.61},
    # Muller et al. 2019, Table 7, reSE (CCSD(T)), in pm there.
    "thioformaldehyde": {"C=S": 1.60909, "C-H": 1.08531, "H-C-S": 121.758},
    # Smith et al. 2023, Table IV, re SE this work.
    "ketene": {"C1=C2": 1.31218, "C1=O": 1.16064, "C2-H": 1.07593,
               "C1=C2-H": 119.086, "O=C1=C2": 180.0},
    # Flaud & Lafferty 1993 via Demaison 2007, 143.078 pm.
    "sulfur_dioxide": {"S-O": 1.43078, "O-S-O": 119.330},
    # Heim et al. 2020, Table IV, r_e^SE CCSD(T); C2...C5 is checked apart,
    # since it is a cross-ring distance and not in the bonds.
    "pyrimidine": {"C2-N1": 1.33333, "C4-C5": 1.38675, "C2-H": 1.08165,
                   "C4-H": 1.08224, "C5-H": 1.07929, "N1-C2-H": 116.332,
                   "C4-C5-H": 121.665, "C5-C4-H": 121.261},
    # Esselman et al. 2025, Table IV, r_e^SE (46 isotopologues), the eight
    # independent parameters; r(C3-C4) 1.4222 is dependent (checked to its
    # printed rounding below).
    "thiophene": {"S1-C2": 1.7104, "C2-C3": 1.3658, "C2-H": 1.0770, "C3-H": 1.0786,
                  "S1-C2-C3": 111.600, "S1-C2-H": 120.087, "C2-C3-H": 123.413,
                  "C2-S1-C5": 92.056},
}


def _offsets_pct(mol, sp, subs=None):
    masses = np.asarray(mol.masses, dtype=float).copy()
    for i, m in (sp.subs if subs is None else subs).items():
        masses[int(i)] = m
    calc = np.asarray(rotational_constants_mhz(mol.geometry, masses), dtype=float)
    return 100.0 * (np.asarray(sp.abc_mhz, dtype=float) - calc) / calc


def _bench():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "accbench_val", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    saved = sys.argv
    try:
        sys.argv = ["accbench_val"]
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_the_set_is_the_verified_molecules():
    assert [m.key for m in VAL.VALIDATION_SET] == list(PUBLISHED)
    assert set(VAL.ENTRIES) == set(PUBLISHED)


@pytest.mark.parametrize("mol", VAL.VALIDATION_SET, ids=lambda m: m.key)
def test_the_geometry_is_the_published_structure(mol):
    got = mol.internal_coordinates(mol.geometry)
    for name, value in PUBLISHED[mol.key].items():
        assert got[name] == pytest.approx(value, abs=1e-9), name
    assert mol.structure_type == "r_e"
    assert mol.equilibrium_geometry is mol.geometry


def test_pyrimidine_cross_ring_distance_and_ring_closure():
    x = VAL.PYRIMIDINE.geometry
    assert np.linalg.norm(x[4] - x[1]) == pytest.approx(2.65395, abs=1e-9)
    # Planar, and the six ring angles sum to 720 deg.
    assert np.ptp(x[:, 1]) < 1e-12
    ang = VAL.PYRIMIDINE.internal_coordinates(x)
    ring = (ang["N1-C2-N3"] + ang["C4-C5-C6"] + 2 * ang["C2-N1-C6"] + 2 * ang["N1-C6-C5"])
    assert ring == pytest.approx(720.0, abs=1e-9)
    assert "C2...C5" not in VAL.PYRIMIDINE.bonds


def test_thiophene_dependent_parameters_come_out_as_printed():
    got = VAL.THIOPHENE.internal_coordinates(VAL.THIOPHENE.geometry)
    assert got["C3-C4"] == pytest.approx(1.4222, abs=0.5e-4)
    assert got["C3-C2-H"] == pytest.approx(128.313, abs=0.5e-3)
    assert got["C4-C3-H"] == pytest.approx(124.215, abs=0.5e-3)
    assert got["C2-C3-C4"] == pytest.approx(112.372, abs=0.5e-3)


def test_thiophene_scores_the_2021_constants_of_the_species_its_structure_used():
    from dev.validation_molecules import thiophene
    assert len(thiophene.SPECIES_2021_S) == 26
    left_out = {s["label"] for s in thiophene.SPECIES_2021_S if not s["used_in_re_SE_2021"]}
    assert left_out == {"[2,3,4,5-2H, 34S]", "[3-2H, 33S]"}
    assert len(VAL.THIOPHENE.species) == 24
    assert all(sp.measured_sigma_mhz is not None for sp in VAL.THIOPHENE.species)
    # The 2025 set, kept for reference, prints no uncertainties at all.
    assert all(s["measured_sigma_mhz"] is None for s in thiophene.SPECIES)


@pytest.mark.parametrize("mol", VAL.VALIDATION_SET, ids=lambda m: m.key)
def test_every_species_sits_just_off_its_rigid_constants(mol):
    # B0 differs from the r_e rigid-rotor constant by the vibrational
    # correction, 0.2-1.2% on these molecules. A mistyped digit, or most
    # substitutions on the wrong atom, move it by percents and fail here; a
    # swap along the symmetry axis can stay inside (pyrimidine's C2/C5 and
    # H2/H5: 0.72-1.27%), which the isotope-shift test below catches for heavy
    # atoms. All are planar, so the inertial defect is
    # small and positive: 0.03-0.14 u A^2 here, the lowest pyrimidine's
    # [2,4-2H] at 0.0287, as its source prints.
    assert not mol.species[0].subs, "the parent comes first"
    for sp in mol.species:
        off = _offsets_pct(mol, sp)
        assert np.all(np.abs(off) < 1.5), (sp.label, off)
        inertia = 505379.009 / np.asarray(sp.abc_mhz, dtype=float)
        defect = inertia[2] - inertia[0] - inertia[1]
        assert 0.01 < defect < 0.2, (sp.label, defect)


def _shift_error_pp(mol, sp, subs):
    """Largest |observed - calculated| isotope shift from the parent, as a
    percentage of the parent's constant. The vibrational part of B0 nearly
    cancels in a heavy-atom shift, so a right site leaves almost nothing."""
    parent = mol.species[0]
    dobs = np.asarray(sp.abc_mhz, float) - np.asarray(parent.abc_mhz, float)
    masses = np.asarray(mol.masses, dtype=float)
    sub = masses.copy()
    for i, m in subs.items():
        sub[int(i)] = m
    dcal = (np.asarray(rotational_constants_mhz(mol.geometry, sub), float)
            - np.asarray(rotational_constants_mhz(mol.geometry, masses), float))
    return float(np.max(np.abs(dobs - dcal) / np.asarray(parent.abc_mhz, float)) * 100)


def _heavy_atom_species(mol):
    return [sp for sp in mol.species[1:]
            if all(abs(m - REF.M_D) > 1e-6 for m in sp.subs.values())]


@pytest.mark.parametrize("mol", VAL.VALIDATION_SET, ids=lambda m: m.key)
def test_heavy_atom_isotope_shifts_match_their_sites(mol):
    # Right sites: at most 0.063 percentage points of the parent's constant
    # on this set (H2C18O); the nearest wrong site is 0.298 (pyrimidine
    # [5-13C] on C2). Deuterium is left out: H/D changes the vibrational
    # averaging itself (D2CO 0.79 at the right site), so its shifts cannot
    # tell an axis swap of an H site; the independent re-read checked those.
    for sp in _heavy_atom_species(mol):
        assert _shift_error_pp(mol, sp, sp.subs) < 0.15, sp.label


def test_every_heavy_atom_wrong_site_fails_the_shift_test():
    # The premise of the test above, over every single-site heavy-atom
    # species that has a non-equivalent atom of its element to be put on.
    tried = 0
    for mol in VAL.VALIDATION_SET:
        masses = np.asarray(mol.masses, dtype=float)
        for sp in _heavy_atom_species(mol):
            if len(sp.subs) != 1:
                continue
            (i, mass), = sp.subs.items()
            right = rotational_constants_mhz(mol.geometry, np.where(np.arange(len(masses)) == i, mass, masses))
            for j, el in enumerate(mol.elems):
                if j == i or el != mol.elems[i]:
                    continue
                moved = np.where(np.arange(len(masses)) == j, mass, masses)
                if np.allclose(rotational_constants_mhz(mol.geometry, moved), right, rtol=1e-9):
                    continue                      # a symmetry-equivalent atom
                tried += 1
                assert _shift_error_pp(mol, sp, {j: mass}) > 0.15, (mol.key, sp.label, j)
    assert tried >= 10


def test_a_wrong_substitution_site_would_fail_that_check():
    # The premise of the test above: ketene's [1-13C] put on C2 instead of C1.
    sp = VAL.KETENE.species[1]
    assert sp.label == "[1-13C]" and set(sp.subs) == {0}
    right = _offsets_pct(VAL.KETENE, sp)
    wrong = _offsets_pct(VAL.KETENE, sp, {1: sp.subs[0]})
    assert np.all(np.abs(right) < 1.5)
    assert np.max(np.abs(wrong)) > 1.5


@pytest.mark.parametrize("module", VAL.ENTRY_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_shared_masses_are_the_benchmark_modules(module):
    shared = [n for n in dir(module) if n.startswith("M_") and hasattr(REF, n)]
    assert shared, "every entry uses some of the module's masses"
    for name in shared:
        assert getattr(module, name) == getattr(REF, name), name
    # Every substituted mass is one the entry module defines, so none is typed
    # inline.
    defined = {float(getattr(module, n)) for n in dir(module) if n.startswith("M_")}
    for sp in module.ENTRY["species"]:
        for mass in sp["subs"].values():
            assert float(mass) in defined, (sp["label"], mass)


@pytest.mark.parametrize("module", VAL.ENTRY_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_entry_has_its_provenance_file(module):
    path = Path(module.__file__).with_name(Path(module.__file__).stem + ".numbers.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data, path.name


def test_every_reference_passes_the_bar_at_one_sigma():
    # D-155: the 0.5 mA bar is a 1-sigma figure. Ketene's source quotes
    # 2-sigma (0.69 mA), which would fail it; halved it passes. Pyrimidine's
    # 2-sigma 0.45 passes either way.
    for mol in VAL.VALIDATION_SET:
        assert mol.equilibrium_reference_quality() == "ok", mol.key
    assert VAL.ENTRIES["ketene"]["geometry_re_uncertainty_ma"] == 0.69
    assert VAL.KETENE.geometry_re_uncertainty_ma == pytest.approx(0.345)
    assert VAL.ENTRIES["pyrimidine"]["geometry_re_uncertainty_ma"] == 0.45
    assert VAL.PYRIMIDINE.geometry_re_uncertainty_ma == pytest.approx(0.225)
    assert VAL.ENTRIES["thiophene"]["geometry_re_uncertainty_ma"] == 0.3
    assert VAL.THIOPHENE.geometry_re_uncertainty_ma == pytest.approx(0.15)
    assert 0.69 > REF.RE_REFERENCE_MAX_UNCERTAINTY_MA


def test_sulfur_dioxide_scores_only_its_species_with_stated_sigmas():
    labels = [sp.label for sp in VAL.SULFUR_DIOXIDE.species]
    assert labels == ["32S16O2", "33S16O2", "34S16O2"]
    assert all(sp.measured_sigma_mhz is not None for sp in VAL.SULFUR_DIOXIDE.species)
    raw = VAL.ENTRIES["sulfur_dioxide"]["species"]
    assert len(raw) == 19
    assert {sp["label"] for sp in raw if sp["measured_sigma_mhz"] is not None} == set(labels)


def test_a_missing_scored_species_is_an_error(monkeypatch):
    monkeypatch.setitem(VAL.SCORED_SPECIES, "sulfur_dioxide", ("32S16O2", "no-such-species"))
    with pytest.raises(KeyError, match="no-such-species"):
        VAL.reference_molecule(VAL.ENTRIES["sulfur_dioxide"])


def test_the_loader_copies_each_species_as_transcribed():
    for mol in VAL.VALIDATION_SET:
        raw = {sp["label"]: sp for sp in VAL.ENTRIES[mol.key]["species"]}
        for sp in mol.species:
            src = raw[sp.label]
            assert sp.abc_mhz == tuple(src["abc_mhz"])
            assert sp.decimals == tuple(src["decimals"])
            assert sp.subs == dict(src["subs"])
            sig = src["measured_sigma_mhz"]
            assert sp.measured_sigma_mhz == (None if sig is None else tuple(sig))


def test_the_set_never_enters_the_offsets_pool():
    b = _bench()
    pool = {m.key for m in b.benchmark_pool()} | {m.key for m in REF.CALIBRATION_ONLY}
    held_out = {m.key for m in b.scored_molecules("validation")}
    assert held_out == set(PUBLISHED)
    assert not (held_out & pool)
    # The default scored set is still the nine, unchanged.
    assert [m.key for m in b.scored_molecules("benchmark")] == [m.key for m in b.benchmark_pool()]
    assert len(b.benchmark_pool()) == 9
    assert b.table_order("benchmark") == b._TABLE_ORDER
    assert b.table_order("validation") == list(PUBLISHED)


def test_validation_rows_get_their_own_results_file():
    b = _bench()
    root = Path("/x")
    paths = {b.results_path(root, "hf_6-31g", ext, s)
             for ext in (True, False) for s in ("benchmark", "validation")}
    assert len(paths) == 4
    assert b.results_path(root, "hf_6-31g", True, "benchmark").name == "accuracy_upgrades_hf_6-31g.json"
    assert b.results_path(root, "hf_6-31g", True, "validation").name == \
        "accuracy_upgrades_hf_6-31g_validation.json"


def _run_script_import(argv, monkeypatch):
    # A fresh import of the benchmark AND of the scripts it imports, which
    # read sys.argv at import time too: a cached import would hide a clash
    # between their options and this one.
    import importlib.util as iu
    for name in [n for n in sys.modules if n.startswith("scripts.")]:
        monkeypatch.delitem(sys.modules, name)
    spec = iu.spec_from_file_location(
        "accbench_argv", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", ["accbench_argv", *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def test_scored_argument_is_checked(monkeypatch):
    with pytest.raises(SystemExit, match="scored= must be"):
        _run_script_import(["scored=validaton"], monkeypatch)


def test_scored_validation_parses_with_every_imported_script(monkeypatch):
    # scripts/monofluoro_benchmark.py reads set= at import; scored= must not
    # be one of the options it, or anything else imported, already takes.
    mod = _run_script_import(["method=hf", "basis=6-31g", "scored=validation"], monkeypatch)
    assert mod.SCORED_SET == "validation"
    assert [m.key for m in mod.scored_molecules()] == list(PUBLISHED)


def test_the_table_reads_the_validation_file(tmp_path, monkeypatch):
    b = _bench()
    row = {"theory_rms_ma": 9.0, "me_rms_ma": 6.0, "hybrid_rms_ma": 3.0, "code_vintage": "abc"}
    data = {"formaldehyde": {"base": row}}
    monkeypatch.setattr(b, "SCORED_SET", "validation")
    b.results_path(tmp_path, "hf_6-31g").write_text(json.dumps(data), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.print_summary_table(None, tmp_path)
    out = buf.getvalue()
    assert "formaldehyde" in out and "vinyl fluoride" not in out
    assert "9.00  6.00    3.00" in out



def _run_validation_main(tmp_path, monkeypatch, pool_spy=None):
    """main() with scored=validation, every validation molecule already on file
    in the validation results file, the theory cache faked and fitting
    forbidden (helpers of test_calibration_references). Returns the keys the
    offsets pool was built from."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_calibration_references as TC
    from backend.spectral.bond_offsets import bond_classes_of
    b = TC._bench("scored=validation")
    pool = b.benchmark_pool()
    fake = {m.key: {"coords": np.asarray(m.geometry, float).tolist(),
                    "offsets": {c: [0.001] for c in bond_classes_of(m).values()}}
            for m in pool + list(REF.CALIBRATION_ONLY) + list(VAL.VALIDATION_SET)}
    seen = {}
    real_pool = b.offsets_pool

    def spy(theory, keys):
        seen["keys"] = list(keys)
        out = real_pool(theory, keys)
        return pool_spy(out) if pool_spy else out
    monkeypatch.setattr(b, "_ROOT", tmp_path)
    monkeypatch.setattr(b, "code_vintage", lambda: "test")
    monkeypatch.setattr(b, "theory_geometries", lambda mols: {m.key: fake[m.key] for m in mols})
    monkeypatch.setattr(b, "offsets_pool", spy)
    monkeypatch.setattr(b, "get_backend", lambda name: TC._NoQuantum)
    monkeypatch.setattr(b, "RUN_DIAGNOSTICS", False)
    configs = [c for c in ["base", "offsets", "elec", "lam", "all"] if c in b.CONFIGS]
    (tmp_path / "output").mkdir(parents=True)
    # Rows only in the validation file: reading any other file would find
    # none, and main() would start fitting, which _NoQuantum forbids.
    (tmp_path / "output" / "accuracy_upgrades_hf_6-31g_validation.json").write_text(json.dumps(
        {m.key: {c: {"code_vintage": "test"} for c in configs} for m in VAL.VALIDATION_SET}))
    with contextlib.redirect_stdout(io.StringIO()):
        b.main()
    return seen["keys"], [m.key for m in pool]


def test_main_scores_the_validation_set_from_the_unchanged_pool(tmp_path, monkeypatch):
    keys, nine = _run_validation_main(tmp_path, monkeypatch)
    # The pool is the nine plus the calibration entries (the faked cache puts
    # each at its reference shape, so all three are usable), exactly as a
    # default run builds it, and holds no validation molecule.
    assert keys == nine + [m.key for m in REF.CALIBRATION_ONLY]
    assert not set(keys) & set(PUBLISHED)


def test_main_refuses_a_pool_that_holds_a_validation_molecule(tmp_path, monkeypatch):
    def leak(pool):
        return {**pool, "formaldehyde": {"C-O": [0.001]}}
    with pytest.raises(SystemExit, match="validation molecules in the offsets pool"):
        _run_validation_main(tmp_path, monkeypatch, pool_spy=leak)


def test_bond_pair_spread_sees_what_the_averaged_score_cannot():
    """A ring broken antisymmetrically scores as well as an intact one on the
    symmetry-averaged bonds; bond_pair_spread_ma is the number that shows it
    (T-78: the offsets prior once broke pyrimidine's ring by 217 mA)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_calibration_references as TC
    b = TC._bench("scored=validation")
    mol = VAL.PYRIMIDINE
    ref = np.asarray(mol.geometry, dtype=float)
    assert b.bond_pair_spread_ma(mol, ref) < 1e-6
    # Lengthen one C-H of an equivalent pair, shorten its partner.
    (i, j), (k, m) = mol.bonds["C4-H"]
    broken = ref.copy()
    for (a, h), d in (((i, j), 0.01), ((k, m), -0.01)):
        u = broken[h] - broken[a]
        broken[h] = broken[a] + u * (1.0 + d / np.linalg.norm(u))
    assert b.rms_bond_error(mol, broken)[0] < 1e-6
    assert b.bond_pair_spread_ma(mol, broken) == pytest.approx(20.0, abs=1e-6)
