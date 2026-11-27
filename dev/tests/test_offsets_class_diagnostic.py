"""What each bond-class offset was pooled from (D-171, backlog N41).

A bond class is an element pair, so pyrimidine's aromatic C-N takes its offset
from isocyanic acid's N=C and hydrogen cyanide's C#N. The diagnostic records,
per class applied to a molecule, how many bonds the offset averaged and how
their theory lengths sit against the molecule's own, and flags a bond when no
member lies within MEMBER_LENGTH_WARN_ANG of it. It must change no number:
offsets, priors and widths are the same with or without it.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.spectral.bond_offsets import (  # noqa: E402
    MEMBER_LENGTH_WARN_ANG,
    bond_classes_of,
    class_diagnostic_warnings,
    leave_one_out_class_diagnostic,
    leave_one_out_offsets,
    measure_offsets,
    measure_theory_lengths,
    offset_corrected_geometry,
    prior_sigma_for_molecule,
)
import dev.monofluoro_references as REF  # noqa: E402
import dev.validation_references as VAL  # noqa: E402
from dev.validation_references import PYRIMIDINE  # noqa: E402

# Theory lengths (A) at RHF/6-31G, from output/theory_geometries_hf_6-31g.json:
# isocyanic acid's N-C and hydrogen cyanide's C-N are the C-N pool; vinyl
# fluoride's C=C and fluoroethane's C-C give a C-C class with one member near
# pyrimidine's C4-C5 and one far from it.
HF_POOL_LENGTHS = {
    "isocyanic_acid": {"H-N": [0.9791], "C-N": [1.1761], "C-O": [1.1795]},
    "hydrogen_cyanide": {"C-H": [1.0527], "C-N": [1.1441]},
    "vinyl_fluoride": {"C-C": [1.3102], "C-F": [1.3777],
                       "C-H": [1.0673, 1.0708, 1.0704]},
    "fluoroethane": {"C-C": [1.5092], "C-F": [1.4249],
                     "C-H": [1.0786, 1.0839, 1.0820]},
}
PYR_X = np.asarray(PYRIMIDINE.geometry, dtype=float)   # C2-N1 1.33333, N1-C6 1.334626


def _fake_mol(lengths: dict[str, float], elems=("C", "C")):
    """A molecule of separate diatomics, one per named bond, at given lengths."""
    coords, bonds, el = [], {}, []
    for k, (name, length) in enumerate(lengths.items()):
        coords += [[10.0 * k, 0.0, 0.0], [10.0 * k + length, 0.0, 0.0]]
        bonds[name] = [(2 * k, 2 * k + 1)]
        el += list(elems)

    def internal_coordinates(x, _b=bonds):
        x = np.asarray(x, dtype=float)
        return {n: float(np.mean([np.linalg.norm(x[j] - x[i]) for i, j in p]))
                for n, p in _b.items()}
    mol = types.SimpleNamespace(key="fake", elems=el, bonds=bonds, angles={},
                                internal_coordinates=internal_coordinates)
    return mol, np.asarray(coords, dtype=float)


# ── the flag ───────────────────────────────────────────────────────────────

def test_pyrimidine_c_n_is_flagged_when_its_offset_comes_from_multiple_bonds():
    diag = leave_one_out_class_diagnostic(PYRIMIDINE, PYR_X, HF_POOL_LENGTHS,
                                          PYRIMIDINE.key)
    assert diag["tol_ang"] == MEMBER_LENGTH_WARN_ANG == 0.08
    assert diag["flagged"] == ["C2-N1", "N1-C6"]
    cn = diag["classes"]["C-N"]
    assert cn["n_pooled"] == 2
    assert cn["pooled_min_ang"] == 1.1441 and cn["pooled_max_ang"] == 1.1761
    b = cn["bonds"]["C2-N1"]
    assert b["theory_ang"] == pytest.approx(1.33333, abs=1e-6)
    assert b["nearest_ang"] == pytest.approx(1.33333 - 1.1761, abs=1e-6)
    assert b["furthest_ang"] == pytest.approx(1.33333 - 1.1441, abs=1e-6)
    assert b["n_within"] == 0 and b["flagged"] is True
    lines = class_diagnostic_warnings(PYRIMIDINE.key, diag, {"C-N": -0.02348})
    assert lines == [
        "[offsets] pyrimidine C2-N1 (C-N): theory length 1.3333 A, offset "
        "-23.48 mA pooled from 2 bond(s) at 1.1441-1.1761 A, none within "
        "0.08 A (nearest 0.1572 A away)",
        "[offsets] pyrimidine N1-C6 (C-N): theory length 1.3346 A, offset "
        "-23.48 mA pooled from 2 bond(s) at 1.1441-1.1761 A, none within "
        "0.08 A (nearest 0.1585 A away)",
    ]


def test_a_class_whose_members_match_in_length_is_not_flagged():
    diag = leave_one_out_class_diagnostic(PYRIMIDINE, PYR_X, HF_POOL_LENGTHS,
                                          PYRIMIDINE.key)
    ch = diag["classes"]["C-H"]
    assert ch["n_pooled"] == 7
    for name in ("C2-H", "C4-H", "C5-H"):
        bond = ch["bonds"][name]
        assert bond["furthest_ang"] < MEMBER_LENGTH_WARN_ANG
        assert bond["n_within"] == 7 and bond["flagged"] is False
    # C4-C5 (1.387 A) has one member within the tolerance (C=C, 1.310) and one
    # outside it (C-C, 1.509): one like bond is enough, so it is not flagged.
    cc = diag["classes"]["C-C"]["bonds"]["C4-C5"]
    assert cc["n_within"] == 1 and cc["flagged"] is False
    assert cc["furthest_ang"] > MEMBER_LENGTH_WARN_ANG
    assert not any("(C-H)" in s or "(C-C)" in s
                   for s in class_diagnostic_warnings(PYRIMIDINE.key, diag,
                                                      {"C-H": 0.0, "C-C": 0.0}))


def test_every_member_is_compared_not_the_members_mean():
    # Members at 1.30 and 1.50 have their mean on a 1.40 bond, yet neither is
    # like it: the flag fires. One near member among far ones does not.
    mol, x = _fake_mol({"mid": 1.40, "near": 1.32})
    pool = {"a": {"C-C": [1.30]}, "b": {"C-C": [1.50]}}
    diag = leave_one_out_class_diagnostic(mol, x, pool, "fake")
    bonds = diag["classes"]["C-C"]["bonds"]
    assert bonds["mid"]["flagged"] is True
    assert bonds["mid"]["nearest_ang"] == pytest.approx(0.10)
    assert bonds["near"]["flagged"] is False and bonds["near"]["n_within"] == 1
    assert diag["flagged"] == ["mid"]


def test_the_tolerance_is_strict_and_comes_from_the_argument():
    mol, x = _fake_mol({"b": 1.30})
    pool = {"a": {"C-C": [1.20]}}               # 0.10 A away
    assert leave_one_out_class_diagnostic(mol, x, pool, "fake", tol_ang=0.11)["flagged"] == []
    assert leave_one_out_class_diagnostic(mol, x, pool, "fake", tol_ang=0.09)["flagged"] == ["b"]
    assert leave_one_out_class_diagnostic(mol, x, pool, "fake")["tol_ang"] == MEMBER_LENGTH_WARN_ANG


def test_the_scored_molecule_is_never_its_own_member():
    own = measure_theory_lengths(PYRIMIDINE, PYR_X)
    pool = {**HF_POOL_LENGTHS, PYRIMIDINE.key: own}
    with_own = leave_one_out_class_diagnostic(PYRIMIDINE, PYR_X, pool, PYRIMIDINE.key)
    without = leave_one_out_class_diagnostic(PYRIMIDINE, PYR_X, HF_POOL_LENGTHS,
                                             PYRIMIDINE.key)
    assert with_own == without


def test_only_classes_that_got_an_offset_appear():
    # Isocyanic acid's N-H has no member elsewhere, so it gets no offset and no
    # entry; its C-N has hydrogen cyanide's.
    x = np.asarray(REF.ISOCYANIC_ACID.geometry, dtype=float)
    diag = leave_one_out_class_diagnostic(REF.ISOCYANIC_ACID, x, HF_POOL_LENGTHS,
                                          REF.ISOCYANIC_ACID.key)
    per_off = {k: {c: [0.0] * len(v) for c, v in by.items()}
               for k, by in HF_POOL_LENGTHS.items()}
    applied = set(leave_one_out_offsets(per_off, REF.ISOCYANIC_ACID.key))
    assert set(diag["classes"]) == applied & set(bond_classes_of(REF.ISOCYANIC_ACID).values())
    assert "H-N" not in diag["classes"]
    assert diag["classes"]["C-N"]["n_pooled"] == 1


# ── alignment with the offsets table ───────────────────────────────────────

_ALL = (list(REF.MOLECULES) + list(REF.MOLECULES_SET2)
        + [REF.WATER_SET[0], REF.OZONE, REF.ISOCYANIC_ACID]
        + list(REF.CALIBRATION_ONLY) + list(VAL.VALIDATION_SET))


def _theory_like(mol):
    """A stand-in theory geometry: the reference, stretched 1% and nudged so
    every class carries a nonzero, bond-dependent error."""
    x = np.asarray(mol.geometry, dtype=float)
    rng = np.random.default_rng(zlib.crc32(mol.key.encode()))
    return 1.01 * x + 0.003 * rng.standard_normal(x.shape)


@pytest.mark.parametrize("mol", _ALL, ids=lambda m: m.key)
def test_lengths_line_up_with_offsets_entry_for_entry(mol):
    x = _theory_like(mol)
    lengths, errs = measure_theory_lengths(mol, x), measure_offsets(mol, x)
    assert {c: len(v) for c, v in lengths.items()} == {c: len(v) for c, v in errs.items()}
    ref = mol.internal_coordinates(np.asarray(mol.geometry, dtype=float))
    names = {c: [n for n, k in bond_classes_of(mol).items() if k == c] for c in lengths}
    for c, ns in names.items():
        for n, L, e in zip(ns, lengths[c], errs[c]):
            assert L - ref[n] == pytest.approx(e, abs=1e-12)


def test_n_pooled_is_the_count_the_offset_averaged():
    pool = _ALL[:12]
    xs = {m.key: _theory_like(m) for m in pool}
    per_off = {m.key: measure_offsets(m, xs[m.key]) for m in pool}
    per_len = {m.key: measure_theory_lengths(m, xs[m.key]) for m in pool}
    for mol in _ALL:
        x = xs.get(mol.key, _theory_like(mol))
        diag = leave_one_out_class_diagnostic(mol, x, per_len, mol.key)
        for cls, entry in diag["classes"].items():
            pooled = [e for k, by in per_off.items() if k != mol.key for e in by.get(cls, [])]
            assert entry["n_pooled"] == len(pooled), (mol.key, cls)


# ── no number changes ──────────────────────────────────────────────────────

def test_offsets_priors_and_widths_are_unchanged_by_the_diagnostic():
    pool = _ALL[:12]
    xs = {m.key: _theory_like(m) for m in _ALL}
    per_off = {m.key: measure_offsets(m, xs[m.key]) for m in pool}
    per_len = {m.key: measure_theory_lengths(m, xs[m.key]) for m in pool}
    snap_off, snap_len = copy.deepcopy(per_off), copy.deepcopy(per_len)
    for mol in (REF.MOLECULES[0], REF.ISOCYANIC_ACID, PYRIMIDINE):
        x = xs[mol.key]
        loo = leave_one_out_offsets(per_off, mol.key)
        geom = offset_corrected_geometry(mol, x, loo)
        sigma = prior_sigma_for_molecule(mol, per_off, mol.key)
        diag = leave_one_out_class_diagnostic(mol, x, per_len, mol.key)
        class_diagnostic_warnings(mol.key, diag, loo)
        assert per_off == snap_off and per_len == snap_len
        loo2 = leave_one_out_offsets(per_off, mol.key)
        assert loo2 == loo
        assert all(np.float64(loo2[c]).tobytes() == np.float64(loo[c]).tobytes() for c in loo)
        assert offset_corrected_geometry(mol, x, loo2).tobytes() == geom.tobytes()
        assert np.float64(prior_sigma_for_molecule(mol, per_off, mol.key)).tobytes() == \
            np.float64(sigma).tobytes()
    # The offsets themselves are still the plain class means.
    toy = {"a": {"C-N": [0.01], "C-H": [-0.01, -0.03]}, "b": {"C-N": [0.03]}}
    assert leave_one_out_offsets(toy, "c") == {"C-N": pytest.approx(0.02), "C-H": pytest.approx(-0.02)}


# ── the benchmark ──────────────────────────────────────────────────────────

def test_pool_theory_lengths_is_keyed_and_aligned_like_offsets_pool():
    import test_calibration_references as TC
    b = TC._bench()
    mols = [REF.ISOCYANIC_ACID, REF.HYDROGEN_CYANIDE]
    theory = {m.key: {"coords": _theory_like(m).tolist(),
                      "offsets": measure_offsets(m, _theory_like(m))} for m in mols}
    theory["water"] = {"coords": [[0, 0, 0]], "offsets": {"H-O": [0.0]}}   # not asked for
    lengths = b.pool_theory_lengths(theory, mols)
    offs = b.offsets_pool(theory, [m.key for m in mols])
    assert list(lengths) == list(offs) == [m.key for m in mols]
    for k in offs:
        assert {c: len(v) for c, v in lengths[k].items()} == {c: len(v) for c, v in offs[k].items()}


class _StubBackend:
    """Returns a zero Hessian: enough for main() to reach and write its rows."""
    def __init__(self, *a, **k):
        pass

    def run_hessian(self, coords):
        n = np.asarray(coords).shape[0]
        return types.SimpleNamespace(hessian_bohr=np.zeros((3 * n, 3 * n)))


def _run_pyrimidine(tmp_path, monkeypatch, todo):
    """main(), scored=validation, with every row on file except pyrimidine's
    ``todo`` configs; theory faked from stretched references, every quantum
    and fitting step stubbed to return the prior. Returns what main() printed,
    pyrimidine's rows as written, and the faked theory cache."""
    import test_calibration_references as TC
    b = TC._bench("scored=validation")
    mols = b.benchmark_pool() + list(REF.CALIBRATION_ONLY) + list(VAL.VALIDATION_SET)
    fake = {}
    for m in mols:
        x = _theory_like(m)
        fake[m.key] = {"coords": x.tolist(),
                       "offsets": {c: list(v) for c, v in measure_offsets(m, x).items()}}
    monkeypatch.setattr(b, "_ROOT", tmp_path)
    monkeypatch.setattr(b, "HESSIAN_CACHE", False)
    monkeypatch.setattr(b, "HESSIAN_CACHE_DIR", tmp_path / "hessian_cache")
    monkeypatch.setattr(b, "code_vintage", lambda: "test")
    monkeypatch.setattr(b, "theory_geometries", lambda ms: {m.key: fake[m.key] for m in ms})
    monkeypatch.setattr(b, "get_backend", lambda name: _StubBackend)
    monkeypatch.setattr(b, "RUN_DIAGNOSTICS", False)
    monkeypatch.setattr(b, "build_correction_table_from_hessian", lambda *a, **k: ("ctbl", {}))
    monkeypatch.setattr(b, "corrected_targets", lambda *a, **k: "targets")
    monkeypatch.setattr(b, "mixed_estimation_fit", lambda mol, t, prior, s: (np.asarray(prior), 0.0))
    monkeypatch.setattr(b, "hybrid_fit", lambda mol, isos, prior, *a, **k: np.asarray(prior))
    configs = [c for c in ["base", "offsets", "elec", "lam", "all"] if c in b.CONFIGS]
    rows = {m.key: {c: {"code_vintage": "test"} for c in configs} for m in VAL.VALIDATION_SET}
    for c in todo:
        del rows["pyrimidine"][c]
    path = tmp_path / "output" / "accuracy_upgrades_hf_6-31g_validation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.main()
    return buf.getvalue(), json.loads(path.read_text())["pyrimidine"], fake


def test_main_warns_and_records_the_diagnostic_on_offsets_rows(tmp_path, monkeypatch):
    printed, rows, fake = _run_pyrimidine(tmp_path, monkeypatch, ["base", "offsets"])
    warns = [s.strip() for s in printed.splitlines() if s.strip().startswith("[offsets]")]
    assert [w.split()[2] for w in warns] == ["C2-N1", "N1-C6"]
    assert all(w.startswith("[offsets] pyrimidine ") and "none within 0.08 A" in w for w in warns)
    diag = rows["offsets"]["offsets_class_diagnostic"]
    assert diag["flagged"] == ["C2-N1", "N1-C6"]
    assert diag["classes"]["C-N"]["n_pooled"] == 2
    assert rows["base"]["offsets_class_diagnostic"] is None
    # The row's offsets are the leave-one-out table of the whole pool, as before.
    pool_keys = [m.key for m in _ALL[:12]]
    assert rows["offsets"]["offsets_pool"] == sorted(pool_keys)
    per_off = {k: fake[k]["offsets"] for k in pool_keys}
    loo = leave_one_out_offsets(per_off, "pyrimidine")
    assert rows["offsets"]["loo_offsets_ma"] == {k: round(v * 1000.0, 2) for k, v in loo.items()}


def test_main_does_not_warn_when_no_offsets_row_is_run(tmp_path, monkeypatch):
    printed, rows, _fake = _run_pyrimidine(tmp_path, monkeypatch, ["base"])
    assert "[offsets]" not in printed
    assert rows["base"]["offsets_class_diagnostic"] is None
