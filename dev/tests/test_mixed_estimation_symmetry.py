"""The mixed-estimation baseline keeps the molecule's symmetry (D-161).

Its predicates sit on symmetry-averaged bonds and angles, so before the
equality terms nothing but the rotational constants held an antisymmetric
distortion, and the fit used it: chlorofluoromethane's two C-H bonds came out
0.96 A apart at RHF/6-31G while the averaged score read 4.26 mA. Two
chlorine isotopologues cannot see that direction, so it shows even with
constants computed from the reference structure itself (old code: the C-H
pair 383 mA apart here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / ".github", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backend.spectral.centrifugal_distortion import rotational_constants_mhz

from dev.monofluoro_references import MOLECULES, MOLECULES_SET2


@pytest.fixture(scope="module")
def me():
    saved = sys.argv
    sys.argv = ["mixed_estimation_baseline.py"]   # the module reads argv at import
    try:
        import scripts.mixed_estimation_baseline as meb
    finally:
        sys.argv = saved
    return meb


def _spread(mol, x):
    return max(max(v) - min(v) for v in (
        [np.linalg.norm(x[j] - x[i]) for i, j in pairs] for pairs in mol.bonds.values()))


def _targets(mol, geom):
    out = []
    for sp in mol.species:
        m = np.asarray(mol.masses, dtype=float).copy()
        for i, v in sp.subs.items():
            m[i] = v
        abc = rotational_constants_mhz(geom, m)
        out += [{"masses": m, "component": c, "value": float(abc[c]),
                 "sigma": 1e-3 * float(abc[c])} for c in range(3)]
    return out


def test_mixed_estimation_keeps_equivalent_bonds_equal(me):
    mol = {m.key: m for m in list(MOLECULES) + list(MOLECULES_SET2)}["chlorofluoromethane"]
    assert len(mol.bonds["C-H"]) == 2
    ref = np.asarray(mol.geometry, dtype=float)
    theory = ref.copy()
    for pairs in mol.bonds.values():         # every bond 20 mA long, symmetrically
        for i, j in pairs:
            u = theory[j] - theory[i]
            theory[j] = theory[i] + u * (1.0 + 0.02 / np.linalg.norm(u))
    theory += 1e-5 * np.random.default_rng(3).standard_normal(theory.shape)

    geom, chi2 = me.mixed_estimation_fit(mol, _targets(mol, ref), theory, 0.02)

    assert np.isfinite(chi2)
    assert _spread(mol, geom) < _spread(mol, theory) + 1e-6
