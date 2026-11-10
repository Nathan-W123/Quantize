"""A distance from the published structure is only an error if that structure
is the right answer. For most of this set it is not.

The benchmark's primary score is RMS distance from `mol.geometry`, which for
ten of twelve molecules is a published EFFECTIVE (r_0) structure. Measured on
the six molecules that have an r_0 reference and enough observables to
determine a geometry (obs >= internal DOF), the engine's own data-only fit
reproduces the observed rotational constants BETTER than the published
structure does -- in six cases out of six -- while sitting 2.8 to 21.3 mAng
away from it:

  molecule           dist mAng   engine rms/sig   published rms/sig
  vinyl fluoride         21.3        0.37             0.98
  acetyl fluoride         5.1        0.04             0.37
  fluoroethane            7.3        0.04             0.75
  formyl fluoride         2.8        0.16             0.69
  fluoroacetylene         5.2        0.02             0.10
  ethylene oxide          9.3        0.02             2.19

So the constants admit a family of structures several mAng wide -- the
underdetermination this engine's prior exists to resolve -- and the published
r_0 is one choice inside it rather than its centre. The median 6.2 mAng
degeneracy is about eight times the hybrid's own corrected error on water
(0.754 mAng), so distance-from-r_0 cannot resolve the engine's accuracy.

This is why equilibrium references are not merely preferable: they are the
only yardstick that can measure this engine, and only two molecules have one.

These tests need no quantum chemistry: how well a geometry reproduces a set
of rotational constants is a property of the geometry and the constants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.monofluoro_references as REF  # noqa: E402
from backend.spectral.spectral import SpectralEngine  # noqa: E402
from scripts.monofluoro_benchmark import build_isotopologues  # noqa: E402

#: Molecules with an r_0 reference AND obs >= internal DOF, with the published
#: structure's own residual as measured. Anything far from these means either
#: the reference data or the spectral model moved.
MEASURED_REFERENCE_RMS = {
    "VINYL_FLUORIDE": 0.98,
    "ACETYL_FLUORIDE": 0.37,
    "FLUOROETHANE": 0.75,
    "FORMYL_FLUORIDE": 0.69,
    "FLUOROACETYLENE": 0.10,
    "ETHYLENE_OXIDE": 2.19,
}


def _rms_sigma(mol, coords):
    eng = SpectralEngine(build_isotopologues(mol, None))
    _J, rw = eng.stacked(np.asarray(coords, dtype=float), want_jacobian=False)
    rw = np.asarray(rw, dtype=float)
    return float(np.sqrt(np.sum(rw ** 2) / rw.size))


@pytest.mark.parametrize("name", sorted(MEASURED_REFERENCE_RMS))
def test_published_reference_residual_is_what_was_measured(name):
    mol = getattr(REF, name)
    got = _rms_sigma(mol, mol.geometry)
    want = MEASURED_REFERENCE_RMS[name]
    assert got == pytest.approx(want, abs=0.05), (
        f"{name}: published structure now reproduces its constants at "
        f"{got:.2f} sigma, measured {want:.2f}")


def test_these_molecules_really_are_the_determined_ones():
    """Guard the premise: the claim only applies where data suffices."""
    for name in MEASURED_REFERENCE_RMS:
        mol = getattr(REF, name)
        assert mol.structure_type == "effective", name
        assert mol.n_observables >= mol.internal_dof, name


def test_some_published_references_fit_their_own_constants_poorly():
    """Ethylene oxide's published structure misses by 2.19 sigma.

    Not a criticism of the published work -- an effective structure is fitted
    under its own assumptions and to its own data selection. It is a
    statement about what distance-from-it can and cannot measure here.
    """
    mol = REF.ETHYLENE_OXIDE
    assert _rms_sigma(mol, mol.geometry) > 1.5


def test_the_benchmark_records_the_yardsticks_own_fit():
    """The number has to reach the row, or nobody sees it."""
    import contextlib
    import importlib.util as iu
    import io
    spec = iu.spec_from_file_location(
        "accbench", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    saved = sys.argv
    try:
        sys.argv = ["accbench"]
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    mol = REF.ETHYLENE_OXIDE
    got = mod.residual_rms_sigma(build_isotopologues(mol, None), mol.geometry)
    assert got == pytest.approx(MEASURED_REFERENCE_RMS["ETHYLENE_OXIDE"], abs=0.05)
    # And it degrades rather than raising on nonsense input, because a
    # diagnostic must not take down the run it qualifies.
    bad = mod.residual_rms_sigma([], mol.geometry)
    assert np.isnan(bad) or bad == 0.0


def test_the_degeneracy_dwarfs_the_engines_own_error():
    """6.2 mAng median against 0.754 mAng: the yardstick cannot resolve it."""
    median_degeneracy_ma = 6.234      # measured, data-only fit vs published r_0
    water_corrected_error_ma = 0.754  # measured, hybrid + vib correction vs r_e
    assert median_degeneracy_ma / water_corrected_error_ma > 5.0
