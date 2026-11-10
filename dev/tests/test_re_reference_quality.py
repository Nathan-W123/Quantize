"""An equilibrium reference is only a yardstick if it is finer than the engine.

Three scales, all measured, decide the bar:

  0.754 mA   the hybrid's own error on water after the vibrational
             correction, scored against a published r_e
  6.2 mA     median distance between the engine's data-only fit and the
             published EFFECTIVE structures, over the six molecules whose
             data determines a geometry. The engine reproduces the constants
             BETTER than those structures do in all six cases, so that is the
             width of the family the constants allow, not an error
  5 mA       what CCCBDB/Landolt-Bornstein actually quote on the same
             structures (formyl fluoride: +/-0.005 A on bonds, +/-0.5 to
             3 deg on angles), which agrees with the 6.2

So a reference coarser than a few tenths of a mAngstrom cannot resolve this
engine, and one wired in without its uncertainty cannot be shown to.

These tests exist because two plausible references were ruled out this round
on exactly that basis -- CCCBDB's formyl fluoride entry at +/-5 mA and of
unstated type, and dev/re_fcho_ccsdt.json, a computed CCSD(T)/cc-pVTZ r_e
sitting in this repo whose own file reports 2-3 mA of method bias. Either
would have scored the reference's error as the engine's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.monofluoro_references as REF  # noqa: E402

#: Measured hybrid error on water, corrected, against its published r_e.
WATER_CORRECTED_ERROR_MA = 0.754
#: Measured median degeneracy of the effective references.
EFFECTIVE_REFERENCE_DEGENERACY_MA = 6.234

ALL_MOLECULES = [
    "VINYL_FLUORIDE", "ACETYL_FLUORIDE", "FLUOROETHANE", "FLUOROBENZENE",
    "FORMYL_FLUORIDE", "FLUOROACETYLENE", "CHLOROFLUOROMETHANE", "WATER",
    "ACETONE", "OZONE", "ETHYLENE_OXIDE", "ISOCYANIC_ACID",
]


def test_the_bar_sits_below_the_engines_own_error():
    """Otherwise it cannot resolve the thing it is meant to measure."""
    assert REF.RE_REFERENCE_MAX_UNCERTAINTY_MA < WATER_CORRECTED_ERROR_MA


def test_the_bar_excludes_the_effective_references_by_a_wide_margin():
    assert (EFFECTIVE_REFERENCE_DEGENERACY_MA
            > 10 * REF.RE_REFERENCE_MAX_UNCERTAINTY_MA)


@pytest.mark.parametrize("name", ALL_MOLECULES)
def test_every_molecule_reports_a_verdict(name):
    mol = getattr(REF, name)
    assert mol.equilibrium_reference_quality() in {
        "none", "ok", "too_coarse", "unstated"}


def test_the_two_usable_references_pass():
    for name in ("WATER", "OZONE"):
        mol = getattr(REF, name)
        assert mol.equilibrium_reference_quality() == "ok", name
        assert mol.equilibrium_geometry is not None
        assert mol.geometry_re_uncertainty_ma <= REF.RE_REFERENCE_MAX_UNCERTAINTY_MA


def test_nothing_else_claims_an_equilibrium_reference():
    """If this starts failing, somebody added one -- check its uncertainty."""
    have = {n for n in ALL_MOLECULES
            if getattr(REF, n).equilibrium_geometry is not None}
    assert have == {"WATER", "OZONE"}, have


def test_an_unstated_uncertainty_is_not_treated_as_good():
    """The failure mode this guards: a geometry wired in with no uncertainty."""
    import copy
    mol = copy.copy(REF.FORMYL_FLUORIDE)
    mol.geometry_re = np.asarray(REF.FORMYL_FLUORIDE.geometry, dtype=float)
    mol.geometry_re_source = "test"
    mol.geometry_re_uncertainty_ma = None
    assert mol.equilibrium_reference_quality() == "unstated"


def test_a_too_coarse_reference_is_rejected():
    """CCCBDB's formyl fluoride entry quotes +/-5 mA; it must not pass."""
    import copy
    mol = copy.copy(REF.FORMYL_FLUORIDE)
    mol.geometry_re = np.asarray(REF.FORMYL_FLUORIDE.geometry, dtype=float)
    mol.geometry_re_source = "CCCBDB / Landolt-Bornstein 1976"
    mol.geometry_re_uncertainty_ma = 5.0
    assert mol.equilibrium_reference_quality() == "too_coarse"


def test_the_repo_s_computed_fcho_structure_would_not_qualify():
    """dev/re_fcho_ccsdt.json is the trap: it is in the repo and looks ready.

    Its own file records that it did not converge and that the method bias is
    2-3 mA. That is three to four times the engine's error, so it cannot be
    the yardstick -- and this test states why, so the file's presence does not
    read as an invitation.
    """
    path = _ROOT / "dev" / "re_fcho_ccsdt.json"
    if not path.exists():
        pytest.skip("re_fcho_ccsdt.json not present")
    data = json.loads(path.read_text())
    assert data.get("converged") is False
    note = str(data.get("localization_note", ""))
    assert "2-3 mA" in note or "2-3 mAng" in note
    # The stated bias, read off that note, is far outside the bar.
    assert 2.0 > REF.RE_REFERENCE_MAX_UNCERTAINTY_MA
    assert REF.FORMYL_FLUORIDE.equilibrium_geometry is None, (
        "formyl fluoride now has an r_e reference -- if it is the computed "
        "CCSD(T) structure, it cannot measure the engine")
