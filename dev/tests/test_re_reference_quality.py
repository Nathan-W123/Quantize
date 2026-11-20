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
    # Calibration-only (no constants; measured for the bond offsets only).
    "METHYL_CHLORIDE", "HYDROGEN_CYANIDE", "HYDROGEN_PEROXIDE",
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


USABLE = {"WATER", "OZONE", "CHLOROFLUOROMETHANE", "VINYL_FLUORIDE",
          "METHYL_CHLORIDE", "HYDROGEN_CYANIDE", "HYDROGEN_PEROXIDE"}


def test_the_usable_references_pass():
    for name in sorted(USABLE):
        mol = getattr(REF, name)
        assert mol.equilibrium_reference_quality() == "ok", name
        assert mol.equilibrium_geometry is not None
        assert mol.geometry_re_uncertainty_ma <= REF.RE_REFERENCE_MAX_UNCERTAINTY_MA


def test_nothing_else_claims_an_equilibrium_reference():
    """If this starts failing, somebody added one -- check its uncertainty."""
    have = {n for n in ALL_MOLECULES
            if getattr(REF, n).equilibrium_geometry is not None}
    assert have == USABLE, have


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


def test_the_one_sourced_equilibrium_parameter_is_recorded():
    """Vinyl fluoride's equilibrium C=C-H angle, the only equilibrium number
    this session could source beyond water and ozone.

    Demaison, J., 2006, J. Mol. Spectrosc. 239, 201, via Table 19 of his 2007
    Mol. Phys. review (open access on HAL). A single angle cannot populate
    geometry_re, so nothing scores against it yet -- it is pinned so the
    value and its provenance survive.
    """
    assert REF.VINYL_FLUORIDE_RE_CCH_DEG == pytest.approx(125.95, abs=0.005)
    assert REF.VINYL_FLUORIDE_RE_CCH_UNC_DEG == pytest.approx(0.20, abs=0.005)
    src = REF.VINYL_FLUORIDE_RE_CCH_SOURCE
    assert "Demaison" in src and "239" in src
    # A parameter, not a geometry -- but since 2026-11-19 vinyl fluoride has a
    # full r_e (Piccardo et al. 2015), and the two equilibrium determinations
    # must agree on this angle (H-C1=C2: 126.18 against 125.95(20)).
    q = REF.VINYL_FLUORIDE.internal_coordinates(REF.VINYL_FLUORIDE.equilibrium_geometry)
    assert abs(q["H4-C1=C2"] - REF.VINYL_FLUORIDE_RE_CCH_DEG) < 1.5 * REF.VINYL_FLUORIDE_RE_CCH_UNC_DEG
    assert REF.VINYL_FLUORIDE.equilibrium_reference_quality() == "ok"


def test_published_r0_scatter_exceeds_our_measured_distance():
    """The published r0 determinations of that same angle span 8.3 degrees.

    From the review's Table 19: three r0 values at 123.7, 129.2 and 120.9
    degrees. Our data-only fit sat 2.29 deg from the published r0 for this
    molecule -- comfortably inside the published scatter, which is why the
    6.2 mA degeneracy measured this session is the state of the references
    rather than an artefact of the fit.
    """
    published_r0_deg = [123.7, 129.2, 120.9]
    scatter = max(published_r0_deg) - min(published_r0_deg)
    our_distance_deg = 2.2918       # measured, data-only fit vs published r_0
    assert scatter > 8.0
    assert our_distance_deg < scatter


def test_chlorofluoromethane_s_r_e_is_the_published_structure():
    """Vogt, Demaison & Rudolph (2014) r_e^SE, wired in 2026-11-19: the
    geometry must reproduce the published parameters, the uncertainty must be
    the largest quoted on a bond, and the provenance must say where the
    numbers were read (a compilation, not the paywalled paper)."""
    mol = REF.CHLOROFLUOROMETHANE
    g = mol.geometry_re

    def ang(i, j, k):
        a, b = g[i] - g[j], g[k] - g[j]
        return np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))

    p = REF.CH2CLF_RE_SE
    assert np.linalg.norm(g[1] - g[0]) == pytest.approx(p["C-Cl"][0], abs=1e-9)
    assert np.linalg.norm(g[2] - g[0]) == pytest.approx(p["C-F"][0], abs=1e-9)
    assert np.linalg.norm(g[3] - g[0]) == pytest.approx(p["C-H"][0], abs=1e-9)
    assert np.linalg.norm(g[4] - g[0]) == pytest.approx(p["C-H"][0], abs=1e-9)
    assert ang(1, 0, 2) == pytest.approx(p["Cl-C-F"][0], abs=1e-7)
    assert ang(1, 0, 3) == pytest.approx(p["Cl-C-H"][0], abs=1e-7)
    assert ang(3, 0, 4) == pytest.approx(p["H-C-H"][0], abs=1e-7)
    assert mol.geometry_re_uncertainty_ma == pytest.approx(
        1e3 * max(p[k][1] for k in ("C-Cl", "C-F", "C-H")))
    src = mol.geometry_re_source
    assert "Vogt" in src and "2873" in src and "NQCC" in src
    # Same frame and atom order as the effective structure: a few mA apart,
    # not a relabelled or mirrored molecule.
    d = np.sqrt(np.mean(np.sum((g - mol.geometry) ** 2, axis=1)))
    assert d < 0.02



def test_vinyl_fluoride_s_r_e_is_the_published_structure():
    """Piccardo et al. (2015) r_e^SE, wired in 2026-11-19: every published
    parameter reproduced, the uncertainty the largest quoted on a bond, H5
    cis to F as in the rest of this module, and the source named."""
    mol = REF.VINYL_FLUORIDE
    q = mol.internal_coordinates(mol.geometry_re)
    p = REF.VINYL_FLUORIDE_RE_SE
    for key, (value, _sd) in p.items():
        assert q[key] == pytest.approx(value, abs=1e-9), key
    # Derived, not tabulated: the angle Demaison 2006 quotes, within 1.2 sigma.
    assert q["H4-C1=C2"] == pytest.approx(360.0 - 121.72 - 112.10, abs=1e-9)
    assert abs(q["H4-C1=C2"] - REF.VINYL_FLUORIDE_RE_CCH_DEG) < 1.5 * 0.20
    g = mol.geometry_re
    assert np.linalg.norm(g[2] - g[4]) < np.linalg.norm(g[2] - g[5])  # H5 is cis
    bond_sd_ma = max(sd for k, (_v, sd) in p.items() if k in mol.bonds) * 1000.0
    assert mol.geometry_re_uncertainty_ma == pytest.approx(bond_sd_ma)
    assert "Piccardo" in mol.geometry_re_source and "2015" in mol.geometry_re_source
