"""
Live PubChem fetch + cheap ORCA minimization—requires network **and** ORCA.

Skip offline with ``pytest -m "not network"`` / ``pytest -m "not orca"``.
"""

from __future__ import annotations

import pytest

from tests.conftest import assert_finite_coords


@pytest.mark.network
@pytest.mark.orca
@pytest.mark.slow
def test_pubchem_cid_naphthalene_relaxed(naphthalene_pubchem_then_relaxed):
    coords, elems = naphthalene_pubchem_then_relaxed
    assert len(elems) == 18
    assert_finite_coords(coords, elems)


@pytest.mark.network
@pytest.mark.orca
def test_pubchem_water_cid_relaxed(water_pubchem_then_relaxed):
    coords, elems = water_pubchem_then_relaxed
    assert elems[0] == "O"
    assert len(elems) == 3
    assert_finite_coords(coords, elems)


@pytest.mark.network
@pytest.mark.orca
def test_pubchem_prefer_name_relaxed(tmp_path):
    from tests.conftest import initial_guess, relax_initial_geometry, require_orca_exe

    exe = require_orca_exe()
    coords0, elems = initial_guess("water", pubchem_prefer="name", pubchem_timeout=45.0)
    coords, elems2 = relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "wat_name_hf",
        orca_exe=exe,
        stem="cheap_water_name",
    )
    assert elems2[0] == "O"
    assert_finite_coords(coords, elems2)


@pytest.mark.network
@pytest.mark.orca
def test_pubchem_smiles_ethanol_relaxed(tmp_path):
    from tests.conftest import initial_guess, relax_initial_geometry, require_orca_exe

    exe = require_orca_exe()
    coords0, elems = initial_guess("CCO", pubchem_prefer="smiles", pubchem_timeout=45.0)
    coords, elems2 = relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "cco_hf",
        orca_exe=exe,
        stem="cheap_ethanol",
    )
    assert len(coords) == len(elems2)
    assert_finite_coords(coords, elems2)
