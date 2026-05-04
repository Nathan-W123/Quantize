"""
Geometry seeds used in behavioural tests flow through::

    coarse guess (bond graph / PubChem) → :func:`backend.orca_cheap_opt.minimize_geometry_cheap_orca`

If ORCA is not installed, tests marked ``@pytest.mark.orca`` are skipped unless you set
``ORCA_EXE`` / PATH and rerun.

Environment:

* ``QUANTIZE_CHEAP_OPT_KEYWORD`` — ORCA preamble after ``!`` (default ``HF-3c Opt TightSCF``).

For coarse connectivity / PubChem only (no QM), continue to call
``guess_geometry_molecular_input`` directly in unit-parse tests or the alias parity test.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from backend.geometryguess import guess_geometry_molecular_input
from backend.orca_cheap_opt import minimize_geometry_cheap_orca


def initial_guess(*args, **kwargs):
    """Alias for :func:`guess_geometry_molecular_input` (coarse connectivity / PubChem)."""
    return guess_geometry_molecular_input(*args, **kwargs)


def require_orca_exe() -> str:
    """Return resolved ORCA path or ``pytest.skip``."""
    from backend.quantize import _find_orca

    ex = os.environ.get("QUANTIZE_ORCA_EXE") or os.environ.get("ORCA_EXE") or None
    try:
        return _find_orca(ex)
    except RuntimeError as e:
        pytest.skip(f"ORCA unavailable (needed for cheap-relax fixtures): {e}")


def relax_initial_geometry(
    coords,
    elems: list[str],
    work_dir: Path,
    *,
    orca_exe: str,
    stem: str = "relax",
    center: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Cheap ORCA optimization inside ``work_dir`` (must exist or parent must)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    return minimize_geometry_cheap_orca(
        coords,
        elems,
        workdir=str(work_dir),
        orca_executable=orca_exe,
        stem=stem,
        center=center,
    )


@pytest.fixture
def water_cheap_relaxed_guess(tmp_path):
    orca_exe = require_orca_exe()
    coords0, elems = initial_guess(
        None,
        elems=["O", "H", "H"],
        bonds=[(0, 1), (0, 2)],
        center=True,
    )
    return relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "water_hf_relax",
        orca_exe=orca_exe,
        stem="cheap_water",
    )


@pytest.fixture
def ocs_cheap_relaxed_guess(tmp_path):
    orca_exe = require_orca_exe()
    coords0, elems = initial_guess(
        None,
        elems=["O", "C", "S"],
        bonds=[(0, 1), (1, 2)],
        center=True,
    )
    return relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "ocs_hf_relax",
        orca_exe=orca_exe,
        stem="cheap_ocs",
    )


@pytest.fixture
def naphthalene_pubchem_then_relaxed(tmp_path):
    """PubChem CID 931 + cheap ORCA relaxation (heavy)."""
    orca_exe = require_orca_exe()
    coords0, elems = initial_guess(
        "931",
        center=False,
        pubchem_timeout=60.0,
    )
    return relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "naph_hf_relax",
        orca_exe=orca_exe,
        stem="cheap_naph",
    )


@pytest.fixture
def water_pubchem_then_relaxed(tmp_path):
    orca_exe = require_orca_exe()
    coords0, elems = initial_guess(
        "962",
        center=False,
        pubchem_prefer="cid",
        pubchem_timeout=45.0,
    )
    return relax_initial_geometry(
        coords0,
        elems,
        tmp_path / "water_pub_hf_relax",
        orca_exe=orca_exe,
        stem="cheap_water_pubchem",
    )


def assert_finite_coords(coords, elems):
    c = np.asarray(coords, dtype=float)
    assert c.ndim == 2 and c.shape[1] == 3
    assert np.all(np.isfinite(c))
    assert len(elems) == len(c)
