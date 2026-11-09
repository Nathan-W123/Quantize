"""The route by which the torsion package reaches the optimiser.

Quantize has a full torsion-rotation Hamiltonian, a torsional fitter and
torsional averaging, all reachable from runner/run_generic.py. None of it was
reachable from MolecularOptimizer, which is what every accuracy benchmark runs
through, so none of that physics had ever been scored. These cover the path
that closes the gap: a computed torsional contribution supplied to the
optimiser must reach the correction table and change the correction.

The failure this guards against is silence. A switch that is wired but inert
reports no error and produces a plausible number; this project has already
spent a benchmark run discovering exactly that about centrifugal-distortion
constants as fit data. So these assert that turning the treatment on CHANGES
something, not merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import dev.analytic_water_backend  # noqa: F401  (registers "analytic_water")
from backend.quantize import MolecularOptimizer
from backend.registry import get_backend
from backend.spectral.harmonic_alpha import (
    TorsionReplacement,
    build_correction_table_from_hessian,
)

ELEMS = ["O", "H", "H"]
MASSES = [15.9949146196, 1.00782503207, 1.00782503207]


def _water():
    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    hess = get_backend("analytic_water")(elems=ELEMS).run_hessian(coords).hessian_bohr
    isos = [{"name": "H2-16O", "masses": MASSES, "component_indices": [0, 1, 2]}]
    return coords, hess, isos


def _alphas(table):
    entry = table["H2-16O"]
    return {k: float(v["alpha_sum_mhz"]) for k, v in entry.items()
            if isinstance(v, dict) and "alpha_sum_mhz" in v}


def _sigmas(table):
    entry = table["H2-16O"]
    return {k: float(v.get("sigma_mhz", 0.0)) for k, v in entry.items()
            if isinstance(v, dict) and "alpha_sum_mhz" in v}


def test_correction_table_is_unchanged_without_a_replacement():
    """Molecules with no rotor must be bit-identical to before this existed."""
    coords, hess, isos = _water()
    base, _ = build_correction_table_from_hessian(hess, coords, isos)
    same, _ = build_correction_table_from_hessian(
        hess, coords, isos, torsion_replacement=None)
    assert _alphas(same) == _alphas(base)
    assert _sigmas(same) == _sigmas(base)


def test_a_replacement_reaches_the_correction_table():
    """The bridge itself: supplied contribution changes the correction."""
    coords, hess, isos = _water()
    base, _ = build_correction_table_from_hessian(hess, coords, isos)
    tors, _ = build_correction_table_from_hessian(
        hess, coords, isos,
        torsion_replacement=TorsionReplacement(
            alpha_mhz={"A": 5000.0, "B": 0.0, "C": 0.0},
            sigma_mhz={"A": 50.0, "B": 0.0, "C": 0.0},
            mode_indices=(0,)),
    )
    a_base, a_tors = _alphas(base), _alphas(tors)
    assert a_tors != a_base, "the replacement never reached the table"
    # Water's mode 0 is the bend, which carries a large share of alpha, so
    # removing it must move every component rather than only the one the
    # replacement names.
    for k in ("A", "B", "C"):
        assert abs(a_tors[k] - a_base[k]) > 1.0, f"{k} did not move"


def test_replacing_a_mode_narrows_the_truncation_sigma():
    """A mode handled explicitly stops contributing its magnitude to sigma.

    sigma here is driven by the size of the terms retained in a truncated
    series. A torsional mode treated by averaging is no longer one of those
    terms, so its magnitude should leave the uncertainty with it. If sigma did
    not fall, the treatment would be paying for precision it never collects.
    """
    coords, hess, isos = _water()
    base, _ = build_correction_table_from_hessian(hess, coords, isos)
    tors, _ = build_correction_table_from_hessian(
        hess, coords, isos,
        torsion_replacement=TorsionReplacement(
            alpha_mhz={}, sigma_mhz={"A": 50.0}, mode_indices=(0,)),
    )
    s_base, s_tors = _sigmas(base), _sigmas(tors)
    assert s_tors["A"] < s_base["A"], (
        f"sigma_A did not fall: {s_tors['A']} against {s_base['A']}"
    )


def test_optimiser_accepts_and_stores_a_replacement():
    """The parameter exists on the class the benchmarks construct."""
    coords, _, _ = _water()
    iso = {
        "name": "H2-16O",
        "masses": MASSES,
        "obs_constants": [835840.0, 435350.0, 278140.0],
        "sigma_constants": [1.0, 1.0, 1.0],
        "component_indices": [0, 1, 2],
    }
    replacement = TorsionReplacement(alpha_mhz={"A": 1.0}, mode_indices=(0,))
    opt = MolecularOptimizer(
        elems=ELEMS, coords=coords, isotopologues=[iso],
        torsion_replacement=replacement,
    )
    assert opt._torsion_replacement is replacement


def test_modes_are_selected_by_frequency_when_indices_are_omitted():
    """freq_cm must pick the same modes lam_freq_cm would, so the two
    treatments never act on different sets."""
    omega = [120.0, 1600.0, 3800.0]
    by_freq = TorsionReplacement(alpha_mhz={}, freq_cm=350.0)
    assert by_freq.modes_for(omega) == [0]
    explicit = TorsionReplacement(alpha_mhz={}, mode_indices=(1, 2))
    assert explicit.modes_for(omega) == [1, 2]
    assert TorsionReplacement(alpha_mhz={}).modes_for(omega) == []
