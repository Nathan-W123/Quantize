"""The rotational g-tensor is isotope-dependent, and the engine has to use that.

g_alpha is the electronic angular momentum divided through by the inertia
tensor, so it is as mass-dependent as a rotational constant is. The correction
it feeds,

    delta_elec = -(m_e/m_p) * g_alpha * B_obs

is applied species by species, and isotopic differences are the entire source of
structural information in the fit -- so reusing the parent's g for every
substituted species writes an isotope-shaped error into exactly the quantity
being determined. That is the worst possible shape for an error here, whatever
its absolute size.

It was happening two ways at once. The caller computed one g-tensor per molecule
and applied it to every isotopologue, and underneath that the masses passed to
rotational_g_tensor were used only to pick the principal axis frame: pyscf's
rotational_gtensor builds its own inertia tensor from mol.atom_mass_list with
isotope_avg=True, which is standard atomic weights. So chlorine came out at
35.45 rather than as either isotope, and a deuterated species was computed as
though it were protiated.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.rovib_corrections import _g_for_species

try:
    from backend.spectral.electronic_g import (
        g_tensors_for_isotopologues,
        rotational_g_tensor,
    )
    import pyscf.prop.rotational_gtensor  # noqa: F401
    _HAS_GTENSOR = True
except Exception:                                      # noqa: BLE001
    _HAS_GTENSOR = False

pytestmark = pytest.mark.skipif(not _HAS_GTENSOR,
                                reason="pyscf-properties not installed")

M_O, M_H, M_D = 15.99491462, 1.00782503, 2.01410178


def _water_coords(r=0.95785, deg=104.508):
    half = np.radians(deg) / 2.0
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.sin(half), r * np.cos(half), 0.0],
                     [-r * np.sin(half), r * np.cos(half), 0.0]])


@pytest.fixture(scope="module")
def water_gs():
    """g-tensors for H2-16O, D2-16O and the parent translated off origin."""
    coords = _water_coords()
    return {
        "h2o": rotational_g_tensor(["O", "H", "H"], coords,
                                   np.array([M_O, M_H, M_H])),
        "d2o": rotational_g_tensor(["O", "H", "H"], coords,
                                   np.array([M_O, M_D, M_D])),
        "shifted": rotational_g_tensor(["O", "H", "H"],
                                       coords + np.array([3.1, -2.4, 0.7]),
                                       np.array([M_O, M_H, M_H])),
    }


# ── the masses have to reach the property code ──────────────────────────────

def test_deuteration_changes_the_g_tensor_by_a_factor_of_two(water_gs):
    """The check that the mass override actually bites.

    Measured at HF/6-31g: g_bb is +0.682 for H2-16O and +0.341 for D2-16O.
    Before mol.nucprop was set, both came back identical -- which is what an
    isotope-averaged mass table gives, and it is not a small error.
    """
    h2o, d2o = water_gs["h2o"], water_gs["d2o"]
    assert h2o["B"] == pytest.approx(0.682, abs=0.01)
    assert d2o["B"] == pytest.approx(0.341, abs=0.01)
    assert h2o["B"] > 1.8 * d2o["B"]
    for comp in "ABC":
        assert abs(h2o[comp] - d2o[comp]) > 0.2


def test_every_component_moves_on_substitution(water_gs):
    h2o, d2o = water_gs["h2o"], water_gs["d2o"]
    assert all(h2o[c] > d2o[c] for c in "ABC")


def test_the_g_tensor_is_translation_invariant(water_gs):
    """It is defined about the centre of mass, so where the molecule sits in
    the input frame must not matter. A gauge-origin slip would show up here and
    nowhere else."""
    a, b = water_gs["h2o"], water_gs["shifted"]
    for comp in "ABC":
        assert a[comp] == pytest.approx(b[comp], abs=1e-4)


# ── the per-species helper ──────────────────────────────────────────────────

def test_g_tensors_for_isotopologues_gives_one_per_species():
    coords = _water_coords()
    isos = [
        {"name": "H2-16O", "masses": np.array([M_O, M_H, M_H])},
        {"name": "D2-16O", "masses": np.array([M_O, M_D, M_D])},
    ]
    got = g_tensors_for_isotopologues(["O", "H", "H"], coords, isos)
    assert set(got) == {"H2-16O", "D2-16O"}
    assert got["H2-16O"]["B"] != got["D2-16O"]["B"]


def test_a_species_without_masses_is_skipped():
    coords = _water_coords()
    isos = [{"name": "good", "masses": np.array([M_O, M_H, M_H])},
            {"name": "bad"}]
    got = g_tensors_for_isotopologues(["O", "H", "H"], coords, isos)
    assert set(got) == {"good"}


# ── resolve_corrections accepts both shapes ─────────────────────────────────

@pytest.mark.parametrize("g,name,want", [
    ({"A": 0.5, "B": 0.6, "C": 0.7}, "anything", {"A": 0.5, "B": 0.6, "C": 0.7}),
    ({"iso1": {"A": 0.1}, "iso2": {"A": 0.2}}, "iso2", {"A": 0.2}),
    ({"iso1": {"A": 0.1}}, "missing", None),
    ({}, "iso", None),
    (None, "iso", None),
])
def test_g_for_species_tells_the_two_shapes_apart(g, name, want):
    """Component-keyed applies to everything; species-keyed is looked up.

    The discrimination is on whether the top-level keys are component labels,
    so a molecule whose isotopologue happened to be named "A" would be
    misread -- which no naming scheme in this repository produces, and which a
    species-keyed dict of one entry would make obvious immediately.
    """
    assert _g_for_species(g, name) == want


def test_per_species_g_produces_different_electronic_deltas():
    from backend.spectral.rovib_corrections import resolve_corrections

    isos = [
        {"name": "light", "masses": [M_O, M_H, M_H],
         "obs_constants": [800000.0], "component_indices": [1],
         "sigma_constants": [10.0]},
        {"name": "heavy", "masses": [M_O, M_D, M_D],
         "obs_constants": [800000.0], "component_indices": [1],
         "sigma_constants": [10.0]},
    ]
    per_species = resolve_corrections(
        isos, correction_elec=True, elems=["O", "H", "H"],
        g_tensor={"light": {"B": 0.682}, "heavy": {"B": 0.341}})
    shared = resolve_corrections(
        isos, correction_elec=True, elems=["O", "H", "H"],
        g_tensor={"B": 0.682})

    def elec(targets, label):
        t = next(t for t in targets if t.isotopologue_label == label)
        return next(r.delta_mhz for r in t.correction_records
                    if "elec" in r.source or "elec" in str(r.method).lower()
                    or "g_B" in str(r.notes))

    assert elec(per_species, "light") != pytest.approx(elec(per_species, "heavy"))
    assert elec(shared, "light") == pytest.approx(elec(shared, "heavy"))
    # And the mis-correction the shared form applies to the substituted species
    # is the size of the whole correction, not a refinement of it.
    assert abs(elec(shared, "heavy") - elec(per_species, "heavy")) \
        > 0.5 * abs(elec(per_species, "heavy"))
