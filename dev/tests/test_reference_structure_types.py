"""Every reference says what KIND of structure it is.

An error measured against an r_e reference is the engine's own: r_e is the
quantity the engine targets. An error measured against an r_s or effective
reference shares the blame with the yardstick, because those carry a
vibrational-averaging offset from r_e. The benchmark docstring has said so in
prose for a while -- ozone reaches 1.00 mA against a reference whose own floor
the literature puts at 4.1 mA -- but a number read out of the results JSON
carried no trace of it. These tests pin the field that fixes that.

They deliberately do NOT test a numeric floor. See ReferenceMolecule for why
one cannot be computed from the final geometry.
"""

from __future__ import annotations

import pytest

from dev.monofluoro_references import (
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    WATER_SET,
)

_ALLOWED = {"r_e", "r_s", "r_0", "r_z", "effective"}


def _pool():
    return list(MOLECULES) + list(MOLECULES_SET2) + [WATER_SET[0], OZONE,
                                                     ISOCYANIC_ACID]


@pytest.mark.parametrize("mol", _pool(), ids=lambda m: m.key)
def test_every_reference_declares_a_known_type(mol):
    assert mol.structure_type in _ALLOWED, mol.structure_type


def test_water_is_the_equilibrium_reference():
    # The only r_e structure in the set, and the reason its row is the one
    # that scores the engine alone.
    assert WATER_SET[0].structure_type == "r_e"
    assert "r_e" in WATER_SET[0].structure_source


def test_the_r_s_references_are_the_ones_whose_sources_say_so():
    # Typed from what the cited source states, never inferred from the date.
    for mol in (OZONE, ISOCYANIC_ACID):
        assert mol.structure_type == "r_s"
        assert "r_s" in mol.structure_source


def test_sources_that_state_no_convention_are_marked_effective():
    # The six monofluoro references cite microwave papers from 1953-1989 whose
    # convention is not recorded in the stored source string. "effective"
    # says exactly that, rather than guessing r_0 or r_s -- a guess that would
    # be invisible once it was in the field.
    for mol in list(MOLECULES) + list(MOLECULES_SET2):
        if mol.structure_type == "effective":
            assert "r_s" not in mol.structure_source
            assert "r_e" not in mol.structure_source


def test_only_one_reference_in_the_set_is_equilibrium():
    # The headline constraint on what this benchmark can demonstrate: eight of
    # nine references are not the quantity the engine targets. If this count
    # changes, the accuracy claims the set can support change with it.
    equilibrium = [m.key for m in _pool() if m.structure_type == "r_e"]
    assert equilibrium == ["water"], equilibrium


def test_a_type_is_never_silently_absent():
    # The field has a default, so a new reference added without thinking gets
    # "effective" -- the conservative answer -- rather than None or a crash.
    for mol in _pool():
        assert isinstance(mol.structure_type, str) and mol.structure_type
