"""Scoring against r_e, where a published equilibrium structure exists.

The engine fits r_e. Eight of the nine references are not r_e, so an error
measured against them is shared with the yardstick. Ozone is the case that
makes this concrete and testable: the repository holds BOTH its r_s reference
and a published r_e (Tyuterev et al. 1999), and they are 1.06 mA apart --
about the size of the best result the benchmark reports for that molecule.

So these tests pin the plumbing that lets a result be scored against both,
and the arithmetic that makes the difference visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from dev.monofluoro_references import (
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    OZONE_R_E_ANG,
    OZONE_THETA_E_DEG,
    WATER_SET,
)
from scripts.mixed_estimation_baseline import rms_angle_error, rms_bond_error

WATER = WATER_SET[0]


def _pool():
    return list(MOLECULES) + list(MOLECULES_SET2) + [WATER, OZONE, ISOCYANIC_ACID]


class TestEquilibriumGeometryProperty:
    def test_water_returns_its_primary_geometry(self):
        # Water's primary reference IS r_e, so there is no second structure.
        assert WATER.structure_type == "r_e"
        np.testing.assert_allclose(WATER.equilibrium_geometry, WATER.geometry)

    def test_ozone_returns_the_separate_r_e_not_its_r_s_primary(self):
        assert OZONE.structure_type == "r_s"
        assert not np.allclose(OZONE.equilibrium_geometry, OZONE.geometry)
        np.testing.assert_allclose(OZONE.equilibrium_geometry, OZONE.geometry_re)

    def test_ozone_r_e_matches_the_published_numbers(self):
        c = np.asarray(OZONE.equilibrium_geometry, dtype=float)
        r = float(np.linalg.norm(c[0] - c[1]))
        u, v = c[1] - c[0], c[2] - c[0]
        ang = float(np.degrees(np.arccos(
            np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))))
        assert r == pytest.approx(OZONE_R_E_ANG, abs=1e-9)
        assert ang == pytest.approx(OZONE_THETA_E_DEG, abs=1e-6)

    def test_ozone_cites_its_equilibrium_source(self):
        assert "Tyuterev" in OZONE.geometry_re_source
        assert "1999" in OZONE.geometry_re_source

    def test_molecules_without_a_published_r_e_return_none(self):
        # Seven of nine. Returning None rather than falling back to r_s is the
        # point: a caller that wants equilibrium must skip them, not be handed
        # a substitution structure wearing the wrong label.
        without = [m.key for m in _pool() if m.equilibrium_geometry is None]
        assert set(without) == {
            "vinyl_fluoride", "acetyl_fluoride", "fluoroethane",
            "formyl_fluoride", "fluoroacetylene", "chlorofluoromethane",
            "isocyanic_acid",
        }, without


class TestTheTwoOzoneReferencesDisagreeByTheReportedAmount:
    def test_the_references_are_about_one_milliangstrom_apart(self):
        # The whole reason for keeping both. If this ever shrinks to nothing,
        # the distinction stops mattering and these tests should go.
        rms, _ = rms_bond_error(OZONE, OZONE.equilibrium_geometry)
        assert rms == pytest.approx(1.06, abs=0.05)

    def test_scoring_the_same_structure_against_both_gives_different_answers(self):
        # One fitted structure, two yardsticks. Here the "fit" is simply the
        # r_e structure itself, so against r_e it is exact and against r_s it
        # is off by the reference difference -- which is the cleanest possible
        # demonstration that the choice of reference is not a detail.
        fit = OZONE.equilibrium_geometry
        vs_re, _ = rms_bond_error(OZONE, fit, OZONE.equilibrium_geometry)
        vs_rs, _ = rms_bond_error(OZONE, fit)
        assert vs_re == pytest.approx(0.0, abs=1e-9)
        assert vs_rs > 1.0

    def test_the_angle_difference_is_small_but_real(self):
        deg, _ = rms_angle_error(OZONE, OZONE.equilibrium_geometry)
        assert deg == pytest.approx(0.026, abs=0.005)


class TestRefCoordsParameter:
    def test_default_is_the_primary_reference(self):
        a, _ = rms_bond_error(OZONE, OZONE.geometry)
        b, _ = rms_bond_error(OZONE, OZONE.geometry, OZONE.geometry)
        assert a == pytest.approx(0.0, abs=1e-9)
        assert b == pytest.approx(0.0, abs=1e-9)

    def test_passing_a_reference_changes_the_answer(self):
        a, _ = rms_bond_error(OZONE, OZONE.geometry)
        b, _ = rms_bond_error(OZONE, OZONE.geometry, OZONE.equilibrium_geometry)
        assert a == pytest.approx(0.0, abs=1e-9)
        assert b > 1.0

    def test_angle_helper_takes_it_too(self):
        a, _ = rms_angle_error(OZONE, OZONE.geometry)
        b, _ = rms_angle_error(OZONE, OZONE.geometry, OZONE.equilibrium_geometry)
        assert a == pytest.approx(0.0, abs=1e-9)
        assert b > 0.01

    def test_water_scores_the_same_either_way(self):
        # Its primary reference already IS the equilibrium one, so passing it
        # explicitly must change nothing. Guards against the two paths drifting.
        fit = np.asarray(WATER.geometry, dtype=float) * 1.001
        a, _ = rms_bond_error(WATER, fit)
        b, _ = rms_bond_error(WATER, fit, WATER.equilibrium_geometry)
        assert a == pytest.approx(b, rel=1e-12)


class TestNoDuplicatedLiteratureValues:
    def test_the_predictor_script_uses_the_reference_module(self):
        # The Tyuterev numbers were written out in two files. A literature
        # value kept in two places is one that will eventually disagree with
        # itself, so this pins that they now come from one source.
        from scripts.correction_error_predictors import R_E
        np.testing.assert_allclose(R_E["ozone"][1], OZONE.equilibrium_geometry)
        np.testing.assert_allclose(R_E["water"][1], WATER.equilibrium_geometry)
