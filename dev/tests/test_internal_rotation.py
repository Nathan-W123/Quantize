"""Tests for backend/torsion/internal_rotation.py.

The module computes F and rho, which until now were hand-entered literature
values in every TorsionHamiltonianSpec in the repository. Two kinds of check
here, and the distinction matters:

CLOSED FORM. Where lambda lies exactly along a principal axis the reduced
moment collapses to r = 1 - I_alpha/I_g, and I_alpha for a methyl is
3 m_H (r_CH sin theta)^2 by hand. Those are exact, so they are asserted to
machine precision and they are what actually pins the relations.

LITERATURE ANCHOR. Methanol's accepted F = 27.6 cm^-1 and rho = 0.81 are the
values the repository's own fixtures carry. Reproducing them confirms the
CONVENTION is the one the rest of the torsion package expects -- but from a
geometry built by hand from published bond lengths and angles, not from
methanol's actual coordinates. F is a ratio whose denominator is a small
difference of larger moments, so it inherits several times the geometry's own
error; rho does not. Hence rho is held to 1% and F to 5%, and
``test_f_is_more_geometry_sensitive_than_rho`` measures that asymmetry rather
than leaving the looser tolerance looking arbitrary.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion.internal_rotation import (
    _INERTIA_TO_CM,
    dihedral_angle,
    internal_rotation_constants,
    rotational_constants_cm,
    top_atoms,
)

_M_H = 1.0078250
_M_D = 2.0141018
_M_C = 12.0
_M_O = 15.9949146


def _methanol(r_co=1.4246, r_oh=0.9451, r_ch=1.0936,
              a_hco=108.0, a_coh=108.5, m_methyl_h=_M_H):
    """Methanol from published internal coordinates, C at origin, O along +z."""
    a_hco, a_coh = np.deg2rad(a_hco), np.deg2rad(a_coh)
    rows = [np.zeros(3), np.array([0.0, 0.0, r_co])]
    for k in range(3):
        phi = np.deg2rad(120.0 * k)
        rows.append(np.array([r_ch * np.sin(a_hco) * np.cos(phi),
                              r_ch * np.sin(a_hco) * np.sin(phi),
                              r_ch * np.cos(a_hco)]))
    rows.append(rows[1] + np.array([r_oh * np.sin(a_coh), 0.0,
                                    r_oh * np.cos(a_coh)]))
    coords = np.vstack(rows)
    elements = ["C", "O", "H", "H", "H", "H"]
    masses = np.array([_M_C, _M_O, m_methyl_h, m_methyl_h, m_methyl_h, _M_H])
    return coords, elements, masses


def _methanol_constants(**kw):
    coords, elements, masses = _methanol(**kw)
    return internal_rotation_constants(coords, masses, frame_atom=1,
                                       top_atom=0, elements=elements)


# ── The closed-form checks, which are what pin the relations ─────────────────

class TestClosedForm:
    def test_i_alpha_is_the_methyl_sum_by_hand(self):
        # I_alpha = Sum m_i d_i^2 with d the PERPENDICULAR distance from the
        # axis line. The pivot carbon sits on the axis and must contribute
        # exactly nothing; the three hydrogens sit at r_CH sin(HCO).
        r_ch, a_hco = 1.0936, 108.0
        res = _methanol_constants(r_ch=r_ch, a_hco=a_hco)
        d = r_ch * np.sin(np.deg2rad(a_hco))
        assert res["I_alpha"] == pytest.approx(3.0 * _M_H * d ** 2, rel=1e-12)

    def test_r_collapses_to_one_minus_i_alpha_over_i_a(self):
        # Methanol's internal axis is within 1.5 degrees of its a axis, so
        # r = 1 - I_alpha/I_a holds to the square of that small cosine.
        res = _methanol_constants()
        I_a = res["principal_moments_amu_ang2"][0]
        assert abs(res["lambda_abc"][0]) > 0.999
        assert res["r"] == pytest.approx(1.0 - res["I_alpha"] / I_a, rel=2e-3)

    def test_f_is_the_reduced_moment_relation(self):
        res = _methanol_constants()
        assert res["F_cm"] == pytest.approx(_INERTIA_TO_CM / res["I_red"], rel=1e-12)
        assert res["I_red"] == pytest.approx(res["r"] * res["I_alpha"], rel=1e-12)

    def test_rho_is_lambda_times_i_alpha_over_i_g(self):
        res = _methanol_constants()
        I_g = res["principal_moments_amu_ang2"]
        expect = res["lambda_abc"] * res["I_alpha"] / I_g
        np.testing.assert_allclose(res["rho_abc"], expect, rtol=1e-12)

    def test_inertia_constant_is_the_familiar_number(self):
        # B[cm^-1] = 16.8576 / I[amu Ang^2]; a wrong constant would scale
        # every F and every fitted barrier with it.
        assert _INERTIA_TO_CM == pytest.approx(16.8576, rel=1e-4)

    def test_mass_on_the_axis_does_not_resist_recoil(self):
        # The obvious reading of "heavy frame" is wrong, and this is the test
        # that says so. Methanol's oxygen lies ON the C-O rotation axis, so it
        # has no moment about that axis and cannot resist the frame's recoil.
        # Making it 1000x heavier must leave r essentially untouched.
        coords, elements, masses = _methanol()
        base = internal_rotation_constants(coords, masses, 1, 0,
                                           elements=elements)
        heavy_axis = masses.copy()
        heavy_axis[1] *= 1000.0
        res = internal_rotation_constants(coords, heavy_axis, 1, 0,
                                          elements=elements)
        assert res["r"] == pytest.approx(base["r"], rel=0.05)

    def test_off_axis_frame_mass_drives_r_up_and_f_down(self):
        # Mass OFF the axis is what resists recoil. Raising the hydroxyl
        # hydrogen's mass must push r monotonically up and F monotonically
        # down towards the top's own constant, 16.8576/I_alpha.
        coords, elements, masses = _methanol()
        rs, fs = [], []
        for factor in (1.0, 2.0, 5.0, 20.0, 100.0, 1000.0):
            m = masses.copy()
            m[5] *= factor
            res = internal_rotation_constants(coords, m, 1, 0, elements=elements)
            rs.append(res["r"])
            fs.append(res["F_cm"])
        assert rs == sorted(rs), rs
        assert fs == sorted(fs, reverse=True), fs
        own = _INERTIA_TO_CM / internal_rotation_constants(
            coords, masses, 1, 0, elements=elements)["I_alpha"]
        assert fs[-1] > own                     # approached from above
        assert fs[-1] < 1.2 * own               # and got close to it


# ── The literature anchor ────────────────────────────────────────────────────

class TestMethanolLiteratureValues:
    """The values the repository's own fixtures carry: F = 27.6, rho = 0.81."""

    def test_rho_matches_to_one_percent(self):
        assert _methanol_constants()["rho"] == pytest.approx(0.81, rel=0.01)

    def test_f_matches_to_five_percent(self):
        assert _methanol_constants()["F_cm"] == pytest.approx(27.6, rel=0.05)

    def test_rotational_constants_are_in_the_right_place(self):
        # Fixture spec: A = 4.25, B = 0.823, C = 0.793 cm^-1. Loose, because
        # this is a hand-built geometry, but a factor-of-two or a transposed
        # ordering would show.
        coords, _, masses = _methanol()
        A, B, C = rotational_constants_cm(coords, masses)
        assert A > B > C
        assert A == pytest.approx(4.25, rel=0.03)
        assert B == pytest.approx(0.823, rel=0.07)
        assert C == pytest.approx(0.793, rel=0.07)

    def test_rho_is_scale_free_and_f_is_not(self):
        """Why rho is held to 1% and F to 5%, as an exact statement.

        rho_g = lambda_g I_alpha/I_g is a RATIO of moments, so a geometry that
        is uniformly the wrong size gives rho exactly right. F carries
        1/I_red, so the same error enters it twice over. A hand-built methanol
        about 1% too large therefore reproduces rho to all digits and F to
        about 2% -- which is the whole of the 26.85 against 27.6 discrepancy,
        and the reason the looser F tolerance is not a fudge.
        """
        coords, elements, masses = _methanol()
        base = internal_rotation_constants(coords, masses, 1, 0,
                                           elements=elements)
        s = 1.01
        scaled = internal_rotation_constants(coords * s, masses, 1, 0,
                                             elements=elements)
        np.testing.assert_allclose(scaled["rho_abc"], base["rho_abc"], rtol=1e-12)
        assert scaled["r"] == pytest.approx(base["r"], rel=1e-12)
        assert scaled["F_cm"] / base["F_cm"] == pytest.approx(1.0 / s ** 2, rel=1e-12)


# ── Invariance: F and rho are properties of the molecule, not of the frame ──

class TestInvariance:
    def test_translation_invariant(self):
        coords, elements, masses = _methanol()
        a = internal_rotation_constants(coords, masses, 1, 0, elements=elements)
        b = internal_rotation_constants(coords + np.array([3.0, -7.0, 11.0]),
                                        masses, 1, 0, elements=elements)
        assert b["F_cm"] == pytest.approx(a["F_cm"], rel=1e-10)
        assert b["rho"] == pytest.approx(a["rho"], rel=1e-10)

    def test_rotation_invariant(self):
        coords, elements, masses = _methanol()
        t = np.deg2rad(37.0)
        R = np.array([[np.cos(t), -np.sin(t), 0.0],
                      [np.sin(t), np.cos(t), 0.0],
                      [0.0, 0.0, 1.0]])
        a = internal_rotation_constants(coords, masses, 1, 0, elements=elements)
        b = internal_rotation_constants(coords @ R.T, masses, 1, 0,
                                        elements=elements)
        assert b["F_cm"] == pytest.approx(a["F_cm"], rel=1e-10)
        assert b["rho"] == pytest.approx(a["rho"], rel=1e-10)

    def test_either_group_may_be_called_the_top(self):
        """I_red is symmetric between the two groups, as it must be.

        I_red = I_alpha (I_a - I_alpha)/I_a is unchanged by
        I_alpha -> I_a - I_alpha, because the relative internal rotation of
        two groups is ONE coordinate. So naming the methyl the top or the
        hydroxyl the top has to give the same reduced moment, even though
        their I_alpha differ by a factor of four. Agreement is to the small
        off-axis terms, not exact.
        """
        coords, elements, masses = _methanol()
        methyl = internal_rotation_constants(coords, masses, 1, 0,
                                             elements=elements)
        hydroxyl = internal_rotation_constants(coords, masses, 0, 1,
                                               elements=elements)
        assert hydroxyl["top_indices"] == (1, 5)
        assert methyl["I_alpha"] > 3.0 * hydroxyl["I_alpha"]
        assert hydroxyl["I_red"] == pytest.approx(methyl["I_red"], rel=0.05)
        assert hydroxyl["F_cm"] == pytest.approx(methyl["F_cm"], rel=0.05)


class TestMassDependence:
    def test_deuterating_only_the_top_barely_moves_f(self):
        """I_alpha doubles exactly, and F moves by a tenth. Not a bug.

        I_red tends to the FRAME's contribution as the top gets heavy, so once
        the top already dominates -- methanol's methyl is 81% of I_a -- making
        it heavier buys little. The naive expectation that CD3 halves F is
        what the next test shows actually requires.
        """
        light = _methanol_constants()
        heavy = _methanol_constants(m_methyl_h=_M_D)
        assert heavy["I_alpha"] == pytest.approx(
            light["I_alpha"] * _M_D / _M_H, rel=1e-12)
        assert heavy["F_cm"] < light["F_cm"]
        assert heavy["F_cm"] / light["F_cm"] == pytest.approx(0.90, abs=0.03)

    def test_deuterating_both_groups_halves_f(self):
        # Every mass in I_red doubles, so I_red doubles and F halves. This is
        # where the remembered "CD3 halves F" belongs: the fully deuterated
        # species, not the methyl alone.
        coords, elements, masses = _methanol(m_methyl_h=_M_D)
        both = masses.copy()
        both[5] = _M_D
        light = _methanol_constants()
        res = internal_rotation_constants(coords, both, 1, 0, elements=elements)
        assert res["F_cm"] / light["F_cm"] == pytest.approx(0.5, abs=0.04)


# ── Top detection and its failure modes ─────────────────────────────────────

class TestTopAtoms:
    def test_finds_the_methyl(self):
        coords, elements, _ = _methanol()
        assert top_atoms(coords, elements, frame_atom=1, top_atom=0) == (0, 2, 3, 4)

    def test_finds_the_hydroxyl_from_the_other_end(self):
        coords, elements, _ = _methanol()
        assert top_atoms(coords, elements, frame_atom=0, top_atom=1) == (1, 5)

    def test_ring_bond_is_rejected(self):
        # Cyclopropane-like triangle: nothing rotates about a ring bond, and
        # the connectivity walk would otherwise return the whole ring as a
        # "top" with an I_alpha covering the molecule.
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0],
                           [-0.6, -0.8, 0.0]])
        elements = ["C", "C", "C", "H"]
        with pytest.raises(ValueError, match="ring"):
            top_atoms(coords, elements, frame_atom=0, top_atom=1)

    def test_same_atom_for_both_ends_is_rejected(self):
        coords, elements, _ = _methanol()
        with pytest.raises(ValueError, match="must be different"):
            top_atoms(coords, elements, frame_atom=0, top_atom=0)

    def test_a_terminal_bond_gives_a_top_without_complaint(self):
        # Cutting C-H leaves everything but that H on the carbon side. It is
        # physically meaningless to rotate a molecule about a terminal bond,
        # but connectivity cannot tell, so this is the r guard's job and not
        # top_atoms'. Recorded so the absence of an error here is deliberate.
        coords, elements, _ = _methanol()
        assert top_atoms(coords, elements, frame_atom=2, top_atom=0) == (0, 1, 3, 4, 5)

    def test_a_dominant_off_axis_mass_breaks_separability_and_raises(self):
        # The real pathology the r guard catches: an off-axis atom so heavy
        # that the "frame" is lighter than the "top". Internal and overall
        # rotation stop being separable and r leaves (0, 1] rather than
        # degrading gently.
        coords, elements, masses = _methanol()
        m = masses.copy()
        m[5] = 1.0e6
        with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
            internal_rotation_constants(coords, m, frame_atom=0, top_atom=1,
                                        elements=elements)

    def test_frame_atom_inside_the_top_is_rejected(self):
        coords, _, masses = _methanol()
        with pytest.raises(ValueError, match="must not be one of the top"):
            internal_rotation_constants(coords, masses, frame_atom=0,
                                        top_atom=2, top_indices=(0, 2, 3, 4))

    def test_bad_top_set_is_caught_by_the_r_guard(self):
        # Handing it the whole molecule makes the recoil term exceed 1, so r
        # leaves (0, 1]. Raised rather than returned: an F built on a bad r is
        # the wrong quantity, not an approximate one.
        coords, _, masses = _methanol()
        with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
            internal_rotation_constants(coords, masses, frame_atom=1,
                                        top_atom=0,
                                        top_indices=(0, 2, 3, 4, 5))

    def test_mismatched_masses_are_caught(self):
        coords, elements, _ = _methanol()
        with pytest.raises(ValueError, match="atom count"):
            internal_rotation_constants(coords, np.ones(3), 1, 0,
                                        elements=elements)


# ── dihedral_angle ───────────────────────────────────────────────────────────

class TestDihedralAngle:
    def test_planar_cis_is_zero(self):
        coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        assert dihedral_angle(coords, 0, 1, 2, 3) == pytest.approx(0.0, abs=1e-9)

    def test_planar_trans_is_one_eighty(self):
        coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
        assert abs(dihedral_angle(coords, 0, 1, 2, 3)) == pytest.approx(180.0, abs=1e-9)

    def test_ninety_degrees(self):
        coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        assert abs(dihedral_angle(coords, 0, 1, 2, 3)) == pytest.approx(90.0, abs=1e-9)

    def test_sign_reverses_with_the_atom_order(self):
        coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.0, 0.6, 0.8]])
        fwd = dihedral_angle(coords, 0, 1, 2, 3)
        rev = dihedral_angle(coords, 3, 2, 1, 0)
        assert fwd == pytest.approx(rev, abs=1e-9)        # dihedral is symmetric
        mirrored = coords.copy()
        mirrored[3, 2] *= -1.0
        assert dihedral_angle(mirrored, 0, 1, 2, 3) == pytest.approx(-fwd, abs=1e-9)

    def test_radians_option(self):
        coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        d = dihedral_angle(coords, 0, 1, 2, 3)
        r = dihedral_angle(coords, 0, 1, 2, 3, degrees=False)
        assert r == pytest.approx(np.deg2rad(d), rel=1e-12)

    def test_methyl_hydrogens_are_one_twenty_apart(self):
        # The physical check the scan generator will depend on: stepping a
        # methyl torsion by 120 degrees must return the same structure.
        coords, _, _ = _methanol()
        a = dihedral_angle(coords, 5, 1, 0, 2)
        b = dihedral_angle(coords, 5, 1, 0, 3)
        gap = (b - a) % 360.0
        assert gap == pytest.approx(120.0, abs=1e-6)

    def test_full_range_is_covered(self):
        # atan2, so the result spans (-180, 180] rather than folding at 0.
        seen = []
        for deg in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0):
            t = np.deg2rad(deg)
            coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [1.0, np.cos(t), np.sin(t)]])
            seen.append(dihedral_angle(coords, 0, 1, 2, 3))
        assert min(seen) < -100.0 and max(seen) > 100.0
        np.testing.assert_allclose(sorted(seen), sorted([-150.0, -90.0, -30.0,
                                                         30.0, 90.0, 150.0]),
                                   atol=1e-9)
