"""Tests for backend/torsion/torsion_scan_qm.py.

The scanner is driven by an ANALYTIC potential rather than a quantum backend,
so the constrained optimiser can be checked against a known answer: the
minimum, the barrier and the relaxed bond length at each angle are all things
the model potential fixes exactly. A test against a real PES could only
compare the scanner with itself.

The model is a methanol-like skeleton with

    V = V3/2 (1 - cos 3 phi) + k/2 (r_CO - r_eq(phi))^2

where r_eq(phi) itself depends on the torsion. That last term is the point: it
makes the frame genuinely want to relax as the top turns, so a rigid scan and
a relaxed scan give measurably different barriers and the test can tell which
one the code performed.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.torsion.internal_rotation import dihedral_angle
from backend.torsion.torsion_scan_qm import (
    _HARTREE_TO_CM,
    relaxed_torsion_scan,
    rotate_top,
    spec_from_scan,
)

_M_H, _M_C, _M_O = 1.0078250, 12.0, 15.9949146
_ELEMENTS = ["C", "O", "H", "H", "H", "H"]
_MASSES = np.array([_M_C, _M_O, _M_H, _M_H, _M_H, _M_H])
_DIHEDRAL = (5, 1, 0, 2)          # H_O - O - C - H_methyl


def _geometry(r_co=1.4246, r_oh=0.9451, r_ch=1.0936,
              a_hco=108.0, a_coh=108.5):
    a_hco, a_coh = np.deg2rad(a_hco), np.deg2rad(a_coh)
    rows = [np.zeros(3), np.array([0.0, 0.0, r_co])]
    for k in range(3):
        phi = np.deg2rad(120.0 * k)
        rows.append(np.array([r_ch * np.sin(a_hco) * np.cos(phi),
                              r_ch * np.sin(a_hco) * np.sin(phi),
                              r_ch * np.cos(a_hco)]))
    rows.append(rows[1] + np.array([r_oh * np.sin(a_coh), 0.0,
                                    r_oh * np.cos(a_coh)]))
    return np.vstack(rows)


class _ModelPES:
    """V = torsional term + a C-O stretch whose equilibrium moves with phi.

    Energies in Hartree, gradients in Hartree/Bohr, matching a backend's
    ``_energy_gradient``. The gradient is ANALYTIC, by the chain rule through
    the dihedral and the C-O length. A finite-differenced version worked
    identically but needed 6N energy evaluations per call and took the file
    from seconds to seven minutes, which is too slow to run often enough to
    be useful; ``test_gradient_matches_finite_difference`` keeps it honest.
    """

    def __init__(self, v3_cm=373.0, k=0.25, dr=0.02, couple=True):
        self.v3 = v3_cm / _HARTREE_TO_CM
        self.k = k
        self.dr = dr
        self.couple = couple
        self.calls = 0

    def r_eq(self, phi_deg):
        if not self.couple:
            return 1.4246
        # Longest at the barrier top (phi = 60 deg mod 120), shortest at the
        # minimum: the frame relaxing as the top turns. 1 - cos(3 phi) is zero
        # at phi = 0 and 2 at phi = 60, so no phase shift -- an earlier +180
        # here put the stretch maximum at the MINIMUM instead, which made the
        # rigid scan cheaper than the relaxed one and inverted the test.
        return 1.4246 + self.dr * 0.5 * (1.0 - np.cos(np.deg2rad(3.0 * phi_deg)))

    def energy(self, coords):
        coords = np.asarray(coords, dtype=float)
        phi = dihedral_angle(coords, *_DIHEDRAL)
        r_co = float(np.linalg.norm(coords[1] - coords[0]))
        tors = 0.5 * self.v3 * (1.0 - np.cos(np.deg2rad(3.0 * phi)))
        stretch = 0.5 * self.k * (r_co - self.r_eq(phi)) ** 2
        return float(tors + stretch)

    def __call__(self, coords):
        from backend.quantum import ANG_TO_BOHR, _dihedral_deriv
        coords = np.asarray(coords, dtype=float)
        self.calls += 1
        phi = dihedral_angle(coords, *_DIHEDRAL)
        bond = coords[1] - coords[0]
        r_co = float(np.linalg.norm(bond))
        req = self.r_eq(phi)

        # dV/dphi, in Hartree per RADIAN, then per degree where needed.
        t = np.deg2rad(3.0 * phi)
        dtors_dphi = 0.5 * self.v3 * 3.0 * np.sin(t)              # per rad(phi)*3
        dreq_dphi = (0.0 if not self.couple else
                     self.dr * 0.5 * 3.0 * np.sin(t))             # per rad
        dstretch_dphi = -self.k * (r_co - req) * dreq_dphi
        dV_dphi_rad = dtors_dphi + dstretch_dphi

        grad = np.zeros_like(coords)
        # Negated: _dihedral_deriv's sign convention is the opposite of
        # dihedral_angle's, which self.energy is written in terms of.
        dphi_dx = -_dihedral_deriv(coords, *_DIHEDRAL)   # per radian
        grad += dV_dphi_rad * dphi_dx

        dV_dr = self.k * (r_co - req)
        u = bond / r_co
        grad[1] += dV_dr * u
        grad[0] -= dV_dr * u
        return self.energy(coords), grad / ANG_TO_BOHR   # Hartree/Ang -> /Bohr

    def numeric_gradient(self, coords, h=1e-6):
        from backend.quantum import ANG_TO_BOHR
        coords = np.asarray(coords, dtype=float)
        grad = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                up = coords.copy(); up[i, j] += h
                dn = coords.copy(); dn[i, j] -= h
                grad[i, j] = (self.energy(up) - self.energy(dn)) / (2.0 * h)
        return grad / ANG_TO_BOHR


class TestSignConventions:
    """The two sign traps this module had to work around, pinned.

    Neither is wrong on its own; they disagree, and the disagreement is
    invisible until something consumes both. If a later change makes them
    agree, these tests fail and the negations in the module come out.
    """

    def test_dihedral_deriv_sign_convention(self):
        # backend.quantum._dihedral_deriv is the NEGATIVE of
        # d(dihedral_angle)/dx. Load bearing in the Wilson B-matrix code, so
        # it is not the one to change; the scanner negates it locally.
        from backend.quantum import _dihedral_deriv
        coords = rotate_top(_geometry(), (0, 2, 3, 4), 1, 0, 25.0)
        analytic = _dihedral_deriv(coords, *_DIHEDRAL)
        h, numeric = 1e-7, np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                up = coords.copy(); up[i, j] += h
                dn = coords.copy(); dn[i, j] -= h
                numeric[i, j] = (np.deg2rad(dihedral_angle(up, *_DIHEDRAL))
                                 - np.deg2rad(dihedral_angle(dn, *_DIHEDRAL))) / (2 * h)
        np.testing.assert_allclose(analytic, -numeric, rtol=1e-5, atol=1e-7)

    def test_scanner_measures_the_rotation_sign(self):
        from backend.torsion.torsion_scan_qm import _rotation_sign
        coords = _geometry()
        assert _rotation_sign(coords, (0, 2, 3, 4), _DIHEDRAL, 1, 0) == -1.0
        assert _rotation_sign(coords, (0, 2, 3, 4), _DIHEDRAL, 0, 1) == +1.0

    def test_rotation_sign_rejects_the_wrong_axis(self):
        # Probing an axis that is not the dihedral's central bond gives a
        # response that is not +/-1, and that is raised rather than rounded.
        from backend.torsion.torsion_scan_qm import _rotation_sign
        coords = _geometry()
        with pytest.raises(ValueError, match="central bond"):
            _rotation_sign(coords, (2,), _DIHEDRAL, 0, 2)


class TestModelPES:
    """The model is test scaffolding, so it gets checked too."""

    def test_gradient_matches_finite_difference(self):
        pes = _ModelPES()
        coords = rotate_top(_geometry(), (0, 2, 3, 4), 1, 0, 25.0)
        coords[1, 2] += 0.03          # off the stretch minimum too
        _, analytic = pes(coords)
        np.testing.assert_allclose(analytic, pes.numeric_gradient(coords),
                                   rtol=1e-5, atol=1e-9)

    def test_energy_is_three_fold_in_the_dihedral(self):
        pes = _ModelPES()
        base = _geometry()
        e = [pes.energy(rotate_top(base, (0, 2, 3, 4), 1, 0, d))
             for d in (0.0, 120.0, 240.0)]
        assert e[1] == pytest.approx(e[0], abs=1e-12)
        assert e[2] == pytest.approx(e[0], abs=1e-12)

    def test_uncoupled_model_has_no_frame_preference(self):
        pes = _ModelPES(couple=False)
        assert pes.r_eq(0.0) == pytest.approx(pes.r_eq(60.0), rel=1e-12)


# ── rotate_top ───────────────────────────────────────────────────────────────

class TestRotateTop:
    def test_turns_the_dihedral_by_the_requested_magnitude(self):
        # The MAGNITUDE is the requested one; the sign follows the axis
        # direction, which is the trap documented on rotate_top.
        coords = _geometry()
        before = dihedral_angle(coords, *_DIHEDRAL)
        for delta in (10.0, 37.0, 90.0, -37.0):
            turned = rotate_top(coords, (0, 2, 3, 4), 1, 0, delta)
            got = (dihedral_angle(turned, *_DIHEDRAL) - before + 180.0) % 360.0 - 180.0
            assert abs(got) == pytest.approx(abs(delta), abs=1e-8)

    def test_axis_direction_sets_the_sign(self):
        coords = _geometry()
        before = dihedral_angle(coords, *_DIHEDRAL)

        def moved(a, b):
            t = rotate_top(coords, (0, 2, 3, 4), a, b, 37.0)
            return (dihedral_angle(t, *_DIHEDRAL) - before + 180.0) % 360.0 - 180.0

        assert moved(1, 0) == pytest.approx(-37.0, abs=1e-8)
        assert moved(0, 1) == pytest.approx(+37.0, abs=1e-8)

    def test_pivot_atoms_do_not_move(self):
        coords = _geometry()
        turned = rotate_top(coords, (0, 2, 3, 4), 1, 0, 50.0)
        np.testing.assert_allclose(turned[0], coords[0], atol=1e-12)
        np.testing.assert_allclose(turned[1], coords[1], atol=1e-12)

    def test_frame_atoms_do_not_move(self):
        coords = _geometry()
        turned = rotate_top(coords, (0, 2, 3, 4), 1, 0, 50.0)
        np.testing.assert_allclose(turned[5], coords[5], atol=1e-12)

    def test_rigid_rotation_preserves_internal_distances(self):
        coords = _geometry()
        turned = rotate_top(coords, (0, 2, 3, 4), 1, 0, 73.0)
        for i, j in [(0, 2), (2, 3), (3, 4), (0, 1), (1, 5)]:
            assert np.linalg.norm(turned[i] - turned[j]) == pytest.approx(
                np.linalg.norm(coords[i] - coords[j]), rel=1e-12)

    def test_one_twenty_degrees_maps_the_methyl_onto_itself(self):
        coords = _geometry()
        turned = rotate_top(coords, (0, 2, 3, 4), 1, 0, 120.0)
        # The set of methyl positions is unchanged, atom labels permuted.
        a = np.sort(np.linalg.norm(coords[[2, 3, 4]] - coords[5], axis=1))
        b = np.sort(np.linalg.norm(turned[[2, 3, 4]] - coords[5], axis=1))
        np.testing.assert_allclose(a, b, atol=1e-10)


# ── The constrained scan ─────────────────────────────────────────────────────

class TestRelaxedScan:
    def test_every_point_lands_on_its_target_dihedral(self):
        # The constraint is the whole mechanism; if it slips, the scan is a
        # set of energies at unknown angles.
        pes = _ModelPES()
        scan, info = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                          _DIHEDRAL, n_points=6)
        assert info["all_converged"], info["per_point"]
        assert info["max_dihedral_error_deg"] < 0.05
        assert info["dihedral_within_tol"]

    def test_grid_covers_a_full_turn_without_duplicating_the_endpoint(self):
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                       _DIHEDRAL, n_points=6)
        offsets = [gp.phi for gp in scan.grid_points]
        assert offsets == pytest.approx([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        assert scan.periodic is True

    def test_recovers_the_three_fold_periodicity(self):
        # The model has V3 only, so energies at 0/120/240 must agree, and at
        # 60/180/300 must agree, and the two groups must differ.
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                       _DIHEDRAL, n_points=6)
        e = np.array([gp.energy for gp in scan.grid_points])
        np.testing.assert_allclose(e[[0, 2, 4]], e[0], atol=1e-9)
        np.testing.assert_allclose(e[[1, 3, 5]], e[1], atol=1e-9)
        assert e[1] > e[0]

    def test_relaxed_barrier_is_below_the_rigid_one(self):
        # The reason the default is relaxed. The model's C-O equilibrium
        # lengthens at the barrier top, so a rigid scan pays a stretch
        # penalty there that a relaxed scan does not.
        pes = _ModelPES()
        _, relaxed = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                          _DIHEDRAL, n_points=6, relax=True)
        _, rigid = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                        _DIHEDRAL, n_points=6, relax=False)
        assert relaxed["barrier_cm"] < rigid["barrier_cm"]
        # The relaxed scan recovers the model's true V3 exactly, because at
        # each angle it sits at the stretch minimum. The rigid scan carries
        # the stretch penalty it has to pay at the barrier on top.
        assert relaxed["barrier_cm"] == pytest.approx(373.0, rel=0.01)
        assert rigid["barrier_cm"] > 380.0

    def test_relaxation_actually_moves_the_frame(self):
        # And it moves it the way the model asks: C-O longest at the barrier.
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                       _DIHEDRAL, n_points=6, relax=True)
        r_co = [float(np.linalg.norm(gp.geometry[1] - gp.geometry[0]))
                for gp in scan.grid_points]
        assert max(r_co) - min(r_co) > 0.01
        assert r_co[1] > r_co[0]        # barrier longer than minimum

    def test_rigid_scan_leaves_the_frame_exactly_alone(self):
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                       _DIHEDRAL, n_points=6, relax=False)
        for gp in scan.grid_points:
            assert float(np.linalg.norm(gp.geometry[1] - gp.geometry[0])) \
                == pytest.approx(1.4246, rel=1e-12)
        assert scan.metadata["relaxed"] is False

    def test_uncoupled_model_gives_equal_barriers(self):
        # Control: with no frame coupling there is nothing to relax, so the
        # relaxed and rigid barriers must agree. Without this, the previous
        # test could be passing on an artefact of the optimiser.
        pes = _ModelPES(couple=False)
        _, relaxed = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                          _DIHEDRAL, n_points=6, relax=True)
        _, rigid = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                        _DIHEDRAL, n_points=6, relax=False)
        assert relaxed["barrier_cm"] == pytest.approx(rigid["barrier_cm"], rel=0.02)

    def test_partial_span_includes_the_endpoint(self):
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS, _DIHEDRAL,
                                       n_points=4, span_deg=120.0)
        offsets = [gp.phi for gp in scan.grid_points]
        assert offsets == pytest.approx([0.0, 40.0, 80.0, 120.0])
        assert scan.periodic is False

    def test_scan_records_the_axis_and_the_top(self):
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS,
                                       _DIHEDRAL, n_points=3)
        assert scan.atoms == _DIHEDRAL
        assert scan.metadata["top_indices"] == (0, 2, 3, 4)
        assert scan.energy_unit == "hartree"

    def test_explicit_top_indices_are_honoured(self):
        pes = _ModelPES()
        scan, _ = relaxed_torsion_scan(pes, _geometry(), _ELEMENTS, _DIHEDRAL,
                                       n_points=3, top_indices=(0, 2, 3, 4))
        assert scan.metadata["top_indices"] == (0, 2, 3, 4)

    def test_constraint_survives_crossing_the_branch_cut(self):
        # Targets run past +180 degrees, where an unwrapped residual would
        # send SLSQP the long way round. Covered by the full-turn scans above
        # but asserted directly, since it is the subtle failure.
        pes = _ModelPES()
        coords = _geometry()
        phi0 = dihedral_angle(coords, *_DIHEDRAL)
        scan, info = relaxed_torsion_scan(pes, coords, _ELEMENTS, _DIHEDRAL,
                                          n_points=12)
        assert info["max_dihedral_error_deg"] < 0.05
        achieved = [dihedral_angle(gp.geometry, *_DIHEDRAL)
                    for gp in scan.grid_points]
        for gp, got in zip(scan.grid_points, achieved):
            want = (phi0 + gp.phi + 180.0) % 360.0 - 180.0
            assert ((got - want + 180.0) % 360.0 - 180.0) == pytest.approx(0.0, abs=0.05)


# ── spec_from_scan ───────────────────────────────────────────────────────────

class TestSpecFromScan:
    def _scan(self, **kw):
        pes = _ModelPES(**kw)
        coords = _geometry()
        scan, _ = relaxed_torsion_scan(pes, coords, _ELEMENTS, _DIHEDRAL,
                                       n_points=12)
        return coords, scan

    def test_fills_f_and_rho_from_the_geometry(self):
        coords, scan = self._scan()
        spec, info = spec_from_scan(coords, _MASSES, _ELEMENTS, scan)
        # The same values test_internal_rotation pins against the literature.
        assert spec.F == pytest.approx(26.85, rel=0.02)
        assert spec.rho == pytest.approx(0.808, rel=0.02)
        assert info["internal_rotation"]["top_indices"] == (0, 2, 3, 4)

    def test_fills_abc_in_descending_order(self):
        coords, scan = self._scan()
        spec, _ = spec_from_scan(coords, _MASSES, _ELEMENTS, scan)
        assert spec.A > spec.B > spec.C
        assert spec.A == pytest.approx(4.17, rel=0.02)

    def test_recovers_the_model_barrier(self):
        coords, scan = self._scan(v3_cm=373.0)
        _, info = spec_from_scan(coords, _MASSES, _ELEMENTS, scan)
        assert info["barrier_cm"] == pytest.approx(373.0, rel=0.03)

    def test_potential_is_three_fold(self):
        coords, scan = self._scan()
        spec, info = spec_from_scan(coords, _MASSES, _ELEMENTS, scan,
                                    symmetry_number=3)
        # Only harmonics that are multiples of 3 may be populated, and the
        # model has V3 alone, so order 3 must carry essentially all of it.
        assert set(spec.potential.vcos) <= {3, 6}
        v3 = abs(spec.potential.vcos.get(3, 0.0))
        v6 = abs(spec.potential.vcos.get(6, 0.0))
        assert v3 > 150.0
        assert v6 < 0.01 * v3
        assert info["n_harmonics_used"] == 2

    def test_harmonics_are_clamped_to_what_the_grid_resolves(self):
        """A 12-point scan cannot carry order 9, so it is not asked to.

        Before the clamp, requesting orders 3/6/9 from a 12-point grid
        returned V3 = V9 = -93.25: one real V3 split in half and half of it
        mislabelled as a ninth harmonic, with no warning. Order 9 needs 18
        points to be distinguishable from order 3.
        """
        coords, scan = self._scan()
        _, info = spec_from_scan(coords, _MASSES, _ELEMENTS, scan,
                                 symmetry_number=3, n_harmonics=3)
        assert len(scan.grid_points) == 12
        assert info["n_harmonics_requested"] == 3
        assert info["n_harmonics_max_for_grid"] == 2
        assert info["n_harmonics_used"] == 2
        assert info["harmonics_clamped"] is True

    def test_a_finer_grid_allows_more_harmonics(self):
        pes = _ModelPES()
        coords = _geometry()
        scan, _ = relaxed_torsion_scan(pes, coords, _ELEMENTS, _DIHEDRAL,
                                       n_points=24)
        _, info = spec_from_scan(coords, _MASSES, _ELEMENTS, scan,
                                 symmetry_number=3, n_harmonics=3)
        assert info["n_harmonics_used"] == 3
        assert info["harmonics_clamped"] is False

    def test_scales_with_the_model_barrier(self):
        coords_a, scan_a = self._scan(v3_cm=200.0)
        coords_b, scan_b = self._scan(v3_cm=600.0)
        _, info_a = spec_from_scan(coords_a, _MASSES, _ELEMENTS, scan_a)
        _, info_b = spec_from_scan(coords_b, _MASSES, _ELEMENTS, scan_b)
        assert info_b["barrier_cm"] / info_a["barrier_cm"] == pytest.approx(3.0, rel=0.05)

    def test_deuterated_top_changes_f_not_the_potential(self):
        # The potential is electronic and mass-independent; F is not. A spec
        # builder that shared F across isotopologues would be wrong, and this
        # is the assertion that would catch it.
        coords, scan = self._scan()
        light, _ = spec_from_scan(coords, _MASSES, _ELEMENTS, scan)
        heavy_masses = _MASSES.copy()
        heavy_masses[[2, 3, 4]] = 2.0141018
        heavy, _ = spec_from_scan(coords, heavy_masses, _ELEMENTS, scan)
        assert heavy.F < light.F
        assert heavy.potential.vcos == light.potential.vcos

    def test_rejects_an_unsupported_energy_unit(self):
        coords, scan = self._scan()
        scan.energy_unit = "kJ/mol"
        with pytest.raises(ValueError, match="energy_unit"):
            spec_from_scan(coords, _MASSES, _ELEMENTS, scan)

    def test_needs_axis_atoms_when_the_scan_carries_none(self):
        coords, scan = self._scan()
        scan.atoms = None
        with pytest.raises(ValueError, match="frame_atom"):
            spec_from_scan(coords, _MASSES, _ELEMENTS, scan)
