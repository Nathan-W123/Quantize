"""Tests for backend/kraitchman.py — rs substitution structures and inertial defects."""

from __future__ import annotations

import numpy as np
import pytest

from backend.kraitchman import (
    COSTAIN_K,
    compare_rs_to_geometry,
    inertial_defect,
    kraitchman_analysis,
    kraitchman_single_substitution,
    moments_from_constants,
    planar_moments,
    principal_axis_coords,
)
from backend.spectral import _rotational_constants

_M_S32, _M_S34 = 31.9720711744, 33.967867004
_M_O16, _M_O18 = 15.99491461956, 17.99915961286
_M_C12, _M_C13 = 12.0, 13.00335483778
_M_H, _M_D = 1.00782503207, 2.01410177785


def _so2_coords(r=1.4308, theta_deg=119.33):
    th = np.deg2rad(theta_deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _iso(name, masses, coords):
    m = np.asarray(masses, dtype=float)
    return {"name": name, "masses": m, "obs_constants": _rotational_constants(coords, m)}


class TestPlanarMoments:
    def test_planar_molecule_has_zero_Pc(self):
        coords = _so2_coords()
        m = np.array([_M_S32, _M_O16, _M_O16])
        P = planar_moments(_rotational_constants(coords, m))
        assert abs(P[2]) < 1e-8

    def test_sum_rule(self):
        """P_a + P_b = I_c for any molecule."""
        coords = _so2_coords()
        m = np.array([_M_S32, _M_O16, _M_O16])
        c = _rotational_constants(coords, m)
        P = planar_moments(c)
        I = moments_from_constants(c)
        assert P[0] + P[1] == pytest.approx(I[2], rel=1e-12)

    def test_rejects_nonpositive_constants(self):
        with pytest.raises(ValueError):
            planar_moments([100.0, -5.0, 3.0])


class TestInertialDefect:
    def test_zero_for_rigid_planar(self):
        coords = _so2_coords()
        m = np.array([_M_S32, _M_O16, _M_O16])
        assert inertial_defect(_rotational_constants(coords, m)) == pytest.approx(0.0, abs=1e-8)

    def test_negative_for_nonplanar(self):
        # methane-like tetrahedron: out-of-plane mass makes Ic < Ia + Ib
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ])
        m = np.array([_M_C12, _M_H, _M_H, _M_H, _M_D])
        assert inertial_defect(_rotational_constants(coords, m)) < -0.1


class TestKraitchmanExactRecovery:
    """Kraitchman equations are exact for rigid synthetic data."""

    def test_on_axis_atom(self):
        coords = _so2_coords()
        pm = np.array([_M_S32, _M_O16, _M_O16])
        sm = np.array([_M_S34, _M_O16, _M_O16])
        res = kraitchman_single_substitution(
            _rotational_constants(coords, pm),
            _rotational_constants(coords, sm),
            pm.sum(), _M_S34 - _M_S32, atom_index=0,
        )
        true_abs = np.abs(principal_axis_coords(coords, pm))[0]
        np.testing.assert_allclose(res.coords_abs, true_abs, atol=1e-6)

    def test_off_axis_atom(self):
        coords = _so2_coords()
        pm = np.array([_M_S32, _M_O16, _M_O16])
        sm = np.array([_M_S32, _M_O18, _M_O16])
        res = kraitchman_single_substitution(
            _rotational_constants(coords, pm),
            _rotational_constants(coords, sm),
            pm.sum(), _M_O18 - _M_O16, atom_index=1,
        )
        true_abs = np.abs(principal_axis_coords(coords, pm))[1]
        np.testing.assert_allclose(res.coords_abs, true_abs, atol=1e-9)

    def test_fully_asymmetric_planar_molecule(self):
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.36, 0.1, 0.0],
            [2.05, 0.95, 0.0],
            [-0.55, 0.87, 0.0],
        ])
        m = np.array([_M_O16, _M_C12, _M_H, _M_H])
        subs = [(1, _M_C13), (2, _M_D), (3, _M_D), (0, _M_O18)]
        isos = [_iso("parent", m, coords)]
        for idx, newm in subs:
            mm = m.copy()
            mm[idx] = newm
            isos.append(_iso(f"sub{idx}", mm, coords))
        out = kraitchman_analysis(isos)
        assert len(out["rows"]) == 4
        cmp_rows = compare_rs_to_geometry(out["rows"], coords, m)
        for row in cmp_rows:
            for ax in ("a", "b", "c"):
                assert abs(row[f"delta_{ax}_angstrom"]) < 1e-6

    def test_costain_sigma(self):
        coords = _so2_coords()
        pm = np.array([_M_S32, _M_O16, _M_O16])
        sm = np.array([_M_S32, _M_O18, _M_O16])
        res = kraitchman_single_substitution(
            _rotational_constants(coords, pm),
            _rotational_constants(coords, sm),
            pm.sum(), _M_O18 - _M_O16,
        )
        # off-axis coordinate: sigma = K/|z|
        assert res.sigma_costain[0] == pytest.approx(COSTAIN_K / res.coords_abs[0], rel=1e-9)

    def test_near_axis_negative_square_clamped(self):
        """Atom exactly on the a-axis gives ~0 (possibly tiny negative) b,c squares."""
        coords = _so2_coords()
        pm = np.array([_M_S32, _M_O16, _M_O16])
        sm = np.array([_M_S34, _M_O16, _M_O16])
        res = kraitchman_single_substitution(
            _rotational_constants(coords, pm),
            _rotational_constants(coords, sm),
            pm.sum(), _M_S34 - _M_S32,
        )
        # S sits on the b axis (C2 axis): a and c must clamp to ~0, no crash
        assert res.coords_abs[0] < 1e-4
        assert res.coords_abs[2] < 1e-4


class TestKraitchmanAnalysisFiltering:
    def test_multi_substitution_skipped(self):
        coords = _so2_coords()
        pm = np.array([_M_S32, _M_O16, _M_O16])
        mm = np.array([_M_S34, _M_O18, _M_O16])  # double substitution
        out = kraitchman_analysis([_iso("parent", pm, coords), _iso("double", mm, coords)])
        assert out["rows"] == []
        assert "double" in out["skipped"]

    def test_zero_delta_mass_rejected(self):
        with pytest.raises(ValueError):
            kraitchman_single_substitution([100.0, 90.0, 50.0], [100.0, 90.0, 50.0], 50.0, 0.0)

    def test_empty_input(self):
        out = kraitchman_analysis([])
        assert out["rows"] == [] and out["defects"] == []


class TestRunIntegration:
    def test_kraitchman_run_analysis_snapshot_format(self):
        from runner.usability import kraitchman_run_analysis
        coords = _so2_coords()
        pm = [_M_S32, _M_O16, _M_O16]
        sm = [_M_S34, _M_O16, _M_O16]
        snap = []
        for name, m in [("parent", pm), ("34S", sm)]:
            snap.append({
                "name": name,
                "masses": m,
                "obs_constants": _rotational_constants(coords, np.asarray(m)).tolist(),
                "component_indices": [0, 1, 2],
            })
        out = kraitchman_run_analysis(coords, snap)
        assert out is not None
        assert len(out["rows"]) == 1
        assert len(out["comparison"]) == 1
        assert abs(out["comparison"][0]["delta_b_angstrom"]) < 1e-5

    def test_returns_none_for_partial_components(self):
        from runner.usability import kraitchman_run_analysis
        coords = _so2_coords()
        snap = [{
            "name": "parent",
            "masses": [_M_S32, _M_O16, _M_O16],
            "obs_constants": [10000.0, 9000.0],
            "component_indices": [1, 2],
        }]
        assert kraitchman_run_analysis(coords, snap) is None
