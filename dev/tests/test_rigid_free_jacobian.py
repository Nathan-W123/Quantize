"""Cartesian steps never chase a finite-difference Jacobian's rigid-motion leak.

Near degenerate moments (B ~ C: a near-linear molecule) the rotational
constants' Jacobian gives way to central finite differences, which displace one
atom at a time and so break translation and rotation invariance slightly
(fluoroacetylene, HF/6-31G: 1.5e-5 of |J|). The joint matrix holds rigid motion
only by its 1e-8 eigenvalue floor and lambda_damp, so the leak asked for a slide
of the whole molecule metres long, the trust radius was spent on it, and the fit
crawled (57 iterations on 6e-3 A steps that were 95% rigid, stopping 2.6 mA
from its optimum). The Cartesian loop now removes the Jacobian's rigid part,
which the true Jacobian does not have (decisions D-94).

The fits here run on a three-atom valence force field, not quantum chemistry.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

import backend.quantize as Q  # noqa: E402
import backend.spectral.spectral as SP  # noqa: E402
from backend.base_backend import GradientResult, HessianResult, QuantumBackend  # noqa: E402
from backend.registry import register_backend  # noqa: E402
from backend.spectral.SVD import SubspaceOptimizer  # noqa: E402

_BOHR = 0.529177210903
# HCN-like stretches and bend (Hartree/A^2, Hartree/A^2, Hartree), minimum
# away from the geometry the data come from so the two halves disagree.
_K_CH, _K_CN, _K_B = 1.42, 4.29, 0.046
_R_CH, _R_CN = 1.060, 1.145
_TRUE = np.array([[0.0, 0.0, -1.0655], [0.0, 0.0, 0.0], [0.0, 0.0, 1.1532]])
_M = dict(H=1.00782503207, D=2.01410177785, C=12.0, C13=13.00335483507,
          N=14.0030740048, N15=15.0001088982)


def _energy(x):
    x = np.asarray(x, float).reshape(3, 3)
    a, b = x[0] - x[1], x[2] - x[1]
    ra, rb = np.linalg.norm(a), np.linalg.norm(b)
    return (0.5 * _K_CH * (ra - _R_CH) ** 2 + 0.5 * _K_CN * (rb - _R_CN) ** 2
            + _K_B * (1.0 + a @ b / (ra * rb)))


def _grad(x, h=1e-5):
    f = np.asarray(x, float).ravel()
    g = np.zeros(f.size)
    for i in range(f.size):
        p, m = f.copy(), f.copy()
        p[i] += h
        m[i] -= h
        g[i] = (_energy(p) - _energy(m)) / (2 * h)
    return g


def _hess(x, h=1e-4):
    f = np.asarray(x, float).ravel()
    H = np.zeros((f.size, f.size))
    for i in range(f.size):
        p, m = f.copy(), f.copy()
        p[i] += h
        m[i] -= h
        H[:, i] = (_grad(p) - _grad(m)) / (2 * h)
    return 0.5 * (H + H.T)


@register_backend
class _LinearTriatomic(QuantumBackend):
    name = "test_linear_triatomic"
    supports_parallel = True

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])

    def run_hessian(self, coords_ang):
        return HessianResult(energy=_energy(coords_ang), gradient_bohr=_grad(coords_ang) * _BOHR,
                             hessian_bohr=_hess(coords_ang) * _BOHR ** 2)

    def run_gradient(self, coords_ang):
        return GradientResult(energy=_energy(coords_ang), gradient_bohr=_grad(coords_ang) * _BOHR)


def _isotopologues():
    out = []
    for name, m in (("HCN", ("H", "C", "N")), ("DCN", ("D", "C", "N")),
                    ("H13CN", ("H", "C13", "N")), ("HC15N", ("H", "C", "N15"))):
        masses = [_M[k] for k in m]
        b = float(SP._rotational_constants(_TRUE, np.array(masses))[1])
        out.append(dict(name=name, masses=masses, obs_constants=[b], sigma_constants=[0.5],
                        alpha_constants=[0.0], component_indices=[1]))
    return out


def _fit(seed, monkeypatch, fixed=True):
    """A Cartesian joint fit from a noisy start; returns the optimizer, the
    rigid fraction of each solver step, and the finite-difference call count."""
    if not fixed:
        monkeypatch.setattr(Q.MolecularOptimizer, "_rigid_free_jacobian", lambda self, J: J)
    calls = {"fd": 0}
    fd = SP._jacobian_full

    def counted(*a, **k):
        calls["fd"] += 1
        return fd(*a, **k)

    monkeypatch.setattr(SP, "_jacobian_full", counted)
    start = _TRUE + np.random.default_rng(seed).normal(0.0, 0.02, _TRUE.shape)
    opt = Q.MolecularOptimizer(elems=["H", "C", "N"], coords=start, isotopologues=_isotopologues(),
                               quantum_backend="test_linear_triatomic", use_autoconfig=False,
                               max_iter=80, hess_recalc_every=10, coordinate_mode="cartesian")
    rigid = []
    joint_step = SubspaceOptimizer._joint_step

    def watched(self, J, r, g, H, a, prior_displacement=None, **kw):
        dp = joint_step(self, J, r, g, H, a, prior_displacement=prior_displacement, **kw)
        P = opt._rigid_mode_projector(np.asarray(opt.coords, float), opt._rigid_ref_masses)
        rigid.append(float(np.linalg.norm(dp - P @ dp) / max(np.linalg.norm(dp), 1e-300)))
        return dp

    monkeypatch.setattr(SubspaceOptimizer, "_joint_step", watched)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt, rigid, calls["fd"]


@pytest.mark.parametrize("seed", [3, 4, 6])
def test_a_near_linear_fit_steps_without_rigid_motion(seed, monkeypatch):
    opt, rigid, n_fd = _fit(seed, monkeypatch)
    assert n_fd > 0, "premise: the fit reached the finite-difference Jacobian"
    # Measured: 1e-9 at most (the solve's rounding).
    assert max(rigid) <= 1e-6, rigid
    h = opt.history
    # Measured (seeds 3-6): 3-5 iterations, at most one rejected, last step
    # <= 1.6e-6 A.
    assert sum(not r["accepted"] for r in h) <= 1
    assert len(h) <= 8 and h[-1]["step_norm"] <= 1e-5, [r["step_norm"] for r in h]


@pytest.mark.parametrize("seed", [3, 4, 6])
def test_without_the_projection_the_steps_are_rigid(seed, monkeypatch):
    # The premise of the test above: the same fits with the Jacobian's leak
    # left in. Measured: steps up to 100% rigid, 6-9 rejections, 11-28
    # iterations, the trust radius down to 3.9e-4 - 3.1e-3 A.
    opt, rigid, n_fd = _fit(seed, monkeypatch, fixed=False)
    assert n_fd > 0
    assert max(rigid) > 0.5, rigid
    assert sum(not r["accepted"] for r in opt.history) >= 5


def _water_rows():
    from test_cd_rows_sign import _water_cd
    opt = _water_cd()
    J, _ = opt.spectral.stacked(opt.coords)
    n = sum(len(np.ravel(i["component_indices"])) for i in opt.spectral.isotopologues)
    assert J.shape[0] > n, "premise: the fixture has distortion rows"
    return opt, J[:n], J[n:]


def test_the_projection_changes_nothing_on_an_analytic_jacobian():
    opt, J, _ = _water_rows()
    np.testing.assert_allclose(opt._rigid_free_jacobian(J), J, rtol=0, atol=1e-12 * np.abs(J).max())


def test_it_removes_a_finite_difference_leak_and_follows_the_flag():
    opt, _, J_cd = _water_rows()
    x = np.asarray(opt.coords, float)
    m = np.asarray(opt._rigid_ref_masses, float)
    P = opt._rigid_mode_projector(x, m)
    # The rotational constants by finite differences, and the distortion
    # rows (always finite differences). Measured: 6e-6 of |J| for the latter.
    for J in (SP._jacobian_full(x, m, 1e-3), J_cd):
        assert np.abs(J - J @ P).max() > 1e-9 * np.abs(J).max(), "premise: a leak"
        JP = opt._rigid_free_jacobian(J)
        assert np.abs(JP - JP @ P).max() <= 1e-12 * np.abs(J).max()
    opt.project_rigid_modes = False
    assert opt._rigid_free_jacobian(J_cd) is J_cd


def test_a_conformer_mixture_keeps_its_rigid_rows():
    # A mixture's conformers are the base plus offsets fixed in the lab
    # frame, so turning the base changes its constants: its Jacobian has a
    # real rigid part, which the step must keep.
    import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)
    from test_cd_rows_sign import _isos, _x0
    x0 = _x0()
    th = 0.3
    rot = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(
            elems=["O", "H", "H"], coords=x0, isotopologues=_isos(None), use_autoconfig=False,
            max_iter=5, auto_systematic_sigma=False, coordinate_mode="cartesian", symmetry=None,
            spectral_only=True, quantum_backend="none", use_conformer_mixture=True,
            conformer_defs=[{"name": "a", "coords": x0, "weight": 0.6},
                            {"name": "b", "coords": _x0(r=0.97, deg=100.0) @ rot.T, "weight": 0.4}])
    J, _ = opt.spectral.stacked(opt.coords)
    P = opt._rigid_mode_projector(np.asarray(opt.coords, float), opt._rigid_ref_masses)
    # Measured: 0.11 of |J|.
    assert np.linalg.norm(J - J @ P) > 1e-2 * np.linalg.norm(J), "premise: rows depend on orientation"
    assert opt._rigid_free_jacobian(J) is J


def test_the_degrees_of_freedom_count_the_same_jacobian(monkeypatch):
    # chi-square's effective degrees of freedom feed the rescaling gate; they
    # must be counted on the Jacobian the step solves with. Measured on the
    # fits above the leak moves them by ~1e-5 (fluoroacetylene: 4.8936
    # against 4.8958), too little to test by value, so the call is checked.
    opt, _, _ = _fit(3, monkeypatch)
    seen = []
    real = Q.MolecularOptimizer._rigid_free_jacobian

    def spy(self, J):
        seen.append(np.asarray(J).shape)
        return real(self, J)

    monkeypatch.setattr(Q.MolecularOptimizer, "_rigid_free_jacobian", spy)
    opt._chi2_dof()
    assert seen and seen[0][1] == 9
