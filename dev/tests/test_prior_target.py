"""Centring the quantum prior somewhere other than the level of theory's minimum.

The joint step solves (J^T J + alpha_q H + lambda I) dp = J^T r - alpha_q g,
and nothing in it mentions the geometry the caller passed in. g is the level of
theory's own gradient, so the prior's minimum is the level of theory's own
minimum -- handing the optimiser a bias-corrected starting structure moves the
starting point and nothing else.

That is not a subtle effect. Measured on acetyl fluoride with the spectral data
suppressed a thousandfold, so only the prior could act: started from a 6.57 mA
bias-corrected structure and relaxed to 15.03 mA, against pure theory's 15.75.
It walked back to the level of theory's answer with nothing pulling it there
but the prior itself.

prior_target_coords replaces g with H (x - x_target), which is the gradient of
a Gaussian centred on the target with the quantum surface's curvature. These
tests pin the substitution rather than the benchmark outcome: the outcome moves
with the level of theory, the algebra should not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.quantize import _kabsch_align  # noqa: E402


def test_kabsch_recovers_a_known_rotation():
    """A rotated copy must come back exactly, or the prior would penalise the
    optimiser's own rigid motion as though it were a structural change."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(6, 3))
    t = 0.9
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    moved = (rot @ ref.T).T + np.array([1.5, -2.0, 0.25])
    assert _kabsch_align(moved, ref) == pytest.approx(ref, abs=1e-10)


def test_kabsch_never_reflects():
    """A determinant fix is required: without it the SVD can return an
    improper rotation, which superimposes a mirror image and reports a
    perfect fit for the wrong structure."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(8, 3))
    mirrored = ref * np.array([1.0, 1.0, -1.0])
    out = _kabsch_align(mirrored, ref)
    assert not np.allclose(out, ref, atol=1e-6), (
        "a mirror image was superimposed on the original"
    )


def test_kabsch_leaves_internal_structure_alone():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(5, 3))
    mobile = rng.normal(size=(5, 3))
    out = _kabsch_align(mobile, ref)
    for i in range(5):
        for j in range(i + 1, 5):
            assert np.linalg.norm(out[i] - out[j]) == pytest.approx(
                np.linalg.norm(mobile[i] - mobile[j]), rel=1e-12)


class _Stub:
    """Just enough of MolecularOptimizer to exercise the substitution."""

    from backend.quantize import MolecularOptimizer as _MO
    _prior_gradient = _MO._prior_gradient

    def __init__(self, coords, target, mode="cartesian"):
        self.coords = np.asarray(coords, dtype=float)
        self._prior_target_coords = (None if target is None
                                     else np.asarray(target, dtype=float))
        self.coordinate_mode = mode


def _h(n):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3 * n, 3 * n))
    return a @ a.T + np.eye(3 * n)


def test_no_target_leaves_the_gradient_untouched():
    """Default behaviour must be bit-identical, or every existing result moves."""
    x = np.arange(9, dtype=float).reshape(3, 3)
    g = np.arange(9, dtype=float)
    out = _Stub(x, None)._prior_gradient(g, _h(3))
    assert out is g


def test_internal_mode_is_skipped_rather_than_applied_in_the_wrong_basis():
    """In internal mode the gradient is already projected into q-space, so a
    Cartesian H (x - x_target) would be nonsense of the right shape."""
    x = np.arange(9, dtype=float).reshape(3, 3)
    g = np.arange(9, dtype=float)
    stub = _Stub(x, x + 0.1, mode="internal")
    assert stub._prior_gradient(g, _h(3)) is g


def test_gradient_vanishes_at_the_target():
    """The defining property: the prior's minimum is the target."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 3))
    stub = _Stub(x, x.copy())
    out = stub._prior_gradient(np.ones(12), _h(4))
    assert np.max(np.abs(out)) < 1e-9


def _strip_rigid(disp, coords):
    """Remove translation and infinitesimal rotation about the centroid.

    What is left is a displacement Kabsch alignment cannot absorb, which is the
    only kind the prior should see.

    The six rigid vectors have to be orthonormalised as a set, not normalised
    one at a time: the three rotation generators are mutually orthogonal only
    when the inertia tensor is isotropic, and projecting with the raw ones
    leaves about 9% of the rotation behind -- measured, after a first version
    of this helper did exactly that.
    """
    n = np.asarray(coords, dtype=float).shape[0]
    r = np.asarray(coords, dtype=float) - np.asarray(coords, dtype=float).mean(axis=0)
    rigid = []
    for axis in np.eye(3):
        t = np.tile(axis, (n, 1))
        rigid.append(t.ravel())
        rigid.append(np.cross(axis, r).ravel())
    basis, _ = np.linalg.qr(np.asarray(rigid).T)
    flat = np.asarray(disp, dtype=float).ravel()
    flat = flat - basis @ (basis.T @ flat)
    return flat.reshape(-1, 3)


def test_gradient_is_the_hessian_times_a_purely_internal_displacement():
    """H (x - x_target) exactly, once the part alignment would remove is gone.

    A naive version of this test asserted it for an arbitrary displacement and
    failed, which is the alignment doing its job: a random displacement carries
    translation and rotation, and those are not structural changes.
    """
    rng = np.random.default_rng(5)
    target = rng.normal(size=(4, 3))
    disp = _strip_rigid(rng.normal(size=(4, 3)) * 0.01, target)
    hess = _h(4)
    stub = _Stub(target + disp, target)
    assert stub._prior_gradient(np.zeros(12), hess) == pytest.approx(
        hess @ disp.ravel(), rel=1e-6, abs=1e-9)


def test_the_rigid_part_of_a_displacement_is_not_penalised():
    """Same structure, shoved sideways: no prior force at all."""
    rng = np.random.default_rng(7)
    target = rng.normal(size=(4, 3))
    stub = _Stub(target + np.array([0.3, -0.2, 0.7]), target)
    out = stub._prior_gradient(np.zeros(12), _h(4))
    assert np.max(np.abs(out)) < 1e-8


def test_a_rigidly_moved_target_still_gives_a_vanishing_gradient():
    """The alignment earning its place: the same structure in another frame is
    not a displacement, and penalising it would fight the optimiser."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(5, 3))
    t = 0.6
    rot = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(t), -np.sin(t)],
                    [0.0, np.sin(t), np.cos(t)]])
    target = (rot @ x.T).T + np.array([3.0, 1.0, -2.0])
    out = _Stub(x, target)._prior_gradient(np.zeros(15), _h(5))
    assert np.max(np.abs(out)) < 1e-8


def test_a_mismatched_target_is_ignored_not_broadcast():
    x = np.zeros((4, 3))
    stub = _Stub(x, np.zeros((3, 3)))
    g = np.ones(12)
    assert stub._prior_gradient(g, _h(4)) is g


# ── isotropic prior: holding the coordinates the data cannot see ───────────

def _iso_opt(sigma_x):
    from backend.spectral.SVD import SubspaceOptimizer
    return SubspaceOptimizer(1e-3, 0.0, 0.1, None, 0.0, objective_mode="joint",
                             alpha_quantum=1.0, quantum_prior_sigma_ang=sigma_x,
                             isotropic_prior_sigma_ang=sigma_x)


def test_isotropic_prior_is_off_unless_asked_for():
    """Every existing result must be reproducible, so this cannot default on."""
    from backend.spectral.SVD import SubspaceOptimizer
    assert SubspaceOptimizer(1e-3, 0.0, 0.1, None, 1e-4,
                             objective_mode="joint").isotropic_prior_sigma_ang is None


def test_a_direction_the_data_cannot_see_is_held_at_the_target():
    """The whole point.

    J sees coordinate 0 strongly and coordinate 1 not at all. Starting
    displaced in both, the step must return the blind coordinate to the target
    rather than let it absorb residual. Without the isotropic term the blind
    direction is governed by alpha_q H alone, which on a real molecule was
    measured losing acetyl fluoride's C1-C2 stretch to -18.49 mA while its
    other four bonds sat at 2.46.
    """
    sigma_x = 0.006
    n = 2
    J = np.array([[1000.0, 0.0]])            # coordinate 1 is invisible
    residual = np.array([5.0])
    hess = np.diag([4.0, 4.0])
    disp = np.array([0.02, 0.02])            # both start 20 mA off target
    opt = _iso_opt(sigma_x)
    opt.trust_radius = 10.0
    dp, *_ = opt.step(J, residual, hess @ disp, hess, prior_displacement=disp)
    # The blind coordinate is pulled back toward the target...
    assert dp[1] == pytest.approx(-disp[1], rel=0.05), dp
    # ...while the seen one is free to follow the data, which wants +x.
    assert dp[0] > 0.0, dp


@pytest.mark.parametrize("s_val,tol", [(1.0e4, 0.01), (1.0e3, 0.05)])
def test_the_isotropic_prior_barely_disturbs_a_well_seen_direction(s_val, tol):
    """It has to fade out where the data is informative, or it would trade good
    coordinates for bad ones -- which is exactly what scaling alpha_q did:
    measured on acetyl fluoride, alpha_q x30 moved C1-C2 from -18.49 to -15.13
    while the other four bonds degraded from 2.46 to 6.35 mA RMS.

    Parameterised because "negligible" is a statement about scale, not an
    absolute: the isotropic precision is 1/sigma_x^2, about 2.8e4 per Angstrom
    squared at sigma_x = 6 mA, so its influence is whatever that is next to
    s^2. At the spectral scale of these molecules (dB/dx of 1e3-1e4 MHz per
    Angstrom against sigmas of a few MHz) that is a fraction of a percent; at
    s = 1e3 it is a few percent, which a first version of this test asserted
    was under one percent and was right to fail.
    """
    from backend.spectral.SVD import SubspaceOptimizer as SO

    sigma_x = 0.006
    J = np.array([[s_val, 0.0], [0.0, s_val]])
    residual = np.array([5.0, 5.0])
    hess = np.diag([4.0, 4.0])
    disp = np.zeros(2)

    bare = SO(1e-3, 0.0, 0.1, None, 0.0, objective_mode="joint",
              alpha_quantum=1.0, quantum_prior_sigma_ang=sigma_x)
    bare.trust_radius = 10.0
    opt = _iso_opt(sigma_x)
    opt.trust_radius = 10.0

    dp_bare, *_ = bare.step(J, residual, hess @ disp, hess)
    dp_iso, *_ = opt.step(J, residual, hess @ disp, hess,
                          prior_displacement=disp)
    assert dp_iso == pytest.approx(dp_bare, rel=tol), (dp_iso, dp_bare)


def test_no_displacement_means_no_isotropic_contribution():
    """Callers that cannot supply a displacement (internal mode) must get the
    old behaviour rather than a silently half-applied prior."""
    from backend.spectral.SVD import SubspaceOptimizer as SO
    sigma_x = 0.006
    J = np.array([[1000.0, 0.0]])
    residual = np.array([5.0])
    hess = np.diag([4.0, 4.0])
    bare = SO(1e-3, 0.0, 0.1, None, 0.0, objective_mode="joint",
              alpha_quantum=1.0, quantum_prior_sigma_ang=sigma_x)
    bare.trust_radius = 10.0
    opt = _iso_opt(sigma_x)
    opt.trust_radius = 10.0
    a, *_ = bare.step(J, residual, np.zeros(2), hess)
    b, *_ = opt.step(J, residual, np.zeros(2), hess, prior_displacement=None)
    assert a == pytest.approx(b, rel=1e-12)
