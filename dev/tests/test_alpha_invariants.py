"""Exact properties the alpha expression must satisfy, whatever the source.

Task #16 asked for a term-by-term comparison against Mills' canonical form.
That form could not be reached: searches return papers that cite the 1972
chapter rather than reproduce it, and writing down a published equation from
memory and calling the result "verified" is the failure this repository
documents elsewhere -- see dev/g_tensor_references.py, which records four text
corruptions found in deposited abstracts.

So these check the implementation against things that are true independently of
any text. Two of the three terms already have exact closed forms pinned in
test_alpha_against_experiment: the harmonic term against Dunham's -6Be^2/we and
harmonic-plus-anharmonic against the Pekeris relation, both on a Morse CO. What
those cannot touch is everything a diatomic does not have -- the Coriolis term
is identically zero with one mode, the harmonic term's sum over off-diagonal
inertia derivatives never engages, and phi_rrs has no s != r.

The invariances below do reach those. alpha is built from an inertia tensor, a
mass-weighted Hessian, a principal-axis rotation and three index sums, and a
wrong axis pairing, a transposed zeta or a sign error in the Coriolis
denominator would almost all show up as a dependence on where the molecule
happens to sit or how its identical atoms happen to be numbered.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.harmonic_alpha import compute_harmonic_alpha
from reference_molecules import (
    CO_COORDS,
    CO_MASSES,
    co_hessian,
    H2O_MASSES,
    h2o_coords,
    h2o_hessian,
)

COMPONENTS = ("A", "B", "C")


def _alpha(coords, masses, hessian, **kw):
    alpha, _, _, info = compute_harmonic_alpha(hessian, coords, masses, **kw)
    return alpha, info


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ── The Coriolis coupling constants themselves ──────────────────────────────

def _zeta_of(coords, masses, hessian):
    """Rebuild the zeta array the way compute_harmonic_alpha does.

    Duplicated rather than imported because the point is to check the
    construction, and a helper shared with the code under test would check
    nothing.
    """
    from backend.spectral.centrifugal_distortion import (
        normal_modes,
        rigid_mode_count,
    )
    omega, L_mw = normal_modes(
        hessian, masses, n_rigid=rigid_mode_count(coords, masses))
    keep = omega >= 50.0
    L = L_mw[:, keep].reshape(len(masses), 3, -1)
    n_vib = L.shape[2]
    zeta = np.zeros((3, n_vib, n_vib))
    for K, (a1, a2) in enumerate(((1, 2), (2, 0), (0, 1))):
        zeta[K] = (np.einsum("ar,as->rs", L[:, a1, :], L[:, a2, :])
                   - np.einsum("ar,as->rs", L[:, a2, :], L[:, a1, :]))
    return zeta


def test_zeta_is_antisymmetric():
    """zeta^a_rs = -zeta^a_sr, exactly, from its definition.

    The Coriolis sum skips s == r and squares every term, so a transposed zeta
    would survive silently. Antisymmetry is what makes the diagonal vanish and
    is the cheapest check that the two einsums are the right way round.
    """
    coords = h2o_coords()
    zeta = _zeta_of(coords, H2O_MASSES, h2o_hessian(coords))
    for K in range(3):
        assert np.allclose(zeta[K], -zeta[K].T, atol=1e-12)
        assert np.allclose(np.diag(zeta[K]), 0.0, atol=1e-12)


def test_zeta_is_bounded_by_orthogonality():
    """|zeta| <= 1, and each row's square-sum <= 1.

    The normal-mode transformation is orthogonal, so zeta is a block of an
    orthogonal rotation generator's matrix element and cannot exceed one. A
    normalisation error in L -- a missing mass weight, say -- shows up here
    immediately, and would otherwise scale the whole Coriolis term.
    """
    coords = h2o_coords()
    zeta = _zeta_of(coords, H2O_MASSES, h2o_hessian(coords))
    assert np.abs(zeta).max() <= 1.0 + 1e-9
    for K in range(3):
        assert (np.sum(zeta[K] ** 2, axis=1) <= 1.0 + 1e-9).all()


def test_a_diatomic_has_no_coriolis_coupling():
    """One vibrational mode cannot couple to another."""
    zeta = _zeta_of(CO_COORDS, CO_MASSES, co_hessian(CO_COORDS))
    assert np.abs(zeta).max() < 1e-10


# ── Invariances of the whole expression ─────────────────────────────────────

@pytest.mark.parametrize("shift", [
    np.array([1.0, 0.0, 0.0]),
    np.array([-3.7, 2.1, 0.9]),
])
def test_alpha_is_invariant_under_translation(shift):
    """alpha is a molecular property; where the molecule sits is not."""
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    base, _ = _alpha(coords, H2O_MASSES, hess)
    moved, _ = _alpha(coords + shift, H2O_MASSES, hess)
    for k in COMPONENTS:
        assert moved[k] == pytest.approx(base[k], rel=1e-9)


@pytest.mark.parametrize("axis,angle", [
    ([0.0, 0.0, 1.0], 0.7),
    ([1.0, 1.0, 0.0], 1.3),
    ([0.3, -0.5, 0.8], 2.4),
])
def test_alpha_is_invariant_under_rigid_rotation(axis, angle):
    """The strongest single check available without a reference text.

    alpha is assembled from a principal-axis rotation, an inertia-tensor
    derivative summed over axis pairs, and a Coriolis sum whose zeta carries an
    axis index. A wrong axis pairing, a transposed rotation or a swapped
    component would almost all break this, because they would tie the answer to
    the frame the coordinates happened to arrive in.

    The Hessian is rotated with the coordinates: it is a second derivative in
    Cartesian space, so R H R^T in each atom block.
    """
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    R = _rotation(axis, angle)
    n = len(H2O_MASSES)
    big = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            big[3 * i:3 * i + 3, 3 * j:3 * j + 3] = (
                R @ hess[3 * i:3 * i + 3, 3 * j:3 * j + 3] @ R.T)

    base, _ = _alpha(coords, H2O_MASSES, hess)
    turned, _ = _alpha(coords @ R.T, H2O_MASSES, big)
    for k in COMPONENTS:
        assert turned[k] == pytest.approx(base[k], rel=1e-7), (
            f"{k}: alpha depends on the input frame"
        )


def test_alpha_is_invariant_under_relabelling_identical_atoms():
    """Swapping water's two hydrogens is a relabelling, not a new molecule.

    Every index sum in the expression runs over atoms, and this is what catches
    one that assumes an ordering.
    """
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    perm = [0, 2, 1]                     # O stays, the two H swap
    idx = np.concatenate([[3 * p, 3 * p + 1, 3 * p + 2] for p in perm])

    base, _ = _alpha(coords, H2O_MASSES, hess)
    swapped, _ = _alpha(coords[perm], H2O_MASSES[perm], hess[np.ix_(idx, idx)])
    for k in COMPONENTS:
        assert swapped[k] == pytest.approx(base[k], rel=1e-7)


def test_alpha_scales_with_the_isotopologue_it_is_asked_for():
    """A guard against the masses being ignored somewhere in the chain.

    Deuterating water must change alpha substantially -- this is the entire
    mechanism the engine runs on, since it is isotopic differences that carry
    the structural information.
    """
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    light, _ = _alpha(coords, H2O_MASSES, hess)
    heavy_masses = H2O_MASSES.copy()
    heavy_masses[1:] = 2.01410177812
    heavy, _ = _alpha(coords, heavy_masses, hess)
    assert any(abs(heavy[k] - light[k]) > 0.05 * abs(light[k]) for k in COMPONENTS)


# ── Each term's exact power of omega ────────────────────────────────────────

def test_each_term_carries_its_own_exact_power_of_omega():
    """Uniform frequency scaling separates the three terms and checks each.

    freq_scale multiplies every harmonic frequency and leaves the normal-mode
    vectors and the cubic force constants alone, so it changes omega and
    nothing else. Each term then has a different exact exponent:

      harmonic    <Q_r^2> ~ 1/omega_r, the second derivative omega-free   1/w
      Coriolis    prefactor -2Be^2/omega_r times a bracket
                  (3w_r^2 + w_s^2)/(w_r^2 - w_s^2) that is homogeneous of
                  degree ZERO, so the bracket contributes nothing            1/w
      anharmonic  <Q_r^2> ~ 1/omega_r times 1/lambda_s with lambda_s = w_s^2  1/w^3

    Doubling every frequency must therefore halve the first two and divide the
    third by eight. This is sharper than it looks. The Coriolis exponent is a
    direct check that its bracket is degree zero, which a non-homogeneous form
    would break; the anharmonic exponent distinguishes lambda_s = w_s^2 from
    w_s, which is otherwise only visible by reading the code.

    What it still cannot see is a dimensionless constant: the 3 in
    3w_r^2 + w_s^2, or the 2 in -2Be^2. For the harmonic and anharmonic terms
    those are pinned by the Dunham and Pekeris closed forms in
    test_alpha_against_experiment. For the Coriolis term they are not pinned by
    anything in this repository, which is the honest remaining gap in task #16.
    """
    coords = h2o_coords()

    def terms(scale):
        _, _, _, info = compute_harmonic_alpha(
            h2o_hessian(coords), coords, H2O_MASSES,
            hessian_fn=h2o_hessian, freq_scale=scale)
        return {name: info[f"alpha_{name}_mhz"]
                for name in ("centrifugal", "coriolis", "anharmonic")}

    one, two = terms(1.0), terms(2.0)
    expected = {"centrifugal": 2.0, "coriolis": 2.0, "anharmonic": 8.0}
    for name, want in expected.items():
        for k in COMPONENTS:
            assert abs(two[name][k]) > 0.0, f"{name}/{k} vanished; test is vacuous"
            ratio = one[name][k] / two[name][k]
            assert ratio == pytest.approx(want, rel=1e-6), (
                f"{name} term scales as {ratio:.4f} on {k}, expected {want}"
            )
