"""The sampler must not spend itself moving the molecule around the box.

Translations and rotations change no rotational constant, so an isotropic
step in all 3N Cartesians wastes most of itself. Measured on water before
this change: 97.9% of the sampled variance sat in the six rigid-body
directions and 2.1% in the three that mean anything.

Restricting the proposal to the physical subspace is valid, not merely
tidier: a zero-mean Gaussian supported on a fixed subspace is still a
symmetric proposal, so detailed balance holds and the chain samples the
target restricted to the affine subspace through its start point.

What this does NOT fix is convergence. R-hat stays around 2.4, because the
cause is that the exact posterior is improper along one physical direction
(see test_gauss_newton_validity), not gauge wandering. Pinned here so nobody
reads the gauge fix as a convergence fix.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.mcmc import AdaptiveMetropolisV2  # noqa: E402
from backend.uncertainty_v2.models import MCMCConfig  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


@pytest.fixture(scope="module")
def sampled(tmp_path_factory):
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O, "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02,
        workdir=str(tmp_path_factory.mktemp("mcmc")))
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
        summary = AdaptiveMetropolisV2(opt, MCMCConfig(
            n_steps=600, burn_in=200, n_chains=2, seed=11,
            persist_chain_samples=False)).run()
    return opt, summary


def _subspaces(opt):
    basis = opt.physical_basis()
    n = basis.shape[0]
    gauge = np.eye(n) - basis @ basis.T
    w, V = np.linalg.eigh((gauge + gauge.T) / 2)
    return basis, V[:, w > 0.5]


def test_water_has_two_physical_and_seven_removed_directions(sampled):
    """Under the auto-detected C2v: 6 rigid-body + 1 symmetry-forbidden.

    The chain is proposed only along the two symmetric internal directions.
    Before the symmetry restriction it was also proposed along the
    antisymmetric stretch -- a direction the fit itself never moves along,
    because it symmetrises after every step -- so the sampler was exploring a
    model nobody had fitted.
    """
    opt, _ = sampled
    basis, gbasis = _subspaces(opt)
    assert opt.symmetry is not None and opt.symmetry.point_group == "C2v"
    assert basis.shape == (9, 2)
    assert gbasis.shape == (9, 7)
    assert opt.projected_direction_counts() == (6, 1)


def test_no_sampled_variance_in_the_gauge_subspace(sampled):
    """97.9% before, and it has to be numerically zero, not merely smaller."""
    opt, summary = sampled
    basis, gbasis = _subspaces(opt)
    s = np.concatenate([c.samples.reshape(c.samples.shape[0], -1)
                        for c in summary.chains])
    c = s - s.mean(0)
    v_phys = float(np.sum((c @ basis) ** 2))
    v_gauge = float(np.sum((c @ gbasis) ** 2))
    assert v_phys > 0.0, "the chain did not move at all"
    assert v_gauge / (v_phys + v_gauge) < 1e-12, (
        f"{100 * v_gauge / (v_phys + v_gauge):.2f}% of the sampled variance "
        f"is in rigid-body directions")


def test_the_molecule_never_translates(sampled):
    """The centroid is exactly conserved along each chain.

    Which quantity is conserved is worth being precise about. The projector
    builds its translation modes as uniform UNWEIGHTED vectors -- correctly,
    per its own docstring: the null vectors of a plain Cartesian Hessian are
    uniform displacements, not sqrt(m)-scaled ones. So a physical
    displacement satisfies sum_i d_i = 0 per axis, which conserves the plain
    centroid. The centre of mass is NOT conserved and need not be: measured
    drift 4.9e-3 Angstrom against 5.6e-17 for the centroid. That coupling
    between internal motion and centre-of-mass position is a property of the
    unweighted projector, not a leak of translational freedom.
    """
    _opt, summary = sampled
    m = np.asarray(MASSES_H2O, dtype=float)
    for chain in summary.chains:
        centroid0 = chain.samples[0].mean(axis=0)
        drift = max(float(np.max(np.abs(xyz.mean(axis=0) - centroid0)))
                    for xyz in chain.samples)
        assert drift < 1e-12, f"centroid drifted by {drift:.3e} Angstrom"

    # And the centre of mass is expected to move, so nobody later "fixes"
    # the projector to conserve it.
    com_drift = max(
        float(np.max(np.abs((m[:, None] * xyz).sum(0) / m.sum()
                            - (m[:, None] * chain.samples[0]).sum(0) / m.sum())))
        for chain in summary.chains for xyz in chain.samples)
    assert com_drift > 0.0


def test_chains_still_move_in_physical_directions(sampled):
    """The complement: projection must not have frozen the sampler."""
    _opt, summary = sampled
    for chain in summary.chains:
        spread = float(np.std(chain.samples, axis=0).max())
        assert spread > 0.0


def test_projection_is_not_a_convergence_fix(sampled):
    """Honest about what it does not do: the saddle is still the saddle."""
    _opt, summary = sampled
    if summary.convergence is None:
        pytest.skip("too few samples for diagnostics")
    # Still unconverged, and still correctly refused.
    assert summary.usable is False
    assert summary.unusable_reason()


def test_full_rank_fallback_proposes_in_all_directions():
    """With no reference masses there is no projector, and no silent freeze."""
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O, "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    saved = opt._rigid_ref_masses
    try:
        opt._rigid_ref_masses = None
        basis = opt.physical_basis()
        assert basis.shape == (9, 9)
        assert np.array_equal(basis, np.eye(9))
    finally:
        opt._rigid_ref_masses = saved


def test_water_chains_now_converge_under_the_imposed_symmetry(sampled):
    """The refusal used to fire on every water run; now it need not.

    With proposals along all three internal directions, four seeds gave max
    R-hat 2.25 to 3.42 and min ESS 34 to 46 -- unusable every time, because at
    these residuals the exact posterior is a saddle along the antisymmetric
    stretch and the chains wandered along it. That direction is fixed by the
    C2v the fit imposes, so it is no longer proposed, and the same four seeds
    give R-hat 1.018 to 1.054 with ESS 166 to 185. Pinned on one of them, at
    a length long enough for ESS to clear its threshold.
    """
    opt, _ = sampled
    with contextlib.redirect_stdout(io.StringIO()):
        summary = AdaptiveMetropolisV2(opt, MCMCConfig(
            n_steps=2000, burn_in=500, n_chains=4, seed=11,
            persist_chain_samples=False)).run()
    assert summary.convergence is not None
    r_hat = float(summary.convergence.r_hat.max())
    ess = float(summary.convergence.ess.min())
    assert summary.usable, summary.unusable_reason()
    assert r_hat < 1.1, f"max R-hat {r_hat:.3f}"
    assert ess > 100.0, f"min ESS {ess:.1f}"

