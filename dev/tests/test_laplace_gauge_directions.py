"""The Laplace report must quote widths a geometry could actually have.

These exist because the suite was green while `--uncertainty-engine v2`
printed `max std err : 1.118297e+02` for water -- 112 Angstrom of uncertainty
on a 0.958 Angstrom bond. Nothing was broken in the covariance; the report
simply included the rigid-body subspace, where the only thing setting a width
is the regularisation ridge, and no test asserted that a reported width was
physically possible.

The ridge is also why it went unnoticed: it lifts the six null directions just
enough that `matrix_rank` returns 9 of 9 and `cond` lands at 2.4e11, under the
1e12 warning threshold. So the report carried two reassuring diagnostics on
either side of a number wrong by four orders of magnitude.
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

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.posterior import laplace_approximation  # noqa: E402
from backend.uncertainty_v2.reporting import format_laplace_summary  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]
R_OH_ANG = 0.95785
THETA_DEG = 104.508


def _coords(r=R_OH_ANG, deg=THETA_DEG):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _water(**kw):
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O,
                        "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


@pytest.fixture(scope="module")
def water_fit():
    """symmetry="auto", the default: C2v is detected and imposed.

    Under that constraint water has two physical directions, not three. The
    third -- the antisymmetric stretch -- is fixed by the symmetrisation the
    fit applies after every step, so it is not a parameter and the posterior
    has no width along it. See MolecularOptimizer._restrict_to_symmetric.
    """
    return _water()


@pytest.fixture(scope="module")
def water_fit_c1():
    """No symmetry imposed: all three internal directions are parameters."""
    return _water(symmetry="C1")


def test_reported_widths_are_smaller_than_the_molecule(water_fit):
    """A sigma larger than the bond it describes is not a wide interval."""
    summary = laplace_approximation(water_fit)
    worst = float(np.max(summary.std_err))
    # The unprojected report gave 1.12e2 here. Anything at or above the bond
    # length is meaningless as a width on that bond, whatever its provenance.
    assert worst < R_OH_ANG, (
        f"worst reported sigma {worst:.3e} Ang exceeds the O-H bond "
        f"({R_OH_ANG} Ang) it is supposed to be an uncertainty on")
    # And it should be small, not merely finite: these are 0.2 MHz constants.
    assert worst < 0.05


def test_rigid_body_directions_are_counted_and_removed(water_fit):
    """Water is 9 Cartesians: 3 translations + 3 rotations go, and under the
    imposed C2v so does the antisymmetric stretch -- counted apart, because a
    symmetry assumption is not a gauge freedom."""
    summary = laplace_approximation(water_fit)
    assert water_fit.symmetry is not None and water_fit.symmetry.point_group == "C2v"
    assert summary.n_params == 9
    assert summary.n_gauge_directions == 7
    assert summary.n_symmetry_directions == 1
    assert summary.n_gauge_directions - summary.n_symmetry_directions == 6
    assert summary.rank == 2, "both symmetric internal directions should be identified"


def test_without_symmetry_all_three_internal_directions_remain(water_fit_c1):
    """The rigid-body count does not depend on the symmetry handling."""
    summary = laplace_approximation(water_fit_c1)
    assert water_fit_c1.symmetry is None
    assert summary.n_params == 9
    assert summary.n_gauge_directions == 6
    assert summary.n_symmetry_directions == 0
    assert summary.rank == 3, "all three internal directions should be identified"


def test_gauge_subspace_carries_no_width(water_fit):
    """Displacing or turning the whole molecule has no posterior width.

    This is the property the report was violating. It is exact, not
    approximate: the projection sends those directions to zero rather than
    making them small.
    """
    summary = laplace_approximation(water_fit)
    cov = summary.covariance
    n_atoms = 3
    coords = water_fit.coords
    masses = np.asarray(MASSES_H2O, dtype=float)
    com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    rel = coords - com

    gauge = []
    for axis in range(3):                       # translations
        v = np.zeros((n_atoms, 3))
        v[:, axis] = 1.0
        gauge.append(v.ravel())
    for omega in np.eye(3):                     # rotations
        gauge.append(np.cross(np.tile(omega, (n_atoms, 1)), rel).ravel())

    scale = float(np.max(np.abs(cov)))
    for k, v in enumerate(gauge):
        v = v / np.linalg.norm(v)
        var = float(v @ cov @ v)
        assert abs(var) < 1e-9 * max(scale, 1.0), (
            f"gauge direction {k} carries variance {var:.3e}, so the "
            f"projection did not remove it")


def test_physical_directions_do_carry_width(water_fit):
    """The complement of the above: the fit is not simply zeroed out."""
    summary = laplace_approximation(water_fit)
    assert float(np.max(summary.std_err)) > 0.0
    # Two nonzero eigenvalues under C2v, one per symmetric internal coordinate.
    eig = np.linalg.eigvalsh((summary.covariance + summary.covariance.T) / 2)
    scale = float(np.max(eig))
    assert int(np.sum(eig > 1e-9 * scale)) == 2


def test_unconstrained_physical_directions_do_carry_width(water_fit_c1):
    summary = laplace_approximation(water_fit_c1)
    eig = np.linalg.eigvalsh((summary.covariance + summary.covariance.T) / 2)
    scale = float(np.max(eig))
    assert int(np.sum(eig > 1e-9 * scale)) == 3


def test_condition_number_describes_the_physical_block(water_fit):
    """The 2.4e11 this engine kept reporting was the gauge subspace.

    Worth stating, because that number has been read as "the fit is badly
    conditioned" for a long time. It is not: over the three physical
    directions the problem is conditioned at 2.4e3. The ill-conditioning was
    entirely the six directions the ridge was propping up.

    The bound below is what makes this test discriminating -- 1e12 would pass
    on the unprojected matrix too, since 2.4e11 slips under it, which is also
    why no warning ever fired.
    """
    summary = laplace_approximation(water_fit)
    assert np.isfinite(summary.condition_number)
    assert summary.condition_number < 1e6, (
        f"cond {summary.condition_number:.3e} looks like the padded matrix "
        f"(2.4e11), not the physical block (2.4e3)")
    # No spurious rank or conditioning warnings: the structural nullity is
    # not a finding. Scoped to those two, because the Gauss-Newton validity
    # warning fires correctly on this fixture (18 sigma residuals) and is
    # pinned by test_gauss_newton_validity.
    noise = [w for w in summary.warnings
             if "Rank-deficient" in w or "Ill-conditioned" in w]
    assert noise == [], noise


def test_v2_covariance_matches_the_optimizer_method(water_fit):
    """F2: one implementation. The raw covariance must be bit-identical."""
    cov_method, _ = water_fit.laplace_approximation()
    cov_phys, _basis, n_gauge = water_fit.physical_covariance(cov_method)
    summary = laplace_approximation(water_fit)
    assert n_gauge == summary.n_gauge_directions
    assert np.array_equal(cov_phys, summary.covariance), (
        "v2 no longer reports what the optimizer method computed")


def test_regularization_and_coords_reach_the_method(water_fit):
    base = laplace_approximation(water_fit)
    loose = laplace_approximation(water_fit, regularization=1e-4)
    assert loose.regularization == pytest.approx(1e-4)
    assert not np.array_equal(loose.covariance, base.covariance)
    # A negative ridge is clamped, not passed through.
    assert laplace_approximation(
        water_fit, regularization=-1.0).regularization == 0.0
    off = laplace_approximation(water_fit, coords=water_fit.coords + 0.01)
    assert not np.array_equal(off.covariance, base.covariance)


def test_report_labels_the_projection(water_fit):
    """"rank 3" beside "n params 9" must not read as a rank-deficient fit."""
    txt = format_laplace_summary(laplace_approximation(water_fit))
    assert "2 physical" in txt
    assert "6 rigid-body + 1 symmetry-forbidden removed" in txt
    assert "of 2 physical" in txt
    # And without a symmetry constraint the label is the plain one.
    txt_c1 = format_laplace_summary(laplace_approximation(_water(symmetry="C1")))
    assert "3 physical" in txt_c1
    assert "6 rigid-body removed" in txt_c1
    assert "symmetry" not in txt_c1


def test_physical_covariance_is_idempotent(water_fit):
    """Projecting twice is projecting once, so double application is safe.

    geometry_uncertainty and the v2 path both call it; this is what makes it
    harmless if a third caller appears downstream of either.

    Tolerances come from the matrix's own scale, not from a round number.
    ||cov||_F is 3.1e4 here, so an element-wise atol of 1e-18 -- which is what
    this test first asserted -- sits fourteen orders below the round-off floor
    and fails on noise in structurally-zero entries. What idempotency actually
    predicts is a relative change at machine precision, and a second pass that
    converges rather than drifting.
    """
    cov, _ = water_fit.laplace_approximation()
    scale = float(np.linalg.norm(cov))
    once, basis, g1 = water_fit.physical_covariance(cov)
    twice, _b2, g2 = water_fit.physical_covariance(once)
    assert g1 == g2 == 7          # 6 rigid-body + 1 symmetry-forbidden

    # Measured 6.4e-18; eps alone would allow 2.2e-16.
    rel = float(np.linalg.norm(once - twice)) / scale
    assert rel < 1e-15, f"second projection moved the matrix by {rel:.3e}"

    # The block that carries the uncertainties agrees to machine precision.
    b1 = basis.T @ once @ basis
    b2 = basis.T @ twice @ basis
    assert np.linalg.norm(b1 - b2) / np.linalg.norm(b1) < 1e-14

    # And it settles instead of walking: ten passes, each change smaller.
    cur = twice
    for _ in range(8):
        nxt, _b, _g = water_fit.physical_covariance(cur)
        assert float(np.linalg.norm(nxt - cur)) / scale < 1e-15
        cur = nxt


def test_physical_covariance_passes_through_without_masses(water_fit):
    """No reference masses means no projector, and no silent zeroing."""
    cov, _ = water_fit.laplace_approximation()
    saved = water_fit._rigid_ref_masses
    try:
        water_fit._rigid_ref_masses = None
        out, basis, n_gauge = water_fit.physical_covariance(cov)
        assert n_gauge == 0
        assert np.array_equal(out, cov)
        assert np.array_equal(basis, np.eye(cov.shape[0]))
    finally:
        water_fit._rigid_ref_masses = saved


def test_geometry_uncertainty_agrees_on_scale(water_fit):
    """The two user-facing paths should not disagree by orders of magnitude.

    geometry_uncertainty propagates to bonds and angles and adds the
    systematic block, so the numbers are not equal -- but they are built from
    the same covariance and must live on the same scale. The 112-Angstrom
    report failed this by four orders.
    """
    summary = laplace_approximation(water_fit)
    with contextlib.redirect_stdout(io.StringIO()):
        unc = water_fit.geometry_uncertainty()
    if unc is None:
        pytest.skip("geometry_uncertainty declined this covariance")
    bond_sigmas = [v for key, v in _flatten(unc)
                   if np.isfinite(v) and "angle" not in str(key).lower()]
    if not bond_sigmas:
        pytest.skip("no finite bond widths reported")
    worst_v2 = float(np.max(summary.std_err))
    worst_geo = float(np.max(bond_sigmas))
    assert worst_geo < R_OH_ANG
    ratio = max(worst_v2, worst_geo) / max(min(worst_v2, worst_geo), 1e-30)
    assert ratio < 1e3, (
        f"v2 worst sigma {worst_v2:.3e} vs geometry_uncertainty "
        f"{worst_geo:.3e}: same covariance, so this gap is a bug")


def _flatten(obj, key=""):
    """Walk whatever shape geometry_uncertainty returns for (label, value)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{key}[{i}]")
    elif isinstance(obj, (int, float, np.floating)):
        yield key, float(obj)
