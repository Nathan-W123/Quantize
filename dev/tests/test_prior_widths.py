"""The prior width the engine ships must be the one it was benchmarked with.

sigma_x is how far the quantum surface is trusted, in Angstrom. It is an
estimate of the level of theory's own geometry error, not a tuning knob, and
this repository measured it for three levels. Those measurements lived in a
benchmark script while MolecularOptimizer shipped one flat constant, so every
benchmark passed the calibrated value explicitly and every CLI and browser run
took the flat one. At B3LYP/cc-pVTZ that is a prior five times looser than the
measurement supports: the shipped engine and the benchmarked engine were not
the same engine.

These pin the three things that keeps true: the calibrated value is used, an
explicit value still wins, and an uncalibrated level degrades loosely rather
than optimistically.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

import dev.pyscf_backend  # noqa: F401  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer
from backend.spectral.prior_widths import (
    DEFAULT_PRIOR_SIGMA_ANG,
    PRIOR_SIGMA_BY_LEVEL,
    level_for_backend,
    prior_sigma_for_level,
)

WATER = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.7575, 0.5868],
                  [0.0, -0.7575, 0.5868]])
ISO = {
    "name": "H2-16O",
    "masses": [15.9949146196, 1.00782503207, 1.00782503207],
    "obs_constants": [835840.0, 435350.0, 278140.0],
    "sigma_constants": [1.0, 1.0, 1.0],
    "sigma_systematic_constants": [1.0, 1.0, 1.0],
    "component_indices": [0, 1, 2],
}


def _build(**kwargs):
    """Construct an optimiser quietly and return it with its [prior] line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=WATER, isotopologues=[ISO],
            quantum_backend="pyscf_hf", **kwargs)
    said = [ln.strip() for ln in buf.getvalue().splitlines() if "[prior]" in ln]
    return opt, (said[0] if said else "")


@pytest.mark.parametrize("method,basis", sorted(PRIOR_SIGMA_BY_LEVEL))
def test_every_calibrated_level_reaches_the_optimiser(method, basis):
    opt, said = _build(orca_method=method, orca_basis=basis)
    expected = PRIOR_SIGMA_BY_LEVEL[(method, basis)]
    assert opt.optimizer.quantum_prior_sigma_ang == pytest.approx(expected)
    assert "calibrated" in said


def test_b3lyp_cc_pvtz_is_not_the_flat_default():
    """The case that motivated this: 0.004 measured against 0.020 shipped."""
    opt, _ = _build(orca_method="b3lyp", orca_basis="cc-pvtz")
    got = opt.optimizer.quantum_prior_sigma_ang
    assert got == pytest.approx(0.004)
    assert got < DEFAULT_PRIOR_SIGMA_ANG / 2.0


def test_an_explicit_width_still_wins():
    """Benchmarks pass sigma_x explicitly; calibration must not override it."""
    opt, said = _build(orca_method="b3lyp", orca_basis="cc-pvtz",
                       quantum_prior_sigma_ang=0.050)
    assert opt.optimizer.quantum_prior_sigma_ang == pytest.approx(0.050)
    assert said == "", "calibration spoke over an explicit value"


def test_an_uncalibrated_level_degrades_loosely_and_says_so():
    """It must also be a float.

    The parameter used to default to the flat constant, so everything
    downstream always received a number. Switching the default to None to mean
    "calibrate me" left the uncalibrated path unassigned, and SubspaceOptimizer
    received a None it had never seen.
    """
    opt, said = _build(orca_method="mp2", orca_basis="cc-pvtz")
    got = opt.optimizer.quantum_prior_sigma_ang
    assert got == pytest.approx(DEFAULT_PRIOR_SIGMA_ANG)
    assert isinstance(got, float)
    assert "uncalibrated" in said


def test_the_default_is_the_loosest_calibrated_value():
    """An unmeasured level has earned no confidence. Asserting a precision the
    surface does not have is what cost water 1.36 -> 4.13 mA."""
    assert DEFAULT_PRIOR_SIGMA_ANG >= max(PRIOR_SIGMA_BY_LEVEL.values())


def test_backend_decides_which_method_basis_pair_counts():
    """psi4 reads its own pair; ORCA, PySCF and third-party backends read the
    orca_* pair, which is why benchmarks set orca_method while running PySCF."""
    assert level_for_backend("psi4", "hf", "6-31g", "b3lyp", "cc-pvdz") == ("b3lyp", "cc-pvdz")
    assert level_for_backend("pyscf_hf", "hf", "6-31g", "b3lyp", "cc-pvdz") == ("hf", "6-31g")
    assert level_for_backend("orca", "ccsd(t)", "cc-pVTZ") == ("ccsd(t)", "cc-pvtz")


def test_lookup_is_case_and_whitespace_insensitive():
    assert prior_sigma_for_level("B3LYP", " cc-pVTZ ") == (0.004, True)


def test_the_benchmark_scripts_share_the_library_table():
    """One source of truth, so a recalibration cannot reach only one of them."""
    from scripts.mixed_estimation_baseline import _SIGMA_X_BY_LEVEL
    assert _SIGMA_X_BY_LEVEL is PRIOR_SIGMA_BY_LEVEL


def test_an_explicit_none_keeps_the_legacy_heuristic():
    """None is a request, not a missing value.

    SubspaceOptimizer documents quantum_prior_sigma_ang=None as "legacy
    heuristic behaviour" and branches on it: with a float it derives alpha_q
    from sigma_x, with None it uses the alpha_quantum heuristic instead. Those
    are different objectives. The YAML runner passes None deliberately.

    The first version of this calibration made the parameter default to None to
    mean "calibrate me", which collapsed the two states and switched every
    None-passing caller -- the whole CLI path -- onto the calibrated-weight
    objective. test_endtoend_water_pipeline caught it by sign-checking an angle
    against theory's own error.
    """
    opt, said = _build(orca_method="b3lyp", orca_basis="cc-pvtz",
                       quantum_prior_sigma_ang=None)
    assert opt.optimizer.quantum_prior_sigma_ang is None, (
        "an explicit None was overwritten by the calibrated value"
    )
    assert said == "", "calibration spoke over an explicit None"


def test_the_quantum_weight_goes_as_one_over_sigma_squared():
    """Why calibrating sigma_x matters more than its size suggests.

    The joint objective is the engine's main solver, and sigma_x sets the
    balance in it through alpha_q = 1 / (0.5 * lambda_bar * sigma_x^2). The
    square is the point: B3LYP/cc-pVTZ's 0.004 against the flat 0.020 is five
    times in sigma_x and twenty-five times in how hard the quantum prior pulls.
    """
    import dev.analytic_water_backend  # noqa: F401
    from backend.registry import get_backend
    from backend.spectral.SVD import SubspaceOptimizer

    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    hess = get_backend("analytic_water")(elems=["O", "H", "H"]).run_hessian(
        coords).hessian_bohr

    def alpha_q(sigma):
        return SubspaceOptimizer(
            objective_mode="joint",
            quantum_prior_sigma_ang=sigma)._calibrated_alpha_q(hess)

    ratio = alpha_q(PRIOR_SIGMA_BY_LEVEL[("b3lyp", "cc-pvtz")]) / alpha_q(
        DEFAULT_PRIOR_SIGMA_ANG)
    assert ratio == pytest.approx((0.020 / 0.004) ** 2, rel=1e-6)
    assert ratio == pytest.approx(25.0, rel=1e-6)
