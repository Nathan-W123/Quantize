"""The hybrid must not do worse than the quantum surface it started from.

Adding experimental information should never cost accuracy. The check runs the
real MolecularOptimizer against a deliberately detuned theory PES over every
subset of the observed constants, including the undersaturated single-constant
cases the SVD split exists to handle.

This is the property that broke when `harmonic_from_hessian` silently did
nothing at hess_recalc_every=1: the fit targeted ground-state B_0 as if it were
B_e, a systematic error of several thousand MHz, and three of the seven subsets
came out worse than theory alone.
"""

import contextlib
import io
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from hybrid_partial_data import (  # noqa: E402
    R_E,
    R_THEORY,
    THETA_E,
    THETA_THEORY,
    internals,
)
from backend.quantize import MolecularOptimizer  # noqa: E402
from reference_molecules import H2O_B0_MHZ, H2O_MASSES, h2o_coords  # noqa: E402

THEORY_DR_MA = abs(R_THEORY - R_E) * 1000.0
THEORY_DTHETA = abs(THETA_THEORY - THETA_E)
ALL_SUBSETS = [
    c for n in (1, 2, 3) for c in itertools.combinations(range(3), n)
]


def _run(components):
    start = h2o_coords(R_THEORY, np.radians(THETA_THEORY))
    comps = list(components)
    iso = [{
        "name": "H2-16O",
        "masses": H2O_MASSES.tolist(),
        "obs_constants": [float(H2O_B0_MHZ[c]) for c in comps],
        "sigma_constants": [0.2] * len(comps),
        "alpha_constants": [0.0] * len(comps),
        "component_indices": comps,
    }]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"],
        coords=start,
        isotopologues=iso,
        quantum_backend="detuned_water",
        harmonic_from_hessian=True,
        anharmonic_from_hessian=True,
        coordinate_mode="cartesian",
        max_iter=40,
        spectral_only=False,
        use_autoconfig=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        coords = opt.run()
    return internals(coords)


@pytest.mark.parametrize("components", ALL_SUBSETS,
                         ids=lambda c: "".join("ABC"[i] for i in c))
def test_hybrid_is_never_worse_than_theory_alone(components):
    r, theta = _run(components)
    assert abs(r - R_E) * 1000.0 <= THEORY_DR_MA, (
        f"bond length worse than theory: {abs(r - R_E) * 1000:.1f} mA "
        f"vs {THEORY_DR_MA:.1f} mA"
    )
    # The angle is allowed a little slack: a single constant genuinely cannot
    # determine both parameters, so one of them leans on the prior.
    assert abs(theta - THETA_E) <= THEORY_DTHETA + 1.0


def test_more_constants_do_not_hurt():
    """Going from one constant to all three should not lose accuracy."""
    r_one, _ = _run((1,))
    r_all, _ = _run((0, 1, 2))
    assert abs(r_all - R_E) <= abs(r_one - R_E) + 5e-4


def test_exact_targets_recover_the_reference_structure_from_any_surface():
    """The optimizer and the SVD split introduce no error of their own.

    Feed B_e taken straight from the reference geometry and the recovered
    structure must be r_e regardless of where the theory surface has its
    minimum. This is what places the residual in the correction-quality rows on
    alpha, and therefore on the Hessian, rather than on the fitting machinery.
    """
    from backend.spectral.centrifugal_distortion import rotational_constants_mhz
    from water_scenarios import run as run_scenario

    be_exact = rotational_constants_mhz(h2o_coords(), H2O_MASSES)
    for backend, start in (
        ("ontarget_water", h2o_coords(0.9800, np.radians(107.0))),
        ("detuned_water", h2o_coords(R_THEORY, np.radians(THETA_THEORY))),
    ):
        r, theta = run_scenario(backend, start, False, False, targets=be_exact)
        assert r == pytest.approx(R_E, abs=1e-4)
        assert theta == pytest.approx(THETA_E, abs=1e-2)


def test_full_correction_chain_improves_the_bond_angle():
    """On the full A/B/C data the corrections roughly a third the angle error.

    Pinned separately from the bond length, which this force field does not
    improve -- see scripts/water_scenarios.py for why.
    """
    from water_scenarios import run as run_scenario

    start = h2o_coords(R_THEORY, np.radians(THETA_THEORY))
    _, theta_raw = run_scenario("detuned_water", start, False, False)
    _, theta_full = run_scenario("detuned_water", start, True, True)
    assert abs(theta_full - THETA_E) < 0.5 * abs(theta_raw - THETA_E)


def test_undersaturated_spectroscopy_alone_is_ambiguous():
    """One constant against two structural parameters has no unique solution,
    which is what makes the quantum prior necessary rather than merely helpful."""
    from scipy.optimize import least_squares

    from backend.spectral.centrifugal_distortion import rotational_constants_mhz

    b_obs = H2O_B0_MHZ[1]

    def fit(start):
        sol = least_squares(
            lambda p: [(b_obs - rotational_constants_mhz(
                h2o_coords(p[0], np.radians(p[1])), H2O_MASSES)[1]) / b_obs],
            np.array(start), bounds=([0.85, 95.0], [1.10, 115.0]),
            xtol=1e-14, ftol=1e-14,
        )
        return internals(h2o_coords(sol.x[0], np.radians(sol.x[1])))

    r_hi, _ = fit((0.99, 108.0))
    r_lo, _ = fit((0.94, 100.0))
    assert abs(r_hi - r_lo) > 0.02, "expected a genuinely ambiguous fit"
