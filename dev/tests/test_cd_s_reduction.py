"""Watson's S reduction, and the checks that need no literature to make.

The A reduction was validated against water's measured DJ, DJK and DK. S cannot
be validated that way for the same molecule -- water is conventionally fit in A,
and an S-reduced DJ is a different number from an A-reduced one, so there is
nothing to compare against. What makes S checkable anyway is that it is a
reparameterisation of the same operator, which forces a set of properties an
incorrect implementation cannot satisfy:

  * the S asymmetry operators must reduce to the identities they are built from,
  * d1 must come out as -delta_J, the one relation between the two reductions
    that is clean at the quartic level,
  * a symmetric top must give zero asymmetry constants in both reductions,
  * feeding fitted parameters back through their own operator must reproduce the
    levels they were fitted to, and
  * S must beat A on a near-symmetric top by a wide margin, because that is the
    entire reason Watson introduced it.

The last one is the strongest. Nothing in this construction was told that S
exists for near-symmetric tops; it either falls out or the implementation is
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import dev.analytic_water_backend  # noqa: F401  (registers "analytic_water")
from backend.registry import get_backend
from backend.spectral.cd_reduction import (
    REDUCTION_PARAMS,
    angular_momentum_operators,
    best_reduction,
    ladder_power_sum,
    levels_from_reduction,
    levels_from_tau,
    reduction_from_tau,
    reduction_residual_mhz,
)
from backend.spectral.centrifugal_distortion import (
    _CM_TO_MHZ,
    _MHZ_TO_CM,
    bk_mode_derivatives,
    normal_modes,
    rotational_constants_mhz,
    tau_prime_from_dB1_cm,
)

#: Ray's asymmetry parameter kappa = (2B - A - C)/(A - C): -1 prolate, +1 oblate.
_PROLATE = np.array([300000.0, 10500.0, 10000.0])
_OBLATE = np.array([10500.0, 10200.0, 10000.0])
_SYMMETRIC = np.array([300000.0, 10000.0, 10000.0])
_ASYMMETRIC = np.array([300000.0, 200000.0, 100000.0])


def _water_tau():
    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    masses = np.array([15.9949146196, 1.00782503207, 1.00782503207])
    hess = get_backend("analytic_water")(
        elems=["O", "H", "H"]).run_hessian(coords).hessian_bohr
    abc = rotational_constants_mhz(coords, masses)
    omega, l_mw = normal_modes(hess, masses, n_rigid=6)
    keep = omega >= 50.0
    omega, l_mw = omega[keep], l_mw[:, keep]
    db1, _ = bk_mode_derivatives(coords, masses, l_mw, omega, 0.05, abc)
    return abc, tau_prime_from_dB1_cm(db1 * _MHZ_TO_CM, omega) * _CM_TO_MHZ


# ── the ladder operator the S reduction is built on ─────────────────────────

@pytest.mark.parametrize("j", [0, 1, 2, 3, 5, 7])
def test_ladder_power_two_reproduces_the_existing_asymmetry_operator(j):
    """P+^2 + P-^2 == 2 (Px^2 - Py^2), which is already validated.

    Anchoring the new operator to the old one means the d2 operator inherits
    that validation rather than asking to be believed on its own.
    """
    _, _, pd = angular_momentum_operators(j)
    assert np.allclose(ladder_power_sum(j, 2), 2.0 * pd)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_ladder_power_is_hermitian_and_has_the_right_bandwidth(n):
    j = 6
    op = ladder_power_sum(j, n)
    assert np.allclose(op, op.T)
    # Non-zero only where |K - K'| == n.
    ks = np.arange(-j, j + 1)
    sep = np.abs(ks[:, None] - ks[None, :])
    assert np.allclose(op[sep != n], 0.0)
    assert np.any(op[sep == n] != 0.0)


def test_ladder_power_zero_is_the_identity():
    assert np.allclose(ladder_power_sum(4, 0), np.eye(9))


@pytest.mark.parametrize("n", [-2, 1, 3])
def test_ladder_power_rejects_odd_and_negative(n):
    with pytest.raises(ValueError):
        ladder_power_sum(3, n)


# ── A is unchanged ──────────────────────────────────────────────────────────

def test_the_a_reduction_did_not_move():
    """Generalising the solve must not perturb the validated reduction."""
    abc, tau = _water_tau()
    got = reduction_from_tau(abc, tau, reduction="A")
    for name, expected in (("DJ", 38.18), ("DJK", -235.39), ("DK", 899.03)):
        assert got[name] == pytest.approx(expected, rel=2e-3)


# ── relations between the two reductions ────────────────────────────────────

def test_d1_equals_minus_delta_j():
    """The one quartic relation between the reductions that is exact.

    A carries -2 delta_J P^2 (Px^2 - Py^2) and S carries +d1 P^2 (P+^2 + P-^2),
    and P+^2 + P-^2 = 2 (Px^2 - Py^2), so the same operator is being fitted with
    opposite sign. Getting this wrong is the most likely single error in the S
    operator set, and it is invisible in the residual.
    """
    abc, tau = _water_tau()
    a = reduction_from_tau(abc, tau, reduction="A")
    s = reduction_from_tau(abc, tau, reduction="S")
    assert s["d1"] == pytest.approx(-a["delta_J"], rel=1e-6)


@pytest.mark.parametrize("reduction,keys", [("A", ("delta_J", "delta_K")),
                                            ("S", ("d1", "d2"))])
def test_a_symmetric_top_has_no_asymmetry_constants(reduction, keys):
    """With B == C there is no asymmetry for those operators to describe."""
    _, tau = _water_tau()
    got = reduction_from_tau(_SYMMETRIC, tau, reduction=reduction)
    for k in keys:
        assert abs(got[k]) < 1e-6


def test_both_reductions_shift_a_b_and_c():
    """The indeterminacy has to go somewhere, and it goes into A, B and C."""
    abc, tau = _water_tau()
    for reduction in ("A", "S"):
        got = reduction_from_tau(abc, tau, reduction=reduction)
        moved = [abs(got[k] - v) for k, v in zip("ABC", abc)]
        assert max(moved) > 1.0


# ── the reduced form must reproduce what it was fitted to ───────────────────

@pytest.mark.parametrize("reduction", ["A", "S"])
def test_fitted_parameters_round_trip_through_their_own_operator(reduction):
    """levels_from_reduction and the design matrix must describe one operator.

    If they drifted apart, a fitted parameter set would not mean the Hamiltonian
    it was fitted for, and every constant would be quietly wrong while the fit
    still looked converged.
    """
    abc, tau = _water_tau()
    params = reduction_from_tau(abc, tau, reduction=reduction)
    got = levels_from_reduction(abc, params, reduction=reduction)
    want = levels_from_tau(abc, tau)
    resid = float(np.sqrt(np.mean((got - want) ** 2)))
    assert resid == pytest.approx(
        reduction_residual_mhz(abc, tau, reduction=reduction), rel=1e-9)
    # And it is a fit, not a coincidence: the mismatch is small against the
    # levels themselves.
    assert resid < 1e-3 * float(np.max(np.abs(want)))


@pytest.mark.parametrize("reduction", ["A", "S"])
def test_constants_are_stable_against_how_many_levels_the_solve_sees(reduction):
    abc, tau = _water_tau()
    low = reduction_from_tau(abc, tau, reduction=reduction, jmax=4)
    high = reduction_from_tau(abc, tau, reduction=reduction, jmax=8)
    for k in REDUCTION_PARAMS[reduction]:
        if k in "ABC":
            continue
        assert low[k] == pytest.approx(high[k], rel=0.25), k


# ── the reason S exists, rediscovered ───────────────────────────────────────

def test_s_beats_a_by_orders_of_magnitude_on_a_near_prolate_top():
    """A is ill-conditioned as B approaches C; S is not. Nothing here was told so.

    Measured: A leaves a residual around 1.8e6 MHz against S's 2.3e2, and A's
    delta_K blows up to ~1.5e4 while S's d2 stays near 3.
    """
    _, tau = _water_tau()
    resid_a = reduction_residual_mhz(_PROLATE, tau, reduction="A")
    resid_s = reduction_residual_mhz(_PROLATE, tau, reduction="S")
    assert resid_s < resid_a / 100.0

    a = reduction_from_tau(_PROLATE, tau, reduction="A")
    s = reduction_from_tau(_PROLATE, tau, reduction="S")
    assert abs(a["delta_K"]) > 100.0 * abs(s["d2"])


def test_a_is_the_right_choice_for_a_strongly_asymmetric_top():
    """Water is fit in A in the literature, and the residual agrees."""
    abc, tau = _water_tau()
    assert (reduction_residual_mhz(abc, tau, reduction="A")
            <= reduction_residual_mhz(abc, tau, reduction="S"))


@pytest.mark.parametrize("abc", [_PROLATE, _OBLATE, _ASYMMETRIC])
def test_best_reduction_returns_the_lower_residual_of_the_two(abc):
    _, tau = _water_tau()
    label, params, resid = best_reduction(abc, tau)
    assert label in REDUCTION_PARAMS
    assert set(params) == set(REDUCTION_PARAMS[label])
    both = [reduction_residual_mhz(abc, tau, reduction=r)
            for r in REDUCTION_PARAMS]
    assert resid == pytest.approx(min(both), rel=1e-9)


def test_unknown_reduction_is_rejected_rather_than_defaulted():
    abc, tau = _water_tau()
    with pytest.raises(ValueError, match="reduction"):
        reduction_from_tau(abc, tau, reduction="B")


# ── the sextic block ────────────────────────────────────────────────────────

@pytest.mark.parametrize("reduction", ["A", "S"])
def test_sextic_order_adds_seven_parameters(reduction):
    from backend.spectral.cd_reduction import reduction_params
    quartic = reduction_params(reduction, 4)
    sextic = reduction_params(reduction, 6)
    assert len(quartic) == 8
    assert len(sextic) == 15
    assert sextic[:8] == quartic


@pytest.mark.parametrize("reduction", ["A", "S"])
def test_sextic_reduces_the_level_residual(reduction):
    """The point of the block: a quartic-only form cannot span the tau levels.

    Measured on water at jmax=8, A goes 762.8 -> 482.6 MHz and S 811.7 -> 371.7.
    """
    abc, tau = _water_tau()
    quartic = reduction_residual_mhz(abc, tau, reduction=reduction, jmax=8, order=4)
    sextic = reduction_residual_mhz(abc, tau, reduction=reduction, jmax=8, order=6)
    assert sextic < 0.8 * quartic


@pytest.mark.parametrize("reduction,keys", [("A", ("phi_J", "phi_JK", "phi_K")),
                                            ("S", ("h1", "h2", "h3"))])
def test_a_symmetric_top_has_no_sextic_asymmetry_constants(reduction, keys):
    _, tau = _water_tau()
    got = reduction_from_tau(_SYMMETRIC, tau, reduction=reduction, order=6)
    for k in keys:
        assert abs(got[k]) < 1e-9


@pytest.mark.parametrize("reduction", ["A", "S"])
def test_sextic_parameters_round_trip_through_their_own_operator(reduction):
    abc, tau = _water_tau()
    params = reduction_from_tau(abc, tau, reduction=reduction, order=6)
    got = levels_from_reduction(abc, params, reduction=reduction, order=6)
    want = levels_from_tau(abc, tau)
    resid = float(np.sqrt(np.mean((got - want) ** 2)))
    assert resid == pytest.approx(
        reduction_residual_mhz(abc, tau, reduction=reduction, order=6), rel=1e-9)


def test_sextic_constants_are_much_smaller_than_quartic_ones():
    """Sixth order in P against fourth: on water the ratio is about 1e-4.

    A sextic constant coming out comparable to a quartic one would mean the
    design matrix is degenerate and the solve has split one physical effect
    across two columns.
    """
    abc, tau = _water_tau()
    p = reduction_from_tau(abc, tau, reduction="A", order=6)
    quartic = max(abs(p[k]) for k in ("DJ", "DJK", "DK"))
    sextic = max(abs(p[k]) for k in ("Phi_J", "Phi_JK", "Phi_KJ", "Phi_K"))
    assert sextic < 1e-2 * quartic


def test_including_sextic_does_not_rescue_the_quartic_constants():
    """A hypothesis worth recording as falsified.

    The quartic-only fit leaves a 139.8 MHz level residual, which suggested the
    quartic constants were absorbing sextic-shaped error -- and DJK is 36% off
    experiment, by far the worst of the three. Adding the sextic block cuts the
    residual by a third and moves DJK from -36.1% to -36.8%: slightly further
    away, and DJ and DK by a tenth of a percent.

    So DJK's error is force-field error, not reduction contamination, exactly as
    compute_cd_constants already claims (the analytic PES's bend sits 4.4% below
    the experimental harmonic frequency and DJK is the most bend-sensitive of
    the three). Sextic constants are worth having on their own merits; they are
    not a fix for the quartic ones.
    """
    abc, tau = _water_tau()
    q = reduction_from_tau(abc, tau, reduction="A", jmax=6, order=4)
    s = reduction_from_tau(abc, tau, reduction="A", jmax=6, order=6)
    for k in ("DJ", "DJK", "DK"):
        assert s[k] == pytest.approx(q[k], rel=0.02), k


@pytest.mark.parametrize("order", [4, 6])
def test_unknown_order_is_rejected(order):
    from backend.spectral.cd_reduction import reduction_params
    assert len(reduction_params("A", order)) in (8, 15)
    with pytest.raises(ValueError, match="order"):
        reduction_params("A", 8)
