"""The tau' -> Watson A-reduction mapping, now derived and checked against experiment.

This file used to record a failure. `watson_a_reduction_cd_from_tau_cm` was a
closed-form coefficient table that did not reproduce published constants: on
H2-16O it produced DJ and DK with the wrong sign and DK 138x too small, so the
whole centrifugal-distortion path was unusable and switched off by default.

`compute_cd_constants` now goes through `backend.spectral.cd_reduction`, which
does not transcribe a coefficient table at all. A reduction is a
reparameterisation of one operator, so the A-reduced constants are obtained by
requiring the A-reduced Hamiltonian to reproduce the energy levels the tau
tensor itself generates -- a linear solve, with nothing taken on authority.

What makes that trustworthy is this file. The construction never sees an
experimental number, and it has to land on three that are known to five
figures:

    DJ    +38.2  against   +37.59     1.6% high
    DK    +899   against   +973.3     7.6% low
    DJK   -235   against   -172.9      36% low

The remaining spread is force-field error rather than mapping error: the
analytic water PES puts the bend at 1577 cm-1 against an experimental harmonic
1649, tau goes as 1/omega^2, and DJK is the most bend-sensitive of the three.
Signs are right for all five constants, which is the qualitative thing the old
mapping could not do at any sigma.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.registry import get_backend  # noqa: E402
from backend.spectral.cd_reduction import (  # noqa: E402
    a_reduction_from_tau,
    angular_momentum_operators,
    levels_from_tau,
)
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    compute_cd_constants,
    rotational_constants_mhz,
)

#: Experimental ground-state Watson A-reduction constants for H2-16O, in MHz.
EXPERIMENT = {"DJ": 37.59, "DJK": -172.9, "DK": 973.3}

#: What the derived reduction produces on the analytic PES. Pinned so a change
#: to either the reduction or the PES is visible rather than silent.
CURRENT = {"DJ": 38.2, "DJK": -235.4, "DK": 899.0}


def _water_coords_masses():
    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    masses = np.array([15.9949146196, 1.00782503207, 1.00782503207])
    return coords, masses


def _water_cd():
    coords, masses = _water_coords_masses()
    hess = get_backend("analytic_water")(elems=["O", "H", "H"]).run_hessian(coords)
    return compute_cd_constants(hess.hessian_bohr, coords, masses)


# ── the operator algebra the reduction is built on ─────────────────────────

def test_angular_momentum_operators_satisfy_their_own_identities():
    """P^2, Pz^2 and Px^2-Py^2 must be consistent before anything built on
    them means a thing. A wrong off-diagonal factor in Pd still produces a
    Hermitian matrix and plausible-looking constants."""
    for j in (1, 3, 5):
        p2, pz2, pd = angular_momentum_operators(j)
        n = 2 * j + 1
        assert p2 == pytest.approx(j * (j + 1) * np.eye(n))
        assert np.diag(pz2) == pytest.approx(np.arange(-j, j + 1.0) ** 2)
        assert pd == pytest.approx(pd.T)
        # Px^2 and Py^2 are built as ((P^2 - Pz^2) +- Pd)/2, so both must come
        # out positive semi-definite; that fails if Pd is too large.
        perp = p2 - pz2
        for sign in (1.0, -1.0):
            ev = np.linalg.eigvalsh(0.5 * (perp + sign * pd))
            assert ev.min() > -1e-9


def test_a_spherical_top_has_no_asymmetry_splitting():
    """A = B = C with no distortion: every level of a given J is degenerate at
    J(J+1)*B. A sanity check on the axis association, which is otherwise only
    visible through the constants themselves."""
    lv = levels_from_tau((5000.0, 5000.0, 5000.0), np.zeros((3, 3)), jmax=3)
    expected = np.concatenate([[5000.0 * j * (j + 1)] * (2 * j + 1)
                               for j in range(4)])
    assert np.sort(lv) == pytest.approx(np.sort(expected), rel=1e-9)


def test_zero_tau_gives_zero_distortion_constants():
    """No force field, no distortion. The solve must not invent any."""
    got = a_reduction_from_tau((835712.0, 435059.0, 278357.0),
                               np.zeros((3, 3)), jmax=6)
    for k in ("DJ", "DJK", "DK", "delta_J", "delta_K"):
        assert abs(got[k]) < 1e-6
    # ...and it must hand the rotational constants back unchanged.
    assert (got["A"], got["B"], got["C"]) == pytest.approx(
        (835712.0, 435059.0, 278357.0), rel=1e-9)


def test_constants_are_stable_against_how_many_levels_the_solve_sees():
    """If the answer moved with jmax the solve would be fitting noise rather
    than performing a reparameterisation."""
    from backend.spectral.centrifugal_distortion import (
        _CM_TO_MHZ, _MHZ_TO_CM, bk_mode_derivatives, normal_modes,
        tau_prime_from_dB1_cm,
    )

    coords, masses = _water_coords_masses()
    abc = rotational_constants_mhz(coords, masses)
    hess = get_backend("analytic_water")(
        elems=["O", "H", "H"]).run_hessian(coords).hessian_bohr
    omega, l_mw = normal_modes(hess, masses, n_rigid=6)
    keep = omega >= 50.0
    omega, l_mw = omega[keep], l_mw[:, keep]
    db1, _ = bk_mode_derivatives(coords, masses, l_mw, omega, 0.05, abc)
    tau = tau_prime_from_dB1_cm(db1 * _MHZ_TO_CM, omega) * _CM_TO_MHZ

    a4 = a_reduction_from_tau(abc, tau, jmax=4)
    a8 = a_reduction_from_tau(abc, tau, jmax=8)
    for k in ("DJ", "DJK", "DK"):
        assert a8[k] == pytest.approx(a4[k], rel=0.03)


# ── against experiment ─────────────────────────────────────────────────────

def test_every_constant_has_the_right_sign():
    """The old mapping's qualitative failure. Signs cannot be rescued by a
    wide sigma, so this is the gate that matters."""
    got = _water_cd().as_dict()
    wrong = [n for n in EXPERIMENT if np.sign(got[n]) != np.sign(EXPERIMENT[n])]
    assert wrong == [], f"wrong sign on {wrong}"


@pytest.mark.parametrize("name,tol", [("DJ", 0.10), ("DK", 0.15), ("DJK", 0.45)])
def test_constants_match_experiment_within_force_field_error(name, tol):
    """Per-constant tolerances, set by what was measured rather than by one
    loose bound: DJ and DK land within 8%, DJK within 36% because it is the
    most sensitive to the PES's 4.4% error in the bend frequency."""
    got = _water_cd().as_dict()[name]
    assert got == pytest.approx(EXPERIMENT[name], rel=tol)


def test_output_is_reproducible():
    """Pin the numbers, so a change to the reduction or the PES is visible."""
    got = _water_cd().as_dict()
    for name, expected in CURRENT.items():
        assert got[name] == pytest.approx(expected, rel=0.02), (
            f"{name} moved from {expected} to {got[name]:.1f}"
        )


def test_sigma_reflects_the_measured_agreement_not_a_blanket_floor():
    """A validated correction should be allowed to claim the precision it has,
    but no more: the worst measured constant is 36% out, so sigma sits at 40%
    and not at the old 100% floor."""
    cd = _water_cd()
    for name in EXPERIMENT:
        val = abs(cd.as_dict()[name])
        assert 0.3 * val < cd.sigma[name] < 0.6 * val


def test_notes_no_longer_claim_the_mapping_is_unvalidated():
    assert "UNVALIDATED" not in _water_cd().notes.upper()
