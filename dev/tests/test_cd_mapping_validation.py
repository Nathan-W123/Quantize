"""The tau' -> Watson A-reduction mapping does not reproduce experiment.

`watson_a_reduction_cd_from_tau_cm` is documented as unvalidated. This measures
how unvalidated, so the claim rests on numbers rather than on a comment, and so
that anyone who fixes the mapping sees these assertions fail and knows to
update them.

Water is the sharpest available test: its distortion constants are large and
precisely known, and the analytic PES removes force-field error, leaving only
the mapping itself under test.

Measured against the experimental ground-state A-reduction constants for
H2-16O the mapping gets:

    DJ    -66.9  against   +37.6   -- wrong sign
    DJK   -18.9  against  -172.9   -- right sign, 9x too small
    DK     -7.0  against  +973.3   -- wrong sign, 138x too small

Two of three have the wrong sign, so this is not an order-of-magnitude
estimate that could be rescued with a wider sigma; it is qualitatively wrong.
The path stays off by default (`harmonic_cd_from_hessian=False`) and warns when
switched on.
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
from backend.spectral.centrifugal_distortion import compute_cd_constants  # noqa: E402

#: Experimental ground-state Watson A-reduction constants for H2-16O, in MHz.
EXPERIMENT = {"DJ": 37.59, "DJK": -172.9, "DK": 973.3}

#: What the current mapping produces, to be updated if the mapping is fixed.
CURRENT = {"DJ": -66.9, "DJK": -18.9, "DK": -7.0}


def _water_cd():
    r, theta = 0.95785, np.radians(104.508)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
        [-r * np.sin(theta / 2), r * np.cos(theta / 2), 0.0],
    ])
    masses = np.array([15.9949146196, 1.00782503207, 1.00782503207])
    hess = get_backend("analytic_water")(elems=["O", "H", "H"]).run_hessian(coords)
    return compute_cd_constants(hess.hessian_bohr, coords, masses)


def test_mapping_is_reproducible():
    """Pin current output, so a change to the mapping is visible."""
    got = _water_cd().as_dict()
    for name, expected in CURRENT.items():
        assert got[name] == pytest.approx(expected, abs=0.5), (
            f"{name} changed from {expected} to {got[name]:.1f}. If the mapping "
            f"was fixed, update CURRENT and re-check the xfails below."
        )


def test_mapping_still_declares_itself_unvalidated():
    """The notes must keep saying so while the numbers say so."""
    assert "UNVALIDATED" in _water_cd().notes


def test_sigma_is_floored_at_one_hundred_percent():
    """An unvalidated correction must not be able to claim precision."""
    cd = _water_cd()
    for name in ("DJ", "DJK", "DK"):
        assert cd.sigma[name] >= abs(cd.as_dict()[name]), (
            f"sigma on {name} is smaller than the value itself, which would let "
            f"an unvalidated correction pull a fit"
        )


@pytest.mark.parametrize("name", ["DJ", "DJK", "DK"])
@pytest.mark.xfail(strict=True, reason="tau' -> A-reduction mapping is wrong")
def test_mapping_would_match_experiment_if_correct(name):
    """The assertion a correct mapping should satisfy. Fails today, by design.

    Strict, so that fixing the mapping turns these red-to-green and forces the
    surrounding documentation to be revisited.
    """
    got = _water_cd().as_dict()[name]
    assert got == pytest.approx(EXPERIMENT[name], rel=0.5)


def test_two_of_three_constants_have_the_wrong_sign():
    """Records the specific failure: this is not a magnitude problem."""
    got = _water_cd().as_dict()
    wrong = [n for n in ("DJ", "DJK", "DK")
             if np.sign(got[n]) != np.sign(EXPERIMENT[n])]
    assert set(wrong) == {"DJ", "DK"}, (
        f"expected DJ and DK to have the wrong sign, got {wrong}"
    )
