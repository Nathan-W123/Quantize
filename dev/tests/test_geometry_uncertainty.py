"""Posterior uncertainties on the fitted bond lengths and angles.

For a molecule with no accepted structure -- the case this engine exists for --
the uncertainty is the result, not a footnote: there is nothing to take a
difference against. So it has to behave sensibly under the things that should
move it, and these tests pin exactly that rather than any particular value.

What is deliberately *not* asserted is that the uncertainty is accurate in the
sense of bracketing the true error. It is a precision propagated from the stated
uncertainties on the constants, and it knows nothing about the B0-versus-Be
offset that dominates the systematic error here. Calibrating it is a separate
piece of work; see the note in `geometry_uncertainty`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = [835840.29, 435351.72, 278138.70]
B0_D2O = [462324.0, 218038.0, 145258.0]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _opt(sigma_mhz=0.2, species=1, prior_sigma=0.02, **kw):
    isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
             "sigma_constants": [sigma_mhz] * 3}]
    if species > 1:
        isos.append({"name": "D2-16O", "masses": MASSES_D2O,
                     "obs_constants": B0_D2O, "sigma_constants": [sigma_mhz] * 3})
    return MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=prior_sigma, **kw)


def _run(opt):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt.geometry_uncertainty()


def test_reports_a_finite_positive_width_for_every_coordinate():
    unc = _run(_opt())
    assert unc, "no uncertainties returned"
    assert set(unc) >= {"O1-H2", "O1-H3", "H2-O1-H3"}
    for name, s in unc.items():
        assert np.isfinite(s) and s > 0, f"{name} got {s}"


def test_looser_constants_give_wider_bonds():
    """The whole point of propagating: worse data must show as worse numbers.

    The relative sigma floor is switched off here. It clamps every sigma to at
    least 0.1% of its own constant, which for water's A of 836 GHz is 836 MHz --
    so both a 0.02 and a 20 MHz input land on the same floored value and the
    propagation would look broken when it is the floor doing its job. The floor
    itself is exercised by the test below.
    """
    tight = _run(_opt(sigma_mhz=0.02, sigma_floor_rel=0.0))
    loose = _run(_opt(sigma_mhz=20.0, sigma_floor_rel=0.0))
    assert loose["O1-H2"] > tight["O1-H2"], (
        f"loosening sigma 1000x did not widen the bond: "
        f"{tight['O1-H2']:.3e} -> {loose['O1-H2']:.3e}"
    )


def test_the_relative_sigma_floor_clamps_optimistic_inputs():
    """Quoted precision on a published constant overstates what it constrains.

    The floor exists so that a constant quoted to six figures cannot claim six
    figures of structural information, since the structural model behind it is
    not that good. Two sigmas far below the floor must therefore give the same
    answer -- that is the floor working, not the propagation failing.
    """
    a = _run(_opt(sigma_mhz=0.02))
    b = _run(_opt(sigma_mhz=20.0))
    assert a["O1-H2"] == pytest.approx(b["O1-H2"], rel=1e-9)
    unfloored = _run(_opt(sigma_mhz=0.02, sigma_floor_rel=0.0))
    assert unfloored["O1-H2"] < a["O1-H2"], (
        "removing the floor should let the tight sigma report a tighter bond"
    )


def test_more_isotopologues_narrow_the_fit():
    one = _run(_opt(species=1))
    two = _run(_opt(species=2))
    assert two["O1-H2"] <= one["O1-H2"] * 1.01, (
        f"adding a species widened the bond: {one['O1-H2']:.3e} -> {two['O1-H2']:.3e}"
    )


def test_a_tighter_prior_narrows_directions_the_data_cannot_see():
    """In a blind direction the posterior reports the prior, so it must track it."""
    loose = _run(_opt(prior_sigma=0.10))
    tight = _run(_opt(prior_sigma=0.002))
    assert tight["H2-O1-H3"] < loose["H2-O1-H3"], (
        f"prior width did not propagate: {loose['H2-O1-H3']:.3e} -> "
        f"{tight['H2-O1-H3']:.3e}"
    )


def test_symmetry_equivalent_bonds_get_equal_uncertainties():
    """Water's two O-H bonds are equal by C2v, so their widths must match."""
    unc = _run(_opt(species=2))
    a, b = unc["O1-H2"], unc["O1-H3"]
    assert abs(a - b) <= 1e-6 * max(a, b, 1e-12), f"{a:.6e} vs {b:.6e}"


def test_rigid_modes_do_not_leak_into_the_widths():
    """Translations and rotations are null in both blocks.

    Left in, they make the covariance singular and the propagated widths blow
    up. A finite answer here is the evidence they were removed.
    """
    unc = _run(_opt())
    assert all(s < 1.0 for s in unc.values()), (
        f"implausibly large widths suggest the rigid subspace was not removed: {unc}"
    )
