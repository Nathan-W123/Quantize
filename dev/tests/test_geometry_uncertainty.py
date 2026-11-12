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

    The sigma floor is off so the stated sigma reaches the weighting. Any
    sigma works now: the covariance is inverted over the physical directions,
    whose block has condition number 16.4 here at every sigma, so nothing is
    refused down to 0.02 MHz (see test_laplace_physical_subspace). Inverting
    the full 3N matrix refused everything at or below 2 MHz.
    """
    tight = _run(_opt(sigma_mhz=20.0, sigma_floor_rel=0.0))
    loose = _run(_opt(sigma_mhz=2000.0, sigma_floor_rel=0.0))
    assert loose["O1-H2"] > tight["O1-H2"], (
        f"loosening sigma 1000x did not widen the bond: "
        f"{tight['O1-H2']:.3e} -> {loose['O1-H2']:.3e}"
    )


def test_the_reported_width_is_data_dominated_here():
    """10x looser data gives a 10x wider bond, until the prior takes over.

    This test used to assert the opposite -- that 1000x worse data moved the
    bond by under a percent, because the width was "set mostly by the prior".
    That prior-set width belonged to the antisymmetric stretch, which the
    imposed C2v holds fixed, so it was never a width the fit had. On the two
    directions the fit can move along, the data is in charge:

        sigma (MHz)    O1-H2 width (Ang)    step ratio
        20             1.333e-05
        200            1.333e-04            10.00
        2000           1.325e-03             9.94
        20000          8.745e-03             6.60   <- prior starts to bite

    which is the linear scaling a data-dominated Laplace width must show.
    """
    w20 = _run(_opt(sigma_mhz=20.0, sigma_floor_rel=0.0))["O1-H2"]
    w200 = _run(_opt(sigma_mhz=200.0, sigma_floor_rel=0.0))["O1-H2"]
    w2000 = _run(_opt(sigma_mhz=2000.0, sigma_floor_rel=0.0))["O1-H2"]
    assert w200 / w20 == pytest.approx(10.0, rel=0.02)
    assert w2000 / w200 == pytest.approx(10.0, rel=0.02)


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
    # And a sigma above the floor is NOT clamped, so the floor is doing
    # something rather than flattening everything.
    above = _run(_opt(sigma_mhz=2000.0))
    assert above["O1-H2"] > a["O1-H2"]


def test_an_indefinite_covariance_is_refused_outright():
    """The guard, tested on the guard rather than through the numerics.

    An earlier version of this test provoked the bad covariance by running a
    real fit with sigma quoted at 0.02 MHz and the floor off, then asserted
    what came back. That is not portable: on Python 3.11 / numpy 2.4.6 the
    coordinate comes back nan, on a 3.12 / numpy 2.5.4 venv the whole result
    is None, and on the CI runner the same collapsed matrix yields a
    plausible-looking 0.0119 because its variance happens to land positive.
    Three machines, three answers, none of them meaningful -- and a test
    pinned to any one of them fails on the other two. That is the same trap
    this file already fell into once.

    So the guard is driven directly: hand geometry_uncertainty a covariance
    that is definitely indefinite and require it to refuse. Platform
    independent, and it tests the thing that is supposed to protect the
    caller.
    """
    opt = _opt()
    _run(opt)
    good = opt.geometry_uncertainty()
    assert good is not None and np.isfinite(good["O1-H2"])

    # Same optimum, but with an unmistakably indefinite covariance -- built on
    # the directions the fit can move along, so that the projection which
    # removes rigid-body and symmetry-forbidden directions cannot discard the
    # negative part. (Placing it on oxygen's in-plane x, as this first did,
    # puts it on the antisymmetric stretch under the imposed C2v, where it is
    # projected away and the guard never sees it.)
    n = opt.coords.size
    basis = opt.physical_basis()
    assert basis.shape[1] == 2
    bad = basis @ np.diag([-1.0, 1e-4]) @ basis.T + 1e-4 * np.eye(n)
    opt.laplace_approximation = lambda *a, **k: (bad, None)
    assert opt.geometry_uncertainty() is None, (
        "an indefinite covariance must yield no widths at all, rather than "
        "numbers taken from it"
    )


def test_indefiniteness_on_a_direction_the_model_fixes_is_projected_away():
    """The other half: a negative variance on a symmetry-forbidden direction
    says nothing about the widths of coordinates the fit can move along.

    Pinned because it is a choice. The guard runs on the covariance the
    widths are taken from, after the forbidden directions are gone; a
    defect confined to a direction the model holds fixed is not a defect in
    what is reported.
    """
    opt = _opt()
    _run(opt)
    n = opt.coords.size
    bad = np.eye(n) * 1e-4
    bad[0, 0] = -1.0                      # oxygen's in-plane x: the B2 stretch
    opt.laplace_approximation = lambda *a, **k: (bad, None)
    unc = opt.geometry_uncertainty()
    assert unc is not None and np.isfinite(unc["O1-H2"])


def test_round_off_indefiniteness_is_tolerated():
    """The guard must not fire on a covariance that is merely imperfect.

    The healthy default path has a negative minimum eigenvalue too, at about
    -2e-16 of its maximum. A guard with an absolute threshold, or a ratio
    threshold set too tight, would reject every real result.
    """
    opt = _opt()
    _run(opt)
    n = opt.coords.size
    nearly = np.eye(n) * 1e-4
    nearly[0, 0] = -1e-4 * 1e-12          # min/max = -1e-12, round-off scale
    opt.laplace_approximation = lambda *a, **k: (nearly, None)
    unc = opt.geometry_uncertainty()
    assert unc is not None, "round-off indefiniteness must still be usable"


def test_the_floored_default_is_unaffected_by_that_guard():
    """The guard must not fire on ordinary round-off.

    The default path's covariance also has a negative minimum eigenvalue, at
    -2.7e-20 against a positive variance of 1.4e-4 -- round-off, 16 orders of
    magnitude below the terms being summed. A guard with an absolute threshold
    would either miss the real failure or reject this, which is why the
    tolerance is relative to the term scale.
    """
    unc = _run(_opt(sigma_mhz=0.02))
    assert np.isfinite(unc["O1-H2"]) and unc["O1-H2"] > 0


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


def test_class_floor_widens_only_its_own_class():
    """The floor says: the fraction of a coordinate resting on theory cannot be
    quoted tighter than the measured class error of the level of theory.

    A single measured constant against three internal DOF leaves the angle
    mostly theory-determined, so an "angle" floor far above the propagated
    width must widen the angle -- and must not touch the bonds, because floors
    are per class and "angle" is not "X-H".
    """
    def one_constant_opt(**kw):
        return MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(),
            isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                            "obs_constants": B0_H2O[:1],
                            "sigma_constants": [0.2],
                            "component_indices": [0]}],
            quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
            quantum_prior_sigma_ang=0.02, **kw)

    base = one_constant_opt()
    u0 = _run(base)
    blind = 1.0 - base.data_support()["H2-O1-H3"]
    assert blind > 0.05, f"angle unexpectedly data-determined (blind={blind:.3f})"

    floored = one_constant_opt(prior_class_sigma={"angle": 30.0})
    u1 = _run(floored)
    assert u1["H2-O1-H3"] >= blind * 30.0, (
        f"angle floor did not bind: {u1['H2-O1-H3']:.3f} < {blind * 30.0:.3f}"
    )
    assert u1["O1-H2"] == pytest.approx(u0["O1-H2"], rel=1e-6), (
        "an angle-class floor moved a bond width"
    )
    _, p_sig = floored.sigma_split()["H2-O1-H3"]
    assert p_sig >= blind * 30.0 * 0.999, (
        "sigma_split prior part does not reflect the floor"
    )


def test_class_floor_defers_to_data():
    """Where the data determines a coordinate, the theory floor has no claim:
    the widths must be unchanged up to the coordinate's tiny blind fraction."""
    base = _opt(species=2)
    u0 = _run(base)
    support = base.data_support()
    floored = _opt(species=2,
                   prior_class_sigma={"X-H": 0.5, "angle": 30.0})
    u1 = _run(floored)
    for name in u0:
        blind = 1.0 - support[name]
        cap = np.hypot(u0[name], blind * (30.0 if name.count("-") == 2 else 0.5))
        assert u0[name] <= u1[name] <= cap * 1.01, (
            f"{name}: base {u0[name]:.4g}, floored {u1[name]:.4g}, "
            f"cap {cap:.4g} (blind {blind:.3f})"
        )
