"""Auto-derived systematic sigmas: verified, and on by default since D-165.

The B0-versus-Be model gap is the dominant error term in this engine, and
until now a user who quoted honest measurement precision got intervals that
omitted it, with only a printed warning to say so. Deriving it was tried
before and reverted because the propagation produced bond widths of 47
Angstrom and an angle width of 10975 degrees -- which turned out to be the
sigma-units mismatch fixed in test_systematic_sigma_units, not a problem with
deriving it.

With that fixed the derived term is sensible (0.0018 Angstrom, 0.32 degrees
on water with the flat estimate, D-67) and it repairs a real under-coverage:
the old angle interval of 0.0273 degrees was NINE TIMES too narrow to reach
the 0.246-degree angle error the engine itself makes.

It was kept off by default until the user decided (decision page item 7,
D-165): the reason these tests used to pin -- an interval that narrowed when
more measurement error was admitted -- no longer occurs
(test_the_interval_is_monotonic_in_sigma), and turning it on changes every
reported interval without a stated model error, which was the user's call.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = [835840.29, 435351.72, 278138.70]
B0_D2O = [462324.0, 218038.0, 145258.0]
R_OH = 0.95785
#: Water angle error the engine itself makes, measured against the accepted
#: structure earlier in development: -0.246 deg at base, +0.182 corrected.
KNOWN_ANGLE_ERROR_DEG = 0.246


def _coords(r=R_OH, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(auto, sigma_mhz=0.2, **kw):
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [sigma_mhz] * 3}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),
                            ("D2-16O", MASSES_D2O, B0_D2O))]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, auto_systematic_sigma=auto, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
        unc = opt.geometry_uncertainty()
    return opt, unc


def _widths(unc):
    return (max(v for k, v in unc.items() if k.count("-") == 1),
            max(v for k, v in unc.items() if k.count("-") == 2))


def test_on_by_default_and_off_on_request():
    """On by default since D-165 (the user's call, decision page item 7): a
    species with no stated systematic sigma gets the flat estimate, so a raw
    B0-as-Be fit does not quote measurement precision as its uncertainty.
    auto_systematic_sigma=False still gives measurement precision only."""
    opt, _unc = _fit(auto=False)
    assert opt.auto_systematic_sigma is False
    assert opt.spectral.isotopologues[0].get("sigma_systematic_constants") is None
    isos = [{"name": "H2-16O", "masses": MASSES_H2O,
             "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}]
    with contextlib.redirect_stdout(io.StringIO()):
        default = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=3, quantum_prior_sigma_ang=0.02)
    assert default.auto_systematic_sigma is True
    sysg = np.asarray(default.spectral.isotopologues[0]["sigma_systematic_constants"],
                      dtype=float)
    # 1% of A, 0.5% of B and C (D-67).
    assert np.allclose(sysg, np.asarray(B0_H2O) * [0.01, 0.005, 0.005])


def test_opt_in_derives_a_systematic_without_touching_the_quoted_sigma():
    opt, _unc = _fit(auto=True)
    iso = opt.spectral.isotopologues[0]
    sysg = np.asarray(iso["sigma_systematic_constants"], dtype=float)
    raw = np.asarray(iso["sigma_constants"], dtype=float)
    assert np.all(sysg > 0.0)
    # The user's measurement precision is left exactly as quoted.
    assert np.allclose(raw, 0.2)
    # And the derived gap is the flat relative default -- 1% of A, 0.5% of B
    # and C -- not a rescaling of the quoted sigma (decisions D-67). It used to
    # be the inertial-defect estimate, 8.5% / 4.4% / 2.8% here (H2-16O):
    # against the B3LYP VPT2 gap 8.7x / 6.3x / 1.06x for H2-16O, and
    # 0.008-0.54x on B and C of every species of the other benchmark
    # molecules.
    frac = sysg / np.asarray(B0_H2O, dtype=float)
    assert np.allclose(frac, (0.010, 0.005, 0.005), rtol=1e-12, atol=0.0), frac


def test_derived_widths_are_physically_sensible():
    """47 Angstrom and 10975 degrees caused the original revert."""
    _opt, unc = _fit(auto=True)
    bond, angle = _widths(unc)
    assert bond < R_OH, f"bond width {bond:.4f} Ang exceeds the bond"
    assert angle < 180.0, f"angle width {angle:.1f} deg exceeds a half turn"
    assert bond > 0.0 and angle > 0.0


def test_it_repairs_the_angle_under_coverage():
    """The point of the change: reach the error the engine actually makes."""
    _o0, unc_off = _fit(auto=False)
    _o1, unc_on = _fit(auto=True)
    _b_off, a_off = _widths(unc_off)
    _b_on, a_on = _widths(unc_on)
    assert a_off < KNOWN_ANGLE_ERROR_DEG, (
        "premise broken: the old interval already covered the known error")
    assert a_on > KNOWN_ANGLE_ERROR_DEG, (
        f"derived interval {a_on:.3f} deg still does not reach the "
        f"{KNOWN_ANGLE_ERROR_DEG} deg error the engine makes")


def test_the_model_error_dominates_both_coordinates():
    """With the derived systematic on, it is the larger term for the bond AND
    the angle: measured 8.8x and 11.8x their statistical widths with the flat
    estimate (60x and 87x with the inertial-defect one it replaced, which ran
    6-9x large on water's A and B against the VPT2 gaps).

    This replaces "the bond is barely affected" (asserted < 2x). The bond only
    looked conservative because its statistical width carried the
    antisymmetric stretch's prior-only variance -- 0.0119 Ang on a direction
    the imposed C2v holds fixed. Its honest statistical width is 0.0002 Ang,
    and against that a B0-versus-Be model error of this size is decisive.
    """
    _o0, unc_off = _fit(auto=False)
    _o1, unc_on = _fit(auto=True)
    b_off, a_off = _widths(unc_off)
    b_on, a_on = _widths(unc_on)
    assert (b_on / b_off) > 5.0
    assert (a_on / a_off) > 5.0


def test_fitted_geometry_does_not_move():
    """This is a sigma on the REPORT. If it moved the answer it would be a
    different change entirely, and would need benchmark re-validation."""
    o_off, _ = _fit(auto=False)
    o_on, _ = _fit(auto=True)
    assert np.array_equal(o_off.coords, o_on.coords)


def test_explicit_systematic_is_not_overwritten():
    isos = [{"name": n, "masses": m, "obs_constants": list(b),
             "sigma_constants": [0.2] * 3,
             "sigma_systematic_constants": list(0.01 * np.asarray(b))}
            for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O),)]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            max_iter=3, quantum_prior_sigma_ang=0.02,
            auto_systematic_sigma=True)
    got = np.asarray(opt.spectral.isotopologues[0]["sigma_systematic_constants"])
    assert np.allclose(got, 0.01 * np.asarray(B0_H2O))


def test_the_interval_is_monotonic_in_sigma():
    """What used to keep this off by default, and what has changed.

    With sigma_floor_rel off, so the quoted sigma reaches the weighting, the
    inertial-defect estimate (to D-67) made admitting 1000x more measurement
    uncertainty NARROW the reported bond width -- by the time it was
    replaced only just, 0.01600 -> 0.01590 Angstrom (the angle widened, 2.565
    -> 2.569 deg); this docstring once recorded 0.0455 -> 0.0180. An
    uncertainty that shrinks when you admit more error cannot be published,
    and this test pinned that as the reason auto stayed off. That
    estimate ran 6-9x large on water's A and B; with the flat one the same
    1000x widens the bond, 0.00199 -> 0.00230 Angstrom (angle 0.320 -> 0.343
    deg).

    Whether to make the derivation the default is a separate decision: it
    would change every reported interval (backlog F13).
    """
    _o_t, unc_tight = _fit(auto=True, sigma_mhz=2.0, sigma_floor_rel=0.0)
    _o_l, unc_loose = _fit(auto=True, sigma_mhz=2000.0, sigma_floor_rel=0.0)
    b_tight, a_tight = _widths(unc_tight)
    b_loose, a_loose = _widths(unc_loose)
    assert b_loose > b_tight and a_loose > a_tight, (b_tight, b_loose, a_tight, a_loose)


def test_with_the_floor_on_it_is_monotonic():
    """The default floor masks the problem, which is exactly why relying on
    it would be the wrong reason to ship this on."""
    widths = []
    for sig in (2.0, 20.0, 200.0, 2000.0):
        _o, unc = _fit(auto=True, sigma_mhz=sig)
        widths.append(_widths(unc)[0])
    for lo, hi in zip(widths, widths[1:]):
        assert hi >= lo - 1e-9, widths


def test_the_caller_s_isotopologues_are_not_written_into():
    """The fills work on copies (T-81 review): writing into the caller's dicts
    made a second optimizer built from the same list -- the runner's dihedral
    probe, then the fit -- see the first one's estimate as stated."""
    isos = [{"name": "H2-16O", "masses": MASSES_H2O,
             "obs_constants": list(B0_H2O), "sigma_constants": [0.2] * 3}]
    kw = {"elems": ["O", "H", "H"], "coords": _coords(), "isotopologues": isos,
          "quantum_backend": "analytic_water", "use_autoconfig": False, "max_iter": 3,
          "quantum_prior_sigma_ang": 0.02}
    with contextlib.redirect_stdout(io.StringIO()):
        first = MolecularOptimizer(**kw)
        second = MolecularOptimizer(**kw)
    assert "sigma_systematic_constants" not in isos[0]
    assert first.model_error_sources() == second.model_error_sources() == {"H2-16O": "estimated"}


def test_a_sigma_that_includes_the_model_error_gets_the_estimate_only_on_request():
    """sigma_constants_include_systematic without a stated split: the flat
    estimate (8358 MHz on A) would exceed the quoted 0.2 MHz and leave no
    measurement part, so the default does not fill it; an explicit True does."""
    def build(**extra):
        isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": list(B0_H2O),
                 "sigma_constants": [0.2] * 3, "sigma_constants_include_systematic": True}]
        with contextlib.redirect_stdout(io.StringIO()):
            return MolecularOptimizer(
                elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
                quantum_prior_sigma_ang=0.02, **extra)
    assert build().spectral.isotopologues[0].get("sigma_systematic_constants") is None
    assert build(auto_systematic_sigma=True).spectral.isotopologues[0].get(
        "sigma_systematic_constants") is not None



def test_the_estimate_does_not_move_a_corrected_fit():
    """A vibrational correction takes a stated model error out of the quoted
    sigma (compute_corrected_targets); the engine's estimate is not part of
    the quoted sigma, so it must not be taken out, or the default estimate
    moves the weights: water with a 5 MHz-sigma correction table and a
    2000 MHz quoted sigma moved 0.98 mA before (T-81 review)."""
    deltas = {"H2-16O": {"A": 34000.0, "B": -4500.0, "C": 5600.0},
              "D2-16O": {"A": 12000.0, "B": -1600.0, "C": 2400.0}}
    table = {n: {c: {"delta_mhz": d[c], "sigma_mhz": 5.0, "method": "VPT2"} for c in "ABC"}
             for n, d in deltas.items()}

    def fit(auto):
        isos = [{"name": n, "masses": m, "obs_constants": list(b), "sigma_constants": [2000.0] * 3}
                for n, m, b in (("H2-16O", MASSES_H2O, B0_H2O), ("D2-16O", MASSES_D2O, B0_D2O))]
        with contextlib.redirect_stdout(io.StringIO()):
            opt = MolecularOptimizer(
                elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                quantum_backend="analytic_water", use_autoconfig=False, max_iter=40,
                quantum_prior_sigma_ang=0.02, auto_systematic_sigma=auto,
                correction_table=table, correction_mode="user_only")
            return np.asarray(opt.run())

    np.testing.assert_array_equal(fit(True), fit(False))
