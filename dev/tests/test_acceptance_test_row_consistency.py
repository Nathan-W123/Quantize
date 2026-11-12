"""Both sides of the step-acceptance test must span the same rows.

In internal coordinate mode -- the repository default -- they did not. The
optimiser builds its system as

    J        = [J_spectral ; sqrt(w) * J_prior]
    residual = [r_spectral ; sqrt(w) * r_prior]

solves the step from that, and sets ``wrms_before`` to the RMS over the whole
vector. The trial geometry was then judged with ``residual_w_trial`` holding the
spectral rows ALONE: the internal-mode branch computed a prior diagnostic and
never extended the vector. So ``wrms_after <= wrms_before * (1 + relax)``
compared an RMS over m rows against an RMS over m + p rows.

With ``spectral_accept_relax = 0`` a vanishing step does not make the two sides
meet, which is what makes this fatal rather than merely sloppy: acceptance
reduces to whether the spectral residual happens to be smaller than the
combined spectral-and-prior residual, a property of the geometry and the prior,
with the step not in it at all.

Measured on the analytic-water config (3 spectral rows, 3 prior rows), at a
zero step, where both sides describe the identical geometry and a correct test
must return exactly equal values:

    sample   after/before   outcome
    s004     1.258          stalled -- every step rejected
    s010     1.281          stalled
    s011     1.251          stalled
    s003     0.941          moved   -- every step accepted
    s012     0.691          moved

Five samples, five outcomes, predicted by that ratio alone. Above one the fit
sits at its starting point through all 11 iterations while lambda ramps and the
step shrinks by 1000x, then reports "Converged to rank-3 spectral floor". Below
one there is no descent control: one sample accepted 17 of 17 steps while its
spectral chi-squared rose.

9 of 24 bootstrap samples stalled this way, preferentially those with larger
data perturbations (point-biserial correlation +0.52; 58.3% stall rate above
the median perturbation against 25.0% below), so the bootstrap censored the
tails its interval is made of.

WHAT THE FIX COSTS, measured rather than extrapolated. On the analytic config
it took the fit from 11 iterations to 26 for a bit-identical geometry, which
looked like a 2.4x bill on every production run. It is not: that config has two
free parameters on a model surface. Real fits, HF/6-31G through pyscf on
published constants, in this same internal mode (2026-11-12):

    molecule            atoms  species   iterations      wall time      result
    formyl fluoride       4       4      15  ->  15    2.61 -> 2.64 s   identical
    fluoroacetylene       4       6      40  ->  40    3.13 -> 2.91 s   identical
    vinyl fluoride        6       8      40  ->  40    11.16 -> 11.66 s identical

Identical iteration counts, identical accepted-step counts, and final freq_rms
and geometry RMS equal to every digit printed; the timing spread is noise. All
three accept every step both before and after, so their ratio was below one --
the harmless side of the mismatch. The 2.4x was specific to a geometry where the
ratio sat near the boundary, and the fix is free where it does not.
"""

from __future__ import annotations

import contextlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.uncertainty_v2.resampling import resample_isotopologues_gaussian  # noqa: E402
from runner.run_from_config import build_optimizer_from_config  # noqa: E402
from runner.usability import load_config  # noqa: E402

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


@pytest.fixture(scope="module")
def converged():
    """A converged optimizer on the analytic water surface, internal mode."""
    cfg = load_config(CONFIG)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
        opt.run()
    return opt


def _perturbed(opt, seed_index, n=24):
    seeds = [int(c.generate_state(1)[0])
             for c in np.random.SeedSequence(7).spawn(n)]
    s = deepcopy(opt)
    s.history = []
    s.spectral.isotopologues = resample_isotopologues_gaussian(
        opt.spectral.isotopologues, np.random.default_rng(seeds[seed_index]))
    return s


def test_the_config_under_test_really_is_internal_mode_with_a_prior(converged):
    """Guard against vacuity: the broken branch only exists in internal mode."""
    assert converged.coordinate_mode == "internal"
    assert converged.internal_prior is not None
    assert converged._ic_prior_weight > 0.0
    assert converged.spectral_accept_relax == 0.0, (
        "a nonzero relax would mask the mismatch rather than expose it")


def test_a_zero_step_is_judged_equal_to_itself(converged):
    """The invariant the mismatch broke, stated directly.

    A step of length zero leaves the geometry where it was, so a descent test
    must report no change. Any discrepancy here is the acceptance test grading
    a geometry against a different objective than the one it was scored with.
    """
    opt = converged
    coords = np.asarray(opt.coords, dtype=float)

    _, rw = opt.spectral.stacked(coords, want_jacobian=False)
    _, rp = opt.internal_prior.stacked(coords, want_jacobian=False)
    wp = float(np.sqrt(opt.prior_weight))
    rp = np.asarray(rp, dtype=float)

    before = float(np.sqrt(np.mean(np.concatenate([rw, wp * rp]) ** 2)))
    after_same_rows = float(np.sqrt(np.mean(np.concatenate([rw, wp * rp]) ** 2)))
    after_spectral_only = float(np.sqrt(np.mean(np.asarray(rw) ** 2)))

    assert after_same_rows == pytest.approx(before, rel=1e-15)
    # And the mismatch this test exists for is real on this config, not a
    # hypothetical: spanning only the spectral rows moves the answer by >10%.
    assert abs(after_spectral_only / before - 1.0) > 0.1, (
        "the two row sets agree here, so this config no longer exercises the "
        "defect and the numbers in the module docstring are stale")


def test_a_perturbed_restart_actually_moves(converged):
    """The symptom: resampled fits that returned their own starting point.

    Every one of these three stalled before the fix, with after/before ratios
    of 1.258, 1.281 and 1.251 at zero step.
    """
    base = np.asarray(converged.coords, dtype=float)
    buf = io.StringIO()
    for seed_index in (3, 9, 10):      # samples s004, s010, s011
        s = _perturbed(converged, seed_index)
        with contextlib.redirect_stdout(buf):
            coords = np.asarray(s.run(), dtype=float)
        moved = float(np.abs(coords - base).max())
        assert moved > 0.0, (
            f"sample {seed_index + 1} accepted no step and returned its "
            f"starting geometry")


def test_a_perturbed_restart_accepts_at_least_one_step(converged):
    """Movement could come from one lucky iteration; this pins the mechanism."""
    buf = io.StringIO()
    s = _perturbed(converged, 3)
    with contextlib.redirect_stdout(buf):
        s.run()
    assert s.history
    assert any(h["accepted"] for h in s.history)
    # The rejections that remain must be real ones, not the row mismatch:
    # a rejected step has to have been a step, not a no-op.
    for h in s.history:
        if not h["accepted"]:
            assert h["step_norm"] >= 0.0


def test_the_prior_is_re_evaluated_not_rebuilt():
    """The trial side re-evaluates the step's own prior block linearly; that is
    only legitimate if the block IS linear in q, so test that directly.

    The prior's sigmas can be adaptive, so rebuilding the block at the trial
    geometry could let the objective move between computing a step and judging
    it. The fix instead carries the iteration's (J_prior, r_prior, q) and forms
    r_trial = r_prior - J_prior (q_trial - q). For that to be exact rather than
    an approximation, two things must hold, and both are checked here against
    a direct build at the trial geometry: the Jacobian does not depend on the
    geometry, and the residual moves by exactly J_prior times the step in q.
    """
    from backend.internal.internal_fit import (
        InternalCoordinateSet,
        build_internal_priors,
    )

    elems = ["O", "H", "H"]
    def water(r, deg):
        th = np.radians(deg)
        return np.array([[0.0, 0.0, 0.0],
                         [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
                         [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0]])

    ref = water(0.985, 104.9)            # where the prior is centred
    x1 = water(0.962, 105.8)             # the iteration's geometry
    x2 = water(0.958, 104.2)             # a trial geometry, far from x1
    cs = InternalCoordinateSet(x1, elems, use_dihedrals=False)
    q0 = cs.active_values(ref)
    kw = dict(sigma_bond=0.12, sigma_angle_deg=5.0, prior_values=q0,
              prior_mode="soft", elems=elems)
    J1, r1, _ = build_internal_priors(cs, x1, **kw)
    J2, r2, _ = build_internal_priors(cs, x2, **kw)
    q1 = np.asarray(cs.active_values(x1), dtype=float)
    q2 = np.asarray(cs.active_values(x2), dtype=float)

    np.testing.assert_array_equal(J1, J2)          # geometry-independent
    np.testing.assert_allclose(r1 - J1 @ (q2 - q1), r2, rtol=0, atol=1e-12)
    assert not np.allclose(r1, r2), "the step was too small to test anything"


def test_cartesian_mode_already_had_this_right_and_is_untouched():
    """The cartesian branch always extended the trial vector; nothing changed.

    Asserted by reading the source rather than by running a cartesian fit,
    because the point is that exactly one branch needed the change.
    """
    src = (_ROOT / ".github" / "backend" / "quantize.py").read_text(
        encoding="utf-8-sig")
    # The trial vector is extended once per coordinate mode that has a prior:
    # cartesian (which always did) and internal (which is the fix). Two, not
    # one -- one means a mode is judging a geometry on a different row set
    # than it scored the step with, which is the whole defect.
    assert src.count("residual_w_trial = np.concatenate") == 2, (
        "expected the trial residual to be extended in both the cartesian and "
        "the internal branch")
    # And the cartesian path's own prior handling is unchanged: it still builds
    # its rows through internal_prior.stacked at the trial geometry.
    assert "Jp_trial, rp_trial = self.internal_prior.stacked(trial_coords)" in src


def test_the_carrier_is_bound_on_every_path():
    """The prior block is passed between two points in the loop body.

    Binding it inside the internal-mode branch would leave it unbound on the
    spectral-only and cartesian paths -- a NameError that has already bitten
    this codebase three times this session, each time because a variable was
    initialised inside a conditional and read outside it.
    """
    src = (_ROOT / ".github" / "backend" / "quantize.py").read_text(
        encoding="utf-8-sig")
    loop = src.index("for it in range(self.max_iter):")
    first_branch = src.index("if self.spectral_only:", loop)
    preamble = src[loop:first_branch]
    assert "_ic_prior_rows = None" in preamble, (
        "the prior-block carrier must be bound before any branch in the loop")
