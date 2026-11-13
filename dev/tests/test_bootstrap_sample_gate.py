"""A bootstrap sample that never moved is not a draw from the posterior.

`BootstrapSampleResult.success` only ever meant "the fit did not raise". A
sample whose optimiser accepted no step returns the geometry it was handed, and
that was counted as a success and put straight into the cloud the credible
interval is computed from -- so the cloud contained copies of its own centre.

Measured on the analytic-water config, twelve samples with seed 7: all twelve
"succeeded", five accepted no step and came back bit-identical to the base
geometry, giving ten duplicate pairs in a cloud of twelve points. Dropping them
widens the reported interval by 19.5% on the bonds and 29.9% on the angle.

The direction matters more than the size. Every failed sample lands exactly on
the distribution's centre, so the more often the sampler fails the narrower the
interval it reports -- the error points the wrong way, and in the limit where
one sample moves the reported width goes to zero while the truth is unchanged.
That cannot be left to a warning, so the summary refuses to be quoted once too
few samples survive.

The second gate this file used to pin -- final residual RMS within 2x the base
fit's -- is gone. Perturbed constants fit worse by construction, so it measured
the draw rather than the fit: it refused 25 of 48 draws on the water config and
39 of 48 on formyl fluoride, nearly all of them at their own optimum, and kept
the draws that happened to fit well, which narrowed the cloud. A sample is now
judged on whether its refit fitted the constants it was given and reached its
own optimum (test_bootstrap_targets.py has the measurements).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.uncertainty_v2.bootstrap import BootstrapEngine  # noqa: E402
from backend.uncertainty_v2.models import (  # noqa: E402
    BootstrapConfig,
    BootstrapRunSummary,
    BootstrapSampleResult,
)
from backend.uncertainty_v2.reporting import format_bootstrap_summary  # noqa: E402

BASE_RMS = 100.0
COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])


def _sample(idx, *, rms, moved, success=True, coords=COORDS,
            termination="plateau", distance=1e-6, kept=True):
    return BootstrapSampleResult(
        idx=idx, seed=idx, success=success,
        coords=None if coords is None else np.asarray(coords, dtype=float),
        rms_mhz=rms, moved_ang=moved, termination=termination,
        mode_distance=distance, draw_kept=kept,
    )


class _StubOptimizer:
    """Only what the gate reads: the base fit's history."""

    def __init__(self, rms=BASE_RMS):
        self.history = [] if rms is None else [{"freq_rms": float(rms)}]
        self.workdir = "."


def _engine(rms=BASE_RMS, n_samples=4):
    return BootstrapEngine(_StubOptimizer(rms), BootstrapConfig(n_samples=n_samples))


def _gate(results, rms=BASE_RMS):
    eng = _engine(rms)
    eng._gate_samples(results, eng._base_rms_mhz())
    return results


# ── the no-accepted-step refusal ─────────────────────────────────────────────

def test_a_sample_that_accepted_no_step_is_refused():
    (r,) = _gate([_sample(1, rms=BASE_RMS, moved=0.0)])
    assert r.success is True, "the fit did not raise; that fact is unchanged"
    assert r.usable is False
    assert "accepted no step" in r.unusable_reason


def test_the_no_step_refusal_does_not_depend_on_a_threshold():
    """It fires even for a sample whose RMS is better than the base fit's."""
    (r,) = _gate([_sample(1, rms=BASE_RMS / 10.0, moved=0.0)])
    assert r.usable is False
    assert "accepted no step" in r.unusable_reason


def test_the_smallest_real_movement_is_still_a_draw():
    """Only exactly zero means no step; a tiny step is a step."""
    (r,) = _gate([_sample(1, rms=BASE_RMS, moved=5e-16)])
    assert r.usable is True
    assert r.unusable_reason is None


# ── the spectral-gate refusal ────────────────────────────────────────────────

def test_a_draw_that_fits_worse_than_the_base_is_still_a_draw():
    """The RMS-ratio gate's false refusal: perturbed data fit worse, and a
    refit sitting at its own optimum is a draw however large its residual."""
    (r,) = _gate([_sample(1, rms=BASE_RMS * 7.8, moved=1e-3, distance=1e-6)])
    assert r.usable is True


def test_the_gate_does_not_read_the_residual_rms():
    a, b = _gate([_sample(1, rms=BASE_RMS * 1e6, moved=1e-3),
                  _sample(2, rms=BASE_RMS * 1e-6, moved=1e-3)])
    assert a.usable is True and b.usable is True


def test_a_refit_that_stopped_short_of_its_own_optimum_is_refused():
    limit = BootstrapConfig().max_mode_distance
    near, far = _gate([_sample(1, rms=BASE_RMS, moved=1e-3, distance=0.5 * limit),
                       _sample(2, rms=BASE_RMS, moved=1e-3, distance=2.0 * limit)])
    assert near.usable is True
    assert far.usable is False and "from its own optimum" in far.unusable_reason


def test_a_refit_that_ran_out_of_iterations_is_refused():
    (r,) = _gate([_sample(1, rms=BASE_RMS, moved=1e-3, termination="max_iter")])
    assert r.usable is False and "ran out of iterations" in r.unusable_reason


def test_a_refit_whose_constants_were_replaced_is_refused():
    """A correction rebuild used to overwrite the drawn constants with the
    unperturbed ones; such a refit fitted the base data."""
    (r,) = _gate([_sample(1, rms=BASE_RMS, moved=1e-3, kept=False)])
    assert r.usable is False and "replaced" in r.unusable_reason


def test_an_unmeasured_distance_is_accepted_only_on_a_converged_fit():
    conv, plat = _gate([_sample(1, rms=BASE_RMS, moved=1e-3, distance=None,
                                termination="converged"),
                        _sample(2, rms=BASE_RMS, moved=1e-3, distance=None,
                                termination="plateau")])
    assert conv.usable is True
    assert plat.usable is False and "could not be measured" in plat.unusable_reason


def test_without_a_base_fit_the_gate_is_unchanged():
    """The certificate is the sample's own; nothing is read from the base."""
    moved, still = _gate([_sample(1, rms=1e9, moved=1e-3),
                          _sample(2, rms=1.0, moved=0.0)], rms=None)
    assert moved.usable is True
    assert still.usable is False and "accepted no step" in still.unusable_reason


# ── a raise, and a sample with no geometry ───────────────────────────────────

def test_a_raising_sample_is_unusable_and_says_so():
    (r,) = _gate([_sample(1, rms=None, moved=None, success=False, coords=None)])
    assert r.usable is False and "raised" in r.unusable_reason


def test_a_sample_with_no_geometry_cannot_enter_the_cloud():
    (r,) = _gate([_sample(1, rms=BASE_RMS, moved=1e-3, coords=None)])
    assert r.usable is False and "no geometry" in r.unusable_reason


# ── the summary's refusal ────────────────────────────────────────────────────

def _summary(results, n_samples=None):
    n = len(results) if n_samples is None else n_samples
    success = [r for r in results if r.success]
    return BootstrapRunSummary(
        config=BootstrapConfig(n_samples=n),
        total_samples=len(results),
        successful_samples=len(success),
        failed_samples=len(results) - len(success),
        mean_rms_mhz=None,
        sample_results=results,
        base_rms_mhz=BASE_RMS,
    )


def test_usable_results_excludes_every_refusal():
    results = _gate([_sample(1, rms=BASE_RMS, moved=1e-3),
                     _sample(2, rms=BASE_RMS, moved=0.0),
                     _sample(3, rms=BASE_RMS, moved=1e-3, termination="max_iter"),
                     _sample(4, rms=BASE_RMS, moved=2e-3)])
    s = _summary(results)
    assert s.successful_samples == 4, "all four ran without raising"
    assert s.usable_samples == 2
    assert s.refused_samples == 2
    assert [r.idx for r in s.usable_results] == [1, 4]


def test_an_interval_is_refused_once_fewer_than_half_the_samples_are_draws():
    results = _gate([_sample(1, rms=BASE_RMS, moved=1e-3)]
                    + [_sample(i, rms=BASE_RMS, moved=0.0) for i in (2, 3, 4)])
    s = _summary(results)
    assert s.usable is False
    reason = s.unusable_reason()
    assert "1 of 4" in reason and "would not describe the draws" in reason


def test_half_the_samples_surviving_is_enough_to_quote():
    results = _gate([_sample(1, rms=BASE_RMS, moved=1e-3),
                     _sample(2, rms=BASE_RMS, moved=1e-3),
                     _sample(3, rms=BASE_RMS, moved=0.0),
                     _sample(4, rms=BASE_RMS, moved=0.0)])
    s = _summary(results)
    assert s.usable is True
    assert s.unusable_reason() is None


def test_the_refusal_counts_against_what_was_asked_for_not_what_came_back():
    """Samples that never returned must not make the survivors look sufficient."""
    results = _gate([_sample(1, rms=BASE_RMS, moved=1e-3),
                     _sample(2, rms=BASE_RMS, moved=1e-3)])
    s = _summary(results, n_samples=10)
    assert s.usable_samples == 2
    assert s.usable is False, "2 draws out of 10 requested is not a sample"


# ── serialisation and reporting ──────────────────────────────────────────────

def test_the_new_fields_survive_serialisation():
    results = _gate([_sample(1, rms=BASE_RMS, moved=0.0),
                     _sample(2, rms=BASE_RMS, moved=1e-3)])
    blob = _summary(results).to_dict()
    assert blob["usable_samples"] == 1 and blob["refused_samples"] == 1
    assert blob["base_rms_mhz"] == pytest.approx(BASE_RMS)
    assert blob["sample_results"][0]["usable"] is False
    assert blob["sample_results"][0]["moved_ang"] == 0.0
    assert "accepted no step" in blob["sample_results"][0]["unusable_reason"]


def test_the_printed_summary_separates_usable_from_merely_non_raising():
    results = _gate([_sample(1, rms=BASE_RMS, moved=0.0),
                     _sample(2, rms=BASE_RMS, moved=1e-3)])
    text = format_bootstrap_summary(_summary(results))
    assert "success: 2" in text
    assert "usable : 1" in text and "refused 1" in text
    assert "base fit RMS" in text


def test_the_printed_summary_carries_the_refusal_when_there_is_one():
    results = _gate([_sample(1, rms=BASE_RMS, moved=0.0),
                     _sample(2, rms=BASE_RMS, moved=0.0)])
    text = format_bootstrap_summary(_summary(results))
    assert "REFUSED" in text


# ── the bias this exists to remove ───────────────────────────────────────────

def test_duplicates_of_the_centre_narrow_the_interval_which_is_why_they_go():
    """The arithmetic behind the refusal, on the measured 5-of-12 failure rate.

    Seven real draws, plus five copies of the geometry they scatter around:
    the ungated standard deviation is smaller than the gated one, so a sampler
    that fails more often reports more confidence.
    """
    rng = np.random.default_rng(0)
    centre = 0.9583
    draws = centre + rng.normal(0.0, 0.0026, size=7)
    pinned = np.full(5, centre)
    gated = float(np.std(draws, ddof=1))
    ungated = float(np.std(np.concatenate([draws, pinned]), ddof=1))
    assert ungated < gated
    assert gated / ungated > 1.1

    # And it gets worse without bound: one mover among many copies reports a
    # width that tends to zero while the draws themselves have not changed.
    worse = float(np.std(np.concatenate([draws[:1], np.full(50, centre)]), ddof=1))
    assert worse < ungated / 5
