"""Unmixed chains must be refused, not printed and passed along.

This sampler returned R-hat between 2.3 and 18 on water depending on the
seed -- it cannot converge, because the exact posterior is improper along one
direction (see test_gauss_newton_validity). The diagnostics were computed
correctly and printed correctly the whole time. What was missing is that
nothing acted on them: the samples went into the reported distribution
regardless, so `analyze_coordinate_distribution` turned the spread of four
chains' starting points into a posterior width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.uncertainty_v2.models import (  # noqa: E402
    ChainRunResult,
    ConvergenceSummary,
    MCMCConfig,
    MCMCRunSummary,
)
from backend.uncertainty_v2.reporting import format_mcmc_summary  # noqa: E402


def _summary(r_hat, ess, n_chains=2):
    chains = [
        ChainRunResult(
            chain_idx=i + 1, seed=i, samples=np.zeros((5, 2, 3)),
            log_posterior=np.zeros(5), acceptance_rate=0.3,
            final_scale=1e-3, n_proposals=10, n_accepted=3)
        for i in range(n_chains)
    ]
    conv = None
    if r_hat is not None:
        conv = ConvergenceSummary(
            r_hat=np.asarray(r_hat, dtype=float),
            ess=np.asarray(ess, dtype=float), warnings=[])
    return MCMCRunSummary(config=MCMCConfig(n_steps=20, burn_in=10),
                          chains=chains, convergence=conv)


def test_converged_chains_are_usable():
    s = _summary([1.00, 1.02, 1.05], [400, 350, 500])
    assert s.usable is True
    assert s.unusable_reason() is None


def test_bad_r_hat_is_refused():
    s = _summary([1.00, 2.33, 1.01], [400, 400, 400])
    assert s.usable is False
    reason = s.unusable_reason()
    assert "R-hat" in reason and "2.33" in reason


def test_low_ess_is_refused_even_with_good_r_hat():
    """49 effective samples out of 12,000 was the measured case."""
    s = _summary([1.00, 1.01, 1.02], [49, 400, 400])
    assert s.usable is False
    assert "ESS" in s.unusable_reason()


def test_both_failures_are_named():
    s = _summary([5.65, 1.0, 1.0], [40, 400, 400])
    reason = s.unusable_reason()
    assert "R-hat" in reason and "ESS" in reason


def test_missing_diagnostics_are_refused_not_assumed_fine():
    """Absence of evidence of mixing is not evidence of mixing."""
    s = _summary(None, None)
    assert s.usable is False
    assert "no evidence" in s.unusable_reason().lower()


def test_threshold_is_the_conventional_one():
    assert MCMCRunSummary.R_HAT_USABLE_MAX == 1.1
    assert MCMCRunSummary.ESS_USABLE_MIN == 100.0
    # Exactly at the line is acceptable; just past it is not.
    assert _summary([1.1], [100]).usable is True
    assert _summary([1.1000001], [100]).usable is False
    assert _summary([1.1], [99.9]).usable is False


def test_verdict_appears_in_the_printed_report():
    bad = format_mcmc_summary(_summary([2.33], [49]))
    assert "NOT USABLE" in bad
    good = format_mcmc_summary(_summary([1.01], [400]))
    assert "USABLE" in good
    assert "NOT USABLE" not in good


def test_verdict_is_serialised():
    """The JSON is what a later reader sees, long after stdout is gone."""
    d = _summary([2.33], [49]).to_dict()
    assert d["usable"] is False
    assert d["unusable_reason"]
    import json
    json.loads(json.dumps(d, default=float))
