from __future__ import annotations

import json

import numpy as np
import pytest

from backend.uncertainty_v2.mcmc import compute_split_rhat_ess
from backend.uncertainty_v2.models import (
    BootstrapConfig,
    BootstrapRunSummary,
    BootstrapSampleResult,
    MCMCConfig,
)
from backend.uncertainty_v2.reporting import (
    format_bootstrap_summary,
    format_laplace_summary,
    format_mcmc_summary,
    format_posterior_state,
)
from backend.uncertainty_v2.resampling import resample_isotopologues_gaussian


def test_bootstrap_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        BootstrapConfig(n_samples=0).validate()
    with pytest.raises(ValueError):
        BootstrapConfig(max_workers=0).validate()
    with pytest.raises(ValueError):
        BootstrapConfig(strategy="unknown").validate()


def test_mcmc_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        MCMCConfig(n_steps=1).validate()
    with pytest.raises(ValueError):
        MCMCConfig(n_steps=10, burn_in=10).validate()
    with pytest.raises(ValueError):
        MCMCConfig(thin=0).validate()
    with pytest.raises(ValueError):
        MCMCConfig(n_chains=0).validate()


def test_resampling_is_deterministic_for_fixed_seed():
    isotopologues = [
        {
            "name": "main",
            "obs_constants": [1000.0, 500.0, 300.0],
            "sigma_constants": [1.0, 2.0, 3.0],
        }
    ]
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    out1 = resample_isotopologues_gaussian(isotopologues, rng1)
    out2 = resample_isotopologues_gaussian(isotopologues, rng2)
    assert np.allclose(out1[0]["obs_constants"], out2[0]["obs_constants"])


def test_split_rhat_ess_reports_converged_for_iid_normal_chains():
    rng = np.random.default_rng(7)
    chains = rng.normal(0.0, 1.0, size=(4, 120, 3))
    conv = compute_split_rhat_ess(chains)
    assert conv is not None
    assert conv.r_hat.shape == (3,)
    assert conv.ess.shape == (3,)
    assert np.all(conv.r_hat < 1.1)
    assert np.all(conv.ess > 100.0)


def test_split_rhat_ess_warns_for_misaligned_chain_means():
    rng = np.random.default_rng(9)
    chains = rng.normal(0.0, 1.0, size=(4, 120, 1))
    chains[0, :, 0] += 4.0
    conv = compute_split_rhat_ess(chains)
    assert conv is not None
    assert conv.r_hat[0] > 1.1
    assert any("R-hat" in w for w in conv.warnings)


def test_reporting_helpers_emit_useful_text():
    bcfg = BootstrapConfig(n_samples=2, max_workers=1)
    bsum = BootstrapRunSummary(
        config=bcfg,
        total_samples=2,
        successful_samples=2,
        failed_samples=0,
        mean_rms_mhz=0.42,
        sample_results=[
            BootstrapSampleResult(idx=1, seed=1, success=True, coords=np.zeros((2, 3)), rms_mhz=0.4),
            BootstrapSampleResult(idx=2, seed=2, success=True, coords=np.zeros((2, 3)), rms_mhz=0.44),
        ],
    )
    txt = format_bootstrap_summary(bsum)
    assert "Bootstrap v2 summary" in txt
    assert "mean RMS" in txt

    # Ensure serializable run summaries remain JSON-friendly for artifact output.
    json.loads(json.dumps(bsum.to_dict()))

    from backend.uncertainty_v2.models import LaplaceSummary, MCMCRunSummary, ChainRunResult, ConvergenceSummary, PosteriorState

    pst = PosteriorState(
        coords=np.zeros((2, 3)),
        chi_sq_spectral=1.0,
        chi_sq_prior=2.0,
        energy_term=0.5,
        log_likelihood=-0.5,
        log_prior=-1.0,
        log_posterior=-1.5,
    )
    lsum = LaplaceSummary(
        n_params=6,
        rank=6,
        condition_number=1.0e4,
        regularization=1.0e-8,
        std_err=np.ones(6),
        covariance=np.eye(6),
        warnings=[],
    )
    msum = MCMCRunSummary(
        config=MCMCConfig(n_steps=20, burn_in=10, n_chains=2),
        chains=[
            ChainRunResult(
                chain_idx=1,
                seed=1,
                samples=np.zeros((3, 2, 3)),
                log_posterior=np.zeros(3),
                acceptance_rate=0.3,
                final_scale=0.01,
                n_proposals=20,
                n_accepted=6,
            )
        ],
        convergence=ConvergenceSummary(r_hat=np.array([1.02]), ess=np.array([150.0]), warnings=[]),
    )
    assert "Posterior state" in format_posterior_state(pst)
    assert "Laplace approximation" in format_laplace_summary(lsum)
    assert "MCMC v2 summary" in format_mcmc_summary(msum)

