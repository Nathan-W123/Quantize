from __future__ import annotations

from .models import BootstrapRunSummary, LaplaceSummary, MCMCRunSummary, PosteriorState


def format_bootstrap_summary(summary: BootstrapRunSummary) -> str:
    """Human-readable bootstrap summary text block."""
    lines = [
        "Bootstrap v2 summary",
        f"  samples: {summary.total_samples}",
        f"  success: {summary.successful_samples}",
        f"  failed : {summary.failed_samples}",
    ]
    if summary.mean_rms_mhz is not None:
        lines.append(f"  mean RMS (MHz): {summary.mean_rms_mhz:.4f}")
    return "\n".join(lines)


def format_posterior_state(state: PosteriorState) -> str:
    lines = [
        "Posterior state (optimum)",
        f"  chi^2 spectral: {state.chi_sq_spectral:.6e}",
        f"  chi^2 prior   : {state.chi_sq_prior:.6e}",
        f"  energy term   : {state.energy_term:.6e}",
        f"  log-likelihood: {state.log_likelihood:.6e}",
        f"  log-prior     : {state.log_prior:.6e}",
        f"  log-posterior : {state.log_posterior:.6e}",
    ]
    return "\n".join(lines)


def format_laplace_summary(summary: LaplaceSummary) -> str:
    lines = [
        "Laplace approximation",
        f"  n params         : {summary.n_params}",
        f"  rank             : {summary.rank}",
        f"  condition number : {summary.condition_number:.3e}",
        f"  regularization   : {summary.regularization:.3e}",
        f"  mean std err     : {float(summary.std_err.mean()):.6e}",
        f"  max  std err     : {float(summary.std_err.max()):.6e}",
    ]
    for w in summary.warnings:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)


def format_mcmc_summary(summary: MCMCRunSummary) -> str:
    lines = [
        "MCMC v2 summary",
        f"  chains: {len(summary.chains)}",
        f"  steps : {summary.config.n_steps} (burn-in={summary.config.burn_in}, thin={summary.config.thin})",
    ]
    for c in summary.chains:
        lines.append(
            f"  chain {c.chain_idx:02d}: accept={c.acceptance_rate:.1%}, "
            f"kept={c.samples.shape[0]}, final_scale={c.final_scale:.2e}"
        )
    if summary.convergence is None:
        lines.append("  convergence: not available")
    else:
        lines.append(f"  max R-hat: {float(summary.convergence.r_hat.max()):.3f}")
        lines.append(f"  min ESS  : {float(summary.convergence.ess.min()):.1f}")
        for w in summary.convergence.warnings:
            lines.append(f"  warning: {w}")
    return "\n".join(lines)
