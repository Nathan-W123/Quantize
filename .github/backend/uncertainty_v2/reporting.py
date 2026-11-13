from __future__ import annotations

from .models import BootstrapRunSummary, LaplaceSummary, MCMCRunSummary, PosteriorState


def _wrap(text: str, indent: str = "  ", width: int = 78) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)


def format_bootstrap_summary(summary: BootstrapRunSummary) -> str:
    """Human-readable bootstrap summary text block."""
    lines = [
        "Bootstrap v2 summary",
        f"  samples: {summary.total_samples}",
        f"  success: {summary.successful_samples}",
        f"  failed : {summary.failed_samples}",
        # "success" only ever meant the fit did not raise. A sample that
        # accepted no step came back as the starting geometry and was counted
        # among the successes, so this line is what separates the two.
        f"  usable : {summary.usable_samples}  (refused {summary.refused_samples})",
    ]
    if summary.mean_rms_mhz is not None:
        lines.append(f"  mean RMS (MHz): {summary.mean_rms_mhz:.4f}")
    if summary.base_rms_mhz is not None:
        lines.append(f"  base fit RMS  : {summary.base_rms_mhz:.4f} MHz")
    refusals: dict[str, int] = {}
    for r in summary.sample_results:
        if r.success and not r.usable and r.unusable_reason:
            key = r.unusable_reason.split(" (")[0]
            if key.startswith("the refit stopped"):
                key = "stopped short of its own optimum"
            refusals[key] = refusals.get(key, 0) + 1
    for why, n in sorted(refusals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  refused {n}: {why}")
    a = summary.analytic
    if a:
        lines.append(f"  estimates: {a['target']}")
        lines.append(f"  {'coordinate':<16}{'cloud':>11}{'analytic':>11}{'posterior':>11}")
        boot = a.get("bootstrap_std") or [None] * len(a["names"])
        for name, unit, b, t_sd, p_sd in zip(a["names"], a["units"], boot,
                                             a["target_std"], a["posterior_std"]):
            f = 1e3 if unit == "angstrom" else 1.0
            u = "mA" if unit == "angstrom" else "deg"
            cloud = "-" if b is None else f"{b * f:.4f}"
            lines.append(f"  {name:<16}{cloud:>11}{t_sd * f:>11.4f}{p_sd * f:>11.4f}  {u}")
        chk = a.get("check")
        if chk:
            se = chk["mc_sd"] if isinstance(chk["mc_sd"], list) else [chk["mc_sd"]] * len(a["names"])
            kurt = chk.get("kurtosis") or [3.0] * len(a["names"])
            pairs = ", ".join(
                f"{nm} {r:.3f} (+-{e:.3f}" + (f", kurtosis {k:.0f}" if k > 4.5 else "") + ")"
                for nm, r, e, k in zip(a["names"], chk["ratio"], se, kurt))
            lines.extend(_wrap(f"cloud/analytic at {chk['n']} refits, with Monte Carlo error: "
                               f"{pairs}"))
            if chk["departs"]:
                imp = dict(zip(a["names"], chk.get("impact") or []))
                what = ", ".join(
                    f"{nm} (the reported width {abs(imp[nm]) * 100:.0f}% too "
                    f"{'narrow' if imp[nm] > 0 else 'wide'} if the cloud is right)"
                    for nm in chk["departs"] if nm in imp)
                lines.extend(_wrap(f"DEPARTS beyond {chk['sigmas']:g}x Monte Carlo error on: "
                                   f"{what}."))
                if a.get("objective") == "split":
                    lines.append("  Under the split objective the refits are the split estimator, so")
                    lines.append("  the reported (split posterior) widths do not describe its spread there.")
                else:
                    lines.append("  The analytic column is a Laplace approximation; a departure says it")
                    lines.append("  does not describe the refits' spread there (nonlinearity, or a")
                    lines.append("  curvature the Gauss-Newton blocks leave out).")
        if a.get("objective") == "split":
            lines.append("  objective: split -- the cloud is the split estimator's spread (the split")
            lines.append("  step as configured: its rank cutoff and its stopping rules define it).")
            if a.get("split_note"):
                note = str(a["split_note"])
                lines.extend(_wrap(note[:1].upper() + note[1:] + "."))
            shift = a.get("mode_shift")
            if shift:
                name, v = max(shift.items(), key=lambda kv: abs(kv[1]) * (1e3 if "bond" in kv[0] else 1.0))
                unit = "mA" if "bond" in name else "deg"
                val = v * 1e3 if unit == "mA" else v
                lines.extend(_wrap(f"One Newton step toward the posterior's mode moves {name} by "
                                   f"{val:+.3g} {unit}: the posterior is centred there, not at this fit."))
        hd = a.get("hessian_distance_ang")
        if hd is not None and a.get("objective") == "split":
            lines.extend(_wrap(f"Split refits refresh the quantum point as the base fit did (only "
                               f"joint refits hold it); the base's Hessian was computed "
                               f"{1e3 * hd:.2f} mA from this geometry."))
        elif hd is not None:
            lines.extend(_wrap(f"Refits hold the base's quantum quadratic, so the cloud cannot "
                               f"detect error in it; its Hessian was computed {1e3 * hd:.2f} mA "
                               f"from this geometry."))
        if a.get("no_posterior"):
            lines.append("  NO REPORTED POSTERIOR at this geometry: " + str(a["no_posterior"]))
            lines.append("  The analytic and posterior columns are the Laplace's arithmetic only.")
        else:
            lines.append("  The reported uncertainty is the posterior column; the cloud is a")
            lines.append("  refitting check of the analytic column beside it.")
    reason = summary.unusable_reason()
    if reason:
        lines.append(f"  REFUSED: {reason}")
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
    n_phys = summary.n_params - summary.n_gauge_directions
    n_sym = int(getattr(summary, "n_symmetry_directions", 0) or 0)
    n_rigid = summary.n_gauge_directions - n_sym
    removed = f"{n_rigid} rigid-body" + (
        f" + {n_sym} symmetry-forbidden" if n_sym else "")
    lines = [
        "Laplace approximation",
        f"  n params         : {summary.n_params}"
        + (f" ({n_phys} physical, {removed} removed)"
           if summary.n_gauge_directions else ""),
        # Rank and condition describe the physical directions only; saying so
        # on the line matters, because "rank 3" beside "n params 9" otherwise
        # reads as a rank-deficient fit when it is a projected one.
        f"  rank             : {summary.rank}"
        + (f" of {n_phys} physical" if summary.n_gauge_directions else ""),
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
    # The verdict, not just the numbers. R-hat and ESS were already printed
    # while the sampler returned 2.3 to 18 run to run; printing a diagnostic
    # is not the same as acting on it.
    reason = summary.unusable_reason()
    if reason is None:
        lines.append("  intervals: USABLE")
    else:
        lines.append("  intervals: NOT USABLE -- " + reason)
    return "\n".join(lines)
