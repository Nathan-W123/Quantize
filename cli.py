from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import UTC, datetime

from paths import ensure_repo_paths

_ROOT = ensure_repo_paths(Path(__file__).resolve().parent)

from dev.benchmarks.conformer_suite import (
    attach_comparison as attach_conformer_comparison,
    compare_to_baseline as compare_conformer_to_baseline,
    default_baseline_path as conformer_default_baseline_path,
    default_history_dir as conformer_default_history_dir,
    default_output_path as conformer_default_output_path,
    load_baseline as load_conformer_baseline,
    render_text_summary as render_conformer_summary,
    run_conformer_benchmark_suite,
    write_history_snapshot as write_conformer_history_snapshot,
    write_result as write_conformer_result,
)
from dev.benchmarks.lam_suite import (
    attach_comparison as attach_lam_comparison,
    compare_to_baseline as compare_lam_to_baseline,
    default_baseline_path as lam_default_baseline_path,
    default_history_dir as lam_default_history_dir,
    default_output_path as lam_default_output_path,
    load_baseline as load_lam_baseline,
    render_text_summary as render_lam_summary,
    run_lam_benchmark_suite,
    write_history_snapshot as write_lam_history_snapshot,
    write_result as write_lam_result,
)
from runner.run_from_config import main as run_from_config_main
from runner.usability import ConfigError, load_config, rebuild_report_from_run_dir, validate_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_config_summary(cfg: dict) -> None:
    mode = str(cfg.get("coordinate_mode", "internal"))
    q = cfg.get("quantum", {}) or {}
    backend = str(q.get("backend", "orca"))
    name = str(cfg.get("name", cfg.get("molecule", "run")))
    n_iso = len(cfg.get("isotopologues", []) or [])
    elems = cfg.get("elements", []) or []
    print("\nConfig Summary")
    print("-" * 60)
    print(f"  Name             : {name}")
    print(f"  Schema version   : {cfg.get('schema_version', 'n/a')}")
    print(f"  Coordinate mode  : {mode}")
    print(f"  Quantum backend  : {backend}")
    print(f"  Elements         : {len(elems)} atoms")
    print(f"  Isotopologues    : {n_iso}")
    print(f"  Preset           : {cfg.get('preset', 'BALANCED')}")
    print("-" * 60)


def _json_safe(v):
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return v.tolist()
        if isinstance(v, (_np.floating, _np.integer)):
            return v.item()
    except Exception:
        pass
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def _load_torsion_spec_from_cfg_path(config_path: Path):
    """Load config and build TorsionHamiltonianSpec from it."""
    from backend.torsion.torsion_hamiltonian import (
        TorsionFourierPotential, TorsionHamiltonianSpec,
    )
    cfg = load_config(config_path)
    th = cfg.get("torsion_hamiltonian") or {}
    if not isinstance(th, dict) or not th.get("enabled", True):
        raise ConfigError("torsion_hamiltonian block not found or disabled in config.")
    pot_cfg = th.get("potential") or {}
    vcos_raw = pot_cfg.get("vcos") or {}
    vsin_raw = pot_cfg.get("vsin") or {}
    pot = TorsionFourierPotential(
        v0=float(pot_cfg.get("v0", 0.0)),
        vcos={int(k): float(v) for k, v in vcos_raw.items()},
        vsin={int(k): float(v) for k, v in vsin_raw.items()},
        units="cm-1",
    )
    spec = TorsionHamiltonianSpec(
        F=float(th["F"]),
        rho=float(th.get("rho", 0.0)),
        A=float(th.get("A", 0.0)),
        B=float(th.get("B", 0.0)),
        C=float(th.get("C", 0.0)),
        potential=pot,
        n_basis=int(th.get("n_basis", 10)),
        units="cm-1",
    )
    return spec, th


# ── lam-scan subcommand ───────────────────────────────────────────────────────

def _cmd_lam_scan(args) -> int:
    """Fit Fourier potential from a CSV scan file and print coefficients."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: scan CSV not found: {csv_path}", file=sys.stderr)
        return 2

    phi_col   = args.phi_col   or "phi_deg"
    e_col     = args.energy_col or "energy_cm1"
    angle_unit = args.angle_unit or "degrees"
    energy_unit = args.energy_unit or "cm-1"
    n_harmonics = int(args.n_harmonics or 3)
    sym_number  = int(args.symmetry_number or 3)

    import numpy as np
    from backend.torsion.scan_preprocess import preprocess_scan
    from backend.torsion.scan_fit import scan_to_torsion_potential

    phi_vals, e_vals = [], []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                phi_vals.append(float(row[phi_col]))
                e_vals.append(float(row[e_col]))
            except (KeyError, ValueError) as exc:
                print(f"Warning: skipping row: {exc}", file=sys.stderr)

    phi_arr = np.asarray(phi_vals, dtype=float)
    e_arr   = np.asarray(e_vals,   dtype=float)

    # --energy-unit was parsed and then never applied: energies went into the
    # fit as read and every line below is labelled cm^-1, so a scan in
    # kcal/mol printed a potential 349.755x too small and one in hartree
    # 219,474.6x too small, under the wrong unit and with no warning. Convert
    # through the converter hindered_rotor already owns; argparse's choices
    # guarantee it knows the unit.
    from backend.torsion.hindered_rotor import convert_energy_to_cm1
    e_arr = e_arr * convert_energy_to_cm1(1.0, energy_unit)

    if angle_unit.lower() in {"deg", "degrees"}:
        phi_arr = np.deg2rad(phi_arr)
    period = 2.0 * np.pi / sym_number

    phi_arr, e_arr, pp_info = preprocess_scan(
        phi_arr, e_arr,
        symmetry_number=sym_number,
        period_rad=period,
        do_sort=True,
        do_deduplicate=True,
        do_extend_by_symmetry=args.extend,
    )
    for w in pp_info.get("warnings", []):
        print(f"[preprocess] {w}")

    # This called fit_fourier_potential, which takes no symmetry_number, so
    # every `quantize lam-scan` raised TypeError before fitting anything.
    # scan_to_torsion_potential is the symmetry-aware wrapper written for
    # exactly this: n_harmonics symmetry-allowed groups, so 3 with sigma = 3
    # fits V3, V6, V9. --cosine-only only ever forces cosine terms on; left
    # off, the wrapper's own default applies (cosine-only when sigma > 1).
    _pot, result = scan_to_torsion_potential(
        phi_arr, e_arr,
        n_harmonics=n_harmonics,
        symmetry_number=sym_number,
        cosine_only=True if args.cosine_only else None,
        zero_at_minimum=True,
    )
    print(f"\nScan Fourier fit  (n_harmonics={n_harmonics}, sym={sym_number}, n_pts={len(phi_arr)})")
    print(f"  RMS residual : {result['rms_cm1']:.4f} cm^-1")
    print(f"  v0           : {result['v0']:.6f} cm^-1")
    for k, v in sorted(result.get("vcos", {}).items()):
        print(f"  Vcos_{k:<3d}   : {float(v):+.6f} cm^-1")
    for k, v in sorted(result.get("vsin", {}).items()):
        print(f"  Vsin_{k:<3d}   : {float(v):+.6f} cm^-1")

    if args.json:
        out = {
            "v0": result["v0"],
            "vcos": {str(k): float(v) for k, v in result.get("vcos", {}).items()},
            "vsin": {str(k): float(v) for k, v in result.get("vsin", {}).items()},
            "rms_cm1": result["rms_cm1"],
            "n_points": int(len(phi_arr)),
        }
        print(json.dumps(out, indent=2))
    return 0


# ── lam-fit subcommand ────────────────────────────────────────────────────────

def _cmd_lam_fit(args) -> int:
    """Fit RAM-lite parameters to observed levels or transitions from a config."""
    try:
        spec, th = _load_torsion_spec_from_cfg_path(Path(args.config))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    from backend.torsion.torsion_fitter import (
        fit_torsion_to_levels,
        fit_torsion_to_transitions,
        select_fit_params,
    )

    param_names = args.params.split(",") if args.params else list(
        th.get("fitting", {}).get("params", ["Vcos_3"])
    )
    params = select_fit_params(spec, [p.strip() for p in param_names])
    max_iter = int(args.max_iter or th.get("fitting", {}).get("max_iter", 50))

    obs_levels     = th.get("targets") or []
    obs_transitions = th.get("transitions") or []

    if not obs_levels and not obs_transitions:
        print("No observed levels (targets) or transitions found in config.", file=sys.stderr)
        return 2

    if obs_levels:
        result = fit_torsion_to_levels(spec, obs_levels, params=params, max_iter=max_iter)
        label = "levels"
    else:
        result = fit_torsion_to_transitions(spec, obs_transitions, params=params, max_iter=max_iter)
        label = "transitions"

    print(f"\nLAM fit to {label}  ({result['n_iter']} iter, "
          f"{'converged' if result['converged'] else 'NOT converged'})")
    print(f"  Initial RMS : {result['rms_cm-1_init']:.6f} cm^-1")
    print(f"  Final   RMS : {result['rms_cm-1']:.6f} cm^-1")
    for name, val_i, val_f in zip(
        result["param_names"], result["param_values_init"], result["param_values"]
    ):
        print(f"  {name:<12} : {float(val_i):+.6f}  →  {float(val_f):+.6f} cm^-1")
    for w in result.get("warnings", []):
        print(f"  [warning] {w}")
    return 0


# ── lam-diagnose subcommand ───────────────────────────────────────────────────

def _cmd_lam_diagnose(args) -> int:
    """Display LAM diagnostics: tunneling splitting, purity, basis convergence."""
    try:
        spec, th = _load_torsion_spec_from_cfg_path(Path(args.config))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    from backend.torsion.torsion_symmetry import (
        predict_tunneling_splitting,
        symmetry_purity_table,
    )

    sym_mode = str(th.get("symmetry_mode", "") or "").lower()
    n_levels = int(th.get("n_levels", 6))

    print(f"\nLAM diagnostics  F={spec.F:.5f} cm^-1  rho={spec.rho:.7f}")
    print(f"  n_basis={spec.n_basis}  potential harmonics: "
          f"v0={spec.potential.v0:.3f}  "
          + "  ".join(f"Vcos{k}={v:.3f}" for k, v in spec.potential.vcos.items()))

    if sym_mode in {"c3", "3fold", "threefold"}:
        print("\nA/E tunneling splittings (J=0, K=0):")
        print(f"  {'vt':>3}  {'E_A cm-1':>12}  {'E_E cm-1':>12}  {'Delta cm-1':>12}  {'Delta MHz':>12}")
        rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=n_levels)
        for r in rows:
            print(f"  {r['vt']:>3}  {r['E_A_cm-1']:>12.4f}  {r['E_E_cm-1']:>12.4f}"
                  f"  {r['splitting_cm-1']:>12.4f}  {r['splitting_MHz']:>12.3f}")

        print("\nSymmetry purity (J=0, K=0):")
        prows = symmetry_purity_table(spec, J=0, K=0, n_levels=n_levels)
        for r in prows:
            print(f"  vt={r['level_index']}  {r['symmetry_label']:<4}  "
                  f"E={r['energy_cm-1']:.4f} cm^-1  purity={r['purity']:.4f}")
    else:
        from backend.torsion.torsion_hamiltonian import solve_ram_lite_levels
        print("\nTorsion levels (J=0, K=0):")
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=n_levels)
        for i, e in enumerate(out["energies_cm-1"]):
            print(f"  vt={i}  E={e:.6f} cm^-1")

    if args.convergence:
        print("\nBasis convergence (vt=0 A-species ground state):")
        from backend.torsion.torsion_symmetry import c3_symmetry_block_energies
        from copy import deepcopy
        for nb in (5, 8, 10, 12, 15, 20):
            s = deepcopy(spec)
            s.n_basis = nb
            try:
                blk = c3_symmetry_block_energies(s, J=0, K=0, n_levels_per_block=1)
                e0 = float(blk["A"]["energies_cm-1"][0]) if blk["A"]["energies_cm-1"].size else float("nan")
            except Exception:
                e0 = float("nan")
            print(f"  n_basis={nb:>3}  E_A0={e0:.6f} cm^-1")
    return 0


# ── uncertainty subcommand ───────────────────────────────────────────────────

#: The legacy sampling stack duplicated the v2 one -- both implemented bootstrap
#: resampling and adaptive Metropolis, reachable from the same command, and could
#: return different answers for the same fit. v2 is kept; v1 is gone.
_V1_REMOVED = (
    "--uncertainty-engine v1 has been removed: it duplicated the v2 sampler and "
    "could give a different answer for the same fit. Use v2, which is the default."
)

#: Geometries drawn from the reported posterior when no sampler produced
#: posterior draws: enough for the 2.5/97.5 percentiles to settle to about
#: a twentieth of a width.
_ANALYTIC_DRAWS = 2000

#: The fewest rto refits an interval is taken from. Percentiles from fewer
#: are mostly the extremes: at the default 20 the water config read 0.70 /
#: 0.79 of the posterior's widths with a bond flagged bimodal; at 200, 0.97
#: / 0.91. Below this the rto cloud stays a check and the interval comes
#: from the reported posterior.
_MIN_RTO_DRAWS = 200


def _interval_cloud(mode, mcmc_cloud, boot_posterior, opt, seed,
                    regularization=1e-8, notes=None):
    """Which cloud the interval table, the predictive check and the plots
    describe, and what it is.

    Only a sample of the posterior. The MCMC chains are one when they pass
    their convergence checks; an "rto" bootstrap is one, refit by refit,
    when there are enough refits and the base geometry has a Laplace width
    at all (at a stationary saddle its refits fall into other basins and
    the cloud comes out bimodal). The other bootstrap strategies are not --
    "error-model" is the reported covariance less its prior part, up to 13x
    narrower (formyl fluoride) -- so they are printed as the check they are,
    and the interval comes from the reported posterior itself. Two clouds
    are never pooled: the CLI used to append the bootstrap draws to the
    MCMC chains, mixing in proportion to two sample counts that have nothing
    to do with each other. ``notes`` collects why a cloud was passed over.
    Raises ValueError when no cloud can be quoted.
    """
    import numpy as np
    from backend.uncertainty_v2 import no_posterior_reason, reported_posterior_draws

    notes = [] if notes is None else notes
    if mode in ("mcmc", "both") and mcmc_cloud:
        return list(mcmc_cloud), "the MCMC chains (posterior samples)"
    if mode == "mcmc":
        return [], None
    if boot_posterior:
        why = no_posterior_reason(opt)
        if why:
            raise ValueError(f"the rto refits are not quoted as an interval: {why}")
        if len(boot_posterior) >= _MIN_RTO_DRAWS:
            return list(boot_posterior), "the rto refits (posterior draws)"
        notes.append(
            f"the rto cloud has {len(boot_posterior)} refits, fewer than the "
            f"{_MIN_RTO_DRAWS} a percentile interval needs (its sd carries a "
            f"{100.0 / np.sqrt(2.0 * len(boot_posterior)):.0f}% Monte Carlo error); "
            f"it is the check printed above, and the interval is the reported posterior's")
    rng = np.random.default_rng(seed)
    draws = reported_posterior_draws(opt, _ANALYTIC_DRAWS, rng, regularization=regularization)
    return list(draws), (f"the reported posterior (Laplace covariance plus shared "
                         f"model error, {_ANALYTIC_DRAWS} draws)")


#: A ridge that moves no Laplace width by more than this (relative) is not
#: one the interval's reader needs telling about: 0 and 1e-9 against the
#: default 1e-8 move water's widths by about 4e-13, 1e3 by 4%.
_RIDGE_NOTE_REL = 1e-3


def _laplace_std(opt, regularization):
    """The Laplace's physical widths at a ridge, without the Gauss-Newton
    check (which the comparison does not need and which costs a second
    exact-curvature pass)."""
    import numpy as np

    cov, _ = opt.laplace_approximation(regularization=float(regularization),
                                       check_gauss_newton=False)
    if cov is None:
        return None
    cov, _basis, _n = opt.physical_covariance(cov)
    return np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, None))


def _ridge_moves_widths(opt, regularization):
    """Whether the Laplace ridge in use moves the reported widths at all,
    against the default ridge. The note it gates used to fire on any value
    but the default, a ridge of 0 included."""
    import numpy as np

    if float(regularization) == 1e-8:
        return False
    try:
        s1 = _laplace_std(opt, regularization)
        s0 = _laplace_std(opt, 1e-8)
    except Exception:
        return True                     # cannot tell: say it
    if s0 is None or s1 is None or s0.shape != s1.shape or not s0.size:
        return True
    keep = np.isfinite(s0) & np.isfinite(s1) & (s0 > 1e-12 * float(np.nanmax(s0)))
    if not np.any(keep):
        return True
    return bool(np.max(np.abs(s1[keep] - s0[keep]) / s0[keep]) > _RIDGE_NOTE_REL)


def _write_bundle(opt, summary_bundle):
    import os
    out_dir = os.path.join(opt.workdir, "uncertainty_v2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "uncertainty_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary_bundle), fh, indent=2)
    return out_path


def _cmd_uncertainty(args) -> int:
    """Run Bayesian/Bootstrap uncertainty analysis from config."""
    from runner.run_from_config import build_optimizer_from_config
    from backend.uncertainty_v2 import (
        AdaptiveMetropolisV2,
        BootstrapConfig,
        BootstrapEngine,
        MCMCConfig,
        evaluate_posterior_state,
        laplace_approximation,
    )
    from backend.uncertainty_v2.reporting import (
        format_bootstrap_summary,
        format_laplace_summary,
        format_mcmc_summary,
        format_posterior_state,
    )
    from backend.uncertainty.analysis import analyze_coordinate_distribution, print_distribution_summary, posterior_predictive_check, print_ppc_summary
    from scripts.uncertainty_plots import plot_coordinate_distributions, plot_mcmc_traces, plot_parameter_corner, plot_autocorrelations
    import numpy as np

    if args.uncertainty_engine != "v2":
        # Before the config is loaded: this used to run the whole base fit
        # first and then exit, and the default was this engine.
        print(f"Error: {_V1_REMOVED}", file=sys.stderr)
        return 2
    if args.samples < 1:
        print("Error: --samples must be >= 1.", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("Error: --workers must be >= 1.", file=sys.stderr)
        return 2
    if args.mode in ["mcmc", "both"]:
        if args.chains < 1:
            print("Error: --chains must be >= 1.", file=sys.stderr)
            return 2
        if args.mcmc_steps < 2:
            print("Error: --mcmc-steps must be >= 2.", file=sys.stderr)
            return 2
        if args.burn_in < 0:
            print("Error: --burn-in must be >= 0.", file=sys.stderr)
            return 2
        if args.burn_in >= args.mcmc_steps:
            print("Error: --burn-in must be smaller than --mcmc-steps.", file=sys.stderr)
            return 2

    boot_cfg = None
    if args.mode in ["bootstrap", "both"]:
        boot_cfg = BootstrapConfig(
            n_samples=args.samples,
            max_workers=args.workers,
            seed=args.seed,
            strategy=args.bootstrap_strategy,
            output_dir=args.bootstrap_output_dir,
            persist_samples=not args.no_bootstrap_persist,
        )

    cfg = load_config(args.config)
    opt = build_optimizer_from_config(cfg)
    if boot_cfg is not None:
        # What can be known before the fit is checked before the fit: rto on
        # a spectral-only config used to fail with a traceback after it.
        problem = BootstrapEngine.strategy_problem(opt, boot_cfg.strategy, fitted=False)
        if problem:
            print(f"Error: --bootstrap-strategy {boot_cfg.strategy}: {problem}", file=sys.stderr)
            return 2
    summary_bundle = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "mode": args.mode,
        "engine": args.uncertainty_engine,
        "seed": args.seed,
        "warnings": [],
    }

    print("\nConverging initial geometry to the global minimum...")
    opt.run()

    posterior_state = evaluate_posterior_state(opt, opt.coords, use_proxy=True)
    laplace = laplace_approximation(opt, regularization=args.laplace_regularization)
    print("\n" + format_posterior_state(posterior_state))
    print(format_laplace_summary(laplace))
    cd_rows = opt.distortion_rows()
    if cd_rows:
        # They drive the fit and its chi-square like the rotational rows;
        # the predictive check below covers the rotational rows only.
        print("\nCentrifugal-distortion rows (fitted)")
        for r in cd_rows:
            print(f"  {r['isotopologue']:<12} {r['constant']:<4} observed {r['observed_mhz']:>12.6g}  "
                  f"predicted {r['predicted_mhz']:>12.6g}  sigma {r['sigma_mhz']:>10.4g}  z {r['z']:+.2f}")
        summary_bundle["distortion_rows"] = cd_rows
    summary_bundle["posterior_state"] = posterior_state.to_dict()
    summary_bundle["laplace"] = laplace.to_dict()
    if not args.no_laplace_persist:
        out_dir = os.path.join(opt.workdir, "uncertainty_v2")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "posterior_laplace_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "posterior_state": posterior_state.to_dict(),
                    "laplace": laplace.to_dict(),
                },
                fh,
                indent=2,
            )
    
    # Hoisted above the mode branches on purpose: all are read after them,
    # and initialising inside `if args.mode in [...]` makes --mode bootstrap
    # raise NameError on the empty-cloud path.
    mcmc_cloud: list = []
    boot_posterior: list = []
    all_chains: list = []
    mcmc_refused = False
    mcmc_refusal_reason = None
    mcmc_no_diagnostics = False
    boot_refused = False

    if boot_cfg is not None:
        engine = BootstrapEngine(opt, boot_cfg,
                                 laplace_regularization=args.laplace_regularization)
        try:
            engine._check_strategy()     # the checks that needed the fit
        except ValueError as exc:
            print(f"Error: --bootstrap-strategy {boot_cfg.strategy}: {exc}", file=sys.stderr)
            summary_bundle["bootstrap_refused"] = str(exc)
            if args.mode == "bootstrap":
                summary_bundle["exit_status"] = 2
                print(f"\n[uncertainty] Summary written: {_write_bundle(opt, summary_bundle)}")
                return 2
            # --mode both: the refusal says no refit can be certified against
            # this posterior; the chains sample that posterior as it is, so
            # they still run (this used to return here and lose the MCMC half
            # as well), and the interval carries the split caveat below.
            print("[bootstrap] refused; running the MCMC half of --mode both.")
            boot_refused = True
            boot_cfg = None
            summary_bundle["partial"] = True
            summary_bundle["warnings"].append(
                f"--bootstrap-strategy {args.bootstrap_strategy} was refused after the base "
                f"fit, so this run's result has no bootstrap: {exc}")
    if boot_cfg is not None:
        summary = engine.run(base_workdir=opt.workdir)
        print("\n" + format_bootstrap_summary(summary))
        summary_bundle["bootstrap"] = summary.to_dict()
        # Usable, not merely non-raising. A sample whose fit accepted no
        # step returns the geometry it started from; counting it put a
        # duplicate of the cloud's own centre into the cloud, which makes
        # the reported interval NARROWER the more often the sampler fails.
        # Measured on water: two of four samples came back bit-identical
        # to the base geometry and were reported as successes.
        successes = summary.usable_results
        n_refused = summary.refused_samples

        if n_refused:
            print(f"\n[warning] {n_refused}/{args.samples} bootstrap samples ran but "
                  f"produced no draw; they are excluded from the cloud.")
            summary_bundle["warnings"].append(
                f"{n_refused}/{args.samples} bootstrap samples ran but produced no draw "
                f"(no accepted step, replaced constants, or a refit that did not reach "
                f"its own optimum); excluded."
            )
        if len(successes) < args.samples:
            print(f"[warning] Only {len(successes)}/{args.samples} bootstrap samples "
                  f"produced a usable draw.")
            summary_bundle["warnings"].append(
                f"Only {len(successes)}/{args.samples} bootstrap samples produced a usable draw."
            )
        if not summary.usable:
            reason = summary.unusable_reason()
            print(f"[warning] {reason}")
            print("          These refits are not used for the interval.")
            summary_bundle["warnings"].append(reason)
            summary_bundle["bootstrap_interval_refused"] = True
            successes = []
        elif len(successes) < args.samples / 2:
            print("          High failure rate suggests the geometry is unstable or poorly constrained.")
            summary_bundle["warnings"].append("High bootstrap failure rate suggests unstable/weakly constrained geometry.")

        if boot_cfg.strategy == "rto":
            boot_posterior = [np.asarray(res.coords, dtype=float) for res in successes]

    if args.mode in ["mcmc", "both"]:
        print(f"\n[mcmc] Running {args.chains} parallel chains...")
        mcfg = MCMCConfig(
            n_steps=args.mcmc_steps,
            burn_in=args.burn_in,
            thin=args.mcmc_thin,
            n_chains=args.chains,
            proposal_scale=args.mcmc_scale,
            adapt_every=args.mcmc_adapt_every,
            seed=args.seed,
            checkpoint_every=args.mcmc_checkpoint_every,
            persist_chain_samples=not args.no_mcmc_persist,
            output_dir=args.mcmc_output_dir,
        )
        mcmc_summary = AdaptiveMetropolisV2(opt, mcfg).run(base_workdir=opt.workdir)
        print("\n" + format_mcmc_summary(mcmc_summary))
        summary_bundle["mcmc"] = mcmc_summary.to_dict()
        # Traces are kept either way -- looking at a bad trace is how you
        # diagnose one. The INTERVAL cloud is what gets refused: feeding
        # unmixed chains into analyze_coordinate_distribution reports the
        # spread of the chains' starting points as a posterior width.
        _mcmc_refusal = mcmc_summary.unusable_reason()
        for c in mcmc_summary.chains:
            all_chains.append(c.samples)
        if _mcmc_refusal is None:
            for c in mcmc_summary.chains:
                for s in c.samples:
                    mcmc_cloud.append(s)
        else:
            print(f"\n[mcmc] REFUSED: {_mcmc_refusal}")
            print("[mcmc] These samples are excluded from the reported "
                  "distribution. Traces are still written so the failure "
                  "can be inspected.")
            summary_bundle["warnings"].append(
                "MCMC samples excluded from the reported distribution: "
                + _mcmc_refusal)
            mcmc_refused = True
            mcmc_refusal_reason = _mcmc_refusal
            mcmc_no_diagnostics = mcmc_summary.convergence is None
        if mcmc_summary.convergence is not None:
            from backend.internal.internal_fit import InternalCoordinateSet

            ic_set = InternalCoordinateSet(opt.coords, opt.elems)
            ic_names = ic_set.active_names()
            print("\n" + "=" * 60)
            print("  MCMC CONVERGENCE DIAGNOSTICS (v2 split-Rhat/ESS)")
            print("=" * 60)
            print(f"{'Parameter':<30} {'R-hat':>10} {'ESS':>10}")
            print("-" * 60)
            for name, r, e in zip(ic_names, mcmc_summary.convergence.r_hat, mcmc_summary.convergence.ess):
                mark = "(!)" if r > 1.1 else "   "
                print(f"{mark}{name:<30} {r:10.3f} {e:10.1f}")
            print("\n  (Target: R-hat < 1.1; ESS > 100)")
            print("=" * 60)
        else:
            print("\n[mcmc] v2 convergence diagnostics unavailable (too few chains/samples).")
            summary_bundle["warnings"].append("MCMC convergence diagnostics unavailable (too few chains/samples).")
    notes: list = []
    try:
        cloud, cloud_source = _interval_cloud(
            args.mode, mcmc_cloud, boot_posterior, opt, args.seed,
            regularization=args.laplace_regularization, notes=notes)
    except ValueError as exc:
        # On stderr, as every other refusal is, and with the bundle written:
        # the bootstrap, MCMC and Laplace blocks above are still results.
        print(f"Error: {exc}", file=sys.stderr)
        summary_bundle["cloud_refused"] = str(exc)
        summary_bundle["exit_status"] = 1
        print(f"\n[uncertainty] Summary written: {_write_bundle(opt, summary_bundle)}")
        return 1
    if (cloud_source and "reported posterior" not in cloud_source
            and _ridge_moves_widths(opt, args.laplace_regularization)):
        # The ridge is a device of the Laplace; the chains and the refits
        # sample the objective itself, so their interval does not carry it.
        notes.append(
            f"--laplace-regularization {args.laplace_regularization:g} applies to the "
            f"Laplace (its summary, the bootstrap's analytic and posterior columns, "
            f"draws of the reported posterior); {cloud_source} sample the objective "
            f"itself and do not carry it, so this interval is not that posterior")
    caveat = None
    if cloud_source:
        try:
            split_gap = opt.split_inconsistency()
        except Exception as exc:
            split_gap = f"the split check could not be made ({type(exc).__name__}: {exc})"
        if split_gap:
            # Under split the interval can only be the chains' or the reported
            # posterior's draws (rto is refused), and both sample the split
            # posterior -- which this check says need not describe the fit.
            caveat = (f"split objective: {split_gap}. This interval ({cloud_source}) is the "
                      f"split posterior's and need not be the split fit's spread; an "
                      f"error-model bootstrap of this fit measures that spread")
            notes.append(caveat)
    for note in notes:
        print(f"\n[note] {note}")
        summary_bundle["warnings"].append(note)
    if not cloud:
        if mcmc_refused and mcmc_no_diagnostics:
            # Nothing was tested, so nothing "failed to converge": with one
            # chain (or too few kept samples) there is no R-hat at all, and
            # more steps alone do not make one.
            why = ("no MCMC convergence diagnostics could be computed (fewer than two "
                   "chains, or too few samples kept after --burn-in), so the chains "
                   "cannot be shown to have mixed and nothing is reported from them. "
                   "Run --chains 2 or more with enough steps after the burn-in, or use "
                   "--mode bootstrap.")
        elif mcmc_refused:
            why = ("every MCMC sample was refused as unconverged ("
                   + str(mcmc_refusal_reason).split(". The chains")[0].rstrip(".")
                   + "), so there is nothing to report a distribution from. Raise "
                   "--mcmc-steps, or use --mode bootstrap, which refits per "
                   "resample and does not depend on the chains mixing.")
        else:
            why = "No uncertainty samples were generated."
        print(f"Error: {why}", file=sys.stderr)
        summary_bundle["cloud_refused"] = why
        summary_bundle["exit_status"] = 1
        print(f"\n[uncertainty] Summary written: {_write_bundle(opt, summary_bundle)}")
        return 1

    print(f"\nProcessing sample cloud: {cloud_source}...")
    stats, corr, names = analyze_coordinate_distribution(np.array(cloud), opt.elems)
    print_distribution_summary(stats)
    summary_bundle["cloud"] = {
        "source": cloud_source,
        "n_samples": int(len(cloud)),
        "n_parameters": int(np.asarray(cloud[0]).size) if cloud else 0,
        "coordinate_stats": _json_safe(stats),
        "parameter_names": _json_safe(names),
    }
    if caveat:
        summary_bundle["cloud"]["caveat"] = caveat
    
    # --- Posterior Predictive Check (PPC) ---
    ppc_stats = posterior_predictive_check(np.array(cloud), opt.spectral)
    print_ppc_summary(ppc_stats)
    summary_bundle["posterior_predictive_check"] = _json_safe(ppc_stats)

    # --- Generate Plots ---
    plot_dir = os.path.join(opt.workdir, "plots", "uncertainty")
    print(f"\nGenerating diagnostic plots in: {plot_dir}")
    plot_coordinate_distributions(np.array(cloud), opt.elems, plot_dir)
    plot_parameter_corner(np.array(cloud), opt.elems, plot_dir)
    if args.mode in ["mcmc", "both"] and all_chains:
        # One chain, not a tail slice of `cloud`. The slice was wrong twice
        # over: in --mode both the cloud starts with bootstrap samples, and
        # the MCMC part is several chains laid end to end, so a "trace" drawn
        # from it stepped discontinuously from one chain to the next. It also
        # vanished whenever the chains were refused, which is precisely when
        # someone needs to look at it.
        _trace = np.asarray(all_chains[0])
        if _trace.size:
            plot_mcmc_traces(_trace, opt.elems, plot_dir)
            plot_autocorrelations(_trace, opt.elems, plot_dir)
    summary_bundle["plot_dir"] = plot_dir

    summary_bundle["exit_status"] = 3 if boot_refused else 0
    print(f"\n[uncertainty] Summary written: {_write_bundle(opt, summary_bundle)}")
    return summary_bundle["exit_status"]


def _cmd_benchmark(args) -> int:
    """Run a named benchmark suite, compare it to a baseline, and write artifacts."""
    suites = {
        "lam": {
            "run": run_lam_benchmark_suite,
            "load_baseline": load_lam_baseline,
            "compare": compare_lam_to_baseline,
            "attach": attach_lam_comparison,
            "write_result": write_lam_result,
            "write_history": write_lam_history_snapshot,
            "render": render_lam_summary,
        },
        "conformer": {
            "run": run_conformer_benchmark_suite,
            "load_baseline": load_conformer_baseline,
            "compare": compare_conformer_to_baseline,
            "attach": attach_conformer_comparison,
            "write_result": write_conformer_result,
            "write_history": write_conformer_history_snapshot,
            "render": render_conformer_summary,
        },
    }
    suite = suites.get(args.suite)
    if suite is None:
        print(f"Unknown benchmark suite: {args.suite}", file=sys.stderr)
        return 2

    result = suite["run"]()
    comparison = None

    baseline_path = Path(args.baseline) if args.baseline else None
    if baseline_path:
        if not baseline_path.exists():
            print(f"Baseline not found: {baseline_path}", file=sys.stderr)
            return 2
        baseline = suite["load_baseline"](baseline_path)
        baseline["baseline_path"] = str(baseline_path)
        comparison = suite["compare"](result, baseline)
        result = suite["attach"](result, comparison)

    output_path = Path(args.output)
    suite["write_result"](output_path, result)

    history_snapshot = None
    if args.history_dir:
        history_snapshot = suite["write_history"](args.history_dir, result)

    print(suite["render"](result))
    print(f"\nWrote benchmark result: {output_path}")
    if history_snapshot is not None:
        print(f"Wrote history snapshot: {history_snapshot}")

    if args.enforce_thresholds and comparison and not comparison.get("passed", False):
        return 1
    return 0


# ── main entry point ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantize", description="Quantize command-line interface.")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run Quantize from a YAML or JSON config.")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--no-run-dir", action="store_true")

    # --- validate ---
    check_p = sub.add_parser("validate", help="Validate a YAML or JSON config without running.")
    check_p.add_argument("config", type=Path)

    # --- lam-scan ---
    scan_p = sub.add_parser("lam-scan", help="Fit Fourier potential from a torsion scan CSV.")
    scan_p.add_argument("csv", help="CSV file with scan data.")
    scan_p.add_argument("--phi-col", dest="phi_col", default="phi_deg")
    scan_p.add_argument("--energy-col", dest="energy_col", default="energy_cm1")
    scan_p.add_argument("--angle-unit", dest="angle_unit", default="degrees",
                        choices=["degrees", "radians"])
    scan_p.add_argument("--energy-unit", dest="energy_unit", default="cm-1",
                        choices=["cm-1", "kcal/mol", "kj/mol", "hartree"])
    scan_p.add_argument("--n-harmonics", dest="n_harmonics", type=int, default=3)
    scan_p.add_argument("--symmetry-number", dest="symmetry_number", type=int, default=3)
    scan_p.add_argument("--cosine-only", dest="cosine_only", action="store_true",
                        help="Fit cosine terms only (Cs-symmetric rotor).")
    scan_p.add_argument("--extend", action="store_true",
                        help="Extend scan by symmetry before fitting.")
    scan_p.add_argument("--json", action="store_true", help="Also print JSON output.")

    # --- lam-fit ---
    fit_p = sub.add_parser("lam-fit", help="Fit RAM-lite parameters to observed data from config.")
    fit_p.add_argument("config", help="YAML config with torsion_hamiltonian block.")
    fit_p.add_argument("--params", help="Comma-separated parameter names (e.g. Vcos_3,F,rho).")
    fit_p.add_argument("--max-iter", dest="max_iter", type=int, default=None)

    # --- lam-diagnose ---
    diag_p = sub.add_parser("lam-diagnose", help="Display LAM diagnostics from config.")
    diag_p.add_argument("config", help="YAML config with torsion_hamiltonian block.")
    diag_p.add_argument("--convergence", action="store_true",
                        help="Show basis convergence table.")

    # --- uncertainty ---
    unc_p = sub.add_parser(
        "uncertainty", help="Run Bayesian/Bootstrap uncertainty analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("exit status:\n"
                "  0  an interval was reported and every requested part ran\n"
                "  1  no interval could be reported (the bundle says why)\n"
                "  2  usage error, or a strategy refused before anything past the base fit ran\n"
                "  3  an interval was reported, but part of the request was refused\n"
                "     (--mode both after a bootstrap refusal; the bundle has 'partial': true)"))
    unc_p.add_argument("config", type=Path)
    unc_p.add_argument("--mode", choices=["bootstrap", "mcmc", "both"], default="both")
    unc_p.add_argument("--samples", type=int, default=20, help="Number of Bootstrap samples.")
    unc_p.add_argument("--mcmc-steps", type=int, default=5000, help="Number of MCMC steps.")
    unc_p.add_argument("--burn-in", type=int, default=1000, help="MCMC burn-in period.")
    unc_p.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    unc_p.add_argument("--workers", type=int, default=1, help="Parallel workers for Bootstrap.")
    unc_p.add_argument(
        "--uncertainty-engine",
        choices=["legacy", "v2"],
        default="v2",
        help=("Uncertainty engine. Only v2 exists; 'legacy' is still accepted so an "
              "old command line stops with an explanation instead of a usage error."),
    )
    unc_p.add_argument(
        "--bootstrap-strategy",
        choices=["error-model", "rto", "isotopologue-gaussian"],
        default="error-model",
        help=("What each refit draws. error-model: the constants from the "
              "engine's error model (measurement + correlated model error), "
              "so the cloud checks the reported covariance less its prior "
              "part. rto: also the quantum prior's centre, so the cloud checks "
              "the reported covariance itself. isotopologue-gaussian: the "
              "legacy independent draw by the quoted sigma. Only rto's cloud "
              "is a posterior sample; for the others the interval is taken "
              "from the reported posterior and the cloud is printed as a check."),
    )
    unc_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for deterministic bootstrap in v2 engine.",
    )
    unc_p.add_argument(
        "--bootstrap-output-dir",
        default="bootstrap_v2",
        help="Output directory name under workdir for v2 bootstrap artifacts.",
    )
    unc_p.add_argument(
        "--no-bootstrap-persist",
        action="store_true",
        help="Disable bootstrap summary JSON persistence in v2 engine.",
    )
    unc_p.add_argument(
        "--laplace-regularization",
        type=float,
        default=1e-8,
        help=("Ridge added to the Laplace over the physical directions. Applies to "
              "what this command takes from the Laplace: its summary, the "
              "bootstrap's analytic and posterior columns, and an interval drawn "
              "from the reported posterior. MCMC chains and rto refits sample the "
              "objective itself and do not carry it (a note says so)."),
    )
    unc_p.add_argument(
        "--no-laplace-persist",
        action="store_true",
        help="Disable posterior/laplace JSON persistence in v2 engine.",
    )
    unc_p.add_argument("--mcmc-thin", type=int, default=1, help="Thinning interval for v2 MCMC.")
    unc_p.add_argument("--mcmc-scale", type=float, default=0.005, help="Proposal scale for v2 MCMC.")
    unc_p.add_argument("--mcmc-adapt-every", type=int, default=100, help="Adapt interval for v2 MCMC.")
    unc_p.add_argument(
        "--mcmc-checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint cadence (steps) for v2 MCMC.",
    )
    unc_p.add_argument(
        "--mcmc-output-dir",
        default="mcmc_v2",
        help="Output directory name under workdir for v2 MCMC artifacts.",
    )
    unc_p.add_argument(
        "--no-mcmc-persist",
        action="store_true",
        help="Disable chain sample persistence for v2 MCMC.",
    )

    # --- report ---
    report_p = sub.add_parser("report", help="Rebuild report and plots from an existing run directory.")
    report_p.add_argument("run_dir", type=Path)

    # --- benchmark ---
    bench_p = sub.add_parser("benchmark", help="Run benchmark suites and compare against baselines.")
    bench_p.add_argument("suite", choices=["lam", "conformer"], help="Benchmark suite name.")
    bench_p.add_argument("--baseline", default=None,
                         help="Checked-in baseline JSON used for drift thresholds.")
    bench_p.add_argument("--output", default=None,
                         help="Path for the latest benchmark result JSON.")
    bench_p.add_argument("--history-dir", default=None,
                         help="Directory for timestamped benchmark history snapshots.")
    bench_p.add_argument("--enforce-thresholds", action="store_true",
                         help="Exit non-zero when the baseline comparison fails.")

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        if args.baseline is None:
            args.baseline = str(
                lam_default_baseline_path() if args.suite == "lam" else conformer_default_baseline_path()
            )
        if args.output is None:
            args.output = str(
                lam_default_output_path() if args.suite == "lam" else conformer_default_output_path()
            )
        if args.history_dir is None:
            args.history_dir = str(
                lam_default_history_dir() if args.suite == "lam" else conformer_default_history_dir()
            )

    if args.command == "validate":
        try:
            cfg = load_config(args.config)
            validate_config(cfg)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        _print_config_summary(cfg)
        print(f"Config OK: {args.config}")
        return 0

    if args.command == "run":
        try:
            sys.argv = [sys.argv[0], str(args.config)] + (["--no-run-dir"] if args.no_run_dir else [])
            run_from_config_main()
            return 0
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 1
            if code != 0:
                print("\nRun failed. Tip: use `quantize validate <config>` for detailed config checks.", file=sys.stderr)
            return code

    if args.command == "lam-scan":
        return _cmd_lam_scan(args)

    if args.command == "lam-fit":
        return _cmd_lam_fit(args)

    if args.command == "lam-diagnose":
        return _cmd_lam_diagnose(args)

    if args.command == "uncertainty":
        return _cmd_uncertainty(args)

    if args.command == "report":
        try:
            out = rebuild_report_from_run_dir(args.run_dir)
        except ConfigError as exc:
            print(f"Report error: {exc}", file=sys.stderr)
            return 2
        print(f"Report rebuilt: {out['report_md']}")
        print(f"HTML report   : {out['report_html']}")
        print(f"Plots updated : {len(out['plots'])}")
        return 0

    if args.command == "benchmark":
        return _cmd_benchmark(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
