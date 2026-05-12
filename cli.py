from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import UTC, datetime

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    print(f"  Name             : {name}")
    print(f"  Schema version   : {cfg.get('schema_version', 'n/a')}")
    print(f"  Coordinate mode  : {mode}")
    print(f"  Quantum backend  : {backend}")
    print(f"  Elements         : {len(elems)} atoms")
    print(f"  Isotopologues    : {n_iso}")
    print(f"  Preset           : {cfg.get('preset', 'BALANCED')}")
    print("-" * 60)

def _load_torsion_spec_from_cfg_path(config_path: Path):
    """Load config and build TorsionHamiltonianSpec from it."""
    from backend.torsion_hamiltonian import (
        TorsionFourierPotential, TorsionHamiltonianSpec, TorsionEffectiveConstantFourier,
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
    from backend.scan_preprocess import preprocess_scan
    from backend.scan_fit import fit_fourier_potential

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

    result = fit_fourier_potential(
        phi_arr, e_arr,
        n_harmonics=n_harmonics,
        symmetry_number=sym_number,
        cosine_only=args.cosine_only,
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

    from backend.torsion_fitter import (
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

    from backend.torsion_symmetry import (
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
        from backend.torsion_hamiltonian import solve_ram_lite_levels
        print("\nTorsion levels (J=0, K=0):")
        out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=n_levels)
        for i, e in enumerate(out["energies_cm-1"]):
            print(f"  vt={i}  E={e:.6f} cm^-1")

    if args.convergence:
        print("\nBasis convergence (vt=0 A-species ground state):")
        from backend.torsion_symmetry import c3_symmetry_block_energies
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

def _cmd_uncertainty(args) -> int:
    """Run Bayesian/Bootstrap uncertainty analysis from config."""
    from runner.run_from_config import build_optimizer_from_config
    from backend.uncertainty_sampling import (
        BootstrapRunner, AdaptiveMetropolisSampler, calculate_convergence_diagnostics
    )
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
    from backend.uncertainty_analysis import analyze_coordinate_distribution, print_distribution_summary, posterior_predictive_check, print_ppc_summary
    from backend.uncertainty_plots import plot_coordinate_distributions, plot_mcmc_traces, plot_parameter_corner, plot_autocorrelations
    import numpy as np

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

    cfg = load_config(args.config)
    opt = build_optimizer_from_config(cfg)
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

    if args.uncertainty_engine == "v2":
        posterior_state = evaluate_posterior_state(opt, opt.coords, use_proxy=True)
        laplace = laplace_approximation(opt, regularization=args.laplace_regularization)
        print("\n" + format_posterior_state(posterior_state))
        print(format_laplace_summary(laplace))
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
    
    cloud = []
    
    if args.mode in ["bootstrap", "both"]:
        if args.uncertainty_engine == "v2":
            boot_cfg = BootstrapConfig(
                n_samples=args.samples,
                max_workers=args.workers,
                seed=args.seed,
                strategy=args.bootstrap_strategy,
                output_dir=args.bootstrap_output_dir,
                persist_samples=not args.no_bootstrap_persist,
            )
            engine = BootstrapEngine(opt, boot_cfg)
            summary = engine.run(base_workdir=opt.workdir)
            print("\n" + format_bootstrap_summary(summary))
            summary_bundle["bootstrap"] = summary.to_dict()
            successes = [r for r in summary.sample_results if r.success and r.coords is not None]
        else:
            runner = BootstrapRunner(opt, n_samples=args.samples, max_workers=args.workers)
            boot_results = runner.run(base_workdir=opt.workdir)
            summary_bundle["bootstrap"] = {
                "total_samples": len(boot_results),
                "successful_samples": int(sum(1 for r in boot_results if r.get("success"))),
                "failed_samples": int(sum(1 for r in boot_results if not r.get("success"))),
            }
            successes = [res for res in boot_results if res["success"]]
        
        # Warning for high bootstrap failure rate
        if len(successes) < args.samples:
            print(f"\n[warning] Only {len(successes)}/{args.samples} bootstrap samples converged.")
            summary_bundle["warnings"].append(
                f"Only {len(successes)}/{args.samples} bootstrap samples converged."
            )
            if len(successes) < args.samples / 2:
                print("          High failure rate suggests the geometry is unstable or poorly constrained.")
                summary_bundle["warnings"].append("High bootstrap failure rate suggests unstable/weakly constrained geometry.")

        for res in successes:
            if args.uncertainty_engine == "v2":
                cloud.append(np.asarray(res.coords, dtype=float))
            elif res["success"]:
                cloud.append(res["coords"])

    if args.mode in ["mcmc", "both"]:
        print(f"\n[mcmc] Running {args.chains} parallel chains...")
        all_chains = []
        if args.uncertainty_engine == "v2":
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
            for c in mcmc_summary.chains:
                all_chains.append(c.samples)
                for s in c.samples:
                    cloud.append(s)
            if mcmc_summary.convergence is not None:
                from backend.internal_fit import InternalCoordinateSet

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
        else:
            # Execute chains sequentially in CLI; process pool recommended for production
            for c in range(args.chains):
                print(f"  Chain {c+1}/{args.chains}:")
                sampler = AdaptiveMetropolisSampler(opt)
                mcmc_samples = sampler.run(n_steps=args.mcmc_steps, burn_in=args.burn_in)
                all_chains.append(mcmc_samples)
                for s in mcmc_samples:
                    cloud.append(s)

            # --- Convergence Validation ---
            if args.chains > 1:
                from backend.internal_fit import InternalCoordinateSet

                ic_set = InternalCoordinateSet(opt.coords, opt.elems)
                q_chains = np.array([[ic_set.active_values(xyz) for xyz in chain] for chain in all_chains])

                r_hat, ess, diag_error = calculate_convergence_diagnostics(q_chains)
                ic_names = ic_set.active_names()

                print("\n" + "="*60)
                print("  MCMC CONVERGENCE DIAGNOSTICS")
                print("="*60)
                if diag_error is not None:
                    print(f"  Skipped: {diag_error}")
                    summary_bundle["warnings"].append(f"MCMC diagnostics skipped: {diag_error}")
                else:
                    print(f"{'Parameter':<30} {'R-hat':>10} {'ESS':>10}")
                    print("-" * 60)
                    for name, r, e in zip(ic_names, r_hat, ess):
                        mark = "(!)" if r > 1.1 else "   "
                        print(f"{mark}{name:<30} {r:10.3f} {e:10.1f}")
                    print("\n  (Target: R-hat < 1.1 for convergence; ESS > 100 per chain)")
                print("="*60)

    if not cloud:
        print("Error: No uncertainty samples were generated.")
        return 1

    print("\nProcessing sample cloud...")
    stats, corr, names = analyze_coordinate_distribution(np.array(cloud), opt.elems)
    print_distribution_summary(stats)
    summary_bundle["cloud"] = {
        "n_samples": int(len(cloud)),
        "n_parameters": int(np.asarray(cloud[0]).size) if cloud else 0,
        "coordinate_stats": _json_safe(stats),
        "parameter_names": _json_safe(names),
    }
    
    # --- Posterior Predictive Check (PPC) ---
    ppc_stats = posterior_predictive_check(np.array(cloud), opt.spectral)
    print_ppc_summary(ppc_stats)
    summary_bundle["posterior_predictive_check"] = _json_safe(ppc_stats)

    # --- Generate Plots ---
    plot_dir = os.path.join(opt.workdir, "plots", "uncertainty")
    print(f"\nGenerating diagnostic plots in: {plot_dir}")
    plot_coordinate_distributions(np.array(cloud), opt.elems, plot_dir)
    plot_parameter_corner(np.array(cloud), opt.elems, plot_dir)
    if args.mode in ["mcmc", "both"]:
        plot_mcmc_traces(np.array(cloud)[-args.mcmc_steps:], opt.elems, plot_dir)
        plot_autocorrelations(np.array(cloud)[-args.mcmc_steps:], opt.elems, plot_dir)
    summary_bundle["plot_dir"] = plot_dir

    out_dir = os.path.join(opt.workdir, "uncertainty_v2" if args.uncertainty_engine == "v2" else "uncertainty_legacy")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "uncertainty_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary_bundle), fh, indent=2)
    print(f"\n[uncertainty] Summary written: {out_path}")
    return 0


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
    scan_p.add_argument("--energy-unit", dest="energy_unit", default="cm-1")
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
    unc_p = sub.add_parser("uncertainty", help="Run Bayesian/Bootstrap uncertainty analysis.")
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
        default="legacy",
        help="Uncertainty engine implementation. 'v2' enables greenfield Phase-1 bootstrap.",
    )
    unc_p.add_argument(
        "--bootstrap-strategy",
        choices=["isotopologue-gaussian"],
        default="isotopologue-gaussian",
        help="Bootstrap resampling strategy for v2 engine.",
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
        help="Diagonal regularization added to Laplace Hessian/JTJ in v2 engine.",
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
