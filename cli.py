from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runner.run_from_config import main as run_from_config_main
from runner.usability import ConfigError, load_config, validate_config


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            cfg = load_config(args.config)
            validate_config(cfg)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        print(f"Config OK: {args.config}")
        return 0

    if args.command == "run":
        sys.argv = [sys.argv[0], str(args.config)] + (["--no-run-dir"] if args.no_run_dir else [])
        run_from_config_main()
        return 0

    if args.command == "lam-scan":
        return _cmd_lam_scan(args)

    if args.command == "lam-fit":
        return _cmd_lam_fit(args)

    if args.command == "lam-diagnose":
        return _cmd_lam_diagnose(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
