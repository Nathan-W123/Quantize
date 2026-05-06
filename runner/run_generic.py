"""
Generic molecule runner — consumes a fully-specified YAML input file.

Called by runner/run_from_config.py when the YAML contains an `elements` key.
Can also be run directly:
    python -m runner.run_generic configs/template.yaml
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.geometryguess import guess_geometry_molecular_input
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from runner.reporting import (
    export_rovib_corrections_csv,
    export_rovib_warnings_json,
    export_semi_experimental_targets_csv,
    generate_rovib_report_section,
)
from runner.run_settings import BASE_SETTINGS, GLOBAL_PRESETS
from runner.usability import write_outputs

_COMPONENT_MAP = {"A": 0, "B": 1, "C": 2}


def _reorder_to_match(
    coords_pub: np.ndarray,
    elems_pub: list[str],
    elems_target: list[str],
) -> np.ndarray:
    """Reorder PubChem atom rows so they match the user's elements list order."""
    from collections import defaultdict
    if sorted(elems_pub) != sorted(elems_target):
        raise ValueError(
            f"SMILES returned elements {elems_pub} but YAML 'elements' is {elems_target}. "
            "Check that the SMILES matches the molecule you specified."
        )
    available: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(elems_pub):
        available[e].append(i)
    new_order = []
    for e in elems_target:
        if not available[e]:
            raise ValueError(f"Not enough '{e}' atoms in SMILES result to match elements list.")
        new_order.append(available[e].pop(0))
    return coords_pub[new_order]


def _resolve_preset(name: str | None) -> tuple[str, dict]:
    key = str(name).strip().upper() if name else str(BASE_SETTINGS["default_preset"]).upper()
    if key not in GLOBAL_PRESETS:
        valid = ", ".join(sorted(GLOBAL_PRESETS))
        raise ValueError(f"Unknown preset '{key}'. Valid: {valid}")
    return key, dict(GLOBAL_PRESETS[key])


def _build_geometry(cfg: dict) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    elems = [str(e).strip() for e in cfg["elements"]]
    geo = cfg.get("geometry", {})
    method = str(geo.get("method", "bonds")).strip().lower()
    bonds: list[tuple[int, int]] = []

    smiles = str(geo.get("smiles", "")).strip()

    if smiles:
        # SMILES encodes bond order → PubChem returns MMFF94-quality 3D geometry.
        # Reorder atoms to match the user's elements list so masses stay consistent.
        coords_pub, elems_pub = guess_geometry_molecular_input(
            identifier=smiles, pubchem_prefer="smiles"
        )
        coords = _reorder_to_match(coords_pub, list(elems_pub), elems)
        from backend.quantum import _detect_bonds
        bonds = _detect_bonds(coords, elems)

    elif method == "bonds":
        raw_bonds = geo.get("bonds", [])
        if not raw_bonds:
            raise ValueError("geometry.method=bonds requires a non-empty 'bonds' list.")
        bonds = [(int(a), int(b)) for a, b in raw_bonds]
        coords, _ = guess_geometry_molecular_input(elems=elems, bonds=bonds)
        raw_lengths = geo.get("bond_lengths")
        if raw_lengths is not None:
            bond_lengths = [float(x) for x in raw_lengths]
            if len(bond_lengths) != len(bonds):
                raise ValueError(
                    f"bond_lengths has {len(bond_lengths)} entries but bonds has {len(bonds)}."
                )
            from backend.geometryguess import _relax_geometry
            targets = {(min(i, j), max(i, j)): bl for (i, j), bl in zip(bonds, bond_lengths)}
            coords = _relax_geometry(coords, bonds, targets, n_steps=500)
            coords -= coords.mean(axis=0, keepdims=True)

    elif method == "pubchem":
        identifier = str(geo.get("identifier", "")).strip()
        if not identifier:
            raise ValueError("geometry.method=pubchem requires geometry.identifier.")
        coords_pub, elems_pub = guess_geometry_molecular_input(identifier=identifier)
        coords = _reorder_to_match(coords_pub, list(elems_pub), elems)
        from backend.quantum import _detect_bonds
        bonds = _detect_bonds(coords, elems)

    elif method == "coords":
        raw = geo.get("coords_angstrom", [])
        coords = np.array(raw, dtype=float)
        if coords.ndim != 2 or coords.shape != (len(elems), 3):
            raise ValueError(
                f"coords_angstrom shape {coords.shape} does not match "
                f"{len(elems)} atoms × 3 columns."
            )
        from backend.quantum import _detect_bonds
        bonds = _detect_bonds(coords, elems)

    else:
        raise ValueError(
            f"Unknown geometry.method '{method}'. Use: smiles, bonds, pubchem, or coords."
        )

    return coords, elems, bonds


def _build_isotopologue(iso: dict) -> dict:
    raw_comps = iso.get("components", ["A", "B", "C"])
    indices = []
    for c in raw_comps:
        s = str(c).strip().upper()
        if s not in _COMPONENT_MAP:
            raise ValueError(f"Unknown component '{c}'. Use A, B, or C.")
        indices.append(_COMPONENT_MAP[s])

    out = {
        "name": str(iso.get("name", "iso")),
        "masses": [float(m) for m in iso["masses"]],
        "component_indices": indices,
        "obs_constants": [float(v) for v in iso["obs_b0_mhz"]],
        "sigma_constants": [float(v) for v in iso["sigma_mhz"]],
        "alpha_constants": [float(v) for v in iso["alpha_mhz"]],
    }
    # Optional decomposed-delta channels (component-aligned to ``indices``).
    for src_key, dst_key in (
        ("delta_vib_mhz", "delta_vib_constants"),
        ("delta_elec_mhz", "delta_elec_constants"),
        ("delta_bob_mhz", "delta_bob_constants"),
        ("sigma_correction_mhz", "sigma_correction_constants"),
    ):
        if src_key in iso and iso[src_key] is not None:
            out[dst_key] = [float(v) for v in iso[src_key]]
    if "rovib_source" in iso:
        out["rovib_source"] = str(iso["rovib_source"])
    return out


def _compute_metrics(
    coords: np.ndarray,
    bonds: list[tuple[int, int]],
    elems: list[str],
) -> dict[str, float]:
    """Bond distances (Å) and valence angles (deg) from the bond graph."""
    metrics: dict[str, float] = {}

    for i, j in bonds:
        key = f"r({elems[i]}{i}-{elems[j]}{j})"
        metrics[key] = float(np.linalg.norm(coords[j] - coords[i]))

    nbrs: dict[int, list[int]] = defaultdict(list)
    for i, j in bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)

    for k, neighbors in sorted(nbrs.items()):
        for i, j in combinations(sorted(neighbors), 2):
            v1 = coords[i] - coords[k]
            v2 = coords[j] - coords[k]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-30:
                continue
            cos_a = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            key = f"ang({elems[i]}{i}-{elems[k]}{k}-{elems[j]}{j})"
            metrics[key] = float(np.degrees(np.arccos(cos_a)))

    return metrics


def main(cfg: dict[str, Any]) -> dict[str, Any]:
    name = str(cfg.get("name", "molecule")).strip()
    managed_run = "_run_dir" in cfg
    run_dir = Path(str(cfg.get("_run_dir") or ".")).resolve()
    base_workdir = str(run_dir) if managed_run else "trials"

    # ── Geometry ──────────────────────────────────────────────────────────────
    coords, elems, bonds = _build_geometry(cfg)

    # ── Isotopologues ─────────────────────────────────────────────────────────
    raw_isos = cfg.get("isotopologues")
    if not raw_isos:
        raise ValueError("At least one isotopologue is required under 'isotopologues:'.")
    isotopologues = [_build_isotopologue(iso) for iso in raw_isos]

    # ── Quantum chemistry ─────────────────────────────────────────────────────
    qsec = cfg.get("quantum", {})
    backend = str(qsec.get("backend", "orca")).strip().lower()
    spectral_only = backend == "none"
    orca_method = str(qsec.get("method", "wB97X-D4")).strip()
    orca_basis = str(qsec.get("basis", "def2-TZVPP")).strip()
    orca_exe = qsec.get("executable") or BASE_SETTINGS["orca_exe"]

    # ── Rovibrational corrections (optional) ──────────────────────────────────
    correction_table = cfg.get("corrections", None) or None
    correction_mode = str(cfg.get("correction_mode", "hybrid_auto")).strip()
    correction_sigma_vib_fraction = float(cfg.get("correction_sigma_vib_fraction", 0.1))
    correction_elec = bool(cfg.get("correction_elec", False))
    correction_sigma_elec_fraction = float(cfg.get("correction_sigma_elec_fraction", 0.1))
    correction_bob_params = cfg.get("correction_bob_params", None) or None

    # ── Preset and run control ────────────────────────────────────────────────
    preset_name, preset = _resolve_preset(cfg.get("preset"))
    rng_seed = int(cfg.get("rng_seed", 42))
    write_xyz = bool(cfg.get("write_xyz", False))
    symmetry_spec = cfg.get("symmetry") or None

    # ── Multi-start seed geometries ───────────────────────────────────────────
    rng = np.random.default_rng(rng_seed)
    starts = [coords.copy()]
    n_atoms = len(elems)
    for _ in range(int(preset["n_starts"]) - 1):
        jitter = rng.normal(0.0, 0.03, size=(n_atoms, 3))
        jitter[0] = 0.0  # keep first (usually heavy) atom anchored
        starts.append(coords + jitter)

    # ── Optimizer kwargs ──────────────────────────────────────────────────────
    optimizer_kwargs = dict(
        quantum_backend=backend if not spectral_only else "orca",
        orca_executable=orca_exe,
        method_preset="fast",
        orca_method=orca_method,
        orca_basis=orca_basis,
        spectral_only=spectral_only,
        max_iter=500,
        conv_step=1e-7,
        conv_freq=float(preset["conv_freq"]),
        spectral_accept_relax=float(preset.get("spectral_accept_relax", 0.0)),
        conv_energy=1e-8,
        conv_step_range=1e-6,
        conv_step_null=1e-5,
        conv_grad_null=1e-4,
        orca_update_thresh=0.005,
        hess_recalc_every=2,
        adaptive_hess_schedule=True,
        hess_recalc_min=1,
        hess_recalc_max=8,
        sv_threshold=1.3980595102624797e-05,
        sv_min_abs=0.0,
        trust_radius=float(preset["trust_radius"]),
        null_trust_radius=None,
        lambda_damp=0.00016370045068111915,
        objective_mode="split",
        alpha_quantum=0.2778639378704326,
        robust_loss="none",
        robust_param=1.0,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        spectral_delta=0.00034930106014707015,
        auto_sanitize_spectral=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        use_internal_preconditioner=False,
        sigma_floor_mhz=float(preset["sigma_floor_mhz"]),
        max_spectral_weight=float(preset["max_spectral_weight"]),
        enable_geometry_guardrails=True,
        enforce_quantum_descent=bool(preset["enforce_quantum_descent"]),
        use_internal_priors=bool(preset["use_internal_priors"]),
        prior_weight=1.0,
        prior_auto_from_initial=True,
        prior_use_dihedrals=False,
        prior_sigma_bond=0.04,
        prior_sigma_angle_deg=2.0,
        use_conformer_mixture=bool(preset["use_conformer_mixture"]),
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        use_orca_rovib=False,
        rovib_recalc_every=1,
        rovib_source_mode="hybrid_auto",
        correction_table=correction_table,
        correction_mode=correction_mode,
        correction_sigma_vib_fraction=correction_sigma_vib_fraction,
        correction_elec=correction_elec,
        correction_sigma_elec_fraction=correction_sigma_elec_fraction,
        correction_bob_params=correction_bob_params,
        symmetry=symmetry_spec,
        project_rigid_modes=True,
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        base_workdir=base_workdir,
        quantum_descent_tol=float(preset.get("quantum_descent_tol", 1e-5)),
    )

    print(f"[{name}] elements     : {elems}")
    print(f"[{name}] isotopologues: {[iso['name'] for iso in isotopologues]}")
    print(f"[{name}] quantum      : {backend} / {orca_method} / {orca_basis}")
    print(f"[{name}] preset       : {preset_name}")
    print(f"[{name}] n_starts     : {preset['n_starts']}  max_workers: {preset['max_workers']}")
    if symmetry_spec:
        print(f"[{name}] symmetry     : {symmetry_spec}")

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
        job_name=name,
    )
    for r in results:
        labeled = _compute_metrics(r["coords"], bonds, elems)
        r["metrics"] = list(labeled.values())   # flat floats for underconstrained_success_score
        r["metrics_labeled"] = labeled

    best = select_best_result(results, spectral_gate_abs=0.05, spectral_gate_rel=2.0)
    best_metrics = best["metrics_labeled"]

    # ── Optional XYZ output ───────────────────────────────────────────────────
    if write_xyz:
        xyz_path = run_dir / f"{name}_optimized.xyz"
        with open(xyz_path, "w", encoding="utf-8") as fh:
            fh.write(f"{len(elems)}\n")
            fh.write(f"Optimized geometry: {name}\n")
            for e, (x, y, z) in zip(elems, best["coords"]):
                fh.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")
        print(f"[{name}] wrote {xyz_path}")

    # ── Collect per-metric multi-start statistics ─────────────────────────────
    all_metric_arrays: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r["metrics_labeled"].items():
            all_metric_arrays[k].append(v)

    # ── Results summary ───────────────────────────────────────────────────────
    w = 64
    print("\n" + "=" * w)
    print(f"  Inferred geometry: {name}")
    print("=" * w)

    dist_items = [(k, v) for k, v in best_metrics.items() if k.startswith("r(")]
    ang_items  = [(k, v) for k, v in best_metrics.items() if k.startswith("ang(")]

    if dist_items:
        print(f"  {'Bond':<26}  {'Best [Å]':>10}  {'± std':>10}")
        print("  " + "-" * 50)
        for k, v in dist_items:
            vals = all_metric_arrays.get(k, [v])
            std = float(np.std(vals)) if len(vals) > 1 else float("nan")
            std_str = f"{std:.6f}" if not np.isnan(std) else "   n/a  "
            print(f"  {k:<26}  {v:>10.6f}  {std_str:>10}")

    if ang_items:
        print(f"\n  {'Angle':<30}  {'Best [°]':>8}  {'± std':>8}")
        print("  " + "-" * 50)
        for k, v in ang_items:
            vals = all_metric_arrays.get(k, [v])
            std = float(np.std(vals)) if len(vals) > 1 else float("nan")
            std_str = f"{std:.4f}" if not np.isnan(std) else " n/a "
            print(f"  {k:<30}  {v:>8.4f}  {std_str:>8}")

    print("=" * w)
    print(f"  Spectral RMS (MHz)   : {best['freq_rms']:.6f}")
    energy = best.get("energy")
    if energy is not None and energy != 0.0:
        print(f"  Final energy (Eh)    : {energy:.10f}")
    score = underconstrained_success_score(results, best, isotopologues)
    print(f"  Success score (0-100): {score['score']:.1f}")
    print(
        f"  Constrained rank     : {score['constrained_rank']}/{score['internal_dof']} "
        f"({100.0 * score['rank_fraction']:.1f}%)"
    )
    if score["score"] >= 80.0:
        verdict = "strong geometry recovery"
    elif score["score"] >= 60.0:
        verdict = "useful recovery — add isotopologues for tighter constraint"
    else:
        verdict = "geometry regularized by quantum prior; low spectral confidence"
    print(f"  Verdict              : {verdict}")
    print("=" * w)

    result_bundle = {
        "name": name,
        "run_dir": str(run_dir),
        "cfg": cfg,
        "elems": elems,
        "bonds": bonds,
        "results": results,
        "best": best,
        "score": score,
        "best_metrics": best_metrics,
        "all_metric_arrays": dict(all_metric_arrays),
        "preset": preset_name,
        "quantum": {
            "backend": backend,
            "method": orca_method,
            "basis": orca_basis,
        },
    }
    output_cfg = cfg.get("output", {}) or {}
    if (managed_run or bool(output_cfg)) and output_cfg.get("artifacts", True):
        artifacts = write_outputs(result_bundle)
        result_bundle["artifacts"] = artifacts
        print(f"[{name}] report       : {artifacts['report_md']}")
        print(f"[{name}] residual CSV : {artifacts['residuals_csv']}")
        print(f"[{name}] geometry CSV : {artifacts['geometry_csv']}")
    return result_bundle


if __name__ == "__main__":
    freeze_support()
    import argparse
    try:
        import yaml
    except ModuleNotFoundError:
        print("PyYAML required: pip install PyYAML", file=sys.stderr)
        raise SystemExit(1) from None

    parser = argparse.ArgumentParser(
        description="Run quantize from a generalized YAML config."
    )
    parser.add_argument("config", type=Path, help="Path to YAML file")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    main(cfg)
