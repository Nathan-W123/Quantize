"""
Generic molecule runner — consumes a fully-specified YAML input file.

Called by runner/run_from_config.py when the YAML contains an `elements` key.
Can also be run directly:
    python -m runner.run_generic configs/template.yaml
"""

from __future__ import annotations

import sys
import csv
import json
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
from backend.internal_fit import InternalCoordinateSet, spectral_jacobian_q, build_internal_priors
from backend.spectral import SpectralEngine
from backend.spectral_model import normalize_spectral_model
from backend.uncertainty import uncertainty_table
from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
    torsion_objective_from_levels,
)
from backend.torsion_uncertainty import (
    covariance_from_matched_level_residuals,
    default_torsion_parameters,
)
from backend.torsion_average import (
    TorsionGridPoint,
    TorsionScan,
    average_torsion_scan_boltzmann,
    average_torsion_scan_quantum,
)
from backend.hindered_rotor import HinderedRotorModel
from runner.reporting import (
    export_rovib_corrections_csv,
    export_rovib_warnings_json,
    export_semi_experimental_targets_csv,
    generate_rovib_report_section,
)
from runner.run_settings import BASE_SETTINGS, GLOBAL_PRESETS
from runner.usability import write_outputs

_COMPONENT_MAP = {"A": 0, "B": 1, "C": 2}
_MHZ_PER_CM1 = 29979.2458


def _torsion_transition_objective_from_levels(
    predicted_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build transition residual table and RMS (cm^-1) from predicted level energies."""
    pred_map: dict[tuple[int, int, int], float] = {}
    for r in predicted_rows:
        key = (int(r["J"]), int(r["K"]), int(r["level_index"]))
        pred_map[key] = float(r["energy_cm-1"])

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for t in transition_rows:
        lo_key = (int(t["J_lo"]), int(t["K_lo"]), int(t["level_lo"]))
        hi_key = (int(t["J_hi"]), int(t["K_hi"]), int(t["level_hi"]))
        if lo_key not in pred_map or hi_key not in pred_map:
            warnings.append(f"Missing predicted level for transition keys lo={lo_key}, hi={hi_key}")
            continue
        obs = float(t["freq_cm-1"])
        pred = float(pred_map[hi_key] - pred_map[lo_key])
        rows.append(
            {
                "J_lo": lo_key[0],
                "K_lo": lo_key[1],
                "level_lo": lo_key[2],
                "J_hi": hi_key[0],
                "K_hi": hi_key[1],
                "level_hi": hi_key[2],
                "observed_cm-1": obs,
                "predicted_cm-1": pred,
                "residual_cm-1": obs - pred,
            }
        )
    if not rows:
        return {
            "rows": [],
            "rms_cm-1": float("inf"),
            "warnings": warnings + ["No matched torsion target transitions."],
        }
    res = np.asarray([r["residual_cm-1"] for r in rows], dtype=float)
    rms = float(np.sqrt(np.mean(res ** 2)))
    return {"rows": rows, "rms_cm-1": rms, "warnings": warnings}


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


def _build_isotopologue(iso: dict, spectral_model: str = "rigid") -> dict:
    model = normalize_spectral_model(spectral_model)
    raw_comps = iso.get("components", ["A", "B", "C"])
    indices = []
    for c in raw_comps:
        s = str(c).strip().upper()
        if s not in _COMPONENT_MAP:
            raise ValueError(f"Unknown component '{c}'. Use A, B, or C.")
        indices.append(_COMPONENT_MAP[s])

    obs = [float(v) for v in iso["obs_b0_mhz"]]
    sig = [float(v) for v in iso["sigma_mhz"]]
    alp = [float(v) for v in iso["alpha_mhz"]]
    if not (len(obs) == len(sig) == len(alp) == len(indices)):
        raise ValueError(
            f"Isotopologue '{iso.get('name', 'iso')}' has mismatched component/value lengths."
        )

    # Proxy model for internal-rotor systems (e.g. methanol): objective uses B,C only.
    if model == "internal_rotor_bc":
        keep = [i for i, comp_idx in enumerate(indices) if comp_idx in (1, 2)]
        if not keep:
            raise ValueError(
                f"Isotopologue '{iso.get('name', 'iso')}' has no B/C components for spectral_model=internal_rotor_bc."
            )
        indices = [indices[i] for i in keep]
        obs = [obs[i] for i in keep]
        sig = [sig[i] for i in keep]
        alp = [0.0 for _ in keep]

    out = {
        "name": str(iso.get("name", "iso")),
        "masses": [float(m) for m in iso["masses"]],
        "component_indices": indices,
        "obs_constants": obs,
        "sigma_constants": sig,
        "alpha_constants": alp,
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


def _print_internal_uncertainty_summary(best: dict, cfg: dict, elems: list[str]) -> None:
    """Print internal-coordinate uncertainty and variance table to CLI."""
    coord_mode = str(cfg.get("coordinate_mode", "internal")).strip().lower()
    if coord_mode != "internal":
        return
    iso_snapshot = best.get("spectral_isotopologues_snapshot", [])
    if not iso_snapshot:
        return

    ic_cfg = cfg.get("internal_coordinates", {}) or {}
    use_dihedrals = bool(ic_cfg.get("use_dihedrals", False))
    damping = max(float(ic_cfg.get("damping", 1e-6)), 1e-14)
    sigma_bond = float(ic_cfg.get("prior_sigma_bond", 0.04))
    sigma_angle_deg = float(ic_cfg.get("prior_sigma_angle_deg", 2.0))
    sigma_dihedral_deg = float(ic_cfg.get("prior_sigma_dihedral_deg", 15.0))

    coords = np.asarray(best["coords"], dtype=float)
    coord_set = InternalCoordinateSet(coords, elems, use_dihedrals=use_dihedrals)
    B_active = coord_set.active_B_matrix(coords)
    if B_active.shape[0] == 0:
        return

    Bplus = InternalCoordinateSet.damped_pseudoinverse(B_active, damping)
    J_spectral, _ = SpectralEngine(iso_snapshot).stacked(coords)
    Jq = spectral_jacobian_q(J_spectral, Bplus)
    _, _, sigma_prior = build_internal_priors(
        coord_set,
        coords,
        sigma_bond=sigma_bond,
        sigma_angle_deg=sigma_angle_deg,
        sigma_dihedral_deg=sigma_dihedral_deg,
    )
    rows = uncertainty_table(
        coord_set,
        coords,
        Jq,
        sigma_prior=sigma_prior,
        lambda_reg=damping,
    )

    print("\n  Internal-coordinate uncertainty summary")
    print(f"  {'Coordinate':<30}  {'Value':>12}  {'StdErr':>12}  {'Variance':>12}  {'Unit':>6}")
    print("  " + "-" * 84)
    for r in rows:
        se = float(r["std_err"])
        var = se * se
        print(
            f"  {r['name']:<30}  {float(r['value']):>12.6f}  {se:>12.6f}  {var:>12.6f}  {r['value_unit']:>6}"
        )


def _run_torsion_phase2_exports(
    *,
    cfg: dict,
    elems: list[str],
    best: dict,
    isotopologues: list[dict],
    run_dir: Path,
) -> dict[str, Any]:
    """
    Phase-2 bridge: config-driven torsion Hamiltonian predictions.

    Reads optional ``torsion_hamiltonian`` block and writes:
      - exports/torsion_levels.csv
      - exports/torsion_summary.json
    """
    tcfg = cfg.get("torsion_hamiltonian", {}) or {}
    if not isinstance(tcfg, dict) or not bool(tcfg.get("enabled", False)):
        return {}

    if "F" not in tcfg:
        raise ValueError("torsion_hamiltonian.enabled=true requires torsion_hamiltonian.F")

    # Choose isotopologue for mass-based reference rotational constants.
    iso_sel = tcfg.get("isotopologue", 0)
    if isinstance(iso_sel, str):
        matches = [i for i, iso in enumerate(isotopologues) if str(iso.get("name")) == iso_sel]
        if not matches:
            raise ValueError(f"torsion_hamiltonian.isotopologue '{iso_sel}' not found.")
        iso_idx = matches[0]
    else:
        iso_idx = int(iso_sel)
    if iso_idx < 0 or iso_idx >= len(isotopologues):
        raise ValueError("torsion_hamiltonian.isotopologue index out of range.")

    masses = np.asarray(isotopologues[iso_idx]["masses"], dtype=float)
    abc_mhz = SpectralEngine([{"name": "tmp", "masses": masses, "obs_constants": [0, 0, 0]}]).rotational_constants(
        np.asarray(best["coords"], dtype=float), masses
    )
    abc_cm1 = abc_mhz / _MHZ_PER_CM1
    scan_abc_cm1, scan_warn = _torsion_scan_feed_constants(
        cfg=cfg, elems=elems, coords=np.asarray(best["coords"], dtype=float), masses=masses
    )
    if scan_abc_cm1 is not None:
        abc_cm1 = np.asarray(scan_abc_cm1, dtype=float)

    A_cm1 = float(tcfg.get("A_cm-1", abc_cm1[0]))
    B_cm1 = float(tcfg.get("B_cm-1", abc_cm1[1]))
    C_cm1 = float(tcfg.get("C_cm-1", abc_cm1[2]))
    F_cm1 = float(tcfg["F"])
    rho = float(tcfg.get("rho", 0.0))
    n_basis = int(tcfg.get("n_basis", 7))
    n_levels = int(tcfg.get("n_levels", 8))
    units = str(tcfg.get("units", "cm-1"))

    p = tcfg.get("potential", {}) or {}
    v0 = float(p.get("v0", 0.0))
    vcos_raw = p.get("vcos", {}) or {}
    vsin_raw = p.get("vsin", {}) or {}
    vcos = {int(k): float(v) for k, v in vcos_raw.items()}
    vsin = {int(k): float(v) for k, v in vsin_raw.items()}

    pot = TorsionFourierPotential(v0=v0, vcos=vcos, vsin=vsin, units=units)
    spec = TorsionHamiltonianSpec(
        F=F_cm1,
        rho=rho,
        A=A_cm1,
        B=B_cm1,
        C=C_cm1,
        potential=pot,
        n_basis=n_basis,
        units=units,
    )

    J_values = [int(x) for x in (tcfg.get("J_values") or [0])]
    K_values = [int(x) for x in (tcfg.get("K_values") or [0])]

    rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for J in J_values:
        for K in K_values:
            out = solve_ram_lite_levels(spec, J=J, K=K, n_levels=n_levels)
            all_warnings.extend(list(out.get("warnings", [])))
            for level_idx, energy in enumerate(out["energies_cm-1"]):
                rows.append(
                    {
                        "J": J,
                        "K": K,
                        "level_index": level_idx,
                        "energy_cm-1": float(energy),
                    }
                )

    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    levels_csv = exports_dir / "torsion_levels.csv"
    with levels_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["J", "K", "level_index", "energy_cm-1"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    transition_targets = tcfg.get("transitions") or []
    transition_obj = _torsion_transition_objective_from_levels(rows, transition_targets) if transition_targets else None
    transition_csv: Path | None = None
    if transition_obj and transition_obj.get("rows"):
        transition_csv = exports_dir / "torsion_transition_objective.csv"
        with transition_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "J_lo",
                    "K_lo",
                    "level_lo",
                    "J_hi",
                    "K_hi",
                    "level_hi",
                    "observed_cm-1",
                    "predicted_cm-1",
                    "residual_cm-1",
                ],
            )
            writer.writeheader()
            for r in transition_obj["rows"]:
                writer.writerow(r)

    uncertainty_csv: Path | None = None
    unc = tcfg.get("uncertainty") or {}
    if isinstance(unc, dict) and bool(unc.get("enabled", False)):
        level_targets = tcfg.get("targets") or []
        if level_targets:
            level_obj = torsion_objective_from_levels(rows, level_targets)
            matched = list(level_obj.get("rows", []))
            if matched:
                include_completeness = bool(unc.get("include_completeness", False))
                params = default_torsion_parameters(spec, include_completeness=include_completeness)
                cov = covariance_from_matched_level_residuals(
                    spec,
                    matched_rows=matched,
                    params=params,
                    damping=float(unc.get("damping", 1e-8)),
                    rank_tol=float(unc.get("rank_tol", 1e-10)),
                    default_sigma_cm1=float(unc.get("default_sigma_cm1", 1.0)),
                )
                uncertainty_csv = exports_dir / "torsion_parameter_uncertainty.csv"
                with uncertainty_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=["parameter", "std_err", "variance"],
                    )
                    writer.writeheader()
                    for pname, se in zip(cov["param_names"], cov["std_err"]):
                        se_f = float(se)
                        writer.writerow(
                            {"parameter": str(pname), "std_err": se_f, "variance": se_f * se_f}
                        )

    summary_json = exports_dir / "torsion_summary.json"
    summary = {
        "model": "ram_lite_phase1",
        "A_cm-1": A_cm1,
        "B_cm-1": B_cm1,
        "C_cm-1": C_cm1,
        "F_cm-1": F_cm1,
        "rho": rho,
        "n_basis": n_basis,
        "J_values": J_values,
        "K_values": K_values,
        "n_levels": n_levels,
        "warnings": list(dict.fromkeys(all_warnings + scan_warn)),
        "levels_csv": str(levels_csv),
    }
    if transition_obj is not None:
        summary["transition_rms_cm-1"] = float(transition_obj["rms_cm-1"])
        summary["transition_warnings"] = list(dict.fromkeys(transition_obj.get("warnings", [])))
        if transition_csv is not None:
            summary["transition_objective_csv"] = str(transition_csv)
    if uncertainty_csv is not None:
        summary["torsion_parameter_uncertainty_csv"] = str(uncertainty_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{cfg.get('name', 'run')}] torsion CSV  : {levels_csv}")
    print(f"[{cfg.get('name', 'run')}] torsion info : {summary_json}")
    return {
        "torsion_levels_csv": levels_csv,
        "torsion_summary_json": summary_json,
        "torsion_summary": summary,
        "torsion_transition_objective_csv": transition_csv,
        "torsion_parameter_uncertainty_csv": uncertainty_csv,
    }


def _predict_torsion_levels_for_coords(
    cfg: dict,
    elems: list[str],
    coords: np.ndarray,
    isotopologues: list[dict],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return predicted torsion level rows for one geometry (in-memory only)."""
    tcfg = cfg.get("torsion_hamiltonian", {}) or {}
    if not isinstance(tcfg, dict) or not bool(tcfg.get("enabled", False)):
        return [], []
    if "F" not in tcfg:
        return [], ["torsion_hamiltonian enabled but missing F."]

    iso_sel = tcfg.get("isotopologue", 0)
    if isinstance(iso_sel, str):
        matches = [i for i, iso in enumerate(isotopologues) if str(iso.get("name")) == iso_sel]
        iso_idx = matches[0] if matches else 0
    else:
        iso_idx = int(iso_sel)
    iso_idx = min(max(iso_idx, 0), len(isotopologues) - 1)
    masses = np.asarray(isotopologues[iso_idx]["masses"], dtype=float)
    abc_mhz = SpectralEngine([{"name": "tmp", "masses": masses, "obs_constants": [0, 0, 0]}]).rotational_constants(
        np.asarray(coords, dtype=float), masses
    )
    abc_cm1 = abc_mhz / _MHZ_PER_CM1
    scan_abc_cm1, scan_warn = _torsion_scan_feed_constants(
        cfg=cfg, elems=elems, coords=np.asarray(coords, dtype=float), masses=masses
    )
    if scan_abc_cm1 is not None:
        abc_cm1 = np.asarray(scan_abc_cm1, dtype=float)

    A_cm1 = float(tcfg.get("A_cm-1", abc_cm1[0]))
    B_cm1 = float(tcfg.get("B_cm-1", abc_cm1[1]))
    C_cm1 = float(tcfg.get("C_cm-1", abc_cm1[2]))
    F_cm1 = float(tcfg["F"])
    rho = float(tcfg.get("rho", 0.0))
    n_basis = int(tcfg.get("n_basis", 7))
    n_levels = int(tcfg.get("n_levels", 8))
    units = str(tcfg.get("units", "cm-1"))
    p = tcfg.get("potential", {}) or {}
    v0 = float(p.get("v0", 0.0))
    vcos = {int(k): float(v) for k, v in (p.get("vcos", {}) or {}).items()}
    vsin = {int(k): float(v) for k, v in (p.get("vsin", {}) or {}).items()}
    pot = TorsionFourierPotential(v0=v0, vcos=vcos, vsin=vsin, units=units)
    spec = TorsionHamiltonianSpec(
        F=F_cm1, rho=rho, A=A_cm1, B=B_cm1, C=C_cm1, potential=pot, n_basis=n_basis, units=units
    )

    J_values = [int(x) for x in (tcfg.get("J_values") or [0])]
    K_values = [int(x) for x in (tcfg.get("K_values") or [0])]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = list(scan_warn)
    for J in J_values:
        for K in K_values:
            out = solve_ram_lite_levels(spec, J=J, K=K, n_levels=n_levels)
            warnings.extend(list(out.get("warnings", [])))
            for level_idx, energy in enumerate(out["energies_cm-1"]):
                rows.append(
                    {"J": J, "K": K, "level_index": level_idx, "energy_cm-1": float(energy)}
                )
    return rows, warnings


def _torsion_scan_feed_constants(
    *,
    cfg: dict,
    elems: list[str],
    coords: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray | None, list[str]]:
    """
    If torsion_hamiltonian.scan is present, compute hindered-rotor torsion-averaged
    rotational constants and return them in cm^-1 for feeding into RAM-lite inputs.
    """
    tcfg = cfg.get("torsion_hamiltonian", {}) or {}
    scan_cfg = tcfg.get("scan")
    if not isinstance(scan_cfg, dict):
        return None, []
    points = scan_cfg.get("grid_points")
    if not isinstance(points, list) or len(points) == 0:
        return None, ["torsion_hamiltonian.scan is present but has no grid_points; using direct geometry constants."]

    gp_list: list[TorsionGridPoint] = []
    warnings: list[str] = []
    for i, p in enumerate(points):
        if not isinstance(p, dict):
            warnings.append(f"scan.grid_points[{i}] is not a mapping; skipping.")
            continue
        if "phi" not in p:
            warnings.append(f"scan.grid_points[{i}] missing phi; skipping.")
            continue
        g = p.get("geometry", coords.tolist())
        rc = p.get("rotational_constants")
        gp_list.append(
            TorsionGridPoint(
                phi=float(p["phi"]),
                geometry=np.asarray(g, dtype=float),
                energy=(None if p.get("energy") is None else float(p["energy"])),
                rotational_constants=(None if rc is None else np.asarray(rc, dtype=float)),
                weight=(None if p.get("weight") is None else float(p["weight"])),
                label=str(p.get("label")) if p.get("label") is not None else None,
                metadata=dict(p.get("metadata", {}) or {}),
            )
        )
    if not gp_list:
        return None, warnings + ["No valid torsion scan points found; using direct geometry constants."]

    scan = TorsionScan(
        name=str(scan_cfg.get("name", f"{cfg.get('name', 'run')}_scan")),
        atoms=tuple(scan_cfg.get("atoms")) if scan_cfg.get("atoms") is not None else None,
        grid_points=gp_list,
        angle_unit=str(scan_cfg.get("angle_unit", "degrees")),
        energy_unit=str(scan_cfg.get("energy_unit", "hartree")),
        periodic=bool(scan_cfg.get("periodic", True)),
        metadata=dict(scan_cfg.get("metadata", {}) or {}),
    )

    mode = str(scan_cfg.get("mode", "quantum")).strip().lower()
    if mode == "quantum":
        hr = scan_cfg.get("hindered_rotor_model") or {}
        if not isinstance(hr, dict):
            return None, warnings + ["hindered_rotor_model is invalid; using direct geometry constants."]
        model = HinderedRotorModel(
            name=str(hr.get("name", "hindered_rotor")),
            symmetry_number=int(hr.get("symmetry_number", 1)),
            rotational_constant_F=hr.get("rotational_constant_F"),
            rotational_constant_unit=str(hr.get("rotational_constant_unit", "cm-1")),
            fourier_terms={int(k): float(v) for k, v in (hr.get("fourier_terms", {}) or {}).items()},
            potential_energy_unit=str(hr.get("potential_energy_unit", "cm-1")),
            basis_size=int(hr.get("basis_size", 41)),
            warnings=list(hr.get("warnings", []) or []),
        )
        out = average_torsion_scan_quantum(
            elems, scan, model, masses=masses, state_index=int(scan_cfg.get("state_index", 0))
        )
    else:
        out = average_torsion_scan_boltzmann(
            elems, scan, masses=masses, temperature_K=float(scan_cfg.get("temperature_K", 298.15))
        )

    avg_mhz = np.asarray(out["averaged_constants"], dtype=float)
    return avg_mhz / _MHZ_PER_CM1, warnings + list(out.get("warnings", []))


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
    spectral_model = normalize_spectral_model(str(cfg.get("spectral_model", "rigid")))
    isotopologues = [_build_isotopologue(iso, spectral_model=spectral_model) for iso in raw_isos]

    # ── Quantum chemistry ─────────────────────────────────────────────────────
    qsec = cfg.get("quantum", {})
    backend = str(qsec.get("backend", "orca")).strip().lower()
    spectral_only = backend == "none"
    orca_method = str(qsec.get("method", "wB97X-D4")).strip()
    orca_basis = str(qsec.get("basis", "def2-TZVPP")).strip()
    orca_exe = qsec.get("executable") or BASE_SETTINGS["orca_exe"]

    # ── Rovibrational corrections (optional) ──────────────────────────────────
    # Canonical key: rovibrational_corrections: {mode, correction_table, ...}
    # Flat legacy keys (corrections, correction_mode, ...) are accepted as fallback.
    _rc_block = cfg.get("rovibrational_corrections") or {}
    correction_table = (
        _rc_block.get("correction_table")
        or cfg.get("corrections")
        or None
    )
    correction_mode = str(
        _rc_block.get("mode") or cfg.get("correction_mode") or "hybrid_auto"
    ).strip()
    correction_sigma_vib_fraction = float(
        _rc_block.get("sigma_vib_fraction")
        or cfg.get("correction_sigma_vib_fraction")
        or 0.1
    )
    correction_elec = bool(
        _rc_block.get("electronic_correction")
        or cfg.get("correction_elec")
        or False
    )
    correction_sigma_elec_fraction = float(
        _rc_block.get("sigma_elec_fraction")
        or cfg.get("correction_sigma_elec_fraction")
        or 0.1
    )
    correction_bob_params = (
        _rc_block.get("bob_params")
        or cfg.get("correction_bob_params")
        or None
    )

    # ── Preset and run control ────────────────────────────────────────────────
    preset_name, preset = _resolve_preset(cfg.get("preset"))
    rng_seed = int(cfg.get("rng_seed", 42))
    write_xyz = bool(cfg.get("write_xyz", False))
    symmetry_spec = cfg.get("symmetry") or None
    coordinate_mode = str(cfg.get("coordinate_mode", "internal")).strip().lower()
    internal_cfg = cfg.get("internal_coordinates", {}) or {}
    ic_use_dihedrals = bool(internal_cfg.get("use_dihedrals", False))
    ic_damping = float(internal_cfg.get("damping", 1e-6))
    ic_micro_iter = int(internal_cfg.get("microiterations", 20))
    ic_prior_weight = float(internal_cfg.get("prior_weight", 1.0))
    ic_prior_sigma_bond = float(internal_cfg.get("prior_sigma_bond", 0.04))
    ic_prior_sigma_angle_deg = float(internal_cfg.get("prior_sigma_angle_deg", 2.0))
    ic_prior_sigma_dihedral_deg = float(internal_cfg.get("prior_sigma_dihedral_deg", 15.0))

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
        coordinate_mode=coordinate_mode,
        ic_use_dihedrals=ic_use_dihedrals,
        ic_damping=ic_damping,
        ic_micro_iter=ic_micro_iter,
        ic_prior_weight=ic_prior_weight,
        ic_prior_sigma_bond=ic_prior_sigma_bond,
        ic_prior_sigma_angle_deg=ic_prior_sigma_angle_deg,
        ic_prior_sigma_dihedral_deg=ic_prior_sigma_dihedral_deg,
        symmetry=symmetry_spec,
        project_rigid_modes=True,
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        base_workdir=base_workdir,
        quantum_descent_tol=float(preset.get("quantum_descent_tol", 1e-5)),
    )

    print(f"[{name}] elements     : {elems}")
    print(f"[{name}] isotopologues: {[iso['name'] for iso in isotopologues]}")
    print(f"[{name}] spectral model: {spectral_model}")
    print(f"[{name}] quantum      : {backend} / {orca_method} / {orca_basis}")
    print(f"[{name}] coord mode   : {coordinate_mode}")
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

    tcfg = cfg.get("torsion_hamiltonian", {}) or {}
    torsion_targets = tcfg.get("targets")
    torsion_transitions = tcfg.get("transitions")
    use_torsion_selection = bool(tcfg.get("use_in_selection", False)) and bool(torsion_transitions or torsion_targets)
    torsion_weight = float(tcfg.get("selection_weight", 0.01))

    if use_torsion_selection:
        for r in results:
            t_rows, t_warn = _predict_torsion_levels_for_coords(
                cfg, elems, np.asarray(r["coords"], dtype=float), isotopologues
            )
            if torsion_transitions:
                t_obj = _torsion_transition_objective_from_levels(t_rows, torsion_transitions)
                r["torsion_objective_kind"] = "transitions"
            else:
                t_obj = torsion_objective_from_levels(t_rows, torsion_targets)
                r["torsion_objective_kind"] = "levels"
            r["torsion_objective_rows"] = t_obj["rows"]
            r["torsion_rms_cm-1"] = float(t_obj["rms_cm-1"])
            r["torsion_warnings"] = list(dict.fromkeys(t_warn + t_obj.get("warnings", [])))
            r["selection_score"] = float(r.get("freq_rms", np.inf)) + torsion_weight * float(r["torsion_rms_cm-1"])
        best = min(results, key=lambda x: float(x.get("selection_score", np.inf)))
    else:
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
    if "torsion_rms_cm-1" in best:
        print(f"  Torsion RMS (cm^-1)  : {float(best['torsion_rms_cm-1']):.6f}")
        if best.get("selection_score") is not None:
            print(f"  Selection score      : {float(best['selection_score']):.6f}")
        if best.get("torsion_warnings"):
            print("  Torsion warnings:")
            for wmsg in best["torsion_warnings"][:5]:
                print(f"    - {wmsg}")
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
    _print_internal_uncertainty_summary(best, cfg, elems)

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

    torsion_artifacts = _run_torsion_phase2_exports(
        cfg=cfg,
        elems=elems,
        best=best,
        isotopologues=isotopologues,
        run_dir=run_dir,
    )
    if torsion_artifacts:
        result_bundle["torsion_artifacts"] = torsion_artifacts
    if "torsion_objective_rows" in best:
        result_bundle["torsion_objective_rows"] = list(best["torsion_objective_rows"])
        result_bundle["torsion_rms_cm-1"] = float(best.get("torsion_rms_cm-1", np.inf))
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
