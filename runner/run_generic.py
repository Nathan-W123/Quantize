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
    TorsionEffectiveConstantFourier,
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    auto_assign_levels_by_proximity,
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
    average_torsion_scan_quantum_thermal,
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
    """Build transition residual table and RMS (cm^-1) from predicted level energies.

    Phase-4 enhancements:
    - ``freq_mhz`` accepted as alternative to ``freq_cm-1``
    - ``symmetry_lo`` / ``symmetry_hi`` (optional) filter by A/E label
    - ``sigma_cm-1`` stored per row for downstream uncertainty weighting
    """
    pred_map: dict[tuple[int, int, int], float] = {}
    pred_sym_map: dict[tuple[int, int, int], str] = {}
    for r in predicted_rows:
        key = (int(r["J"]), int(r["K"]), int(r["level_index"]))
        pred_map[key] = float(r["energy_cm-1"])
        if "symmetry_label" in r:
            pred_sym_map[key] = str(r["symmetry_label"])

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for t in transition_rows:
        lo_key = (int(t["J_lo"]), int(t["K_lo"]), int(t["level_lo"]))
        hi_key = (int(t["J_hi"]), int(t["K_hi"]), int(t["level_hi"]))
        if lo_key not in pred_map or hi_key not in pred_map:
            warnings.append(f"Missing predicted level for transition keys lo={lo_key}, hi={hi_key}")
            continue

        # Resolve observed frequency: freq_cm-1 takes priority; freq_mhz as fallback.
        obs_raw = t.get("freq_cm-1")
        if obs_raw is None:
            obs_mhz = t.get("freq_mhz")
            if obs_mhz is None:
                warnings.append(
                    f"Transition lo={lo_key} hi={hi_key} has neither freq_cm-1 nor freq_mhz; skipped."
                )
                continue
            obs_raw = float(obs_mhz) / _MHZ_PER_CM1
        obs = float(obs_raw)

        # Symmetry selection rule (Phase 4): skip if predicted label doesn't match spec.
        sym_lo_spec = t.get("symmetry_lo")
        sym_hi_spec = t.get("symmetry_hi")
        if sym_lo_spec is not None and lo_key in pred_sym_map:
            if str(pred_sym_map[lo_key]).upper() != str(sym_lo_spec).upper():
                warnings.append(
                    f"Symmetry mismatch at lo_key={lo_key}: "
                    f"predicted '{pred_sym_map[lo_key]}' != spec '{sym_lo_spec}'; transition skipped."
                )
                continue
        if sym_hi_spec is not None and hi_key in pred_sym_map:
            if str(pred_sym_map[hi_key]).upper() != str(sym_hi_spec).upper():
                warnings.append(
                    f"Symmetry mismatch at hi_key={hi_key}: "
                    f"predicted '{pred_sym_map[hi_key]}' != spec '{sym_hi_spec}'; transition skipped."
                )
                continue

        pred = float(pred_map[hi_key] - pred_map[lo_key])
        row: dict[str, Any] = {
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
        if lo_key in pred_sym_map:
            row["symmetry_lo"] = pred_sym_map[lo_key]
        if hi_key in pred_sym_map:
            row["symmetry_hi"] = pred_sym_map[hi_key]
        sigma_raw = t.get("sigma_cm-1")
        if sigma_raw is not None:
            row["sigma_cm-1"] = float(sigma_raw)
        rows.append(row)

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


def _fit_scan_potential(
    tcfg: dict,
    exports_dir: Path | None = None,
) -> tuple[dict | None, dict, list[str]]:
    """
    Fit a Fourier potential from scan grid_point energies if fit_potential is enabled.

    Returns (fitted_potential_dict | None, fit_diagnostics, warnings).
    fitted_potential_dict has keys 'v0', 'vcos', 'vsin' — compatible with
    the torsion_hamiltonian.potential block format so it can be merged into tcfg.
    """
    from backend.scan_fit import (
        energies_to_cm1 as _e2cm1,
        ingest_scan_csv,
        scan_to_torsion_potential,
        validate_scan_coverage,
        export_scan_fit_csv,
    )

    scan_cfg = tcfg.get("scan")
    if not isinstance(scan_cfg, dict):
        return None, {}, []

    fp_cfg = scan_cfg.get("fit_potential")
    if fp_cfg is None or fp_cfg is False:
        return None, {}, []
    if fp_cfg is True:
        fp_cfg = {}
    if isinstance(fp_cfg, dict) and not bool(fp_cfg.get("enabled", True)):
        return None, {}, []
    if not isinstance(fp_cfg, dict):
        fp_cfg = {}

    angle_unit = str(scan_cfg.get("angle_unit", "degrees")).strip().lower()
    energy_unit = str(scan_cfg.get("energy_unit", "hartree")).strip().lower()

    phi_list: list[float] = []
    energy_list: list[float] = []
    warnings: list[str] = []

    csv_path_raw = scan_cfg.get("csv_path") or scan_cfg.get("path")
    if csv_path_raw:
        csv_path = Path(str(csv_path_raw))
        if not csv_path.is_absolute():
            csv_path = (_ROOT / csv_path).resolve()
        try:
            phi_arr, energies_cm1 = ingest_scan_csv(
                csv_path,
                phi_col=str(scan_cfg.get("phi_col", "phi_deg")),
                energy_col=str(scan_cfg.get("energy_col", "energy_cm1")),
                angle_unit=str(scan_cfg.get("csv_angle_unit", scan_cfg.get("angle_unit", "degrees"))),
                energy_unit=str(scan_cfg.get("csv_energy_unit", scan_cfg.get("energy_unit", "cm-1"))),
            )
        except Exception as exc:
            return None, {}, [f"Could not ingest torsion scan CSV '{csv_path}': {exc}"]
    else:
        gps = scan_cfg.get("grid_points")
        if not isinstance(gps, list) or len(gps) == 0:
            return None, {}, ["scan.fit_potential enabled but no grid_points or csv_path found."]

        for i, gp in enumerate(gps):
            if not isinstance(gp, dict) or "phi" not in gp or gp.get("energy") is None:
                warnings.append(
                    f"scan.grid_points[{i}] missing phi or energy; skipping for potential fit."
                )
                continue
            phi_val = float(gp["phi"])
            if angle_unit in {"deg", "degrees", "degree"}:
                phi_list.append(phi_val * np.pi / 180.0)
            else:
                phi_list.append(phi_val)
            energy_list.append(float(gp["energy"]))

        if len(phi_list) < 3:
            return None, {}, warnings + [
                "Fewer than 3 scan points with energy data; cannot fit potential."
            ]

        phi_arr = np.array(phi_list, dtype=float)
        energies_raw = np.array(energy_list, dtype=float)
        energies_cm1 = _e2cm1(energies_raw, energy_unit)

    # Phase-3 scan preprocessing (sort, deduplicate endpoint, extend by symmetry)
    pp_cfg = scan_cfg.get("preprocess") or {}
    if isinstance(pp_cfg, dict) and pp_cfg:
        from backend.scan_preprocess import preprocess_scan as _preprocess_scan
        sym_num_pp = int(fp_cfg.get("symmetry_number", 1)) if isinstance(fp_cfg, dict) else 1
        period_pp = 2.0 * np.pi / sym_num_pp if sym_num_pp > 1 else 2.0 * np.pi
        phi_arr, energies_cm1, pp_info = _preprocess_scan(
            phi_arr,
            energies_cm1,
            symmetry_number=sym_num_pp,
            period_rad=period_pp,
            do_sort=bool(pp_cfg.get("sort", True)),
            do_deduplicate=bool(pp_cfg.get("deduplicate", True)),
            do_extend_by_symmetry=bool(pp_cfg.get("extend_by_symmetry", False)),
            endpoint_tol_rad=float(pp_cfg.get("endpoint_tol_rad", 0.05)),
        )
        warnings.extend(pp_info.get("warnings", []))

    energies_cm1 = energies_cm1 - float(np.min(energies_cm1))  # relative

    n_harmonics = int(fp_cfg.get("n_harmonics", 6))
    symmetry_number = int(fp_cfg.get("symmetry_number", 1))
    cosine_only_raw = fp_cfg.get("cosine_only")
    cosine_only = cosine_only_raw if isinstance(cosine_only_raw, bool) else None
    zero_at_minimum = bool(fp_cfg.get("zero_at_minimum", True))

    period_rad = 2.0 * np.pi / symmetry_number if symmetry_number > 1 else 2.0 * np.pi
    cov = validate_scan_coverage(phi_arr, energies_cm1, period_rad=period_rad)
    warnings.extend(cov.get("warnings", []))
    if not cov["ok"]:
        for err in cov.get("errors", []):
            warnings.append(f"Scan coverage error: {err}")

    fit_kwargs: dict = {
        "n_harmonics": n_harmonics,
        "symmetry_number": symmetry_number,
        "zero_at_minimum": zero_at_minimum,
    }
    if cosine_only is not None:
        fit_kwargs["cosine_only"] = cosine_only

    pot, fit_result = scan_to_torsion_potential(phi_arr, energies_cm1, **fit_kwargs)
    warnings.extend(fit_result.get("warnings", []))

    if exports_dir is not None:
        try:
            csv_path = exports_dir / "torsion_scan_fit.csv"
            export_scan_fit_csv(csv_path, phi_arr, energies_cm1, pot.v0, pot.vcos, pot.vsin)
        except Exception as exc:
            warnings.append(f"Could not write torsion_scan_fit.csv: {exc}")
        else:
            fit_result["scan_fit_csv"] = str(csv_path)

    fitted_pot_dict = {
        "v0": pot.v0,
        "vcos": pot.vcos,
        "vsin": pot.vsin,
    }
    return fitted_pot_dict, fit_result, warnings


def _apply_fitted_scan_potential(
    tcfg: dict,
    exports_dir: Path | None = None,
) -> tuple[dict, dict, list[str]]:
    """Return a torsion config with scan-fitted potential applied when requested."""
    fitted_pot_dict, scan_fit_result, scan_fit_warn = _fit_scan_potential(tcfg, exports_dir)
    if fitted_pot_dict is None:
        return tcfg, scan_fit_result, scan_fit_warn
    out = dict(tcfg)
    out["potential"] = {
        "v0": fitted_pot_dict["v0"],
        "vcos": {str(k): v for k, v in fitted_pot_dict["vcos"].items()},
        "vsin": {str(k): v for k, v in fitted_pot_dict["vsin"].items()},
    }
    return out, scan_fit_result, scan_fit_warn


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
      - exports/torsion_scan_fit.csv  (if scan.fit_potential enabled)
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
    scan_abc_cm1, scan_warn, scan_average = _torsion_scan_feed_constants(
        cfg=cfg, elems=elems, coords=np.asarray(best["coords"], dtype=float), masses=masses
    )
    if scan_abc_cm1 is not None:
        abc_cm1 = np.asarray(scan_abc_cm1, dtype=float)

    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Phase-2 scan potential fit: override the potential block if fit_potential is enabled.
    original_tcfg = tcfg
    tcfg, scan_fit_result, scan_fit_warn = _apply_fitted_scan_potential(tcfg, exports_dir)
    scan_warn = list(scan_warn) + scan_fit_warn
    fitted_pot_dict = None if tcfg is original_tcfg else tcfg.get("potential")
    if fitted_pot_dict is not None:
        print(
            f"[{cfg.get('name', 'run')}] scan potential fitted: "
            f"RMS={scan_fit_result.get('rms_cm1', float('nan')):.3f} cm^-1, "
            f"harmonics={scan_fit_result.get('harmonics', [])}"
        )

    spec, symmetry_mode, label_levels = _build_torsion_spec_from_config(tcfg, abc_cm1)
    export_blocks = bool(tcfg.get("export_symmetry_blocks", False))
    n_levels = int(tcfg.get("n_levels", 8))
    J_values = [int(x) for x in (tcfg.get("J_values") or [0])]
    K_values = [int(x) for x in (tcfg.get("K_values") or [0])]

    rows, all_warnings, block_rows = _collect_level_rows(
        spec, J_values, K_values, n_levels, symmetry_mode, label_levels,
        return_blocks=export_blocks,
    )
    # Unpack commonly reported scalars for summary
    F_cm1 = spec.F
    rho = spec.rho
    n_basis = spec.n_basis
    A_cm1 = spec.A
    B_cm1 = spec.B
    C_cm1 = spec.C

    levels_csv = exports_dir / "torsion_levels.csv"
    _has_sym = any("symmetry_label" in r for r in rows)
    _level_fields = ["J", "K", "level_index", "energy_cm-1"]
    if _has_sym:
        _level_fields += ["symmetry_label", "symmetry_sublabel", "symmetry_purity"]
    with levels_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_level_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    symmetry_blocks_csv: Path | None = None
    if block_rows:
        symmetry_blocks_csv = exports_dir / "torsion_symmetry_blocks.csv"
        with symmetry_blocks_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["J", "K", "block", "residue", "block_level_index", "energy_cm-1"],
            )
            writer.writeheader()
            for r in block_rows:
                writer.writerow(r)

    # Phase-9 tunneling splitting export (C3 symmetry mode)
    tunneling_csv: Path | None = None
    tunneling_rows: list[dict] = []
    if symmetry_mode == "c3":
        try:
            from backend.torsion_symmetry import (
                predict_tunneling_splitting as _predict_splitting,
                tunneling_splitting_to_csv_rows as _split_csv_rows,
            )
            for J_val in J_values:
                for K_val in K_values:
                    split_rows = _predict_splitting(
                        spec, J=int(J_val), K=int(K_val), n_levels=int(n_levels)
                    )
                    tunneling_rows.extend(split_rows)
            if tunneling_rows:
                tunneling_csv = exports_dir / "torsion_tunneling_splitting.csv"
                with tunneling_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=["vt", "J", "K", "E_A_cm-1", "E_E_cm-1",
                                    "splitting_cm-1", "splitting_MHz"],
                    )
                    writer.writeheader()
                    for r in _split_csv_rows(tunneling_rows):
                        writer.writerow(r)
        except Exception as exc:
            all_warnings.append(f"tunneling_splitting: {exc}")

    # Phase-5 auto-assign by proximity
    auto_assign_cfg = tcfg.get("auto_assign") or {}
    auto_assign_csv: Path | None = None
    auto_assign_result: dict[str, Any] = {}
    if isinstance(auto_assign_cfg, dict) and bool(auto_assign_cfg.get("enabled", False)):
        obs_list = auto_assign_cfg.get("observed_cm1") or []
        if obs_list:
            sym_filter = auto_assign_cfg.get("symmetry_filter") or None
            max_delta = float(auto_assign_cfg.get("max_delta_cm1", float("inf")))
            auto_assign_result = auto_assign_levels_by_proximity(
                rows,
                obs_list,
                max_delta_cm1=max_delta,
                symmetry_filter=sym_filter,
                method=str(auto_assign_cfg.get("method", "global")),
                ambiguity_tol_cm1=float(auto_assign_cfg.get("ambiguity_tol_cm1", 0.05)),
            )
            all_warnings.extend(auto_assign_result.get("warnings", []))
            auto_assign_csv = exports_dir / "torsion_auto_assign.csv"
            with auto_assign_csv.open("w", newline="", encoding="utf-8") as fh:
                aa_fields = [
                    "observed_cm-1", "predicted_cm-1", "delta_cm-1",
                    "matched", "predicted_row_index", "assignment_method",
                ]
                if any("J" in r for r in auto_assign_result.get("assignments", [])):
                    aa_fields += ["J", "K", "level_index"]
                if any("symmetry_label" in r for r in auto_assign_result.get("assignments", [])):
                    aa_fields.append("symmetry_label")
                writer = csv.DictWriter(fh, fieldnames=aa_fields, extrasaction="ignore")
                writer.writeheader()
                for r in auto_assign_result.get("assignments", []):
                    writer.writerow(r)

    scan_average_csv: Path | None = None
    thermal_states_csv: Path | None = None
    if scan_average:
        scan_average_csv = _write_torsion_scan_average_csv(exports_dir, scan_average)
        scan_average["scan_average_csv"] = str(scan_average_csv)
        thermal_states_csv = _write_torsion_thermal_states_csv(exports_dir, scan_average)
        if thermal_states_csv is not None:
            scan_average["thermal_states_csv"] = str(thermal_states_csv)

    plot_paths = _write_torsion_phase2_plots(
        run_dir=run_dir,
        scan_fit_result=scan_fit_result,
        scan_average=scan_average,
    )

    transition_targets = tcfg.get("transitions") or []
    transition_obj = _torsion_transition_objective_from_levels(rows, transition_targets) if transition_targets else None
    transition_csv: Path | None = None
    transition_summary_csv: Path | None = None
    _trans_fields = [
        "J_lo", "K_lo", "level_lo", "J_hi", "K_hi", "level_hi",
        "observed_cm-1", "predicted_cm-1", "residual_cm-1",
    ]
    if transition_obj and transition_obj.get("rows"):
        transition_csv = exports_dir / "torsion_transition_objective.csv"
        _has_trans_sym = any(
            "symmetry_lo" in r or "symmetry_hi" in r
            for r in transition_obj["rows"]
        )
        if _has_trans_sym:
            _trans_fields += ["symmetry_lo", "symmetry_hi"]
        if any("sigma_cm-1" in r for r in transition_obj["rows"]):
            _trans_fields.append("sigma_cm-1")
        with transition_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_trans_fields, extrasaction="ignore")
            writer.writeheader()
            for r in transition_obj["rows"]:
                writer.writerow(r)
        transition_summary_csv = _write_transition_group_summary_csv(exports_dir, transition_obj["rows"])

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

    # Phase-6 parameter fitting
    fitting_cfg = tcfg.get("fitting") or {}
    fitting_csv: Path | None = None
    fit_covariance_csv: Path | None = None
    fit_correlation_csv: Path | None = None
    fitted_levels_csv: Path | None = None
    fitted_transition_csv: Path | None = None
    fit_result: dict[str, Any] = {}
    if isinstance(fitting_cfg, dict) and bool(fitting_cfg.get("enabled", False)):
        from backend.torsion_fitter import fit_torsion_to_levels as _fit_levels
        from backend.torsion_fitter import fit_torsion_to_transitions as _fit_trans
        from backend.torsion_fitter import select_fit_params as _select_params

        def _run_one_fit(stage_cfg: dict[str, Any], current_spec: TorsionHamiltonianSpec) -> dict[str, Any]:
            raw_param_names = stage_cfg.get("params") or None
            fit_params = _select_params(current_spec, raw_param_names) if raw_param_names else None
            fit_kwargs = {
                "params": fit_params,
                "max_iter": int(stage_cfg.get("max_iter", fitting_cfg.get("max_iter", 50))),
                "xtol": float(stage_cfg.get("xtol", fitting_cfg.get("xtol", 1e-8))),
                "ftol": float(stage_cfg.get("ftol", fitting_cfg.get("ftol", 1e-8))),
                "damping": float(stage_cfg.get("damping", fitting_cfg.get("damping", 1e-6))),
                "bounds": stage_cfg.get("bounds", fitting_cfg.get("bounds")),
                "priors": stage_cfg.get("priors", fitting_cfg.get("priors")),
            }
            if "default_sigma_cm1" in stage_cfg or "default_sigma_cm1" in fitting_cfg:
                fit_kwargs["default_sigma_cm1"] = float(
                    stage_cfg.get("default_sigma_cm1", fitting_cfg.get("default_sigma_cm1", 1.0))
                )
            use_levels = bool(stage_cfg.get("use_levels", fitting_cfg.get("use_levels", True)))
            use_transitions = bool(stage_cfg.get("use_transitions", fitting_cfg.get("use_transitions", True)))
            level_targets = tcfg.get("targets") or []
            transition_targets = tcfg.get("transitions") or []
            if use_levels and level_targets:
                return _fit_levels(current_spec, level_targets, **fit_kwargs)
            if use_transitions and transition_targets:
                return _fit_trans(current_spec, transition_targets, **fit_kwargs)
            return {
                "fitted_spec": current_spec,
                "param_names": [],
                "warnings": ["No fitting targets available for this stage."],
            }

        try:
            stages = fitting_cfg.get("stages")
            if isinstance(stages, list) and stages:
                current_spec = spec
                stage_results = []
                for i, raw_stage in enumerate(stages):
                    stage_cfg = raw_stage if isinstance(raw_stage, dict) else {}
                    stage_result = _run_one_fit(stage_cfg, current_spec)
                    stage_result["stage_index"] = i
                    stage_results.append(stage_result)
                    current_spec = stage_result.get("fitted_spec", current_spec)
                fit_result = dict(stage_results[-1])
                fit_result["stages"] = [
                    {
                        "stage_index": int(r.get("stage_index", i)),
                        "rms_cm-1": float(r.get("rms_cm-1", float("nan"))),
                        "converged": bool(r.get("converged", False)),
                        "param_names": list(r.get("param_names", [])),
                    }
                    for i, r in enumerate(stage_results)
                ]
            else:
                fit_result = _run_one_fit(fitting_cfg, spec)
        except ValueError as exc:
            all_warnings.append(f"fitting.params error: {exc}")
        except Exception as exc:
            all_warnings.append(f"Parameter fitting failed: {exc}")

        if fit_result:
            all_warnings.extend(fit_result.get("warnings", []))
            param_names = list(fit_result.get("param_names", []))
            fitting_csv = exports_dir / "torsion_fit_params.csv"
            with fitting_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "parameter", "initial_value", "fitted_value", "std_err",
                        "lower_bound", "upper_bound",
                    ],
                )
                writer.writeheader()
                for idx, (name, v0, vf) in enumerate(zip(
                    param_names,
                    fit_result.get("param_values_init", []),
                    fit_result.get("param_values", []),
                )):
                    std = fit_result.get("std_err", [])
                    lo = fit_result.get("bounds_lower", [])
                    hi = fit_result.get("bounds_upper", [])
                    writer.writerow(
                        {
                            "parameter": name,
                            "initial_value": float(v0),
                            "fitted_value": float(vf),
                            "std_err": float(std[idx]) if idx < len(std) else "",
                            "lower_bound": float(lo[idx]) if idx < len(lo) and np.isfinite(lo[idx]) else "",
                            "upper_bound": float(hi[idx]) if idx < len(hi) and np.isfinite(hi[idx]) else "",
                        }
                    )
            fit_covariance_csv = _write_fit_matrix_csv(
                exports_dir, "torsion_fit_covariance.csv", param_names, fit_result.get("covariance")
            )
            fit_correlation_csv = _write_fit_matrix_csv(
                exports_dir, "torsion_fit_correlation.csv", param_names, fit_result.get("correlation")
            )

            fitted_spec = fit_result.get("fitted_spec")
            if isinstance(fitted_spec, TorsionHamiltonianSpec):
                fitted_rows, fit_row_warnings, fitted_block_rows = _collect_level_rows(
                    fitted_spec, J_values, K_values, n_levels, symmetry_mode, label_levels,
                    return_blocks=export_blocks,
                )
                all_warnings.extend(fit_row_warnings)
                fitted_levels_csv = exports_dir / "torsion_levels_fitted.csv"
                with fitted_levels_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_level_fields, extrasaction="ignore")
                    writer.writeheader()
                    for r in fitted_rows:
                        writer.writerow(r)
                if transition_targets:
                    fitted_transition_obj = _torsion_transition_objective_from_levels(
                        fitted_rows, transition_targets
                    )
                    if fitted_transition_obj.get("rows"):
                        fitted_transition_csv = exports_dir / "torsion_transition_objective_fitted.csv"
                        with fitted_transition_csv.open("w", newline="", encoding="utf-8") as fh:
                            writer = csv.DictWriter(fh, fieldnames=_trans_fields, extrasaction="ignore")
                            writer.writeheader()
                            for r in fitted_transition_obj["rows"]:
                                writer.writerow(r)
                        fit_result["fitted_transition_rms_cm-1"] = fitted_transition_obj.get("rms_cm-1")
                        all_warnings.extend(fitted_transition_obj.get("warnings", []))
            print(
                f"[{cfg.get('name', 'run')}] torsion fit  : "
                f"RMS={fit_result.get('rms_cm-1', float('nan')):.4f} cm^-1 "
                f"({'converged' if fit_result.get('converged') else 'not converged'} "
                f"in {fit_result.get('n_iter', 0)} iter)"
            )

    # Phase-10 LAM correction report
    lam_report: dict[str, Any] = {}
    try:
        from backend.torsion_lam_integration import (
            classify_constant_source as _classify_source,
            format_lam_report_for_summary as _fmt_lam,
            lam_correction_report as _lam_report,
        )
        lam_source = _classify_source(cfg)
        B_rigid = np.array([A_cm1, B_cm1, C_cm1])
        B_avg_arr: np.ndarray | None = None
        if scan_average:
            avg_cm1 = scan_average.get("averaged_constants_cm-1")
            if avg_cm1 is not None:
                try:
                    B_avg_arr = np.asarray(avg_cm1, dtype=float).ravel()[:3]
                except Exception:
                    B_avg_arr = None
        fit_rms = float(fit_result.get("rms_cm-1", 0.0)) if fit_result else 0.0
        n_obs_levels = len(tcfg.get("targets") or [])
        lam_raw = _lam_report(
            B_rigid,
            B_torsion_avg_cm1=B_avg_arr,
            torsion_rms_cm1=fit_rms,
            n_torsion_levels=max(n_obs_levels, 1),
            source=lam_source,
        )
        lam_report = _fmt_lam(lam_raw)
    except Exception as exc:
        all_warnings.append(f"lam_correction_report: {exc}")

    # Phase-7 geometry coupling: recompute F/rho from geometry if configured
    geom_coupling_cfg = tcfg.get("geometry_coupling") or {}
    geom_coupling_result: dict[str, Any] = {}
    if isinstance(geom_coupling_cfg, dict) and bool(geom_coupling_cfg.get("enabled", False)):
        top_idxs = geom_coupling_cfg.get("top_indices")
        axis_idxs = geom_coupling_cfg.get("axis_atom_indices")
        if top_idxs is not None and axis_idxs is not None:
            coords = best.get("coords")
            masses_list = isotopologues[0].get("masses") if isotopologues else None
            if coords is not None and masses_list is not None:
                try:
                    from backend.torsion_geometry import (
                        compute_F_rho_from_geometry as _compute_F_rho,
                        torsion_geometry_jacobian as _tor_geom_jac,
                    )
                    coords_arr = np.asarray(coords, dtype=float)
                    masses_arr = np.asarray(masses_list, dtype=float)
                    gc_F, gc_rho = _compute_F_rho(
                        coords_arr, masses_arr,
                        [int(i) for i in top_idxs],
                        (int(axis_idxs[0]), int(axis_idxs[1])),
                    )
                    geom_coupling_result["F_geometry_cm-1"] = gc_F
                    geom_coupling_result["rho_geometry"] = gc_rho
                    geom_coupling_result["F_spec_cm-1"] = float(spec.F)
                    geom_coupling_result["rho_spec"] = float(spec.rho)
                    geom_coupling_result["F_delta_cm-1"] = gc_F - float(spec.F)
                    geom_coupling_result["rho_delta"] = gc_rho - float(spec.rho)
                    all_warnings.extend([])
                    print(
                        f"[{cfg.get('name', 'run')}] geom coupling: "
                        f"F_geom={gc_F:.4f} cm^-1 (spec={spec.F:.4f}), "
                        f"rho_geom={gc_rho:.6f} (spec={spec.rho:.6f})"
                    )
                except Exception as exc:
                    all_warnings.append(f"geometry_coupling: {exc}")

    summary_json = exports_dir / "torsion_summary.json"
    summary: dict[str, Any] = {
        "model": "ram_lite",
        "A_cm-1": A_cm1,
        "B_cm-1": B_cm1,
        "C_cm-1": C_cm1,
        "F_cm-1": F_cm1,
        "rho": rho,
        "F4_cm-1": spec.F4,
        "F6_cm-1": spec.F6,
        "c_mk_cm-1": spec.c_mk,
        "c_k2_cm-1": spec.c_k2,
        "n_basis": n_basis,
        "J_values": J_values,
        "K_values": K_values,
        "n_levels": n_levels,
        "symmetry_mode": symmetry_mode,
        "label_levels": label_levels,
        "has_symmetry_labels": _has_sym,
        "export_symmetry_blocks": export_blocks,
        "warnings": list(dict.fromkeys(all_warnings + scan_warn)),
        "levels_csv": str(levels_csv),
    }
    if symmetry_blocks_csv is not None:
        summary["symmetry_blocks_csv"] = str(symmetry_blocks_csv)
    if tunneling_csv is not None:
        summary["tunneling_splitting_csv"] = str(tunneling_csv)
        summary["tunneling_n_rows"] = len(tunneling_rows)
    if scan_average:
        summary["scan_average_csv"] = str(scan_average_csv)
        summary["scan_average_method"] = scan_average.get("method")
        summary["scan_averaged_constants_mhz"] = scan_average.get("averaged_constants_mhz")
        summary["scan_averaged_constants_cm-1"] = scan_average.get("averaged_constants_cm-1")
        if thermal_states_csv is not None:
            summary["thermal_states_csv"] = str(thermal_states_csv)
    if transition_obj is not None:
        summary["transition_rms_cm-1"] = float(transition_obj["rms_cm-1"])
        summary["transition_warnings"] = list(dict.fromkeys(transition_obj.get("warnings", [])))
        if transition_csv is not None:
            summary["transition_objective_csv"] = str(transition_csv)
        if transition_summary_csv is not None:
            summary["transition_summary_csv"] = str(transition_summary_csv)
    if uncertainty_csv is not None:
        summary["torsion_parameter_uncertainty_csv"] = str(uncertainty_csv)
    if fitted_pot_dict is not None:
        summary["scan_fit_rms_cm-1"] = float(scan_fit_result.get("rms_cm1", float("nan")))
        summary["scan_fit_harmonics"] = scan_fit_result.get("harmonics", [])
        summary["scan_fit_n_points"] = scan_fit_result.get("n_points", 0)
        if scan_fit_result.get("scan_fit_csv"):
            summary["scan_fit_csv"] = scan_fit_result["scan_fit_csv"]
    if plot_paths:
        summary["torsion_plots"] = [str(p) for p in plot_paths]
    if lam_report:
        summary["lam_correction"] = lam_report
    if geom_coupling_result:
        summary["geometry_coupling"] = geom_coupling_result
    if auto_assign_csv is not None:
        summary["auto_assign_csv"] = str(auto_assign_csv)
        summary["auto_assign_n_matched"] = int(auto_assign_result.get("n_matched", 0))
        summary["auto_assign_n_unmatched"] = int(auto_assign_result.get("n_unmatched", 0))
        summary["auto_assign_rms_cm-1"] = float(auto_assign_result.get("rms_cm-1", float("nan")))
        summary["auto_assign_method"] = str(auto_assign_result.get("method_used", ""))
    if fit_result:
        summary["fitting_rms_cm-1"] = float(fit_result.get("rms_cm-1", float("nan")))
        summary["fitting_rms_cm-1_init"] = float(fit_result.get("rms_cm-1_init", float("nan")))
        summary["fitting_converged"] = bool(fit_result.get("converged", False))
        summary["fitting_n_iter"] = int(fit_result.get("n_iter", 0))
        summary["fitting_param_names"] = list(fit_result.get("param_names", []))
        if fit_result.get("stages"):
            summary["fitting_stages"] = fit_result["stages"]
        if fit_result.get("fitted_transition_rms_cm-1") is not None:
            summary["fitted_transition_rms_cm-1"] = float(fit_result["fitted_transition_rms_cm-1"])
        if fitting_csv is not None:
            summary["fitting_csv"] = str(fitting_csv)
        if fit_covariance_csv is not None:
            summary["fitting_covariance_csv"] = str(fit_covariance_csv)
        if fit_correlation_csv is not None:
            summary["fitting_correlation_csv"] = str(fit_correlation_csv)
        if fitted_levels_csv is not None:
            summary["fitted_levels_csv"] = str(fitted_levels_csv)
        if fitted_transition_csv is not None:
            summary["fitted_transition_objective_csv"] = str(fitted_transition_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{cfg.get('name', 'run')}] torsion CSV  : {levels_csv}")
    print(f"[{cfg.get('name', 'run')}] torsion info : {summary_json}")
    return {
        "torsion_levels_csv": levels_csv,
        "torsion_summary_json": summary_json,
        "torsion_summary": summary,
        "torsion_transition_objective_csv": transition_csv,
        "torsion_parameter_uncertainty_csv": uncertainty_csv,
        "torsion_symmetry_blocks_csv": symmetry_blocks_csv,
        "torsion_tunneling_splitting_csv": tunneling_csv,
        "torsion_scan_average_csv": scan_average_csv,
        "torsion_thermal_states_csv": thermal_states_csv,
        "torsion_auto_assign_csv": auto_assign_csv,
        "torsion_plots": plot_paths,
        "torsion_fit_params_csv": fitting_csv,
        "torsion_fit_covariance_csv": fit_covariance_csv,
        "torsion_fit_correlation_csv": fit_correlation_csv,
        "torsion_fitted_levels_csv": fitted_levels_csv,
        "torsion_fitted_transition_objective_csv": fitted_transition_csv,
        "torsion_transition_summary_csv": transition_summary_csv,
    }


def _build_torsion_spec_from_config(
    tcfg: dict,
    abc_cm1: np.ndarray,
) -> tuple["TorsionHamiltonianSpec", str | None, bool]:
    """
    Build a TorsionHamiltonianSpec from a torsion_hamiltonian config block.

    Returns (spec, symmetry_mode, label_levels).
    Supports Phase-1 fields: F4, F6, c_mk, c_k2, F_alpha, symmetry_mode, label_levels.
    """
    A_cm1 = float(tcfg.get("A_cm-1", abc_cm1[0]))
    B_cm1 = float(tcfg.get("B_cm-1", abc_cm1[1]))
    C_cm1 = float(tcfg.get("C_cm-1", abc_cm1[2]))
    F_cm1 = float(tcfg["F"])
    rho = float(tcfg.get("rho", 0.0))
    F4 = float(tcfg.get("F4", 0.0))
    F6 = float(tcfg.get("F6", 0.0))
    c_mk = float(tcfg.get("c_mk", 0.0))
    c_k2 = float(tcfg.get("c_k2", 0.0))
    n_basis = int(tcfg.get("n_basis", 7))
    units = str(tcfg.get("units", "cm-1"))

    p = tcfg.get("potential", {}) or {}
    v0 = float(p.get("v0", 0.0))
    vcos = {int(k): float(v) for k, v in (p.get("vcos", {}) or {}).items()}
    vsin = {int(k): float(v) for k, v in (p.get("vsin", {}) or {}).items()}
    pot = TorsionFourierPotential(v0=v0, vcos=vcos, vsin=vsin, units=units)

    F_alpha = None
    fa_cfg = tcfg.get("F_alpha")
    if isinstance(fa_cfg, dict) and "f0" in fa_cfg:
        F_alpha = TorsionEffectiveConstantFourier(
            f0=float(fa_cfg["f0"]),
            fcos={int(k): float(v) for k, v in (fa_cfg.get("fcos", {}) or {}).items()},
            fsin={int(k): float(v) for k, v in (fa_cfg.get("fsin", {}) or {}).items()},
            units=str(fa_cfg.get("units", "cm-1")),
        )

    spec = TorsionHamiltonianSpec(
        F=F_cm1, rho=rho, F4=F4, F6=F6, c_mk=c_mk, c_k2=c_k2,
        A=A_cm1, B=B_cm1, C=C_cm1,
        potential=pot, F_alpha=F_alpha,
        n_basis=n_basis, units=units,
    )

    symmetry_mode = tcfg.get("symmetry_mode") or None
    label_levels = bool(tcfg.get("label_levels", False))
    return spec, symmetry_mode, label_levels


def _collect_level_rows(
    spec: "TorsionHamiltonianSpec",
    J_values: list[int],
    K_values: list[int],
    n_levels: int,
    symmetry_mode: str | None,
    label_levels: bool,
    return_blocks: bool = False,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Solve and collect torsion level rows for a grid of (J, K) blocks."""
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for J in J_values:
        for K in K_values:
            out = solve_ram_lite_levels(
                spec, J=J, K=K, n_levels=n_levels,
                symmetry_mode=symmetry_mode, label_levels=label_levels,
                return_blocks=return_blocks,
            )
            all_warnings.extend(list(out.get("warnings", [])))
            sym_labels = out.get("symmetry_labels")
            sym_sublabels = out.get("symmetry_sublabels")
            sym_purity = out.get("symmetry_purity")
            for level_idx, energy in enumerate(out["energies_cm-1"]):
                row: dict[str, Any] = {
                    "J": J, "K": K,
                    "level_index": level_idx,
                    "energy_cm-1": float(energy),
                }
                if sym_labels is not None and level_idx < len(sym_labels):
                    row["symmetry_label"] = str(sym_labels[level_idx])
                if sym_sublabels is not None and level_idx < len(sym_sublabels):
                    row["symmetry_sublabel"] = str(sym_sublabels[level_idx])
                if sym_purity is not None and level_idx < len(sym_purity):
                    row["symmetry_purity"] = float(sym_purity[level_idx])
                rows.append(row)

            if return_blocks:
                for block_name, block in (out.get("symmetry_blocks") or {}).items():
                    for block_idx, energy in enumerate(block.get("energies_cm-1", [])):
                        block_rows.append(
                            {
                                "J": J,
                                "K": K,
                                "block": str(block_name),
                                "residue": int(block.get("residue", -1)),
                                "block_level_index": int(block_idx),
                                "energy_cm-1": float(energy),
                            }
                        )
    return rows, all_warnings, block_rows


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
    scan_abc_cm1, scan_warn, _scan_average = _torsion_scan_feed_constants(
        cfg=cfg, elems=elems, coords=np.asarray(coords, dtype=float), masses=masses
    )
    if scan_abc_cm1 is not None:
        abc_cm1 = np.asarray(scan_abc_cm1, dtype=float)

    tcfg, _scan_fit_result, scan_fit_warn = _apply_fitted_scan_potential(tcfg, exports_dir=None)
    spec, symmetry_mode, label_levels = _build_torsion_spec_from_config(tcfg, abc_cm1)
    n_levels = int(tcfg.get("n_levels", 8))
    J_values = [int(x) for x in (tcfg.get("J_values") or [0])]
    K_values = [int(x) for x in (tcfg.get("K_values") or [0])]
    rows, level_warnings, _block_rows = _collect_level_rows(
        spec, J_values, K_values, n_levels, symmetry_mode, label_levels
    )
    return rows, list(scan_warn) + list(scan_fit_warn) + level_warnings


def _torsion_scan_feed_constants(
    *,
    cfg: dict,
    elems: list[str],
    coords: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    """
    If torsion_hamiltonian.scan is present, compute hindered-rotor torsion-averaged
    rotational constants and return them in cm^-1 for feeding into RAM-lite inputs.
    """
    tcfg = cfg.get("torsion_hamiltonian", {}) or {}
    scan_cfg = tcfg.get("scan")
    if not isinstance(scan_cfg, dict):
        return None, [], {}
    points = scan_cfg.get("grid_points")
    if not isinstance(points, list) or len(points) == 0:
        return None, ["torsion_hamiltonian.scan is present but has no grid_points; using direct geometry constants."], {}

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
        return None, warnings + ["No valid torsion scan points found; using direct geometry constants."], {}

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
            return None, warnings + ["hindered_rotor_model is invalid; using direct geometry constants."], {}
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
    elif mode in {"quantum_thermal", "thermal_quantum"}:
        hr = scan_cfg.get("hindered_rotor_model") or {}
        if not isinstance(hr, dict):
            return None, warnings + ["hindered_rotor_model is invalid; using direct geometry constants."], {}
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
        out = average_torsion_scan_quantum_thermal(
            elems,
            scan,
            model,
            masses=masses,
            temperature_K=float(scan_cfg.get("temperature_K", 298.15)),
            max_states=int(scan_cfg.get("max_states", 6)),
        )
    else:
        out = average_torsion_scan_boltzmann(
            elems, scan, masses=masses, temperature_K=float(scan_cfg.get("temperature_K", 298.15))
        )

    avg_mhz = np.asarray(out["averaged_constants"], dtype=float)
    avg_cm1 = avg_mhz / _MHZ_PER_CM1
    scan_average = {
        "method": str(out.get("method", mode)),
        "phi_radians": np.asarray(out.get("phi_radians", []), dtype=float).tolist(),
        "weights": np.asarray(out.get("weights", []), dtype=float).tolist(),
        "grid_constants_mhz": np.asarray(out.get("grid_constants", []), dtype=float).tolist(),
        "averaged_constants_mhz": avg_mhz.tolist(),
        "averaged_constants_cm-1": avg_cm1.tolist(),
        "state_index": int(out.get("state_index", scan_cfg.get("state_index", 0))) if "state_populations" not in out else None,
        "states_used": out.get("states_used"),
        "state_populations": np.asarray(out.get("state_populations", []), dtype=float).tolist(),
        "torsional_energies_cm1": np.asarray(out.get("torsional_energies_cm1", []), dtype=float).tolist(),
    }
    return avg_cm1, warnings + list(out.get("warnings", [])), scan_average


def _write_torsion_scan_average_csv(exports_dir: Path, scan_average: dict[str, Any]) -> Path:
    """Write scan-grid rotational constants and torsional weights."""
    path = exports_dir / "torsion_scan_average.csv"
    phi = np.asarray(scan_average.get("phi_radians", []), dtype=float)
    weights = np.asarray(scan_average.get("weights", []), dtype=float)
    grid = np.asarray(scan_average.get("grid_constants_mhz", []), dtype=float)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "grid_index",
                "phi_deg",
                "weight",
                "A_mhz",
                "B_mhz",
                "C_mhz",
            ],
        )
        writer.writeheader()
        n = min(phi.size, weights.size, grid.shape[0] if grid.ndim == 2 else 0)
        for i in range(n):
            writer.writerow(
                {
                    "grid_index": i,
                    "phi_deg": float(np.degrees(phi[i])),
                    "weight": float(weights[i]),
                    "A_mhz": float(grid[i, 0]),
                    "B_mhz": float(grid[i, 1]),
                    "C_mhz": float(grid[i, 2]),
                }
            )
    return path


def _write_torsion_thermal_states_csv(exports_dir: Path, scan_average: dict[str, Any]) -> Path | None:
    """Write state populations from quantum-thermal torsion scan averaging."""
    pops = np.asarray(scan_average.get("state_populations", []), dtype=float)
    energies = np.asarray(scan_average.get("torsional_energies_cm1", []), dtype=float)
    if pops.size == 0:
        return None
    path = exports_dir / "torsion_thermal_states.csv"
    e0 = float(np.min(energies)) if energies.size else 0.0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "state_index",
                "vt_label",
                "energy_cm-1",
                "relative_energy_cm-1",
                "population",
            ],
        )
        writer.writeheader()
        for i, pop in enumerate(pops):
            energy = float(energies[i]) if i < energies.size else float("nan")
            writer.writerow(
                {
                    "state_index": i,
                    "vt_label": f"vt{i}",
                    "energy_cm-1": energy,
                    "relative_energy_cm-1": energy - e0 if np.isfinite(energy) else float("nan"),
                    "population": float(pop),
                }
            )
    return path


def _write_transition_group_summary_csv(exports_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    """Write Phase-4 residual diagnostics grouped by branch and symmetry pair."""
    if not rows:
        return None
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        branch = f"dJ={int(r['J_hi']) - int(r['J_lo'])},dK={int(r['K_hi']) - int(r['K_lo'])}"
        symmetry = f"{r.get('symmetry_lo', '')}->{r.get('symmetry_hi', '')}"
        level_pair = f"{int(r['level_lo'])}->{int(r['level_hi'])}"
        groups[(branch, symmetry, level_pair)].append(float(r["residual_cm-1"]))
    path = exports_dir / "torsion_transition_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["branch", "symmetry_pair", "level_pair", "n", "rms_cm-1", "mean_residual_cm-1"],
        )
        writer.writeheader()
        for (branch, symmetry, level_pair), vals in sorted(groups.items()):
            arr = np.asarray(vals, dtype=float)
            writer.writerow(
                {
                    "branch": branch,
                    "symmetry_pair": symmetry,
                    "level_pair": level_pair,
                    "n": int(arr.size),
                    "rms_cm-1": float(np.sqrt(np.mean(arr ** 2))),
                    "mean_residual_cm-1": float(np.mean(arr)),
                }
            )
    return path


def _write_fit_matrix_csv(exports_dir: Path, name: str, param_names: list[str], matrix: Any) -> Path | None:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    path = exports_dir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["parameter"] + [str(p) for p in param_names]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pname, row in zip(param_names, arr):
            out = {"parameter": pname}
            out.update({col: float(val) for col, val in zip(param_names, row)})
            writer.writerow(out)
    return path


def _write_torsion_phase2_plots(
    *,
    run_dir: Path,
    scan_fit_result: dict,
    scan_average: dict[str, Any],
) -> list[Path]:
    """Best-effort torsion diagnostics plots; returns paths written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[Path] = []
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scan_fit_csv = scan_fit_result.get("scan_fit_csv")
    if scan_fit_csv:
        try:
            data = np.genfromtxt(scan_fit_csv, delimiter=",", names=True, dtype=float)
            if data.size > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(data["phi_deg"], data["energy_cm1"], "o", label="scan")
                ax.plot(data["phi_deg"], data["V_fitted_cm1"], "-", label="fit")
                ax.set_xlabel("torsion angle (deg)")
                ax.set_ylabel("V(phi) (cm^-1)")
                ax.set_title("Torsion Scan Potential")
                ax.legend()
                fig.tight_layout()
                path = plots_dir / "torsion_scan_fit.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
        except Exception:
            pass

    phi = np.asarray(scan_average.get("phi_radians", []), dtype=float)
    weights = np.asarray(scan_average.get("weights", []), dtype=float)
    grid = np.asarray(scan_average.get("grid_constants_mhz", []), dtype=float)
    if phi.size and weights.size == phi.size:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(np.degrees(phi), weights, marker="o")
            ax.set_xlabel("torsion angle (deg)")
            ax.set_ylabel("weight")
            ax.set_title("Torsional Averaging Weights")
            fig.tight_layout()
            path = plots_dir / "torsion_weights.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
        except Exception:
            pass
    if phi.size and grid.ndim == 2 and grid.shape[0] == phi.size and grid.shape[1] == 3:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            for idx, label in enumerate(("A", "B", "C")):
                ax.plot(np.degrees(phi), grid[:, idx], marker="o", label=label)
            ax.set_xlabel("torsion angle (deg)")
            ax.set_ylabel("rotational constant (MHz)")
            ax.set_title("Scan Rotational Constants")
            ax.legend()
            fig.tight_layout()
            path = plots_dir / "torsion_scan_constants.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
        except Exception:
            pass
    return paths


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
