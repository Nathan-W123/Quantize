from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from backend.spectral_model import normalize_spectral_model

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled by load_config error path
    yaml = None


COMPONENT_LABELS = ("A", "B", "C")
VALID_PRESETS = {"FAST_DEBUG", "BALANCED", "STRICT"}
VALID_BACKENDS = {"orca", "psi4", "none"}
VALID_GEOMETRY_METHODS = {"bonds", "pubchem", "coords"}
_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ConfigError(ValueError):
    """Raised when a Quantize config is invalid."""


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a Quantize YAML or JSON config file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError("PyYAML is required to read YAML configs. Install PyYAML.")
            data = yaml.safe_load(text)
        else:
            raise ConfigError("Config file must end in .yaml, .yml, or .json.")
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config {cfg_path} must contain a mapping/object at top level.")
    return data


def _expect_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping/object.")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{path}' must be a list.")
    return value


def _check_numeric_list(value: Any, path: str, n: int | None = None, positive: bool = False) -> None:
    items = _as_list(value, path)
    if n is not None and len(items) != n:
        raise ConfigError(f"'{path}' must contain {n} values; got {len(items)}.")
    for i, item in enumerate(items):
        try:
            x = float(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"'{path}[{i}]' must be numeric; got {item!r}.") from exc
        if not np.isfinite(x):
            raise ConfigError(f"'{path}[{i}]' must be finite; got {item!r}.")
        if positive and x <= 0.0:
            raise ConfigError(f"'{path}[{i}]' must be positive; got {item!r}.")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate supported legacy and generalized config shapes with clear errors."""
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a mapping/object.")

    has_elements = "elements" in cfg
    has_molecule = "molecule" in cfg
    if has_elements and has_molecule:
        raise ConfigError("Set either 'elements' for a generalized run or 'molecule' for legacy mode, not both.")
    if not has_elements and not has_molecule:
        raise ConfigError("Config must set either 'elements' or 'molecule'.")

    preset = cfg.get("preset")
    if preset is not None and str(preset).strip().upper() not in VALID_PRESETS:
        raise ConfigError(f"'preset' must be one of {', '.join(sorted(VALID_PRESETS))}.")

    output = _expect_mapping(cfg, "output")
    for key in ("root", "run_dir"):
        if key in output and output[key] is not None and not str(output[key]).strip():
            raise ConfigError(f"'output.{key}' cannot be blank.")

    if has_molecule:
        molecule = str(cfg.get("molecule", "")).strip()
        if not molecule:
            raise ConfigError("'molecule' cannot be blank.")
        return

    elements = _as_list(cfg.get("elements"), "elements")
    if not elements:
        raise ConfigError("'elements' must contain at least one atom.")
    for i, elem in enumerate(elements):
        if not isinstance(elem, str) or not _ELEMENT_RE.match(elem.strip()):
            raise ConfigError(f"'elements[{i}]' must be an element symbol like C, H, O, or Cl.")
    n_atoms = len(elements)

    geometry = _expect_mapping(cfg, "geometry")
    if geometry.get("smiles"):
        pass
    else:
        method = str(geometry.get("method", "bonds")).strip().lower()
        if method not in VALID_GEOMETRY_METHODS:
            raise ConfigError("'geometry.method' must be one of bonds, pubchem, or coords.")
        if method == "bonds":
            bonds = _as_list(geometry.get("bonds"), "geometry.bonds")
            if not bonds:
                raise ConfigError("'geometry.bonds' must contain at least one [i, j] pair.")
            for b_i, pair in enumerate(bonds):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ConfigError(f"'geometry.bonds[{b_i}]' must be a two-item list.")
                for atom_i in pair:
                    if not isinstance(atom_i, int) or atom_i < 0 or atom_i >= n_atoms:
                        raise ConfigError(
                            f"'geometry.bonds[{b_i}]' contains atom index {atom_i!r}, "
                            f"but valid indices are 0..{n_atoms - 1}."
                        )
            if "bond_lengths" in geometry and geometry["bond_lengths"] is not None:
                _check_numeric_list(geometry["bond_lengths"], "geometry.bond_lengths", len(bonds), positive=True)
        elif method == "pubchem":
            if not str(geometry.get("identifier", "")).strip():
                raise ConfigError("'geometry.identifier' is required when geometry.method is pubchem.")
        elif method == "coords":
            rows = _as_list(geometry.get("coords_angstrom"), "geometry.coords_angstrom")
            if len(rows) != n_atoms:
                raise ConfigError(f"'geometry.coords_angstrom' must have {n_atoms} coordinate rows.")
            for row_i, row in enumerate(rows):
                _check_numeric_list(row, f"geometry.coords_angstrom[{row_i}]", 3)

    isotopologues = _as_list(cfg.get("isotopologues"), "isotopologues")
    if not isotopologues:
        raise ConfigError("'isotopologues' must contain at least one entry.")
    for iso_i, iso in enumerate(isotopologues):
        if not isinstance(iso, dict):
            raise ConfigError(f"'isotopologues[{iso_i}]' must be a mapping/object.")
        prefix = f"isotopologues[{iso_i}]"
        _check_numeric_list(iso.get("masses"), f"{prefix}.masses", n_atoms, positive=True)
        comps = iso.get("components", ["A", "B", "C"])
        comps = _as_list(comps, f"{prefix}.components")
        if not comps:
            raise ConfigError(f"'{prefix}.components' must contain at least one component.")
        for comp in comps:
            if str(comp).strip().upper() not in COMPONENT_LABELS:
                raise ConfigError(f"'{prefix}.components' values must be A, B, or C.")
        n_comp = len(comps)
        _check_numeric_list(iso.get("obs_b0_mhz"), f"{prefix}.obs_b0_mhz", n_comp)
        _check_numeric_list(iso.get("alpha_mhz"), f"{prefix}.alpha_mhz", n_comp)
        _check_numeric_list(iso.get("sigma_mhz"), f"{prefix}.sigma_mhz", n_comp, positive=True)

    if "spectral_model" in cfg and cfg.get("spectral_model") is not None:
        try:
            normalize_spectral_model(str(cfg.get("spectral_model")))
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    quantum = _expect_mapping(cfg, "quantum")
    backend = str(quantum.get("backend", "orca")).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError("'quantum.backend' must be one of orca, psi4, or none.")

    _validate_rovibrational_corrections_block(cfg)
    _validate_torsion_block(cfg)


_VALID_ROVIB_MODES = {
    "hybrid_auto", "user_only", "orca_only",
    "manual_alpha", "manual_delta", "none",
    "strict_user", "strict_backend",
}


def _validate_rovibrational_corrections_block(cfg: dict[str, Any]) -> None:
    """Validate the optional rovibrational_corrections: block."""
    rc = cfg.get("rovibrational_corrections")
    if rc is None:
        return
    if not isinstance(rc, dict):
        raise ConfigError("'rovibrational_corrections' must be a mapping/object.")

    mode = rc.get("mode")
    if mode is not None and str(mode).strip().lower() not in _VALID_ROVIB_MODES:
        raise ConfigError(
            f"'rovibrational_corrections.mode' must be one of "
            f"{sorted(_VALID_ROVIB_MODES)}, got '{mode}'."
        )

    correction_table = rc.get("correction_table")
    if correction_table is not None:
        p = Path(str(correction_table)).expanduser()
        if not p.is_file():
            raise ConfigError(
                f"'rovibrational_corrections.correction_table' not found: {p}"
            )
        suf = p.suffix.lower()
        if suf not in (".csv", ".yaml", ".yml"):
            raise ConfigError(
                "'rovibrational_corrections.correction_table' must be a .csv, .yaml, or .yml file."
            )

    for frac_key in ("sigma_vib_fraction", "sigma_elec_fraction"):
        v = rc.get(frac_key)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'rovibrational_corrections.{frac_key}' must be numeric."
                ) from exc
            if fv < 0.0:
                raise ConfigError(
                    f"'rovibrational_corrections.{frac_key}' must be >= 0."
                )

    elec = rc.get("electronic_correction")
    if elec is not None and not isinstance(elec, bool):
        raise ConfigError(
            "'rovibrational_corrections.electronic_correction' must be true or false."
        )

    bob = rc.get("bob_params")
    if bob is not None and not isinstance(bob, dict):
        raise ConfigError(
            "'rovibrational_corrections.bob_params' must be a mapping of element → component → u-value."
        )


def _validate_torsion_block(cfg: dict[str, Any]) -> None:
    t = cfg.get("torsion_hamiltonian")
    if t is None:
        return
    if not isinstance(t, dict):
        raise ConfigError("'torsion_hamiltonian' must be a mapping/object.")
    if "enabled" in t and not isinstance(t.get("enabled"), bool):
        raise ConfigError("'torsion_hamiltonian.enabled' must be true or false.")
    scan = t.get("scan")
    if scan is None:
        return
    if not isinstance(scan, dict):
        raise ConfigError("'torsion_hamiltonian.scan' must be a mapping/object.")
    gps = scan.get("grid_points")
    if gps is None or not isinstance(gps, list) or len(gps) == 0:
        raise ConfigError("'torsion_hamiltonian.scan.grid_points' must be a non-empty list.")
    for i, gp in enumerate(gps):
        if not isinstance(gp, dict):
            raise ConfigError(f"'torsion_hamiltonian.scan.grid_points[{i}]' must be a mapping/object.")
        if "phi" not in gp:
            raise ConfigError(f"'torsion_hamiltonian.scan.grid_points[{i}].phi' is required.")
        try:
            float(gp["phi"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"'torsion_hamiltonian.scan.grid_points[{i}].phi' must be numeric."
            ) from exc
    mode = str(scan.get("mode", "quantum")).strip().lower()
    if mode not in {"quantum", "boltzmann"}:
        raise ConfigError("'torsion_hamiltonian.scan.mode' must be 'quantum' or 'boltzmann'.")
    if mode == "quantum":
        hr = scan.get("hindered_rotor_model")
        if hr is None or not isinstance(hr, dict):
            raise ConfigError(
                "'torsion_hamiltonian.scan.hindered_rotor_model' is required in quantum scan mode."
            )
        if hr.get("rotational_constant_F") is None:
            raise ConfigError(
                "'torsion_hamiltonian.scan.hindered_rotor_model.rotational_constant_F' is required in quantum mode."
            )


def safe_run_name(name: str | None) -> str:
    raw = str(name or "quantize_run").strip().lower()
    safe = _SAFE_NAME_RE.sub("_", raw).strip("._-")
    return safe or "quantize_run"


def prepare_run_directory(cfg: dict[str, Any], config_path: Path | None = None) -> Path:
    """Create and annotate an output run directory."""
    output = _expect_mapping(cfg, "output")
    explicit = output.get("run_dir")
    if explicit:
        run_dir = Path(str(explicit)).expanduser()
    else:
        root = Path(str(output.get("root", "runs"))).expanduser()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{stamp}_{safe_run_name(cfg.get('name') or cfg.get('molecule'))}"
        if run_dir.exists():
            base = run_dir
            for i in range(2, 1000):
                candidate = base.with_name(f"{base.name}_{i:03d}")
                if not candidate.exists():
                    run_dir = candidate
                    break
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "exports").mkdir(exist_ok=True)
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, run_dir / f"input{config_path.suffix.lower()}")
    cfg["_run_dir"] = str(run_dir.resolve())
    return run_dir.resolve()


def write_final_geometry_csv(path: Path, elems: list[str], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atom_index", "element", "x_angstrom", "y_angstrom", "z_angstrom"])
        for i, (elem, xyz) in enumerate(zip(elems, np.asarray(coords, dtype=float))):
            writer.writerow([i, elem, f"{xyz[0]:.10f}", f"{xyz[1]:.10f}", f"{xyz[2]:.10f}"])


def residual_rows(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from backend.spectral import SpectralEngine

    engine = SpectralEngine(spectral_isotopologues)
    rows: list[dict[str, Any]] = []
    for iso in engine.isotopologues:
        calc_all = engine.rotational_constants(coords, iso["masses"])
        idx = np.asarray(iso["component_indices"], dtype=int)
        target = iso["obs_constants"] + 0.5 * iso["alpha_constants"]
        for j, comp in enumerate(idx):
            calc = float(calc_all[int(comp)])
            rows.append(
                {
                    "isotopologue": iso["name"],
                    "component": COMPONENT_LABELS[int(comp)],
                    "target_mhz": float(target[j]),
                    "calculated_mhz": calc,
                    "residual_mhz": float(target[j] - calc),
                    "sigma_mhz": float(iso["sigma_constants"][j]),
                }
            )
    return rows


def write_residuals_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["isotopologue", "component", "target_mhz", "calculated_mhz", "residual_mhz", "sigma_mhz"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def singular_values(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> np.ndarray:
    from backend.spectral import SpectralEngine

    engine = SpectralEngine(spectral_isotopologues)
    J, _ = engine.stacked_unweighted(coords)
    return np.linalg.svd(J, compute_uv=False)


def write_markdown_report(path: Path, result: dict[str, Any], artifacts: dict[str, Any] | None = None) -> None:
    from runner.reporting import generate_rovib_report_section

    best = result["best"]
    score = result.get("score", {})
    lines = [
        f"# Quantize Report: {result.get('name', 'run')}",
        "",
        f"- Run directory: `{result.get('run_dir', '.')}`",
        f"- Best start: `{best.get('idx', 'n/a')}`",
        f"- Spectral RMS: `{float(best.get('freq_rms', np.nan)):.6f}` MHz",
        f"- Final energy: `{float(best.get('energy', np.nan)):.10g}` Eh",
    ]
    if score:
        lines.extend(
            [
                f"- Success score: `{float(score.get('score', np.nan)):.1f}`",
                f"- Constrained rank: `{score.get('constrained_rank', 'n/a')}/{score.get('internal_dof', 'n/a')}`",
            ]
        )

    iso_snapshot = best.get("spectral_isotopologues_snapshot", [])
    if iso_snapshot:
        lines.extend(["", generate_rovib_report_section(iso_snapshot)])

    lines.extend(["", "## Final Geometry", "", "| atom | element | x (Ang) | y (Ang) | z (Ang) |", "|---:|---|---:|---:|---:|"])
    for i, (elem, xyz) in enumerate(zip(result["elems"], np.asarray(best["coords"], dtype=float))):
        lines.append(f"| {i} | {elem} | {xyz[0]:.8f} | {xyz[1]:.8f} | {xyz[2]:.8f} |")
    lines.extend(["", "## Residuals", "", "| isotopologue | component | target MHz | calculated MHz | residual MHz |", "|---|---:|---:|---:|---:|"])
    for row in result.get("residual_rows", []):
        lines.append(
            f"| {row['isotopologue']} | {row['component']} | {row['target_mhz']:.6f} | "
            f"{row['calculated_mhz']:.6f} | {row['residual_mhz']:.6f} |"
        )
    if result.get("torsion_objective_rows"):
        torsion_rows = result["torsion_objective_rows"]
        is_transition_mode = "J_lo" in torsion_rows[0]
        lines.extend(
            [
                "",
                "## Torsion Objective",
                "",
                f"- Torsion RMS (cm^-1): `{float(result.get('torsion_rms_cm-1', np.nan)):.6f}`",
                "",
            ]
        )
        if is_transition_mode:
            lines.extend(
                [
                    "| J_lo | K_lo | level_lo | J_hi | K_hi | level_hi | Observed (cm^-1) | Predicted (cm^-1) | Residual (cm^-1) |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in torsion_rows:
                lines.append(
                    f"| {int(row['J_lo'])} | {int(row['K_lo'])} | {int(row['level_lo'])} | "
                    f"{int(row['J_hi'])} | {int(row['K_hi'])} | {int(row['level_hi'])} | "
                    f"{float(row['observed_cm-1']):.6f} | {float(row['predicted_cm-1']):.6f} | "
                    f"{float(row['residual_cm-1']):.6f} |"
                )
        else:
            lines.extend(
                [
                    "| J | K | Level | Observed (cm^-1) | Predicted (cm^-1) | Residual (cm^-1) |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in torsion_rows:
                lines.append(
                    f"| {int(row['J'])} | {int(row['K'])} | {int(row['level_index'])} | "
                    f"{float(row['observed_cm-1']):.6f} | {float(row['predicted_cm-1']):.6f} | "
                    f"{float(row['residual_cm-1']):.6f} |"
                )
    lines.extend(["", "## Outputs", "", "- `exports/final_geometry.csv`", "- `exports/residuals.csv`"])
    if artifacts:
        if artifacts.get("rovib_corrections_csv") is not None:
            lines.append("- `exports/rovib_corrections.csv`")
        if artifacts.get("semi_experimental_targets_csv") is not None:
            lines.append("- `exports/semi_experimental_targets.csv`")
        if artifacts.get("rovib_warnings_json") is not None:
            lines.append("- `exports/rovib_warnings.json`")
        if artifacts.get("internal_uncertainty_csv") is not None:
            lines.append("- `exports/internal_uncertainty.csv`")
        if artifacts.get("internal_covariance_csv") is not None:
            lines.append("- `exports/internal_covariance.csv`")
        if artifacts.get("internal_identifiability_csv") is not None:
            lines.append("- `exports/internal_identifiability.csv`")
        if artifacts.get("torsion_objective_csv") is not None:
            lines.append("- `exports/torsion_objective.csv`")
    lines.extend(["- `plots/residuals.png`", "- `plots/singular_values.png`", "- `plots/convergence.png`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(run_dir: Path, result: dict[str, Any]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    paths: list[Path] = []
    plots_dir = run_dir / "plots"
    rows = result.get("residual_rows", [])
    if rows:
        labels = [f"{r['isotopologue']} {r['component']}" for r in rows]
        values = [float(r["residual_mhz"]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(range(len(values)), values, color="#2f6f9f")
        ax.set_ylabel("Residual (MHz)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        path = plots_dir / "residuals.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    sv = np.asarray(result.get("singular_values", []), dtype=float)
    if sv.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.arange(1, sv.size + 1), sv, marker="o", color="#007c7a")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value")
        ax.set_title("Spectral Jacobian Singular Values")
        fig.tight_layout()
        path = plots_dir / "singular_values.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    history = result["best"].get("history") or []
    if history:
        it = [int(h.get("iteration", i + 1)) for i, h in enumerate(history)]
        freq = [float(h.get("freq_rms", np.nan)) for h in history]
        step = [float(h.get("step_norm", np.nan)) for h in history]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(it, freq, marker="o", color="#1d5fd1", label="freq RMS")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Frequency RMS (MHz)")
        ax2 = ax1.twinx()
        ax2.semilogy(it, step, marker="s", color="#c55a11", label="step norm")
        ax2.set_ylabel("Step norm (Ang)")
        fig.tight_layout()
        path = plots_dir / "convergence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_outputs(result: dict[str, Any]) -> dict[str, Path | list[Path]]:
    """Write CSV, Markdown, and plot artifacts for a completed generic run."""
    from runner.reporting import (
        export_rovib_corrections_csv,
        export_rovib_warnings_json,
        export_semi_experimental_targets_csv,
    )

    run_dir = Path(result.get("run_dir") or ".").resolve()
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    geom_csv = exports_dir / "final_geometry.csv"
    residual_csv = exports_dir / "residuals.csv"
    write_final_geometry_csv(geom_csv, result["elems"], result["best"]["coords"])

    iso_snapshot = result["best"].get("spectral_isotopologues_snapshot", [])
    rows = residual_rows(result["best"]["coords"], iso_snapshot)
    result["residual_rows"] = rows
    result["singular_values"] = singular_values(result["best"]["coords"], iso_snapshot).tolist()
    write_residuals_csv(residual_csv, rows)

    artifacts: dict[str, Any] = {
        "geometry_csv": geom_csv,
        "residuals_csv": residual_csv,
    }

    # Rovib correction exports (written whenever isotopologue data exists).
    if iso_snapshot:
        rovib_csv = export_rovib_corrections_csv(iso_snapshot, exports_dir / "rovib_corrections.csv")
        semi_csv = export_semi_experimental_targets_csv(iso_snapshot, exports_dir / "semi_experimental_targets.csv")
        warn_json = export_rovib_warnings_json(iso_snapshot, exports_dir / "rovib_warnings.json")
        artifacts["rovib_corrections_csv"] = rovib_csv
        artifacts["semi_experimental_targets_csv"] = semi_csv
        artifacts["rovib_warnings_json"] = warn_json

    # Internal-coordinate uncertainty / identifiability exports.
    cfg = result.get("cfg", {}) or {}
    coord_mode = str(cfg.get("coordinate_mode", "cartesian")).strip().lower()
    if coord_mode == "internal" and iso_snapshot:
        from backend.internal_fit import InternalCoordinateSet, spectral_jacobian_q, build_internal_priors
        from backend.spectral import SpectralEngine
        from backend.uncertainty import uncertainty_table, compute_uncertainty
        from backend.identifiability import identifiability_table

        ic_cfg = cfg.get("internal_coordinates", {}) or {}
        use_dihedrals = bool(ic_cfg.get("use_dihedrals", False))
        damping = max(float(ic_cfg.get("damping", 1e-6)), 1e-14)
        sigma_bond = float(ic_cfg.get("prior_sigma_bond", 0.04))
        sigma_angle_deg = float(ic_cfg.get("prior_sigma_angle_deg", 2.0))
        sigma_dihedral_deg = float(ic_cfg.get("prior_sigma_dihedral_deg", 15.0))

        coord_set = InternalCoordinateSet(result["best"]["coords"], result["elems"], use_dihedrals=use_dihedrals)
        B_active = coord_set.active_B_matrix(result["best"]["coords"])
        if B_active.shape[0] > 0:
            Bplus = InternalCoordinateSet.damped_pseudoinverse(B_active, damping)
            J_spectral, _ = SpectralEngine(iso_snapshot).stacked(result["best"]["coords"])
            Jq = spectral_jacobian_q(J_spectral, Bplus)
            _, _, sigma_prior = build_internal_priors(
                coord_set,
                result["best"]["coords"],
                sigma_bond=sigma_bond,
                sigma_angle_deg=sigma_angle_deg,
                sigma_dihedral_deg=sigma_dihedral_deg,
            )

            unc_rows = uncertainty_table(
                coord_set,
                result["best"]["coords"],
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
            )
            unc_csv = exports_dir / "internal_uncertainty.csv"
            with unc_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "name",
                        "value",
                        "value_unit",
                        "std_err",
                        "std_err_unit",
                        "ci_lo",
                        "ci_hi",
                        "ci_unit",
                    ],
                )
                writer.writeheader()
                for r in unc_rows:
                    writer.writerow(r)
            artifacts["internal_uncertainty_csv"] = unc_csv

            cov, _, _ = compute_uncertainty(
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
            )
            cov_csv = exports_dir / "internal_covariance.csv"
            with cov_csv.open("w", newline="", encoding="utf-8") as fh:
                active_names = [ic.name for ic in coord_set.active_coords()]
                writer = csv.writer(fh)
                writer.writerow(["coordinate"] + active_names)
                for i, row_name in enumerate(active_names):
                    writer.writerow([row_name] + [f"{float(v):.12e}" for v in cov[i]])
            artifacts["internal_covariance_csv"] = cov_csv

            id_rows, sv, rank = identifiability_table(coord_set, Jq, sigma_prior)
            id_csv = exports_dir / "internal_identifiability.csv"
            with id_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["name", "score", "label", "sv_rank"],
                )
                writer.writeheader()
                for r in id_rows:
                    writer.writerow(r)
            artifacts["internal_identifiability_csv"] = id_csv
            artifacts["internal_rank"] = rank
            artifacts["internal_singular_values"] = [float(x) for x in sv]

    if result.get("torsion_objective_rows"):
        torsion_csv = exports_dir / "torsion_objective.csv"
        torsion_rows = result["torsion_objective_rows"]
        first = torsion_rows[0]
        if "J_lo" in first:
            fieldnames = [
                "J_lo",
                "K_lo",
                "level_lo",
                "J_hi",
                "K_hi",
                "level_hi",
                "observed_cm-1",
                "predicted_cm-1",
                "residual_cm-1",
            ]
        else:
            fieldnames = [
                "J",
                "K",
                "level_index",
                "observed_cm-1",
                "predicted_cm-1",
                "residual_cm-1",
            ]
        with torsion_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for r in torsion_rows:
                writer.writerow(r)
        artifacts["torsion_objective_csv"] = torsion_csv

    report_md = run_dir / "report.md"
    write_markdown_report(report_md, result, artifacts=artifacts)
    plot_paths = write_plots(run_dir, result)
    artifacts["report_md"] = report_md
    artifacts["plots"] = plot_paths
    return artifacts
