from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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

    quantum = _expect_mapping(cfg, "quantum")
    backend = str(quantum.get("backend", "orca")).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError("'quantum.backend' must be one of orca, psi4, or none.")


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


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
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
    lines.extend(["", "## Final Geometry", "", "| atom | element | x (Ang) | y (Ang) | z (Ang) |", "|---:|---|---:|---:|---:|"])
    for i, (elem, xyz) in enumerate(zip(result["elems"], np.asarray(best["coords"], dtype=float))):
        lines.append(f"| {i} | {elem} | {xyz[0]:.8f} | {xyz[1]:.8f} | {xyz[2]:.8f} |")
    lines.extend(["", "## Residuals", "", "| isotopologue | component | target MHz | calculated MHz | residual MHz |", "|---|---:|---:|---:|---:|"])
    for row in result.get("residual_rows", []):
        lines.append(
            f"| {row['isotopologue']} | {row['component']} | {row['target_mhz']:.6f} | "
            f"{row['calculated_mhz']:.6f} | {row['residual_mhz']:.6f} |"
        )
    lines.extend(["", "## Outputs", "", "- `exports/final_geometry.csv`", "- `exports/residuals.csv`", "- `plots/residuals.png`", "- `plots/singular_values.png`", "- `plots/convergence.png`"])
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
    run_dir = Path(result.get("run_dir") or ".").resolve()
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    geom_csv = exports_dir / "final_geometry.csv"
    residual_csv = exports_dir / "residuals.csv"
    write_final_geometry_csv(geom_csv, result["elems"], result["best"]["coords"])

    rows = residual_rows(result["best"]["coords"], result["best"]["spectral_isotopologues_snapshot"])
    result["residual_rows"] = rows
    result["singular_values"] = singular_values(result["best"]["coords"], result["best"]["spectral_isotopologues_snapshot"]).tolist()
    write_residuals_csv(residual_csv, rows)

    report_md = run_dir / "report.md"
    write_markdown_report(report_md, result)
    plot_paths = write_plots(run_dir, result)
    return {"geometry_csv": geom_csv, "residuals_csv": residual_csv, "report_md": report_md, "plots": plot_paths}
