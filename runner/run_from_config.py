#!/usr/bin/env python3
"""
Run a hybrid quantize job from a YAML config file.

Two input styles are supported:

  New-style — fully generalized (any molecule):
    Supply ``elements``, ``geometry``, and ``isotopologues`` directly in the YAML.
    See configs/template.yaml for a fully-documented example.

  Legacy style — named molecule shortcut:
    Supply a ``molecule`` key (water, so2, co2, …) to dispatch to a pre-built
    runner in molecule_runners/.  Accepts optional ``preset``, ``orca_method``, ``orca_basis``.

Usage
-----
  python runner/run_from_config.py configs/template.yaml          # new-style
  python runner/run_from_config.py configs/example_water.yaml     # legacy

Environment overrides (legacy style, highest priority in ``get_run_settings``):
  QUANTIZE_ORCA_METHOD, QUANTIZE_ORCA_BASIS (aliases: ORCA_METHOD, ORCA_BASIS)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    print("NumPy is required. Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from None

try:
    import yaml
except ModuleNotFoundError:
    print(
        "PyYAML is required for run_from_config.py.\n"
        "  pip install PyYAML\n"
        "or: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

# Same molecule keys as ``run_molecule.py`` (maps CLI name → runner module).
RUNNER_MODULES: dict[str, str] = {
    "so2": "molecule_runners.run_SO2",
    "ocs": "molecule_runners.run_OCS",
    "co2": "molecule_runners.run_CO2",
    "water": "molecule_runners.run_water",
    "methanol": "molecule_runners.run_methanol_vt0_staggered",
    "methanol_vt0_staggered": "molecule_runners.run_methanol_vt0_staggered",
    "benzene": "molecule_runners.run_benzene",
    "formaldehyde": "molecule_runners.run_formaldehyde",
    "naphthalene": "molecule_runners.run_naphthalene",
}

_QUANTIZE_ORCA_ENV = ("QUANTIZE_ORCA_METHOD", "QUANTIZE_ORCA_BASIS")
_ORCA_EXE_KEY = "ORCA_EXE"


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping (dictionary).")
    return data


def _apply_env_overrides(cfg: dict) -> dict[str, str | None]:
    """Apply YAML env overrides; return previous values to restore."""
    saved: dict[str, str | None] = {}
    for k in _QUANTIZE_ORCA_ENV:
        saved[k] = os.environ.get(k)
    for k in _QUANTIZE_ORCA_ENV:
        os.environ.pop(k, None)

    method = cfg.get("orca_method")
    basis = cfg.get("orca_basis")
    if method is not None and str(method).strip():
        os.environ["QUANTIZE_ORCA_METHOD"] = str(method).strip()
    if basis is not None and str(basis).strip():
        os.environ["QUANTIZE_ORCA_BASIS"] = str(basis).strip()

    exe = cfg.get("orca_exe")
    if exe is not None and str(exe).strip():
        saved[_ORCA_EXE_KEY] = os.environ.get(_ORCA_EXE_KEY)
        os.environ[_ORCA_EXE_KEY] = str(exe).strip()

    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for k in _QUANTIZE_ORCA_ENV:
        os.environ.pop(k, None)
    for k in _QUANTIZE_ORCA_ENV:
        v = saved.get(k)
        if v is not None:
            os.environ[k] = v
    if _ORCA_EXE_KEY in saved:
        v = saved[_ORCA_EXE_KEY]
        if v is None:
            os.environ.pop(_ORCA_EXE_KEY, None)
        else:
            os.environ[_ORCA_EXE_KEY] = v


def _run_generic(cfg: dict) -> None:
    """Dispatch to the generic runner for new-style YAML (has 'elements' key)."""
    from runner.run_generic import main as generic_main
    generic_main(cfg)


def _run_legacy(cfg: dict) -> None:
    """Dispatch to a named molecule runner for legacy YAML (has 'molecule' key)."""
    molecule = str(cfg.get("molecule", "")).strip().lower()
    if not molecule:
        raise SystemExit("YAML must set either 'elements' (new-style) or 'molecule' (legacy).")
    if molecule not in RUNNER_MODULES:
        valid = ", ".join(sorted(RUNNER_MODULES.keys()))
        raise SystemExit(f"Unknown molecule '{molecule}'. Valid: {valid}")

    preset = cfg.get("preset")
    if preset is not None:
        preset = str(preset).strip().upper()
        if preset not in ("FAST_DEBUG", "BALANCED", "STRICT"):
            raise SystemExit(f"Unknown preset '{preset}'. Use FAST_DEBUG, BALANCED, or STRICT.")

    saved_env = _apply_env_overrides(cfg)
    try:
        mod_name = RUNNER_MODULES[molecule]
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "PRESET_OVERRIDE"):
            mod.PRESET_OVERRIDE = preset
        print(f"[run_from_config] molecule={molecule} module={mod_name} preset={preset}")
        if os.environ.get("QUANTIZE_ORCA_METHOD"):
            print(f"[run_from_config] QUANTIZE_ORCA_METHOD={os.environ['QUANTIZE_ORCA_METHOD']}")
        if os.environ.get("QUANTIZE_ORCA_BASIS"):
            print(f"[run_from_config] QUANTIZE_ORCA_BASIS={os.environ['QUANTIZE_ORCA_BASIS']}")
        if os.environ.get(_ORCA_EXE_KEY):
            print(f"[run_from_config] {_ORCA_EXE_KEY}={os.environ[_ORCA_EXE_KEY]}")
        if not hasattr(mod, "main"):
            raise SystemExit(f"Runner {mod_name} has no main().")
        mod.main()
    finally:
        _restore_env(saved_env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run quantize from a YAML config. "
            "New-style: supply 'elements' + 'isotopologues' (see configs/template.yaml). "
            "Legacy: supply 'molecule' key to use a pre-built runner."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML file",
    )
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg = _load_config(cfg_path)

    if "elements" in cfg:
        _run_generic(cfg)
    else:
        _run_legacy(cfg)


if __name__ == "__main__":
    main()
