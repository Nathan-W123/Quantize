#!/usr/bin/env python3
"""
Run a hybrid quantize job from a YAML or JSON config file.

Supply ``elements``, ``geometry``, and ``isotopologues`` in the config.
See ``configs/example_CO2.yaml`` or ``configs/template.yaml`` for examples.

Usage
-----
  python runner/run_from_config.py configs/example_CO2.yaml
  python -m cli run configs/example_CO2.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from paths import ensure_repo_paths

_ROOT = ensure_repo_paths(Path(__file__).resolve().parent.parent)

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    print("NumPy is required. Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from None

from runner.usability import ConfigError, load_config, prepare_run_directory, validate_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run quantize from a YAML or JSON config (requires 'elements' + 'isotopologues').",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument(
        "--no-run-dir",
        action="store_true",
        help="Do not create a managed output run directory.",
    )
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    try:
        cfg = load_config(cfg_path)
        validate_config(cfg)
        if not args.no_run_dir:
            run_dir = prepare_run_directory(cfg, cfg_path)
            print(f"[run_from_config] run_dir={run_dir}")
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from None

    from runner.run_generic import main as generic_main

    generic_main(cfg)


if __name__ == "__main__":
    main()
