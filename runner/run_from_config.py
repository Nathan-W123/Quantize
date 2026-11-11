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

import os
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


def build_optimizer_from_config(
    cfg: dict,
    *,
    multistart: bool | None = None,
    verbose: bool = True,
):
    """Build the :class:`MolecularOptimizer` a config describes, unrun.

    ``quantize uncertainty`` imported this name for as long as the subcommand
    has existed, and it was never written: the import failed, so every
    uncertainty path in the CLI was dead on arrival. The work it needs is the
    same config translation a fit does, which lived inline in
    :func:`runner.run_generic.main` and is now :func:`build_run_inputs`.

    Which geometry the optimizer starts from is the one real decision here, and
    the codebase had already made it. A config asks for ``preset["n_starts"]``
    seed geometries, and :func:`backend.multistart.select_best_result` is the
    canonical rule for choosing among their results -- lowest quantum energy
    within a spectral gate. So with ``n_starts > 1`` this runs the multistart
    the config asked for and seeds the returned optimizer at the selected
    result's geometry; the caller's own ``run()`` then converges from there,
    which is the point the uncertainty is meant to describe. Pass
    ``multistart=False`` to skip that and seed from the config geometry alone,
    which is much cheaper and gives a posterior around whatever local minimum
    that start reaches -- not the engine's answer.

    The returned optimizer is *not* run: the caller decides when to spend the
    quantum chemistry.
    """
    import numpy as np

    from backend.multistart import run_multistart, select_best_result
    from backend.quantize import MolecularOptimizer
    from runner.run_generic import build_run_inputs

    inputs = build_run_inputs(cfg, verbose=verbose)
    kwargs = dict(inputs["optimizer_kwargs"])
    starts = inputs["starts"]
    elems = inputs["elems"]
    isotopologues = inputs["isotopologues"]
    name = inputs["name"]
    preset = inputs["preset"]

    if multistart is None:
        multistart = len(starts) > 1

    coords = np.asarray(starts[0], dtype=float)
    if multistart and len(starts) > 1:
        results = run_multistart(
            starts=starts,
            elems=elems,
            isotopologues=isotopologues,
            optimizer_kwargs=kwargs,
            max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
            job_name=name,
        )
        best = select_best_result(results, spectral_gate_abs=0.05, spectral_gate_rel=2.0)
        coords = np.asarray(best["coords"], dtype=float)
        if verbose:
            print(
                f"[{name}] seeding from the best of {len(starts)} starts "
                f"(select_best_result: energy within the spectral gate)"
            )
    elif verbose and len(starts) > 1:
        print(
            f"[{name}] multistart skipped: seeding from the config geometry, so "
            f"this is a local minimum, not the {len(starts)}-start selection"
        )

    # run_multistart pops these two and turns them into a per-start workdir;
    # MolecularOptimizer itself takes neither, so a single build has to do the
    # same or the constructor raises TypeError on base_workdir.
    base_workdir = kwargs.pop("base_workdir", ".")
    kwargs.pop("job_name", None)
    workdir = os.path.join(str(base_workdir), "single", str(name))
    os.makedirs(workdir, exist_ok=True)
    kwargs["workdir"] = workdir
    kwargs.setdefault("psi4_output_file", os.path.join(workdir, "psi4.out"))

    return MolecularOptimizer(
        coords=coords, elems=elems, isotopologues=isotopologues, **kwargs
    )


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
