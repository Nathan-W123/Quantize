"""Estimate the geometry error of a level of theory without knowing the answer.

`quantum_prior_sigma_ang` is defined as the displacement over which the quantum
surface is trusted -- roughly the geometry error of the method. Setting it from
the measured error against a reference structure would be circular: the whole
point is to handle molecules that have no accepted structure.

It can be estimated honestly instead, from the spread between two levels of
theory. If a cheap method and a better one disagree by d, the cheap one is
unlikely to be much better than d. That requires no experimental structure at
all, so it is available for exactly the molecules this project is aimed at.

This script reports that spread alongside the true error against the published
reference, so the estimator can be checked where a reference does exist.

    python scripts/estimate_theory_error.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from dev.monofluoro_references import MOLECULES  # noqa: E402
from scripts.monofluoro_benchmark import errors, start_geometry  # noqa: E402

#: (label, method, basis). The first is the level used throughout the benchmark.
LEVELS = (
    ("RHF/6-31G", "hf", "6-31g"),
    ("B3LYP/6-31G(d)", "b3lyp", "6-31g(d)"),
)

OUT = _ROOT / "output" / "theory_error_estimate.json"


def bond_rms_difference(mol, a, b) -> float:
    """RMS difference in bond lengths between two geometries, in mA."""
    ia, ib = mol.internal_coordinates(a), mol.internal_coordinates(b)
    d = [(ia[k] - ib[k]) * 1000 for k in mol.bonds]
    return float(np.sqrt(np.mean(np.square(d))))


def main() -> None:
    print("  Estimating the geometry error of RHF/6-31G without using the answer.\n")
    print(f"  {'molecule':<18}{'spread':>10}{'true err':>10}{'ratio':>8}"
          f"{'  (both RMS bond, mA)'}")
    print("  " + "-" * 62)

    out = []
    for mol in MOLECULES:
        start = start_geometry(mol)
        geoms = {}
        for label, method, basis in LEVELS:
            t0 = time.time()
            b = get_backend("pyscf_hf")(elems=list(mol.elems), method=method,
                                        basis=basis)
            geoms[label] = b.optimise(start)
            print(f"    {mol.name} {label}: {time.time() - t0:.0f}s", flush=True)

        cheap = LEVELS[0][0]
        spread = bond_rms_difference(mol, geoms[cheap], geoms[LEVELS[1][0]])
        true_err = errors(mol, geoms[cheap])["rms_bond_ma"]
        better_err = errors(mol, geoms[LEVELS[1][0]])["rms_bond_ma"]
        out.append({
            "molecule": mol.name, "key": mol.key,
            "spread_ma": spread, "true_error_ma": true_err,
            "better_level_error_ma": better_err,
            "cf_err_cheap_ma": errors(mol, geoms[cheap])["cf_err_ma"],
            "cf_err_better_ma": errors(mol, geoms[LEVELS[1][0]])["cf_err_ma"],
        })
        print(f"  {mol.name:<18}{spread:>10.1f}{true_err:>10.1f}"
              f"{true_err / spread:>8.2f}")

    sp = np.mean([r["spread_ma"] for r in out])
    tr = np.mean([r["true_error_ma"] for r in out])
    print("  " + "-" * 62)
    print(f"  {'mean':<18}{sp:>10.1f}{tr:>10.1f}{tr / sp:>8.2f}")
    print(f"\n  The spread between the two levels is an answer-free estimate of the")
    print(f"  cheap level's error. Here it {'under' if sp < tr else 'over'}states it "
          f"by a factor of {tr / sp:.2f}.")
    print(f"  => quantum_prior_sigma_ang ~ {sp / 1000:.3f} A from the estimator, "
          f"{tr / 1000:.3f} A from the truth.")

    print(f"\n  {'molecule':<18}{'C-F cheap':>12}{'C-F better':>12}")
    print("  " + "-" * 42)
    for r in out:
        print(f"  {r['molecule']:<18}{r['cf_err_cheap_ma']:>+12.1f}"
              f"{r['cf_err_better_ma']:>+12.1f}")
    print(f"\n  Better-level RMS bond error: "
          f"{np.mean([r['better_level_error_ma'] for r in out]):.1f} mA "
          f"(vs {tr:.1f} for the cheap level).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
