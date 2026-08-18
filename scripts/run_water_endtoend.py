"""Drive a real end-to-end water run on the analytic PES.

Registers the analytic backend, then hands the config to the same runner the
CLI uses, so the whole pipeline is exercised: validation, run directory,
internal coordinates, autoconfig, multistart, the B_0 -> B_e correction chain,
reporting and exports.

    python scripts/run_water_endtoend.py [config.yaml]

Prints the recovered structure against the accepted equilibrium geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers "analytic_water")
from runner.run_generic import main as run_generic_main  # noqa: E402
from runner.usability import load_config, prepare_run_directory, validate_config  # noqa: E402

R_E, THETA_E = 0.95784, 104.508


def structure(coords):
    x = np.asarray(coords, dtype=float)
    r1 = np.linalg.norm(x[1] - x[0])
    r2 = np.linalg.norm(x[2] - x[0])
    u1 = (x[1] - x[0]) / r1
    u2 = (x[2] - x[0]) / r2
    theta = np.degrees(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
    return 0.5 * (r1 + r2), theta


def main() -> None:
    cfg_path = Path(
        sys.argv[1] if len(sys.argv) > 1
        else _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"
    )
    cfg = load_config(cfg_path)
    validate_config(cfg)
    prepare_run_directory(cfg, cfg_path)
    result = run_generic_main(cfg)

    r, theta = structure(result["best"]["coords"])
    print("\n" + "=" * 64)
    print("  End-to-end result vs the accepted equilibrium structure")
    print("=" * 64)
    print(f"  r(O-H)  = {r:.5f} A     ({(r - R_E) * 1000:+.1f} mA from r_e = {R_E})")
    print(f"  angle   = {theta:.3f} deg  ({theta - THETA_E:+.3f} deg from "
          f"{THETA_E})")
    print(f"  run dir = {result.get('run_dir')}")


if __name__ == "__main__":
    main()
