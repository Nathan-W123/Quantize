"""How much authority should the quantum surface have?

The split objective partitions the parameter space hard: whatever survives the
SVD rank cutoff is handed entirely to the data, and the quantum surface gets the
rest. That is fine when the retained directions are well determined, but the
cutoff is purely *relative* (sv_threshold * s_max, with sv_min_abs defaulting to
0), so a direction the data pins to only a degree or so is still treated as
fully constrained and theory gets no vote in it. For water's bond angle that is
exactly what happens.

Two ways to give theory back some say:

  sv_min_abs                a hard absolute floor on the singular value. Since
                            the Jacobian is sigma-weighted, 1/s is the parameter
                            uncertainty along a direction, so the floor is
                            "only trust what the data resolves this well". All
                            or nothing per direction, and it saturates quickly.

  objective_mode: joint     solve (J^T J + alpha_q H + lambda I) dp = J^T r -
  quantum_prior_sigma_ang   alpha_q g, with alpha_q calibrated from the
                            displacement over which the theory surface is
                            trusted. Every direction stays contested, weighted
                            by how well each source knows it. Smooth, and it is
                            a proper MAP estimate rather than a threshold.

    python scripts/tune_quantum_prior.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT, _ROOT / "dev" / "tests", _ROOT / "scripts"):
    sys.path.insert(0, str(_p))

import dev.analytic_water_backend  # noqa: F401,E402
from runner.run_generic import main as run_generic_main  # noqa: E402
from runner.usability import prepare_run_directory, validate_config  # noqa: E402
from theory_vs_experiment_vs_hybrid import R_E, THETA_E, structure  # noqa: E402

_CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"

# Reference points for this test case, from theory_vs_experiment_vs_hybrid.py
THEORY_DR, THEORY_DTHETA = 10.9, -0.808
EXPT_DR, EXPT_DTHETA = 13.2, -0.643


def run(**optimizer_overrides):
    cfg = copy.deepcopy(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    cfg["quantum"]["backend"] = "analytic_water_detuned"
    cfg.setdefault("optimizer", {}).update(optimizer_overrides)
    validate_config(cfg)
    prepare_run_directory(cfg, _CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        return structure(run_generic_main(cfg)["best"]["coords"])


def _combined(r, theta):
    """Each error as a fraction of the theory-alone error, summed. Below 2.0
    means the hybrid beats theory averaged over both parameters."""
    return abs((r - R_E) * 1000) / THEORY_DR + abs(theta - THETA_E) / abs(THEORY_DTHETA)


def _row(label, r, theta):
    print(f"  {label:<24}{r:>10.5f}{(r - R_E) * 1000:>+9.1f}"
          f"{theta:>10.3f}{theta - THETA_E:>+9.3f}{_combined(r, theta):>11.2f}")


def main() -> None:
    print(f"  THEORY alone       dr = {THEORY_DR:+.1f} mA   dtheta = {THEORY_DTHETA:+.3f} deg")
    print(f"  EXPERIMENT alone   dr = {EXPT_DR:+.1f} mA   dtheta = {EXPT_DTHETA:+.3f} deg\n")
    print("  quantum_prior_sigma_ang is the displacement over which the theory surface")
    print("  is trusted -- roughly the geometry error of the electronic-structure method.")
    print("  Smaller means a more confident prior, holding more directions.\n")

    hdr = (f"  {'setting':<24}{'r (A)':>10}{'dr (mA)':>9}"
           f"{'theta':>10}{'dtheta':>9}{'combined':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    _row("split (current default)", *run())
    _row("split, sv_min_abs 1e5", *run(sv_min_abs=1e5))
    for sigma in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        _row(f"joint, prior {sigma:.3f} A",
             *run(objective_mode="joint", quantum_prior_sigma_ang=sigma))

    print("\n  'combined' is each error as a fraction of the theory-alone error, summed;")
    print("  below 2.00 the hybrid beats theory averaged over both parameters.")
    print("\n  The hard sv_min_abs floor fixes the angle but costs the bond length: it")
    print("  removes a direction from the fit outright. The calibrated prior keeps every")
    print("  direction contested and does better on both at once.")
    print("\n  The sweep is not perfectly monotonic -- this is a nonlinear fit with")
    print("  multistart, so neighbouring values can land in different basins. Treat it")
    print("  as a range to scan on your own system, not a curve to interpolate.")


if __name__ == "__main__":
    main()
