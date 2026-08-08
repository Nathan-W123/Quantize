"""Water: theory alone vs experiment alone vs the hybrid, against the accepted r_e.

The three-way comparison the package exists to justify. Each leg answers a
different question, and each has a characteristic failure:

  THEORY ALONE      minimise the quantum surface, no spectroscopy. Gives an
                    equilibrium structure, but carries the method's systematic
                    error -- here a deliberate ~11 mA, hybrid-DFT sized.

  EXPERIMENT ALONE  fit the observed rotational constants with no quantum
                    prior. Exact where the data reaches, but B_0 is a
                    vibrationally averaged quantity, so what comes out is an
                    r_0-like effective structure, not r_e. It also cannot
                    resolve directions the data does not see.

  HYBRID            map B_0 to B_e with the correction chain, then let the SVD
                    split give spectroscopy the directions it constrains and the
                    quantum surface the rest.

Everything is run through the real pipeline -- the same path `python -m cli run`
takes -- via the analytic backends in dev/analytic_water_backend.py, so it works
with no Psi4 or ORCA installed.

    python scripts/theory_vs_experiment_vs_hybrid.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from dev.analytic_water_backend import (  # noqa: E402
    R_DETUNED, THETA_DETUNED, _detuned_energy_hartree,
)
from reference_molecules import h2o_coords  # noqa: E402
from runner.run_generic import main as run_generic_main  # noqa: E402
from runner.usability import prepare_run_directory, validate_config  # noqa: E402

# Accepted equilibrium structure (Csaszar et al.; Benedict et al.)
R_E, THETA_E = 0.95784, 104.508
_CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


def structure(coords):
    x = np.asarray(coords, dtype=float)
    r1 = np.linalg.norm(x[1] - x[0])
    r2 = np.linalg.norm(x[2] - x[0])
    u1 = (x[1] - x[0]) / r1
    u2 = (x[2] - x[0]) / r2
    theta = np.degrees(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
    return 0.5 * (r1 + r2), theta


def _pipeline(backend, harmonic, anharmonic):
    cfg = copy.deepcopy(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    cfg["quantum"]["backend"] = backend
    cfg["rovibrational_corrections"]["harmonic_from_hessian"] = harmonic
    cfg["rovibrational_corrections"]["anharmonic_from_hessian"] = anharmonic
    validate_config(cfg)
    prepare_run_directory(cfg, _CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_generic_main(cfg)
    return structure(result["best"]["coords"]), result


def theory_alone():
    """Minimise the quantum surface. No experimental input of any kind."""
    sol = minimize(
        lambda p: _detuned_energy_hartree(h2o_coords(p[0], np.radians(p[1]))),
        np.array([0.98, 106.0]), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-18, "maxiter": 20000},
    )
    return structure(h2o_coords(sol.x[0], np.radians(sol.x[1])))


def _row(label, r, theta, note=""):
    print(f"  {label:<34}{r:>9.5f}{(r - R_E) * 1000:>+9.1f}"
          f"{theta:>10.3f}{theta - THETA_E:>+9.3f}   {note}")


def main() -> None:
    print("  Water, single isotopologue (H2-16O), all three legs through the real pipeline.")
    print(f"  Quantum method's own minimum is displaced to r = {R_DETUNED} A, "
          f"theta = {THETA_DETUNED} deg,")
    print("  standing in for the systematic error a real electronic-structure method carries.\n")

    hdr = (f"  {'':<34}{'r (A)':>9}{'dr (mA)':>9}{'theta':>10}{'dtheta':>9}")
    print(hdr)
    print("  " + "-" * 96)

    _row("EXPERIMENTAL reference (r_e)", R_E, THETA_E, "accepted equilibrium structure")
    print()

    r, t = theory_alone()
    _row("THEORY alone", r, t, "quantum surface minimised, no spectroscopy")

    (r, t), _ = _pipeline("none", False, False)
    _row("EXPERIMENT alone", r, t, "fits observed B_0, no quantum prior")

    (r, t), res = _pipeline("analytic_water_detuned", True, True)
    score = float((res.get("score") or {}).get("score", float("nan")))
    _row("HYBRID (theory + experiment)", r, t, f"corrected B_e + null-space prior")

    print("\n  Experiment alone converges on an r_0-like effective structure: B_0 is a")
    print("  vibrationally averaged quantity, so fitting it directly returns the average,")
    print("  not the equilibrium geometry, however precisely the constants are measured.")
    print("  Theory alone returns an equilibrium structure but carries its method error.")
    print("  The hybrid corrects B_0 toward B_e and lets the quantum surface hold the")
    print("  directions the data cannot see.")
    print(f"\n  Run confidence for the hybrid leg: score {score:.1f}/100. One isotopologue")
    print("  cannot determine water's structure; the angle in particular stays")
    print("  prior-dominated, and the run reports that rather than hiding it.")


if __name__ == "__main__":
    main()
