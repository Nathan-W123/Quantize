"""Fluorobenzene: hybrid and theory against the published experimental structure.

Ground truth is the microwave substitution structure of

    Bak, Christensen, Hansen-Nygaard & Tannenbaum,
    J. Chem. Phys. 26, 134 (1957)   [Cartesians via NIST CCCBDB casno 462066]

together with the ground-state constants from the same source,
A = 0.18892, B = 0.08575, C = 0.05897 cm-1. The two are mutually consistent: the
published geometry reproduces the published constants to 0.19-0.23%, the size of
the r_s versus r_0 difference, and the constants give an inertial defect of
+0.046 amu*A^2 as a planar molecule requires.

This is deliberately *not* configs/fluorobenzene.yaml, whose constants give an
inertial defect near +5.7 amu*A^2 and cannot come from any real structure.

Only the parent species is used -- three observables against thirty internal
degrees of freedom. That is the regime the SVD split exists for: spectroscopy
fixes what it can see, the quantum surface holds the rest. Theory is a genuine
RHF calculation through PySCF.

    python scripts/fluorobenzene_vs_published.py [basis]
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from dev.reference_structures import (  # noqa: E402
    FLUOROBENZENE_B0_MHZ,
    FLUOROBENZENE_ELEMS,
    FLUOROBENZENE_GEOM,
    FLUOROBENZENE_MASSES,
    FLUOROBENZENE_SOURCE,
    internal_coordinates,
)

_K = 505379.0084353526


def _defect(abc):
    inertia = _K / np.asarray(abc, dtype=float)
    return float(inertia[2] - inertia[0] - inertia[1])


def run_hybrid(backend_name, basis, start, **kwargs):
    iso = [{
        "name": "C6H5F",
        "masses": FLUOROBENZENE_MASSES.tolist(),
        "obs_constants": FLUOROBENZENE_B0_MHZ.tolist(),
        # Published uncertainties are far tighter, but the r_s/r_0 gap is the
        # real error floor on fitting B_0 as if it were B_e, so weight to that.
        "sigma_constants": [10.0, 6.0, 4.0],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    }]
    opt = MolecularOptimizer(
        elems=list(FLUOROBENZENE_ELEMS),
        coords=np.asarray(start, dtype=float),
        isotopologues=iso,
        quantum_backend=backend_name,
        orca_method="hf",
        orca_basis=basis,
        coordinate_mode="cartesian",
        spectral_only=(backend_name == "none"),
        use_autoconfig=False,
        **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def main() -> None:
    basis = sys.argv[1] if len(sys.argv) > 1 else "6-31g"
    truth = FLUOROBENZENE_GEOM
    ref = internal_coordinates(truth)

    print(f"  Ground truth: {FLUOROBENZENE_SOURCE}")
    abc_geom = rotational_constants_mhz(truth, FLUOROBENZENE_MASSES)
    print(f"  Published B0 (MHz): {np.round(FLUOROBENZENE_B0_MHZ, 2)}"
          f"   inertial defect {_defect(FLUOROBENZENE_B0_MHZ):+.3f} amu*A^2")
    print(f"  From the structure: {np.round(abc_geom, 2)}"
          f"   agreement {np.max(np.abs(abc_geom / FLUOROBENZENE_B0_MHZ - 1)) * 100:.2f}%")
    print(f"\n  Theory: RHF/{basis} via PySCF. Experimental input: the parent A, B, C only")
    print("  -- 3 observables against 30 internal degrees of freedom.\n")

    # Start displaced from the truth so nothing is handed the answer.
    rng = np.random.default_rng(11)
    start = truth + rng.normal(0.0, 0.03, size=truth.shape)
    start[:, 0] = 0.0                                   # keep the molecule planar

    t0 = time.time()
    backend = get_backend("pyscf_hf")(
        elems=list(FLUOROBENZENE_ELEMS), method="hf", basis=basis)
    theory = backend.optimise(start)
    t_theory = time.time() - t0

    # Joint objective with a calibrated prior. With three observables against
    # thirty degrees of freedom the split objective drives the residual to zero
    # in the directions it owns and distorts the C-H angles doing it; leaving
    # every direction contested costs almost nothing in bond length and recovers
    # most of that. See scripts/tune_quantum_prior.py.
    t0 = time.time()
    hybrid = run_hybrid("pyscf_hf", basis, start, max_iter=30, hess_recalc_every=10,
                        objective_mode="joint", quantum_prior_sigma_ang=0.005)
    t_hybrid = time.time() - t0

    got_t = internal_coordinates(theory)
    got_h = internal_coordinates(hybrid)

    print(f"  {'parameter':<24}{'published':>11}{'theory':>10}{'err':>9}"
          f"{'hybrid':>10}{'err':>9}")
    print("  " + "-" * 73)
    bond_t, bond_h, ang_t, ang_h = [], [], [], []
    bond_tags = ("C-F", "ipso-ortho", "ortho-meta", "meta-para", "C-H")
    for name in ref:
        is_bond = any(tag in name for tag in bond_tags)
        r, a, b = ref[name], got_t[name], got_h[name]
        if is_bond:
            bond_t.append((a - r) * 1000)
            bond_h.append((b - r) * 1000)
            print(f"  {name:<24}{r:>11.4f}{a:>10.4f}{(a - r) * 1000:>+8.1f}m"
                  f"{b:>10.4f}{(b - r) * 1000:>+8.1f}m")
        else:
            ang_t.append(a - r)
            ang_h.append(b - r)
            print(f"  {name:<24}{r:>11.2f}{a:>10.2f}{a - r:>+9.2f}"
                  f"{b:>10.2f}{b - r:>+9.2f}")

    print("  " + "-" * 73)
    print(f"  {'RMS bond error (mA)':<24}{'':>11}{'':>10}"
          f"{np.sqrt(np.mean(np.square(bond_t))):>9.1f}{'':>10}"
          f"{np.sqrt(np.mean(np.square(bond_h))):>9.1f}")
    print(f"  {'RMS angle error (deg)':<24}{'':>11}{'':>10}"
          f"{np.sqrt(np.mean(np.square(ang_t))):>9.2f}{'':>10}"
          f"{np.sqrt(np.mean(np.square(ang_h))):>9.2f}")
    print(f"\n  bond errors are in milliangstrom (m), angles in degrees.")
    print(f"  theory {t_theory:.0f}s, hybrid {t_hybrid:.0f}s")

    for label, geom in (("theory", theory), ("hybrid", hybrid)):
        abc = rotational_constants_mhz(geom, FLUOROBENZENE_MASSES)
        rel = (abc / FLUOROBENZENE_B0_MHZ - 1) * 100
        print(f"  {label:<7} reproduces the observed constants to "
              f"{np.max(np.abs(rel)):.2f}%  {np.round(abc, 1)}")


if __name__ == "__main__":
    main()
