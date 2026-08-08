"""Fit water's structure under each correction scheme and score it against r_e.

Answers the practical question the correction chain exists to answer: does
mapping B_0 -> B_e actually move the recovered geometry toward the accepted
equilibrium structure?

Two structural parameters (r_OH, theta) are fitted to A, B, C of H2-16O. The
alpha used to build the B_e targets is recomputed at each new geometry, the same
way the optimizer re-derives it after each Hessian.

    python scripts/geometry_accuracy.py

Uses the analytic Hoy/Mills/Strey + Morse water potential from
dev/tests/reference_molecules.py, so it runs without Psi4 or ORCA. That force
field is not spectroscopic quality -- its bend frequency is ~4% low -- so treat
these numbers as a floor on achievable accuracy, not a ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from backend.spectral.harmonic_alpha import compute_harmonic_alpha  # noqa: E402
from reference_molecules import (  # noqa: E402
    H2O_B0_MHZ,
    H2O_MASSES,
    h2o_coords,
    h2o_hessian,
)

# Accepted equilibrium structure (Csaszar et al.; Benedict et al.)
R_E_ANG = 0.95784
THETA_E_DEG = 104.508

_INERTIA_TO_MHZ = 505379.0084353526


def inertial_defect(constants_mhz) -> float:
    """I_c - I_a - I_b in amu*A^2. Exactly zero for a rigid planar molecule."""
    inertia = _INERTIA_TO_MHZ / np.asarray(constants_mhz, dtype=float)
    return float(inertia[2] - inertia[0] - inertia[1])


def _constants(params):
    return rotational_constants_mhz(
        h2o_coords(params[0], np.radians(params[1])), H2O_MASSES
    )


def _delta_vib(params, scheme):
    """Delta_vib = 0.5 * sum_r alpha_r, evaluated at the current geometry."""
    if scheme == "none":
        return np.zeros(3)
    coords = h2o_coords(params[0], np.radians(params[1]))
    kwargs = {"hessian_fn": h2o_hessian} if scheme == "anharmonic" else {}
    alpha, _, _, _ = compute_harmonic_alpha(
        h2o_hessian(coords), coords, H2O_MASSES, **kwargs
    )
    return 0.5 * np.array([alpha["A"], alpha["B"], alpha["C"]])


def fit(scheme: str, max_outer: int = 8) -> dict:
    params = np.array([0.97, 104.5])
    target = H2O_B0_MHZ.copy()
    for _ in range(max_outer):
        target = H2O_B0_MHZ + _delta_vib(params, scheme)
        sol = least_squares(
            lambda q: (target - _constants(q)) / H2O_B0_MHZ,
            params,
            bounds=([0.85, 95.0], [1.10, 115.0]),
            xtol=1e-14,
            ftol=1e-14,
        )
        if np.allclose(sol.x, params, atol=1e-10):
            params = sol.x
            break
        params = sol.x
    residual = target - _constants(params)
    return {
        "r_ang": float(params[0]),
        "theta_deg": float(params[1]),
        "rms_mhz": float(np.sqrt(np.mean(residual ** 2))),
        "defect": inertial_defect(target),
    }


def main() -> None:
    schemes = [
        ("observed B_0, uncorrected", "none"),
        ("harmonic + Coriolis", "harmonic"),
        ("harmonic + Coriolis + anharmonic", "anharmonic"),
    ]
    print(f"Reference equilibrium structure: r_e = {R_E_ANG:.5f} A, "
          f"theta_e = {THETA_E_DEG:.3f} deg")
    print(f"Inertial defect of observed B_0: {inertial_defect(H2O_B0_MHZ):+.5f} amu*A^2 "
          "(rigid planar = 0)\n")
    header = (f"  {'correction scheme':<34}{'r (A)':>10}{'dr (mA)':>10}"
              f"{'theta':>10}{'dtheta':>9}{'RMS MHz':>10}{'defect':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, scheme in schemes:
        res = fit(scheme)
        print(
            f"  {label:<34}{res['r_ang']:>10.5f}"
            f"{(res['r_ang'] - R_E_ANG) * 1000:>+10.1f}"
            f"{res['theta_deg']:>10.3f}"
            f"{res['theta_deg'] - THETA_E_DEG:>+9.3f}"
            f"{res['rms_mhz']:>10.0f}"
            f"{res['defect']:>+10.5f}"
        )
    print("\n  dr/dtheta are signed errors against the accepted equilibrium structure.")
    print("  RMS is the residual of the fitted rigid geometry against its own targets;")
    print("  a large value means no single rigid structure reproduces them.")


if __name__ == "__main__":
    main()
