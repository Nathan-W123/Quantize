"""Does the correction need to follow the geometry the optimiser moves to?

The engine computes alpha at the quantum minimum x_QM, folds it into
B_e = B_0 + 1/2 sum alpha, and then optimises. From that point on the
correction is a constant. But the joint objective exists precisely to move the
geometry OFF x_QM -- the spectra pull it towards the true equilibrium -- so the
frozen correction ends up applied at a geometry it was never expanded about.

This script measures whether that matters, in three steps and without building
anything:

  1. Fit, and measure the displacement u = x_joint - x_QM.
  2. Measure d(alpha)/ds along u, at FIXED force field, by displacing +-|u|/2.
     A directional derivative rather than a full gradient: the optimiser only
     travels one direction, so two extra alpha evaluations suffice whatever the
     molecule's size. That is also exactly what an implementation would need.
  3. Predict alpha(x_joint) = alpha(x_QM) + (d alpha/ds)|u| and ask whether it
     is closer to the truth than alpha(x_QM) is.

Step 2 evaluates a Hessian off the stationary point, which VPT2 strictly does
not allow. Over the few mA involved the residual gradient's contamination of
the normal-mode analysis is second order, which is good enough to size an
effect -- it is not good enough to ship as a production alpha, and the point of
the exercise is to decide whether shipping one is worth the work.

Truth is the reference structure's correction, B_e(x_ref) - B_0. Note what
x_ref is per molecule: water's is a genuine r_e, ozone's is an r_s substitution
structure, which is close to r_e but not equal to it, so ozone's "error"
columns carry that systematic on top of everything else.

Run:  python scripts/correction_selfconsistency.py [water|ozone] [method] [basis]
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
    compute_harmonic_alpha,
    normal_mode_hessian_derivatives,
)
from dev.monofluoro_references import OZONE, WATER  # noqa: E402
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

_MOLS = {"water": WATER, "ozone": OZONE}


def internal_coords(coords):
    """Bond length (A) and apex angle (deg) of a bent XY2, atom 0 central."""
    coords = np.asarray(coords, dtype=float)
    v1, v2 = coords[1] - coords[0], coords[2] - coords[0]
    r = float(0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2)))
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return r, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main():
    argv = sys.argv[1:]
    name = (argv[0] if argv else "water").lower()
    method = argv[1] if len(argv) > 1 else "b3lyp"
    basis = argv[2] if len(argv) > 2 else "cc-pvtz"
    if name not in _MOLS:
        raise SystemExit(f"unknown molecule '{name}'; choose from {sorted(_MOLS)}")
    mol = _MOLS[name]

    sys.argv = ["bench", f"method={method}", f"basis={basis}"]
    import scripts.accuracy_upgrades_benchmark as bench

    masses = np.asarray(mol.masses, dtype=float)
    backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=method, basis=basis)
    cache: dict = {}

    def hessian_fn(coords, _b=backend, _c=cache):
        key = np.asarray(coords, dtype=float).round(9).tobytes()
        if key not in _c:
            _c[key] = _b.run_hessian(coords).hessian_bohr
        return _c[key]

    def alpha_at(coords):
        """The parent isotopologue's correction 1/2 sum alpha, expanded at coords."""
        with contextlib.redirect_stdout(io.StringIO()):
            hess = hessian_fn(coords)
            derivs = normal_mode_hessian_derivatives(hessian_fn, coords, hess, list(masses))
            alpha, _, _, _ = compute_harmonic_alpha(hess, coords, masses, mode_derivs=derivs)
        return np.array([0.5 * alpha[c] for c in "ABC"])

    obs = np.asarray(mol.species[0].abc_mhz, dtype=float)
    x_ref = np.asarray(mol.geometry, dtype=float)
    truth = rotational_constants_mhz(x_ref, masses) - obs

    # ---- step 1: the displacement the joint objective actually performs -----
    with contextlib.redirect_stdout(io.StringIO()):
        x_qm = backend.optimise(start_geometry(mol))
    isos = build_isotopologues(mol, None)
    with contextlib.redirect_stdout(io.StringIO()):
        ctbl, _info = build_correction_table_from_hessian(
            hessian_fn(x_qm), x_qm, isos, hessian_fn=hessian_fn,
            cubic_scheme="normal_mode", lam_freq_cm=0.0,
            freq_scale=bench.FREQ_SCALE, harmonic_scheme="watson")
    sigma_x = bench._SIGMA_X_BY_LEVEL.get((method.lower(), basis.lower()), 0.020)
    with contextlib.redirect_stdout(io.StringIO()):
        x_joint = np.asarray(bench.hybrid_fit(mol, isos, x_qm, ctbl, sigma_x), dtype=float)

    step = x_joint - x_qm
    dist = float(np.linalg.norm(step))
    r_ref, th_ref = internal_coords(x_ref)
    r_qm, th_qm = internal_coords(x_qm)
    r_j, th_j = internal_coords(x_joint)

    print(f"  {mol.name}   {method}/{basis}   sigma_x = {sigma_x:.4f} A"
          f"   {len(isos)} isotopologues\n")
    print(f"  {'':16s}{'r (A)':>10s}{'angle (deg)':>13s}{'dr vs ref mA':>15s}{'dang vs ref':>13s}")
    print(f"  {'QM minimum':16s}{r_qm:10.5f}{th_qm:13.3f}"
          f"{1000 * (r_qm - r_ref):15.2f}{th_qm - th_ref:13.3f}")
    print(f"  {'joint optimum':16s}{r_j:10.5f}{th_j:13.3f}"
          f"{1000 * (r_j - r_ref):15.2f}{th_j - th_ref:13.3f}")
    print(f"  {'reference':16s}{r_ref:10.5f}{th_ref:13.3f}")
    print(f"\n  displacement joint - QM:  dr = {1000 * (r_j - r_qm):+.2f} mA"
          f"   dangle = {th_j - th_qm:+.3f} deg"
          f"   |u| = {dist:.5f} A (cartesian)")

    if dist < 1e-6:
        print("\n  the optimiser did not move: nothing to correct.")
        return

    # ---- step 2: d(alpha)/ds along that direction, force field held fixed ---
    unit = step / dist
    half = 0.5 * dist
    d_alpha = (alpha_at(x_qm + half * unit) - alpha_at(x_qm - half * unit)) / dist

    a_qm = alpha_at(x_qm)
    predicted = a_qm + d_alpha * dist

    print(f"\n  d(correction)/ds along u, at fixed {method}/{basis}"
          f"   (+-{1000 * half:.2f} mA cartesian):")
    print(f"  {'':4s}{'MHz per mA':>14s}{'drift over |u|':>17s}")
    for i, comp in enumerate("ABC"):
        print(f"  {comp:4s}{d_alpha[i] / 1000.0:+14.1f}{d_alpha[i] * dist:+17.1f}")

    # ---- step 3: does following the geometry help? -------------------------
    err_now = a_qm - truth
    err_fixed = predicted - truth
    print(f"\n  {'':4s}{'corr at x_QM':>14s}{'corr at x_joint':>17s}{'truth':>11s}"
          f"{'err now':>10s}{'err fixed':>11s}{'change':>9s}")
    for i, comp in enumerate("ABC"):
        change = (100.0 * (abs(err_fixed[i]) - abs(err_now[i])) / abs(err_now[i])
                  if abs(err_now[i]) > 1e-9 else float("nan"))
        print(f"  {comp:4s}{a_qm[i]:+14.1f}{predicted[i]:+17.1f}{truth[i]:+11.1f}"
              f"{err_now[i]:+10.1f}{err_fixed[i]:+11.1f}{change:+8.0f}%")

    rms_now = float(np.sqrt(np.mean(err_now ** 2)))
    rms_fixed = float(np.sqrt(np.mean(err_fixed ** 2)))
    print(f"\n  rms over A,B,C:  {rms_now:.1f} -> {rms_fixed:.1f} MHz"
          f"   ({100 * (rms_fixed - rms_now) / rms_now:+.0f}%)")
    if name == "ozone":
        print("  (ozone's reference is an r_s substitution structure, not r_e:"
              " the error columns carry that systematic too)")


if __name__ == "__main__":
    main()
