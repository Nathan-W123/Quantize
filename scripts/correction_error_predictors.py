"""Can the engine tell, for free, which of its own corrections are wrong?

The rovibrational correction is 5 sigma out on water's B and under 2 sigma on
its A and C. If some quantity the engine already computes separates those
cases, the fit can down-weight the bad components at no extra compute -- which
is the only lever left that does not mean more Hessians.

Truth comes from published equilibrium structures, so only molecules with a
real r_e reference can appear here: water and ozone, 21 constants between them.

    python scripts/correction_error_predictors.py [method=b3lyp] [basis=6-31g(d)]

Result: there is no such quantity. B3LYP/6-31G(d), 21 constants.
---------------------------------------------------------------
Eight components are beyond 2 sigma and thirteen within, and every candidate
predictor puts the two groups on top of each other:

    predictor                  beyond 2 sigma        within
    anharmonic / harmonic       0.64 -  13.73     0.62 -  27.71   overlaps
    |anharmonic| / |total|      1.08 -   1.92     0.95 -   1.90   overlaps
    |correction| / B_obs        4062 -  27787 ppm 8270 - 33574    overlaps
    sigma / |correction|        0.15               0.15           no signal
    cross-mode fraction         0.22 -   0.69     0.34 -   0.60   overlaps

The fourth row is the one to read twice. The reported sigma is a flat 15% of
the correction on all 21 rows, so it carries no information about which
correction to believe at all -- yet the actual relative error ranges from 2% to
81%. "5 sigma" here means only "the correction is more than 30% wrong".

The failures are water's B (81% wrong) and ozone's A (75%), and they have
nothing in common: water's B is anharmonic-dominated with little cancellation,
ozone's A is the opposite. A scaling factor cannot fit both either -- the
overshoot runs 1.02x to 5.38x within the same two molecules.

What the whole set does share is the sign. The computed correction overshoots
on 21 rows out of 21; the truth is always between B_0 and B_0 + delta. Six of
those rows are independent (two molecules x three components), so this is
p = 0.016 under a null of random signs -- suggestive, not established, and it
would need molecules beyond these two to rest anything on.

Three internal consistency checks are blind to the bias, and for one reason.
Reduced chi-square never leaves [0.25, 4] on this set, the planarity constraint
moves ozone's A by 0.65 MHz against the ~315 MHz the angle error needs, and the
inertial defect recovers only 81% of water's zero-point defect. A correction
error confined to one component maps almost exactly onto a change in one
geometric parameter -- ozone's A is the angle -- so the fit absorbs it into the
structure and leaves no residual for any internal test to find.

That is the boundary of what free engineering can reach here: the remaining
correction error is visible only against an external r_e structure or to a
better force field, and neither is free.
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
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    rotational_constants_mhz,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    compute_harmonic_alpha,
    normal_mode_hessian_derivatives,
)
from dev.monofluoro_references import (  # noqa: E402
    OZONE,
    WATER,
    WATER_R_E,
    WATER_THETA_E_DEG,
    _bent_xy,
)
from scripts.monofluoro_benchmark import start_geometry  # noqa: E402

#: Published equilibrium structures, the only ones in the benchmark set.
#: Water: Csaszar et al., J. Chem. Phys. 122, 214305 (2005).
#: Ozone: Tyuterev et al., J. Mol. Spectrosc. 198, 57 (1999) -- r_e = 1.27276 A,
#: angle 116.7542 deg, quoted to +/-0.00015 A and +/-0.0025 deg.
R_E = {
    "water": (WATER, _bent_xy(WATER_R_E, WATER_THETA_E_DEG)),
    "ozone": (OZONE, _bent_xy(1.27276, 116.7542)),
}


def measure(mol, r_e_coords, method: str, basis: str) -> list[dict]:
    """One row per (isotopologue, component): the correction, its error, and
    every quantity the engine could have used to distrust it."""
    backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                      method=method, basis=basis)
    cache: dict = {}

    def hessian_fn(coords_ang):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in cache:
            cache[key] = backend.run_hessian(coords_ang).hessian_bohr
        return cache[key]

    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(start_geometry(mol))
        hess = hessian_fn(coords)
        parent = np.asarray(mol.masses, dtype=float)
        mode_derivs = normal_mode_hessian_derivatives(
            hessian_fn, coords, hess, list(parent))

    rows = []
    for sp in mol.species:
        masses = sp.masses(parent)
        obs = np.asarray(sp.abc_mhz, dtype=float)
        truth = rotational_constants_mhz(r_e_coords, masses) - obs
        with contextlib.redirect_stdout(io.StringIO()):
            alpha, _, sigma, info = compute_harmonic_alpha(
                hess, coords, masses, mode_derivs=mode_derivs)
        for i, comp in enumerate("ABC"):
            if comp not in alpha:
                continue
            harm = 0.5 * float(info["alpha_centrifugal_mhz"][comp])
            cori = 0.5 * float(info["alpha_coriolis_mhz"][comp])
            anh = 0.5 * float(info["alpha_anharmonic_mhz"][comp])
            calc = 0.5 * float(alpha[comp])
            sig = 0.5 * float(sigma[comp])
            rows.append({
                "mol": mol.key, "iso": sp.label, "comp": comp,
                "harm": harm, "cori": cori, "anh": anh,
                "calc": calc, "sigma": sig,
                "true": float(truth[i]), "err": calc - float(truth[i]),
                "ratio": float(info["anharmonic_ratio"][comp]),
                "anh_err": float((info.get("alpha_anharmonic_err_mhz") or {})
                                 .get(comp, float("nan"))) * 0.5,
                "obs": float(obs[i]),
            })
    return rows


def main() -> None:
    method, basis = "b3lyp", "6-31g(d)"
    for tok in sys.argv[1:]:
        if tok.startswith("method="):
            method = tok.split("=", 1)[1]
        elif tok.startswith("basis="):
            basis = tok.split("=", 1)[1]

    rows = []
    for key, (mol, r_e_coords) in R_E.items():
        print(f"  [{key}] {method}/{basis} ...", flush=True)
        rows += measure(mol, r_e_coords, method, basis)

    print()
    print(f"  {'mol':6s} {'iso':24s} {'c':2s} "
          f"{'harm':>9s} {'cori':>9s} {'anh':>10s} {'calc':>10s} "
          f"{'true':>10s} {'err':>10s} {'sigma':>9s} {'n_sig':>6s} "
          f"{'anh/hrm':>8s} {'|err|/anh':>10s}")
    for r in rows:
        n_sig = abs(r["err"]) / r["sigma"] if r["sigma"] else float("nan")
        per_anh = abs(r["err"]) / abs(r["anh"]) if r["anh"] else float("nan")
        print(f"  {r['mol']:6s} {r['iso'][:24]:24s} {r['comp']:2s} "
              f"{r['harm']:9.1f} {r['cori']:9.1f} {r['anh']:10.1f} "
              f"{r['calc']:10.1f} {r['true']:10.1f} {r['err']:+10.1f} "
              f"{r['sigma']:9.1f} {n_sig:6.1f} {r['ratio']:8.2f} {per_anh:10.2f}")

    bad = [r for r in rows if abs(r["err"]) / r["sigma"] > 2.0]
    good = [r for r in rows if abs(r["err"]) / r["sigma"] <= 2.0]
    print()
    print(f"  {len(bad)} components beyond 2 sigma, {len(good)} within.")
    for label, fn in (
        ("anharmonic / harmonic", lambda r: r["ratio"]),
        ("|anh| / |calc|", lambda r: abs(r["anh"]) / abs(r["calc"]) if r["calc"] else np.nan),
        ("|calc| / B_obs (ppm)", lambda r: 1e6 * abs(r["calc"]) / r["obs"]),
        ("sigma / |calc|", lambda r: r["sigma"] / abs(r["calc"]) if r["calc"] else np.nan),
    ):
        b = np.array([fn(r) for r in bad], dtype=float)
        g = np.array([fn(r) for r in good], dtype=float)
        print(f"  {label:24s} beyond 2sig: {np.nanmin(b):8.2f} - {np.nanmax(b):8.2f}"
              f"    within: {np.nanmin(g):8.2f} - {np.nanmax(g):8.2f}"
              f"    {'SEPARATES' if np.nanmin(b) > np.nanmax(g) or np.nanmax(b) < np.nanmin(g) else 'overlaps'}")


if __name__ == "__main__":
    main()
