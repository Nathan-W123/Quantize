"""Are the engine's uncertainties honest?

For a fluorinated molecule with no accepted structure -- the case this project
targets -- there is no error to report, only an uncertainty. That makes the
uncertainty the deliverable, and an uncertainty nobody has checked is worse than
none: it invites a reader to trust a number by exactly the amount it is wrong.

So this measures the only thing that can be measured: on molecules that *do*
have published structures, how often does the quoted interval actually contain
the true value? A well calibrated 1-sigma interval covers about 68% of the time.
Coverage far below that means the engine is overconfident and its intervals must
not be quoted on unknown molecules until the cause is fixed.

    python scripts/uncertainty_calibration.py
    python scripts/uncertainty_calibration.py method=b3lyp "basis=6-31g(d)"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from dev.monofluoro_references import MOLECULES_SET2  # noqa: E402
from backend.quantum import _detect_angles, _detect_bonds  # noqa: E402
from scripts.monofluoro_benchmark import (  # noqa: E402
    build_isotopologues,
    start_geometry,
)


def measure(coords, elems, bonds, angles):
    """Bond lengths and angles keyed the way geometry_uncertainty keys them.

    The reference module names coordinates chemically ("C=O"); the engine
    detects them from the geometry and names them by atom index ("C1-O2").
    Deriving both from the same index pairs is what lets an error be paired
    with the uncertainty that belongs to it.
    """
    out = {}
    for i, j in bonds:
        out[f"{elems[i]}{i + 1}-{elems[j]}{j + 1}"] = (
            float(np.linalg.norm(coords[i] - coords[j])), True)
    for i, j, k in angles:
        v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
        cosa = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        out[f"{elems[i]}{i + 1}-{elems[j]}{j + 1}-{elems[k]}{k + 1}"] = (
            float(np.degrees(np.arccos(np.clip(cosa, -1.0, 1.0)))), False)
    return out

METHOD, BASIS = "b3lyp", "6-31g(d)"
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

OUT = _ROOT / "output" / "uncertainty_calibration.json"


def run_one(mol, limit):
    b = get_backend("pyscf_hf")(elems=list(mol.elems), method=METHOD, basis=BASIS)
    gt = b.optimise(start_geometry(mol))
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(gt, dtype=float),
        isotopologues=build_isotopologues(mol, limit),
        quantum_backend="pyscf_hf", orca_method=METHOD, orca_basis=BASIS,
        coordinate_mode="cartesian", use_autoconfig=False, max_iter=40,
        hess_recalc_every=10,
        # The systematic now has its own channel, so the weighting sigma is
        # measurement precision and the blanket relative floor would only put
        # the model error back into the weights.
        sigma_floor_rel=0.0, chi2_rescale=True)
    with contextlib.redirect_stdout(io.StringIO()):
        res = opt.run()
    coords = res["coords"] if isinstance(res, dict) and "coords" in res else res
    unc = opt.geometry_uncertainty()
    return coords, unc, opt.point_group, opt.data_support(), opt.reduced_chi_square()


def main() -> None:
    print(f"  {METHOD.upper()}/{BASIS}. Do the quoted uncertainties cover the truth?\n")
    rows, records = [], []
    for mol in MOLECULES_SET2:
        elems = list(mol.elems)
        for label, limit in (("parent only", 1), ("all species", None)):
            coords, unc, pg, support, chi2 = run_one(mol, limit)
            bonds = _detect_bonds(coords, elems)
            angles = _detect_angles(bonds)
            got = measure(np.asarray(coords, float), elems, bonds, angles)
            ref = measure(np.asarray(mol.geometry, float), elems, bonds, angles)
            print(f"  {mol.name} / {label}   (point group {pg or '-'}, "
                  f"chi2/nu = {chi2:.2f})")
            print(f"    {'coordinate':<14}{'accepted':>10}{'fitted':>10}"
                  f"{'error':>11}{'quoted 1σ':>12}{'':>3}")
            for name, (value, is_bond) in got.items():
                if name not in ref or name not in unc:
                    continue
                scale = 1000.0 if is_bond else 1.0
                unit = "mÅ" if is_bond else "°"
                err = (value - ref[name][0]) * scale
                sig = unc[name] * scale
                covered = abs(err) <= sig
                rows.append((covered, abs(err), sig, is_bond))
                records.append({"molecule": mol.name, "level": label,
                                "coordinate": name, "error": err, "sigma": sig,
                                "covered": bool(covered), "is_bond": bool(is_bond),
                                "data_support": float(support.get(name, float("nan"))),
                                "chi2_nu": float(chi2)})
                print(f"    {name:<14}{ref[name][0]:>10.4f}{value:>10.4f}"
                      f"{err:>+9.2f} {unit:<2}{sig:>10.2f} {unit:<2}"
                      f"{'  ok' if covered else '  MISS'}"
                      f"{'' if support.get(name, 1.0) > 0.5 else '  [theory-determined]'}")
            print()

    n = len(rows)
    cov = sum(1 for c, *_ in rows if c)
    print(f"  {'=' * 70}")
    print(f"  Coverage of the quoted 1σ interval: {cov}/{n} = {100 * cov / n:.0f}%"
          f"   (well calibrated ≈ 68%)")
    ratios = [e / s for _, e, s, _ in rows if s > 0]
    print(f"  Median |error| / σ: {np.median(ratios):.2f}   (well calibrated ≈ 0.67)")
    for kind, flag in (("bonds", True), ("angles", False)):
        sub = [(c, e, s) for c, e, s, b in rows if b is flag]
        if sub:
            k = sum(1 for c, *_ in sub if c)
            print(f"    {kind:<8}{k}/{len(sub)} covered, "
                  f"median |error|/σ = {np.median([e / s for _, e, s in sub if s > 0]):.2f}")
    print(f"  {'=' * 70}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"method": METHOD, "basis": BASIS,
                               "coverage": cov / n, "n": n,
                               "records": records}, indent=2), encoding="utf-8")
    print(f"\n  written to {OUT}")


if __name__ == "__main__":
    main()
