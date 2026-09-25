"""Validate the rotational g-tensor against every measured value we could verify.

Until now one number tested this: g_perp ~ -0.028 for OCS, in a code comment
without a citation. One datum cannot separate a correct implementation from a
wrong one that lands near a small negative number, and it tests neither the
sign, nor the anisotropy, nor the axis labelling, nor -- the failure that was
actually present -- the mass dependence.

dev/g_tensor_references.py now holds 59 measured components across 20 molecules
and isotopologues, spanning four orders of magnitude and both signs. This script
computes each one and reports the comparison.

Geometry is optimised once at a reference level and the g-tensor is then
evaluated at that one geometry across a basis ladder, which separates the two
error sources: geometry error moves every basis together, basis error does not.

    python scripts/g_tensor_validation.py [geom=b3lyp/6-31g(d)]
                                          [bases=6-31g,6-31g(d),cc-pvdz]
                                          [method=hf] [only=water,ozone]
"""

from __future__ import annotations

import contextlib
import io
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
from backend.spectral.electronic_g import rotational_g_tensor  # noqa: E402
from dev.g_tensor_references import primary_tensor  # noqa: E402

M_H, M_D = 1.00782503207, 2.01410177812
M_C, M_C13 = 12.0, 13.00335483507
M_N, M_O = 14.0030740048, 15.9949146196
M_F, M_LI = 18.99840322, 7.0160034366
M_SI, M_S, M_S34 = 27.9769265325, 31.97207117, 33.967867
M_CL = 34.96885268


def _bent(r1, r2, deg):
    """XY2-type: central atom first, then the two ligands in the xy plane."""
    half = np.radians(deg) / 2.0
    return np.array([[0.0, 0.0, 0.0],
                     [r1 * np.sin(half), r1 * np.cos(half), 0.0],
                     [-r2 * np.sin(half), r2 * np.cos(half), 0.0]])


def _linear(*bonds):
    z = [0.0]
    for b in bonds:
        z.append(z[-1] + b)
    return np.array([[0.0, 0.0, zi] for zi in z])


def _planar_x2co(cx, ch, deg_hcx):
    """H2C=X: C at origin, X along +y, two H below. Order C, X, H, H."""
    a = np.radians(deg_hcx)
    return np.array([[0.0, 0.0, 0.0],
                     [0.0, cx, 0.0],
                     [ch * np.sin(a), ch * np.cos(a), 0.0],
                     [-ch * np.sin(a), ch * np.cos(a), 0.0]])


def _methyl_x(cx, ch, deg_xch):
    """CH3X along z: C at origin, X at +z, three H splayed below."""
    a = np.radians(deg_xch)
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, cx]]
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        out.append([ch * np.sin(a) * np.cos(phi),
                    ch * np.sin(a) * np.sin(phi),
                    ch * np.cos(a)])
    return np.array(out)


#: Starting geometries only -- every one is optimised before the g-tensor is
#: taken, so these are seeds for the optimiser and not structural claims.
#: (reference name, elems, start geometry, masses)
CASES = [
    ("water", "Water-d2, D2O", ["O", "H", "H"],
     _bent(0.958, 0.958, 104.5), [M_O, M_D, M_D]),
    ("ozone", "Ozone, O3", ["O", "O", "O"],
     _bent(1.272, 1.272, 116.8), [M_O, M_O, M_O]),
    ("so2", "Sulfur dioxide, SO2", ["S", "O", "O"],
     _bent(1.431, 1.431, 119.3), [M_S, M_O, M_O]),
    ("of2", "Oxygen difluoride, OF2", ["O", "F", "F"],
     _bent(1.405, 1.405, 103.1), [M_O, M_F, M_F]),
    ("h2co", "Formaldehyde, H2CO", ["C", "O", "H", "H"],
     _planar_x2co(1.205, 1.111, 121.9), [M_C, M_O, M_H, M_H]),
    ("d2co", "Formaldehyde-d2, D2CO", ["C", "O", "H", "H"],
     _planar_x2co(1.205, 1.111, 121.9), [M_C, M_O, M_D, M_D]),
    ("h2cs", "Thioformaldehyde, H2CS", ["C", "S", "H", "H"],
     _planar_x2co(1.611, 1.093, 121.9), [M_C, M_S, M_H, M_H]),
    ("ocs32", "Carbonyl sulfide, OCS", ["S", "C", "O"],
     _linear(1.5614, 1.1600), [M_S, M_C, M_O]),
    ("ocs34", "Carbonyl sulfide, OCS", ["S", "C", "O"],
     _linear(1.5614, 1.1600), [M_S34, M_C, M_O]),
    ("co", "Carbon monoxide, CO", ["C", "O"],
     _linear(1.128), [M_C, M_O]),
    ("cs", "Carbon monosulfide, CS", ["C", "S"],
     _linear(1.535), [M_C, M_S]),
    ("sio", "Silicon monoxide, SiO", ["Si", "O"],
     _linear(1.510), [M_SI, M_O]),
    ("sis", "Silicon monosulfide, SiS", ["Si", "S"],
     _linear(1.929), [M_SI, M_S]),
    ("lih", "Lithium hydride, LiH", ["Li", "H"],
     _linear(1.595), [M_LI, M_H]),
    ("lif", "Lithium fluoride, LiF", ["Li", "F"],
     _linear(1.564), [M_LI, M_F]),
    ("ch3f", "Methyl fluoride, CH3F", ["C", "F", "H", "H", "H"],
     _methyl_x(1.383, 1.087, 108.5), [M_C, M_F, M_H, M_H, M_H]),
]


def _is_linear(coords):
    c = np.asarray(coords, dtype=float)
    c = c - c.mean(axis=0)
    return float(np.linalg.svd(c, compute_uv=False)[1]) < 1e-3


def compare(case, geom_method, geom_basis, bases):
    key, refname, elems, start, masses = case
    masses = np.asarray(masses, dtype=float)
    backend = get_backend("pyscf_hf")(elems=list(elems), method=geom_method,
                                      basis=geom_basis)
    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(np.asarray(start, dtype=float))
    refs = primary_tensor(refname)
    linear = _is_linear(coords)

    rows = []
    for basis in bases:
        with contextlib.redirect_stdout(io.StringIO()):
            g = rotational_g_tensor(list(elems), coords, masses,
                                    method="hf", basis=basis)
        for comp, ref in refs.items():
            if comp == "perp":
                # A linear molecule has no g along the axis; a symmetric top's
                # perpendicular pair is B == C. Either way B is the one to use.
                got = g["B"]
            elif comp == "par":
                got = g["A"]
            else:
                got = g[comp]
            if linear and comp == "A":
                continue
            rows.append({
                "case": key, "molecule": refname, "basis": basis,
                "component": comp, "calc": float(got),
                "exp": float(ref.value), "exp_sigma": float(ref.sigma),
                "printed": ref.printed,
            })
    return rows, coords


def main() -> None:
    geom = "b3lyp/6-31g(d)"
    bases = ["6-31g", "6-31g(d)", "cc-pvdz"]
    only: list[str] = []
    for tok in sys.argv[1:]:
        if tok.startswith("geom="):
            geom = tok.split("=", 1)[1]
        elif tok.startswith("bases="):
            bases = tok.split("=", 1)[1].split(",")
        elif tok.startswith("only="):
            only = tok.split("=", 1)[1].split(",")
    geom_method, geom_basis = geom.split("/", 1)

    print(f"  geometry: {geom};  g at HF/{{{', '.join(bases)}}}")
    print(f"  {len(CASES)} species against dev/g_tensor_references.py\n")

    out, rows = [], []
    for case in CASES:
        if only and case[0] not in only:
            continue
        t0 = time.time()
        try:
            got, coords = compare(case, geom_method, geom_basis, bases)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {case[0]:8s} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        rows += got
        for r in got:
            err = r["calc"] - r["exp"]
            rel = 100.0 * err / abs(r["exp"]) if r["exp"] else float("nan")
            flag = "" if np.sign(r["calc"]) == np.sign(r["exp"]) else "  SIGN"
            print(f"  {r['case']:8s} {r['basis']:9s} g_{r['component']:4s} "
                  f"calc {r['calc']:+9.4f}  exp {r['exp']:+9.4f} "
                  f"+-{r['exp_sigma']:.4f}   err {err:+8.4f} ({rel:+7.1f}%)"
                  f"{flag}", flush=True)
        out.append(case[0])
        print(f"    ({time.time() - t0:.0f} s)", flush=True)

    if not rows:
        return
    print("\n  summary by basis (signed mean and RMS of the relative error):")
    for basis in bases:
        sel = [r for r in rows if r["basis"] == basis and r["exp"]]
        if not sel:
            continue
        rel = np.array([100.0 * (r["calc"] - r["exp"]) / abs(r["exp"])
                        for r in sel])
        signs = sum(1 for r in sel
                    if np.sign(r["calc"]) != np.sign(r["exp"]))
        print(f"    {basis:9s} n={len(sel):3d}  mean {rel.mean():+7.1f}%  "
              f"RMS {np.sqrt((rel ** 2).mean()):6.1f}%  "
              f"median |err| {np.median(np.abs(rel)):5.1f}%  "
              f"sign errors {signs}")

    path = _ROOT / "output" / "g_tensor_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  written to {path}")


if __name__ == "__main__":
    main()
