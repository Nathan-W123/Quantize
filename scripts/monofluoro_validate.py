"""Validate the calibrated hybrid on a molecule it was not calibrated on.

`monofluoro_tune.py` chose the joint objective's prior width by scanning it over
vinyl fluoride, acetyl fluoride and fluoroethane. Reporting that it beats theory
on those same three would prove very little, since the setting was picked to do
exactly that.

Fluorobenzene took no part in that choice. It is also a harder problem than any
of them: four measured species give 12 rotational constants against 30 internal
degrees of freedom, every substitution is deuterium so no carbon is ever located
directly, and the molecule is planar, which makes a further three directions
invisible to a linearised fit. If the calibrated setting generalises, it should
beat theory here too, with nothing adjusted.

    python scripts/monofluoro_validate.py
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
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from dev.monofluoro_references import HELDOUT  # noqa: E402
from scripts.monofluoro_benchmark import (  # noqa: E402
    BASIS,
    DATA_LEVELS,
    METHOD,
    build_isotopologues,
    errors,
    start_geometry,
)

#: The value chosen by the scan, applied here with nothing else changed.
CALIBRATED_PRIOR_SIGMA_ANG = 0.030

OUT = _ROOT / "output" / "monofluoro_validate.json"


def effective_rank(mol, isos, h=1e-5) -> int:
    x0 = np.asarray(mol.geometry, dtype=float).ravel()
    rows = []
    for iso in isos:
        masses = np.asarray(iso["masses"], dtype=float)
        keep = iso["component_indices"]
        blk = np.zeros((len(keep), x0.size))
        for k in range(x0.size):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            d = (rotational_constants_mhz(xp.reshape(-1, 3), masses)
                 - rotational_constants_mhz(xm.reshape(-1, 3), masses)) / (2 * h)
            blk[:, k] = d[keep]
        rows.append(blk / np.asarray(iso["sigma_constants"])[:, None])
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int((s > 1e-5 * s.max()).sum())


def run(mol, isos, start, backend, **kwargs):
    opt = MolecularOptimizer(
        elems=list(mol.elems),
        coords=np.asarray(start, dtype=float),
        isotopologues=isos,
        quantum_backend=backend,
        orca_method=METHOD,
        orca_basis=BASIS,
        coordinate_mode="cartesian",
        spectral_only=(backend == "none"),
        use_autoconfig=False,
        **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def main() -> None:
    print(f"  Held-out validation. RHF/{BASIS}, "
          f"quantum_prior_sigma_ang = {CALIBRATED_PRIOR_SIGMA_ANG} A, "
          f"nothing tuned per molecule.\n")
    out = []
    for mol in HELDOUT:
        start = start_geometry(mol)
        b = get_backend("pyscf_hf")(elems=list(mol.elems), method=METHOD,
                                    basis=BASIS)
        t0 = time.time()
        theory_geom = b.optimise(start)
        theory = errors(mol, theory_geom)
        theory["seconds"] = time.time() - t0

        print(f"{'=' * 86}")
        print(f"  {mol.name} ({mol.formula}) — {mol.n_atoms} atoms, "
              f"{mol.internal_dof} internal DOF")
        print(f"  structure: {mol.structure_source}")
        print(f"  constants: {mol.constants_source}")
        print(f"{'=' * 86}")
        print(f"\n  {'method':<26}{'RMS bond':>10}{'max bond':>10}"
              f"{'RMS angle':>11}{'C-F err':>10}{'vs theory':>11}")
        print(f"  {'':<26}{'(mA)':>10}{'(mA)':>10}{'(deg)':>11}{'(mA)':>10}")
        print("  " + "-" * 78)
        e0 = errors(mol, start)
        print(f"  {'start':<26}{e0['rms_bond_ma']:>10.1f}{e0['max_bond_ma']:>10.1f}"
              f"{e0['rms_angle_deg']:>11.2f}{e0['cf_err_ma']:>+10.1f}{'':>11}")
        print(f"  {'THEORY (no data)':<26}{theory['rms_bond_ma']:>10.1f}"
              f"{theory['max_bond_ma']:>10.1f}{theory['rms_angle_deg']:>11.2f}"
              f"{theory['cf_err_ma']:>+10.1f}{'—':>11}")

        rec = {"molecule": mol.name, "key": mol.key,
               "internal_dof": mol.internal_dof,
               "structure_source": mol.structure_source,
               "constants_source": mol.constants_source,
               "prior_sigma_ang": CALIBRATED_PRIOR_SIGMA_ANG,
               "reference_internals": mol.internal_coordinates(mol.geometry),
               "start_errors": {k: v for k, v in e0.items() if k != "internals"},
               "theory": {k: v for k, v in theory.items() if k != "internals"},
               "levels": {}}

        for label, limit in DATA_LEVELS:
            isos = build_isotopologues(mol, limit)
            n_obs = sum(len(i["component_indices"]) for i in isos)
            rank = effective_rank(mol, isos)
            print("  " + "-" * 78)
            print(f"  {label}: {len(isos)} species, {n_obs} constants -> rank "
                  f"{rank} of {mol.internal_dof} "
                  f"({mol.internal_dof - rank} undetermined)")
            lvl = {"n_observables": n_obs, "rank": rank,
                   "deficit": mol.internal_dof - rank, "legs": {}}
            for leg, backend, kw in (
                ("experiment", "none", dict(max_iter=60)),
                ("hybrid, joint prior", "pyscf_hf",
                 dict(max_iter=40, hess_recalc_every=10, objective_mode="joint",
                      quantum_prior_sigma_ang=CALIBRATED_PRIOR_SIGMA_ANG)),
            ):
                t0 = time.time()
                g = run(mol, isos, start, backend, **kw)
                e = errors(mol, g)
                e["seconds"] = time.time() - t0
                lvl["legs"][leg] = {k: v for k, v in e.items() if k != "internals"}
                lvl["legs"][leg]["internals"] = e["internals"]
                delta = e["rms_bond_ma"] - theory["rms_bond_ma"]
                mark = "BEATS" if delta < 0 else "loses"
                print(f"  {'  ' + leg:<26}{e['rms_bond_ma']:>10.1f}"
                      f"{e['max_bond_ma']:>10.1f}{e['rms_angle_deg']:>11.2f}"
                      f"{e['cf_err_ma']:>+10.1f}{delta:>+8.1f} {mark}")
            rec["levels"][label] = lvl

        ref = mol.internal_coordinates(mol.geometry)
        print(f"\n  {'parameter':<12}{'published':>11}{'theory':>10}"
              f"{'experiment':>12}{'hybrid':>10}")
        print("  " + "-" * 56)
        full = rec["levels"]["all species"]["legs"]
        for name in list(mol.bonds) + list(mol.angles):
            fmt = "{:>10.4f}" if name in mol.bonds else "{:>10.2f}"
            wide = "{:>11.4f}" if name in mol.bonds else "{:>11.2f}"
            row = f"  {name:<12}" + wide.format(ref[name])
            row += fmt.format(mol.internal_coordinates(theory_geom)[name])
            row += fmt.replace(">10", ">12").format(
                full["experiment"]["internals"][name])
            row += fmt.format(full["hybrid, joint prior"]["internals"][name])
            print(row)
        out.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  written to {OUT}")


if __name__ == "__main__":
    main()
