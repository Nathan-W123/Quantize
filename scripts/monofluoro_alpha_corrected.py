"""Does applying the vibration-rotation correction change the verdict?

The main benchmark fits measured ground-state constants as they stand, with no
correction. That leaves a systematic ~0.5% one-signed residual, because the
constants are r_0 quantities and the reference structures are r_s. This script
adds the correction and re-runs the three data-using legs.

What is corrected
-----------------
B_0 = B_e - (1/2) sum_r alpha_r, so subtracting the alpha term converts each
measured ground-state constant into an equilibrium one and the fit then targets
r_e. That is how semi-experimental equilibrium structures are built in the
literature: measured B_0 plus a computed alpha.

Fairness
--------
alpha needs a force field, which only the quantum backend can supply, so
switching on the optimizer's own `harmonic_from_hessian` path would correct the
hybrid legs and leave the spectroscopy-only leg uncorrected -- rigging the
comparison. Instead the correction table is computed ONCE per molecule and
passed to all three legs through `correction_table=`, which the spectral-only
path honours too. Every leg therefore sees identical targets.

Where the force field is evaluated
----------------------------------
At the theory-optimised geometry. That is a genuine stationary point, which the
alpha formulae assume, and it requires no knowledge of the reference structure.

    python scripts/monofluoro_alpha_corrected.py [molecule_key ...]
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
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from dev.monofluoro_references import MOLECULES  # noqa: E402
from scripts.monofluoro_benchmark import (  # noqa: E402
    BASIS,
    DATA_LEVELS,
    METHOD,
    OBJECTIVES,
    build_isotopologues,
    errors,
    start_geometry,
)

BASELINE = _ROOT / "output" / "monofluoro_benchmark.json"
OUT = _ROOT / "output" / "monofluoro_alpha_corrected.json"
LEGS = ["experiment"] + [label for label, _ in OBJECTIVES]


def _cached_hessian_fn(backend, cache):
    """Hessians keyed by geometry.

    The cubic force field costs 6N Hessians and depends only on the electronic
    surface, not on which isotopologues are being fitted -- so the same
    displaced geometries come up again for every data level. Caching turns the
    second and later calls into lookups.
    """
    def hessian_fn(coords_ang):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in cache:
            cache[key] = backend.run_hessian(coords_ang).hessian_bohr
        return cache[key]
    return hessian_fn


def correction_table(mol, geom, isos, backend, cache):
    """Cubic-force-field alpha for every species."""
    t0 = time.time()
    hessian_fn = _cached_hessian_fn(backend, cache)
    hess = hessian_fn(geom)

    with contextlib.redirect_stdout(io.StringIO()):
        ctbl, info = build_correction_table_from_hessian(
            hess, np.asarray(geom, dtype=float), isos,
            hessian_fn=hessian_fn, fd_delta_cubic=0.01,
            nonconvergent_policy="warn",
        )
    return ctbl, info, time.time() - t0


def run_leg(mol, isos, backend, start, ctbl, **kwargs):
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
        correction_table=ctbl,
        **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def main() -> None:
    if not BASELINE.exists():
        sys.exit(f"missing {BASELINE} - run scripts/monofluoro_benchmark.py first")
    base = {m["key"]: m for m in json.loads(BASELINE.read_text(encoding="utf-8"))}
    wanted = sys.argv[1:] or [m.key for m in MOLECULES]

    print(f"  RHF/{BASIS}. Vibration-rotation correction from a Cartesian cubic")
    print("  force field at the theory-optimised geometry, applied identically to")
    print("  every data-using leg.\n")

    out = []
    for mol in MOLECULES:
        if mol.key not in wanted:
            continue
        b = base[mol.key]
        start = start_geometry(mol)
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        theory_geom = backend.optimise(start)
        hess_cache: dict = {}

        print(f"{'=' * 84}")
        print(f"  {mol.name} ({mol.formula}) — {mol.internal_dof} internal DOF")
        print(f"{'=' * 84}")

        rec = {"molecule": mol.name, "key": mol.key,
               "internal_dof": mol.internal_dof,
               "theory": b["theory"], "levels": {}}

        for level_label, limit in DATA_LEVELS:
            isos = build_isotopologues(mol, limit)
            ctbl, info, t_ff = correction_table(mol, theory_geom, isos, backend,
                                                hess_cache)
            n_obs = sum(len(i["component_indices"]) for i in isos)

            nonconv = info.get("nonconvergent", {}) or {}
            statuses = sorted(set(info.get("anharmonic_statuses", [])))
            all_ratios = [r for comps in nonconv.values() for r in comps.values()
                          if np.isfinite(r)]
            print(f"\n  {level_label}: {len(isos)} species, {n_obs} constants "
                  f"(force field {t_ff / 60:.1f} min)")
            print(f"    anharmonic term: {', '.join(statuses) or 'unknown'}")
            if nonconv:
                worst = max(all_ratios) if all_ratios else float("nan")
                comps = sorted({c for v in nonconv.values() for c in v})
                print(f"    WARNING: cubic term exceeds harmonic in "
                      f"{len(nonconv)}/{len(isos)} species "
                      f"(components {', '.join(comps)}; worst ratio {worst:.1f}) "
                      f"-> VPT2 series diverging, correction unreliable")

            print(f"    {'leg':<22}{'uncorrected':>13}{'corrected':>12}"
                  f"{'change':>10}{'C-F unc':>10}{'C-F cor':>10}")
            print("    " + "-" * 77)
            lvl = {"n_observables": n_obs, "rank": b["levels"][level_label]["rank"],
                   "anharmonic_statuses": statuses,
                   "n_species": len(isos),
                   "n_nonconvergent_species": len(nonconv),
                   "worst_divergence_ratio": (max(all_ratios) if all_ratios
                                              else None),
                   "nonconvergent": {k: v for k, v in nonconv.items()},
                   "legs": {}}
            for leg in LEGS:
                kwargs = dict(OBJECTIVES).get(leg, {})
                is_hybrid = leg != "experiment"
                extra = dict(max_iter=40, hess_recalc_every=10) if is_hybrid \
                    else dict(max_iter=60)
                geom = run_leg(mol, isos, "pyscf_hf" if is_hybrid else "none",
                               start, ctbl, **extra, **kwargs)
                e = errors(mol, geom)
                old = b["levels"][level_label]["legs"][leg]
                lvl["legs"][leg] = {**e, "uncorrected_rms_bond_ma": old["rms_bond_ma"],
                                    "uncorrected_cf_err_ma": old["cf_err_ma"]}
                print(f"    {leg:<22}{old['rms_bond_ma']:>13.1f}"
                      f"{e['rms_bond_ma']:>12.1f}"
                      f"{e['rms_bond_ma'] - old['rms_bond_ma']:>+10.1f}"
                      f"{old['cf_err_ma']:>+10.1f}{e['cf_err_ma']:>+10.1f}")
            print(f"    {'theory (unchanged)':<22}{b['theory']['rms_bond_ma']:>13.1f}"
                  f"{'—':>12}{'—':>10}{b['theory']['cf_err_ma']:>+10.1f}{'—':>10}")
            rec["levels"][level_label] = lvl
        out.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  summary written to {OUT}")


if __name__ == "__main__":
    main()
