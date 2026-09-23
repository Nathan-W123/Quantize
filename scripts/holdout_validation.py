"""Validate the engine without any reference structure at all.

Every accuracy number in this repository is scored against a published
structure, and those are the weak link: they are r_s substitution structures
while the engine targets r_e, and on fluoroacetylene the two reference types
disagree by 2.85 mA -- larger than the 2.27 mA the hybrid scored against one of
them. The benchmark cannot resolve the engine's accuracy because the yardstick
is coarser than the thing being measured, and the r_e structures that would fix
that are behind journal paywalls.

Held-out prediction needs no yardstick. Fit the structure to all isotopologues
but one, then predict the rotational constants of the species that was left
out. A structure closer to the truth predicts unseen constants better, whatever
the truth happens to be -- so this ranks the methods using only data the
molecule itself provides.

It is also a harder test than fitting. Every method can reproduce the constants
it was fitted to; only a structure that is actually right reproduces the ones it
has never seen.

Three legs, exactly as in the main benchmark:

    theory   the level of theory's own geometry -- uses no data, so its
             prediction is the same whichever species is held out, which makes
             it the natural control
    ME       mixed estimation fitted to the remaining species
    hybrid   the engine fitted to the remaining species

    python scripts/holdout_validation.py [method=b3lyp] [basis=6-31g(d)]
                                         [molecule_key ...]

Results, B3LYP/6-31G(d), RMS relative error on the withheld species
-------------------------------------------------------------------

    molecule          spec   theory       ME   hybrid   reference-based said
    ozone                5   2.7298   0.0700   0.0694   hybrid
    vinyl fluoride       8   1.4052   0.5040   0.5014   theory
    acetyl fluoride     10   1.8070   0.2677   0.2485   theory
    fluoroethane         6   1.6181   0.8871   0.0944   theory
    fluoroacetylene      6   1.0973   0.0040   0.0163   hybrid
    MEAN                     1.7315   0.3466   0.1860

The hybrid beats the level of theory on 5 of 5, by a median factor of 17, and
is best of the three on 4 of 5.

What makes this worth having is the last column. On vinyl fluoride, acetyl
fluoride and fluoroethane the reference-based benchmark says theory alone wins
-- and all three predict isotopologues they have never seen 2.8x, 7.3x and 17x
better than theory does. Those are the molecules whose published references are
r_s structures, so the disagreement is evidence about the references rather
than about the engine.

Limitation, and it is a real one. This measures consistency with the
*corrected* constants, not with the truth. A bias shared across every
isotopologue of a molecule is absorbed into the fitted structure and then
predicts the withheld species just as well, so a confident result here does not
by itself establish that a structure is closer to the true equilibrium
geometry. What it does establish is that the fitted structures capture the
isotopic dependence of the constants and the level of theory does not.

The one independent check available points the same way: water is the only
molecule in the set with a genuine r_e reference, and there the hybrid scores
4.13 mA against theory's 10.89.
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

import dev.pyscf_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    rotational_constants_mhz,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from dev.monofluoro_references import (  # noqa: E402
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    WATER_SET,
)
from scripts.mixed_estimation_baseline import (  # noqa: E402
    _SIGMA_X_BY_LEVEL,
    corrected_targets,
    mixed_estimation_fit,
    rms_bond_error,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "b3lyp", "6-31g(d)"
_args: list[str] = []
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]
    else:
        _args.append(_tok)

SIGMA_X_ANG = _SIGMA_X_BY_LEVEL.get((METHOD.lower(), BASIS.lower()), 0.020)

#: Below this many species the exercise is not meaningful: dropping one from a
#: two-species molecule leaves nothing to fit.
MIN_SPECIES = 5


def predict_error(geom, iso, targets_by_species):
    """Relative error in predicting one species' corrected constants, in %.

    The fitted structure is rigid and equilibrium, so what it predicts is B_e,
    and the corrected target is the B_e,SE the other methods were fitted to --
    a like-for-like comparison that does not re-charge anyone for the
    vibrational correction.
    """
    want = targets_by_species.get(iso["name"])
    if not want:
        return None
    masses = np.asarray(iso["masses"], dtype=float)
    calc = rotational_constants_mhz(np.asarray(geom, dtype=float), masses)
    errs = [100.0 * (float(calc[c]) - v) / abs(v) for c, v in want.items()]
    return float(np.sqrt(np.mean(np.square(errs))))


def main() -> None:
    pool = (list(MOLECULES) + list(MOLECULES_SET2)
            + [WATER_SET[0], OZONE, ISOCYANIC_ACID])
    wanted = _args or [m.key for m in pool]
    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    geo_path = _ROOT / "output" / f"theory_geometries_{tag}.json"
    cache = json.loads(geo_path.read_text(encoding="utf-8")) \
        if geo_path.exists() else {}
    out_path = _ROOT / "output" / f"holdout_{tag}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}

    print(f"  {METHOD.upper()}/{BASIS}, sigma_x = {SIGMA_X_ANG} A")
    print("  Fit on all isotopologues but one; predict the one left out.\n")

    for mol in pool:
        if mol.key not in wanted or mol.key in out:
            continue
        isos = build_isotopologues(mol, None)
        if len(isos) < MIN_SPECIES:
            continue
        t0 = time.time()
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        if mol.key in cache:
            theory = np.asarray(cache[mol.key]["coords"], dtype=float)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                theory = backend.optimise(start_geometry(mol))

        hcache: dict = {}

        def hessian_fn(coords_ang, _b=backend, _c=hcache):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _b.run_hessian(coords_ang).hessian_bohr
            return _c[key]

        with contextlib.redirect_stdout(io.StringIO()):
            ctbl, _ = build_correction_table_from_hessian(
                hessian_fn(theory), theory, isos, hessian_fn=hessian_fn,
                cubic_scheme="normal_mode")

        # Corrected targets for every species, indexed for lookup.
        all_targets = corrected_targets(mol, isos, ctbl)
        by_species: dict = {}
        for iso in isos:
            want = {}
            for t in all_targets:
                # corrected_targets carries masses, not labels, so match on them
                if np.allclose(np.asarray(t["masses"], dtype=float),
                               np.asarray(iso["masses"], dtype=float)):
                    want[int(t["component"])] = float(t["value"])
            by_species[iso["name"]] = want

        rows = []
        for held in range(len(isos)):
            keep = [isos[i] for i in range(len(isos)) if i != held]
            kept_targets = [t for t in all_targets
                            if not np.allclose(
                                np.asarray(t["masses"], dtype=float),
                                np.asarray(isos[held]["masses"], dtype=float))]
            me_geom, _ = mixed_estimation_fit(mol, kept_targets, theory,
                                              SIGMA_X_ANG)
            opt = MolecularOptimizer(
                elems=list(mol.elems), coords=theory.copy(),
                isotopologues=keep, quantum_backend="pyscf_hf",
                orca_method=METHOD, orca_basis=BASIS,
                coordinate_mode="cartesian", use_autoconfig=False,
                max_iter=40, hess_recalc_every=10, correction_table=ctbl,
                quantum_prior_sigma_ang=SIGMA_X_ANG,
                prior_target_coords=theory.copy(),
                chi2_rescale=True, chi2_rescale_max_passes=3)
            with contextlib.redirect_stdout(io.StringIO()):
                hyb = opt.run()

            rows.append({
                "held_out": isos[held]["name"],
                "theory": predict_error(theory, isos[held], by_species),
                "me": predict_error(me_geom, isos[held], by_species),
                "hybrid": predict_error(hyb, isos[held], by_species),
            })
            print(f"    hold out {isos[held]['name']:<22}"
                  f"theory {rows[-1]['theory']:7.4f}%  "
                  f"ME {rows[-1]['me']:7.4f}%  "
                  f"hybrid {rows[-1]['hybrid']:7.4f}%", flush=True)

        good = [r for r in rows if None not in (r["theory"], r["me"], r["hybrid"])]
        summary = {k: float(np.sqrt(np.mean([r[k] ** 2 for r in good])))
                   for k in ("theory", "me", "hybrid")}
        out[mol.key] = {"n_species": len(isos), "rows": rows,
                        "rms_percent": summary}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  {mol.name:<22} RMS  theory {summary['theory']:.4f}%  "
              f"ME {summary['me']:.4f}%  hybrid {summary['hybrid']:.4f}%"
              f"   ({(time.time()-t0)/60:.1f} min)\n", flush=True)

    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
