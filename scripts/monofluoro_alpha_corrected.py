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
from scripts.monofluoro_benchmark import (  # noqa: E402
    BASIS,
    MOLECULES,
    DATA_LEVELS,
    METHOD,
    OBJECTIVES,
    build_isotopologues,
    errors,
    start_geometry,
)

BASELINE = _ROOT / "output" / "monofluoro_benchmark.json"
for _tok in sys.argv[1:]:
    if _tok.startswith("baseline="):
        BASELINE = _ROOT / _tok.split("=", 1)[1]
_ALPHA_TAG = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
OUT = _ROOT / "output" / f"monofluoro_alpha_corrected_{_ALPHA_TAG}.json"
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


def correction_table(mol, geom, isos, backend, cache, policy="warn"):
    """Cubic-force-field alpha for every species.

    ``policy`` decides what happens to a component whose cubic term exceeds its
    harmonic one, i.e. whose perturbation series is not converging:
    ``warn`` applies it anyway, ``inflate`` scales its sigma by the divergence
    ratio so the fit discounts it, ``drop`` omits it entirely.
    """
    t0 = time.time()
    hessian_fn = _cached_hessian_fn(backend, cache)
    hess = hessian_fn(geom)

    with contextlib.redirect_stdout(io.StringIO()):
        ctbl, info = build_correction_table_from_hessian(
            hess, np.asarray(geom, dtype=float), isos,
            hessian_fn=hessian_fn, fd_delta_cubic=0.01,
            nonconvergent_policy=policy,
            # Corrections here are evaluated at the theory-optimised geometry,
            # which is the stationary point the normal-mode scheme requires.
            cubic_scheme="normal_mode",
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


def run_level(mol, b, level_label, policy, isos, n_obs, theory_geom,
              backend, hess_cache) -> dict:
    """One data level under one nonconvergence policy, all three legs.

    Every data-using leg warm-starts from the theory geometry, exactly as the
    uncorrected baseline it is compared against does. This script originally
    cold-started from the noisy reference displacement, and for
    fluoroacetylene that alone produced a 39 mA "blowup" that was blamed on
    the correction: a cold start leaves a linear molecule slightly bent, the
    bend directions are invisible to the data (their singular values sit
    below any sane cutoff), and the residual O(bend^2) contamination of the
    constants drags the fitted bonds tens of mA off. Corrected and
    uncorrected legs must differ by the correction, not by the start.
    """
    ctbl, info, t_ff = correction_table(mol, theory_geom, isos, backend,
                                        hess_cache, policy)
    nonconv = info.get("nonconvergent", {}) or {}
    dropped = info.get("dropped_components", {}) or {}
    statuses = sorted(set(info.get("anharmonic_statuses", [])))
    ratios = [r for comps in nonconv.values() for r in comps.values()
              if np.isfinite(r)]
    worst = max(ratios) if ratios else None

    print(f"\n  {level_label} — policy '{policy}': {len(isos)} species, "
          f"{n_obs} constants (force field {t_ff / 60:.1f} min)")
    print(f"    anharmonic term: {', '.join(statuses) or 'unknown'}")
    if nonconv:
        comps = sorted({c for v in nonconv.values() for c in v})
        print(f"    cubic exceeds harmonic in {len(nonconv)}/{len(isos)} species "
              f"(components {', '.join(comps)}; worst ratio {worst:.1f})")
    if dropped:
        n_dropped = sum(len(v) for v in dropped.values())
        print(f"    dropped {n_dropped} component(s) across "
              f"{len(dropped)} species")

    print(f"    {'leg':<22}{'uncorrected':>13}{'corrected':>12}"
          f"{'change':>10}{'C-F unc':>10}{'C-F cor':>10}")
    print("    " + "-" * 77)
    lvl = {"level": level_label, "policy": policy, "n_observables": n_obs,
           "rank": b["levels"][level_label]["rank"],
           "anharmonic_statuses": statuses, "n_species": len(isos),
           "n_nonconvergent_species": len(nonconv),
           "worst_divergence_ratio": worst,
           "n_dropped_components": sum(len(v) for v in dropped.values()),
           "legs": {}}
    for leg in LEGS:
        kwargs = dict(OBJECTIVES).get(leg, {})
        is_hybrid = leg != "experiment"
        extra = dict(max_iter=40, hess_recalc_every=10) if is_hybrid \
            else dict(max_iter=60)
        geom = run_leg(mol, isos, "pyscf_hf" if is_hybrid else "none",
                       theory_geom, ctbl, **extra, **kwargs)
        e = errors(mol, geom)
        old = b["levels"][level_label]["legs"][leg]
        lvl["legs"][leg] = {**e,
                            "uncorrected_rms_bond_ma": old["rms_bond_ma"],
                            "uncorrected_cf_err_ma": old["cf_err_ma"]}
        print(f"    {leg:<22}{old['rms_bond_ma']:>13.1f}{e['rms_bond_ma']:>12.1f}"
              f"{e['rms_bond_ma'] - old['rms_bond_ma']:>+10.1f}"
              f"{old['cf_err_ma']:>+10.1f}{e['cf_err_ma']:>+10.1f}")
    print(f"    {'theory (unchanged)':<22}{b['theory']['rms_bond_ma']:>13.1f}"
          f"{'—':>12}{'—':>10}{b['theory']['cf_err_ma']:>+10.1f}{'—':>10}")
    return lvl


def main() -> None:
    if not BASELINE.exists():
        sys.exit(f"missing {BASELINE} - run scripts/monofluoro_benchmark.py first")
    base = {m["key"]: m for m in json.loads(BASELINE.read_text(encoding="utf-8"))}
    argv = sys.argv[1:]
    pol_tokens = [t.split("=", 1)[1] for t in argv if t.startswith("policy=")]
    policies = pol_tokens[0].split(",") if pol_tokens else ["warn"]
    wanted = [t for t in argv
              if not t.startswith(("policy=", "baseline="))] \
        or [m.key for m in MOLECULES]

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
            n_obs = sum(len(i["component_indices"]) for i in isos)
            for policy in policies:
                key = f"{level_label} [{policy}]"
                rec["levels"][key] = run_level(
                    mol, b, level_label, policy, isos, n_obs,
                    theory_geom, backend, hess_cache)
        out.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  summary written to {OUT}")


if __name__ == "__main__":
    main()
