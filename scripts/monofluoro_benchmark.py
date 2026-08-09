"""Theory vs experiment vs hybrid on three monofluorinated molecules.

Every molecule is run identically:

  * ground truth is a published microwave structure, paired with the rotational
    constants from the same study and checked for mutual consistency;
  * theory is RHF/6-31G through PySCF -- the same method and basis for all three,
    and the same one used for fluorobenzene, so the numbers are comparable;
  * the fit starts from a geometry displaced from the truth, so nothing is handed
    the answer;
  * the same optimizer settings are used throughout.

Each molecule is run at two data levels, because the fluorobenzene work showed
that how much the quantum step contributes depends entirely on whether the
spectral data alone can pin the structure down:

  * "parent only"  -- 3 observables, far fewer than the internal degrees of
    freedom, so the spectral problem is badly under-determined;
  * "all species"  -- every symmetry-unique single substitution, which brings
    the count up to or past the degrees of freedom.

    python scripts/monofluoro_benchmark.py            # all three
    python scripts/monofluoro_benchmark.py fluoroethane

Writes a JSON summary next to the printed tables so the report can be built from
the same numbers.
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
from dev.monofluoro_references import (  # noqa: E402
    HELDOUT,
    MOLECULES,
    MOLECULES_SET2,
    ReferenceMolecule,
)

#: Which molecule set to run. `set=2` is the independent second set, which
#: took no part in calibrating the hybrid's prior width.
_SETS = {"1": MOLECULES, "2": MOLECULES_SET2, "heldout": HELDOUT}
MOLECULES = next((_SETS[t.split('=', 1)[1]] for t in sys.argv[1:]
                  if t.startswith('set=')), MOLECULES)

METHOD, BASIS = "hf", "6-31g"
DISPLACEMENT_ANG = 0.03
SEED = 11
_K = 505379.0084353526

#: The level of theory applies to BOTH the theory-alone leg and the hybrid's
#: quantum half. Changing it for one and not the other would make the comparison
#: meaningless, so there is a single setting for both.
#:
#:     python scripts/monofluoro_benchmark.py method=b3lyp "basis=6-31g(d)" \
#:         sigma=0.007 objectives=joint
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

#: label -> how many isotopologues the fit is allowed to see. None means all.
DATA_LEVELS = (("parent only", 1), ("all species", None))

#: The two ways of combining the two information sources. `split` partitions the
#: parameter space hard -- data owns the directions it can see, theory owns the
#: rest. `joint` leaves every direction contested and weights each source by how
#: well it knows that direction; `quantum_prior_sigma_ang` is the displacement
#: over which the quantum surface is trusted, so it is set to the geometry error
#: RHF/6-31G actually shows rather than tuned per molecule.
#: How far the quantum surface is trusted, in Angstrom.
#:
#: This is the single most consequential setting in the hybrid, and it must be
#: set to the geometry error of the level of theory actually being used. RHF/6-31G
#: is off by 15-19 mA RMS on these molecules and ~29 mA on C-F, so the 0.005 A
#: carried over from earlier water and fluorobenzene work overstated its accuracy
#: five-fold and left the joint hybrid barely distinguishable from plain theory.
#:
#: `scripts/monofluoro_tune.py` scans it: at 0.005-0.015 A the hybrid beats theory
#: in 5 of 6 cases, and from 0.020 A to 0.080 A in all 6. The plateau is wide, so
#: the exact value matters much less than being in the right decade.
#:
#: For a molecule with no accepted structure -- the case this project is aimed at
#: -- it cannot be read off a reference. `scripts/estimate_theory_error.py`
#: estimates it instead from the spread between two levels of theory, which needs
#: no experimental structure at all.
QUANTUM_PRIOR_SIGMA_ANG = 0.030
for _tok in sys.argv[1:]:
    if _tok.startswith("sigma="):
        QUANTUM_PRIOR_SIGMA_ANG = float(_tok.split("=", 1)[1])

_ALL_OBJECTIVES = (
    ("hybrid, split", {}),
    ("hybrid, joint prior", {"objective_mode": "joint",
                             "quantum_prior_sigma_ang": QUANTUM_PRIOR_SIGMA_ANG}),
)
#: `objectives=joint` skips the split partition, which a scan over its
#: singular-value cutoff showed reaches at most 3 of 6 cases better than theory.
_WANT_OBJ = next((t.split("=", 1)[1].split(",") for t in sys.argv[1:]
                  if t.startswith("objectives=")), None)
OBJECTIVES = tuple(
    (label, kw) for label, kw in _ALL_OBJECTIVES
    if _WANT_OBJ is None or any(w in label for w in _WANT_OBJ)
)


def build_isotopologues(mol: ReferenceMolecule, limit=None) -> list[dict]:
    """Measured constants only, with the components that were actually observed.

    Nothing is synthesised: a constant the original study did not determine is
    absent rather than filled in, and the uncertainties come from the reference
    module's model (dominated by the r_s-versus-r_0 gap, not by measurement
    precision). alpha_constants are zero because no vibration-rotation
    correction is applied -- that offset is carried by sigma instead.
    """
    out = []
    for sp in mol.species[:limit]:
        out.append({
            "name": sp.label,
            "masses": sp.masses(mol.masses).tolist(),
            "obs_constants": sp.observed(),
            "sigma_constants": sp.sigmas(),
            "alpha_constants": [0.0] * len(sp.observed()),
            "component_indices": list(sp.component_indices),
        })
    return out


def start_geometry(mol: ReferenceMolecule) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return mol.geometry + rng.normal(0.0, DISPLACEMENT_ANG, size=mol.geometry.shape)


def run_fit(mol, isotopologues, backend, start, **kwargs):
    opt = MolecularOptimizer(
        elems=list(mol.elems),
        coords=np.asarray(start, dtype=float),
        isotopologues=isotopologues,
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


def errors(mol, coords) -> dict:
    got, ref = mol.internal_coordinates(coords), mol.internal_coordinates(mol.geometry)
    bonds = [(got[k] - ref[k]) * 1000 for k in mol.bonds]
    angles = [got[k] - ref[k] for k in mol.angles]
    cf_key = next(k for k in mol.bonds if k.endswith("-F"))
    return {
        "rms_bond_ma": float(np.sqrt(np.mean(np.square(bonds)))),
        "max_bond_ma": float(np.max(np.abs(bonds))),
        "rms_angle_deg": float(np.sqrt(np.mean(np.square(angles)))),
        "max_angle_deg": float(np.max(np.abs(angles))),
        "cf_err_ma": (got[cf_key] - ref[cf_key]) * 1000,
        "internals": got,
    }


def consistency(mol) -> float:
    """Worst percentage gap between the published geometry and any measured constant."""
    worst = 0.0
    for sp in mol.species:
        calc = rotational_constants_mhz(mol.geometry, sp.masses(mol.masses))
        for k, v in enumerate(sp.abc_mhz):
            if v is not None:
                worst = max(worst, abs(calc[k] / v - 1) * 100)
    return float(worst)


def effective_rank(mol, isos, h=1e-5) -> int:
    """Independent constraints in a set of isotopologues, not the raw count."""
    x0 = np.asarray(mol.geometry, dtype=float).ravel()
    rows = []
    for iso in isos:
        masses = np.asarray(iso["masses"], dtype=float)
        keep = iso["component_indices"]
        block = np.zeros((len(keep), x0.size))
        for k in range(x0.size):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            d = (rotational_constants_mhz(xp.reshape(-1, 3), masses)
                 - rotational_constants_mhz(xm.reshape(-1, 3), masses)) / (2 * h)
            block[:, k] = d[keep]
        rows.append(block / np.asarray(iso["sigma_constants"])[:, None])
    s = np.linalg.svd(np.vstack(rows), compute_uv=False)
    return int((s > 1e-5 * s.max()).sum())


def defect(abc) -> float:
    inertia = _K / np.asarray(abc, dtype=float)
    return float(inertia[2] - inertia[0] - inertia[1])


_HDR = (f"  {'method':<22}{'RMS bond':>10}{'max bond':>10}{'RMS angle':>11}"
        f"{'max angle':>11}{'C-F err':>10}{'time':>8}\n"
        f"  {'':<22}{'(mA)':>10}{'(mA)':>10}{'(deg)':>11}{'(deg)':>11}{'(mA)':>10}")


def _row(label, e, seconds=None):
    t = f"{seconds:>7.0f}s" if seconds is not None else f"{'-':>8}"
    return (f"  {label:<22}{e['rms_bond_ma']:>10.1f}{e['max_bond_ma']:>10.1f}"
            f"{e['rms_angle_deg']:>11.2f}{e['max_angle_deg']:>11.2f}"
            f"{e['cf_err_ma']:>+10.1f}{t}")


def benchmark(mol: ReferenceMolecule) -> dict:
    start = start_geometry(mol)
    n_species = len(mol.species)

    print(f"\n{'=' * 82}")
    print(f"  {mol.name} ({mol.formula}) — {mol.n_atoms} atoms, "
          f"{mol.internal_dof} internal DOF")
    print(f"  structure: {mol.structure_source}")
    print(f"  constants: {mol.constants_source}")
    print(f"  Measured parent B0 (MHz): {np.round(mol.b0_mhz, 1)}   "
          f"inertial defect {defect(mol.b0_mhz):+.3f} amu A^2")
    print(f"  Geometry reproduces every measured constant to {consistency(mol):.2f}%")
    print(f"  {n_species} measured species -> {mol.n_observables} measured constants")
    print(f"{'=' * 82}")

    results = {"molecule": mol.name, "formula": mol.formula, "key": mol.key,
               "n_atoms": mol.n_atoms, "internal_dof": mol.internal_dof,
               "n_isotopologues": n_species,
               "structure_source": mol.structure_source,
               "constants_source": mol.constants_source,
               "b0_mhz": mol.b0_mhz.tolist(),
               "inertial_defect": defect(mol.b0_mhz),
               "consistency_pct": consistency(mol),
               "bond_names": list(mol.bonds), "angle_names": list(mol.angles),
               "reference_internals": mol.internal_coordinates(mol.geometry),
               "start_internals": mol.internal_coordinates(start),
               "start_errors": {k: v for k, v in errors(mol, start).items()},
               "levels": {}}

    # Theory sees no spectral data at all, so it is run once and reused.
    t0 = time.time()
    backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=METHOD, basis=BASIS)
    theory_geom = backend.optimise(start)
    theory_time = time.time() - t0
    results["theory"] = {**errors(mol, theory_geom), "seconds": theory_time}

    print(f"\n{_HDR}")
    print("  " + "-" * 82)
    print(_row("start", errors(mol, start)))
    print(_row("THEORY (no data)", errors(mol, theory_geom), theory_time))

    for level_label, limit in DATA_LEVELS:
        isos = build_isotopologues(mol, limit)
        n_obs = sum(len(i["component_indices"]) for i in isos)
        rank = effective_rank(mol, isos)
        geoms, times = {"theory": theory_geom}, {"theory": theory_time}

        t0 = time.time()
        geoms["experiment"] = run_fit(mol, isos, "none", start, max_iter=60)
        times["experiment"] = time.time() - t0

        for obj_label, kwargs in OBJECTIVES:
            t0 = time.time()
            geoms[obj_label] = run_fit(mol, isos, "pyscf_hf", start,
                                       max_iter=40, hess_recalc_every=10, **kwargs)
            times[obj_label] = time.time() - t0

        legs = ["theory", "experiment"] + [label for label, _ in OBJECTIVES]

        print("  " + "-" * 82)
        print(f"  {level_label}: {len(isos)} species, {n_obs} measured constants "
              f"-> rank {rank} vs {mol.internal_dof} DOF "
              f"({mol.internal_dof - rank} direction(s) undetermined)")
        for leg in legs[1:]:
            print(_row(f"  {leg}", errors(mol, geoms[leg]), times[leg]))

        level = {"n_isotopologues": len(isos), "n_observables": n_obs,
                 "rank": rank, "deficit": mol.internal_dof - rank, "legs": {}}
        for leg in legs:
            level["legs"][leg] = {**errors(mol, geoms[leg]), "seconds": times[leg]}
        results["levels"][level_label] = level

        ref = mol.internal_coordinates(mol.geometry)
        print(f"\n  {level_label} — parameter table")
        print(f"  {'parameter':<14}{'published':>11}" + "".join(
            f"{leg:>20}" for leg in legs))
        print("  " + "-" * (25 + 20 * len(legs)))
        for name in list(mol.bonds) + list(mol.angles):
            wide, cell = ("{:>11.4f}", "{:>20.4f}") if name in mol.bonds \
                else ("{:>11.2f}", "{:>20.2f}")
            row = f"  {name:<14}" + wide.format(ref[name])
            row += "".join(cell.format(mol.internal_coordinates(geoms[leg])[name])
                           for leg in legs)
            print(row)
        print()
    return results


def main() -> None:
    wanted = [t for t in sys.argv[1:] if "=" not in t] or [m.key for m in MOLECULES]
    print(f"  Level of theory: {METHOD.upper()}/{BASIS} (PySCF) — the same for the")
    print(f"  theory-alone leg and the hybrid's quantum half, for every molecule.")
    print(f"  Quantum prior sigma: {QUANTUM_PRIOR_SIGMA_ANG} A.")
    print(f"  Start displaced from the published structure by "
          f"{DISPLACEMENT_ANG} A rms, seed {SEED}.")
    out = [benchmark(m) for m in MOLECULES if m.key in wanted]
    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    path = _ROOT / "output" / f"monofluoro_benchmark_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  summary written to {path}")


if __name__ == "__main__":
    main()
