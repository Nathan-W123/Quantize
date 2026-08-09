"""Find a hybrid configuration that beats theory alone in every case.

The joint objective weights the quantum surface by `quantum_prior_sigma_ang`,
documented as "the displacement over which the quantum surface is trusted,
roughly the geometry error of the method". The benchmark used 0.005 A, carried
over from earlier water and fluorobenzene work. That is far too tight: RHF/6-31G
is off by 15-19 mA RMS on these molecules and by ~29 mA on C-F, so 0.005 A tells
the fit that theory is four to six times better than it is, and the joint hybrid
consequently barely moves off theory.

This scans the parameter and reports, for each setting, how many of the six
molecule/data-level cases beat theory. A setting is only useful if it can be
chosen without knowing the answer -- and this one can, because it is a property
of the level of theory rather than of the molecule.

    python scripts/monofluoro_tune.py                 # scan the prior
    python scripts/monofluoro_tune.py --split         # scan the split cutoff too
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
from dev.monofluoro_references import MOLECULES  # noqa: E402
from scripts.monofluoro_benchmark import (  # noqa: E402
    BASIS as _DEFAULT_BASIS,
    DATA_LEVELS,
    METHOD as _DEFAULT_METHOD,
    build_isotopologues,
    errors,
    start_geometry,
)

#: The level of theory is overridable, because it has to be: the hybrid must be
#: compared against theory alone at the SAME level, or the comparison says
#: nothing about whether combining the sources helps. RHF/6-31G is off by 16.6 mA
#: RMS on these molecules and B3LYP/6-31G(d) by 6.8 mA, so "hybrid beats theory"
#: at one level does not carry over to the other.
#:
#:     python scripts/monofluoro_tune.py method=b3lyp basis=6-31g(d)
METHOD, BASIS = _DEFAULT_METHOD, _DEFAULT_BASIS
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

#: Scanned values of quantum_prior_sigma_ang, in Angstrom. The useful range is
#: bracketed by the actual geometry error of the level of theory.
#: Overridable, because the useful range scales with the level of theory's own
#: error: around 0.02 A for RHF/6-31G, around 0.007 A for B3LYP/6-31G(d).
#:     python scripts/monofluoro_tune.py sigmas=0.003,0.005,0.008,0.012,0.020
PRIOR_SIGMAS = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080)
for _tok in sys.argv[1:]:
    if _tok.startswith("sigmas="):
        PRIOR_SIGMAS = tuple(float(v) for v in _tok.split("=", 1)[1].split(","))

#: Scanned relative singular-value cutoffs for the split objective.
SPLIT_CUTOFFS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1)

_TAG = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
OUT = _ROOT / "output" / f"monofluoro_tune_{_TAG}.json"


def run(mol, isos, start, **kwargs):
    opt = MolecularOptimizer(
        elems=list(mol.elems),
        coords=np.asarray(start, dtype=float),
        isotopologues=isos,
        quantum_backend="pyscf_hf",
        orca_method=METHOD,
        orca_basis=BASIS,
        coordinate_mode="cartesian",
        use_autoconfig=False,
        max_iter=40,
        hess_recalc_every=10,
        **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def cases():
    """(molecule, level label, isotopologue list) for all six cases."""
    for mol in MOLECULES:
        for label, limit in DATA_LEVELS:
            yield mol, label, build_isotopologues(mol, limit)


def main() -> None:
    do_split = "--split" in sys.argv
    print(f"  {METHOD.upper()}/{BASIS}. Scanning the hybrid's trust in theory.")
    print("  Theory alone and the hybrid's quantum half use the same level, so"
          " the\n  comparison isolates whether combining the sources helps.\n")

    theory = {}
    starts = {}
    for mol in MOLECULES:
        starts[mol.key] = start_geometry(mol)
        b = get_backend("pyscf_hf")(elems=list(mol.elems), method=METHOD,
                                    basis=BASIS)
        g = b.optimise(starts[mol.key])
        theory[mol.key] = errors(mol, g)
        print(f"  theory {mol.name:<18} RMS bond {theory[mol.key]['rms_bond_ma']:5.1f} mA"
              f"   C-F {theory[mol.key]['cf_err_ma']:+6.1f} mA")

    all_cases = list(cases())
    results = {"prior_sigmas": list(PRIOR_SIGMAS), "theory": theory, "scan": {}}

    print(f"\n  {'quantum_prior_sigma_ang scan (joint objective)':<44}")
    header = f"  {'case':<30}{'theory':>9}" + "".join(
        f"{s:>9.3f}" for s in PRIOR_SIGMAS)
    print(header)
    print("  " + "-" * (39 + 9 * len(PRIOR_SIGMAS)))

    beats = {s: 0 for s in PRIOR_SIGMAS}
    cf_beats = {s: 0 for s in PRIOR_SIGMAS}
    for mol, label, isos in all_cases:
        th = theory[mol.key]
        row, key = [], f"{mol.name} / {label}"
        rec = {}
        for s in PRIOR_SIGMAS:
            t0 = time.time()
            g = run(mol, isos, starts[mol.key], objective_mode="joint",
                    quantum_prior_sigma_ang=float(s))
            e = errors(mol, g)
            rec[str(s)] = {k: v for k, v in e.items() if k != "internals"}
            rec[str(s)]["seconds"] = time.time() - t0
            row.append(e["rms_bond_ma"])
            beats[s] += e["rms_bond_ma"] < th["rms_bond_ma"]
            cf_beats[s] += abs(e["cf_err_ma"]) < abs(th["cf_err_ma"])
        results["scan"][key] = rec
        best = min(range(len(row)), key=lambda i: row[i])
        cells = "".join(
            (f"{v:>8.1f}*" if i == best else f"{v:>9.1f}")
            for i, v in enumerate(row))
        print(f"  {key:<30}{th['rms_bond_ma']:>9.1f}{cells}")

    print("  " + "-" * (39 + 9 * len(PRIOR_SIGMAS)))
    print(f"  {'cases beating theory (of 6)':<30}{'':>9}"
          + "".join(f"{beats[s]:>9d}" for s in PRIOR_SIGMAS))
    print(f"  {'  ... on C-F':<30}{'':>9}"
          + "".join(f"{cf_beats[s]:>9d}" for s in PRIOR_SIGMAS))
    results["beats_theory"] = {str(s): beats[s] for s in PRIOR_SIGMAS}
    results["cf_beats_theory"] = {str(s): cf_beats[s] for s in PRIOR_SIGMAS}

    if do_split:
        print(f"\n  sv_threshold scan (split objective)")
        header = f"  {'case':<30}{'theory':>9}" + "".join(
            f"{c:>9.0e}" for c in SPLIT_CUTOFFS)
        print(header)
        print("  " + "-" * (39 + 9 * len(SPLIT_CUTOFFS)))
        sbeats = {c: 0 for c in SPLIT_CUTOFFS}
        results["split_cutoffs"] = list(SPLIT_CUTOFFS)
        results["split_scan"] = {}
        for mol, label, isos in all_cases:
            th = theory[mol.key]
            row, key = [], f"{mol.name} / {label}"
            rec = {}
            for c in SPLIT_CUTOFFS:
                g = run(mol, isos, starts[mol.key], sv_threshold=float(c))
                e = errors(mol, g)
                rec[str(c)] = {k: v for k, v in e.items() if k != "internals"}
                row.append(e["rms_bond_ma"])
                sbeats[c] += e["rms_bond_ma"] < th["rms_bond_ma"]
            results["split_scan"][key] = rec
            best = min(range(len(row)), key=lambda i: row[i])
            cells = "".join((f"{v:>8.1f}*" if i == best else f"{v:>9.1f}")
                            for i, v in enumerate(row))
            print(f"  {key:<30}{th['rms_bond_ma']:>9.1f}{cells}")
        print("  " + "-" * (39 + 9 * len(SPLIT_CUTOFFS)))
        print(f"  {'cases beating theory (of 6)':<30}{'':>9}"
              + "".join(f"{sbeats[c]:>9d}" for c in SPLIT_CUTOFFS))
        results["split_beats_theory"] = {str(c): sbeats[c] for c in SPLIT_CUTOFFS}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written to {OUT}")


if __name__ == "__main__":
    main()
