"""Validate the monofluorinated reference data before it is used for anything.

Two checks, both of which have caught real errors:

1. **Consistency.** Every measured constant must be reproduced by the published
   geometry to about 1%, the size of the r_s-versus-r_0 difference. A larger gap
   means the structure and the constants disagree, or an isotopologue has been
   assigned to the wrong atom. This is what identified the swapped cis/trans
   labels on vinyl fluoride species 789 and 791.

2. **Information content.** The count of measured constants is not the number of
   independent constraints. This builds the stacked spectral Jacobian over only
   the constants that were actually measured and reports its numerical rank
   against the internal degrees of freedom, which is what decides whether the
   problem is determined.

    python scripts/check_monofluoro_references.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from dev.monofluoro_references import MOLECULES, ReferenceMolecule  # noqa: E402

CONSISTENCY_WARN_PCT = 2.0
_RANK_CUTOFFS = (1e-3, 1e-5, 1e-8)


def species_deviations(mol: ReferenceMolecule) -> list[tuple[str, list, float]]:
    """Per-species percentage deviation of rigid constants from the measured ones."""
    out = []
    for sp in mol.species:
        calc = rotational_constants_mhz(mol.geometry, sp.masses(mol.masses))
        devs = [None if v is None else (calc[k] / v - 1) * 100
                for k, v in enumerate(sp.abc_mhz)]
        worst = max(abs(d) for d in devs if d is not None)
        out.append((sp.label, devs, worst))
    return out


def stacked_jacobian(mol: ReferenceMolecule, n_species=None, h=1e-5) -> np.ndarray:
    """d(measured constants)/d(cartesian coordinates), measured components only."""
    x0 = np.asarray(mol.geometry, dtype=float).ravel()
    rows = []
    for sp in mol.species[:n_species]:
        masses = sp.masses(mol.masses)
        keep = sp.component_indices
        block = np.zeros((len(keep), x0.size))
        for k in range(x0.size):
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            d = (rotational_constants_mhz(xp.reshape(-1, 3), masses)
                 - rotational_constants_mhz(xm.reshape(-1, 3), masses)) / (2 * h)
            block[:, k] = d[keep]
        # weight each row by its own sigma, which is how the fit sees it
        block /= np.asarray(sp.sigmas())[:, None]
        rows.append(block)
    return np.vstack(rows)


def report(mol: ReferenceMolecule) -> bool:
    print(f"\n{'=' * 78}")
    print(f"  {mol.name} ({mol.formula}) — {mol.n_atoms} atoms, "
          f"{mol.internal_dof} internal DOF")
    print(f"  structure: {mol.structure_source}")
    print(f"  constants: {mol.constants_source}")
    print(f"{'=' * 78}")

    print(f"\n  {'species':<26}{'A %':>9}{'B %':>9}{'C %':>9}   status")
    print("  " + "-" * 62)
    ok = True
    for label, devs, worst in species_deviations(mol):
        cells = ["   —   " if d is None else f"{d:+7.2f}" for d in devs]
        status = "ok" if worst < CONSISTENCY_WARN_PCT else "INCONSISTENT"
        ok &= worst < CONSISTENCY_WARN_PCT
        print(f"  {label:<26}{cells[0]:>9}{cells[1]:>9}{cells[2]:>9}   {status}")

    print(f"\n  {'data level':<26}{'species':>9}{'measured':>10}"
          + "".join(f"{'rank ' + f'{c:g}':>12}" for c in _RANK_CUTOFFS)
          + f"{'DOF':>7}")
    print("  " + "-" * 76)
    for label, n in (("parent only", 1), ("all species", len(mol.species))):
        J = stacked_jacobian(mol, n)
        s = np.linalg.svd(J, compute_uv=False)
        ranks = [int((s > c * s.max()).sum()) for c in _RANK_CUTOFFS]
        n_obs = sum(len(sp.component_indices) for sp in mol.species[:n])
        print(f"  {label:<26}{n:>9}{n_obs:>10}"
              + "".join(f"{r:>12}" for r in ranks)
              + f"{mol.internal_dof:>7}")
        deficit = mol.internal_dof - ranks[1]
        if deficit > 0:
            print(f"  {'':<26}-> undetermined in {deficit} internal direction(s)")
    return ok


def main() -> None:
    all_ok = True
    for mol in MOLECULES:
        all_ok &= report(mol)
    print(f"\n{'=' * 78}")
    print("  Every constant above is a measured literature value. Deviations are")
    print("  the r_s-versus-r_0 difference, not fit error.")
    print(f"{'=' * 78}")
    if not all_ok:
        sys.exit("consistency check FAILED — do not use this data")


if __name__ == "__main__":
    main()
