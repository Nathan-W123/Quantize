"""Geometries for the 3D viewer, and the comparison between them.

Two geometries matter to someone looking at a run: the one it STARTED from --
whatever SMILES, PubChem, a bond list or typed coordinates produced -- and the
one the fit landed on. Showing both is the only way to see what the spectra
actually did, as opposed to what the quantum chemistry already knew.

The starting geometry comes from run_generic._build_geometry, the same function
the runner uses, so the preview is the structure that would really be used
rather than a second guess at it. The fitted one is read back from the run
directory's exports/final_geometry.csv.

Comparing them needs care: a fit is free to translate and rotate the molecule,
and those move every atom without changing the structure at all. So the two are
superposed before anything is measured -- centred, then rotated onto each other
by the Kabsch algorithm -- or a rigid rotation would read as a large change in
every coordinate and the bond and angle differences would be meaningless.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from paths import ensure_repo_paths

ensure_repo_paths(Path(__file__).resolve().parent.parent)

#: Covalent radii in Angstrom, for deciding which atoms are bonded when the
#: geometry source did not say. Rounded; a bond test only needs to separate
#: bonded from non-bonded, and the 1.3 tolerance below does the real work.
_COVALENT = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Br": 1.20, "I": 1.39,
}
#: Element colours, following the convention every chemist already reads.
_COLOUR = {
    "H": "#e8e8e8", "C": "#404040", "N": "#2f5fd0", "O": "#d03a2f",
    "F": "#4fbf6f", "Cl": "#3fa83f", "Br": "#9a4a1e", "I": "#6a2fa0",
    "S": "#d0b020", "P": "#d08020", "Si": "#b0a090",
}


def element_style(symbol: str) -> dict[str, Any]:
    sym = str(symbol).strip().capitalize()
    return {"radius": _COVALENT.get(sym, 0.75), "colour": _COLOUR.get(sym, "#9a7fb0")}


def detect_bonds(coords, elems, tolerance: float = 1.3) -> list[list[int]]:
    """Pairs closer than ``tolerance`` times the sum of their covalent radii."""
    xyz = np.asarray(coords, dtype=float)
    out: list[list[int]] = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            cut = tolerance * (_COVALENT.get(str(elems[i]).capitalize(), 0.75)
                               + _COVALENT.get(str(elems[j]).capitalize(), 0.75))
            if float(np.linalg.norm(xyz[i] - xyz[j])) <= cut:
                out.append([i, j])
    return out


def kabsch(mobile, target):
    """Rotation putting ``mobile`` onto ``target``, both already centred.

    Without this a fit that merely rotated the molecule would look like it had
    moved every atom, and the per-atom and per-bond differences below would be
    reporting the orientation rather than the structure.
    """
    a = np.asarray(mobile, dtype=float)
    b = np.asarray(target, dtype=float)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    # Guard against the reflection SVD will happily hand back: a mirror image
    # is not a rotation, and accepting one would silently superpose a molecule
    # onto its enantiomer.
    corr = np.diag([1.0, 1.0, d])
    return vt.T @ corr @ u.T


def superpose(mobile, target):
    """Centre both and rotate the first onto the second. Returns (moved, rmsd)."""
    a = np.asarray(mobile, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    moved = a @ kabsch(a, b).T
    rmsd = float(np.sqrt(((moved - b) ** 2).sum(axis=1).mean()))
    return moved, rmsd


def _angle_deg(xyz, i, j, k) -> float:
    v1, v2 = xyz[i] - xyz[j], xyz[k] - xyz[j]
    c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.degrees(math.acos(max(-1.0, min(1.0, c)))))


def compare(before, after, elems, bonds) -> dict[str, Any]:
    """What the fit changed, in internal coordinates rather than Cartesians.

    Bond lengths and angles are what a structure IS; Cartesian displacements
    also contain the molecule's position and orientation, which carry no
    structural information at all.
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    rows = []
    for i, j in bonds:
        r0 = float(np.linalg.norm(b[i] - b[j]))
        r1 = float(np.linalg.norm(a[i] - a[j]))
        rows.append({"kind": "bond",
                     "label": f"{elems[i]}{i + 1}–{elems[j]}{j + 1}",
                     "before": r0, "after": r1, "delta": r1 - r0, "unit": "Å"})
    # Angles at every atom with two or more bonds to it.
    nbr: dict[int, list[int]] = {}
    for i, j in bonds:
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    for centre, ns in sorted(nbr.items()):
        for x in range(len(ns)):
            for y in range(x + 1, len(ns)):
                i, k = ns[x], ns[y]
                t0, t1 = _angle_deg(b, i, centre, k), _angle_deg(a, i, centre, k)
                rows.append({"kind": "angle",
                             "label": f"{elems[i]}{i + 1}–{elems[centre]}{centre + 1}–{elems[k]}{k + 1}",
                             "before": t0, "after": t1, "delta": t1 - t0, "unit": "°"})
    return {"rows": rows}


def starting_geometry(cfg: dict) -> dict[str, Any]:
    """The structure a run would actually start from, via the runner's own code."""
    from runner.run_generic import _build_geometry

    coords, elems, bonds = _build_geometry(cfg)
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords.mean(axis=0)
    if not bonds:
        bonds = detect_bonds(coords, elems)
    return {
        "elements": list(elems),
        "coords": [[float(x) for x in row] for row in coords],
        "bonds": [list(map(int, b)) for b in bonds],
        "styles": [element_style(e) for e in elems],
    }


def fitted_geometry(run_dir: str | Path) -> dict[str, Any] | None:
    """Read exports/final_geometry.csv back, or None if the run wrote none."""
    path = Path(run_dir) / "exports" / "final_geometry.csv"
    if not path.is_file():
        return None
    elems, coords = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            elems.append(str(row["element"]).strip())
            coords.append([float(row["x_angstrom"]), float(row["y_angstrom"]),
                           float(row["z_angstrom"])])
    if not elems:
        return None
    return {"elements": elems, "coords": coords}


def before_and_after(cfg: dict, run_dir: str | Path | None) -> dict[str, Any]:
    """Both structures, superposed, with what changed between them."""
    start = starting_geometry(cfg)
    out: dict[str, Any] = {"before": start, "after": None, "comparison": None,
                           "rmsd": None}
    fitted = fitted_geometry(run_dir) if run_dir else None
    if fitted is None or len(fitted["elements"]) != len(start["elements"]):
        return out
    moved, rmsd = superpose(fitted["coords"], start["coords"])
    centred_before = np.asarray(start["coords"], dtype=float)
    out["after"] = {"elements": fitted["elements"],
                    "coords": [[float(x) for x in row] for row in moved]}
    out["rmsd"] = rmsd
    out["comparison"] = compare(centred_before, moved, start["elements"],
                                start["bonds"])
    return out


#: The exports a finished run leaves behind, and what each one feeds. Reading
#: the CSVs rather than holding results in memory means a panel can be filled
#: for a run from any earlier session, and that the UI shows exactly the
#: numbers that were written to disk rather than a parallel copy of them.
_RESULT_EXPORTS = {
    "corrections": "rovib_corrections.csv",
    "residuals": "residuals.csv",
    "uncertainty": "internal_uncertainty.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str):
    """A float, or None for a blank or a NaN -- both mean 'not determined'."""
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def run_results(run_dir: str | Path) -> dict[str, Any]:
    """The three result panels, read back from a run directory's exports."""
    exports = Path(run_dir) / "exports"

    corrections = [{
        "iso": r.get("isotopologue", ""),
        "comp": r.get("component", ""),
        "b0": _f(r, "B0_exp_MHz"),
        "delta": _f(r, "delta_total_MHz"),
        "be": _f(r, "Be_target_MHz"),
        "sigma": _f(r, "sigma_Be_target_MHz"),
        "source": r.get("source", ""),
    } for r in _read_csv(exports / _RESULT_EXPORTS["corrections"])]

    residuals = []
    for r in _read_csv(exports / _RESULT_EXPORTS["residuals"]):
        resid, sigma = _f(r, "residual_mhz"), _f(r, "sigma_mhz")
        residuals.append({
            "iso": r.get("isotopologue", ""),
            "comp": r.get("component", ""),
            "resid": resid,
            "sigma": sigma,
            # Residual in units of its own sigma is the only form that compares
            # across components: A runs to hundreds of GHz and C to tens, so
            # raw MHz says nothing about whether a fit is good.
            "pull": (resid / sigma) if (resid is not None and sigma) else None,
        })

    uncertainty = [{
        "name": r.get("name", ""),
        "value": _f(r, "value"),
        "std_err": _f(r, "std_err"),
        "unit": (r.get("value_unit") or "").strip(),
        "dominance": r.get("prior_dominance", ""),
        "sensitivity": r.get("prior_sensitivity", ""),
    } for r in _read_csv(exports / _RESULT_EXPORTS["uncertainty"])]

    pulls = [abs(x["pull"]) for x in residuals if x["pull"] is not None]
    return {
        "corrections": corrections,
        "residuals": residuals,
        "uncertainty": uncertainty,
        # rms pull is the headline number: about 1 means the fit agrees with
        # the data to within the uncertainty it was given.
        "rms_pull": (float(np.sqrt(np.mean(np.square(pulls)))) if pulls else None),
        "max_pull": (max(pulls) if pulls else None),
    }
