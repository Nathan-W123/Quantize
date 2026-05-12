from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from backend.geometryguess import _relax_geometry
from backend.quantum import _detect_bonds, _detect_dihedrals
from backend.spectral import _rotational_constants

_KCAL_PER_MOL_PER_K = 0.00198720425864083
_HARTREE_TO_KCAL_PER_MOL = 627.509474
_CM1_TO_KCAL_PER_MOL = 0.002859144
_DEFAULT_ANGLE_GRID_DEG = (60.0, 180.0, 300.0)

_COV_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Si": 1.11,
    "B": 0.84,
}


@dataclass(frozen=True)
class RotatableBond:
    atom_i: int
    atom_j: int
    dihedral_atoms: tuple[int, int, int, int]
    rotating_atoms: tuple[int, ...]


def _unit(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / nrm


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = _unit(np.asarray(axis, dtype=float))
    x, y, z = ax
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _graph_from_bonds(n_atoms: int, bonds: list[tuple[int, int]]) -> dict[int, set[int]]:
    graph = {i: set() for i in range(n_atoms)}
    for i, j in bonds:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    return graph


def _component_after_cut(
    n_atoms: int,
    bonds: list[tuple[int, int]],
    fixed_atom: int,
    rotating_atom: int,
) -> tuple[int, ...]:
    graph = _graph_from_bonds(n_atoms, bonds)
    graph[fixed_atom].discard(rotating_atom)
    graph[rotating_atom].discard(fixed_atom)
    seen = {rotating_atom}
    queue = [rotating_atom]
    while queue:
        node = queue.pop(0)
        for nbr in sorted(graph[node]):
            if nbr in seen:
                continue
            seen.add(nbr)
            queue.append(nbr)
    return tuple(sorted(seen))


def _kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("RMSD comparison requires matching coordinate shapes.")
    a0 = a - np.mean(a, axis=0, keepdims=True)
    b0 = b - np.mean(b, axis=0, keepdims=True)
    cov = a0.T @ b0
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    diff = a0 @ rot - b0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _dihedral_deg(coords: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    i, j, k, l = atoms
    p0 = np.asarray(coords[i], dtype=float)
    p1 = np.asarray(coords[j], dtype=float)
    p2 = np.asarray(coords[k], dtype=float)
    p3 = np.asarray(coords[l], dtype=float)
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1u = _unit(b1)
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1u, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def _wrap_deg(angle_deg: float) -> float:
    out = float(angle_deg)
    while out <= -180.0:
        out += 360.0
    while out > 180.0:
        out -= 360.0
    return out


def rotate_atoms_around_bond(
    coords: np.ndarray,
    atom_i: int,
    atom_j: int,
    rotating_atoms: tuple[int, ...],
    angle_deg: float,
) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).copy()
    axis_origin = xyz[atom_i]
    axis_vec = xyz[atom_j] - xyz[atom_i]
    rot = _rotation_matrix(axis_vec, np.radians(float(angle_deg)))
    for atom in rotating_atoms:
        shifted = xyz[atom] - axis_origin
        xyz[atom] = axis_origin + rot @ shifted
    return xyz


def _nonbonded_pairs(n_atoms: int, bonds: list[tuple[int, int]]) -> set[tuple[int, int]]:
    graph = _graph_from_bonds(n_atoms, bonds)
    excluded: set[tuple[int, int]] = set()
    for i in range(n_atoms):
        for j in graph[i]:
            excluded.add((min(i, j), max(i, j)))
        for j in graph[i]:
            for k in graph[j]:
                if k == i:
                    continue
                excluded.add((min(i, k), max(i, k)))
    pairs: set[tuple[int, int]] = set()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) not in excluded:
                pairs.add((i, j))
    return pairs


def estimate_conformer_energy_kcal_mol(
    coords: np.ndarray,
    elems: list[str],
    bonds: list[tuple[int, int]],
) -> float:
    xyz = np.asarray(coords, dtype=float)
    nonbonded = _nonbonded_pairs(len(xyz), bonds)
    energy = 0.0
    for i, j in nonbonded:
        rij = float(np.linalg.norm(xyz[i] - xyz[j]))
        cutoff = 1.15 * (_COV_RADII.get(elems[i], 0.8) + _COV_RADII.get(elems[j], 0.8))
        if rij < 1e-8:
            energy += 1e6
            continue
        if rij < cutoff:
            x = cutoff / rij
            energy += 12.0 * (x - 1.0) ** 2
        else:
            energy += 0.02 * (cutoff / rij) ** 6
    return float(energy)


def optimize_conformer_geometry(
    coords: np.ndarray,
    elems: list[str],
    bonds: list[tuple[int, int]],
    *,
    n_steps: int = 120,
    dt: float = 0.01,
) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float)
    targets = {(min(i, j), max(i, j)): float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in bonds}
    return _relax_geometry(
        xyz,
        bonds,
        targets,
        n_steps=int(n_steps),
        dt=float(dt),
        k_bond=6.0,
        k_rep=0.02,
        rep_cut=1.35,
    )


def detect_rotatable_bonds(
    coords: np.ndarray,
    elems: list[str],
    bonds: list[tuple[int, int]] | None = None,
    *,
    max_bonds: int | None = None,
    explicit_bonds: list[tuple[int, int]] | None = None,
) -> list[RotatableBond]:
    xyz = np.asarray(coords, dtype=float)
    all_bonds = list(bonds if bonds is not None else _detect_bonds(xyz, elems))
    graph = _graph_from_bonds(len(xyz), all_bonds)
    dihedrals = _detect_dihedrals(all_bonds)
    dihedral_by_bond: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for atoms in dihedrals:
        _, j, k, _ = atoms
        dihedral_by_bond.setdefault((min(j, k), max(j, k)), atoms)

    requested = None
    if explicit_bonds:
        requested = {tuple(sorted((int(i), int(j)))) for i, j in explicit_bonds}

    out: list[RotatableBond] = []
    for i, j in sorted((min(a, b), max(a, b)) for a, b in all_bonds):
        if requested is not None and (i, j) not in requested:
            continue
        if elems[i] == "H" or elems[j] == "H":
            continue
        if len(graph[i]) < 2 or len(graph[j]) < 2:
            continue
        dih = dihedral_by_bond.get((i, j))
        if dih is None:
            continue
        rotating = _component_after_cut(len(xyz), all_bonds, i, j)
        if len(rotating) >= len(xyz) or len(rotating) == 0:
            continue
        out.append(
            RotatableBond(
                atom_i=i,
                atom_j=j,
                dihedral_atoms=dih,
                rotating_atoms=rotating,
            )
        )
    if max_bonds is not None:
        out = out[: max(int(max_bonds), 0)]
    return out


def _prune_candidates(
    candidates: list[dict[str, Any]],
    *,
    prune_rmsd_ang: float,
    prune_max_delta_abc_mhz: float,
    max_conformers: int,
    energy_window_kcal_mol: float | None,
) -> tuple[list[dict[str, Any]], int]:
    if not candidates:
        return [], 0
    ordered = sorted(candidates, key=lambda row: (float(row["energy_kcal_mol"]), str(row["name"])))
    kept: list[dict[str, Any]] = []
    pruned = 0
    e_min = float(ordered[0]["energy_kcal_mol"])
    for cand in ordered:
        if energy_window_kcal_mol is not None and float(cand["energy_kcal_mol"]) > e_min + float(energy_window_kcal_mol):
            pruned += 1
            continue
        duplicate = False
        for ref in kept:
            rmsd = _kabsch_rmsd(cand["coords"], ref["coords"])
            d_abc = float(np.max(np.abs(np.asarray(cand["rotational_constants_mhz"]) - np.asarray(ref["rotational_constants_mhz"]))))
            if rmsd <= float(prune_rmsd_ang) or d_abc <= float(prune_max_delta_abc_mhz):
                duplicate = True
                break
        if duplicate:
            pruned += 1
            continue
        kept.append(cand)
        if len(kept) >= int(max_conformers):
            break
    return kept, pruned + max(0, len(ordered) - len(kept) - pruned)


def _explicit_conformer_defs(
    cfg: dict[str, Any],
    reference_coords: np.ndarray,
    *,
    default_energy_unit: str,
) -> list[dict[str, Any]]:
    rows = cfg.get("conformers") or cfg.get("entries") or []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", f"explicit_{idx + 1}"))
        coords = row.get("coords_angstrom") or row.get("coords")
        offset = row.get("offset_angstrom") or row.get("offset")
        entry: dict[str, Any] = {
            "name": name,
            "weight": float(row.get("weight", 1.0)),
            "energy": float(row.get("energy", 0.0)),
            "energy_unit": str(row.get("energy_unit", default_energy_unit)),
            "source": "explicit",
            "metadata": {"source": "explicit"},
        }
        if coords is not None:
            entry["coords"] = np.asarray(coords, dtype=float)
        elif offset is not None:
            entry["offset"] = np.asarray(offset, dtype=float)
            entry["coords"] = np.asarray(reference_coords, dtype=float) + np.asarray(offset, dtype=float)
        else:
            entry["coords"] = np.asarray(reference_coords, dtype=float).copy()
        out.append(entry)
    return out


def generate_conformer_candidates(
    reference_coords: np.ndarray,
    elems: list[str],
    bonds: list[tuple[int, int]] | None,
    generation_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = generation_cfg or {}
    xyz0 = np.asarray(reference_coords, dtype=float)
    all_bonds = list(bonds if bonds is not None else _detect_bonds(xyz0, elems))
    rotatable = detect_rotatable_bonds(
        xyz0,
        elems,
        all_bonds,
        max_bonds=cfg.get("max_rotatable_bonds"),
        explicit_bonds=cfg.get("rotatable_bonds"),
    )
    angle_grid = tuple(float(v) for v in (cfg.get("angle_grid_deg") or _DEFAULT_ANGLE_GRID_DEG))
    include_input = bool(cfg.get("include_input_geometry", True))
    optimize = bool(cfg.get("optimize", True))
    optimize_steps = int(cfg.get("optimization_steps", 120))
    max_comb = int(cfg.get("max_combination_count", 128))
    energy_window = cfg.get("energy_window_kcal_mol")
    energy_window = None if energy_window is None else float(energy_window)
    prune_rmsd = float(cfg.get("prune_rmsd_ang", 0.15))
    prune_abc = float(cfg.get("prune_constants_mhz", 1.0))
    max_confs = int(cfg.get("max_conformers", 12))

    combos = list(product(angle_grid, repeat=len(rotatable))) if rotatable else [tuple()]
    if len(combos) > max_comb:
        combos = combos[:max_comb]

    candidates: list[dict[str, Any]] = []
    if include_input:
        xyz = optimize_conformer_geometry(xyz0, elems, all_bonds, n_steps=optimize_steps) if optimize else xyz0.copy()
        candidates.append(
            {
                "name": "conf_ref",
                "coords": xyz,
                "energy_kcal_mol": estimate_conformer_energy_kcal_mol(xyz, elems, all_bonds),
                "rotational_constants_mhz": _rotational_constants(xyz, np.ones(len(elems), dtype=float)),
                "source": "reference",
                "metadata": {"source": "reference", "dihedrals_deg": {}},
            }
        )

    for combo_idx, combo in enumerate(combos, start=1):
        if not combo:
            continue
        xyz = xyz0.copy()
        dihedral_meta: dict[str, float] = {}
        for rb, target in zip(rotatable, combo):
            current = _dihedral_deg(xyz, rb.dihedral_atoms)
            delta = _wrap_deg(float(target) - current)
            xyz = rotate_atoms_around_bond(xyz, rb.atom_i, rb.atom_j, rb.rotating_atoms, delta)
            dihedral_meta[f"{rb.dihedral_atoms[0]}-{rb.dihedral_atoms[1]}-{rb.dihedral_atoms[2]}-{rb.dihedral_atoms[3]}"] = float(target)
        if optimize:
            xyz = optimize_conformer_geometry(xyz, elems, all_bonds, n_steps=optimize_steps)
        candidates.append(
            {
                "name": f"conf_gen_{combo_idx:02d}",
                "coords": xyz,
                "energy_kcal_mol": estimate_conformer_energy_kcal_mol(xyz, elems, all_bonds),
                "rotational_constants_mhz": _rotational_constants(xyz, np.ones(len(elems), dtype=float)),
                "source": "generated",
                "metadata": {
                    "source": "generated",
                    "dihedrals_deg": dihedral_meta,
                },
            }
        )

    kept, pruned = _prune_candidates(
        candidates,
        prune_rmsd_ang=prune_rmsd,
        prune_max_delta_abc_mhz=prune_abc,
        max_conformers=max_confs,
        energy_window_kcal_mol=energy_window,
    )
    diag = {
        "rotatable_bonds": [
            {
                "bond": [rb.atom_i, rb.atom_j],
                "dihedral_atoms": list(rb.dihedral_atoms),
                "rotating_atoms": list(rb.rotating_atoms),
            }
            for rb in rotatable
        ],
        "generated_candidates": len(candidates),
        "kept_candidates": len(kept),
        "pruned_candidates": int(pruned),
        "angle_grid_deg": [float(v) for v in angle_grid],
        "warnings": [],
    }
    if not rotatable:
        diag["warnings"].append("No eligible rotatable bonds detected; using the reference conformer only.")
    if len(combos) >= max_comb:
        diag["warnings"].append("Conformer generation truncated by max_combination_count.")
    return kept, diag


def build_conformer_ensemble(
    reference_coords: np.ndarray,
    elems: list[str],
    bonds: list[tuple[int, int]] | None,
    conformer_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = conformer_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    weight_mode = str(cfg.get("weight_mode", "boltzmann")).strip().lower()
    temperature_k = float(cfg.get("temperature_k", 298.15))
    energy_unit = str(cfg.get("energy_unit", "kcal/mol")).strip().lower()

    explicit_defs = _explicit_conformer_defs(
        cfg,
        np.asarray(reference_coords, dtype=float),
        default_energy_unit=energy_unit,
    )
    generation_cfg = cfg.get("generation") if isinstance(cfg.get("generation"), dict) else {}
    generation_enabled = bool(generation_cfg.get("enabled", False))

    generated_defs: list[dict[str, Any]] = []
    generation_diag = {
        "rotatable_bonds": [],
        "generated_candidates": 0,
        "kept_candidates": 0,
        "pruned_candidates": 0,
        "angle_grid_deg": list(_DEFAULT_ANGLE_GRID_DEG),
        "warnings": [],
    }
    if generation_enabled:
        generated_defs, generation_diag = generate_conformer_candidates(
            np.asarray(reference_coords, dtype=float),
            list(elems),
            bonds,
            generation_cfg,
        )

    ensemble_defs: list[dict[str, Any]] = []
    for row in explicit_defs:
        ensemble_defs.append(dict(row))
    for row in generated_defs:
        ensemble_defs.append(
            {
                "name": str(row["name"]),
                "coords": np.asarray(row["coords"], dtype=float),
                "weight": 1.0,
                "energy": float(row["energy_kcal_mol"]),
                "energy_unit": "kcal/mol",
                "source": row["source"],
                "metadata": dict(row.get("metadata") or {}),
            }
        )

    use_mixture = enabled and len(ensemble_defs) > 0
    summary = {
        "enabled": bool(use_mixture),
        "requested_enabled": bool(enabled),
        "weight_mode": weight_mode,
        "temperature_k": float(temperature_k),
        "energy_unit": energy_unit,
        "explicit_count": len(explicit_defs),
        "generated_count": len(generated_defs),
        "n_conformers": len(ensemble_defs),
        "generation": generation_diag,
        "warnings": list(generation_diag.get("warnings", [])),
        "conformers": [
            {
                "name": str(row.get("name", f"conf_{idx+1}")),
                "source": str(row.get("source", "explicit")),
                "energy": float(row.get("energy", 0.0)),
                "energy_unit": str(row.get("energy_unit", "kcal/mol")),
                "weight": float(row.get("weight", 1.0)),
                "metadata": dict(row.get("metadata") or {}),
            }
            for idx, row in enumerate(ensemble_defs)
        ],
    }
    if enabled and not ensemble_defs:
        summary["warnings"].append("Conformer block enabled but produced no conformers; falling back to single-structure mode.")

    return {
        "enabled": bool(use_mixture),
        "weight_mode": weight_mode,
        "temperature_k": float(temperature_k),
        "energy_unit": energy_unit,
        "conformer_defs": ensemble_defs if use_mixture else None,
        "summary": summary,
    }
