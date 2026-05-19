"""
First-pass Large-Amplitude Motion averaging utilities.

This module provides an additive geometry-plus-dynamics correction layer for
non-rigid motion (torsion, inversion, puckering, floppy coordinates) by
averaging rotational constants over a supplied coordinate grid:

    <B_i> = integral |psi(q)|^2 B_i(q) dq

or discrete approximation:

    <B_i> ~= sum_g w_g B_i(q_g)

Important: this is an effective averaging model and NOT a full
torsion-rotation Hamiltonian treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.spectral import _rotational_constants

# --- constants / unit conversions -------------------------------------------------

# Boltzmann constant in Hartree / K
K_B_HARTREE_PER_K = 3.166811563e-6

KCAL_PER_MOL_TO_HARTREE = 1.0 / 627.5094740631
KJ_PER_MOL_TO_HARTREE = 1.0 / 2625.4996394799
CM_INV_TO_HARTREE = 1.0 / 219474.6313705


@dataclass
class LargeAmplitudeGridPoint:
    coordinate_value: float
    geometry: np.ndarray
    energy: Optional[float] = None
    weight: Optional[float] = None
    label: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LargeAmplitudeModel:
    name: str
    coordinate_type: str
    grid_points: list[LargeAmplitudeGridPoint]
    weighting: str = "boltzmann"  # boltzmann | user
    temperature_K: float = 298.15
    units: str = "hartree"  # hartree | kcal/mol | kj/mol | cm-1
    warnings: list[str] = field(default_factory=list)


# Limited default isotopic-average masses (amu) for common elements.
_DEFAULT_ELEMENT_MASS = {
    "H": 1.00794,
    "D": 2.0141017781,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.998403163,
    "P": 30.973761998,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
    "Si": 28.0855,
}


def convert_energy_to_hartree(value: float, units: str) -> float:
    u = str(units).strip().lower()
    v = float(value)
    if u == "hartree":
        return v
    if u == "kcal/mol":
        return v * KCAL_PER_MOL_TO_HARTREE
    if u == "kj/mol":
        return v * KJ_PER_MOL_TO_HARTREE
    if u == "cm-1":
        return v * CM_INV_TO_HARTREE
    raise ValueError(
        f"Unknown energy units '{units}'. Allowed: hartree, kcal/mol, kj/mol, cm-1."
    )


def normalize_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("No weights supplied.")
    if np.any(~np.isfinite(w)):
        raise ValueError("Weights must be finite.")
    if np.any(w < 0.0):
        raise ValueError("Weights must be non-negative.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("All weights are zero; cannot normalize.")
    return w / total


def boltzmann_weights(energies, temperature_K: float, units: str = "hartree") -> np.ndarray:
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be > 0 for Boltzmann weighting.")
    e = np.asarray(energies, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("No energies supplied for Boltzmann weighting.")
    if np.any(~np.isfinite(e)):
        raise ValueError("Energies must be finite for Boltzmann weighting.")
    e_h = np.array([convert_energy_to_hartree(x, units) for x in e], dtype=float)
    e_rel = e_h - np.min(e_h)
    beta = 1.0 / (K_B_HARTREE_PER_K * float(temperature_K))
    raw = np.exp(-beta * e_rel)
    return normalize_weights(raw)


def _masses_from_elements(elements: list[str]) -> np.ndarray:
    masses = []
    for e in elements:
        key = str(e).strip()
        if key not in _DEFAULT_ELEMENT_MASS:
            raise ValueError(
                f"No default mass for element '{key}'. Provide explicit masses."
            )
        masses.append(_DEFAULT_ELEMENT_MASS[key])
    return np.asarray(masses, dtype=float)


def rotational_constants_for_geometry(elements, geometry, masses=None) -> np.ndarray:
    """
    Compute [A,B,C] rotational constants (MHz) for one geometry.

    Reuses backend.spectral rotational-constant machinery.
    """
    xyz = np.asarray(geometry, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("geometry must have shape (N, 3).")
    if masses is None:
        m = _masses_from_elements(list(elements))
    else:
        m = np.asarray(masses, dtype=float).ravel()
    if m.size != xyz.shape[0]:
        raise ValueError("masses length must match number of atoms in geometry.")
    return np.asarray(_rotational_constants(xyz, m), dtype=float)


def _torsion_range_warning(qvals: np.ndarray) -> Optional[str]:
    if qvals.size < 2:
        return "torsion grid has too few points to assess periodic coverage."
    span = float(np.max(qvals) - np.min(qvals))
    # Heuristic: less than ~70% of 2*pi likely not representative.
    if span < 0.7 * 2.0 * np.pi:
        return (
            "torsion grid span is limited; periodic averaging may be incomplete."
        )
    return None


def average_rotational_constants(elements, model: LargeAmplitudeModel, masses=None) -> dict:
    if not model.grid_points:
        raise ValueError("LargeAmplitudeModel has no grid_points.")

    warnings = list(model.warnings or [])
    if len(model.grid_points) < 3:
        warnings.append("fewer than 3 grid points; motion average may be unreliable.")

    qvals = np.array([float(g.coordinate_value) for g in model.grid_points], dtype=float)
    geoms = [np.asarray(g.geometry, dtype=float) for g in model.grid_points]
    gconst = np.array(
        [rotational_constants_for_geometry(elements, g, masses=masses) for g in geoms],
        dtype=float,
    )

    weighting = str(model.weighting).strip().lower()
    if weighting == "boltzmann":
        energies = [gp.energy for gp in model.grid_points]
        if any(e is None for e in energies):
            raise ValueError("Boltzmann weighting requires energy at every grid point.")
        w = boltzmann_weights(energies, model.temperature_K, units=model.units)
    elif weighting == "user":
        user_w = [gp.weight for gp in model.grid_points]
        if any(v is None for v in user_w):
            raise ValueError("User weighting selected but one or more weights are missing.")
        w = normalize_weights(user_w)
    else:
        raise ValueError("weighting must be one of: boltzmann, user")

    ctype = str(model.coordinate_type).strip().lower()
    if ctype == "torsion":
        tw = _torsion_range_warning(qvals)
        if tw:
            warnings.append(tw)
    if ctype in {"floppy", "custom"}:
        warnings.append(
            "coordinate_type is model-dependent; interpretation is approximate."
        )

    # Weight-collapse warning.
    if float(np.max(w)) > 0.95:
        warnings.append(
            "weights collapse strongly to one grid point; effective motion correction may be negligible/unstable."
        )

    # Energy spread warning for Boltzmann model.
    if weighting == "boltzmann":
        e_h = np.array([convert_energy_to_hartree(gp.energy, model.units) for gp in model.grid_points], dtype=float)
        spread = float(np.max(e_h) - np.min(e_h))
        if spread > 0.05:  # ~31 kcal/mol
            warnings.append(
                "energy spread across grid is very large; population may be dominated by minima."
            )

    averaged = np.sum(w[:, None] * gconst, axis=0)
    return {
        "averaged_constants": np.asarray(averaged, dtype=float),
        "grid_constants": gconst,
        "weights": np.asarray(w, dtype=float),
        "coordinate_values": qvals,
        "warnings": warnings,
    }


def effective_motion_correction(elements, reference_geometry, model: LargeAmplitudeModel, masses=None) -> dict:
    ref = rotational_constants_for_geometry(elements, reference_geometry, masses=masses)
    avg = average_rotational_constants(elements, model, masses=masses)
    delta = np.asarray(avg["averaged_constants"], dtype=float) - ref
    return {
        "reference_constants": ref,
        "averaged_constants": np.asarray(avg["averaged_constants"], dtype=float),
        "delta": np.asarray(delta, dtype=float),
        "weights": np.asarray(avg["weights"], dtype=float),
        "grid_constants": np.asarray(avg["grid_constants"], dtype=float),
        "coordinate_values": np.asarray(avg["coordinate_values"], dtype=float),
        "warnings": list(avg["warnings"]),
    }


def build_torsion_scan_model(
    name: str,
    coordinate_values,
    geometries,
    energies=None,
    temperature_K: float = 298.15,
    units: str = "hartree",
) -> LargeAmplitudeModel:
    """
    Build a torsion scan model from precomputed geometry/energy arrays.

    This helper does not run quantum scans; callers must supply geometries and
    optional energies/weights.
    """
    q = np.asarray(coordinate_values, dtype=float).ravel()
    if len(q) != len(geometries):
        raise ValueError("coordinate_values and geometries must have equal length.")
    if energies is not None and len(energies) != len(geometries):
        raise ValueError("energies and geometries must have equal length.")

    gps: list[LargeAmplitudeGridPoint] = []
    for i, qv in enumerate(q):
        gps.append(
            LargeAmplitudeGridPoint(
                coordinate_value=float(qv),
                geometry=np.asarray(geometries[i], dtype=float),
                energy=(None if energies is None else float(energies[i])),
                label=f"grid_{i}",
            )
        )
    return LargeAmplitudeModel(
        name=name,
        coordinate_type="torsion",
        grid_points=gps,
        weighting=("boltzmann" if energies is not None else "user"),
        temperature_K=float(temperature_K),
        units=units,
    )

