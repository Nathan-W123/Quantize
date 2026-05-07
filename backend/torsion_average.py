"""
First-pass torsional averaging for effective rotational constants.

This module computes effective motion-averaged A/B/C constants from a supplied
1D torsion scan using either:
- quantum hindered-rotor weights, or
- Boltzmann weights.

Important limitation:
- This is not a full torsion-rotation Hamiltonian treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.hindered_rotor import boltzmann_torsion_weights
from backend.spectral import _rotational_constants
from backend.torsion_hamiltonian import (
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
    torsion_probability_density,
)


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


@dataclass
class TorsionGridPoint:
    phi: float
    geometry: np.ndarray
    energy: float | None = None
    rotational_constants: np.ndarray | None = None
    sigma_abc: np.ndarray | None = None  # (3,) uncertainty on [A,B,C] in cm^-1
    weight: float | None = None
    label: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TorsionScan:
    name: str
    atoms: tuple[int, int, int, int] | None
    grid_points: list[TorsionGridPoint]
    angle_unit: str = "degrees"
    energy_unit: str = "hartree"
    periodic: bool = True
    metadata: dict = field(default_factory=dict)


def ensure_phi_radians(scan: TorsionScan) -> np.ndarray:
    """Return torsion grid values in radians."""
    phi = np.asarray([float(gp.phi) for gp in scan.grid_points], dtype=float)
    unit = str(scan.angle_unit).strip().lower()
    if unit in {"radian", "radians", "rad"}:
        return phi
    if unit in {"degree", "degrees", "deg"}:
        return np.deg2rad(phi)
    raise ValueError("Unsupported angle_unit. Allowed: degrees or radians.")


def _masses_from_elements(elements: list[str]) -> np.ndarray:
    masses = []
    for e in elements:
        key = str(e).strip()
        if key not in _DEFAULT_ELEMENT_MASS:
            raise ValueError(f"No default mass for element '{key}'. Provide explicit masses.")
        masses.append(_DEFAULT_ELEMENT_MASS[key])
    return np.asarray(masses, dtype=float)


def get_or_compute_grid_rotational_constants(elements, scan: TorsionScan, masses=None) -> np.ndarray:
    """
    Return grid rotational constants array shape (G,3).

    Uses prefilled `grid_point.rotational_constants` when present; otherwise
    computes from geometry with backend.spectral rotational constants.
    """
    if masses is None:
        m = _masses_from_elements(list(elements))
    else:
        m = np.asarray(masses, dtype=float).ravel()

    out = []
    for gp in scan.grid_points:
        if gp.rotational_constants is not None:
            abc = np.asarray(gp.rotational_constants, dtype=float).ravel()
            if abc.size != 3:
                raise ValueError("Prefilled rotational_constants must have length 3.")
            out.append(abc)
            continue

        xyz = np.asarray(gp.geometry, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Each torsion grid geometry must have shape (N,3).")
        if xyz.shape[0] != m.size:
            raise ValueError("masses length must match number of atoms in grid geometries.")
        out.append(np.asarray(_rotational_constants(xyz, m), dtype=float))

    return np.asarray(out, dtype=float)


def get_grid_sigma_abc(scan: TorsionScan) -> np.ndarray | None:
    """
    Extract per-grid-point rotational constant uncertainties from scan.

    Returns (G,3) float array when any grid point has sigma_abc set,
    or None when no uncertainties are provided (triggers heuristic fallback).
    """
    sigmas = []
    any_set = False
    for gp in scan.grid_points:
        if gp.sigma_abc is not None:
            s = np.asarray(gp.sigma_abc, dtype=float).ravel()
            if s.size != 3:
                raise ValueError("TorsionGridPoint.sigma_abc must have length 3 (A, B, C).")
            sigmas.append(s)
            any_set = True
        else:
            sigmas.append(np.zeros(3, dtype=float))
    return np.asarray(sigmas, dtype=float) if any_set else None


def propagate_averaging_uncertainty(
    constants_grid: np.ndarray,
    weights: np.ndarray,
    averaged: np.ndarray,
    sigma_grid: np.ndarray | None,
) -> dict:
    """
    Propagate grid-point uncertainties through weighted averaging.

    Two independent contributions:
      sigma_statistical    = sqrt(Σ_g w_g² σ_g²)   — from grid-point measurement errors
      sigma_representational = sqrt(Σ_g w_g (C_g - C_avg)²) — physical spread of constants

    Both are combined in quadrature as sigma_total.  When sigma_grid is None only the
    representational term is computed (sigma_statistical = 0).

    Returns
    -------
    dict with keys sigma_statistical, sigma_representational, sigma_total — each (3,) array.
    """
    C = np.asarray(constants_grid, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    avg = np.asarray(averaged, dtype=float)

    var_repr = np.sum(w[:, None] * (C - avg[None, :]) ** 2, axis=0)

    if sigma_grid is not None:
        sg = np.asarray(sigma_grid, dtype=float)
        var_stat = np.sum((w[:, None] ** 2) * (sg ** 2), axis=0)
    else:
        var_stat = np.zeros(3, dtype=float)

    return {
        "sigma_statistical": np.sqrt(var_stat),
        "sigma_representational": np.sqrt(var_repr),
        "sigma_total": np.sqrt(var_stat + var_repr),
    }


def average_rotational_constants_with_weights(constants_grid, weights) -> np.ndarray:
    """Weighted average of constants grid (G,3) with normalized weights (G,)."""
    C = np.asarray(constants_grid, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("constants_grid must have shape (G,3).")
    if w.size != C.shape[0]:
        raise ValueError("weights length must match number of grid points.")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative.")
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights sum must be > 0.")
    w = w / s
    return np.sum(w[:, None] * C, axis=0)


def _scan_warnings(
    scan: TorsionScan,
    spec: TorsionHamiltonianSpec | None = None,
    symmetry_number: int = 1,
) -> list[str]:
    warnings: list[str] = []
    G = len(scan.grid_points)
    if G < 5:
        warnings.append("fewer than 5 torsion grid points; averaging may be unreliable.")

    phi = ensure_phi_radians(scan)
    if scan.periodic:
        span = float(np.max(phi) - np.min(phi)) if phi.size > 1 else 0.0
        if span < 1.7 * np.pi:
            warnings.append("periodic torsion grid does not cover close to 0..2pi range.")

    if spec is not None:
        if int(symmetry_number) > 1:
            vcos_keys = [int(n) for n in spec.potential.vcos.keys() if int(n) > 0]
            bad_terms = [n for n in vcos_keys if n % int(symmetry_number) != 0]
            if bad_terms:
                warnings.append(
                    "symmetry_number > 1 but Fourier terms include non-multiple harmonics; "
                    "verify torsion symmetry assumptions."
                )
        max_n = max((int(n) for n in spec.potential.vcos.keys() if int(n) > 0), default=0)
        if max_n > 0 and int(spec.n_basis) < max_n:
            warnings.append("n_basis may be too small relative to highest Fourier term.")

    if not scan.periodic and scan.atoms is not None:
        warnings.append("torsion scan is marked nonperiodic; torsional averaging interpretation may be limited.")

    return warnings


def _prob_weights_from_spec(
    spec: TorsionHamiltonianSpec,
    phi_radians: np.ndarray,
    state_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized probability weights on the scan phi grid for one eigenstate.

    Returns (weights, energies_cm1) where weights sum to 1.
    """
    result = solve_ram_lite_levels(spec, J=0, K=0)
    eigvecs = result["eigenvectors"]
    m_vals = result["m_values"]
    e_cm1 = result["energies_cm-1"]
    if state_index < 0 or state_index >= eigvecs.shape[1]:
        raise ValueError(f"state_index {state_index} out of range for basis size {eigvecs.shape[1]}.")
    psi = eigvecs[:, int(state_index)]
    p = torsion_probability_density(psi, phi_radians, m_vals)
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError("Computed torsional probabilities sum to zero.")
    return p / s, e_cm1


def average_torsion_scan_quantum(
    elements,
    scan: TorsionScan,
    spec: TorsionHamiltonianSpec,
    masses=None,
    state_index: int = 0,
    symmetry_number: int = 1,
) -> dict:
    """Average A/B/C using quantum torsion-Hamiltonian weights from eigenstate `state_index`."""
    phi = ensure_phi_radians(scan)
    C = get_or_compute_grid_rotational_constants(elements, scan, masses=masses)

    warnings = _scan_warnings(scan, spec=spec, symmetry_number=symmetry_number)
    w, e_cm1 = _prob_weights_from_spec(spec, phi, state_index=state_index)

    if float(np.max(w)) > 0.95:
        warnings.append("torsional weights collapse to one grid point; correction may be fragile.")

    avg = average_rotational_constants_with_weights(C, w)
    sigma_grid = get_grid_sigma_abc(scan)
    unc = propagate_averaging_uncertainty(C, w, avg, sigma_grid)
    return {
        "averaged_constants": np.asarray(avg, dtype=float),
        "sigma_averaged": unc["sigma_total"],
        "uncertainty_breakdown": unc,
        "grid_constants": np.asarray(C, dtype=float),
        "weights": np.asarray(w, dtype=float),
        "phi_radians": np.asarray(phi, dtype=float),
        "state_index": int(state_index),
        "torsional_energies_cm1": np.asarray(e_cm1, dtype=float),
        "warnings": list(dict.fromkeys(warnings)),
        "method": "quantum_hindered_rotor",
    }


def average_torsion_scan_quantum_thermal(
    elements,
    scan: TorsionScan,
    spec: TorsionHamiltonianSpec,
    masses=None,
    temperature_K: float = 298.15,
    max_states: int = 6,
    symmetry_number: int = 1,
) -> dict:
    """
    Thermal quantum averaging over multiple torsional eigenstates.

    Uses state populations from torsion-Hamiltonian eigen-energies and combines
    per-state probability weights on the supplied phi grid.
    """
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be > 0.")
    phi = ensure_phi_radians(scan)
    C = get_or_compute_grid_rotational_constants(elements, scan, masses=masses)
    warnings = _scan_warnings(scan, spec=spec, symmetry_number=symmetry_number)

    result = solve_ram_lite_levels(spec, J=0, K=0)
    e_cm1 = np.asarray(result["energies_cm-1"], dtype=float)
    eigvecs = result["eigenvectors"]
    m_vals = result["m_values"]
    if e_cm1.size == 0:
        raise ValueError("No torsional eigen-energies available.")
    n_states = min(max(1, int(max_states)), int(e_cm1.size))
    e_sel = e_cm1[:n_states]
    # Boltzmann populations — energy offset cancels
    e_h = e_sel / 219474.6313705
    beta = 1.0 / (3.166811563e-6 * float(temperature_K))
    pop_raw = np.exp(-beta * (e_h - np.min(e_h)))
    pops = pop_raw / np.sum(pop_raw)

    w_total = np.zeros(phi.size, dtype=float)
    for i in range(n_states):
        psi_i = eigvecs[:, i]
        p_i = torsion_probability_density(psi_i, phi, m_vals)
        s_i = float(np.sum(p_i))
        if s_i > 0:
            p_i = p_i / s_i
        w_total += float(pops[i]) * p_i

    w_total = w_total / np.sum(w_total)
    if float(np.max(w_total)) > 0.95:
        warnings.append("thermal quantum weights collapse strongly to one grid point.")
    avg = average_rotational_constants_with_weights(C, w_total)
    sigma_grid = get_grid_sigma_abc(scan)
    unc = propagate_averaging_uncertainty(C, w_total, avg, sigma_grid)
    return {
        "averaged_constants": np.asarray(avg, dtype=float),
        "sigma_averaged": unc["sigma_total"],
        "uncertainty_breakdown": unc,
        "grid_constants": np.asarray(C, dtype=float),
        "weights": np.asarray(w_total, dtype=float),
        "phi_radians": np.asarray(phi, dtype=float),
        "state_populations": np.asarray(pops, dtype=float),
        "states_used": int(n_states),
        "torsional_energies_cm1": np.asarray(e_cm1, dtype=float),
        "warnings": list(dict.fromkeys(warnings)),
        "method": "quantum_hindered_rotor_thermal",
    }


def average_torsion_scan_boltzmann(
    elements,
    scan: TorsionScan,
    masses=None,
    temperature_K: float = 298.15,
) -> dict:
    """Average A/B/C using Boltzmann weights from grid energies."""
    phi = ensure_phi_radians(scan)
    C = get_or_compute_grid_rotational_constants(elements, scan, masses=masses)

    energies = [gp.energy for gp in scan.grid_points]
    if any(e is None for e in energies):
        raise ValueError("Boltzmann torsion averaging requires energy on every grid point.")

    w = boltzmann_torsion_weights(
        np.asarray(energies, dtype=float),
        temperature_K=float(temperature_K),
        energy_unit=scan.energy_unit,
    )

    warnings = _scan_warnings(scan, spec=None)
    if float(np.max(w)) > 0.95:
        warnings.append("Boltzmann weights collapse to one grid point; correction may be fragile.")

    avg = average_rotational_constants_with_weights(C, w)
    sigma_grid = get_grid_sigma_abc(scan)
    unc = propagate_averaging_uncertainty(C, w, avg, sigma_grid)
    return {
        "averaged_constants": np.asarray(avg, dtype=float),
        "sigma_averaged": unc["sigma_total"],
        "uncertainty_breakdown": unc,
        "grid_constants": np.asarray(C, dtype=float),
        "weights": np.asarray(w, dtype=float),
        "phi_radians": np.asarray(phi, dtype=float),
        "state_index": 0,
        "torsional_energies_cm1": np.array([], dtype=float),
        "warnings": list(dict.fromkeys(warnings)),
        "method": "boltzmann_torsion_average",
    }


def torsional_motion_correction(
    elements,
    reference_geometry,
    scan: TorsionScan,
    spec: TorsionHamiltonianSpec | None = None,
    masses=None,
    mode: str = "quantum",
    temperature_K: float = 298.15,
) -> dict:
    """Return effective torsional correction: delta_constants = averaged - reference."""
    if masses is None:
        m = _masses_from_elements(list(elements))
    else:
        m = np.asarray(masses, dtype=float).ravel()

    xyz_ref = np.asarray(reference_geometry, dtype=float)
    if xyz_ref.ndim != 2 or xyz_ref.shape[1] != 3:
        raise ValueError("reference_geometry must have shape (N,3).")
    if xyz_ref.shape[0] != m.size:
        raise ValueError("masses length must match reference geometry atom count.")

    ref = np.asarray(_rotational_constants(xyz_ref, m), dtype=float)

    mode_key = str(mode).strip().lower()
    if mode_key == "quantum":
        if spec is None:
            raise ValueError("quantum mode requires a TorsionHamiltonianSpec.")
        avg_out = average_torsion_scan_quantum(elements, scan, spec, masses=masses, state_index=0)
    elif mode_key == "boltzmann":
        avg_out = average_torsion_scan_boltzmann(
            elements, scan, masses=masses, temperature_K=float(temperature_K),
        )
    else:
        raise ValueError("mode must be 'quantum' or 'boltzmann'.")

    avg = np.asarray(avg_out["averaged_constants"], dtype=float)
    delta = avg - ref
    return {
        "reference_constants": ref,
        "averaged_constants": avg,
        "delta_constants": delta,
        "weights": np.asarray(avg_out["weights"], dtype=float),
        "grid_constants": np.asarray(avg_out["grid_constants"], dtype=float),
        "warnings": list(avg_out.get("warnings", [])),
        "mode": mode_key,
    }
