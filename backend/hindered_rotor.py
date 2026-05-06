"""
First-pass 1D hindered-rotor utilities for torsional averaging.

This module builds and solves a simplified 1D hindered-rotor Hamiltonian
for an internal torsion coordinate phi and returns probability weights that
can be used to average rotational constants over a torsion scan.

Important limitation:
- This is not a full torsion-rotation Hamiltonian.
- It does not model full rovibrational/tunneling splitting structure.
- It is intended as an additive approximation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Unit conversions to cm^-1
_HARTREE_TO_CM1 = 219474.6313705
_KCAL_PER_MOL_TO_CM1 = 349.7550874793
_KJ_PER_MOL_TO_CM1 = 83.5934722514

# k_B in Hartree/K, used in Boltzmann fallback
_K_B_HARTREE_PER_K = 3.166811563e-6


@dataclass
class HinderedRotorModel:
    name: str
    symmetry_number: int = 1
    rotational_constant_F: float | None = None
    rotational_constant_unit: str = "cm-1"
    fourier_terms: dict[int, float] = field(default_factory=dict)
    potential_energy_unit: str = "cm-1"
    basis_size: int = 41
    warnings: list[str] = field(default_factory=list)


def model_diagnostics(model: HinderedRotorModel) -> list[str]:
    """Return non-fatal diagnostic warnings for hindered-rotor settings."""
    warnings = list(model.warnings or [])
    if int(model.symmetry_number) < 1:
        warnings.append("symmetry_number < 1 is invalid; expected positive integer.")
    if model.rotational_constant_F is not None:
        F_cm1 = convert_energy_to_cm1(model.rotational_constant_F, model.rotational_constant_unit)
        if F_cm1 <= 0.0:
            warnings.append("rotational_constant_F <= 0 is unusual; verify kinetic parameter.")
    terms = {int(k): float(v) for k, v in (model.fourier_terms or {}).items() if int(k) > 0}
    if int(model.symmetry_number) > 1 and terms:
        bad = [n for n in terms if n % int(model.symmetry_number) != 0]
        if bad:
            warnings.append(
                "fourier_terms include harmonics inconsistent with symmetry_number; "
                "verify torsion symmetry assumptions."
            )
    n_basis = int(model.basis_size)
    if n_basis < 11:
        warnings.append("very small basis_size may under-resolve barrier coupling.")
    M = max((n_basis - 1) // 2, 0)
    max_n = max(terms.keys(), default=0)
    if max_n > 0 and M < max_n:
        warnings.append("basis_size is likely too small relative to highest Fourier harmonic.")
    return warnings


def convert_energy_to_cm1(value, unit: str) -> float:
    """Convert scalar energy value to cm^-1."""
    u = str(unit).strip().lower()
    v = float(value)
    if u == "cm-1":
        return v
    if u == "kcal/mol":
        return v * _KCAL_PER_MOL_TO_CM1
    if u == "kj/mol":
        return v * _KJ_PER_MOL_TO_CM1
    if u == "hartree":
        return v * _HARTREE_TO_CM1
    raise ValueError(
        f"Unsupported energy unit '{unit}'. Allowed: cm-1, kcal/mol, kj/mol, hartree."
    )


def build_fourier_potential(phi_rad, fourier_terms: dict[int, float], unit: str = "cm-1") -> np.ndarray:
    """
    Build V(phi) in cm^-1 from terms:

        V(phi) = sum_n (V_n / 2) * [1 - cos(n phi)]
    """
    phi = np.asarray(phi_rad, dtype=float)
    V = np.zeros_like(phi, dtype=float)
    for n_raw, amp_raw in (fourier_terms or {}).items():
        n = int(n_raw)
        if n <= 0:
            continue
        v_n_cm1 = convert_energy_to_cm1(amp_raw, unit)
        V += 0.5 * v_n_cm1 * (1.0 - np.cos(n * phi))
    return V


def build_hindered_rotor_hamiltonian(model: HinderedRotorModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Fourier-basis Hamiltonian for a 1D hindered rotor:

        H = -F d^2/dphi^2 + V(phi)

    In |m> basis (m=-M..M):
      T_mm = F * m^2
      V_diag += sum_n V_n/2
      V_{m,m±n} += -V_n/4
    """
    if model.rotational_constant_F is None:
        raise ValueError("rotational_constant_F is required for quantum hindered-rotor weights.")

    n_basis = int(model.basis_size)
    if n_basis < 3:
        raise ValueError("basis_size must be >= 3.")
    if n_basis % 2 == 0:
        raise ValueError("basis_size must be odd.")

    M = (n_basis - 1) // 2
    m_values = np.arange(-M, M + 1, dtype=int)

    F_cm1 = convert_energy_to_cm1(model.rotational_constant_F, model.rotational_constant_unit)
    H = np.diag(F_cm1 * (m_values.astype(float) ** 2)).astype(float)

    terms = {int(k): convert_energy_to_cm1(v, model.potential_energy_unit) for k, v in (model.fourier_terms or {}).items() if int(k) > 0}

    if terms:
        const_diag = 0.5 * float(sum(terms.values()))
        H += const_diag * np.eye(n_basis, dtype=float)

    idx = {int(m): i for i, m in enumerate(m_values)}
    for n, v_n_cm1 in terms.items():
        coup = -0.25 * float(v_n_cm1)
        for i, m in enumerate(m_values):
            j_plus = idx.get(int(m + n))
            j_minus = idx.get(int(m - n))
            if j_plus is not None:
                H[i, j_plus] += coup
            if j_minus is not None:
                H[i, j_minus] += coup

    H = 0.5 * (H + H.T)
    return H, m_values


def solve_hindered_rotor(model: HinderedRotorModel) -> dict:
    """Solve the 1D hindered-rotor Hamiltonian with numpy.linalg.eigh."""
    H, m_values = build_hindered_rotor_hamiltonian(model)
    energies, eigvecs = np.linalg.eigh(H)
    warnings = model_diagnostics(model)
    return {
        "energies_cm1": np.asarray(energies, dtype=float),
        "eigenvectors": np.asarray(eigvecs, dtype=complex),
        "m_values": np.asarray(m_values, dtype=int),
        "hamiltonian": np.asarray(H, dtype=float),
        "warnings": warnings,
    }


def basis_convergence_report(
    model: HinderedRotorModel,
    *,
    basis_sizes: list[int] | None = None,
    state_index: int = 0,
) -> dict:
    """
    Compare target-state energy across basis sizes as a convergence diagnostic.
    """
    sizes = basis_sizes or [max(3, int(model.basis_size) - 10), int(model.basis_size), int(model.basis_size) + 10]
    energies = []
    used_sizes = []
    warnings: list[str] = []
    for n in sizes:
        if int(n) < 3 or int(n) % 2 == 0:
            warnings.append(f"Skipping invalid basis size {n}; require odd >=3.")
            continue
        tmp = HinderedRotorModel(
            name=model.name,
            symmetry_number=model.symmetry_number,
            rotational_constant_F=model.rotational_constant_F,
            rotational_constant_unit=model.rotational_constant_unit,
            fourier_terms=dict(model.fourier_terms),
            potential_energy_unit=model.potential_energy_unit,
            basis_size=int(n),
            warnings=list(model.warnings),
        )
        out = solve_hindered_rotor(tmp)
        e = np.asarray(out["energies_cm1"], dtype=float)
        if state_index < 0 or state_index >= e.size:
            warnings.append(f"state_index {state_index} out of range for basis {n}.")
            continue
        used_sizes.append(int(n))
        energies.append(float(e[int(state_index)]))
    if len(energies) < 2:
        return {
            "basis_sizes": np.asarray(used_sizes, dtype=int),
            "energies_cm1": np.asarray(energies, dtype=float),
            "delta_last_cm1": float("nan"),
            "warnings": warnings + ["insufficient valid basis sizes for convergence estimate."],
        }
    delta_last = abs(float(energies[-1] - energies[-2]))
    if delta_last > 1e-2:
        warnings.append("basis convergence appears loose for selected state; consider larger basis_size.")
    return {
        "basis_sizes": np.asarray(used_sizes, dtype=int),
        "energies_cm1": np.asarray(energies, dtype=float),
        "delta_last_cm1": float(delta_last),
        "warnings": warnings,
    }


def torsional_probability_on_grid(
    model: HinderedRotorModel,
    phi_grid,
    state_index: int = 0,
) -> dict:
    """
    Evaluate normalized |psi(phi)|^2 weights on a real-space torsion grid.

    Returns dict with keys: weights, probabilities, phi_grid, state_index, warnings.
    """
    phi = np.asarray(phi_grid, dtype=float).ravel()
    if phi.size < 3:
        raise ValueError("phi_grid must contain at least 3 points.")

    solved = solve_hindered_rotor(model)
    eigvecs = np.asarray(solved["eigenvectors"], dtype=complex)
    m_values = np.asarray(solved["m_values"], dtype=int)

    if state_index < 0 or state_index >= eigvecs.shape[1]:
        raise ValueError("state_index out of range for solved hindered-rotor basis.")

    coeff = eigvecs[:, int(state_index)]
    phase = np.exp(1j * np.outer(phi, m_values))
    psi = (phase @ coeff) / np.sqrt(2.0 * np.pi)
    prob = np.abs(psi) ** 2

    if np.any(~np.isfinite(prob)):
        raise ValueError("Computed torsional probabilities are non-finite.")
    s = float(np.sum(prob))
    if s <= 0.0:
        raise ValueError("Computed torsional probabilities sum to zero.")

    weights = prob / s
    warnings = list(solved.get("warnings", []))
    if phi.size < 7:
        warnings.append("torsion probability grid is small; weights may be under-resolved.")

    return {
        "weights": np.asarray(weights, dtype=float),
        "probabilities": np.asarray(prob, dtype=float),
        "phi_grid": phi,
        "state_index": int(state_index),
        "warnings": warnings,
        "energies_cm1": np.asarray(solved["energies_cm1"], dtype=float),
    }


def boltzmann_torsion_weights(energies, temperature_K: float, energy_unit: str = "hartree") -> np.ndarray:
    """Compute normalized Boltzmann weights from grid energies."""
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be > 0.")

    e = np.asarray(energies, dtype=float).ravel()
    if e.size == 0:
        raise ValueError("No energies provided.")
    if np.any(~np.isfinite(e)):
        raise ValueError("Energies must be finite for Boltzmann weights.")

    unit = str(energy_unit).strip().lower()
    if unit == "hartree":
        e_h = e
    elif unit == "cm-1":
        e_h = e / _HARTREE_TO_CM1
    elif unit == "kcal/mol":
        e_h = e / 627.5094740631
    elif unit == "kj/mol":
        e_h = e / 2625.4996394799
    else:
        raise ValueError(
            f"Unsupported energy unit '{energy_unit}'. Allowed: hartree, cm-1, kcal/mol, kj/mol."
        )

    e_rel = e_h - np.min(e_h)
    beta = 1.0 / (_K_B_HARTREE_PER_K * float(temperature_K))
    raw = np.exp(-beta * e_rel)
    total = float(np.sum(raw))
    if total <= 0.0:
        raise ValueError("Boltzmann weights are numerically zero.")
    return raw / total
