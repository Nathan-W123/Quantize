"""
2D torsion scan averaging for molecules with two coupled internal rotors.

Computes quantum-weighted averages of effective rotational constants over a
2D (α1, α2) torsion scan using the probability density |ψ(α1,α2)|² from
the 2D Fourier-basis Hamiltonian in torsion_hamiltonian_2d.

Typical usage
-------------
1. Build a TorsionScan2D from QC scan data (geometries, energies, A/B/C).
2. Construct a TorsionHamiltonianSpec2D with fitted F1, F2, V potential.
3. Call average_torsion_scan_2d_quantum() to get Boltzmann-like quantum
   weighted effective rotational constants and representational uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.torsion_hamiltonian_2d import (
    TorsionHamiltonianSpec2D,
    solve_2d_torsion_levels,
)


@dataclass
class TorsionGridPoint2D:
    """Single point on a 2D (α1, α2) torsion scan."""

    phi1: float
    phi2: float
    rotational_constants: np.ndarray | None = None  # (3,) [A, B, C] in cm^-1
    energy: float | None = None
    sigma_abc: np.ndarray | None = None             # (3,) per-point uncertainty
    geometry: np.ndarray | None = None              # (N_atoms, 3) in Å
    label: str | None = None


@dataclass
class TorsionScan2D:
    """Ordered collection of grid points spanning a 2D torsion scan."""

    name: str
    grid_points: list[TorsionGridPoint2D]
    angle_unit: str = "degrees"   # "degrees" or "radians"
    energy_unit: str = "cm-1"
    periodic: bool = True
    metadata: dict = field(default_factory=dict)


def average_torsion_scan_2d_quantum(
    scan: TorsionScan2D,
    spec: TorsionHamiltonianSpec2D,
    state_index: int = 0,
) -> dict:
    """
    Quantum-weight average of rotational constants over a 2D torsion scan.

    Each scan point g = (α1_g, α2_g) is weighted by |ψ_{state_index}(α1_g, α2_g)|²
    from the 2D Fourier-basis Hamiltonian, then normalised to sum to 1.

    For scatter-point scans (arbitrary (α1,α2) pairs, not a regular grid) the
    density is evaluated at each pair directly via matrix-vector products —
    O(N1·N2·G) without building a G×G matrix.

    Parameters
    ----------
    scan         : TorsionScan2D
    spec         : TorsionHamiltonianSpec2D
    state_index  : which torsional eigenstate to use (0 = ground state)

    Returns
    -------
    dict with keys:
      averaged_constants  (3,) — weighted average [A, B, C] in cm^-1
      weights             (G,) — normalised density weights
      phi1_radians        (G,) — rotor 1 angles used (radians)
      phi2_radians        (G,) — rotor 2 angles used (radians)
      state_index         int
      sigma_averaged      (3,) — representational scatter uncertainty
      method              'quantum_2d_rotor'
      warnings            list[str]
    """
    if not scan.grid_points:
        raise ValueError("TorsionScan2D has no grid points.")

    G = len(scan.grid_points)
    warnings: list[str] = []

    # ── Parse angles ─────────────────────────────────────────────────────────
    unit = str(scan.angle_unit).strip().lower()
    phi1_raw = np.array([float(gp.phi1) for gp in scan.grid_points])
    phi2_raw = np.array([float(gp.phi2) for gp in scan.grid_points])
    if unit in {"degree", "degrees", "deg"}:
        phi1_rad = np.deg2rad(phi1_raw)
        phi2_rad = np.deg2rad(phi2_raw)
    elif unit in {"radian", "radians", "rad"}:
        phi1_rad = phi1_raw
        phi2_rad = phi2_raw
    else:
        raise ValueError(f"Unsupported angle_unit: {scan.angle_unit!r}")

    # ── Solve 2D Hamiltonian and extract eigenstate ───────────────────────────
    out = solve_2d_torsion_levels(spec, J=0, K=0)
    n_states = out["eigenvectors"].shape[1]
    if state_index >= n_states:
        raise ValueError(f"state_index={state_index} >= n_states={n_states}")
    eigenvec = out["eigenvectors"][:, state_index]  # (N,)

    # ── Evaluate |ψ(α1_g, α2_g)|² at each scan point ────────────────────────
    n1 = int(spec.n_basis_1)
    n2 = int(spec.n_basis_2)
    N1 = 2 * n1 + 1
    N2 = 2 * n2 + 1
    m1_vals = np.arange(-n1, n1 + 1)
    m2_vals = np.arange(-n2, n2 + 1)

    C = eigenvec.astype(complex).reshape(N1, N2)  # (N1, N2) coefficients

    # Phase matrices for the G scan points
    phi1_mat = np.exp(1j * np.outer(m1_vals, phi1_rad))  # (N1, G)
    phi2_mat = np.exp(1j * np.outer(m2_vals, phi2_rad))  # (N2, G)

    # inner[m1, g] = Σ_{m2} C[m1, m2] * exp(i*m2*α2_g)
    inner = C @ phi2_mat  # (N1, G)

    # ψ[g] = 1/(2π) Σ_{m1} exp(i*m1*α1_g) * inner[m1, g]
    psi_pts = np.einsum("mg,mg->g", phi1_mat, inner) / (2.0 * np.pi)  # (G,)
    dens_pts = np.real(np.abs(psi_pts) ** 2)  # (G,), non-negative

    # ── Normalise to weights ──────────────────────────────────────────────────
    w_sum = float(np.sum(dens_pts))
    if w_sum < 1e-30:
        warnings.append("Total probability density near zero; falling back to uniform weights.")
        weights = np.ones(G, dtype=float) / G
    else:
        weights = dens_pts / w_sum

    # ── Collect rotational constants ──────────────────────────────────────────
    abc_list = []
    for i, gp in enumerate(scan.grid_points):
        if gp.rotational_constants is not None:
            abc_list.append(np.asarray(gp.rotational_constants, dtype=float).ravel())
        else:
            abc_list.append(np.zeros(3, dtype=float))
            if not warnings or "rotational_constants" not in warnings[-1]:
                warnings.append(f"Grid point {i} has no rotational_constants; using zeros.")
    C_arr = np.array(abc_list, dtype=float)  # (G, 3)

    # ── Weighted average and representational scatter uncertainty ─────────────
    C_avg = np.sum(weights[:, None] * C_arr, axis=0)
    var_repr = np.sum(weights[:, None] * (C_arr - C_avg[None, :]) ** 2, axis=0)
    sigma_avg = np.sqrt(var_repr)

    return {
        "averaged_constants": C_avg,
        "weights": weights,
        "phi1_radians": phi1_rad,
        "phi2_radians": phi2_rad,
        "state_index": state_index,
        "sigma_averaged": sigma_avg,
        "method": "quantum_2d_rotor",
        "warnings": list(dict.fromkeys(warnings)),
    }
