"""
Two-rotor RAM-lite torsion Hamiltonian in a 2D Fourier basis.

Handles molecules with two coupled internal rotors (e.g., dimethyl ether,
acetic acid). The basis is the direct product |m1, m2⟩ with m1 ∈ [-n1, n1]
and m2 ∈ [-n2, n2], giving a matrix of size (2n1+1)(2n2+1).

H = F1*(m1-ρ1*K)² + F2*(m2-ρ2*K)² + F12*(m1-ρ1*K)*(m2-ρ2*K) + V(α1, α2)

The 2D potential uses a double Fourier series in (α1, α2):
  V(α1,α2) = v0 + Σ_{(n1,n2)} vcos[(n1,n2)] * cos(n1*α1 + n2*α2)

Separable potentials V1(α1) + V2(α2) use keys (n,0) and (0,n) respectively.
Coupling terms use keys (n1,n2) with both nonzero.

Flat basis ordering: index i = (m1 + n_basis_1) * N2 + (m2 + n_basis_2),
where N2 = 2*n_basis_2 + 1. m1 is the slow (row) index, m2 the fast (column)
index — consistent with numpy meshgrid indexing='ij'.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TorsionFourierPotential2D:
    """
    Two-dimensional torsional Fourier potential.

    V(α1, α2) = v0 + Σ_{(n1,n2)} vcos[(n1,n2)] * cos(n1*α1 + n2*α2)

    Keys of vcos are (n1, n2) integer tuples (n1 indexes rotor 1, n2 rotor 2).
    Separable potentials use (n,0) for V1(α1) and (0,n) for V2(α2) terms.
    Coupling terms have both n1 and n2 nonzero.
    """

    v0: float = 0.0
    vcos: dict = field(default_factory=dict)  # {(n1, n2): coeff}
    units: str = "cm-1"


@dataclass
class TorsionHamiltonianSpec2D:
    """
    Specification for the 2D two-rotor RAM-lite torsion Hamiltonian.

    H = F1*(m1-ρ1*K)² + F2*(m2-ρ2*K)² + F12*(m1-ρ1*K)*(m2-ρ2*K) + V(α1,α2)

    A symmetric-top rotational energy E_rot = B*J(J+1) + (A-B)*K² is added to
    every diagonal element when J or K is nonzero.
    """

    F1: float
    F2: float
    rho1: float
    rho2: float
    potential: TorsionFourierPotential2D
    n_basis_1: int = 10
    n_basis_2: int = 10
    F12: float = 0.0
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    units: str = "cm-1"


def build_2d_torsion_hamiltonian(
    spec: TorsionHamiltonianSpec2D,
    J: int = 0,
    K: int = 0,
) -> np.ndarray:
    """
    Build the 2D two-rotor RAM-lite Hamiltonian matrix.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec2D
    J    : total angular momentum quantum number
    K    : projection quantum number

    Returns
    -------
    H : real symmetric (N, N) array, N = (2*n_basis_1+1) * (2*n_basis_2+1)
    """
    n1 = int(spec.n_basis_1)
    n2 = int(spec.n_basis_2)
    N1 = 2 * n1 + 1
    N2 = 2 * n2 + 1
    N = N1 * N2

    m1_vals = np.arange(-n1, n1 + 1)  # (N1,)
    m2_vals = np.arange(-n2, n2 + 1)  # (N2,)

    # Flat index i = (m1+n1)*N2 + (m2+n2), m1 slow, m2 fast (indexing='ij')
    M1, M2 = np.meshgrid(m1_vals, m2_vals, indexing="ij")
    M1_flat = M1.ravel()  # (N,)
    M2_flat = M2.ravel()  # (N,)

    H = np.zeros((N, N), dtype=float)

    # ── Kinetic diagonal ─────────────────────────────────────────────────────
    x1 = M1_flat - float(spec.rho1) * float(K)
    x2 = M2_flat - float(spec.rho2) * float(K)
    diag = (
        float(spec.F1) * x1 ** 2
        + float(spec.F2) * x2 ** 2
        + float(spec.F12) * x1 * x2
    )
    if J != 0 or K != 0:
        diag += float(spec.B) * J * (J + 1) + (float(spec.A) - float(spec.B)) * K ** 2
    np.fill_diagonal(H, diag)

    # ── Constant potential offset ────────────────────────────────────────────
    v0 = float(spec.potential.v0)
    if v0 != 0.0:
        H += v0 * np.eye(N)

    # ── Fourier potential terms ──────────────────────────────────────────────
    # ⟨m1',m2'| cos(n1_k*α1 + n2_k*α2) |m1,m2⟩
    #   = (δ_{m1',m1+n1_k}·δ_{m2',m2+n2_k}
    #      + δ_{m1',m1-n1_k}·δ_{m2',m2-n2_k}) / 2
    # For each column index i=(m1,m2) add coeff/2 at the two coupled rows.
    for (n1_key, n2_key), coeff in spec.potential.vcos.items():
        n1_k = int(n1_key)
        n2_k = int(n2_key)
        c = float(coeff)
        if c == 0.0:
            continue
        if n1_k == 0 and n2_k == 0:
            H += c * np.eye(N)
            continue
        for sign in (+1, -1):
            m1p = M1_flat + sign * n1_k
            m2p = M2_flat + sign * n2_k
            valid = (np.abs(m1p) <= n1) & (np.abs(m2p) <= n2)
            if not np.any(valid):
                continue
            cols = np.where(valid)[0]
            rows = (m1p[valid] + n1) * N2 + (m2p[valid] + n2)
            np.add.at(H, (rows, cols), c / 2.0)

    return H


def solve_2d_torsion_levels(
    spec: TorsionHamiltonianSpec2D,
    J: int = 0,
    K: int = 0,
    n_levels: int | None = None,
) -> dict:
    """
    Diagonalize the 2D torsion Hamiltonian and return energy levels.

    Returns
    -------
    dict with keys:
      energies_cm-1  : (n_levels,) eigenvalues in ascending order
      eigenvectors   : (N, n_levels) eigenvectors (columns)
      n_basis_1, n_basis_2, J, K
    """
    H = build_2d_torsion_hamiltonian(spec, J=J, K=K)
    vals, vecs = np.linalg.eigh(H)
    N = H.shape[0]
    k = min(int(n_levels), N) if n_levels is not None else N
    return {
        "energies_cm-1": vals[:k],
        "eigenvectors": vecs[:, :k],
        "n_basis_1": int(spec.n_basis_1),
        "n_basis_2": int(spec.n_basis_2),
        "J": J,
        "K": K,
    }


def torsion_probability_density_2d(
    eigenvec: np.ndarray,
    n_basis_1: int,
    n_basis_2: int,
    alpha1_grid: np.ndarray,
    alpha2_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate 2D probability density |ψ(α1,α2)|² on a rectangular grid.

    Uses the |m1,m2⟩ basis with flat m1-major ordering.

    Parameters
    ----------
    eigenvec   : (N,) eigenvector coefficients (m1-major flat order)
    n_basis_1  : half-range for rotor 1 (N1 = 2*n_basis_1+1)
    n_basis_2  : half-range for rotor 2 (N2 = 2*n_basis_2+1)
    alpha1_grid: (G1,) α1 values in radians
    alpha2_grid: (G2,) α2 values in radians

    Returns
    -------
    density : (G1, G2) non-negative real array
              normalised so ∫∫|ψ|² dα1 dα2 ≈ 1 on a uniform grid covering
              [0, 2π) × [0, 2π).

    Note: for scatter-point evaluation at G paired (α1[g], α2[g]) samples,
    compute np.diag(torsion_probability_density_2d(v, n1, n2, α1, α2))
    only when α1 and α2 are the same 1-D arrays; otherwise use the inline
    computation in average_torsion_scan_2d_quantum.
    """
    n1 = int(n_basis_1)
    n2 = int(n_basis_2)
    N1 = 2 * n1 + 1
    N2 = 2 * n2 + 1
    m1_vals = np.arange(-n1, n1 + 1)
    m2_vals = np.arange(-n2, n2 + 1)

    C = np.asarray(eigenvec, dtype=complex).reshape(N1, N2)
    A1 = np.asarray(alpha1_grid, dtype=float)
    A2 = np.asarray(alpha2_grid, dtype=float)

    # ψ(α1,α2) = 1/(2π) Σ_{m1,m2} C[m1,m2] * exp(i*m1*α1) * exp(i*m2*α2)
    # Exploit separability of the exponential:
    #   inner[m1, g2] = Σ_{m2} C[m1,m2] * exp(i*m2*α2[g2])   (N1, G2)
    #   ψ[g1, g2]     = Σ_{m1} exp(i*m1*α1[g1]) * inner[m1,g2] / (2π)
    phi2 = np.exp(1j * np.outer(m2_vals, A2))  # (N2, G2)
    inner = C @ phi2                             # (N1, G2)
    phi1 = np.exp(1j * np.outer(m1_vals, A1))  # (N1, G1)
    psi = phi1.T @ inner / (2.0 * np.pi)        # (G1, G2)
    return np.real(np.abs(psi) ** 2)
