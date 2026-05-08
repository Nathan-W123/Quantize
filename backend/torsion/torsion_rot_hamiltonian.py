"""
Full torsion-rotation Hamiltonian extending RAM-lite.

Implements a coupled torsion-rotation Hamiltonian in the direct-product
basis |J, K, m> (symmetric-top projection K, Fourier torsion index m).

Hamiltonian (Watson A-reduction, prolate convention z = a-axis):

  H = H_rot + F*(m - rho*K)^2 + V(alpha) + H_asym + H_cd

  H_rot   = (B+C)/2 * J(J+1) + [A - (B+C)/2] * K^2
  H_asym  : ΔK=±2,  (B-C)/4 * f(J,K)   where f = sqrt[J(J+1)-K(K+1)][J(J+1)-(K+1)(K+2)]
  H_cd    : Watson A-reduction quartic centrifugal distortion
              diagonal: -DJ*J(J+1)^2 - DJK*J(J+1)*K^2 - DK*K^4
              ΔK=±2:   (-d1*J(J+1) - d2*(K^2+(K±2)^2)/4) * f(J,K)

Alpha-dependent constants (optional):
  When spec.A_alpha / B_alpha / C_alpha are provided as Fourier series,
  the diagonal rotational block and the ΔK=±2 coupling become full
  n_m×n_m Fourier matrices, yielding complete torsion-rotation K-mixing.

RAM-lite (single-K block) remains available via ``solve_ram_lite_levels``
in torsion_hamiltonian.py and is the recommended fast path for most tasks.

Notes
-----
- Full-matrix size: (2J+1) * (2*n_basis+1).  For J=5, n_basis=10 this is
  11*21 = 231 x 231 — still fast with numpy.linalg.eigh.
- Ka/Kc assignment is approximate (dominant-K heuristic), sufficient for
  labeling low-J levels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.torsion_hamiltonian import (
    TorsionHamiltonianSpec,
    _validate_units,
    basis_m_values,
    effective_torsion_constant_matrix,
    fourier_potential_matrix,
)


def _jk_asym_coupling(J: int, K: int) -> float:
    """
    Asymmetric-top off-diagonal factor for delta_K = +2 transition.

    <J,K+2|H_asym|J,K> = (B-C)/4 * _jk_asym_coupling(J, K)

    sqrt([J(J+1)-K(K+1)] * [J(J+1)-(K+1)(K+2)])
    """
    jj = int(J) * (int(J) + 1)
    k = int(K)
    t1 = jj - k * (k + 1)
    t2 = jj - (k + 1) * (k + 2)
    if t1 <= 0 or t2 <= 0:
        return 0.0
    return float(np.sqrt(t1 * t2))


def build_full_torsion_rotation_hamiltonian(
    spec: TorsionHamiltonianSpec,
    J: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build the full torsion-rotation Hamiltonian in |J, K, m> direct-product basis.

    Basis ordering: for each K in [-J, ..., J], for each m in m_vals.
    Total dimension = (2J+1) * (2*n_basis+1).

    Hamiltonian includes:
      H = H_rot(J,K) + F*(m - rho*K)^2 + V(alpha) + H_asym(J,K,K+/-2)
          + H_cd  (Watson A-reduction quartic centrifugal distortion)

    Centrifugal distortion (Watson A-reduction):
      Diagonal:      -DJ*J(J+1)^2 - DJK*J(J+1)*K^2 - DK*K^4
      Off-diag ΔK=2: [-d1*J(J+1) - d2*(K^2+(K+2)^2)/4] * f(J,K)

    When spec.A_alpha / B_alpha / C_alpha are set (alpha-dependent constants),
    the diagonal rotational blocks and the K±2 off-diagonal coupling become
    full n_m×n_m Fourier matrices, enabling complete torsion-K mixing.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec — A, B, C, F, rho, DJ/DJK/DK/d1/d2,
           potential, n_basis, and optional A_alpha/B_alpha/C_alpha
    J : rotational quantum number (>= 0)

    Returns
    -------
    H : complex ndarray (dim, dim) — Hermitian Hamiltonian matrix
    K_vals : int ndarray (2J+1,) — K quantum numbers in basis order
    m_vals : int ndarray (2*n_basis+1,) — m quantum numbers
    warnings : list[str]
    """
    _validate_units(spec.units)
    J_int = max(0, int(J))

    m_vals = basis_m_values(spec.n_basis)
    K_vals = np.arange(-J_int, J_int + 1, dtype=int)
    n_m = m_vals.size
    n_K = K_vals.size
    dim = n_K * n_m
    warns: list[str] = list(spec.warnings or [])

    A = float(spec.A)
    B = float(spec.B)
    C = float(spec.C)
    rho = float(spec.rho)
    F = float(spec.F)
    DJ = float(spec.DJ)
    DJK = float(spec.DJK)
    DK = float(spec.DK)
    d1 = float(spec.d1)
    d2 = float(spec.d2)

    jj = float(J_int * (J_int + 1))  # J(J+1)

    V_mat = fourier_potential_matrix(m_vals, spec.potential)  # (n_m, n_m)

    # Build Fourier matrices for alpha-dependent constants when provided.
    # These replace scalar A/B/C with full n_m×n_m matrices in the Hamiltonian.
    use_alpha = (spec.A_alpha is not None or
                 spec.B_alpha is not None or
                 spec.C_alpha is not None)
    if use_alpha:
        A_mat = (effective_torsion_constant_matrix(m_vals, spec.A_alpha)
                 if spec.A_alpha is not None else A * np.eye(n_m, dtype=complex))
        B_mat = (effective_torsion_constant_matrix(m_vals, spec.B_alpha)
                 if spec.B_alpha is not None else B * np.eye(n_m, dtype=complex))
        C_mat = (effective_torsion_constant_matrix(m_vals, spec.C_alpha)
                 if spec.C_alpha is not None else C * np.eye(n_m, dtype=complex))
        BpC_half = (B_mat + C_mat) * 0.5   # (B(α)+C(α))/2 matrix
        BmC_qtr = (B_mat - C_mat) * 0.25   # (B(α)-C(α))/4 matrix
        warns.append(
            "Alpha-dependent rotational constants active: full K-torsion mixing in off-diagonal blocks."
        )

    H = np.zeros((dim, dim), dtype=complex)

    for ki, K in enumerate(K_vals):
        k = int(K)
        rs = ki * n_m
        re = rs + n_m

        # Watson A-reduction CD diagonal correction (scalar, same for all m)
        E_cd = -DJ * jj * jj - DJK * jj * k * k - DK * k ** 4

        # Kinetic F*(m - rho*K)^2 — diagonal in m
        x = m_vals - rho * k
        kin_diag = np.diag((F * x ** 2 + E_cd).astype(complex))

        if use_alpha:
            # Rotational energy as full n_m×n_m matrix
            rot_mat = BpC_half * jj + (A_mat - BpC_half) * float(k * k)
            H[rs:re, rs:re] += rot_mat.astype(complex) + kin_diag + V_mat
        else:
            E_rot_k = (B + C) / 2.0 * jj + (A - (B + C) / 2.0) * k * k
            H[rs:re, rs:re] += np.diag(np.full(n_m, E_rot_k, dtype=complex)) + kin_diag + V_mat

        # Off-diagonal K±2 coupling
        ki2 = ki + 2
        if ki2 < n_K:
            k2 = int(K_vals[ki2])   # = k + 2
            cs = ki2 * n_m
            ce = cs + n_m
            asym_elem = _jk_asym_coupling(J_int, k)
            if abs(asym_elem) > 1e-15:
                # CD corrections to K±2 (scalar regardless of alpha-dep mode)
                d1_corr = -d1 * jj * asym_elem
                d2_corr = -d2 * (k * k + k2 * k2) / 4.0 * asym_elem

                if use_alpha:
                    # Full matrix: (B(α)-C(α))/4 mixes torsion states
                    cpl_block = (BmC_qtr * asym_elem
                                 + (d1_corr + d2_corr) * np.eye(n_m)).astype(complex)
                else:
                    asym_scalar = (B - C) / 4.0 * asym_elem + d1_corr + d2_corr
                    cpl_block = (asym_scalar * np.eye(n_m, dtype=complex))

                H[rs:re, cs:ce] += cpl_block
                H[cs:ce, rs:re] += cpl_block.conj().T

    H = 0.5 * (H + H.conj().T)
    return H, K_vals, m_vals, warns


def _dominant_K_assignment(
    eigvecs: np.ndarray,
    K_vals: np.ndarray,
    n_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Ka/Kc labels from dominant K-block weight in each eigenvector.

    For prolate convention (z = a-axis): dominant |K| ~ Ka.
    Kc ~ J - Ka (asymmetric top identity for lowest levels).
    """
    J = (K_vals.size - 1) // 2
    n_levels = eigvecs.shape[1]
    Ka_arr = np.zeros(n_levels, dtype=int)
    Kc_arr = np.zeros(n_levels, dtype=int)

    for lv in range(n_levels):
        v = eigvecs[:, lv]
        K_weight = np.array([
            float(np.sum(np.abs(v[ki * n_m:(ki + 1) * n_m]) ** 2))
            for ki in range(K_vals.size)
        ])
        best_ki = int(np.argmax(K_weight))
        Ka = abs(int(K_vals[best_ki]))
        Ka_arr[lv] = Ka
        Kc_arr[lv] = max(0, J - Ka)

    return Ka_arr, Kc_arr


def solve_full_torsion_rotation_levels(
    spec: TorsionHamiltonianSpec,
    J: int,
    *,
    n_levels: Optional[int] = None,
) -> dict:
    """
    Diagonalize the full torsion-rotation Hamiltonian for quantum number J.

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    J : rotational quantum number
    n_levels : number of lowest levels to return (None = all)

    Returns
    -------
    dict with:
      J : int
      energies_cm-1 : (n_levels,) float ndarray — sorted eigenvalues
      Ka_labels : (n_levels,) int ndarray — dominant Ka (approximate)
      Kc_labels : (n_levels,) int ndarray — dominant Kc (approximate)
      eigenvectors : (dim, n_levels) complex ndarray
      K_vals : int ndarray — K quantum numbers in basis order
      m_vals : int ndarray — m quantum numbers
      dim : int — total Hamiltonian dimension
      warnings : list[str]
    """
    H, K_vals, m_vals, warns = build_full_torsion_rotation_hamiltonian(spec, J)
    e_all, U_all = np.linalg.eigh(H)
    e_all = np.real_if_close(e_all).astype(float)

    Ka_arr, Kc_arr = _dominant_K_assignment(U_all, K_vals, m_vals.size)

    n = e_all.size if n_levels is None else min(int(n_levels), e_all.size)
    return {
        "J": int(J),
        "energies_cm-1": e_all[:n],
        "Ka_labels": Ka_arr[:n],
        "Kc_labels": Kc_arr[:n],
        "eigenvectors": U_all[:, :n],
        "K_vals": K_vals,
        "m_vals": m_vals,
        "dim": int(H.shape[0]),
        "warnings": warns + [
            "Full torsion-rotation: asymmetric-top K-coupling included."
        ],
    }


def compare_ram_lite_vs_full(
    spec: TorsionHamiltonianSpec,
    J: int,
    K: int,
    *,
    n_levels: int = 5,
) -> dict:
    """
    Compare RAM-lite single-K block energies vs full coupled solution.

    Useful for assessing the importance of K-coupling for a given molecule.

    Returns
    -------
    dict with:
      ram_lite_energies_cm-1 : from solve_ram_lite_levels
      full_energies_cm-1     : from solve_full_torsion_rotation_levels
      max_diff_cm-1          : max absolute difference over n_levels
      rms_diff_cm-1          : RMS difference
    """
    from backend.torsion_hamiltonian import solve_ram_lite_levels

    rl = solve_ram_lite_levels(spec, J=J, K=K, n_levels=n_levels)
    full = solve_full_torsion_rotation_levels(spec, J, n_levels=n_levels)

    e_rl = rl["energies_cm-1"]
    e_full = full["energies_cm-1"]
    n = min(len(e_rl), len(e_full), n_levels)

    diff = np.abs(e_rl[:n] - e_full[:n])
    return {
        "J": J,
        "K": K,
        "n_compared": n,
        "ram_lite_energies_cm-1": e_rl[:n],
        "full_energies_cm-1": e_full[:n],
        "max_diff_cm-1": float(np.max(diff)) if n > 0 else float("nan"),
        "rms_diff_cm-1": float(np.sqrt(np.mean(diff ** 2))) if n > 0 else float("nan"),
    }
