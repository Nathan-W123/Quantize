"""
Line intensities and nuclear-spin statistical weights for the RAM torsion model.

Implements:
  - torsion_cos_alpha_matrix: <m'|cos(α)|m> in the |m> Fourier basis
  - torsion_dipole_matrix_elements: |<ψ_hi|cos(α)|ψ_lo>|² for eigenvector pairs
  - honl_london_factor: symmetric-top Hönl-London line strength factors
  - compute_torsion_line_list: complete line list with selection rules and weights

Physics notes
-------------
The torsional part of the transition dipole moment is approximated as proportional
to cos(α), where α is the internal rotation angle.  This is a first-order Fourier
expansion of the angle-dependent dipole component.

  <m'|cos(α)|m> = (δ_{m',m+1} + δ_{m',m-1}) / 2

The total line strength is factored as:
  S_total = S_torsion × S_HL × g_nsw

where:
  S_torsion = |<ψ_hi|cos(α)|ψ_lo>|²   (torsional Franck-Condon-like factor)
  S_HL      = Hönl-London factor        (symmetric-top approximation)
  g_nsw     = nuclear-spin statistical weight

Hönl-London factors use the symmetric-top expressions (Gordy & Cook 1984, Ch. 3)
as an approximation valid for near-symmetric-top molecules.  The 'a'-type
selection rules (ΔK=0) are the default for torsion-rotation transitions where the
torsional axis is approximately collinear with a principal axis.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from backend.torsion_hamiltonian import (
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion_symmetry import nuclear_spin_weight, symmetry_selection_rules

_MHZ_PER_CM1 = 29979.2458


# ── cos(α) matrix in |m> basis ──────────────────────────────────────────────

def torsion_cos_alpha_matrix(m_values: np.ndarray) -> np.ndarray:
    """
    Construct <m'|cos(α)|m> in the Fourier |m> basis.

    cos(α)|m> = (|m+1> + |m-1>) / 2  →  real tridiagonal with 0.5 on ±1 diagonals.

    Parameters
    ----------
    m_values : (N,) integer array of basis m quantum numbers

    Returns
    -------
    (N, N) real array
    """
    m = np.asarray(m_values, dtype=int).ravel()
    N = m.size
    C = np.zeros((N, N), dtype=float)
    m_to_idx = {int(mi): i for i, mi in enumerate(m)}
    for j, mj in enumerate(m):
        for shift in (+1, -1):
            i = m_to_idx.get(int(mj) + shift)
            if i is not None:
                C[i, j] = 0.5
    return C


# ── Torsional matrix elements ────────────────────────────────────────────────

def torsion_dipole_matrix_elements(
    eigvecs_lo: np.ndarray,
    eigvecs_hi: np.ndarray,
    m_values: np.ndarray,
) -> np.ndarray:
    """
    Compute |<ψ_hi|cos(α)|ψ_lo>|² for all (hi, lo) eigenstate pairs.

    Parameters
    ----------
    eigvecs_lo : (N_basis, n_lo) array — eigenvectors of the lower block
    eigvecs_hi : (N_basis, n_hi) array — eigenvectors of the upper block
    m_values : (N_basis,) integer array — basis m quantum numbers

    Returns
    -------
    (n_hi, n_lo) float array of |<hi|cos(α)|lo>|²
    """
    C = torsion_cos_alpha_matrix(m_values)
    U_lo = np.asarray(eigvecs_lo)
    U_hi = np.asarray(eigvecs_hi)
    # Apply cos(α) to each lower eigenstate: (N, n_lo)
    C_lo = C @ U_lo if np.isrealobj(U_lo) else C @ U_lo.real
    # Project onto upper eigenstates: <hi|C|lo> = U_hi† · C_lo, shape (n_hi, n_lo)
    if np.isrealobj(U_hi):
        ME = U_hi.T @ C_lo
    else:
        ME = U_hi.conj().T @ C_lo
    return np.abs(ME) ** 2


# ── Hönl-London factors ─────────────────────────────────────────────────────

def honl_london_factor(
    J_lo: int,
    K_lo: int,
    J_hi: int,
    K_hi: int,
    transition_type: str = "a",
) -> float:
    """
    Hönl-London factor for a symmetric-top transition.

    Supported types:
      'a' : parallel (ΔK = 0)       — torsional axis ≈ a-axis
      'b' : perpendicular (|ΔK| = 1) — b-type selection rules
      'c' : perpendicular (|ΔK| = 1) — c-type (same HL formula as b-type)

    Branch is determined by ΔJ = J_hi - J_lo:
      +1 → R-branch,  0 → Q-branch,  −1 → P-branch.

    Formulas from Gordy & Cook (1984) Microwave Molecular Spectra, Ch. 3.

    Returns
    -------
    float ≥ 0.  Returns 0 if the selection rule for this type is violated.
    """
    Jl, Kl, Jh, Kh = int(J_lo), int(K_lo), int(J_hi), int(K_hi)
    dJ = Jh - Jl
    dK = Kh - Kl
    t = str(transition_type).strip().lower()

    if t == "a":
        if dK != 0:
            return 0.0
        K = abs(Kl)
        if dJ == +1:
            J = Jl
            return float((J + 1) ** 2 - K ** 2) / float(J + 1) if J + 1 > 0 else 0.0
        if dJ == 0:
            if Jl == 0:
                return 0.0
            J = Jl
            denom = float(J * (J + 1))
            return float(K ** 2 * (2 * J + 1)) / denom if denom > 0 else 0.0
        if dJ == -1:
            J = Jl
            return float(J ** 2 - K ** 2) / float(J) if J > 0 else 0.0
        return 0.0

    if t in ("b", "c"):
        if abs(dK) != 1:
            return 0.0
        K = abs(Kl)
        s = dK  # +1 or -1
        if dJ == +1:
            J = Jl
            # (J + 1 + s*K + 1)(J + 1 + s*K + 2) → (J ± K + 2)(J ± K + 1)
            a = J + s * K + 2
            b = J + s * K + 1
            return float(a * b) / (4.0 * float(J + 1)) if J + 1 > 0 else 0.0
        if dJ == 0:
            J = Jl
            if J == 0:
                return 0.0
            denom = 4.0 * float(J) * float(J + 1)
            term = float((J - s * K) * (J + s * K + 1) * (2 * J + 1))
            return term / denom if denom > 0 else 0.0
        if dJ == -1:
            J = Jl
            if J == 0:
                return 0.0
            a = J - s * K
            b = J - s * K - 1
            return float(a * b) / (4.0 * float(J)) if J > 0 else 0.0
        return 0.0

    return 0.0


# ── Line list ────────────────────────────────────────────────────────────────

def compute_torsion_line_list(
    spec: TorsionHamiltonianSpec,
    J_values: Sequence[int] = (0, 1),
    K_values: Sequence[int] = (0,),
    n_levels: int = 8,
    *,
    symmetry_mode: Optional[str] = "c3",
    rotor_fold: int = 3,
    transition_type: str = "a",
    max_freq_mhz: Optional[float] = None,
    min_line_strength: float = 0.0,
    include_pure_torsional: bool = True,
    include_rotational: bool = True,
) -> list[dict]:
    """
    Compute a line list for torsion-rotation transitions.

    Iterates over all combinations of lower (J_lo, K_lo, vt_lo) and upper
    (J_hi, K_hi, vt_hi) quantum numbers, applying:
      - ΔJ ∈ {−1, 0, +1}
      - ΔK consistent with transition_type
      - Symmetry selection rules (A↔A, E↔E; A↔E forbidden for C3)
      - Positive frequency (E_hi > E_lo)

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    J_values : J quantum numbers to compute
    K_values : K quantum numbers to compute
    n_levels : torsional levels per (J, K) block
    symmetry_mode : 'c3' enables C3 symmetry labeling; None disables
    rotor_fold : 3 for CH3 (C3), 2 for CH2 (C2) — governs nuclear spin weights
    transition_type : 'a' (ΔK=0), 'b' or 'c' (|ΔK|=1)
    max_freq_mhz : discard lines above this frequency [MHz]
    min_line_strength : discard lines with line_strength below this threshold
    include_pure_torsional : include ΔJ=0, ΔK=0, Δvt≠0 (far-IR torsional band)
    include_rotational : include ΔJ=±1 (microwave/mm-wave rotational spectrum)

    Returns
    -------
    list of dicts, sorted by frequency, each with:
      freq_cm-1, freq_mhz       : transition frequency
      J_lo, K_lo, vt_lo, symmetry_lo : lower-state quantum numbers
      J_hi, K_hi, vt_hi, symmetry_hi : upper-state quantum numbers
      line_strength              : |<ψ_hi|cos(α)|ψ_lo>|²
      honl_london                : Hönl-London factor
      nuclear_spin_weight        : integer nuclear-spin weight (lower state)
      relative_intensity         : nsw × line_strength × honl_london (or 0 if forbidden)
      allowed                    : bool — symmetry selection rule result
    """
    sym_mode = str(symmetry_mode).strip().lower() if symmetry_mode else None
    label = sym_mode == "c3"

    # Solve and cache all (J, K) blocks
    blocks: dict[tuple[int, int], dict] = {}
    for J in J_values:
        for K in K_values:
            if abs(int(K)) > int(J):
                continue
            out = solve_ram_lite_levels(
                spec, J=int(J), K=int(K), n_levels=int(n_levels),
                symmetry_mode=sym_mode, label_levels=label,
            )
            blocks[(int(J), int(K))] = out

    t = str(transition_type).strip().lower()
    lines: list[dict] = []

    for (J_lo, K_lo), blk_lo in blocks.items():
        e_lo = blk_lo["energies_cm-1"]
        U_lo = blk_lo["eigenvectors"]
        m_lo = blk_lo["m_values"]
        sym_lo_arr = blk_lo.get("symmetry_labels", np.full(len(e_lo), "A", dtype=object))

        for (J_hi, K_hi), blk_hi in blocks.items():
            dJ = J_hi - J_lo
            dK = K_hi - K_lo

            if abs(dJ) > 1:
                continue
            if not include_pure_torsional and dJ == 0 and dK == 0:
                continue
            if not include_rotational and abs(dJ) == 1:
                continue

            # Enforce transition_type ΔK rule
            if t == "a" and dK != 0:
                continue
            if t in ("b", "c") and abs(dK) != 1:
                continue

            e_hi = blk_hi["energies_cm-1"]
            U_hi = blk_hi["eigenvectors"]
            m_hi = blk_hi["m_values"]
            sym_hi_arr = blk_hi.get("symmetry_labels", np.full(len(e_hi), "A", dtype=object))

            # Basis m_values must match for matrix element computation
            if m_hi.size != m_lo.size or not np.all(m_hi == m_lo):
                continue

            HL = honl_london_factor(J_lo, K_lo, J_hi, K_hi, transition_type=t)
            ME2 = torsion_dipole_matrix_elements(U_lo, U_hi, m_lo)

            for vt_hi in range(len(e_hi)):
                s_hi = str(sym_hi_arr[vt_hi]) if vt_hi < len(sym_hi_arr) else "A"
                for vt_lo in range(len(e_lo)):
                    s_lo = str(sym_lo_arr[vt_lo]) if vt_lo < len(sym_lo_arr) else "A"

                    freq_cm1 = float(e_hi[vt_hi]) - float(e_lo[vt_lo])
                    if freq_cm1 <= 0.0:
                        continue
                    freq_mhz = freq_cm1 * _MHZ_PER_CM1
                    if max_freq_mhz is not None and freq_mhz > float(max_freq_mhz):
                        continue

                    sel = symmetry_selection_rules(s_lo, s_hi, rotor_fold=int(rotor_fold))
                    allowed = sel.get("allowed") is not False

                    ls = float(ME2[vt_hi, vt_lo])
                    if ls < float(min_line_strength):
                        continue

                    try:
                        nsw = nuclear_spin_weight(s_lo, rotor_fold=int(rotor_fold))
                    except ValueError:
                        nsw = 1

                    rel_int = float(nsw) * ls * float(HL) if allowed else 0.0

                    lines.append({
                        "freq_cm-1": freq_cm1,
                        "freq_mhz": freq_mhz,
                        "J_lo": J_lo,
                        "K_lo": K_lo,
                        "vt_lo": vt_lo,
                        "symmetry_lo": s_lo,
                        "J_hi": J_hi,
                        "K_hi": K_hi,
                        "vt_hi": vt_hi,
                        "symmetry_hi": s_hi,
                        "line_strength": ls,
                        "honl_london": float(HL),
                        "nuclear_spin_weight": nsw,
                        "relative_intensity": rel_int,
                        "allowed": allowed,
                    })

    lines.sort(key=lambda r: r["freq_mhz"])
    return lines


def format_line_list_for_csv(lines: list[dict]) -> list[dict]:
    """
    Format line list rows for CSV export (round floats, keep all columns).
    """
    out = []
    for r in lines:
        out.append({
            "freq_cm-1": f"{r['freq_cm-1']:.6f}",
            "freq_mhz": f"{r['freq_mhz']:.4f}",
            "J_lo": int(r["J_lo"]),
            "K_lo": int(r["K_lo"]),
            "vt_lo": int(r["vt_lo"]),
            "symmetry_lo": str(r["symmetry_lo"]),
            "J_hi": int(r["J_hi"]),
            "K_hi": int(r["K_hi"]),
            "vt_hi": int(r["vt_hi"]),
            "symmetry_hi": str(r["symmetry_hi"]),
            "line_strength": f"{r['line_strength']:.8f}",
            "honl_london": f"{r['honl_london']:.6f}",
            "nuclear_spin_weight": int(r["nuclear_spin_weight"]),
            "relative_intensity": f"{r['relative_intensity']:.8f}",
            "allowed": bool(r["allowed"]),
        })
    return out
