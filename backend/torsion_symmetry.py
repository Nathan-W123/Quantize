"""
Symmetry blocks and tunneling splitting for the RAM torsion-rotation Hamiltonian.

Implements:
- C3 symmetry block decomposition (A / E1 / E2 species) and per-block diagonalization
- A-E tunneling splitting prediction (Delta_vt = E_E - E_A for each torsional level)
- Wang-like permutation matrix that block-diagonalizes H under C3 symmetry
- Symmetry selection rules for A/E transitions
- Symmetry purity diagnostics

Cn symmetry labeling convention
---------------------------------
C3 rotor: m mod 3 == 0 → A species
          m mod 3 == 1 → E1 species
          m mod 3 == 2 → E2 species
A and E have different nuclear spin statistics; A<->E transitions are forbidden.
E1 and E2 are degenerate for symmetric potentials.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.torsion_hamiltonian import (
    TorsionHamiltonianSpec,
    _validate_units,
    basis_m_values,
    fourier_potential_matrix,
    solve_ram_lite_levels,
)

_C3_RESIDUE_LABEL: dict[int, str] = {0: "A", 1: "E1", 2: "E2"}
_C3_LABEL_RESIDUE: dict[str, int] = {"A": 0, "E1": 1, "E2": 2, "E": 1}


# ── Wang transformation ───────────────────────────────────────────────────────

def wang_transformation_c3(m_values: np.ndarray) -> np.ndarray:
    """
    Permutation matrix W that groups basis vectors by C3 residue (A/E1/E2).

    W[new_i, old_i] = 1 exactly for the atom permutation that sorts |m> vectors
    into [A-block | E1-block | E2-block] order.

    W @ H @ W.T is block-diagonal (A, E1, E2) for a C3-symmetric Hamiltonian.

    Parameters
    ----------
    m_values : int ndarray — basis m quantum numbers

    Returns
    -------
    W : (N, N) complex ndarray — unitary permutation matrix
    """
    m = np.asarray(m_values, dtype=int).ravel()
    N = m.size
    groups: list[list[int]] = [[], [], []]
    for i, mi in enumerate(m):
        groups[int(mi) % 3].append(i)
    order = groups[0] + groups[1] + groups[2]
    W = np.zeros((N, N), dtype=complex)
    for new_i, old_i in enumerate(order):
        W[new_i, old_i] = 1.0
    return W


# ── Per-block energies ─────────────────────────────────────────────────────────

def c3_symmetry_block_energies(
    spec: TorsionHamiltonianSpec,
    J: int = 0,
    K: int = 0,
    *,
    n_levels_per_block: Optional[int] = None,
) -> dict:
    """
    Diagonalize the RAM-lite Hamiltonian within each C3 symmetry block.

    Restricts the |m> basis to m ≡ r (mod 3) for each residue r = 0, 1, 2,
    giving exact block diagonalization for a C3-symmetric potential (V_n with
    n ≡ 0 mod 3 only).

    Parameters
    ----------
    spec : TorsionHamiltonianSpec
    J, K : rotational quantum numbers for the block
    n_levels_per_block : truncate each block to this many levels

    Returns
    -------
    dict with keys 'A', 'E1', 'E2' (each with 'energies_cm-1', 'm_values',
    'residue') and 'warnings'.
    """
    result = solve_ram_lite_levels(
        spec, J=J, K=K, symmetry_mode="c3", return_blocks=True, n_levels=None
    )
    blocks = result.get("symmetry_blocks") or {}
    warns = list(result.get("warnings", []))
    out: dict = {"warnings": warns}

    for r, label in _C3_RESIDUE_LABEL.items():
        if label in blocks:
            b = blocks[label]
            e = b["energies_cm-1"]
            if n_levels_per_block is not None:
                e = e[:int(n_levels_per_block)]
            out[label] = {
                "energies_cm-1": e,
                "m_values": b["m_values"],
                "residue": int(r),
            }
        else:
            out[label] = {"energies_cm-1": np.array([]), "m_values": np.array([]), "residue": r}

    return out


# ── Tunneling splittings ──────────────────────────────────────────────────────

def predict_tunneling_splitting(
    spec: TorsionHamiltonianSpec,
    *,
    J: int = 0,
    K: int = 0,
    n_levels: int = 5,
) -> list[dict]:
    """
    Predict A/E tunneling splittings for the lowest torsional levels.

    For each pair of A-species and E1-species levels (indexed by vt = 0, 1, 2, ...),
    the splitting is Delta = E(E1) - E(A).  Positive Delta means the A level is
    lower (standard for C3 methyl rotors below the barrier).

    Parameters
    ----------
    spec : TorsionHamiltonianSpec with C3-compatible potential
    J, K : rotational quantum numbers
    n_levels : number of A/E level pairs to include

    Returns
    -------
    list of dicts with:
      vt : torsional level index (0 = ground)
      J, K
      E_A_cm-1 : A-species energy
      E_E_cm-1 : E1-species energy
      splitting_cm-1 : E_E - E_A
      splitting_MHz  : same in MHz
    """
    _MHZ_PER_CM1 = 29979.2458
    blocks = c3_symmetry_block_energies(spec, J=J, K=K, n_levels_per_block=n_levels)
    E_A = blocks.get("A", {}).get("energies_cm-1", np.array([]))
    E_E = blocks.get("E1", {}).get("energies_cm-1", np.array([]))

    rows = []
    n = min(int(n_levels), len(E_A), len(E_E))
    for vt in range(n):
        ea = float(E_A[vt])
        ee = float(E_E[vt])
        split = ee - ea
        rows.append({
            "vt": vt,
            "J": int(J),
            "K": int(K),
            "E_A_cm-1": ea,
            "E_E_cm-1": ee,
            "splitting_cm-1": split,
            "splitting_MHz": split * _MHZ_PER_CM1,
        })
    return rows


# ── Symmetry selection rules ──────────────────────────────────────────────────

def symmetry_selection_rules(
    symmetry_lo: str,
    symmetry_hi: str,
    rotor_fold: int = 3,
) -> dict:
    """
    Determine whether a transition between symmetry species is allowed.

    For a C3 rotor (rotor_fold=3) the nuclear spin statistics require:
      A <-> A : allowed
      E <-> E (including E1<->E1, E2<->E2, E1<->E2) : allowed
      A <-> E : forbidden

    Parameters
    ----------
    symmetry_lo, symmetry_hi : species labels ('A', 'E', 'E1', 'E2')
    rotor_fold : rotational symmetry order (currently only 3 implemented)

    Returns
    -------
    dict with 'allowed' (bool or None) and 'reason' (str)
    """
    lo = str(symmetry_lo).strip().upper()
    hi = str(symmetry_hi).strip().upper()

    if rotor_fold == 3:
        lo_A = lo == "A"
        hi_A = hi == "A"
        lo_E = lo in {"E", "E1", "E2"}
        hi_E = hi in {"E", "E1", "E2"}

        if lo_A and hi_A:
            return {"allowed": True, "reason": "A↔A: same nuclear spin species (C3 rotor)."}
        if lo_E and hi_E:
            return {"allowed": True, "reason": "E↔E: same nuclear spin statistics (C3 rotor)."}
        if (lo_A and hi_E) or (lo_E and hi_A):
            return {
                "allowed": False,
                "reason": "A↔E forbidden: different nuclear spin statistics for C3 rotor.",
            }
        return {"allowed": None, "reason": f"Unknown symmetry labels: lo={symmetry_lo!r}, hi={symmetry_hi!r}"}

    return {"allowed": None, "reason": f"Selection rules not implemented for rotor_fold={rotor_fold}."}


# ── Diagnostics ───────────────────────────────────────────────────────────────

def symmetry_purity_table(
    spec: TorsionHamiltonianSpec,
    J: int = 0,
    K: int = 0,
    *,
    n_levels: int = 6,
) -> list[dict]:
    """
    Per-level symmetry purity table from the full (unsplit) Hamiltonian.

    Each row reports the dominant symmetry label and the purity (fractional weight
    in the dominant C3 residue block) for each eigenstate.

    Returns
    -------
    list of dicts with:
      level_index, energy_cm-1, symmetry_label, symmetry_sublabel, purity
    """
    out = solve_ram_lite_levels(
        spec, J=J, K=K, symmetry_mode="c3", label_levels=True, n_levels=n_levels
    )
    e = out["energies_cm-1"]
    labels = out.get("symmetry_labels", np.full(len(e), "unknown", dtype=object))
    sublabels = out.get("symmetry_sublabels", np.full(len(e), "unknown", dtype=object))
    purities = out.get("symmetry_purity", np.zeros(len(e)))

    rows = []
    for i, (ei, lbl, sub, pur) in enumerate(zip(e, labels, sublabels, purities)):
        rows.append({
            "level_index": i,
            "energy_cm-1": float(ei),
            "symmetry_label": str(lbl),
            "symmetry_sublabel": str(sub),
            "purity": float(pur),
        })
    return rows


def tunneling_splitting_to_csv_rows(rows: list[dict]) -> list[dict]:
    """Format tunneling splitting rows for CSV export (string-formatted floats)."""
    out = []
    for r in rows:
        out.append({
            "vt": int(r["vt"]),
            "J": int(r["J"]),
            "K": int(r["K"]),
            "E_A_cm-1": f"{r['E_A_cm-1']:.6f}",
            "E_E_cm-1": f"{r['E_E_cm-1']:.6f}",
            "splitting_cm-1": f"{r['splitting_cm-1']:.6f}",
            "splitting_MHz": f"{r['splitting_MHz']:.4f}",
        })
    return out
