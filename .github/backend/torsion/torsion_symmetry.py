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

import re

from math import gcd
from typing import Optional

import numpy as np

from backend.torsion.torsion_hamiltonian import (
    TorsionHamiltonianSpec,
    _validate_units,
    basis_m_values,
    fourier_potential_matrix,
    solve_ram_lite_levels,
)

_C3_RESIDUE_LABEL: dict[int, str] = {0: "A", 1: "E1", 2: "E2"}
_C3_LABEL_RESIDUE: dict[str, int] = {"A": 0, "E1": 1, "E2": 2, "E": 1}

# ── Nuclear-spin statistical weights, derived rather than tabulated ─────────
#
# These used to be two hard-coded dicts, and one of them was wrong. C3 carried
# A:E = 1:2, which matches no convention: the published value for methanol, the
# canonical C3 methyl rotor, is A weight 4 (the I=3/2 quartet) against E weight
# 4 once its double degeneracy is counted, so A:E = 1:1 in total or 2:1 against
# a single E component. The table also contradicted its own C2 entry, which was
# right -- A:B = 1:3 IS a nuclear-spin weight -- so whichever way the C3 numbers
# were meant, the two folds did not share a convention.
#
# Deriving them removes the question. For n equivalent nuclei of spin I
# permuted cyclically:
#
#   1. The permutation C_n^m decomposes n objects into gcd(m, n) cycles, and a
#      product spin state is invariant under it exactly when it is constant on
#      each cycle. So the trace of that permutation on the spin space is
#      (2I+1)^gcd(m,n), with no character table required.
#   2. Projecting onto the k-th irrep of the cyclic group gives the number of
#      spin functions of that symmetry:
#
#          a_k = (1/n) sum_m exp(-2 pi i k m / n) (2I+1)^gcd(m,n)
#
#   3. An n-cycle is an EVEN permutation exactly when n is odd. For fermions
#      the total wavefunction is symmetric under even permutations and
#      antisymmetric under odd ones, so it must carry irrep k=0 for odd n and
#      k=n/2 for even n. A torsional level of species k therefore takes the
#      weight of the spin species that completes it.
#
# The derivation is checked two ways in test_torsion_symmetry: it reproduces
# the C2 entry this module already had and agreed with, and it reproduces the
# published methanol C3 weights that the old C3 entry did not.
#
# A CAVEAT THE OLD TABLE HID. The fold of the POTENTIAL is not the number of
# equivalent nuclei in the top. Nitromethane and toluene have six-fold
# torsional potentials because a three-fold methyl turns against a two-fold
# frame, and their spin statistics are still the three protons' C3 ones. Key
# the weights off the nuclei, not off the barrier, whenever the two differ;
# nuclear_spin_weight's rotor_fold argument assumes they do not, which is true
# for CH2 and CH3 and is why it was never wrong before.


def _spin_function_counts(n_equivalent: int, spin: float = 0.5) -> dict[int, int]:
    """Spin functions of each cyclic symmetry species, by character projection."""
    n = int(n_equivalent)
    if n < 1:
        raise ValueError("n_equivalent must be >= 1.")
    d = int(round(2.0 * float(spin) + 1.0))
    if d < 2:
        raise ValueError("spin must be at least 1/2 for a spin-bearing rotor.")

    counts = {}
    for k in range(n):
        total = sum(np.exp(-2j * np.pi * k * m / n) * d ** gcd(m, n)
                    for m in range(n))
        counts[k] = int(round(float(total.real / n)))
    if sum(counts.values()) != d ** n:
        raise ArithmeticError(
            f"spin function counts {counts} sum to {sum(counts.values())}, "
            f"expected {d}^{n} = {d ** n}"
        )
    return counts


def species_label(k: int, n_equivalent: int) -> str:
    """Cyclic-group label for symmetry species k of a C_n rotor.

    k = 0 is the totally symmetric A. For even n, k = n/2 has character -1
    under every generator and is the non-degenerate B. Everything else pairs
    with its conjugate into a doubly degenerate E.
    """
    n, k = int(n_equivalent), int(k) % int(n_equivalent)
    if k == 0:
        return "A"
    if n % 2 == 0 and k == n // 2:
        return "B"
    return f"E{k}"


def nuclear_spin_weights(n_equivalent: int, spin: float = 0.5) -> dict[str, int]:
    """Statistical weight of each torsional species, from first principles.

    Fermion statistics assumed, which is right for the protons these rotors are
    made of. Returns weights keyed by species label, with "E" aliased to the
    lowest E species so existing callers keep working.
    """
    n = int(n_equivalent)
    counts = _spin_function_counts(n, spin)
    target = 0 if n % 2 == 1 else n // 2
    weights = {species_label(k, n): counts[(target - k) % n] for k in range(n)}
    for k in range(1, n):
        label = species_label(k, n)
        if label.startswith("E"):
            weights.setdefault("E", weights[label])
            break
    return weights


#: Folds with equivalent spin-1/2 nuclei in the top, derived once at import.
#: 4 and 6 are included for completeness; see the caveat above before using
#: fold 6 for a molecule whose six-fold barrier comes from a three-fold top.
_ROTOR_FOLD_WEIGHTS: dict[int, dict[str, int]] = {
    n: nuclear_spin_weights(n, spin=0.5) for n in (2, 3, 4, 6)
}

C2_NUCLEAR_SPIN_WEIGHTS: dict[str, int] = _ROTOR_FOLD_WEIGHTS[2]
C3_NUCLEAR_SPIN_WEIGHTS: dict[str, int] = _ROTOR_FOLD_WEIGHTS[3]


def nuclear_spin_weight(symmetry_label: str, rotor_fold: int = 3) -> int:
    """
    Return the nuclear-spin statistical weight for a symmetry species.

    Parameters
    ----------
    symmetry_label : 'A', 'E', 'E1', 'E2' (C3 rotor) or 'A', 'B' (C2 rotor)
    rotor_fold : 3 for CH3-type, 2 for CH2-type

    Returns
    -------
    int weight

    Raises
    ------
    ValueError for unknown rotor_fold or label
    """
    label = str(symmetry_label).strip().upper()
    table = _ROTOR_FOLD_WEIGHTS.get(int(rotor_fold))
    if table is None:
        raise ValueError(
            f"nuclear_spin_weight not implemented for rotor_fold={rotor_fold}. "
            f"Supported: {sorted(_ROTOR_FOLD_WEIGHTS.keys())}."
        )
    w = table.get(label)
    if w is None:
        raise ValueError(
            f"Unknown symmetry label {symmetry_label!r} for rotor_fold={rotor_fold}. "
            f"Known: {sorted(table.keys())}."
        )
    return int(w)


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

def _species_index(label: str, rotor_fold: int):
    """Cyclic irrep index k for a species label, or None if unrecognised.

    Accepts the bare "E" as the lowest E species, which is what C2 and C3
    callers have always passed.
    """
    text = str(label).strip().upper()
    n = int(rotor_fold)
    if text == "A":
        return 0
    if text == "B":
        return n // 2 if n % 2 == 0 else None
    if text == "E":
        # The bare "E" means the lowest degenerate species. A C2 rotor has
        # none: its only two species are k=0 (A) and k=1 = n/2 (B).
        return 1 if n >= 3 else None
    m = re.fullmatch(r"E(\d+)", text)
    if m:
        k = int(m.group(1))
        if not 0 < k < n:
            return None
        # An E label is only meaningful where species k is actually part of a
        # degenerate pair. For even n, k = n/2 is the non-degenerate B, so
        # "E1" on a C2 rotor is not a C2 label at all -- it used to be read as
        # B, silently answering a question about a species that does not exist.
        if n % 2 == 0 and k == n // 2:
            return None
        return k
    return None


def symmetry_selection_rules(
    symmetry_lo: str,
    symmetry_hi: str,
    rotor_fold: int = 3,
) -> dict:
    """
    Determine whether a transition between torsional symmetry species is allowed.

    ONE RULE, FOR EVERY FOLD. A radiative transition does not reorient nuclear
    spins, so it cannot change the nuclear spin species. A transition is
    allowed exactly when the two levels belong to the same species, and the
    degenerate pairs count as one: species k and n-k are complex conjugates of
    each other and describe the same spin modification, which is why
    E1 <-> E2 is allowed for a C3 rotor.

    So the test is k_lo = +/- k_hi (mod n). For C3 that reproduces the
    hand-written rules this function used to carry -- A<->A and E<->E allowed,
    A<->E forbidden -- and for C2, A<->A and B<->B allowed, A<->B forbidden. It
    extends to C4 and C6 without new cases, where it correctly separates E1
    from E2: for a six-fold rotor those are different spin species with
    different weights (11 against 9), not two halves of one.

    Returns
    -------
    dict with 'allowed' (bool or None when a label is unrecognised) and
    'reason' (str).
    """
    n = int(rotor_fold)
    if n < 2:
        return {"allowed": None, "reason": f"rotor_fold must be >= 2, got {rotor_fold}."}

    k_lo = _species_index(symmetry_lo, n)
    k_hi = _species_index(symmetry_hi, n)
    if k_lo is None or k_hi is None:
        return {
            "allowed": None,
            "reason": (f"Unknown symmetry labels for a C{n} rotor: "
                       f"lo={symmetry_lo!r}, hi={symmetry_hi!r}"),
        }

    lo_name = species_label(k_lo, n)
    hi_name = species_label(k_hi, n)
    if k_lo == k_hi or k_lo == (-k_hi) % n:
        return {
            "allowed": True,
            "reason": (f"{lo_name} <-> {hi_name}: same nuclear spin species "
                       f"(C{n} rotor)."),
        }
    return {
        "allowed": False,
        "reason": (f"{lo_name} <-> {hi_name} forbidden: different nuclear spin "
                   f"statistics for a C{n} rotor."),
    }


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
