from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal


PriorKind = Literal["bond", "angle", "dihedral"]


@dataclass(frozen=True)
class PriorRecord:
    """Normalized prior descriptor for a single internal coordinate."""

    name: str
    kind: PriorKind
    atoms: tuple[int, ...]
    target_value: float
    sigma: float
    units: str
    source: str = "default_library"
    confidence: Optional[float] = None
    notes: Optional[str] = None
    atom_types: Optional[tuple[str, ...]] = None
    bond_order: Optional[int] = None


def normalize_user_prior_atoms(raw_atoms: Sequence[int], n_atoms: int) -> tuple[int, ...]:
    """
    Normalize user-provided atom indices to 0-based indexing.

    Accepts either 0-based or 1-based lists. If any value is 0, treats input as
    0-based. Otherwise, treats as 1-based.
    """
    atoms = tuple(int(a) for a in raw_atoms)
    if not atoms:
        raise ValueError("Prior atoms list cannot be empty.")
    zero_based = any(a == 0 for a in atoms)
    if zero_based:
        norm = atoms
    else:
        norm = tuple(a - 1 for a in atoms)
    for a in norm:
        if a < 0 or a >= n_atoms:
            raise ValueError(f"Prior atom index {a} is outside valid range [0, {n_atoms - 1}].")
    return norm

