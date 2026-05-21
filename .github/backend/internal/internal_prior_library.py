from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from backend.priors.prior_models import PriorRecord


@dataclass(frozen=True)
class _BondRule:
    pair: tuple[str, str]
    bond_order: int
    target_A: float
    sigma_A: float


@dataclass(frozen=True)
class _AngleRule:
    pattern: tuple[str, str, str]
    target_deg: float
    sigma_deg: float
    label: str


_BOND_RULES = [
    _BondRule(("C", "H"), 1, 1.090, 0.015),
    _BondRule(("C", "C"), 1, 1.540, 0.020),
    _BondRule(("C", "C"), 2, 1.340, 0.020),
    _BondRule(("C", "O"), 1, 1.430, 0.020),
    _BondRule(("C", "O"), 2, 1.210, 0.015),
    _BondRule(("C", "N"), 1, 1.470, 0.020),
    _BondRule(("O", "H"), 1, 0.960, 0.015),
    _BondRule(("N", "H"), 1, 1.010, 0.015),
]

_ANGLE_RULES = [
    _AngleRule(("H", "C", "H"), 109.5, 2.0, "tetrahedral_HCH"),
    _AngleRule(("X", "C", "X"), 109.5, 2.0, "tetrahedral"),
    _AngleRule(("X", "Csp2", "X"), 120.0, 2.0, "trigonal_planar"),
    _AngleRule(("X", "X", "X"), 180.0, 3.0, "linear"),
]


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _infer_bond_order(e1: str, e2: str, length_a: float) -> int:
    pair = _sorted_pair(e1, e2)
    if pair == ("C", "O"):
        return 2 if length_a < 1.30 else 1
    if pair == ("C", "C"):
        return 2 if length_a < 1.42 else 1
    return 1


def _resolve_bond_rule(e1: str, e2: str, length_a: float) -> _BondRule | None:
    pair = _sorted_pair(e1, e2)
    bo = _infer_bond_order(e1, e2, length_a)
    for r in _BOND_RULES:
        if _sorted_pair(*r.pair) == pair and r.bond_order == bo:
            return r
    for r in _BOND_RULES:
        if _sorted_pair(*r.pair) == pair:
            return r
    return None


def _angle_hybrid_tag(center: str, angle_deg: float) -> str:
    if center == "C":
        if angle_deg > 116.0:
            return "Csp2"
        return "C"
    return center


def resolve_default_prior(
    ic: Any,
    elems: list[str],
    q_value: float,
) -> Optional[PriorRecord]:
    """
    Build default chemistry-inspired prior for one internal coordinate.

    q_value is expected in natural units (A for bond, rad for angles/dihedrals).
    """
    kind = ic.kind
    atoms = ic.atoms
    if kind == "bond" and len(atoms) == 2:
        i, j = atoms
        e1, e2 = elems[i], elems[j]
        rule = _resolve_bond_rule(e1, e2, float(q_value))
        if rule is None:
            return None
        return PriorRecord(
            name=ic.name,
            kind="bond",
            atoms=atoms,
            atom_types=(e1, e2),
            bond_order=rule.bond_order,
            target_value=float(rule.target_A),
            sigma=float(rule.sigma_A),
            units="angstrom",
            source="default_library",
            confidence=0.7,
            notes="chem-default bond prior",
        )

    if kind == "angle" and len(atoms) == 3:
        i, j, k = atoms
        e1, ec, e3 = elems[i], elems[j], elems[k]
        ang_deg = float(np.degrees(q_value))
        center_tag = _angle_hybrid_tag(ec, ang_deg)

        if (e1, ec, e3) == ("H", "C", "H") or (e3, ec, e1) == ("H", "C", "H"):
            target_deg, sigma_deg, label = 109.5, 2.0, "H-C-H"
        elif center_tag == "Csp2":
            target_deg, sigma_deg, label = 120.0, 2.0, "trigonal_planar"
        elif ang_deg > 165.0:
            target_deg, sigma_deg, label = 180.0, 3.0, "linear"
        else:
            target_deg, sigma_deg, label = 109.5, 2.0, "tetrahedral"

        return PriorRecord(
            name=ic.name,
            kind="angle",
            atoms=atoms,
            atom_types=(e1, ec, e3),
            target_value=float(np.deg2rad(target_deg)),
            sigma=float(np.deg2rad(sigma_deg)),
            units="radian",
            source="default_library",
            confidence=0.65,
            notes=f"chem-default angle prior ({label})",
        )

    return None
