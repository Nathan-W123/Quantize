"""
RAM-lite reliability scoring for torsion-rotation Hamiltonians.

Provides assess_ram_lite_reliability() which evaluates a TorsionHamiltonianSpec
against heuristics for known RAM-lite failure modes and returns a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.torsion_hamiltonian import TorsionHamiltonianSpec


@dataclass
class TorsionReliabilityScore:
    """
    Reliability assessment for a RAM-lite torsion calculation.

    score                : 'high', 'moderate', 'low', or 'unreliable'
    flags                : list of human-readable reason strings (empty when score='high')
    ram_lite_error_bound_cm1 : rough upper bound on RAM-lite truncation error in cm^-1
                           (inf for 'unreliable')
    """
    score: str
    flags: list[str] = field(default_factory=list)
    ram_lite_error_bound_cm1: float = 0.0


def _potential_barrier(spec: TorsionHamiltonianSpec) -> float:
    """Estimate peak-to-trough barrier height by sampling the Fourier potential."""
    alpha = np.linspace(0.0, 2.0 * np.pi, 1000, endpoint=False)
    V = np.full_like(alpha, float(spec.potential.v0))
    for n, c in spec.potential.vcos.items():
        V += float(c) * np.cos(int(n) * alpha)
    for n, s in (spec.potential.vsin or {}).items():
        V += float(s) * np.sin(int(n) * alpha)
    return float(np.max(V) - np.min(V))


def assess_ram_lite_reliability(spec: TorsionHamiltonianSpec) -> TorsionReliabilityScore:
    """
    Assess the reliability of a RAM-lite calculation for the given spec.

    Scoring criteria (applied in order of severity):

    'unreliable':
      - Any centrifugal distortion constant > 1% of min(A, B, C)

    'low':
      - V/F < 2  (nearly free rotor — basis and convergence degrade)
      - n_basis < max Fourier harmonic order (basis inadequate)

    'moderate':
      - |B − C| > 0.1 × B  (strong asymmetry — K-mixing matters)
      - |ρ| > 0.95  (near-axis rotor — F·(m−ρK)² approximation degrades)

    'high':
      - None of the above triggered

    Parameters
    ----------
    spec : TorsionHamiltonianSpec

    Returns
    -------
    TorsionReliabilityScore
    """
    flags: list[str] = []

    F = float(spec.F)
    rho = abs(float(spec.rho))
    A = float(spec.A)
    B = float(spec.B)
    C = float(spec.C)
    n_basis = int(spec.n_basis)

    barrier = _potential_barrier(spec)
    v_over_f = barrier / F if F > 0.0 else float("inf")

    # ── Criterion 1: nearly free rotor ───────────────────────────────────────
    if v_over_f < 2.0:
        flags.append(
            f"V/F = {v_over_f:.2f} < 2: nearly free rotor; "
            "basis completeness and Fourier convergence may degrade."
        )

    # ── Criterion 2: basis inadequacy ────────────────────────────────────────
    all_n = [int(n) for n in spec.potential.vcos.keys() if int(n) > 0]
    all_n += [int(n) for n in (spec.potential.vsin or {}).keys() if int(n) > 0]
    max_n = max(all_n, default=0)
    if max_n > 0 and n_basis < max_n:
        flags.append(
            f"n_basis={n_basis} is smaller than the highest Fourier harmonic order {max_n}: "
            "the basis is inadequate; increase n_basis."
        )

    # ── Criterion 3: strong asymmetry ─────────────────────────────────────────
    if B > 0.0 and abs(B - C) > 0.1 * B:
        flags.append(
            f"B − C = {abs(B - C):.4f} cm⁻¹ > 0.1 × B ({0.1 * B:.4f}): "
            "strong asymmetric-top character; K-mixing between torsional states may be significant."
        )

    # ── Criterion 4: near-axis rotor ──────────────────────────────────────────
    if rho > 0.95:
        flags.append(
            f"ρ = {rho:.3f} > 0.95: near-axis internal rotor; "
            "the F·(m − ρK)² kinetic approximation degrades for large K."
        )

    # ── Criterion 5: centrifugal distortion dominance ────────────────────────
    cd_names = ("DJ", "DJK", "DK", "d1", "d2")
    cd_vals = {name: float(getattr(spec, name, 0.0) or 0.0) for name in cd_names}
    large_cd: list[str] = []
    rot_constants = [x for x in (A, B, C) if x > 0.0]
    if rot_constants:
        b_min = min(rot_constants)
        for name, val in cd_vals.items():
            if abs(val) > 0.01 * b_min:
                large_cd.append(f"{name}={val:.2e}")
    if large_cd:
        flags.append(
            f"Large centrifugal distortion constants ({', '.join(large_cd)}) exceed "
            "1% of the smallest rotational constant: "
            "RAM-lite may be inadequate for high-J transitions."
        )

    # ── Assign score ──────────────────────────────────────────────────────────
    if large_cd:
        score = "unreliable"
        error_bound = float("inf")
    elif any("n_basis" in f or "V/F" in f for f in flags):
        score = "low"
        error_bound = 2.0 * F
    elif flags:
        score = "moderate"
        error_bound = max(0.5 * barrier / max(v_over_f, 1.0), 0.05)
    else:
        score = "high"
        error_bound = max(0.1 * barrier / max(v_over_f, 1.0), 0.005)

    return TorsionReliabilityScore(
        score=score,
        flags=flags,
        ram_lite_error_bound_cm1=error_bound,
    )
