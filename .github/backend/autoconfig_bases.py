"""
Problem-shape heuristics for optimizer base hyperparameters.

At job startup, ``infer_optimizer_bases`` rescales trust region, LM damping,
SVD rank cutoff, and quantum weight from static counts (atoms, spectral rows,
DOF) before :class:`AutoConfigEngine` applies runtime diagnostics.

Disable via YAML ``autoconfig.heuristic_bases: false`` and pin values under
``optimizer:`` when you want fixed knobs for a specific molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# Align with bayes_tune.default_molecular_optimizer_space() clamps
SV_MIN = 1e-6
SV_MAX = 0.05
ALPHA_MIN = 0.1
ALPHA_MAX = 2.0
TRUST_MIN = 0.005
TRUST_MAX = 0.25
LAMBDA_MIN = 1e-6
LAMBDA_MAX = 0.5

_REF_ATOMS = 3


@dataclass(frozen=True)
class ProblemShape:
    """Static problem dimensions (no QM)."""

    n_atoms: int
    n_params: int
    n_jacobian_rows: int
    has_internal_priors: bool = False
    prior_weight: float = 0.0
    coordinate_mode: str = "cartesian"
    use_dihedrals: bool = False

    @property
    def constraint_frac(self) -> float:
        return float(self.n_jacobian_rows) / float(max(self.n_params, 1))


def count_spectral_rows(isotopologues: Sequence[Mapping[str, Any]]) -> int:
    """Jacobian row count from spectral isotopologue entries only."""
    n = 0
    for iso in isotopologues:
        idx = iso.get("component_indices", [])
        if idx is None:
            continue
        n += len(list(idx))
    return n


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def infer_optimizer_bases(
    shape: ProblemShape,
    *,
    trust_radius: float,
    null_trust_radius: float | None,
    lambda_damp: float,
    sv_threshold: float,
    alpha_quantum: float,
) -> dict[str, float]:
    """
    Rescale optimizer bases from ``shape``; inputs are starting values (e.g. from YAML or run_generic).

    Returns keys: trust_radius, null_trust_radius, lambda_damp, sv_threshold, alpha_quantum.
    """
    n_atoms = max(1, int(shape.n_atoms))
    n_params = max(1, int(shape.n_params))
    cf = shape.constraint_frac

    # Underdetermined spectral fit → more damping, less quantum pull, smaller steps
    under = max(0.0, 1.0 - min(cf, 1.0))
    lambda_scale = 1.0 + 0.8 * under
    alpha_scale = 1.0 / (1.0 + 0.6 * under)
    trust_scale = 1.0 / (1.0 + 0.25 * under)

    # Larger molecules: smaller trust (same Å step is a larger relative deformation)
    atom_scale = (float(_REF_ATOMS) / float(n_atoms)) ** 0.25
    trust_scale *= atom_scale

    # More DOF: slightly looser SVD cutoff (rank boundary less brittle)
    sv_log = 1.0 + 0.15 * max(0.0, np.log10(n_params / float(_REF_ATOMS * 3)))
    sv_scale = float(np.clip(sv_log, 1.0, 2.5))

    if shape.has_internal_priors and shape.prior_weight > 0.0:
        lambda_scale *= 1.15

    out_trust = _clip(float(trust_radius) * trust_scale, TRUST_MIN, TRUST_MAX)
    base_null = float(null_trust_radius) if null_trust_radius is not None else 0.5 * float(trust_radius)
    out_null = _clip(base_null * trust_scale, TRUST_MIN, TRUST_MAX)
    out_lambda = _clip(float(lambda_damp) * lambda_scale, LAMBDA_MIN, LAMBDA_MAX)
    out_sv = _clip(float(sv_threshold) * sv_scale, SV_MIN, SV_MAX)
    out_alpha = _clip(float(alpha_quantum) * alpha_scale, ALPHA_MIN, ALPHA_MAX)

    return {
        "trust_radius": out_trust,
        "null_trust_radius": out_null,
        "lambda_damp": out_lambda,
        "sv_threshold": out_sv,
        "alpha_quantum": out_alpha,
    }


__all__ = [
    "ProblemShape",
    "count_spectral_rows",
    "infer_optimizer_bases",
]
