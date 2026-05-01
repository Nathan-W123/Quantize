"""
Bayesian optimization for Quantize geometry workflows.

Two common uses:

1. **Hybrid optimizer hyperparameters** — tune ``MolecularOptimizer`` knobs
   (trust region, LM damping, SVD cutoff, quantum weight, finite-difference
   step, etc.) to minimize final rotational-constant RMS after ``run()``.
   Requires repeated QM calls (Psi4/ORCA); keep ``n_calls`` modest.

2. **Initial-guess relaxation** — tune spring/repulsion parameters in
   ``geometryguess._relax_geometry`` against a cheap surrogate (default:
   mean squared bond-length strain after relaxation). No QM in the loop.

Dependencies: ``numpy``, ``scikit-optimize`` (``pip install scikit-optimize``).
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
except ImportError:  # pragma: no cover - optional dependency
    gp_minimize = None  # type: ignore[misc, assignment]
    Real = Integer = None  # type: ignore[misc, assignment]


def _require_skopt() -> None:
    if gp_minimize is None:
        raise ImportError(
            "Bayesian tuning requires scikit-optimize. Install with:\n"
            "  pip install scikit-optimize"
        )


def default_molecular_optimizer_space():
    """Reasonable box bounds for hybrid ``MolecularOptimizer`` tuning."""
    _require_skopt()
    return [
        Real(0.005, 0.25, prior="log-uniform", name="trust_radius"),
        Real(1e-6, 0.5, prior="log-uniform", name="lambda_damp"),
        Real(1e-5, 0.05, prior="log-uniform", name="sv_threshold"),
        Real(0.15, 5.0, name="alpha_quantum"),
        Real(1e-4, 5e-3, prior="log-uniform", name="spectral_delta"),
        Integer(1, 12, name="hess_recalc_every"),
    ]


def default_relax_geometry_space():
    """Bounds for ``geometryguess._relax_geometry`` / ``guess_geometry(..., relax_kwargs=...)``."""
    _require_skopt()
    return [
        Integer(50, 400, name="n_steps"),
        Real(0.002, 0.05, prior="log-uniform", name="dt"),
        Real(0.5, 20.0, name="k_bond"),
        Real(1e-4, 0.2, prior="log-uniform", name="k_rep"),
        Real(0.8, 2.5, name="rep_cut"),
    ]


def _vector_to_params(
    dimensions: Sequence,
    x: Sequence[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dim, val in zip(dimensions, x):
        name = dim.name
        if isinstance(dim, Integer):
            out[name] = int(round(float(val)))
        else:
            out[name] = float(val)
    return out


def _surrogate_relax_objective(
    elems: Sequence[str],
    bonds: Sequence[Tuple[int, int]],
    relax_params: Mapping[str, Any],
) -> float:
    """Mean squared fractional bond strain after relaxation (minimize)."""
    from backend.geometryguess import guess_geometry

    coords = guess_geometry(list(elems), list(bonds), relax_kwargs=dict(relax_params))
    # Lazy import: targets match geometryguess internal guesses
    from backend.geometryguess import _bond_length_guess  # pylint: disable=import-outside-toplevel

    errs = []
    for i, j in bonds:
        t = _bond_length_guess(elems[i], elems[j])
        d = float(np.linalg.norm(coords[j] - coords[i]))
        if t > 1e-9:
            errs.append((d - t) / t)
    if not errs:
        return 0.0
    return float(np.mean(np.square(errs)))


def tune_relax_geometry(
    elems: Sequence[str],
    bonds: Sequence[Tuple[int, int]],
    dimensions=None,
    n_calls: int = 40,
    random_state: Optional[int] = 0,
    verbose: bool = True,
):
    """
    Bayesian optimization of guess-geometry relaxation hyperparameters.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        ``skopt`` optimization result (``result.x``, ``result.fun``, ...).
    best_params : dict
        Best parameter dictionary for ``relax_kwargs``.
    """
    _require_skopt()
    dimensions = dimensions or default_relax_geometry_space()

    def objective(x):
        params = _vector_to_params(dimensions, x)
        return _surrogate_relax_objective(elems, bonds, params)

    res = gp_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        random_state=random_state,
        verbose=verbose,
    )
    best_params = _vector_to_params(dimensions, res.x)
    return res, best_params


def tune_molecular_optimizer(
    coords,
    elems: Sequence[str],
    isotopologues: Sequence[Mapping[str, Any]],
    base_kwargs: Optional[Mapping[str, Any]] = None,
    dimensions=None,
    n_calls: int = 20,
    random_state: Optional[int] = 0,
    verbose: bool = True,
    objective_metric: str = "freq_rms",
    freq_weight: float = 1.0,
    wrms_weight: float = 0.0,
    failure_penalty: float = 1e6,
    workdir_suffix_fn: Optional[Callable[[int], str]] = None,
):
    """
    Bayesian optimization of ``MolecularOptimizer`` hyperparameters.

    Each trial constructs ``MolecularOptimizer(..., **merged_kwargs)``, runs
    ``run()``, and returns a scalar loss.

    Parameters
    ----------
    coords, elems, isotopologues
        Passed through to ``MolecularOptimizer``.
    base_kwargs : dict, optional
        Fixed keyword arguments (method, basis, paths, convergence limits, ...).
    dimensions : list, optional
        ``skopt`` space dimensions; default from ``default_molecular_optimizer_space()``.
    objective_metric : {'freq_rms', 'wrms', 'combined'}
        ``combined`` uses ``freq_weight * freq_rms + wrms_weight * wrms`` from the last iteration.
    workdir_suffix_fn : callable, optional
        ``callable(trial_index) -> str`` appended under ``workdir`` so parallel QM
        jobs do not clash. Default: ``trial_{i}`` subfolder inside ``base_workdir``.
    """
    _require_skopt()
    from backend.quantize import MolecularOptimizer  # pylint: disable=import-outside-toplevel

    base_kwargs = dict(base_kwargs or {})
    dimensions = dimensions or default_molecular_optimizer_space()
    base_workdir = str(base_kwargs.get("workdir", "."))

    def objective(x):
        params = _vector_to_params(dimensions, x)
        kw = {**base_kwargs, **params}
        trial = getattr(objective, "_trial_count", 0)
        objective._trial_count = trial + 1  # type: ignore[attr-defined]
        if workdir_suffix_fn is not None:
            sub = workdir_suffix_fn(trial)
        else:
            sub = f"bayes_trial_{trial:04d}"
        kw["workdir"] = os.path.join(base_workdir, sub)

        try:
            opt = MolecularOptimizer(
                coords=np.asarray(coords, dtype=float),
                elems=list(elems),
                isotopologues=list(isotopologues),
                **kw,
            )
            opt.run()
            last = opt.history[-1] if opt.history else {}
            freq = float(last.get("freq_rms", failure_penalty))
            wrms = float(last.get("wrms", failure_penalty))
            if objective_metric == "freq_rms":
                val = freq_weight * freq
            elif objective_metric == "wrms":
                val = wrms_weight * wrms if wrms_weight else wrms
            elif objective_metric == "combined":
                val = freq_weight * freq + wrms_weight * wrms
            else:
                raise ValueError(
                    f"objective_metric must be 'freq_rms', 'wrms', or 'combined'; got {objective_metric!r}"
                )
            if not math.isfinite(val):
                return failure_penalty
            return val
        except Exception:
            return failure_penalty

    objective._trial_count = 0  # type: ignore[attr-defined]

    res = gp_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        random_state=random_state,
        verbose=verbose,
    )
    best_params = _vector_to_params(dimensions, res.x)
    return res, best_params


__all__ = [
    "default_molecular_optimizer_space",
    "default_relax_geometry_space",
    "tune_molecular_optimizer",
    "tune_relax_geometry",
]
