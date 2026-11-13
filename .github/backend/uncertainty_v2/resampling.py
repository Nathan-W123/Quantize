from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def resample_isotopologues_gaussian(
    isotopologues: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Perturb observed rotational constants by their reported sigma values."""
    out: list[dict[str, Any]] = []
    for iso in isotopologues:
        entry = deepcopy(iso)
        obs = np.asarray(iso.get("obs_constants", []), dtype=float)
        sigma = np.asarray(iso.get("sigma_constants", np.ones_like(obs) * 0.1), dtype=float)
        if obs.shape != sigma.shape:
            raise ValueError(
                f"obs_constants and sigma_constants shape mismatch for isotopologue "
                f"{iso.get('name', '<unnamed>')}: {obs.shape} vs {sigma.shape}"
            )
        entry["obs_constants"] = obs + rng.normal(0.0, sigma)
        out.append(entry)
    return out



def symmetric_sqrt(cov: np.ndarray) -> np.ndarray:
    """The symmetric square root S of a covariance (S S = cov, S = S^T).

    Draws are made as S z. An eigenvector factor V sqrt(w) has the same
    covariance but is not unique: LAPACK may return any sign for each
    eigenvector, and any rotation inside a (near-)degenerate eigenspace, so
    the same seed gave different draws on different BLAS builds or thread
    counts -- measured on formyl fluoride, rto shifts differing by 1.1-2.0x
    their own size between OMP_NUM_THREADS=1 and 4 at base geometries 2e-14
    Angstrom apart. S is a function of the matrix alone, so it is
    insensitive to all of that, and a draw is reproducible from its seed.
    """
    cov = np.asarray(cov, dtype=float)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def resample_error_model(
    isotopologues: list[dict[str, Any]],
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Perturb the constants by one draw from their full error covariance.

    ``sigma`` is MolecularOptimizer.constant_error_covariance(): MHz^2 over
    the stacked constants, isotopologue by isotopologue in component order --
    the measurement part plus the model error, correlated across
    isotopologues. Drawing from it, rather than from each constant's quoted
    sigma independently, makes the cloud a check of the covariance the engine
    reports: the same error model, propagated by refitting instead of
    linearly.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = sum(len(np.asarray(iso.get("obs_constants", []), dtype=float)) for iso in isotopologues)
    if sigma.shape != (n, n):
        raise ValueError(f"error covariance is {sigma.shape}, the constants are {n}")
    delta = symmetric_sqrt(sigma) @ rng.standard_normal(n)
    out: list[dict[str, Any]] = []
    k = 0
    for iso in isotopologues:
        entry = deepcopy(iso)
        obs = np.asarray(iso.get("obs_constants", []), dtype=float)
        entry["obs_constants"] = obs + delta[k:k + obs.size]
        k += obs.size
        out.append(entry)
    return out


def draw_quantum_prior_shift(optimizer, rng: np.random.Generator) -> np.ndarray:
    """One draw of the quantum prior's centre, as the gradient shift it makes.

    The fit's quantum term is alpha_q E_q with curvature alpha_q H, a prior of
    precision alpha_q H about its centre. Moving that centre by
    eta ~ N(0, (alpha_q H)^-1) changes E_q's gradient by -H eta, so
    xi = H eta ~ N(0, H / alpha_q), drawn on the physical directions (rigid
    and symmetry-forbidden ones excluded, as the posterior excludes them).
    Refitting with xi (MolecularOptimizer._rto_quantum_shift) and the
    constants drawn from their error model gives a posterior draw.
    """
    x = np.asarray(optimizer.coords, dtype=float)
    _g, Hp = optimizer._project_quantum_terms(optimizer.quantum.gradient,
                                              optimizer.quantum.hessian)
    Hs = np.asarray(optimizer.optimizer._spd_hessian(Hp), dtype=float)
    alpha = float(optimizer.quantum_prior_weight(refresh=True))
    if not (alpha > 0.0):
        raise ValueError("randomise-then-optimise needs a quantum prior (alpha_q > 0)")
    basis = optimizer.physical_basis(coords=x, n=Hs.shape[0])
    # (B S_b B^T) z over all 3N Cartesian directions, with S_b the symmetric
    # square root of the physical block: the same covariance B Hb B^T / alpha
    # whatever basis physical_basis returns for the subspace, so the draw
    # depends on the seed and the Hessian only.
    root = basis @ symmetric_sqrt(basis.T @ Hs @ basis / alpha) @ basis.T
    return root @ rng.standard_normal(Hs.shape[0])
