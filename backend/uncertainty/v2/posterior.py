from __future__ import annotations

import numpy as np

from backend.quantize import MolecularOptimizer

from .models import LaplaceSummary, PosteriorState


def evaluate_posterior_state(
    optimizer: MolecularOptimizer,
    coords: np.ndarray,
    *,
    use_proxy: bool = True,
) -> PosteriorState:
    """Evaluate posterior decomposition at a geometry."""
    trial_coords = np.asarray(coords, dtype=float)

    _, residual_w = optimizer.spectral.stacked(trial_coords)
    chi_sq_spectral = float(np.sum(residual_w ** 2))

    chi_sq_prior = 0.0
    if optimizer.internal_prior is not None and optimizer.coordinate_mode == "cartesian":
        _, rp = optimizer.internal_prior.stacked(trial_coords)
        chi_sq_prior = float(optimizer.prior_weight * np.sum(rp ** 2))

    energy_term = 0.0
    if not optimizer.spectral_only and optimizer.quantum is not None:
        if use_proxy:
            dx = (trial_coords - optimizer._orca_ref_coords).ravel()
            g = optimizer.quantum.gradient
            h = optimizer.quantum.hessian
            energy_term = float(optimizer.quantum.energy + np.dot(g, dx) + 0.5 * np.dot(dx, h @ dx))
        else:
            energy_term = float(optimizer.quantum.energy)

    log_likelihood = -0.5 * chi_sq_spectral
    log_prior = -0.5 * chi_sq_prior - float(optimizer.optimizer.alpha_quantum * energy_term)
    log_posterior = log_likelihood + log_prior

    return PosteriorState(
        coords=trial_coords,
        chi_sq_spectral=chi_sq_spectral,
        chi_sq_prior=chi_sq_prior,
        energy_term=energy_term,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        log_posterior=log_posterior,
    )


def laplace_approximation(
    optimizer: MolecularOptimizer,
    *,
    coords: np.ndarray | None = None,
    regularization: float = 1e-8,
) -> LaplaceSummary:
    """Build Laplace covariance approximation around an optimum geometry."""
    x = optimizer.coords if coords is None else np.asarray(coords, dtype=float)
    jac, _ = optimizer.spectral.stacked(x)
    h_total = jac.T @ jac

    if not optimizer.spectral_only and optimizer.quantum is not None:
        h_total = h_total + float(optimizer.optimizer.alpha_quantum) * np.asarray(optimizer.quantum.hessian, dtype=float)

    reg = float(max(regularization, 0.0))
    h_reg = h_total + reg * np.eye(h_total.shape[0], dtype=float)
    cov = np.linalg.pinv(h_reg)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))

    rank = int(np.linalg.matrix_rank(h_reg))
    cond = float(np.linalg.cond(h_reg))
    warnings: list[str] = []
    if rank < h_reg.shape[0]:
        warnings.append("Rank-deficient Hessian/JTJ; covariance may be weakly identified.")
    if not np.isfinite(cond) or cond > 1e12:
        warnings.append("Ill-conditioned Hessian/JTJ; uncertainties may be unstable.")

    return LaplaceSummary(
        n_params=int(h_reg.shape[0]),
        rank=rank,
        condition_number=cond,
        regularization=reg,
        std_err=np.asarray(std, dtype=float),
        covariance=np.asarray(cov, dtype=float),
        warnings=warnings,
    )

