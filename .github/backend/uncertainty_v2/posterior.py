from __future__ import annotations

import numpy as np

from backend.covariance_checks import std_err_from_covariance

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
    # ONE IMPLEMENTATION, NOT TWO. This used to rebuild the posterior
    # covariance itself: jac.T @ jac, the quantum block weighted by
    # effective_quantum_weight, _spd_hessian, and a 1e-8 ridge -- the same
    # fifteen lines as MolecularOptimizer.laplace_approximation, reaching into
    # that class's private _project_quantum_terms and _spd_hessian to do it.
    #
    # The two agreed to about 5.8e-6 relative, which is not a difference of
    # method: the only divergence was inv() here against pinv() there, and at
    # cond(H) = 2.4e11 with full rank that is LU-versus-SVD round-off amplified
    # by the conditioning. The reported standard errors agreed to seven
    # figures. So nothing was wrong with the numbers -- the hazard was that the
    # same logic lived twice, and the sqrt(max(diag, 0)) defect had to be found
    # and fixed in both copies independently.
    #
    # The method owns the calculation now, because it owns the state it needs.
    cov, _ = optimizer.laplace_approximation(
        coords=coords, regularization=regularization)
    if cov is None:
        return LaplaceSummary(
            n_params=0, rank=0, condition_number=float("inf"),
            regularization=float(max(regularization, 0.0)),
            std_err=np.zeros(0), covariance=np.zeros((0, 0)),
            warnings=["Posterior covariance could not be formed; "
                      "see the optimizer's own warning for why."],
        )
    # GAUGE DIRECTIONS OUT BEFORE ANY WIDTH IS REPORTED.
    #
    # This is what the report was getting wrong, and it was not subtle.
    # Translations and rotations are null in both blocks of the objective, so
    # the only thing giving them a finite width is the ridge. On water that
    # width is sigma = 112 Angstrom, on a 0.958 Angstrom bond, and
    # format_laplace_summary printed it as "max std err" -- 13,400 times the
    # largest physical width (8.3e-3 Angstrom), with no warning, because the
    # ridge also makes matrix_rank report 9 of 9 and puts cond at 2.4e11,
    # just under the 1e12 threshold. Two reassuring diagnostics either side
    # of a number wrong by four orders of magnitude.
    #
    # geometry_uncertainty always projected first. This path did not, which is
    # the entire defect; both now go through optimizer.physical_covariance.
    n_full = int(cov.shape[0])
    cov, basis, n_gauge = optimizer.physical_covariance(cov, coords=coords)
    std, _cov_info = std_err_from_covariance(
        cov, context="uncertainty_v2.posterior (via MolecularOptimizer)")

    # Rank and condition over the PHYSICAL block, via the projector's own
    # basis. Taken on the padded matrix instead they would report nullity
    # n_gauge and cond = inf on every run -- both true, neither informative,
    # since that deficiency is the projection working rather than a problem
    # with the fit. What matters is whether the directions that survive are
    # themselves identified.
    #
    # Reporting these from the covariance rather than from the Hessian it came
    # from is not a looser check: for an invertible H the 2-norm condition
    # number is the same number, cond(H^-1) = cond(H), both being
    # s_max/s_min of the same singular values, and inversion preserves rank.
    # Checked over 500 random symmetric matrices with spectra spanning
    # 1e-12..1e2: the two agree to 1.5e-5 relative at worst, which is inv()'s
    # own round-off at cond 1e14, not a difference in what is measured.
    block = basis.T @ cov @ basis
    n_phys = int(block.shape[0])
    rank = int(np.linalg.matrix_rank(block)) if n_phys else 0
    cond = float(np.linalg.cond(block)) if n_phys else float("inf")

    warnings: list[str] = []
    if rank < n_phys:
        warnings.append(
            f"Rank-deficient posterior: only {rank} of {n_phys} physical "
            f"directions are identified, so the rest are as wide as the "
            f"prior and the ridge make them, not as wide as the data says.")
    if not np.isfinite(cond) or cond > 1e12:
        warnings.append(
            f"Ill-conditioned posterior over the physical directions "
            f"(cond = {cond:.2e}); treat these widths as order-of-magnitude.")
    if not _cov_info["ok"]:
        warnings.append(
            f"{_cov_info['n_negative']} variance(s) came out negative, so "
            f"those standard errors are nan rather than numbers.")

    # Whether the Gauss-Newton Hessian these widths come from is valid at all.
    # The optimizer emits these to its own warning channel; they belong on the
    # summary too, because this is the object that gets serialised and read
    # later, when nobody is watching stdout.
    warnings.extend(optimizer.gauss_newton_warnings(coords=coords))

    return LaplaceSummary(
        n_params=n_full,
        rank=rank,
        condition_number=cond,
        regularization=float(max(regularization, 0.0)),
        std_err=np.asarray(std, dtype=float),
        covariance=np.asarray(cov, dtype=float),
        warnings=warnings,
        n_gauge_directions=n_gauge,
    )

