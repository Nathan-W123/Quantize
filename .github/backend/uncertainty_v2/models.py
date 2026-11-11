from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Runtime configuration for bootstrap uncertainty execution."""

    n_samples: int = 50
    max_workers: int = 1
    seed: int | None = None
    strategy: str = "isotopologue-gaussian"
    output_dir: str = "bootstrap_v2"
    persist_samples: bool = True

    def validate(self) -> None:
        if self.n_samples < 1:
            raise ValueError("BootstrapConfig.n_samples must be >= 1.")
        if self.max_workers < 1:
            raise ValueError("BootstrapConfig.max_workers must be >= 1.")
        if self.strategy not in {"isotopologue-gaussian"}:
            raise ValueError(f"Unsupported bootstrap strategy: {self.strategy}")


@dataclass
class BootstrapSampleResult:
    """Single bootstrap job output."""

    idx: int
    seed: int
    success: bool
    coords: np.ndarray | None = None
    rms_mhz: float | None = None
    error: str | None = None
    workdir: str | None = None
    elapsed_s: float | None = None
    #: How far this sample's geometry moved from the geometry it started at,
    #: as max|dx| in Angstrom. Exactly 0.0 means the optimiser accepted no
    #: step at all, so the "sample" is a copy of the starting point.
    moved_ang: float | None = None
    #: Whether this sample may enter the cloud an interval is taken from.
    #: `success` only says the fit did not raise; these are different facts
    #: and were conflated, which is the defect these fields exist for.
    usable: bool = True
    unusable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        blob = asdict(self)
        if self.coords is not None:
            blob["coords"] = np.asarray(self.coords, dtype=float).tolist()
        return blob


@dataclass
class BootstrapRunSummary:
    """Aggregate bootstrap run metadata and outcomes."""

    config: BootstrapConfig
    total_samples: int
    successful_samples: int
    failed_samples: int
    mean_rms_mhz: float | None
    sample_results: list[BootstrapSampleResult] = field(default_factory=list)
    #: The base fit's residual RMS the samples were gated against, when the
    #: optimizer had been run before sampling. None means only the
    #: no-accepted-step gate could be applied.
    base_rms_mhz: float | None = None

    #: Fewer usable samples than this fraction of the requested count and the
    #: cloud is not a sample of anything. The CLI already used one half as the
    #: line for "high failure rate suggests the geometry is unstable or poorly
    #: constrained"; this applies the same line to samples that ran but did
    #: not produce a draw.
    USABLE_FRACTION_MIN = 0.5

    @property
    def usable_results(self) -> list[BootstrapSampleResult]:
        """The samples an interval may be quoted from."""
        return [r for r in self.sample_results
                if r.success and r.usable and r.coords is not None]

    @property
    def usable_samples(self) -> int:
        return len(self.usable_results)

    @property
    def refused_samples(self) -> int:
        """Samples that ran without raising but are not draws."""
        return sum(1 for r in self.sample_results if r.success and not r.usable)

    @property
    def usable(self) -> bool:
        """Whether an interval may be quoted from this run at all.

        False is a refusal, not a hedge. A sample that accepted no step is a
        copy of the starting geometry, and a cloud padded with copies of its
        own centre reports a narrower interval the more often the sampler
        fails -- the error points the wrong way, so it cannot be left to a
        warning.
        """
        n = int(self.config.n_samples) if self.config is not None else self.total_samples
        if n <= 0:
            return False
        return self.usable_samples >= max(1, int(np.ceil(self.USABLE_FRACTION_MIN * n)))

    def unusable_reason(self) -> str | None:
        """Why an interval is refused, or None when it is not."""
        if self.usable:
            return None
        n = int(self.config.n_samples) if self.config is not None else self.total_samples
        need = max(1, int(np.ceil(self.USABLE_FRACTION_MIN * n)))
        return (
            f"Only {self.usable_samples} of {n} bootstrap samples produced a "
            f"draw ({self.refused_samples} ran but were refused, "
            f"{self.failed_samples} raised); {need} are needed. An interval "
            f"from what is left would be narrower than the truth, because a "
            f"refused sample sits exactly on the starting geometry."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "mean_rms_mhz": self.mean_rms_mhz,
            "base_rms_mhz": self.base_rms_mhz,
            "usable_samples": self.usable_samples,
            "refused_samples": self.refused_samples,
            "usable": self.usable,
            "unusable_reason": self.unusable_reason(),
            "sample_results": [s.to_dict() for s in self.sample_results],
        }


@dataclass
class PosteriorState:
    """Posterior decomposition at a geometry."""

    coords: np.ndarray
    chi_sq_spectral: float
    chi_sq_prior: float
    energy_term: float
    log_likelihood: float
    log_prior: float
    log_posterior: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": np.asarray(self.coords, dtype=float).tolist(),
            "chi_sq_spectral": self.chi_sq_spectral,
            "chi_sq_prior": self.chi_sq_prior,
            "energy_term": self.energy_term,
            "log_likelihood": self.log_likelihood,
            "log_prior": self.log_prior,
            "log_posterior": self.log_posterior,
        }


@dataclass
class LaplaceSummary:
    """Laplace posterior approximation at the optimum."""

    n_params: int
    rank: int
    condition_number: float
    regularization: float
    std_err: np.ndarray
    covariance: np.ndarray
    warnings: list[str] = field(default_factory=list)
    # Rigid-body directions removed before any width was reported. `rank` and
    # `condition_number` describe the remaining n_params - n_gauge_directions
    # physical directions, not the padded matrix: the nullity is structural,
    # so reporting it as rank deficiency or as cond = inf says nothing.
    n_gauge_directions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_params": self.n_params,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "regularization": self.regularization,
            "std_err": np.asarray(self.std_err, dtype=float).tolist(),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "warnings": list(self.warnings),
            "n_gauge_directions": self.n_gauge_directions,
        }


@dataclass(frozen=True)
class MCMCConfig:
    """Runtime configuration for MCMC sampling."""

    n_steps: int = 5000
    burn_in: int = 1000
    thin: int = 1
    n_chains: int = 4
    proposal_scale: float = 0.005
    adapt_every: int = 100
    target_accept_low: float = 0.23
    target_accept_high: float = 0.44
    seed: int | None = None
    checkpoint_every: int = 1000
    persist_chain_samples: bool = True
    output_dir: str = "mcmc_v2"

    def validate(self) -> None:
        if self.n_steps < 2:
            raise ValueError("MCMCConfig.n_steps must be >= 2.")
        if self.burn_in < 0 or self.burn_in >= self.n_steps:
            raise ValueError("MCMCConfig.burn_in must be >= 0 and < n_steps.")
        if self.thin < 1:
            raise ValueError("MCMCConfig.thin must be >= 1.")
        if self.n_chains < 1:
            raise ValueError("MCMCConfig.n_chains must be >= 1.")
        if self.proposal_scale <= 0.0:
            raise ValueError("MCMCConfig.proposal_scale must be > 0.")
        if self.adapt_every < 1:
            raise ValueError("MCMCConfig.adapt_every must be >= 1.")
        if self.checkpoint_every < 1:
            raise ValueError("MCMCConfig.checkpoint_every must be >= 1.")


@dataclass
class ChainRunResult:
    chain_idx: int
    seed: int
    samples: np.ndarray
    log_posterior: np.ndarray
    acceptance_rate: float
    final_scale: float
    n_proposals: int
    n_accepted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_idx": self.chain_idx,
            "seed": self.seed,
            "acceptance_rate": self.acceptance_rate,
            "final_scale": self.final_scale,
            "n_proposals": self.n_proposals,
            "n_accepted": self.n_accepted,
            "n_samples": int(self.samples.shape[0]),
        }


@dataclass
class ConvergenceSummary:
    """Convergence diagnostics over q-space chains."""

    r_hat: np.ndarray
    ess: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_hat": np.asarray(self.r_hat, dtype=float).tolist(),
            "ess": np.asarray(self.ess, dtype=float).tolist(),
            "warnings": list(self.warnings),
        }


@dataclass
class MCMCRunSummary:
    config: MCMCConfig
    chains: list[ChainRunResult]
    convergence: ConvergenceSummary | None = None

    #: Above this, the chains have not mixed and no interval taken from them
    #: means anything. 1.1 is the conventional line (Gelman et al.); measured
    #: here, this sampler returns 2.3 to 18 on water depending on the seed,
    #: because the exact posterior is improper along one direction -- see
    #: MolecularOptimizer.gauss_newton_discrepancy. A number that moves by 8x
    #: with the seed is not a diagnostic to print and move past.
    R_HAT_USABLE_MAX = 1.1
    #: Below this there are too few independent draws for a quantile to mean
    #: anything, whatever R-hat says.
    ESS_USABLE_MIN = 100.0

    @property
    def usable(self) -> bool:
        """Whether intervals may be quoted from these chains at all.

        False is not a hedge, it is a refusal: quoting a credible interval
        from unmixed chains reports the sampler's starting spread as if it
        were the posterior's width.
        """
        if self.convergence is None:
            return False
        try:
            return bool(
                float(self.convergence.r_hat.max()) <= self.R_HAT_USABLE_MAX
                and float(self.convergence.ess.min()) >= self.ESS_USABLE_MIN)
        except (ValueError, AttributeError):
            return False

    def unusable_reason(self) -> str | None:
        """Why intervals are refused, or None when they are not."""
        if self.convergence is None:
            return ("No convergence diagnostics were computed (fewer than two "
                    "chains, or too few kept samples), so there is no evidence "
                    "these chains mixed.")
        r_hat = float(self.convergence.r_hat.max())
        ess = float(self.convergence.ess.min())
        bad = []
        if r_hat > self.R_HAT_USABLE_MAX:
            bad.append(f"max R-hat {r_hat:.2f} exceeds {self.R_HAT_USABLE_MAX}")
        if ess < self.ESS_USABLE_MIN:
            bad.append(f"min ESS {ess:.0f} is below {self.ESS_USABLE_MIN:.0f}")
        if not bad:
            return None
        return (" and ".join(bad) + ". The chains have not mixed, so any "
                "interval from them measures where they started rather than "
                "the posterior's width.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "chains": [c.to_dict() for c in self.chains],
            "convergence": None if self.convergence is None else self.convergence.to_dict(),
            "usable": self.usable,
            "unusable_reason": self.unusable_reason(),
        }
