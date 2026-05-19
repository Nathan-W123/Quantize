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

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "mean_rms_mhz": self.mean_rms_mhz,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_params": self.n_params,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "regularization": self.regularization,
            "std_err": np.asarray(self.std_err, dtype=float).tolist(),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "warnings": list(self.warnings),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "chains": [c.to_dict() for c in self.chains],
            "convergence": None if self.convergence is None else self.convergence.to_dict(),
        }
