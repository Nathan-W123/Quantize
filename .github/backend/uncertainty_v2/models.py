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
    #: What each refit draws, and so what the cloud estimates:
    #:   "error-model"  the constants from the engine's own error model
    #:                  (constant_error_covariance: measurement part plus the
    #:                  model error, correlated across isotopologues). The
    #:                  cloud is the spread the data and the model error alone
    #:                  produce -- the reported covariance less its prior part.
    #:   "rto"          the same, plus the quantum prior's centre drawn from
    #:                  the prior (randomise-then-optimise). The cloud is the
    #:                  posterior: a sampling check of the reported covariance.
    #:   "isotopologue-gaussian"  the legacy draw: each constant independently
    #:                  by its quoted sigma_constants, whatever that contains.
    strategy: str = "error-model"
    output_dir: str = "bootstrap_v2"
    persist_samples: bool = True
    #: A sample is a draw only if its refit reached its own optimum: it is
    #: refused when it ends further than this many posterior widths from it
    #: (distance_to_mode, the largest Newton-step component in that
    #: direction's own width). For a cloud narrower than the posterior --
    #: every strategy but "rto" -- the engine scales it by the cloud's width
    #: over the posterior's, so it is stated in the cloud's own widths.
    max_mode_distance: float = 0.25

    STRATEGIES = ("error-model", "rto", "isotopologue-gaussian")

    def validate(self) -> None:
        if self.n_samples < 1:
            raise ValueError("BootstrapConfig.n_samples must be >= 1.")
        if self.max_workers < 1:
            raise ValueError("BootstrapConfig.max_workers must be >= 1.")
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Unsupported bootstrap strategy: {self.strategy}")
        if not (self.max_mode_distance > 0.0):
            raise ValueError("BootstrapConfig.max_mode_distance must be > 0.")


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
    #: Why the refit's loop ended: "converged", "plateau", "stall" or
    #: "max_iter" (MolecularOptimizer.termination).
    termination: str | None = None
    #: The refit's distance to its own optimum in posterior widths, measured
    #: in the worker on the sample's own objective: the Newton distance to
    #: the posterior's mode (distance_to_mode) under the joint objective,
    #: and under split how far one more run of the step moves it, since the
    #: split step's fixed point is not the posterior's mode (see
    #: ``certificate``).
    mode_distance: float | None = None
    #: How mode_distance was measured: "gradient" or "restart".
    certificate: str | None = None
    #: Whether the refit ended fitting the model it started with: the drawn
    #: constants, the B0 -> Be targets built on them, their sigmas and the
    #: effective sigma. A correction rebuild during the refit used to
    #: overwrite the constants with the unperturbed ones, so the "sample"
    #: refitted the base data.
    draw_kept: bool | None = None
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
    #: The base fit's residual RMS, for information. Samples are no longer
    #: gated on it: perturbed data fit worse by construction, and the ratio
    #: refused draws that had reached their own optimum (25 of 48 on the
    #: water config, 39 of 48 on formyl fluoride).
    base_rms_mhz: float | None = None
    #: What the cloud estimates, and the engine's analytic value of it beside
    #: the reported posterior: {"target", "names", "kinds", "units",
    #: "target_std", "posterior_std", "bootstrap_std"} (bonds in Angstrom,
    #: angles in degrees), "objective", under split "null_response",
    #: "split_note" and (off its mode) "mode_shift", and with a cloud "check"
    #: ({"n", "ratio", "mc_sd", "kurtosis", "impact", "departs", "sigmas",
    #: "impact_min"}: cloud over analytic width per coordinate, its Monte
    #: Carlo error from the cloud's kurtosis, the change it implies for the
    #: reported width, and where it departs). None when it could not be formed.
    analytic: dict | None = None

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
            f"from what is left would not describe the draws that were made."
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
            "analytic": self.analytic,
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
    #: Of those, how many were removed because the imposed point group forbids
    #: them rather than because they are rigid-body motions. Counted inside
    #: n_gauge_directions so n_params - n_gauge_directions stays the size of
    #: the reported block; reported apart because a symmetry assumption is a
    #: modelling choice, not a gauge freedom.
    n_symmetry_directions: int = 0

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
            "n_symmetry_directions": self.n_symmetry_directions,
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
