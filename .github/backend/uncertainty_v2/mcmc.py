from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from backend.internal.internal_fit import InternalCoordinateSet
from backend.quantize import MolecularOptimizer

from .models import ChainRunResult, ConvergenceSummary, MCMCConfig, MCMCRunSummary


def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x0 = np.asarray(x, dtype=float) - float(np.mean(x))
    var = float(np.var(x0))
    if var <= 1e-16:
        return np.zeros(max_lag + 1, dtype=float)
    ac = np.zeros(max_lag + 1, dtype=float)
    ac[0] = 1.0
    n = x0.size
    for lag in range(1, max_lag + 1):
        ac[lag] = float(np.dot(x0[:-lag], x0[lag:]) / ((n - lag) * var))
    return ac


def _ess_for_param(chains_2d: np.ndarray) -> float:
    # chains_2d shape: (m, n)
    m, n = chains_2d.shape
    if n < 3:
        return float(m * n)
    max_lag = min(n - 1, 200)
    rho = np.zeros(max_lag + 1, dtype=float)
    for c in range(m):
        rho += _autocorr_1d(chains_2d[c], max_lag)
    rho /= max(m, 1)

    # Initial positive sequence using pairwise lag sums.
    tau = 1.0
    for k in range(1, max_lag, 2):
        pair_sum = rho[k] + rho[k + 1]
        if pair_sum <= 0.0:
            break
        tau += 2.0 * pair_sum
    tau = max(tau, 1.0)
    return float((m * n) / tau)


def compute_split_rhat_ess(chains: np.ndarray) -> ConvergenceSummary | None:
    """
    Split-Rhat and multi-lag ESS for chains shaped (n_chains, n_samples, n_params).
    """
    if chains.ndim != 3:
        return None
    m, n, p = chains.shape
    if m < 2 or n < 4 or p < 1:
        return None

    half = n // 2
    if half < 2:
        return None
    split = np.concatenate([chains[:, :half, :], chains[:, half : 2 * half, :]], axis=0)
    m2, n2, _ = split.shape

    chain_means = np.mean(split, axis=1)
    mean_all = np.mean(chain_means, axis=0)
    b = (n2 / (m2 - 1)) * np.sum((chain_means - mean_all) ** 2, axis=0)
    w = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    var_hat = ((n2 - 1) / n2) * w + (1.0 / n2) * b
    rhat = np.sqrt(var_hat / np.maximum(w, 1e-16))

    ess = np.zeros(p, dtype=float)
    for i in range(p):
        ess[i] = _ess_for_param(split[:, :, i])

    warnings: list[str] = []
    if np.any(rhat > 1.1):
        warnings.append("One or more parameters have R-hat > 1.1.")
    if np.any(ess < 100):
        warnings.append("One or more parameters have ESS < 100.")
    return ConvergenceSummary(r_hat=rhat, ess=ess, warnings=warnings)


class AdaptiveMetropolisV2:
    def __init__(self, optimizer: MolecularOptimizer, config: MCMCConfig):
        self.optimizer = optimizer
        self.config = config
        self.config.validate()

    def run(self, base_workdir: str | None = None) -> MCMCRunSummary:
        base = os.path.abspath(base_workdir or self.optimizer.workdir or ".")
        out_dir = os.path.join(base, self.config.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        seeds = np.random.SeedSequence(self.config.seed).spawn(self.config.n_chains)
        chain_results: list[ChainRunResult] = []
        for chain_idx, ss in enumerate(seeds, start=1):
            seed = int(ss.generate_state(1)[0])
            rng = np.random.default_rng(seed)
            chain = self._run_single_chain(chain_idx, seed, rng, out_dir)
            chain_results.append(chain)

        # Build q-space tensors for convergence diagnostics.
        sample_counts = [c.samples.shape[0] for c in chain_results]
        n_min = min(sample_counts) if sample_counts else 0
        convergence = None
        if len(chain_results) >= 2 and n_min >= 4:
            ic_set = InternalCoordinateSet(self.optimizer.coords, self.optimizer.elems)
            q_chains = []
            for c in chain_results:
                clipped = c.samples[:n_min]
                q_chains.append(np.array([ic_set.active_values(xyz) for xyz in clipped]))
            convergence = compute_split_rhat_ess(np.array(q_chains))

        summary = MCMCRunSummary(config=self.config, chains=chain_results, convergence=convergence)
        self._persist_summary(summary, out_dir)
        return summary

    def _run_single_chain(
        self,
        chain_idx: int,
        seed: int,
        rng: np.random.Generator,
        out_dir: str,
    ) -> ChainRunResult:
        x_curr = np.asarray(self.optimizer.coords, dtype=float).copy()
        # Overdispersed starts around optimum for chain diversity.
        x_curr = x_curr + rng.normal(0.0, self.config.proposal_scale * 5.0, size=x_curr.shape)
        lp_curr = float(self.optimizer.log_probability(x_curr, use_proxy=True))

        scale = float(self.config.proposal_scale)
        accepted = 0
        # Proposals whose log-probability could not be evaluated. Kept apart
        # from rejections so they never enter the acceptance rate.
        failed = 0
        first_failure: str | None = None
        kept_samples: list[np.ndarray] = []
        kept_lp: list[float] = []

        for step in range(1, self.config.n_steps + 1):
            proposal = x_curr + rng.normal(0.0, scale, size=x_curr.shape)
            try:
                lp_prop = float(self.optimizer.log_probability(proposal, use_proxy=True))
                dlp = lp_prop - lp_curr
                if dlp >= 0.0 or rng.random() < np.exp(dlp):
                    x_curr = proposal
                    lp_curr = lp_prop
                    accepted += 1
            except Exception as exc:
                # A proposal that raises is NOT a rejection. Counting it as one
                # depresses the measured acceptance rate, which drives the
                # adaptation below to shrink the step scale, which biases the
                # sampled width and so the reported uncertainties. Track them
                # separately and exclude them from the rate.
                failed += 1
                if first_failure is None:
                    first_failure = repr(exc)

            if step % self.config.adapt_every == 0 and step <= self.config.burn_in:
                evaluated = max(step - failed, 1)
                acc_rate = accepted / evaluated
                if acc_rate < self.config.target_accept_low:
                    scale *= 0.85
                elif acc_rate > self.config.target_accept_high:
                    scale *= 1.15

            if step > self.config.burn_in and ((step - self.config.burn_in) % self.config.thin == 0):
                kept_samples.append(x_curr.copy())
                kept_lp.append(lp_curr)

            if step % self.config.checkpoint_every == 0:
                self._write_chain_checkpoint(out_dir, chain_idx, step,
                                             accepted / max(step - failed, 1),
                                             scale, len(kept_samples))

        samples = np.array(kept_samples, dtype=float)
        lp = np.array(kept_lp, dtype=float)
        n_props = self.config.n_steps - failed
        acc_rate = accepted / max(n_props, 1)
        print(
            f"  [chain {chain_idx:02d}] acceptance={acc_rate:.1%} "
            f"final_scale={scale:.2e} kept={samples.shape[0]}"
        )
        if failed:
            print(
                f"  [chain {chain_idx:02d}] WARNING: {failed} of "
                f"{self.config.n_steps} proposals could not be evaluated and "
                f"were excluded from the acceptance rate. First failure: "
                f"{first_failure}"
            )
        if self.config.persist_chain_samples:
            np.save(os.path.join(out_dir, f"chain_{chain_idx:02d}_samples.npy"), samples)
            np.save(os.path.join(out_dir, f"chain_{chain_idx:02d}_logp.npy"), lp)

        return ChainRunResult(
            chain_idx=chain_idx,
            seed=seed,
            samples=samples,
            log_posterior=lp,
            acceptance_rate=acc_rate,
            final_scale=scale,
            n_proposals=n_props,
            n_accepted=accepted,
        )

    @staticmethod
    def _write_chain_checkpoint(
        out_dir: str,
        chain_idx: int,
        step: int,
        acceptance_rate: float,
        scale: float,
        kept: int,
    ) -> None:
        path = os.path.join(out_dir, f"chain_{chain_idx:02d}_checkpoint.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "chain_idx": chain_idx,
                    "step": step,
                    "acceptance_rate": acceptance_rate,
                    "proposal_scale": scale,
                    "kept_samples": kept,
                },
                fh,
                indent=2,
            )

    @staticmethod
    def _persist_summary(summary: MCMCRunSummary, out_dir: str) -> None:
        with open(os.path.join(out_dir, "mcmc_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)
        with open(os.path.join(out_dir, "mcmc_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(summary.config), fh, indent=2)

