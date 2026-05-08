import os
import time
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from backend.quantize import MolecularOptimizer

class BootstrapRunner:
    """Handles parallel re-optimization with perturbed experimental data."""
    def __init__(self, optimizer: MolecularOptimizer, n_samples=50, max_workers=1):
        self.base_opt = optimizer
        self.n_samples = n_samples
        self.max_workers = max_workers
        self.results = []

    def run(self, base_workdir="runs"):
        print(f"\n[bootstrap] Launching {self.n_samples} re-optimizations...")
        tasks = []
        for i in range(self.n_samples):
            iso_copy = []
            for iso in self.base_opt.spectral.isotopologues:
                entry = deepcopy(iso)
                # Perturb targets by experimental sigma
                sigma = np.asarray(iso.get("sigma_constants", [0.1, 0.1, 0.1]))
                noise = np.random.normal(0, sigma)
                entry["obs_constants"] = np.array(iso["obs_constants"]) + noise
                iso_copy.append(entry)
            
            wdir = os.path.join(base_workdir, "bootstrap", f"sample_{i+1:03d}")
            tasks.append((iso_copy, wdir, i+1))

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._worker, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res["success"]:
                    self.results.append(res)
                    print(f"  [sample {res['idx']:03d}] Finished. RMS: {res['rms']:.4f} MHz")
                else:
                    print(f"  [sample {res['idx']:03d}] FAILED: {res['error']}")
        return self.results

    def _worker(self, args):
        isos, wdir, idx = args
        opt = deepcopy(self.base_opt)
        opt.spectral.isotopologues = isos
        opt.workdir = wdir
        os.makedirs(wdir, exist_ok=True)
        try:
            coords = opt.run()
            return {
                "idx": idx, 
                "coords": coords, 
                "rms": opt.history[-1]["freq_rms"] if opt.history else 0.0, 
                "success": True
            }
        except Exception as e:
            return {"idx": idx, "success": False, "error": str(e)}

class AdaptiveMetropolisSampler:
    """
    MCMC Sampler for the Bayesian Posterior.
    Adjusts step size dynamically to maintain optimal acceptance rates (~23-44%).
    """
    def __init__(self, optimizer: MolecularOptimizer, walk_scale=0.005):
        self.opt = optimizer
        self.walk_scale = walk_scale
        self.samples = []
        self.log_p_history = []

    def run(self, n_steps=5000, burn_in=1000, tune_every=100):
        curr_coords = self.opt.coords.copy()
        curr_log_p = self.opt.log_probability(curr_coords, use_proxy=True)
        
        accepted = 0
        print(f"[mcmc] Sampling posterior for {n_steps} steps (scale={self.walk_scale})...")
        
        for i in range(1, n_steps + 1):
            # Propose move in Cartesian space
            proposal = curr_coords + np.random.normal(0, self.walk_scale, curr_coords.shape)
            
            try:
                prop_log_p = self.opt.log_probability(proposal, use_proxy=True)
                
                # Metropolis Acceptance Criterion
                if prop_log_p > curr_log_p or np.random.rand() < np.exp(prop_log_p - curr_log_p):
                    curr_coords = proposal
                    curr_log_p = prop_log_p
                    accepted += 1
            except Exception:
                pass # Reject unphysical geometries

            if i > burn_in:
                self.samples.append(curr_coords.copy())
                self.log_p_history.append(curr_log_p)

            # Adaptive tuning of the proposal width
            if i % tune_every == 0:
                rate = accepted / i
                if rate < 0.23: 
                    self.walk_scale *= 0.85
                elif rate > 0.44: 
                    self.walk_scale *= 1.15
                
            if i % (max(1, n_steps // 10)) == 0:
                print(f"  [step {i:5d}] Acceptance: {accepted/i:.1%} Scale: {self.walk_scale:.2e}")
        
        return np.array(self.samples)

def calculate_convergence_diagnostics(chains: np.ndarray):
    """
    Calculate R-hat (Gelman-Rubin) and Effective Sample Size (ESS).
    
    Parameters
    ----------
    chains : np.ndarray
        MCMC samples of shape (n_chains, n_samples, n_params).
    """
    if chains.ndim != 3:
        return None, None, "Invalid chain array shape. Expected (n_chains, n_samples, n_params)."

    n_chains, n_samples, n_params = chains.shape
    if n_chains < 2:
        return None, None, "Convergence diagnostics require at least 2 chains."
    if n_samples < 2:
        return None, None, "Convergence diagnostics require at least 2 post-burn-in samples per chain."
    if n_params < 1:
        return None, None, "No parameters available for convergence diagnostics."

    # 1. R-hat (Gelman-Rubin)
    # Measures the ratio of between-chain variance to within-chain variance.
    # Values near 1.0 indicate convergence.
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means, axis=0)
    
    B = (n_samples / (n_chains - 1)) * np.sum((chain_means - overall_mean)**2, axis=0)
    W = (1 / (n_chains * (n_samples - 1))) * np.sum((chains - chain_means[:, None, :])**2, axis=(0, 1))
    
    var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
    r_hat = np.sqrt(var_plus / np.maximum(W, 1e-12))

    # 2. ESS (Effective Sample Size)
    # Estimates the number of independent samples accounting for autocorrelation.
    ess_vals = []
    for p in range(n_params):
        param_chains = chains[:, :, p]
        rho_sum = 0.0
        for c in range(n_chains):
            c_data = param_chains[c]
            c_data = c_data - np.mean(c_data)
            # Lag-1 autocorrelation as a first-order proxy for ESS
            if np.var(c_data) > 1e-12:
                rho = np.corrcoef(c_data[:-1], c_data[1:])[0, 1]
                rho_sum += max(0, rho)
        
        rho_avg = rho_sum / n_chains
        # n_eff = N / (1 + 2 * sum(rho_k))
        ess_vals.append((n_chains * n_samples) / (1 + 2 * rho_avg))

    return r_hat, np.array(ess_vals), None
