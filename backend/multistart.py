import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from backend.quantize import MolecularOptimizer


def _run_one_start(payload):
    idx = payload["idx"]
    coords = np.asarray(payload["coords"], dtype=float)
    elems = payload["elems"]
    isotopologues = payload["isotopologues"]
    kwargs = dict(payload["optimizer_kwargs"])

    base_workdir = kwargs.pop("base_workdir", ".")
    job_name = kwargs.pop("job_name", "multistart")
    start_dir = os.path.join(base_workdir, "multistart_runs", job_name, f"start_{idx:02d}")
    os.makedirs(start_dir, exist_ok=True)
    kwargs["workdir"] = start_dir
    kwargs.setdefault("psi4_output_file", os.path.join(start_dir, "psi4.out"))

    opt = MolecularOptimizer(coords=coords, elems=elems, isotopologues=isotopologues, **kwargs)
    final_coords = opt.run()
    last = opt.history[-1] if opt.history else {}
    return {
        "idx": idx,
        "coords": final_coords,
        "freq_rms": float(last.get("freq_rms", np.inf)),
        "energy": float(last.get("energy", np.inf)),
        "history": opt.history,
        "workdir": start_dir,
    }


def run_multistart(starts, elems, isotopologues, optimizer_kwargs, max_workers=1, job_name="job"):
    payloads = [
        {
            "idx": i + 1,
            "coords": np.asarray(c, dtype=float),
            "elems": list(elems),
            "isotopologues": isotopologues,
            "optimizer_kwargs": {**optimizer_kwargs, "job_name": job_name},
        }
        for i, c in enumerate(starts)
    ]
    if max_workers <= 1:
        return [_run_one_start(p) for p in payloads]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_run_one_start, p): p["idx"] for p in payloads}
        for fut in as_completed(fut_map):
            results.append(fut.result())
    results.sort(key=lambda r: r["idx"])
    return results


def select_best_result(results, spectral_gate_abs=0.01, spectral_gate_rel=2.0):
    """
    Select best result using a spectral-quality gate followed by lowest energy.

    Strategy:
    1) Find best spectral RMS across starts.
    2) Keep starts with freq_rms <= max(spectral_gate_abs, spectral_gate_rel * best_freq_rms).
    3) Pick the lowest-energy run within that gated set.
    4) Fallback to spectral-first sort if energies are unavailable/non-finite.
    """
    if not results:
        raise ValueError("No multistart results to select from.")
    best_freq = min(float(r.get("freq_rms", np.inf)) for r in results)
    gate = max(float(spectral_gate_abs), float(spectral_gate_rel) * best_freq)
    gated = [r for r in results if float(r.get("freq_rms", np.inf)) <= gate]
    if not gated:
        gated = list(results)
    finite_energy = [r for r in gated if np.isfinite(float(r.get("energy", np.inf)))]
    if finite_energy:
        return min(finite_energy, key=lambda r: float(r.get("energy", np.inf)))
    return min(gated, key=lambda r: float(r.get("freq_rms", np.inf)))


def _infer_internal_dof(coords):
    """Estimate internal DOF from geometry (3N-6 nonlinear, 3N-5 linear)."""
    arr = np.asarray(coords, dtype=float)
    n_atoms = int(arr.shape[0])
    if n_atoms <= 1:
        return 0
    if n_atoms == 2:
        return 1
    centered = arr - arr.mean(axis=0, keepdims=True)
    geom_rank = int(np.linalg.matrix_rank(centered, tol=1e-8))
    linear = geom_rank <= 1
    dof = 3 * n_atoms - (5 if linear else 6)
    return max(dof, 1)


def underconstrained_success_score(results, best_result, isotopologues, internal_dof=None):
    """
    Compute rank-aware success metrics for underconstrained inversion.
    Returns a dict with scores and diagnostic scalars.
    """
    if not results:
        raise ValueError("No multistart results available.")
    if best_result is None:
        raise ValueError("best_result is required.")
    if internal_dof is None:
        internal_dof = _infer_internal_dof(best_result["coords"])
    internal_dof = max(int(internal_dof), 1)

    last = best_result["history"][-1] if best_result.get("history") else {}
    constrained_rank = int(last.get("rank", 0))
    rank_fraction = min(1.0, max(0.0, constrained_rank / float(internal_dof)))

    metric_arrays = [np.asarray(r.get("metrics", []), dtype=float) for r in results if "metrics" in r]
    if metric_arrays:
        all_metrics = np.vstack(metric_arrays)
        metric_std = np.std(all_metrics, axis=0)
    else:
        metric_std = np.array([], dtype=float)
    if metric_std.size > 0:
        # Normalize by loose chemistry-level tolerances (0.01 A or 0.2 deg typical).
        # Unknown metric types are conservatively treated like 0.01-scale lengths.
        denom = np.full(metric_std.shape, 0.01, dtype=float)
        if denom.size > 0:
            denom[-1] = 0.2
        stability_penalty = float(np.linalg.norm(metric_std / denom))
        stability_score = float(np.exp(-stability_penalty))
    else:
        stability_score = 0.5

    sigma_all = np.concatenate([np.asarray(iso["sigma_constants"], dtype=float) for iso in isotopologues])
    sigma_scale = float(np.sqrt(np.mean(sigma_all ** 2))) if sigma_all.size else 1.0
    freq_rms = float(best_result.get("freq_rms", np.inf))
    sigma_ratio = freq_rms / max(sigma_scale, 1e-12)
    spectral_agreement_score = float(np.exp(-sigma_ratio / 5.0))

    # Penalize repeated geometry guardrail violations (if available in history).
    history = best_result.get("history", [])
    guardrail_violation_rate = 0.0
    if history:
        flags = []
        for h in history:
            viol = int(h.get("guardrail_violations", 0))
            flags.append(1.0 if viol > 0 else 0.0)
        guardrail_violation_rate = float(np.mean(flags))
    guardrail_score = float(np.exp(-4.0 * guardrail_violation_rate))

    # Penalize unstable energy trajectories (large |dE| spikes).
    delta_e = []
    for h in history:
        de = h.get("delta_energy", None)
        if de is not None and np.isfinite(float(de)):
            delta_e.append(float(de))
    if delta_e:
        e_scale = float(np.median(np.abs(delta_e)) + 1e-12)
        e_spike = float(np.max(np.abs(delta_e)) / e_scale)
        energy_stability_score = float(np.exp(-0.2 * max(0.0, e_spike - 1.0)))
    else:
        energy_stability_score = 1.0

    prior_wrms = [float(h.get("prior_wrms")) for h in history if h.get("prior_wrms") is not None]
    prior_stability_score = 1.0
    if prior_wrms:
        p_med = float(np.median(prior_wrms))
        p_span = float(max(prior_wrms) - min(prior_wrms))
        prior_stability_score = float(np.exp(-p_span / max(1.0, p_med)))

    score = 100.0 * (
        0.45 * stability_score +
        0.35 * rank_fraction +
        0.20 * spectral_agreement_score
    )
    score *= (0.85 + 0.15 * guardrail_score) * (0.90 + 0.10 * energy_stability_score)
    score *= (0.95 + 0.05 * prior_stability_score)

    conf_weight_stability = 1.0
    conf_entries = [h.get("conformer_weights") for h in history if h.get("conformer_weights") is not None]
    if conf_entries:
        arr = np.asarray(conf_entries, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1:
            std = np.std(arr, axis=0)
            conf_weight_stability = float(np.exp(-5.0 * float(np.mean(std))))
        score *= (0.95 + 0.05 * conf_weight_stability)
    return {
        "score": float(score),
        "constrained_rank": constrained_rank,
        "internal_dof": internal_dof,
        "rank_fraction": float(rank_fraction),
        "stability_score": float(stability_score),
        "spectral_agreement_score": float(spectral_agreement_score),
        "sigma_scale": sigma_scale,
        "sigma_ratio": float(sigma_ratio),
        "guardrail_violation_rate": float(guardrail_violation_rate),
        "guardrail_score": float(guardrail_score),
        "energy_stability_score": float(energy_stability_score),
        "prior_stability_score": float(prior_stability_score),
        "prior_wrms_median": float(np.median(prior_wrms)) if prior_wrms else None,
        "conformer_weight_stability": float(conf_weight_stability),
        "conformer_weights_final": (
            [float(v) for v in np.asarray(conf_entries[-1], dtype=float)] if conf_entries else None
        ),
        "metric_std": metric_std.tolist(),
    }
