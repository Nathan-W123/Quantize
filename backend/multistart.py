import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from backend.quantize import MolecularOptimizer


def _spectral_isotopologue_snapshot(opt):
    """Copy optimizer spectral targets after run (e.g. ORCA rovib-updated alphas)."""
    snap = []
    for iso in opt.spectral.isotopologues:
        snap.append(
            {
                "name": str(iso["name"]),
                "masses": np.asarray(iso["masses"], dtype=float).tolist(),
                "obs_constants": np.asarray(iso["obs_constants"], dtype=float).tolist(),
                "sigma_constants": np.asarray(iso["sigma_constants"], dtype=float).tolist(),
                "alpha_constants": np.asarray(iso["alpha_constants"], dtype=float).tolist(),
                "component_indices": np.asarray(iso["component_indices"], dtype=int).tolist(),
                "torsion_sensitive": bool(iso.get("torsion_sensitive", False)),
            }
        )
    return snap


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
    _e = last.get("energy", np.inf)
    _fr = last.get("freq_rms", np.inf)
    return {
        "idx": idx,
        "coords": final_coords,
        "start_coords": np.asarray(coords, dtype=float).copy(),
        "freq_rms": float(_fr if _fr is not None else np.inf),
        "energy": float(_e if _e is not None else np.inf),
        "history": opt.history,
        "workdir": start_dir,
        "spectral_isotopologues_snapshot": _spectral_isotopologue_snapshot(opt),
    }


def run_multistart(starts, elems, isotopologues, optimizer_kwargs, max_workers=1, job_name="job"):
    """
    Parallel workers run independent ORCA/Psi4 jobs. Many ORCA installations (especially
    academic licenses) allow only **one** ORCA process at a time; parallel runs then exit
    without writing ``.engrad`` / ``.hess``. By default we force ``max_workers=1`` when
    ``quantum_backend='orca'``. Set ``QUANTIZE_ALLOW_PARALLEL_ORCA=1`` to keep preset parallelism.
    """
    kw = optimizer_kwargs or {}
    qb = str(kw.get("quantum_backend", "")).strip().lower()
    spectral_only = bool(kw.get("spectral_only", False))
    allow_parallel_orca = os.environ.get("QUANTIZE_ALLOW_PARALLEL_ORCA", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if qb == "orca" and not spectral_only and max_workers > 1 and not allow_parallel_orca:
        print(
            "[multistart] quantum_backend=orca: using max_workers=1 (parallel ORCA often fails "
            "with single-seat licenses). Export QUANTIZE_ALLOW_PARALLEL_ORCA=1 to allow "
            f"parallel workers (requested {max_workers})."
        )
        max_workers = 1

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


def multistart_seed_metrics(result):
    """
    Per-start diagnostics: Cartesian RMS displacement from `start_coords`, and
    change in reported spectral RMS from first to last optimizer history entry.

    Returns
    -------
    dict with keys:
        coord_rms_disp_ang, spectral_gain_mhz, freq_rms_first, freq_rms_last
    """
    start = result.get("start_coords")
    final = np.asarray(result.get("coords"), dtype=float)
    if start is not None:
        start = np.asarray(start, dtype=float)
        coord_rms = float(np.sqrt(np.mean((final - start) ** 2)))
    else:
        coord_rms = float("nan")
    hist = result.get("history") or []
    if len(hist) >= 1:
        f_first = float(hist[0].get("freq_rms", np.inf))
        f_last = float(hist[-1].get("freq_rms", np.inf))
        gain = f_first - f_last
    else:
        f_first = f_last = float(result.get("freq_rms", np.inf))
        gain = 0.0
    return {
        "coord_rms_disp_ang": coord_rms,
        "spectral_gain_mhz": gain,
        "freq_rms_first": f_first,
        "freq_rms_last": f_last,
    }


def _effective_spectral_rms_for_selection(
    result,
    penalize_stagnant,
    stagnant_coord_rms_tol_ang,
    stagnant_max_spectral_gain_mhz,
    stagnant_penalty_mhz,
):
    """Internal rank key: penalize runs that barely moved and barely improved the spectrum."""
    base = float(result.get("freq_rms", np.inf))
    if not penalize_stagnant:
        return base
    m = multistart_seed_metrics(result)
    cr = m["coord_rms_disp_ang"]
    gain = m["spectral_gain_mhz"]
    if not np.isfinite(cr):
        return base
    stagnant = (cr < float(stagnant_coord_rms_tol_ang)) and (
        gain < float(stagnant_max_spectral_gain_mhz)
    )
    return base + (float(stagnant_penalty_mhz) if stagnant else 0.0)


def rank_key_spectral_value(
    result,
    *,
    penalize_stagnant=False,
    stagnant_coord_rms_tol_ang=5e-6,
    stagnant_max_spectral_gain_mhz=0.05,
    stagnant_penalty_mhz=500.0,
):
    """
    Effective spectral RMS used by ``select_best_result(..., primary_objective='spectral')``.
    Exposed for printing/debugging alongside ``multistart_seed_metrics``.
    """
    return _effective_spectral_rms_for_selection(
        result,
        penalize_stagnant,
        stagnant_coord_rms_tol_ang,
        stagnant_max_spectral_gain_mhz,
        stagnant_penalty_mhz,
    )


def select_best_result(
    results,
    spectral_gate_abs=0.01,
    spectral_gate_rel=2.0,
    *,
    primary_objective="energy",
    penalize_stagnant=False,
    stagnant_coord_rms_tol_ang=5e-6,
    stagnant_max_spectral_gain_mhz=0.05,
    stagnant_penalty_mhz=500.0,
):
    """
    Select best multistart result.

    Parameters
    ----------
    primary_objective : {"energy", "spectral"}
        * ``energy`` (default): within the spectral gate, pick **lowest ORCA energy**
          (original behaviour).
        * ``spectral``: pick **lowest effective spectral RMS** (optionally after
          ``penalize_stagnant``), then lowest energy as tie-breaker.

    penalize_stagnant : bool
        If True, add ``stagnant_penalty_mhz`` to the comparison RMS when the run
        moved less than ``stagnant_coord_rms_tol_ang`` Å (RMS over atoms) **and**
        improved spectral RMS by less than ``stagnant_max_spectral_gain_mhz`` from
        the first to the last history record — i.e. reward starts that actually
        explore geometry or improve the fit.

    Strategy (spectral gate unchanged):
    1) Find best spectral RMS across starts.
    2) Keep starts with freq_rms <= max(spectral_gate_abs, spectral_gate_rel * best_freq_rms).
    3) Rank within the gated set using ``primary_objective`` / stagnation penalty.
    """
    if not results:
        raise ValueError("No multistart results to select from.")
    best_freq = min(float(r.get("freq_rms", np.inf)) for r in results)
    gate = max(float(spectral_gate_abs), float(spectral_gate_rel) * best_freq)
    gated = [r for r in results if float(r.get("freq_rms", np.inf)) <= gate]
    if not gated:
        gated = list(results)

    po = str(primary_objective).strip().lower()
    if po == "spectral":

        def sort_key(r):
            eff = _effective_spectral_rms_for_selection(
                r,
                penalize_stagnant,
                stagnant_coord_rms_tol_ang,
                stagnant_max_spectral_gain_mhz,
                stagnant_penalty_mhz,
            )
            e = float(r.get("energy", np.inf))
            if not np.isfinite(e):
                e = np.inf
            return (eff, e)

        return min(gated, key=sort_key)

    # Legacy: lowest energy within gate, then spectral RMS fallback.
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
    Compute hybrid-inversion success metrics.

    Designed for underconstrained problems where spectral data alone does not
    fully determine the geometry — the quantum prior fills in the null space.
    Low spectral rank is expected and is NOT penalized.

    Score components (sum to 100):
      50% — multi-start geometric stability  (do all starts agree?)
      35% — spectral fit quality             (RMS vs experimental sigma)
      15% — quantum null-space quality       (is the null-space at an energy minimum?)

    Multiplied by health penalties for guardrail violations and energy spikes.
    """
    if not results:
        raise ValueError("No multistart results available.")
    if best_result is None:
        raise ValueError("best_result is required.")
    if internal_dof is None:
        internal_dof = _infer_internal_dof(best_result["coords"])
    internal_dof = max(int(internal_dof), 1)

    history = best_result.get("history", [])
    last = history[-1] if history else {}
    constrained_rank = int(last.get("rank", 0))
    rank_fraction = min(1.0, max(0.0, constrained_rank / float(internal_dof)))

    # ── Multi-start geometric stability ───────────────────────────────────────
    # How consistent is the recovered geometry across independent starts?
    # Tolerances: 0.01 Å for bonds, 0.2° for angles (last metric assumed angular).
    metric_arrays = [np.asarray(r.get("metrics", []), dtype=float) for r in results if "metrics" in r]
    if metric_arrays:
        all_metrics = np.vstack(metric_arrays)
        metric_std = np.std(all_metrics, axis=0)
    else:
        metric_std = np.array([], dtype=float)
    if metric_std.size > 0:
        denom = np.full(metric_std.shape, 0.01, dtype=float)
        denom[-1] = 0.2
        stability_penalty = float(np.linalg.norm(metric_std / denom))
        stability_score = float(np.exp(-stability_penalty))
    else:
        stability_score = 0.5

    # ── Spectral fit quality ───────────────────────────────────────────────────
    # How well does the final geometry reproduce observed rotational constants?
    # Measured in units of experimental sigma — fit within ~5σ scores near 1.
    sigma_all = np.concatenate([np.asarray(iso["sigma_constants"], dtype=float) for iso in isotopologues])
    sigma_scale = float(np.sqrt(np.mean(sigma_all ** 2))) if sigma_all.size else 1.0
    freq_rms = float(best_result.get("freq_rms", np.inf))
    sigma_ratio = freq_rms / max(sigma_scale, 1e-12)
    spectral_agreement_score = float(np.exp(-sigma_ratio / 5.0))

    # ── Quantum null-space quality ─────────────────────────────────────────────
    # For directions invisible to the spectral data, the quantum prior steers
    # the geometry toward the energy minimum. A small final null-space gradient
    # norm (|g_n|) means the null-space directions are properly converged.
    # We use the median |g_n| over the last 20% of iterations as the signal.
    g_null_vals = [float(h["g_null_norm"]) for h in history if h.get("g_null_norm") is not None and np.isfinite(float(h["g_null_norm"]))]
    if g_null_vals:
        tail = g_null_vals[max(0, len(g_null_vals) - max(1, len(g_null_vals) // 5)):]
        g_null_final = float(np.median(tail))
        quantum_quality_score = float(np.exp(-g_null_final / 0.1))
    else:
        quantum_quality_score = 0.5

    # ── Health penalties (multiplicative) ─────────────────────────────────────
    guardrail_violation_rate = 0.0
    if history:
        flags = [1.0 if int(h.get("guardrail_violations", 0)) > 0 else 0.0 for h in history]
        guardrail_violation_rate = float(np.mean(flags))
    guardrail_score = float(np.exp(-4.0 * guardrail_violation_rate))

    delta_e = [float(h["delta_energy"]) for h in history if h.get("delta_energy") is not None and np.isfinite(float(h["delta_energy"]))]
    if delta_e:
        e_scale = float(np.median(np.abs(delta_e)) + 1e-12)
        e_spike = float(np.max(np.abs(delta_e)) / e_scale)
        energy_stability_score = float(np.exp(-0.2 * max(0.0, e_spike - 1.0)))
    else:
        energy_stability_score = 1.0

    prior_wrms = [float(h["prior_wrms"]) for h in history if h.get("prior_wrms") is not None]
    prior_stability_score = 1.0
    if prior_wrms:
        p_med = float(np.median(prior_wrms))
        p_span = float(max(prior_wrms) - min(prior_wrms))
        prior_stability_score = float(np.exp(-p_span / max(1.0, p_med)))

    # ── Final score ───────────────────────────────────────────────────────────
    score = 100.0 * (
        0.50 * stability_score +
        0.35 * spectral_agreement_score +
        0.15 * quantum_quality_score
    )
    score *= (0.85 + 0.15 * guardrail_score)
    score *= (0.90 + 0.10 * energy_stability_score)
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
        "quantum_quality_score": float(quantum_quality_score),
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
