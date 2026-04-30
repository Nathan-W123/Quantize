import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from quantize import MolecularOptimizer


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
