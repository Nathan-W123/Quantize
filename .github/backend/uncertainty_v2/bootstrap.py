from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import numpy as np

from backend.quantize import MolecularOptimizer

from .models import BootstrapConfig, BootstrapRunSummary, BootstrapSampleResult
from .resampling import resample_isotopologues_gaussian


def _run_bootstrap_worker(args: tuple[MolecularOptimizer, list[dict[str, Any]], str, int, int]) -> BootstrapSampleResult:
    base_opt, isotopologues, workdir, idx, seed = args
    start = time.time()
    opt = deepcopy(base_opt)
    opt.spectral.isotopologues = isotopologues
    opt.workdir = workdir
    os.makedirs(workdir, exist_ok=True)
    try:
        coords = opt.run()
        rms = float(opt.history[-1]["freq_rms"]) if opt.history else None
        return BootstrapSampleResult(
            idx=idx,
            seed=seed,
            success=True,
            coords=np.asarray(coords, dtype=float),
            rms_mhz=rms,
            workdir=workdir,
            elapsed_s=time.time() - start,
        )
    except Exception as exc:
        return BootstrapSampleResult(
            idx=idx,
            seed=seed,
            success=False,
            error=str(exc),
            workdir=workdir,
            elapsed_s=time.time() - start,
        )


class BootstrapEngine:
    """Phase-1 bootstrap orchestration with deterministic seeding and persistence."""

    def __init__(self, optimizer: MolecularOptimizer, config: BootstrapConfig):
        self.optimizer = optimizer
        self.config = config
        self.config.validate()

    def run(self, base_workdir: str | None = None) -> BootstrapRunSummary:
        base = os.path.abspath(base_workdir or self.optimizer.workdir or ".")
        run_dir = os.path.join(base, self.config.output_dir)
        os.makedirs(run_dir, exist_ok=True)

        print(
            f"\n[bootstrap-v2] Launching {self.config.n_samples} samples "
            f"(strategy={self.config.strategy}, workers={self.config.max_workers})..."
        )

        seed_seq = np.random.SeedSequence(self.config.seed)
        child_sequences = seed_seq.spawn(self.config.n_samples)
        jobs: list[tuple[MolecularOptimizer, list[dict[str, Any]], str, int, int]] = []

        for i, child in enumerate(child_sequences, start=1):
            seed = int(child.generate_state(1)[0])
            rng = np.random.default_rng(seed)
            if self.config.strategy == "isotopologue-gaussian":
                perturbed = resample_isotopologues_gaussian(self.optimizer.spectral.isotopologues, rng)
            else:
                raise ValueError(f"Unsupported bootstrap strategy: {self.config.strategy}")

            sample_dir = os.path.join(run_dir, f"sample_{i:03d}")
            jobs.append((self.optimizer, perturbed, sample_dir, i, seed))

        results: list[BootstrapSampleResult] = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = [pool.submit(_run_bootstrap_worker, job) for job in jobs]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res.success:
                    rms_label = "n/a" if res.rms_mhz is None else f"{res.rms_mhz:.4f} MHz"
                    print(f"  [sample {res.idx:03d}] Finished. RMS: {rms_label}")
                else:
                    print(f"  [sample {res.idx:03d}] FAILED: {res.error}")

        results.sort(key=lambda r: r.idx)
        success = [r for r in results if r.success]
        mean_rms = float(np.mean([r.rms_mhz for r in success if r.rms_mhz is not None])) if success else None
        summary = BootstrapRunSummary(
            config=self.config,
            total_samples=len(results),
            successful_samples=len(success),
            failed_samples=len(results) - len(success),
            mean_rms_mhz=mean_rms,
            sample_results=results,
        )

        if self.config.persist_samples:
            self._persist_summary(summary, run_dir)
        return summary

    @staticmethod
    def _persist_summary(summary: BootstrapRunSummary, run_dir: str) -> None:
        out_path = os.path.join(run_dir, "bootstrap_summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)

