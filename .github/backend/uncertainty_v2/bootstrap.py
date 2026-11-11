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
    start_coords = np.asarray(opt.coords, dtype=float).copy()
    try:
        coords = opt.run()
        rms = float(opt.history[-1]["freq_rms"]) if opt.history else None
        coords = np.asarray(coords, dtype=float)
        # Measured in the worker, where the starting geometry is in hand.
        # Exactly 0.0 means run() accepted no step, so this "sample" is the
        # starting point again -- see BootstrapRunSummary.usable.
        moved = float(np.abs(coords - start_coords).max())
        return BootstrapSampleResult(
            idx=idx,
            seed=seed,
            success=True,
            coords=coords,
            rms_mhz=rms,
            workdir=workdir,
            elapsed_s=time.time() - start,
            moved_ang=moved,
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
        base_rms = self._base_rms_mhz()
        self._gate_samples(results, base_rms)
        summary = BootstrapRunSummary(
            config=self.config,
            total_samples=len(results),
            successful_samples=len(success),
            failed_samples=len(results) - len(success),
            mean_rms_mhz=mean_rms,
            sample_results=results,
            base_rms_mhz=base_rms,
        )
        for r in results:
            if r.success and not r.usable:
                print(f"  [sample {r.idx:03d}] REFUSED: {r.unusable_reason}")

        if self.config.persist_samples:
            self._persist_summary(summary, run_dir)
        return summary

    #: A sample is a draw from the posterior only if its own fit converged.
    #: These are select_best_result's gate, as run_generic passes it when
    #: choosing among multistart results -- the repository's existing answer to
    #: "is this fit good enough to be considered", reused here rather than a
    #: second threshold that could drift away from it.
    SPECTRAL_GATE_REL = 2.0
    SPECTRAL_GATE_ABS = 0.05

    def _base_rms_mhz(self) -> float | None:
        """The base fit's residual RMS, when the optimizer was run first."""
        hist = getattr(self.optimizer, "history", None) or []
        if not hist:
            return None
        try:
            return float(hist[-1]["freq_rms"])
        except (KeyError, TypeError, ValueError):
            return None

    def _gate_samples(
        self,
        results: list[BootstrapSampleResult],
        base_rms: float | None,
    ) -> None:
        """Mark which samples may enter the cloud, and why the rest may not.

        Two refusals, in order of how certain they are:

        1. ``moved_ang == 0.0``. The optimiser accepted no step, so the
           returned geometry is the one it was handed. Measured on water with
           four samples: two of four came back bit-identical to the base
           geometry (max|dx| = 0.000e+00), were reported as successes, and put
           a duplicate of the distribution's own centre into the cloud. This
           needs no threshold and admits no argument.
        2. The sample's final residual RMS is outside select_best_result's
           spectral gate around the base fit's. A sample whose fit ended 3.9x
           worse than the base fit (54,925 MHz against 14,061 MHz, measured on
           the same run) did not find its own optimum, so its geometry is
           wherever the optimiser gave up, not where the perturbed data put the
           minimum.
        """
        for r in results:
            if not r.success:
                r.usable = False
                r.unusable_reason = "the fit raised"
                continue
            if r.coords is None:
                r.usable = False
                r.unusable_reason = "no geometry was returned"
                continue
            if r.moved_ang is not None and r.moved_ang == 0.0:
                r.usable = False
                r.unusable_reason = (
                    "the optimiser accepted no step, so this geometry is the "
                    "starting point rather than a draw")
                continue
            if base_rms is not None and r.rms_mhz is not None:
                gate = max(base_rms * self.SPECTRAL_GATE_REL,
                           base_rms + self.SPECTRAL_GATE_ABS)
                if float(r.rms_mhz) > gate:
                    ratio = float(r.rms_mhz) / base_rms if base_rms else float("inf")
                    r.usable = False
                    r.unusable_reason = (
                        f"final residual RMS {r.rms_mhz:.1f} MHz is {ratio:.1f}x "
                        f"the base fit's {base_rms:.1f} MHz, outside the "
                        f"{self.SPECTRAL_GATE_REL:g}x spectral gate, so this "
                        f"fit did not reach its own minimum")
                    continue
            r.usable = True
            r.unusable_reason = None

    @staticmethod
    def _persist_summary(summary: BootstrapRunSummary, run_dir: str) -> None:
        out_path = os.path.join(run_dir, "bootstrap_summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)

