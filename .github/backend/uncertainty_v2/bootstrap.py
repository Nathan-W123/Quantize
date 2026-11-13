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
from .resampling import (
    draw_quantum_prior_shift,
    resample_error_model,
    resample_isotopologues_gaussian,
)


def _freeze_corrections(opt: MolecularOptimizer) -> None:
    """Hold the refit to the model the reported covariance was formed with.

    The corrections. A Hessian refresh during a refit used to run
    _apply_harmonic_alpha_corrections, which rebuilds the targets from the
    UNPERTURBED raw isotopologues and writes them over the drawn ones: the
    sample then refitted the base data. On the water config 51% of draws
    were wiped this way, almost all on the bond-shortening side, so one tail
    of the cloud went missing. Frozen, every refit fits the constants it was
    given, with the base's corrections, and refits are cheaper.

    The quantum reference. The reported covariance, the target printed
    beside the cloud and the rto shift xi are all built from the base's
    quadratic model of the surface: its last gradient, Hessian and weight.
    A refit that drifted past orca_update_thresh used to refresh the
    gradient (and on schedule the Hessian and weight), so it fitted a model
    none of those describe. That Hessian is the one from the base fit's
    start, and the real slope differs from it (formyl fluoride C-H secant
    curvature 1.13x the stored one), so live rto came out 7% narrow on
    C-H against the analytic column, the water bond 2.6% narrow, and the
    water error-model angle 2.2% wide, with the refits refreshing the
    quantum point 0.6 times each at 92% of their cost. Held on the base's
    model the refits reproduce the linear prediction of their own draws to
    within 2.2% on every coordinate, at the cost of an error-model refit.
    Whether that quadratic describes the surface near the fit is a property
    of the base fit, not something a refit can check.
    """
    opt._harmonic_from_hessian = False
    opt._anharmonic_from_hessian = False
    opt._harmonic_cd_from_hessian = False
    opt.use_orca_rovib = False
    opt.orca_update_thresh = float("inf")


#: A refit that stopped further than this from its own optimum, in posterior
#: widths, is continued from where it stopped, at most _POLISH_PASSES times.
#: The plateau stop ends a fit on one small step, which is loose for a refit
#: that has a full posterior width to travel: under "rto" on water 5% of
#: refits stopped beyond 0.1 widths short, up to 0.2, which narrows the cloud
#: (the bond came out 4% narrow). Refusing them instead would remove exactly
#: the draws that travelled furthest.
_POLISH_DISTANCE = 0.02
_POLISH_PASSES = 3


def _distance(opt, coords):
    try:
        return float(opt.distance_to_mode(coords=coords))
    except Exception:
        return None


def _run_bootstrap_worker(args: tuple) -> BootstrapSampleResult:
    base_opt, isotopologues, workdir, idx, seed = args[:5]
    rto_shift = args[5] if len(args) > 5 else None
    start = time.time()
    opt = deepcopy(base_opt)
    _freeze_corrections(opt)
    opt.spectral.isotopologues = isotopologues
    opt._rto_quantum_shift = rto_shift
    opt.workdir = workdir
    os.makedirs(workdir, exist_ok=True)
    start_coords = np.asarray(opt.coords, dtype=float).copy()
    drawn = [np.asarray(iso["obs_constants"], dtype=float).copy() for iso in isotopologues]
    try:
        coords = opt.run()
        rms = float(opt.history[-1]["freq_rms"]) if opt.history else None
        coords = np.asarray(coords, dtype=float)
        # Measured in the worker, where the starting geometry is in hand.
        # Exactly 0.0 means run() accepted no step, so this "sample" is the
        # starting point again -- see BootstrapRunSummary.usable.
        moved = float(np.abs(coords - start_coords).max())
        kept = all(
            np.allclose(np.asarray(iso["obs_constants"], dtype=float), d,
                        rtol=0.0, atol=1e-9 * max(1.0, float(np.max(np.abs(d)))))
            for iso, d in zip(opt.spectral.isotopologues, drawn))
        dist = _distance(opt, coords)
        passes = 0
        while (dist is not None and dist > _POLISH_DISTANCE and passes < _POLISH_PASSES
               and (getattr(opt, "termination", None) or {}).get("reason") != "max_iter"):
            coords = np.asarray(opt.run(), dtype=float)
            dist = _distance(opt, coords)
            passes += 1
        rms = float(opt.history[-1]["freq_rms"]) if opt.history else rms
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
            termination=(getattr(opt, "termination", None) or {}).get("reason"),
            mode_distance=dist,
            draw_kept=bool(kept),
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
    """Refit perturbed copies of a fitted optimizer, and say what the spread is.

    The analytic answer is the reported covariance
    (MolecularOptimizer.internal_coordinate_posterior), which costs
    microseconds. This is a check of it by refitting: "error-model" draws the
    constants from the error model that covariance propagates, so its cloud
    is the reported covariance less its prior part; "rto" also draws the
    quantum prior's centre, so its cloud is the reported covariance itself.
    Each run carries the analytic value of its own target beside the cloud,
    so a disagreement is visible rather than inferred.

    Every refit is held on the base's model (_freeze_corrections): its
    corrections and its quadratic model of the quantum surface. What the
    cloud tests is therefore the linearisation of the data side and the
    error model, which is what the analytic column assumes; it does not test
    whether the base's Hessian describes the surface near the fit.
    """

    def __init__(self, optimizer: MolecularOptimizer, config: BootstrapConfig):
        self.optimizer = optimizer
        self.config = config
        self.config.validate()

    def _check_strategy(self) -> None:
        opt = self.optimizer
        if self.config.strategy != "rto":
            return
        if opt.spectral_only or opt.quantum is None:
            raise ValueError("strategy 'rto' needs a quantum prior to randomise")
        if getattr(opt, "_prior_target_coords", None) is not None:
            raise ValueError(
                "strategy 'rto' does not support prior_target_coords: the prior "
                "target recomputes the quantum gradient, so the shift would be lost")
        anchored = opt.internal_prior is not None and opt.coordinate_mode == "cartesian"
        if opt._internal_prior_rows(opt.coords) is not None or anchored:
            raise ValueError(
                "strategy 'rto' randomises the data and the quantum prior only; an "
                "internal-coordinate prior is active and would not be sampled")

    def run(self, base_workdir: str | None = None) -> BootstrapRunSummary:
        base = os.path.abspath(base_workdir or self.optimizer.workdir or ".")
        run_dir = os.path.join(base, self.config.output_dir)
        os.makedirs(run_dir, exist_ok=True)
        self._check_strategy()

        print(
            f"\n[bootstrap-v2] Launching {self.config.n_samples} samples "
            f"(strategy={self.config.strategy}, workers={self.config.max_workers})..."
        )

        seed_seq = np.random.SeedSequence(self.config.seed)
        child_sequences = seed_seq.spawn(self.config.n_samples)
        error_cov = None
        if self.config.strategy in ("error-model", "rto"):
            error_cov = self.optimizer.constant_error_covariance()
        jobs: list[tuple] = []

        for i, child in enumerate(child_sequences, start=1):
            seed = int(child.generate_state(1)[0])
            rng = np.random.default_rng(seed)
            shift = None
            if self.config.strategy == "isotopologue-gaussian":
                perturbed = resample_isotopologues_gaussian(self.optimizer.spectral.isotopologues, rng)
            elif self.config.strategy in ("error-model", "rto"):
                perturbed = resample_error_model(self.optimizer.spectral.isotopologues, error_cov, rng)
                if self.config.strategy == "rto":
                    shift = draw_quantum_prior_shift(self.optimizer, rng)
            else:
                raise ValueError(f"Unsupported bootstrap strategy: {self.config.strategy}")

            sample_dir = os.path.join(run_dir, f"sample_{i:03d}")
            jobs.append((self.optimizer, perturbed, sample_dir, i, seed, shift))

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
        try:
            summary.analytic = self.analytic_reference(results)
        except Exception as exc:  # a comparison must not cost the run
            print(f"  [bootstrap-v2] analytic reference unavailable: {type(exc).__name__}: {exc}")
        for r in results:
            if r.success and not r.usable:
                print(f"  [sample {r.idx:03d}] REFUSED: {r.unusable_reason}")

        if self.config.persist_samples:
            self._persist_summary(summary, run_dir)
        return summary

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
        base_rms: float | None = None,
    ) -> None:
        """Mark which samples may enter the cloud, and why the rest may not.

        A sample is a draw when its refit fitted the constants it was given
        and reached its own optimum. In order:

        1. It raised, or returned no geometry.
        2. ``moved_ang == 0.0``: no step was accepted, so the geometry is the
           start. Measured on water with four samples, two came back
           bit-identical to the base geometry and were counted as successes.
        3. The drawn constants were not the targets it ended with -- a
           correction rebuild overwrote them -- so it refitted other data.
        4. Its loop ran out of iterations.
        5. It stopped more than ``max_mode_distance`` posterior widths from
           its own optimum (distance_to_mode on the sample's own objective),
           or that distance could not be measured on a fit that did not
           report convergence.

        This replaces a gate on the final residual RMS relative to the base
        fit's. Perturbed constants fit worse by construction, so the ratio
        measured the draw, not the fit: it refused 25 of 48 draws on the
        water config and 39 of 48 on formyl fluoride, most of which sat at
        their own optimum, and kept the ones that happened to fit well.
        """
        limit = float(self.config.max_mode_distance)
        for r in results:
            r.usable = False
            if not r.success:
                r.unusable_reason = "the fit raised"
                continue
            if r.coords is None:
                r.unusable_reason = "no geometry was returned"
                continue
            if r.moved_ang is not None and r.moved_ang == 0.0:
                r.unusable_reason = (
                    "the optimiser accepted no step, so this geometry is the "
                    "starting point rather than a draw")
                continue
            if r.draw_kept is False:
                r.unusable_reason = (
                    "the drawn constants were replaced during the refit, so it "
                    "fitted other data")
                continue
            if r.termination == "max_iter":
                r.unusable_reason = "the refit ran out of iterations"
                continue
            if r.mode_distance is None:
                if r.termination not in (None, "converged"):
                    r.unusable_reason = (
                        "its distance to its own optimum could not be measured "
                        f"and the refit ended on '{r.termination}'")
                    continue
            elif r.mode_distance > limit:
                r.unusable_reason = (
                    f"the refit stopped {r.mode_distance:.2f} posterior widths "
                    f"from its own optimum (limit {limit:g})")
                continue
            r.usable = True
            r.unusable_reason = None

    def analytic_reference(self, results=None) -> dict:
        """The engine's analytic value of what this cloud estimates.

        Per internal coordinate, at the base geometry: the target standard
        deviation (the reported covariance less its prior part for
        "error-model"; the reported covariance for "rto"; the sandwich of
        independent quoted-sigma draws for "isotopologue-gaussian"), the
        reported posterior's, and the cloud's own when samples are given.
        Bonds in Angstrom, angles in degrees.
        """
        from backend.internal.internal_fit import InternalCoordinateSet

        opt = self.optimizer
        x = np.asarray(opt.coords, dtype=float)
        cov, _ = opt.laplace_approximation(coords=x, check_gauss_newton=False)
        cov, basis, _n = opt.physical_covariance(cov, coords=x)
        J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
        c_sys = opt._systematic_covariance(J, cov, coords=x, row_scale=scale)
        c_total = cov + c_sys
        strategy = self.config.strategy
        if strategy == "rto":
            target_cov = c_total
            label = "posterior (reported covariance)"
        elif strategy == "error-model":
            _hs, h_prior, _parts = opt._posterior_blocks(x)
            target_cov = c_total - cov @ h_prior @ cov
            label = "data and model error alone (reported covariance less its prior part)"
        else:
            sig = np.concatenate([np.asarray(iso["sigma_constants"], dtype=float).ravel()
                                  for iso in opt.spectral.isotopologues])
            n_rot = sig.size
            A = cov @ np.asarray(J, dtype=float)[:n_rot].T
            target_cov = (A * (np.asarray(scale, dtype=float) * sig) ** 2) @ A.T
            label = "independent quoted-sigma draws (legacy)"
        cs = InternalCoordinateSet(x, opt.elems, opt._ic_use_dihedrals)
        Bq = np.asarray(cs.active_B_matrix(x), dtype=float)
        kinds = [ic.kind for ic in cs.active_coords()]
        conv = np.array([1.0 if k == "bond" else np.degrees(1.0) for k in kinds])

        def sd(C):
            return (np.sqrt(np.clip(np.diag(Bq @ C @ Bq.T), 0.0, None)) * conv).tolist()

        out = {"target": label, "names": list(cs.active_names()), "kinds": kinds,
               "units": ["angstrom" if k == "bond" else "degree" for k in kinds],
               "target_std": sd(target_cov), "posterior_std": sd(c_total),
               "bootstrap_std": None}
        use = [r for r in (results or []) if r.success and r.usable and r.coords is not None]
        if len(use) > 2:
            q = np.array([np.asarray(cs.active_values(np.asarray(r.coords, dtype=float)), dtype=float)
                          for r in use])
            out["bootstrap_std"] = (q.std(axis=0, ddof=1) * conv).tolist()
        return out

    @staticmethod
    def _persist_summary(summary: BootstrapRunSummary, run_dir: str) -> None:
        out_path = os.path.join(run_dir, "bootstrap_summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)
