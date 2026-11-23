from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np

from backend.quantize import MolecularOptimizer

from .models import BootstrapConfig, BootstrapRunSummary, BootstrapSampleResult, json_ready
from .resampling import (
    quantum_prior_shift_sampler,
    resample_cd_observed,
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
    model the refits reproduce the linear prediction of their own draws --
    0.976-1.004 over formyl fluoride's six coordinates and two seeds,
    1.000-1.010 on the water config -- at the cost of an error-model refit.
    A coordinate the prior dominates can stray further when the model-error
    draws are large, from the data side's own nonlinearity (one-constant
    water, joint: angle 1.12).
    Whether that quadratic describes the surface near the fit is a property
    of the base fit, not something a refit can check.

    Under the joint objective only. The split step's quantum gradient is
    Pn (g + H dx), while the split posterior's energy has gradient
    Pn (g + H Pn dx); they agree only while dx, the distance from the
    reference, stays small, which the refreshes ensure. Held, error-model
    refits under split stopped a median 0.18 widths from their optimum by
    the posterior's measure and 4 of 12 were refused (12 of 12 usable,
    median 0.004, with the refreshes), on water with one constant.

    The targets and the weighting. The draw was made with the base's sigmas
    (already rescaled, when chi2_rescale fired on the base), so a refit
    that rescaled itself fitted with sigmas the error model does not
    describe: one formyl fluoride refit in 40 widened all twelve by 2.1x
    and was admitted. And the rm1 mass-dependent model refits its B0 -> Be
    offsets at every iteration, which the reported covariance and the
    convergence certificate both hold fixed: such refits all ended on
    max_iter, and continued to a plateau they moved a further 0.04-0.32
    widths. Held at the base's offsets, a refit has one objective.
    """
    opt._harmonic_from_hessian = False
    opt._anharmonic_from_hessian = False
    opt._harmonic_cd_from_hessian = False
    opt.use_orca_rovib = False
    if opt.optimizer.objective_mode == "joint":
        opt.orca_update_thresh = float("inf")
    opt.chi2_rescale = False
    opt._mass_dependent_model = None


def _refit_model(opt) -> list:
    """What a refit must still be fitting when it ends, per isotopologue:
    the drawn constants, the B0 -> Be target built on them, the sigmas and
    the effective sigma the weighting makes of them (and the structural
    prior's weight when that prior is in the objective). Anything that
    rewrites one of these mid-refit -- a correction rebuild, a mass-dependent
    refit, a chi-square rescale, an adaptive sigma floor -- makes the sample
    a draw from another model."""
    sp = opt.spectral
    out = []
    for iso in sp.isotopologues:
        sig = np.asarray(iso["sigma_constants"], dtype=float)
        out.append(np.asarray(iso["obs_constants"], dtype=float).copy())
        out.append(np.asarray(sp._be_target(iso), dtype=float).copy())
        out.append(sig.copy())
        out.append(np.asarray(sp._effective_sigma(sig, iso=iso), dtype=float).copy())
    if opt.internal_prior is not None and opt.coordinate_mode == "cartesian":
        out.append(np.array([float(opt.prior_weight)]))
    return out


def _same_model(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        x.shape == y.shape and np.allclose(
            x, y, rtol=0.0, atol=1e-9 * max(1.0, float(np.max(np.abs(x))) if x.size else 1.0))
        for x, y in zip(a, b))


#: A refit that stopped further than this from its own optimum, in posterior
#: widths (scaled to the cloud's own widths, see BootstrapEngine._gate_limits),
#: is continued from where it stopped, at most _POLISH_PASSES times, whatever
#: ended its loop. A safety net: on the shipped settings every refit measured
#: (about 10,000 on water and formyl fluoride) ended within 5e-4 widths, so it
#: does not fire. It does when a fit is stopped early: with the plateau stop
#: loosened, 18 of 20 water refits were polished, at most 3 passes each, and
#: a restart confirmed the ones it then refused. Refusing without polishing
#: would remove exactly the draws that travelled furthest.
_POLISH_DISTANCE = 0.02
_POLISH_PASSES = 3
#: A coordinate whose cloud width is under this fraction of its posterior
#: width is held by the prior, and the cloud does not resolve it, so it does
#: not set the gate's scale.
_RESOLVED_FRACTION = 0.02


def _distance(opt, coords):
    try:
        return float(opt.distance_to_mode(coords=coords))
    except Exception:
        return None


#: A restart step at this fraction of the trust radius it was taken under
#: (in the step's own space: internal coordinates in internal mode, where
#: the Cartesian norm read 0.49 of a capped step) counts as trust-limited.
_TRUST_CAPPED = 0.99
#: ... and a restart whose net move is at least this share of its summed
#: accepted steps is marching in one direction rather than oscillating.
#: Measured: capped restarts of converged split refits at default settings
#: read 0-0.5 (true distance <= 0.07 widths); the 1e-5 A attack read 1.0
#: (3-30 widths).
_MARCHING = 0.9
#: A march takes several steps: one accepted step has net = path trivially
#: (21 of 100 converged C1 refits restarted with one capped step).
_MARCH_STEPS = 3


def _trust_limited(h) -> bool:
    """Whether a history step was cut to the trust radius (the whole step)
    or to the null-space trust radius (its null part)."""
    r = float(h.get("dx_range_norm") or 0.0)
    n = float(h.get("dx_null_norm") or 0.0)
    trust = float(h.get("trust_radius") or np.inf)
    null_trust = float(h.get("null_trust_radius") or np.inf)
    return bool(np.hypot(r, n) >= _TRUST_CAPPED * trust or n >= _TRUST_CAPPED * null_trust)


def _split_objective(opt) -> bool:
    return (not opt.spectral_only) and opt.optimizer.objective_mode != "joint"


def _restart_distance(opt, before, after):
    """How far one more run of the step moved a refit, in posterior widths
    along each eigendirection of the posterior precision (the units of
    distance_to_mode).

    The split refit's certificate. The split step stops where
    Pn (g + H dx) = 0, the posterior's gradient is Pn (g + H Pn dx), and the
    two differ by exactly the range-null coupling the split bootstrap is run
    to measure -- so the distance to the posterior's mode refused converged
    split refits as unconverged (26 of 40 on one-constant water through the
    CLI, at 0.06-0.15 widths against a limit of 0.055). A converged step
    does not move when run again, wherever its fixed point is."""
    try:
        _z, basis, w, V = opt._mode_decomposition(np.asarray(after, dtype=float))
    except Exception:
        return None
    d = basis.T @ (np.asarray(after, dtype=float) - np.asarray(before, dtype=float)).ravel()
    return float(np.max(np.abs(V.T @ d) * np.sqrt(w))) if d.size else 0.0


def _run_bootstrap_worker(args: tuple) -> BootstrapSampleResult:
    base_opt, isotopologues, workdir, idx, seed = args[:5]
    rto_shift = args[5] if len(args) > 5 else None
    polish = float(args[6]) if len(args) > 6 else _POLISH_DISTANCE
    start = time.time()
    opt = deepcopy(base_opt)
    _freeze_corrections(opt)
    opt.spectral.isotopologues = isotopologues
    opt._rto_quantum_shift = rto_shift
    opt.workdir = workdir
    os.makedirs(workdir, exist_ok=True)
    start_coords = np.asarray(opt.coords, dtype=float).copy()
    model = _refit_model(opt)
    try:
        coords = np.asarray(opt.run(), dtype=float)
        passes = 0
        if _split_objective(opt):
            certificate = "restart"
            dist = None
            while passes < _POLISH_PASSES:
                n0 = len(opt.history)
                again = np.asarray(opt.run(), dtype=float)
                steps = [h for h in opt.history[n0:] if h.get("accepted")]
                taken = len(steps)
                end = (getattr(opt, "termination", None) or {}).get("reason")
                # A restart still marching at the trust radius -- its last step
                # cut to the cap, and its steps adding up rather than going
                # back and forth -- moved as far as it was allowed, not as far
                # as its fixed point: with the radius at 1e-5 A and a loose
                # plateau rule such refits passed up to 55 widths from it.
                # A step capped while oscillating about the fixed point (the
                # null step of converged split refits often is) still
                # certifies: its net move is a small part of its path.
                path = sum(float(h.get("step_norm") or 0.0) for h in steps)
                net = float(np.linalg.norm(again - coords))
                capped = (len(steps) >= _MARCH_STEPS and _trust_limited(steps[-1])
                          and net >= _MARCHING * path)
                # A restart that accepted no step and then ran out of
                # iterations says nothing: each run starts from the base
                # damping and trust radius and re-proposes the same rejected
                # step, so at max_iter 2 such refits passed at "0.0" while
                # 0.4-14.7 widths from their fixed point. One that stopped on
                # its own rule (converged, stall, plateau) after rejecting
                # every step is at a standstill, which is what a fixed point
                # looks like to the step -- and is most converged refits.
                dist = (None if (taken == 0 and end == "max_iter") or capped
                        else _restart_distance(opt, coords, again))
                coords = again
                passes += 1
                if dist is not None and dist <= polish:
                    break
        else:
            certificate = "gradient"
            dist = _distance(opt, coords)
            # Whatever ended the loop, max_iter included: the certificate,
            # not the stopping rule, says whether the refit is at its
            # optimum. With max_iter 8 every water refit ended on it (the
            # plateau stop cannot fire before iteration 11) and 17 of 20
            # were within 0.002 widths.
            while dist is not None and dist > polish and passes < _POLISH_PASSES:
                coords = np.asarray(opt.run(), dtype=float)
                dist = _distance(opt, coords)
                passes += 1
        rms = float(opt.history[-1]["freq_rms"]) if opt.history else None
        # Measured in the worker, where the starting geometry is in hand.
        # Exactly 0.0 means run() accepted no step, so this "sample" is the
        # starting point again -- see BootstrapRunSummary.usable.
        moved = float(np.abs(coords - start_coords).max())
        # After the polish passes, which are refits too.
        kept = _same_model(model, _refit_model(opt))
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
            certificate=certificate,
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

    def __init__(self, optimizer: MolecularOptimizer, config: BootstrapConfig,
                 laplace_regularization: float = 1e-8):
        self.optimizer = optimizer
        self.config = config
        self.config.validate()
        #: The ridge the analytic column's Laplace is formed with (the CLI's
        #: --laplace-regularization), so it is the covariance reported beside it.
        self.laplace_regularization = float(laplace_regularization)

    @staticmethod
    def strategy_problem(opt: MolecularOptimizer, strategy: str,
                         fitted: bool = True) -> str | None:
        """Why ``strategy`` cannot sample this optimizer, or None.

        ``fitted=False`` asks only what is known before the base fit -- so a
        caller can refuse before paying for it -- and leaves out the checks
        that need the fit's state (a quantum point, an internal prior the
        fit weighted)."""
        if not opt.spectral_only and opt.optimizer.objective_mode != "joint":
            # Under split the refits ARE the split estimator, so their cloud
            # measures what no closed form here predicts: whether the split
            # posterior (whose quantum term Pn H Pn leaves out the step's
            # range-null coupling, and which may disagree with the step about
            # the data's rank) describes the fit's spread. Refusing on the
            # coupling refused accurate checks (C1 water from an asymmetric
            # start, null response 0.058: refits within 1% of the analytic
            # column) and passed bad ones; the summary flags a departure
            # instead, and split refits are certified by a restart, not by
            # the posterior's gradient (_restart_distance). Refused: rto,
            # whose prior centre would have to be drawn in the null space
            # alone, and a fit whose posterior widths cannot be formed.
            if strategy == "rto":
                return ("strategy 'rto' samples the joint objective only: under the split "
                        "objective the quantum prior acts in the data's null space alone, "
                        "so its centre would have to be drawn there")
            if fitted:
                try:
                    opt.distance_to_mode()
                except Exception as exc:
                    return (f"strategy '{strategy}' measures refits in the posterior's widths, "
                            f"which could not be formed at this split fit "
                            f"({type(exc).__name__}: {exc})")
            return None
        if strategy != "rto":
            return None
        if opt.spectral_only or (fitted and opt.quantum is None):
            return "strategy 'rto' needs a quantum prior to randomise"
        if getattr(opt, "_prior_target_coords", None) is not None:
            return ("strategy 'rto' does not support prior_target_coords: the prior "
                    "target recomputes the quantum gradient, so the shift would be lost")
        anchored = opt.internal_prior is not None and opt.coordinate_mode == "cartesian"
        if anchored or (fitted and opt._internal_prior_rows(opt.coords) is not None):
            return ("strategy 'rto' randomises the data and the quantum prior only; an "
                    "internal-coordinate prior is active and would not be sampled")
        return None

    def _check_strategy(self) -> None:
        problem = self.strategy_problem(self.optimizer, self.config.strategy)
        if problem:
            raise ValueError(problem)

    def run(self, base_workdir: str | None = None) -> BootstrapRunSummary:
        base = os.path.abspath(base_workdir or self.optimizer.workdir or ".")
        run_dir = os.path.join(base, self.config.output_dir)
        os.makedirs(run_dir, exist_ok=True)
        self._check_strategy()

        print(
            f"\n[bootstrap-v2] Launching {self.config.n_samples} samples "
            f"(strategy={self.config.strategy}, workers={self.config.max_workers})..."
        )

        limit, polish = self._gate_limits()
        seed_seq = np.random.SeedSequence(self.config.seed)
        child_sequences = seed_seq.spawn(self.config.n_samples)
        error_cov = None
        cd_sigmas: list = []
        if self.config.strategy in ("error-model", "rto"):
            error_cov = self.optimizer.constant_error_covariance()
            cd_sigmas = self.optimizer.spectral.cd_row_sigmas(self.optimizer.coords)
        jobs: list[tuple] = []

        # The prior-centre draw's square root is fixed at the fitted
        # geometry: formed once, not once per sample.
        draw_shift = (quantum_prior_shift_sampler(self.optimizer)
                      if self.config.strategy == "rto" else None)
        for i, child in enumerate(child_sequences, start=1):
            seed = int(child.generate_state(1)[0])
            rng = np.random.default_rng(seed)
            shift = None
            if self.config.strategy == "isotopologue-gaussian":
                perturbed = resample_isotopologues_gaussian(self.optimizer.spectral.isotopologues, rng)
            elif self.config.strategy in ("error-model", "rto"):
                perturbed = resample_error_model(self.optimizer.spectral.isotopologues, error_cov, rng)
                if any(cd_sigmas):
                    perturbed = resample_cd_observed(perturbed, cd_sigmas, rng)
                if self.config.strategy == "rto":
                    shift = draw_shift(rng)
            else:
                raise ValueError(f"Unsupported bootstrap strategy: {self.config.strategy}")

            sample_dir = os.path.join(run_dir, f"sample_{i:03d}")
            jobs.append((self.optimizer, perturbed, sample_dir, i, seed, shift, polish))

        results: list[BootstrapSampleResult] = []

        def _done(res):
            results.append(res)
            if res.success:
                rms_label = "n/a" if res.rms_mhz is None else f"{res.rms_mhz:.4f} MHz"
                print(f"  [sample {res.idx:03d}] Finished. RMS: {rms_label}")
            else:
                print(f"  [sample {res.idx:03d}] FAILED: {res.error}")

        if self.config.max_workers <= 1:
            # One worker runs the refits here, not in a forked child. A child
            # forked from a process whose quantum backend has started threads
            # (pyscf with OMP_NUM_THREADS=4) inherits their locks and can hang
            # for ever: the water split check hung 3 of 3 times at 50 samples,
            # and finishes in 6-7 s run here. Each refit deep-copies the
            # optimizer either way, and the clouds are bit for bit the pool's
            # (water, formyl fluoride, 20 samples); time 0.8-1.3 s against
            # 1.0-1.1 s (T-42). Several workers still fork: see the README.
            for job in jobs:
                _done(_run_bootstrap_worker(job))
        else:
            if not self.optimizer.spectral_only:
                from backend.registry import fork_hang_warning
                why = fork_hang_warning(
                    self.optimizer.quantum_backend, self.config.max_workers,
                    backend_threads=(getattr(self.optimizer, "psi4_num_threads", None)
                                     if self.optimizer.quantum_backend == "psi4" else None))
                if why:
                    print(f"  [bootstrap-v2] warning: {why}")
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = [pool.submit(_run_bootstrap_worker, job) for job in jobs]
                for fut in as_completed(futures):
                    _done(fut.result())

        results.sort(key=lambda r: r.idx)
        success = [r for r in results if r.success]
        mean_rms = float(np.mean([r.rms_mhz for r in success if r.rms_mhz is not None])) if success else None
        base_rms = self._base_rms_mhz()
        self._gate_samples(results, base_rms, limit=limit)
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

    def _gate_limits(self) -> tuple[float, float]:
        """The refusal limit and the polish threshold, in posterior widths.

        distance_to_mode is measured in widths of the posterior, which is
        "rto"'s cloud, so for it these are max_mode_distance and
        _POLISH_DISTANCE as they stand. The other clouds are narrower --
        "error-model"'s is the reported covariance less its prior part, up to
        13x narrower on formyl fluoride, where 0.25 posterior widths let a
        refit stop up to 3.2 of the cloud's own widths short. So for them
        both are scaled by the smallest cloud-to-posterior width ratio over
        the coordinates the cloud resolves, which states them, near enough,
        in the cloud's own widths.

        A heuristic: one marginal ratio scales a test that is a maximum over
        posterior eigendirections, so it is stricter than needed where the
        data dominate, and a coordinate under _RESOLVED_FRACTION sets
        nothing. Converged refits sit about 500x below the smallest limit it
        can produce (formyl fluoride, one isotopologue: limit 0.013, refits
        at 1e-5), so it does not refuse them.
        """
        limit = float(self.config.max_mode_distance)
        if self.config.strategy == "rto":
            return limit, _POLISH_DISTANCE
        try:
            ref = self.analytic_reference()
            t = np.asarray(ref["target_std"], dtype=float)
            p = np.asarray(ref["posterior_std"], dtype=float)
            ratio = t[p > 0.0] / p[p > 0.0]
            ratio = ratio[ratio >= _RESOLVED_FRACTION]
            scale = float(min(1.0, ratio.min())) if ratio.size else 1.0
        except Exception as exc:  # no analytic reference: the unscaled gate
            print(f"  [bootstrap-v2] gate left in posterior widths (no analytic "
                  f"reference: {type(exc).__name__}: {exc})")
            scale = 1.0
        return limit * scale, _POLISH_DISTANCE * scale

    def _gate_samples(
        self,
        results: list[BootstrapSampleResult],
        base_rms: float | None = None,
        limit: float | None = None,
    ) -> None:
        """Mark which samples may enter the cloud, and why the rest may not.

        A sample is a draw when its refit fitted the model it was given and
        reached its own optimum. In order:

        1. It raised, or returned no geometry.
        2. ``moved_ang == 0.0``: no step was accepted, so the geometry is the
           start. Measured on water with four samples, two came back
           bit-identical to the base geometry and were counted as successes.
        3. Its targets or weighting were not the ones it started with -- a
           correction rebuild, a mass-dependent refit or a rescale rewrote
           them -- so it fitted another model.
        4. It stopped more than ``limit`` posterior widths from its own
           optimum (distance_to_mode on the sample's own objective under
           joint; under split, how far a restart of the step moved it; the
           config's max_mode_distance unless the engine scaled it to the
           cloud, see _gate_limits), or that distance could not be measured
           on a fit that did not report convergence. How the loop ended is
           not itself a reason: a refit that ran out of iterations at its
           optimum is a draw.

        This replaces a gate on the final residual RMS relative to the base
        fit's. Perturbed constants fit worse by construction, so the ratio
        measured the draw, not the fit: it refused 25 of 48 draws on the
        water config and 39 of 48 on formyl fluoride, most of which sat at
        their own optimum, and kept the ones that happened to fit well.
        """
        limit = float(self.config.max_mode_distance if limit is None else limit)
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
                    "the drawn constants, their B0->Be targets or their weighting "
                    "were replaced during the refit, so it fitted another model")
                continue
            if r.mode_distance is None:
                if r.termination not in (None, "converged"):
                    r.unusable_reason = (
                        "its distance to its own optimum could not be measured "
                        f"and the refit ended on '{r.termination}'")
                    continue
            elif r.mode_distance > limit:
                if r.certificate == "restart":
                    r.unusable_reason = (
                        f"the refit stopped short of its own optimum: one more run "
                        f"of the split step moved it {r.mode_distance:.3g} posterior "
                        f"widths (limit {limit:.3g})")
                else:
                    r.unusable_reason = (
                        f"the refit stopped {r.mode_distance:.3g} posterior widths "
                        f"from its own optimum (limit {limit:.3g})")
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

        from .posterior import no_posterior_reason, reported_covariance

        opt = self.optimizer
        x = np.asarray(opt.coords, dtype=float)
        c_total, cov = reported_covariance(opt, x, regularization=self.laplace_regularization)
        J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
        strategy = self.config.strategy
        if strategy == "rto":
            target_cov = c_total
            label = "posterior (reported covariance)"
        elif strategy == "error-model":
            _hs, h_prior, _parts = opt._posterior_blocks(x, exact_curvature=False)
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
               "bootstrap_std": None,
               # Set where the Gauss-Newton check found a stationary saddle:
               # the columns are then the Laplace's arithmetic, not a width the
               # engine reports (internal_coordinate_posterior refuses).
               "no_posterior": no_posterior_reason(opt, x),
               # Refits hold the base's quadratic, so this cloud cannot see
               # how far its Hessian is from the curvature here; say how old
               # that Hessian is instead.
               "hessian_distance_ang": opt.hessian_distance(x)}
        split = (not opt.spectral_only) and opt.optimizer.objective_mode != "joint"
        out["objective"] = "spectral only" if opt.spectral_only else opt.optimizer.objective_mode
        if split:
            out["null_response"] = opt.split_coupling()
            try:
                out["split_note"] = opt.split_inconsistency()
            except Exception as exc:  # a note must not cost the comparison
                out["split_note"] = f"the split check could not be made ({type(exc).__name__})"
            if opt.split_off_mode() is not None:
                out["mode_shift"] = self._mode_shift(opt, x, cs, Bq, conv)
        use = [r for r in (results or []) if r.success and r.usable and r.coords is not None]
        if len(use) > 2:
            q = np.array([np.asarray(cs.active_values(np.asarray(r.coords, dtype=float)), dtype=float)
                          for r in use])
            # Dihedrals as deviations from the base, wrapped: a cloud around
            # +-180 deg otherwise has a spread of ~180 deg (fluoroethane with
            # dihedrals: 181.1 against 0.87 wrapped, T-41 verifier).
            dih = np.array([k == "dihedral" for k in kinds], dtype=bool)
            if dih.any():
                q0 = np.asarray(cs.active_values(x), dtype=float)
                d = q[:, dih] - q0[dih]
                q[:, dih] = q0[dih] + (d + np.pi) % (2.0 * np.pi) - np.pi
            out["bootstrap_std"] = (q.std(axis=0, ddof=1) * conv).tolist()
            out["check"] = self._cloud_check(out["names"], out["bootstrap_std"],
                                             out["target_std"], out["posterior_std"], q)
        return out

    @staticmethod
    def _mode_shift(opt, x, cs, Bq, conv):
        """One Newton step on the posterior's own quadratic model, from x
        toward its mode, per internal coordinate (Angstrom, degrees): where
        the posterior is centred when the fit is not there (internal-prior
        C-only water: the angle +4.4 deg)."""
        try:
            _z, basis, w, V = opt._mode_decomposition(x)
            g = V.T @ (basis.T @ opt._posterior_gradient(x))
            dx = basis @ (V @ (-g / w))
            return dict(zip(cs.active_names(), (Bq @ dx * conv).tolist()))
        except Exception:
            return None

    #: A cloud width departs from the analytic one when its log ratio is
    #: beyond the normal quantile that would put the chance of a false flag
    #: anywhere among the m coordinates at the two-sided 3-sigma rate
    #: (0.27%) -- one 3-sigma test per coordinate flagged 14% of Gaussian
    #: nine-coordinate clouds at 20 refits. The Monte Carlo error is itself
    #: estimated, so the rate measured on Gaussian clouds is higher: 1.07%
    #: at 20 refits and 1.00% at 40, nine coordinates (seed 20261116). The
    #: price is power: at 20 refits and nine coordinates a ratio of 0.56
    #: or 1.80 is flagged about half the time (48% and 39% measured), one
    #: nearer 1 seldom, which the report states ...
    _CHECK_FAMILY_RATE = 0.0027
    #: ... and only where, if the cloud is right, the reported (posterior)
    #: width would move by at least this fraction: a 15% departure on a part
    #: holding 1% of the reported variance (joint C-only water's angle)
    #: cannot change what is reported.
    _CHECK_IMPACT = 0.02

    @classmethod
    def _cloud_check(cls, names, cloud_sd, target_sd, posterior_sd, q):
        """Cloud over analytic width per coordinate, its Monte Carlo error,
        the change it implies for the reported width, and where it departs.

        The error of an sd ratio comes from the exact variance of a sample
        variance, 2/(n-1) + excess kurtosis/n, with the sample excess
        kurtosis corrected for its small-n bias (G2) and never let below the
        Gaussian value: sqrt((k-1)/(4n)) from the raw sample kurtosis
        flagged 1.65% of Gaussian coordinates at 20 refits (nominal 0.27%)
        and 6-17% of kurtosis-20 ones at 40-200. The test is on the log
        ratio, whose sampling distribution is near-symmetric where the ratio
        itself is skewed, at the family-wise quantile _CHECK_FAMILY_RATE. The
        printed error is that of the printed ratio (r times the relative
        error). The cloud is refits of the estimator and the analytic column
        the engine's account of their spread, so a departure says that
        account does not describe the fit there.

        A heavy-tailed cloud is the limit: with few refits in its tails its
        sd, and its kurtosis with it, read low. Clouds with 3-5% of refits
        in a tail 5-7x wider were flagged on 7-35% of single coordinates and
        34-92% of nine-coordinate clouds at 20-200 refits, almost all (99%)
        as narrower than the analytic width; the report says so beside such
        a flag. A cloud that collapsed (sd exactly zero against a positive
        analytic width) departs; a coordinate that does not vary has no
        kurtosis (NaN).

        Also returned: "detectable" (low, high), the ratios at which the
        test's bar sits for a Gaussian cloud at this many refits -- its
        50%-power points, not a guarantee: a true ratio there is flagged
        about half the time (at 20 refits, nine coordinates, 2000 Gaussian
        clouds, seed 20261117: 48% at 0.556, 39% at 1.798 -- the high side
        lower because the test's error uses the cloud's own kurtosis; 5%
        further out, 57% and 49%; at ratio 1, 0.1%).

        A flag also needs the ratio to move the reported width by
        _CHECK_IMPACT, which takes a wider ratio where the analytic width is
        a small share s of the posterior variance: high r >= sqrt(1 + ((1 +
        g)^2 - 1) / s), low r <= sqrt(1 - (1 - (1 - g)^2) / s), none when s <
        1 - (1 - g)^2. "detectable_by_coordinate" is (low, high) per
        coordinate with both bars, None for a side it cannot be flagged on:
        at 20 refits with the posterior 5x the analytic width (s = 0.04) the
        ratios 0.556 and 1.798 were flagged 0% and 37% of the time, and with
        it 13x (s = 0.0059, the error-model strategy on formyl fluoride) 0%
        and 0%, though "detectable" said about half."""
        from scipy.stats import norm

        q = np.asarray(q, dtype=float)
        n = int(q.shape[0])
        m = max(len(names), 1)
        d = q - q.mean(axis=0)
        # Exactly constant columns: the mean's round-off would otherwise
        # give them a spread of 1e-33 and a kurtosis read off it.
        m2 = np.where(np.ptp(q, axis=0) > 0.0, (d ** 2).mean(axis=0), 0.0)
        safe = np.where(m2 > 0.0, m2, 1.0)
        g2 = np.where(m2 > 0.0, (d ** 4).mean(axis=0) / safe ** 2 - 3.0, 0.0)
        G2 = (((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))) if n > 3 else np.zeros_like(g2)
        rel = 0.5 * np.sqrt(2.0 / max(n - 1, 1) + np.maximum(G2, 0.0) / max(n, 1))
        G2 = np.where(m2 > 0.0, G2, np.nan)      # no spread, no kurtosis
        z = float(norm.ppf(1.0 - cls._CHECK_FAMILY_RATE / (2.0 * m)))
        rel0 = 0.5 * np.sqrt(2.0 / max(n - 1, 1))
        ratio, impact = [], []
        for b, t, p in zip(cloud_sd, target_sd, posterior_sd):
            r = float(b) / float(t) if t and t > 0 else float("nan")
            ratio.append(r)
            share = (float(t) / float(p)) ** 2 if p and p > 0 else float("nan")
            impact.append(float(np.sqrt(max(1.0 + (r * r - 1.0) * share, 0.0)) - 1.0)
                          if np.isfinite(r) and np.isfinite(share) else float("nan"))
        departs = [nm for nm, r, e, imp in zip(names, ratio, rel, impact)
                   if np.isfinite(r) and (r == 0.0 or abs(np.log(r)) > z * float(e))
                   and np.isfinite(imp) and abs(imp) >= cls._CHECK_IMPACT]
        g = float(cls._CHECK_IMPACT)
        lo0, hi0 = float(np.exp(-z * rel0)), float(np.exp(z * rel0))
        per_coord = []
        for t, p in zip(target_sd, posterior_sd):
            share = (float(t) / float(p)) ** 2 if (t and t > 0 and p and p > 0) else float("nan")
            if not (np.isfinite(share) and share > 0.0):
                per_coord.append([None, None])     # no impact, never flagged
                continue
            hi = max(hi0, float(np.sqrt(1.0 + ((1.0 + g) ** 2 - 1.0) / share)))
            lo_sq = 1.0 - (1.0 - (1.0 - g) ** 2) / share
            per_coord.append([min(lo0, float(np.sqrt(lo_sq))) if lo_sq >= 0.0 else None, hi])
        return {"n": n, "ratio": ratio,
                "mc_sd": [float(r * e) if np.isfinite(r) else float("nan") for r, e in zip(ratio, rel)],
                "kurtosis": [float(k + 3.0) for k in np.atleast_1d(G2)], "impact": impact,
                "departs": departs, "sigmas": z, "impact_min": float(cls._CHECK_IMPACT),
                "detectable": [lo0, hi0], "detectable_by_coordinate": per_coord}

    @staticmethod
    def _persist_summary(summary: BootstrapRunSummary, run_dir: str) -> None:
        out_path = os.path.join(run_dir, "bootstrap_summary.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(json_ready(summary.to_dict()), fh, indent=2, allow_nan=False)
