import numpy as np

from backend.autoconfig_bases import ALPHA_MAX, ALPHA_MIN, SV_MAX, SV_MIN


class AutoConfigEngine:
    """
    Runtime policy engine that adapts optimizer controls from diagnostics.

    This is intentionally molecule-agnostic: it only consumes numerical
    diagnostics (rank, conditioning, residual distribution, acceptance trend).
    """

    # Stage-classification thresholds — dimensionless ratios, scale-invariant
    # by construction.  These encode optimizer policy and should not vary per
    # molecule; the per-molecule dials are the base_* constructor arguments.
    _EXPLORE_REJECT_STREAK = 3     # consecutive rejections → far from basin, widen search
    _EXPLORE_SIGMA_RATIO   = 25.0  # residual / σ > 25 → solution likely in wrong basin
    _FIT_RANK_FRAC         = 0.7   # effective rank / n_params < 0.7 → underdetermined
    _FIT_SIGMA_RATIO       = 8.0   # residual / σ > 8 → still actively fitting, not refining

    _SV_GAP_UNSTABLE = 10.0
    _SV_GAP_STABLE = 100.0

    def __init__(
        self,
        n_params,
        base_trust_radius,
        base_null_trust_radius,
        base_lambda_damp,
        base_prior_weight,
        base_sigma_floor_mhz,
        base_max_spectral_weight,
        base_torsion_a_weight,
        base_sv_threshold=1e-3,
        base_alpha_quantum=1.0,
        tune_sv_threshold=True,
        tune_alpha_quantum=True,
        smoothing=0.4,
    ):
        self.n_params = max(1, int(n_params))
        self.base_trust_radius = float(base_trust_radius)
        self.base_null_trust_radius = float(base_null_trust_radius)
        self.base_lambda_damp = max(float(base_lambda_damp), 1e-8)
        self.base_prior_weight = max(float(base_prior_weight), 0.0)
        self.base_sigma_floor_mhz = max(float(base_sigma_floor_mhz), 0.0)
        self.base_max_spectral_weight = (
            None if base_max_spectral_weight is None else max(float(base_max_spectral_weight), 1e-6)
        )
        self.base_torsion_a_weight = float(base_torsion_a_weight)
        self.base_sv_threshold = max(float(base_sv_threshold), SV_MIN)
        self.base_alpha_quantum = float(np.clip(float(base_alpha_quantum), ALPHA_MIN, ALPHA_MAX))
        self.tune_sv_threshold = bool(tune_sv_threshold)
        self.tune_alpha_quantum = bool(tune_alpha_quantum)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.state = None

    @staticmethod
    def _blend(old, new, w):
        return float((1.0 - w) * old + w * new)

    @staticmethod
    def _safe_median_abs(values):
        if values.size == 0:
            return 0.0
        return float(np.median(np.abs(values)))

    @staticmethod
    def _sv_gap_at_rank(singular_values, rank):
        s = np.asarray(singular_values, dtype=float)
        if rank <= 0 or rank >= s.size:
            return None
        denom = float(s[rank])
        if denom <= 0.0:
            return None
        return float(s[rank - 1] / denom)

    def _classify_stage(self, rank_frac, sigma_ratio, reject_streak):
        if reject_streak >= self._EXPLORE_REJECT_STREAK or sigma_ratio > self._EXPLORE_SIGMA_RATIO:
            return "explore"
        if rank_frac < self._FIT_RANK_FRAC or sigma_ratio > self._FIT_SIGMA_RATIO:
            return "fit"
        return "refine"

    def _stage_scales(self, stage):
        if stage == "explore":
            return {
                "trust": 1.0,
                "null_trust": 1.0,
                "lambda": 4.0,
                "sigma_floor": 2.0,
                "max_weight": 0.4,
                "prior": 1.5,
                "spectral_relax": 0.03,
                "alpha": 0.85,
            }
        if stage == "fit":
            return {
                "trust": 0.8,
                "null_trust": 0.8,
                "lambda": 2.0,
                "sigma_floor": 1.4,
                "max_weight": 0.7,
                "prior": 1.25,
                "spectral_relax": 0.015,
                "alpha": 0.95,
            }
        return {
            "trust": 0.55,
            "null_trust": 0.55,
            "lambda": 1.0,
            "sigma_floor": 1.0,
            "max_weight": 1.0,
            "prior": 1.0,
            "spectral_relax": 0.0,
            "alpha": 1.0,
        }

    def _target_sv_threshold(self, sv_gap, cond_factor):
        base = self.base_sv_threshold
        if not self.tune_sv_threshold:
            return base
        factor = 1.0
        if sv_gap is not None:
            if sv_gap < self._SV_GAP_UNSTABLE:
                factor *= 1.15
            elif sv_gap > self._SV_GAP_STABLE:
                factor *= 0.92
        if cond_factor > 0.5:
            factor *= 1.0 + 0.1 * min(cond_factor, 1.5)
        target = base * factor
        return float(np.clip(target, max(SV_MIN, 0.5 * base), min(SV_MAX, 2.0 * base)))

    def _target_alpha_quantum(self, rank_frac, stage_scale_alpha):
        base = self.base_alpha_quantum
        if not self.tune_alpha_quantum:
            return base
        # Large null space (low rank_frac) → weaker quantum base; stage scales explore/fit/refine.
        null_frac = max(0.0, 1.0 - float(rank_frac))
        target = base * stage_scale_alpha * (0.55 + 0.45 * rank_frac) * (1.0 - 0.25 * null_frac)
        return float(np.clip(target, max(ALPHA_MIN, 0.5 * base), min(ALPHA_MAX, 2.0 * base)))

    def suggest(
        self,
        rank,
        singular_values,
        residual_mhz,
        sigma_scale_mhz,
        torsion_a_residuals,
        torsion_bc_residuals,
        reject_streak,
        has_internal_priors,
    ):
        s = np.asarray(singular_values, dtype=float)
        resid = np.asarray(residual_mhz, dtype=float)
        sigma_scale = max(float(sigma_scale_mhz), 1e-9)
        rank_frac = float(rank) / float(self.n_params)
        cond = 1e12
        if s.size > 0 and s[0] > 0:
            keep = s[s > 0]
            if keep.size > 0:
                cond = float(s[0] / keep[-1])
        sigma_ratio = float(np.sqrt(np.mean(resid ** 2)) / sigma_scale) if resid.size else 0.0
        abs_r = np.abs(resid)
        med = self._safe_median_abs(abs_r)
        mad = float(np.median(np.abs(abs_r - med))) if abs_r.size else 0.0
        robust_scale = max(1e-9, 1.4826 * mad)
        outlier_frac = float(np.mean(abs_r > (med + 3.0 * robust_scale))) if abs_r.size else 0.0
        stage = self._classify_stage(rank_frac, sigma_ratio, reject_streak)
        scales = self._stage_scales(stage)
        sv_gap = self._sv_gap_at_rank(s, int(rank))

        torsion_a = np.asarray(torsion_a_residuals, dtype=float)
        torsion_bc = np.asarray(torsion_bc_residuals, dtype=float)
        torsion_factor = 1.0
        if torsion_a.size > 0:
            a_med = self._safe_median_abs(torsion_a)
            bc_med = max(self._safe_median_abs(torsion_bc), 1e-6)
            ratio = a_med / bc_med
            # If A residual dominates B/C, reduce A influence globally.
            torsion_factor = 1.0 / (1.0 + max(0.0, ratio - 1.0))

        cond_factor = float(np.clip(np.log10(max(cond, 1.0)) / 6.0, 0.0, 1.5))
        sigma_factor = float(np.clip(1.0 + 0.8 * outlier_frac + 0.2 * cond_factor, 1.0, 2.5))
        prior_factor = 1.0 + 1.2 * max(0.0, 1.0 - rank_frac)

        target_trust = max(1e-4, self.base_trust_radius * scales["trust"] / (1.0 + 0.2 * cond_factor))
        target_null_trust = max(
            1e-4,
            self.base_null_trust_radius * scales["null_trust"] / (1.0 + 0.2 * cond_factor),
        )
        target_lambda = self.base_lambda_damp * scales["lambda"] * (1.0 + 2.0 * cond_factor + outlier_frac)
        target_lambda = float(np.clip(target_lambda, 1e-8, 1e2))
        target_sigma_floor = self.base_sigma_floor_mhz * scales["sigma_floor"] * sigma_factor
        if self.base_sigma_floor_mhz <= 0.0:
            target_sigma_floor = 0.0
        target_max_weight = None
        if self.base_max_spectral_weight is not None:
            target_max_weight = self.base_max_spectral_weight * scales["max_weight"] / sigma_factor
            target_max_weight = float(np.clip(target_max_weight, 1.0, self.base_max_spectral_weight))
        target_prior = 0.0
        if has_internal_priors and self.base_prior_weight > 0.0:
            target_prior = self.base_prior_weight * scales["prior"] * prior_factor
            target_prior = float(np.clip(target_prior, 0.25 * self.base_prior_weight, 4.0 * self.base_prior_weight))
        target_torsion_a = float(np.clip(self.base_torsion_a_weight * torsion_factor, 0.05, 1.0))
        target_relax = float(np.clip(scales["spectral_relax"] * (1.0 + outlier_frac), 0.0, 0.05))
        target_sv = self._target_sv_threshold(sv_gap, cond_factor)
        target_alpha = self._target_alpha_quantum(rank_frac, scales["alpha"])

        if self.state is None:
            self.state = {
                "trust_radius": target_trust,
                "null_trust_radius": target_null_trust,
                "lambda_damp": target_lambda,
                "sigma_floor_mhz": target_sigma_floor,
                "max_spectral_weight": target_max_weight,
                "prior_weight": target_prior,
                "torsion_a_weight": target_torsion_a,
                "spectral_accept_relax": target_relax,
                "sv_threshold": target_sv,
                "alpha_quantum": target_alpha,
                "sv_gap": sv_gap,
                "stage": stage,
                "rank_fraction": rank_frac,
                "sigma_ratio": sigma_ratio,
                "condition_est": cond,
                "outlier_fraction": outlier_frac,
            }
            return dict(self.state)

        w = self.smoothing
        self.state["trust_radius"] = self._blend(self.state["trust_radius"], target_trust, w)
        self.state["null_trust_radius"] = self._blend(self.state["null_trust_radius"], target_null_trust, w)
        self.state["lambda_damp"] = self._blend(self.state["lambda_damp"], target_lambda, w)
        self.state["sigma_floor_mhz"] = self._blend(self.state["sigma_floor_mhz"], target_sigma_floor, w)
        if target_max_weight is None:
            self.state["max_spectral_weight"] = None
        else:
            prev = self.state["max_spectral_weight"]
            self.state["max_spectral_weight"] = target_max_weight if prev is None else self._blend(prev, target_max_weight, w)
        self.state["prior_weight"] = self._blend(self.state["prior_weight"], target_prior, w)
        self.state["torsion_a_weight"] = self._blend(self.state["torsion_a_weight"], target_torsion_a, w)
        self.state["spectral_accept_relax"] = self._blend(self.state["spectral_accept_relax"], target_relax, w)
        self.state["sv_threshold"] = self._blend(self.state["sv_threshold"], target_sv, w)
        self.state["alpha_quantum"] = self._blend(self.state["alpha_quantum"], target_alpha, w)
        self.state["sv_gap"] = sv_gap
        self.state["stage"] = stage
        self.state["rank_fraction"] = rank_frac
        self.state["sigma_ratio"] = sigma_ratio
        self.state["condition_est"] = cond
        self.state["outlier_fraction"] = outlier_frac
        return dict(self.state)

    def reseed_bases(
        self,
        *,
        n_params=None,
        base_trust_radius=None,
        base_null_trust_radius=None,
        base_lambda_damp=None,
        base_sv_threshold=None,
        base_alpha_quantum=None,
    ):
        """Update base_* after internal-coordinate DOF count is known (iteration 0)."""
        if n_params is not None:
            self.n_params = max(1, int(n_params))
        if base_trust_radius is not None:
            self.base_trust_radius = float(base_trust_radius)
        if base_null_trust_radius is not None:
            self.base_null_trust_radius = float(base_null_trust_radius)
        if base_lambda_damp is not None:
            self.base_lambda_damp = max(float(base_lambda_damp), 1e-8)
        if base_sv_threshold is not None:
            self.base_sv_threshold = max(float(base_sv_threshold), SV_MIN)
        if base_alpha_quantum is not None:
            self.base_alpha_quantum = float(np.clip(float(base_alpha_quantum), ALPHA_MIN, ALPHA_MAX))
        self.state = None
