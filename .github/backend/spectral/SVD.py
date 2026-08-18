"""
SVD subspace optimizer for hybrid spectroscopic / quantum structure determination.

Operates in any parameter space p (Cartesian x or internal q):

  Cartesian mode (coordinate_mode="cartesian"):
    J  : (3k × 3N)   stacked rotational-constant Jacobian [MHz/Å]
    p  : (3N,)        Cartesian coordinates [Å]
    dp : (3N,)        Cartesian step returned by step()

  Internal-coordinate mode (coordinate_mode="internal"):
    J  : (3k × n_q)  spectral Jacobian in internal-coordinate space [MHz/Å or MHz/rad]
    p  : (n_q,)       internal coordinates [Å, rad]
    dp : (n_q,)       internal-coordinate step returned by step()
    The caller is responsible for converting dp → dx via apply_internal_step().

Decomposes J to separate:
  Range space  — parameter directions that move rotational constants;
                 the experimental step Δp_range drives these toward observed values.
  Null space   — parameter directions invisible to rotational constants;
                 the quantum step Δp_null performs damped-Newton energy minimisation
                 here, governed entirely by the QC gradient and Hessian.

Combined step:  Δp = Δp_range + Δp_null  (clipped to trust radius).
"""

import numpy as np


class SubspaceOptimizer:
    """
    Parameters
    ----------
    sv_threshold : float
        Rank cutoff: singular values below sv_threshold * σ_max are treated as
        null-space directions.  Default 1e-3.
    trust_radius : float
        Maximum allowed step norm in Angstroms.  Default 0.1 Å.
    lambda_damp : float
        Levenberg–Marquardt regularisation added to the null-space Hessian.
        Prevents blow-up when the Hessian has near-zero eigenvalues.  Default 1e-4.
    quantum_prior_sigma_ang : float or None
        Displacement, in Angstroms, over which the quantum surface is trusted —
        roughly the geometry error of the electronic-structure method. When set,
        it replaces the ``alpha_quantum`` heuristics with a calibrated weight
        that puts the quantum prior and the spectral chi-square on a common
        scale, so growing correction uncertainty genuinely returns authority to
        theory. Only meaningful with ``objective_mode="joint"``; the split
        objective hands each range-space direction entirely to the data.
        Default None (legacy heuristic behaviour).
    null_hessian_floor : float
        Eigenvalue floor applied to the projected null-space Hessian before
        the Newton solve.  Clips curvature that is numerically zero at the
        scale of the problem, preventing floating-point noise from producing a
        spuriously large null-space step.  Units match the Hessian eigenvalue
        units (energy / coordinate²); adjust if the optimizer is run at a very
        different scale (e.g. cm⁻¹/Å² vs. MHz/Å²).  Default 1e-8.
    """

    def __init__(
        self,
        sv_threshold=1e-3,
        sv_min_abs=0.0,
        trust_radius=0.1,
        null_trust_radius=None,
        lambda_damp=1e-4,
        objective_mode="split",
        alpha_quantum=1.0,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        use_internal_preconditioner=False,
        null_hessian_floor=1e-8,
        quantum_prior_sigma_ang=None,
    ):
        self.sv_threshold = sv_threshold
        self.sv_min_abs = max(0.0, float(sv_min_abs))
        self.trust_radius = trust_radius
        self.null_trust_radius = (
            float(null_trust_radius) if null_trust_radius is not None else 0.5 * float(trust_radius)
        )
        self.lambda_damp = lambda_damp
        self.null_hessian_floor = float(null_hessian_floor)
        self.objective_mode = objective_mode
        self.alpha_quantum = float(alpha_quantum)
        self.dynamic_quantum_weight = bool(dynamic_quantum_weight)
        self.quantum_weight_beta = float(quantum_weight_beta)
        self.quantum_weight_min = float(quantum_weight_min)
        self.quantum_weight_max = float(quantum_weight_max)
        self.use_internal_preconditioner = use_internal_preconditioner
        self.quantum_prior_sigma_ang = (
            None if quantum_prior_sigma_ang is None else float(quantum_prior_sigma_ang)
        )

    # ── Decomposition ─────────────────────────────────────────────────────────

    def decompose(self, J):
        """
        Full SVD of J (3k × 3N).

        Returns
        -------
        U    : (3k, 3k)   left singular vectors
        s    : (min(3k,3N),)  singular values, descending
        Vt   : (3N, 3N)   right singular vectors (rows)
        rank : int         number of experimentally constrained directions
        """
        U, s, Vt = np.linalg.svd(J, full_matrices=True)
        if s[0] > 0:
            cutoff = max(self.sv_threshold * s[0], self.sv_min_abs)
            rank = int(np.sum(s > cutoff))
        else:
            rank = 0
        return U, s, Vt, rank

    # ── Range-space step (experimental) ──────────────────────────────────────

    def range_step(self, U, s, Vt, rank, residual):
        """
        Least-squares step in the experimental range space.

        Δp_range = V_r Σ_r⁻¹ U_r^T Δν

        Parameters
        ----------
        residual : (m,)   observed − calculated [observable units]

        Returns
        -------
        dp_range : (n_p,)  parameter step in range space
        """
        if rank == 0:
            return np.zeros(Vt.shape[1])
        s_r = s[:rank]
        U_r = U[:, :rank]    # (3k, rank)
        V_r = Vt[:rank].T    # (3N, rank)
        return V_r @ (U_r.T @ residual / s_r)

    # ── Null-space step (quantum) ─────────────────────────────────────────────

    def null_step(self, Vt, rank, gradient, hessian):
        """
        Damped-Newton step in the quantum null space.

        Δp_null = −V_⊥ (V_⊥^T H V_⊥ + λI)⁻¹ V_⊥^T g

        Parameters
        ----------
        gradient : (n_p,)      energy gradient in parameter space
        hessian  : (n_p, n_p)  energy Hessian in parameter space

        Returns
        -------
        dp_null : (n_p,)  parameter step in null space
        """
        n = Vt.shape[1]
        if rank >= n:
            return np.zeros(n)
        V_null = Vt[rank:].T                             # (3N, 3N−rank)
        g_null = V_null.T @ gradient                     # (3N−rank,)
        H_null = V_null.T @ hessian @ V_null             # (3N−rank, 3N−rank)
        # Stabilize indefinite/near-singular null-space curvature.
        evals, evecs = np.linalg.eigh(H_null)
        evals = np.maximum(evals, self.null_hessian_floor)
        H_null_spd = evecs @ np.diag(evals) @ evecs.T
        H_reg  = H_null_spd + self.lambda_damp * np.eye(H_null.shape[0])
        dq     = np.linalg.solve(H_reg, -g_null)
        return V_null @ dq

    # ── Combined step ─────────────────────────────────────────────────────────

    def _apply_internal_preconditioner(self, dx, B):
        if B is None or B.size == 0:
            return dx
        BBt = B @ B.T
        reg = 1e-8 * np.eye(BBt.shape[0])
        q = np.linalg.solve(BBt + reg, B @ dx)
        return B.T @ q

    def _spd_hessian(self, hessian):
        """Floor the Hessian's eigenvalues so the prior can only pull, never push.

        The prior term alpha_q*H is a quadratic model of the quantum surface
        around the current geometry. Along an eigenvector with negative
        curvature that model is an inverted parabola, and solving with it moves
        *away* from the quantum minimum -- the prior becomes repulsive in
        exactly the directions where the surface is least trustworthy.

        This is not hypothetical. The hybrid sits off theory's minimum by
        construction, and the Cartesian Hessian is only free of rotational
        curvature *at* a stationary point; away from one it is indefinite, with
        the negative eigenvalues lying almost entirely (measured overlap 0.985)
        in the translation/rotation block. `null_step` has always floored its
        projected Hessian for this reason; the joint step needs it too, and now
        more, since it is the default.

        The floor is relative to the largest eigenvalue, so it carries the
        Hessian's units and holds up across the many orders of magnitude these
        problems span.
        """
        H = np.asarray(hessian, dtype=float)
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        if evals.size == 0:
            return H
        scale = float(np.max(np.abs(evals)))
        if not np.isfinite(scale) or scale <= 0.0:
            return H
        floor = self.null_hessian_floor * scale
        if float(evals.min()) >= floor:
            return H
        return evecs @ np.diag(np.maximum(evals, floor)) @ evecs.T

    def _joint_step(self, J, residual, gradient, hessian, alpha_q):
        """MAP step: (JᵀJ + α_q H + λI) dp = Jᵀr − α_q g.

        Unlike the split step this has no hard range/null partition, so the
        quantum surface retains influence over every direction in proportion to
        its curvature and the spectral block's weight. That is what lets the
        result fall back toward theory as the spectral targets lose credibility.
        """
        JTJ = J.T @ J
        rhs = J.T @ residual - alpha_q * gradient
        prior = alpha_q * self._spd_hessian(hessian)
        A = JTJ + prior
        n = max(A.shape[0], 1)
        # lambda_damp is a dimensionless relative damping. An absolute constant
        # cannot regularise this matrix: ‖JᵀJ‖ carries the units of the
        # observables over the parameters and the 1/sigma² weighting, so it
        # ranges over many orders of magnitude between problems, and a fixed
        # 1e-4 is silently negligible against it.
        #
        # It must be scaled by the PRIOR block, not by ‖A‖. ‖A‖ is dominated by
        # the data block wherever the data is informative, and using it sets the
        # damping from the data's magnitude everywhere -- including directions
        # the data cannot see at all. In those directions the equation reduces to
        # (alpha_q*H + lambda*I) dp = -alpha_q*g, so a lambda carrying the data's
        # scale swamps the prior and the step is decided by regularisation rather
        # than by the quantum surface. With tight sigma on the observations that
        # factor reaches ~500, which silently disables the theory half of the
        # hybrid exactly where it is the only source of information.
        #
        # Scaling by the prior is also sufficient to regularise: the six rigid
        # modes are null in JᵀJ and in H alike, and both Jᵀr and g are orthogonal
        # to them, so any positive lambda resolves that singularity.
        scale = np.linalg.norm(prior) / n
        if not np.isfinite(scale) or scale <= 0.0:
            scale = np.linalg.norm(A) / n
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        A = A + self.lambda_damp * scale * np.eye(A.shape[0])
        try:
            return np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # H carries six zero modes (translations/rotations), so A can still
            # be singular to working precision.
            return np.linalg.pinv(A) @ rhs

    def effective_quantum_weight(self, J, rank, hessian=None):
        """Weight the quantum block actually carries in the objective.

        Public because anything that reports on the fit -- a covariance, a
        posterior, a log-probability -- has to weight the two halves the same
        way the step did. Reading ``alpha_quantum`` directly gives the legacy
        heuristic, which is not what the joint objective minimises whenever
        ``quantum_prior_sigma_ang`` is set (the default), so a covariance built
        from it would describe a different problem than the one that was solved.
        """
        return self._effective_quantum_weight(J, rank, hessian=hessian)

    def _effective_quantum_weight(self, J, rank, hessian=None):
        """
        Dynamic quantum dominance factor.
        Stronger when spectral constraints are sparse relative to coordinate space.

        When ``quantum_prior_sigma_ang`` is set, alpha_q is instead derived from
        it and ignores the heuristics below — see :meth:`_calibrated_alpha_q`.
        """
        if self.quantum_prior_sigma_ang is not None and hessian is not None:
            return self._calibrated_alpha_q(hessian)
        if not self.dynamic_quantum_weight:
            return self.alpha_quantum
        n_params = max(1, J.shape[1])  # 3N coordinates
        rank_frac = float(rank) / float(n_params)
        scale = 1.0 + self.quantum_weight_beta * max(0.0, 1.0 - rank_frac)
        alpha_eff = self.alpha_quantum * scale
        return float(np.clip(alpha_eff, self.quantum_weight_min, self.quantum_weight_max))

    def _calibrated_alpha_q(self, hessian):
        """Put the quantum prior on the same statistical footing as the data.

        The spectral block is a sum of squared sigma-normalised residuals, so it
        is dimensionless. The quantum block is an energy. Adding them with
        alpha_quantum = 1.0 compares a chi-square to Hartrees, and the spectral
        side wins by whatever factor the units happen to produce — for water
        about 3e5, irrespective of how uncertain the targets are. That is why an
        honest sigma cannot hand authority back to theory on its own.

        Reading the quantum term as a Gaussian prior on geometry centred at the
        theory minimum, E ~ 1/2 dp^T H dp, one prior standard deviation must cost
        1/2 chi-square units. With sigma_x the displacement over which the theory
        surface is trusted and lambda_bar a typical curvature,

            E_scale = 1/2 * lambda_bar * sigma_x^2,   alpha_q = 1 / E_scale

        so alpha_quantum becomes an interpretable statement about how good the
        electronic structure method is, in Angstroms.
        """
        evals = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        positive = evals[evals > 1e-8]
        if positive.size == 0:
            return self.alpha_quantum
        lam_bar = float(np.mean(positive))
        sigma_x = float(self.quantum_prior_sigma_ang)
        e_scale = 0.5 * lam_bar * sigma_x * sigma_x
        if e_scale <= 0.0:
            return self.alpha_quantum
        return float(self.alpha_quantum / e_scale)

    def step(self, J, residual, gradient, hessian, B=None):
        """
        Full hybrid step in whatever parameter space J is defined over.

        Parameters
        ----------
        J        : (m, n_p)  stacked Jacobian [observable / parameter unit]
                   Cartesian mode: (3k, 3N) in MHz/Å
                   Internal mode : (3k, n_q) in MHz/Å or MHz/rad
        residual : (m,)     observed − calculated [same observable unit]
        gradient : (n_p,)   energy gradient in parameter space
                   Cartesian: Hartree/Å;  Internal: already transformed via B+^T gx
        hessian  : (n_p, n_p) energy Hessian in parameter space
                   Cartesian: Hartree/Å²; Internal: B+^T Hx B+

        Returns
        -------
        dp          : (n_p,)  parameter step (trust-radius clipped)
                      Cartesian mode: Cartesian step dx [Å]
                      Internal mode : internal step dq [Å, rad] — caller back-transforms
        rank        : int     SVD rank
        s           : array   full singular-value spectrum
        alpha_q_eff : float   effective quantum weight used
        Vt          : (n_p, n_p) right singular vectors (reuse to avoid recomputing SVD)
        """
        U, s, Vt, rank = self.decompose(J)
        alpha_q_eff = self._effective_quantum_weight(J, rank, hessian=hessian)
        if self.objective_mode == "joint":
            dp = self._joint_step(J, residual, gradient, hessian, alpha_q_eff)
        else:
            dp_range = self.range_step(U, s, Vt, rank, residual)
            dp_null = self.null_step(Vt, rank, gradient, hessian)
            # Numerical safeguard: keep quantum correction strictly in J-null space.
            dp_null = self.null_projector(Vt, rank) @ dp_null
            dp_null = alpha_q_eff * dp_null
            null_norm = np.linalg.norm(dp_null)
            if null_norm > self.null_trust_radius:
                dp_null *= self.null_trust_radius / null_norm
            dp = dp_range + dp_null

        if self.use_internal_preconditioner:
            dp = self._apply_internal_preconditioner(dp, B)

        norm = np.linalg.norm(dp)
        if norm > self.trust_radius:
            dp *= self.trust_radius / norm

        return dp, rank, s, alpha_q_eff, Vt

    def adapt_lambda(self, accepted, min_lambda=1e-8, max_lambda=1e2):
        """
        Simple trust-style damping adaptation: reduce lambda when step is accepted,
        increase when rejected/no progress.
        """
        if accepted:
            self.lambda_damp = max(min_lambda, self.lambda_damp * 0.5)
        else:
            self.lambda_damp = min(max_lambda, self.lambda_damp * 2.0)
            self.trust_radius = max(1e-4, self.trust_radius * 0.5)
            self.null_trust_radius = max(1e-4, self.null_trust_radius * 0.5)

    # ── Projectors (diagnostic / master use) ─────────────────────────────────

    @staticmethod
    def null_projector(Vt, rank):
        """P_null = I − V_r V_r^T  —  projects onto null space of J."""
        n = Vt.shape[1]
        V_r = Vt[:rank].T
        return np.eye(n) - V_r @ V_r.T

    @staticmethod
    def range_projector(Vt, rank):
        """P_range = V_r V_r^T  —  projects onto range space of J."""
        V_r = Vt[:rank].T
        return V_r @ V_r.T
