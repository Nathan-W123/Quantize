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
    ):
        self.sv_threshold = sv_threshold
        self.sv_min_abs = max(0.0, float(sv_min_abs))
        self.trust_radius = trust_radius
        self.null_trust_radius = (
            float(null_trust_radius) if null_trust_radius is not None else 0.5 * float(trust_radius)
        )
        self.lambda_damp = lambda_damp
        self.objective_mode = objective_mode
        self.alpha_quantum = float(alpha_quantum)
        self.dynamic_quantum_weight = bool(dynamic_quantum_weight)
        self.quantum_weight_beta = float(quantum_weight_beta)
        self.quantum_weight_min = float(quantum_weight_min)
        self.quantum_weight_max = float(quantum_weight_max)
        self.use_internal_preconditioner = use_internal_preconditioner

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
        floor = 1e-8
        evals = np.maximum(evals, floor)
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

    def _joint_step(self, J, residual, gradient, hessian, alpha_q):
        JTJ = J.T @ J
        rhs = J.T @ residual - alpha_q * gradient
        A = JTJ + alpha_q * hessian + self.lambda_damp * np.eye(JTJ.shape[0])
        return np.linalg.solve(A, rhs)

    def _effective_quantum_weight(self, J, rank):
        """
        Dynamic quantum dominance factor.
        Stronger when spectral constraints are sparse relative to coordinate space.
        """
        if not self.dynamic_quantum_weight:
            return self.alpha_quantum
        n_params = max(1, J.shape[1])  # 3N coordinates
        rank_frac = float(rank) / float(n_params)
        scale = 1.0 + self.quantum_weight_beta * max(0.0, 1.0 - rank_frac)
        alpha_eff = self.alpha_quantum * scale
        return float(np.clip(alpha_eff, self.quantum_weight_min, self.quantum_weight_max))

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
        alpha_q_eff = self._effective_quantum_weight(J, rank)
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
