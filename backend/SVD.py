"""
SVD subspace optimizer for hybrid spectroscopic / quantum structure determination.

Decomposes the stacked rotational-constant Jacobian J (3k × 3N) to separate:
  Range space  — Cartesian directions that move rotational constants;
                 the experimental step Δx_range drives these toward observed values.
  Null space   — Cartesian directions invisible to rotational constants;
                 the quantum step Δx_null performs damped-Newton energy minimisation
                 here, governed entirely by the ORCA gradient and Hessian.

Combined step:  Δx = Δx_range + Δx_null  (clipped to trust radius).
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

        Δx_range = V_r Σ_r⁻¹ U_r^T Δν

        Parameters
        ----------
        residual : (3k,)  observed − calculated rotational constants [MHz]

        Returns
        -------
        dx_range : (3N,) in Angstroms
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

        Δx_null = −V_⊥ (V_⊥^T H V_⊥ + λI)⁻¹ V_⊥^T g

        Parameters
        ----------
        gradient : (3N,)    energy gradient [Hartree/Å]
        hessian  : (3N, 3N) energy Hessian  [Hartree/Å²]

        Returns
        -------
        dx_null : (3N,) in Angstroms
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
        Full hybrid step.

        Parameters
        ----------
        J        : (3k, 3N) stacked rotational-constant Jacobian [MHz/Å]
        residual : (3k,)    observed − calculated rotational constants [MHz]
        gradient : (3N,)    energy gradient [Hartree/Å]
        hessian  : (3N, 3N) energy Hessian [Hartree/Å²]

        Returns
        -------
        dx   : (3N,)  Cartesian step in Å (trust-radius clipped)
        rank : int    SVD rank
        s    : array  full singular-value spectrum
        """
        U, s, Vt, rank = self.decompose(J)
        alpha_q_eff = self._effective_quantum_weight(J, rank)
        if self.objective_mode == "joint":
            dx = self._joint_step(J, residual, gradient, hessian, alpha_q_eff)
        else:
            dx_range = self.range_step(U, s, Vt, rank, residual)
            dx_null = self.null_step(Vt, rank, gradient, hessian)
            # Numerical safeguard: explicitly keep quantum correction in J-null space.
            dx_null = self.null_projector(Vt, rank) @ dx_null
            dx_null = alpha_q_eff * dx_null
            null_norm = np.linalg.norm(dx_null)
            if null_norm > self.null_trust_radius:
                dx_null *= self.null_trust_radius / null_norm
            dx = dx_range + dx_null

        if self.use_internal_preconditioner:
            dx = self._apply_internal_preconditioner(dx, B)

        norm = np.linalg.norm(dx)
        if norm > self.trust_radius:
            dx *= self.trust_radius / norm

        return dx, rank, s, alpha_q_eff

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
