import numpy as np
from scipy import constants

# h / (8π² · amu · Å²) → MHz; converts principal moments [amu·Å²] to rotational constants [MHz]
_INERTIA_TO_MHZ = (constants.h / (8 * np.pi**2 * constants.atomic_mass * (1e-10)**2)) * 1e-6


def _inertia_tensor(coords, masses):
    """Inertia tensor (3×3) in amu·Å², centered at center of mass."""
    cm = np.dot(masses, coords) / masses.sum()
    r = coords - cm
    r2 = np.einsum("ij,ij->i", r, r)
    return np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)


def _rotational_constants(coords, masses):
    """
    Rotational constants A ≥ B ≥ C in MHz from Cartesian coords (Å) and masses (amu).
    Returns shape (3,).
    """
    eigvals = np.sort(np.linalg.eigvalsh(_inertia_tensor(coords, masses)))
    eigvals = np.where(eigvals > 1e-10, eigvals, np.inf)
    return _INERTIA_TO_MHZ / eigvals


class SpectralEngine:
    """
    Rotational constants and Jacobian for an arbitrary number of isotopologues.

    Parameters
    ----------
    isotopologues : list of dict
        Each entry requires:
            'masses'        : array-like (N,)  atomic masses in amu
            'obs_constants' : array-like (3,)  observed A, B, C in MHz
    delta : float
        Central finite-difference step in Angstroms.
    """

    def __init__(self, isotopologues, delta=1e-3, robust_loss="none", robust_param=1.0):
        if not isotopologues:
            raise ValueError("At least one isotopologue is required.")
        self.isotopologues = [
            {
                "masses": np.asarray(iso["masses"], dtype=float),
                "obs_constants": np.asarray(iso["obs_constants"], dtype=float),
                "component_indices": np.asarray(
                    iso.get("component_indices", list(range(len(iso["obs_constants"])))),
                    dtype=int,
                ),
                "sigma_constants": np.asarray(
                    iso.get("sigma_constants", np.ones(len(iso["obs_constants"]))), dtype=float
                ),
                "alpha_constants": np.asarray(
                    iso.get("alpha_constants", np.zeros(len(iso["obs_constants"]))), dtype=float
                ),
            }
            for iso in isotopologues
        ]
        for iso in self.isotopologues:
            n = len(iso["obs_constants"])
            if len(iso["sigma_constants"]) != n or len(iso["alpha_constants"]) != n:
                raise ValueError("obs_constants, sigma_constants, and alpha_constants must match in length.")
            if len(iso["component_indices"]) != n:
                raise ValueError("component_indices length must match obs_constants length.")
        self.delta = delta
        self.robust_loss = robust_loss.lower()
        self.robust_param = max(float(robust_param), 1e-12)

    def rotational_constants(self, coords, masses):
        """Computed (A, B, C) in MHz for given geometry and masses."""
        return _rotational_constants(np.asarray(coords), np.asarray(masses))

    def jacobian(self, coords, masses, component_indices=None):
        """
        (3 × 3N) Jacobian ∂(A,B,C)/∂(x₁,y₁,z₁,…,xₙ,yₙ,zₙ) via central differences.
        Units: MHz / Å.
        """
        coords = np.asarray(coords, dtype=float)
        masses = np.asarray(masses, dtype=float)
        N = len(coords)
        J_full = np.zeros((3, 3 * N))
        flat = coords.ravel()
        abs_flat = np.abs(flat)
        local_delta = self.delta * np.maximum(abs_flat, 1.0)
        for i in range(3 * N):
            di = local_delta[i]
            fwd = flat.copy(); fwd[i] += di
            bwd = flat.copy(); bwd[i] -= di
            J_full[:, i] = (
                _rotational_constants(fwd.reshape(N, 3), masses)
                - _rotational_constants(bwd.reshape(N, 3), masses)
            ) / (2 * di)
        if component_indices is None:
            return J_full
        return J_full[np.asarray(component_indices, dtype=int)]

    def residuals(self, coords, masses, obs_constants, alpha_constants=None, component_indices=None):
        """
        Δ(A,B,C) = target equilibrium constants − calculated constants in MHz.
        If alpha_constants are supplied, applies Be ≈ B0 + 0.5 * alpha.
        """
        if alpha_constants is None:
            alpha_constants = np.zeros(len(obs_constants))
        be_target = obs_constants + 0.5 * np.asarray(alpha_constants, dtype=float)
        calc = _rotational_constants(np.asarray(coords), np.asarray(masses))
        if component_indices is not None:
            calc = calc[np.asarray(component_indices, dtype=int)]
        return be_target - calc

    def _robust_weight(self, scaled_residual):
        """
        Return diagonal robust reweighting for scaled residuals.
        """
        a = np.abs(scaled_residual)
        if self.robust_loss == "none":
            return np.ones_like(scaled_residual)
        if self.robust_loss == "huber":
            c = self.robust_param
            return np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))
        if self.robust_loss == "cauchy":
            c = self.robust_param
            return 1.0 / (1.0 + (a / c) ** 2)
        raise ValueError(f"Unknown robust_loss='{self.robust_loss}'. Use none|huber|cauchy.")

    def stacked(self, coords):
        """
        Stacked (3k × 3N) Jacobian and (3k,) residual vector across all k isotopologues.
        The SVD of the Jacobian determines which structural parameters are experimentally
        constrained vs. assigned to the quantum null space.
        """
        coords = np.asarray(coords, dtype=float)
        J_blocks, r_blocks = [], []
        for iso in self.isotopologues:
            J = self.jacobian(coords, iso["masses"], iso["component_indices"])
            r = self.residuals(
                coords,
                iso["masses"],
                iso["obs_constants"],
                iso["alpha_constants"],
                iso["component_indices"],
            )
            sigma = np.maximum(iso["sigma_constants"], 1e-12)
            Jw = J / sigma[:, None]
            rw = r / sigma
            robust_w = np.sqrt(self._robust_weight(rw))
            J_blocks.append(robust_w[:, None] * Jw)
            r_blocks.append(robust_w * rw)
        return np.vstack(J_blocks), np.concatenate(r_blocks)

    def stacked_unweighted(self, coords):
        """
        Return unweighted stacked Jacobian and residual vector in physical units.
        Jacobian units: MHz/Å, residual units: MHz.
        """
        coords = np.asarray(coords, dtype=float)
        J_blocks, r_blocks = [], []
        for iso in self.isotopologues:
            J_blocks.append(self.jacobian(coords, iso["masses"], iso["component_indices"]))
            r_blocks.append(
                self.residuals(
                    coords,
                    iso["masses"],
                    iso["obs_constants"],
                    iso["alpha_constants"],
                    iso["component_indices"],
                )
            )
        return np.vstack(J_blocks), np.concatenate(r_blocks)
