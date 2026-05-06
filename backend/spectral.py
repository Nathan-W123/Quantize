import numpy as np
from scipy import constants
from backend.conformer_mixture import ConformerMixture
from backend.correction_models import RovibCorrection
from backend.rovib_corrections import effective_sigma_constants

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


def _jacobian_full(coords, masses, delta):
    """Full 3x(3N) Jacobian for (A,B,C) via central finite differences."""
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(coords)
    j_full = np.zeros((3, 3 * n))
    flat = coords.ravel()
    abs_flat = np.abs(flat)
    local_delta = float(delta) * np.maximum(abs_flat, 1.0)
    for i in range(3 * n):
        di = local_delta[i]
        fwd = flat.copy()
        bwd = flat.copy()
        fwd[i] += di
        bwd[i] -= di
        j_full[:, i] = (
            _rotational_constants(fwd.reshape(n, 3), masses)
            - _rotational_constants(bwd.reshape(n, 3), masses)
        ) / (2 * di)
    return j_full


def _jacobian_full_analytic(coords, masses, delta, degeneracy_rel_tol=1e-4):
    """
    Full (3 × 3N) Jacobian ∂(A,B,C)/∂(flat x) in MHz/Å using dλ/dx = v^T (dI/dx) v
    for principal moments λ of the inertia tensor (same ordering as ``_rotational_constants``).

    Falls back to finite differences when moments are nearly degenerate or non-positive
    (linear / pathological geometries).
    """
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(coords)
    I = _inertia_tensor(coords, masses)
    evals, evecs = np.linalg.eigh(I)
    if np.any(evals <= 1e-10):
        return _jacobian_full(coords, masses, delta)
    gaps = (evals[1] - evals[0], evals[2] - evals[1])
    scale = max(float(np.mean(evals)), 1e-12)
    if min(gaps) / scale < float(degeneracy_rel_tol):
        return _jacobian_full(coords, masses, delta)

    Mtot = float(masses.sum())
    mfrac = masses / Mtot
    cm = masses @ coords / Mtot
    r = coords - cm

    j_full = np.zeros((3, 3 * n))
    for j_atom in range(n):
        for a in range(3):
            p = 3 * j_atom + a
            dI = np.zeros((3, 3))
            for i in range(n):
                ci = (1.0 if i == j_atom else 0.0) - mfrac[j_atom]
                ri = r[i]
                dr2 = 2.0 * ri[a] * ci
                for p_ax in range(3):
                    for q_ax in range(3):
                        delta_pq = 1.0 if p_ax == q_ax else 0.0
                        d_inner = ci * (
                            (1.0 if p_ax == a else 0.0) * ri[q_ax]
                            + (1.0 if q_ax == a else 0.0) * ri[p_ax]
                        )
                        dI[p_ax, q_ax] += masses[i] * (delta_pq * dr2 - d_inner)
            for s in range(3):
                lam = evals[s]
                v = evecs[:, s]
                dlam = float(v @ dI @ v)
                j_full[s, p] = -_INERTIA_TO_MHZ * dlam / (lam * lam)
    return j_full


def sanitize_isotopologues(
    isotopologues,
    coords,
    delta=1e-3,
    jacobian_row_norm_max=1e9,
    tiny_target_mhz=1e-3,
):
    """
    Remove or downweight numerically unstable spectral components.

    Returns
    -------
    cleaned : list[dict]
    notes   : list[str]
    """
    cleaned = []
    notes = []
    labels = ["A", "B", "C"]
    for iso_idx, iso in enumerate(isotopologues, start=1):
        masses = np.asarray(iso["masses"], dtype=float)
        obs = np.asarray(iso["obs_constants"], dtype=float)
        idx = np.asarray(iso.get("component_indices", list(range(len(obs)))), dtype=int)
        sig = np.asarray(iso.get("sigma_constants", np.ones(len(obs))), dtype=float)
        alpha = np.asarray(iso.get("alpha_constants", np.zeros(len(obs))), dtype=float)
        delta_total_in = iso.get("delta_total_constants")
        delta_total = (
            np.asarray(delta_total_in, dtype=float).ravel()
            if delta_total_in is not None
            else None
        )

        def _target(k_local, comp_local):
            if delta_total is not None and k_local < delta_total.size and np.isfinite(delta_total[k_local]):
                return float(obs[k_local] + delta_total[k_local])
            return float(obs[k_local] + 0.5 * alpha[k_local])

        calc_abc = _rotational_constants(coords, masses)
        j_full = _jacobian_full_analytic(coords, masses, delta)
        keep = []
        out_obs, out_sig, out_alpha = [], [], []
        out_delta_total = []
        dropped = []
        for k, comp in enumerate(idx):
            comp = int(comp)
            target = _target(k, comp)
            calc = float(calc_abc[comp]) if 0 <= comp < 3 else np.nan
            jn = float(np.linalg.norm(j_full[comp])) if 0 <= comp < 3 else np.inf
            unstable = (not np.isfinite(calc)) or (not np.isfinite(jn))
            unstable = unstable or (jn > jacobian_row_norm_max)
            # Linear/near-linear A-like targets near zero are often ill-conditioned.
            unstable = unstable or (abs(target) < tiny_target_mhz and comp == 0)
            if unstable:
                dropped.append(labels[comp] if 0 <= comp < 3 else f"R{comp}")
                continue
            keep.append(comp)
            out_obs.append(float(obs[k]))
            out_sig.append(max(float(sig[k]), 1e-12))
            out_alpha.append(float(alpha[k]))
            if delta_total is not None and k < delta_total.size:
                out_delta_total.append(float(delta_total[k]))
            else:
                out_delta_total.append(np.nan)

        if not keep:
            # Fail safe: keep the single most stable original component.
            best_i = 0
            best_norm = np.inf
            for k, comp in enumerate(idx):
                comp = int(comp)
                if 0 <= comp < 3:
                    nrm = float(np.linalg.norm(j_full[comp]))
                    if np.isfinite(nrm) and nrm < best_norm:
                        best_norm = nrm
                        best_i = k
            comp = int(idx[best_i])
            keep = [comp]
            out_obs = [float(obs[best_i])]
            out_sig = [max(float(sig[best_i]), 1e-12)]
            out_alpha = [float(alpha[best_i])]
            if delta_total is not None and best_i < delta_total.size:
                out_delta_total = [float(delta_total[best_i])]
            else:
                out_delta_total = [np.nan]
            dropped = [labels[int(c)] if 0 <= int(c) < 3 else f"R{int(c)}" for i, c in enumerate(idx) if i != best_i]

        if dropped:
            notes.append(
                f"Iso {iso_idx}: dropped unstable components {', '.join(dropped)}; "
                f"kept {', '.join(labels[c] if 0 <= c < 3 else f'R{c}' for c in keep)}."
            )

        # Slice the optional component-aligned correction vectors to keep them
        # aligned with the surviving indices.
        def _slice_optional(key):
            v = iso.get(key)
            if v is None:
                return None
            arr = np.asarray(v, dtype=float).ravel()
            kept_idx = []
            for k, comp in enumerate(idx):
                if int(comp) in keep:
                    kept_idx.append(k)
            kept_idx = [k for k in kept_idx if k < arr.size]
            return np.asarray([arr[k] for k in kept_idx], dtype=float) if kept_idx else None

        cleaned_iso = {
            "name": iso.get("name", f"iso_{iso_idx}"),
            "masses": masses,
            "obs_constants": np.asarray(out_obs, dtype=float),
            "component_indices": np.asarray(keep, dtype=int),
            "sigma_constants": np.asarray(out_sig, dtype=float),
            "alpha_constants": np.asarray(out_alpha, dtype=float),
            "torsion_sensitive": bool(iso.get("torsion_sensitive", False)),
            "rovib_table": iso.get("rovib_table", None),
        }
        # Carry through optional rovib correction fields (may be None).
        for key in (
            "delta_vib_constants",
            "delta_elec_constants",
            "delta_bob_constants",
            "sigma_correction_constants",
            "sigma_effective_constants",
        ):
            sliced = _slice_optional(key)
            if sliced is not None:
                cleaned_iso[key] = sliced
        if any(np.isfinite(out_delta_total)):
            cleaned_iso["delta_total_constants"] = np.asarray(out_delta_total, dtype=float)
        if iso.get("rovib_correction") is not None:
            cleaned_iso["rovib_correction"] = iso["rovib_correction"]
        cleaned.append(cleaned_iso)
    return cleaned, notes


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
        Step scale used only when the Jacobian falls back to finite differences
        (``analytic_jacobian=False`` or near-degenerate principal moments).
    analytic_jacobian : bool
        If True (default), use the analytic inertia derivative for ∂(A,B,C)/∂x.
    jacobian_degeneracy_tol : float
        If relative gaps between sorted principal moments are below this, use FD.
    """

    def __init__(
        self,
        isotopologues,
        delta=1e-3,
        robust_loss="none",
        robust_param=1.0,
        sigma_floor_mhz=0.0,
        sigma_cap_mhz=None,
        max_weight=None,
        component_weight_map=None,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        analytic_jacobian=True,
        jacobian_degeneracy_tol=1e-4,
    ):
        if not isotopologues:
            raise ValueError("At least one isotopologue is required.")
        def _opt_arr(iso_dict, key):
            v = iso_dict.get(key)
            if v is None:
                return None
            return np.asarray(v, dtype=float)

        self.isotopologues = []
        for k, iso in enumerate(isotopologues):
            entry = {
                "name": str(iso.get("name", f"iso_{k+1}")),
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
                "torsion_sensitive": bool(iso.get("torsion_sensitive", False)),
                "rovib_table": iso.get("rovib_table", None),
            }
            for opt_key in (
                "delta_vib_constants",
                "delta_elec_constants",
                "delta_bob_constants",
                "delta_total_constants",
                "sigma_correction_constants",
                "sigma_effective_constants",
            ):
                arr = _opt_arr(iso, opt_key)
                if arr is not None:
                    entry[opt_key] = arr
            if iso.get("rovib_correction") is not None:
                entry["rovib_correction"] = iso["rovib_correction"]
            self.isotopologues.append(entry)
        for iso in self.isotopologues:
            n = len(iso["obs_constants"])
            if len(iso["sigma_constants"]) != n or len(iso["alpha_constants"]) != n:
                raise ValueError("obs_constants, sigma_constants, and alpha_constants must match in length.")
            if len(iso["component_indices"]) != n:
                raise ValueError("component_indices length must match obs_constants length.")
        self.delta = delta
        self.robust_loss = robust_loss.lower()
        self.robust_param = max(float(robust_param), 1e-12)
        self.sigma_floor_mhz = max(float(sigma_floor_mhz), 0.0)
        self.sigma_cap_mhz = None if sigma_cap_mhz is None else max(float(sigma_cap_mhz), self.sigma_floor_mhz)
        self.max_weight = None if max_weight is None else max(float(max_weight), 1e-12)
        cwm = component_weight_map or {}
        self.component_weight_map = {
            int(k): float(v) for k, v in cwm.items() if int(k) in (0, 1, 2)
        }
        self.torsion_aware_weighting = bool(torsion_aware_weighting)
        self.torsion_a_weight = float(torsion_a_weight)
        self.conformer_mixture = None
        if conformer_defs is not None:
            self.conformer_mixture = ConformerMixture(
                reference_coords=np.zeros((len(self.isotopologues[0]["masses"]), 3), dtype=float),
                conformer_defs=conformer_defs,
                weight_mode=conformer_weight_mode,
                temperature_k=conformer_temperature_k,
            )
        self.analytic_jacobian = bool(analytic_jacobian)
        self.jacobian_degeneracy_tol = max(float(jacobian_degeneracy_tol), 1e-15)

    def _be_target(self, iso):
        """Return the Be target vector aligned with iso's component_indices.

        Uses ``delta_total_constants`` when present and finite for an entry,
        falling back to the legacy ``0.5 * alpha`` formula otherwise.
        """
        obs = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso.get("alpha_constants", np.zeros_like(obs)), dtype=float)
        dt = iso.get("delta_total_constants")
        if dt is None:
            return obs + 0.5 * alpha
        dt_arr = np.asarray(dt, dtype=float).ravel()
        out = obs + 0.5 * alpha
        n = min(out.size, dt_arr.size)
        for i in range(n):
            if np.isfinite(dt_arr[i]):
                out[i] = obs[i] + dt_arr[i]
        return out

    def effective_sigma_with_correction(self, iso):
        """Return per-row effective sigma combining obs noise and correction uncertainty."""
        sigma_obs = np.asarray(iso.get("sigma_constants", []), dtype=float)
        sigma_corr = self._correction_sigma(iso, sigma_obs.size)
        return np.sqrt(np.maximum(sigma_obs, 0.0) ** 2 + sigma_corr ** 2)

    def _correction_sigma(self, iso, n):
        """Return length-``n`` correction sigma aligned with the iso's components."""
        out = np.zeros(int(n), dtype=float)
        rc = iso.get("rovib_correction")
        idx = np.asarray(iso.get("component_indices", list(range(int(n)))), dtype=int)
        if isinstance(rc, RovibCorrection):
            sd = rc.sigma_delta_vector()
            for k in range(int(n)):
                c = int(idx[k]) if k < len(idx) else -1
                if 0 <= c < 3 and np.isfinite(sd[c]):
                    out[k] = float(sd[c])
            return out
        sc = iso.get("sigma_correction_constants")
        if sc is not None:
            sc_arr = np.asarray(sc, dtype=float).ravel()
            for k in range(int(n)):
                if k < sc_arr.size and np.isfinite(sc_arr[k]):
                    out[k] = max(float(sc_arr[k]), 0.0)
        return out

    def _effective_sigma(self, sigma, iso=None):
        """
        Apply optional sigma floor/cap and maximum weighting.

        When ``iso`` is provided and contains correction-uncertainty data, the
        observation sigma is first combined in quadrature with the correction
        sigma so downstream weights reflect the full uncertainty budget on Be.
        """
        sigma_eff = np.asarray(sigma, dtype=float).copy()
        if iso is not None:
            sigma_corr = self._correction_sigma(iso, sigma_eff.size)
            if np.any(sigma_corr > 0.0):
                sigma_eff = np.sqrt(np.maximum(sigma_eff, 0.0) ** 2 + sigma_corr ** 2)
        sigma_eff = np.maximum(sigma_eff, 1e-12)
        if self.sigma_floor_mhz > 0.0:
            sigma_eff = np.maximum(sigma_eff, self.sigma_floor_mhz)
        if self.sigma_cap_mhz is not None:
            sigma_eff = np.minimum(sigma_eff, self.sigma_cap_mhz)
        if self.max_weight is not None:
            sigma_eff = np.maximum(sigma_eff, 1.0 / self.max_weight)
        return sigma_eff

    def set_adaptive_controls(self, sigma_floor_mhz=None, max_weight=None, torsion_a_weight=None):
        """
        Update runtime weighting controls from external adaptive policy.
        """
        if sigma_floor_mhz is not None:
            self.sigma_floor_mhz = max(float(sigma_floor_mhz), 0.0)
            if self.sigma_cap_mhz is not None:
                self.sigma_cap_mhz = max(self.sigma_cap_mhz, self.sigma_floor_mhz)
        if max_weight is not None:
            self.max_weight = max(float(max_weight), 1e-12)
        if torsion_a_weight is not None:
            self.torsion_a_weight = max(float(torsion_a_weight), 1e-12)

    def _component_weights(self, iso):
        idx = np.asarray(iso["component_indices"], dtype=int)
        w = np.ones(len(idx), dtype=float)
        for i, comp in enumerate(idx):
            if int(comp) in self.component_weight_map:
                w[i] *= float(self.component_weight_map[int(comp)])
        if self.torsion_aware_weighting and iso.get("torsion_sensitive", False):
            for i, comp in enumerate(idx):
                if int(comp) == 0:
                    w[i] *= self.torsion_a_weight
        return np.maximum(w, 1e-12)

    def rotational_constants(self, coords, masses):
        """Computed (A, B, C) in MHz for given geometry and masses."""
        return _rotational_constants(np.asarray(coords), np.asarray(masses))

    def jacobian(self, coords, masses, component_indices=None):
        """
        (3 × 3N) Jacobian ∂(A,B,C)/∂(x₁,y₁,z₁,…,xₙ,yₙ,zₙ).
        Uses an analytic inertia derivative by default; finite differences when
        ``analytic_jacobian`` is False or when principal moments are nearly degenerate.
        Units: MHz / Å.
        """
        coords = np.asarray(coords, dtype=float)
        masses = np.asarray(masses, dtype=float)
        N = len(coords)
        if self.analytic_jacobian:
            J_full = _jacobian_full_analytic(
                coords, masses, self.delta, self.jacobian_degeneracy_tol
            )
        else:
            J_full = _jacobian_full(coords, masses, self.delta)
        if component_indices is None:
            return J_full
        return J_full[np.asarray(component_indices, dtype=int)]

    def residuals(self, coords, masses, obs_constants, alpha_constants=None, component_indices=None, delta_total_constants=None):
        """
        Δ(A,B,C) = target equilibrium constants − calculated constants in MHz.
        If ``delta_total_constants`` are supplied, applies Be ≈ B0 + δ_total.
        Otherwise, if alpha_constants are supplied, applies Be ≈ B0 + 0.5 * alpha.
        """
        if alpha_constants is None:
            alpha_constants = np.zeros(len(obs_constants))
        if delta_total_constants is not None:
            dt = np.asarray(delta_total_constants, dtype=float)
            be_target = np.asarray(obs_constants, dtype=float).copy()
            n = min(be_target.size, dt.size)
            be_fallback = np.asarray(obs_constants, dtype=float) + 0.5 * np.asarray(alpha_constants, dtype=float)
            for i in range(be_target.size):
                if i < n and np.isfinite(dt[i]):
                    be_target[i] = float(obs_constants[i] + dt[i])
                else:
                    be_target[i] = float(be_fallback[i])
        else:
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
        conf_coords = [coords]
        conf_weights = np.array([1.0], dtype=float)
        if self.conformer_mixture is not None:
            conf_coords = self.conformer_mixture.conformer_coords(coords)
            conf_weights = self.conformer_mixture.weights()
        for iso in self.isotopologues:
            j_mix = None
            calc_mix = None
            be_target = self._be_target(iso)
            idx = np.asarray(iso["component_indices"], dtype=int)
            for w, cxyz in zip(conf_weights, conf_coords):
                Jc = self.jacobian(cxyz, iso["masses"], idx)
                calc_c = _rotational_constants(np.asarray(cxyz), np.asarray(iso["masses"]))[idx]
                if j_mix is None:
                    j_mix = w * Jc
                    calc_mix = w * calc_c
                else:
                    j_mix += w * Jc
                    calc_mix += w * calc_c
            J = j_mix
            r = be_target - calc_mix
            sigma = self._effective_sigma(iso["sigma_constants"], iso=iso)
            Jw = J / sigma[:, None]
            rw = r / sigma
            comp_w = self._component_weights(iso)
            Jw = comp_w[:, None] * Jw
            rw = comp_w * rw
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
        conf_coords = [coords]
        conf_weights = np.array([1.0], dtype=float)
        if self.conformer_mixture is not None:
            conf_coords = self.conformer_mixture.conformer_coords(coords)
            conf_weights = self.conformer_mixture.weights()
        for iso in self.isotopologues:
            idx = np.asarray(iso["component_indices"], dtype=int)
            j_mix = None
            calc_mix = None
            be_target = self._be_target(iso)
            for w, cxyz in zip(conf_weights, conf_coords):
                Jc = self.jacobian(cxyz, iso["masses"], idx)
                calc_c = _rotational_constants(np.asarray(cxyz), np.asarray(iso["masses"]))[idx]
                if j_mix is None:
                    j_mix = w * Jc
                    calc_mix = w * calc_c
                else:
                    j_mix += w * Jc
                    calc_mix += w * calc_c
            J_blocks.append(j_mix)
            r_blocks.append(be_target - calc_mix)
        return np.vstack(J_blocks), np.concatenate(r_blocks)

    def conformer_diagnostics(self):
        if self.conformer_mixture is None:
            return None
        return self.conformer_mixture.diagnostics()
