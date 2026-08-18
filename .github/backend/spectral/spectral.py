import numpy as np
from scipy import constants
from backend.conformers.conformer_mixture import ConformerMixture
#: Relative floor on every observation sigma, as a fraction of the constant.
#: Represents the irreducible error of fitting a rigid rotor to ground-state
#: constants. Set to 0.0 to disable (only sensible for synthetic data whose
#: generating model matches the fitted one exactly).
DEFAULT_SIGMA_FLOOR_REL = 1.0e-3

#: Auto-derived observation sigma, as a fraction of the constant, per component
#: (A, B, C). Used when an isotopologue supplies no ``sigma_constants``.
#:
#: These are model error, not measurement error, and that is why they can be
#: defaulted at all. A spectrometer determines a rotational constant to roughly
#: one part in 1e7; what limits the fit is that the measured quantity is a
#: ground-state constant while the fitted structure is a rigid one, and the
#: r_0-versus-r_e gap is a few tenths of a percent. That gap is a property of
#: the physics, not of the molecule at hand, so a single calibrated fraction
#: transfers across molecules -- which is exactly the argument that lets the
#: quantum prior width be looked up from the level of theory instead of typed
#: in. A is given twice the fraction of B and C: it is the least well
#: determined by a rigid-rotor model, being most sensitive to the light-atom
#: positions that zero-point motion displaces most.
#:
#: Calibrated on the reference molecules (see dev/monofluoro_references.py,
#: whose SIGMA_REL carries the same values) and validated by the coverage runs
#: in scripts/uncertainty_calibration.py. Supplying sigma_constants explicitly
#: always wins; do so whenever the source paper quotes a real uncertainty.
DEFAULT_SIGMA_REL_ABC = (0.010, 0.005, 0.005)

#: Absolute floor so a pathological (zero or near-zero) constant cannot produce
#: a zero sigma and a division by zero downstream.
_SIGMA_ABS_FLOOR_MHZ = 1.0e-6


def default_sigma_constants(obs_constants, component_indices=None):
    """Model-error sigma for constants supplied without one, in MHz.

    Returns ``DEFAULT_SIGMA_REL_ABC[component] * |constant|``, aligned to the
    isotopologue's ``component_indices`` so a B-only species is weighted as a B
    and not as an A. The previous fallback was a flat 1 MHz for every
    component, which is not a calibrated statement about anything: on a 90 GHz
    A constant it claims one part in 90,000, and on a 500 MHz C constant one
    part in 500 -- a 180-fold difference in implied confidence created purely
    by the size of the number.
    """
    obs = np.asarray(obs_constants, dtype=float).ravel()
    idx = (np.arange(obs.size) if component_indices is None
           else np.asarray(component_indices, dtype=int).ravel())
    out = np.empty(obs.size, dtype=float)
    for k in range(obs.size):
        comp = int(idx[k]) if k < idx.size else k
        rel = (DEFAULT_SIGMA_REL_ABC[comp] if 0 <= comp < 3
               else DEFAULT_SIGMA_REL_ABC[1])
        out[k] = max(rel * abs(float(obs[k])), _SIGMA_ABS_FLOOR_MHZ)
    return out

from backend.spectral.centrifugal_distortion import (
    CD_NAMES,
    CDConstants,
    build_cd_table_from_hessian,
    cd_observed_from_iso,
    compute_cd_constants,
)
from backend.spectral.correction_models import RovibCorrection

# h / (8Ï€Â² Â· amu Â· Ã…Â²) â†’ MHz; converts principal moments [amuÂ·Ã…Â²] to rotational constants [MHz]
_INERTIA_TO_MHZ = (constants.h / (8 * np.pi**2 * constants.atomic_mass * (1e-10)**2)) * 1e-6


def _inertia_tensor(coords, masses):
    """Inertia tensor (3Ã—3) in amuÂ·Ã…Â², centered at center of mass."""
    cm = np.dot(masses, coords) / masses.sum()
    r = coords - cm
    r2 = np.einsum("ij,ij->i", r, r)
    return np.einsum("i,jk->jk", masses * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", masses, r, r)


def _principal_moments(coords, masses):
    """Principal moments I_a <= I_b <= I_c in amu Ang^2, ordered to match A >= B >= C."""
    return np.sort(np.linalg.eigvalsh(
        _inertia_tensor(np.asarray(coords, dtype=float),
                        np.asarray(masses, dtype=float))))


def _rotational_constants(coords, masses):
    """
    Rotational constants A â‰¥ B â‰¥ C in MHz from Cartesian coords (Ã…) and masses (amu).
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
    Full (3 Ã— 3N) Jacobian âˆ‚(A,B,C)/âˆ‚(flat x) in MHz/Ã… using dÎ»/dx = v^T (dI/dx) v
    for principal moments Î» of the inertia tensor (same ordering as ``_rotational_constants``).

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
        if iso.get("sigma_constants") is not None:
            sig = np.asarray(iso["sigma_constants"], dtype=float)
        else:
            sig = default_sigma_constants(obs, idx)
            rel = ", ".join(f"{100 * r:g}%" for r in DEFAULT_SIGMA_REL_ABC)
            notes.append(
                f"{iso.get('name', f'iso_{iso_idx}')}: no sigma_constants given; "
                f"using the calibrated model-error default ({rel} of |A|,|B|,|C|) "
                f"-- the r_0-vs-r_e gap, not measurement precision. "
                f"Supply sigma_constants to override."
            )
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
            "sigma_systematic_constants",
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
        If True (default), use the analytic inertia derivative for âˆ‚(A,B,C)/âˆ‚x.
    jacobian_degeneracy_tol : float
        If relative gaps between sorted principal moments are below this, use FD.
    """

    def __init__(
        self,
        isotopologues,
        delta=1e-3,
        robust_loss="none",
        robust_param=1.0,
        sigma_floor_rel=DEFAULT_SIGMA_FLOOR_REL,
        sigma_floor_mhz=0.0,
        sigma_cap_mhz=None,
        max_weight=None,
        component_weight_map=None,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        conformer_defs=None,
        conformer_reference_coords=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        conformer_energy_unit="kcal/mol",
        analytic_jacobian=True,
        jacobian_degeneracy_tol=1e-4,
        cd_weight=0.0,
        fit_cd_constants=False,
        hess_bohr_for_cd=None,
        cd_fd_delta=0.05,
        cd_min_freq_cm=50.0,
        cd_sigma_fraction=0.05,
    ):
        if not isotopologues:
            raise ValueError("At least one isotopologue is required.")
        def _opt_arr(iso_dict, key):
            v = iso_dict.get(key)
            if v is None:
                return None
            return np.asarray(v, dtype=float)

        self.isotopologues = []
        _auto_sigma = []
        for k, iso in enumerate(isotopologues):
            if iso.get("sigma_constants") is None:
                _auto_sigma.append(str(iso.get("name", f"iso_{k+1}")))
            entry = {
                "name": str(iso.get("name", f"iso_{k+1}")),
                "masses": np.asarray(iso["masses"], dtype=float),
                "obs_constants": np.asarray(iso["obs_constants"], dtype=float),
                "component_indices": np.asarray(
                    iso.get("component_indices", list(range(len(iso["obs_constants"])))),
                    dtype=int,
                ),
                "sigma_constants": np.asarray(
                    iso["sigma_constants"] if iso.get("sigma_constants") is not None
                    else default_sigma_constants(
                        iso["obs_constants"],
                        iso.get("component_indices",
                                list(range(len(iso["obs_constants"]))))),
                    dtype=float,
                ),
                "alpha_constants": np.asarray(
                    iso.get("alpha_constants", np.zeros(len(iso["obs_constants"]))), dtype=float
                ),
                # Model error, carried for reporting only. Deliberately absent
                # from every weighting path: a systematic that shifts every
                # isotopologue the same way does not make any single constant
                # less worth fitting, and folding it into the weights makes the
                # fit under-use the data and over-state the answer at once.
                "sigma_systematic_constants": _opt_arr(iso, "sigma_systematic_constants"),
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
            cd_obs, cd_sig = cd_observed_from_iso(iso)
            if cd_obs:
                entry["cd_observed"] = cd_obs
            if cd_sig:
                entry["cd_sigma"] = cd_sig
            self.isotopologues.append(entry)
        for iso in self.isotopologues:
            n = len(iso["obs_constants"])
            if len(iso["sigma_constants"]) != n or len(iso["alpha_constants"]) != n:
                raise ValueError("obs_constants, sigma_constants, and alpha_constants must match in length.")
            if len(iso["component_indices"]) != n:
                raise ValueError("component_indices length must match obs_constants length.")
        if _auto_sigma:
            rel = ", ".join(f"{100 * r:g}%" for r in DEFAULT_SIGMA_REL_ABC)
            print(
                f"  [sigma] No sigma_constants given for {len(_auto_sigma)} "
                f"isotopologue(s): {', '.join(_auto_sigma[:4])}"
                f"{' ...' if len(_auto_sigma) > 4 else ''}.\n"
                f"  [sigma] Using the calibrated model-error default "
                f"({rel} of |A|, |B|, |C|). This is the r_0-vs-r_e modelling\n"
                f"  [sigma] gap, not measurement precision; supply "
                f"sigma_constants to override."
            )
        self.delta = delta
        self.sigma_floor_rel = max(float(sigma_floor_rel), 0.0)
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
            reference_coords = (
                np.asarray(conformer_reference_coords, dtype=float)
                if conformer_reference_coords is not None
                else np.zeros((len(self.isotopologues[0]["masses"]), 3), dtype=float)
            )
            self.conformer_mixture = ConformerMixture(
                reference_coords=reference_coords,
                conformer_defs=conformer_defs,
                weight_mode=conformer_weight_mode,
                temperature_k=conformer_temperature_k,
                energy_unit=conformer_energy_unit,
            )
        self.analytic_jacobian = bool(analytic_jacobian)
        self.jacobian_degeneracy_tol = max(float(jacobian_degeneracy_tol), 1e-15)
        self.cd_weight = max(float(cd_weight), 0.0)
        self.fit_cd_constants = bool(fit_cd_constants)
        self._hess_bohr_for_cd = (
            np.asarray(hess_bohr_for_cd, dtype=float).copy()
            if hess_bohr_for_cd is not None
            else None
        )
        self._cd_fd_delta = float(cd_fd_delta)
        self._cd_min_freq_cm = float(cd_min_freq_cm)
        self._cd_sigma_fraction = max(float(cd_sigma_fraction), 1e-6)
        self._cd_jacobian_cache: dict[tuple, np.ndarray] = {}

    def set_hessian_for_cd(self, hess_bohr) -> None:
        """Update Hessian used to predict harmonic CD constants during optimization."""
        if hess_bohr is None:
            self._hess_bohr_for_cd = None
        else:
            self._hess_bohr_for_cd = np.asarray(hess_bohr, dtype=float).copy()
        self._cd_jacobian_cache.clear()

    def _predict_cd_for_iso(self, iso: dict, coords) -> CDConstants | None:
        corr = iso.get("rovib_correction")
        if isinstance(corr, RovibCorrection) and corr.pred_cd is not None:
            return corr.pred_cd
        if self._hess_bohr_for_cd is None:
            return None
        return compute_cd_constants(
            self._hess_bohr_for_cd,
            coords,
            iso["masses"],
            min_freq_cm=self._cd_min_freq_cm,
            fd_delta=self._cd_fd_delta,
            sigma_fraction=self._cd_sigma_fraction,
        )

    def cd_residuals_mhz(self, coords) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        CD residual vector and per-term labels for isotopologues with observations.

        Returns (residuals, sigmas, labels). Empty arrays if CD fitting is inactive.
        """
        coords = np.asarray(coords, dtype=float)
        if self.cd_weight <= 0.0 or not self.fit_cd_constants:
            return np.array([]), np.array([]), []
        r_list: list[float] = []
        s_list: list[float] = []
        labels: list[str] = []
        for iso in self.isotopologues:
            obs = iso.get("cd_observed") or {}
            if not obs:
                continue
            pred = self._predict_cd_for_iso(iso, coords)
            if pred is None:
                continue
            sig_block = iso.get("cd_sigma") or pred.sigma
            for k in CD_NAMES:
                if k not in obs:
                    continue
                r_list.append(float(pred.as_dict()[k]) - float(obs[k]))
                if k in sig_block:
                    s_list.append(max(float(sig_block[k]), 1e-6))
                else:
                    s_list.append(max(abs(float(obs[k])) * self._cd_sigma_fraction, 0.01))
                labels.append(f"{iso['name']}:{k}")
        if not r_list:
            return np.array([]), np.array([]), []
        return np.asarray(r_list, dtype=float), np.asarray(s_list, dtype=float), labels

    def _cd_jacobian(self, coords, iso: dict) -> np.ndarray:
        """(n_cd_terms × 3N) Jacobian of predicted CD constants w.r.t. coordinates (FD)."""
        if self._hess_bohr_for_cd is None:
            return np.zeros((0, 3 * len(coords)))
        masses = np.asarray(iso["masses"], dtype=float)
        obs = iso.get("cd_observed") or {}
        keys = [k for k in CD_NAMES if k in obs]
        if not keys:
            return np.zeros((0, 3 * len(coords)))
        cache_key = (
            iso["name"],
            tuple(np.round(coords.ravel(), 8)),
            tuple(np.round(masses, 8)),
            tuple(keys),
        )
        if cache_key in self._cd_jacobian_cache:
            return self._cd_jacobian_cache[cache_key]

        n = len(coords)
        flat = coords.ravel()
        eps = max(self.delta, 1e-4)
        base = self._predict_cd_for_iso(iso, coords)
        if base is None:
            J = np.zeros((len(keys), 3 * n))
            self._cd_jacobian_cache[cache_key] = J
            return J
        base_v = base.vector()
        name_to_idx = {k: i for i, k in enumerate(CD_NAMES)}
        row_idx = [name_to_idx[k] for k in keys]
        J = np.zeros((len(keys), 3 * n))
        for p in range(3 * n):
            step = eps * max(abs(flat[p]), 1.0)
            fwd = flat.copy()
            bwd = flat.copy()
            fwd[p] += step
            bwd[p] -= step
            pf = self._predict_cd_for_iso(iso, fwd.reshape(n, 3))
            pb = self._predict_cd_for_iso(iso, bwd.reshape(n, 3))
            if pf is None or pb is None:
                continue
            J[:, p] = (pf.vector()[row_idx] - pb.vector()[row_idx]) / (2.0 * step)
        self._cd_jacobian_cache[cache_key] = J
        return J

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
        # Relative floor: no stated uncertainty can claim the model is better
        # than the model is. These constants are fitted with a rigid rotor and
        # no centrifugal-distortion term, so the residual model error is a
        # fraction of each constant, not an absolute number of MHz. Measured
        # against published structures it runs from 0.06% (fluoroacetylene) to
        # 2.4% (water), so a floor of a tenth of a percent is conservative and
        # does not bite on honestly-quoted data.
        #
        # Without it, a sigma quoted far below the model error hands the data
        # block a weight the physics cannot support: at 0.2 MHz on water's
        # 435 GHz B constant the observation claims 5e-7 relative precision, the
        # data term then outweighs the quantum prior by seven orders of
        # magnitude, and the fit follows the data into directions -- such as a
        # soft bending angle -- where it has no business leading.
        if self.sigma_floor_rel > 0.0 and iso is not None:
            obs = np.asarray(iso.get("obs_constants", []), dtype=float)
            if obs.size == sigma_eff.size:
                sigma_eff = np.maximum(sigma_eff,
                                       self.sigma_floor_rel * np.abs(obs))
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

    def fit_mass_dependent_correction(self, coords, frac_sigma=0.01):
        """Fit Watson's mass-dependent offset and install it as the B0 -> Be target.

        The gap between the measured B_0 and the equilibrium B_e is the largest
        systematic in any of these fits. Computing it needs a cubic force field,
        which at affordable levels of theory was measured to hurt more than it
        helped. Watson's alternative is to fit it: the rovibrational contribution
        to a moment of inertia is well described by

            I_obs = I_m + c_x sqrt(I_m)

        with three coefficients shared across every isotopologue.

        Fitting them *here* rather than as extra optimiser parameters is exact,
        not an approximation: the model is linear in c, so for any structure the
        best c follows in closed form, and substituting it back gives the same
        answer as optimising over structure and c jointly. This is variable
        projection. It also means the parameter vector, the trust radius, the
        symmetry projector and the SVD step are all untouched.

        The reason to do it inside the hybrid at all is that spectroscopy alone
        usually cannot afford these three parameters. A trifluorinated molecule
        has three heavy atoms that can never be isotopically substituted, and a
        standalone r_m fit on one comes out short by exactly three -- the c's.
        With the quantum surface holding the structure, the data are free to
        determine them.

        Parameters
        ----------
        coords : (N, 3)
            Current geometry, whose moments define I_m.
        frac_sigma : float
            Weak prior on the size of the correction, as a fraction of the
            moment. Without it a single isotopologue would let three
            coefficients absorb all three residuals exactly, driving the
            spectral term to zero and silently handing the fit to theory.

        Returns
        -------
        c : (3,) array in amu^(1/2) Angstrom, indexed by principal axis.
        """
        coords = np.asarray(coords, dtype=float)
        num = np.zeros(3)
        den = np.zeros(3)
        typ_sum = np.zeros(3)
        typ_n = np.zeros(3)
        rows = []
        for iso in self.isotopologues:
            masses = np.asarray(iso["masses"], dtype=float)
            i_m = _principal_moments(coords, masses)
            obs = np.asarray(iso["obs_constants"], dtype=float)
            sig = np.asarray(iso["sigma_constants"], dtype=float)
            idx = np.asarray(iso["component_indices"], dtype=int)
            for k, comp in enumerate(idx):
                comp = int(comp)
                if not (0 <= comp < 3) or obs[k] <= 0 or not np.isfinite(obs[k]):
                    continue
                if i_m[comp] <= 0 or not np.isfinite(i_m[comp]):
                    continue
                i_obs = _INERTIA_TO_MHZ / obs[k]
                # A fractional error on the constant is the same fractional
                # error on the moment, since I = k / B.
                s_i = abs(i_obs) * (sig[k] / obs[k]) if sig[k] > 0 else abs(i_obs) * 1e-4
                w = 1.0 / (s_i * s_i)
                root = np.sqrt(i_m[comp])
                num[comp] += w * root * (i_obs - i_m[comp])
                den[comp] += w * i_m[comp]
                typ_sum[comp] += i_m[comp]
                typ_n[comp] += 1
            rows.append((iso, i_m, idx))

        c = np.zeros(3)
        for comp in range(3):
            if typ_n[comp] == 0 or den[comp] <= 0:
                continue
            i_typ = typ_sum[comp] / typ_n[comp]
            # Prior expressed on the correction as a fraction of the moment:
            # c*sqrt(I) ~ frac_sigma * I, so sigma_c = frac_sigma * sqrt(I).
            sigma_c = frac_sigma * np.sqrt(max(i_typ, 1e-12))
            ridge = 1.0 / (sigma_c * sigma_c) if sigma_c > 0 else 0.0
            c[comp] = num[comp] / (den[comp] + ridge)

        # Install as delta_total_constants: the target constant is the one the
        # corrected moment implies, so the optimiser keeps comparing rigid
        # predictions against a target that already carries the offset.
        for iso, i_m, idx in rows:
            obs = np.asarray(iso["obs_constants"], dtype=float)
            delta = np.zeros(obs.size)
            for k, comp in enumerate(idx):
                comp = int(comp)
                if not (0 <= comp < 3) or obs[k] <= 0 or i_m[comp] <= 0:
                    continue
                i_obs = _INERTIA_TO_MHZ / obs[k]
                i_target = i_obs - c[comp] * np.sqrt(i_m[comp])
                if i_target > 0:
                    delta[k] = _INERTIA_TO_MHZ / i_target - obs[k]
            iso["delta_total_constants"] = delta
        return c

    def jacobian(self, coords, masses, component_indices=None):
        """
        (3 Ã— 3N) Jacobian âˆ‚(A,B,C)/âˆ‚(xâ‚,yâ‚,zâ‚,â€¦,xâ‚™,yâ‚™,zâ‚™).
        Uses an analytic inertia derivative by default; finite differences when
        ``analytic_jacobian`` is False or when principal moments are nearly degenerate.
        Units: MHz / Ã….
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
        Î”(A,B,C) = target equilibrium constants âˆ’ calculated constants in MHz.
        If ``delta_total_constants`` are supplied, applies Be â‰ˆ B0 + Î´_total.
        Otherwise, if alpha_constants are supplied, applies Be â‰ˆ B0 + 0.5 * alpha.
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

    def scale_sigma(self, factor: float) -> None:
        """Multiply every observation sigma by ``factor``.

        Scaling down is floored at each constant's systematic sigma when one is
        supplied. A common-mode model offset -- the B0-versus-Be gap that
        dominates here -- produces no isotopologue-to-isotopologue scatter, so
        a small chi-square is structurally incapable of proving the systematic
        part of sigma too large; rescaling below that floor calibrates away an
        error the residuals cannot see. Measured consequence before this guard:
        ethylene oxide's ring bonds rescaled to sigma 1.7 mA against a +12 mA
        error (z = +4.6) on a chi2/nu of 0.05.
        """
        """Multiply every observation sigma by `factor`.

        Used for chi-square rescaling: when a converged fit cannot reach a
        reduced chi-square near one, the stated uncertainties were optimistic,
        and the honest response is to widen them rather than to keep chasing
        residuals the model cannot represent.
        """
        f = float(factor)
        if not np.isfinite(f) or f <= 0.0:
            raise ValueError(f"sigma scale factor must be positive and finite, got {factor!r}")
        for iso in self.isotopologues:
            scaled = np.asarray(iso["sigma_constants"], dtype=float) * f
            if f < 1.0:
                floor = iso.get("sigma_systematic_constants")
                if floor is not None:
                    floor = np.asarray(floor, dtype=float).ravel()
                    if floor.size == scaled.size:
                        scaled = np.maximum(scaled, floor)
            iso["sigma_constants"] = scaled

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
        Stacked (3k Ã— 3N) Jacobian and (3k,) residual vector across all k isotopologues.
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

        if self.cd_weight > 0.0 and self.fit_cd_constants and self._hess_bohr_for_cd is not None:
            w_cd = np.sqrt(self.cd_weight)
            for iso in self.isotopologues:
                obs = iso.get("cd_observed") or {}
                if not obs:
                    continue
                pred = self._predict_cd_for_iso(iso, coords)
                if pred is None:
                    continue
                keys = [k for k in CD_NAMES if k in obs]
                if not keys:
                    continue
                pred_d = pred.as_dict()
                r_cd = np.array([float(pred_d[k]) - float(obs[k]) for k in keys], dtype=float)
                sig_block = iso.get("cd_sigma") or pred.sigma
                sigma_cd = np.array(
                    [
                        max(
                            float(sig_block.get(k, abs(obs[k]) * self._cd_sigma_fraction)),
                            0.01,
                        )
                        for k in keys
                    ],
                    dtype=float,
                )
                J_cd = self._cd_jacobian(coords, iso)
                Jw_cd = w_cd * J_cd / sigma_cd[:, None]
                rw_cd = w_cd * r_cd / sigma_cd
                J_blocks.append(Jw_cd)
                r_blocks.append(rw_cd)

        if not J_blocks:
            n = len(np.asarray(coords, dtype=float)) * 3
            return np.zeros((0, n)), np.array([])
        return np.vstack(J_blocks), np.concatenate(r_blocks)

    def stacked_unweighted(self, coords):
        """
        Return unweighted stacked Jacobian and residual vector in physical units.
        Jacobian units: MHz/Ã…, residual units: MHz.
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
