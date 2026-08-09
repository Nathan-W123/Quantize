"""
Master optimizer â€” ties SpectralEngine, QuantumEngine, and SubspaceOptimizer together.

Workflow
--------
1. Provide initial geometry, element list, isotopologue data, and either
   the path to your ORCA binary or "orca" if it is on PATH.
2. MolecularOptimizer.run() iterates:
     a. Recompute spectral Jacobian J and residuals Î”Î½ at current geometry.
     b. When geometry drifts beyond `orca_update_thresh`, refresh ORCA data:
          - Every `hess_recalc_every` ORCA calls: full Freq job (Hessian + gradient).
          - In between: cheap EnGrad job (gradient only, reuse existing Hessian).
     c. Compute the hybrid SVD step and update coordinates.
     d. Stop when step-norm < conv_step AND freq-RMS < conv_freq.
3. write_xyz() and report() expose the final result.

Units
-----
  Coordinates        : Angstroms
  Rotational constants : MHz
  Gradient           : Hartree / Ã…  (converted internally from Hartree / Bohr)
  Hessian            : Hartree / Ã…Â² (converted internally)
"""

import os
import numpy as np

from backend.spectral.spectral import SpectralEngine, sanitize_isotopologues
from backend.internal.internal_prior import InternalPriorEngine
from backend.spectral.rovib_corrections import (
    resolve_corrections,
    apply_corrections_to_isotopologues,
    validate_correction_quality,
    correction_summary,
)
from backend.spectral.correction_models import parse_correction_table, RovibCorrection, ParsedRovibResult


#: Default width of the quantum prior, in Angstrom -- the displacement over
#: which the electronic-structure surface is trusted.
#:
#: This must match the geometry error of the level of theory in use. Benchmarks
#: over six molecules put RHF/6-31G at 15-19 mA RMS and B3LYP/6-31G(d) at about
#: 7 mA, and the hybrid beats theory alone across a broad plateau either side of
#: the right value. 0.020 A suits a Hartree-Fock or small-basis calculation; for
#: a good DFT or post-HF surface, set it lower.
#:
#: The previous default of None was worse than any number. It routed the weight
#: through a heuristic that adds a dimensionless chi-square to an energy in
#: Hartrees, and the units alone hand the spectral side a factor of ~1e5 (for
#: water), so no honest sigma on the data could give theory a say.
#:
#: `scripts/estimate_theory_error.py` estimates the right value for a molecule
#: with no known structure, from the spread between two levels of theory.
DEFAULT_QUANTUM_PRIOR_SIGMA_ANG = 0.020
from backend.autoconfig import AutoConfigEngine
from backend.autoconfig_bases import ProblemShape, count_spectral_rows, infer_optimizer_bases
from backend.quantum import (
    QuantumEngine,
    parse_orca_rovib_alpha,
    _detect_bonds,
    _detect_angles,
    wilson_B,
)
from backend.base_backend import QuantumState
from backend.registry import get_backend
from backend.spectral.SVD import SubspaceOptimizer
from backend.internal.internal_fit import (
    InternalCoordinateSet,
    apply_internal_step,
    spectral_jacobian_q,
    quantum_terms_q,
    build_internal_priors,
)


def build_correction_from_iso(iso, method=None, basis=None, backend=None):
    """Build a :class:`RovibCorrection` from isotopologue dict legacy/new fields.

    This keeps backward compatibility for runs that provide correction vectors
    directly on isotopologue inputs without an explicit `rovib_correction` object.
    """
    name = str(iso.get("name", "iso"))
    corr = RovibCorrection(
        isotopologue=name,
        method=method,
        basis=basis,
        backend=backend,
        source=str(iso.get("rovib_source", "iso_input")),
        status="ok",
    )

    idx = np.asarray(
        iso.get("component_indices", list(range(len(iso.get("obs_constants", []))))),
        dtype=int,
    )
    alpha = np.asarray(iso.get("alpha_constants", np.zeros(len(idx))), dtype=float).ravel()
    dv = iso.get("delta_vib_constants")
    de = iso.get("delta_elec_constants")
    db = iso.get("delta_bob_constants")
    sc = iso.get("sigma_correction_constants")

    def _pick(vec, k):
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=float).ravel()
        if k >= arr.size:
            return None
        v = float(arr[k])
        return v if np.isfinite(v) else None

    for k, comp in enumerate(idx):
        c = int(comp)
        if c == 0:
            if k < alpha.size and np.isfinite(alpha[k]):
                corr.alpha_A = float(alpha[k])
            v = _pick(dv, k)
            if v is not None:
                corr.delta_vib_A = v
            v = _pick(de, k)
            if v is not None:
                corr.delta_elec_A = v
            v = _pick(db, k)
            if v is not None:
                corr.delta_bob_A = v
            v = _pick(sc, k)
            if v is not None:
                corr.sigma_delta_A = v
        elif c == 1:
            if k < alpha.size and np.isfinite(alpha[k]):
                corr.alpha_B = float(alpha[k])
            v = _pick(dv, k)
            if v is not None:
                corr.delta_vib_B = v
            v = _pick(de, k)
            if v is not None:
                corr.delta_elec_B = v
            v = _pick(db, k)
            if v is not None:
                corr.delta_bob_B = v
            v = _pick(sc, k)
            if v is not None:
                corr.sigma_delta_B = v
        elif c == 2:
            if k < alpha.size and np.isfinite(alpha[k]):
                corr.alpha_C = float(alpha[k])
            v = _pick(dv, k)
            if v is not None:
                corr.delta_vib_C = v
            v = _pick(de, k)
            if v is not None:
                corr.delta_elec_C = v
            v = _pick(db, k)
            if v is not None:
                corr.delta_bob_C = v
            v = _pick(sc, k)
            if v is not None:
                corr.sigma_delta_C = v

    if iso.get("pred_cd") is not None:
        corr.pred_cd = iso.get("pred_cd")

    return corr


# Backward-compatible re-export so callers that imported _find_orca from this
# module (e.g. orca_cheap_opt.py before it was updated) continue to work.
from backend.orca.orca_backend import _find_orca  # noqa: F401


class MolecularOptimizer:
    """
    Parameters
    ----------
    coords : array-like (N, 3)
        Initial Cartesian coordinates in Angstroms.
    elems : list[str]
        Element symbols in atom order.
    isotopologues : list[dict]
        Passed to SpectralEngine.  Each dict needs
        'masses' (N,) [amu] and 'obs_constants' (3,) [MHz].
    orca_executable : str or None
        Path to the ORCA binary, bare name 'orca' if on PATH, or None to
        auto-detect.  If ORCA cannot be found and no pre-computed files are
        loaded via load_orca(), run() will raise an error.
    orca_method : str
        ORCA method keyword, e.g. 'CCSD(T)' or 'wB97X-D3'.
    orca_basis : str
        Basis set keyword, e.g. 'cc-pVTZ'.
    charge : int
    multiplicity : int
    workdir : str
        Directory where ORCA input/output files are written.
    max_iter : int
        Maximum optimisation iterations.
    conv_step : float
        Convergence threshold on Cartesian step norm [Ã…].
    conv_freq : float
        Convergence threshold on rotational-constant RMS residual [MHz].
    conv_energy : float
        Convergence threshold on absolute energy change between iterations
        [Hartree]. Used for hybrid stall detection and optionally for null-space
        convergence when ``null_convergence_requires_energy`` is True.
    spectral_analytic_jacobian : bool
        If True (default), ``SpectralEngine`` uses an analytic âˆ‚(A,B,C)/âˆ‚x with
        finite-difference fallback for degenerate principal moments.
    spectral_jacobian_degeneracy_tol : float
        Relative moment gap below which the Jacobian falls back to finite differences.
    null_convergence_requires_energy : bool
        If True, null-space convergence also requires ``|Î”E| < conv_energy``.
        Default False avoids stalling when energy differences fluctuate iteration-to-iteration.
    conv_step_range : float
        Convergence threshold on the range-space component of the Cartesian
        step norm [Ã…].
    conv_step_null : float
        Convergence threshold on the null-space component of the Cartesian
        step norm [Ã…].
    conv_grad_null : float
        Convergence threshold on the projected null-space gradient norm
        [Hartree/Ã…].
    orca_update_thresh : float
        Re-run ORCA when RMS geometry drift from last ORCA point exceeds
        this value [Ã…].  Default 0.005 Ã….
    hess_recalc_every : int
        Recalculate the full Hessian every N ORCA calls.  Between recalculations
        only a cheap gradient (EnGrad) job is run.  Default 1 (always recalculate).
    sv_threshold : float
        Relative singular-value cutoff for range/null-space split.
    trust_radius : float
        Maximum step size [Ã…].
    lambda_damp : float
        Levenbergâ€“Marquardt regularisation on the null-space Hessian.
    """

    def __init__(
        self,
        coords,
        elems,
        isotopologues,
        quantum_backend="Psi4",
        orca_executable=None,
        orca_method="CCSD(T)",
        orca_basis="cc-pVTZ",
        psi4_method="B3LYP",
        psi4_basis="cc-pVDZ",
        psi4_memory="2 GB",
        psi4_num_threads=1,
        psi4_output_file=os.devnull,
        charge=0,
        multiplicity=1,
        workdir=".",
        max_iter=500,
        conv_step=1e-7,
        conv_freq=1.0,
        conv_energy=1e-8,
        spectral_accept_relax=0.0,
        conv_step_range=1e-6,
        conv_step_null=1e-5,
        conv_grad_null=1e-4,
        orca_update_thresh=0.005,
        hess_recalc_every=1,
        adaptive_hess_schedule=True,
        hess_recalc_min=1,
        hess_recalc_max=8,
        sv_threshold=1e-3,
        sv_min_abs=None,
        trust_radius=0.1,
        null_trust_radius=None,
        lambda_damp=1e-4,
        objective_mode="joint",
        alpha_quantum=1.0,
        quantum_prior_sigma_ang=DEFAULT_QUANTUM_PRIOR_SIGMA_ANG,
        robust_loss="none",
        robust_param=1.0,
        sigma_floor_mhz=0.0,
        sigma_cap_mhz=None,
        max_spectral_weight=None,
        component_weight_map=None,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        use_internal_priors=False,
        prior_weight=1.0,
        prior_auto_from_initial=True,
        prior_use_dihedrals=False,
        prior_sigma_bond=0.05,
        prior_sigma_angle_deg=3.0,
        prior_sigma_dihedral_deg=15.0,
        use_conformer_mixture=False,
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        conformer_energy_unit="kcal/mol",
        conformer_summary=None,
        spectral_delta=1e-3,
        spectral_analytic_jacobian=True,
        spectral_jacobian_degeneracy_tol=1e-4,
        null_convergence_requires_energy=False,
        auto_sanitize_spectral=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        enable_geometry_guardrails=False,
        guardrail_bond_scale_min=0.65,
        guardrail_bond_scale_max=1.45,
        guardrail_clash_scale=0.60,
        guardrail_max_violations=0,
        accept_requires_geometry_valid=True,
        guardrail_lambda_boost=2.0,
        guardrail_trust_shrink=0.8,
        use_internal_preconditioner=False,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        method_preset=None,
        use_orca_rovib=False,
        rovib_recalc_every=1,
        rovib_source_mode="hybrid_auto",
        spectral_only=False,
        symmetry=None,
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        project_rigid_modes=False,
        enforce_quantum_descent=False,
        quantum_descent_tol=1e-10,
        use_autoconfig=True,
        use_autoconfig_heuristic_bases=True,
        autoconfig_tune_sv_threshold=True,
        autoconfig_tune_alpha_quantum=True,
        autoconfig_update_every=1,
        autoconfig_smoothing=0.4,
        correction_table=None,
        correction_mode="hybrid_auto",
        correction_sigma_vib_fraction=0.1,
        correction_elec=False,
        correction_sigma_elec_fraction=0.1,
        correction_bob_params=None,
        correction_g_tensor=None,
        harmonic_from_hessian=False,
        harmonic_sigma_fraction=0.02,
        anharmonic_from_hessian=False,
        anharmonic_fd_delta_ang=0.01,
        nonconvergent_policy="warn",
        harmonic_cd_from_hessian=False,
        cd_sigma_fraction=0.05,
        fit_cd_constants=False,
        cd_weight=0.0,
        coordinate_mode="internal",
        ic_damping=1e-6,
        ic_use_dihedrals=False,
        ic_micro_iter=20,
        ic_prior_weight=1.0,
        ic_prior_sigma_bond=0.05,
        ic_prior_sigma_angle_deg=3.0,
        ic_prior_sigma_dihedral_deg=15.0,
        ic_prior_mode="soft",
        ic_user_priors=None,
        ic_freeze_sigma_floor=1e-6,
        ic_prior_adaptive=None,
    ):
        self.coords = np.asarray(coords, dtype=float).copy()
        self.elems = list(elems)
        _valid_coord_modes = ("cartesian", "internal")
        _coord_mode = str(coordinate_mode).strip().lower()
        if _coord_mode not in _valid_coord_modes:
            raise ValueError(
                f"coordinate_mode must be one of {_valid_coord_modes}, got '{coordinate_mode}'"
            )
        self.coordinate_mode = _coord_mode
        self.conformer_summary = conformer_summary
        if method_preset is not None:
            preset_method, preset_basis = self._method_preset(method_preset)
            orca_method = preset_method
            orca_basis = preset_basis

        # â”€â”€ Rovibrational corrections (M1-M4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._harmonic_from_hessian = bool(harmonic_from_hessian)
        self._harmonic_sigma_fraction = max(float(harmonic_sigma_fraction), 1e-6)
        self._anharmonic_from_hessian = bool(anharmonic_from_hessian)
        self._anharmonic_fd_delta_ang = max(float(anharmonic_fd_delta_ang), 1e-4)
        self._nonconvergent_policy = str(nonconvergent_policy or "warn").strip().lower()
        self._harmonic_cd_from_hessian = bool(harmonic_cd_from_hessian)
        self._warned_cd_unvalidated = False
        self._cd_sigma_fraction = max(float(cd_sigma_fraction), 1e-6)
        self._fit_cd_constants = bool(fit_cd_constants)
        self._cd_weight = max(float(cd_weight), 0.0)
        self._prev_harmonic_alpha_sum: dict = {}   # {(iso_name, comp): float} for drift tracking
        if self._fit_cd_constants and self._cd_weight <= 0.0:
            print(
                "[cd-warning] fit_cd_constants=True but cd_weight=0; "
                "CD observations will not affect the optimization."
            )
        self._correction_elec = bool(correction_elec)
        self._correction_sigma_elec_fraction = float(correction_sigma_elec_fraction)
        self._correction_bob_params = correction_bob_params or None
        self._correction_g_tensor = correction_g_tensor or None
        self._raw_isotopologues = list(isotopologues)   # preserved for harmonic updates
        self._corrected_targets = None
        _ctbl = parse_correction_table(correction_table)
        _apply_corrections = bool(_ctbl) or correction_mode != "hybrid_auto"
        if _apply_corrections or (correction_table is not None):
            _ctbl = parse_correction_table(correction_table)
        if _ctbl or correction_elec or correction_bob_params or correction_g_tensor:
            _corrected_targets = resolve_corrections(
                isotopologues,
                correction_table=_ctbl,
                mode=str(correction_mode).strip().lower(),
                sigma_vib_fraction=float(correction_sigma_vib_fraction),
                elems=list(elems),
                correction_elec=bool(correction_elec),
                sigma_elec_fraction=float(correction_sigma_elec_fraction),
                correction_bob_params=correction_bob_params or None,
                g_tensor=correction_g_tensor or None,
            )
            _qc_warnings = validate_correction_quality(_corrected_targets)
            print("\nRovibrational corrections applied:")
            print(correction_summary(_corrected_targets))
            for w in _qc_warnings:
                print(f"[correction-warning] {w}")
            isotopologues = apply_corrections_to_isotopologues(isotopologues, _corrected_targets)
            self._corrected_targets = _corrected_targets
            if use_orca_rovib:
                print(
                    "[correction-warning] use_orca_rovib=True is ignored when correction_table "
                    "is supplied â€” corrections are pre-applied and alpha_constants are zeroed."
                )
                use_orca_rovib = False

        self.auto_sanitize_spectral = bool(auto_sanitize_spectral)
        self.sanitize_jacobian_row_norm_max = float(sanitize_jacobian_row_norm_max)
        self.sanitize_tiny_target_mhz = float(sanitize_tiny_target_mhz)
        # Materialize per-iso RovibCorrection from user-supplied keys (deltas,
        # alpha tables, sigma_correction). This populates delta_total upfront
        # so the sanitizer carries it through cleanly.
        prepped_isos = []
        for iso in isotopologues:
            iso_copy = dict(iso)
            corr = iso_copy.get("rovib_correction")
            if not isinstance(corr, RovibCorrection):
                corr = build_correction_from_iso(
                    iso_copy,
                    method=orca_method,
                    basis=orca_basis,
                    backend=quantum_backend,
                )
            iso_copy["rovib_correction"] = corr
            if iso_copy.get("pred_cd") is not None:
                corr.pred_cd = iso_copy["pred_cd"]
            # Compute delta_total aligned to the iso's component_indices.
            idx = np.asarray(
                iso_copy.get(
                    "component_indices",
                    list(range(len(iso_copy.get("obs_constants", [])))),
                ),
                dtype=int,
            )
            total = corr.delta_total_vector()  # length 3
            dt = np.full(len(idx), np.nan, dtype=float)
            for k, comp in enumerate(idx):
                c = int(comp)
                if 0 <= c < 3 and np.isfinite(total[c]):
                    dt[k] = float(total[c])
            if np.any(np.isfinite(dt)):
                iso_copy["delta_total_constants"] = dt
            # sigma_correction_constants from the correction object (if any).
            sd = corr.sigma_delta_vector()
            sc = np.zeros(len(idx), dtype=float)
            any_sc = False
            for k, comp in enumerate(idx):
                c = int(comp)
                if 0 <= c < 3 and np.isfinite(sd[c]):
                    sc[k] = float(sd[c])
                    any_sc = True
            if any_sc and "sigma_correction_constants" not in iso_copy:
                iso_copy["sigma_correction_constants"] = sc
            prepped_isos.append(iso_copy)
        isotopologues = prepped_isos
        spectral_isotopologues = isotopologues
        if self.auto_sanitize_spectral:
            spectral_isotopologues, sanitize_notes = sanitize_isotopologues(
                isotopologues=isotopologues,
                coords=self.coords,
                delta=spectral_delta,
                jacobian_row_norm_max=self.sanitize_jacobian_row_norm_max,
                tiny_target_mhz=self.sanitize_tiny_target_mhz,
            )
            for msg in sanitize_notes:
                print(f"[spectral-sanitize] {msg}")
        self.spectral = SpectralEngine(
            spectral_isotopologues,
            delta=spectral_delta,
            robust_loss=robust_loss,
            robust_param=robust_param,
            sigma_floor_mhz=sigma_floor_mhz,
            sigma_cap_mhz=sigma_cap_mhz,
            max_weight=max_spectral_weight,
            component_weight_map=component_weight_map,
            torsion_aware_weighting=torsion_aware_weighting,
            torsion_a_weight=torsion_a_weight,
            conformer_defs=(conformer_defs if use_conformer_mixture else None),
            conformer_reference_coords=self.coords,
            conformer_weight_mode=conformer_weight_mode,
            conformer_temperature_k=conformer_temperature_k,
            conformer_energy_unit=conformer_energy_unit,
            analytic_jacobian=bool(spectral_analytic_jacobian),
            jacobian_degeneracy_tol=float(spectral_jacobian_degeneracy_tol),
            cd_weight=self._cd_weight,
            fit_cd_constants=self._fit_cd_constants,
            cd_sigma_fraction=self._cd_sigma_fraction,
        )
        if any(bool(iso.get("torsion_sensitive", False)) for iso in self.spectral.isotopologues):
            self.spectral.torsion_aware_weighting = True
        self.use_conformer_mixture = bool(use_conformer_mixture)
        self.use_internal_priors = bool(use_internal_priors)
        self.prior_weight = max(float(prior_weight), 0.0)
        self._base_prior_weight = self.prior_weight
        self._base_spectral_accept_relax = max(0.0, float(spectral_accept_relax))
        self._base_sigma_floor_mhz = max(float(sigma_floor_mhz), 0.0)
        self._base_max_spectral_weight = (
            None if max_spectral_weight is None else max(float(max_spectral_weight), 1e-12)
        )
        self._base_torsion_a_weight = max(float(torsion_a_weight), 1e-12)
        self.internal_prior = None
        if self.use_internal_priors and self.prior_weight > 0.0:
            self.internal_prior = InternalPriorEngine(
                coords=self.coords,
                elems=self.elems,
                use_dihedrals=bool(prior_use_dihedrals),
                prior_targets=None,
                prior_sigmas=None,
                auto_from_initial=bool(prior_auto_from_initial),
                sigma_bond=float(prior_sigma_bond),
                sigma_angle_deg=float(prior_sigma_angle_deg),
                sigma_dihedral_deg=float(prior_sigma_dihedral_deg),
            )
        # An absolute floor on the singular values, in the same units as the
        # sigma-weighted Jacobian: 1/s is the parameter uncertainty along a
        # direction, in Angstrom. Tying the floor to the prior width states the
        # rule plainly -- the data owns a direction only where it resolves that
        # direction better than the quantum surface is trusted to. A floor of
        # 0.0 (the old default) let any direction above the *relative* cutoff go
        # to the data, including ones it barely resolves, which is how hydrogen
        # positions ended up tens of milli-Angstrom out.
        if sv_min_abs is None:
            sigma_x = quantum_prior_sigma_ang or DEFAULT_QUANTUM_PRIOR_SIGMA_ANG
            sv_min_abs = 1.0 / float(sigma_x) if sigma_x > 0.0 else 0.0

        self.optimizer = SubspaceOptimizer(
            sv_threshold,
            sv_min_abs,
            trust_radius,
            null_trust_radius,
            lambda_damp,
            objective_mode=objective_mode,
            alpha_quantum=alpha_quantum,
            quantum_prior_sigma_ang=quantum_prior_sigma_ang,
            dynamic_quantum_weight=dynamic_quantum_weight,
            quantum_weight_beta=quantum_weight_beta,
            quantum_weight_min=quantum_weight_min,
            quantum_weight_max=quantum_weight_max,
            use_internal_preconditioner=use_internal_preconditioner,
        )
        self._base_trust_radius = float(self.optimizer.trust_radius)
        self._base_null_trust_radius = float(self.optimizer.null_trust_radius)
        self._base_lambda_damp = float(self.optimizer.lambda_damp)
        self._base_sv_threshold = float(self.optimizer.sv_threshold)
        self._base_alpha_quantum = float(self.optimizer.alpha_quantum)

        self.use_autoconfig = bool(use_autoconfig)
        self.use_autoconfig_heuristic_bases = bool(use_autoconfig_heuristic_bases)
        self.autoconfig_tune_sv_threshold = bool(autoconfig_tune_sv_threshold)
        self.autoconfig_tune_alpha_quantum = bool(autoconfig_tune_alpha_quantum)
        # Declared before the heuristic-bases call below, which reads it. The
        # AutoConfigEngine itself is built further down, once every base value
        # it seeds from is final; the reseed inside the call is therefore a
        # no-op here and the engine picks up these bases directly.
        self.autoconfig = None
        if self.use_autoconfig and self.use_autoconfig_heuristic_bases:
            self._apply_heuristic_optimizer_bases(isotopologues)

        self.orca_method = orca_method
        self.orca_basis = orca_basis
        self.quantum_backend = str(quantum_backend).strip().lower()
        self.psi4_method = psi4_method
        self.psi4_basis = psi4_basis
        self.psi4_memory = psi4_memory
        self.psi4_num_threads = int(psi4_num_threads)
        self.psi4_output_file = psi4_output_file
        self.charge = charge
        self.multiplicity = multiplicity
        self.workdir = os.path.abspath(workdir)

        self.max_iter = max_iter
        self.conv_step = conv_step
        self.conv_freq = conv_freq
        self.conv_energy = float(conv_energy)
        self.null_convergence_requires_energy = bool(null_convergence_requires_energy)
        self.spectral_accept_relax = self._base_spectral_accept_relax
        self.conv_step_range = float(conv_step_range)
        self.conv_step_null = float(conv_step_null)
        self.conv_grad_null = float(conv_grad_null)
        self.orca_update_thresh = orca_update_thresh
        self.hess_recalc_every = max(1, int(hess_recalc_every))
        self.adaptive_hess_schedule = bool(adaptive_hess_schedule)
        self.hess_recalc_min = max(1, int(hess_recalc_min))
        self.hess_recalc_max = max(self.hess_recalc_min, int(hess_recalc_max))
        self.use_orca_rovib = use_orca_rovib
        self.rovib_recalc_every = max(1, int(rovib_recalc_every))
        self.rovib_source_mode = str(rovib_source_mode).strip().lower()

        self.spectral_only = bool(spectral_only)
        self.symmetry = symmetry
        if isinstance(self.symmetry, str) or self.symmetry is None:
            if self.symmetry is not None:
                from backend.symmetry import create_symmetry  # pylint: disable=import-outside-toplevel
                self.symmetry = create_symmetry(
                    self.symmetry,
                    self.elems,
                    self.coords,
                )
        self.debug_rank_diagnostics = bool(debug_rank_diagnostics)
        self.debug_sv_count = max(1, int(debug_sv_count))
        self.project_rigid_modes = bool(project_rigid_modes)
        self.enforce_quantum_descent = bool(enforce_quantum_descent)
        self.quantum_descent_tol = float(quantum_descent_tol)
        self.enable_geometry_guardrails = bool(enable_geometry_guardrails)
        self.guardrail_bond_scale_min = float(guardrail_bond_scale_min)
        self.guardrail_bond_scale_max = float(guardrail_bond_scale_max)
        self.guardrail_clash_scale = float(guardrail_clash_scale)
        self.guardrail_max_violations = max(0, int(guardrail_max_violations))
        self.accept_requires_geometry_valid = bool(accept_requires_geometry_valid)
        self.guardrail_lambda_boost = max(1.0, float(guardrail_lambda_boost))
        self.guardrail_trust_shrink = min(1.0, max(0.1, float(guardrail_trust_shrink)))
        self.autoconfig_update_every = max(1, int(autoconfig_update_every))
        if self.use_autoconfig:
            self.autoconfig = AutoConfigEngine(
                n_params=3 * len(self.elems),
                base_trust_radius=self._base_trust_radius,
                base_null_trust_radius=self._base_null_trust_radius,
                base_lambda_damp=self._base_lambda_damp,
                base_prior_weight=self._base_prior_weight,
                base_sigma_floor_mhz=self._base_sigma_floor_mhz,
                base_max_spectral_weight=self._base_max_spectral_weight,
                base_torsion_a_weight=self._base_torsion_a_weight,
                base_sv_threshold=self._base_sv_threshold,
                base_alpha_quantum=self._base_alpha_quantum,
                tune_sv_threshold=self.autoconfig_tune_sv_threshold,
                tune_alpha_quantum=self.autoconfig_tune_alpha_quantum,
                smoothing=float(autoconfig_smoothing),
            )
        self._ic_damping = max(float(ic_damping), 1e-14)
        self._ic_use_dihedrals = bool(ic_use_dihedrals)
        self._ic_micro_iter = max(1, int(ic_micro_iter))
        self._ic_prior_weight = max(float(ic_prior_weight), 0.0)
        self._ic_prior_sigma_bond = float(ic_prior_sigma_bond)
        self._ic_prior_sigma_angle_deg = float(ic_prior_sigma_angle_deg)
        self._ic_prior_sigma_dihedral_deg = float(ic_prior_sigma_dihedral_deg)
        self._ic_prior_mode = str(ic_prior_mode or "soft").strip().lower()
        self._ic_user_priors = list(ic_user_priors or [])
        self._ic_freeze_sigma_floor = max(float(ic_freeze_sigma_floor), 1e-12)
        self._ic_prior_adaptive = dict(ic_prior_adaptive or {})
        _valid_prior_modes = {"off", "soft", "adaptive", "hard_freeze"}
        if self._ic_prior_mode not in _valid_prior_modes:
            raise ValueError(
                f"ic_prior_mode must be one of {_valid_prior_modes}, got '{self._ic_prior_mode}'"
            )
        self._ic_initial_coords = None   # captured once after first geometry is confirmed
        if self.coordinate_mode == "internal":
            print(f"Internal-coordinate mode enabled: bonds+{'dihedrals+' if self._ic_use_dihedrals else ''}angles, "
                  f"damping={self._ic_damping:.1e}, prior_weight={self._ic_prior_weight:.2f}")

        self._rigid_ref_masses = None
        if len(self.spectral.isotopologues) > 0:
            m = np.asarray(self.spectral.isotopologues[0]["masses"], dtype=float)
            if m.size == len(self.elems):
                self._rigid_ref_masses = m.copy()
        self.quantum = None
        self._backend = None
        self._orca_ref_coords = None
        self._orca_call_count = 0
        self.history = []
        self._guardrail_bonds = _detect_bonds(self.coords, self.elems) if self.enable_geometry_guardrails else []

        if self.spectral_only:
            print("Spectral-only mode: ORCA disabled. Null-space step will be zero.")
        else:
            backend_cls = get_backend(self.quantum_backend)
            if self.quantum_backend == "orca":
                self._backend = backend_cls(
                    elems=self.elems,
                    workdir=self.workdir,
                    method=self.orca_method,
                    basis=self.orca_basis,
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    executable=orca_executable,
                    rovib_source_mode=self.rovib_source_mode,
                )
            elif self.quantum_backend == "psi4":
                try:
                    self._backend = backend_cls(
                        elems=self.elems,
                        method=self.psi4_method,
                        basis=self.psi4_basis,
                        charge=self.charge,
                        multiplicity=self.multiplicity,
                        memory=self.psi4_memory,
                        num_threads=self.psi4_num_threads,
                        output_file=self.psi4_output_file,
                    )
                    print("Psi4 backend initialized.")
                except Exception as e:
                    self._backend = None
                    print(f"Note: Could not initialize Psi4 backend: {e}")
            else:
                # Generic path for registered third-party backends. Pass the
                # method/basis/charge the config actually specified; constructing
                # with elems alone silently discarded them, so a new backend could
                # only ever run at its own defaults.
                self._backend = backend_cls(
                    elems=self.elems,
                    method=self.orca_method,
                    basis=self.orca_basis,
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                )

    @staticmethod
    def _covalent_radius(elem):
        table = {
            "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
            "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39,
        }
        key = str(elem).strip().upper()
        return table.get(key, 0.77)

    def _geometry_validity(self, trial_coords):
        if not self.enable_geometry_guardrails:
            return True, {"violations": 0, "bond_ratio_min": np.nan, "bond_ratio_max": np.nan, "clash_ratio_min": np.nan}
        coords = np.asarray(trial_coords, dtype=float)
        n = len(self.elems)
        bonded = {tuple(sorted((int(i), int(j)))) for i, j in self._guardrail_bonds}
        bond_ratios = []
        clash_ratios = []
        violations = 0

        for i, j in bonded:
            ri = self._covalent_radius(self.elems[i])
            rj = self._covalent_radius(self.elems[j])
            ref = max(ri + rj, 1e-6)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            ratio = d / ref
            bond_ratios.append(ratio)
            if ratio < self.guardrail_bond_scale_min or ratio > self.guardrail_bond_scale_max:
                violations += 1

        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bonded:
                    continue
                ri = self._covalent_radius(self.elems[i])
                rj = self._covalent_radius(self.elems[j])
                ref = max((ri + rj) * self.guardrail_clash_scale, 1e-6)
                d = float(np.linalg.norm(coords[i] - coords[j]))
                ratio = d / ref
                clash_ratios.append(ratio)
                if ratio < 1.0:
                    violations += 1

        valid = violations <= self.guardrail_max_violations
        stats = {
            "violations": int(violations),
            "bond_ratio_min": float(np.min(bond_ratios)) if bond_ratios else np.nan,
            "bond_ratio_max": float(np.max(bond_ratios)) if bond_ratios else np.nan,
            "clash_ratio_min": float(np.min(clash_ratios)) if clash_ratios else np.nan,
        }
        return valid, stats

    @staticmethod
    def _method_preset(name):
        key = str(name).strip().lower()
        presets = {
            "fast": ("r2SCAN-3c", ""),
            "balanced": ("wB97X-D4", "def2-TZVPP"),
            "high": ("wB97X-D4", "def2-QZVPP"),
            "mp2": ("MP2", "cc-pVTZ"),
        }
        if key not in presets:
            raise ValueError(f"Unknown method_preset '{name}'.")
        return presets[key]

    def _run_hessian(self):
        """Full Freq job: refreshes both gradient and Hessian via the active backend."""
        if self._backend is None:
            raise RuntimeError("No quantum backend initialized.")
        result = self._backend.run_hessian(self.coords)
        self.quantum = QuantumState(result.energy, result.gradient_bohr, result.hessian_bohr)
        self._orca_ref_coords = self.coords.copy()

    def _run_gradient(self):
        """Cheap gradient-only job: refreshes gradient, keeps existing Hessian."""
        if self._backend is None:
            raise RuntimeError("No quantum backend initialized.")
        result = self._backend.run_gradient(self.coords)
        self.quantum.energy = result.energy
        self.quantum._gradient_bohr = result.gradient_bohr
        self._orca_ref_coords = self.coords.copy()


    def _refresh_iso_delta_total(self, iso, correction: RovibCorrection):
        """Compute delta_total_constants from a correction and store on iso."""
        idx = np.asarray(iso["component_indices"], dtype=int)
        total = correction.delta_total_vector()  # length 3
        out = np.full(len(idx), np.nan, dtype=float)
        for k, comp in enumerate(idx):
            c = int(comp)
            if 0 <= c < 3 and np.isfinite(total[c]):
                out[k] = float(total[c])
        if np.any(np.isfinite(out)):
            iso["delta_total_constants"] = out
        # Also fold sigma_delta -> sigma_correction_constants.
        sd = correction.sigma_delta_vector()
        sigma_corr = np.zeros(len(idx), dtype=float)
        any_set = False
        for k, comp in enumerate(idx):
            c = int(comp)
            if 0 <= c < 3 and np.isfinite(sd[c]):
                sigma_corr[k] = float(sd[c])
                any_set = True
        if any_set:
            iso["sigma_correction_constants"] = sigma_corr

    def _run_rovib(self):
        """Delegate VPT2 rovibrational corrections to the active backend."""
        if self._backend is None:
            return
        result = self._backend.run_rovib(
            self.coords, isotopologues=self.spectral.isotopologues
        )
        if result is None:
            return
        iso_by_name = {str(iso.get("name", "iso")): iso for iso in self.spectral.isotopologues}
        for corr_dict in result.isotopologue_corrections:
            label = corr_dict["name"]
            iso = iso_by_name.get(label)
            if iso is None:
                continue
            iso["alpha_constants"] = corr_dict["alpha_constants"]
            iso["rovib_correction"] = corr_dict["rovib_correction"]
            self._refresh_iso_delta_total(iso, corr_dict["rovib_correction"])
        # Cross-check: compare VPT2 alpha with harmonic Hessian alpha (diagnostic only).
        if self.quantum is not None and getattr(self.quantum, "_hessian_bohr", None) is not None:
            self._reconcile_vpt2_vs_harmonic(result.isotopologue_corrections)

    def _reconcile_vpt2_vs_harmonic(self, isotopologue_corrections):
        """Compare VPT2 alpha (from ORCA) with harmonic alpha computed from the Hessian.

        Prints a per-isotopologue, per-component table.  Flags components where
        |Î”Î±| > 2 Ã— harmonic_sigma â€” which suggests the harmonic uncertainty estimate
        may be too tight or a resonance is affecting the VPT2 result.
        """
        from backend.spectral.harmonic_alpha import compute_harmonic_alpha  # pylint: disable=import-outside-toplevel

        print("  [vpt2-check] VPT2 vs harmonic-Hessian alpha cross-check:")
        _labels = ["A", "B", "C"]
        for corr_dict in isotopologue_corrections:
            label = str(corr_dict["name"])
            iso_raw = next(
                (i for i in self._raw_isotopologues if str(i.get("name", "iso")) == label),
                None,
            )
            if iso_raw is None:
                continue
            masses = list(iso_raw.get("masses", []))
            if not masses:
                continue
            try:
                h_alpha, _, h_sigma, _ = compute_harmonic_alpha(
                    self.quantum._hessian_bohr,
                    self.coords,
                    masses,
                    sigma_fraction=self._harmonic_sigma_fraction,
                )
            except Exception as exc:
                print(f"  [vpt2-check]   {label}: harmonic computation failed ({exc})")
                continue
            corr = corr_dict.get("rovib_correction")
            if corr is None:
                continue
            vpt2_map = {
                "A": getattr(corr, "alpha_A", None),
                "B": getattr(corr, "alpha_B", None),
                "C": getattr(corr, "alpha_C", None),
            }
            parts = []
            for lbl in _labels:
                v_vpt2 = vpt2_map.get(lbl)
                if v_vpt2 is None:
                    continue
                v_harm = h_alpha.get(lbl, 0.0)
                sig = h_sigma.get(lbl, 1.0)
                diff = abs(float(v_vpt2) - float(v_harm))
                flag = "  [>2Ïƒ]" if diff > 2.0 * sig else ""
                parts.append(f"{lbl}: VPT2={float(v_vpt2):+.1f} harm={v_harm:+.1f} Î”={diff:.1f} MHz{flag}")
            if parts:
                print(f"  [vpt2-check]   {label}: {';  '.join(parts)}")

    def _apply_harmonic_alpha_corrections(self):
        """Recompute harmonic alpha from the current Hessian and update spectral targets.

        Called once after the first Hessian computation when harmonic_from_hessian=True.
        Re-applies rovibrational + electronic corrections to the raw (uncorrected)
        isotopologue data using the current harmonic alpha values.
        """
        from backend.spectral.harmonic_alpha import build_correction_table_from_hessian  # pylint: disable=import-outside-toplevel

        if self.quantum is None:
            return
        hess_bohr = self.quantum._hessian_bohr

        # The anharmonic (cubic) term is the dominant contribution to alpha and
        # carries the opposite sign to the harmonic one, so omitting it biases
        # B_e systematically. It costs 6N extra Hessians, hence opt-in.
        hessian_fn = None
        if self._anharmonic_from_hessian:
            if self._backend is None:
                print(
                    "  [anharmonic] Requested but no quantum backend is available; "
                    "falling back to harmonic+Coriolis alpha only."
                )
            else:
                n_hess = 6 * len(self.elems)
                print(
                    f"  [anharmonic] Building Cartesian cubic force field "
                    f"({n_hess} Hessian evaluations)..."
                )

                def hessian_fn(coords_ang):
                    return self._backend.run_hessian(coords_ang).hessian_bohr

        label = "alpha (harmonic+Coriolis+anharmonic)" if hessian_fn else "harmonic alpha"
        print(f"\n  [harmonic-alpha] Computing {label} from Hessian...")
        ctbl_raw, _res_info = build_correction_table_from_hessian(
            hess_bohr,
            self.coords,
            self._raw_isotopologues,
            sigma_fraction=self._harmonic_sigma_fraction,
            hessian_fn=hessian_fn,
            fd_delta_cubic=self._anharmonic_fd_delta_ang,
            nonconvergent_policy=self._nonconvergent_policy,
        )
        for status in dict.fromkeys(_res_info.get("anharmonic_statuses", [])):
            if status not in ("cubic_fd", "not_requested"):
                print(f"  [anharmonic] WARNING: {status}")

        # A component whose cubic term exceeds its harmonic one has a diverging
        # perturbation series, and its correction is unreliable no matter how
        # small the formal sigma is. Worth saying loudly: when the spectral block
        # is exactly determined the fit reproduces its targets exactly, weights
        # never enter, and a bad correction goes straight into the geometry.
        _nonconv = _res_info.get("nonconvergent", {})
        if _nonconv:
            for _iso_name, _comps in _nonconv.items():
                _detail = ", ".join(
                    f"{c} (cubic/harmonic = {r:.1f})" for c, r in sorted(_comps.items())
                )
                print(
                    f"  [anharmonic] WARNING: {_iso_name}: perturbation series not "
                    f"converging for {_detail}."
                )
            _policy = _res_info.get("nonconvergent_policy", "warn")
            _dropped = _res_info.get("dropped_components", {})
            if _policy == "drop" and _dropped:
                for _iso_name, _comps in _dropped.items():
                    print(
                        f"  [anharmonic] {_iso_name}: dropped {', '.join(sorted(_comps))} "
                        "from the correction table (nonconvergent_policy=drop)."
                    )
            elif _policy == "inflate":
                print(
                    "  [anharmonic] Their sigma has been inflated by the divergence "
                    "ratio. Note this only\n  [anharmonic] changes the fit where the "
                    "spectral block is over-determined."
                )
            else:
                print(
                    "  [anharmonic] These corrections are unreliable. Prefer components "
                    "with a converging\n  [anharmonic] series, or improve the Hessian; "
                    "reweighting cannot compensate for a biased target. Set\n"
                    "  [anharmonic] rovibrational_corrections.nonconvergent_policy: drop "
                    "to exclude them."
                )
        _near_degen = _res_info.get("total_near_degen_skips", 0)
        if _near_degen > 0:
            print(
                f"  [harmonic-alpha] WARNING: {_near_degen} near-degenerate Coriolis pair(s) "
                "skipped (|Ï‰_sÂ²âˆ’Ï‰_rÂ²| < 0.01 cmâ»Â²). Alpha values may be less reliable for "
                "these modes (Fermi/Coriolis resonance region)."
            )
        if not ctbl_raw:
            print("  [harmonic-alpha] Warning: no alpha values computed; skipping update.")
            return

        # Print summary and track drift from previous computation
        max_delta = 0.0
        for iso_name, comps in ctbl_raw.items():
            parts = []
            for comp_lbl in ("A", "B", "C"):
                if comp_lbl not in comps:
                    continue
                v = comps[comp_lbl]["alpha_sum_mhz"]
                s = comps[comp_lbl]["sigma_mhz"]
                key = (iso_name, comp_lbl)
                prev = self._prev_harmonic_alpha_sum.get(key)
                if prev is not None:
                    delta = abs(v - prev)
                    max_delta = max(max_delta, delta)
                    parts.append(f"{comp_lbl}={v:+.1f}Â±{s:.1f} (Î”={delta:+.2f})")
                else:
                    parts.append(f"{comp_lbl}={v:+.1f}Â±{s:.1f}")
                self._prev_harmonic_alpha_sum[key] = v
            print(f"  [harmonic-alpha]   {iso_name}: Î£Î± = {', '.join(parts)} MHz")
        if max_delta > 0.0:
            print(f"  [harmonic-alpha] Max alpha drift since last Hessian: {max_delta:.2f} MHz")

        from backend.spectral.correction_models import parse_correction_table  # pylint: disable=import-outside-toplevel
        ctbl = parse_correction_table(ctbl_raw)
        corrected_targets = resolve_corrections(
            self._raw_isotopologues,
            correction_table=ctbl,
            mode="user_only",
            sigma_vib_fraction=0.0,
            elems=list(self.elems),
            correction_elec=self._correction_elec,
            sigma_elec_fraction=self._correction_sigma_elec_fraction,
            correction_bob_params=self._correction_bob_params,
            g_tensor=self._correction_g_tensor,
        )
        _qc_warnings = validate_correction_quality(corrected_targets)
        print("\n  Rovibrational corrections (harmonic from Hessian):")
        print(correction_summary(corrected_targets))
        for w in _qc_warnings:
            print(f"  [correction-warning] {w}")
        self._corrected_targets = corrected_targets
        corrected_isos = apply_corrections_to_isotopologues(
            self._raw_isotopologues, corrected_targets
        )
        # Update the SpectralEngine isotopologue data in-place
        iso_by_name = {str(iso.get("name", "iso")): iso for iso in self.spectral.isotopologues}
        for new_iso in corrected_isos:
            name = str(new_iso.get("name", "iso"))
            old_iso = iso_by_name.get(name)
            if old_iso is None:
                continue
            old_iso["obs_constants"]   = new_iso["obs_constants"]
            old_iso["alpha_constants"] = new_iso["alpha_constants"]
            old_iso["sigma_constants"] = new_iso["sigma_constants"]

    def _apply_harmonic_cd_corrections(self):
        """Attach harmonic CD predictions from the current Hessian to rovib_correction.pred_cd."""
        from backend.spectral.centrifugal_distortion import build_cd_table_from_hessian

        if self.quantum is None:
            return
        hess_bohr = self.quantum._hessian_bohr
        print("\n  [harmonic-cd] Computing harmonic CD constants from Hessian...")
        if not self._warned_cd_unvalidated:
            self._warned_cd_unvalidated = True
            print(
                "  [harmonic-cd] WARNING: the tau' -> Watson A-reduction mapping "
                "is not validated.\n"
                "                Measured against water's experimental constants it "
                "gets DJ and DK\n"
                "                with the WRONG SIGN (-66.9 vs +37.6, -7.0 vs +973.3) "
                "and DJK nine\n"
                "                times too small. These are not order-of-magnitude "
                "estimates; treat\n"
                "                them as diagnostics only. See "
                "dev/tests/test_cd_mapping_validation.py."
            )
        cd_table = build_cd_table_from_hessian(
            hess_bohr,
            self.coords,
            self._raw_isotopologues,
            sigma_fraction=self._cd_sigma_fraction,
        )
        if not cd_table:
            print("  [harmonic-cd] Warning: no CD values computed; skipping update.")
            return
        for iso_name, cd in cd_table.items():
            parts = ", ".join(f"{k}={cd.as_dict()[k]:.4f}" for k in ("DJ", "DJK", "DK"))
            print(f"  [harmonic-cd]   {iso_name}: {parts} MHz (…)")
        iso_by_name = {str(iso.get("name", "iso")): iso for iso in self.spectral.isotopologues}
        for name, cd in cd_table.items():
            old_iso = iso_by_name.get(name)
            if old_iso is None:
                continue
            corr = old_iso.get("rovib_correction")
            if isinstance(corr, RovibCorrection):
                corr.pred_cd = cd
            else:
                old_iso["pred_cd"] = cd
        self.spectral.set_hessian_for_cd(hess_bohr)

    @staticmethod
    def _due_every(call_count: int, period: int) -> bool:
        """True on call 1, 1+period, 1+2*period, ...

        ``call_count % period == 1`` looks equivalent but silently fails for
        period 1: every integer mod 1 is 0, never 1, so a period of "every call"
        fired on no call at all.
        """
        n = max(1, int(period))
        return (int(call_count) - 1) % n == 0

    def _update_orca(self):
        """Decide whether to do a full Hessian recalculation or gradient-only update."""
        self._orca_call_count += 1
        if self.quantum is None or self._due_every(self._orca_call_count, self.hess_recalc_every):
            self._run_hessian()
            if self.use_orca_rovib and self._due_every(self._orca_call_count, self.rovib_recalc_every):
                self._run_rovib()
            if self._harmonic_from_hessian and self._due_every(self._orca_call_count, self.hess_recalc_every):
                self._apply_harmonic_alpha_corrections()
            if self._harmonic_cd_from_hessian and self._due_every(self._orca_call_count, self.hess_recalc_every):
                self._apply_harmonic_cd_corrections()
            elif self._fit_cd_constants and self._cd_weight > 0.0 and self.quantum is not None:
                self.spectral.set_hessian_for_cd(self.quantum._hessian_bohr)
        else:
            self._run_gradient()
            if self._fit_cd_constants and self._cd_weight > 0.0 and self.quantum is not None:
                self.spectral.set_hessian_for_cd(self.quantum._hessian_bohr)

    # â”€â”€ Pre-computed files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load_orca(self, engrad_path, hess_path):
        """
        Load pre-computed ORCA output files instead of running ORCA.
        Call this before run() when you have existing .engrad / .hess files.
        """
        eng = QuantumEngine(engrad_path, hess_path, self.elems)
        self.quantum = QuantumState(eng.energy, eng._gradient_bohr.copy(), eng._hessian_bohr.copy())
        self._orca_ref_coords = self.coords.copy()
        print(f"Loaded ORCA files.  Energy = {self.quantum.energy:.10f} Hartree")

    # â”€â”€ Drift check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _orca_drift(self):
        if self._orca_ref_coords is None:
            return np.inf
        return float(np.sqrt(np.mean((self.coords - self._orca_ref_coords) ** 2)))

    def _rigid_mode_projector(self, coords, masses):
        """
        Build Cartesian projector that removes rigid-body translation/rotation modes.
        """
        coords = np.asarray(coords, dtype=float)
        masses = np.asarray(masses, dtype=float)
        n = coords.shape[0]
        if masses.size != n:
            return np.eye(3 * n)

        msum = float(np.sum(masses))
        if msum <= 0.0:
            return np.eye(3 * n)
        com = (masses[:, None] * coords).sum(axis=0) / msum
        rel = coords - com
        sq_m = np.sqrt(np.maximum(masses, 1e-16))

        modes = []
        # Translations
        for axis in range(3):
            v = np.zeros((n, 3), dtype=float)
            v[:, axis] = sq_m
            modes.append(v.reshape(-1))
        # Rotations about x/y/z: omega x r
        axes = np.eye(3)
        for omega in axes:
            v = np.cross(np.tile(omega, (n, 1)), rel) * sq_m[:, None]
            modes.append(v.reshape(-1))

        if not modes:
            return np.eye(3 * n)
        M = np.column_stack(modes)  # (3N, <=6)
        Q, _ = np.linalg.qr(M)
        keep = []
        for j in range(Q.shape[1]):
            if np.linalg.norm(Q[:, j]) > 1e-12:
                keep.append(Q[:, j])
        if not keep:
            return np.eye(3 * n)
        Qk = np.column_stack(keep)
        return np.eye(3 * n) - Qk @ Qk.T

    def _project_quantum_terms(self, gradient, hessian):
        """
        Optionally remove rigid-body modes from quantum gradient/Hessian.
        """
        if (not self.project_rigid_modes) or (self._rigid_ref_masses is None):
            return gradient, hessian
        P = self._rigid_mode_projector(self.coords, self._rigid_ref_masses)
        g = P @ gradient
        H = P @ hessian @ P
        # Symmetrize after projection to reduce numerical asymmetry.
        H = 0.5 * (H + H.T)
        return g, H

    def _split_torsion_residuals(self, residual_mhz):
        """
        Partition residual vector into torsion-sensitive A vs non-A components.
        """
        a_vals = []
        bc_vals = []
        start = 0
        r = np.asarray(residual_mhz, dtype=float)
        for iso in self.spectral.isotopologues:
            idx = np.asarray(iso.get("component_indices", []), dtype=int)
            n = len(idx)
            ri = r[start:start + n]
            start += n
            is_torsion = bool(iso.get("torsion_sensitive", False))
            for k, comp in enumerate(idx):
                if is_torsion and int(comp) == 0:
                    a_vals.append(float(ri[k]))
                else:
                    bc_vals.append(float(ri[k]))
        return np.asarray(a_vals, dtype=float), np.asarray(bc_vals, dtype=float)

    def _problem_shape(self, isotopologues, n_params=None):
        n_atoms = len(self.elems)
        if n_params is None:
            n_params = 3 * n_atoms if self.coordinate_mode == "cartesian" else 3 * n_atoms
        return ProblemShape(
            n_atoms=n_atoms,
            n_params=int(n_params),
            n_jacobian_rows=count_spectral_rows(isotopologues),
            has_internal_priors=(self.internal_prior is not None and self.prior_weight > 0.0),
            prior_weight=float(self.prior_weight),
            coordinate_mode=self.coordinate_mode,
            use_dihedrals=bool(getattr(self, "_ic_use_dihedrals", False)),
        )

    def _apply_heuristic_optimizer_bases(self, isotopologues, n_params=None, label=""):
        shape = self._problem_shape(isotopologues, n_params=n_params)
        bases = infer_optimizer_bases(
            shape,
            trust_radius=self._base_trust_radius,
            null_trust_radius=self._base_null_trust_radius,
            lambda_damp=self._base_lambda_damp,
            sv_threshold=self._base_sv_threshold,
            alpha_quantum=self._base_alpha_quantum,
        )
        self.optimizer.trust_radius = bases["trust_radius"]
        self.optimizer.null_trust_radius = bases["null_trust_radius"]
        self.optimizer.lambda_damp = bases["lambda_damp"]
        self.optimizer.sv_threshold = bases["sv_threshold"]
        self.optimizer.alpha_quantum = bases["alpha_quantum"]
        self._base_trust_radius = bases["trust_radius"]
        self._base_null_trust_radius = bases["null_trust_radius"]
        self._base_lambda_damp = bases["lambda_damp"]
        self._base_sv_threshold = bases["sv_threshold"]
        self._base_alpha_quantum = bases["alpha_quantum"]
        suffix = f" {label}" if label else ""
        print(
            f"[autoconfig-bases]{suffix} atoms={shape.n_atoms} rows={shape.n_jacobian_rows} "
            f"n_params={shape.n_params} cf={shape.constraint_frac:.3f} -> "
            f"trust={bases['trust_radius']:.4e} sv={bases['sv_threshold']:.4e} "
            f"alpha={bases['alpha_quantum']:.4f} lambda={bases['lambda_damp']:.4e}"
        )
        if self.autoconfig is not None:
            self.autoconfig.reseed_bases(
                n_params=shape.n_params,
                base_trust_radius=bases["trust_radius"],
                base_null_trust_radius=bases["null_trust_radius"],
                base_lambda_damp=bases["lambda_damp"],
                base_sv_threshold=bases["sv_threshold"],
                base_alpha_quantum=bases["alpha_quantum"],
            )

    def _apply_autoconfig(self, rank, sv, residual_mhz, reject_streak):
        if self.autoconfig is None:
            return None
        sigma_vals = []
        for iso in self.spectral.isotopologues:
            sigma_vals.extend(np.asarray(iso.get("sigma_constants", []), dtype=float).tolist())
        sigma_scale = float(np.median(np.maximum(np.asarray(sigma_vals, dtype=float), 1e-12))) if sigma_vals else 1.0
        torsion_a_residuals, torsion_bc_residuals = self._split_torsion_residuals(residual_mhz)
        controls = self.autoconfig.suggest(
            rank=rank,
            singular_values=sv,
            residual_mhz=residual_mhz,
            sigma_scale_mhz=sigma_scale,
            torsion_a_residuals=torsion_a_residuals,
            torsion_bc_residuals=torsion_bc_residuals,
            reject_streak=reject_streak,
            has_internal_priors=(self.internal_prior is not None),
        )
        target_tr = max(1e-4, float(controls["trust_radius"]))
        target_ntr = max(1e-4, float(controls["null_trust_radius"]))
        target_lam = float(np.clip(controls["lambda_damp"], 1e-8, 1e2))
        # During rejection streaks, preserve trust-region shrink and damping growth
        # from adapt_lambda()/guardrails instead of immediately resetting.
        if reject_streak > 0:
            self.optimizer.trust_radius = min(self.optimizer.trust_radius, target_tr)
            self.optimizer.null_trust_radius = min(self.optimizer.null_trust_radius, target_ntr)
            self.optimizer.lambda_damp = max(self.optimizer.lambda_damp, target_lam)
        else:
            self.optimizer.trust_radius = target_tr
            self.optimizer.null_trust_radius = target_ntr
            self.optimizer.lambda_damp = target_lam
        self.prior_weight = max(0.0, float(controls["prior_weight"]))
        self.spectral_accept_relax = max(0.0, float(controls["spectral_accept_relax"]))
        self.spectral.set_adaptive_controls(
            sigma_floor_mhz=controls["sigma_floor_mhz"],
            max_weight=controls["max_spectral_weight"],
            torsion_a_weight=controls["torsion_a_weight"],
        )
        target_sv = float(controls.get("sv_threshold", self.optimizer.sv_threshold))
        target_alpha = float(controls.get("alpha_quantum", self.optimizer.alpha_quantum))
        if reject_streak > 0:
            self.optimizer.sv_threshold = max(self.optimizer.sv_threshold, target_sv)
            self.optimizer.alpha_quantum = min(self.optimizer.alpha_quantum, target_alpha)
        else:
            self.optimizer.sv_threshold = target_sv
            self.optimizer.alpha_quantum = target_alpha
        return controls

    # â”€â”€ Optimisation loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run(self):
        """
        Run the hybrid optimisation loop.

        Returns
        -------
        coords : (N, 3) ndarray   Final optimised coordinates in Angstroms.
        """
        header = (
            f"{'Iter':>5}  {'|dx| Ang':>12}  {'RMS_w':>12}  {'RMS MHz':>12}  "
            f"{'Rank':>6}  {'sig_kept':>12}  {'|dx_r|':>10}  {'|dx_n|':>10}  "
            f"{'|g_n|':>10}  {'alpha_q':>8}  {'|dE| Eh':>12}"
        )
        print("\n" + header)
        print("-" * len(header))
        if self.debug_rank_diagnostics:
            labels = ["A", "B", "C"]
            print("[rank-debug] Active spectral components in optimizer:")
            for i, iso in enumerate(self.spectral.isotopologues, start=1):
                idx = np.asarray(iso.get("component_indices", []), dtype=int)
                comps = [labels[c] if 0 <= int(c) < 3 else f"R{int(c)}" for c in idx]
                print(f"[rank-debug]   iso {i}: {comps}")

        # #5: Per-component observability â€” how many isotopologues constrain each constant
        _comp_labels = ["A", "B", "C"]
        _obs_count = {"A": 0, "B": 0, "C": 0}
        for _iso in self.spectral.isotopologues:
            for _cidx in _iso.get("component_indices", []):
                _c = int(_cidx)
                if 0 <= _c < 3:
                    _obs_count[_comp_labels[_c]] += 1
        _n_iso = len(self.spectral.isotopologues)
        print(f"\n[observability] {_n_iso} isotopologue(s); rotational constant constraints:")
        for _k in _comp_labels:
            _n = _obs_count[_k]
            _status = "constrained" if _n >= 1 else "UNCONSTRAINED"
            print(f"  {_k}: {_n} isotopologue(s)  [{_status}]")

        converged = False
        prev_energy = None
        prev_freq_rms = None
        _plateau_window = 10
        _plateau_count  = 0
        _hybrid_stall_count = 0
        _reject_streak = 0
        for it in range(self.max_iter):

            if self.spectral_only:
                n_dof = 3 * len(self.elems)
                g = np.zeros(n_dof)
                H = np.eye(n_dof)
            else:
                if self._orca_drift() > self.orca_update_thresh:
                    self._update_orca()
                g = self.quantum.gradient
                H = self.quantum.hessian
                g, H = self._project_quantum_terms(g, H)

            J, residual_w = self.spectral.stacked(self.coords)
            _, residual_mhz = self.spectral.stacked_unweighted(self.coords)
            _n_spectral_rows = int(J.shape[0])  # row count before any prior rows are appended
            prior_wrms_before = None
            if self.internal_prior is not None and self.coordinate_mode == "cartesian":
                Jp, rp = self.internal_prior.stacked(self.coords)
                wp = float(np.sqrt(self.prior_weight))
                J = np.vstack([J, wp * Jp])
                residual_w = np.concatenate([residual_w, wp * rp])
                prior_wrms_before = float(np.sqrt(np.mean(rp ** 2))) if rp.size else 0.0
            elif self.internal_prior is not None and self.coordinate_mode == "internal":
                # Native q-space priors are added in Phase 6; skip Cartesian prior in internal mode
                prior_wrms_before = self.internal_prior.diagnostics(self.coords).get("prior_wrms", 0.0)
            B, _ = wilson_B(self.coords, self.elems)

            # â”€â”€ Internal-coordinate mode: transform J and quantum terms to q-space â”€â”€
            _ic_coord_set = None
            _ic_Bplus = None
            _ic_g = g
            _ic_H = H
            _ic_prior_wrms = None
            if self.coordinate_mode == "internal":
                _ic_coord_set = InternalCoordinateSet(self.coords, self.elems, self._ic_use_dihedrals)
                _ic_B_active = _ic_coord_set.active_B_matrix(self.coords)
                _ic_Bplus = InternalCoordinateSet.damped_pseudoinverse(_ic_B_active, self._ic_damping)
                J = spectral_jacobian_q(J, _ic_Bplus)           # (m, n_active)
                _ic_g, _ic_H = quantum_terms_q(g, H, _ic_Bplus) # (n_active,), (n_active, n_active)
                # AutoConfig was initialised with 3N (Cartesian DOF); correct to n_q
                # on the first iteration so rank_frac reflects internal DOF, not Cartesian.
                if it == 0:
                    b_diag = _ic_coord_set.b_rank_diagnostics(self.coords)
                    kappa_b_str = f"{b_diag['kappa_B']:.2e}" if b_diag['kappa_B'] is not None else "n/a"
                    print(
                        f"[B-matrix] n_coords={b_diag['n_coords']}  rank={b_diag['rank']}"
                        f"  of {b_diag['n_dof']} DOF  Îº(B)={kappa_b_str}"
                    )
                    if b_diag["kappa_B"] is not None and b_diag["kappa_B"] > 1e4:
                        print(
                            f"  [warn] B-matrix ill-conditioned Îº(B)={b_diag['kappa_B']:.2e}; "
                            "some internal coordinates may be linearly dependent â€” "
                            "consider increasing ic_damping."
                        )
                    if self.autoconfig is not None:
                        n_active = len(_ic_coord_set.active_coords())
                        if n_active > 0:
                            self.autoconfig.n_params = n_active
                            if self.use_autoconfig_heuristic_bases:
                                self._apply_heuristic_optimizer_bases(
                                    self.spectral.isotopologues,
                                    n_params=n_active,
                                    label="internal",
                                )

                # Phase 6: native q-space internal priors
                if self._ic_prior_weight > 0.0:
                    if self._ic_initial_coords is None:
                        self._ic_initial_coords = self.coords.copy()
                    _J_prior, _r_prior, _ = build_internal_priors(
                        _ic_coord_set, self.coords,
                        sigma_bond=self._ic_prior_sigma_bond,
                        sigma_angle_deg=self._ic_prior_sigma_angle_deg,
                        sigma_dihedral_deg=self._ic_prior_sigma_dihedral_deg,
                        prior_values=_ic_coord_set.active_values(self._ic_initial_coords),
                        prior_mode=self._ic_prior_mode,
                        prior_specs=self._ic_user_priors,
                        elems=self.elems,
                        freeze_sigma_floor=self._ic_freeze_sigma_floor,
                        spectral_jacobian_q=J,
                        adaptive_config=self._ic_prior_adaptive,
                        sv_rel_threshold=self.optimizer.sv_threshold,
                    )
                    _wp = float(np.sqrt(self._ic_prior_weight))
                    J = np.vstack([J, _wp * _J_prior])
                    residual_w = np.concatenate([residual_w, _wp * _r_prior])
                    _ic_prior_wrms = float(np.sqrt(np.mean(_r_prior ** 2))) if _r_prior.size else 0.0

            wrms_before = float(np.sqrt(np.mean(residual_w ** 2)))
            mhz_rms_before = float(np.sqrt(np.mean(residual_mhz ** 2)))
            _svd_B = None if self.coordinate_mode == "internal" else B
            dp, rank, sv, alpha_q_eff, Vt = self.optimizer.step(J, residual_w, _ic_g, _ic_H, B=_svd_B)

            # â”€â”€ Back-transform and compute trial geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            _orig_coords = self.coords  # reference before update (never mutated here)
            if self.coordinate_mode == "internal":
                # dp is a q-space step; back-transform to Cartesian via micro-iterations
                _q_curr = _ic_coord_set.active_values(self.coords)
                _q_target = _q_curr + dp
                trial_coords, _bt_err = apply_internal_step(
                    self.coords, _q_target, _ic_coord_set,
                    max_micro=self._ic_micro_iter, damping=self._ic_damping,
                )
                # Project the back-transformed Cartesian geometry into the symmetric
                # subspace so spectral residuals are evaluated on a geometry that
                # respects symmetry constraints (mirrors Cartesian-mode behaviour).
                if self.symmetry is not None:
                    trial_coords = self.symmetry.symmetrize(trial_coords)
                    # Recompute the effective q-space step so model_delta uses the
                    # actual geometry change (symmetrization may have shortened dp).
                    _q_trial = _ic_coord_set.active_values(trial_coords)
                    dp = _q_trial - _q_curr
                dx = (trial_coords - _orig_coords).ravel()      # Cartesian displacement for diagnostics
            else:
                if self.symmetry is not None:
                    dp = self.symmetry.project_step(dp)
                dx = dp
                trial_coords = self.coords + dx.reshape(-1, 3)

            P_range = self.optimizer.range_projector(Vt, rank)
            P_null = self.optimizer.null_projector(Vt, rank)
            dx_range_norm = float(np.linalg.norm(P_range @ dp))
            dx_null_norm = float(np.linalg.norm(P_null @ dp))
            g_null_norm = None if self.spectral_only else float(np.linalg.norm(P_null @ _ic_g))
            autoconfig_controls = None
            if self.autoconfig is not None and ((it % self.autoconfig_update_every) == 0):
                autoconfig_controls = self._apply_autoconfig(rank, sv, residual_mhz, _reject_streak)
            geometry_valid, guardrail_stats = self._geometry_validity(trial_coords)
            _, residual_w_trial = self.spectral.stacked(trial_coords)
            prior_wrms_after = None
            if self.internal_prior is not None and self.coordinate_mode == "cartesian":
                Jp_trial, rp_trial = self.internal_prior.stacked(trial_coords)
                wp = float(np.sqrt(self.prior_weight))
                residual_w_trial = np.concatenate([residual_w_trial, wp * rp_trial])
                prior_wrms_after = float(np.sqrt(np.mean(rp_trial ** 2))) if rp_trial.size else 0.0
            elif self.internal_prior is not None and self.coordinate_mode == "internal":
                prior_wrms_after = self.internal_prior.diagnostics(trial_coords).get("prior_wrms", 0.0)
            _, residual_mhz_trial = self.spectral.stacked_unweighted(trial_coords)
            wrms_after = float(np.sqrt(np.mean(residual_w_trial ** 2)))
            mhz_rms_after = float(np.sqrt(np.mean(residual_mhz_trial ** 2)))

            # Dual acceptance gate:
            # 1) spectral improvement
            # 2) optional quantum-descent consistency check from local quadratic model
            spectral_accept = wrms_after <= wrms_before * (1.0 + self.spectral_accept_relax)
            model_delta = None
            quantum_accept = True
            quantum_gate_active = False
            quantum_descent_tol_eff = self.quantum_descent_tol
            if not self.spectral_only:
                # In internal mode g/_ic_H are in q-space and dp is the q-step.
                # In Cartesian mode they are the original Cartesian terms.
                model_delta = float(np.dot(_ic_g, dp) + 0.5 * dp @ (_ic_H @ dp))
                if self.enforce_quantum_descent:
                    quantum_gate_active = True
                    # Recovery fallback: in persistent rejection during explore stage,
                    # relax strict descent gating to allow exit from frozen loops.
                    if (
                        autoconfig_controls is not None
                        and autoconfig_controls.get("stage") == "explore"
                        and _reject_streak >= 3
                    ):
                        quantum_descent_tol_eff = max(self.quantum_descent_tol, 1e-4)
                    quantum_accept = model_delta <= quantum_descent_tol_eff
            geometry_accept = (not self.accept_requires_geometry_valid) or geometry_valid
            accepted = spectral_accept and quantum_accept and geometry_accept
            if accepted:
                self.coords = trial_coords
                if self.symmetry is not None:
                    self.coords = self.symmetry.symmetrize(self.coords)
                _reject_streak = 0
            else:
                _reject_streak += 1
                if self.enable_geometry_guardrails and not geometry_valid:
                    self.optimizer.lambda_damp = min(1e2, self.optimizer.lambda_damp * self.guardrail_lambda_boost)
                    self.optimizer.trust_radius = max(1e-4, self.optimizer.trust_radius * self.guardrail_trust_shrink)
                    self.optimizer.null_trust_radius = max(
                        1e-4, self.optimizer.null_trust_radius * self.guardrail_trust_shrink
                    )
            self.optimizer.adapt_lambda(accepted)
            step_norm = float(np.linalg.norm(dx))
            wrms = wrms_after if accepted else wrms_before
            freq_rms = mhz_rms_after if accepted else mhz_rms_before
            sv_kept   = float(sv[rank - 1]) if rank > 0 else 0.0
            kappa_J   = float(sv[0] / sv[rank - 1]) if rank > 1 and sv[rank - 1] > 0 else None
            if kappa_J is not None and kappa_J > 1e6:
                print(f"  [warn] Îº(J)={kappa_J:.2e} â€” Jacobian ill-conditioned; "
                      "consider adding isotopologues or checking input consistency.")
            if self.debug_rank_diagnostics:
                cutoff = float(self.optimizer.sv_threshold * sv[0]) if len(sv) and sv[0] > 0 else 0.0
                shown = np.asarray(sv[: self.debug_sv_count], dtype=float)
                sv_str = ", ".join(f"{x:.3e}" for x in shown)
                print(
                    f"[rank-debug] iter {it+1:03d} cutoff={cutoff:.3e} rank={rank} "
                    f"sv[:{len(shown)}]=[{sv_str}]"
                )
            energy = None if self.spectral_only else float(self.quantum.energy)
            delta_energy = None
            if energy is not None and prev_energy is not None:
                delta_energy = abs(energy - prev_energy)
            prev_energy = energy

            self.history.append(
                dict(
                    iteration=it + 1,
                    step_norm=step_norm,
                    wrms=wrms,
                    freq_rms=freq_rms,
                    rank=rank,
                    kappa_J=kappa_J,
                    spectral_rows=_n_spectral_rows,
                    lambda_damp=self.optimizer.lambda_damp,
                    accepted=accepted,
                    energy=energy,
                    delta_energy=delta_energy,
                    dx_range_norm=dx_range_norm,
                    dx_null_norm=dx_null_norm,
                    g_null_norm=g_null_norm,
                    alpha_q_eff=alpha_q_eff,
                    model_delta=model_delta,
                    backtransform_error=(_bt_err if self.coordinate_mode == "internal" else None),
                    internal_rank=(rank if self.coordinate_mode == "internal" else None),
                    internal_singular_values=(sv.tolist() if self.coordinate_mode == "internal" else None),
                    dq_range_norm=(dx_range_norm if self.coordinate_mode == "internal" else None),
                    dq_null_norm=(dx_null_norm if self.coordinate_mode == "internal" else None),
                    spectral_accept=spectral_accept,
                    quantum_accept=quantum_accept,
                    quantum_gate_active=quantum_gate_active,
                    quantum_descent_tol_eff=quantum_descent_tol_eff,
                    geometry_valid=geometry_valid,
                    guardrail_violations=guardrail_stats["violations"],
                    guardrail_bond_ratio_min=guardrail_stats["bond_ratio_min"],
                    guardrail_bond_ratio_max=guardrail_stats["bond_ratio_max"],
                    guardrail_clash_ratio_min=guardrail_stats["clash_ratio_min"],
                    prior_wrms=(prior_wrms_after if accepted else prior_wrms_before),
                    prior_wrms_by_conformer=(
                        self.internal_prior.diagnostics_for_conformers(
                            self.spectral.conformer_mixture.conformer_coords(self.coords),
                            self.spectral.conformer_mixture.weights(),
                        )["prior_wrms_by_conformer"]
                        if (self.internal_prior is not None and self.spectral.conformer_mixture is not None)
                        else None
                    ),
                    conformer_weights=(
                        self.spectral.conformer_diagnostics()["weights"]
                        if self.spectral.conformer_diagnostics() is not None else None
                    ),
                    mix_freq_rms=freq_rms if self.use_conformer_mixture else None,
                    autoconfig_stage=(autoconfig_controls["stage"] if autoconfig_controls is not None else None),
                    autoconfig_sigma_ratio=(
                        autoconfig_controls["sigma_ratio"] if autoconfig_controls is not None else None
                    ),
                    autoconfig_condition=(
                        autoconfig_controls["condition_est"] if autoconfig_controls is not None else None
                    ),
                    sv_threshold=self.optimizer.sv_threshold,
                    alpha_quantum=self.optimizer.alpha_quantum,
                    autoconfig_sv_gap=(
                        autoconfig_controls.get("sv_gap") if autoconfig_controls is not None else None
                    ),
                )
            )
            if accepted:
                status = "ok"
            elif self.enable_geometry_guardrails and not geometry_valid:
                status = "rej-geom"
            else:
                status = "rej"
            dE_str = f"{delta_energy:>12.3e}" if delta_energy is not None else f"{'n/a':>12}"
            gnull_str = f"{g_null_norm:>10.3e}" if g_null_norm is not None else f"{'n/a':>10}"
            kappa_str = f"{kappa_J:.2e}" if kappa_J is not None else "n/a"
            stage_suffix = ""
            if autoconfig_controls is not None:
                stage_suffix = f" [{autoconfig_controls['stage']}]"
            print(
                f"{it+1:>5}  {step_norm:>12.4e}  {wrms:>12.4f}  {freq_rms:>12.4f}  "
                f"{rank:>6d}  {sv_kept:>12.4e}  Îº(J)={kappa_str}  "
                f"{dx_range_norm:>10.3e}  {dx_null_norm:>10.3e}  "
                f"{gnull_str}  {alpha_q_eff:>8.3f}  {dE_str}  "
                f"lambda={self.optimizer.lambda_damp:.2e} {status}{stage_suffix}"
            )

            # If we keep rejecting, force a fresh Hessian sooner to recover local model quality.
            if _reject_streak >= 5:
                self._orca_call_count = 0

            # Adaptive Hessian schedule (efficiency): when spectral progress is
            # smooth, increase interval between full Hessians; when rejected or
            # progress stalls, tighten back toward frequent Hessian updates.
            if self.adaptive_hess_schedule and not self.spectral_only:
                spectral_improve = np.inf if prev_freq_rms is None else abs(prev_freq_rms - freq_rms)
                if accepted and spectral_improve > max(5e-3, 0.01 * self.conv_freq):
                    self.hess_recalc_every = min(self.hess_recalc_max, self.hess_recalc_every + 1)
                elif (not accepted) or spectral_improve < max(1e-3, 0.001 * self.conv_freq):
                    self.hess_recalc_every = max(self.hess_recalc_min, self.hess_recalc_every - 1)

            # Split convergence:
            # - range space: spectral residual + range-space step stabilization
            # - null space (hybrid mode): null-space step + null-space gradient + energy stabilization
            energy_ok = True
            if not self.spectral_only:
                energy_ok = delta_energy is not None and delta_energy < self.conv_energy
            spectral_ok = (
                freq_rms < self.conv_freq and
                dx_range_norm < self.conv_step_range
            )
            null_ok = True
            if not self.spectral_only:
                null_ok = (
                    dx_null_norm < self.conv_step_null and
                    g_null_norm is not None and g_null_norm < self.conv_grad_null
                )
                if self.null_convergence_requires_energy:
                    null_ok = null_ok and energy_ok
            if spectral_ok and null_ok:
                print(f"\nConverged in {it + 1} iterations.")
                converged = True
                break

            # Hybrid stall guard: if energy is converged and spectral improvement
            # is negligible for many iterations, stop instead of null-space marching.
            if not self.spectral_only:
                spectral_improve = np.inf if prev_freq_rms is None else abs(prev_freq_rms - freq_rms)
                if energy_ok and spectral_improve < max(1e-3, 0.001 * self.conv_freq):
                    _hybrid_stall_count += 1
                else:
                    _hybrid_stall_count = 0
                if _hybrid_stall_count >= 20:
                    print(
                        f"\nConverged to hybrid spectral/energy stall floor in {it + 1} iterations "
                        f"(freq RMS = {freq_rms:.4f} MHz, |dE| < {self.conv_energy:.1e} Eh)."
                    )
                    converged = True
                    break
            prev_freq_rms = freq_rms

            # Plateau convergence: freq_rms hasn't moved within conv_freq AND the geometry
            # has also settled (step_norm small).  Fall back after 40 flat iterations even
            # if geometry is still drifting (null-space oscillation without further progress).
            if it >= _plateau_window:
                recent = [h["freq_rms"] for h in self.history[-_plateau_window:]]
                freq_flat = max(recent) - min(recent) < self.conv_freq
                geom_flat = step_norm < 1e-3
                if freq_flat:
                    _plateau_count += 1
                else:
                    _plateau_count = 0
                if (freq_flat and geom_flat) or _plateau_count >= 40:
                    print(
                        f"\nConverged to rank-{rank} spectral floor in {it + 1} iterations "
                        f"(freq RMS = {freq_rms:.4f} MHz).\n"
                        f"To reduce residuals further, add isotopologues to increase the "
                        f"experimentally constrained rank."
                    )
                    converged = True
                    break

        if not converged:
            print(f"\nWarning: did not converge within {self.max_iter} iterations.")

        return self.coords.copy()

    # â”€â”€ Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_xyz(self, path):
        """Write final geometry to an XYZ file."""
        with open(path, "w") as f:
            f.write(f"{len(self.elems)}\n")
            f.write("R_se geometry from quantize hybrid optimizer\n")
            for elem, (x, y, z) in zip(self.elems, self.coords):
                f.write(f"{elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    def report(self):
        """
        Print a human-readable convergence and structure summary.
        Returns the final (3k,) residual vector.
        """
        J, residual = self.spectral.stacked(self.coords)
        U, s, Vt, rank = self.optimizer.decompose(J)
        n_null = 3 * len(self.elems) - rank
        cond = float(s[0] / s[-1]) if len(s) > 0 and s[-1] > 0 else np.inf
        JTJ = J.T @ J
        reg = 1e-8 * np.eye(JTJ.shape[0])
        cov = np.linalg.pinv(JTJ + reg)
        param_std = np.sqrt(np.maximum(np.diag(cov), 0.0))

        print("\n" + "=" * 52)
        print("  Final Structure Report")
        print("=" * 52)
        if self.symmetry is not None:
            for line in self.symmetry.summary().splitlines():
                print(f"  {line}")
            print()
        print(f"  Experimentally constrained directions : {rank}")
        print(f"  Theory-filled null-space directions   : {n_null}")
        print(f"  Largest singular value                : {s[0]:.6e}")
        if rank:
            print(f"  Smallest retained singular value      : {s[rank-1]:.6e}")
        print(f"  Jacobian condition estimate           : {cond:.6e}")
        print(f"  Mean parameter uncertainty (arb.)     : {np.mean(param_std):.6e}")
        print()
        print("  Rotational constant residuals (MHz)")
        print(f"  {'Iso':>4}  {'Const':>5}  {'Target':>12}  {'Calc':>12}  {'diff':>10}")
        print("  " + "-" * 50)
        labels = ["A", "B", "C"]
        for k, iso in enumerate(self.spectral.isotopologues):
            calc_all = self.spectral.rotational_constants(self.coords, iso["masses"])
            idx = iso["component_indices"]
            calc = calc_all[idx]
            target = self.spectral._be_target(iso)
            for i, comp in enumerate(idx):
                lbl = labels[int(comp)] if 0 <= int(comp) < len(labels) else f"R{int(comp)}"
                print(
                    f"  {k+1:>4}  {lbl:>5}  {target[i]:>12.4f}  "
                    f"{calc[i]:>12.4f}  {target[i]-calc[i]:>10.4f}"
                )

        print()
        print("  Bond lengths (Ang)")
        print(f"  {'Bond':>10}  {'Length':>10}")
        print("  " + "-" * 24)
        bonds = _detect_bonds(self.coords, self.elems)
        for i, j in bonds:
            d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
            print(f"  {self.elems[i]}{i+1}-{self.elems[j]}{j+1}:{'':>5}  {d:>10.6f}")

        angles = _detect_angles(bonds)
        if angles:
            print()
            print("  Bond angles (deg)")
            print(f"  {'Angle':>14}  {'Degrees':>10}")
            print("  " + "-" * 28)
            for i, j, k in angles:
                v1 = self.coords[i] - self.coords[j]
                v2 = self.coords[k] - self.coords[j]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                print(f"  {self.elems[i]}{i+1}-{self.elems[j]}{j+1}-{self.elems[k]}{k+1}:{'':>5}  {deg:>10.4f}")

        print("=" * 52)

        return residual

    def report_internal(self):
        """
        Extended report for internal-coordinate mode (Phase 7).

        Prints bond/angle table with uncertainties and identifiability labels
        derived from the internal-coordinate Jacobian at the final geometry.
        Falls back to report() if not in internal mode.
        """
        if self.coordinate_mode != "internal":
            return self.report()

        from backend.uncertainty import uncertainty_table, print_uncertainty_table
        from backend.internal.identifiability import identifiability_table, print_identifiability_table
        from backend.priors.prior_sensitivity import classify_prior_dominance, prior_sensitivity_analysis

        coord_set = InternalCoordinateSet(self.coords, self.elems, self._ic_use_dihedrals)
        B_active = coord_set.active_B_matrix(self.coords)
        Bplus = InternalCoordinateSet.damped_pseudoinverse(B_active, self._ic_damping)
        J_spectral, residual = self.spectral.stacked(self.coords)
        Jq = spectral_jacobian_q(J_spectral, Bplus)

        # Collect prior sigmas for uncertainty and identifiability
        if self._ic_prior_weight > 0.0:
            _, _, sigma_prior = build_internal_priors(
                coord_set, self.coords,
                sigma_bond=self._ic_prior_sigma_bond,
                sigma_angle_deg=self._ic_prior_sigma_angle_deg,
                sigma_dihedral_deg=self._ic_prior_sigma_dihedral_deg,
                prior_mode=self._ic_prior_mode,
                prior_specs=self._ic_user_priors,
                elems=self.elems,
                freeze_sigma_floor=self._ic_freeze_sigma_floor,
                spectral_jacobian_q=Jq,
                adaptive_config=self._ic_prior_adaptive,
                sv_rel_threshold=self.optimizer.sv_threshold,
            )
        else:
            sigma_prior = None

        # Print base report header and residuals
        self.report()

        # Print uncertainty table
        _J_prior_tmp, _r_prior_tmp, sigma_prior_for_labels, _meta_tmp = build_internal_priors(
            coord_set, self.coords,
            sigma_bond=self._ic_prior_sigma_bond,
            sigma_angle_deg=self._ic_prior_sigma_angle_deg,
            sigma_dihedral_deg=self._ic_prior_sigma_dihedral_deg,
            prior_mode=self._ic_prior_mode,
            prior_specs=self._ic_user_priors,
            elems=self.elems,
            freeze_sigma_floor=self._ic_freeze_sigma_floor,
            spectral_jacobian_q=Jq,
            adaptive_config=self._ic_prior_adaptive,
            sv_rel_threshold=self.optimizer.sv_threshold,
            return_metadata=True,
        )
        dominance = classify_prior_dominance(Jq, sigma_prior_for_labels, coord_set.active_names())
        sensitivity = prior_sensitivity_analysis(
            coord_set,
            self.coords,
            Jq,
            residual,
            sigma_bond=self._ic_prior_sigma_bond,
            sigma_angle_deg=self._ic_prior_sigma_angle_deg,
            sigma_dihedral_deg=self._ic_prior_sigma_dihedral_deg,
            prior_mode=self._ic_prior_mode,
            prior_specs=self._ic_user_priors,
            elems=self.elems,
            freeze_sigma_floor=self._ic_freeze_sigma_floor,
            spectral_jacobian_q=Jq,
            adaptive_config=self._ic_prior_adaptive,
            sv_rel_threshold=self.optimizer.sv_threshold,
            lambda_reg=self._ic_damping,
        )
        unc_rows = uncertainty_table(
            coord_set, self.coords, Jq,
            sigma_prior=sigma_prior,
            lambda_reg=self._ic_damping,
            residual_w=residual,
            dominance_labels=dominance,
            sensitivity_rows=sensitivity,
        )
        print("\n  Internal-coordinate uncertainties")
        print_uncertainty_table(unc_rows)

        # Print identifiability table
        id_rows, sv, rank = identifiability_table(coord_set, Jq, sigma_prior)
        print()
        print_identifiability_table(id_rows, sv, rank)
        print()

        return residual

    def log_probability(self, trial_coords, use_proxy=True):
        """
        Compute the log-posterior probability P(geometry | Spectra, Theory).

        Components:
          1. Spectral Likelihood (Gaussian noise model on B0)
          2. Theory Prior (QC energy surface)
          3. Structural Priors (Internal coordinate constraints)
        """
        # 1. Spectral Likelihood
        _, residual_w = self.spectral.stacked(trial_coords)
        chi_sq_spectral = np.sum(residual_w**2)

        # 2. Structural Priors
        chi_sq_prior = 0.0
        if self.internal_prior is not None and self.coordinate_mode == "cartesian":
            _, rp = self.internal_prior.stacked(trial_coords)
            chi_sq_prior = self.prior_weight * np.sum(rp**2)
        elif self.coordinate_mode == "internal" and self._ic_prior_weight > 0.0:
            # q-space priors are evaluated inside the sampler if using internal coords
            pass 

        energy_term = 0.0
        if not self.spectral_only and self.quantum is not None:
            if use_proxy:
                dx = (trial_coords - self._orca_ref_coords).ravel()
                g = self.quantum.gradient
                H = self.quantum.hessian
                energy_term = self.quantum.energy + np.dot(g, dx) + 0.5 * np.dot(dx, H @ dx)
            else:
                if self.quantum_backend == "psi4" and self._psi4_engine:
                    energy_term, _ = self._psi4_engine.run_gradient(trial_coords)
                else:
                    energy_term = self.quantum.energy

        log_p = -0.5 * (chi_sq_spectral + chi_sq_prior)
        return log_p - (self.optimizer.alpha_quantum * energy_term)

    def laplace_approximation(self):
        """Analytical estimate of the posterior covariance at the optimum."""
        J, _ = self.spectral.stacked(self.coords)
        H_total = J.T @ J
        if not self.spectral_only and self.quantum is not None:
            H_total += self.optimizer.alpha_quantum * self.quantum.hessian
            
        reg = 1e-8 * np.eye(H_total.shape[0])
        try:
            cov = np.linalg.inv(H_total + reg)
            std_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
            return cov, std_err
        except np.linalg.LinAlgError:
            return None, None
