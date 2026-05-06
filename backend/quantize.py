"""
Master optimizer — ties SpectralEngine, QuantumEngine, and SubspaceOptimizer together.

Workflow
--------
1. Provide initial geometry, element list, isotopologue data, and either
   the path to your ORCA binary or "orca" if it is on PATH.
2. MolecularOptimizer.run() iterates:
     a. Recompute spectral Jacobian J and residuals Δν at current geometry.
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
  Gradient           : Hartree / Å  (converted internally from Hartree / Bohr)
  Hessian            : Hartree / Å² (converted internally)
"""

import os
import shutil
import subprocess
import numpy as np

from backend.spectral import SpectralEngine, sanitize_isotopologues
from backend.internal_prior import InternalPriorEngine
from backend.rovib_corrections import (
    resolve_alpha_components,
    resolve_corrections,
    apply_corrections_to_isotopologues,
    validate_correction_quality,
    correction_summary,
)
from backend.correction_models import parse_correction_table, RovibCorrection, ParsedRovibResult
from backend.autoconfig import AutoConfigEngine
from backend.quantum import (
    QuantumEngine,
    parse_engrad,
    parse_orca_rovib,
    parse_orca_rovib_alpha,
    _detect_bonds,
    _detect_angles,
    wilson_B,
)
from backend.SVD import SubspaceOptimizer
from backend.internal_fit import (
    InternalCoordinateSet,
    apply_internal_step,
    spectral_jacobian_q,
    quantum_terms_q,
    build_internal_priors,
)


def _find_orca(executable):
    """
    Resolve the ORCA executable to an absolute path.
    Accepts a full path, a bare name ('orca'), or None (auto-detect).
    Search order: PATH (:func:`shutil.which`), explicit path if it exists as a file,
    then ``./orca`` or ``./orca.exe`` in the **current working directory**.
    Raises RuntimeError if not found.
    """
    if executable is None:
        executable = "orca"
    found = shutil.which(executable)
    if found:
        return found
    if os.path.isfile(executable):
        return os.path.abspath(executable)
    for name in ("orca", "orca.exe"):
        local = os.path.join(os.getcwd(), name)
        if os.path.isfile(local):
            return os.path.abspath(local)
    raise RuntimeError(
        f"ORCA executable '{executable}' not found on PATH, filesystem, or current directory.\n"
        "Install ORCA, add it to PATH, place an ``orca`` binary in the working directory, or "
        "set orca_executable to the full path, e.g. r'C:\\orca\\orca.exe'."
    )


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
        Convergence threshold on Cartesian step norm [Å].
    conv_freq : float
        Convergence threshold on rotational-constant RMS residual [MHz].
    conv_energy : float
        Convergence threshold on absolute energy change between iterations
        [Hartree]. Used for hybrid stall detection and optionally for null-space
        convergence when ``null_convergence_requires_energy`` is True.
    spectral_analytic_jacobian : bool
        If True (default), ``SpectralEngine`` uses an analytic ∂(A,B,C)/∂x with
        finite-difference fallback for degenerate principal moments.
    spectral_jacobian_degeneracy_tol : float
        Relative moment gap below which the Jacobian falls back to finite differences.
    null_convergence_requires_energy : bool
        If True, null-space convergence also requires ``|ΔE| < conv_energy``.
        Default False avoids stalling when energy differences fluctuate iteration-to-iteration.
    conv_step_range : float
        Convergence threshold on the range-space component of the Cartesian
        step norm [Å].
    conv_step_null : float
        Convergence threshold on the null-space component of the Cartesian
        step norm [Å].
    conv_grad_null : float
        Convergence threshold on the projected null-space gradient norm
        [Hartree/Å].
    orca_update_thresh : float
        Re-run ORCA when RMS geometry drift from last ORCA point exceeds
        this value [Å].  Default 0.005 Å.
    hess_recalc_every : int
        Recalculate the full Hessian every N ORCA calls.  Between recalculations
        only a cheap gradient (EnGrad) job is run.  Default 1 (always recalculate).
    sv_threshold : float
        Relative singular-value cutoff for range/null-space split.
    trust_radius : float
        Maximum step size [Å].
    lambda_damp : float
        Levenberg–Marquardt regularisation on the null-space Hessian.
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
        sv_min_abs=0.0,
        trust_radius=0.1,
        null_trust_radius=None,
        lambda_damp=1e-4,
        objective_mode="split",
        alpha_quantum=1.0,
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
        autoconfig_update_every=1,
        autoconfig_smoothing=0.4,
        correction_table=None,
        correction_mode="hybrid_auto",
        correction_sigma_vib_fraction=0.1,
        correction_elec=False,
        correction_sigma_elec_fraction=0.1,
        correction_bob_params=None,
        coordinate_mode="cartesian",
        ic_damping=1e-6,
        ic_use_dihedrals=False,
        ic_micro_iter=20,
        ic_prior_weight=1.0,
        ic_prior_sigma_bond=0.05,
        ic_prior_sigma_angle_deg=3.0,
        ic_prior_sigma_dihedral_deg=15.0,
    ):
        self.coords = np.asarray(coords, dtype=float).copy()
        self.elems = list(elems)
        if method_preset is not None:
            preset_method, preset_basis = self._method_preset(method_preset)
            orca_method = preset_method
            orca_basis = preset_basis

        # ── Rovibrational corrections (M1-M4) ────────────────────────────────
        self._corrected_targets = None
        _ctbl = parse_correction_table(correction_table)
        _apply_corrections = bool(_ctbl) or correction_mode != "hybrid_auto"
        if _apply_corrections or (correction_table is not None):
            _ctbl = parse_correction_table(correction_table)
        if _ctbl or correction_elec or correction_bob_params:
            _corrected_targets = resolve_corrections(
                isotopologues,
                correction_table=_ctbl,
                mode=str(correction_mode).strip().lower(),
                sigma_vib_fraction=float(correction_sigma_vib_fraction),
                elems=list(elems),
                correction_elec=bool(correction_elec),
                sigma_elec_fraction=float(correction_sigma_elec_fraction),
                correction_bob_params=correction_bob_params or None,
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
                    "is supplied — corrections are pre-applied and alpha_constants are zeroed."
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
            conformer_weight_mode=conformer_weight_mode,
            conformer_temperature_k=conformer_temperature_k,
            analytic_jacobian=bool(spectral_analytic_jacobian),
            jacobian_degeneracy_tol=float(spectral_jacobian_degeneracy_tol),
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
        self.optimizer = SubspaceOptimizer(
            sv_threshold,
            sv_min_abs,
            trust_radius,
            null_trust_radius,
            lambda_damp,
            objective_mode=objective_mode,
            alpha_quantum=alpha_quantum,
            dynamic_quantum_weight=dynamic_quantum_weight,
            quantum_weight_beta=quantum_weight_beta,
            quantum_weight_min=quantum_weight_min,
            quantum_weight_max=quantum_weight_max,
            use_internal_preconditioner=use_internal_preconditioner,
        )
        self._base_trust_radius = float(self.optimizer.trust_radius)
        self._base_null_trust_radius = float(self.optimizer.null_trust_radius)
        self._base_lambda_damp = float(self.optimizer.lambda_damp)

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
        self.use_autoconfig = bool(use_autoconfig)
        self.autoconfig_update_every = max(1, int(autoconfig_update_every))
        self.autoconfig = None
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
                smoothing=float(autoconfig_smoothing),
            )
        _valid_modes = ("cartesian", "internal")
        if str(coordinate_mode).strip().lower() not in _valid_modes:
            raise ValueError(f"coordinate_mode must be one of {_valid_modes}, got '{coordinate_mode}'")
        self.coordinate_mode = str(coordinate_mode).strip().lower()
        self._ic_damping = max(float(ic_damping), 1e-14)
        self._ic_use_dihedrals = bool(ic_use_dihedrals)
        self._ic_micro_iter = max(1, int(ic_micro_iter))
        self._ic_prior_weight = max(float(ic_prior_weight), 0.0)
        self._ic_prior_sigma_bond = float(ic_prior_sigma_bond)
        self._ic_prior_sigma_angle_deg = float(ic_prior_sigma_angle_deg)
        self._ic_prior_sigma_dihedral_deg = float(ic_prior_sigma_dihedral_deg)
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
        self._psi4_engine = None
        self._orca_ref_coords = None
        self._orca_call_count = 0
        self.history = []
        self._guardrail_bonds = _detect_bonds(self.coords, self.elems) if self.enable_geometry_guardrails else []

        if self.spectral_only:
            self._orca_exe = None
            print("Spectral-only mode: ORCA disabled. Null-space step will be zero.")
        else:
            if self.quantum_backend == "orca":
                # Resolve executable — deferred error if not found and load_orca used instead
                try:
                    self._orca_exe = _find_orca(orca_executable)
                    print(f"ORCA found: {self._orca_exe}")
                except RuntimeError as e:
                    self._orca_exe = None
                    print(f"Note: {e}\nCall load_orca() to use pre-computed files.")
            elif self.quantum_backend == "psi4":
                self._orca_exe = None
                try:
                    from backend.Psi4 import Psi4Engine  # pylint: disable=import-outside-toplevel
                    self._psi4_engine = Psi4Engine(
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
                    self._psi4_engine = None
                    print(f"Note: Could not initialize Psi4 backend: {e}")
            else:
                raise ValueError(
                    f"Unknown quantum_backend '{quantum_backend}'. Use 'orca' or 'psi4'."
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

    # ── File paths ────────────────────────────────────────────────────────────

    def _inp_path(self):
        return os.path.join(self.workdir, "quantize_orca.inp")

    def _engrad_path(self):
        return os.path.join(self.workdir, "quantize_orca.engrad")

    def _hess_path(self):
        return os.path.join(self.workdir, "quantize_orca.hess")

    def _out_path(self):
        return os.path.join(self.workdir, "quantize_orca.out")

    def _rovib_out_path(self):
        return os.path.join(self.workdir, "quantize_orca_rovib.out")

    def _err_path(self):
        return os.path.join(self.workdir, "quantize_orca.err")

    # ── ORCA input generation ─────────────────────────────────────────────────

    def _write_orca_input(self, job="hessian"):
        if job == "hessian":
            # Use Freq (not NumFreq) so ORCA 6 dispatches to orca_numfreq.exe.
            # NumFreq explicitly calls orca_autoci which was removed in ORCA 6.
            keyword = "Freq EnGrad"
        elif job == "gradient":
            keyword = "EnGrad"
        elif job == "rovib":
            # Use VPT2 workflow for rovibrational constants; AnFreq in ORCA 6
            # often prints only harmonic thermochemistry for some setups.
            keyword = "VPT2"
        else:
            raise ValueError(f"Unknown ORCA job type: {job}")
        method_line = f"{self.orca_method} {self.orca_basis}".strip()
        lines = [f"! {method_line} TightSCF {keyword}"]
        if job == "hessian":
            lines += ["%freq", "  Temp 298.15", "end"]
        elif job == "rovib":
            lines += [
                "%vpt2",
                "  VPT2 On",
                "  PrintLevel 2",
                "  MinimiseOrcaPrint True",
                "end",
                "%method",
                "  Z_Tol 1e-12",
                "end",
            ]
        # Force single-process; avoids MPI dependency for helper executables.
        lines += ["%pal", "  nprocs 1", "end"]
        lines += [f"* xyz {self.charge} {self.multiplicity}"]
        for elem, (x, y, z) in zip(self.elems, self.coords):
            lines.append(f"  {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
        lines.append("*\n")
        os.makedirs(self.workdir, exist_ok=True)
        with open(self._inp_path(), "w") as f:
            f.write("\n".join(lines))

    # ── ORCA execution ────────────────────────────────────────────────────────

    def _exec_orca(self):
        if self._orca_exe is None:
            raise RuntimeError(
                "ORCA executable not found.  Provide orca_executable= or call load_orca()."
            )
        # Ensure ORCA's own directory is on PATH so helper programs
        # (orca_autoci, orca_mp2, etc.) are found when called as subprocesses.
        env = os.environ.copy()
        orca_dir = os.path.dirname(os.path.abspath(self._orca_exe))
        if orca_dir not in env.get("PATH", ""):
            env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
        # Pass only the input *filename* (not an absolute path). ORCA's helper
        # executables often break on spaces in paths (e.g. ".../YC Hack/...") when
        # the full path is passed; cwd is already the job directory.
        workdir = os.path.abspath(self.workdir)
        inp_rel = os.path.basename(self._inp_path())
        result = subprocess.run(
            [self._orca_exe, inp_rel],
            capture_output=True,
            text=True,
            cwd=workdir,
            env=env,
        )
        os.makedirs(self.workdir, exist_ok=True)
        with open(self._out_path(), "w", encoding="utf-8", errors="ignore") as f:
            f.write(result.stdout or "")
        with open(self._err_path(), "w", encoding="utf-8", errors="ignore") as f:
            f.write(result.stderr or "")
        if result.returncode != 0:
            raise RuntimeError(
                f"ORCA terminated with a non-zero exit code.\n"
                f"--- ORCA stderr (last 3000 chars) ---\n{result.stderr[-3000:]}"
            )

    def _require_orca_artefacts(self, need=("engrad", "hess")):
        """
        ORCA sometimes exits 0 without writing expected files (license limits, helper crash).
        Fail fast with directory listing and tail of quantize_orca.out for debugging.
        """
        labels_paths = []
        if "engrad" in need:
            labels_paths.append(("engrad", self._engrad_path()))
        if "hess" in need:
            labels_paths.append(("hess", self._hess_path()))
        missing = [(lab, p) for lab, p in labels_paths if not os.path.isfile(p)]
        if not missing:
            return
        out_tail = ""
        outp = self._out_path()
        try:
            if os.path.isfile(outp):
                with open(outp, encoding="utf-8", errors="ignore") as f:
                    out_tail = f.read()[-6000:]
        except OSError:
            out_tail = "(could not read quantize_orca.out)"
        try:
            names = sorted(os.listdir(self.workdir))
            listing = "\n".join(names) if names else "(empty)"
        except OSError as e:
            listing = f"(could not list: {e})"
        miss_str = "\n".join(f"  missing {lab}: {p}" for lab, p in missing)
        raise RuntimeError(
            "ORCA ran but expected output files were not found.\n"
            f"{miss_str}\n"
            f"workdir: {self.workdir}\n"
            "Files present:\n"
            f"{listing}\n\n"
            "Common causes: (1) spaces in the full path to the job directory break some ORCA "
            "helpers — clone the repo to a path without spaces, or use this codebase version "
            "that invokes ORCA with a relative input name; (2) academic license allows only "
            "one ORCA job — use max_workers=1; (3) see quantize_orca.out below.\n"
            f"--- tail of quantize_orca.out ---\n{out_tail}"
        )

    def _run_hessian(self):
        """Full Freq job: refreshes both gradient and Hessian."""
        if self.quantum_backend == "psi4":
            if self._psi4_engine is None:
                raise RuntimeError("Psi4 backend not initialized.")
            print("  [Psi4] Running Hessian calculation (gradient + Hessian)...")
            self.quantum = self._psi4_engine.run_hessian(self.coords)
            self._orca_ref_coords = self.coords.copy()
            print(f"  [Psi4] Done.  Energy = {self.quantum.energy:.10f} Hartree")
            return
        print("  [ORCA] Running frequency calculation (gradient + Hessian)...")
        self._write_orca_input(job="hessian")
        self._exec_orca()
        self._require_orca_artefacts(need=("engrad", "hess"))
        self.quantum = QuantumEngine(self._engrad_path(), self._hess_path(), self.elems)
        self._orca_ref_coords = self.coords.copy()
        print(f"  [ORCA] Done.  Energy = {self.quantum.energy:.10f} Hartree")

    def _run_gradient(self):
        """Cheap EnGrad job: refreshes gradient only, keeps existing Hessian."""
        if self.quantum_backend == "psi4":
            if self._psi4_engine is None:
                raise RuntimeError("Psi4 backend not initialized.")
            print("  [Psi4] Running gradient update...")
            energy, grad = self._psi4_engine.run_gradient(self.coords)
            self.quantum.energy = energy
            self.quantum._gradient_bohr = grad
            self._orca_ref_coords = self.coords.copy()
            print(f"  [Psi4] Done.  Energy = {energy:.10f} Hartree")
            return
        print("  [ORCA] Running gradient update (EnGrad)...")
        self._write_orca_input(job="gradient")
        self._exec_orca()
        self._require_orca_artefacts(need=("engrad",))
        energy, grad = parse_engrad(self._engrad_path())
        self.quantum.energy = energy
        self.quantum._gradient_bohr = grad
        self._orca_ref_coords = self.coords.copy()
        print(f"  [ORCA] Done.  Energy = {energy:.10f} Hartree")

    # ── Isotopologue-specific VPT2 helpers ────────────────────────────────────

    def _iso_rovib_inp_path(self, label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(label))
        return os.path.join(self.workdir, f"quantize_orca_rovib_{safe}.inp")

    def _iso_rovib_out_path(self, label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(label))
        return os.path.join(self.workdir, f"quantize_orca_rovib_{safe}.out")

    def _write_orca_rovib_input_for_iso(self, iso_masses, label):
        """Write a VPT2 input that overrides per-atom masses for isotope substitution."""
        method_line = f"{self.orca_method} {self.orca_basis}".strip()
        lines = [f"! {method_line} TightSCF VPT2"]
        lines += [
            "%vpt2",
            "  VPT2 On",
            "  PrintLevel 2",
            "  MinimiseOrcaPrint True",
            "end",
            "%method",
            "  Z_Tol 1e-12",
            "end",
            "%pal",
            "  nprocs 1",
            "end",
        ]
        masses = np.asarray(iso_masses, dtype=float).ravel()
        lines.append(f"* xyz {self.charge} {self.multiplicity}")
        for elem, (x, y, z) in zip(self.elems, self.coords):
            lines.append(f"  {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
        lines.append("*")
        # Apply explicit isotope masses through %coords M block. Use the
        # current geometry too so the block is self-contained for some ORCA
        # versions that re-read coords from %coords.
        lines.append("%coords")
        lines.append(f"  CTyp xyz")
        lines.append(f"  Charge {self.charge}")
        lines.append(f"  Mult {self.multiplicity}")
        lines.append("  Coords")
        for (elem, (x, y, z)), m in zip(zip(self.elems, self.coords), masses):
            lines.append(
                f"    {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}  M = {float(m):.8f}"
            )
        lines.append("  end")
        lines.append("end\n")
        os.makedirs(self.workdir, exist_ok=True)
        with open(self._iso_rovib_inp_path(label), "w") as f:
            f.write("\n".join(lines))

    def _exec_orca_named(self, inp_path, out_path):
        """Run ORCA on a specific input and capture its output to ``out_path``."""
        if self._orca_exe is None:
            raise RuntimeError(
                "ORCA executable not found.  Provide orca_executable= or call load_orca()."
            )
        env = os.environ.copy()
        orca_dir = os.path.dirname(os.path.abspath(self._orca_exe))
        if orca_dir not in env.get("PATH", ""):
            env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
        workdir = os.path.abspath(self.workdir)
        inp_rel = os.path.basename(inp_path)
        result = subprocess.run(
            [self._orca_exe, inp_rel],
            capture_output=True,
            text=True,
            cwd=workdir,
            env=env,
        )
        os.makedirs(self.workdir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(result.stdout or "")
        if result.returncode != 0:
            raise RuntimeError(
                f"ORCA terminated with non-zero exit code while running rovib job '{inp_rel}'.\n"
                f"--- ORCA stderr (last 3000 chars) ---\n{(result.stderr or '')[-3000:]}"
            )

    def _run_rovib_isotopologue_specific(self):
        """Run one ORCA VPT2 job per isotopologue, with mass overrides."""
        cache_dir = self.workdir
        for iso in self.spectral.isotopologues:
            label = str(iso.get("name", "iso"))
            masses = np.asarray(iso["masses"], dtype=float)
            cache_key = make_rovib_cache_key(
                self.coords,
                masses,
                self.orca_method,
                self.orca_basis,
                "orca",
                self.rovib_source_mode,
            )
            cached = load_cached_correction(cache_dir, cache_key, label)
            parsed_alpha = None
            warnings_list: list[str] = []
            if cached is not None:
                parsed_alpha = cached.alpha_vector()
                warnings_list = list(cached.warnings or [])
                print(f"  [ORCA] Cache hit for isotopologue '{label}'.")
            else:
                print(f"  [ORCA] Running VPT2 for isotopologue '{label}' (mass-overridden)...")
                self._write_orca_rovib_input_for_iso(masses, label)
                inp = self._iso_rovib_inp_path(label)
                outp = self._iso_rovib_out_path(label)
                try:
                    self._exec_orca_named(inp, outp)
                    parsed = parse_orca_rovib(outp)
                    parsed_alpha = parsed.alpha_abc
                    warnings_list = list(parsed.warnings)
                    if parsed.parse_status == "parse_failed":
                        print(
                            f"  [ORCA] Warning: VPT2 parse failed for '{label}'; "
                            "falling back to existing alpha."
                        )
                except Exception as exc:  # noqa: BLE001
                    warnings_list.append(f"VPT2 run failed: {exc}")
                    print(f"  [ORCA] Warning: VPT2 run failed for '{label}': {exc}")

            idx = np.asarray(iso["component_indices"], dtype=int)
            user_tbl = iso.get("rovib_table", None)
            try:
                resolved, correction = resolve_alpha_components(
                    existing_alpha_by_component=iso.get(
                        "alpha_constants", np.zeros(len(idx), dtype=float)
                    ),
                    component_indices=idx,
                    parsed_alpha_abc=parsed_alpha,
                    user_alpha_abc=user_tbl,
                    mode=self.rovib_source_mode,
                    isotopologue_name=label,
                    method=self.orca_method,
                    basis=self.orca_basis,
                    backend="orca",
                )
            except ValueError as e:
                print(f"  [ORCA] Strict mode rejected isotopologue '{label}': {e}")
                continue

            correction.warnings = list(correction.warnings) + warnings_list
            correction.geometry_hash = cache_key
            iso["alpha_constants"] = resolved
            iso["rovib_correction"] = correction
            self._refresh_iso_delta_total(iso, correction)
            if cached is None and parsed_alpha is not None and np.isfinite(parsed_alpha).any():
                try:
                    save_cached_correction(cache_dir, cache_key, label, correction)
                except OSError:
                    pass

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
        """
        Optional ORCA anharmonic run to extract alpha(A/B/C) and populate
        isotopologue alpha_constants by selected rotational components.
        """
        if self.quantum_backend != "orca":
            return
        if self.rovib_source_mode == "orca_vpt2_isotopologue_specific":
            self._run_rovib_isotopologue_specific()
            return

        print("  [ORCA] Running rovibrational correction calculation (VPT2)...")
        self._write_orca_input(job="rovib")
        self._exec_orca()
        # Preserve the rovib output for debugging; later jobs overwrite quantize_orca.out.
        try:
            shutil.copyfile(self._out_path(), self._rovib_out_path())
        except OSError:
            pass
        parsed = parse_orca_rovib(self._out_path())
        alpha_abc = parsed.alpha_abc
        warnings_list = list(parsed.warnings)
        if parsed.parse_status == "parse_failed":
            vpt2_path = os.path.join(self.workdir, "quantize_orca.vpt2")
            if os.path.isfile(vpt2_path):
                parsed_fallback = parse_orca_rovib(vpt2_path)
                alpha_abc = parsed_fallback.alpha_abc
                warnings_list += list(parsed_fallback.warnings)
        if not np.isfinite(alpha_abc).any():
            print(
                "  [ORCA] Warning: could not parse alpha constants from rovibrational output; "
                "keeping existing alpha_constants.\n"
                f"  [ORCA] Check files: {self._rovib_out_path()} and "
                f"{os.path.join(self.workdir, 'quantize_orca.vpt2')}"
            )
            return

        # Identify the parent isotopologue (assume index 0) so we can warn when
        # a parent-only correction is broadcast onto isotopically-substituted ones.
        if self.spectral.isotopologues:
            parent_masses = np.asarray(
                self.spectral.isotopologues[0]["masses"], dtype=float
            )
        else:
            parent_masses = None

        for iso_idx, iso in enumerate(self.spectral.isotopologues):
            label = str(iso.get("name", f"iso_{iso_idx + 1}"))
            idx = np.asarray(iso["component_indices"], dtype=int)
            user_tbl = iso.get("rovib_table", None)
            try:
                resolved, correction = resolve_alpha_components(
                    existing_alpha_by_component=iso.get("alpha_constants", np.zeros(len(idx), dtype=float)),
                    component_indices=idx,
                    parsed_alpha_abc=alpha_abc,
                    user_alpha_abc=user_tbl,
                    mode=self.rovib_source_mode,
                    isotopologue_name=label,
                    method=self.orca_method,
                    basis=self.orca_basis,
                    backend="orca",
                )
            except ValueError as e:
                print(f"  [ORCA] Strict mode rejected isotopologue '{label}': {e}")
                continue
            iso_warnings = list(warnings_list)
            iso_masses = np.asarray(iso["masses"], dtype=float)
            if (
                parent_masses is not None
                and iso_idx > 0
                and (
                    iso_masses.shape != parent_masses.shape
                    or not np.allclose(iso_masses, parent_masses)
                )
                and self.rovib_source_mode in ("hybrid_auto", "orca_only")
            ):
                iso_warnings.append(
                    "parent-only VPT2 correction applied to non-parent isotopologue"
                )
            correction.warnings = list(correction.warnings) + iso_warnings
            iso["alpha_constants"] = resolved
            iso["rovib_correction"] = correction
            self._refresh_iso_delta_total(iso, correction)
        print(f"  [ORCA] Updated isotopologue alpha_constants using mode={self.rovib_source_mode}.")

    def _update_orca(self):
        """Decide whether to do a full Hessian recalculation or gradient-only update."""
        self._orca_call_count += 1
        if self.quantum is None or self._orca_call_count % self.hess_recalc_every == 1:
            self._run_hessian()
            if self.use_orca_rovib and self._orca_call_count % self.rovib_recalc_every == 1:
                self._run_rovib()
        else:
            self._run_gradient()

    # ── Pre-computed files ────────────────────────────────────────────────────

    def load_orca(self, engrad_path, hess_path):
        """
        Load pre-computed ORCA output files instead of running ORCA.
        Call this before run() when you have existing .engrad / .hess files.
        """
        self.quantum = QuantumEngine(engrad_path, hess_path, self.elems)
        self._orca_ref_coords = self.coords.copy()
        print(f"Loaded ORCA files.  Energy = {self.quantum.energy:.10f} Hartree")

    # ── Drift check ───────────────────────────────────────────────────────────

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
        return controls

    # ── Optimisation loop ─────────────────────────────────────────────────────

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

            # ── Internal-coordinate mode: transform J and quantum terms to q-space ──
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
                    )
                    _wp = float(np.sqrt(self._ic_prior_weight))
                    J = np.vstack([J, _wp * _J_prior])
                    residual_w = np.concatenate([residual_w, _wp * _r_prior])
                    _ic_prior_wrms = float(np.sqrt(np.mean(_r_prior ** 2))) if _r_prior.size else 0.0

            wrms_before = float(np.sqrt(np.mean(residual_w ** 2)))
            mhz_rms_before = float(np.sqrt(np.mean(residual_mhz ** 2)))
            _svd_B = None if self.coordinate_mode == "internal" else B
            dp, rank, sv, alpha_q_eff, Vt = self.optimizer.step(J, residual_w, _ic_g, _ic_H, B=_svd_B)

            # ── Back-transform and compute trial geometry ─────────────────────
            _orig_coords = self.coords  # reference before update (never mutated here)
            if self.coordinate_mode == "internal":
                # dp is a q-space step; back-transform to Cartesian via micro-iterations
                _q_curr = _ic_coord_set.active_values(self.coords)
                _q_target = _q_curr + dp
                trial_coords, _bt_err = apply_internal_step(
                    self.coords, _q_target, _ic_coord_set,
                    max_micro=self._ic_micro_iter, damping=self._ic_damping,
                )
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
            stage_suffix = ""
            if autoconfig_controls is not None:
                stage_suffix = f" [{autoconfig_controls['stage']}]"
            print(
                f"{it+1:>5}  {step_norm:>12.4e}  {wrms:>12.4f}  {freq_rms:>12.4f}  "
                f"{rank:>6d}  {sv_kept:>12.4e}  {dx_range_norm:>10.3e}  {dx_null_norm:>10.3e}  "
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

    # ── Output ────────────────────────────────────────────────────────────────

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
        from backend.identifiability import identifiability_table, print_identifiability_table

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
            )
        else:
            sigma_prior = None

        # Print base report header and residuals
        self.report()

        # Print uncertainty table
        unc_rows = uncertainty_table(
            coord_set, self.coords, Jq,
            sigma_prior=sigma_prior,
            lambda_reg=self._ic_damping,
        )
        print("\n  Internal-coordinate uncertainties")
        print_uncertainty_table(unc_rows)

        # Print identifiability table
        id_rows, sv, rank = identifiability_table(coord_set, Jq, sigma_prior)
        print()
        print_identifiability_table(id_rows, sv, rank)
        print()

        return residual
