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

import numpy as np

from backend.covariance_checks import std_err_from_covariance

from backend.spectral.spectral import (
    DEFAULT_SIGMA_FLOOR_REL,
    SpectralEngine,
    sanitize_isotopologues,
)
from backend.internal.internal_prior import InternalPriorEngine
from backend.spectral.rovib_corrections import (
    resolve_corrections,
    apply_corrections_to_isotopologues,
    validate_correction_quality,
    correction_summary,
)
from backend.spectral.correction_models import parse_correction_table, RovibCorrection
from backend.spectral.spectral import default_sigma_constants


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
from backend.spectral.prior_widths import (  # noqa: E402
    level_for_backend,
    prior_sigma_for_level,
)

#: Distinguishes "the caller said nothing" from "the caller said None".
#: They are not the same: None is a documented request for the legacy
#: alpha_quantum heuristic in place of the calibrated weight (SubspaceOptimizer
#: branches on it in _effective_quantum_weight), and the YAML runner passes it
#: deliberately. Only silence means "calibrate from the
#: level of theory". Collapsing the two switched every None-passing caller onto
#: the calibrated-weight objective, which the end-to-end water test caught.
_PRIOR_SIGMA_UNSET = object()

#: How negative a propagated variance may be, relative to the size of the terms
#: summed to produce it, before it counts as a failed propagation rather than
#: round-off. Round-off on a well-conditioned covariance lands near 1e-16 of the
#: term scale; a collapsed one lands near -1. Anything in between is already
#: untrustworthy, so the threshold is deliberately far below the round-off it
#: has to tolerate and far above the failure it has to catch.
_VAR_NEGATIVE_REL_TOL = 1e-9
_TINY = 1e-300

#: How indefinite the posterior covariance may be before its propagated widths
#: are refused outright, as min(eig)/max(eig). A symmetric covariance should be
#: positive semi-definite; a small negative eigenvalue is round-off, a large one
#: means the normal matrix was too ill-conditioned to invert meaningfully and
#: every width taken from it is noise -- whatever sign that noise happens to
#: carry on a given BLAS.
#:
#: Measured on water, min(eig)/max(eig), across two interpreters, when the
#: Laplace inverted the full 3N matrix:
#:
#:     floored, sigma 0.02      -2.2e-16  (3.11)   -1.4e-15  (3.12)
#:     floored, sigma 2000      -6.6e-16  (3.11)   -2.9e-16  (3.12)
#:     unfloored, sigma 2.0     -6.7e-10  (3.11)   -7.5e-10  (3.12)   usable
#:     unfloored, sigma 0.02    -1.2e+02  (3.11)   rejected  (3.12)   collapsed
#:
#: so 1e-6 sits in an empty four-order gap above anything usable and far below
#: the collapse. It has to be a RATIO: the same absolute eigenvalue means
#: round-off on one matrix and catastrophe on another.
#:
#: The collapse rows came from that inverse, not from the data. Inverted over
#: the physical directions, the same cases stay between -1.3e-16 and -2e-18
#: down to 0.02 MHz, so this is now a round-off backstop for covariances that
#: arrive from elsewhere rather than something the Laplace can trip.
_COV_INDEFINITE_REL_TOL = 1e-6

#: Fallback when the level of theory has no calibrated width. See
#: spectral.prior_widths for the calibrated table and why an unmeasured level
#: falls back to the loosest value rather than an interpolated one.
DEFAULT_QUANTUM_PRIOR_SIGMA_ANG = 0.020

#: Guards on chi-square rescaling. Below this many degrees of freedom the ratio
#: is dominated by how much freedom the model has rather than by whether the
#: sigmas are right, and the adjustment is skipped. The factor is clamped
#: because an unbounded one drives sigma to zero on a near-perfect fit and every
#: quantity that divides by sigma diverges with it.
CHI2_RESCALE_MIN_DOF = 1
CHI2_RESCALE_MAX_FACTOR = 4.0
from backend.autoconfig import AutoConfigEngine
from backend.autoconfig_bases import ProblemShape, count_spectral_rows, infer_optimizer_bases
from backend.quantum import (
    QuantumEngine,
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


def _kabsch_align(mobile, reference):
    """``mobile`` rotated and translated onto ``reference``, least squares.

    Both are (N, 3) in Angstrom. Unweighted by mass on purpose: this aligns a
    prior target to a trial geometry so their difference is purely internal,
    and mass weighting would let a heavy atom absorb the fit.
    """
    a = np.asarray(mobile, dtype=float)
    b = np.asarray(reference, dtype=float)
    rot = _kabsch_rotation(a, b)
    return (rot @ (a - a.mean(axis=0)).T).T + b.mean(axis=0)


def _kabsch_rotation(mobile, reference):
    """The proper rotation (3x3, acting on column vectors) that _kabsch_align
    applies to ``mobile`` about its centroid."""
    a = np.asarray(mobile, dtype=float)
    b = np.asarray(reference, dtype=float)
    p, q = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _rotate_cartesian_hessian(hessian, rot):
    """A Cartesian Hessian (3N x 3N, atom-major) of a geometry rotated by
    ``rot``: each 3x3 block becomes rot H_ij rot^T. Exact -- the energy does
    not change under a rigid rotation."""
    h = np.asarray(hessian, dtype=float)
    k = np.kron(np.eye(h.shape[0] // 3), np.asarray(rot, dtype=float))
    return k @ h @ k.T



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
        sv_min_abs=None,
        trust_radius=0.1,
        null_trust_radius=None,
        lambda_damp=1e-4,
        objective_mode="joint",
        prior_target_coords=None,
        isotropic_prior=False,
        alpha_quantum=1.0,
        # Unset means "calibrate from the level of theory" (see
        # spectral.prior_widths); an explicit None still means the legacy
        # heuristic. It previously defaulted to the flat constant, which made
        # "the caller said nothing" indistinguishable from "the caller asked
        # for 0.020", so the calibration below could never fire.
        quantum_prior_sigma_ang=_PRIOR_SIGMA_UNSET,
        prior_class_sigma=None,
        robust_loss="none",
        robust_param=1.0,
        chi2_rescale=False,
        chi2_rescale_threshold=4.0,
        chi2_rescale_max_passes=1,
        sigma_floor_mhz=0.0,
        sigma_floor_rel=DEFAULT_SIGMA_FLOOR_REL,
        sigma_cap_mhz=None,
        # OPT-IN, not the default, and the reason is measured rather than
        # cautious. Deriving the model gap and propagating it makes the
        # reported interval ANTI-MONOTONIC in the quoted sigma whenever
        # sigma_floor_rel is off: on water the bond width falls from 0.0455
        # to 0.0180 Angstrom as the quoted sigma loosens by 1000x, because
        # tighter data is weighted more heavily and a coherent model shift
        # then moves the optimum further. That is defensible as physics and
        # indefensible as a published uncertainty -- admitting more error
        # must not narrow an interval.
        #
        # With the default floor on it happens to come out monotonic, because
        # the floor pins the weighting sigma. A default that is sound only
        # because another default masks it is not a default.
        #
        # The prerequisite is the other half of the note below: let the
        # systematic weight the fit as well as the report, so the two sigmas
        # cannot disagree by 1e5 in the first place. Until then this is
        # available and verified, and off.
        auto_systematic_sigma=False,
        systematic_in_weighting=False,
        # How the model error is shared between the stacked constants, for the
        # reported covariance (the fit's weights stay diagonal). The
        # B0-versus-Be gap of one constant type moves every isotopologue
        # together -- one potential surface, one force-field error, smooth mass
        # scaling -- measured correlation 0.94-1.00 across isotopologues on
        # water, ozone, formyl fluoride and vinyl fluoride, from level-of-theory
        # differences and, for water and ozone, from the experimental gap.
        # Across A, B and C it has no consistent sign (formyl fluoride A-B
        # -0.30 against one reference level, +0.73 against another), so the
        # components are independent unless a 3x3 correlation matrix says
        # otherwise (B-C runs +0.98 to +1.00 in the near-prolate tops). 0.99,
        # not 1: the isotopologue-specific remainder is what sets the width in
        # an ill-conditioned fit (formyl fluoride spectral-only rms z 0.74 at
        # 0.99, 4.55 at 1). Against ozone's real gap (r_e to 0.15 mA) the
        # independent model leaves the shift at z 2.9 / 3.5 on bond / angle,
        # this one at 1.3 / 1.6 (test_systematic_correlation). 0 restores the
        # independent model.
        systematic_isotopologue_correlation=0.99,
        systematic_component_correlation=None,
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
        symmetry="auto",
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        project_rigid_modes=True,
        mass_dependent_model=None,
        mass_dependent_frac_sigma=0.01,
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
        correction_method=None,
        correction_basis=None,
        nonconvergent_policy="warn",
        harmonic_scheme="watson",
    #: normal_mode, not cartesian, and the difference is not small. Both compute the
    #: same cubic force field; they differ in how it is sampled and therefore in how
    #: well it is known. The Cartesian tensor takes raw coordinate displacements,
    #: which mix in rotation and use one step for every mode; the normal-mode route
    #: displaces along each mode with a ZPE-scaled step, which is what keeps the cubic
    #: term meaningful on a noisy surface. The code has said "preferred" in
    #: compute_harmonic_alpha's docstring since it was written.
    #:
    #: It also earns a different sigma. The normal-mode path gets a calibrated
    #: nm_sigma_fraction of the cancellation sum plus its own measured noise; the
    #: Cartesian path is floored at the FULL magnitude of the anharmonic term.
    #: Measured on water:
    #:
    #:     cartesian     alpha_A -8411   sigma 35720 / 7965 / 7703
    #:     normal_mode   alpha_A -8344   sigma 11352 / 2792 / 1541
    #:
    #: The same alpha to 0.8%, carried with three to five times the uncertainty. The
    #: fit weights every target by 1/sigma^2, so that is nine to twenty-five times
    #: less weight on the corrections -- the default threw away most of what the
    #: correction chain had computed.
    #:
    #: Every benchmark in this repository passed cubic_scheme="normal_mode"
    #: explicitly, so none of them ever exercised the default, and the shipped engine
    #: was not the one the results came from. It is also the cheaper of the two:
    #: 2*n_vib Hessians against 6N. It degrades to cartesian on its own when there
    #: are no isotopologues to build the mode decomposition from.
        cubic_scheme="normal_mode",
        freq_scale=1.0,
        lam_freq_cm=0.0,
        torsion_replacement=None,
        defect_bias_scale=False,
        harmonic_cd_from_hessian=False,
        cd_sigma_fraction=0.05,
        fit_cd_constants=False,
        cd_weight=0.0,
        coordinate_mode="internal",
        ic_damping=1e-6,
        ic_use_dihedrals=False,
        ic_micro_iter=20,
        ic_cartesian_fallback=False,
        ic_prior_weight=None,
        ic_prior_sigma_bond=None,
        ic_prior_sigma_angle_deg=None,
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

        # ── Rovibrational corrections (M1-M4) ────────────────────────────────
        self._harmonic_from_hessian = bool(harmonic_from_hessian)
        self._harmonic_sigma_fraction = max(float(harmonic_sigma_fraction), 1e-6)
        self._anharmonic_from_hessian = bool(anharmonic_from_hessian)
        self._anharmonic_fd_delta_ang = max(float(anharmonic_fd_delta_ang), 1e-4)
        self._nonconvergent_policy = str(nonconvergent_policy or "warn").strip().lower()
        # These four reach build_correction_table_from_hessian. Until now they
        # were only settable by calling that function directly, which meant the
        # benchmark scripts could pick a scheme and a config file could not --
        # so the Watson harmonic term, the normal-mode cubic transform,
        # frequency scaling and the large-amplitude cut were all unreachable
        # from the CLI and the web UI.
        self._harmonic_scheme = str(harmonic_scheme or "watson").strip().lower()
        self._cubic_scheme = str(cubic_scheme or "normal_mode").strip().lower()
        self._freq_scale = freq_scale
        self._lam_freq_cm = max(float(lam_freq_cm or 0.0), 0.0)
        # An explicitly computed torsional contribution, replacing the
        # perturbative alpha for the modes it covers. This is the only route by
        # which the torsion package reaches the optimiser: until now the full
        # torsion-rotation Hamiltonian, the averaging and the fitter were
        # reachable only from runner/run_generic.py, so none of that physics
        # could be scored by any benchmark, all of which run through this class.
        self._torsion_replacement = torsion_replacement
        # The level the Hessian-built corrections come from when it is not the
        # fit's own: None keeps the fit's level, re-expanded at the current
        # geometry on every Hessian refresh, as before. See
        # _correction_level_table for what a separate level does and why.
        self._correction_level = (
            None if correction_method is None and correction_basis is None
            else (None if correction_method is None else str(correction_method).strip(),
                  None if correction_basis is None else str(correction_basis).strip()))
        self._corr_backend = None
        self._corr_table = None     # (geometry, raw table, info), built once
        if self._correction_level is not None:
            if not self._harmonic_from_hessian:
                raise ValueError(
                    "correction_method / correction_basis set the level of the "
                    "Hessian-built corrections, but harmonic_from_hessian is off, so "
                    "there are none for them to change: turn it on, or drop them.")
            if torsion_replacement is not None:
                raise ValueError(
                    "torsion_replacement is computed at the fit's level; with a "
                    "separate correction level it would mix two force fields.")
        self._defect_bias_scale = bool(defect_bias_scale)
        self._harmonic_cd_from_hessian = bool(harmonic_cd_from_hessian)
        self._warned_cd_unvalidated = False
        self.chi2_rescale = bool(chi2_rescale)
        self.chi2_rescale_threshold = float(chi2_rescale_threshold)
        self.chi2_rescale_max_passes = int(chi2_rescale_max_passes)
        #: One record per rescale check of the last run() (empty when the
        #: rescale is off): reduced chi-square, effective dof, what the check
        #: did, and the factor when it refitted. Kept so a caller can see a
        #: fit that sits near a switch point (N19) without instrumenting run().
        self.chi2_rescale_checks = []
        self._last_chi2_dof = None
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
        # Fill in a missing observation sigma before anything reads it.
        #
        # The estimate is the flat relative default (DEFAULT_SIGMA_REL_ABC: 1%
        # of A, 0.5% of B and C). It used to be the inertial-defect estimate
        # (defect_model_sigma), on the reasoning that a species' own
        # vibrational contamination beats a fraction that is the same for
        # every molecule. Measured, it does not. Against the B3LYP VPT2 gaps
        # 0.5 |alpha_sum| on seven molecules (per-molecule medians; over all 37
        # species in brackets) it ran 0.17-8.9x (0.03-9.1) on A, 0.05-6.1x
        # (0.008-6.3) on B and 0.04-1.06x (0.007-1.06) on C -- too small on B
        # and C everywhere but water, fluoroethane's 20-25x too small -- where
        # the flat fractions are 1.0-2.5x (0.96-2.6) / 0.58-1.17x (0.57-1.4)
        # / 0.23-0.83x (0.20-0.84). Against equilibrium structures (r_e^SE for
        # vinyl fluoride and chlorofluoromethane, an r_e for ozone and water)
        # it is 0.14-0.40x on B and C for the first three and 8.2x / 1.08x for
        # water, the flat 0.58-1.18x / 0.24-0.80x; on A the flat is worst
        # there (ozone 8.6x, water 0.69x) and the defect worse still (16.6x,
        # 4.8x). The flat is closer in 19 of 21 molecule-components on the
        # VPT2 yardstick and 11 of 12 on the r_e one; the exceptions are
        # water's C and acetyl fluoride's A. VPT2 itself is good to ~10%
        # except on water's A (0.54x of its r_e gap) and ozone's A (1.8-3.9x).
        # The defect puts the whole residue on every component's moment,
        # which is not how a B0 - Be gap divides among A, B and C (decisions
        # D-67).
        #
        # sigma_systematic_constants is filled from the same estimate: it is
        # the model-error part, which for this error source is all of it, and
        # the post-correction sigma logic in resolve_corrections needs to know
        # how much of sigma the vibrational correction supersedes. (It is
        # propagated as shared across isotopologues per constant type; see
        # systematic_isotopologue_correlation.)
        # NOT auto-derived when sigma_constants was supplied, and the reason is
        # worth recording because the obvious fix does not work.
        #
        # _systematic_covariance returns zero when no systematic sigma is given,
        # so a user who quotes only measurement uncertainties gets a reported
        # uncertainty with the model-error term silently missing -- and model
        # error is the dominant term here. The engine's own benchmark harness
        # passes them explicitly, which is why that never showed up in testing.
        #
        # Deriving them automatically was tried and reverted, because the
        # propagation A = C J^T Sigma^-1 amplifies the systematic by
        # (sigma_sys/sigma_meas)^2 in variance and the reported width became
        # absurd -- bond widths of 47 Angstrom, an angle width of 10975
        # degrees. The diagnosis recorded here was right, and so was the
        # remedy it named: "floor the weighting sigma at the model error ...
        # rather than feed one into the other. That is a real piece of work
        # and not done yet."
        #
        # IT IS DONE NOW, and in two halves that were added years apart.
        # `sigma_floor_rel` is that floor, and it has been in
        # `_effective_sigma` for some time. What was missing is that
        # `_systematic_covariance` went on dividing by the RAW quoted
        # sigma_constants rather than by the floored one that `stacked`
        # actually weights J with -- on water a mismatch of 1090x to 4179x,
        # which is precisely where the absurd widths came from. With the two
        # reconciled, a 2.4% systematic (water's measured model error) gives
        # a bond width of 0.0128 Angstrom against 9.4045 before, and an
        # independent estimate from B ~ 1/r^2 puts it at 0.0115.
        #
        # So the blocker on auto-derivation is gone: run through this path
        # (test_auto_systematic_sigma's water fit) it yields 0.0018 Angstrom
        # and 0.32 degrees with the flat estimate (0.012 and 2.4 with the
        # inertial-defect one it replaced), not 47 and 10975.
        # Reinstating it is still a change of default -- every user who
        # quotes only measurement sigmas would start seeing the model-error
        # term included -- so it is left off pending that decision, and the
        # user is TOLD rather than handed a number that omits the dominant
        # term without saying so.
        self.auto_systematic_sigma = bool(auto_systematic_sigma)
        self.systematic_in_weighting = bool(systematic_in_weighting)
        _rho = float(systematic_isotopologue_correlation)
        if not (np.isfinite(_rho) and 0.0 <= _rho <= 1.0):
            raise ValueError("systematic_isotopologue_correlation must lie in "
                             f"[0, 1], got {systematic_isotopologue_correlation!r}")
        self.systematic_isotopologue_correlation = _rho
        if systematic_component_correlation is None:
            self.systematic_component_correlation = None
        else:
            _Qc = np.asarray(systematic_component_correlation, dtype=float)
            if (_Qc.shape != (3, 3) or not np.all(np.isfinite(_Qc))
                    or not np.allclose(_Qc, _Qc.T, atol=1e-12)
                    or not np.allclose(np.diag(_Qc), 1.0, atol=1e-12)):
                raise ValueError("systematic_component_correlation must be a "
                                 "symmetric 3x3 correlation matrix (unit "
                                 "diagonal) over A, B, C")
            _Qc = 0.5 * (_Qc + _Qc.T)
            if float(np.linalg.eigvalsh(_Qc).min()) < -1e-10:
                raise ValueError("systematic_component_correlation is not "
                                 "positive semidefinite, so it is not a "
                                 "correlation matrix")
            self.systematic_component_correlation = _Qc
        _filled = []
        _sys_filled = []
        _both_from_model = set()
        for _iso in isotopologues:
            _obs = np.asarray(_iso["obs_constants"], dtype=float)
            _ci = _iso.get("component_indices", list(range(len(_obs))))
            _needs_meas = _iso.get("sigma_constants") is None
            _needs_sys = (_iso.get("sigma_systematic_constants") is None
                          and self.auto_systematic_sigma)
            if not (_needs_meas or _needs_sys):
                continue
            _sig = np.asarray(default_sigma_constants(_obs, _ci), dtype=float)
            _how = "flat model default"
            if _needs_meas:
                _iso["sigma_constants"] = _sig.copy()
                _filled.append(f"{_iso.get('name', 'iso')} [{_how}]")
            if _needs_sys:
                # The model gap is a property of the physics, not of how
                # precisely anyone measured the constants, so it is derived
                # whether or not sigma_constants was supplied. Previously this
                # ran only when sigma_constants was absent, which meant a user
                # who quoted honest measurement precision -- the normal case --
                # silently got intervals missing the dominant term.
                _iso["sigma_systematic_constants"] = _sig.copy()
                _sys_filled.append(f"{_iso.get('name', 'iso')} [{_how}]")
                if _needs_meas:
                    # Both from one model estimate: sigma_constants IS the
                    # model error, and none of it is measurement.
                    _both_from_model.add(id(_iso))
        # Which species' model error the engine estimated, for
        # model_error_sources. Whether a species carries any model error is
        # known only once its corrections are applied (a correction with a
        # stated sigma carries it), so it is reported with the widths
        # (model_error_warnings), not here: this said "no
        # sigma_systematic_constants ... measurement precision ONLY" on every
        # config run, including those whose corrections carried it.
        _estimated = {str(n.split(" [")[0]) for n in _sys_filled}
        # Kept as given: a vibrational correction later writes its own sigma
        # into sigma_systematic_constants (resolve_corrections), so the field
        # alone cannot say which rows the caller stated.
        # Recorded by component (0, 1, 2 for A, B, C), not by position: the
        # sanitizer can drop a component, and a positional record then
        # matched no row, so stated values were labelled "corrections".
        self._model_error_given = {}
        for _i in isotopologues:
            if _i.get("sigma_systematic_constants") is None:
                continue
            _vals = np.abs(np.asarray(_i["sigma_systematic_constants"], dtype=float).ravel())
            _comps = np.asarray(_i.get("component_indices", range(_vals.size)), dtype=int).ravel()
            _name = str(_i.get("name", "iso"))
            self._model_error_given[_name] = (
                "estimated" if _name in _estimated else "stated",
                {int(c): float(v) for c, v in zip(_comps, _vals)})
        if _filled:
            print(f"  [sigma] derived for {len(_filled)} species: "
                  f"{', '.join(_filled[:3])}{' ...' if len(_filled) > 3 else ''}")
        if _sys_filled:
            print(f"  [sigma] systematic (B0-vs-Be model gap) derived for "
                  f"{len(_sys_filled)} species: {', '.join(_sys_filled[:3])}"
                  f"{' ...' if len(_sys_filled) > 3 else ''}. Reported "
                  f"intervals include it; pass auto_systematic_sigma=False "
                  f"for measurement precision only.")

        # WHICH PART OF sigma_constants IS MEASUREMENT.
        #
        # _systematic_covariance counts the model error once: it takes out
        # whatever of it the weighting sigma already holds and adds it back
        # with its correlation. So it has to know how much of sigma_constants
        # is model error. None of it, unless the species says so:
        # sigma_constants_include_systematic=True (the reference module's
        # sigmas() are hypot(model error, quoted precision)), or both were just
        # filled from the same model estimate. sigma_measurement_constants,
        # when given, states the measurement part directly and wins. Without
        # this, those runs carried the model error twice -- once as independent
        # noise in the weights, once more as the systematic term: on the
        # reference water fit the reported O-H width was 1.36x the Laplace,
        # nearly twice the variance.
        _normalised = []
        for _iso in isotopologues:
            if (_iso.get("sigma_measurement_constants") is None
                    and (bool(_iso.get("sigma_constants_include_systematic", False))
                         or id(_iso) in _both_from_model)):
                _sc = np.asarray(_iso["sigma_constants"], dtype=float).ravel()
                _ss = _iso.get("sigma_systematic_constants")
                _ss = (np.zeros_like(_sc) if _ss is None
                       else np.abs(np.asarray(_ss, dtype=float).ravel()))
                if _ss.size != _sc.size:
                    _ss = np.zeros_like(_sc)
                _iso = dict(_iso)
                # Factored, so the difference of two nearly equal numbers is
                # taken once (they differ by 1e5 on the reference sigmas).
                _iso["sigma_measurement_constants"] = np.sqrt(np.maximum(
                    (np.abs(_sc) - _ss) * (np.abs(_sc) + _ss), 0.0))
            _normalised.append(_iso)
        isotopologues = _normalised

        self._raw_isotopologues = list(isotopologues)   # preserved for harmonic updates
        # Fitting distortion constants needs OBSERVED ones to fit against, and
        # nothing anywhere supplies them by default. cd_residuals_mhz skips any
        # isotopologue without a cd_observed block, so with none present the
        # whole switch is a bit-exact no-op -- measured across all nine
        # benchmark molecules in both prior configurations, the fitted geometry
        # moved by at most 8e-9 mA, which is solver noise.
        #
        # That is a silence worth breaking. An hour of compute went into
        # measuring a feature that was switched on, reported no error, and did
        # nothing, and the next person to try would spend the same hour.
        if self._fit_cd_constants and self._cd_weight > 0.0:
            _fixed = [str(_iso.get("name", "iso")) for _iso in self._raw_isotopologues
                      if _iso.get("pred_cd") is not None]
            if _fixed:
                print(
                    "[cd-warning] pred_cd on " + ", ".join(_fixed) + " is not used for "
                    "fitting: the distortion rows predict from the Hessian at each\n"
                    "             trial geometry, since a fixed prediction cannot "
                    "constrain the geometry.")
            _n_cd = sum(
                1 for _iso in self._raw_isotopologues
                if (_iso.get("cd_observed") or _iso.get("centrifugal_distortion"))
            )
            if _n_cd == 0:
                print(
                    "[cd-warning] fit_cd_constants=True but no isotopologue "
                    "carries a cd_observed\n"
                    "             block, so there are no distortion rows to "
                    "fit and the switch\n"
                    "             will have NO effect. Add observed DJ/DJK/DK "
                    "(with the reduction\n"
                    "             they were fitted in) to the isotopologues "
                    "that have them."
                )
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
                defect_bias_scale=bool(defect_bias_scale),
                # The defect probe subtracts the structure's own inertial
                # defect, so the geometry it is read at sets the weights.
                # At the start geometry a restart from the answer re-weighted
                # the data and moved the fit (chlorofluoromethane, HF/6-31G,
                # benchmark recipe with defect_bias_scale: 5.85 mA); the prior
                # target is the fixed point, where the prior is centred on it.
                coords_ang=self._defect_probe_coords(prior_target_coords),
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
            sigma_floor_rel=sigma_floor_rel,
            systematic_in_weighting=systematic_in_weighting,
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
        #
        # That rule presupposes a prior standing behind the truncated
        # directions. In spectral-only mode there is none -- whatever the data
        # resolves, however weakly, is the entire answer -- so the floor must
        # not apply. Measured on fluoroacetylene's six B constants: their
        # weighted Jacobian's true rank is 3 (singular values 485 / 22.7 /
        # 5.2), the floor of 1/sigma_x = 50 truncated it to rank 1, and the
        # "fit" left 4 MHz cross-species residuals and C-F 22 mA wrong where
        # a direct least-squares on the same six numbers leaves 0.04 MHz and
        # 2.5 mA.
        # Calibrate the prior width to the level of theory when the caller did
        # not state one. sigma_x is an estimate of this level's own geometry
        # error, and the repository has measured that for several levels -- but
        # until now those measurements lived in a benchmark script while the
        # engine shipped one flat constant. Every benchmark passed sigma_x
        # explicitly and got the calibrated value; every CLI and browser run
        # took the flat 0.020. Through the joint objective -- the engine's
        # main solver -- that reaches alpha_q as 1/sigma_x^2, so at
        # B3LYP/cc-pVTZ the shipped engine weighted the quantum prior 25 times
        # more weakly than the benchmarked one. See spectral.prior_widths.
        if quantum_prior_sigma_ang is _PRIOR_SIGMA_UNSET:
            _lvl = level_for_backend(
                quantum_backend, orca_method, orca_basis,
                psi4_method, psi4_basis)
            _sig, _calibrated = prior_sigma_for_level(
                *_lvl, default=DEFAULT_QUANTUM_PRIOR_SIGMA_ANG)
            # Assign in BOTH branches. Everything downstream expects a float
            # here, and it always got one while the default was the flat
            # constant; leaving the uncalibrated path unassigned handed
            # SubspaceOptimizer a None it had never seen.
            quantum_prior_sigma_ang = _sig
            if _calibrated:
                print(f"  [prior] sigma_x = {_sig:.4f} A, calibrated for "
                      f"{_lvl[0]}/{_lvl[1]}")
            else:
                print(f"  [prior] sigma_x = {_sig:.4f} A, the uncalibrated "
                      f"default ({_lvl[0]}/{_lvl[1]} has not been benchmarked)")

        if sv_min_abs is None:
            if spectral_only:
                sv_min_abs = 0.0
            else:
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
            isotropic_prior_sigma_ang=(
                (quantum_prior_sigma_ang or DEFAULT_QUANTUM_PRIOR_SIGMA_ANG)
                if isotropic_prior else None),
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
        #: Geometry the quantum prior is centred on, when it is not the level
        #: of theory's own minimum. See _prior_gradient.
        self._prior_target_coords = (
            np.asarray(prior_target_coords, dtype=float).reshape(-1, 3)
            if prior_target_coords is not None else None)
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
        # Symmetry constraints. "auto" (the default) detects the point group from
        # the starting geometry and enforces it for the rest of the fit; pass
        # None to switch the constraint off.
        #
        # This is on by default because the rotational constants cannot see the
        # difference between a symmetric structure and a slightly distorted one,
        # so an unconstrained fit spends its information breaking symmetry that
        # physics fixes exactly. Water is the clean demonstration: its two O-H
        # bonds are equal by C2v, and an unconstrained parent-only fit splits
        # them by 3.9 mA -- larger than the whole remaining bond error.
        # Constraining it cuts that fit's RMS bond error from 3.29 to 0.83 mA.
        #
        # Detection only ever returns a group whose every operation the geometry
        # already satisfies, so the worst case is a subgroup of the true group,
        # which constrains less rather than more. C1 means no constraint was
        # found, and is dropped so the step avoids a 3N x 3N multiply by I.
        self.symmetry = symmetry
        self.point_group = None
        if isinstance(self.symmetry, str) or self.symmetry is None:
            if self.symmetry is not None:
                from backend.symmetry import create_symmetry  # pylint: disable=import-outside-toplevel
                self.symmetry = create_symmetry(
                    self.symmetry,
                    self.elems,
                    self.coords,
                )
                self.point_group = self.symmetry.point_group
                if self.symmetry.point_group == "C1":
                    self.symmetry = None
        elif self.symmetry is not None:
            self.point_group = getattr(self.symmetry, "point_group", None)
        self.debug_rank_diagnostics = bool(debug_rank_diagnostics)
        self.debug_sv_count = max(1, int(debug_sv_count))
        self.project_rigid_modes = bool(project_rigid_modes)
        # Watson mass-dependent B0 -> Be correction, fitted alongside the
        # structure. None disables it; "rm1" fits the three sqrt(I) coefficients.
        #
        # Off by default because it was measured not to help. On four molecules
        # at two data levels (RHF/6-31G, warm start) the mean RMS bond error
        # went 11.49 mA with it off to 12.23 / 13.08 / 17.85 at frac_sigma
        # 0.003 / 0.01 / 0.03 -- monotonically worse, and worse in every
        # fluorinated case.
        #
        # The reason is worth recording, because the same correction fitted
        # *standalone* does help, and markedly: 4.62 -> 2.33 mA on formyl
        # fluoride. Inside the hybrid the residual I_obs - I_m carries two
        # things, the rovibrational offset and the quantum method's structural
        # error. A standalone fit separates them because the structure is free
        # and only isotopic substitution moves it; here the quantum prior
        # anchors the structure, so the coefficients mop up the method's error
        # instead. The symptoms are visible in the fitted values: c comes out
        # negative on formyl fluoride where the standalone fit gives it
        # positive, and on water the signs disagree with the standalone fit on
        # two of three axes.
        #
        # Water at frac_sigma 0.03 does reach 0.14 mA from 1.75, but that is a
        # knife edge -- 1.05 mA by 0.05 and 5.43 mA by 0.1 -- so it is
        # coincidental cancellation rather than the correction doing its job.
        #
        # Turning it on would need the coefficients fitted from isotopic
        # differences alone, so that the method's error cancels as common mode.
        self._mass_dependent_model = (
            None if mass_dependent_model is None
            else str(mass_dependent_model).strip().lower())
        if self._mass_dependent_model not in (None, "rm1"):
            raise ValueError(
                f"mass_dependent_model must be None or 'rm1', got "
                f"{mass_dependent_model!r}")
        self._mass_dependent_frac_sigma = float(mass_dependent_frac_sigma)
        self._mass_dependent_c = None
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
        # Bonds and angles, plus dihedrals: True or False as stated, or "auto"
        # -- bonds and angles, with dihedrals added only when those do not
        # reach every direction the fit can move, decided once below
        # (_resolve_ic_dihedrals). The default stays False: "auto" moves two
        # benchmark answers (D-100), so it is asked for, not assumed.
        if isinstance(ic_use_dihedrals, str):
            _d = ic_use_dihedrals.strip().lower()
            if _d not in ("auto", "true", "false"):
                raise ValueError(f"ic_use_dihedrals must be True, False or 'auto', got {ic_use_dihedrals!r}")
            self._ic_dihedrals_setting = "auto" if _d == "auto" else (_d == "true")
        else:
            self._ic_dihedrals_setting = bool(ic_use_dihedrals)
        self._ic_use_dihedrals = self._ic_dihedrals_setting is True
        self._ic_dihedrals_reason = None
        self._ic_micro_iter = max(1, int(ic_micro_iter))
        # Opt-in: retake in Cartesians an internal step the back-transform
        # could not realise, and step in Cartesians for the rest of the pass
        # (_cartesian_fallback_step). Off by default: it moves acetyl
        # fluoride's answer and costs it a Hessian (D-106).
        self._ic_cartesian_fallback = bool(ic_cartesian_fallback)
        self._ic_fallback_steps = 0
        self._ic_switched = False
        # THE INTERNAL-COORDINATE PRIOR IS A THIRD TERM, so by default it is
        # only there when nothing else holds the undetermined directions.
        # The objective this engine is built on is spectra + quantum chemistry;
        # the library prior is neither. It pulls every bond and angle toward a
        # rule-book value (vinyl fluoride's H-C-H toward 109.5 deg, against
        # 120.1 published; formyl fluoride's O-C-H toward 120 against 127.3),
        # and with it on formyl fluoride's angle error was 2.03 deg against
        # 1.08 with it off. So with a quantum prior active the default weight
        # is 0; spectral-only fits, which have no other regulariser, keep 1.
        # An explicit ic_prior_weight is used as given.
        _qm_prior_on = (not self.spectral_only
                        and self.quantum_backend not in {"none", ""})
        if ic_prior_weight is None:
            ic_prior_weight = 0.0 if _qm_prior_on else 1.0
            self._ic_prior_weight_reason = (
                "off: the quantum prior is on" if _qm_prior_on
                else "on: spectral-only fit")
        else:
            self._ic_prior_weight_reason = "as given"
        self._ic_prior_weight = max(float(ic_prior_weight), 0.0)
        # WIDTHS AS GIVEN ARE WIDTHS AS USED. The rule library used to replace
        # the width of every coordinate it had a rule for, so a configured
        # 0.12 Angstrom / 5 deg became 0.015 / 2 without a word. Widths passed
        # here now win over the library's for their kind of coordinate; left
        # unset, the library's apply where it has a rule and these fallbacks
        # (0.05 Angstrom, 3 deg) elsewhere. Library centres apply either way.
        self._ic_prior_configured_kinds = {
            kind for kind, given in (("bond", ic_prior_sigma_bond),
                                     ("angle", ic_prior_sigma_angle_deg))
            if given is not None}
        self._ic_prior_sigma_bond = (
            0.05 if ic_prior_sigma_bond is None else float(ic_prior_sigma_bond))
        self._ic_prior_sigma_angle_deg = (
            3.0 if ic_prior_sigma_angle_deg is None else float(ic_prior_sigma_angle_deg))
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
                  f"damping={self._ic_damping:.1e}, prior_weight={self._ic_prior_weight:.2f} "
                  f"({self._ic_prior_weight_reason})")

        self._rigid_ref_masses = None
        if len(self.spectral.isotopologues) > 0:
            m = np.asarray(self.spectral.isotopologues[0]["masses"], dtype=float)
            if m.size == len(self.elems):
                self._rigid_ref_masses = m.copy()
        if self.coordinate_mode == "internal" and self._ic_dihedrals_setting == "auto":
            self._resolve_ic_dihedrals()
        self.quantum = None
        self._backend = None
        self._orca_ref_coords = None
        # Where the current Hessian was computed. Gradient-only updates move
        # _orca_ref_coords but keep the Hessian, so the two can be far apart.
        self._hessian_ref_coords = None
        # What the last fit iteration actually used, kept so that everything
        # reporting on the fit -- the Laplace, the log-probability, the run
        # report -- describes the objective that was minimised rather than
        # rebuilding its own (see _posterior_blocks).
        self._last_ic_prior = None
        self._last_alpha_q_eff = None
        # Quantum updates since the last full Hessian / rovib job. None means
        # "never computed", which forces one on the next update.
        self._since_hessian = None
        self._since_rovib = None
        self._last_data_support = {}
        self._last_sigma_split = {}
        self._prior_class_sigma = dict(prior_class_sigma) if prior_class_sigma else None
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
        if self._correction_level is not None:
            self._corr_backend = self._make_correction_backend()

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

    def _target_prior_active(self):
        """Whether the quantum prior is the Gaussian centred on
        prior_target_coords (Cartesian mode; see _prior_gradient)."""
        t = self._prior_target_coords
        return (t is not None and self.coordinate_mode == "cartesian"
                and np.shape(t) == np.shape(self.coords))

    def _defect_probe_coords(self, target):
        """Where resolve_corrections reads the defect probe (coords_ang).

        The probe subtracts the structure's own inertial defect, so the
        geometry it is read at sets the weights. When the quantum prior is the
        Gaussian centred on ``target`` (the condition _target_prior_active
        tests: Cartesian mode, the target's shape matching) the target is the
        fit's fixed point and the probe is read there, so a restart from the
        answer does not re-weight the data. Otherwise the target is not the
        prior's centre (internal mode ignores it) and the probe stays at the
        current geometry. A flat (3N,) target is reshaped as the attribute is.
        """
        coords = np.asarray(self.coords, dtype=float)
        if target is None or self.coordinate_mode != "cartesian":
            return coords
        t = np.asarray(target, dtype=float).reshape(-1, 3)
        return t if t.shape == coords.shape else coords

    def _target_hessian_frame(self):
        """The prior target placed in the current frame: where a target
        prior's curvature belongs."""
        return _kabsch_align(self._prior_target_coords, self.coords)

    def _run_hessian(self):
        """Full Freq job: refreshes both gradient and Hessian via the active backend.

        With a target-centred prior the Hessian is the prior's precision, so
        it is the one at the target, computed once and only rotated into the
        current frame afterwards; the energy and gradient are still the
        current geometry's. A Hessian at wherever the fit had got to made the
        prior move with the fit: on isocyanic acid (HF/6-31G, the benchmark
        recipe) a fresh Hessian at the answer moved it 97 mA, the next 38 mA,
        and three refreshes did not settle. Every shipped script starts at its
        target and is joint with one Hessian, so their answers are unchanged
        (to 1e-8 mA on the benchmark's nine molecules). A fit that recomputes
        its Hessian on schedule does change: the split objective with a target
        (no shipped script) moved isocyanic acid 0.91 mA.
        """
        if self._backend is None:
            raise RuntimeError("No quantum backend initialized.")
        if self._target_prior_active():
            at = self._target_hessian_frame()
            key = np.asarray(self._prior_target_coords, dtype=float).tobytes()
            held = getattr(self, "_target_hessian", None)
            if held is not None and held[0] == key:
                hessian = _rotate_cartesian_hessian(held[2], _kabsch_rotation(held[1], at))
            elif not np.allclose(at, self.coords, rtol=0.0, atol=1e-9):
                hessian = np.asarray(self._backend.run_hessian(at).hessian_bohr, dtype=float)
                self._target_hessian = (key, at.copy(), hessian.copy())
            else:
                hessian = None                # at the target: one job below
            if hessian is not None:
                result = self._backend.run_gradient(self.coords)
                self.quantum = QuantumState(result.energy, result.gradient_bohr, hessian)
                self._orca_ref_coords = self.coords.copy()
                self._hessian_ref_coords = at.copy()
                self._quantum_prior_weight_cache = None
                self._proj_ref_terms_cache = None
                return
        result = self._backend.run_hessian(self.coords)
        self.quantum = QuantumState(result.energy, result.gradient_bohr, result.hessian_bohr)
        self._orca_ref_coords = self.coords.copy()
        self._hessian_ref_coords = self.coords.copy()
        if self._target_prior_active():
            self._target_hessian = (np.asarray(self._prior_target_coords, dtype=float).tobytes(),
                                    self.coords.copy(),
                                    np.asarray(result.hessian_bohr, dtype=float).copy())
        # A new reference: a new weight, and new projected reference terms
        # (that cache is keyed on the coordinates alone, so an update at the
        # same coordinates paired the new reference point with the old
        # gradient and Hessian; distance_to_mode read 0.035 for 0.002).
        self._quantum_prior_weight_cache = None
        self._proj_ref_terms_cache = None

    def _run_gradient(self):
        """Cheap gradient-only job: refreshes gradient, keeps existing Hessian."""
        if self._backend is None:
            raise RuntimeError("No quantum backend initialized.")
        result = self._backend.run_gradient(self.coords)
        self.quantum.energy = result.energy
        self.quantum._gradient_bohr = result.gradient_bohr
        self._orca_ref_coords = self.coords.copy()
        self._quantum_prior_weight_cache = None
        self._proj_ref_terms_cache = None


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
        |Δα| > 2 × harmonic_sigma — which suggests the harmonic uncertainty estimate
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
                flag = "  [>2σ]" if diff > 2.0 * sig else ""
                parts.append(f"{lbl}: VPT2={float(v_vpt2):+.1f} harm={v_harm:+.1f} Δ={diff:.1f} MHz{flag}")
            if parts:
                print(f"  [vpt2-check]   {label}: {';  '.join(parts)}")

    def _make_correction_backend(self):
        """The backend the corrections come from: the fit's backend class at
        correction_method / correction_basis (each defaulting to the fit's)."""
        method = self._correction_level[0] or self.orca_method
        basis = self._correction_level[1] or self.orca_basis
        if self.spectral_only:
            raise ValueError("a correction level needs a quantum backend; this fit is "
                             "spectral-only")
        if self.quantum_backend in ("orca", "psi4"):
            raise ValueError(
                f"a separate correction level is evaluated at that level's own minimum, "
                f"so its backend must optimise a geometry; the '{self.quantum_backend}' "
                f"backend here does not (pyscf_hf does)")
        be = get_backend(self.quantum_backend)(
            elems=self.elems, method=method, basis=basis,
            charge=self.charge, multiplicity=self.multiplicity)
        if not callable(getattr(be, "optimise", None)):
            raise ValueError(
                f"a separate correction level is evaluated at that level's own minimum, "
                f"so its backend must optimise a geometry; '{self.quantum_backend}' has "
                f"no optimise()")
        return be

    def _correction_level_table(self):
        """The Hessian-built correction table at the correction level, at that
        level's own minimum: built once, then kept across Hessian refreshes.

        alpha is a property of a force field and the masses, so it is
        evaluated where that force field is stationary -- the benchmark's corr
        protocol (scripts/accuracy_upgrades_benchmark.py), whose notes record
        that evaluating another level's Hessians at the fit's geometry, off
        that level's minimum, made the level look irrelevant: an artefact of
        the protocol. And it is kept, not re-expanded at each refresh as the
        fit's own corrections are: it describes the correction level's
        surface, not where the fit has got to.

        Measured on water (fit and prior CCSD/cc-pVDZ, sigma_x 0.020, H2O +
        D2O, against r_e): bond -0.58 mA with the fit level's own
        corrections, -0.03 with them at MP2/cc-pVTZ (7 Hessians, 3.7 min),
        -0.08 at CCSD/cc-pVTZ (33 min); angle +0.164, +0.136, +0.102 deg.
        """
        if self._corr_table is not None:
            return self._corr_table[1], self._corr_table[2]
        from backend.spectral.harmonic_alpha import build_correction_table_from_hessian  # pylint: disable=import-outside-toplevel

        be = self._corr_backend
        label = f"{be.method}/{be.basis}" if hasattr(be, "method") else "the correction level"
        print(f"\n  [correction-level] Optimising at {label} for the corrections' own minimum...")
        geom = np.asarray(be.optimise(np.asarray(self.coords, dtype=float)), dtype=float)
        hess = np.asarray(be.run_hessian(geom).hessian_bohr, dtype=float)
        hessian_fn = None
        if self._anharmonic_from_hessian:
            def hessian_fn(coords_ang):
                return be.run_hessian(coords_ang).hessian_bohr
        ctbl_raw, info = build_correction_table_from_hessian(
            hess, geom, self._raw_isotopologues,
            sigma_fraction=self._harmonic_sigma_fraction,
            hessian_fn=hessian_fn,
            fd_delta_cubic=self._anharmonic_fd_delta_ang,
            nonconvergent_policy=self._nonconvergent_policy,
            harmonic_scheme=self._harmonic_scheme,
            cubic_scheme=self._cubic_scheme,
            freq_scale=self._freq_scale,
            lam_freq_cm=self._lam_freq_cm,
        )
        here = np.asarray(self.coords, dtype=float)
        shift = float(np.sqrt(np.mean(np.sum((here - _kabsch_align(geom, here)) ** 2, axis=1))))
        print(f"  [correction-level] corrections at {label}, at its minimum "
              f"({1e3 * shift:.2f} mA RMS from the fit's geometry); kept for the whole fit.")
        self._corr_table = (geom, ctbl_raw, info)
        return ctbl_raw, info

    def _apply_harmonic_alpha_corrections(self):
        """Recompute harmonic alpha from the current Hessian and update spectral targets.

        Re-applies rovibrational + electronic corrections to the raw
        (uncorrected) isotopologue data using the current harmonic alpha.

        This fires on EVERY Hessian recalculation, not once. An earlier version
        of this docstring claimed otherwise, and there is no guard that would
        make it true. The distinction matters: self.coords is the CURRENT
        geometry, so each rebuild re-expands alpha about wherever the optimiser
        has reached. That is the expansion-point tracking the benchmark calls
        the "direct" scheme -- measured there as the best of three, since it is
        the wanted quantity rather than an extrapolation towards it -- and it
        means hess_recalc_every, not a separate switch, is what controls how
        closely the correction follows the fit.
        """
        from backend.spectral.harmonic_alpha import build_correction_table_from_hessian  # pylint: disable=import-outside-toplevel

        if self.quantum is None:
            return
        if self._corr_backend is not None:
            # A separate correction level: its own minimum, built once.
            ctbl_raw, _res_info = self._correction_level_table()
        else:
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
                harmonic_scheme=self._harmonic_scheme,
                cubic_scheme=self._cubic_scheme,
                freq_scale=self._freq_scale,
                lam_freq_cm=self._lam_freq_cm,
                torsion_replacement=self._torsion_replacement,
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
                "skipped (|ω_s²−ω_r²| < 0.01 cm⁻²). Alpha values may be less reliable for "
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
                    parts.append(f"{comp_lbl}={v:+.1f}±{s:.1f} (Δ={delta:+.2f})")
                else:
                    parts.append(f"{comp_lbl}={v:+.1f}±{s:.1f}")
                self._prev_harmonic_alpha_sum[key] = v
            print(f"  [harmonic-alpha]   {iso_name}: Σα = {', '.join(parts)} MHz")
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
            defect_bias_scale=self._defect_bias_scale,
            # As at construction: the prior target when the prior is centred on it.
            coords_ang=self._defect_probe_coords(self._prior_target_coords),
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
            # And the split of that sigma into model error and measurement, so
            # the reported covariance does not charge a gap just corrected away.
            for _key in ("sigma_measurement_constants", "sigma_systematic_constants"):
                _v = new_iso.get(_key)
                if _v is not None and np.asarray(_v).size == np.asarray(new_iso["obs_constants"]).size:
                    old_iso[_key] = np.asarray(_v, dtype=float)

    def _apply_harmonic_cd_corrections(self):
        """Attach harmonic CD predictions from the current Hessian to rovib_correction.pred_cd."""
        from backend.spectral.centrifugal_distortion import build_cd_table_from_hessian

        if self.quantum is None:
            return
        hess_bohr = self.quantum._hessian_bohr
        print("\n  [harmonic-cd] Computing harmonic CD constants from Hessian...")
        if not self._warned_cd_unvalidated:
            self._warned_cd_unvalidated = True
            # This used to warn that the mapping produced DJ and DK with the
            # wrong sign (-66.9 against +37.6, -7.0 against +973.3) and DJK
            # nine times too small. That described the old closed-form
            # coefficient table, which cd_reduction replaced with a linear
            # solve against the levels tau itself generates. The signs are now
            # right on all five constants and the magnitudes are measured, so
            # the warning says what is actually known.
            print(
                "  [harmonic-cd] NOTE: validated on water only, where the "
                "A-reduced constants\n"
                "                come out DJ +38.2 against +37.59 (1.6% high), "
                "DK +899 against\n"
                "                +973.3 (7.6% low) and DJK -235 against -172.9 "
                "(36% low) --\n"
                "                force-field error, not mapping error, and DJK "
                "is the most\n"
                "                bend-sensitive of the three. One molecule is "
                "not a validation\n"
                "                set; see dev/tests/test_cd_mapping_validation.py."
            )
        # The reduction (or the linear branch) each isotopologue's rows use,
        # pinned here if not yet, so the table is in the same one: decided per
        # geometry instead, a jittered start of a linear molecule printed and
        # attached DJ ~1e8 MHz.
        _at0 = np.asarray(self.coords, dtype=float)
        pinned = {str(i.get("name", "iso")): self.spectral._cd_reduction(i, _at0)
                  for i in self.spectral.isotopologues}
        # In the Hessian's own frame (SpectralEngine.cd_frame_coords says why).
        ref = self._hessian_ref_coords
        at = (_kabsch_align(self.coords, ref)
              if ref is not None and np.shape(ref) == np.shape(self.coords) else self.coords)
        cd_table = build_cd_table_from_hessian(
            hess_bohr,
            at,
            [dict(i, cd_reduction=pinned.get(str(i.get("name", "iso"))) or i.get("cd_reduction"))
             for i in self._raw_isotopologues],
            sigma_fraction=self._cd_sigma_fraction,
        )
        if not cd_table:
            print("  [harmonic-cd] Warning: no CD values computed; skipping update.")
            return
        for iso_name, cd in cd_table.items():
            if getattr(cd, "reduction", None) == "linear":
                print(f"  [harmonic-cd]   {iso_name}: D={cd.DJ:.6f} MHz (linear molecule: "
                      f"DJK and DK do not exist)")
                continue
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
        self.spectral.set_hessian_for_cd(hess_bohr, coords=self._hessian_ref_coords)

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
        """Decide whether to do a full Hessian recalculation or gradient-only update.

        The schedule counts calls since the last full Hessian rather than
        testing ``(call_count - 1) % period``. A modulus only describes a
        periodic schedule while the period is constant, and this one is not:
        the adaptive block below rewrites ``hess_recalc_every`` between calls,
        which shifts the phase and can step the counter straight over the
        firing residue. Measured on the benchmark molecules, a nominal period
        of 5 produced one Hessian in 20 steps for fluoroacetylene and none at
        all in 14 for formyl fluoride. That matters more under the joint
        objective than it used to, because there the Hessian *is* the quantum
        prior -- a stale one anchors the prior to a geometry the fit has left.
        """
        # Count the call first, then test: `_since_hessian` is the number of
        # updates that have happened since the last Hessian, so a period of 1
        # has to fire on the very next call. Testing before incrementing makes
        # every period one longer than asked for, which for period 1 means every
        # other call rather than every call.
        if self._since_hessian is not None:
            self._since_hessian += 1
        if self._since_rovib is not None:
            self._since_rovib += 1
        hess_due = (self.quantum is None
                    or self._since_hessian is None
                    or self._since_hessian >= self.hess_recalc_every)
        if hess_due:
            self._run_hessian()
            self._since_hessian = 0
            rovib_due = (self._since_rovib is None
                         or self._since_rovib >= self.rovib_recalc_every)
            if self.use_orca_rovib and rovib_due:
                self._run_rovib()
                self._since_rovib = 0
            if self._harmonic_from_hessian:
                self._apply_harmonic_alpha_corrections()
            if self._harmonic_cd_from_hessian:
                self._apply_harmonic_cd_corrections()
            elif self._fit_cd_constants and self._cd_weight > 0.0 and self.quantum is not None:
                self.spectral.set_hessian_for_cd(self.quantum._hessian_bohr,
                                                 coords=self._hessian_ref_coords)
        else:
            self._run_gradient()
            if self._fit_cd_constants and self._cd_weight > 0.0 and self.quantum is not None:
                self.spectral.set_hessian_for_cd(self.quantum._hessian_bohr,
                                                 coords=self._hessian_ref_coords)

    def _force_hessian_refresh(self):
        """Make the next quantum update recompute the Hessian rather than reuse it."""
        self._since_hessian = None

    # ── Pre-computed files ────────────────────────────────────────────────────

    def load_orca(self, engrad_path, hess_path):
        """
        Load pre-computed ORCA output files instead of running ORCA.
        Call this before run() when you have existing .engrad / .hess files.
        """
        eng = QuantumEngine(engrad_path, hess_path, self.elems)
        self.quantum = QuantumState(eng.energy, eng._gradient_bohr.copy(), eng._hessian_bohr.copy())
        self._orca_ref_coords = self.coords.copy()
        self._hessian_ref_coords = self.coords.copy()
        self._quantum_prior_weight_cache = None
        self._proj_ref_terms_cache = None
        print(f"Loaded ORCA files.  Energy = {self.quantum.energy:.10f} Hartree")

    # ── Drift check ───────────────────────────────────────────────────────────

    def _orca_drift(self):
        if self._orca_ref_coords is None:
            return np.inf
        return float(np.sqrt(np.mean((self.coords - self._orca_ref_coords) ** 2)))

    def distortion_rows(self, coords=None):
        """The centrifugal-distortion rows the fit carries, as the report shows
        them: per isotopologue and constant, observed and predicted (MHz), the
        noise their weight implies (sigma / sqrt(cd_weight)) and the pull z.
        Empty when the rows are off. They drive the fit and the chi-square
        like the rotational rows, and used to appear in no report at all."""
        sp = self.spectral
        if not sp._cd_rows_active():
            return []
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        root_w = float(np.sqrt(sp.cd_weight))
        out = []
        for iso in sp.isotopologues:
            rows = sp._cd_rows(iso, x)
            if rows is None:
                continue
            keys, obs_v, pred_v, sig_v = rows
            for k, o, pr, sg in zip(keys, obs_v, pred_v, sig_v):
                eff = float(sg) / root_w
                out.append({"isotopologue": str(iso.get("name", "iso")), "constant": k,
                            "observed_mhz": float(o), "predicted_mhz": float(pr),
                            "residual_mhz": float(o - pr), "sigma_mhz": eff,
                            "z": float((o - pr) / eff),
                            "reduction": str(iso.get("cd_reduction", "")),
                            "reduction_source": str(iso.get("cd_reduction_source", ""))})
        return out

    def _model_error_rows(self):
        """Per isotopologue: the kinds of model error its rows carry (see
        model_error_sources) and the components whose given value -- stated,
        or the engine's estimate -- a correction's sigma replaced."""
        out = {}
        given = getattr(self, "_model_error_given", {})
        for iso in self.spectral.isotopologues:
            name = str(iso.get("name", "iso"))
            comps = np.asarray(iso["component_indices"], dtype=int).ravel()
            n = comps.size
            sys_sig = iso.get("sigma_systematic_constants")
            sys_sig = (np.zeros(n) if sys_sig is None
                       else np.abs(np.asarray(sys_sig, dtype=float).ravel()))
            if sys_sig.size != n:
                sys_sig = np.zeros(n)
            corr = np.asarray(self.spectral._correction_sigma(iso, n), dtype=float)
            kind_given, vals_given = given.get(name, (None, {}))
            kinds, replaced = set(), []
            for c, cs, s in zip(comps, corr, sys_sig):
                g = vals_given.get(int(c))
                if cs > 0.0 or (s > 0.0 and not (
                        g is not None and np.isclose(s, g, rtol=1e-12, atol=0.0))):
                    # A correction's sigma: it replaces the gap's in
                    # _systematic_covariance, and a vibrational correction
                    # writes its own into sigma_systematic_constants.
                    kinds.add("corrections")
                    if g is not None and g > 0.0:
                        replaced.append(int(c))
                elif s > 0.0:
                    kinds.add(kind_given)
                else:
                    kinds.add(None)
            out[name] = (kinds, kind_given, replaced)
        return out

    def model_error_sources(self):
        """Per isotopologue, where its rows' model error -- the B0-versus-Be
        gap, propagated as shared across isotopologues -- comes from:
        "stated" (sigma_systematic_constants given), "estimated"
        (auto_systematic_sigma), "corrections" (a B0 -> Be correction with a
        stated sigma on its rows), a "+" join where rows differ, or None
        where no row carries any: its widths cover measurement precision
        only. "(stated replaced)" marks a species whose given value a
        correction's sigma replaced on some row (model_error_warnings names
        the rows). Distortion rows carry none by design and are not counted."""
        out = {}
        for name, (kinds, kind_given, replaced) in self._model_error_rows().items():
            if kinds == {None}:
                out[name] = None
                continue
            out[name] = ("+".join(sorted(k for k in kinds if k))
                         + (" (some rows none)" if None in kinds else "")
                         + (f" ({kind_given} replaced)" if replaced else ""))
        return out

    def model_error_warnings(self):
        """The report's statement of what the widths leave out, from the rows
        actually fitted (model_error_sources)."""
        if self.spectral_only and not self.spectral.isotopologues:
            return []
        src = self.model_error_sources()
        bare = [n for n, k in src.items() if k is None]
        part = [n for n, k in src.items() if k is not None and "some rows none" in k]
        msgs = []
        if bare and len(bare) == len(src):
            msgs.append(
                "no model-error term: no row carries the B0-versus-Be gap, so these "
                "widths cover the stated measurement precision only, and the gap "
                "usually dominates (analytic water: 0.28 mA without it, 6.9 with its "
                "corrections' sigma). State sigma_systematic_mhz per isotopologue, apply "
                "B0 -> Be corrections with a stated sigma, or set model_error: {auto: "
                "true} for the engine's flat estimate (1% of A, 0.5% of B and C)")
        elif bare or part:
            msgs.append(
                "the model-error term covers only part of the data: "
                + ", ".join(bare + part) + " carry it on none or only some of their rows, "
                "and their widths there cover measurement precision only")
        # A given value a correction replaced was dropped without a word: the
        # widths did not move for a stated 1% or 10% on corrected water.
        swapped = [(n, kg, r) for n, (_k, kg, r) in self._model_error_rows().items() if r]
        if swapped:
            what = "; ".join(f"{n} ({', '.join('ABC'[c] if 0 <= c < 3 else str(c) for c in r)})"
                             for n, _kg, r in swapped)
            given_as = sorted({kg for _n, kg, _r in swapped})
            msgs.append(
                f"the {' and '.join(given_as)} model error is not used on rows a B0 -> Be "
                f"correction with its own sigma was applied to: {what}. The correction "
                f"removes the gap that value stood for, and its sigma is the error left; "
                f"to widen those rows, raise the correction's sigma "
                f"(harmonic_sigma_fraction, sigma_correction_mhz)")
        return msgs

    def distortion_warnings(self):
        """What the distortion rows' weights and coverage mean, for the
        report: printed once when the engine is built, and lost from any
        record kept after that."""
        from backend.spectral.spectral import CD_FIT_KEYS
        sp = self.spectral
        if not sp._cd_rows_active() or not any(
                k in (iso.get("cd_observed") or {})
                for iso in sp.isotopologues for k in CD_FIT_KEYS):
            return []
        out = []
        for iso in sp.isotopologues:
            dropped = [k for k in ("DJK", "DK") if k in (iso.get("cd_observed") or {})]
            if iso.get("cd_reduction") == "linear" and dropped:
                out.append(f"{iso.get('name', 'iso')} is predicted as a linear molecule, "
                           f"which has one quartic constant, D (fitted as DJ); the stated "
                           f"{', '.join(dropped)} are not fitted.")
        w = float(sp.cd_weight)
        # The model part each row actually carries, per key, as a share of
        # the observed value: 40% by default, the larger cd_sigma_fraction
        # or a stated cd_model_sigma otherwise, each divided by sqrt(w).
        parts = []
        for iso in sp.isotopologues:
            obs = iso.get("cd_observed") or {}
            model = iso.get("cd_model_sigma") or {}
            src = iso.get("cd_model_sigma_source") or {}
            for k in CD_FIT_KEYS:
                if k in obs and k in model and abs(float(obs[k])) > 0.0:
                    parts.append((iso.get("name", "iso"), k,
                                  100.0 * float(model[k]) / abs(float(obs[k])),
                                  src.get(k, "default")))
        default_only = all(s_ == "default" for *_r, s_ in parts)
        if parts and (abs(w - 1.0) > 1e-12 or not default_only):
            names = {n for n, *_r in parts}
            shown = ", ".join(
                (f"{n} " if len(names) > 1 else "") + f"{k} {f / np.sqrt(w):.3g}%"
                + ("" if s_ == "default" else f" ({s_})")
                for n, k, f, s_ in parts)
            if abs(w - 1.0) > 1e-12:
                out.append(
                    f"distortion rows carry cd_weight {w:g}: each row's noise is its sigma / "
                    f"sqrt({w:g}), so the harmonic model's error enters as {shown} of the "
                    f"observed values; only cd_weight 1 keeps the stated model sigma")
            else:
                out.append(
                    f"distortion rows' model error is {shown} of the observed values (the "
                    f"mapping is validated at 40% on water)")
        if sp.cd_unfitted_keys:
            out.append(
                f"observed {', '.join(sp.cd_unfitted_keys)} not fitted: the harmonic "
                f"mapping is validated on DJ, DJK and DK only")
        return out

    #: Below this, split_coupling is round-off: a split fit whose data reach
    #: every physical direction has no null space to couple (exactly 0), and
    #: one whose null direction is decoupled by symmetry reads about 1e-11.
    #: Above it the null response is a DETECTOR, not a measure of the error:
    #: one-constant water reads 0.46 and its split refits put the angle at
    #: 0.04x the posterior's spread, but C1 water from an asymmetric start
    #: reads 0.058 with refits within 1% of it, and two-isotopologue B,C
    #: water reads 0.0195 with refits 1.3-1.4x off. Only a bootstrap of the
    #: split fit says which (its summary flags a departure).
    SPLIT_COUPLING_TOL = 1e-6

    def split_coupling(self):
        """The split step's null response, or None off the split objective.

        The split step's quantum gradient is Pn (g + H dx); the posterior it
        is reported with keeps only Pn H Pn. At the step's fixed point a
        range displacement dx_r moves the null coordinates by -K dx_r,
        K = (Vn^T Hb Vn)^-1 Vn^T Hb (I - Vn Vn^T), over the physical
        directions B (Hb = B^T Hs B, Vn the physical part of the null space
        of the stacked Jacobian). This returns |K|_2: dimensionless, exactly
        zero when the data reach every physical direction (or symmetry
        decouples the rest), so the split posterior is exact. Nonzero, it
        says the posterior leaves out a coupling, not how much that matters
        (see SPLIT_COUPLING_TOL); and it reads the posterior's null space,
        not the one the step decides (split_off_mode covers a disagreement).
        """
        if (self.spectral_only or self.quantum is None
                or self.optimizer.objective_mode == "joint"):
            return None
        x = np.asarray(self.coords, dtype=float)
        Pn = self._split_energy_projector()
        _g, Hp = self._projected_reference_terms()
        Hs = np.asarray(self.optimizer._spd_hessian(Hp), dtype=float)
        B = self.physical_basis(coords=x, n=Hs.shape[0])
        N = B.T @ Pn @ B
        w, V = np.linalg.eigh(0.5 * (N + N.T))
        Vn = V[:, w > 0.5]
        if Vn.shape[1] == 0:
            return 0.0
        Hb = B.T @ Hs @ B
        rest = np.eye(Hb.shape[0]) - Vn @ Vn.T
        K = np.linalg.solve(Vn.T @ Hb @ Vn, Vn.T @ Hb @ rest)
        return float(np.linalg.norm(K, 2))

    #: A split fit whose null-space energy has its minimum further than this
    #: from the fit, in posterior widths (as distance_to_mode counts them),
    #: stopped short of the split step's fixed point (split_stall).
    SPLIT_STALL_LIMIT = 0.25

    def _split_record(self):
        """The split step's last record, carried to this geometry, or None.

        The fit loop records what each split step solved with
        (_last_split_step, at the geometry it was computed at). A fit ends
        on a refused step, at that geometry, or on an accepted one, a step
        away from it: the quadratic model's gradient is then carried here
        exactly (g + H d, d that step in the step's own parameters), and the
        null space, a Jacobian's, is taken as unchanged over one step.
        Asking for the record's own geometry instead hid stalls: acetyl
        fluoride (HF/6-31G, Cartesian) ends on an accepted 1e-4 A step after
        eleven refusals at the trust floor, 4.0 widths short, and
        fluoroacetylene (internal) likewise (T-50 verifier)."""
        rec = getattr(self, "_last_split_step", None)
        x = np.asarray(self.coords, dtype=float)
        if rec is None or rec["coords"].shape != x.shape:
            return None
        if np.array_equal(rec["coords"], x):
            return rec
        if rec["Bplus"] is None:
            d = (x - rec["coords"]).ravel()
        else:
            cs = rec["coord_set"]
            d = (np.asarray(cs.active_values(x), dtype=float)
                 - np.asarray(cs.active_values(rec["coords"]), dtype=float))
            for k, ic in enumerate(cs.active_coords()):
                if ic.kind == "dihedral":
                    d[k] = (d[k] + np.pi) % (2.0 * np.pi) - np.pi
        return dict(rec, g=rec["g"] + rec["H"] @ d)

    def _split_step_newton(self):
        """One Newton step on what the split step last solved with, over its
        own null space and in its own parameters: (Cartesian step, number of
        null directions along which the quadratic model has no minimum, the
        step in the step's own parameters), or None without a record.

        The record (_split_record) holds the step's stacked Jacobian's
        right singular vectors and rank, and the gradient and Hessian it was
        handed -- after the pull toward a prior target, in q-space in
        internal mode -- so this reads the null space the step decides, not
        the one the posterior reads: in internal mode the two differ (the
        step's is on the q-space stack), and an internal fit standing at its
        own fixed point read as stalled on the posterior's (T-49 verifier:
        two-constant water at sigma_x 0.5 mA, 4.1 widths against 0.03 on
        its own; acetyl fluoride 3.6 against 0.015). Directions of negative
        curvature have no minimum to measure a distance to: they are counted
        and left out, where an eigenvalue floor would have set the length
        (acetyl fluoride and isocyanic acid, Cartesian: 1.5-2.2 A at a floor
        of 1e-8, 0.05-1.7 A for floors of 1e-4 to 1e-2; T-49 verifier). In
        internal mode the step reaches Cartesians through B+, linearly:
        formyl fluoride's 8.8 mA realises as 6.8 by back-transformation, and
        fluoroacetylene's step as 186 mA in norm against 278 linearised
        (T-50 verifier)."""
        rec = self._split_record()
        x = np.asarray(self.coords, dtype=float)
        if rec is None:
            return None
        Vn = rec["Vt"][rec["rank"]:].T
        if Vn.shape[1] == 0:
            return np.zeros(x.size), 0, np.zeros(Vn.shape[0])
        Hn = Vn.T @ rec["H"] @ Vn
        w, U = np.linalg.eigh(0.5 * (Hn + Hn.T))
        tol = 1e-6 * max(float(np.max(np.abs(w))), 1e-300)
        z = U.T @ (Vn.T @ rec["g"])
        dz = np.zeros_like(z)
        pos = w > tol
        dz[pos] = -z[pos] / w[pos]
        dq = Vn @ (U @ dz)
        dx = dq if rec["Bplus"] is None else rec["Bplus"] @ dq
        return np.asarray(dx, dtype=float).ravel(), int(np.sum(w < -tol)), dq

    def _split_null_mismatch(self):
        """How far the null space the split step last decided is from the
        one its posterior reads, as the sine of their largest principal angle
        over the physical directions (1 when their dimensions differ, 0 when
        they agree), or None without a record. They differ where the step's
        own cutoff, or its q-space stack in internal mode, sees the data
        differently from the posterior (split_off_mode); equal ranks are not
        enough in internal mode (one-constant water with an internal prior:
        rank 2 of 3 on the step's q-space stack against 3 of 9 in
        Cartesians, null spaces of different dimension, 2.2 widths off the
        posterior's mode). Both sides are cut by one rule -- the eigenvalues
        of B^T P B above 0.5, P the null space's projector in Cartesians --
        so that where the step and the posterior decompose one Jacobian
        (Cartesian mode) they read the same null space: cut differently, a
        null vector mixing symmetry-allowed and forbidden parts counted on
        one side only, and water with an HDO constant under C2v read 1.0 for
        projectors equal to 2e-16 (T-50 verifier)."""
        rec = self._split_record()
        x = np.asarray(self.coords, dtype=float)
        if rec is None:
            return None
        B = self.physical_basis(coords=x, n=x.size)

        def cut(P):
            N = B.T @ P @ B
            w, V = np.linalg.eigh(0.5 * (N + N.T))
            return V[:, w > 0.5]

        Vp = cut(self._split_energy_projector())
        Vn = rec["Vt"][rec["rank"]:].T
        if rec["Bplus"] is not None:
            Vn = rec["Bplus"] @ Vn
        if Vn.size == 0:
            Q = np.zeros((x.size, 0))
        else:
            # Its span in Cartesians, relative to the columns' own length:
            # redundant combinations (q-space) have no image.
            ref = max(float(np.linalg.norm(Vn, axis=0).max()), 1e-300)
            U, sv, _ = np.linalg.svd(Vn, full_matrices=False)
            Q = U[:, sv > 1e-6 * ref]
        Vs = cut(Q @ Q.T)
        if Vs.shape[1] != Vp.shape[1]:
            return 1.0
        if Vp.shape[1] == 0:
            return 0.0
        return float(np.linalg.norm(Vs @ Vs.T - Vp @ Vp.T, 2))

    #: Null spaces closer than this (sine of the largest principal angle)
    #: are the same one for split_inconsistency's purposes.
    SPLIT_NULL_AGREE = 0.1

    def split_stall(self):
        """How far the split fit stopped from the minimum of its null-space
        energy: (posterior widths, largest atom move in Angstrom, null
        directions with no minimum), or None off the split objective or
        without a record of its last step.

        The split step fits the data on the range of its stacked Jacobian
        and lets the quantum surface alone decide the rest, so its fixed point
        has no energy gradient left along its null space. In the Python API
        its rank is cut at sv_min_abs = 1/sigma_x, which puts directions the
        data still weigh (singular values 0.2-6% of the largest on formyl
        fluoride, vinyl fluoride, chlorofluoromethane at HF/6-31G) on the
        energy's side, while the acceptance test still scores the data there
        -- formyl fluoride's null step takes its residual from 69 to 180 MHz
        -- so the step is refused, the trust radius falls to its floor, and
        the fit stops with that gradient unspent. One Newton step on what the
        step last solved with (_split_step_newton) measures how far. On the
        benchmark molecules (HF/6-31G, Python API defaults), where the
        step's null space is its posterior's, it read 2.9-44 mA (largest atom)
        in Cartesian coordinates -- acetyl fluoride's 44 and isocyanic acid's
        28 with a direction of negative curvature besides -- and 0.8-28 in
        internal ones, through B+ (fluoroacetylene's 115, linearised, is far
        from what back-transforms); the note fires on 9 of the 18 fits (T-50).
        A trust radius of 0.05 or 0.2 instead of 0.1 moved such answers up to
        1.5 of their reported widths (T-48), so neither the split posterior nor any
        linear response describes the answer or its spread (a coupling-only
        response matched T-40's refit clouds no better than the posterior).
        Zero where the step's null space is empty or its gradient spent.
        """
        if (self.spectral_only or self.quantum is None
                or self.optimizer.objective_mode == "joint"):
            return None
        out = self._split_step_newton()
        if out is None:
            return None
        dx, n_neg, _dq = out
        x = np.asarray(self.coords, dtype=float)
        if not np.any(dx):
            return 0.0, 0.0, n_neg
        B = self.physical_basis(coords=x, n=x.size)
        H_spec, H_prior = self._posterior_blocks(x)[:2]
        M = B.T @ (H_spec + H_prior) @ B
        wm, Vm = np.linalg.eigh(0.5 * (M + M.T))
        db = B.T @ dx       # its physical part: rigid motion is no distance
        widths = float(np.max(np.abs(Vm.T @ db) * np.sqrt(np.maximum(wm, 0.0) + 1e-8)))
        atom = float(np.abs((B @ db).reshape(-1, 3)).max())
        return widths, atom, n_neg

    #: A split fit further than this from the mode its posterior describes,
    #: in posterior widths (the bootstrap's default refusal limit), is one
    #: whose step and posterior disagree about the objective -- once it has
    #: converged; stopped at max_iter it has simply not got there.
    SPLIT_MODE_LIMIT = 0.25

    def split_off_mode(self):
        """Why this split fit does not stand at its posterior's mode, or None
        (also None off the split objective).

        A converged split fit more than SPLIT_MODE_LIMIT posterior widths
        from its posterior's mode: the step has settled where the
        posterior's gradient is not zero (one-constant water with an internal
        prior: 2.2 widths; two-isotopologue C-only water, whose step settles
        at rank 1 against a rank-2 posterior: 3.8 widths), so the posterior
        is centred somewhere the fit is not. A fit stopped at max_iter is
        not counted here; split_inconsistency says it has not converged."""
        if self.split_coupling() is None:
            return None
        try:
            d = float(self.distance_to_mode())
        except Exception as exc:
            return (f"the split fit's distance to its posterior's mode could not be "
                    f"computed ({type(exc).__name__}: {exc}), so neither it nor a refit "
                    f"can be certified")
        term = (getattr(self, "termination", None) or {}).get("reason")
        if d > self.SPLIT_MODE_LIMIT and term != "max_iter":
            mismatch = self._split_null_mismatch()
            if mismatch is not None and mismatch < self.SPLIT_NULL_AGREE:
                # Same null space, no stall left: what moves the posterior's
                # mode is the coupling its quantum term leaves out (a prior
                # target at sigma_x 5 mA: 1.0 widths; T-50 verifier).
                return (f"the converged split fit stands {d:.2g} posterior widths from the mode "
                        f"its posterior describes: the step and its posterior read the same null "
                        f"space, but the posterior leaves out the step's range-null coupling "
                        f"(null response {self.split_coupling():.2g}), which moves its mode")
            how = ("decides which directions are null on its own q-space stack"
                   if self.coordinate_mode == "internal"
                   else "decides the data's rank with its own cutoff")
            return (f"the converged split fit stands {d:.2g} posterior widths from the mode "
                    f"its posterior describes: the step {how}, and here the two disagree")
        return None

    def split_inconsistency(self):
        """What may keep the split posterior from describing this split fit,
        for the report, or None (also None off the split objective): a fit
        stalled short of its null-space energy's minimum (split_stall), one
        off its posterior's mode (split_off_mode), one stopped at max_iter
        short of it, or a nonzero null response (a detector: an error-model
        bootstrap measures whether it matters)."""
        coupling = self.split_coupling()
        if coupling is None:
            return None
        term = (getattr(self, "termination", None) or {}).get("reason")
        # Checked first: a stalled fit is also off its posterior's mode, and
        # the coupling can read zero while it stalls (every direction on the
        # energy's side: one-constant water at sigma_x 0.5 mA). Measured on
        # the step's own null space, so a fit at its own fixed point whose
        # posterior reads another null space keeps the off-mode note.
        stall = self.split_stall()
        mismatch = self._split_null_mismatch()
        # Where the step and its posterior read different null spaces, that
        # is the larger gap (two-isotopologue C-only water: 3.8 widths off
        # the posterior's mode, 1.3 short of its own null-space minimum), and
        # the off-mode note below says so.
        agree = mismatch is not None and mismatch < self.SPLIT_NULL_AGREE
        if (stall is not None and agree and stall[0] > self.SPLIT_STALL_LIMIT
                and term != "max_iter"):
            neg = (f"; along {stall[2]} more the quantum model has no minimum at all"
                   if stall[2] else "")
            return (f"the split fit stopped {stall[0]:.2g} posterior widths ({1e3 * stall[1]:.2g} mA "
                    f"on its largest atom) short of the minimum of its null-space energy{neg}: "
                    f"the split step leaves directions the data still weigh to the quantum "
                    f"surface, its acceptance test refuses the moves there, and the fit stops "
                    f"where they are refused, so its answer depends on the step schedule "
                    f"(formyl fluoride, HF/6-31G: up to 1.5 widths between trust radii) and "
                    f"these widths describe neither this answer nor its spread")
        off = self.split_off_mode()
        if off is not None:
            return off
        d = float(self.distance_to_mode())
        if term == "max_iter" and d > self.SPLIT_MODE_LIMIT:
            return (f"the split fit stopped at max_iter {d:.2g} posterior widths from its "
                    f"posterior's mode: it has not converged")
        if coupling > self.SPLIT_COUPLING_TOL:
            return (f"the split step couples the data's range and null space (null response "
                    f"{coupling:.2g}), which its posterior leaves out; whether that matters "
                    f"here only an error-model bootstrap of this fit measures (one-constant "
                    f"water, 0.46: the angle's spread 0.04x the posterior's; C1 water from "
                    f"an asymmetric start, 0.058: within 1%)")
        return None

    #: With corrections built from the Hessian (the B0 -> Be targets, the
    #: distortion prediction) the targets move with it, so the refresh is
    #: repeated until the answer moves less than this (RMS): on the water
    #: config 0.678, then 0.050, then 0.0036 mA (2.8x inside; 1.75x with B
    #: and C only), so three passes settle it -- and if they do not, the
    #: result says so (``settled``).
    REFRESH_SETTLED_ANG = 1e-5

    def refresh_hessian_at_answer(self, max_passes=3):
        """Recompute the quantum Hessian at the converged geometry, and
        re-converge on it.

        The fit refreshes its quantum gradient whenever it drifts
        orca_update_thresh, but the Hessian only on its own schedule, so the
        curvature behind the quantum prior's weight and the reported widths
        is usually the one from the start of the fit. Measured at HF/6-31G
        on nine molecules: 4-20 mA from the answer; recomputing it here moved
        alpha_q by -14..+24% and the widths by up to 7%, the geometry by
        under 0.8 mA, and made a restart from the answer stationary (about
        0.001 mA against 0.13-0.26 without). It costs one Hessian, so the
        plain fit does not do it; `quantize uncertainty` does, before any
        width. Measured again through `cli run` (HF/6-31G, formyl fluoride
        and chlorofluoromethane, from their RHF minima and 0.03 A-displaced
        starts): the plain answer is 0.03-0.6 mA rms (1.4 max) from where
        one pass lands, which is the fit's own optimum (a restart chain
        agrees to 1.4e-6 A); the bond RMS moves up to 0.5 mA; the pass adds
        15-21% to the run. Left out of the plain fit for that cost
        (decisions D-63).

        The first pass always runs. Another runs if re-converging carried the
        geometry past orca_update_thresh from the new Hessian, or -- where
        corrections are built from the Hessian and so move with it -- more
        than REFRESH_SETTLED_ANG; at most max_passes. It re-converges with the
        optimisation loop alone: run() would apply the chi-square rescale
        again (x4 per call on water with chi2_rescale on). ORCA's VPT2
        corrections (use_orca_rovib) keep their own schedule -- they cost far
        more than a Hessian.

        Where the corrections follow the Hessian, each pass also rebuilds
        their sigmas from the configured isotopologues (water config: B's
        9470 -> 8039 MHz), so the widths move with them; with a fixed
        correction table the sigmas are left exactly as they were.

        Returns {"passes", "age_before_ang", "age_after_ang", "moved_ang",
        "last_move_ang", "termination", "settled"} (RMS distances), or None
        without a quantum model. ``settled`` is False when the passes ran out
        with the answer still moving (past orca_update_thresh from its
        Hessian, or past REFRESH_SETTLED_ANG where the corrections follow the
        Hessian) or the last pass stopped at max_iter.
        """
        if self.spectral_only or self.quantum is None or self._backend is None:
            return None
        age = self.hessian_distance()
        start = np.asarray(self.coords, dtype=float).copy()
        follows = bool(self._harmonic_from_hessian or self._harmonic_cd_from_hessian
                       or (self._fit_cd_constants and self._cd_weight > 0.0))
        passes, last = 0, None
        while passes < int(max_passes):
            if passes:
                drift = float(self.hessian_distance() or 0.0)
                if drift <= float(self.orca_update_thresh) and (
                        not follows or (last is not None and last <= self.REFRESH_SETTLED_ANG)):
                    break
            before = np.asarray(self.coords, dtype=float).copy()
            self._force_hessian_refresh()
            self._update_orca()          # a full update: Hessian, and what follows it
            self._optimise_once()
            last = float(np.sqrt(np.mean((np.asarray(self.coords, dtype=float) - before) ** 2)))
            passes += 1
        moved = float(np.sqrt(np.mean((np.asarray(self.coords, dtype=float) - start) ** 2)))
        after = self.hessian_distance()
        ended = (getattr(self, "termination", None) or {}).get("reason")
        settled = bool(ended != "max_iter" and float(after or 0.0) <= float(self.orca_update_thresh)
                       and (not follows or (last is not None and last <= self.REFRESH_SETTLED_ANG)))
        return {"passes": passes, "age_before_ang": age, "age_after_ang": after,
                "moved_ang": moved, "last_move_ang": last, "termination": ended,
                "settled": settled}

    def hessian_distance(self, coords=None):
        """RMS distance (Angstrom, the _orca_drift metric) from ``coords`` to
        where the quantum Hessian was computed, or None without one.

        The adaptive schedule refreshes the gradient far more often than the
        Hessian, and the reported widths, the quantum prior's weight and the
        bootstrap's quadratic all take their curvature from the Hessian:
        formyl fluoride's benchmark fit ends 5.1 mA from its only Hessian
        (computed at the start), the water config 10.8 mA, and a fresh
        Hessian at the answer moves their widths by up to 3%.

        With a target-centred prior the curvature belongs at the target, not
        at the answer (_run_hessian), so this is the distance, after
        alignment, from the target to where the Hessian was computed --
        whatever ``coords`` is.
        """
        ref = getattr(self, "_hessian_ref_coords", None)
        if ref is None or self.spectral_only or self.quantum is None:
            return None
        if self._target_prior_active():
            t = np.asarray(self._prior_target_coords, dtype=float)
            return float(np.sqrt(np.mean((_kabsch_align(ref, t) - t) ** 2)))
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        return float(np.sqrt(np.mean((np.asarray(x, dtype=float) - ref) ** 2)))

    def _cartesian_fallback_step(self, J_cart, residual_w, g, H, B, ic_prior_rows,
                                 coord_set, B_active, alpha_q):
        """The internal step's objective, stepped in Cartesians; None if refused.

        An internal step is a step in bonds and angles, carried to Cartesians
        by micro-iterations. Near 180 deg a valence angle stops describing the
        bend: it is not differentiable at 180 and one angle stands for two
        bend directions, so the step can ask for what no nearby geometry has,
        the back-transform realises something else (fluoroacetylene, HF/6-31G:
        error 1.6e-3 against 7e-5 for an accepted step), and that step raises
        the objective and is refused -- unchanged next time, since it lies
        inside the trust radius. Fluoroacetylene, whose optimum lies within
        0.05 deg of linear (179.98 and 179.96 deg), stopped so at 179.995 deg,
        0.008 posterior widths short, and its bootstrap kept about half its
        refits (T-43, D-105; the mode located by the T-45 verifier). Joint
        objective only: the split step is not tried.

        Same objective: the spectral rows (rigid part removed, as in
        Cartesian mode), the q-space prior rows carried through Wilson B
        (exact for their gradient: r = r_ref - J_q dq, dq = B dx), and the
        Cartesian quantum terms; accepted on the same ratio test. Once one is
        accepted the rest of the pass steps in Cartesians: alternating let
        the internal steps undo the fallback's, and acetyl fluoride (whose
        methyl torsion bonds and angles cannot turn) ended 0.40 widths from
        its optimum against 0.19 without; continuing reaches 2.8e-6.
        "Could not realise": an error over a tenth of the step and over 1e-6
        -- without the floor, a converged fit's zero-length steps passed on
        round-off and switched (formyl fluoride, chlorofluoromethane).

        Measured on nine molecules (HF/6-31G, joint, benchmark starts):
        fluoroacetylene reaches its optimum (1.9e-6 widths; its bootstrap
        keeps 50 of 50 refits, from 27; atoms move up to 2.6e-4 A); acetyl
        fluoride's base fit too (2.8e-6, bond RMS -0.18 mA, one more Hessian:
        13 -> 24 s) but only 9 of its 50 refits; the other seven are
        bit-identical. A free torsion is not this: bonds and angles cannot
        express it at all, its steps are realised (fluoroethane never
        triggers), and use_dihedrals: auto is its remedy (D-100). Opt-in
        (ic_cartesian_fallback, internal_coordinates.cartesian_fallback),
        because it moves acetyl fluoride's answer and costs it a Hessian
        (D-106).
        """
        Jx = self._rigid_free_jacobian(np.asarray(J_cart, dtype=float))
        if ic_prior_rows is not None:
            Jp, _r_ref, _q_ref, wp = ic_prior_rows
            Jx = np.vstack([Jx, wp * (np.asarray(Jp, dtype=float) @ np.asarray(B_active, dtype=float))])
        if Jx.shape[0] != np.asarray(residual_w).size:
            return None
        gx = None if g is None else self._prior_gradient(g, H)
        # The internal step's own quantum weight: the legacy rule (no
        # quantum_prior_sigma_ang) scales it by the Jacobian's rank fraction,
        # which differs between q-space and Cartesians (2.667 against 2.2,
        # T-45 verifier), so recomputing it would fit another objective.
        opt = self.optimizer
        dx = np.asarray(opt._joint_step(Jx, residual_w, gx, H, float(alpha_q),
                                        prior_displacement=self._prior_displacement()),
                        dtype=float).ravel()
        if opt.use_internal_preconditioner:
            dx = opt._apply_internal_preconditioner(dx, B)
        norm = float(np.linalg.norm(dx))
        if norm > opt.trust_radius:
            dx = dx * (opt.trust_radius / norm)
        if self.symmetry is not None:
            dx = self.symmetry.project_step(dx)
        trial = self.coords + dx.reshape(-1, 3)
        valid, stats = self._geometry_validity(trial)
        _, r1 = self.spectral.stacked(trial, want_jacobian=self.spectral._cd_rows_active())
        prior_wrms = None
        if ic_prior_rows is not None:
            rp = self._ic_prior_trial_rows(ic_prior_rows, coord_set, trial)
            prior_wrms = float(np.sqrt(np.mean(rp ** 2))) if rp.size else 0.0
            r1 = np.concatenate([r1, ic_prior_rows[3] * rp])
        ok, rho = self._joint_step_ratio(residual_w, r1, Jx, dx, gx, H, alpha_q, trial)
        if self.accept_requires_geometry_valid and not valid:
            ok = False
        _, r_mhz = self.spectral.stacked_unweighted(trial, want_jacobian=False)
        model_delta = (None if gx is None
                       else float(np.dot(gx, dx) + 0.5 * dx @ (np.asarray(H, dtype=float) @ dx)))
        # The step's own range/null split, on the Jacobian it was solved with
        # (as the loop does for its step): convergence, the history and the
        # bootstrap's trust-capped test read it.
        _U, _s, Vt, rank = opt.decompose(Jx)
        P_null = opt.null_projector(Vt, rank)
        return dict(accepted=bool(ok), trial=trial, dx=dx, step_ratio=rho,
                    dx_range_norm=float(np.linalg.norm(opt.range_projector(Vt, rank) @ dx)),
                    dx_null_norm=float(np.linalg.norm(P_null @ dx)),
                    g_null_norm=None if (self.spectral_only or gx is None)
                    else float(np.linalg.norm(P_null @ np.asarray(gx, dtype=float))),
                    wrms=float(np.sqrt(np.mean(r1 ** 2))),
                    mhz_rms=float(np.sqrt(np.mean(np.asarray(r_mhz) ** 2))),
                    prior_wrms=prior_wrms, model_delta=model_delta,
                    geometry_valid=valid, guardrail_stats=stats)

    @staticmethod
    def _ic_prior_trial_rows(rows, coord_set, trial_coords):
        """The q-space prior's residuals at a trial geometry, from this step's rows.

        The prior is linear in q, r = r_ref - J (q_trial - q_ref), with the
        dihedral differences wrapped to (-pi, pi] as the rows themselves were
        built: a step across +-180 deg otherwise reads as a 360 deg move away
        from the prior, and every such step was refused (spectral-only
        fluoroethane with dihedrals: 17 of 17 crossing steps, T-41 verifier).
        """
        J, r_ref, q_ref, _w = rows
        dq = np.asarray(coord_set.active_values(trial_coords), dtype=float) - np.asarray(q_ref, dtype=float)
        for k, ic in enumerate(coord_set.active_coords()):
            if ic.kind == "dihedral":
                dq[k] = (dq[k] + np.pi) % (2.0 * np.pi) - np.pi
        return np.asarray(r_ref, dtype=float) - np.asarray(J, dtype=float) @ dq

    def _resolve_ic_dihedrals(self):
        """Add dihedrals when bonds and angles miss a direction the fit can move.

        Internal mode steps in bonds and angles. Those do not reach every
        internal motion: a methyl group's rotation changes no bond and no
        angle, so a step in them cannot turn it. Acetyl fluoride (HF/6-31G)
        reached 14 of its 15 directions and fluoroethane 17 of 18; their fits
        stopped 0.19 and 0.16 posterior widths from their own optimum, 94% and
        99.9% of the remaining step along the missing torsion, and the
        bootstrap kept 0/50 and 4/50 refits. With dihedrals both reach the
        optimum (1.7e-6 and 1.6e-7 widths), and acetyl fluoride agrees with
        the Cartesian fit of the same objective (14.584 vs 14.602 mA bond RMS)
        (T-41, D-99). Opt-in (ic_use_dihedrals="auto"; the default is False),
        because it moves those answers -- fluoroethane's bond RMS by up to
        +0.37 mA, acetyl fluoride's by -0.19 -- and costs acetyl fluoride one
        more Hessian, as its fit now travels the torsion (D-100).

        Decided once, at the start geometry: the count is set by the bonding,
        and a set that changed during a fit would change the objective's
        coordinates under its step. Molecules whose bonds and angles already
        reach every direction -- every one without a free torsion -- keep
        them alone, so their fits are unchanged. Directions the imposed point
        group forbids are not counted (physical_basis leaves them out).
        """
        x = np.asarray(self.coords, dtype=float)
        try:
            basis = np.asarray(self.physical_basis(coords=x), dtype=float)
            B0 = np.asarray(InternalCoordinateSet(x, self.elems, False).active_B_matrix(x),
                            dtype=float)
            B1 = np.asarray(InternalCoordinateSet(x, self.elems, True).active_B_matrix(x),
                            dtype=float)
        except Exception:  # an unusual geometry keeps the stated set
            return
        if basis.size == 0 or B0.size == 0 or B0.shape[1] != basis.shape[0]:
            return
        # A direction counts as reached only where the step's damped inverse
        # realises at least half of a move along it (s^2 > ic_damping): a
        # direction bonds and angles barely see is as unreachable as one they
        # cannot see. Fluorobenzene's benchmark start (no symmetry) reaches all
        # 30 directions above a relative 1e-6, but its weakest at 7.2e-5 is
        # realised 5%, and the fit stalled 0.056 widths short (T-41 verifier).
        damp = float(self._ic_damping)

        def reach(B):
            sv = np.linalg.svd(B @ basis, compute_uv=False)
            return int((sv * sv > damp).sum())

        n = int(basis.shape[1])
        r0 = reach(B0)
        if r0 >= n:
            return
        # Only when dihedrals reach more: a linear chain's dihedrals are
        # undefined (fluoroacetylene's added row is zero) and a molecule with
        # no four-atom chain has none (formyl fluoride).
        r1 = reach(B1) if B1.size and B1.shape[1] == basis.shape[0] else r0
        if r1 > r0:
            self._ic_use_dihedrals = True
            self._ic_dihedrals_reason = (
                f"bonds and angles reach {r0} of the {n} directions the fit can move, "
                f"with dihedrals {r1} (a torsion has no bond or angle of its own)")
            print(f"  [internal] dihedrals added: {self._ic_dihedrals_reason}; "
                  f"state internal_coordinates.use_dihedrals to choose.")

    def _rigid_mode_projector(self, coords, masses):
        """
        Build Cartesian projector that removes rigid-body translation/rotation modes.

        The modes are built *unweighted*, because the gradient and Hessian this
        projector is applied to are plain Cartesian derivatives in Angstrom, not
        mass-weighted ones. The two differ: the null vectors of a mass-weighted
        Hessian are sqrt(m)-scaled translations, while the null vectors of the
        plain Cartesian Hessian are uniform ones -- displacing every atom by the
        same amount is what leaves the energy unchanged. Weighting by sqrt(m)
        here would project out a subspace that is not the rigid one and would
        leave genuine rigid contamination behind.

        `masses` is still used, but only to place the origin at the centre of
        mass, which is where the rotation generators belong.
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

        modes = []
        # Translations
        for axis in range(3):
            v = np.zeros((n, 3), dtype=float)
            v[:, axis] = 1.0
            modes.append(v.reshape(-1))
        # Rotations about x/y/z: omega x r
        axes = np.eye(3)
        for omega in axes:
            v = np.cross(np.tile(omega, (n, 1)), rel)
            modes.append(v.reshape(-1))

        if not modes:
            return np.eye(3 * n)
        M = np.column_stack(modes)  # (3N, <=6)
        # Rank has to come from the singular values, not from column norms after
        # a QR: reduced QR returns orthonormal -- hence unit-norm -- columns even
        # where the input was rank deficient, so a norm test never rejects
        # anything. A linear molecule has five rigid modes, not six (rotation
        # about the molecular axis moves no atom), and the sixth column of Q is
        # then an arbitrary unit vector spanning a genuine vibration. Projecting
        # it out would delete a real degree of freedom from the quantum prior.
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        if s.size == 0 or s[0] <= 0.0:
            return np.eye(3 * n)
        keep = int((s > 1e-8 * s[0]).sum())
        if keep == 0:
            return np.eye(3 * n)
        Qk = U[:, :keep]
        return np.eye(3 * n) - Qk @ Qk.T

    def _rigid_free_jacobian(self, J):
        """The Cartesian Jacobian with its rigid-body part removed.

        No row the step solves with depends on where the molecule sits or how
        it is turned (but see the conformer mixture below) -- rotational
        constants, distortion constants predicted
        in the Hessian's own frame, internal-coordinate priors -- so the true
        Jacobian satisfies J = J P exactly and this changes nothing on the
        analytic path (1e-15 of |J|, eight benchmark molecules). The finite-
        difference path does not satisfy it: it displaces one atom at a time,
        and near degenerate moments (B ~ C, where the analytic derivative
        gives way to central differences) fluoroacetylene's B rows leaked
        1.5e-5 of |J| into translation and rotation. The joint matrix holds
        those directions only by its 1e-8 eigenvalue floor and lambda_damp,
        so that leak asked for a 3.75 A slide of the whole molecule, the
        trust radius was spent on it, and the fit (HF/6-31G, Cartesian)
        crawled for 57 iterations on 6e-3 A steps 95% rigid, stopping 2.6 mA
        from its own optimum. With the projection it converges in 19, as it
        does with the analytic Jacobian (T-38, D-94). The distortion rows'
        finite-difference Jacobian leaks the same way (water's, at its fit:
        6e-6 of |J|).

        Under the same flag as the quantum terms' projection: one switch
        decides whether the step sees rigid motion at all.

        Not with a conformer mixture. Its conformers are the base geometry
        plus offsets fixed in the lab frame, so its rows do depend on the
        base's orientation, and turning the base is a real move of that
        model: projected, a water mixture (second conformer turned 0.3 rad,
        spectral only) stopped at chi-square 718 instead of 58, on a
        rotational gradient of 664 against 1.4e-4 in the internal directions
        (T-38 verifier).
        """
        if (not self.project_rigid_modes) or (self._rigid_ref_masses is None):
            return J
        if getattr(self.spectral, "conformer_mixture", None) is not None:
            return J
        J = np.asarray(J, dtype=float)
        if J.ndim != 2 or J.shape[1] != 3 * len(self.coords):
            return J
        return J @ self._rigid_mode_projector(self.coords, self._rigid_ref_masses)

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

    def _prior_displacement(self):
        """``x - x_target``, Kabsch-aligned, flattened -- or None.

        The isotropic prior needs the displacement itself, not the
        curvature-weighted gradient of it, so it is computed here rather than
        recovered from ``_prior_gradient``'s output (which would require
        inverting H and is singular in the six rigid directions).
        """
        target = self._prior_target_coords
        if (target is None or self.coordinate_mode == "internal"
                or self.optimizer.isotropic_prior_sigma_ang is None):
            return None
        x = np.asarray(self.coords, dtype=float)
        if target.shape != x.shape:
            return None
        return (x - _kabsch_align(target, x)).ravel()

    def _expanded_quantum_gradient(self, gradient, hessian):
        """The quantum gradient at the current geometry, to first order.

        The stored gradient belongs to the geometry of the last quantum update
        (``_orca_ref_coords``), and updates only happen once the fit has
        drifted ``orca_update_thresh`` (5 mA RMS) away from it. Between them
        the step used that stale gradient as if it were current, so the
        quantum force did not change as the fit moved -- and a fit that
        converged under a stale force is not at the optimum of its own
        objective. Measured in internal mode from the published start: a
        fresh restart from formyl fluoride's "converged" answer moved it 5.83
        mA, and vinyl fluoride cycled for 80 iterations without converging.

            g(x) ~= g(x_ref) + H (x - x_ref)

        is the same quadratic model of the surface the step already solves
        with, applied to the gradient too. It costs one matrix-vector product
        and no quantum chemistry. Rigid motion between the two geometries is
        harmless: H acting on an infinitesimal rotation rotates the gradient
        with the molecule, and translations are null in H.
        """
        if gradient is None:
            return gradient
        shift = self._rto_shift(np.asarray(gradient).size)
        ref = self._orca_ref_coords
        if ref is None or hessian is None:
            return np.asarray(gradient, dtype=float) - shift
        x = np.asarray(self.coords, dtype=float)
        ref = np.asarray(ref, dtype=float)
        g = np.asarray(gradient, dtype=float)
        H = np.asarray(hessian, dtype=float)
        if ref.shape != x.shape or H.shape != (g.size, g.size):
            return g - shift
        return g + H @ (x - ref).ravel() - shift

    #: Randomise-then-optimise: a fixed vector subtracted from the quantum
    #: prior's gradient, which moves the prior's centre by H^-1 xi. With
    #: xi ~ N(0, H / alpha_q) and the constants drawn from their error model,
    #: each refit's optimum is a draw from the posterior (exactly, for a
    #: linear-Gaussian model). Set by the bootstrap's "rto" strategy on its
    #: own copy of the optimizer; None everywhere else.
    _rto_quantum_shift = None

    def _rto_shift(self, n):
        xi = self._rto_quantum_shift
        if xi is None:
            return np.zeros(int(n))
        xi = np.asarray(xi, dtype=float).ravel()
        return xi if xi.size == int(n) else np.zeros(int(n))

    #: Smallest ratio of actual to predicted decrease of the joint objective at
    #: which a step is accepted: the conventional trust-region value, which
    #: takes any step that delivers some of what the model promised.
    _JOINT_RATIO_ETA = 1e-4

    def _joint_step_ratio(self, r0, r1, J, dp, g, H, alpha_q, trial_coords):
        """Accept a joint step when the objective it minimises actually fell.

        The joint step is the damped Gauss-Newton step of

            Phi = 1/2 |r|^2 + alpha_q E_q  [+ 1/2 |x - x_t|^2 / sigma_iso^2]

        with r the weighted spectral residuals plus any internal-prior rows.
        Accepting on |r| alone -- as this did -- does not test that step. At
        the optimum J^T r = alpha_q g, which is not zero, so on the
        low-residual side every step that lowers Phi raises |r| to first
        order and was rejected; the trust radius halved toward its floor and
        the plateau test called that convergence. Conversely a step that
        raised Phi was accepted whenever it lowered |r|. Measured on the
        benchmark's Cartesian recipe started at the theory optimum: one step
        accepted, then every one after it refused (formyl fluoride 1 of 11,
        chlorofluoromethane 2 of 18), and stalls of up to 33 mA / 6.5 deg
        from a start near the spectral optimum.

        rho = [Phi(x) - Phi(x + dp)] / [m(0) - m(dp)], with m the step's own
        quadratic model. Accepted when the model promised a decrease and
        rho > eta, or, where it promised none, when Phi fell anyway.

        No quantum chemistry is spent. With a prior target the quantum prior
        IS the quadratic 1/2 d^T H d (d the displacement from the
        Kabsch-aligned target), so its change is evaluated exactly. Otherwise
        the change comes from the quadratic model the step solved, with the
        gradient re-expanded at the current geometry
        (_expanded_quantum_gradient); without that re-expansion the model
        would describe the surface where it was last computed.

        ``dp`` must be the step actually taken. Returns (accept, rho).
        """
        r0 = np.asarray(r0, dtype=float).ravel()
        r1 = np.asarray(r1, dtype=float).ravel()
        dp = np.asarray(dp, dtype=float).ravel()
        if r0.size != r1.size:
            # Both sides must span the same rows; if they somehow do not, a
            # comparison of objectives is meaningless, so fall back to the
            # residual test rather than invent one.
            self._warn_once(
                "joint step test: the current and trial residuals have "
                f"{r0.size} and {r1.size} rows; falling back to the residual "
                "test for this step.", key="joint_ratio_row_mismatch")
            return bool(float(r1 @ r1) <= float(r0 @ r0)), None
        Jdp = np.asarray(J, dtype=float) @ dp
        actual = 0.5 * float(r0 @ r0) - 0.5 * float(r1 @ r1)
        pred = float(r0 @ Jdp) - 0.5 * float(Jdp @ Jdp)
        target = self._prior_target_coords
        x0 = np.asarray(self.coords, dtype=float)
        x1 = np.asarray(trial_coords, dtype=float)
        exact_target = (target is not None and self.coordinate_mode == "cartesian"
                        and np.shape(target) == x0.shape)
        if not self.spectral_only and alpha_q:
            Hq = np.asarray(H, dtype=float)
            dq_model = float(np.asarray(g, dtype=float) @ dp
                             + 0.5 * dp @ (self.optimizer._spd_hessian(Hq) @ dp))
            if exact_target:
                d0 = (x0 - _kabsch_align(target, x0)).ravel()
                d1 = (x1 - _kabsch_align(target, x1)).ravel()
                dq_actual = 0.5 * float(d1 @ Hq @ d1) - 0.5 * float(d0 @ Hq @ d0)
            else:
                dq_actual = dq_model
            actual -= float(alpha_q) * dq_actual
            pred -= float(alpha_q) * dq_model
        d0_iso = self._prior_displacement()
        sig_iso = self.optimizer.isotropic_prior_sigma_ang
        if (sig_iso is not None and float(sig_iso) > 0.0 and d0_iso is not None
                and d0_iso.size == dp.size and exact_target):
            p_iso = 1.0 / (float(sig_iso) ** 2)
            d1_iso = (x1 - _kabsch_align(target, x1)).ravel()
            actual -= 0.5 * p_iso * (float(d1_iso @ d1_iso) - float(d0_iso @ d0_iso))
            pred -= p_iso * (float(d0_iso @ dp) + 0.5 * float(dp @ dp))
        if pred > 0.0:
            rho = actual / pred
            return bool(rho > self._JOINT_RATIO_ETA), float(rho)
        return bool(actual > 0.0), None

    def _prior_gradient(self, gradient, hessian):
        """Gradient of the quantum prior, centred where the prior actually is.

        The joint step solves (J^T J + alpha_q H + lambda I) dp = J^T r -
        alpha_q g. Nothing in that references the geometry the caller handed
        in, so g -- the level of theory's own gradient -- places the prior's
        minimum at the level of theory's own minimum. Handing the optimiser a
        bias-corrected starting structure therefore moves the starting point
        and nothing else, and -alpha_q g pulls straight back to the
        uncorrected geometry. Measured on acetyl fluoride with the spectral
        data suppressed a thousandfold: started at a 6.57 mA corrected
        structure, relaxed to 15.03 mA against pure theory's 15.75.

        When a target is supplied the prior becomes a Gaussian centred on it
        with the quantum surface's own curvature,

            E_prior(x) = 1/2 (x - x_target)^T H (x - x_target)

        whose gradient is H (x - x_target). That is the statement the
        bond-class offsets justify -- the equilibrium structure is near the
        bias-corrected geometry, and H says how sharply -- where the raw
        gradient asserts the stronger and measurably false claim that it is at
        the level of theory's minimum.

        The target is Kabsch-aligned to the current geometry first: the prior
        is about internal structure, and a rigid rotation accumulated by the
        optimiser would otherwise register as a displacement to be penalised.
        Only Cartesian mode is handled; in internal mode the gradient has
        already been projected into q-space, so the substitution is skipped
        rather than applied in the wrong basis.
        """
        target = self._prior_target_coords
        if target is None or self.coordinate_mode == "internal":
            return gradient
        x = np.asarray(self.coords, dtype=float)
        if target.shape != x.shape:
            return gradient
        aligned = _kabsch_align(target, x)
        return np.asarray(hessian, dtype=float) @ (x - aligned).ravel()

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
        # A calibrated quantum weight is a statement about the level of theory
        # (sigma_x, in Angstrom), and alpha_quantum multiplies it. A heuristic
        # that rescales alpha_quantum by problem shape therefore silently
        # re-states how good the theory is -- so with sigma_x set it is left
        # where the caller put it.
        if self.optimizer.quantum_prior_sigma_ang is not None:
            bases = dict(bases, alpha_quantum=self._base_alpha_quantum)
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
        # Not with a calibrated weight: there alpha_quantum multiplies
        # 1 / (1/2 lambda_bar sigma_x^2), so retuning it mid-fit changes the
        # objective under the fit. Measured on formyl fluoride (class
        # defaults, internal mode): alpha_q fell 3438 -> 2216 within one fit,
        # and a fresh restart from its "converged" answer moved 15.5 mA.
        calibrated = self.optimizer.quantum_prior_sigma_ang is not None
        if reject_streak > 0:
            self.optimizer.sv_threshold = max(self.optimizer.sv_threshold, target_sv)
            if not calibrated:
                self.optimizer.alpha_quantum = min(self.optimizer.alpha_quantum, target_alpha)
        else:
            self.optimizer.sv_threshold = target_sv
            if not calibrated:
                self.optimizer.alpha_quantum = target_alpha
        return controls

    # ── Optimisation loop ─────────────────────────────────────────────────────

    def _chi2_dof(self):
        """Effective degrees of freedom of the regularised fit.

        nu = n_obs - tr(hat matrix), with the hat built from the same
        regularised normal matrix the fit solves: tr[(J^T J + alpha_q H)^-1
        J^T J]. Counting a raw SVD rank instead treats every data-touched
        direction as fully spent, which both overstates nu where the prior
        shares a direction and understates it where the data barely grazes one.
        For an interpolating fit the trace approaches n_obs, nu approaches 0,
        and chi-square rescaling is correctly skipped rather than clamped
        mid-correction.
        """
        J, rw = self.spectral.stacked(self.coords)
        n_obs = int(rw.size)
        if n_obs == 0:
            return 0.0
        # The Jacobian the step solves with (see _rigid_free_jacobian): a
        # finite-difference leak into rigid motion would otherwise count as
        # data-touched directions (fluoroacetylene: nu 4.8936 against 4.8958).
        J = self._rigid_free_jacobian(np.asarray(J, dtype=float))
        JTJ = J.T @ J
        if self.spectral_only or self.quantum is None:
            sv = np.linalg.svd(J, compute_uv=False)
            rank = int((sv > 1e-8 * sv[0]).sum()) if sv.size and sv[0] > 0 else 0
            return float(max(0, n_obs - rank))
        g = self.quantum.gradient
        H = self.quantum.hessian
        g, H = self._project_quantum_terms(g, H)
        rank = int(np.linalg.matrix_rank(J))
        alpha_q = self.optimizer.effective_quantum_weight(J, rank, hessian=H)
        H_total = JTJ + alpha_q * self.optimizer._spd_hessian(H)
        try:
            C = np.linalg.pinv(H_total)
        except np.linalg.LinAlgError:
            return float(max(0, n_obs - rank))
        tr_hat = float(np.trace(C @ JTJ))
        return float(max(0.0, n_obs - min(tr_hat, float(n_obs))))

    def reduced_chi_square(self):
        """Chi-square per effective degree of freedom at the current geometry.

        nu comes from the hat-matrix trace (see _chi2_dof), so the prior's
        share of each direction is counted fractionally. When nu_eff is
        numerically zero -- an interpolating fit -- the rank-based count is
        reported instead so the caller sees a finite diagnostic, but the
        rescaling gate in run() skips such fits entirely.
        """
        _, rw = self.spectral.stacked(self.coords)
        n_obs = int(rw.size)
        if n_obs == 0:
            self._last_chi2_dof = 0.0
            return 0.0
        nu_eff = self._chi2_dof()
        # Kept for run(): its dof gate and its check record read this instead
        # of computing the hat-matrix trace a second time at the same point.
        self._last_chi2_dof = float(nu_eff)
        if nu_eff > 1e-9:
            return float(np.dot(rw, rw)) / nu_eff
        try:
            J, _ = self.spectral.stacked(self.coords)
            s = np.linalg.svd(J, compute_uv=False)
            rank = int(np.sum(s > 1e-5 * s[0])) if s.size and s[0] > 0 else 0
        except np.linalg.LinAlgError:
            rank = 0
        nu = max(n_obs - rank, 1)
        return float(np.sum(rw ** 2) / nu)

    def run(self):
        """
        Run the hybrid optimisation loop, then rescale sigma if the fit says so.

        Off by default. It is a real capability but an unvalidated one: on the
        end-to-end water case it left the structure worse than theory alone
        (14.6 mA against 10.9), so it should be switched on only where it has
        been shown to help. Enable with chi2_rescale=True.

        MEASURED AGAIN, 2026-11-09, because an audit flagged that every
        benchmark passes chi2_rescale=True while this ships False and that
        looked like the same shipped-versus-benchmarked gap that cubic_scheme
        turned out to be. It is not; the default here is deliberate.

        Forced to fire -- threshold set to 1e-6, confirmed firing with dof >= 1
        and reduced chi-square of 0.24, 0.032 and 0.0014 -- it moved the fitted
        geometry by less than 0.005 mA on water, ozone and formyl fluoride at
        RHF/6-31G. It rescales the SPECTRAL sigmas only (spectral.scale_sigma),
        leaving the quantum prior width alone, so it can genuinely shift the
        data-versus-prior balance and on these three it simply does not change
        where the fit lands.

        So the benchmarks have been enabling a switch that does nothing on the
        molecules they run, and the one case where it is known to do something
        it makes the structure worse. Leaving it off is the supported choice,
        not an oversight.

        A converged fit whose reduced chi-square is far above one is telling you
        the stated uncertainties were optimistic: the model cannot represent the
        data to the precision claimed. Continuing to weight the data at that
        precision makes the fit chase residuals it cannot remove, and it removes
        them by distorting the structure. Widening sigma by the Birge ratio,
        sqrt(chi2/nu), and refitting is the standard response.

        Returns
        -------
        coords : (N, 3) ndarray   Final optimised coordinates in Angstroms.
        """
        self.chi2_rescale_checks = []
        coords = self._optimise_once()
        if not self.chi2_rescale:
            return coords
        # Two-sided. The test used to be `chi2_nu > threshold`, which can only
        # catch sigmas that are too small. Sigmas that are too *large* are just
        # as wrong and are the failure this benchmark actually shows: measured
        # chi-square per degree of freedom came out at 0.05, meaning the fit was
        # under-using its data by a factor of about four and reporting intervals
        # several times too wide -- and the one-sided test could never fire.
        lo = 1.0 / float(self.chi2_rescale_threshold)
        for _pass in range(int(self.chi2_rescale_max_passes)):
            self._last_chi2_dof = None
            chi2_nu = self.reduced_chi_square()
            # None when reduced_chi_square was replaced by something that
            # does not keep it; then the gate below computes it, at the same
            # point in the pass as it always did.
            dof = self._last_chi2_dof
            check = {"chi2_nu": float(chi2_nu), "dof": dof, "outcome": None}
            self.chi2_rescale_checks.append(check)
            if not np.isfinite(chi2_nu) or chi2_nu <= 0.0:
                check["outcome"] = "no finite chi-square"
                break
            if lo <= chi2_nu <= self.chi2_rescale_threshold:
                check["outcome"] = "in band"
                break
            # A near-zero chi-square is not evidence that the sigmas are
            # enormously too large; it is what a model with enough freedom does
            # to a handful of observations. Rescaling on it drives sigma toward
            # zero, and anything that divides by sigma then blows up -- measured
            # on fluoroacetylene, whose six B-only observations gave chi2/nu of
            # 0.00 and a reported bond uncertainty of 1e13 Angstrom.
            #
            # So the adjustment is clamped, and only made when there are enough
            # degrees of freedom for the ratio to mean anything.
            if dof is None:
                dof = check["dof"] = float(self._chi2_dof())
            if dof < CHI2_RESCALE_MIN_DOF:
                check["outcome"] = "too few dof"
                break
            factor = float(np.clip(np.sqrt(chi2_nu),
                                   1.0 / CHI2_RESCALE_MAX_FACTOR,
                                   CHI2_RESCALE_MAX_FACTOR))
            direction = "optimistic" if chi2_nu > 1.0 else "conservative"
            verb = "widening" if chi2_nu > 1.0 else "tightening"
            # A downward scale stops at each row's model-error floor (see
            # SpectralEngine.scale_sigma). Once every row is there it changes
            # nothing, and refitting the same objective is a no-op that used to
            # run, and say "refitting", up to chi2_rescale_max_passes times.
            # Measured 2026-11-19 on water, ozone, CH2ClF and vinyl fluoride:
            # passes after the first moved the fit by under 2e-7 A.
            before = [np.array(iso["sigma_constants"], dtype=float)
                      for iso in self.spectral.isotopologues]
            self.spectral.scale_sigma(factor)
            after = [np.asarray(iso["sigma_constants"], dtype=float)
                     for iso in self.spectral.isotopologues]
            if all(np.allclose(a, b, rtol=1e-12, atol=0.0) for a, b in zip(after, before)):
                print(
                    f"\n  [chi2-rescale] reduced chi-square = {chi2_nu:.3g}, but every "
                    f"sigma is already at its model-error floor, which a small "
                    f"chi-square cannot lower; not refitting."
                )
                check["outcome"] = "at floor"
                break
            # The stated sigmas moved, so this refits even when every weighting
            # sigma, max(stated, 0.1% of the constant, ...), is unchanged.
            # Skipping that refit was tried and reverted (2026-11-22): it is a
            # no-op only after a fit that stopped at a stationary point. After
            # a plateau or max_iter stop it keeps optimising -- water at 5 MHz
            # stated from a shifted start, chi2/nu 201.96614 -> 201.96841 ->
            # 201.96874 over two such refits -- so a skip trades convergence for
            # time. No real fit measured reaches it: 0 of 86 script fits and 0
            # of 35 benchmark fits refit on unchanged weights.
            check["outcome"] = "refit"
            check["factor"] = factor
            print(
                f"\n  [chi2-rescale] reduced chi-square = {chi2_nu:.3g}, outside "
                f"[{lo:.3g}, {self.chi2_rescale_threshold:g}].\n"
                f"                 The stated sigmas are {direction} by about "
                f"{max(factor, 1 / factor):.1f}x; {verb} them and refitting."
            )
            coords = self._optimise_once()
        return coords

    def _optimise_once(self):
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
        self._ic_switched = False     # every pass starts in its own coordinates
        if self.debug_rank_diagnostics:
            labels = ["A", "B", "C"]
            print("[rank-debug] Active spectral components in optimizer:")
            for i, iso in enumerate(self.spectral.isotopologues, start=1):
                idx = np.asarray(iso.get("component_indices", []), dtype=int)
                comps = [labels[c] if 0 <= int(c) < 3 else f"R{int(c)}" for c in idx]
                print(f"[rank-debug]   iso {i}: {comps}")

        # #5: Per-component observability — how many isotopologues constrain each constant
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

        # EVERY PASS STARTS FROM THE BASE TRUST REGION. adapt_lambda only ever
        # shrinks the radius (floor 1e-4 Angstrom) and nothing grew it back, so
        # a second pass on the same optimizer -- a chi2-rescale refit, or a
        # bootstrap refit of a deep copy -- inherited the radius the first
        # pass had ended on and could barely move. Measured on the benchmark
        # configuration (autoconfig off): 1.95e-4 Angstrom inherited, and
        # bootstrap clouds 30-50% too narrow (HCOF C-H 5.43 against 7.41 mA).
        # With autoconfig on, _apply_autoconfig re-seeds these each iteration
        # anyway, so this changes nothing there.
        self.optimizer.trust_radius = self._base_trust_radius
        self.optimizer.null_trust_radius = self._base_null_trust_radius
        self.optimizer.lambda_damp = self._base_lambda_damp
        self.optimizer.alpha_quantum = self._base_alpha_quantum

        converged = False
        # Why the loop ended, for callers that must tell a converged fit from
        # one that ran out of iterations (the bootstrap's sample gate).
        self.termination = {"reason": "max_iter", "iterations": 0}
        prev_energy = None
        prev_freq_rms = None
        _plateau_window = 10
        _plateau_count  = 0
        _hybrid_stall_count = 0
        _reject_streak = 0
        for it in range(self.max_iter):

            # Carries the iteration's internal-prior rows from the point where
            # the step is computed to the point where the trial geometry is
            # judged, so both sides of the acceptance test span the same rows.
            # Declared here, unconditionally: setting it inside the
            # internal-mode branch leaves it unbound on every other path.
            _ic_prior_rows = None

            if self.spectral_only:
                n_dof = 3 * len(self.elems)
                g = np.zeros(n_dof)
                H = np.eye(n_dof)
            else:
                if self._orca_drift() > self.orca_update_thresh:
                    self._update_orca()
                g = self.quantum.gradient
                H = self.quantum.hessian
                g = self._expanded_quantum_gradient(g, H)
                g, H = self._project_quantum_terms(g, H)

            # Re-fit the mass-dependent B0 -> Be offset at the current geometry.
            # It is linear in its coefficients, so the best ones follow in closed
            # form and substituting them back is equivalent to carrying them as
            # optimiser parameters (variable projection) -- but the parameter
            # vector, trust radius and symmetry projector stay untouched. The
            # coefficients depend on the structure, so this has to be redone as
            # the structure moves.
            if self._mass_dependent_model is not None:
                self._mass_dependent_c = self.spectral.fit_mass_dependent_correction(
                    self.coords, frac_sigma=self._mass_dependent_frac_sigma)

            J, residual_w = self.spectral.stacked(self.coords)
            # Residuals only: the unweighted Jacobian is never read here, and
            # building it (analytically by default -- an inertia-derivative
            # assembly per isotopologue, finite differences only near
            # degenerate moments) was a quarter to a third of a refit's time,
            # per iteration and again per trial step below.
            _, residual_mhz = self.spectral.stacked_unweighted(self.coords, want_jacobian=False)
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
            if self.coordinate_mode == "cartesian":
                # The step, its rank and the ratio test below all read this J.
                # Internal mode needs nothing: J B+ already excludes rigid motion.
                J = self._rigid_free_jacobian(J)
            B, _ = wilson_B(self.coords, self.elems)

            # ── Internal-coordinate mode: transform J and quantum terms to q-space ──
            _ic_coord_set = None
            _ic_Bplus = None
            _ic_g = g
            _ic_H = H
            _J_cart, _g_cart, _H_cart = J, g, H     # for _cartesian_fallback_step
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
                        f"  of {b_diag['n_dof']} DOF  κ(B)={kappa_b_str}"
                    )
                    if b_diag["kappa_B"] is not None and b_diag["kappa_B"] > 1e4:
                        print(
                            f"  [warn] B-matrix ill-conditioned κ(B)={b_diag['kappa_B']:.2e}; "
                            "some internal coordinates may be linearly dependent — "
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
                    _J_prior, _r_prior, _sig_prior = build_internal_priors(
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
                        configured_kinds=self._ic_prior_configured_kinds,
                    )
                    _wp = float(np.sqrt(self._ic_prior_weight))
                    J = np.vstack([J, _wp * _J_prior])
                    residual_w = np.concatenate([residual_w, _wp * _r_prior])
                    # Keep this iteration's prior block for the acceptance test.
                    # Its sigmas may be adaptive, so rebuilding it at the trial
                    # geometry could change the objective between the step and
                    # the judgement of the step. The prior is exactly linear in
                    # q -- r_prior = (q0 - q)/sigma with J_prior = diag(1/sigma)
                    # -- so the same prior can be re-evaluated anywhere from
                    # these three pieces with no approximation at all.
                    _ic_prior_rows = (
                        _J_prior,
                        _r_prior,
                        np.asarray(_ic_coord_set.active_values(self.coords), dtype=float),
                        _wp,
                    )
                    # The prior exactly as this iteration used it -- widths
                    # possibly adaptive -- for the posterior to read back.
                    # r = (q0 - q) / sigma, so the centre is q + sigma * r.
                    _sig_eff = 1.0 / np.maximum(np.abs(np.diag(np.asarray(_J_prior, dtype=float))), 1e-300)
                    self._last_ic_prior = dict(
                        coord_set=_ic_coord_set,
                        sigma=_sig_eff,
                        q0=_ic_prior_rows[2] + _sig_eff * np.asarray(_r_prior, dtype=float),
                        weight=float(self._ic_prior_weight),
                    )

            wrms_before = float(np.sqrt(np.mean(residual_w ** 2)))
            mhz_rms_before = float(np.sqrt(np.mean(residual_mhz ** 2)))
            _svd_B = None if self.coordinate_mode == "internal" else B
            _ic_g = self._prior_gradient(_ic_g, _ic_H)
            # The cap this step is computed under (it adapts between steps):
            # a step at it says how far the step was allowed to go, not how
            # far the optimum is, which a restart certificate must know.
            _trust_used = float(self.optimizer.trust_radius)
            _null_trust_used = float(self.optimizer.null_trust_radius)
            dp, rank, sv, alpha_q_eff, Vt = self.optimizer.step(
                J, residual_w, _ic_g, _ic_H, B=_svd_B,
                prior_displacement=self._prior_displacement(),
                weight_hessian=(H if self.coordinate_mode == "internal"
                                and not self.spectral_only else None))
            self._last_alpha_q_eff = float(alpha_q_eff)
            if self.optimizer.objective_mode != "joint":
                # What the split step solved with at this geometry, in its own
                # parameters (q-space in internal mode), for split_stall.
                self._last_split_step = dict(
                    coords=np.asarray(self.coords, dtype=float).copy(), Vt=np.asarray(Vt, dtype=float),
                    rank=int(rank), g=np.asarray(_ic_g, dtype=float).ravel(),
                    H=np.asarray(_ic_H, dtype=float),
                    Bplus=None if _ic_Bplus is None else np.asarray(_ic_Bplus, dtype=float),
                    coord_set=_ic_coord_set)

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

            # The step actually taken, in the parameters it was solved in. In
            # internal mode the back-transform realises only the part of dq
            # that a geometry can have -- the redundant angle sum of a planar
            # centre, for one, cannot move -- so the step test below judges the
            # step that was taken, not the one that was asked for.
            if self.coordinate_mode == "internal":
                _dp_realised = (np.asarray(_ic_coord_set.active_values(trial_coords), dtype=float)
                                - np.asarray(_q_curr, dtype=float))
                for _k, _ic in enumerate(_ic_coord_set.active_coords()):
                    if _ic.kind == "dihedral":
                        _dp_realised[_k] = (_dp_realised[_k] + np.pi) % (2.0 * np.pi) - np.pi
            else:
                _dp_realised = np.asarray(dp, dtype=float).ravel()

            P_range = self.optimizer.range_projector(Vt, rank)
            P_null = self.optimizer.null_projector(Vt, rank)
            dx_range_norm = float(np.linalg.norm(P_range @ dp))
            dx_null_norm = float(np.linalg.norm(P_null @ dp))
            g_null_norm = None if self.spectral_only else float(np.linalg.norm(P_null @ _ic_g))
            geometry_valid, guardrail_stats = self._geometry_validity(trial_coords)
            # Residuals only: the trial's Jacobian was built and discarded
            # (30-90 ms per benchmark fit); a step taken computes its own.
            # Not with distortion rows: their FD Jacobian is memoised on the
            # coordinates rounded to 1e-8, so the trial's entry is what the
            # next iteration reads at an accepted (symmetrised) geometry, and
            # skipping it moved CD fits within their stopping tolerance
            # (1.8e-10 A; a multistart spread by 1e-5 relative).
            _, residual_w_trial = self.spectral.stacked(
                trial_coords, want_jacobian=self.spectral._cd_rows_active())
            prior_wrms_after = None
            if self.internal_prior is not None and self.coordinate_mode == "cartesian":
                Jp_trial, rp_trial = self.internal_prior.stacked(trial_coords)
                wp = float(np.sqrt(self.prior_weight))
                residual_w_trial = np.concatenate([residual_w_trial, wp * rp_trial])
                prior_wrms_after = float(np.sqrt(np.mean(rp_trial ** 2))) if rp_trial.size else 0.0
            elif self.internal_prior is not None and self.coordinate_mode == "internal":
                prior_wrms_after = self.internal_prior.diagnostics(trial_coords).get("prior_wrms", 0.0)
            if _ic_prior_rows is not None:
                # THE TWO SIDES OF THE ACCEPTANCE TEST MUST SPAN THE SAME ROWS.
                #
                # They did not. In internal mode wrms_before is the RMS over
                # [spectral; sqrt(w) * prior] -- the vector the step is solved
                # from -- while this branch left residual_w_trial as the
                # spectral rows alone and only computed a prior diagnostic. So
                # `wrms_after <= wrms_before` compared an RMS over m rows with
                # an RMS over m + p rows, and with spectral_accept_relax = 0 a
                # vanishing step does not make the two sides meet: whether a
                # step is accepted came out of the row counts, not the step.
                #
                # Measured on the analytic-water config (3 spectral rows, 3
                # prior rows), at a ZERO step, where both sides describe the
                # identical geometry and a correct test must give exactly
                # equal values: after/before = 1.258, 1.281, 1.251 on three
                # bootstrap samples and 0.941, 0.691 on two others. Above one
                # every step is rejected however small or however good, so the
                # fit stalls at its starting point and reports "converged";
                # below one every step is accepted however bad, so there is no
                # descent control. Five samples, five outcomes, predicted by
                # the ratio alone -- the gate was testing the prior's size
                # against the residual's, which is a property of the geometry.
                #
                # 9 of 24 bootstrap samples stalled this way, preferentially
                # the ones with larger data perturbations (correlation +0.52),
                # which censored the tails the interval is made of.
                #
                # Unconditional, now. The q-space prior rows come from
                # ic_prior_weight, not from an InternalPriorEngine, so the
                # class default (use_internal_priors=False, ic_prior_weight=1)
                # carries them in the step while self.internal_prior is None.
                # The first version of this fix sat under that engine's
                # branch and so left exactly the default path mismatched.
                _wp_acc = _ic_prior_rows[3]
                _rp_trial = self._ic_prior_trial_rows(_ic_prior_rows, _ic_coord_set, trial_coords)
                residual_w_trial = np.concatenate(
                    [residual_w_trial, _wp_acc * _rp_trial])
                prior_wrms_after = (
                    float(np.sqrt(np.mean(_rp_trial ** 2))) if _rp_trial.size else 0.0)
            _, residual_mhz_trial = self.spectral.stacked_unweighted(trial_coords, want_jacobian=False)
            wrms_after = float(np.sqrt(np.mean(residual_w_trial ** 2)))
            mhz_rms_after = float(np.sqrt(np.mean(residual_mhz_trial ** 2)))

            # Autoconfig only after the trial has been weighed. It can move the
            # spectral sigma floor, the weight cap and the prior weight, and it
            # used to run between the step and the trial evaluation -- so the
            # two sides of the acceptance test could be weighted differently,
            # and the test then measured the change of weights as well as the
            # step. Everything it reads (rank, singular values, the current
            # residuals, the rejection streak) is already fixed here, and what
            # it sets takes effect from the next step either way.
            autoconfig_controls = None
            if self.autoconfig is not None and ((it % self.autoconfig_update_every) == 0):
                autoconfig_controls = self._apply_autoconfig(rank, sv, residual_mhz, _reject_streak)

            # Acceptance.
            #
            # Joint objective: the standard trust-region ratio test on the
            # objective the step minimises (see _joint_step_ratio). Split
            # objective: there is no single objective, so the residual test
            # and the optional quantum-descent check stay as they were.
            joint = (self.optimizer.objective_mode == "joint")
            step_ratio = None
            if joint:
                spectral_accept, step_ratio = self._joint_step_ratio(
                    residual_w, residual_w_trial, J, _dp_realised, _ic_g, _ic_H,
                    alpha_q_eff, trial_coords)
            else:
                spectral_accept = wrms_after <= wrms_before * (1.0 + self.spectral_accept_relax)
            model_delta = None
            quantum_accept = True
            quantum_gate_active = False
            quantum_descent_tol_eff = self.quantum_descent_tol
            if not self.spectral_only:
                # In internal mode g/_ic_H are in q-space and dp is the q-step.
                # In Cartesian mode they are the original Cartesian terms.
                model_delta = float(np.dot(_ic_g, dp) + 0.5 * dp @ (_ic_H @ dp))
                if self.enforce_quantum_descent and joint:
                    # Requiring the energy to fall on its own forbids exactly
                    # the trade the joint objective exists to make, and ANDed
                    # with a descent test on the whole objective it freezes a
                    # fit that starts at the prior centre. The ratio test
                    # already requires the joint objective to descend.
                    self._warn_once(
                        "enforce_quantum_descent has no effect with the joint "
                        "objective: its step test already requires the joint "
                        "objective (spectral + quantum) to descend, and a "
                        "separate energy-descent test would forbid the "
                        "data-for-energy trade the objective exists to make.",
                        key="enforce_quantum_descent_joint")
                elif self.enforce_quantum_descent:
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
            _fb_used = False
            if joint and self.coordinate_mode == "internal" and self._ic_cartesian_fallback and (
                    self._ic_switched
                    or (not accepted and _bt_err is not None
                        and _bt_err > max(0.1 * float(np.linalg.norm(dp)), 1e-6))):
                if self._ic_switched:
                    accepted = spectral_accept = False
                _fb = self._cartesian_fallback_step(
                    _J_cart, residual_w, _g_cart, _H_cart, B, _ic_prior_rows,
                    _ic_coord_set, _ic_B_active, alpha_q_eff)
                # Once switched the pass's step IS the Cartesian one, refused
                # or not, so the history describes it; before, only an
                # accepted retake replaces the refused internal step.
                if _fb is not None and (_fb["accepted"] or self._ic_switched):
                    trial_coords, dx = _fb["trial"], _fb["dx"]
                    wrms_after, mhz_rms_after = _fb["wrms"], _fb["mhz_rms"]
                    step_ratio, model_delta = _fb["step_ratio"], _fb["model_delta"]
                    prior_wrms_after = _fb["prior_wrms"]
                    geometry_valid, guardrail_stats = _fb["geometry_valid"], _fb["guardrail_stats"]
                    dx_range_norm, dx_null_norm = _fb["dx_range_norm"], _fb["dx_null_norm"]
                    g_null_norm = _fb["g_null_norm"]
                    _bt_err = None   # no back-transform: the step is Cartesian
                    _fb_used = True
                if _fb is not None and _fb["accepted"]:
                    spectral_accept = geometry_accept = accepted = True
                    self._ic_fallback_steps += 1
                    self._ic_switched = True
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
                print(f"  [warn] κ(J)={kappa_J:.2e} — Jacobian ill-conditioned; "
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
                    trust_radius=_trust_used,
                    null_trust_radius=_null_trust_used,
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
                    step_ratio=step_ratio,
                    backtransform_error=(_bt_err if self.coordinate_mode == "internal" else None),
                    cartesian_fallback=_fb_used,
                    internal_rank=(rank if self.coordinate_mode == "internal" else None),
                    internal_singular_values=(sv.tolist() if self.coordinate_mode == "internal" else None),
                    dq_range_norm=(dx_range_norm if self.coordinate_mode == "internal" and not _fb_used else None),
                    dq_null_norm=(dx_null_norm if self.coordinate_mode == "internal" and not _fb_used else None),
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

            def _fmt_opt(v, spec):
                # None on a Cartesian fallback step, which has no q-space split.
                return format(v, spec) if v is not None else format("n/a", spec[:3])
            stage_suffix = ""
            if autoconfig_controls is not None:
                stage_suffix = f" [{autoconfig_controls['stage']}]"
            print(
                f"{it+1:>5}  {step_norm:>12.4e}  {wrms:>12.4f}  {freq_rms:>12.4f}  "
                f"{rank:>6d}  {sv_kept:>12.4e}  κ(J)={kappa_str}  "
                f"{_fmt_opt(dx_range_norm, '>10.3e')}  {_fmt_opt(dx_null_norm, '>10.3e')}  "
                f"{gnull_str}  {alpha_q_eff:>8.3f}  {dE_str}  "
                f"lambda={self.optimizer.lambda_damp:.2e} {status}{stage_suffix}"
            )

            # If we keep rejecting, force a fresh Hessian sooner to recover local model quality.
            if _reject_streak >= 5:
                self._force_hessian_refresh()

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
                self.termination = {"reason": "converged", "iterations": it + 1}
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
                    self.termination = {"reason": "stall", "iterations": it + 1}
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
                    self.termination = {"reason": "plateau", "iterations": it + 1}
                    break

        if not converged:
            print(f"\nWarning: did not converge within {self.max_iter} iterations.")
            self.termination = {"reason": "max_iter", "iterations": len(self.history)}

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
        param_std, _ = std_err_from_covariance(
            cov, context="parameter report (pinv of J^T J + reg)")

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

        kr = self.kraitchman_report()
        if kr and kr.get("rows"):
            print()
            print("  Kraitchman substitution coordinates vs fitted geometry (Ang)")
            print(f"  {'species':>16}{'atom':>6}{'|a| sub':>10}{'|a| fit':>10}"
                  f"{'d(a)':>9}{'|b| sub':>10}{'|b| fit':>10}{'d(b)':>9}")
            print("  " + "-" * 80)
            for r in kr["rows"]:
                print(f"  {r['name'][:16]:>16}"
                      f"{self.elems[int(r['atom_index'])] + str(int(r['atom_index']) + 1):>6}"
                      f"{r['abs_a_angstrom']:>10.4f}{r.get('fit_abs_a_angstrom', float('nan')):>10.4f}"
                      f"{r.get('delta_a_angstrom', float('nan')):>+9.4f}"
                      f"{r['abs_b_angstrom']:>10.4f}{r.get('fit_abs_b_angstrom', float('nan')):>10.4f}"
                      f"{r.get('delta_b_angstrom', float('nan')):>+9.4f}")
            for w in kr.get("warnings", [])[:4]:
                print(f"    note: {w}")

        print("=" * 52)

        return residual

    @staticmethod
    def _coordinate_class(elem_i, elem_j):
        """Bond class for the per-class prior widths.

        The priority (hydrogen, then chlorine, then fluorine, else skeleton)
        matches the classification under which the class widths were measured;
        a coordinate must land in the same bin here that its calibration error
        landed in there, or the floor is calibrated against nothing.
        """
        pair = {str(elem_i).strip().capitalize(), str(elem_j).strip().capitalize()}
        if pair & {"H", "D", "T"}:
            return "X-H"
        if "Cl" in pair:
            return "X-Cl"
        if "F" in pair:
            return "C-F"
        return "skeleton"

    def _warn_once(self, message: str, key: str | None = None) -> None:
        """Print a warning the first time only, per optimizer instance.

        geometry_uncertainty propagates one coordinate at a time, so a single
        bad covariance would otherwise print the same paragraph once per bond
        and angle and bury itself in its own repetition. ``key`` is what makes
        two warnings "the same": the messages carry per-coordinate numbers, so
        keying on the text alone would defeat the purpose.
        """
        seen = getattr(self, "_warned_messages", None)
        if seen is None:
            seen = set()
            self._warned_messages = seen
        k = message if key is None else key
        if k in seen:
            return
        seen.add(k)
        print(f"  [uncertainty] WARNING: {message}", flush=True)

    def geometry_uncertainty(self, check_gauss_newton=True):
        """Posterior standard deviation on each bond length and angle.

        For a molecule that already has an accepted structure, the interesting
        number is the error. For one that does not -- which is the case this
        engine exists for, since fluorine has no second stable isotope and the
        species needed to pin a structure are usually not available -- there is
        nothing to take a difference against, and the uncertainty *is* the
        result. A structure quoted without one cannot be used.

        The posterior covariance comes from the Laplace approximation at the
        optimum, C = (JᵀJ + alpha_q H)⁻¹, weighted exactly as the fit weighted
        them. Bond lengths and angles are then propagated through it in the
        usual way, sigma_q² = gᵀ C g, with g the gradient of that coordinate
        with respect to the Cartesians.

        Two properties of this number are worth stating plainly, because they
        decide how it should be read:

        * It is a *posterior* width, so it includes the prior. In a direction
          the data cannot see, it reports the prior's width rather than
          infinity -- which is honest only to the extent that
          `quantum_prior_sigma_ang` is honest. It is a modelling choice, not a
          measurement. When ``prior_class_sigma`` is supplied (a dict of
          measured per-class theory errors: "X-H", "C-F", "X-Cl", "skeleton"
          in Angstrom and "angle" in degrees), the prior-side width of each
          coordinate is floored at its class error scaled by the fraction of
          the coordinate the data cannot see, replacing the isotropic
          assumption with the measured anisotropy of the level of theory.
        * It includes the model error only where one is supplied
          (sigma_systematic_constants, or auto_systematic_sigma): the
          B0-versus-Be gap left uncorrected, or a vibrational correction's own
          uncertainty where one was applied. That error is propagated as
          shared across isotopologues for each constant type
          (systematic_isotopologue_correlation, default 0.99) and counted once,
          even when the weighting sigma already contains it
          (_systematic_covariance). Without one it is a precision, not an
          accuracy -- expect it to be optimistic, and calibrate it against
          molecules whose structures are known before trusting it on one whose
          structure is not.

        Rigid-body directions, and any the imposed point group forbids, are
        excluded before inversion: translations and rotations are null in
        both blocks, so leaving them in makes the covariance singular and the
        propagated widths meaningless.
        """
        # The Gauss-Newton validity check costs about 2 ms here -- 0.1% of a
        # real pyscf fit, but 40% of this call on a cheap backend, so it is
        # switchable. On by default: it is what catches a reported width
        # resting on an approximation that has failed.
        cov, _ = self.laplace_approximation(
            check_gauss_newton=check_gauss_newton)
        if cov is None:
            return None
        # A stationary saddle (gauss_newton_warnings, filter 3): the posterior
        # is improper along a direction a reported width depends on, so there
        # is no Laplace width to give for any coordinate.
        saddle = getattr(self, "_gn_saddle", None)
        if (check_gauss_newton and saddle is not None
                and saddle[0] == np.asarray(self.coords, dtype=float).tobytes()):
            return None
        n = self.coords.size
        if cov.shape[0] != n:
            return None

        # Drop the rigid-body subspace, which no observable constrains.
        cov, _basis, _n_gauge = self.physical_covariance(cov)

        J, _, row_scale = self.spectral.stacked(self.coords, return_row_scale=True)
        cov = cov + self._systematic_covariance(J, cov, coords=self.coords,
                                                row_scale=row_scale)

        # IS THIS COVARIANCE USABLE AT ALL? Checked once, on the matrix, rather
        # than per coordinate on the sign of a quadratic form.
        #
        # The per-coordinate check below catches a variance that comes out
        # negative, which is what happens on some platforms. On others the same
        # collapsed matrix yields a POSITIVE variance and so a plausible-looking
        # width -- measured: 0.0119 Ang on a covariance whose condition number
        # is 1.2e32 and whose minimum eigenvalue is -120 times its maximum.
        # Guarding on the sign alone therefore catches one symptom on one
        # machine and lets the same meaningless number through on another,
        # which is exactly how this survived in the first place.
        sym = (cov + cov.T) / 2.0
        try:
            eigs = np.linalg.eigvalsh(sym)
        except np.linalg.LinAlgError:
            eigs = None
        if eigs is not None and eigs.size:
            hi = float(np.max(eigs))
            lo = float(np.min(eigs))
            if hi > 0.0 and lo / hi < -_COV_INDEFINITE_REL_TOL:
                self._warn_once(
                    "geometry_uncertainty: the posterior covariance is not "
                    f"positive semi-definite (min/max eigenvalue = "
                    f"{lo / hi:.2e}), so no width taken from it is meaningful. "
                    "Returning None rather than numbers. The Laplace "
                    "covariance is factorised over the physical directions "
                    "and is positive definite by construction, so this "
                    "points at a covariance supplied some other way or at "
                    "the systematic block added to it, not at the data.",
                    key="geometry_uncertainty_indefinite_covariance",
                )
                return None

        null_frac = self._data_null_fractions(J)

        bonds = _detect_bonds(self.coords, self.elems)
        angles = _detect_angles(bonds)
        x0 = np.asarray(self.coords, dtype=float).ravel()

        def propagate(fn, h=1e-5):
            g = np.zeros(n)
            for k in range(n):
                xp, xm = x0.copy(), x0.copy()
                xp[k] += h
                xm[k] -= h
                g[k] = (fn(xp.reshape(-1, 3)) - fn(xm.reshape(-1, 3))) / (2 * h)
            var = float(g @ cov @ g)

            # A NEGATIVE VARIANCE IS A FAILURE, NOT A ZERO.
            #
            # This used to be `sqrt(max(var, 0.0))`, which is right for a
            # negative of round-off size and catastrophically wrong for
            # anything larger. Measured on water with the relative sigma floor
            # switched off and sigmas quoted at 0.02 MHz on constants near
            # 1e6 MHz: the covariance reaches a condition number of 1.2e32
            # with a minimum eigenvalue of -2.0e13, this variance comes out at
            # -2.3e13, and the old clamp reported it as sigma = 0.000000.
            #
            # That is the worst possible answer. A function whose own docstring
            # says the uncertainty IS the result, and that a structure quoted
            # without one cannot be used, was quoting PERFECT PRECISION at the
            # exact point where its arithmetic had collapsed. It also made the
            # behaviour platform-dependent -- on another LAPACK the round-off
            # lands slightly positive and you get a plausible 0.0119 instead,
            # which is how it survived: a test pinned the comparison and
            # passed on one machine and failed on another, with neither value
            # meaning anything.
            #
            # The tolerance is relative to the size of the terms being summed,
            # because an absolute one cannot tell 1e-20 on a small matrix from
            # 1e-20 on a huge one. The floored default path sits at
            # min_eig = -2.7e-20 against a positive variance of 1.4e-4 and is
            # unaffected.
            scale = float(np.abs(g) @ np.abs(cov) @ np.abs(g))
            if var < -_VAR_NEGATIVE_REL_TOL * max(scale, _TINY):
                self._warn_once(
                    "geometry_uncertainty: propagated variance came out "
                    f"negative ({var:.3e} against a term scale of "
                    f"{scale:.3e}), so the posterior covariance is not "
                    "positive semi-definite and this coordinate's width "
                    "cannot be propagated. Reported as nan rather than 0. "
                    "The Laplace covariance is positive definite by "
                    "construction, so this points at a covariance supplied "
                    "some other way or at the systematic block added to it.",
                    key="geometry_uncertainty_negative_variance",
                )
                return float("nan"), g
            return float(np.sqrt(max(var, 0.0))), g

        out = {}
        coord_fns = []
        for i, j in bonds:
            coord_fns.append((f"{self.elems[i]}{i + 1}-{self.elems[j]}{j + 1}",
                              lambda c, i=i, j=j: float(np.linalg.norm(c[i] - c[j])),
                              self._coordinate_class(self.elems[i], self.elems[j])))
        for i, j, k in angles:
            def ang(c, i=i, j=j, k=k):
                v1, v2 = c[i] - c[j], c[k] - c[j]
                cosa = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                return float(np.degrees(np.arccos(np.clip(cosa, -1.0, 1.0))))

            coord_fns.append((f"{self.elems[i]}{i + 1}-{self.elems[j]}{j + 1}"
                              f"-{self.elems[k]}{k + 1}", ang, "angle"))

        for name, fn, cls in coord_fns:
            sigma, g = propagate(fn)
            # E3: the blended sigma hides which source it came from, and the two
            # parts are calibrated against different things -- the data part
            # against residuals and resampling, the prior part against the
            # measured geometry error of the level of theory. Split by
            # projecting the coordinate gradient onto the data's range and null
            # spaces and propagating each part through the same covariance.
            # The cross term is dropped, so the parts need not add in
            # quadrature to the total; they are diagnostics, not a partition.
            g_null = null_frac @ g
            g_range = g - g_null
            # Same clamp question as the total above, on two projections of
            # it. These are diagnostics rather than reported widths, and the
            # indefinite-covariance guard earlier in this method has already
            # refused anything badly broken, so a round-off clamp is all that
            # is needed here -- but it is nan rather than 0 for the same
            # reason: a zero split would read as "this source contributes no
            # uncertainty", which is a claim, not an absence.
            _split = []
            for _g in (g_range, g_null):
                _v = float(_g @ cov @ _g)
                _scale = float(np.abs(_g) @ np.abs(cov) @ np.abs(_g))
                _split.append(float("nan")
                              if _v < -_VAR_NEGATIVE_REL_TOL * max(_scale, _TINY)
                              else float(np.sqrt(max(_v, 0.0))))
            self._last_sigma_split[name] = (_split[0], _split[1])
            # How much of this coordinate the spectrum cannot see. 1.0 means the
            # value came from the quantum surface and nothing else -- which is
            # the honest caveat to attach to a structure nobody can check, and
            # is invisible in the sigma alone, since there the prior's width is
            # reported exactly as if it had been measured.
            gn = float(np.linalg.norm(g))
            blind = float(np.linalg.norm(null_frac @ g) / gn) if gn > 0 else 0.0
            blind = min(max(blind, 0.0), 1.0)
            # Per-class prior floor. The isotropic quantum_prior_sigma_ang says
            # every theory-determined coordinate is equally trustworthy; the
            # measured geometry error of a level of theory says otherwise (at
            # B3LYP/6-31G(d), C-Cl errs several times worse than C-F). Where a
            # coordinate rests on the prior -- and only to that fraction -- its
            # prior-side width must not be quoted below the measured class
            # error. The total is topped up by the variance deficit, so a
            # coordinate the data determines is untouched, and no interval ever
            # narrows.
            if self._prior_class_sigma:
                d_sig, p_sig = self._last_sigma_split[name]
                floor = blind * float(self._prior_class_sigma.get(cls, 0.0))
                if floor > p_sig:
                    sigma = float(np.sqrt(sigma * sigma + floor * floor - p_sig * p_sig))
                    self._last_sigma_split[name] = (d_sig, floor)
            out[name] = sigma
            self._last_data_support[name] = 1.0 - blind
        return out

    #: Data-information fractions at which a coordinate is labelled
    #: data-constrained / prior-dominated -- the thresholds the former
    #: priors.prior_sensitivity module used (removed 2026-11-19; this method
    #: is what the product reports).
    _DATA_FRACTION_LABELS = (0.8, 0.2)

    def internal_coordinate_posterior(self):
        """The fit's posterior over its bonds, angles and dihedrals.

        What the run report and its exports quote. It is the covariance
        geometry_uncertainty propagates -- the Laplace over the physical
        directions, from every block of the objective the fit minimised
        (_posterior_blocks), plus the systematic block -- carried into the
        active internal coordinates through Wilson's B.

        It used to come from a separate rebuild (uncertainty.core
        .compute_uncertainty) that carried the configured internal prior and
        no quantum prior at all, kept the symmetry-forbidden directions, and
        rescaled every width by the spectral chi-square -- downward too, when
        chi-square was below one -- while the report said the widths included
        the quantum prior. No rescaling here: the spectral chi-square per
        observation is returned as a diagnostic.

        Each coordinate is labelled by how much of its precision the data
        supplies, 1 - var(posterior) / var(priors alone): >= 0.8
        "data-constrained", <= 0.2 "prior-dominated", otherwise "mixed".

        Returns a dict of plain lists -- it crosses the multistart process
        boundary -- or None when the Laplace has no answer.
        """
        x = np.asarray(self.coords, dtype=float)
        cov_lap, _ = self.laplace_approximation(coords=x, check_gauss_newton=False)
        if cov_lap is None:
            return None
        try:
            gn = list(self.gauss_newton_warnings(coords=x))
        except Exception as exc:  # a diagnostic must not cost the report
            gn = [f"Gauss-Newton check unavailable: {type(exc).__name__}"]
        saddle = getattr(self, "_gn_saddle", None)
        if saddle is not None and saddle[0] == x.tobytes():
            return {"error": "no Laplace width exists at this geometry -- " + saddle[1],
                    "warnings": gn}
        hess_d = self.hessian_distance(x)
        if hess_d is not None and hess_d > float(self.orca_update_thresh):
            gn = gn + [
                f"the quantum Hessian behind these widths was computed "
                f"{1e3 * hess_d:.2f} mA (RMS) from this geometry, beyond the "
                f"{1e3 * float(self.orca_update_thresh):.2f} mA at which the fit "
                f"refreshes its quantum point: the widths and the quantum prior's "
                f"weight carry that curvature, not the curvature here (formyl "
                f"fluoride: a Hessian at the answer moves its widths by up to 3%)"]
        gn = gn + self.distortion_warnings() + self.model_error_warnings()
        try:
            split_gap = self.split_inconsistency()
        except Exception as exc:  # a diagnostic must not cost the report
            split_gap = f"the split check could not be made ({type(exc).__name__}: {exc})"
        if split_gap is not None:
            gn = gn + [
                f"split objective: {split_gap}. These widths are the split posterior's, "
                f"which need not be the split estimator's spread; the joint objective "
                f"has no such gap."]
        cov_lap, basis, n_removed = self.physical_covariance(cov_lap, coords=x)
        J, rw, row_scale = self.spectral.stacked(x, return_row_scale=True)
        rw = np.asarray(rw, dtype=float)
        cd_rows = self.distortion_rows(x)
        C_sys = self._systematic_covariance(J, cov_lap, coords=x, row_scale=row_scale)
        cov_x = cov_lap + C_sys
        _H_spec, H_prior, parts = self._posterior_blocks(x)

        cs = InternalCoordinateSet(x, self.elems, self._ic_use_dihedrals)
        Bq = np.asarray(cs.active_B_matrix(x), dtype=float)
        q = np.asarray(cs.active_values(x), dtype=float)
        C_q = Bq @ cov_x @ Bq.T
        C_q = 0.5 * (C_q + C_q.T)
        std, _info = std_err_from_covariance(
            C_q, context="internal_coordinate_posterior")

        # Prior-only variance along each coordinate, over the same physical
        # directions; a direction no prior constrains has infinite variance.
        Mp = basis.T @ H_prior @ basis
        w_p, V_p = np.linalg.eigh(0.5 * (Mp + Mp.T))
        tiny = 1e-12 * max(float(np.max(np.abs(w_p))) if w_p.size else 0.0, 1e-300)
        hi, lo = self._DATA_FRACTION_LABELS
        fractions, labels = [], []
        for b in Bq:
            v_post = float(b @ cov_lap @ b)
            proj = V_p.T @ (basis.T @ b)
            nb = float(proj @ proj)
            free = w_p <= tiny
            if nb <= 0.0 or v_post <= 0.0:
                frac = float("nan")
            elif np.any(proj[free] ** 2 > 1e-12 * nb):
                frac = 1.0
            else:
                v_prior = float(np.sum(proj[~free] ** 2 / w_p[~free]))
                frac = float(np.clip(1.0 - v_post / v_prior, 0.0, 1.0))
            fractions.append(frac)
            if not np.isfinite(frac):
                labels.append("unidentifiable")
            elif v_post > 1e4:
                labels.append("unidentifiable")      # held by the ridge alone
            elif frac >= hi:
                labels.append("data-constrained")
            elif frac <= lo:
                labels.append("prior-dominated")
            else:
                labels.append("mixed information")

        blocks = ["spectral data"] + list(parts)
        has_sys = bool(np.any(C_sys != 0.0))
        if has_sys:
            blocks.append("systematic model error")
        st = self._last_ic_prior
        ic_sigma = (st["sigma"].tolist()
                    if st is not None and st["weight"] > 0.0
                    and st["sigma"].size == q.size else None)
        # Where each internal-prior centre and width came from, for the fit
        # that had one: the rule library's provenance, with the centre and
        # width overwritten by the ones the fit actually used.
        provenance = None
        if ic_sigma is not None:
            try:
                q_init = (np.asarray(st["coord_set"].active_values(self._ic_initial_coords), dtype=float)
                          if self._ic_initial_coords is not None else None)
                _jp, _rp, _sp, meta = build_internal_priors(
                    st["coord_set"], x,
                    sigma_bond=self._ic_prior_sigma_bond,
                    sigma_angle_deg=self._ic_prior_sigma_angle_deg,
                    sigma_dihedral_deg=self._ic_prior_sigma_dihedral_deg,
                    prior_values=q_init, prior_mode=self._ic_prior_mode,
                    prior_specs=self._ic_user_priors, elems=self.elems,
                    freeze_sigma_floor=self._ic_freeze_sigma_floor,
                    configured_kinds=self._ic_prior_configured_kinds,
                    return_metadata=True)
                provenance = []
                for k, m in enumerate(meta):
                    m = dict(m)
                    m["atoms"] = [int(a) for a in (m.get("atoms") or ())]
                    m["atom_types"] = list(m.get("atom_types") or ())
                    m["target_value"] = float(st["q0"][k])
                    m["sigma"] = float(st["sigma"][k])
                    provenance.append(m)
            except Exception:  # provenance is a nicety; the widths are not
                provenance = None
        return {
            "names": list(cs.active_names()),
            # Whether the set includes dihedrals (stated, or added because
            # bonds and angles missed a direction: dihedrals_reason).
            "use_dihedrals": bool(self._ic_use_dihedrals),
            "dihedrals_reason": self._ic_dihedrals_reason,
            "kinds": [ic.kind for ic in cs.active_coords()],
            "values": q.tolist(),
            "covariance": C_q.tolist(),
            "std_err": np.asarray(std, dtype=float).tolist(),
            "data_fraction": fractions,
            "labels": labels,
            "blocks": blocks,
            "objective_mode": str(self.optimizer.objective_mode),
            "hessian_distance_ang": hess_d,
            "alpha_q": (None if self.spectral_only or self.quantum is None
                        else float(self.quantum_prior_weight(refresh=True))),
            "chi2_per_observation": float(rw @ rw) / max(rw.size, 1),
            "n_observations": int(rw.size),
            # The observations behind that chi-square: the rotational rows and,
            # when they are on, the distortion rows (listed).
            "n_distortion_rows": len(cd_rows),
            "n_rotational_rows": int(rw.size) - len(cd_rows),
            "distortion_rows": cd_rows,
            "n_removed_directions": int(n_removed),
            "internal_prior_sigma": ic_sigma,
            "internal_prior_provenance": provenance,
            # The correlation the model error was propagated with, when it was.
            "systematic_isotopologue_correlation": (
                self.systematic_isotopologue_correlation if has_sys else None),
            "systematic_component_correlation": (
                None if not has_sys or self.systematic_component_correlation is None
                else self.systematic_component_correlation.tolist()),
            "warnings": gn,
        }

    def _systematic_covariance(self, J, cov, coords=None, row_scale=None):
        """The model-error term of the reported covariance, counted once.

        Model error -- the B0-versus-Be gap a rigid structure cannot
        reproduce or, where a vibrational correction removed that gap, the
        correction's own uncertainty -- is not measurement noise, but it moves
        the answer: a shift e in the constants moves the optimum by A D e,
        with A = C J^T the estimator's response and D the row scale J carries
        (stacked divides each row by its effective sigma and multiplies it by
        its component and robust weights).

        The covariance of the constants is then Sigma_meas + S R S rather than
        the diag(sigma_eff^2) the Laplace assumed, so the reported covariance
        is the sandwich C J^T D (Sigma_meas + S R S) D J C + C H_prior C. This
        returns its difference from the Laplace, which callers add to C:

            A [D (Sigma_meas + S R S) D - I] A^T

        * S: the model-error sigma of each row -- sigma_systematic_constants,
          or the correction sigma on a row a vibrational correction with a
          stated sigma was applied to (the gap it stood for is gone; a
          correction of unstated quality does not retire it).
        * R: its correlation between rows, R_ab = Q[comp_a, comp_b] * (1 if
          the same isotopologue else rho). rho
          (systematic_isotopologue_correlation, 0.99) couples the same
          constant type across isotopologues, which share one potential
          surface and so one force-field error; Q
          (systematic_component_correlation, identity by default) couples A,
          B and C, whose gaps have no consistent sign.
        * Sigma_meas: the measurement part of the weighting sigma -- the
          weighting with its model-error channels left out (the systematic
          floor, the correction sigma) and, where sigma_constants itself
          carries model error (sigma_measurement_constants), that too.

        When the weighting holds no model error, D Sigma_meas D = I and this
        is A D S R S D A^T. When it does -- reference sigmas that are
        hypot(model error, quoted precision), systematic_in_weighting, a
        correction sigma -- the copy the Laplace already counted as
        independent noise is taken out before the correlated one goes in;
        the old form added it on top, nearly doubling the variance. With
        rho = 0 and no model error in the weights this is the independent
        model, A diag((D s)^2) A^T. Centrifugal-distortion rows carry no
        model error and keep their Laplace weighting. Exactly zero when no
        model error is given.

        ``row_scale`` is the third value of ``stacked(coords,
        return_row_scale=True)``; callers that have just stacked J at
        ``coords`` pass it rather than paying for a second evaluation.
        """
        J = np.asarray(J, dtype=float)
        cov = np.asarray(cov, dtype=float)
        rows = self._constant_error_rows(coords=coords, row_scale=row_scale)
        if rows is None:
            return np.zeros_like(cov)
        s_model, m2, n_rot = rows["s_model"], rows["m2"], rows["n_rot"]
        if J.shape[0] < n_rot or (not np.any(s_model > 0.0) and np.all(m2 == 1.0)):
            return np.zeros_like(cov)
        scale = rows["scale"]
        if scale is None:
            return np.zeros_like(cov)
        t = scale * s_model
        K = (t[:, None] * rows["R"]) * t[None, :]
        K[np.diag_indices_from(K)] += m2 - 1.0
        A = cov @ J[:n_rot].T
        out = A @ K @ A.T
        out = 0.5 * (out + out.T)
        if not np.all(np.isfinite(out)):
            return np.zeros_like(cov)
        return out

    def _constant_error_rows(self, coords=None, row_scale=None):
        """Per stacked constant: the pieces of the constants' error model.

        s_model (model-error sigma), m2 = (measurement part / weighting
        sigma)^2, the row scale J carries, and the correlation R of the model
        error between rows -- what _systematic_covariance and
        constant_error_covariance are both built from, so the reported
        covariance and the bootstrap's draws describe one error model. None
        when there are no constants.
        """
        sp = self.spectral
        x = np.asarray(self.coords if coords is None else coords, dtype=float)
        s_rows, m2_rows, iso_rows, comp_rows = [], [], [], []
        for i_iso, iso in enumerate(sp.isotopologues):
            idx = np.asarray(iso["component_indices"], dtype=int)
            n = idx.size

            def _row_vec(key, _iso=iso, _n=n):
                v = _iso.get(key)
                if v is None:
                    return None
                v = np.asarray(v, dtype=float).ravel()
                return v if v.size == _n else None

            sig_eff = np.asarray(sp._effective_sigma(iso["sigma_constants"], iso=iso),
                                 dtype=float).ravel()
            meas_raw = _row_vec("sigma_measurement_constants")
            if meas_raw is None:
                meas_raw = np.asarray(iso["sigma_constants"], dtype=float).ravel()
            sig_meas = np.asarray(sp._effective_sigma(meas_raw, iso=iso, model_error=False),
                                  dtype=float).ravel()
            sys_sig = _row_vec("sigma_systematic_constants")
            s_model = np.zeros(n) if sys_sig is None else np.abs(sys_sig)
            # A row a vibrational correction with a stated sigma was applied
            # to: the gap is gone, and what is left of the model error is the
            # correction's sigma. A correction of unstated quality does not
            # retire the gap -- the same rule the weighting's floor follows.
            corr = np.asarray(sp._correction_sigma(iso, n), dtype=float)
            s_model = np.where(corr > 0.0, corr, s_model)
            s_rows.append(s_model)
            m2_rows.append((sig_meas / sig_eff) ** 2)
            iso_rows.append(np.full(n, i_iso))
            comp_rows.append(idx)
        if not s_rows:
            return None
        s_model = np.concatenate(s_rows)
        m2 = np.concatenate(m2_rows)
        n_rot = s_model.size
        iso_i = np.concatenate(iso_rows)
        comp = np.concatenate(comp_rows)
        Qc = (np.eye(3) if self.systematic_component_correlation is None
              else self.systematic_component_correlation)
        R = Qc[np.ix_(comp, comp)] * np.where(
            iso_i[:, None] == iso_i[None, :], 1.0, self.systematic_isotopologue_correlation)
        scale = None
        if not np.all(s_model == 0.0) or not np.all(m2 == 1.0) or row_scale is not None:
            if row_scale is None:
                _J0, _r0, row_scale = sp.stacked(x, want_jacobian=False, return_row_scale=True)
            scale = np.asarray(row_scale, dtype=float).ravel()
            if scale.size != n_rot:
                scale = None
        return {"s_model": s_model, "m2": m2, "n_rot": n_rot, "iso": iso_i,
                "comp": comp, "R": R, "scale": scale}

    def constant_error_covariance(self, coords=None):
        """Covariance of the stacked constants' errors, in MHz^2: the error
        model the reported covariance propagates.

            Sigma = diag((sigma_meas / c)^2) + S R S

        with sigma_meas the measurement part of the weighting sigma (what
        _systematic_covariance keeps; c the component and robust weights J
        carries, 1 by default), S the model-error sigmas and R their
        correlation (rho across isotopologues for one constant type, Q
        across types). In the linear limit the optimum's covariance under
        this Sigma is the reported covariance less its prior part -- so a
        bootstrap drawing the constants from it checks the reported number
        rather than describing a third one. Rows are in stacked order
        (isotopologue, then component_indices); centrifugal-distortion rows
        are not included.
        """
        x = np.asarray(self.coords if coords is None else coords, dtype=float)
        _J0, _r0, row_scale = self.spectral.stacked(x, want_jacobian=False,
                                                    return_row_scale=True)
        rows = self._constant_error_rows(coords=x, row_scale=row_scale)
        if rows is None:
            return np.zeros((0, 0))
        scale = rows["scale"]
        s = rows["s_model"]
        meas_var = rows["m2"] / np.maximum(scale, 1e-300) ** 2
        Sigma = (s[:, None] * rows["R"]) * s[None, :]
        Sigma[np.diag_indices_from(Sigma)] += meas_var
        return 0.5 * (Sigma + Sigma.T)

    def _data_null_fractions(self, J):
        """Projector onto the directions the spectrum does not own.

        The cutoff matches the fit's own range/null boundary -- the relative
        sv_threshold and the absolute sv_min_abs floor from the optimiser --
        rather than a bare numerical-rank tolerance. With a 1e-8 tolerance a
        direction the data barely grazes is labelled "data" while its variance
        in the very same covariance is prior-dominated, so the data/prior split
        and the theory-determined flag disagree with the fit that produced
        them; and the boundary would silently stand still when
        quantum_prior_sigma_ang (which sets sv_min_abs) moves it.
        """
        J = np.asarray(J, dtype=float)
        n = self.coords.size
        if J.size == 0 or J.shape[1] != n:
            return np.zeros((n, n))
        _, s, Vt = np.linalg.svd(J, full_matrices=True)
        if s.size == 0 or s[0] <= 0:
            return np.eye(n)
        cut = max(float(self.optimizer.sv_threshold) * float(s[0]),
                  float(getattr(self.optimizer, "sv_min_abs", 0.0) or 0.0))
        rank = int((s > cut).sum())
        V_null = Vt[rank:]
        return V_null.T @ V_null if V_null.size else np.zeros((n, n))


    def sigma_split(self):
        """Per-coordinate (data_sigma, prior_sigma) from the last uncertainty call.

        See the E3 comment in geometry_uncertainty: the first number is the
        width along directions the spectrum resolves, the second along the
        directions only the quantum prior constrains.
        """
        return dict(self._last_sigma_split)

    def data_support(self):
        """Fraction of each coordinate the spectrum actually determines.

        Populated by :meth:`geometry_uncertainty`. 1.0 means fully constrained
        by measurement, 0.0 means the value is the quantum surface's alone.
        """
        return dict(self._last_data_support)

    def kraitchman_report(self):
        """Substitution (r_s) coordinates from the data, next to the fitted geometry.

        This is an independent check on the fit rather than an input to it.
        Kraitchman's equations give |a|, |b|, |c| for each substituted atom
        directly from the measured constants, using no force field, no level of
        theory and no structural model -- so where a substitution coordinate and
        the fitted one disagree by much more than the Costain uncertainty, the
        disagreement is the fit's, and it localises the problem to a specific
        atom instead of to the structure as a whole.

        Returns the analysis dict (see :func:`kraitchman_analysis`) with each row
        augmented by the fitted principal-axis coordinates, or None when the data
        cannot support it. Only species with all three constants measured are
        usable, since the equations need all three moments.
        """
        try:
            from backend.kraitchman import (  # pylint: disable=import-outside-toplevel
                compare_rs_to_geometry, kraitchman_analysis,
            )
        except ImportError:
            return None
        isos = []
        for iso in self.spectral.isotopologues:
            if len(iso["component_indices"]) != 3:
                continue
            const = np.asarray(iso["obs_constants"], dtype=float)
            if const.size != 3 or not np.all(np.isfinite(const)) or np.any(const <= 0):
                continue
            isos.append({"name": iso["name"], "masses": np.asarray(iso["masses"], dtype=float),
                         "obs_constants": const})
        if len(isos) < 2:
            return None
        try:
            res = kraitchman_analysis(isos)
            res["rows"] = compare_rs_to_geometry(
                res["rows"], self.coords,
                np.asarray(isos[0]["masses"], dtype=float))
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            return {"rows": [], "defects": [], "skipped": [],
                    "warnings": [f"Kraitchman analysis failed: {exc}"]}
        return res

    def fit_mass_dependent_structure(self, model="rm1", coords0=None):
        """Fit a Watson mass-dependent (r_m) structure to the loaded constants.

        This is an alternative route to a near-equilibrium structure that uses
        no force field and no level of theory: the rovibrational offset is
        fitted as a handful of isotope-independent coefficients rather than
        computed from anharmonic derivatives. That matters here because the
        computed route -- `rovib_corrections` -- needs a cubic force field, and
        at the levels of theory that are affordable for these molecules it was
        measured to hurt more than it helped.

        It needs several isotopologues, since the structure and the coefficients
        are separated only by how differently they respond to substitution. The
        returned object carries the degrees of freedom and reduced chi-square so
        that an under-determined fit is visible rather than silently reported.

        Independent of the hybrid fit: it reads the same constants but does not
        use the quantum surface, so comparing the two is a real check.
        """
        from backend.spectral.mass_dependent import (  # pylint: disable=import-outside-toplevel
            fit_mass_dependent_structure,
        )
        start = self.coords if coords0 is None else np.asarray(coords0, dtype=float)
        return fit_mass_dependent_structure(
            self.spectral.isotopologues, start, model=model, symmetry=self.symmetry,
        )


    def log_probability(self, trial_coords, use_proxy=True):
        """
        Compute the log-posterior probability P(geometry | Spectra, Theory).

        Components:
          1. Spectral Likelihood (Gaussian noise model on B0)
          2. Theory Prior (QC energy surface)
          3. Structural Priors (Internal coordinate constraints)
        """
        # 1. Spectral Likelihood.
        #
        # Residuals only. Both of these calls discarded the Jacobian they
        # asked for, and the spectral one builds it per isotopologue: measured at
        # about 400 of this method's 413 microseconds, paid once per MCMC
        # proposal -- 16,000 times for four chains of 4,000 steps -- for a
        # matrix thrown away on the next line.
        _, residual_w = self.spectral.stacked(trial_coords, want_jacobian=False)
        chi_sq_spectral = np.sum(residual_w**2)

        # 2. Structural priors -- every non-quantum prior the fit carried,
        # including the q-space internal prior this used to skip with `pass`.
        chi_sq_prior = self.structural_prior_chi2(trial_coords)

        # The quantum prior, weighted as the fit weighted it rather than with
        # the legacy alpha_quantum. These two readings differ by 1811x on
        # water, so this used to be the log-posterior of an objective nobody
        # minimised -- and the MCMC sampled it. Both pieces now come from the
        # same place laplace_approximation gets them.
        energy_term = self.quantum_prior_energy(trial_coords, use_proxy=use_proxy)

        log_p = -0.5 * (chi_sq_spectral + chi_sq_prior)
        return log_p - (self.quantum_prior_weight() * energy_term)

    # ── The objective the fit minimised, for everything that reports on it ──

    def _fit_alpha(self, J, H_projected):
        """The quantum weight the fit's step used.

        With a calibrated sigma_x it depends on the Hessian alone, which the
        step reads in Cartesian form in both coordinate modes, so recomputing
        it here gives the step's own number. The legacy heuristic depends on
        the rank of the step's Jacobian -- q-space in internal mode -- so there
        the value the last step recorded is the one the fit used.
        """
        if (self.optimizer.quantum_prior_sigma_ang is None
                and self._last_alpha_q_eff is not None):
            return float(self._last_alpha_q_eff)
        return float(self.optimizer.effective_quantum_weight(
            J, int(np.linalg.matrix_rank(J)), hessian=H_projected))

    def _internal_prior_rows(self, x):
        """sqrt(w) diag(1/sigma) B(x): the q-space internal prior's
        Gauss-Newton rows in Cartesian coordinates, as the fit last weighted
        it -- or None when the fit carried no internal prior."""
        st = self._last_ic_prior
        if st is None or self.coordinate_mode != "internal" or st["weight"] <= 0.0:
            return None
        B = np.asarray(st["coord_set"].active_B_matrix(x), dtype=float)
        if B.shape[0] != st["sigma"].size:
            return None
        return np.sqrt(st["weight"]) * (B / st["sigma"][:, None])

    def _isotropic_prior_active(self):
        sig = self.optimizer.isotropic_prior_sigma_ang
        return (self.optimizer.objective_mode == "joint" and sig is not None
                and float(sig) > 0.0 and self._prior_target_coords is not None
                and self.coordinate_mode == "cartesian")

    def structural_prior_chi2(self, coords):
        """Every non-quantum prior in the fitted objective, as chi-square.

        The fit minimises 1/2 |r|^2 over the spectral rows and the rows of
        each of these priors, so -1/2 of this is their log-density: the
        InternalPriorEngine rows (Cartesian mode), the q-space internal prior
        (internal mode, as the fit last weighted it) and the isotropic prior
        on the displacement from the prior target (joint, Cartesian). The
        log-probability used to skip the q-space prior with a bare `pass`,
        so the sampler and the posterior-state report described an objective
        with one term fewer than the fit's.
        """
        x = np.asarray(coords, dtype=float)
        chi2 = 0.0
        if self.internal_prior is not None and self.coordinate_mode == "cartesian":
            _, rp = self.internal_prior.stacked(x, want_jacobian=False)
            chi2 += float(self.prior_weight) * float(np.sum(np.asarray(rp) ** 2))
        st = self._last_ic_prior
        if st is not None and self.coordinate_mode == "internal" and st["weight"] > 0.0:
            cs = st["coord_set"]
            d = np.asarray(st["q0"], dtype=float) - np.asarray(cs.active_values(x), dtype=float)
            for k, ic in enumerate(cs.active_coords()):
                if ic.kind == "dihedral":
                    d[k] = (d[k] + np.pi) % (2.0 * np.pi) - np.pi
            chi2 += st["weight"] * float(np.sum((d / st["sigma"]) ** 2))
        if self._isotropic_prior_active():
            dd = (x - _kabsch_align(self._prior_target_coords, x)).ravel()
            chi2 += float(dd @ dd) / float(self.optimizer.isotropic_prior_sigma_ang) ** 2
        return chi2

    def _split_null_projector(self, x, J_spec, prior_rows):
        """Null space of the split step's stacked Jacobian, in Cartesians."""
        J_st = np.vstack([J_spec] + list(prior_rows)) if prior_rows else J_spec
        _U, _s, Vt, rank = self.optimizer.decompose(J_st)
        return self.optimizer.null_projector(Vt, rank)

    def _posterior_blocks(self, coords=None):
        """Gauss-Newton curvature of the objective the fit minimised.

        Returns ``(H_spec, H_prior, parts)``: the spectral block J^T J, the sum
        of every other block, and those blocks by name. One builder, so the
        Laplace, the Gauss-Newton check and the run report cannot drift apart
        again -- they had: the Laplace carried a quantum block the split
        objective only uses in the data's null space, and none of the
        internal prior that internal-mode fits did carry.

        * quantum: alpha_q spd(H), as the joint step weights it; under the
          split objective only on the null space of the stacked Jacobian,
          which is where that step lets it act;
        * internal prior: the q-space prior's rows, when the fit had one;
        * structural prior: the InternalPriorEngine rows (Cartesian mode);
        * isotropic prior: I / sigma_iso^2 (joint, Cartesian, prior target).
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        J, _ = self.spectral.stacked(x)
        H_spec = J.T @ J
        parts = {}
        rows = []
        R_ic = self._internal_prior_rows(x)
        if R_ic is not None:
            parts["internal prior"] = R_ic.T @ R_ic
            rows.append(R_ic)
        if self.internal_prior is not None and self.coordinate_mode == "cartesian":
            Jp, _rp = self.internal_prior.stacked(x)
            R_cp = np.sqrt(float(self.prior_weight)) * np.asarray(Jp, dtype=float)
            parts["structural prior"] = R_cp.T @ R_cp
            rows.append(R_cp)
        if not self.spectral_only and self.quantum is not None:
            _g, Hp = self._project_quantum_terms(self.quantum.gradient, self.quantum.hessian)
            Hq = self._fit_alpha(J, Hp) * self.optimizer._spd_hessian(Hp)
            if self.optimizer.objective_mode == "joint":
                parts["quantum"] = Hq
            else:
                Pn = self._split_null_projector(x, J, rows)
                parts["quantum (data null space)"] = Pn @ Hq @ Pn
        if self._isotropic_prior_active():
            sig = float(self.optimizer.isotropic_prior_sigma_ang)
            parts["isotropic prior"] = np.eye(H_spec.shape[0]) / (sig * sig)
        H_prior = np.zeros_like(H_spec)
        for blk in parts.values():
            H_prior = H_prior + blk
        return H_spec, H_prior, parts

    def _posterior_gradient(self, coords=None):
        """Gradient of the objective _posterior_blocks gives the curvature of.

        Phi = 1/2 |r_spec|^2 + 1/2 chi2_structural + alpha_q E_q, each term as
        the fit weighted it, with E_q the same quadratic model of the surface
        the step solves (about the last quantum point, or about the prior
        target when there is one; on the split objective only in the data's
        null space). It is what says how far a geometry is from the mode the
        Laplace widths describe.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        J, r = self.spectral.stacked(x)
        grad = -(np.asarray(J, dtype=float).T @ np.asarray(r, dtype=float))
        R_ic = self._internal_prior_rows(x)
        if R_ic is not None:
            st = self._last_ic_prior
            cs = st["coord_set"]
            d = np.asarray(st["q0"], dtype=float) - np.asarray(cs.active_values(x), dtype=float)
            for k, ic in enumerate(cs.active_coords()):
                if ic.kind == "dihedral":
                    d[k] = (d[k] + np.pi) % (2.0 * np.pi) - np.pi
            grad = grad - R_ic.T @ (np.sqrt(st["weight"]) * d / st["sigma"])
        if self.internal_prior is not None and self.coordinate_mode == "cartesian":
            Jp, rp = self.internal_prior.stacked(x)
            grad = grad - float(self.prior_weight) * (np.asarray(Jp, dtype=float).T
                                                      @ np.asarray(rp, dtype=float))
        if not self.spectral_only and self.quantum is not None:
            g, H = self._projected_reference_terms()
            target = self._prior_target_coords
            if (target is not None and self.coordinate_mode == "cartesian"
                    and np.shape(target) == x.shape):
                gq = H @ (x - _kabsch_align(target, x)).ravel()
            else:
                dx = (x - self._orca_ref_coords).ravel()
                if self.optimizer.objective_mode != "joint":
                    Pn = self._split_energy_projector()
                    gq = Pn @ (g + H @ (Pn @ dx))
                else:
                    gq = g + H @ dx - self._rto_shift(dx.size)
            grad = grad + self.quantum_prior_weight() * gq
        if self._isotropic_prior_active():
            sig = float(self.optimizer.isotropic_prior_sigma_ang)
            grad = grad + (x - _kabsch_align(self._prior_target_coords, x)).ravel() / (sig * sig)
        return grad

    def laplace_approximation(self, coords=None, regularization: float = 1e-8,
                              check_gauss_newton: bool = True):
        """Analytical estimate of the posterior covariance at the optimum.

        The quantum block is weighted exactly as the optimiser weighted it, via
        :meth:`SubspaceOptimizer.effective_quantum_weight`, and its Hessian gets
        the same eigenvalue floor. Using ``alpha_quantum`` and the raw Hessian
        instead -- as this did -- describes a different objective than the one
        that was minimised whenever ``quantum_prior_sigma_ang`` is set, which is
        now the default, so the reported uncertainties would not belong to the
        reported geometry.

        INVERTED OVER THE PHYSICAL DIRECTIONS ONLY: C = B (BᵀHB + reg·I)⁻¹ Bᵀ,
        with B from :meth:`physical_basis`. The rigid-body directions, and any
        the imposed point group forbids, never enter the inverse, so the
        covariance comes back already projected and :meth:`physical_covariance`
        leaves it unchanged.

        This used to invert the whole 3N matrix and project afterwards. In
        exact arithmetic that is the same matrix; in floating point it is
        not. The rigid directions carry only the quantum block's eigenvalue
        floor (8.0e-5 on water) against 3e12 along the data, so their
        variances sit sixteen orders above the physical ones, and the
        inverse's round-off -- about eps times the largest entry -- swamps
        every physical entry. Measured on water with the sigma floor off,
        against an exact rational inverse: under C2v it refused at 2 MHz and
        below, although the physical block's condition number is 16.4; under
        C1 it returned widths silently wrong by 67% at 2 MHz and an angle
        width of exactly 0.0 at 0.02 MHz. The subspace inverse is exact to
        5e-14 in all of those cases, and moves ordinary floored results by
        under 1e-5 relative.

        ``regularization`` is now a ridge on the physical directions only --
        an isotropic N(0, 1/reg) prior in Å⁻², immaterial whenever the
        quantum prior is on. It matters only where the data and the prior
        together leave a physical direction unidentified, and that is
        reported below from the unridged block.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        # Every block of the objective the fit minimised (_posterior_blocks):
        # the internal prior when the fit carried one, the quantum block where
        # the objective lets it act. Kept apart so the Gauss-Newton check
        # below can be handed both without rebuilding the spectral
        # Jacobian.
        H_spec, H_prior, _parts = self._posterior_blocks(x)
        H_total = H_spec + H_prior

        reg_scalar = float(max(regularization, 0.0))
        basis = self.physical_basis(coords=x, n=H_total.shape[0])
        n_phys = int(basis.shape[1])
        if n_phys == 0:
            self._warn_once(
                "laplace_approximation: no physical directions remain once "
                "the rigid-body and symmetry-forbidden ones are removed, so "
                "there is no posterior to report. Returning None.",
                key="laplace_no_physical_directions")
            return None, None
        M = basis.T @ H_total @ basis
        M = 0.5 * (M + M.T)

        # CONDITIONING, REPORTED -- over the physical block, without the
        # ridge. On the 3N matrix both numbers were dominated by the
        # rigid-body directions and said nothing about the fit ("rank 3 of 9",
        # cond 3.9e20, while the physical block's cond was 16.4). With the
        # ridge added, a direction nothing identifies looks identified: rank
        # 1 becomes rank 2.
        _rank = int(np.linalg.matrix_rank(M))
        _cond = float(np.linalg.cond(M))
        if _rank < n_phys:
            self._warn_once(
                f"laplace_approximation: over the {n_phys} physical "
                f"directions the Hessian is rank {_rank}, so "
                f"{n_phys - _rank} of them are not identified by the data or "
                f"the prior and their widths are set by the "
                f"{reg_scalar:.1e} ridge rather than by anything physical.",
                key="laplace_rank_deficient")
        elif not np.isfinite(_cond) or _cond > 1e12:
            self._warn_once(
                f"laplace_approximation: the Hessian over the physical "
                f"directions has condition number {_cond:.2e}, so round-off "
                f"in its inverse can reach about {_cond * np.finfo(float).eps:.0e} "
                f"relative, and the widths taken from it are no more precise "
                f"than that.",
                key="laplace_ill_conditioned")

        # IS GAUSS-NEWTON EVEN VALID HERE? H_total above is JtJ, which drops
        # sum_i r_i Hess(r_i) from the exact Hessian of 0.5*chi^2. At this
        # engine's residual levels that omission can exceed the curvature it
        # perturbs and flip its sign, in which case the width below is an
        # artefact rather than an uncertainty. Reported, not assumed away.
        if check_gauss_newton:
            for _msg in self.gauss_newton_warnings(
                    coords=x, _blocks=(H_spec, H_prior)):
                self._warn_once("laplace_approximation: " + _msg,
                                key="laplace_gauss_newton_" + _msg[:48])

        # Cholesky succeeds exactly when the ridged physical block is
        # positive definite, so it is the check and the factorisation at once.
        # A Gaussian posterior needs that; anything else has no width to give.
        try:
            L = np.linalg.cholesky(M + reg_scalar * np.eye(n_phys))
        except np.linalg.LinAlgError:
            self._warn_once(
                "laplace_approximation: the Hessian over the physical "
                "directions is not positive definite, so no Gaussian "
                "posterior exists at this point. Returning None.",
                key="laplace_singular")
            return None, None
        L_inv = np.linalg.inv(L)
        M_inv = L_inv.T @ L_inv
        cov = basis @ M_inv @ basis.T
        cov = 0.5 * (cov + cov.T)
        std_err, _ = std_err_from_covariance(
            cov, context="Laplace approximation (physical subspace)")
        return cov, std_err

    def quantum_prior_weight(self, coords=None, refresh=False):
        """The weight the quantum prior carries, as the fit itself weighted it.

        ONE WEIGHT, NOT TWO. `laplace_approximation` asks
        :meth:`SubspaceOptimizer.effective_quantum_weight` for this, exactly as
        that method's own docstring instructs -- "anything that reports on the
        fit -- a covariance, a posterior, a log-probability -- has to weight
        the two halves the same way the step did". `log_probability` instead
        read ``alpha_quantum`` directly, which is the legacy heuristic that
        docstring warns against. On water the two differ by a factor of 1811:
        alpha_quantum is 1.0 while the calibrated weight is 1810.58. So the
        sampler, and `evaluate_posterior_state` with it, reported a
        log-posterior for an objective nobody minimised.

        COMPUTED ONCE, DELIBERATELY. The returned value is cached, and that is
        not a micro-optimisation. Under the legacy ``dynamic_quantum_weight``
        path the weight depends on ``rank(J)``, which depends on the geometry;
        recomputing it at each trial point would make `log_probability` a
        moving target rather than a fixed distribution, and a Metropolis chain
        against a moving target samples nothing in particular. The default
        path (``quantum_prior_sigma_ang`` set) already depends only on the
        fixed reference Hessian, so caching changes no number there -- it
        makes the guarantee hold in the legacy path too.

        Pass ``refresh=True`` after deliberately changing the weighting inputs.

        KEYED TO THE QUANTUM REFERENCE. The calibrated weight depends on the
        Hessian at the last quantum point, and that point moves whenever the
        fit drifts past orca_update_thresh. The cache was never cleared when
        it did, so a refit that travelled -- every randomise-then-optimise
        sample, any fit after a quantum update -- was described with the
        weight of a reference it had left: 2097 against the 1848 its own
        steps used on the water config, which put converged RTO samples up to
        0.2 posterior widths from their "optimum". Each quantum update now
        clears it.
        """
        cached = getattr(self, "_quantum_prior_weight_cache", None)
        if cached is not None and not refresh:
            return cached
        if self.spectral_only or self.quantum is None:
            self._quantum_prior_weight_cache = 0.0
            return 0.0
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        J, _ = self.spectral.stacked(x, want_jacobian=True)
        _g, H = self._project_quantum_terms(
            self.quantum.gradient, self.quantum.hessian)
        alpha = self._fit_alpha(J, H)
        self._quantum_prior_weight_cache = alpha
        return alpha

    def quantum_prior_energy(self, trial_coords, use_proxy=True):
        """The quantum prior's energy at a geometry, as the Hessian treats it.

        The gradient and Hessian are passed through
        :meth:`_project_quantum_terms` first, for the same reason
        `laplace_approximation` does: the quantum proxy is a quadratic
        expansion about a fixed Cartesian reference, and a quadratic about a
        fixed frame is not invariant under rotating the molecule. Unprojected,
        turning the whole molecule by 10 degrees moved the log-posterior by
        0.0004 -- small, but it is a restoring force along a direction that
        has no physics in it, and the sampler was free to wander there.

        No ``_spd_hessian`` floor here: that floor exists to keep a *curvature*
        matrix invertible, and clamping eigenvalues would change the energy
        value rather than condition a solve.
        """
        if self.spectral_only or self.quantum is None:
            return 0.0
        xx = np.asarray(trial_coords, dtype=float)
        if not use_proxy:
            if self.quantum_backend == "psi4" and self._psi4_engine:
                energy, _ = self._psi4_engine.run_gradient(xx)
                return float(energy)
            return float(self.quantum.energy)
        g, H = self._projected_reference_terms()
        target = self._prior_target_coords
        if (target is not None and self.coordinate_mode == "cartesian"
                and np.shape(target) == xx.shape):
            # With a prior target the fit's quantum prior IS this quadratic,
            # centred on the Kabsch-aligned target (see _prior_gradient) --
            # not the surface's expansion about the last quantum point, which
            # put the sampler's mode somewhere the fit's was not.
            d = (xx - _kabsch_align(target, xx)).ravel()
            return float(0.5 * np.dot(d, H @ d))
        dx = (xx - self._orca_ref_coords).ravel()
        if self.optimizer.objective_mode != "joint":
            # The split step lets the quantum surface act only on the null
            # space of its stacked Jacobian; so does its posterior.
            dx = self._split_energy_projector() @ dx
        return float(self.quantum.energy + np.dot(g - self._rto_shift(dx.size), dx)
                     + 0.5 * np.dot(dx, H @ dx))

    def _split_energy_projector(self):
        """The split objective's null-space projector at the fitted geometry,
        cached on it (a spectral Jacobian per call otherwise)."""
        key = np.asarray(self.coords, dtype=float)
        cached = getattr(self, "_split_proj_cache", None)
        if cached is not None and np.array_equal(cached[0], key):
            return cached[1]
        J, _ = self.spectral.stacked(key)
        rows = [r for r in (self._internal_prior_rows(key),) if r is not None]
        if self.internal_prior is not None and self.coordinate_mode == "cartesian":
            Jp, _rp = self.internal_prior.stacked(key)
            rows.append(np.sqrt(float(self.prior_weight)) * np.asarray(Jp, dtype=float))
        Pn = self._split_null_projector(key, J, rows)
        self._split_proj_cache = (key.copy(), Pn)
        return Pn

    def _projected_reference_terms(self):
        """The reference gradient and Hessian, rigid-projected, cached.

        `_project_quantum_terms` builds the rigid-mode projector every call,
        and that is an SVD: measured at 146 microseconds, against 591 for the
        whole of `log_probability`. A Metropolis chain evaluates
        log_probability once per proposal -- 16,000 times for four chains of
        4,000 steps -- so paying for the same SVD each time cost 26% of the
        sampler's runtime for an answer that cannot change while it runs.

        The inputs are the reference gradient and Hessian, which change only
        when a quantum update replaces them (and every update clears this
        cache), and `self.coords`, which moves during the fit. So the cache
        is validated against the coordinates it was built at rather than
        assumed. np.array_equal on 3N floats costs about a microsecond
        against the 146 it saves.
        """
        key = np.asarray(self.coords, dtype=float)
        cached = getattr(self, "_proj_ref_terms_cache", None)
        if cached is not None:
            coords_at, g_c, H_c = cached
            if coords_at.shape == key.shape and np.array_equal(coords_at, key):
                return g_c, H_c
        g, H = self._project_quantum_terms(
            self.quantum.gradient, self.quantum.hessian)
        self._proj_ref_terms_cache = (key.copy(), g, H)
        return g, H

    def gauss_newton_discrepancy(self, coords=None, n_directions=3,
                                 step_frac=0.1, rel_warn=0.5,
                                 _blocks=None):
        """Is the Gauss-Newton Hessian a valid stand-in for the real one here?

        `laplace_approximation` builds JtJ, which is the Gauss-Newton form of
        Hess(0.5*chi^2). The exact Hessian is JtJ + sum_i r_i Hess(r_i), and
        Gauss-Newton drops the second term. That is standard and harmless when
        the residuals are small -- but this engine is not built for small
        residuals. It runs with an uncorrected B0-versus-Be offset against
        sigma_constants quoted in fractions of a MHz, so the residuals are
        large by design: 18 sigma rms on water at the default settings.

        Measured at water's starting geometry (r_e against raw B0, 18 sigma
        rms), the omitted term is 3.0% of ||JtJ|| in norm, which sounds
        ignorable, and is not: the softest physical curvature is +7,995,
        0.04% of ||JtJ||, and the 3% perturbation flips its sign, to -372,388.
        That number was once quoted as the optimum's, because the fit then
        never left its start; at the fitted point of that one-isotopologue fit
        the two agree within 10% (7,995 against 8,758). A genuine stationary
        saddle needs the symmetry left off a symmetric molecule: H2O + D2O
        with C1 forced gives exact -1,087 at the mode, where the posterior is
        improper along the antisymmetric stretch and no sampler can converge.
        gauss_newton_warnings tells these cases apart.

        Gauss-Newton hides all of this, reporting an all-positive curvature
        and a finite width, with nothing to say it is resting on an
        approximation that has failed. This method is what says so.

        The comparison is directional on purpose. A full exact Hessian costs
        O(n^2) residual evaluations; a stiff direction cannot be flipped by a
        small perturbation anyway, so only the soft end of the spectrum is
        worth probing, at two extra evaluations each. On water the softest
        direction turns out to have essentially zero Gauss-Newton curvature
        from the spectral data (-2.5e-11): it is held up entirely by the
        quantum prior, and the omitted term is 48 times that prior.

        Returns a list of per-direction dicts, softest first, each with the
        Gauss-Newton and exact curvature, the omitted term, whether the sign
        flipped, and whether the finite difference itself was stable. An
        empty list means the check could not be run.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        if _blocks is not None:
            # Handed down by laplace_approximation, which has just built both.
            # Rebuilding them here would mean a second spectral Jacobian for
            # no new information.
            H_spec, H_quant = _blocks
        else:
            # H_quant here is every non-spectral block of the objective --
            # the quantum prior, and the internal or isotropic prior when the
            # fit had one. All are exactly quadratic in the step's own model,
            # so the omitted term this check measures is the spectral one.
            H_spec, H_quant, _parts = self._posterior_blocks(x)
        H_gn = H_spec + H_quant
        n = H_gn.shape[0]

        # The physical block's own eigenvectors are the directions probed, so
        # the posterior width along each is 1/sqrt(eigenvalue + ridge) exactly
        # -- the same 1e-8 ridge laplace_approximation defaults to. This used
        # to come from an inverse of the full 3N matrix, which at tight sigma
        # is round-off: measured 4.965e-3 against the exact 5.645e-9 on water
        # at 0.02 MHz, so the finite-difference step was 4.96e-4 Angstrom
        # instead of its 1e-7 floor.
        basis = self.physical_basis(coords=x, n=n)
        if basis.shape[1] == 0:
            return []

        block = basis.T @ H_gn @ basis
        w, V = np.linalg.eigh((block + block.T) / 2.0)   # ascending: softest first

        x0 = x.ravel()

        def half_chi_sq(vec):
            # Residuals only: this probe never looks at the Jacobian, and
            # building it (analytic by default, per isotopologue) costs far
            # more than the residuals. Skipping it makes the whole check
            # cheaper than the Laplace call it qualifies.
            _J, r = self.spectral.stacked(vec.reshape(-1, 3), want_jacobian=False)
            return 0.5 * float(np.sum(np.asarray(r) ** 2))

        f0 = half_chi_sq(x0)
        out = []
        for k in range(min(int(n_directions), basis.shape[1])):
            v = basis @ V[:, k]
            nv = float(np.linalg.norm(v))
            if nv <= 0.0:
                continue
            v = v / nv

            # Step from the posterior width along this direction, so the probe
            # is local in the units the problem itself sets. A fixed step
            # cannot be right for both a stiff and a soft direction.
            _prec = float(w[k]) + 1e-8
            sigma_v = float(1.0 / np.sqrt(_prec)) if _prec > 0.0 else 0.0
            step = float(np.clip(step_frac * sigma_v, 1e-7, 1e-3))

            def curv(t):
                return (half_chi_sq(x0 + t * v) - 2.0 * f0
                        + half_chi_sq(x0 - t * v)) / (t * t)

            c_true_spec = curv(step)
            c_quant = float(v @ H_quant @ v)      # proxy is exactly quadratic
            c_gn = float(w[k])
            c_true = c_true_spec + c_quant
            omitted = c_true_spec - float(v @ H_spec @ v)

            # A second step at half the size, to tell a real curvature from a
            # cancelling subtraction -- without it, finite-difference noise
            # could be reported as a sign flip. Only computed where it would
            # change what is said, so a direction with nothing to report
            # costs one probe rather than two.
            flipped = c_gn > 0.0 and c_true <= 0.0
            rel = abs(c_true - c_gn) / max(abs(c_gn), 1e-300)
            fd_stable = True
            if flipped or rel > rel_warn:
                c_half = curv(step / 2.0)
                denom = max(abs(c_true_spec), abs(c_half), 1.0)
                fd_stable = abs(c_true_spec - c_half) / denom < 0.25

            out.append({
                "index": k,
                "direction": v,
                "curvature_gauss_newton": c_gn,
                "curvature_exact": c_true,
                "curvature_spectral_gauss_newton": float(v @ H_spec @ v),
                "curvature_spectral_exact": c_true_spec,
                "curvature_quantum": c_quant,
                "omitted_term": omitted,
                "sigma_along": sigma_v,
                "step": step,
                "fd_stable": bool(fd_stable),
                "sign_flipped": bool(flipped),
                "relative_discrepancy": rel,
            })
        return out

    #: Newton distance to the mode, in Gauss-Newton widths, beyond which a
    #: curvature comparison describes a point other than the one the Laplace
    #: widths are about, and is reported as that rather than as a failure.
    _GN_MODE_DISTANCE = 2.0

    def distance_to_mode(self, coords=None, _blocks=None):
        """How far a geometry is from the mode of its own objective, in widths.

        The Newton step to the stationary point of the objective the Laplace
        describes, measured along each eigendirection of its Gauss-Newton
        precision M in that direction's own width 1/sqrt(w): max |g_i| /
        sqrt(w_i), with g the posterior gradient (_posterior_gradient). 0 at
        the mode; above 1, the geometry is further from its optimum than the
        optimum is uncertain. No quantum chemistry: the quantum part is the
        quadratic model the step solves.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        return self._mode_decomposition(np.asarray(x, dtype=float), _blocks)[0]

    def _mode_decomposition(self, x, _blocks=None):
        """(distance_to_mode, physical basis, eigenvalues, eigenvectors) of the
        Gauss-Newton precision at x -- the Gauss-Newton check reuses the
        decomposition for its relevance filter."""
        H_spec, H_prior = (_blocks if _blocks is not None
                           else self._posterior_blocks(x)[:2])
        basis = self.physical_basis(coords=x, n=H_spec.shape[0])
        M = basis.T @ (H_spec + H_prior) @ basis
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        w = np.maximum(w, 0.0) + 1e-8
        gp = V.T @ (basis.T @ self._posterior_gradient(x))
        z = float(np.max(np.abs(gp) / np.sqrt(w))) if gp.size else 0.0
        return z, basis, w, V
    #: Share of some reported bond's or angle's variance a direction must
    #: carry for a finding on it to be reported at all.
    _GN_RELEVANT_SHARE = 0.10

    def gauss_newton_warnings(self, coords=None, n_directions=3,
                              rel_warn=0.5, _blocks=None) -> list[str]:
        """Plain-language findings from :meth:`gauss_newton_discrepancy`.

        Three filters stand between a curvature gap and a warning, because a
        gap is only a Gauss-Newton failure where the Laplace widths are
        actually being taken (the item-7 checker found 7 of 9 warnings were
        false alarms without them):

        1. Is this the mode? The check assumes the geometry is the mode of
           the objective the widths describe. Measured as the Newton distance
           to that mode in Gauss-Newton widths; beyond 2 the residuals there
           are not the mode's, the omitted term with them, and the finding is
           reported as "not the mode" -- Gauss-Newton transfers to the mode
           (within 9% of the true posterior in every such case measured) and
           its widths are kept.
        2. Does any reported width depend on it? A direction carrying under
           10% of every bond's and angle's variance -- the out-of-plane motion
           of a planar molecule fitted without its symmetry, say -- is not
           reported on.
        3. A sign flip that survives both is a stationary saddle: the
           posterior is improper there and no Laplace width exists for any
           coordinate. That is recorded, and geometry_uncertainty refuses
           (returns None) rather than quote widths -- measured, they were
           0.85x and 0.12x the true posterior's on real water with C1 forced.
        """
        msgs: list[str] = []
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        self._gn_saddle = None
        try:
            rows = self.gauss_newton_discrepancy(
                coords=x, n_directions=n_directions, _blocks=_blocks)
        except Exception as exc:                 # a diagnostic must not break a fit
            self._warn_once(
                f"gauss_newton_discrepancy could not run ({exc!r}), so the "
                f"validity of the Gauss-Newton Hessian is unchecked.",
                key="gn_check_failed")
            return msgs
        flagged = [r for r in rows if r["fd_stable"]
                   and (r["sign_flipped"] or r["relative_discrepancy"] > rel_warn)]
        if not flagged:
            return msgs

        # Everything below runs only when something was flagged.
        try:
            z_mode, basis, w, V = self._mode_decomposition(x, _blocks)
        except Exception:                        # cannot tell: say what was seen
            z_mode, basis, w, V = 0.0, None, None, None
        if z_mode > self._GN_MODE_DISTANCE:
            msgs.append(
                f"the curvature check found Gauss-Newton and the exact Hessian "
                f"apart on {len(flagged)} direction(s), but this geometry is "
                f"{z_mode:.1f} Gauss-Newton widths from the mode of the "
                f"objective the widths describe (the fit stopped short of it, "
                f"or the check was asked about another point), so the gap "
                f"describes residuals the mode does not have. The Gauss-Newton "
                f"widths are kept: they transfer to the mode, the exact "
                f"curvature does not.")
            return msgs

        # Relevance: the share each flagged direction carries of the
        # variance of the bonds and angles a report quotes.
        try:
            cs = InternalCoordinateSet(x, self.elems, False)
            Bq = np.asarray(cs.active_B_matrix(x), dtype=float)
            coord_names = list(cs.active_names())
            C = basis @ (V / w) @ V.T @ basis.T
            var = np.einsum("ij,jk,ik->i", Bq, C, Bq)
        except Exception:
            Bq, var, coord_names = None, None, []
        saddle = []
        for row in flagged:
            share, which = 1.0, ""
            if Bq is not None and Bq.size:
                v = np.asarray(row["direction"], dtype=float)
                load = (Bq @ v) ** 2 / max(float(row["curvature_gauss_newton"]) + 1e-8, 1e-300)
                with np.errstate(invalid="ignore", divide="ignore"):
                    shares = np.where(var > 0, load / var, 0.0)
                k = int(np.argmax(shares))
                share, which = float(shares[k]), coord_names[k]
            if share < self._GN_RELEVANT_SHARE:
                continue
            if row["sign_flipped"]:
                saddle.append(row)
                msgs.append(
                    f"Gauss-Newton is invalid along physical direction "
                    f"{row['index']}: it reports curvature "
                    f"{row['curvature_gauss_newton']:.3e} but the exact "
                    f"Hessian gives {row['curvature_exact']:.3e}, at the mode "
                    f"of the objective ({z_mode:.1f} widths from it). The "
                    f"fitted point is a saddle of the exact posterior along "
                    f"that direction -- it carries {100 * share:.0f}% of "
                    f"{which}'s variance -- so no Laplace width exists for any "
                    f"coordinate here, and geometry_uncertainty returns None."
                    + self._symmetry_hint(x))
            else:
                msgs.append(
                    f"Gauss-Newton is doubtful along physical direction "
                    f"{row['index']}: curvature "
                    f"{row['curvature_gauss_newton']:.3e} against an exact "
                    f"{row['curvature_exact']:.3e} "
                    f"({100 * row['relative_discrepancy']:.0f}% discrepancy), "
                    f"because the term Gauss-Newton omits is "
                    f"{row['omitted_term']:.3e}. That direction carries "
                    f"{100 * share:.0f}% of {which}'s variance; treat that "
                    f"width as order-of-magnitude.")
        if saddle:
            self._gn_saddle = (x.tobytes(), msgs[-1])
        return msgs

    def _symmetry_hint(self, x):
        """Name the symmetry a saddle's geometry has, if more than is imposed."""
        try:
            from backend.symmetry import detect_symmetry
            found, _R = detect_symmetry(list(self.elems), np.asarray(x, dtype=float))
        except Exception:
            return ""
        imposed = None if self.symmetry is None else self.symmetry.point_group
        if not found or found == "C1" or found == imposed:
            return ""
        return (f" The geometry satisfies {found} while "
                f"{imposed or 'no symmetry'} is imposed: with symmetry='auto' "
                f"(the default) that direction is fixed by the constraint and "
                f"the posterior is proper.")

    def physical_covariance(self, cov, coords=None):
        """Strip the rigid-body subspace from a Cartesian posterior covariance.

        Translations and rotations are null directions of both blocks of the
        objective: moving or turning the whole molecule changes no rotational
        constant and no energy. An inverse of the full 3N matrix nevertheless
        hands them a finite width, and that width is enormous, because it
        comes from the quantum block's eigenvalue floor (alpha_q x 1e-8 x
        lambda_max) rather than anything the data said. Reporting it as a
        parameter uncertainty is not conservative, it is wrong by orders of
        magnitude -- measured on water: sigma = 112 Angstrom on a 0.958
        Angstrom bond, a factor of 13,400, while the physical widths are
        8.3e-3 Angstrom.

        laplace_approximation no longer produces such a matrix: it inverts
        over the physical directions only, so its covariance passes through
        here unchanged. The projection stays for covariances built any other
        way, and because the indefiniteness guard in geometry_uncertainty is
        defined on what survives it.

        Returns ``(cov_phys, basis, n_gauge)``. ``cov_phys`` is P C P, which is
        singular by construction: the gauge directions are exactly zero rather
        than merely small. ``basis`` spans the directions that survive, so
        rank and condition number can be reported over the block that means
        something instead of over a matrix whose nullity is structural.

        ONE IMPLEMENTATION. geometry_uncertainty projected here first and
        uncertainty_v2 did not, which is why the v2 report printed the 112.
        Both now call this.
        """
        cov = np.asarray(cov, dtype=float)
        n = cov.shape[0]
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        if self._rigid_ref_masses is None or x.size != n:
            return cov, np.eye(n), 0

        basis = self.physical_basis(coords=x, n=n)
        # Project with the basis itself rather than the rigid projector alone,
        # so the symmetry-forbidden directions go too. Without a symmetry
        # constraint this is the same matrix: the rigid projector is
        # orthogonal, so basis @ basis.T reproduces it exactly.
        Q = basis @ basis.T
        cov_phys = Q @ cov @ Q
        return cov_phys, basis, int(n - basis.shape[1])

    def physical_basis(self, coords=None, n=None):
        """Orthonormal basis for the directions that are not rigid-body.

        Columns span the complement of the translations and rotations: 3N - 6
        of them for a non-linear molecule, 3N - 5 for a linear one. Anything
        that needs to work in physical directions only -- reporting a width,
        proposing an MCMC step -- takes its basis from here, so there is one
        definition of "physical" rather than one per caller.

        Taken from the projector's own spectrum rather than from a threshold
        on some projected matrix: P is an orthogonal projector, so its
        eigenvalues are exactly 0 and 1 up to round-off, and a cut at 0.5
        cannot land on anything ambiguous.

        Cached on everything it depends on -- the geometry, the reference
        masses and the imposed symmetry's projector -- because one
        uncertainty call asks for it three times at the same point (the
        Laplace inverse, the Gauss-Newton check, the projection the
        indefiniteness guard runs on). The array returned is read-only, so a
        caller cannot corrupt the cached copy by writing into it.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        size = int(x.size if n is None else n)
        if self._rigid_ref_masses is None or x.size != size:
            return np.eye(size)
        Pi = self._symmetry_projector(size)
        key = (size, x.tobytes(), np.asarray(self._rigid_ref_masses).tobytes(),
               None if Pi is None else Pi.tobytes())
        cached = getattr(self, "_physical_basis_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        P = self._rigid_mode_projector(x, self._rigid_ref_masses)
        w, V = np.linalg.eigh((P + P.T) / 2.0)
        V = V[:, w > 0.5]
        basis = np.array(self._restrict_to_symmetric(V), dtype=float)
        basis.setflags(write=False)
        self._physical_basis_cache = (key, basis)
        return basis

    def _symmetry_projector(self, size):
        """The imposed point group's orthogonal projector, or None.

        Only a constraint the fit actually enforces counts. ``symmetrize`` is
        applied after every accepted step, so the geometry can never leave
        this subspace; on centroid-preserving displacements -- which is all
        that survives the rigid-body projection -- its Jacobian is exactly this
        matrix.
        """
        sym = getattr(self, "symmetry", None)
        Pi = getattr(sym, "_projection", None) if sym is not None else None
        if Pi is None:
            return None
        Pi = np.asarray(Pi, dtype=float)
        if Pi.shape != (size, size):
            return None
        return Pi

    def _restrict_to_symmetric(self, V):
        """Keep only the columns of ``V`` the imposed symmetry allows.

        WHY THIS EXISTS. With a point group imposed, the fit symmetrises after
        every step, so a symmetry-breaking direction is not a parameter of the
        model at all -- it is fixed, exactly, by the constraint. Reporting a
        width along it describes a model nobody fitted.

        It was not a cosmetic leak. On water under C2v the one such direction
        is the antisymmetric stretch, and rotational constants are even in it,
        so at a symmetric geometry the data has no first-order sensitivity to
        it at all: J v = 1.3e-11 against 7.8e3 and 1.3e4 on the two allowed
        directions. Gauss-Newton then gave it curvature 1.90 (the quantum prior
        alone), the Laplace reported a bond-length sigma of 0.775 Angstrom on a
        0.956 Angstrom bond, and the Gauss-Newton check fired a 26,012%
        discrepancy -- all of it sum_i r_i d2r_i = 571.48 on this one
        direction. Restricted to the two directions the fit can move along,
        Gauss-Newton and the exact Hessian agree to 2.4% and the bond sigma is
        0.0082 Angstrom.

        An exact-Hessian covariance would have been the wrong repair: it gives
        that direction a finite width (0.042 Angstrom), which is just as
        meaningless for a coordinate the model holds fixed.

        Both projectors are orthogonal and they commute, so ``V^T Pi V`` has
        eigenvalues of exactly 0 and 1 -- measured {0, 1, 1} on water to 1e-17
        -- and the 0.5 cut cannot land on anything ambiguous.
        """
        V = np.asarray(V, dtype=float)
        if V.ndim != 2 or V.shape[1] == 0:
            return V
        Pi = self._symmetry_projector(V.shape[0])
        if Pi is None:
            return V
        M = V.T @ Pi @ V
        w, U = np.linalg.eigh((M + M.T) / 2.0)
        return V @ U[:, w > 0.5]

    def projected_direction_counts(self, coords=None, n=None):
        """How many directions the reported posterior leaves out, and why.

        Returns ``(n_rigid, n_symmetry)``: rigid-body directions (translations
        and rotations, null in every block of the objective) and directions
        the imposed point group forbids. Kept apart because they mean
        different things -- one is gauge freedom, the other is a modelling
        constraint -- and a report that lumped them would call a symmetry
        assumption a gauge choice.
        """
        x = self.coords if coords is None else np.asarray(coords, dtype=float)
        x = np.asarray(x, dtype=float)
        size = int(x.size if n is None else n)
        if self._rigid_ref_masses is None or x.size != size:
            return 0, 0
        P = self._rigid_mode_projector(x, self._rigid_ref_masses)
        w, V = np.linalg.eigh((P + P.T) / 2.0)
        V = V[:, w > 0.5]
        n_rigid = size - V.shape[1]
        n_symmetry = V.shape[1] - self._restrict_to_symmetric(V).shape[1]
        return int(n_rigid), int(n_symmetry)
