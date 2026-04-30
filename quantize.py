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

from spectral import SpectralEngine, sanitize_isotopologues
from quantum import (
    QuantumEngine,
    parse_engrad,
    parse_orca_rovib_alpha,
    _detect_bonds,
    _detect_angles,
    wilson_B,
)
from SVD import SubspaceOptimizer


def _find_orca(executable):
    """
    Resolve the ORCA executable to an absolute path.
    Accepts a full path, a bare name ('orca'), or None (auto-detect).
    Raises RuntimeError if not found.
    """
    if executable is None:
        executable = "orca"
    found = shutil.which(executable)
    if found:
        return found
    if os.path.isfile(executable):
        return os.path.abspath(executable)
    raise RuntimeError(
        f"ORCA executable '{executable}' not found on PATH or filesystem.\n"
        "Set orca_executable to the full path, e.g. r'C:\\orca\\orca.exe'."
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
        [Hartree]. Used only when quantum prior is enabled.
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
        spectral_delta=1e-3,
        auto_sanitize_spectral=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        use_internal_preconditioner=False,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        method_preset=None,
        use_orca_rovib=False,
        rovib_recalc_every=1,
        spectral_only=False,
        debug_rank_diagnostics=False,
        debug_sv_count=6,
    ):
        self.coords = np.asarray(coords, dtype=float).copy()
        self.elems = list(elems)
        if method_preset is not None:
            preset_method, preset_basis = self._method_preset(method_preset)
            orca_method = preset_method
            orca_basis = preset_basis
        self.auto_sanitize_spectral = bool(auto_sanitize_spectral)
        self.sanitize_jacobian_row_norm_max = float(sanitize_jacobian_row_norm_max)
        self.sanitize_tiny_target_mhz = float(sanitize_tiny_target_mhz)
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

        self.spectral_only = bool(spectral_only)
        self.debug_rank_diagnostics = bool(debug_rank_diagnostics)
        self.debug_sv_count = max(1, int(debug_sv_count))
        self.quantum = None
        self._psi4_engine = None
        self._orca_ref_coords = None
        self._orca_call_count = 0
        self.history = []

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
                    from Psi4 import Psi4Engine  # pylint: disable=import-outside-toplevel
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

    def _err_path(self):
        return os.path.join(self.workdir, "quantize_orca.err")

    # ── ORCA input generation ─────────────────────────────────────────────────

    def _write_orca_input(self, job="hessian"):
        if job == "hessian":
            keyword = "NumFreq EnGrad"
        elif job == "gradient":
            keyword = "EnGrad"
        elif job == "rovib":
            keyword = "AnFreq"
        else:
            raise ValueError(f"Unknown ORCA job type: {job}")
        method_line = f"{self.orca_method} {self.orca_basis}".strip()
        lines = [f"! {method_line} TightSCF {keyword}"]
        if job == "hessian":
            lines += ["%freq", "  Temp 298.15", "end"]
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
        result = subprocess.run(
            [self._orca_exe, self._inp_path()],
            capture_output=True,
            text=True,
            cwd=self.workdir,
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
        energy, grad = parse_engrad(self._engrad_path())
        self.quantum.energy = energy
        self.quantum._gradient_bohr = grad
        self._orca_ref_coords = self.coords.copy()
        print(f"  [ORCA] Done.  Energy = {energy:.10f} Hartree")

    def _run_rovib(self):
        """
        Optional ORCA anharmonic run to extract alpha(A/B/C) and populate
        isotopologue alpha_constants by selected rotational components.
        """
        if self.quantum_backend != "orca":
            return
        print("  [ORCA] Running rovibrational correction calculation (AnFreq)...")
        self._write_orca_input(job="rovib")
        self._exec_orca()
        alpha_abc = parse_orca_rovib_alpha(self._out_path())
        if not np.isfinite(alpha_abc).any():
            print("  [ORCA] Warning: could not parse alpha constants from AnFreq output; keeping existing alpha_constants.")
            return
        for iso in self.spectral.isotopologues:
            idx = np.asarray(iso["component_indices"], dtype=int)
            alpha = np.zeros(len(idx), dtype=float)
            for i, comp in enumerate(idx):
                if 0 <= comp < 3 and np.isfinite(alpha_abc[comp]):
                    alpha[i] = alpha_abc[comp]
            iso["alpha_constants"] = alpha
        print("  [ORCA] Updated isotopologue alpha_constants from AnFreq output.")

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

    # ── Optimisation loop ─────────────────────────────────────────────────────

    def run(self):
        """
        Run the hybrid optimisation loop.

        Returns
        -------
        coords : (N, 3) ndarray   Final optimised coordinates in Angstroms.
        """
        header = (
            f"{'Iter':>5}  {'|Δx| Å':>12}  {'RMS_w':>12}  {'RMS MHz':>12}  "
            f"{'Rank':>6}  {'σ_kept':>12}  {'|Δx_r|':>10}  {'|Δx_n|':>10}  "
            f"{'|g_n|':>10}  {'α_q':>8}  {'|ΔE| Eh':>12}"
        )
        print("\n" + header)
        print("─" * len(header))
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

            J, residual_w = self.spectral.stacked(self.coords)
            _, residual_mhz = self.spectral.stacked_unweighted(self.coords)
            B, _ = wilson_B(self.coords, self.elems)

            wrms_before = float(np.sqrt(np.mean(residual_w ** 2)))
            mhz_rms_before = float(np.sqrt(np.mean(residual_mhz ** 2)))
            dx, rank, sv, alpha_q_eff = self.optimizer.step(J, residual_w, g, H, B=B)
            _, _, Vt, _ = self.optimizer.decompose(J)
            P_range = self.optimizer.range_projector(Vt, rank)
            P_null = self.optimizer.null_projector(Vt, rank)
            dx_range_norm = float(np.linalg.norm(P_range @ dx))
            dx_null_norm = float(np.linalg.norm(P_null @ dx))
            g_null_norm = None if self.spectral_only else float(np.linalg.norm(P_null @ g))
            trial_coords = self.coords + dx.reshape(-1, 3)
            _, residual_w_trial = self.spectral.stacked(trial_coords)
            _, residual_mhz_trial = self.spectral.stacked_unweighted(trial_coords)
            wrms_after = float(np.sqrt(np.mean(residual_w_trial ** 2)))
            mhz_rms_after = float(np.sqrt(np.mean(residual_mhz_trial ** 2)))

            # Accept based on the weighted objective used to construct the step.
            accepted = wrms_after <= wrms_before
            if accepted:
                self.coords = trial_coords
                _reject_streak = 0
            else:
                _reject_streak += 1
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
                )
            )
            status = "ok" if accepted else "rej"
            dE_str = f"{delta_energy:>12.3e}" if delta_energy is not None else f"{'n/a':>12}"
            gnull_str = f"{g_null_norm:>10.3e}" if g_null_norm is not None else f"{'n/a':>10}"
            print(
                f"{it+1:>5}  {step_norm:>12.4e}  {wrms:>12.4f}  {freq_rms:>12.4f}  "
                f"{rank:>6d}  {sv_kept:>12.4e}  {dx_range_norm:>10.3e}  {dx_null_norm:>10.3e}  "
                f"{gnull_str}  {alpha_q_eff:>8.3f}  {dE_str}  "
                f"λ={self.optimizer.lambda_damp:.2e} {status}"
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
                    g_null_norm is not None and g_null_norm < self.conv_grad_null and
                    energy_ok
                )
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
                        f"(freq RMS = {freq_rms:.4f} MHz, |ΔE| < {self.conv_energy:.1e} Eh)."
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

        print("\n" + "═" * 52)
        print("  Final Structure Report")
        print("═" * 52)
        print(f"  Experimentally constrained directions : {rank}")
        print(f"  Theory-filled null-space directions   : {n_null}")
        print(f"  Largest singular value                : {s[0]:.6e}")
        if rank:
            print(f"  Smallest retained singular value      : {s[rank-1]:.6e}")
        print(f"  Jacobian condition estimate           : {cond:.6e}")
        print(f"  Mean parameter uncertainty (arb.)     : {np.mean(param_std):.6e}")
        print()
        print("  Rotational constant residuals (MHz)")
        print(f"  {'Iso':>4}  {'Const':>5}  {'Target':>12}  {'Calc':>12}  {'Δ':>10}")
        print("  " + "─" * 50)
        labels = ["A", "B", "C"]
        for k, iso in enumerate(self.spectral.isotopologues):
            calc_all = self.spectral.rotational_constants(self.coords, iso["masses"])
            idx = iso["component_indices"]
            calc = calc_all[idx]
            target = iso["obs_constants"] + 0.5 * iso["alpha_constants"]
            for i, comp in enumerate(idx):
                lbl = labels[int(comp)] if 0 <= int(comp) < len(labels) else f"R{int(comp)}"
                print(
                    f"  {k+1:>4}  {lbl:>5}  {target[i]:>12.4f}  "
                    f"{calc[i]:>12.4f}  {target[i]-calc[i]:>10.4f}"
                )

        print()
        print("  Bond lengths (Å)")
        print(f"  {'Bond':>10}  {'Length':>10}")
        print("  " + "─" * 24)
        bonds = _detect_bonds(self.coords, self.elems)
        for i, j in bonds:
            d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
            print(f"  {self.elems[i]}{i+1}-{self.elems[j]}{j+1}:{'':>5}  {d:>10.6f}")

        angles = _detect_angles(bonds)
        if angles:
            print()
            print("  Bond angles (°)")
            print(f"  {'Angle':>14}  {'Degrees':>10}")
            print("  " + "─" * 28)
            for i, j, k in angles:
                v1 = self.coords[i] - self.coords[j]
                v2 = self.coords[k] - self.coords[j]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                print(f"  {self.elems[i]}{i+1}-{self.elems[j]}{j+1}-{self.elems[k]}{k+1}:{'':>5}  {deg:>10.4f}")

        print("═" * 52)

        return residual
