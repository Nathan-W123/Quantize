"""
ORCA quantum chemistry backend.

Encapsulates all ORCA subprocess management, input-file generation, output
parsing, and VPT2 rovibrational calculations that previously lived inside
``MolecularOptimizer`` in ``backend/quantize.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from backend.base_backend import GradientResult, HessianResult, QuantumBackend, RovibResult
from backend.quantum import QuantumEngine, parse_engrad, parse_orca_rovib
from backend.rovib_cache import (
    load_cached_correction,
    make_rovib_cache_key,
    save_cached_correction,
)
from backend.rovib_corrections import resolve_alpha_components
from backend.registry import register_backend


# ── Executable discovery ──────────────────────────────────────────────────────

def _find_orca(executable):
    """
    Resolve the ORCA executable to an absolute path.

    Search order: PATH (:func:`shutil.which`), explicit path if it exists as a
    file, then ``./orca`` or ``./orca.exe`` in the current working directory.
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


# ── Backend class ─────────────────────────────────────────────────────────────

@register_backend
class OrcaBackend(QuantumBackend):
    """ORCA-based quantum chemistry computations (gradients, Hessians, VPT2)."""

    name = "orca"
    supports_parallel = False

    def __init__(
        self,
        elems,
        workdir,
        method,
        basis,
        charge,
        multiplicity,
        executable=None,
        nprocs=1,
        rovib_source_mode="hybrid_auto",
    ):
        self.elems = list(elems)
        self.workdir = os.path.abspath(workdir)
        self.method = str(method).strip()
        self.basis = str(basis).strip()
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.nprocs = int(nprocs)
        self.rovib_source_mode = str(rovib_source_mode).strip().lower()
        self._exe = None
        try:
            self._exe = _find_orca(executable)
            print(f"ORCA found: {self._exe}")
        except RuntimeError as e:
            self._exe = None
            print(f"Note: {e}\nCall load_orca() to use pre-computed files.")

    # ── File paths ─────────────────────────────────────────────────────────────

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

    def _iso_rovib_inp_path(self, label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(label))
        return os.path.join(self.workdir, f"quantize_orca_rovib_{safe}.inp")

    def _iso_rovib_out_path(self, label):
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(label))
        return os.path.join(self.workdir, f"quantize_orca_rovib_{safe}.out")

    # ── Input generation ───────────────────────────────────────────────────────

    def _write_input(self, coords, job="hessian"):
        if job == "hessian":
            keyword = "Freq EnGrad"
        elif job == "gradient":
            keyword = "EnGrad"
        elif job == "rovib":
            keyword = "VPT2"
        else:
            raise ValueError(f"Unknown ORCA job type: {job}")
        method_line = f"{self.method} {self.basis}".strip()
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
        lines += ["%pal", "  nprocs 1", "end"]
        lines += [f"* xyz {self.charge} {self.multiplicity}"]
        for elem, (x, y, z) in zip(self.elems, coords):
            lines.append(f"  {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
        lines.append("*\n")
        os.makedirs(self.workdir, exist_ok=True)
        with open(self._inp_path(), "w") as f:
            f.write("\n".join(lines))

    def _write_rovib_input_for_iso(self, coords, iso_masses, label):
        """Write a VPT2 input with per-atom mass overrides for isotopologue substitution."""
        method_line = f"{self.method} {self.basis}".strip()
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
        for elem, (x, y, z) in zip(self.elems, coords):
            lines.append(f"  {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}")
        lines.append("*")
        lines.append("%coords")
        lines.append("  CTyp xyz")
        lines.append(f"  Charge {self.charge}")
        lines.append(f"  Mult {self.multiplicity}")
        lines.append("  Coords")
        for (elem, (x, y, z)), m in zip(zip(self.elems, coords), masses):
            lines.append(
                f"    {elem:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}  M = {float(m):.8f}"
            )
        lines.append("  end")
        lines.append("end\n")
        os.makedirs(self.workdir, exist_ok=True)
        with open(self._iso_rovib_inp_path(label), "w") as f:
            f.write("\n".join(lines))

    # ── Execution ──────────────────────────────────────────────────────────────

    def _exec(self):
        if self._exe is None:
            raise RuntimeError(
                "ORCA executable not found.  Provide orca_executable= or call load_orca()."
            )
        env = os.environ.copy()
        orca_dir = os.path.dirname(os.path.abspath(self._exe))
        if orca_dir not in env.get("PATH", ""):
            env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
        workdir = os.path.abspath(self.workdir)
        inp_rel = os.path.basename(self._inp_path())
        result = subprocess.run(
            [self._exe, inp_rel],
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

    def _exec_named(self, inp_path, out_path):
        """Run ORCA on a specific input and capture its output to ``out_path``."""
        if self._exe is None:
            raise RuntimeError(
                "ORCA executable not found.  Provide orca_executable= or call load_orca()."
            )
        env = os.environ.copy()
        orca_dir = os.path.dirname(os.path.abspath(self._exe))
        if orca_dir not in env.get("PATH", ""):
            env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
        workdir = os.path.abspath(self.workdir)
        inp_rel = os.path.basename(inp_path)
        result = subprocess.run(
            [self._exe, inp_rel],
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

    def _require_artefacts(self, need=("engrad", "hess")):
        """Fail fast if expected ORCA output files are missing after a run."""
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

    # ── QuantumBackend interface ───────────────────────────────────────────────

    def run_hessian(self, coords_ang: np.ndarray) -> HessianResult:
        """Full Freq job: returns energy, gradient, and Hessian."""
        print("  [ORCA] Running frequency calculation (gradient + Hessian)...")
        self._write_input(coords_ang, job="hessian")
        self._exec()
        self._require_artefacts(need=("engrad", "hess"))
        eng = QuantumEngine(self._engrad_path(), self._hess_path(), self.elems)
        print(f"  [ORCA] Done.  Energy = {eng.energy:.10f} Hartree")
        return HessianResult(
            energy=eng.energy,
            gradient_bohr=eng._gradient_bohr.copy(),
            hessian_bohr=eng._hessian_bohr.copy(),
        )

    def run_gradient(self, coords_ang: np.ndarray) -> GradientResult:
        """Cheap EnGrad job: returns energy and gradient only."""
        print("  [ORCA] Running gradient update (EnGrad)...")
        self._write_input(coords_ang, job="gradient")
        self._exec()
        self._require_artefacts(need=("engrad",))
        energy, grad_bohr = parse_engrad(self._engrad_path())
        print(f"  [ORCA] Done.  Energy = {energy:.10f} Hartree")
        return GradientResult(energy=energy, gradient_bohr=np.asarray(grad_bohr, dtype=float).ravel())

    def run_rovib(
        self,
        coords_ang: np.ndarray,
        isotopologues: list[dict] | None = None,
    ) -> RovibResult | None:
        """VPT2 anharmonic corrections for all isotopologues.

        Dispatches to isotopologue-specific runs when ``rovib_source_mode`` is
        ``"orca_vpt2_isotopologue_specific"``; otherwise runs a single shared
        VPT2 job and broadcasts the result.

        Returns ``None`` if no corrections could be extracted.
        """
        if not isotopologues:
            return None
        if self.rovib_source_mode == "orca_vpt2_isotopologue_specific":
            return self._run_rovib_iso_specific(coords_ang, isotopologues)
        return self._run_rovib_shared(coords_ang, isotopologues)

    def run_cheap_opt(self, coords_ang: np.ndarray) -> np.ndarray | None:
        """Fast geometry pre-optimisation using ORCA with a lightweight method."""
        try:
            from backend.orca_cheap_opt import minimize_geometry_cheap_orca  # pylint: disable=import-outside-toplevel
            return minimize_geometry_cheap_orca(
                coords=coords_ang,
                elems=self.elems,
                charge=self.charge,
                multiplicity=self.multiplicity,
                orca_exe=self._exe,
                workdir=os.path.join(self.workdir, "cheap_opt"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [ORCA] cheap_opt failed: {exc}")
            return None

    # ── Rovib helpers ──────────────────────────────────────────────────────────

    def _run_rovib_shared(self, coords_ang, isotopologues):
        """Single VPT2 job; broadcast alpha to all isotopologues."""
        print("  [ORCA] Running rovibrational correction calculation (VPT2)...")
        self._write_input(coords_ang, job="rovib")
        self._exec()
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
            return None

        parent_masses = (
            np.asarray(isotopologues[0]["masses"], dtype=float) if isotopologues else None
        )
        mode_norm = self.rovib_source_mode
        strict_mode = mode_norm.startswith("strict_")
        corrections = []
        for iso_idx, iso in enumerate(isotopologues):
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
                    method=self.method,
                    basis=self.basis,
                    backend="orca",
                )
            except ValueError as e:
                if strict_mode:
                    raise RuntimeError(
                        f"Strict rovib mode failed for isotopologue '{label}': {e}"
                    ) from e
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
            if parsed.parse_status:
                correction.status = str(parsed.parse_status)
            if correction.status == "ok" and correction.warnings:
                correction.status = "partial"
            corrections.append({"name": label, "alpha_constants": resolved, "rovib_correction": correction})
        print(f"  [ORCA] Updated isotopologue alpha_constants using mode={self.rovib_source_mode}.")
        return RovibResult(isotopologue_corrections=corrections)

    def _run_rovib_iso_specific(self, coords_ang, isotopologues):
        """One VPT2 job per isotopologue with mass overrides."""
        mode_norm = self.rovib_source_mode
        strict_mode = mode_norm.startswith("strict_")
        corrections = []
        for iso in isotopologues:
            label = str(iso.get("name", "iso"))
            masses = np.asarray(iso["masses"], dtype=float)
            cache_key = make_rovib_cache_key(
                coords_ang,
                masses,
                self.method,
                self.basis,
                "orca",
                self.rovib_source_mode,
            )
            cached = load_cached_correction(self.workdir, cache_key, label)
            parsed_alpha = None
            warnings_list: list[str] = []
            run_status = "unknown"
            if cached is not None:
                parsed_alpha = cached.alpha_vector()
                warnings_list = list(cached.warnings or [])
                run_status = str(cached.status or "ok")
                print(f"  [ORCA] Cache hit for isotopologue '{label}'.")
            else:
                print(f"  [ORCA] Running VPT2 for isotopologue '{label}' (mass-overridden)...")
                self._write_rovib_input_for_iso(coords_ang, masses, label)
                inp = self._iso_rovib_inp_path(label)
                outp = self._iso_rovib_out_path(label)
                try:
                    self._exec_named(inp, outp)
                    parsed = parse_orca_rovib(outp)
                    parsed_alpha = parsed.alpha_abc
                    warnings_list = list(parsed.warnings)
                    run_status = str(parsed.parse_status or "unknown")
                    if parsed.parse_status == "parse_failed":
                        print(
                            f"  [ORCA] Warning: VPT2 parse failed for '{label}'; "
                            "falling back to existing alpha."
                        )
                except Exception as exc:  # noqa: BLE001
                    warnings_list.append(f"VPT2 run failed: {exc}")
                    run_status = "vpt2_failed"
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
                    method=self.method,
                    basis=self.basis,
                    backend="orca",
                )
            except ValueError as e:
                if strict_mode:
                    raise RuntimeError(
                        f"Strict rovib mode failed for isotopologue '{label}': {e}"
                    ) from e
                print(f"  [ORCA] Strict mode rejected isotopologue '{label}': {e}")
                continue

            correction.warnings = list(correction.warnings) + warnings_list
            if run_status not in ("", "unknown", None):
                correction.status = run_status
            if correction.status == "ok" and correction.warnings:
                correction.status = "partial"
            correction.geometry_hash = cache_key
            if cached is None and parsed_alpha is not None and np.isfinite(parsed_alpha).any():
                try:
                    save_cached_correction(self.workdir, cache_key, label, correction)
                except OSError:
                    pass
            corrections.append({"name": label, "alpha_constants": resolved, "rovib_correction": correction})
        return RovibResult(isotopologue_corrections=corrections)
