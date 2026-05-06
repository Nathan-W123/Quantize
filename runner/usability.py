from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from backend.spectral_model import normalize_spectral_model

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled by load_config error path
    yaml = None


COMPONENT_LABELS = ("A", "B", "C")
VALID_PRESETS = {"FAST_DEBUG", "BALANCED", "STRICT"}
VALID_BACKENDS = {"orca", "psi4", "none"}
VALID_GEOMETRY_METHODS = {"bonds", "pubchem", "coords"}
_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ConfigError(ValueError):
    """Raised when a Quantize config is invalid."""


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a Quantize YAML or JSON config file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError("PyYAML is required to read YAML configs. Install PyYAML.")
            data = yaml.safe_load(text)
        else:
            raise ConfigError("Config file must end in .yaml, .yml, or .json.")
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config {cfg_path} must contain a mapping/object at top level.")
    return data


def _expect_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping/object.")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"'{path}' must be a list.")
    return value


def _check_numeric_list(value: Any, path: str, n: int | None = None, positive: bool = False) -> None:
    items = _as_list(value, path)
    if n is not None and len(items) != n:
        raise ConfigError(f"'{path}' must contain {n} values; got {len(items)}.")
    for i, item in enumerate(items):
        try:
            x = float(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"'{path}[{i}]' must be numeric; got {item!r}.") from exc
        if not np.isfinite(x):
            raise ConfigError(f"'{path}[{i}]' must be finite; got {item!r}.")
        if positive and x <= 0.0:
            raise ConfigError(f"'{path}[{i}]' must be positive; got {item!r}.")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate supported legacy and generalized config shapes with clear errors."""
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a mapping/object.")

    has_elements = "elements" in cfg
    has_molecule = "molecule" in cfg
    if has_elements and has_molecule:
        raise ConfigError("Set either 'elements' for a generalized run or 'molecule' for legacy mode, not both.")
    if not has_elements and not has_molecule:
        raise ConfigError("Config must set either 'elements' or 'molecule'.")

    preset = cfg.get("preset")
    if preset is not None and str(preset).strip().upper() not in VALID_PRESETS:
        raise ConfigError(f"'preset' must be one of {', '.join(sorted(VALID_PRESETS))}.")

    output = _expect_mapping(cfg, "output")
    for key in ("root", "run_dir"):
        if key in output and output[key] is not None and not str(output[key]).strip():
            raise ConfigError(f"'output.{key}' cannot be blank.")

    if has_molecule:
        molecule = str(cfg.get("molecule", "")).strip()
        if not molecule:
            raise ConfigError("'molecule' cannot be blank.")
        return

    elements = _as_list(cfg.get("elements"), "elements")
    if not elements:
        raise ConfigError("'elements' must contain at least one atom.")
    for i, elem in enumerate(elements):
        if not isinstance(elem, str) or not _ELEMENT_RE.match(elem.strip()):
            raise ConfigError(f"'elements[{i}]' must be an element symbol like C, H, O, or Cl.")
    n_atoms = len(elements)

    geometry = _expect_mapping(cfg, "geometry")
    if geometry.get("smiles"):
        pass
    else:
        method = str(geometry.get("method", "bonds")).strip().lower()
        if method not in VALID_GEOMETRY_METHODS:
            raise ConfigError("'geometry.method' must be one of bonds, pubchem, or coords.")
        if method == "bonds":
            bonds = _as_list(geometry.get("bonds"), "geometry.bonds")
            if not bonds:
                raise ConfigError("'geometry.bonds' must contain at least one [i, j] pair.")
            for b_i, pair in enumerate(bonds):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ConfigError(f"'geometry.bonds[{b_i}]' must be a two-item list.")
                for atom_i in pair:
                    if not isinstance(atom_i, int) or atom_i < 0 or atom_i >= n_atoms:
                        raise ConfigError(
                            f"'geometry.bonds[{b_i}]' contains atom index {atom_i!r}, "
                            f"but valid indices are 0..{n_atoms - 1}."
                        )
            if "bond_lengths" in geometry and geometry["bond_lengths"] is not None:
                _check_numeric_list(geometry["bond_lengths"], "geometry.bond_lengths", len(bonds), positive=True)
        elif method == "pubchem":
            if not str(geometry.get("identifier", "")).strip():
                raise ConfigError("'geometry.identifier' is required when geometry.method is pubchem.")
        elif method == "coords":
            rows = _as_list(geometry.get("coords_angstrom"), "geometry.coords_angstrom")
            if len(rows) != n_atoms:
                raise ConfigError(f"'geometry.coords_angstrom' must have {n_atoms} coordinate rows.")
            for row_i, row in enumerate(rows):
                _check_numeric_list(row, f"geometry.coords_angstrom[{row_i}]", 3)

    isotopologues = _as_list(cfg.get("isotopologues"), "isotopologues")
    if not isotopologues:
        raise ConfigError("'isotopologues' must contain at least one entry.")
    for iso_i, iso in enumerate(isotopologues):
        if not isinstance(iso, dict):
            raise ConfigError(f"'isotopologues[{iso_i}]' must be a mapping/object.")
        prefix = f"isotopologues[{iso_i}]"
        _check_numeric_list(iso.get("masses"), f"{prefix}.masses", n_atoms, positive=True)
        comps = iso.get("components", ["A", "B", "C"])
        comps = _as_list(comps, f"{prefix}.components")
        if not comps:
            raise ConfigError(f"'{prefix}.components' must contain at least one component.")
        for comp in comps:
            if str(comp).strip().upper() not in COMPONENT_LABELS:
                raise ConfigError(f"'{prefix}.components' values must be A, B, or C.")
        n_comp = len(comps)
        _check_numeric_list(iso.get("obs_b0_mhz"), f"{prefix}.obs_b0_mhz", n_comp)
        _check_numeric_list(iso.get("alpha_mhz"), f"{prefix}.alpha_mhz", n_comp)
        _check_numeric_list(iso.get("sigma_mhz"), f"{prefix}.sigma_mhz", n_comp, positive=True)

    if "spectral_model" in cfg and cfg.get("spectral_model") is not None:
        try:
            normalize_spectral_model(str(cfg.get("spectral_model")))
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    quantum = _expect_mapping(cfg, "quantum")
    backend = str(quantum.get("backend", "orca")).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError("'quantum.backend' must be one of orca, psi4, or none.")

    _validate_rovibrational_corrections_block(cfg)
    _validate_torsion_block(cfg)


_VALID_ROVIB_MODES = {
    "hybrid_auto", "user_only", "orca_only",
    "manual_alpha", "manual_delta", "none",
    "strict_user", "strict_backend",
}


def _validate_rovibrational_corrections_block(cfg: dict[str, Any]) -> None:
    """Validate the optional rovibrational_corrections: block."""
    rc = cfg.get("rovibrational_corrections")
    if rc is None:
        return
    if not isinstance(rc, dict):
        raise ConfigError("'rovibrational_corrections' must be a mapping/object.")

    mode = rc.get("mode")
    if mode is not None and str(mode).strip().lower() not in _VALID_ROVIB_MODES:
        raise ConfigError(
            f"'rovibrational_corrections.mode' must be one of "
            f"{sorted(_VALID_ROVIB_MODES)}, got '{mode}'."
        )

    correction_table = rc.get("correction_table")
    if correction_table is not None:
        p = Path(str(correction_table)).expanduser()
        if not p.is_file():
            raise ConfigError(
                f"'rovibrational_corrections.correction_table' not found: {p}"
            )
        suf = p.suffix.lower()
        if suf not in (".csv", ".yaml", ".yml"):
            raise ConfigError(
                "'rovibrational_corrections.correction_table' must be a .csv, .yaml, or .yml file."
            )

    for frac_key in ("sigma_vib_fraction", "sigma_elec_fraction"):
        v = rc.get(frac_key)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'rovibrational_corrections.{frac_key}' must be numeric."
                ) from exc
            if fv < 0.0:
                raise ConfigError(
                    f"'rovibrational_corrections.{frac_key}' must be >= 0."
                )

    elec = rc.get("electronic_correction")
    if elec is not None and not isinstance(elec, bool):
        raise ConfigError(
            "'rovibrational_corrections.electronic_correction' must be true or false."
        )

    bob = rc.get("bob_params")
    if bob is not None and not isinstance(bob, dict):
        raise ConfigError(
            "'rovibrational_corrections.bob_params' must be a mapping of element → component → u-value."
        )


_VALID_TORSION_SYMMETRY_MODES = {"c3", "3fold", "threefold", "none", "off", "null", ""}
_VALID_SCAN_ANGLE_UNITS = {"degrees", "deg", "degree", "radians", "rad", "radian"}
_VALID_SCAN_ENERGY_UNITS = {"cm-1", "cm_1", "hartree", "ha", "kcal/mol", "kcal", "kj/mol", "kj"}


def _validate_torsion_block(cfg: dict[str, Any]) -> None:
    """Validate the optional torsion_hamiltonian: block (Phase 0+1+2 fields)."""
    t = cfg.get("torsion_hamiltonian")
    if t is None:
        return
    if not isinstance(t, dict):
        raise ConfigError("'torsion_hamiltonian' must be a mapping/object.")

    # --- enabled ---
    enabled_raw = t.get("enabled")
    if enabled_raw is not None and not isinstance(enabled_raw, bool):
        raise ConfigError("'torsion_hamiltonian.enabled' must be true or false.")
    enabled = bool(enabled_raw) if enabled_raw is not None else False

    # --- units ---
    if "units" in t and str(t["units"]).strip().lower() != "cm-1":
        raise ConfigError("'torsion_hamiltonian.units' must be 'cm-1'.")

    # --- F required if enabled ---
    if enabled and "F" not in t:
        raise ConfigError("'torsion_hamiltonian.F' is required when enabled is true.")

    # --- scalar numeric fields ---
    for fkey in ("F", "rho", "F4", "F6", "c_mk", "c_k2"):
        v = t.get(fkey)
        if v is not None:
            try:
                float(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'torsion_hamiltonian.{fkey}' must be numeric.") from exc

    # --- positive integer fields ---
    for ikey in ("n_basis", "n_levels"):
        v = t.get(ikey)
        if v is not None:
            try:
                iv = int(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'torsion_hamiltonian.{ikey}' must be an integer.") from exc
            if iv < 1:
                raise ConfigError(f"'torsion_hamiltonian.{ikey}' must be >= 1.")

    # --- J_values, K_values ---
    for jkkey in ("J_values", "K_values"):
        v = t.get(jkkey)
        if v is not None:
            if not isinstance(v, list):
                raise ConfigError(f"'torsion_hamiltonian.{jkkey}' must be a list.")
            for i, jkv in enumerate(v):
                try:
                    iv = int(jkv)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"'torsion_hamiltonian.{jkkey}[{i}]' must be an integer.") from exc
                if iv < 0:
                    raise ConfigError(f"'torsion_hamiltonian.{jkkey}[{i}]' must be >= 0.")

    # --- symmetry_mode ---
    sym = t.get("symmetry_mode")
    if sym is not None and str(sym).strip().lower() not in _VALID_TORSION_SYMMETRY_MODES:
        raise ConfigError(
            f"'torsion_hamiltonian.symmetry_mode' must be one of "
            f"{sorted(_VALID_TORSION_SYMMETRY_MODES)} or null."
        )

    # --- boolean flags ---
    for bool_key in ("label_levels", "export_symmetry_blocks", "use_in_selection"):
        v = t.get(bool_key)
        if v is not None and not isinstance(v, bool):
            raise ConfigError(f"'torsion_hamiltonian.{bool_key}' must be true or false.")

    # --- selection_weight ---
    sw = t.get("selection_weight")
    if sw is not None:
        try:
            sw_f = float(sw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("'torsion_hamiltonian.selection_weight' must be numeric.") from exc
        if sw_f <= 0.0:
            raise ConfigError("'torsion_hamiltonian.selection_weight' must be positive.")

    # --- potential block ---
    pot = t.get("potential")
    if pot is not None:
        if not isinstance(pot, dict):
            raise ConfigError("'torsion_hamiltonian.potential' must be a mapping/object.")
        if "v0" in pot:
            try:
                float(pot["v0"])
            except (TypeError, ValueError) as exc:
                raise ConfigError("'torsion_hamiltonian.potential.v0' must be numeric.") from exc
        for vc_key in ("vcos", "vsin"):
            vc = pot.get(vc_key)
            if vc is not None:
                if not isinstance(vc, dict):
                    raise ConfigError(f"'torsion_hamiltonian.potential.{vc_key}' must be a mapping.")
                for k, v in vc.items():
                    try:
                        ki = int(k)
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"'torsion_hamiltonian.potential.{vc_key}' keys must be integers, got {k!r}."
                        ) from exc
                    if ki <= 0:
                        raise ConfigError(
                            f"'torsion_hamiltonian.potential.{vc_key}' keys must be positive integers."
                        )
                    try:
                        float(v)
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"'torsion_hamiltonian.potential.{vc_key}[{k}]' must be numeric."
                        ) from exc

    # --- F_alpha block ---
    fa = t.get("F_alpha")
    if fa is not None:
        if not isinstance(fa, dict):
            raise ConfigError("'torsion_hamiltonian.F_alpha' must be a mapping/object.")
        if "f0" not in fa:
            raise ConfigError("'torsion_hamiltonian.F_alpha.f0' is required.")
        try:
            f0_val = float(fa["f0"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("'torsion_hamiltonian.F_alpha.f0' must be numeric.") from exc
        if f0_val <= 0.0:
            raise ConfigError(
                "'torsion_hamiltonian.F_alpha.f0' must be positive (mean torsion constant)."
            )
        for fa_vc in ("fcos", "fsin"):
            vc = fa.get(fa_vc)
            if vc is not None:
                if not isinstance(vc, dict):
                    raise ConfigError(f"'torsion_hamiltonian.F_alpha.{fa_vc}' must be a mapping.")
                for k, v in vc.items():
                    try:
                        int(k)
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"'torsion_hamiltonian.F_alpha.{fa_vc}' keys must be integers."
                        ) from exc
                    try:
                        float(v)
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"'torsion_hamiltonian.F_alpha.{fa_vc}[{k}]' must be numeric."
                        ) from exc

    # --- targets list ---
    targets = t.get("targets")
    if targets is not None:
        if not isinstance(targets, list):
            raise ConfigError("'torsion_hamiltonian.targets' must be a list.")
        for ti, targ in enumerate(targets):
            if not isinstance(targ, dict):
                raise ConfigError(f"'torsion_hamiltonian.targets[{ti}]' must be a mapping/object.")
            for req_key in ("J", "K"):
                if req_key not in targ:
                    raise ConfigError(f"'torsion_hamiltonian.targets[{ti}].{req_key}' is required.")
                try:
                    iv = int(targ[req_key])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.targets[{ti}].{req_key}' must be an integer."
                    ) from exc
                if iv < 0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.targets[{ti}].{req_key}' must be >= 0."
                    )
            if "level_index" not in targ:
                raise ConfigError(f"'torsion_hamiltonian.targets[{ti}].level_index' is required.")
            try:
                li = int(targ["level_index"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'torsion_hamiltonian.targets[{ti}].level_index' must be an integer."
                ) from exc
            if li < 0:
                raise ConfigError(
                    f"'torsion_hamiltonian.targets[{ti}].level_index' must be >= 0."
                )
            if "energy_cm-1" not in targ:
                raise ConfigError(f"'torsion_hamiltonian.targets[{ti}].energy_cm-1' is required.")
            try:
                float(targ["energy_cm-1"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'torsion_hamiltonian.targets[{ti}].energy_cm-1' must be numeric."
                ) from exc

    # --- transitions list ---
    transitions = t.get("transitions")
    if transitions is not None:
        if not isinstance(transitions, list):
            raise ConfigError("'torsion_hamiltonian.transitions' must be a list.")
        for ti, trans in enumerate(transitions):
            if not isinstance(trans, dict):
                raise ConfigError(
                    f"'torsion_hamiltonian.transitions[{ti}]' must be a mapping/object."
                )
            for req_key in ("J_lo", "K_lo", "level_lo", "J_hi", "K_hi", "level_hi"):
                if req_key not in trans:
                    raise ConfigError(
                        f"'torsion_hamiltonian.transitions[{ti}].{req_key}' is required."
                    )
                try:
                    float(trans[req_key])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.transitions[{ti}].{req_key}' must be numeric."
                    ) from exc
            # freq_cm-1 or freq_mhz required (Phase 4: allow MHz input)
            has_freq = "freq_cm-1" in trans or "freq_mhz" in trans
            if not has_freq:
                raise ConfigError(
                    f"'torsion_hamiltonian.transitions[{ti}]' must have 'freq_cm-1' or 'freq_mhz'."
                )
            for freq_key in ("freq_cm-1", "freq_mhz"):
                v = trans.get(freq_key)
                if v is not None:
                    try:
                        float(v)
                    except (TypeError, ValueError) as exc:
                        raise ConfigError(
                            f"'torsion_hamiltonian.transitions[{ti}].{freq_key}' must be numeric."
                        ) from exc
            # optional symmetry selection fields (Phase 4)
            for sym_key in ("symmetry_lo", "symmetry_hi"):
                v = trans.get(sym_key)
                if v is not None and not isinstance(v, str):
                    raise ConfigError(
                        f"'torsion_hamiltonian.transitions[{ti}].{sym_key}' must be a string (e.g. 'A', 'E')."
                    )
            # optional per-transition uncertainty
            sig = trans.get("sigma_cm-1")
            if sig is not None:
                try:
                    sv = float(sig)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.transitions[{ti}].sigma_cm-1' must be numeric."
                    ) from exc
                if sv <= 0.0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.transitions[{ti}].sigma_cm-1' must be positive."
                    )

    # --- uncertainty block ---
    unc = t.get("uncertainty")
    if unc is not None:
        if not isinstance(unc, dict):
            raise ConfigError("'torsion_hamiltonian.uncertainty' must be a mapping/object.")
        if "enabled" in unc and not isinstance(unc["enabled"], bool):
            raise ConfigError("'torsion_hamiltonian.uncertainty.enabled' must be true or false.")
        if "include_completeness" in unc and not isinstance(unc["include_completeness"], bool):
            raise ConfigError(
                "'torsion_hamiltonian.uncertainty.include_completeness' must be true or false."
            )
        for fkey in ("damping", "rank_tol", "default_sigma_cm1"):
            v = unc.get(fkey)
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.uncertainty.{fkey}' must be numeric."
                    ) from exc
                if fv <= 0.0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.uncertainty.{fkey}' must be positive."
                    )

    # --- auto_assign block (Phase 5) ---
    aa = t.get("auto_assign")
    if aa is not None:
        if not isinstance(aa, dict):
            raise ConfigError("'torsion_hamiltonian.auto_assign' must be a mapping/object.")
        if "enabled" in aa and not isinstance(aa["enabled"], bool):
            raise ConfigError("'torsion_hamiltonian.auto_assign.enabled' must be true or false.")
        if "method" in aa and str(aa["method"]).strip().lower() not in {"global", "greedy", "auto"}:
            raise ConfigError("'torsion_hamiltonian.auto_assign.method' must be global, greedy, or auto.")
        v = aa.get("max_delta_cm1")
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "'torsion_hamiltonian.auto_assign.max_delta_cm1' must be numeric."
                ) from exc
            if fv <= 0.0:
                raise ConfigError(
                    "'torsion_hamiltonian.auto_assign.max_delta_cm1' must be positive."
                )
        v = aa.get("ambiguity_tol_cm1")
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "'torsion_hamiltonian.auto_assign.ambiguity_tol_cm1' must be numeric."
                ) from exc
            if fv < 0.0:
                raise ConfigError(
                    "'torsion_hamiltonian.auto_assign.ambiguity_tol_cm1' must be >= 0."
                )
        obs = aa.get("observed_cm1")
        if obs is not None:
            if not isinstance(obs, list):
                raise ConfigError(
                    "'torsion_hamiltonian.auto_assign.observed_cm1' must be a list of energies."
                )
            for i, v in enumerate(obs):
                try:
                    float(v)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.auto_assign.observed_cm1[{i}]' must be numeric."
                    ) from exc

    # --- fitting block (Phase 6) ---
    fit = t.get("fitting")
    if fit is not None:
        if not isinstance(fit, dict):
            raise ConfigError("'torsion_hamiltonian.fitting' must be a mapping/object.")
        if "enabled" in fit and not isinstance(fit["enabled"], bool):
            raise ConfigError("'torsion_hamiltonian.fitting.enabled' must be true or false.")
        for bkey in ("use_levels", "use_transitions"):
            v = fit.get(bkey)
            if v is not None and not isinstance(v, bool):
                raise ConfigError(f"'torsion_hamiltonian.fitting.{bkey}' must be true or false.")
        mi = fit.get("max_iter")
        if mi is not None:
            try:
                iv = int(mi)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "'torsion_hamiltonian.fitting.max_iter' must be an integer."
                ) from exc
            if iv < 1:
                raise ConfigError("'torsion_hamiltonian.fitting.max_iter' must be >= 1.")
        for fkey in ("xtol", "ftol", "damping"):
            v = fit.get(fkey)
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.fitting.{fkey}' must be numeric."
                    ) from exc
                if fv <= 0.0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.fitting.{fkey}' must be positive."
                    )
        params = fit.get("params")
        if params is not None:
            if not isinstance(params, list):
                raise ConfigError("'torsion_hamiltonian.fitting.params' must be a list of parameter names.")
            for i, pn in enumerate(params):
                if not isinstance(pn, str) or not str(pn).strip():
                    raise ConfigError(
                        f"'torsion_hamiltonian.fitting.params[{i}]' must be a non-empty string."
                    )
        for map_key in ("bounds", "priors"):
            if map_key in fit and fit[map_key] is not None and not isinstance(fit[map_key], dict):
                raise ConfigError(f"'torsion_hamiltonian.fitting.{map_key}' must be a mapping/object.")
        stages = fit.get("stages")
        if stages is not None:
            if not isinstance(stages, list):
                raise ConfigError("'torsion_hamiltonian.fitting.stages' must be a list.")
            for i, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    raise ConfigError(f"'torsion_hamiltonian.fitting.stages[{i}]' must be a mapping/object.")

    # --- fitting cross-validation (Phase 6 troubleshooting) ---
    fit = t.get("fitting")
    if isinstance(fit, dict) and fit.get("enabled", False):
        has_targets = bool(t.get("targets"))
        has_transitions = bool(t.get("transitions"))
        if not has_targets and not has_transitions:
            raise ConfigError(
                "'torsion_hamiltonian.fitting.enabled' is true but neither 'targets' nor "
                "'transitions' are provided. Add observed levels or transition frequencies to fit against."
            )

    # --- potential sign-convention warning ---
    pot = t.get("potential")
    if isinstance(pot, dict):
        vcos = pot.get("vcos") or {}
        if isinstance(vcos, dict):
            for k, v in vcos.items():
                try:
                    ki, vi = int(k), float(v)
                except (TypeError, ValueError):
                    continue
                if ki % 3 == 0 and vi > 0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.potential.vcos[{k}]' = {vi:.4f} is positive. "
                        f"In this codebase the Fourier convention is V(a) = v0 + sum vcos_n*cos(n*a). "
                        f"A 3-fold barrier V3 maps to vcos3 = -V3/2 (negative). "
                        f"If your barrier is {vi:.1f} cm^-1, set v0={vi:.4f} and vcos3={-vi:.4f}."
                    )

    # --- n_basis adequacy check ---
    n_basis_v = t.get("n_basis")
    pot_check = t.get("potential")
    if n_basis_v is not None and isinstance(pot_check, dict):
        nb_ok = True
        try:
            nb = int(n_basis_v)
        except (TypeError, ValueError):
            nb_ok = False
        if nb_ok:
            vcos_c = pot_check.get("vcos") or {}
            if isinstance(vcos_c, dict):
                coeff_vals = []
                for v in vcos_c.values():
                    try:
                        coeff_vals.append(abs(float(v)))
                    except (TypeError, ValueError):
                        pass
                max_barrier = max(coeff_vals, default=0.0)
                if nb < 8 and max_barrier > 150.0:
                    raise ConfigError(
                        f"'torsion_hamiltonian.n_basis' = {nb} may be too small for a barrier "
                        f"of ~{max_barrier:.0f} cm^-1. Use n_basis >= 10 for moderate barriers "
                        f"(~200-500 cm^-1) and n_basis >= 15 for high barriers (> 500 cm^-1). "
                        f"Run 'quantize lam-diagnose --convergence' to check basis convergence."
                    )

    # --- geometry_coupling block (Phase 7) ---
    gc = t.get("geometry_coupling")
    if gc is not None:
        if not isinstance(gc, dict):
            raise ConfigError("'torsion_hamiltonian.geometry_coupling' must be a mapping/object.")
        if "enabled" in gc and not isinstance(gc["enabled"], bool):
            raise ConfigError("'torsion_hamiltonian.geometry_coupling.enabled' must be true or false.")
        gc_enabled = bool(gc.get("enabled", False))
        if gc_enabled:
            if not gc.get("top_indices"):
                raise ConfigError(
                    "'torsion_hamiltonian.geometry_coupling.top_indices' is required when "
                    "geometry_coupling.enabled is true. "
                    "Specify the atom indices (0-based) of the rotating top (e.g. [2, 3, 4] for methyl H atoms)."
                )
            if not gc.get("axis_atom_indices"):
                raise ConfigError(
                    "'torsion_hamiltonian.geometry_coupling.axis_atom_indices' is required when "
                    "geometry_coupling.enabled is true. "
                    "Specify two atom indices (0-based) defining the rotation axis (e.g. [0, 1] for C-O)."
                )
        top = gc.get("top_indices")
        if top is not None:
            if not isinstance(top, list) or not top:
                raise ConfigError(
                    "'torsion_hamiltonian.geometry_coupling.top_indices' must be a non-empty list of integers."
                )
            for i, t_idx in enumerate(top):
                try:
                    int(t_idx)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.geometry_coupling.top_indices[{i}]' must be an integer."
                    ) from exc
        axis = gc.get("axis_atom_indices")
        if axis is not None:
            if not isinstance(axis, (list, tuple)) or len(axis) != 2:
                raise ConfigError(
                    "'torsion_hamiltonian.geometry_coupling.axis_atom_indices' must be a 2-element list of integers."
                )
            for i, a_idx in enumerate(axis):
                try:
                    int(a_idx)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"'torsion_hamiltonian.geometry_coupling.axis_atom_indices[{i}]' must be an integer."
                    ) from exc
        dx = gc.get("dx_ang")
        if dx is not None:
            try:
                dxv = float(dx)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "'torsion_hamiltonian.geometry_coupling.dx_ang' must be numeric."
                ) from exc
            if dxv <= 0.0:
                raise ConfigError("'torsion_hamiltonian.geometry_coupling.dx_ang' must be positive.")

    # --- scan block ---
    scan = t.get("scan")
    if scan is None:
        return
    if not isinstance(scan, dict):
        raise ConfigError("'torsion_hamiltonian.scan' must be a mapping/object.")

    au = scan.get("angle_unit")
    if au is not None and str(au).strip().lower() not in _VALID_SCAN_ANGLE_UNITS:
        raise ConfigError(
            "'torsion_hamiltonian.scan.angle_unit' must be 'degrees' or 'radians'."
        )
    eu = scan.get("energy_unit")
    if eu is not None and str(eu).strip().lower() not in _VALID_SCAN_ENERGY_UNITS:
        raise ConfigError(
            "'torsion_hamiltonian.scan.energy_unit' must be one of: cm-1, hartree, kcal/mol, kj/mol."
        )
    per = scan.get("periodic")
    if per is not None and not isinstance(per, bool):
        raise ConfigError("'torsion_hamiltonian.scan.periodic' must be true or false.")

    gps = scan.get("grid_points")
    csv_path = scan.get("csv_path") or scan.get("path")
    if csv_path is not None and not isinstance(csv_path, (str, Path)):
        raise ConfigError("'torsion_hamiltonian.scan.csv_path' must be a path string.")
    if gps is None:
        if not csv_path:
            raise ConfigError("'torsion_hamiltonian.scan.grid_points' must be a non-empty list.")
    elif not isinstance(gps, list):
        raise ConfigError("'torsion_hamiltonian.scan.grid_points' must be a list.")
    elif len(gps) == 0 and not csv_path:
        raise ConfigError("'torsion_hamiltonian.scan.grid_points' must be a non-empty list.")
    for i, gp in enumerate(gps or []):
        if not isinstance(gp, dict):
            raise ConfigError(f"'torsion_hamiltonian.scan.grid_points[{i}]' must be a mapping/object.")
        if "phi" not in gp:
            raise ConfigError(f"'torsion_hamiltonian.scan.grid_points[{i}].phi' is required.")
        try:
            float(gp["phi"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"'torsion_hamiltonian.scan.grid_points[{i}].phi' must be numeric."
            ) from exc
        if "energy" in gp and gp["energy"] is not None:
            try:
                float(gp["energy"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'torsion_hamiltonian.scan.grid_points[{i}].energy' must be numeric."
                ) from exc

    mode = str(scan.get("mode", "quantum")).strip().lower()
    if mode not in {"quantum", "boltzmann", "quantum_thermal", "thermal_quantum"}:
        raise ConfigError(
            "'torsion_hamiltonian.scan.mode' must be 'quantum', 'boltzmann', or 'quantum_thermal'."
        )
    has_grid_points = isinstance(gps, list) and len(gps) > 0
    if mode in {"quantum", "quantum_thermal", "thermal_quantum"} and has_grid_points:
        hr = scan.get("hindered_rotor_model")
        if hr is None or not isinstance(hr, dict):
            raise ConfigError(
                "'torsion_hamiltonian.scan.hindered_rotor_model' is required in quantum scan mode."
            )
        if hr.get("rotational_constant_F") is None:
            raise ConfigError(
                "'torsion_hamiltonian.scan.hindered_rotor_model.rotational_constant_F'"
                " is required in quantum mode."
            )

    # --- preprocess block (Phase 3) ---
    pp = scan.get("preprocess")
    if pp is not None:
        if not isinstance(pp, dict):
            raise ConfigError("'torsion_hamiltonian.scan.preprocess' must be a mapping/object.")
        for bkey in ("sort", "deduplicate", "extend_by_symmetry"):
            v = pp.get(bkey)
            if v is not None and not isinstance(v, bool):
                raise ConfigError(
                    f"'torsion_hamiltonian.scan.preprocess.{bkey}' must be true or false."
                )
        tol = pp.get("endpoint_tol_rad")
        if tol is not None:
            try:
                tv = float(tol)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "'torsion_hamiltonian.scan.preprocess.endpoint_tol_rad' must be numeric."
                ) from exc
            if tv <= 0.0:
                raise ConfigError(
                    "'torsion_hamiltonian.scan.preprocess.endpoint_tol_rad' must be positive."
                )

    # --- fit_potential block (Phase 2) ---
    fp = scan.get("fit_potential")
    if fp is None or fp is False:
        return
    if fp is True:
        return
    if not isinstance(fp, dict):
        raise ConfigError("'torsion_hamiltonian.scan.fit_potential' must be a mapping or bool.")
    if "enabled" in fp and not isinstance(fp["enabled"], bool):
        raise ConfigError("'torsion_hamiltonian.scan.fit_potential.enabled' must be true or false.")
    for ikey in ("n_harmonics", "symmetry_number"):
        v = fp.get(ikey)
        if v is not None:
            try:
                iv = int(v)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'torsion_hamiltonian.scan.fit_potential.{ikey}' must be an integer."
                ) from exc
            if iv < 1:
                raise ConfigError(
                    f"'torsion_hamiltonian.scan.fit_potential.{ikey}' must be >= 1."
                )
    for bkey in ("cosine_only", "zero_at_minimum"):
        v = fp.get(bkey)
        if v is not None and not isinstance(v, bool):
            raise ConfigError(
                f"'torsion_hamiltonian.scan.fit_potential.{bkey}' must be true or false."
            )


def safe_run_name(name: str | None) -> str:
    raw = str(name or "quantize_run").strip().lower()
    safe = _SAFE_NAME_RE.sub("_", raw).strip("._-")
    return safe or "quantize_run"


def prepare_run_directory(cfg: dict[str, Any], config_path: Path | None = None) -> Path:
    """Create and annotate an output run directory."""
    output = _expect_mapping(cfg, "output")
    explicit = output.get("run_dir")
    if explicit:
        run_dir = Path(str(explicit)).expanduser()
    else:
        root = Path(str(output.get("root", "runs"))).expanduser()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{stamp}_{safe_run_name(cfg.get('name') or cfg.get('molecule'))}"
        if run_dir.exists():
            base = run_dir
            for i in range(2, 1000):
                candidate = base.with_name(f"{base.name}_{i:03d}")
                if not candidate.exists():
                    run_dir = candidate
                    break
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "exports").mkdir(exist_ok=True)
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, run_dir / f"input{config_path.suffix.lower()}")
    cfg["_run_dir"] = str(run_dir.resolve())
    return run_dir.resolve()


def write_final_geometry_csv(path: Path, elems: list[str], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atom_index", "element", "x_angstrom", "y_angstrom", "z_angstrom"])
        for i, (elem, xyz) in enumerate(zip(elems, np.asarray(coords, dtype=float))):
            writer.writerow([i, elem, f"{xyz[0]:.10f}", f"{xyz[1]:.10f}", f"{xyz[2]:.10f}"])


def residual_rows(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from backend.spectral import SpectralEngine

    engine = SpectralEngine(spectral_isotopologues)
    rows: list[dict[str, Any]] = []
    for iso in engine.isotopologues:
        calc_all = engine.rotational_constants(coords, iso["masses"])
        idx = np.asarray(iso["component_indices"], dtype=int)
        target = iso["obs_constants"] + 0.5 * iso["alpha_constants"]
        for j, comp in enumerate(idx):
            calc = float(calc_all[int(comp)])
            rows.append(
                {
                    "isotopologue": iso["name"],
                    "component": COMPONENT_LABELS[int(comp)],
                    "target_mhz": float(target[j]),
                    "calculated_mhz": calc,
                    "residual_mhz": float(target[j] - calc),
                    "sigma_mhz": float(iso["sigma_constants"][j]),
                }
            )
    return rows


def write_residuals_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["isotopologue", "component", "target_mhz", "calculated_mhz", "residual_mhz", "sigma_mhz"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def singular_values(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> np.ndarray:
    from backend.spectral import SpectralEngine

    engine = SpectralEngine(spectral_isotopologues)
    J, _ = engine.stacked_unweighted(coords)
    return np.linalg.svd(J, compute_uv=False)


def write_markdown_report(path: Path, result: dict[str, Any], artifacts: dict[str, Any] | None = None) -> None:
    from runner.reporting import generate_lam_report_section, generate_rovib_report_section

    best = result["best"]
    score = result.get("score", {})
    lines = [
        f"# Quantize Report: {result.get('name', 'run')}",
        "",
        f"- Run directory: `{result.get('run_dir', '.')}`",
        f"- Best start: `{best.get('idx', 'n/a')}`",
        f"- Spectral RMS: `{float(best.get('freq_rms', np.nan)):.6f}` MHz",
        f"- Final energy: `{float(best.get('energy', np.nan)):.10g}` Eh",
    ]
    if score:
        lines.extend(
            [
                f"- Success score: `{float(score.get('score', np.nan)):.1f}`",
                f"- Constrained rank: `{score.get('constrained_rank', 'n/a')}/{score.get('internal_dof', 'n/a')}`",
            ]
        )

    iso_snapshot = best.get("spectral_isotopologues_snapshot", [])
    if iso_snapshot:
        lines.extend(["", generate_rovib_report_section(iso_snapshot)])

    torsion_summary = result.get("torsion_summary") or {}
    if torsion_summary:
        lines.extend(["", generate_lam_report_section(torsion_summary)])

    lines.extend(["", "## Final Geometry", "", "| atom | element | x (Ang) | y (Ang) | z (Ang) |", "|---:|---|---:|---:|---:|"])
    for i, (elem, xyz) in enumerate(zip(result["elems"], np.asarray(best["coords"], dtype=float))):
        lines.append(f"| {i} | {elem} | {xyz[0]:.8f} | {xyz[1]:.8f} | {xyz[2]:.8f} |")
    lines.extend(["", "## Residuals", "", "| isotopologue | component | target MHz | calculated MHz | residual MHz |", "|---|---:|---:|---:|---:|"])
    for row in result.get("residual_rows", []):
        lines.append(
            f"| {row['isotopologue']} | {row['component']} | {row['target_mhz']:.6f} | "
            f"{row['calculated_mhz']:.6f} | {row['residual_mhz']:.6f} |"
        )
    if result.get("torsion_objective_rows"):
        torsion_rows = result["torsion_objective_rows"]
        is_transition_mode = "J_lo" in torsion_rows[0]
        lines.extend(
            [
                "",
                "## Torsion Objective",
                "",
                f"- Torsion RMS (cm^-1): `{float(result.get('torsion_rms_cm-1', np.nan)):.6f}`",
                "",
            ]
        )
        if is_transition_mode:
            lines.extend(
                [
                    "| J_lo | K_lo | level_lo | J_hi | K_hi | level_hi | Observed (cm^-1) | Predicted (cm^-1) | Residual (cm^-1) |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in torsion_rows:
                lines.append(
                    f"| {int(row['J_lo'])} | {int(row['K_lo'])} | {int(row['level_lo'])} | "
                    f"{int(row['J_hi'])} | {int(row['K_hi'])} | {int(row['level_hi'])} | "
                    f"{float(row['observed_cm-1']):.6f} | {float(row['predicted_cm-1']):.6f} | "
                    f"{float(row['residual_cm-1']):.6f} |"
                )
        else:
            lines.extend(
                [
                    "| J | K | Level | Observed (cm^-1) | Predicted (cm^-1) | Residual (cm^-1) |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in torsion_rows:
                lines.append(
                    f"| {int(row['J'])} | {int(row['K'])} | {int(row['level_index'])} | "
                    f"{float(row['observed_cm-1']):.6f} | {float(row['predicted_cm-1']):.6f} | "
                    f"{float(row['residual_cm-1']):.6f} |"
                )
    lines.extend(["", "## Outputs", "", "- `exports/final_geometry.csv`", "- `exports/residuals.csv`"])
    if artifacts:
        if artifacts.get("rovib_corrections_csv") is not None:
            lines.append("- `exports/rovib_corrections.csv`")
        if artifacts.get("semi_experimental_targets_csv") is not None:
            lines.append("- `exports/semi_experimental_targets.csv`")
        if artifacts.get("rovib_warnings_json") is not None:
            lines.append("- `exports/rovib_warnings.json`")
        if artifacts.get("internal_uncertainty_csv") is not None:
            lines.append("- `exports/internal_uncertainty.csv`")
        if artifacts.get("internal_covariance_csv") is not None:
            lines.append("- `exports/internal_covariance.csv`")
        if artifacts.get("internal_identifiability_csv") is not None:
            lines.append("- `exports/internal_identifiability.csv`")
        if artifacts.get("torsion_objective_csv") is not None:
            lines.append("- `exports/torsion_objective.csv`")
    lines.extend(["- `plots/residuals.png`", "- `plots/singular_values.png`", "- `plots/convergence.png`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(run_dir: Path, result: dict[str, Any]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    paths: list[Path] = []
    plots_dir = run_dir / "plots"
    rows = result.get("residual_rows", [])
    if rows:
        labels = [f"{r['isotopologue']} {r['component']}" for r in rows]
        values = [float(r["residual_mhz"]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(range(len(values)), values, color="#2f6f9f")
        ax.set_ylabel("Residual (MHz)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        path = plots_dir / "residuals.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    sv = np.asarray(result.get("singular_values", []), dtype=float)
    if sv.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.arange(1, sv.size + 1), sv, marker="o", color="#007c7a")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value")
        ax.set_title("Spectral Jacobian Singular Values")
        fig.tight_layout()
        path = plots_dir / "singular_values.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    history = result["best"].get("history") or []
    if history:
        it = [int(h.get("iteration", i + 1)) for i, h in enumerate(history)]
        freq = [float(h.get("freq_rms", np.nan)) for h in history]
        step = [float(h.get("step_norm", np.nan)) for h in history]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(it, freq, marker="o", color="#1d5fd1", label="freq RMS")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Frequency RMS (MHz)")
        ax2 = ax1.twinx()
        ax2.semilogy(it, step, marker="s", color="#c55a11", label="step norm")
        ax2.set_ylabel("Step norm (Ang)")
        fig.tight_layout()
        path = plots_dir / "convergence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_outputs(result: dict[str, Any]) -> dict[str, Path | list[Path]]:
    """Write CSV, Markdown, and plot artifacts for a completed generic run."""
    from runner.reporting import (
        export_rovib_corrections_csv,
        export_rovib_warnings_json,
        export_semi_experimental_targets_csv,
    )

    run_dir = Path(result.get("run_dir") or ".").resolve()
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    geom_csv = exports_dir / "final_geometry.csv"
    residual_csv = exports_dir / "residuals.csv"
    write_final_geometry_csv(geom_csv, result["elems"], result["best"]["coords"])

    iso_snapshot = result["best"].get("spectral_isotopologues_snapshot", [])
    rows = residual_rows(result["best"]["coords"], iso_snapshot)
    result["residual_rows"] = rows
    result["singular_values"] = singular_values(result["best"]["coords"], iso_snapshot).tolist()
    write_residuals_csv(residual_csv, rows)

    artifacts: dict[str, Any] = {
        "geometry_csv": geom_csv,
        "residuals_csv": residual_csv,
    }

    # Rovib correction exports (written whenever isotopologue data exists).
    if iso_snapshot:
        rovib_csv = export_rovib_corrections_csv(iso_snapshot, exports_dir / "rovib_corrections.csv")
        semi_csv = export_semi_experimental_targets_csv(iso_snapshot, exports_dir / "semi_experimental_targets.csv")
        warn_json = export_rovib_warnings_json(iso_snapshot, exports_dir / "rovib_warnings.json")
        artifacts["rovib_corrections_csv"] = rovib_csv
        artifacts["semi_experimental_targets_csv"] = semi_csv
        artifacts["rovib_warnings_json"] = warn_json

    # Internal-coordinate uncertainty / identifiability exports.
    cfg = result.get("cfg", {}) or {}
    coord_mode = str(cfg.get("coordinate_mode", "cartesian")).strip().lower()
    if coord_mode == "internal" and iso_snapshot:
        from backend.internal_fit import InternalCoordinateSet, spectral_jacobian_q, build_internal_priors
        from backend.spectral import SpectralEngine
        from backend.uncertainty import uncertainty_table, compute_uncertainty
        from backend.identifiability import identifiability_table

        ic_cfg = cfg.get("internal_coordinates", {}) or {}
        use_dihedrals = bool(ic_cfg.get("use_dihedrals", False))
        damping = max(float(ic_cfg.get("damping", 1e-6)), 1e-14)
        sigma_bond = float(ic_cfg.get("prior_sigma_bond", 0.04))
        sigma_angle_deg = float(ic_cfg.get("prior_sigma_angle_deg", 2.0))
        sigma_dihedral_deg = float(ic_cfg.get("prior_sigma_dihedral_deg", 15.0))

        coord_set = InternalCoordinateSet(result["best"]["coords"], result["elems"], use_dihedrals=use_dihedrals)
        B_active = coord_set.active_B_matrix(result["best"]["coords"])
        if B_active.shape[0] > 0:
            Bplus = InternalCoordinateSet.damped_pseudoinverse(B_active, damping)
            J_spectral, _ = SpectralEngine(iso_snapshot).stacked(result["best"]["coords"])
            Jq = spectral_jacobian_q(J_spectral, Bplus)
            _, _, sigma_prior = build_internal_priors(
                coord_set,
                result["best"]["coords"],
                sigma_bond=sigma_bond,
                sigma_angle_deg=sigma_angle_deg,
                sigma_dihedral_deg=sigma_dihedral_deg,
            )

            unc_rows = uncertainty_table(
                coord_set,
                result["best"]["coords"],
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
            )
            unc_csv = exports_dir / "internal_uncertainty.csv"
            with unc_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "name",
                        "value",
                        "value_unit",
                        "std_err",
                        "std_err_unit",
                        "ci_lo",
                        "ci_hi",
                        "ci_unit",
                    ],
                )
                writer.writeheader()
                for r in unc_rows:
                    writer.writerow(r)
            artifacts["internal_uncertainty_csv"] = unc_csv

            cov, _, _ = compute_uncertainty(
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
            )
            cov_csv = exports_dir / "internal_covariance.csv"
            with cov_csv.open("w", newline="", encoding="utf-8") as fh:
                active_names = [ic.name for ic in coord_set.active_coords()]
                writer = csv.writer(fh)
                writer.writerow(["coordinate"] + active_names)
                for i, row_name in enumerate(active_names):
                    writer.writerow([row_name] + [f"{float(v):.12e}" for v in cov[i]])
            artifacts["internal_covariance_csv"] = cov_csv

            id_rows, sv, rank = identifiability_table(coord_set, Jq, sigma_prior)
            id_csv = exports_dir / "internal_identifiability.csv"
            with id_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["name", "score", "label", "sv_rank"],
                )
                writer.writeheader()
                for r in id_rows:
                    writer.writerow(r)
            artifacts["internal_identifiability_csv"] = id_csv
            artifacts["internal_rank"] = rank
            artifacts["internal_singular_values"] = [float(x) for x in sv]

    if result.get("torsion_objective_rows"):
        torsion_csv = exports_dir / "torsion_objective.csv"
        torsion_rows = result["torsion_objective_rows"]
        first = torsion_rows[0]
        if "J_lo" in first:
            fieldnames = [
                "J_lo",
                "K_lo",
                "level_lo",
                "J_hi",
                "K_hi",
                "level_hi",
                "observed_cm-1",
                "predicted_cm-1",
                "residual_cm-1",
            ]
        else:
            fieldnames = [
                "J",
                "K",
                "level_index",
                "observed_cm-1",
                "predicted_cm-1",
                "residual_cm-1",
            ]
        with torsion_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for r in torsion_rows:
                writer.writerow(r)
        artifacts["torsion_objective_csv"] = torsion_csv

    report_md = run_dir / "report.md"
    write_markdown_report(report_md, result, artifacts=artifacts)
    plot_paths = write_plots(run_dir, result)
    artifacts["report_md"] = report_md
    artifacts["plots"] = plot_paths
    return artifacts
