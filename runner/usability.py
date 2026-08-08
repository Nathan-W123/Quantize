from __future__ import annotations

import csv
import json
import re
import shutil
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
from backend.spectral.spectral_model import normalize_spectral_model
from runner.config_schema import CONFIG_SCHEMA_VERSION, normalize_config

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
    return normalize_config(data)


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
    schema_v = str(cfg.get("schema_version", CONFIG_SCHEMA_VERSION)).strip()
    if schema_v != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"'schema_version' must be '{CONFIG_SCHEMA_VERSION}' for this build (got '{schema_v}')."
        )

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
    seen_iso_names: set[str] = set()
    for iso_i, iso in enumerate(isotopologues):
        if not isinstance(iso, dict):
            raise ConfigError(f"'isotopologues[{iso_i}]' must be a mapping/object.")
        prefix = f"isotopologues[{iso_i}]"
        iso_name = str(iso.get("name", "")).strip()
        if not iso_name:
            raise ConfigError(f"'{prefix}.name' is required and cannot be blank.")
        if iso_name in seen_iso_names:
            raise ConfigError(f"Duplicate isotopologue name '{iso_name}' in '{prefix}.name'.")
        seen_iso_names.add(iso_name)
        _check_numeric_list(iso.get("masses"), f"{prefix}.masses", n_atoms, positive=True)
        comps = iso.get("components", ["A", "B", "C"])
        comps = _as_list(comps, f"{prefix}.components")
        if not comps:
            raise ConfigError(f"'{prefix}.components' must contain at least one component.")
        norm_comps: list[str] = []
        for comp in comps:
            c = str(comp).strip().upper()
            if c not in COMPONENT_LABELS:
                raise ConfigError(f"'{prefix}.components' values must be A, B, or C.")
            norm_comps.append(c)
        if len(set(norm_comps)) != len(norm_comps):
            raise ConfigError(f"'{prefix}.components' contains duplicate entries; each of A/B/C can appear once.")
        n_comp = len(comps)
        if "obs_b0_mhz" not in iso:
            raise ConfigError(f"'{prefix}.obs_b0_mhz' is required (one value per listed component).")
        if "alpha_mhz" not in iso:
            raise ConfigError(f"'{prefix}.alpha_mhz' is required (one value per listed component).")
        if "sigma_mhz" not in iso:
            raise ConfigError(f"'{prefix}.sigma_mhz' is required (one value per listed component).")
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
    _validate_conformer_mixture_block(cfg, n_atoms=n_atoms)
    _validate_torsion_block(cfg)
    _validate_internal_priors_block(cfg, n_atoms=n_atoms)


_VALID_ROVIB_MODES = {
    "hybrid_auto", "user_only", "orca_only",
    "manual_alpha", "manual_delta", "none",
    "strict_user", "strict_backend",
}


def _validate_internal_priors_block(cfg: dict[str, Any], n_atoms: int) -> None:
    ip = cfg.get("internal_priors")
    if ip is None:
        return
    if not isinstance(ip, dict):
        raise ConfigError("'internal_priors' must be a mapping/object.")

    mode = str(ip.get("mode", "soft")).strip().lower()
    valid_modes = {"off", "soft", "adaptive", "hard_freeze"}
    if mode not in valid_modes:
        raise ConfigError(f"'internal_priors.mode' must be one of {sorted(valid_modes)}.")

    fsf = ip.get("freeze_sigma_floor")
    if fsf is not None:
        try:
            fv = float(fsf)
        except (TypeError, ValueError) as exc:
            raise ConfigError("'internal_priors.freeze_sigma_floor' must be numeric.") from exc
        if fv <= 0.0:
            raise ConfigError("'internal_priors.freeze_sigma_floor' must be positive.")

    ad = ip.get("adaptive")
    if ad is not None:
        if not isinstance(ad, dict):
            raise ConfigError("'internal_priors.adaptive' must be a mapping/object.")
        for key in ("identifiability_gamma", "min_sigma_scale", "max_sigma_scale"):
            if key in ad and ad[key] is not None:
                try:
                    fv = float(ad[key])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"'internal_priors.adaptive.{key}' must be numeric.") from exc
                if fv < 0.0:
                    raise ConfigError(f"'internal_priors.adaptive.{key}' must be >= 0.")
        if "min_sigma_scale" in ad and "max_sigma_scale" in ad:
            if float(ad["max_sigma_scale"]) < float(ad["min_sigma_scale"]):
                raise ConfigError(
                    "'internal_priors.adaptive.max_sigma_scale' must be >= min_sigma_scale."
                )

    priors = ip.get("user_priors", [])
    if priors is None:
        return
    if not isinstance(priors, list):
        raise ConfigError("'internal_priors.user_priors' must be a list.")
    for i, p in enumerate(priors):
        path = f"internal_priors.user_priors[{i}]"
        if not isinstance(p, dict):
            raise ConfigError(f"'{path}' must be a mapping/object.")
        kind = str(p.get("type", p.get("kind", ""))).strip().lower()
        if kind not in {"bond", "angle", "dihedral"}:
            raise ConfigError(f"'{path}.type' must be bond, angle, or dihedral.")
        atoms = p.get("atoms")
        if not isinstance(atoms, list):
            raise ConfigError(f"'{path}.atoms' must be a list of atom indices.")
        need = 2 if kind == "bond" else 3 if kind == "angle" else 4
        if len(atoms) != need:
            raise ConfigError(f"'{path}.atoms' must contain {need} indices for type '{kind}'.")
        for a in atoms:
            if not isinstance(a, int):
                raise ConfigError(f"'{path}.atoms' indices must be integers.")
            # Accept either 0-based or 1-based user indices.
            if a < 0 or a > n_atoms:
                raise ConfigError(f"'{path}.atoms' index {a} is out of range.")
        for req_key in ("target", "sigma"):
            if req_key not in p:
                raise ConfigError(f"'{path}.{req_key}' is required.")
            try:
                fv = float(p[req_key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'{path}.{req_key}' must be numeric.") from exc
            if req_key == "sigma" and fv <= 0.0:
                raise ConfigError(f"'{path}.sigma' must be positive.")
        units = p.get("units")
        if units is not None and str(units).strip().lower() not in {
            "angstrom", "a", "å", "radian", "rad", "degree", "degrees", "deg"
        }:
            raise ConfigError(f"'{path}.units' must be angstrom, radian, or degree variants.")


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

    g_tensor = rc.get("g_tensor")
    if g_tensor is not None:
        if not isinstance(g_tensor, dict):
            raise ConfigError(
                "'rovibrational_corrections.g_tensor' must be a mapping of "
                "component (A/B/C) → g-value."
            )
        for comp, val in g_tensor.items():
            comp_u = str(comp).strip().upper()
            if comp_u not in {"A", "B", "C"}:
                raise ConfigError(
                    f"'rovibrational_corrections.g_tensor' key '{comp}' must be A, B, or C."
                )
            try:
                float(val)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"'rovibrational_corrections.g_tensor.{comp_u}' must be numeric."
                ) from exc

    for flag_key in ("harmonic_from_hessian", "anharmonic_from_hessian",
                     "harmonic_cd_from_hessian", "fit_cd_constants"):
        v = rc.get(flag_key)
        if v is not None and not isinstance(v, bool):
            raise ConfigError(
                f"'rovibrational_corrections.{flag_key}' must be true or false."
            )

    if rc.get("anharmonic_from_hessian") and not rc.get("harmonic_from_hessian"):
        raise ConfigError(
            "'rovibrational_corrections.anharmonic_from_hessian' requires "
            "'harmonic_from_hessian: true' — the anharmonic term is added to the "
            "Hessian-derived alpha, which is only computed in that mode."
        )

    step = rc.get("anharmonic_fd_delta_ang")
    if step is not None:
        try:
            sv = float(step)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "'rovibrational_corrections.anharmonic_fd_delta_ang' must be numeric."
            ) from exc
        if not 1e-4 <= sv <= 0.1:
            raise ConfigError(
                "'rovibrational_corrections.anharmonic_fd_delta_ang' must be between "
                "1e-4 and 0.1 Angstrom; smaller amplifies Hessian noise, larger "
                "loses the third-derivative signal to truncation error."
            )


_VALID_TORSION_SYMMETRY_MODES = {"c3", "3fold", "threefold", "c2", "2fold", "twofold", "none", "off", "null", ""}
_VALID_SCAN_ANGLE_UNITS = {"degrees", "deg", "degree", "radians", "rad", "radian"}
_VALID_SCAN_ENERGY_UNITS = {"cm-1", "cm_1", "hartree", "ha", "kcal/mol", "kcal", "kj/mol", "kj"}


def _validate_conformer_mixture_block(cfg: dict[str, Any], n_atoms: int) -> None:
    """Validate both explicit-conformer blocks.

    ``conformer_mixture`` (fixed/Boltzmann mixture) and ``conformers`` (the
    generation workflow) are parsed by the same ``_explicit_conformer_defs``
    reader, so both accept a ``conformers:`` list of ``coords_angstrom`` /
    ``offset_angstrom`` entries and both need the same shape checks.
    """
    for key in ("conformer_mixture", "conformers"):
        _validate_conformer_block(cfg.get(key), key, n_atoms)


def _validate_conformer_block(cm: Any, key: str, n_atoms: int) -> None:
    if cm is None:
        return
    if not isinstance(cm, dict):
        raise ConfigError(f"'{key}' must be a mapping/object.")
    if "enabled" in cm and not isinstance(cm["enabled"], bool):
        raise ConfigError(f"'{key}.enabled' must be true or false.")
    mode = str(cm.get("weight_mode", "fixed")).strip().lower()
    if mode not in {"fixed", "boltzmann"}:
        raise ConfigError(f"'{key}.weight_mode' must be fixed or boltzmann.")
    if "temperature_k" in cm:
        try:
            t = float(cm["temperature_k"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"'{key}.temperature_k' must be numeric.") from exc
        if t <= 0.0:
            raise ConfigError(f"'{key}.temperature_k' must be > 0.")
    conformers = cm.get("conformers")
    if conformers is None:
        return
    if not isinstance(conformers, list):
        raise ConfigError(f"'{key}.conformers' must be a list.")
    for i, c in enumerate(conformers):
        p = f"{key}.conformers[{i}]"
        if not isinstance(c, dict):
            raise ConfigError(f"'{p}' must be a mapping/object.")
        if "weight" in c and c["weight"] is not None:
            try:
                w = float(c["weight"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'{p}.weight' must be numeric.") from exc
            if w < 0.0:
                raise ConfigError(f"'{p}.weight' must be >= 0.")
        for ekey in ("energy", "relative_energy_cm1", "relative_energy_kcal_mol"):
            if ekey in c and c[ekey] is not None:
                try:
                    float(c[ekey])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"'{p}.{ekey}' must be numeric.") from exc
        if "coords_angstrom" in c and c["coords_angstrom"] is not None:
            rows = _as_list(c["coords_angstrom"], f"{p}.coords_angstrom")
            if len(rows) != n_atoms:
                raise ConfigError(f"'{p}.coords_angstrom' must have {n_atoms} rows.")
            for j, row in enumerate(rows):
                _check_numeric_list(row, f"{p}.coords_angstrom[{j}]", 3)
        if "offset_angstrom" in c and c["offset_angstrom"] is not None:
            rows = _as_list(c["offset_angstrom"], f"{p}.offset_angstrom")
            if len(rows) != n_atoms:
                raise ConfigError(f"'{p}.offset_angstrom' must have {n_atoms} rows.")
            for j, row in enumerate(rows):
                _check_numeric_list(row, f"{p}.offset_angstrom[{j}]", 3)


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
    # Physics-level check: only meaningful when the torsion pipeline will run.
    pot = t.get("potential")
    if enabled and isinstance(pot, dict):
        vcos = pot.get("vcos") or {}
        if isinstance(vcos, dict):
            for k, v in vcos.items():
                try:
                    ki, vi = int(k), float(v)
                except (TypeError, ValueError):
                    continue
                # Only check the fundamental 3-fold term — higher harmonics (V6, V9…)
                # are overtone corrections whose sign is physically independent.
                if ki == 3 and vi > 0:
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
            vsin_c = pot_check.get("vsin") or {}
            coeff_vals = []
            max_harmonic = 0
            for src in (vcos_c, vsin_c):
                if not isinstance(src, dict):
                    continue
                for k, v in src.items():
                    try:
                        ki = int(k)
                        vi = abs(float(v))
                    except (TypeError, ValueError):
                        continue
                    coeff_vals.append(vi)
                    if ki > max_harmonic:
                        max_harmonic = ki
            max_barrier = max(coeff_vals, default=0.0)
            # n_basis must be >= highest harmonic order to represent V_n correctly
            if max_harmonic > 0 and nb < max_harmonic:
                raise ConfigError(
                    f"'torsion_hamiltonian.n_basis' = {nb} is smaller than the highest "
                    f"potential harmonic order {max_harmonic}. Set n_basis >= {max_harmonic} "
                    f"so that V_{max_harmonic} is correctly represented in the Fourier basis."
                )
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
        copied = run_dir / f"input{config_path.suffix.lower()}"
        shutil.copy2(config_path, copied)
    else:
        copied = None
    # Run metadata snapshot for reproducibility.
    cfg_blob = json.dumps(cfg, sort_keys=True, default=str)
    meta = {
        "schema_version": cfg.get("schema_version", CONFIG_SCHEMA_VERSION),
        "created_at": datetime.now(UTC).isoformat(),
        "name": cfg.get("name", cfg.get("molecule", "quantize_run")),
        "preset": cfg.get("preset"),
        "coordinate_mode": cfg.get("coordinate_mode"),
        "config_path": (str(config_path.resolve()) if config_path else None),
        "copied_config": (str(copied) if copied else None),
        "config_sha256": hashlib.sha256(cfg_blob.encode("utf-8")).hexdigest(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cfg["_run_dir"] = str(run_dir.resolve())
    return run_dir.resolve()


def write_final_geometry_csv(path: Path, elems: list[str], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["atom_index", "element", "x_angstrom", "y_angstrom", "z_angstrom"])
        for i, (elem, xyz) in enumerate(zip(elems, np.asarray(coords, dtype=float))):
            writer.writerow([i, elem, f"{xyz[0]:.10f}", f"{xyz[1]:.10f}", f"{xyz[2]:.10f}"])


def residual_rows(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from backend.spectral.spectral import SpectralEngine

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
    from backend.spectral.spectral import SpectralEngine

    engine = SpectralEngine(spectral_isotopologues)
    J, _ = engine.stacked_unweighted(coords)
    return np.linalg.svd(J, compute_uv=False)


def kraitchman_run_analysis(coords: np.ndarray, spectral_isotopologues: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Kraitchman rs analysis for a completed run.

    Uses only isotopologues that observe all three components (A, B, C); the
    substitution coordinates are compared against the fitted geometry in its
    principal axis system.  Returns None when fewer than two full-ABC species
    are available.
    """
    from backend.kraitchman import compare_rs_to_geometry, kraitchman_analysis

    full_abc = []
    for iso in spectral_isotopologues:
        comp = sorted(int(c) for c in iso.get("component_indices", []))
        obs = np.asarray(iso.get("obs_constants", []), dtype=float)
        if comp == [0, 1, 2] and obs.size == 3 and np.all(obs > 0):
            full_abc.append({
                "name": iso.get("name"),
                "masses": np.asarray(iso["masses"], dtype=float),
                "obs_constants": obs,
            })
    if len(full_abc) < 2:
        return None
    out = kraitchman_analysis(full_abc)
    parent_masses = full_abc[0]["masses"]
    out["comparison"] = compare_rs_to_geometry(out["rows"], np.asarray(coords, dtype=float), parent_masses)
    return out


def write_kraitchman_csv(path: Path, kr: dict[str, Any]) -> None:
    """Write rs coordinates (+fit comparison) and inertial defects to one CSV."""
    fields = [
        "name", "atom_index", "delta_mass_amu",
        "abs_a_angstrom", "abs_b_angstrom", "abs_c_angstrom",
        "sigma_a_angstrom", "sigma_b_angstrom", "sigma_c_angstrom",
        "fit_abs_a_angstrom", "fit_abs_b_angstrom", "fit_abs_c_angstrom",
        "delta_a_angstrom", "delta_b_angstrom", "delta_c_angstrom",
        "imaginary_axes", "inertial_defect_amuA2",
    ]
    rows = list(kr.get("comparison") or kr.get("rows") or [])
    defects = {d["name"]: d["inertial_defect_amuA2"] for d in kr.get("defects", [])}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["inertial_defect_amuA2"] = defects.get(row.get("name"), "")
            writer.writerow(out)
        # Species without substitution rows (e.g. the parent) still get defects.
        named = {r.get("name") for r in rows}
        for name, defect in defects.items():
            if name not in named:
                writer.writerow({"name": name, "inertial_defect_amuA2": defect})


def generate_kraitchman_report_section(kr: dict[str, Any]) -> str:
    """Markdown section comparing Kraitchman rs coordinates with the fit."""
    lines = ["## Kraitchman Substitution Analysis (rs)", ""]
    comparison = kr.get("comparison") or []
    if comparison:
        lines.extend([
            "Substitution coordinates |a|, |b|, |c| from Kraitchman's equations vs the fitted geometry",
            "(parent principal axis system; sigma = Costain rule 0.0015/|z|):",
            "",
            "| species | atom | rs |a| | rs |b| | rs |c| | fit |a| | fit |b| | fit |c| | max dev (Å) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in comparison:
            max_dev = max(abs(row[f"delta_{ax}_angstrom"]) for ax in ("a", "b", "c"))
            lines.append(
                f"| {row['name']} | {row['atom_index']} | "
                f"{row['abs_a_angstrom']:.5f} | {row['abs_b_angstrom']:.5f} | {row['abs_c_angstrom']:.5f} | "
                f"{row['fit_abs_a_angstrom']:.5f} | {row['fit_abs_b_angstrom']:.5f} | {row['fit_abs_c_angstrom']:.5f} | "
                f"{max_dev:.5f} |"
            )
    defects = kr.get("defects") or []
    if defects:
        lines.extend([
            "",
            "Inertial defects (Δ = I_c − I_a − I_b; ≈ 0 for rigid planar molecules):",
            "",
            "| species | Δ (amu·Å²) |",
            "|---|---:|",
        ])
        for d in defects:
            lines.append(f"| {d['name']} | {d['inertial_defect_amuA2']:.5f} |")
    for w in kr.get("warnings", []):
        lines.append(f"- ⚠ {w}")
    return "\n".join(lines)


def write_markdown_report(path: Path, result: dict[str, Any], artifacts: dict[str, Any] | None = None) -> None:
    from runner.reporting import (
        generate_conformer_report_section,
        generate_lam_report_section,
        generate_rovib_report_section,
    )

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

    if result.get("kraitchman"):
        lines.extend(["", generate_kraitchman_report_section(result["kraitchman"])])

    torsion_summary = result.get("torsion_summary") or {}
    if torsion_summary:
        lines.extend(["", generate_lam_report_section(torsion_summary)])

    conformer_summary = best.get("conformer_summary") or {}
    if conformer_summary:
        lines.extend(["", generate_conformer_report_section(conformer_summary)])

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
        if artifacts.get("kraitchman_csv") is not None:
            lines.append("- `exports/kraitchman_rs.csv`")
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
        if artifacts.get("internal_prior_sensitivity_csv") is not None:
            lines.append("- `exports/internal_prior_sensitivity.csv`")
        if artifacts.get("internal_prior_provenance_csv") is not None:
            lines.append("- `exports/internal_prior_provenance.csv`")
        if artifacts.get("conformer_weights_history_csv") is not None:
            lines.append("- `exports/conformer_weights_history.csv`")
        if artifacts.get("conformer_summary_json") is not None:
            lines.append("- `exports/conformer_summary.json`")
        if artifacts.get("torsion_objective_csv") is not None:
            lines.append("- `exports/torsion_objective.csv`")
    lines.extend(["- `plots/residuals.png`", "- `plots/singular_values.png`", "- `plots/convergence.png`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(path: Path, result: dict[str, Any], artifacts: dict[str, Any] | None = None) -> None:
    """Write a lightweight publication-friendly HTML report."""
    best = result["best"]
    score = result.get("score", {})
    rows = result.get("residual_rows", [])
    geom_rows = []
    for i, (elem, xyz) in enumerate(zip(result["elems"], np.asarray(best["coords"], dtype=float))):
        geom_rows.append(
            "<tr>"
            f"<td>{i}</td><td>{elem}</td>"
            f"<td>{xyz[0]:.8f}</td><td>{xyz[1]:.8f}</td><td>{xyz[2]:.8f}</td>"
            "</tr>"
        )

    residual_rows_html = []
    for r in rows:
        residual_rows_html.append(
            "<tr>"
            f"<td>{r['isotopologue']}</td><td>{r['component']}</td>"
            f"<td>{float(r['target_mhz']):.6f}</td><td>{float(r['calculated_mhz']):.6f}</td>"
            f"<td>{float(r['residual_mhz']):.6f}</td><td>{float(r.get('sigma_mhz', np.nan)):.6f}</td>"
            "</tr>"
        )

    out_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Quantize Report: {result.get('name', 'run')}</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#111;}",
        "h1,h2{margin:.6rem 0;}",
        ".meta{background:#f7f7f7;border:1px solid #ddd;padding:.8rem;}",
        "table{border-collapse:collapse;width:100%;margin:.8rem 0;}",
        "th,td{border:1px solid #ddd;padding:.35rem .45rem;text-align:left;font-size:.95rem;}",
        "th{background:#f2f2f2;}",
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}",
        ".muted{color:#555;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Quantize Report: {result.get('name', 'run')}</h1>",
        "<div class='meta'>",
        f"<div><strong>Run directory:</strong> {result.get('run_dir', '.')}</div>",
        f"<div><strong>Best start:</strong> {best.get('idx', 'n/a')}</div>",
        f"<div><strong>Spectral RMS:</strong> {float(best.get('freq_rms', np.nan)):.6f} MHz</div>",
        f"<div><strong>Final energy:</strong> {float(best.get('energy', np.nan)):.10g} Eh</div>",
        f"<div><strong>Success score:</strong> {float(score.get('score', np.nan)):.1f}</div>",
        f"<div><strong>Constrained rank:</strong> {score.get('constrained_rank', 'n/a')}/{score.get('internal_dof', 'n/a')}</div>",
        "</div>",
        "<h2>Final Geometry</h2>",
        "<table><thead><tr><th>atom</th><th>element</th><th>x (Ang)</th><th>y (Ang)</th><th>z (Ang)</th></tr></thead><tbody>",
        *geom_rows,
        "</tbody></table>",
        "<h2>Residuals</h2>",
        "<table><thead><tr><th>isotopologue</th><th>component</th><th>target MHz</th><th>calculated MHz</th><th>residual MHz</th><th>sigma MHz</th></tr></thead><tbody>",
        *residual_rows_html,
        "</tbody></table>",
        "<h2>Artifacts</h2>",
        "<ul>",
        "<li>exports/final_geometry.csv</li>",
        "<li>exports/residuals.csv</li>",
        "<li>plots/residuals.png</li>",
        "<li>plots/singular_values.png</li>",
        "<li>plots/convergence.png</li>",
        "<li>plots/predicted_vs_observed.png</li>",
        "<li>plots/normalized_residuals.png</li>",
        "</ul>",
    ]
    if artifacts and artifacts.get("internal_prior_sensitivity_csv") is not None:
        out_lines.extend(
            [
                "<h2>Prior Sensitivity</h2>",
                "<p class='muted'>See <code>exports/internal_prior_sensitivity.csv</code> for coordinate-level sensitivity labels.</p>",
            ]
        )
    out_lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


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

    if rows:
        obs = [float(r["target_mhz"]) for r in rows]
        pred = [float(r["calculated_mhz"]) for r in rows]
        lo = min(obs + pred)
        hi = max(obs + pred)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(obs, pred, c="#264653", edgecolors="white", linewidth=0.6, s=55)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#e76f51", linewidth=1.1)
        ax.set_xlabel("Observed (MHz)")
        ax.set_ylabel("Predicted (MHz)")
        ax.set_title("Predicted vs Observed")
        fig.tight_layout()
        path = plots_dir / "predicted_vs_observed.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        sigma = [max(float(r.get("sigma_mhz", 1.0)), 1e-12) for r in rows]
        z = [float(r["residual_mhz"]) / s for r, s in zip(rows, sigma)]
        labels = [f"{r['isotopologue']} {r['component']}" for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(+3.0, color="#b22222", linewidth=0.7, linestyle="--")
        ax.axhline(-3.0, color="#b22222", linewidth=0.7, linestyle="--")
        ax.bar(range(len(z)), z, color="#1f7a8c")
        ax.set_ylabel("Normalized residual (sigma)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        path = plots_dir / "normalized_residuals.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_outputs(result: dict[str, Any]) -> dict[str, Path | list[Path]]:
    """Write CSV, Markdown, and plot artifacts for a completed generic run."""
    from runner.reporting import (
        export_conformer_summary_json,
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

    # Kraitchman rs analysis + inertial defects (needs full A/B/C per species).
    kraitchman_csv = None
    try:
        kr = kraitchman_run_analysis(result["best"]["coords"], iso_snapshot)
        result["kraitchman"] = kr
        if kr and (kr["rows"] or kr["defects"]):
            kraitchman_csv = exports_dir / "kraitchman_rs.csv"
            write_kraitchman_csv(kraitchman_csv, kr)
            artifacts["kraitchman_csv"] = kraitchman_csv
    except Exception as exc:  # diagnostics must never break the run
        result["kraitchman"] = None
        result.setdefault("warnings", []).append(f"kraitchman: {exc}")

    # Conformer-mixture diagnostics (if present in optimization history).
    hist = list(result.get("best", {}).get("history", []) or [])
    conformer_rows = []
    for h in hist:
        weights = h.get("conformer_weights")
        if weights is None:
            continue
        for j, w in enumerate(list(weights)):
            conformer_rows.append(
                {
                    "iteration": int(h.get("iteration", 0)),
                    "conformer_index": int(j),
                    "weight": float(w),
                    "freq_rms_mhz": float(h.get("freq_rms", np.nan)),
                    "prior_wrms": float(h.get("prior_wrms", np.nan))
                    if h.get("prior_wrms") is not None
                    else np.nan,
                }
            )
    if conformer_rows:
        conf_csv = exports_dir / "conformer_weights_history.csv"
        with conf_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["iteration", "conformer_index", "weight", "freq_rms_mhz", "prior_wrms"],
            )
            writer.writeheader()
            for r in conformer_rows:
                writer.writerow(r)
        artifacts["conformer_weights_history_csv"] = conf_csv

    # Ensemble summary (names, sources, energies, generation diagnostics) as built
    # by run_generic. Written independently of the per-iteration weight history,
    # which is only populated when the optimizer refits weights each iteration.
    conformer_summary = dict(result.get("best", {}).get("conformer_summary") or {})
    if conformer_rows:
        final_iter = max(x["iteration"] for x in conformer_rows)
        final = [r for r in conformer_rows if r["iteration"] == final_iter]
        conformer_summary["n_iterations_with_conformer_data"] = len(
            {r["iteration"] for r in conformer_rows}
        )
        conformer_summary["final_weights"] = [
            {"conformer_index": int(r["conformer_index"]), "weight": float(r["weight"])}
            for r in final
        ]
    if conformer_summary:
        artifacts["conformer_summary_json"] = export_conformer_summary_json(
            conformer_summary, exports_dir / "conformer_summary.json"
        )

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
        from backend.internal.internal_fit import InternalCoordinateSet, spectral_jacobian_q, build_internal_priors
        from backend.spectral.spectral import SpectralEngine
        from backend.uncertainty import uncertainty_table, compute_uncertainty
        from backend.internal.identifiability import identifiability_table
        from backend.priors.prior_sensitivity import classify_prior_dominance, prior_sensitivity_analysis

        ic_cfg = cfg.get("internal_coordinates", {}) or {}
        use_dihedrals = bool(ic_cfg.get("use_dihedrals", False))
        damping = max(float(ic_cfg.get("damping", 1e-6)), 1e-14)
        sigma_bond = float(ic_cfg.get("prior_sigma_bond", 0.04))
        sigma_angle_deg = float(ic_cfg.get("prior_sigma_angle_deg", 2.0))
        sigma_dihedral_deg = float(ic_cfg.get("prior_sigma_dihedral_deg", 15.0))
        ip_cfg = cfg.get("internal_priors", {}) or {}

        coord_set = InternalCoordinateSet(result["best"]["coords"], result["elems"], use_dihedrals=use_dihedrals)
        B_active = coord_set.active_B_matrix(result["best"]["coords"])
        if B_active.shape[0] > 0:
            Bplus = InternalCoordinateSet.damped_pseudoinverse(B_active, damping)
            J_spectral, residual_w = SpectralEngine(iso_snapshot).stacked(result["best"]["coords"])
            Jq = spectral_jacobian_q(J_spectral, Bplus)
            _, _, sigma_prior, prior_meta = build_internal_priors(
                coord_set,
                result["best"]["coords"],
                sigma_bond=sigma_bond,
                sigma_angle_deg=sigma_angle_deg,
                sigma_dihedral_deg=sigma_dihedral_deg,
                prior_mode=str(ip_cfg.get("mode", "soft")).strip().lower(),
                prior_specs=list(ip_cfg.get("user_priors", []) or []),
                elems=result["elems"],
                freeze_sigma_floor=float(ip_cfg.get("freeze_sigma_floor", 1e-6)),
                spectral_jacobian_q=Jq,
                adaptive_config=dict(ip_cfg.get("adaptive", {}) or {}),
                return_metadata=True,
            )
            prior_meta_map = {str(m.get("name", "")): m for m in prior_meta}

            dominance = classify_prior_dominance(Jq, sigma_prior, coord_set.active_names())
            sensitivity_rows = prior_sensitivity_analysis(
                coord_set,
                result["best"]["coords"],
                Jq,
                residual_w,
                sigma_bond=sigma_bond,
                sigma_angle_deg=sigma_angle_deg,
                sigma_dihedral_deg=sigma_dihedral_deg,
                prior_mode=str(ip_cfg.get("mode", "soft")).strip().lower(),
                prior_specs=list(ip_cfg.get("user_priors", []) or []),
                elems=result["elems"],
                freeze_sigma_floor=float(ip_cfg.get("freeze_sigma_floor", 1e-6)),
                spectral_jacobian_q=Jq,
                adaptive_config=dict(ip_cfg.get("adaptive", {}) or {}),
                sv_rel_threshold=1e-3,
                lambda_reg=damping,
            )
            unc_rows = uncertainty_table(
                coord_set,
                result["best"]["coords"],
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
                dominance_labels=dominance,
                sensitivity_rows=sensitivity_rows,
                residual_w=residual_w,
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
                        "prior_dominance",
                        "prior_sensitivity",
                        "prior_delta",
                        "prior_delta_unit",
                        "prior_source",
                        "prior_mode",
                        "prior_target",
                        "prior_target_unit",
                        "prior_sigma",
                        "prior_sigma_unit",
                        "prior_confidence",
                        "prior_notes",
                        "prior_identifiability_score",
                        "prior_sigma_scale",
                        "chi2_scale",
                    ],
                )
                writer.writeheader()
                for r in unc_rows:
                    meta = prior_meta_map.get(str(r.get("name", "")), {})
                    target = meta.get("target_value")
                    sig_v = meta.get("sigma")
                    units = str(meta.get("units", ""))
                    kind = str(meta.get("kind", ""))
                    if kind in {"angle", "dihedral"} and units in {"radian", "rad"}:
                        target = np.degrees(float(target)) if target is not None else target
                        sig_v = np.degrees(float(sig_v)) if sig_v is not None else sig_v
                        units = "deg"
                    elif kind == "bond" and units in {"angstrom", "a", "å"}:
                        units = "Å"
                    row = dict(r)
                    row.update(
                        {
                            "prior_source": meta.get("source", ""),
                            "prior_mode": meta.get("mode", ""),
                            "prior_target": target,
                            "prior_target_unit": units,
                            "prior_sigma": sig_v,
                            "prior_sigma_unit": units,
                            "prior_confidence": meta.get("confidence"),
                            "prior_notes": meta.get("notes"),
                            "prior_identifiability_score": meta.get("identifiability_score"),
                            "prior_sigma_scale": meta.get("sigma_scale"),
                        }
                    )
                    writer.writerow(row)
            artifacts["internal_uncertainty_csv"] = unc_csv

            prior_csv = exports_dir / "internal_prior_provenance.csv"
            with prior_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "name",
                        "kind",
                        "atoms",
                        "atom_types",
                        "bond_order",
                        "source",
                        "mode",
                        "target_value",
                        "target_unit",
                        "sigma",
                        "sigma_unit",
                        "confidence",
                        "notes",
                        "identifiability_score",
                        "sigma_scale",
                        "global_sigma_scale",
                    ],
                )
                writer.writeheader()
                for m in prior_meta:
                    kind = str(m.get("kind", ""))
                    units = str(m.get("units", ""))
                    target = m.get("target_value")
                    sig_v = m.get("sigma")
                    if kind in {"angle", "dihedral"} and units in {"radian", "rad"}:
                        target = np.degrees(float(target)) if target is not None else target
                        sig_v = np.degrees(float(sig_v)) if sig_v is not None else sig_v
                        units = "deg"
                    elif kind == "bond" and units in {"angstrom", "a", "å"}:
                        units = "Å"
                    writer.writerow(
                        {
                            "name": m.get("name"),
                            "kind": kind,
                            "atoms": "-".join(str(int(a) + 1) for a in (m.get("atoms") or ())),
                            "atom_types": "-".join(str(a) for a in (m.get("atom_types") or ())),
                            "bond_order": m.get("bond_order"),
                            "source": m.get("source"),
                            "mode": m.get("mode"),
                            "target_value": target,
                            "target_unit": units,
                            "sigma": sig_v,
                            "sigma_unit": units,
                            "confidence": m.get("confidence"),
                            "notes": m.get("notes"),
                            "identifiability_score": m.get("identifiability_score"),
                            "sigma_scale": m.get("sigma_scale"),
                            "global_sigma_scale": m.get("global_sigma_scale"),
                        }
                    )
            artifacts["internal_prior_provenance_csv"] = prior_csv

            sens_csv = exports_dir / "internal_prior_sensitivity.csv"
            with sens_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "name",
                        "delta",
                        "unit",
                        "sensitivity_label",
                        "q_0p5x",
                        "q_1p0x",
                        "q_2p0x",
                    ],
                )
                writer.writeheader()
                for r in sensitivity_rows:
                    writer.writerow(r)
            artifacts["internal_prior_sensitivity_csv"] = sens_csv

            # Same chi2 inflation as the uncertainty table above, so the exported
            # covariance and the exported std_errs describe the same posterior.
            cov, _, _, _ = compute_uncertainty(
                Jq,
                sigma_prior=sigma_prior,
                lambda_reg=damping,
                residual_w=residual_w,
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
    report_html = run_dir / "report.html"
    write_html_report(report_html, result, artifacts=artifacts)
    plot_paths = write_plots(run_dir, result)
    artifacts["report_md"] = report_md
    artifacts["report_html"] = report_html
    artifacts["plots"] = plot_paths
    report_payload = {
        "name": result.get("name"),
        "run_dir": str(run_dir),
        "elems": result.get("elems", []),
        "best": {
            "idx": result.get("best", {}).get("idx"),
            "freq_rms": result.get("best", {}).get("freq_rms"),
            "energy": result.get("best", {}).get("energy"),
            "coords": np.asarray(result.get("best", {}).get("coords", []), dtype=float).tolist(),
            "history": list(result.get("best", {}).get("history", []) or []),
        },
        "score": result.get("score", {}),
        "residual_rows": rows,
        "singular_values": list(result.get("singular_values", [])),
    }
    report_payload_path = run_dir / "report_payload.json"
    report_payload_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    artifacts["report_payload_json"] = report_payload_path
    # Artifact manifest for quick navigation.
    def _jsonify_artifact_value(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [str(p) if isinstance(p, Path) else p for p in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    manifest = {
        "written_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "artifacts": {k: _jsonify_artifact_value(v) for k, v in artifacts.items()},
    }
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifacts["artifact_manifest_json"] = manifest_path
    return artifacts


def rebuild_report_from_run_dir(run_dir: Path | str) -> dict[str, Any]:
    """Regenerate report + plots from an existing managed run directory."""
    run_path = Path(run_dir).expanduser().resolve()
    payload_path = run_path / "report_payload.json"
    if not payload_path.is_file():
        raise ConfigError(
            f"Cannot rebuild report for '{run_path}': missing report_payload.json. "
            "Run `quantize run <config>` first with artifacts enabled."
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {"name", "run_dir", "elems", "best", "residual_rows"}
    missing = [k for k in required if k not in payload]
    if missing:
        raise ConfigError(f"Malformed report payload: missing keys {missing}.")

    report_md = run_path / "report.md"
    report_html = run_path / "report.html"
    write_markdown_report(report_md, payload, artifacts=None)
    write_html_report(report_html, payload, artifacts=None)
    plot_paths = write_plots(run_path, payload)
    return {
        "report_md": report_md,
        "report_html": report_html,
        "plots": plot_paths,
    }
