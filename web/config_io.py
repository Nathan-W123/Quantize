"""Form state to config dict, and back.

This is the join between the two ways of running Quantize. The browser builds
the SAME dict a YAML input file loads to, so it goes through the same
validate_config and the same runner; there is no second interpretation of a
case. ``config_to_yaml`` lets a case built in the browser be saved and run with
``python -m cli run case.yaml``, and ``config_to_form`` lets a hand-written
YAML file be opened in the form. Round-tripping either direction is covered by
tests, because the moment the two drift the UI starts lying about what it will
run.

Nothing here validates. Validation is runner.usability.validate_config, called
on the result, so the browser reports exactly the errors the CLI would.
"""

from __future__ import annotations

from typing import Any

#: Keys written only when they differ from what normalize_config would supply.
#: A config full of restated defaults is hard to read and hard to diff, and the
#: point of the download button is a file someone can keep.
_OMIT_IF_BLANK = ("symmetry", "method", "basis")


def _floats(value, name: str) -> list[float]:
    """Parse a comma/space separated list of numbers from a form field."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = str(value or "").replace(",", " ").split()
    out = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {item!r} is not a number") from None
    return out


def _coords(value) -> list[list[float]]:
    """Parse an XYZ-ish block: one atom per line, three numbers each."""
    if isinstance(value, (list, tuple)):
        return [[float(x) for x in row] for row in value]
    rows = []
    for n, line in enumerate(str(value or "").strip().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        # Tolerate a leading element symbol, as pasted from an .xyz file.
        if parts and not parts[0].lstrip("+-").replace(".", "", 1).isdigit():
            parts = parts[1:]
        if len(parts) != 3:
            raise ValueError(f"geometry line {n}: expected 3 numbers, got {len(parts)}")
        rows.append([float(x) for x in parts])
    return rows


def _elements(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [tok for tok in str(value or "").replace(",", " ").split() if tok]


def form_to_config(form: dict[str, Any]) -> dict[str, Any]:
    """Build the config dict the runner consumes from the browser's form state.

    Raises ValueError with a field-shaped message for anything unparseable, so
    the browser can say which box is wrong before the runner is involved.
    """
    get = form.get
    cfg: dict[str, Any] = {
        "name": str(get("name") or "quantize_run").strip(),
        "coordinate_mode": str(get("coordinate_mode") or "internal"),
        "preset": str(get("preset") or "BALANCED"),
        "elements": _elements(get("elements")),
    }

    seed = get("rng_seed")
    if seed not in (None, ""):
        cfg["rng_seed"] = int(seed)

    # How often the Hessian -- and so the correction's expansion point -- is
    # recomputed. This is the real control over how closely the correction
    # follows the fitted geometry: _apply_harmonic_alpha_corrections re-expands
    # alpha at the CURRENT geometry on every Hessian rebuild, which is the
    # "direct" tracking scheme the benchmark measured as the best of three.
    #
    # It belongs under "optimizer", not at the top level: run_generic only
    # copies a key into the optimiser when it appears in cfg["optimizer"], so a
    # top-level one is read by nothing and silently ignored.
    if get("hess_recalc_every") not in (None, ""):
        cfg["optimizer"] = {"hess_recalc_every": int(get("hess_recalc_every"))}

    symmetry = str(get("symmetry") or "").strip()
    if symmetry:
        cfg["symmetry"] = symmetry

    # Four ways in, and typing Cartesian coordinates is the last resort rather
    # than the default: run_generic can fetch a PubChem 3D conformer from a
    # name, a CID or a SMILES string, or build one from a bond list. The key
    # each branch reads was checked against that code -- "pubchem" wants
    # "identifier", and an earlier version of this wrote "name", which the
    # runner would have rejected at run time with the form reporting no error.
    geom_method = str(get("geometry_method") or "smiles")
    geometry: dict[str, Any] = {"method": geom_method}
    if geom_method == "coords":
        geometry["coords_angstrom"] = _coords(get("coords_angstrom"))
    elif geom_method == "pubchem":
        geometry["identifier"] = str(get("pubchem_identifier") or "").strip()
    elif geom_method == "smiles":
        geometry["smiles"] = str(get("smiles") or "").strip()
    elif geom_method == "bonds":
        pairs = []
        for line in str(get("bonds") or "").replace(",", " ").split("\n"):
            toks = line.split()
            if len(toks) == 2:
                pairs.append([int(toks[0]), int(toks[1])])
        geometry["bonds"] = pairs
    cfg["geometry"] = geometry

    isos = []
    for i, row in enumerate(get("isotopologues") or [], start=1):
        label = str(row.get("name") or f"iso{i}").strip()
        comps = [c for c in (row.get("components") or ["A", "B", "C"]) if c in ("A", "B", "C")]
        entry: dict[str, Any] = {
            "name": label,
            "masses": _floats(row.get("masses"), f"{label} masses"),
            "components": comps,
            "obs_b0_mhz": _floats(row.get("obs_b0_mhz"), f"{label} obs_b0_mhz"),
        }
        sigma = _floats(row.get("sigma_mhz"), f"{label} sigma_mhz")
        if sigma:
            entry["sigma_mhz"] = sigma
        # validate_config requires alpha_mhz on every isotopologue, one value
        # per listed component -- it is the slot the correction writes into, so
        # it has to exist even when nothing has been computed yet. Omitting it
        # for a blank form field made a freshly opened page fail validation.
        alpha = _floats(row.get("alpha_mhz"), f"{label} alpha_mhz")
        entry["alpha_mhz"] = alpha if alpha else [0.0] * len(comps)
        isos.append(entry)
    cfg["isotopologues"] = isos

    quantum: dict[str, Any] = {"backend": str(get("quantum_backend") or "pyscf_hf")}
    for key in ("method", "basis"):
        val = str(get(key) or "").strip()
        if val:
            quantum[key] = val
    cfg["quantum"] = quantum

    # Key names here are exactly the ones run_generic reads out of the
    # rovibrational_corrections block. They were checked against that code
    # rather than guessed: an invented key is silently ignored by the runner,
    # which is the one failure mode a settings UI must not have.
    corrections: dict[str, Any] = {
        "mode": str(get("corrections_mode") or "hybrid_auto"),
        "harmonic_from_hessian": bool(get("harmonic_from_hessian", True)),
        "anharmonic_from_hessian": bool(get("anharmonic_from_hessian", True)),
        "nonconvergent_policy": str(get("nonconvergent_policy") or "warn"),
        "electronic_correction": bool(get("electronic_correction", False)),
        "use_builtin_bob": bool(get("use_builtin_bob", True)),
        # Scheme choices. Reachable from a config only since the runner was
        # taught to read them; before that they were benchmark-script-only.
        "harmonic_scheme": str(get("harmonic_scheme") or "watson"),
        "cubic_scheme": str(get("cubic_scheme") or "cartesian"),
    }
    for key in ("harmonic_sigma_fraction", "anharmonic_fd_delta_ang",
                "sigma_vib_fraction", "sigma_elec_fraction", "cd_sigma_fraction",
                "freq_scale", "lam_freq_cm"):
        val = get(key)
        if val not in (None, ""):
            corrections[key] = float(val)
    if get("rovib_recalc_every") not in (None, ""):
        corrections["rovib_recalc_every"] = int(get("rovib_recalc_every"))

    # Centrifugal distortion lives in this same block -- there is no separate
    # centrifugal_distortion section, whatever its own vocabulary suggests.
    if get("harmonic_cd_from_hessian"):
        corrections["harmonic_cd_from_hessian"] = True
    if get("fit_cd_constants"):
        corrections["fit_cd_constants"] = True
        # The runner warns and does nothing when cd_weight is left at 0, so a
        # UI that offers the switch has to offer the weight with it.
        corrections["cd_weight"] = float(get("cd_weight") or 1.0)
    cfg["rovibrational_corrections"] = corrections

    ic: dict[str, Any] = {"use_dihedrals": bool(get("use_dihedrals", False))}
    for key in ("damping", "prior_sigma_bond", "prior_sigma_angle_deg",
                "prior_sigma_dihedral_deg"):
        val = get(key)
        if val not in (None, ""):
            ic[key] = float(val)
    cfg["internal_coordinates"] = ic

    cfg["internal_priors"] = {"mode": str(get("internal_priors_mode") or "soft")}

    cfg["output"] = {
        "root": str(get("output_root") or "output/runs"),
        "artifacts": bool(get("artifacts", True)),
    }
    return cfg


def config_to_form(cfg: dict[str, Any]) -> dict[str, Any]:
    """The inverse: pre-fill the form from a config someone wrote by hand."""
    geom = cfg.get("geometry") or {}
    quantum = cfg.get("quantum") or {}
    corr = cfg.get("rovibrational_corrections") or {}
    ic = cfg.get("internal_coordinates") or {}
    ip = cfg.get("internal_priors") or {}
    out = cfg.get("output") or {}

    coords = geom.get("coords_angstrom") or []
    form: dict[str, Any] = {
        "name": cfg.get("name", ""),
        "coordinate_mode": cfg.get("coordinate_mode", "internal"),
        "preset": cfg.get("preset", "BALANCED"),
        "rng_seed": cfg.get("rng_seed", ""),
        "symmetry": cfg.get("symmetry", ""),
        "elements": " ".join(str(e) for e in (cfg.get("elements") or [])),
        "geometry_method": geom.get("method", "smiles"),
        "coords_angstrom": "\n".join(
            " ".join(f"{float(x):.6f}" for x in row) for row in coords),
        "pubchem_identifier": geom.get("identifier", ""),
        "smiles": geom.get("smiles", ""),
        "bonds": "\n".join(f"{a} {b}" for a, b in (geom.get("bonds") or [])),
        "quantum_backend": quantum.get("backend", "pyscf_hf"),
        "method": quantum.get("method", ""),
        "basis": quantum.get("basis", ""),
        "corrections_mode": corr.get("mode", "hybrid_auto"),
        "harmonic_from_hessian": bool(corr.get("harmonic_from_hessian", True)),
        "anharmonic_from_hessian": bool(corr.get("anharmonic_from_hessian", True)),
        "nonconvergent_policy": corr.get("nonconvergent_policy", "warn"),
        "harmonic_scheme": corr.get("harmonic_scheme", "watson"),
        "cubic_scheme": corr.get("cubic_scheme", "cartesian"),
        "freq_scale": corr.get("freq_scale", ""),
        "lam_freq_cm": corr.get("lam_freq_cm", ""),
        "hess_recalc_every": (cfg.get("optimizer") or {}).get("hess_recalc_every", ""),
        "electronic_correction": bool(corr.get("electronic_correction", False)),
        "use_builtin_bob": bool(corr.get("use_builtin_bob", True)),
        "harmonic_sigma_fraction": corr.get("harmonic_sigma_fraction", ""),
        "anharmonic_fd_delta_ang": corr.get("anharmonic_fd_delta_ang", ""),
        "sigma_vib_fraction": corr.get("sigma_vib_fraction", ""),
        "sigma_elec_fraction": corr.get("sigma_elec_fraction", ""),
        "rovib_recalc_every": corr.get("rovib_recalc_every", ""),
        "harmonic_cd_from_hessian": bool(corr.get("harmonic_cd_from_hessian", False)),
        "cd_sigma_fraction": corr.get("cd_sigma_fraction", ""),
        "fit_cd_constants": bool(corr.get("fit_cd_constants", False)),
        "cd_weight": corr.get("cd_weight", ""),
        "use_dihedrals": bool(ic.get("use_dihedrals", False)),
        "damping": ic.get("damping", ""),
        "prior_sigma_bond": ic.get("prior_sigma_bond", ""),
        "prior_sigma_angle_deg": ic.get("prior_sigma_angle_deg", ""),
        "prior_sigma_dihedral_deg": ic.get("prior_sigma_dihedral_deg", ""),
        "internal_priors_mode": ip.get("mode", "soft"),
        "output_root": out.get("root", "output/runs"),
        "artifacts": bool(out.get("artifacts", True)),
        "isotopologues": [],
    }
    for iso in cfg.get("isotopologues") or []:
        form["isotopologues"].append({
            "name": iso.get("name", ""),
            "masses": " ".join(str(m) for m in (iso.get("masses") or [])),
            "components": list(iso.get("components") or ["A", "B", "C"]),
            "obs_b0_mhz": " ".join(str(v) for v in (iso.get("obs_b0_mhz") or [])),
            "sigma_mhz": " ".join(str(v) for v in (iso.get("sigma_mhz") or [])),
            "alpha_mhz": " ".join(str(v) for v in (iso.get("alpha_mhz") or [])),
        })
    return form


def config_to_yaml(cfg: dict[str, Any]) -> str:
    """Serialise a config as YAML, falling back to JSON when PyYAML is absent.

    JSON is valid YAML, so the fallback still loads through the CLI's
    load_config -- it is only less pleasant to read.
    """
    header = ("# Written by the Quantize web UI. Run it with:\n"
              "#   python -m cli run <this file>\n")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        import json
        return header + json.dumps(cfg, indent=2) + "\n"
    class _Dumper(yaml.SafeDumper):
        """Keep short numeric lists on one line.

        Block style turns a coordinate into five lines of "- - 0.0" and masses
        into one line per atom, which makes the downloaded file far harder to
        read or edit than the hand-written configs it sits beside. Numbers go
        inline; everything else stays block.
        """

    def _numeric_list(dumper, data):
        flow = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in data) and 0 < len(data) <= 12
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=flow)

    _Dumper.add_representer(list, _numeric_list)
    body = yaml.dump(cfg, Dumper=_Dumper, sort_keys=False,
                     default_flow_style=False, allow_unicode=True)
    return header + body
