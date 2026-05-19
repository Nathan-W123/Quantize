"""
Loader for user-supplied rovibrational correction tables.

Two CSV layouts are supported, distinguished by their column names:

1. Alpha-constant format::

       isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status

   Each row contributes a single A/B/C alpha value to the named isotopologue.

2. Direct-delta format::

       isotopologue,component,delta_vib_MHz,sigma_delta_vib_MHz,source,method,basis,status

   Each row contributes a vibrational delta directly (Be ≈ B0 + delta_vib).

Both layouts are validated; on any error, a :class:`ValueError` with a clear
message is raised.  The loader returns a ``dict[str, RovibCorrection]`` keyed
by isotopologue name.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from backend.correction_models import COMPONENTS, RovibCorrection


_ALPHA_HEADERS = {
    "isotopologue",
    "component",
    "alpha_MHz",
    "sigma_alpha_MHz",
    "source",
}
_DELTA_HEADERS = {
    "isotopologue",
    "component",
    "delta_vib_MHz",
    "sigma_delta_vib_MHz",
    "source",
}


def _detect_layout(headers: Iterable[str]) -> str:
    h = set(headers)
    if "alpha_MHz" in h:
        missing = _ALPHA_HEADERS - h
        if missing:
            raise ValueError(
                f"alpha-format CSV missing columns: {sorted(missing)}"
            )
        return "alpha"
    if "delta_vib_MHz" in h:
        missing = _DELTA_HEADERS - h
        if missing:
            raise ValueError(
                f"delta-format CSV missing columns: {sorted(missing)}"
            )
        return "delta"
    raise ValueError(
        "CSV must contain either 'alpha_MHz' or 'delta_vib_MHz' column."
    )


def _validate_component(name: str, comp: str) -> str:
    cu = (comp or "").strip().upper()
    if cu not in COMPONENTS:
        raise ValueError(
            f"Invalid component '{comp}' for isotopologue '{name}'. "
            f"Use one of {COMPONENTS}."
        )
    return cu


def _to_finite_float(value: str, field: str, name: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Missing value for '{field}' on isotopologue '{name}'."
        )
    try:
        v = float(value)
    except ValueError as e:
        raise ValueError(
            f"Non-numeric '{field}' for isotopologue '{name}': {value!r}"
        ) from e
    if not np.isfinite(v):
        raise ValueError(
            f"Non-finite '{field}' for isotopologue '{name}': {value!r}"
        )
    return v


def _to_optional_nonneg(value: str, field: str, name: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        v = float(value)
    except ValueError as e:
        raise ValueError(
            f"Non-numeric '{field}' for isotopologue '{name}': {value!r}"
        ) from e
    if not np.isfinite(v):
        raise ValueError(
            f"Non-finite '{field}' for isotopologue '{name}': {value!r}"
        )
    if v < 0.0:
        raise ValueError(
            f"Negative uncertainty '{field}' for isotopologue '{name}': {v}"
        )
    return v


def _validate_source_method(source: str, method: Optional[str], where: str) -> None:
    src = (source or "").strip()
    if not src:
        raise ValueError(f"{where}: 'source' is required.")
    meth = (method or "").strip()
    if src.lower() != "manual" and not meth:
        raise ValueError(
            f"{where}: 'method' is required when source is not 'manual'."
        )


def _load_rovib_correction_table_yaml(
    path: "Path",
    known_isotopologues: Optional[Iterable[str]] = None,
) -> "dict[str, RovibCorrection]":
    """Load corrections from a YAML file into RovibCorrection objects.

    Expected YAML shape::

        H2O:
          B:
            alpha_MHz: 1.234
            sigma_alpha_MHz: 0.05
            source: orca
            method: VPT2
            basis: def2-TZVPP
          C:
            delta_vib_MHz: 0.617
            sigma_delta_vib_MHz: 0.025
            source: user

        H2_18O:
          B:
            alpha_MHz: 1.100
            source: orca
            method: VPT2

    Each isotopologue is a key mapping to component dicts (A/B/C).
    Each component dict must have either ``alpha_MHz`` or ``delta_vib_MHz``.
    """
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML correction tables: pip install PyYAML"
        ) from exc

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML correction table must be a top-level mapping, got {type(data).__name__}"
        )

    known_set = set(known_isotopologues) if known_isotopologues is not None else None
    out: dict[str, RovibCorrection] = {}

    for iso_name, comp_specs in data.items():
        name = str(iso_name).strip()
        if not name:
            raise ValueError("YAML correction table: isotopologue name cannot be blank.")
        if known_set is not None and name not in known_set:
            raise ValueError(
                f"YAML correction table: isotopologue '{name}' not in known list "
                f"{sorted(known_set)}."
            )
        if not isinstance(comp_specs, dict):
            raise ValueError(
                f"YAML correction table: '{name}' must map component labels to spec dicts, "
                f"got {type(comp_specs).__name__}."
            )

        corr = out.setdefault(
            name,
            RovibCorrection(isotopologue=name),
        )

        for comp_raw, spec in comp_specs.items():
            comp = _validate_component(name, str(comp_raw))
            if not isinstance(spec, dict):
                raise ValueError(
                    f"YAML correction table: '{name}.{comp_raw}' must be a dict, "
                    f"got {type(spec).__name__}."
                )

            source = str(spec.get("source", "")).strip()
            method = str(spec.get("method", "")).strip() or None
            basis = str(spec.get("basis", "")).strip() or None
            status = str(spec.get("status", "ok")).strip() or "ok"
            _validate_source_method(
                source, method, f"YAML correction table '{name}.{comp_raw}'"
            )

            if not corr.source or corr.source == "unknown":
                corr.source = source
            elif source and source not in corr.source:
                corr.source = corr.source + "+" + source
            if method and not corr.method:
                corr.method = method
            if basis and not corr.basis:
                corr.basis = basis
            corr.status = status

            if "alpha_MHz" in spec:
                v = _to_finite_float(str(spec["alpha_MHz"]), "alpha_MHz", name)
                sigma_str = str(spec.get("sigma_alpha_MHz", "")).strip()
                sigma = _to_optional_nonneg(sigma_str or None, "sigma_alpha_MHz", name)
                setattr(corr, f"alpha_{comp}", v)
                if sigma is not None:
                    setattr(corr, f"sigma_delta_{comp}", 0.5 * sigma)
            elif "delta_vib_MHz" in spec:
                v = _to_finite_float(str(spec["delta_vib_MHz"]), "delta_vib_MHz", name)
                sigma_str = str(spec.get("sigma_delta_vib_MHz", "")).strip()
                sigma = _to_optional_nonneg(sigma_str or None, "sigma_delta_vib_MHz", name)
                setattr(corr, f"delta_vib_{comp}", v)
                if sigma is not None:
                    setattr(corr, f"sigma_delta_{comp}", sigma)
            else:
                raise ValueError(
                    f"YAML correction table: '{name}.{comp_raw}' must have "
                    f"'alpha_MHz' or 'delta_vib_MHz'."
                )

    return out


def load_rovib_correction_table(
    path,
    known_isotopologues: Optional[Iterable[str]] = None,
) -> dict[str, RovibCorrection]:
    """Load a correction table from a CSV or YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the file. Extension determines format:
        ``.csv`` uses the two-layout CSV parser;
        ``.yaml`` / ``.yml`` uses the YAML loader.
    known_isotopologues : iterable of str, optional
        If supplied, every ``isotopologue`` value in the table must appear
        here, otherwise a :class:`ValueError` is raised.

    Returns
    -------
    dict[str, RovibCorrection]
        One :class:`RovibCorrection` per isotopologue name.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Correction table not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_rovib_correction_table_yaml(path, known_isotopologues)
    # Default: CSV

    known_set = set(known_isotopologues) if known_isotopologues is not None else None
    out: dict[str, RovibCorrection] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header row.")
        layout = _detect_layout(reader.fieldnames)

        for row_idx, row in enumerate(reader, start=2):  # start=2: header is row 1
            name = (row.get("isotopologue") or "").strip()
            if not name:
                raise ValueError(
                    f"Row {row_idx}: missing 'isotopologue' name."
                )
            if known_set is not None and name not in known_set:
                raise ValueError(
                    f"Row {row_idx}: isotopologue '{name}' not in known list "
                    f"{sorted(known_set)}."
                )

            comp = _validate_component(name, row.get("component", ""))
            source = (row.get("source") or "").strip()
            method_row = (row.get("method") or "").strip() or None
            _validate_source_method(
                source, method_row, f"Row {row_idx} isotopologue '{name}'"
            )

            corr = out.setdefault(
                name,
                RovibCorrection(
                    isotopologue=name,
                    method=method_row,
                    basis=(row.get("basis") or "").strip() or None,
                    backend=None,
                    source=source,
                    status=(row.get("status") or "").strip() or "ok",
                ),
            )
            # Update method/basis/source/status if previously empty.
            if not corr.method and (row.get("method") or "").strip():
                corr.method = row["method"].strip()
            if not corr.basis and (row.get("basis") or "").strip():
                corr.basis = row["basis"].strip()
            if (row.get("status") or "").strip():
                corr.status = row["status"].strip()
            if source and source not in corr.source:
                # Preserve mixed-source provenance.
                if corr.source and corr.source != source:
                    corr.source = corr.source + "+" + source
                else:
                    corr.source = source

            if layout == "alpha":
                v = _to_finite_float(row.get("alpha_MHz"), "alpha_MHz", name)
                sigma = _to_optional_nonneg(
                    row.get("sigma_alpha_MHz"), "sigma_alpha_MHz", name
                )
                setattr(corr, f"alpha_{comp}", v)
                if sigma is not None:
                    # Convert sigma_alpha to sigma_delta_vib via the same factor.
                    setattr(corr, f"sigma_delta_{comp}", 0.5 * sigma)
            else:  # delta
                v = _to_finite_float(
                    row.get("delta_vib_MHz"), "delta_vib_MHz", name
                )
                sigma = _to_optional_nonneg(
                    row.get("sigma_delta_vib_MHz"),
                    "sigma_delta_vib_MHz",
                    name,
                )
                setattr(corr, f"delta_vib_{comp}", v)
                if sigma is not None:
                    setattr(corr, f"sigma_delta_{comp}", sigma)

    return out
