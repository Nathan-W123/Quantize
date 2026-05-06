"""
Report generation and CSV/JSON exports for rovibrational corrections.

These functions consume the same isotopologue dictionaries used by
:mod:`backend.spectral.SpectralEngine`.  When a ``rovib_correction`` field
(:class:`backend.correction_models.RovibCorrection`) is attached, full
provenance is included.  Otherwise the legacy ``alpha_constants`` path is
used and missing fields are reported as ``n/a``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.correction_models import COMPONENTS, RovibCorrection


def _fmt(v, fmt=".4f"):
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "n/a"
    return format(x, fmt)


def _component_label(comp):
    c = int(comp)
    return COMPONENTS[c] if 0 <= c < 3 else f"R{c}"


def _correction_components(iso: dict):
    """Yield per-row tuples (label, B0, sigma, alpha, sigma_alpha, delta_vib,
    delta_elec, delta_bob, delta_total, sigma_delta, source, method, basis,
    backend, status, warnings)."""
    obs = np.asarray(iso.get("obs_constants", []), dtype=float)
    sigma = np.asarray(iso.get("sigma_constants", np.zeros_like(obs)), dtype=float)
    alpha_iso = np.asarray(iso.get("alpha_constants", np.zeros_like(obs)), dtype=float)
    idx = np.asarray(iso.get("component_indices", list(range(obs.size))), dtype=int)

    rc = iso.get("rovib_correction")
    if isinstance(rc, RovibCorrection):
        alpha_full = rc.alpha_vector()
        delta_vib = rc.delta_vib_vector()
        delta_elec = rc.delta_elec_vector()
        delta_bob = rc.delta_bob_vector()
        delta_total = rc.delta_total_vector()
        sigma_delta = rc.sigma_delta_vector()
        source = rc.source
        method = rc.method
        basis = rc.basis
        backend = rc.backend
        status = rc.status
        warns = "; ".join(rc.warnings) if rc.warnings else ""
    else:
        alpha_full = np.full(3, np.nan)
        for k, c in enumerate(idx):
            if 0 <= int(c) < 3 and k < alpha_iso.size:
                alpha_full[int(c)] = float(alpha_iso[k])
        delta_vib = 0.5 * np.where(np.isfinite(alpha_full), alpha_full, np.nan)
        delta_elec = np.zeros(3)
        delta_bob = np.zeros(3)
        delta_total = delta_vib + delta_elec + delta_bob
        sigma_delta = np.full(3, np.nan)
        source = iso.get("rovib_source", "iso_alpha")
        method = None
        basis = None
        backend = None
        status = "legacy_alpha"
        warns = ""

    sigma_corr_iso = iso.get("sigma_correction_constants")
    sigma_corr_arr = (
        np.asarray(sigma_corr_iso, dtype=float).ravel()
        if sigma_corr_iso is not None
        else None
    )

    delta_total_iso = iso.get("delta_total_constants")
    delta_total_arr = (
        np.asarray(delta_total_iso, dtype=float).ravel()
        if delta_total_iso is not None
        else None
    )

    rows = []
    for k, c in enumerate(idx):
        ci = int(c)
        if not (0 <= ci < 3):
            continue
        b0 = float(obs[k]) if k < obs.size else float("nan")
        sig = float(sigma[k]) if k < sigma.size else float("nan")
        a_v = float(alpha_full[ci])
        sig_alpha = (
            2.0 * float(sigma_delta[ci])
            if np.isfinite(sigma_delta[ci])
            else float("nan")
        )
        dv = float(delta_vib[ci])
        de = float(delta_elec[ci])
        db = float(delta_bob[ci])
        dt = float(delta_total[ci])
        if delta_total_arr is not None and k < delta_total_arr.size and np.isfinite(delta_total_arr[k]):
            dt = float(delta_total_arr[k])
        sd = float(sigma_delta[ci])
        if sigma_corr_arr is not None and k < sigma_corr_arr.size and np.isfinite(sigma_corr_arr[k]):
            sd = float(sigma_corr_arr[k])
        be = b0 + dt if np.isfinite(dt) else b0 + (0.5 * a_v if np.isfinite(a_v) else 0.0)
        sigma_be = (
            float(np.sqrt(max(sig, 0.0) ** 2 + max(sd, 0.0) ** 2))
            if np.isfinite(sd)
            else sig
        )
        rows.append(
            {
                "label": _component_label(ci),
                "B0_exp_MHz": b0,
                "sigma_B0_MHz": sig,
                "alpha_MHz": a_v,
                "sigma_alpha_MHz": sig_alpha,
                "delta_vib_MHz": dv,
                "delta_elec_MHz": de,
                "delta_bob_MHz": db,
                "delta_total_MHz": dt,
                "Be_target_MHz": be,
                "sigma_Be_target_MHz": sigma_be,
                "source": source,
                "method": method or "",
                "basis": basis or "",
                "backend": backend or "",
                "status": status,
                "warnings": warns,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Markdown report section
# ---------------------------------------------------------------------------


def generate_rovib_report_section(isotopologues: Iterable[dict]) -> str:
    """Return a markdown string summarising the rovib corrections per isotopologue."""
    lines = ["## Rovibrational corrections", ""]
    header = (
        "| Iso | Comp | B0 (MHz) | sigma_B0 | delta_vib | delta_elec | delta_bob "
        "| delta_total | Be target | source | status |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    any_rows = False
    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        for r in _correction_components(iso):
            any_rows = True
            lines.append(
                "| {iso} | {comp} | {b0} | {sb0} | {dv} | {de} | {db} | {dt} "
                "| {be} | {src} | {st} |".format(
                    iso=name,
                    comp=r["label"],
                    b0=_fmt(r["B0_exp_MHz"]),
                    sb0=_fmt(r["sigma_B0_MHz"]),
                    dv=_fmt(r["delta_vib_MHz"]),
                    de=_fmt(r["delta_elec_MHz"]),
                    db=_fmt(r["delta_bob_MHz"]),
                    dt=_fmt(r["delta_total_MHz"]),
                    be=_fmt(r["Be_target_MHz"]),
                    src=r["source"],
                    st=r["status"],
                )
            )

    if not any_rows:
        lines.append("| _no isotopologue rows_ | | | | | | | | | | |")

    # Warnings block.
    warn_lines = []
    for iso in isotopologues:
        rc = iso.get("rovib_correction")
        if isinstance(rc, RovibCorrection) and rc.warnings:
            warn_lines.append(f"- **{iso.get('name', 'iso')}**:")
            for w in rc.warnings:
                warn_lines.append(f"    - {w}")
    if warn_lines:
        lines.append("")
        lines.append("### Warnings")
        lines.extend(warn_lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV / JSON exports
# ---------------------------------------------------------------------------


_CSV_FIELDS = [
    "isotopologue",
    "component",
    "B0_exp_MHz",
    "sigma_B0_MHz",
    "alpha_MHz",
    "sigma_alpha_MHz",
    "delta_vib_MHz",
    "delta_elec_MHz",
    "delta_bob_MHz",
    "delta_total_MHz",
    "Be_target_MHz",
    "sigma_Be_target_MHz",
    "source",
    "method",
    "basis",
    "backend",
    "status",
    "warnings",
]


def export_rovib_corrections_csv(isotopologues: Iterable[dict], path) -> Path:
    """Write the per-component correction table to ``path`` (CSV)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for iso in isotopologues:
            name = str(iso.get("name", "iso"))
            for r in _correction_components(iso):
                writer.writerow(
                    {
                        "isotopologue": name,
                        "component": r["label"],
                        "B0_exp_MHz": r["B0_exp_MHz"],
                        "sigma_B0_MHz": r["sigma_B0_MHz"],
                        "alpha_MHz": r["alpha_MHz"],
                        "sigma_alpha_MHz": r["sigma_alpha_MHz"],
                        "delta_vib_MHz": r["delta_vib_MHz"],
                        "delta_elec_MHz": r["delta_elec_MHz"],
                        "delta_bob_MHz": r["delta_bob_MHz"],
                        "delta_total_MHz": r["delta_total_MHz"],
                        "Be_target_MHz": r["Be_target_MHz"],
                        "sigma_Be_target_MHz": r["sigma_Be_target_MHz"],
                        "source": r["source"],
                        "method": r["method"],
                        "basis": r["basis"],
                        "backend": r["backend"],
                        "status": r["status"],
                        "warnings": r["warnings"],
                    }
                )
    return p


def export_semi_experimental_targets_csv(isotopologues: Iterable[dict], path) -> Path:
    """Write the (B0, delta_total, Be) targets used by the optimizer."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "isotopologue",
        "component",
        "B0_exp_MHz",
        "delta_total_MHz",
        "Be_target_MHz",
        "sigma_Be_target_MHz",
    ]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for iso in isotopologues:
            name = str(iso.get("name", "iso"))
            for r in _correction_components(iso):
                writer.writerow(
                    {
                        "isotopologue": name,
                        "component": r["label"],
                        "B0_exp_MHz": r["B0_exp_MHz"],
                        "delta_total_MHz": r["delta_total_MHz"],
                        "Be_target_MHz": r["Be_target_MHz"],
                        "sigma_Be_target_MHz": r["sigma_Be_target_MHz"],
                    }
                )
    return p


def export_rovib_warnings_json(isotopologues: Iterable[dict], path) -> Path:
    """Write a JSON dump of warnings + provenance per isotopologue."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"isotopologues": []}
    for iso in isotopologues:
        rc = iso.get("rovib_correction")
        item = {
            "name": str(iso.get("name", "iso")),
            "warnings": [],
            "source": None,
            "status": None,
            "method": None,
            "basis": None,
            "backend": None,
            "geometry_hash": None,
        }
        if isinstance(rc, RovibCorrection):
            item.update(
                warnings=list(rc.warnings or []),
                source=rc.source,
                status=rc.status,
                method=rc.method,
                basis=rc.basis,
                backend=rc.backend,
                geometry_hash=rc.geometry_hash,
            )
        blob["isotopologues"].append(item)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    return p
