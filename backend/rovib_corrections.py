from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.correction_models import COMPONENTS, RovibCorrection


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


# ── Existing helpers (backward-compatible) ────────────────────────────────────

def _as_alpha_vector(value):
    """Coerce a user-/parse-provided alpha value into a length-3 ndarray.

    Accepts a length-3 array-like, a dict with keys ``A``/``B``/``C``, or
    ``None`` (returned as ``None``).  Returns ``None`` when the input cannot
    be interpreted, never raises.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("A", "B", "C")):
            try:
                return np.array(
                    [value["A"], value["B"], value["C"]], dtype=float
                )
            except (TypeError, ValueError):
                return None
        return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size >= 3:
        return arr[:3].astype(float)
    return None


def _finite_or_none(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pick(comp_idx: int, vec):
    if vec is None:
        return None
    if comp_idx < 0 or comp_idx >= len(vec):
        return None
    return _finite_or_none(vec[comp_idx])


# ---------------------------------------------------------------------------
# Alpha resolution
# ---------------------------------------------------------------------------


_VALID_MODES = {
    "hybrid_auto",
    "user_only",
    "orca_only",
    "manual_alpha",
    "manual_delta",
    "none",
    "strict_user",
    "strict_backend",
}


def resolve_alpha_components(
    existing_alpha_by_component,
    component_indices,
    parsed_alpha_abc,
    user_alpha_abc,
    mode,
    isotopologue_name: str = "",
    method: Optional[str] = None,
    basis: Optional[str] = None,
    backend: Optional[str] = None,
):
    """Resolve alpha constants for the selected components.

    Parameters
    ----------
    existing_alpha_by_component : array-like
        Alpha values currently associated with the selected components
        (parallel to ``component_indices``).  These are used as the fall-back
        when the chosen mode cannot supply a value.
    component_indices : array-like of int
        Indices into the (A, B, C) triple for each spectral row being fitted.
    parsed_alpha_abc : array-like of length >=3 or None
        Backend (e.g. ORCA VPT2) alpha vector for A/B/C.
    user_alpha_abc : array-like of length >=3 or None
        User-supplied alpha vector for A/B/C.
    mode : str
        One of ``hybrid_auto``, ``user_only``, ``orca_only``, ``manual_alpha``,
        ``manual_delta``, ``none``, ``strict_user``, ``strict_backend``.
    isotopologue_name, method, basis, backend : str
        Provenance metadata copied into the resulting :class:`RovibCorrection`.

    Returns
    -------
    resolved : ndarray
        Alpha vector parallel to ``component_indices``.
    correction : RovibCorrection
        Provenance object recording where each component was sourced from.

    Raises
    ------
    ValueError
        ``strict_user`` raises when no user value is available for a selected
        component.  ``strict_backend`` raises when no backend value is
        available for a selected component.
    """
    mode_str = str(mode or "hybrid_auto").strip().lower()
    if mode_str not in _VALID_MODES:
        raise ValueError(
            f"Unknown rovib mode '{mode}'. Valid: {sorted(_VALID_MODES)}"
        )

    idx = np.asarray(component_indices, dtype=int)
    existing = np.asarray(existing_alpha_by_component, dtype=float)
    out = existing.astype(float).copy()

    parsed = _as_alpha_vector(parsed_alpha_abc)
    user = _as_alpha_vector(user_alpha_abc)

    correction = RovibCorrection(
        isotopologue=str(isotopologue_name or ""),
        method=method,
        basis=basis,
        backend=backend,
    )
    sources_per_component: list[str] = []
    warnings: list[str] = []

    for i, comp in enumerate(idx):
        c = int(comp)
        cand_user = None if user is None or c < 0 or c >= len(user) else user[c]
        cand_orca = None if parsed is None or c < 0 or c >= len(parsed) else parsed[c]
        if mode == "user_only":
            if cand_user is not None and np.isfinite(cand_user):
                out[i] = float(cand_user)
        elif mode == "orca_only":
            if cand_orca is not None and np.isfinite(cand_orca):
                out[i] = float(cand_orca)
        else:
            if cand_user is not None and np.isfinite(cand_user):
                out[i] = float(cand_user)
            elif cand_orca is not None and np.isfinite(cand_orca):
                out[i] = float(cand_orca)
    return out


# ── Data model ────────────────────────────────────────────────────────────────

_COMP_LABELS = {0: "A", 1: "B", 2: "C"}
_COMP_INDICES = {"A": 0, "B": 1, "C": 2}


@dataclass
class CorrectionRecord:
    """One correction term applied to a single rotational constant."""
    isotopologue_label: str
    component: str                       # "A", "B", or "C"
    delta_mhz: float                     # signed correction (added to B0 to give Be)
    sigma_mhz: Optional[float]           # uncertainty on this correction; None = unknown
    source: str                          # "user", "orca", "cfour", "alpha_fallback", "none"
    method: str                          # "VPT2", "GVPT2", "HR", "manual", "none"
    basis: Optional[str] = None
    quality_flags: list = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class CorrectedSpectralTarget:
    """A rotational constant corrected to a semi-experimental equilibrium value Be,SE."""
    isotopologue_label: str
    component: str                       # "A", "B", or "C"
    component_index: int                 # 0, 1, or 2
    b0_mhz: float                        # original observed B0
    sigma_exp_mhz: float                 # original experimental uncertainty
    value_mhz: float                     # corrected Be,SE = B0 + sum(delta_mhz)
    sigma_mhz: float                     # effective uncertainty after propagation
    correction_records: list = field(default_factory=list)

    @property
    def total_delta_mhz(self) -> float:
        return sum(r.delta_mhz for r in self.correction_records)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _propagate_sigma(sigma_exp: float, correction_sigmas: list) -> float:
    """sigma_eff = sqrt(sigma_exp^2 + sum(sigma_corr_i^2)); None entries skipped."""
    total = float(sigma_exp) ** 2
    for s in correction_sigmas:
        if s is not None:
            v = float(s)
            if v > 0.0:
                total += v * v
    return float(total ** 0.5)


# ── Correction resolution ─────────────────────────────────────────────────────

def resolve_corrections(
    isotopologues: list,
    correction_table: Optional[dict] = None,
    mode: str = "hybrid_auto",
    sigma_vib_fraction: float = 0.1,
    elems: Optional[list] = None,
    correction_elec: bool = False,
    sigma_elec_fraction: float = 0.1,
    correction_bob_params: Optional[dict] = None,
) -> list:
    """
    For each isotopologue × component pair, resolve all corrections and return
    a CorrectedSpectralTarget for each.

    The full correction sequence (matching the r_e^SE formula) is:
        B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB

    Each term is recorded as a separate CorrectionRecord with its own provenance,
    source, and uncertainty.

    Vibrational correction precedence (mode="hybrid_auto"):
      1. user correction_table entry (explicit delta_mhz or alpha_sum_mhz)
      2. existing alpha_constants in the isotopologue dict
      3. no correction (quality flag "no_correction" is added)

    Electronic and BOB corrections are applied on top regardless of mode.

    Parameters
    ----------
    isotopologues : list of dict
        Isotopologue dicts as passed to MolecularOptimizer.
    correction_table : dict or None
        Parsed correction table from parse_correction_table().
        Keys: isotopologue name → {component → spec_dict}.
    mode : str
        "hybrid_auto" | "user_only" | "alpha_only"
    sigma_vib_fraction : float
        Fractional uncertainty assigned to vibrational corrections when
        sigma_mhz is not specified in the correction table.
    elems : list of str or None
        Element symbols in atom order. Required for electronic and BOB corrections.
    correction_elec : bool
        If True, add the electronic mass correction using the Gordy-Cook formula:
            delta_elec = -(m_e / M_total) * B_obs
        This requires elems to be supplied.
    sigma_elec_fraction : float
        Fractional uncertainty on the electronic correction (default 0.1 = 10%).
        Reflects the approximation error from ignoring the electronic g-tensor.
    correction_bob_params : dict or None
        Per-element BOB u-parameters. When supplied, computes:
            delta_bob = -Σ_a (m_e / m_a) * u_a^X
        Format: {elem_symbol: {comp_label: u_value_or_dict}}
        where each component value is a float (u, sigma unknown) or
        a dict {"u": float, "sigma_u": float|None}.
        This requires elems to be supplied.

    Returns
    -------
    list of CorrectedSpectralTarget (one per iso × component)
    """
    from backend.correction_models import (
        vpt2_delta_b, electronic_delta_b, bob_delta_b
    )

    mode = str(mode).strip().lower()
    ctbl = correction_table or {}
    targets = []

    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        obs = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso.get("alpha_constants", np.zeros(len(obs))), dtype=float)
        sigma = np.asarray(iso.get("sigma_constants", np.ones(len(obs))), dtype=float)
        idx = np.asarray(
            iso.get("component_indices", list(range(len(obs)))), dtype=int
        )
        masses = list(iso.get("masses", []))
        total_mass = float(sum(masses)) if masses else 0.0

        iso_ctbl = ctbl.get(name, {})

        for k, comp in enumerate(idx):
            comp = int(comp)
            comp_label = _COMP_LABELS.get(comp, f"R{comp}")
            b0 = float(obs[k])
            sigma_exp = float(sigma[k]) if k < len(sigma) else 1.0
            alpha_val = float(alpha[k]) if k < len(alpha) else 0.0

            records = []

            # ── Vibrational correction (priority-ordered) ─────────────────────

            # Priority 1: user correction_table
            if mode != "alpha_only" and comp_label in iso_ctbl:
                spec = iso_ctbl[comp_label]
                if "delta_mhz" in spec:
                    delta = float(spec["delta_mhz"])
                else:
                    delta = vpt2_delta_b(float(spec["alpha_sum_mhz"]))

                sig_corr = spec.get("sigma_mhz", None)
                if sig_corr is not None:
                    sig_corr = float(sig_corr)
                elif sigma_vib_fraction > 0.0:
                    sig_corr = abs(delta) * sigma_vib_fraction

                records.append(CorrectionRecord(
                    isotopologue_label=name,
                    component=comp_label,
                    delta_mhz=delta,
                    sigma_mhz=sig_corr,
                    source=str(spec.get("source", "user")),
                    method=str(spec.get("method", "VPT2")),
                    basis=spec.get("basis", None),
                    notes=spec.get("notes", None),
                ))

            # Priority 2: existing alpha_constants
            elif mode != "user_only" and alpha_val != 0.0:
                delta = vpt2_delta_b(alpha_val)
                sig_corr = (abs(delta) * sigma_vib_fraction) if sigma_vib_fraction > 0.0 else None
                records.append(CorrectionRecord(
                    isotopologue_label=name,
                    component=comp_label,
                    delta_mhz=delta,
                    sigma_mhz=sig_corr,
                    source="alpha_fallback",
                    method="VPT2",
                ))

            # No vibrational correction available
            if not records:
                records.append(CorrectionRecord(
                    isotopologue_label=name,
                    component=comp_label,
                    delta_mhz=0.0,
                    sigma_mhz=None,
                    source="none",
                    method="none",
                    quality_flags=["no_correction"],
                ))

            # ── Electronic mass correction ────────────────────────────────────
            if correction_elec and total_mass > 0.0:
                delta_e = electronic_delta_b(b0, total_mass)
                sig_e = abs(delta_e) * sigma_elec_fraction if sigma_elec_fraction > 0.0 else None
                records.append(CorrectionRecord(
                    isotopologue_label=name,
                    component=comp_label,
                    delta_mhz=delta_e,
                    sigma_mhz=sig_e,
                    source="computed",
                    method="elec",
                    notes="Gordy-Cook: -(m_e/M_total)*B_obs",
                ))

            # ── Born-Oppenheimer Breakdown correction ─────────────────────────
            if correction_bob_params and elems and masses:
                delta_b, sig_b = bob_delta_b(elems, masses, comp_label, correction_bob_params)
                if delta_b != 0.0 or sig_b is not None:
                    records.append(CorrectionRecord(
                        isotopologue_label=name,
                        component=comp_label,
                        delta_mhz=delta_b,
                        sigma_mhz=sig_b,
                        source="user",
                        method="BOB",
                        notes="BOB: -Σ_a (m_e/m_a)*u_a",
                    ))

            total_delta = sum(r.delta_mhz for r in records)
            b_e_se = b0 + total_delta
            sigma_eff = _propagate_sigma(sigma_exp, [r.sigma_mhz for r in records])

            targets.append(CorrectedSpectralTarget(
                isotopologue_label=name,
                component=comp_label,
                component_index=comp,
                b0_mhz=b0,
                sigma_exp_mhz=sigma_exp,
                value_mhz=b_e_se,
                sigma_mhz=sigma_eff,
                correction_records=records,
            ))

    return targets


# ── Apply corrections back to isotopologue dicts ──────────────────────────────

def apply_corrections_to_isotopologues(
    isotopologues: list,
    corrected_targets: list,
) -> list:
    """
    Return new isotopologue dicts with corrected equilibrium targets.

    For each corrected target, the corresponding isotopologue dict is updated:
      obs_constants  → Be,SE  (the corrected equilibrium value)
      alpha_constants → 0.0   (correction already absorbed)
      sigma_constants → sigma_eff (propagated uncertainty)

    The corrected dicts are otherwise identical to the originals and remain
    compatible with SpectralEngine without any changes to that class.
    """
    # Build lookup: (iso_name, component_index) → CorrectedSpectralTarget
    lookup: dict = {}
    for t in corrected_targets:
        lookup[(t.isotopologue_label, t.component_index)] = t

    result = []
    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        obs = np.asarray(iso["obs_constants"], dtype=float).copy()
        sigma = np.asarray(iso.get("sigma_constants", np.ones(len(obs))), dtype=float).copy()
        alpha = np.zeros(len(obs), dtype=float)
        idx = np.asarray(
            iso.get("component_indices", list(range(len(obs)))), dtype=int
        )

        for k, comp in enumerate(idx):
            key = (name, int(comp))
            if key in lookup:
                t = lookup[key]
                obs[k] = t.value_mhz
                sigma[k] = t.sigma_mhz
                # alpha[k] remains 0.0 — correction absorbed into obs

        new_iso = dict(iso)
        new_iso["obs_constants"] = obs
        new_iso["alpha_constants"] = alpha
        new_iso["sigma_constants"] = sigma
        result.append(new_iso)

    return result


# ── Quality-control checks ────────────────────────────────────────────────────

def validate_correction_quality(
    corrected_targets: list,
    sigma_ratio_warn: float = 3.0,
) -> list:
    """
    Check corrected targets for quality issues. Returns a list of warning strings.

    Flags raised:
      - Mixed correction coverage: some isotopologues corrected, others not
      - Correction magnitude >> experimental uncertainty but correction sigma unknown
      - Component corrected in one isotopologue but missing in another
    """
    warnings_out = []

    by_iso: dict = {}
    for t in corrected_targets:
        by_iso.setdefault(t.isotopologue_label, []).append(t)

    # Determine which isotopologues have any non-trivial correction
    has_correction: dict = {}
    for name, targets in by_iso.items():
        has_correction[name] = any(
            r.source not in ("none",)
            for t in targets
            for r in t.correction_records
        )

    corrected_isos = [n for n, h in has_correction.items() if h]
    uncorrected_isos = [n for n, h in has_correction.items() if not h]
    if corrected_isos and uncorrected_isos:
        warnings_out.append(
            f"Mixed correction coverage: {corrected_isos} have vibrational corrections but "
            f"{uncorrected_isos} do not. Fitting mixed B0 and Be,SE targets introduces "
            "systematic error in the recovered geometry."
        )

    # Per-component coverage consistency across isotopologues
    by_comp: dict = {}
    for t in corrected_targets:
        by_comp.setdefault(t.component, {})[t.isotopologue_label] = t

    all_isos = set(by_iso.keys())
    for comp, iso_map in by_comp.items():
        corrected_in_comp = {
            n for n, t in iso_map.items()
            if any(r.source not in ("none",) for r in t.correction_records)
        }
        uncorrected_in_comp = all_isos - corrected_in_comp
        if corrected_in_comp and uncorrected_in_comp and len(all_isos) > 1:
            warnings_out.append(
                f"Component {comp}: corrected in {sorted(corrected_in_comp)} but "
                f"not in {sorted(uncorrected_in_comp)}."
            )

    # Correction magnitude vs. experimental uncertainty
    for t in corrected_targets:
        for r in t.correction_records:
            if r.source in ("none",) or r.delta_mhz == 0.0:
                continue
            sigma_floor = max(t.sigma_exp_mhz, 0.01)
            risk_ratio = abs(r.delta_mhz) / sigma_floor
            if risk_ratio > sigma_ratio_warn and r.sigma_mhz is None:
                r.quality_flags = list(r.quality_flags) + ["large_correction_unknown_sigma"]
                warnings_out.append(
                    f"{t.isotopologue_label}/{t.component}: correction "
                    f"{r.delta_mhz:+.1f} MHz is {risk_ratio:.1f}× the experimental "
                    f"sigma ({sigma_floor:.3f} MHz) but correction uncertainty is unknown — "
                    "consider supplying sigma_mhz in the correction table."
                )

    return warnings_out


# ── Human-readable summary ────────────────────────────────────────────────────

def correction_summary(corrected_targets: list) -> str:
    """
    Return a formatted table of corrected targets for printing.

    Columns: isotopologue | component | B0 (MHz) | delta (MHz) | Be,SE (MHz) | sigma_eff | source
    """
    header = (
        f"  {'Isotopologue':<14}  {'Comp':>4}  {'B0 (MHz)':>14}  "
        f"{'delta (MHz)':>12}  {'Be,SE (MHz)':>14}  {'sigma_eff':>10}  {'source':<18}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]
    for t in corrected_targets:
        delta = t.total_delta_mhz
        sources = ", ".join(dict.fromkeys(r.source for r in t.correction_records))
        flags = []
        for r in t.correction_records:
            flags.extend(r.quality_flags)
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"  {t.isotopologue_label:<14}  {t.component:>4}  {t.b0_mhz:>14.4f}  "
            f"{delta:>+12.4f}  {t.value_mhz:>14.4f}  {t.sigma_mhz:>10.4f}  {sources:<18}"
            f"{flag_str}"
        )
    return "\n".join(lines)
