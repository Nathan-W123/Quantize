from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.spectral.correction_models import COMPONENTS, RovibCorrection


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


# â”€â”€ Existing helpers (backward-compatible) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        comp_label = _COMP_LABELS.get(c, f"R{c}")

        if mode_str == "none":
            out[i] = 0.0
            sources_per_component.append("none")

        elif mode_str in ("user_only", "manual_alpha"):
            if cand_user is not None and np.isfinite(cand_user):
                out[i] = float(cand_user)
                sources_per_component.append("user")
            else:
                sources_per_component.append("none")

        elif mode_str == "strict_user":
            if cand_user is None or not np.isfinite(cand_user):
                raise ValueError(
                    f"strict_user mode: no user alpha value for component {comp_label} "
                    f"of isotopologue '{isotopologue_name}'."
                )
            out[i] = float(cand_user)
            sources_per_component.append("user")

        elif mode_str == "orca_only":
            if cand_orca is not None and np.isfinite(cand_orca):
                out[i] = float(cand_orca)
                sources_per_component.append("orca")
            else:
                sources_per_component.append("none")

        elif mode_str == "strict_backend":
            if cand_orca is None or not np.isfinite(cand_orca):
                raise ValueError(
                    f"strict_backend mode: no backend alpha value for component {comp_label} "
                    f"of isotopologue '{isotopologue_name}'."
                )
            out[i] = float(cand_orca)
            sources_per_component.append("orca")

        elif mode_str == "manual_delta":
            # Delta is supplied directly via correction_table; alpha is not used.
            # Force alpha contribution to zero to avoid stale legacy alpha carry-through.
            out[i] = 0.0
            sources_per_component.append("manual_delta")

        else:  # hybrid_auto
            if cand_user is not None and np.isfinite(cand_user):
                out[i] = float(cand_user)
                sources_per_component.append("user")
            elif cand_orca is not None and np.isfinite(cand_orca):
                out[i] = float(cand_orca)
                sources_per_component.append("orca")
            else:
                sources_per_component.append("none")
                warnings.append(f"no alpha for component {comp_label} in hybrid_auto mode")

    # Populate RovibCorrection fields from resolved values.
    alpha_full = np.full(3, np.nan)
    for i, comp in enumerate(idx):
        c = int(comp)
        if 0 <= c < 3:
            alpha_full[c] = out[i]
    correction.alpha_A = None if not np.isfinite(alpha_full[0]) else float(alpha_full[0])
    correction.alpha_B = None if not np.isfinite(alpha_full[1]) else float(alpha_full[1])
    correction.alpha_C = None if not np.isfinite(alpha_full[2]) else float(alpha_full[2])
    correction.source = "+".join(dict.fromkeys(s for s in sources_per_component if s != "none")) or "none"
    if all(s == "none" for s in sources_per_component):
        correction.status = "missing_component"
    elif any(s == "none" for s in sources_per_component) or warnings:
        correction.status = "partial"
    else:
        correction.status = "ok"
    correction.warnings = warnings

    return out, correction


# â”€â”€ Data model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _propagate_sigma(sigma_exp: float, correction_sigmas: list) -> float:
    """sigma_eff = sqrt(sigma_exp^2 + sum(sigma_corr_i^2)); None entries skipped."""
    total = float(sigma_exp) ** 2
    for s in correction_sigmas:
        if s is not None:
            v = float(s)
            if v > 0.0:
                total += v * v
    return float(total ** 0.5)


# â”€â”€ Correction resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#: Electron-to-proton mass ratio, and the largest rotational g-tensor
#: component a molecule plausibly shows. Together these bound the electronic
#: correction when no g-tensor is supplied: |delta| <= (m_e/m_p) * g_max * B.
#: Typical |g| runs from about 0.01 for heavy molecules to a few tenths for
#: light hydrides, so 0.7 is a conservative ceiling.
_M_E_OVER_M_P = 5.446170214e-4
_G_MAX_TYPICAL = 0.7


def resolve_corrections(
    isotopologues: list,
    correction_table: Optional[dict] = None,
    mode: str = "hybrid_auto",
    sigma_vib_fraction: float = 0.1,
    elems: Optional[list] = None,
    correction_elec: bool = False,
    sigma_elec_fraction: float = 0.1,
    correction_bob_params: Optional[dict] = None,
    g_tensor: Optional[dict] = None,
) -> list:
    """
    For each isotopologue Ã— component pair, resolve all corrections and return
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
        Keys: isotopologue name â†’ {component â†’ spec_dict}.
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
        Floored at 100% when no g_tensor is supplied, since the fallback formula
        is only order-of-magnitude.
    g_tensor : dict or None
        Rotational g-tensor components, ``{"A": g_aa, "B": g_bb, "C": g_cc}``.
        When given, the standard delta_elec = -(m_e/m_p) * g * B_obs is used
        instead of the crude 1/M_total fallback. Components may be negative.
    correction_bob_params : dict or None
        Per-element BOB u-parameters. When supplied, computes:
            delta_bob = -Î£_a (m_e / m_a) * u_a^X
        Format: {elem_symbol: {comp_label: u_value_or_dict}}
        where each component value is a float (u, sigma unknown) or
        a dict {"u": float, "sigma_u": float|None}.
        This requires elems to be supplied.

    Returns
    -------
    list of CorrectedSpectralTarget (one per iso Ã— component)
    """
    from backend.spectral.correction_models import (
        vpt2_delta_b, electronic_delta_b, bob_delta_b
    )

    mode = str(mode).strip().lower()
    ctbl = correction_table or {}
    targets = []

    for iso in isotopologues:
        name = str(iso.get("name", "iso"))
        obs = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso.get("alpha_constants", np.zeros(len(obs))), dtype=float)
        if iso.get("sigma_constants") is not None:
            sigma = np.asarray(iso["sigma_constants"], dtype=float)
        else:
            from backend.spectral.spectral import default_sigma_constants
            sigma = default_sigma_constants(
                obs, iso.get("component_indices", list(range(len(obs)))))
        # The coherent model error -- the B0-versus-Be gap -- as distinct from
        # measurement precision. This is the part a vibrational correction
        # supersedes; see the sigma_exp adjustment below.
        sys_sig_in = iso.get("sigma_systematic_constants")
        sigma_sys = (np.asarray(sys_sig_in, dtype=float).ravel()
                     if sys_sig_in is not None else None)
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

            # â”€â”€ Vibrational correction (priority-ordered) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

            # â”€â”€ Electronic correction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if correction_elec and total_mass > 0.0:
                g_val = (g_tensor or {}).get(comp_label)
                g_val = _finite_or_none(g_val)
                delta_e = electronic_delta_b(b0, total_mass, g_value=g_val)
                if g_val is not None:
                    notes_e = f"-(m_e/m_p)*g*B_obs with g_{comp_label}={g_val:g}"
                    frac = sigma_elec_fraction
                    flags: list = []
                else:
                    # No g-tensor: apply nothing, and say how much is unknown.
                    #
                    # The old fallback, -(m_e/M_total)*B_obs, is not the standard
                    # formula. It is off by roughly (M_total * g) and takes the
                    # wrong sign whenever g is negative, which is common. A
                    # wrong-signed point estimate cannot be rescued by widening
                    # its sigma: a value -x with sigma x spans [-2x, 0] and never
                    # reaches a true value of +y. Reporting zero with an honest
                    # bound is unbiased and strictly better.
                    #
                    # The bound comes from the real formula, delta = -(m_e/m_p)*g*B,
                    # evaluated at the largest g a molecule plausibly shows.
                    delta_e = 0.0
                    notes_e = (
                        "no g_tensor supplied: electronic correction not applied; "
                        f"sigma covers |g| up to {_G_MAX_TYPICAL:g}. Supply "
                        "g_tensor for the standard -(m_e/m_p)*g*B_obs correction."
                    )
                    frac = 0.0
                    flags = ["electronic_no_g_tensor", "electronic_not_applied"]
                if g_val is not None:
                    sig_e = abs(delta_e) * frac if frac > 0.0 else None
                else:
                    sig_e = _M_E_OVER_M_P * _G_MAX_TYPICAL * abs(float(b0))
                records.append(CorrectionRecord(
                    isotopologue_label=name,
                    component=comp_label,
                    delta_mhz=delta_e,
                    sigma_mhz=sig_e,
                    source="computed",
                    method="elec",
                    notes=notes_e,
                    quality_flags=flags,
                ))

            # â”€â”€ Born-Oppenheimer Breakdown correction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                        notes="BOB: -Î£_a (m_e/m_a)*u_a",
                    ))

            total_delta = sum(r.delta_mhz for r in records)
            b_e_se = b0 + total_delta

            # Do not charge for an error that has just been removed.
            #
            # sigma_exp is dominated by the r_0-versus-r_e model gap: the
            # measured constant is a ground-state one, the fitted structure is
            # rigid, and that mismatch is a few tenths of a percent while the
            # frequency itself is measured to about one part in 1e7. Applying a
            # vibrational correction is precisely the act of removing that gap
            # -- so continuing to carry it in the weighting sigma double-counts
            # it, and what remains is only how well the correction is known.
            #
            # Measured on formyl fluoride's B at B3LYP/6-31G(d): sigma_exp 58.8
            # MHz against a correction sigma of 11.7, so the corrected target
            # was weighted 5x too loosely -- and weight goes as 1/sigma^2, so
            # the data carried 25x less influence than it had earned. The
            # engine was paying for the correction and then discarding it.
            #
            # Only the coherent (systematic) part is removed, and only when a
            # real vibrational correction was applied. Whatever the caller did
            # not declare systematic stays, and the relative sigma floor in
            # SpectralEngine still applies downstream, so no weight can run
            # away.
            sigma_exp_eff = sigma_exp
            # Vibrational records are whatever is neither the electronic nor
            # the BOB term nor the explicit no-correction placeholder. Matching
            # on a list of method names instead silently missed
            # "VPT2_semidiag", which is what the Hessian-built table stamps --
            # i.e. every correction this engine generates itself.
            corrected_vib = any(
                r.method not in ("elec", "BOB", "none") and r.source != "none"
                and r.delta_mhz != 0.0
                for r in records
            )
            if corrected_vib:
                s_sys = None
                if sigma_sys is not None and k < sigma_sys.size:
                    s_sys = float(sigma_sys[k])
                elif iso.get("sigma_constants") is None:
                    # Auto-derived sigma is model gap end to end.
                    s_sys = sigma_exp
                if s_sys is not None and s_sys > 0.0:
                    sigma_exp_eff = float(
                        max(sigma_exp ** 2 - min(s_sys, sigma_exp) ** 2, 0.0) ** 0.5
                    )
            sigma_eff = _propagate_sigma(sigma_exp_eff, [r.sigma_mhz for r in records])

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


# â”€â”€ Apply corrections back to isotopologue dicts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def apply_corrections_to_isotopologues(
    isotopologues: list,
    corrected_targets: list,
) -> list:
    """
    Return new isotopologue dicts with corrected equilibrium targets.

    For each corrected target, the corresponding isotopologue dict is updated:
      obs_constants  â†’ Be,SE  (the corrected equilibrium value)
      alpha_constants â†’ 0.0   (correction already absorbed)
      sigma_constants â†’ sigma_eff (propagated uncertainty)

    The corrected dicts are otherwise identical to the originals and remain
    compatible with SpectralEngine without any changes to that class.
    """
    # Build lookup: (iso_name, component_index) â†’ CorrectedSpectralTarget
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
                # alpha[k] remains 0.0 â€” correction absorbed into obs

        new_iso = dict(iso)
        new_iso["obs_constants"] = obs
        new_iso["alpha_constants"] = alpha
        new_iso["sigma_constants"] = sigma
        result.append(new_iso)

    return result


# â”€â”€ Quality-control checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                    f"{r.delta_mhz:+.1f} MHz is {risk_ratio:.1f}Ã— the experimental "
                    f"sigma ({sigma_floor:.3f} MHz) but correction uncertainty is unknown â€” "
                    "consider supplying sigma_mhz in the correction table."
                )

    return warnings_out


# â”€â”€ Human-readable summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
