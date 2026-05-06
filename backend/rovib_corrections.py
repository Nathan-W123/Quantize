"""
Rovibrational correction resolution.

Given user-supplied corrections, an ORCA VPT2 parse, and any pre-existing
alpha values stored on an isotopologue, this module produces:

* A resolved alpha vector for the components actually being fitted.
* A :class:`RovibCorrection` describing where each component came from.

It also helps build :class:`RovibCorrection` instances directly from
isotopologue dictionaries (for cases where the user supplies decomposed
deltas instead of alpha constants) and computes effective sigmas that fold
in correction uncertainty in quadrature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.correction_models import COMPONENTS, RovibCorrection


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
        cand_user = _pick(c, user)
        cand_orca = _pick(c, parsed)

        chosen: Optional[float] = None
        chosen_source = "existing"

        if mode_str == "none":
            chosen = None
        elif mode_str == "manual_delta":
            # Delta-mode: the user supplies decomposed deltas elsewhere; do
            # not modify the alpha array at all.
            chosen = None
        elif mode_str == "manual_alpha":
            if cand_user is not None:
                chosen, chosen_source = cand_user, "user_manual"
        elif mode_str == "user_only":
            if cand_user is not None:
                chosen, chosen_source = cand_user, "user"
        elif mode_str == "orca_only":
            if cand_orca is not None:
                chosen, chosen_source = cand_orca, "orca"
        elif mode_str == "strict_user":
            if cand_user is None:
                raise ValueError(
                    f"strict_user mode: no user-supplied alpha for component "
                    f"'{COMPONENTS[c] if 0 <= c < 3 else c}' on "
                    f"isotopologue '{isotopologue_name}'."
                )
            chosen, chosen_source = cand_user, "user"
        elif mode_str == "strict_backend":
            if cand_orca is None:
                raise ValueError(
                    f"strict_backend mode: no backend-supplied alpha for component "
                    f"'{COMPONENTS[c] if 0 <= c < 3 else c}' on "
                    f"isotopologue '{isotopologue_name}'."
                )
            chosen, chosen_source = cand_orca, "orca"
        else:  # hybrid_auto
            if cand_user is not None:
                chosen, chosen_source = cand_user, "user"
            elif cand_orca is not None:
                chosen, chosen_source = cand_orca, "orca"

        if chosen is not None:
            out[i] = float(chosen)
        sources_per_component.append(chosen_source)

        # Populate the full A/B/C alpha attributes on the correction object.
        if 0 <= c < 3:
            attr = f"alpha_{COMPONENTS[c]}"
            current = getattr(correction, attr)
            if current is None and chosen is not None:
                setattr(correction, attr, float(chosen))

    if all(s == "existing" for s in sources_per_component):
        correction.status = "fallback_existing"
        correction.source = "existing"
    elif all(s == "user" or s == "user_manual" for s in sources_per_component):
        correction.status = "ok"
        correction.source = "user"
    elif all(s == "orca" for s in sources_per_component):
        correction.status = "ok"
        correction.source = "orca"
    elif "user" in sources_per_component or "user_manual" in sources_per_component:
        correction.status = "ok"
        correction.source = "hybrid"
    elif "orca" in sources_per_component:
        correction.status = "ok"
        correction.source = "orca"
    else:
        correction.status = "unknown"
        correction.source = "unknown"

    correction.warnings = warnings
    return out, correction


# ---------------------------------------------------------------------------
# Building corrections from isotopologue dictionaries
# ---------------------------------------------------------------------------


def _maybe_set(corr: RovibCorrection, prefix: str, vec):
    """Set ``{prefix}_{A,B,C}`` on ``corr`` from a length-3 vector if finite."""
    if vec is None:
        return
    arr = np.asarray(vec, dtype=float).ravel()
    if arr.size < 3:
        return
    for i, comp in enumerate(COMPONENTS):
        v = float(arr[i])
        if np.isfinite(v):
            setattr(corr, f"{prefix}_{comp}", v)


def build_correction_from_iso(
    iso: dict,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    backend: Optional[str] = None,
) -> RovibCorrection:
    """Build a :class:`RovibCorrection` from an isotopologue dictionary.

    Recognises both the legacy ``alpha_constants`` (length matches selected
    components) and the new fully-specified keys:
    ``delta_vib_constants``, ``delta_elec_constants``, ``delta_bob_constants``,
    ``sigma_correction_constants``, plus an existing ``rovib_correction`` field
    that overrides everything else.
    """
    existing = iso.get("rovib_correction")
    if isinstance(existing, RovibCorrection):
        return existing

    name = str(iso.get("name", ""))
    corr = RovibCorrection(
        isotopologue=name, method=method, basis=basis, backend=backend
    )

    # ---- alpha_constants is component-aligned, expand to A/B/C -------------
    idx = iso.get("component_indices")
    alpha_iso = iso.get("alpha_constants")
    if alpha_iso is not None and idx is not None:
        idx_arr = np.asarray(idx, dtype=int)
        alpha_arr = np.asarray(alpha_iso, dtype=float).ravel()
        for k, comp_idx in enumerate(idx_arr):
            if 0 <= int(comp_idx) < 3 and k < alpha_arr.size:
                v = float(alpha_arr[k])
                if np.isfinite(v):
                    setattr(corr, f"alpha_{COMPONENTS[int(comp_idx)]}", v)

    # ---- explicit decomposed deltas ----------------------------------------
    _maybe_set(corr, "delta_vib", iso.get("delta_vib_constants"))
    _maybe_set(corr, "delta_elec", iso.get("delta_elec_constants"))
    _maybe_set(corr, "delta_bob", iso.get("delta_bob_constants"))

    # ---- correction uncertainty --------------------------------------------
    sigma_corr = iso.get("sigma_correction_constants")
    if sigma_corr is not None:
        sigma_arr = np.asarray(sigma_corr, dtype=float).ravel()
        if idx is not None:
            idx_arr = np.asarray(idx, dtype=int)
            for k, comp_idx in enumerate(idx_arr):
                if 0 <= int(comp_idx) < 3 and k < sigma_arr.size:
                    v = float(sigma_arr[k])
                    if np.isfinite(v):
                        setattr(
                            corr,
                            f"sigma_delta_{COMPONENTS[int(comp_idx)]}",
                            v,
                        )
        elif sigma_arr.size >= 3:
            for i, comp in enumerate(COMPONENTS):
                v = float(sigma_arr[i])
                if np.isfinite(v):
                    setattr(corr, f"sigma_delta_{comp}", v)

    # Heuristic provenance: source comes from the iso dict if provided.
    source = iso.get("rovib_source")
    if source:
        corr.source = str(source)
    elif iso.get("delta_vib_constants") is not None:
        corr.source = "user_delta"
        corr.status = "ok"
    elif alpha_iso is not None:
        corr.source = "iso_alpha"
        corr.status = "ok"

    return corr


# ---------------------------------------------------------------------------
# Effective sigma (quadrature with correction uncertainty)
# ---------------------------------------------------------------------------


def effective_sigma_constants(iso: dict) -> np.ndarray:
    """Return component-aligned effective sigma in MHz.

    Combines ``sigma_constants`` (observation noise) with correction
    uncertainty via quadrature.  The correction uncertainty source is, in
    priority order:

    1. ``rovib_correction.sigma_delta_vector()`` mapped onto the selected
       ``component_indices``.
    2. ``sigma_correction_constants`` aligned to component_indices.
    3. zero (no correction uncertainty contribution).
    """
    sigma_obs = np.asarray(
        iso.get("sigma_constants", []), dtype=float
    ).ravel()
    n = sigma_obs.size
    sigma_corr = np.zeros(n, dtype=float)

    rc = iso.get("rovib_correction")
    idx = np.asarray(iso.get("component_indices", list(range(n))), dtype=int)
    if isinstance(rc, RovibCorrection):
        sd = rc.sigma_delta_vector()  # length 3
        for k in range(n):
            c = int(idx[k]) if k < len(idx) else -1
            if 0 <= c < 3 and np.isfinite(sd[c]):
                sigma_corr[k] = float(sd[c])
    else:
        sc = iso.get("sigma_correction_constants")
        if sc is not None:
            sc_arr = np.asarray(sc, dtype=float).ravel()
            for k in range(n):
                if k < sc_arr.size and np.isfinite(sc_arr[k]):
                    sigma_corr[k] = max(float(sc_arr[k]), 0.0)

    return np.sqrt(np.maximum(sigma_obs, 0.0) ** 2 + sigma_corr ** 2)
