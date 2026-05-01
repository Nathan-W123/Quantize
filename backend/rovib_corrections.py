import numpy as np


def _as_alpha_vector(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("A", "B", "C")):
            return np.array([value["A"], value["B"], value["C"]], dtype=float)
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size >= 3:
        return arr[:3].astype(float)
    return None


def resolve_alpha_components(existing_alpha_by_component, component_indices, parsed_alpha_abc, user_alpha_abc, mode):
    """
    Resolve alpha constants for selected components using user table and ORCA parse.
    Precedence by mode:
      hybrid_auto: user -> orca -> existing
      user_only  : user -> existing
      orca_only  : orca -> existing
    """
    mode = str(mode or "hybrid_auto").strip().lower()
    idx = np.asarray(component_indices, dtype=int)
    existing = np.asarray(existing_alpha_by_component, dtype=float)
    out = existing.copy()
    parsed = _as_alpha_vector(parsed_alpha_abc)
    user = _as_alpha_vector(user_alpha_abc)

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
