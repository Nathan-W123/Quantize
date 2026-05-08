from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG_SCHEMA_VERSION = "1.0"


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize config shape with stable defaults.

    This is intentionally lightweight and backward-compatible.
    """
    c = deepcopy(cfg)
    c.setdefault("schema_version", CONFIG_SCHEMA_VERSION)
    c.setdefault("name", str(c.get("molecule", "quantize_run")))
    c.setdefault("preset", "BALANCED")
    c.setdefault("rng_seed", 42)
    c.setdefault("write_xyz", False)

    output = c.setdefault("output", {}) or {}
    output.setdefault("root", "runs")
    output.setdefault("artifacts", True)
    c["output"] = output

    if "coordinate_mode" not in c:
        c["coordinate_mode"] = "internal"

    ic = c.setdefault("internal_coordinates", {}) or {}
    ic.setdefault("use_dihedrals", False)
    ic.setdefault("damping", 1e-6)
    ic.setdefault("microiterations", 20)
    ic.setdefault("prior_weight", 1.0)
    ic.setdefault("prior_sigma_bond", 0.04)
    ic.setdefault("prior_sigma_angle_deg", 2.0)
    ic.setdefault("prior_sigma_dihedral_deg", 15.0)
    c["internal_coordinates"] = ic

    ip = c.setdefault("internal_priors", {}) or {}
    ip.setdefault("mode", "soft")
    ip.setdefault("freeze_sigma_floor", 1e-6)
    ip.setdefault("user_priors", [])
    ad = ip.setdefault("adaptive", {}) or {}
    ad.setdefault("identifiability_gamma", 2.0)
    ad.setdefault("min_sigma_scale", 0.5)
    ad.setdefault("max_sigma_scale", 3.0)
    ip["adaptive"] = ad
    c["internal_priors"] = ip

    cm = c.setdefault("conformer_mixture", {}) or {}
    cm.setdefault("enabled", False)
    cm.setdefault("weight_mode", "fixed")
    cm.setdefault("temperature_k", 298.15)
    cm.setdefault("conformers", [])
    c["conformer_mixture"] = cm

    return c
