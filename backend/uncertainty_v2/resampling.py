from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def resample_isotopologues_gaussian(
    isotopologues: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Perturb observed rotational constants by their reported sigma values."""
    out: list[dict[str, Any]] = []
    for iso in isotopologues:
        entry = deepcopy(iso)
        obs = np.asarray(iso.get("obs_constants", []), dtype=float)
        sigma = np.asarray(iso.get("sigma_constants", np.ones_like(obs) * 0.1), dtype=float)
        if obs.shape != sigma.shape:
            raise ValueError(
                f"obs_constants and sigma_constants shape mismatch for isotopologue "
                f"{iso.get('name', '<unnamed>')}: {obs.shape} vs {sigma.shape}"
            )
        entry["obs_constants"] = obs + rng.normal(0.0, sigma)
        out.append(entry)
    return out

