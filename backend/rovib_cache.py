"""
File-based cache for ORCA rovib (VPT2) corrections.

A cache key is the SHA-256 hash of the geometry, isotopologue masses, method,
basis, backend, and resolution mode.  Hits avoid expensive ORCA reruns when
the user re-fits the same molecule.

Layout::

    cache_dir/rovib_cache/<key[:16]>/<isotopologue_label>.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from backend.correction_models import RovibCorrection


def make_rovib_cache_key(
    coords,
    masses,
    method: Optional[str],
    basis: Optional[str],
    backend: Optional[str],
    mode: Optional[str],
) -> str:
    """Return a hexadecimal SHA-256 cache key for the given parameters."""
    payload = {
        "coords": np.asarray(coords, dtype=float).tolist(),
        "masses": np.asarray(masses, dtype=float).tolist(),
        "method": str(method) if method is not None else None,
        "basis": str(basis) if basis is not None else None,
        "backend": str(backend) if backend is not None else None,
        "mode": str(mode) if mode is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(cache_dir, key: str, isotopologue_label: str) -> Path:
    safe_label = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(isotopologue_label)
    ) or "iso"
    return Path(cache_dir) / "rovib_cache" / key[:16] / f"{safe_label}.json"


def load_cached_correction(
    cache_dir,
    key: str,
    isotopologue_label: str,
) -> Optional[RovibCorrection]:
    """Load a previously serialised :class:`RovibCorrection` or ``None`` if absent."""
    p = _cache_path(cache_dir, key, isotopologue_label)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    valid_keys = {f.name for f in fields(RovibCorrection)}
    payload = {k: v for k, v in data.get("correction", {}).items() if k in valid_keys}
    if "warnings" in payload and not isinstance(payload["warnings"], list):
        payload["warnings"] = []
    if "isotopologue" not in payload:
        payload["isotopologue"] = str(isotopologue_label)
    try:
        return RovibCorrection(**payload)
    except TypeError:
        return None


def save_cached_correction(
    cache_dir,
    key: str,
    isotopologue_label: str,
    correction: RovibCorrection,
    raw_output_paths: Optional[Iterable[str]] = None,
) -> Path:
    """Serialise a :class:`RovibCorrection` to JSON in the cache directory."""
    p = _cache_path(cache_dir, key, isotopologue_label)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "key": key,
        "isotopologue": str(isotopologue_label),
        "raw_output_paths": (
            [str(x) for x in raw_output_paths] if raw_output_paths else []
        ),
        "correction": asdict(correction),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, sort_keys=True, default=str)
    return p
