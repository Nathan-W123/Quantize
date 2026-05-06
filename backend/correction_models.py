"""
Data classes describing rovibrational corrections used to convert observed B0
rotational constants to semi-experimental Be targets.

This module is the bottom-most layer in the rovib correction stack and must not
import from any other backend module.  It provides:

* :class:`RovibCorrection` -- per-isotopologue container for alpha constants and
  the decomposed deltas (vibrational, electronic, BOB) plus uncertainty,
  provenance, and warnings.
* :class:`ParsedRovibResult` -- the output of an ORCA VPT2 parser pass with a
  parse status and the list of warnings/source files consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


COMPONENTS = ("A", "B", "C")


def _component_value(value: Optional[float]) -> float:
    """Return ``value`` as a float or ``np.nan`` if missing."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class RovibCorrection:
    """Per-isotopologue rotational-vibrational correction data.

    The correction can be expressed either as alpha constants (where the
    vibrational delta is ``0.5 * alpha`` to first order) or as direct
    decomposed deltas with optional electronic and Born-Oppenheimer-breakdown
    contributions.  Components are stored individually for A, B, and C; a
    ``None`` value signals that the component is not available.
    """

    isotopologue: str
    alpha_A: Optional[float] = None
    alpha_B: Optional[float] = None
    alpha_C: Optional[float] = None
    delta_vib_A: Optional[float] = None
    delta_vib_B: Optional[float] = None
    delta_vib_C: Optional[float] = None
    delta_elec_A: float = 0.0
    delta_elec_B: float = 0.0
    delta_elec_C: float = 0.0
    delta_bob_A: float = 0.0
    delta_bob_B: float = 0.0
    delta_bob_C: float = 0.0
    sigma_delta_A: Optional[float] = None
    sigma_delta_B: Optional[float] = None
    sigma_delta_C: Optional[float] = None
    source: str = "unknown"
    backend: Optional[str] = None
    method: Optional[str] = None
    basis: Optional[str] = None
    geometry_hash: Optional[str] = None
    status: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ vectors

    def alpha_vector(self) -> np.ndarray:
        """Return alpha_A, alpha_B, alpha_C as a length-3 array (NaN if missing)."""
        return np.array(
            [
                _component_value(self.alpha_A),
                _component_value(self.alpha_B),
                _component_value(self.alpha_C),
            ],
            dtype=float,
        )

    def delta_vib_vector(self) -> np.ndarray:
        """Return the vibrational delta for A/B/C.

        Falls back to ``0.5 * alpha`` when ``delta_vib_X`` is unset and the
        corresponding alpha value is finite; otherwise the entry is NaN.
        """
        alpha = self.alpha_vector()
        deltas = (self.delta_vib_A, self.delta_vib_B, self.delta_vib_C)
        out = np.full(3, np.nan, dtype=float)
        for i, (d, a) in enumerate(zip(deltas, alpha)):
            if d is not None:
                v = _component_value(d)
                if np.isfinite(v):
                    out[i] = v
                    continue
            if np.isfinite(a):
                out[i] = 0.5 * float(a)
        return out

    def delta_elec_vector(self) -> np.ndarray:
        """Return the electronic delta for A/B/C (defaults to zero)."""
        return np.array(
            [
                float(self.delta_elec_A or 0.0),
                float(self.delta_elec_B or 0.0),
                float(self.delta_elec_C or 0.0),
            ],
            dtype=float,
        )

    def delta_bob_vector(self) -> np.ndarray:
        """Return the Born-Oppenheimer-breakdown delta for A/B/C (defaults to zero)."""
        return np.array(
            [
                float(self.delta_bob_A or 0.0),
                float(self.delta_bob_B or 0.0),
                float(self.delta_bob_C or 0.0),
            ],
            dtype=float,
        )

    def delta_total_vector(self) -> np.ndarray:
        """Sum of vibrational, electronic, and BOB deltas.

        NaN entries from the vibrational delta propagate.
        """
        vib = self.delta_vib_vector()
        elec = self.delta_elec_vector()
        bob = self.delta_bob_vector()
        # Treat NaN in vib as missing; preserve NaN in result so callers can
        # detect that a component has no usable correction.
        total = vib + elec + bob
        return total

    def sigma_delta_vector(self) -> np.ndarray:
        """Return the uncertainty on the total delta for A/B/C (NaN if unknown)."""
        return np.array(
            [
                _component_value(self.sigma_delta_A),
                _component_value(self.sigma_delta_B),
                _component_value(self.sigma_delta_C),
            ],
            dtype=float,
        )


@dataclass
class ParsedRovibResult:
    """Outcome of parsing an ORCA VPT2 (or equivalent) output for alpha constants."""

    alpha_abc: np.ndarray
    frequencies: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    parse_status: str = "unknown"
    units: str = "MHz"
