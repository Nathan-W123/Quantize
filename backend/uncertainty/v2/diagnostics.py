from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunDiagnostics:
    """Common diagnostics container for uncertainty workflows."""

    warnings: list[str]
    notes: list[str]

