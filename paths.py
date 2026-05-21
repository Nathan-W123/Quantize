"""Repo import paths and generated-artifact directories under ``output/``."""

from __future__ import annotations

import sys
from pathlib import Path

# All local generated files live under output/ (gitignored except .gitkeep stubs).
OUTPUT_DIR = Path("output")
OUTPUT_RUNS_DIR = OUTPUT_DIR / "runs"
OUTPUT_RESULTS_DIR = OUTPUT_DIR / "results"
OUTPUT_TRIALS_DIR = OUTPUT_DIR / "trials"
OUTPUT_BENCHMARKS_DIR = OUTPUT_RESULTS_DIR / "benchmarks"
OUTPUT_BENCHMARK_HISTORY_DIR = OUTPUT_BENCHMARKS_DIR / "history"

_DEFAULT_OUTPUT_SUBDIRS = (
    OUTPUT_RUNS_DIR,
    OUTPUT_RESULTS_DIR,
    OUTPUT_TRIALS_DIR,
    OUTPUT_BENCHMARKS_DIR,
    OUTPUT_BENCHMARK_HISTORY_DIR,
)


def ensure_repo_paths(repo_root: Path) -> Path:
    """Insert ``<repo>/.github`` then ``<repo>`` so ``import backend`` resolves."""
    root = repo_root.resolve()
    for entry in (root / ".github", root):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def ensure_output_layout(repo_root: Path | None = None) -> Path:
    """Create ``output/`` subdirectories if missing."""
    root = (repo_root or Path(__file__).resolve().parent).resolve()
    for sub in _DEFAULT_OUTPUT_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root / OUTPUT_DIR
