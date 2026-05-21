#!/usr/bin/env python3
"""Entry-point shim — delegates to runner/run_from_config.py."""
from pathlib import Path

from paths import ensure_repo_paths

ensure_repo_paths(Path(__file__).resolve().parent)
from runner.run_from_config import main

if __name__ == "__main__":
    main()
