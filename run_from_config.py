#!/usr/bin/env python3
"""Entry-point shim — delegates to runner/run_from_config.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner.run_from_config import main

if __name__ == "__main__":
    main()
