from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runner.run_from_config import main as run_from_config_main
from runner.usability import ConfigError, load_config, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantize", description="Quantize command-line interface.")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run Quantize from a YAML or JSON config.")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--no-run-dir", action="store_true")

    check_p = sub.add_parser("validate", help="Validate a YAML or JSON config without running.")
    check_p.add_argument("config", type=Path)

    args = parser.parse_args(argv)
    if args.command == "validate":
        try:
            cfg = load_config(args.config)
            validate_config(cfg)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        print(f"Config OK: {args.config}")
        return 0

    if args.command == "run":
        sys.argv = [sys.argv[0], str(args.config)] + (["--no-run-dir"] if args.no_run_dir else [])
        run_from_config_main()
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
