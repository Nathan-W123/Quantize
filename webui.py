#!/usr/bin/env python3
"""Browser front end for Quantize.

    python3 webui.py                 # http://localhost:8018
    python3 webui.py --port 9000
    python3 webui.py --host 0.0.0.0  # reachable from the local network

Build a case from dropdowns, validate it, run it, and read the fitted
structure back. Needs nothing beyond the package's own dependencies -- no
Flask, no Qt, no display server -- so it works over SSH and in containers.

It does not replace the config-file route. The form builds the same config a
YAML file loads to and runs it through the same validator and runner, and
"Download YAML" hands back a file that runs unchanged under:

    python -m cli run case.yaml
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web.server import serve  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Quantize - local web UI")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (0.0.0.0 exposes it to your network)")
    p.add_argument("--port", type=int, default=8018, help="port to listen on")
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = p.parse_args()

    if not args.no_open and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
