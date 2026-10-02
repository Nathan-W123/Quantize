"""Local HTTP server for the Quantize web UI.

Standard library only -- http.server, json, threading -- so it runs anywhere
the package itself runs: over SSH, inside a container, on a machine with no
display and no Flask. Aero's web UI takes the same approach and for the same
reason.

Every route that produces or consumes a case goes through
``web.config_io``, and ``/api/run`` goes through the same
validate -> prepare_run_directory -> run_generic.main sequence that
``runner/run_from_config.py`` uses for a YAML file on disk. The browser is a
config editor with a run button, not a second pipeline.

Bind address defaults to localhost. The server executes quantum chemistry and
writes into the output tree on request, with no authentication, so exposing it
with --host 0.0.0.0 hands that to anyone who can reach the port.
"""

from __future__ import annotations

import io
import json
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from paths import ensure_repo_paths

_ROOT = ensure_repo_paths(Path(__file__).resolve().parent.parent)
_INDEX = Path(__file__).resolve().parent / "index.html"

#: Registering the quantum backends has to happen before anything offers them
#: in a dropdown or validates a name against the registry, or the two disagree
#: depending on which module imported first. Cost of being wrong: the UI lists
#: pyscf_hf and the validator then rejects it.
try:
    import dev.pyscf_backend  # noqa: F401
except Exception:  # pragma: no cover - a pyscf-less install is still usable
    pass


class _Run:
    """One job: its captured output, its state, and its result."""

    def __init__(self, run_id: str, config: dict[str, Any]) -> None:
        self.id = run_id
        self.config = config
        self.buffer = io.StringIO()
        self.lock = threading.Lock()
        self.done = False
        self.error: str | None = None
        self.run_dir: str | None = None

    def text(self) -> str:
        with self.lock:
            return self.buffer.getvalue()


class _Tee(io.TextIOBase):
    """Collect written text under the run's lock so the poller can read it."""

    def __init__(self, run: _Run) -> None:
        self._run = run

    def write(self, s):  # noqa: D102
        with self._run.lock:
            self._run.buffer.write(s)
        return len(s)

    def flush(self):  # noqa: D102
        return None


_RUNS: dict[str, _Run] = {}
_RUNS_LOCK = threading.Lock()


def _execute(run: _Run) -> None:
    """Run one job exactly the way the CLI runs a config file."""
    from runner.run_generic import main as generic_main
    from runner.usability import prepare_run_directory, validate_config

    stream = _Tee(run)
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            validate_config(run.config)
            run_dir = prepare_run_directory(run.config, None)
            run.run_dir = str(run_dir)
            # prepare_run_directory copies the source file when a run came from
            # one, and a browser case has no source file -- so it would leave
            # the run directory without the one artifact that makes the run
            # repeatable. Write it from the dict instead: input.yaml here is a
            # file "python -m cli run" accepts unchanged, so a case built by
            # clicking can be re-run, diffed and committed like any other.
            from web.config_io import config_to_yaml
            (run_dir / "input.yaml").write_text(
                config_to_yaml(run.config), encoding="utf-8")
            print(f"[web] run_dir={run_dir}")
            print(f"[web] rerun with: python -m cli run {run_dir / 'input.yaml'}")
            generic_main(run.config)
    except BaseException as exc:  # noqa: BLE001 - surfaced to the browser
        run.error = f"{type(exc).__name__}: {exc}"
        with run.lock:
            run.buffer.write("\n" + traceback.format_exc())
    finally:
        run.done = True


class Handler(BaseHTTPRequestHandler):
    server_version = "Quantize"

    # Keep the console readable: one line per run, not per asset fetch.
    def log_message(self, fmt, *args):  # noqa: D102
        return

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ── routes ───────────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802, D102
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                body = _INDEX.read_bytes()
            except FileNotFoundError:
                self._json({"error": f"index.html missing at {_INDEX}"}, 500)
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/options":
            self._options()
            return
        if path == "/api/progress":
            self._progress()
            return
        self._json({"error": f"no route {path}"}, 404)

    def do_POST(self):  # noqa: N802, D102
        path = self.path.split("?", 1)[0]
        routes = {
            "/api/validate": self._validate,
            "/api/config": self._config,
            "/api/load": self._load,
            "/api/run": self._run,
            "/api/structure": self._structure,
            "/api/results": self._results,
        }
        fn = routes.get(path)
        if fn is None:
            self._json({"error": f"no route {path}"}, 404)
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    # ── handlers ─────────────────────────────────────────────────────────────

    def _options(self) -> None:
        from runner.usability import valid_backends

        from web.options import all_options

        opts = all_options()
        # What the validator will accept, which is the registry plus "none".
        # Offering anything else would build a config that cannot be run.
        opts["quantum_backend"] = sorted(valid_backends())
        self._json(opts)

    def _validate(self) -> None:
        from runner.usability import ConfigError, validate_config

        from web.config_io import form_to_config

        try:
            cfg = form_to_config(self._body().get("form") or {})
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)})
            return
        try:
            validate_config(cfg)
        except ConfigError as exc:
            self._json({"ok": False, "error": str(exc), "config": cfg})
            return
        self._json({"ok": True, "config": cfg})

    def _config(self) -> None:
        """The YAML this form represents, for download and for the CLI."""
        from web.config_io import config_to_yaml, form_to_config

        cfg = form_to_config(self._body().get("form") or {})
        self._json({"yaml": config_to_yaml(cfg), "config": cfg})

    def _load(self) -> None:
        """Pre-fill the form from a config file someone wrote by hand."""
        from web.config_io import config_to_form

        text = str(self._body().get("text") or "")
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(text)
        except ModuleNotFoundError:
            cfg = json.loads(text)
        if not isinstance(cfg, dict):
            self._json({"ok": False, "error": "config must be a mapping"})
            return
        self._json({"ok": True, "form": config_to_form(cfg)})

    def _run(self) -> None:
        from runner.usability import ConfigError, validate_config

        from web.config_io import form_to_config

        cfg = form_to_config(self._body().get("form") or {})
        try:
            validate_config(cfg)
        except ConfigError as exc:
            self._json({"ok": False, "error": str(exc)})
            return

        # Stop here rather than inside the first Hessian. validate_config only
        # checks that the backend NAME is registered, and registration happens
        # on import while the dependency is imported lazily later -- so a
        # machine without pyscf validates fine, prints a correction table,
        # starts optimising and only then raises ModuleNotFoundError. The run
        # is lost either way; failing now says why and what to do about it.
        from web.options import backend_availability

        name = str((cfg.get("quantum") or {}).get("backend") or "")
        info = backend_availability().get(name, {})
        if info.get("ok") is False:
            hint = info.get("hint") or ""
            self._json({"ok": False, "error": (
                f"backend '{name}' cannot run here: {info.get('why')}."
                + (f" {hint}" if hint else ""))})
            return

        run_id = uuid.uuid4().hex[:12]
        run = _Run(run_id, cfg)
        with _RUNS_LOCK:
            _RUNS[run_id] = run
        threading.Thread(target=_execute, args=(run,), daemon=True).start()
        print(f"[web] started run {run_id}: {cfg.get('name')}")
        self._json({"ok": True, "id": run_id})

    def _structure(self) -> None:
        """The starting structure, and the fitted one when a run has produced it.

        Resolving the geometry can reach the network -- a SMILES string or a
        PubChem name is fetched -- so this is a POST the browser asks for, not
        something computed on every keystroke.
        """
        from web.config_io import form_to_config
        from web.structure import before_and_after

        body = self._body()
        cfg = form_to_config(body.get("form") or {})
        run_dir = None
        run_id = str(body.get("run_id") or "")
        if run_id:
            with _RUNS_LOCK:
                run = _RUNS.get(run_id)
            run_dir = run.run_dir if run else None
        try:
            self._json({"ok": True, **before_and_after(cfg, run_dir)})
        except Exception as exc:  # noqa: BLE001 - a bad SMILES lands here
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _results(self) -> None:
        """Corrections, residuals and uncertainties for a finished run."""
        from web.structure import run_results

        run_id = str(self._body().get("run_id") or "")
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if run is None or not run.run_dir:
            self._json({"ok": False, "error": "no run directory"})
            return
        self._json({"ok": True, **run_results(run.run_dir)})

    def _progress(self) -> None:
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(self.path).query)
        run_id = (q.get("id") or [""])[0]
        since = int((q.get("since") or ["0"])[0])
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if run is None:
            self._json({"error": f"unknown run {run_id}"}, 404)
            return
        text = run.text()
        self._json({
            "text": text[since:],
            "length": len(text),
            "done": run.done,
            "error": run.error,
            "run_dir": run.run_dir,
        })


def serve(host: str = "127.0.0.1", port: int = 8018) -> None:
    """Serve until interrupted."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    where = "localhost" if host in ("127.0.0.1", "localhost") else host
    print(f"Quantize web UI on http://{where}:{port}")
    if host not in ("127.0.0.1", "localhost"):
        print("  WARNING: bound beyond localhost. This server runs quantum "
              "chemistry and writes files on request, with no authentication.")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
