import argparse
import importlib
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

try:
    import numpy  # noqa: F401 — required by runners; fail fast with a helpful hint
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parent
    _venv_root = (_root / ".venv").resolve()
    _venv_py = (
        _venv_root / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else _venv_root / "bin" / "python"
    )
    # `.venv/bin/python` often symlinks to the same Homebrew binary as sys.executable;
    # comparing paths wrongly skips re-exec. Use sys.prefix: in a venv it equals `.venv`.
    _running_in_project_venv = Path(sys.prefix).resolve() == _venv_root
    if _venv_py.is_file() and not _running_in_project_venv:
        _script = Path(sys.argv[0]).resolve()
        os.execv(str(_venv_py), [str(_venv_py), str(_script), *sys.argv[1:]])
    sys.stderr.write(
        "NumPy is not installed for this Python executable:\n"
        f"  {sys.executable}\n"
        f"  sys.prefix = {sys.prefix}\n\n"
    )
    if _running_in_project_venv:
        _pip = _venv_root / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
        sys.stderr.write(
            "You are using this project's .venv but dependencies are missing. Run:\n"
            f"  {_pip} install -r {_root / 'requirements.txt'}\n"
        )
    else:
        sys.stderr.write(
            "Create the venv and install deps:\n"
            f"  cd {_root}\n"
            "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
        )
    raise SystemExit(1) from None

RUNNER_MODULES = {
    "so2": "molecule_runners.run_SO2",
    "ocs": "molecule_runners.run_OCS",
    "co2": "molecule_runners.run_CO2",
    "water": "molecule_runners.run_water",
    "methanol": "molecule_runners.run_methanol_vt0_staggered",
    "methanol_vt0_staggered": "molecule_runners.run_methanol_vt0_staggered",
    "benzene": "molecule_runners.run_benzene",
    "formaldehyde": "molecule_runners.run_formaldehyde",
    "naphthalene": "molecule_runners.run_naphthalene",
}

# Edit these defaults, then run this script directly.
MOLECULE = "ocs"  # so2 | ocs | co2 | water | methanol
PRESET = "STRICT"          # None | FAST_DEBUG | BALANCED | STRICT


def main():
    parser = argparse.ArgumentParser(description="Run molecule driver by name.")
    parser.add_argument("molecule", nargs="?", default=None, choices=sorted(RUNNER_MODULES.keys()))
    parser.add_argument("--preset", default=None, help="Preset override: FAST_DEBUG|BALANCED|STRICT")
    args = parser.parse_args()

    molecule = args.molecule if args.molecule is not None else MOLECULE
    preset = args.preset if args.preset is not None else PRESET

    if molecule not in RUNNER_MODULES:
        valid = ", ".join(sorted(RUNNER_MODULES.keys()))
        raise ValueError(f"Unknown molecule '{molecule}'. Valid options: {valid}")

    mod = importlib.import_module(RUNNER_MODULES[molecule])
    if hasattr(mod, "PRESET_OVERRIDE"):
        mod.PRESET_OVERRIDE = preset
    mod.main()


if __name__ == "__main__":
    freeze_support()
    main()
