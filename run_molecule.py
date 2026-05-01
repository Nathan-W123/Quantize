import argparse
import importlib
from multiprocessing import freeze_support

RUNNER_MODULES = {
    "so2": "runs.run_SO2",
    "ocs": "runs.run_OCS",
    "co2": "runs.run_CO2",
    "water": "runs.run_water",
    "methanol": "runs.run_n",
    "run_n": "runs.run_n",
}

# Edit these defaults, then run this script directly.
MOLECULE = "methanol"  # so2 | ocs | co2 | water | methanol
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
