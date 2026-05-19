"""
Canonical map of CLI molecule names to their runner module paths.

Both run_molecule.py and runner/run_from_config.py import from here so the
registry stays in one place.
"""

RUNNER_MODULES: dict[str, str] = {
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
