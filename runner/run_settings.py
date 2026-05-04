import os
from copy import deepcopy

from backend.spectral_model import normalize_spectral_model


def _default_orca_executable():
    """
    Full path to the ORCA binary, or None to let MolecularOptimizer search PATH for ``orca``.

    Override without editing code: ``export ORCA_EXE=/path/to/orca`` (or ``QUANTIZE_ORCA_EXE``).
    """
    return os.environ.get("QUANTIZE_ORCA_EXE") or os.environ.get("ORCA_EXE") or None


# Default ORCA SCF/DFT method and basis for each molecule driver (runs/run_*.py).
# Change these to retarget expensive quantum-chemistry steps. Optional global overrides
# in BASE_SETTINGS (`orca_method`, `orca_basis`) replace the corresponding entry here for
# every molecule when set to a non-None string.
MOLECULE_ORCA_DEFAULTS = {
    "methanol": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "water": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "so2": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "ocs": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "co2": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "benzene": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "formaldehyde": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
    "naphthalene": {"orca_method": "wB97X-D4", "orca_basis": "def2-TZVPP"},
}

# Spectral objective: ``rigid`` = fit Watson A,B,C from principal moments; ``internal_rotor_bc``
# = methanol may fit B,C only (proxy — see backend/spectral_model.py). Methanol default is ``rigid``.
MOLECULE_SPECTRAL_MODEL_DEFAULTS = {
    "methanol": "rigid",
    "water": "rigid",
    "so2": "rigid",
    "ocs": "rigid",
    "co2": "rigid",
    "benzene": "rigid",
    "formaldehyde": "rigid",
    "naphthalene": "rigid",
}


BASE_SETTINGS = {
    # None → resolve ``orca`` on PATH (macOS/Linux). Windows users may set ORCA_EXE or paste a path here.
    # Per-run ORCA overrides without editing this file: set ``QUANTIZE_ORCA_METHOD`` /
    # ``QUANTIZE_ORCA_BASIS`` (see ``run_from_config.py``), or ``ORCA_METHOD`` / ``ORCA_BASIS``.
    "orca_exe": _default_orca_executable(),
    "quantum_backend": "orca",
    "use_quantum_prior": True,
    "write_xyz": False,
    "default_preset": "BALANCED",
    # Global ORCA overrides: None → use MOLECULE_ORCA_DEFAULTS for the selected molecule.
    "orca_method": None,
    "orca_basis": None,
    # None → use MOLECULE_SPECTRAL_MODEL_DEFAULTS for the selected molecule.
    "spectral_model": None,
}


GLOBAL_PRESETS = {
    "FAST_DEBUG": dict(
        n_starts=1,
        max_workers=1,
        conv_freq=5000.0,
        spectral_accept_relax=0.08,
        trust_radius=0.006,
        sigma_floor_mhz=0.05,
        max_spectral_weight=50.0,
        use_internal_priors=False,
        use_conformer_mixture=False,
        enforce_quantum_descent=False,
        quantum_descent_tol=1e-4,
        enable_geometry_guardrails=True,
    ),
    "BALANCED": dict(
        n_starts=3,
        max_workers=2,
        conv_freq=1000.0,
        spectral_accept_relax=0.03,
        trust_radius=0.005,
        sigma_floor_mhz=0.02,
        max_spectral_weight=100.0,
        use_internal_priors=True,
        use_conformer_mixture=True,
        enforce_quantum_descent=False,
        quantum_descent_tol=1e-5,
        enable_geometry_guardrails=True,
    ),
    "STRICT": dict(
        n_starts=5,
        max_workers=3,
        conv_freq=500.0,
        spectral_accept_relax=0.0,
        trust_radius=0.003,
        sigma_floor_mhz=0.01,
        max_spectral_weight=250.0,
        use_internal_priors=True,
        use_conformer_mixture=True,
        enforce_quantum_descent=True,
        quantum_descent_tol=1e-10,
        enable_geometry_guardrails=True,
    ),
}

VALID_MOLECULES = {"so2", "ocs", "co2", "water", "methanol", "benzene", "formaldehyde", "naphthalene"}


def get_run_settings(molecule_name, preset_override=None):
    key = str(molecule_name).strip().lower()
    if key not in VALID_MOLECULES:
        raise ValueError(f"Unknown molecule key '{molecule_name}'.")
    out = deepcopy(BASE_SETTINGS)
    selected = preset_override if preset_override is not None else BASE_SETTINGS["default_preset"]
    selected = str(selected).strip().upper()
    if selected not in GLOBAL_PRESETS:
        valid = ", ".join(sorted(GLOBAL_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{selected}'. Valid presets: {valid}")
    out["molecule"] = key
    out["presets"] = deepcopy(GLOBAL_PRESETS)
    out["selected_preset"] = selected
    out["preset_values"] = out["presets"][selected]

    qc = deepcopy(MOLECULE_ORCA_DEFAULTS[key])
    om_override = out.get("orca_method")
    basis_override = out.get("orca_basis")
    if om_override is not None:
        qc["orca_method"] = str(om_override).strip()
    if basis_override is not None:
        qc["orca_basis"] = str(basis_override).strip()
    # Highest priority: one-shot overrides (e.g. ``run_from_config.py`` / CI).
    env_m = os.environ.get("QUANTIZE_ORCA_METHOD") or os.environ.get("ORCA_METHOD")
    env_b = os.environ.get("QUANTIZE_ORCA_BASIS") or os.environ.get("ORCA_BASIS")
    if env_m:
        qc["orca_method"] = str(env_m).strip()
    if env_b:
        qc["orca_basis"] = str(env_b).strip()
    out["orca_method"] = qc["orca_method"]
    out["orca_basis"] = qc["orca_basis"]

    sm = out.get("spectral_model")
    if sm is None:
        sm = MOLECULE_SPECTRAL_MODEL_DEFAULTS[key]
    out["spectral_model"] = normalize_spectral_model(sm)

    # Re-read so per-run env changes (e.g. ``run_from_config.py``) are visible without reloading module.
    exe_env = os.environ.get("QUANTIZE_ORCA_EXE") or os.environ.get("ORCA_EXE")
    if exe_env:
        out["orca_exe"] = str(exe_env).strip()

    return out
