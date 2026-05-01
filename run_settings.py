from copy import deepcopy


BASE_SETTINGS = {
    "orca_exe": r"C:\ORCA_6.1.1\orca.exe",
    "quantum_backend": "orca",
    "use_quantum_prior": True,
    "write_xyz": False,
    "default_preset": "BALANCED",
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

VALID_MOLECULES = {"so2", "ocs", "co2", "water", "methanol"}


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
    return out
