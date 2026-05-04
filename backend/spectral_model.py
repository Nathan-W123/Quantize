"""
Spectral objective modes for geometry inversion.

The default rigid asymmetric-rotor map (principal moments → A, B, C) matches
laboratory Watson constants well for semi-rigid tops. For molecules with strong
methyl internal rotation (e.g. methanol), catalog "A, B, C" often come from a
torsion–rotation global fit; comparing them to a single-structure rigid rotor
then systematically mismatches, especially for A.

Mode ``internal_rotor_bc`` does **not** embed a full RAM/IAM Hamiltonian (that
would require transition-line fitting and many more parameters). Instead it is a
documented **proxy**: fit only **B and C**, which are typically less polluted by
torsional coupling in this workflow, and treat targets as **B0-style** by using
zero vibrational ``alpha`` on those components unless mode ``rigid`` supplies
your Be-correction vector.

A future extension could diagonalize a reduced torsion–rotation Hamiltonian or
call an external SPFIT/SPCAT workflow when line data exist.
"""

from __future__ import annotations

_VALID_MODES = frozenset({"rigid", "internal_rotor_bc"})
_ALIASES = {
    "internal_rotor": "internal_rotor_bc",
    "bc_only": "internal_rotor_bc",
    "semi_rigid_bc": "internal_rotor_bc",
}


def normalize_spectral_model(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES | set(_ALIASES.keys())))
        raise ValueError(f"Unknown spectral_model '{name}'. Use one of: {valid}")
    return key


def methanol_isotopologue_row(
    *,
    name: str,
    masses,
    obs_abc_mhz,
    sigma_abc_mhz,
    alpha_abc_mhz,
    mode: str,
    torsion_sensitive: bool = True,
) -> dict:
    """
    Build one isotopologue dict for ``runs/run_n`` from full A,B,C tables.

    ``internal_rotor_bc`` keeps full observables in storage order [B,C] with
    ``component_indices`` [1,2] into the principal-axis (A,B,C) vector.
    """
    mode = normalize_spectral_model(mode)
    obs = list(obs_abc_mhz)
    sig = list(sigma_abc_mhz)
    alp = list(alpha_abc_mhz)
    if len(obs) != 3 or len(sig) != 3 or len(alp) != 3:
        raise ValueError("obs_abc_mhz, sigma_abc_mhz, alpha_abc_mhz must have length 3")

    if mode == "rigid":
        return {
            "name": name,
            "masses": list(masses),
            "component_indices": [0, 1, 2],
            "obs_constants": obs,
            "sigma_constants": sig,
            "alpha_constants": alp,
            "torsion_sensitive": torsion_sensitive,
        }

    # internal_rotor_bc: B0 targets on B,C; drop uncertain nu8 alpha on these unless user switches back to rigid.
    return {
        "name": name,
        "masses": list(masses),
        "component_indices": [1, 2],
        "obs_constants": [obs[1], obs[2]],
        "sigma_constants": [sig[1], sig[2]],
        "alpha_constants": [0.0, 0.0],
        "torsion_sensitive": torsion_sensitive,
    }
