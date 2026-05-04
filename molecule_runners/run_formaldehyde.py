"""
Formaldehyde (H₂CO) — planar asymmetric rotor benchmark for rigid Watson A,B,C inversion.

Experimental targets are laboratory B₀-style constants (MHz); the spectral engine forms
Be ≈ B₀ + ½α per component (here α = 0 unless you extend the tables).

Data provenance (see comments below): NIST CCCBDB + CDMS Cologne.
"""

import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_planar_formaldehyde
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from backend.spectral import sanitize_isotopologues
from runner.run_settings import get_run_settings

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_H = 1.00782503207
m_D = 2.01410177785
m_C12 = 12.0
m_O16 = 15.99491461957

# Conversion for Herzberg-style rotational constants tabulated in cm⁻¹ (NIST CCCBDB).
_CM_TO_MHZ = 29979.2458

# Atom order for every row: O, C, H_a, H_b (planar C₂ᵥ frame; HDCO substitutes one H site).
elems = ["O", "C", "H", "H"]

all_masses = {
    # H212C16O — NIST CCCBDB experimental rotational constants (cm⁻¹) × CM_TO_MHZ,
    #   https://cccbdb.nist.gov/exp2x.asp?casno=50000  → A,B,C from 1966Herzberg (see CCCBDB page).
    "H212C16O": np.array([m_O16, m_C12, m_H, m_H]),
    # HDCO — CDMS species 031501, https://cdms.astro.uni-koeln.de/cgi-bin/cdmsinfo?file=e031501.cat
    #   Bocquet et al., J. Mol. Spectrosc. 1999; THz data Zakharenko et al., JMS 2015.
    "HDCO": np.array([m_O16, m_C12, m_H, m_D]),
}

obs_b0_values = {
    "H212C16O": np.array([9.40530, 1.29530, 1.13420], dtype=float) * _CM_TO_MHZ,
    "HDCO": np.array([198118.36, 34909.107, 29562.871]),
}

sigma_table = {
    "H212C16O": np.array([2.0, 0.05, 0.05]),
    "HDCO": np.array([2.0, 0.05, 0.05]),
}

alpha_table = {k: np.zeros(3, dtype=float) for k in obs_b0_values}

_ISO_ORDER = ["H212C16O", "HDCO"]
component_idx = [0, 1, 2]

coords = guess_planar_formaldehyde()

isotopologues = [
    {
        "name": name,
        "masses": all_masses[name].tolist(),
        "component_indices": component_idx,
        "obs_constants": obs_b0_values[name][component_idx].tolist(),
        "sigma_constants": sigma_table[name][component_idx].tolist(),
        "alpha_constants": alpha_table[name][component_idx].tolist(),
        "torsion_sensitive": False,
    }
    for name in _ISO_ORDER
]

USE_QUANTUM_PRIOR = True
RNG_SEED = 19
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _angle_deg(vertex, arm_a, arm_b):
    """Angle arm_a — vertex — arm_b in degrees (atoms as 3D positions)."""
    v1 = arm_a - vertex
    v2 = arm_b - vertex
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _metrics(arr):
    o, c, h1, h2 = arr[0], arr[1], arr[2], arr[3]
    r_co = float(np.linalg.norm(o - c))
    r_ch1 = float(np.linalg.norm(h1 - c))
    r_ch2 = float(np.linalg.norm(h2 - c))
    v1 = h1 - c
    v2 = h2 - c
    hch = float(
        np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    )
    # ∠(O–C–H) at carbon (report both hydrogens; equal in C₂ᵥ isotopologues).
    hco_1 = _angle_deg(c, o, h1)
    hco_2 = _angle_deg(c, o, h2)
    hco_avg = 0.5 * (hco_1 + hco_2)
    return r_co, (r_ch1 + r_ch2) * 0.5, hch, hco_avg, hco_1, hco_2


def _print_input_diagnostics():
    labels = ["A", "B", "C"]
    print("\nFormaldehyde benchmark — input spectral targets (B₀ + 0.5·α = Be with α=0 here):")
    for iso in isotopologues:
        name = iso.get("name", "iso")
        idx = np.asarray(iso["component_indices"], dtype=int)
        b0 = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso["alpha_constants"], dtype=float)
        be = b0 + 0.5 * alpha
        comps = [labels[i] if 0 <= i < 3 else f"R{i}" for i in idx]
        print(f"  {name} components {comps}: B0 MHz ≈ {np.round(be, 4).tolist()}")

    cleaned, notes = sanitize_isotopologues(isotopologues, coords, delta=1e-4)
    print("\nSanitizer pre-check at initial geometry:")
    for iso_in, iso_out in zip(isotopologues, cleaned):
        name = iso_in.get("name", "iso")
        kept = [labels[i] if 0 <= int(i) < 3 else f"R{int(i)}" for i in np.asarray(iso_out["component_indices"], dtype=int)]
        print(f"  {name}: kept {kept}")
    if notes:
        for n in notes:
            print(f"    note: {n}")
    else:
        print("    note: no components dropped at initial geometry")


def main():
    _print_input_diagnostics()
    settings = get_run_settings("formaldehyde", PRESET_OVERRIDE)
    preset = settings["preset_values"]
    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        starts.append(coords + rng.normal(0.0, 0.04, size=coords.shape))

    optimizer_kwargs = dict(
        quantum_backend=settings["quantum_backend"],
        orca_executable=settings["orca_exe"],
        spectral_only=not USE_QUANTUM_PRIOR,
        max_iter=500,
        conv_step=1e-7,
        conv_freq=float(preset["conv_freq"]),
        spectral_accept_relax=float(preset.get("spectral_accept_relax", 0.0)),
        conv_energy=1e-8,
        conv_step_range=1e-6,
        conv_step_null=1e-5,
        conv_grad_null=1e-4,
        orca_update_thresh=0.005,
        hess_recalc_every=2,
        adaptive_hess_schedule=True,
        hess_recalc_min=1,
        hess_recalc_max=8,
        sv_threshold=1e-4,
        sv_min_abs=0.0,
        trust_radius=float(preset["trust_radius"]),
        null_trust_radius=0.01,
        lambda_damp=1e-3,
        objective_mode="split",
        alpha_quantum=0.45,
        spectral_delta=2e-4,
        robust_loss="none",
        robust_param=1.0,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        auto_sanitize_spectral=True,
        sigma_floor_mhz=float(preset["sigma_floor_mhz"]),
        max_spectral_weight=float(preset["max_spectral_weight"]),
        enable_geometry_guardrails=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        use_internal_preconditioner=False,
        project_rigid_modes=False,
        enforce_quantum_descent=bool(preset.get("enforce_quantum_descent", False)),
        quantum_descent_tol=float(preset.get("quantum_descent_tol", 1e-5)),
        use_internal_priors=bool(preset["use_internal_priors"]),
        prior_weight=1.0,
        prior_auto_from_initial=True,
        prior_use_dihedrals=False,
        prior_sigma_bond=0.04,
        prior_sigma_angle_deg=2.0,
        use_conformer_mixture=bool(preset["use_conformer_mixture"]),
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        method_preset=None,
        orca_method=settings["orca_method"],
        orca_basis=settings["orca_basis"],
        use_orca_rovib=False,
        rovib_recalc_every=5,
        rovib_source_mode="hybrid_auto",
        symmetry=None,
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        base_workdir=".",
    )

    print(f"Using preset: {settings['selected_preset']}  |  ORCA: {settings['orca_method']}/{settings['orca_basis']}")
    print(f"Running {int(preset['n_starts'])} starts with max_workers={int(preset['max_workers'])} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
        job_name="formaldehyde",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])
    best = select_best_result(results, spectral_gate_abs=0.05, spectral_gate_rel=1.5)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("formaldehyde_optimized.xyz", "w") as f:
            f.write(f"{len(elems)}\n")
            f.write("Best formaldehyde geometry from multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_co, r_ch_avg, hch, hco_avg, hco_1, hco_2 = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_m = all_metrics.mean(axis=0)
    std_m = all_metrics.std(axis=0)

    print("\n" + "=" * 62)
    print("  Formaldehyde — inferred geometry (planar asymmetric-rotor fit)")
    print("=" * 62)
    print(f"  {'Parameter':<22}  {'Recovered':>12}")
    print("  " + "-" * 38)
    print(f"  {'r(C=O) [A]':<22}  {r_co:>12.6f}")
    print(f"  {'r(C-H)_avg [A]':<22}  {r_ch_avg:>12.6f}")
    print(f"  {'angle H-C-H [deg]':<22}  {hch:>12.4f}")
    print(f"  {'angle H-C-O_avg [deg]':<22}  {hco_avg:>12.4f}")
    print(f"  {'angle H-C-O (site 1,2)':<22}  {hco_1:>6.4f}, {hco_2:>6.4f}")
    print("=" * 62)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C=O):        {mean_m[0]:.6f} ± {std_m[0]:.6f} Å")
    print(f"    r(C-H)_avg:    {mean_m[1]:.6f} ± {std_m[1]:.6f} Å")
    print(f"    angle HCH:     {mean_m[2]:.4f} ± {std_m[2]:.4f} deg")
    print(f"    angle HCO_avg: {mean_m[3]:.4f} ± {std_m[3]:.4f} deg")
    print(f"    angle HCO (H1): {mean_m[4]:.4f} ± {std_m[4]:.4f} deg")
    print(f"    angle HCO (H2): {mean_m[5]:.4f} ± {std_m[5]:.4f} deg")
    print(f"  Best run spectral RMS MHz: {best['freq_rms']:.6f}")
    print(f"  Best run energy (Eh):      {best['energy']:.10f}")
    score = underconstrained_success_score(results, best, isotopologues)
    std_vals = np.asarray(score["metric_std"], dtype=float)
    print("  Underconstrained success score (rank-aware):")
    print(f"    Score (0-100): {score['score']:.1f}")
    print(
        f"    Constrained rank: {score['constrained_rank']}/{score['internal_dof']} "
        f"({100.0 * score['rank_fraction']:.1f}%)"
    )
    if std_vals.size > 0:
        std_line = ", ".join(f"m{i+1}_std={v:.6f}" for i, v in enumerate(std_vals))
        print(f"    Multi-start metric std: {std_line}")
    print(
        f"    Spectral vs uncertainty: RMS/sigma ~= {score['sigma_ratio']:.1f}x "
        f"(sigma scale {score['sigma_scale']:.4f} MHz)"
    )
    print("=" * 62)


if __name__ == "__main__":
    freeze_support()
    main()
