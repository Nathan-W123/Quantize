"""
Naphthalene (C₁₀H₈) — larger polycyclic benchmark (18 atoms) for rigid Watson A,B,C inversion.

Spectral targets
----------------
Experimental rotational constants (cm⁻¹) from NIST CCCBDB — *Experimental data for C₁₀H₈*
https://cccbdb.nist.gov/exp2x.asp?casno=91203

Reference on that page: Kabir, M. H.; Kasahara, S.; Demtröder, W.; et al.
*J. Chem. Phys.* **2003**, *119*, 3691 (doi:10.1063/1.1590961).

Converted to MHz via 29979.2458 MHz/(cm⁻¹) (same convention as other drivers).

Initial geometry
----------------
Fetched at run time via ``backend.pubchem_geometry.coords_elems_from_pubchem`` (**CID 931**,
same 3-D conformer family as PubChem MMFF-relaxed structures). Offline/air-gapped use requires a
cached SDF/step or swapping in a frozen ``coords`` source.

See: https://pubchem.ncbi.nlm.nih.gov/compound/931

This driver uses **one** fully substituted isotopologue (all ¹²C, all ¹H). Additional rows (e.g.
perdeuterated or ¹³C-site-specific species) can be appended once experimental A,B,C are transcribed
from the same literature families (e.g. Pirali et al., *Phys. Chem. Chem. Phys.* **2013**, *15*, 10141).
"""

import numpy as np
from multiprocessing import freeze_support

from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from backend.pubchem_geometry import coords_elems_from_pubchem
from backend.spectral import sanitize_isotopologues
from runner.run_settings import get_run_settings

m_H = 1.00782503207
m_C12 = 12.00000000000

_CM_TO_MHZ = 29979.2458

# NIST CCCBDB experimental rotational constants (cm⁻¹), Kabir et al. 2003 — see module docstring.
_ABC_CM_KABIR2003 = np.array([0.10405, 0.04113, 0.02948], dtype=float)

elems = ["C"] * 10 + ["H"] * 8


def load_initial_coordinates() -> np.ndarray:
    """PubChem CID 931 3-D record; validates element order matches ``elems``."""
    coords_pub, elems_pub = coords_elems_from_pubchem("931", prefer="cid", timeout=120.0)
    if elems_pub != elems:
        raise ValueError(
            "PubChem atom symbols/order mismatch vs driver `elems` — check CID 931 parsing."
            f"\n   driver: {elems}\n  pubchem: {elems_pub}"
        )
    return np.asarray(coords_pub, dtype=float)


all_masses = {
    "12C10H8": np.array([m_C12] * 10 + [m_H] * 8),
}

obs_b0_values = {
    "12C10H8": _ABC_CM_KABIR2003 * _CM_TO_MHZ,
}

sigma_table = {
    # Conservative — CCCBDB entry does not propagate quoted uncertainties into this driver.
    "12C10H8": np.array([0.5, 0.05, 0.05]),
}

alpha_table = {k: np.zeros(3, dtype=float) for k in obs_b0_values}

_ISO_ORDER = ["12C10H8"]
component_idx = [0, 1, 2]

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
RNG_SEED = 41
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _metrics(arr):
    """Fusion C–C distance (Å) and rms height of carbons from best-fit plane (Å)."""
    c = np.asarray(arr[:10], dtype=float)
    g = c.mean(axis=0)
    x = c - g
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    n = vh[-1]
    n = n / max(np.linalg.norm(n), 1e-15)
    h = np.abs(np.dot(x, n))
    rms_plane = float(np.sqrt(np.mean(h**2)))
    r_fusion = float(np.linalg.norm(c[0] - c[1]))
    return r_fusion, rms_plane


def _print_input_diagnostics(coords):
    labels = ["A", "B", "C"]
    print("\nNaphthalene benchmark — input spectral targets (B₀ + 0.5·α = Be with α=0 here):")
    for iso in isotopologues:
        name = iso.get("name", "iso")
        b0 = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso["alpha_constants"], dtype=float)
        be = b0 + 0.5 * alpha
        comps = [labels[i] for i in np.asarray(iso["component_indices"], dtype=int)]
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
    coords = load_initial_coordinates()
    _print_input_diagnostics(coords)
    settings = get_run_settings("naphthalene", PRESET_OVERRIDE)
    preset = settings["preset_values"]
    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        starts.append(coords + rng.normal(0.0, 0.03, size=coords.shape))

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
        orca_update_thresh=0.01,
        hess_recalc_every=3,
        adaptive_hess_schedule=True,
        hess_recalc_min=1,
        hess_recalc_max=10,
        sv_threshold=1e-4,
        sv_min_abs=0.0,
        trust_radius=float(preset["trust_radius"]),
        null_trust_radius=0.02,
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
        prior_sigma_bond=0.06,
        prior_sigma_angle_deg=4.0,
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
        job_name="naphthalene",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])
    best = select_best_result(results, spectral_gate_abs=0.2, spectral_gate_rel=2.0)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("naphthalene_optimized.xyz", "w") as f:
            f.write(f"{len(elems)}\n")
            f.write("Best naphthalene geometry from multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_fus, rms_p = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_m = all_metrics.mean(axis=0)
    std_m = all_metrics.std(axis=0)

    print("\n" + "=" * 66)
    print("  Naphthalene — inferred geometry (18-atom polycyclic benchmark)")
    print("=" * 66)
    print(f"  {'Parameter':<38}  {'Recovered':>12}")
    print("  " + "-" * 54)
    print(f"  {'fusion C–C distance (PubChem fusion pair) [A]':<38}  {r_fus:>12.6f}")
    print(f"  {'C₁₀ rms deviation from plane [A]':<38}  {rms_p:>12.6f}")
    print("=" * 66)
    print("  Multi-start consensus (mean ± std):")
    print(f"    fusion r(C–C): {mean_m[0]:.6f} ± {std_m[0]:.6f} Å")
    print(f"    plane rms:     {mean_m[1]:.6f} ± {std_m[1]:.6f} Å")
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
    print("=" * 66)


if __name__ == "__main__":
    freeze_support()
    main()
