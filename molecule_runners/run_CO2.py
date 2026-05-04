import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_linear_triatomic
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from runner.run_settings import get_run_settings

# ── CO2 inversion scaffold (fill with real isotopologue data) ─────────────────
#
# Recommended for linear molecules:
# - Start with B-only constraints (component index 1) for stable conditioning.
# - Add more isotopologues to improve constrained rank.
#
# Replace placeholder B0/sigma/alpha values below with literature values.

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_O16 = 15.99491461956
m_O17 = 16.99913175650
m_O18 = 17.99915961286
m_C12 = 12.00000000000
m_C13 = 13.00335483507

# ── Candidate isotopologue masses (expand as needed) ─────────────────────────
all_masses = {
    "16O12C16O": np.array([m_O16, m_C12, m_O16]),
    "16O13C16O": np.array([m_O16, m_C13, m_O16]),
    "16O12C18O": np.array([m_O16, m_C12, m_O18]),
    "18O12C18O": np.array([m_O18, m_C12, m_O18]),
    "16O12C17O": np.array([m_O16, m_C12, m_O17]),
}

# ── Fill these with real data (MHz) ───────────────────────────────────────────
# For linear CO2, B-only is usually the most stable:
component_idx = [1]  # 0=A, 1=B, 2=C

alpha_table = {
    "16O12C16O": np.array([94.42]),
    "18O12C18O": np.array([79.80]),
    "16O12C18O": np.array([87.10]),
    "16O13C16O": np.array([94.40]),
}

sigma_table = {
    "16O12C16O": np.array([0.01]),
    "18O12C18O": np.array([0.02]),
    "16O12C18O": np.array([0.02]),
    "16O13C16O": np.array([0.01]),
}

obs_b0_values = {
    "16O12C16O": np.array([11698.472]),
    "18O12C18O": np.array([10398.052]),
    "16O12C18O": np.array([11048.254]),
    "16O13C16O": np.array([11698.471]),
}

# ── Active isotopologue set for the run ──────────────────────────────────────
active_isotopes = ["16O12C16O", "16O13C16O", "18O12C18O", "16O12C18O"]

isotopologues = []
for name in active_isotopes:
    isotopologues.append(
        {
            "name": name,
            "masses": all_masses[name].tolist(),
            "component_indices": component_idx,
            "obs_constants": obs_b0_values[name].tolist(),
            "sigma_constants": sigma_table[name].tolist(),
            "alpha_constants": alpha_table[name].tolist(),
        }
    )

# ── Initial geometry guess (linear O-C-O) ────────────────────────────────────
elems = ["O", "C", "O"]
coords = guess_linear_triatomic(
    left_elem="O",
    center_elem="C",
    right_elem="O",
    r_left_center=1.16,
    r_center_right=1.16,
    bend_deg=0.0,
)

# ── Backend / optimization controls ──────────────────────────────────────────
USE_QUANTUM_PRIOR = True
RNG_SEED = 29
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _metrics(arr):
    o1 = arr[0]
    c = arr[1]
    o2 = arr[2]
    r1 = float(np.linalg.norm(o1 - c))
    r2 = float(np.linalg.norm(o2 - c))
    v1 = o1 - c
    v2 = o2 - c
    ang = float(np.degrees(np.arccos(
        np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    )))
    return r1, r2, ang


def _print_input_diagnostics():
    labels = ["A", "B", "C"]
    print("\nInput spectral targets (B0 + 0.5*alpha = Be):")
    for iso in isotopologues:
        name = iso.get("name", "iso")
        idx = np.asarray(iso["component_indices"], dtype=int)
        b0 = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso["alpha_constants"], dtype=float)
        be = b0 + 0.5 * alpha
        comps = [labels[i] if 0 <= i < 3 else f"R{i}" for i in idx]
        print(f"  {name} components: {comps}")
        print(f"    B0    = {np.round(b0, 6).tolist()}")
        print(f"    alpha = {np.round(alpha, 6).tolist()}")
        print(f"    Be    = {np.round(be, 6).tolist()}")


def main():
    _print_input_diagnostics()
    settings = get_run_settings("co2", PRESET_OVERRIDE)
    preset = settings["preset_values"]

    # Safety warning if placeholders were not replaced.
    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        jitter = np.zeros_like(coords)
        jitter[0] = rng.normal(0.0, [0.04, 0.02, 0.02])
        jitter[2] = rng.normal(0.0, [0.04, 0.02, 0.02])
        starts.append(coords + jitter)

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
        sv_threshold=1.3980595102624797e-05,
        sv_min_abs=0.0,
        trust_radius=float(preset["trust_radius"]),
        null_trust_radius=None,
        lambda_damp=0.00016370045068111915,
        objective_mode="split",
        alpha_quantum=0.2778639378704326,
        robust_loss="none",
        robust_param=1.0,
        torsion_aware_weighting=False,
        torsion_a_weight=1.0,
        spectral_delta=0.00034930106014707015,
        auto_sanitize_spectral=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        use_internal_preconditioner=False,
        sigma_floor_mhz=float(preset["sigma_floor_mhz"]),
        max_spectral_weight=float(preset["max_spectral_weight"]),
        enable_geometry_guardrails=True,
        enforce_quantum_descent=True,
        use_internal_priors=bool(preset["use_internal_priors"]),
        prior_weight=1.0,
        prior_auto_from_initial=True,
        prior_use_dihedrals=False,
        prior_sigma_bond=0.03,
        prior_sigma_angle_deg=1.5,
        use_conformer_mixture=bool(preset["use_conformer_mixture"]),
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        method_preset="fast",
        orca_method=settings["orca_method"],
        orca_basis=settings["orca_basis"],
        use_orca_rovib=False,
        rovib_recalc_every=1,
        rovib_source_mode="hybrid_auto",
        symmetry="Dinf_h",
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        base_workdir=".",
    )

    print(f"Using preset: {settings['selected_preset']}")
    print(f"Running {int(preset['n_starts'])} starts with max_workers={int(preset['max_workers'])} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
        job_name="co2",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])

    best = select_best_result(results, spectral_gate_abs=0.01, spectral_gate_rel=2.0)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("co2_optimized.xyz", "w") as f:
            f.write("3\n")
            f.write("Best CO2 geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_co1, r_co2, ang_oco = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_r1, mean_r2, mean_ang = all_metrics.mean(axis=0)
    std_r1, std_r2, std_ang = all_metrics.std(axis=0)

    print("\n" + "=" * 60)
    print("  CO2 inferred geometry from isotopologue data")
    print("=" * 60)
    print(f"  {'Parameter':<16}  {'Recovered':>12}")
    print("  " + "-" * 32)
    print(f"  {'r(C-O1) [A]':<16}  {r_co1:>12.6f}")
    print(f"  {'r(C-O2) [A]':<16}  {r_co2:>12.6f}")
    print(f"  {'angle [deg]':<16}  {ang_oco:>12.4f}")
    print("=" * 60)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C-O1): {mean_r1:.6f} ± {std_r1:.6f} Å")
    print(f"    r(C-O2): {mean_r2:.6f} ± {std_r2:.6f} Å")
    print(f"    angle  : {mean_ang:.4f} ± {std_ang:.4f} deg")
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
    if score.get("conformer_weights_final") is not None:
        print(f"    Conformer weights (final): {np.round(score['conformer_weights_final'], 4).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    main()
