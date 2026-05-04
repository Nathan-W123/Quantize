import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_bent_triatomic
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from runner.run_settings import get_run_settings

# ── Nonlinear triatomic example: SO2 (experimental inversion benchmark) ───────
# This script uses experimental isotopologue rotational constants (B0) and
# mode-resolved alpha corrections to recover geometry.
#
# Atom order: S, O1, O2

# Initial guess intentionally offset from reference.
coords = guess_bent_triatomic(
    central_elem="S",
    terminal1_elem="O",
    terminal2_elem="O",
    r1=1.50,
    r2=1.47,
    angle_deg=111.0,
)

elems = ["S", "O", "O"]

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_O16 = 15.99491461956
m_S32 = 31.9720711744
m_S34 = 33.967867004

all_masses = {
    "32S-16O2": np.array([m_S32, m_O16, m_O16]),
    "34S-16O2": np.array([m_S34, m_O16, m_O16]),
}

# Full summed alpha(A/B/C) in MHz.
# Source provided as half-sums (sum(2*alpha_i) values):
#   32S-16O2: [193.50, 35.48, 58.53]
#   34S-16O2: [184.40, 35.50, 56.78]
# This script uses Be = B0 + 0.5 * alpha_constants, so alpha_constants must be
# full alpha totals (2x the half-sum values above).
alpha_table = {
    "32S-16O2": np.array([387.00, 70.96, 117.06], dtype=float),
    "34S-16O2": np.array([368.80, 71.00, 113.56], dtype=float),
}
sigma_table = {
    # Uncertainties from reported constants in parentheses:
    # 32S-16O2: A0 60778.516(16), B0 10318.0747(24), C0 8799.3076(24)
    # 34S-16O2: A0 58544.11(11), B0 10318.08(1),  C0 8767.12(1)
    "32S-16O2": np.array([0.0160, 0.0024, 0.0024]),
    "34S-16O2": np.array([0.1100, 0.0100, 0.0100]),
}

active_isotopes = ["32S-16O2", "34S-16O2"]

# Experimental observed B0 values (MHz).
obs_b0_values = {
    "32S-16O2": np.array([60778.5160, 10318.0747, 8799.3076], dtype=float),
    "34S-16O2": np.array([58544.1100, 10318.0800, 8767.1200], dtype=float),
}

component_idx = [0, 1, 2]
RNG_SEED = 41
rng = np.random.default_rng(RNG_SEED)

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

WRITE_XYZ = False
PRESET_OVERRIDE = None


def _metrics(arr):
    s = arr[0]
    o1 = arr[1]
    o2 = arr[2]
    r1 = float(np.linalg.norm(o1 - s))
    r2 = float(np.linalg.norm(o2 - s))
    v1 = o1 - s
    v2 = o2 - s
    ang = float(np.degrees(np.arccos(
        np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    )))
    return r1, r2, ang


def _print_inputs():
    labels = ["A", "B", "C"]
    print("SO2 experimental target model:")
    print("  B0 = measured rotational constants (MHz)")
    print("  Be = B0 + 0.5*alpha")
    for iso in isotopologues:
        idx = np.asarray(iso["component_indices"], dtype=int)
        b0 = np.asarray(iso["obs_constants"], dtype=float)
        alpha = np.asarray(iso["alpha_constants"], dtype=float)
        comps = [labels[i] for i in idx]
        print(f"  {iso['name']} components: {comps}")
        print(f"    B0    = {np.round(b0, 4).tolist()}")
        print(f"    alpha = {np.round(alpha, 4).tolist()}")
        print(f"    Be    = {np.round(b0 + 0.5 * alpha, 4).tolist()}")


def main():
    _print_inputs()
    settings = get_run_settings("so2", PRESET_OVERRIDE)
    preset = settings["preset_values"]
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        jitter = np.zeros_like(coords)
        jitter[1] = rng.normal(0.0, [0.05, 0.03, 0.03])
        jitter[2] = rng.normal(0.0, [0.05, 0.03, 0.03])
        starts.append(coords + jitter)

    optimizer_kwargs = dict(
        quantum_backend=settings["quantum_backend"],
        orca_executable=settings["orca_exe"],
        method_preset="fast",
        orca_method=settings["orca_method"],
        orca_basis=settings["orca_basis"],
        spectral_only=False,
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
        enable_geometry_guardrails=bool(preset["enable_geometry_guardrails"]),
        enforce_quantum_descent=bool(preset["enforce_quantum_descent"]),
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
        use_orca_rovib=False,
        rovib_recalc_every=1,
        rovib_source_mode="hybrid_auto",
        symmetry="C2v",
        debug_rank_diagnostics=False,
        debug_sv_count=6,
        base_workdir=".",
    )

    print(f"\nUsing preset: {settings['selected_preset']}")
    print(f"Running {int(preset['n_starts'])} starts with max_workers={int(preset['max_workers'])} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
        job_name="so2",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])

    best = select_best_result(results, spectral_gate_abs=0.05, spectral_gate_rel=2.0)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("so2_optimized.xyz", "w") as f:
            f.write("3\n")
            f.write("Best SO2 geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r1, r2, ang = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_r1, mean_r2, mean_ang = all_metrics.mean(axis=0)
    std_r1, std_r2, std_ang = all_metrics.std(axis=0)

    print("\n" + "=" * 66)
    print("  SO2 fit: recovered geometry from experimental constants")
    print("=" * 66)
    print(f"  {'Parameter':<18}  {'Recovered':>12}")
    print("  " + "-" * 60)
    print(f"  {'r(S-O1) [A]':<18}  {r1:>12.6f}")
    print(f"  {'r(S-O2) [A]':<18}  {r2:>12.6f}")
    print(f"  {'angle O-S-O':<18}  {ang:>12.4f}")
    print("=" * 66)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(S-O1): {mean_r1:.6f} ± {std_r1:.6f} Ang")
    print(f"    r(S-O2): {mean_r2:.6f} ± {std_r2:.6f} Ang")
    print(f"    angle  : {mean_ang:.4f} ± {std_ang:.4f} deg")
    print(f"  Best run spectral RMS MHz: {best['freq_rms']:.6f}")
    print(f"  Best run energy (Eh):      {best['energy']:.10f}")
    score = underconstrained_success_score(results, best, isotopologues)
    std_vals = np.asarray(score["metric_std"], dtype=float)
    std_r1 = float(std_vals[0]) if std_vals.size > 0 else float("nan")
    std_r2 = float(std_vals[1]) if std_vals.size > 1 else float("nan")
    std_ang = float(std_vals[2]) if std_vals.size > 2 else float("nan")
    print("-" * 66)
    print("  Underconstrained success score (rank-aware):")
    print(f"    Score (0-100): {score['score']:.1f}")
    print(
        f"    Constrained rank: {score['constrained_rank']}/{score['internal_dof']} "
        f"({100.0 * score['rank_fraction']:.1f}%)"
    )
    print(
        f"    Multi-start stability std: "
        f"dr={std_r1:.6f} A, dr2={std_r2:.6f} A, dang={std_ang:.4f} deg"
    )
    print(
        f"    Spectral vs uncertainty: RMS/sigma ~= {score['sigma_ratio']:.1f}x "
        f"(sigma scale {score['sigma_scale']:.4f} MHz)"
    )
    if score.get("conformer_weights_final") is not None:
        print(f"    Conformer weights (final): {np.round(score['conformer_weights_final'], 4).tolist()}")
    if score["score"] >= 80.0:
        verdict = "strong geometry recovery under partial data"
    elif score["score"] >= 60.0:
        verdict = "useful geometry recovery; add isotopologues for tighter spectroscopy"
    else:
        verdict = "geometry likely regularized by quantum prior; low spectral confidence"
    print(f"    Verdict: {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    freeze_support()
    main()
