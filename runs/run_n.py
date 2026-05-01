import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_geometry
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from run_settings import get_run_settings

# ── Methanol inversion from input isotopologue rotational constants ───────────
# Atom order: C, O, Hc1, Hc2, Hc3, Ho
elems = ["C", "O", "H", "H", "H", "H"]
bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_H = 1.00782503207
m_D = 2.01410177785
m_C12 = 12.0
m_C13 = 13.00335483507
m_O16 = 15.99491461957
m_O18 = 17.99915961286

all_masses = {
    "12CH3-16OH": np.array([m_C12, m_O16, m_H, m_H, m_H, m_H]),
    "13CH3-16OH": np.array([m_C13, m_O16, m_H, m_H, m_H, m_H]),
    "12CH3-18OH": np.array([m_C12, m_O18, m_H, m_H, m_H, m_H]),
    "CH3OD": np.array([m_C12, m_O16, m_H, m_H, m_H, m_D]),
    "CD3OH": np.array([m_C12, m_O16, m_D, m_D, m_D, m_H]),
}

# Input A/B/C constants (MHz) provided by user.
obs_b0_values = {
    "12CH3-16OH": np.array([127484.0, 24679.908, 23769.704]),
    "13CH3-16OH": np.array([126903.0, 23735.61, 22875.02]),
    "12CH3-18OH": np.array([127000.0, 23238.82, 22423.48]),
    "CH3OD": np.array([127510.0, 22826.5, 22126.3]),
    "CD3OH": np.array([99840.0, 12093.82, 11411.29]),
}

# Parenthesized uncertainties converted to MHz.
# Estimated 12CH3-18OH A value gets larger sigma.
sigma_table = {
    "12CH3-16OH": np.array([40.0, 0.001, 0.001]),
    "13CH3-16OH": np.array([2.0, 0.01, 0.01]),
    "12CH3-18OH": np.array([200.0, 0.05, 0.05]),
    "CH3OD": np.array([50.0, 0.2, 0.2]),
    "CD3OH": np.array([60.0, 0.04, 0.04]),
}

# User-provided vibrational correction (MHz) for nu8 (CO-stretch) only:
#   Delta A ~= -450, Delta B ~= -210, Delta C ~= -195
# These are partial mode contributions rather than full summed alpha over all
# modes. We apply them as the current alpha constants baseline for each
# isotopologue; ORCA rovibrational updates can still refine during optimization.
nu8_alpha = np.array([-450.0, -210.0, -195.0], dtype=float)
alpha_table = {k: nu8_alpha.copy() for k in obs_b0_values}

# ── Initial guess from general graph-based guesser ────────────────────────────
coords = guess_geometry(elems, bonds)

isotopologues = []
for name in ["12CH3-16OH", "13CH3-16OH", "12CH3-18OH", "CH3OD", "CD3OH"]:
    isotopologues.append(
        {
            "name": name,
            "masses": all_masses[name].tolist(),
            "component_indices": [0, 1, 2],
            "obs_constants": obs_b0_values[name].tolist(),
            "sigma_constants": sigma_table[name].tolist(),
            "alpha_constants": alpha_table[name].tolist(),
            "torsion_sensitive": True,
        }
    )

USE_QUANTUM_PRIOR = True
N_STARTS = 5
RNG_STARTS = 31
MAX_WORKERS = 3
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _metrics(arr):
    def _dist(a, b):
        return float(np.linalg.norm(arr[a] - arr[b]))

    def _angle(i, j, k):
        v1 = arr[i] - arr[j]
        v2 = arr[k] - arr[j]
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    r_co = _dist(0, 1)
    r_oh = _dist(1, 5)
    r_ch_avg = float(np.mean([_dist(0, 2), _dist(0, 3), _dist(0, 4)]))
    ang_coh = _angle(0, 1, 5)
    ang_hch_avg = float(np.mean([_angle(2, 0, 3), _angle(2, 0, 4), _angle(3, 0, 4)]))
    return r_co, r_oh, r_ch_avg, ang_coh, ang_hch_avg


def main():
    print("Methanol isotopologue input constants (MHz):")
    print(f"  {'Isotopologue':<12}  {'A':>12}  {'B':>12}  {'C':>12}")
    print("  " + "-" * 54)
    for name in ["12CH3-16OH", "13CH3-16OH", "12CH3-18OH", "CH3OD", "CD3OH"]:
        vals = obs_b0_values[name]
        print(f"  {name:<12}  {vals[0]:>12.3f}  {vals[1]:>12.3f}  {vals[2]:>12.3f}")
    print()

    settings = get_run_settings("methanol", PRESET_OVERRIDE)
    preset = settings["preset_values"]
    rng_starts = np.random.default_rng(RNG_STARTS)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        starts.append(coords + rng_starts.normal(0.0, 0.05, size=coords.shape))
    conformer_defs = [
        {"name": "anti_like", "offset": np.zeros_like(coords), "weight": 0.60, "energy": 0.0},
        {"name": "gauche_like", "offset": np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.08, -0.04],
            [0.0, -0.08, -0.04],
            [0.0, 0.00, 0.06],
            [0.0, 0.05, 0.03],
        ], dtype=float), "weight": 0.40, "energy": 0.5},
    ]

    optimizer_kwargs = dict(
        quantum_backend=settings["quantum_backend"],
        orca_executable=settings["orca_exe"],
        spectral_only=not USE_QUANTUM_PRIOR,
        max_iter=500,
        conv_step=1e-7,
        conv_freq=float(preset["conv_freq"]),
        spectral_accept_relax=float(preset.get("spectral_accept_relax", 0.0)),
        conv_energy=1e-8,
        conv_step_range=1e-3,
        conv_step_null=1e-3,
        conv_grad_null=1e-1,
        orca_update_thresh=0.01,
        hess_recalc_every=4,
        adaptive_hess_schedule=True,
        hess_recalc_min=1,
        hess_recalc_max=10,
        sv_threshold=1e-4,
        sv_min_abs=0.0,
        trust_radius=float(preset["trust_radius"]),
        null_trust_radius=0.01,
        lambda_damp=1e-3,
        objective_mode="split",
        alpha_quantum=1.0,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=5.0,
        spectral_delta=2e-4,
        robust_loss="none",
        robust_param=1.0,
        torsion_aware_weighting=True,
        torsion_a_weight=1.0,
        use_autoconfig=True,
        auto_sanitize_spectral=True,
        sigma_floor_mhz=float(preset["sigma_floor_mhz"]),
        max_spectral_weight=float(preset["max_spectral_weight"]),
        enable_geometry_guardrails=True,
        guardrail_bond_scale_min=0.70,
        guardrail_bond_scale_max=1.35,
        guardrail_clash_scale=0.65,
        accept_requires_geometry_valid=True,
        sanitize_jacobian_row_norm_max=1e9,
        sanitize_tiny_target_mhz=1e-3,
        use_internal_preconditioner=False,
        enforce_quantum_descent=bool(preset.get("enforce_quantum_descent", False)),
        quantum_descent_tol=float(preset.get("quantum_descent_tol", 1e-5)),
        use_internal_priors=bool(preset["use_internal_priors"]),
        prior_weight=1.0,
        prior_auto_from_initial=True,
        prior_use_dihedrals=True,
        prior_sigma_bond=0.06,
        prior_sigma_angle_deg=4.0,
        prior_sigma_dihedral_deg=20.0,
        use_conformer_mixture=bool(preset["use_conformer_mixture"]),
        conformer_defs=conformer_defs,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
        method_preset=None,
        orca_method="B3LYP",
        orca_basis="def2-SVP",
        use_orca_rovib=False,
        rovib_recalc_every=2,
        rovib_source_mode="hybrid_auto",
        symmetry=None,
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
        job_name="methanol",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])
    best = select_best_result(results, spectral_gate_abs=0.1, spectral_gate_rel=2.0)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("methanol_optimized.xyz", "w") as f:
            f.write(f"{len(elems)}\n")
            f.write("Best methanol geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_co, r_oh, r_ch_avg, ang_coh, ang_hch_avg = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_vals = all_metrics.mean(axis=0)
    std_vals = all_metrics.std(axis=0)

    print("\n" + "=" * 66)
    print("  Methanol inferred geometry from isotopologue constants")
    print("=" * 66)
    print(f"  {'Parameter':<20}  {'Recovered':>12}")
    print("  " + "-" * 36)
    print(f"  {'r(C-O) [A]':<20}  {r_co:>12.6f}")
    print(f"  {'r(O-H) [A]':<20}  {r_oh:>12.6f}")
    print(f"  {'r(C-H)_avg [A]':<20}  {r_ch_avg:>12.6f}")
    print(f"  {'angle C-O-H [deg]':<20}  {ang_coh:>12.4f}")
    print(f"  {'angle H-C-H_avg':<20}  {ang_hch_avg:>12.4f}")
    print("=" * 66)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C-O):        {mean_vals[0]:.6f} ± {std_vals[0]:.6f} Å")
    print(f"    r(O-H):        {mean_vals[1]:.6f} ± {std_vals[1]:.6f} Å")
    print(f"    r(C-H)_avg:    {mean_vals[2]:.6f} ± {std_vals[2]:.6f} Å")
    print(f"    angle C-O-H:   {mean_vals[3]:.4f} ± {std_vals[3]:.4f} deg")
    print(f"    angle H-C-H*:  {mean_vals[4]:.4f} ± {std_vals[4]:.4f} deg")
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


if __name__ == "__main__":
    freeze_support()
    main()
