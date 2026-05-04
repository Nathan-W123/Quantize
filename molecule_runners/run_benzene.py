import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_planar_benzene
from backend.multistart import run_multistart, select_best_result, underconstrained_success_score
from backend.symmetry import PointGroupSymmetry
from runner.run_settings import get_run_settings

# ── Benzene (D6h) hybrid inversion — experimental rotational targets ──────────
#
# Primary source: In Heo et al., RSC Adv., 2022, 12, 21406–21416.
#   doi:10.1039/D2RA03431J (PMC: PMC9347355)
#
# Table 1 (symmetric oblate tops): only the degenerate in-plane constant B0 was
# resolved (K structure not resolved); we constrain the calculated Watson B
# component only for these species (component index 1).
#
# Table 2 (mono-13C isotopologues): full asymmetric Watson constants A0, B0, C0.
# Atom order: C0 is the unique 13C site (Ip ~ C2v local symmetry); ring numbered
# clockwise in the xy plane with Hi bonded to Ci.

# ── Isotopic masses (amu, NIST/CODATA style) ───────────────────────────────────
m_H = 1.00782503207
m_D = 2.01410177785
m_C12 = 12.00000000000
m_C13 = 13.00335483507

# ── Equilibrium guess from Heo et al. inferred r0 (Å) — Table 1 structure fit ─
coords = guess_planar_benzene(r_cc=1.3971, r_ch=1.0804)

elems = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

# Enforce D6h nuclear positions over the course of the optimisation.
symmetry = PointGroupSymmetry("D6h", elems, coords)

all_masses = {
    "12C6H6": np.array([m_C12] * 6 + [m_H] * 6),
    "12C6D6": np.array([m_C12] * 6 + [m_D] * 6),
    "13C6H6": np.array([m_C13] * 6 + [m_H] * 6),
    # 13C at C0; remaining carbons 12C.
    "13C1-12C5H6": np.array([m_C13] + [m_C12] * 5 + [m_H] * 6),
}

# MHz — Table 1 / Table 2 (Heo 2022); uncertainties from parentheses in source.
sigma_sym_B = np.array([0.0054])  # B0(54) on last digits → 0.0054 MHz
obs_sym_B = {
    "12C6H6": np.array([5689.2855]),
    "12C6D6": np.array([4707.3175]),
    "13C6H6": np.array([5337.884]),
}
sigma_sym_B_table = {
    "12C6H6": sigma_sym_B,
    "12C6D6": np.array([0.0034]),
    "13C6H6": np.array([0.051]),
}

obs_mono_abc = np.array([5689.474, 5568.473, 2868.6])
sigma_mono_abc = np.array([0.018, 0.023, 0.73])

component_sym_B = [1]
component_abc = [0, 1, 2]

isotopologues = [
    {
        "name": "12C6H6",
        "masses": all_masses["12C6H6"].tolist(),
        "component_indices": component_sym_B,
        "obs_constants": obs_sym_B["12C6H6"].tolist(),
        "sigma_constants": sigma_sym_B_table["12C6H6"].tolist(),
        "alpha_constants": [0.0],
    },
    {
        "name": "12C6D6",
        "masses": all_masses["12C6D6"].tolist(),
        "component_indices": component_sym_B,
        "obs_constants": obs_sym_B["12C6D6"].tolist(),
        "sigma_constants": sigma_sym_B_table["12C6D6"].tolist(),
        "alpha_constants": [0.0],
    },
    {
        "name": "13C6H6",
        "masses": all_masses["13C6H6"].tolist(),
        "component_indices": component_sym_B,
        "obs_constants": obs_sym_B["13C6H6"].tolist(),
        "sigma_constants": sigma_sym_B_table["13C6H6"].tolist(),
        "alpha_constants": [0.0],
    },
    {
        "name": "13C1-12C5H6",
        "masses": all_masses["13C1-12C5H6"].tolist(),
        "component_indices": component_abc,
        "obs_constants": obs_mono_abc.tolist(),
        "sigma_constants": sigma_mono_abc.tolist(),
        "alpha_constants": [0.0, 0.0, 0.0],
    },
]

USE_QUANTUM_PRIOR = True
RNG_SEED = 41
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _metrics(arr):
    c = arr[:6]
    h = arr[6:]
    r_cc = [float(np.linalg.norm(c[i] - c[(i + 1) % 6])) for i in range(6)]
    r_ch = [float(np.linalg.norm(h[i] - c[i])) for i in range(6)]
    return float(np.mean(r_cc)), float(np.mean(r_ch))


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
    print(symmetry.summary())
    _print_input_diagnostics()
    settings = get_run_settings("benzene", PRESET_OVERRIDE)
    preset = settings["preset_values"]

    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        jitter = rng.normal(0.0, 0.015, size=coords.shape)
        starts.append(coords + jitter)

    optimizer_kwargs = dict(
        symmetry=symmetry,
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
        job_name="benzene",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])

    best = select_best_result(results, spectral_gate_abs=0.01, spectral_gate_rel=2.0)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("benzene_optimized.xyz", "w") as f:
            f.write("12\n")
            f.write("Best benzene geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_cc, r_ch = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_cc, mean_ch = all_metrics.mean(axis=0)
    std_cc, std_ch = all_metrics.std(axis=0)

    print("\n" + "=" * 60)
    print("  Benzene inferred geometry from isotopologue data")
    print("=" * 60)
    print(f"  {'Parameter':<16}  {'Recovered':>12}")
    print("  " + "-" * 32)
    print(f"  {'mean r(C-C) [A]':<16}  {r_cc:>12.6f}")
    print(f"  {'mean r(C-H) [A]':<16}  {r_ch:>12.6f}")
    print("=" * 60)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C-C): {mean_cc:.6f} ± {std_cc:.6f} Å")
    print(f"    r(C-H): {mean_ch:.6f} ± {std_ch:.6f} Å")
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
