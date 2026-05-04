import numpy as np
from multiprocessing import freeze_support

from backend.geometryguess import guess_staggered_methanol
from backend.multistart import (
    multistart_seed_metrics,
    rank_key_spectral_value,
    run_multistart,
    select_best_result,
    underconstrained_success_score,
)

# Multistart selection: prefer lower spectral RMS over lowest energy; penalize runs that
# never moved from the seed and never improved freq_rms (iter 1 → last).
_SELECTION_STAGNANT_TOL_ANG = 5e-6
_SELECTION_MIN_GAIN_MHZ = 0.05
_SELECTION_STAGNANT_PENALTY_MHZ = 500.0
from backend.spectral import SpectralEngine
from backend.spectral_model import methanol_isotopologue_row
from runner.run_settings import get_run_settings

# ── Methanol, single torsional well: νt = 0, staggered methyl (C_s global minimum) ─
#
# Unlike ethanol, methanol has one deep torsional minimum (often called “staggered”
# in the CH3–OH dihedral sense).  The large-amplitude torsion still splits A/E
# tunneling manifolds; the reduced constants below come from **global
# torsion–rotation / RAM** fits to **νt = 0** (ground torsional state) as published
# in CDMS and the cited papers — not a 50/50 mix of two **geometric** “conformers.”
#
# This driver **disables** the old two-offset `conformer_defs` mixture so the
# optimization targets **one** internal-rotation/ground-torsional dataset only.
#
# Watson A, B, C (MHz), A ≥ B ≥ C.
#
# References (catalog metadata / papers):
#   • 12CH3-16OH — CDMS https://cdms.astro.uni-koeln.de/cgi-bin/cdmsinfo?file=e032504.cat
#       (species tag 032504, v3 May 2016); Xu et al., J. Mol. Spectrosc. 251, 305 (2008).
#   • 12CH3-18OH — CDMS e034504 (Sep. 2012); global fit J. Fisher et al.,
#       J. Mol. Spectrosc. 245, 7 (2007); header constants from cdmsinfo A,B,C table.
#   • CD3OH — CDMS e035505 (Mar. 2022); Ilyushin et al., Astron. Astrophys. 658, A127 (2022).
#   • 13CH3-16OH — Global millimeter/THz analysis for 13CH3OH (Pearson et al.,
#       J. Mol. Spectrosc. 2014, doi:10.1016/j.jms.2014.02.012; RAM parameters as in
#       the CDMS-oriented fit described therein).
#   • CH3OD — Same torsion–rotation family as 12CH3OH (ground νt); values from the
#       long-running Cologne/Xu–Hougen global-fit parameter sets used alongside CDMS.
#
# Atom order: C, O, Hc1, Hc2, Hc3, Ho
elems = ["C", "O", "H", "H", "H", "H"]

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

obs_b0_values = {
    "12CH3-16OH": np.array([127523.4, 24692.5, 23760.3]),
    "13CH3-16OH": np.array([126941.4, 23747.3, 22865.9]),
    "12CH3-18OH": np.array([127285.3749, 23651.2578, 22787.1066]),
    "CH3OD": np.array([127549.4, 22839.1, 22116.9]),
    "CD3OH": np.array([70819.0, 19856.0, 19266.0]),
}

sigma_table = {
    "12CH3-16OH": np.array([2.0, 0.05, 0.05]),
    "13CH3-16OH": np.array([2.0, 0.05, 0.05]),
    "12CH3-18OH": np.array([2.0, 0.05, 0.05]),
    "CH3OD": np.array([10.0, 0.2, 0.2]),
    "CD3OH": np.array([5.0, 0.05, 0.05]),
}

# B₀ catalog targets vs rigid equilibrium (A,B,C); avoid hand-picking partial ν₈ α only.
# Effective equilibrium targets use Be ≈ B₀ + ½α per component (`backend/spectral.py`).
# Default α = 0 here. ORCA AnFreq rovib is off by default (`use_orca_rovib=False`) — turn on only if you accept long runtimes.
alpha_table = {k: np.zeros(3, dtype=float) for k in obs_b0_values}

# Staggered CH3OH template (tetrahedral methyl + ∠COH); not the generic graph guesser.
coords = guess_staggered_methanol()

_ISO_ORDER = ["12CH3-16OH", "13CH3-16OH", "12CH3-18OH", "CH3OD", "CD3OH"]


def _make_isotopologues(spectral_model: str):
    return [
        methanol_isotopologue_row(
            name=name,
            masses=all_masses[name].tolist(),
            obs_abc_mhz=obs_b0_values[name].tolist(),
            sigma_abc_mhz=sigma_table[name].tolist(),
            alpha_abc_mhz=alpha_table[name].tolist(),
            mode=spectral_model,
            torsion_sensitive=True,
        )
        for name in _ISO_ORDER
    ]


USE_QUANTUM_PRIOR = True
N_STARTS = 5
RNG_STARTS = 31
MAX_WORKERS = 3
WRITE_XYZ = False
PRESET_OVERRIDE = None


def _print_spectral_validation_table(coords, isotopologue_rows, spectral_model: str):
    """
    Per-isotopologue residuals (MHz) using the same Be targets and component subset as the objective.
    ``isotopologue_rows`` should match the optimizer (e.g. multistart snapshot with ORCA rovib α).
    """
    se = SpectralEngine(
        isotopologue_rows,
        delta=2e-4,
        torsion_aware_weighting=True,
        torsion_a_weight=1.0,
    )
    labels = ["A", "B", "C"]
    print("\nSpectral validation (target − rigid calc), same rows as optimizer:")
    print(f"  spectral_model={spectral_model}")
    hdr = (
        f"  {'iso':<12}  {'comp':>5}  {'B₀+½α target':>14}  {'calc':>14}  {'resid':>10}  {'σ':>8}  {'|res|/σ':>10}"
    )
    print(hdr)
    print("  " + "-" * 88)
    for iso in se.isotopologues:
        name = iso["name"]
        masses = iso["masses"]
        idx = np.asarray(iso["component_indices"], dtype=int)
        calc_abc = se.rotational_constants(coords, masses)
        be_target = np.asarray(iso["obs_constants"], dtype=float) + 0.5 * np.asarray(
            iso["alpha_constants"], dtype=float
        )
        sig = np.asarray(iso["sigma_constants"], dtype=float)
        for k in range(len(idx)):
            j = int(idx[k])
            lbl = labels[j] if 0 <= j < 3 else f"R{j}"
            tgt = float(be_target[k])
            c = float(calc_abc[j])
            res = tgt - c
            s = max(float(sig[k]), 1e-12)
            print(
                f"  {name:<12}  {lbl:>5}  {tgt:>14.4f}  {c:>14.4f}  {res:>10.4f}  {s:>8.4f}  {abs(res) / s:>10.3f}"
            )
    print("  (calc = principal-axis Watson A,B,C from geometry and masses; target uses α from snapshot)")


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
    settings = get_run_settings("methanol", PRESET_OVERRIDE)
    spectral_model = settings["spectral_model"]
    isotopologues = _make_isotopologues(spectral_model)

    print(
        "Methanol driver: νt = 0 (ground torsional state), staggered methyl vs OH "
        "(single potential minimum). No dual-offset conformer mixture.\n"
    )
    print("Methanol isotopologue input constants (MHz):")
    print(f"  spectral_model={spectral_model}  (from run_settings / MOLECULE_SPECTRAL_MODEL_DEFAULTS)")
    if spectral_model == "internal_rotor_bc":
        print(
            "  Note: internal-rotation proxy — objective uses Watson B,C only (rigid principal-axis); "
            "A shown for reference."
        )
        print(f"  {'Isotopologue':<12}  {'A(ref)':>12}  {'B(fit)':>12}  {'C(fit)':>12}")
    else:
        print(
            "  Default rigid model: full Watson A,B,C vs principal moments (see spectral_model if overridden)."
        )
        print(f"  {'Isotopologue':<12}  {'A(fit)':>12}  {'B(fit)':>12}  {'C(fit)':>12}")
    print("  " + "-" * 54)
    for name in _ISO_ORDER:
        vals = obs_b0_values[name]
        print(f"  {name:<12}  {vals[0]:>12.3f}  {vals[1]:>12.3f}  {vals[2]:>12.3f}")
    print()

    preset = settings["preset_values"]
    rng_starts = np.random.default_rng(RNG_STARTS)
    starts = [coords.copy()]
    for _ in range(int(preset["n_starts"]) - 1):
        starts.append(coords + rng_starts.normal(0.0, 0.05, size=coords.shape))

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
        alpha_quantum=0.45,
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
        use_conformer_mixture=False,
        conformer_defs=None,
        conformer_weight_mode="fixed",
        conformer_temperature_k=298.15,
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

    print(f"Using preset: {settings['selected_preset']}  |  spectral_model: {spectral_model}")
    print(f"Running {int(preset['n_starts'])} starts with max_workers={int(preset['max_workers'])} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(int(preset["max_workers"]), int(preset["n_starts"])),
        job_name="methanol_vt0_staggered",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])
    best = select_best_result(
        results,
        spectral_gate_abs=0.1,
        spectral_gate_rel=2.0,
        primary_objective="spectral",
        penalize_stagnant=True,
        stagnant_coord_rms_tol_ang=_SELECTION_STAGNANT_TOL_ANG,
        stagnant_max_spectral_gain_mhz=_SELECTION_MIN_GAIN_MHZ,
        stagnant_penalty_mhz=_SELECTION_STAGNANT_PENALTY_MHZ,
    )
    final_coords = best["coords"]

    print("\nMultistart seed diagnostics (displacement vs initial start, spectral gain):")
    print(
        f"  Selection: primary_objective=spectral; stagnant penalty "
        f"+{_SELECTION_STAGNANT_PENALTY_MHZ:.0f} MHz if "
        f"|Δx|_rms < {_SELECTION_STAGNANT_TOL_ANG:.1e} Å and gain < {_SELECTION_MIN_GAIN_MHZ:.2f} MHz"
    )
    hdr = f"  {'start':>6}  {'|Δx|_rms Å':>12}  {'Δfreq_1→last MHz':>18}  {'final RMS MHz':>14}  {'rank RMS*':>12}"
    print(hdr)
    print("  " + "-" * 76)
    for r in results:
        m = multistart_seed_metrics(r)
        rk = rank_key_spectral_value(
            r,
            penalize_stagnant=True,
            stagnant_coord_rms_tol_ang=_SELECTION_STAGNANT_TOL_ANG,
            stagnant_max_spectral_gain_mhz=_SELECTION_MIN_GAIN_MHZ,
            stagnant_penalty_mhz=_SELECTION_STAGNANT_PENALTY_MHZ,
        )
        star = "*" if r.get("idx") == best.get("idx") else " "
        print(
            f"  {int(r.get('idx', -1)):>6}  {m['coord_rms_disp_ang']:>12.6f}  "
            f"{m['spectral_gain_mhz']:>18.4f}  {float(r.get('freq_rms', np.inf)):>14.4f}  {rk:>12.4f}{star}"
        )
    print("  *rank RMS = effective value used for ordering (includes stagnant penalty when marked)")

    iso_for_validation = best.get("spectral_isotopologues_snapshot") or isotopologues
    _print_spectral_validation_table(final_coords, iso_for_validation, spectral_model)

    if WRITE_XYZ:
        with open("methanol_vt0_staggered_optimized.xyz", "w") as f:
            f.write(f"{len(elems)}\n")
            f.write("Best methanol (νt=0 staggered) geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    r_co, r_oh, r_ch_avg, ang_coh, ang_hch_avg = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_vals = all_metrics.mean(axis=0)
    std_vals = all_metrics.std(axis=0)

    print("\n" + "=" * 66)
    print("  Methanol (νt=0, staggered) inferred geometry from isotopologue constants")
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
