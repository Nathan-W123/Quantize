import numpy as np
from multiprocessing import freeze_support
from geometryguess import guess_bent_triatomic
from multistart import run_multistart, select_best_result
from spectral import sanitize_isotopologues

# ── Water inversion from observed isotopologue constants (unknown geometry) ───
#
# This script is intentionally not anchored to any known geometry target.
# You provide observed B0 constants (from experiment/literature), sigma, and alpha.
# The optimizer then infers a geometry from those constraints.
#
# For undersaturated datasets, enable the quantum prior (ORCA) so unconstrained
# directions are stabilized by energy gradient/Hessian information.

# ── Exact isotopic masses (amu) ───────────────────────────────────────────────
m_H   = 1.00782503207
m_D   = 2.01410177785
m_O16 = 15.99491461956
m_O18 = 17.99915961286

all_masses = {
    "H2-16O": np.array([m_O16, m_H,  m_H]),
    "H2-18O": np.array([m_O18, m_H,  m_H]),
    "HDO":    np.array([m_O16, m_H,  m_D]),
    "D2O":    np.array([m_O16, m_D,  m_D]),
}

# ── Observed data from provided literature-style table ─────────────────────────
# obs_constants are B0 (MHz); Be is formed internally as B0 + 0.5*alpha.
alpha_table = {
    "H2-16O": np.array([-43390.0, 10560.0, 6240.0]),
    "H2-18O": np.array([-42350.0, 10540.0, 6210.0]),
    "HDO":    np.array([-28450.0, 5920.0, 3880.0]),
    "D2O":    np.array([-15320.0, 3740.0, 2210.0]),
}

sigma_table = {
    "H2-16O": np.array([0.2, 0.2, 0.2]),
    "H2-18O": np.array([0.2, 0.2, 0.2]),
    "HDO":    np.array([0.3, 0.2, 0.2]),
    "D2O":    np.array([0.3, 0.2, 0.2]),
}

obs_b0_values = {
    "H2-16O": np.array([835840.3, 435351.7, 278138.7]),
    "H2-18O": np.array([825366.1, 435331.6, 276950.5]),
    "HDO":    np.array([701932.3, 272913.3, 192055.2]),
    "D2O":    np.array([462291.7, 217982.3, 145301.1]),
}

elems = ["O", "H", "H"]
coords = guess_bent_triatomic(
    central_elem="O",
    terminal1_elem="H",
    terminal2_elem="H",
    r1=1.01,
    r2=1.01,
    angle_deg=108.0,
)

# ── Data selection: corrected isotopologue set with full ABC ──────────────────
component_idx = [0, 1, 2]

# ── Isotopologue definitions from observed constants ──────────────────────────
isotopologues = [
    {
        "name":              "H2-16O",
        "masses":            all_masses["H2-16O"].tolist(),
        "component_indices": component_idx,
        "obs_constants":     obs_b0_values["H2-16O"][component_idx].tolist(),
        "sigma_constants":   sigma_table["H2-16O"][component_idx].tolist(),
        "alpha_constants":   alpha_table["H2-16O"][component_idx].tolist(),
    },
    {
        "name":              "H2-18O",
        "masses":            all_masses["H2-18O"].tolist(),
        "component_indices": component_idx,
        "obs_constants":     obs_b0_values["H2-18O"][component_idx].tolist(),
        "sigma_constants":   sigma_table["H2-18O"][component_idx].tolist(),
        "alpha_constants":   alpha_table["H2-18O"][component_idx].tolist(),
    },
]

# ── Prior mode control ─────────────────────────────────────────────────────────
# Set True to use ORCA gradient/Hessian and fill undersaturated directions.
# Set False for spectral-only test mode.
USE_QUANTUM_PRIOR = True
QUANTUM_BACKEND = "psi4"  # "orca" or "psi4"
N_STARTS = 5
RNG_SEED = 11
MAX_WORKERS = 3
WRITE_XYZ = False

def _metrics(arr):
    o = arr[0]
    h1 = arr[1]
    h2 = arr[2]
    r1 = float(np.linalg.norm(h1 - o))
    r2 = float(np.linalg.norm(h2 - o))
    v1 = h1 - o
    v2 = h2 - o
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
        print(f"    B0    = {np.round(b0, 4).tolist()}")
        print(f"    alpha = {np.round(alpha, 4).tolist()}")
        print(f"    Be    = {np.round(be, 4).tolist()}")

    cleaned, notes = sanitize_isotopologues(isotopologues, coords, delta=1e-4)
    print("\nSanitizer pre-check at initial geometry:")
    for iso_in, iso_out in zip(isotopologues, cleaned):
        name = iso_in.get("name", "iso")
        kept = [labels[i] if 0 <= int(i) < 3 else f"R{int(i)}" for i in np.asarray(iso_out["component_indices"], dtype=int)]
        print(f"  {name}: kept components {kept}")
    if notes:
        for n in notes:
            print(f"    note: {n}")
    else:
        print("    note: no components dropped at initial geometry")


def main():
    _print_input_diagnostics()
    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(N_STARTS - 1):
        jitter = np.zeros_like(coords)
        jitter[1] = rng.normal(0.0, [0.03, 0.03, 0.02])
        jitter[2] = rng.normal(0.0, [0.03, 0.03, 0.02])
        starts.append(coords + jitter)

    optimizer_kwargs = dict(
        quantum_backend=QUANTUM_BACKEND,
        spectral_only=not USE_QUANTUM_PRIOR,
        max_iter=150,
        conv_step=1e-10,
        conv_freq=250.0,
        conv_energy=1e-8,
        conv_step_range=1e-6,
        conv_step_null=2e-2,
        conv_grad_null=1e-5,
        trust_radius=0.05,
        null_trust_radius=0.01,
        lambda_damp=1e-3,
        hess_recalc_every=2,
        sv_threshold=3e-4,
        sv_min_abs=7.5e3,
        objective_mode="split",
        alpha_quantum=1.3,
        quantum_weight_max=2.5,
        spectral_delta=1e-4,
        robust_loss="none",
        psi4_basis="cc-pVTZ",
        base_workdir=".",
        debug_rank_diagnostics=True,
        debug_sv_count=6,
    )

    print(f"Running {N_STARTS} starts with max_workers={MAX_WORKERS} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(MAX_WORKERS, N_STARTS),
        job_name="water",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])
    best = select_best_result(results, spectral_gate_abs=0.02, spectral_gate_rel=1.2)
    final_coords = best["coords"]

    if WRITE_XYZ:
        with open("water_optimized.xyz", "w") as f:
            f.write("3\n")
            f.write("Best water geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    # ── Recovered geometry summary (no reference target assumed) ─────────────────
    r_OH1, r_OH2, angle_HOH = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_r1, mean_r2, mean_ang = all_metrics.mean(axis=0)
    std_r1, std_r2, std_ang = all_metrics.std(axis=0)

    print("\n" + "=" * 56)
    print("  Inferred geometry from isotopologue data")
    print("=" * 56)
    print(f"  {'Parameter':<16}  {'Recovered':>12}")
    print("  " + "-" * 30)
    print(f"  {'r(O-H1) [A]':<16}  {r_OH1:>12.6f}")
    print(f"  {'r(O-H2) [A]':<16}  {r_OH2:>12.6f}")
    print(f"  {'angle [deg]':<16}  {angle_HOH:>12.4f}")
    print("=" * 56)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(O-H1): {mean_r1:.6f} ± {std_r1:.6f} Å")
    print(f"    r(O-H2): {mean_r2:.6f} ± {std_r2:.6f} Å")
    print(f"    angle  : {mean_ang:.4f} ± {std_ang:.4f} deg")
    print(f"  Best run spectral RMS MHz: {best['freq_rms']:.6f}")
    print(f"  Best run energy (Eh):      {best['energy']:.10f}")
    print()
    print("  Note:")
    print("    - This run does not use a known geometry target.")
    print("    - Use independent literature/fit B0, sigma, and alpha tables above.")
    if not USE_QUANTUM_PRIOR:
        print("    - Quantum prior is OFF (spectral-only).")
        print("      For undersaturated datasets, set USE_QUANTUM_PRIOR=True.")
    print("=" * 56)


if __name__ == "__main__":
    freeze_support()
    main()
