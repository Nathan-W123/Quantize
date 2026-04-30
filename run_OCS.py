import numpy as np
from multiprocessing import freeze_support
from geometryguess import guess_linear_triatomic
from multistart import run_multistart, select_best_result

# ── OCS validation with two isotopologues (real-style B0 inputs) ──────────────
#
# This script validates hybrid inversion on a linear triatomic using two
# isotopologues with B0 inputs and alpha corrections.
# The expected geometry is the accepted near-equilibrium linear OCS structure.
#
# Accepted OCS geometry targets (reference values for validation):
#   r(C=O) = 1.1563 Å
#   r(C=S) = 1.5610 Å
#   angle O-C-S = 180.0°
#
# Real-style source values used below:
#   16O12C32S: B0 = 6081.4921 MHz, sigma = 0.0001, alpha = 25.24 MHz
#   16O12C34S: B0 = 5932.8159 MHz, sigma = 0.0002, alpha = 24.32 MHz

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_O16 = 15.99491
m_C12 = 12.00000
m_S32 = 31.97207
m_S34 = 33.96787

# ── Accepted geometry (reference for validation printout) ─────────────────────
r_CO_ref = 1.1563
r_CS_ref = 1.5610
ang_ref = 180.0

# ── Observed rotational data model ─────────────────────────────────────────────
# For linear OCS, use B component only for stable conditioning.
component_idx = [1]
alpha_table = {
    "16O12C32S": np.array([25.24]),  # MHz
    "16O12C34S": np.array([24.32]),  # MHz
}
sigma_table = {
    "16O12C32S": np.array([0.0001]),  # MHz
    "16O12C34S": np.array([0.0002]),  # MHz
}
obs_b0_values = {
    "16O12C32S": np.array([6081.4921]),  # MHz
    "16O12C34S": np.array([5932.8159]),  # MHz
}

elems = ["O", "C", "S"]
coords = guess_linear_triatomic(
    left_elem="O",
    center_elem="C",
    right_elem="S",
    r_left_center=1.22,
    r_center_right=1.50,
    bend_deg=-1.1,
)

all_masses = {
    "16O12C32S": np.array([m_O16, m_C12, m_S32]),
    "16O12C34S": np.array([m_O16, m_C12, m_S34]),
}

isotopologues = [
    {
        "masses": all_masses["16O12C32S"].tolist(),
        "component_indices": component_idx,
        "obs_constants": obs_b0_values["16O12C32S"].tolist(),
        "sigma_constants": sigma_table["16O12C32S"].tolist(),
        "alpha_constants": alpha_table["16O12C32S"].tolist(),
    },
    {
        "masses": all_masses["16O12C34S"].tolist(),
        "component_indices": component_idx,
        "obs_constants": obs_b0_values["16O12C34S"].tolist(),
        "sigma_constants": sigma_table["16O12C34S"].tolist(),
        "alpha_constants": alpha_table["16O12C34S"].tolist(),
    },
]

# ── Backend / optimization controls ────────────────────────────────────────────
USE_QUANTUM_PRIOR = True
QUANTUM_BACKEND = "psi4"  # "orca" or "psi4"
N_STARTS = 5
RNG_SEED = 23
MAX_WORKERS = 3
WRITE_XYZ = False

def _metrics(arr):
    o = arr[0]
    c = arr[1]
    s = arr[2]
    r_co = float(np.linalg.norm(o - c))
    r_cs = float(np.linalg.norm(s - c))
    v1 = o - c
    v2 = s - c
    ang = float(np.degrees(np.arccos(
        np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    )))
    return r_co, r_cs, ang


def main():
    rng = np.random.default_rng(RNG_SEED)
    starts = [coords.copy()]
    for _ in range(N_STARTS - 1):
        jitter = np.zeros_like(coords)
        jitter[0] = rng.normal(0.0, [0.06, 0.03, 0.03])  # O
        jitter[2] = rng.normal(0.0, [0.06, 0.03, 0.03])  # S
        starts.append(coords + jitter)

    optimizer_kwargs = dict(
        quantum_backend=QUANTUM_BACKEND,
        spectral_only=not USE_QUANTUM_PRIOR,
        max_iter=150,
        conv_freq=150.0,
        conv_energy=1e-8,
        conv_step_range=1e-6,
        conv_step_null=5e-2,
        conv_grad_null=1e-5,
        trust_radius=0.05,
        null_trust_radius=0.025,
        lambda_damp=1e-3,
        hess_recalc_every=2,
        sv_threshold=1e-3,
        objective_mode="split",
        alpha_quantum=1.0,
        spectral_delta=1e-4,
        robust_loss="none",
        # Higher-quality default quantum prior for geometry fidelity.
        psi4_method="wB97X-D",
        psi4_basis="def2-TZVPP",
        base_workdir=".",
    )

    print(f"Running {N_STARTS} starts with max_workers={MAX_WORKERS} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(MAX_WORKERS, N_STARTS),
        job_name="ocs",
    )
    for r in results:
        r["metrics"] = _metrics(r["coords"])

    best = select_best_result(results, spectral_gate_abs=0.01, spectral_gate_rel=2.0)
    final_coords = best["coords"]
    if WRITE_XYZ:
        with open("ocs_optimized.xyz", "w") as f:
            f.write("3\n")
            f.write("Best OCS geometry from parallel multistart\n")
            for e, (x, y, z) in zip(elems, final_coords):
                f.write(f"{e:2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n")

    # ── Validation summary vs accepted geometry ───────────────────────────────────
    r_CO, r_CS, ang_ocs = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_rco, mean_rcs, mean_ang = all_metrics.mean(axis=0)
    std_rco, std_rcs, std_ang = all_metrics.std(axis=0)

    print("\n" + "=" * 64)
    print("  OCS validation: recovered geometry vs accepted reference")
    print("=" * 64)
    print(f"  {'Parameter':<18}  {'Recovered':>12}  {'Reference':>12}  {'Error':>10}")
    print("  " + "-" * 58)
    print(f"  {'r(C=O) [A]':<18}  {r_CO:>12.6f}  {r_CO_ref:>12.6f}  {r_CO-r_CO_ref:>+10.6f}")
    print(f"  {'r(C=S) [A]':<18}  {r_CS:>12.6f}  {r_CS_ref:>12.6f}  {r_CS-r_CS_ref:>+10.6f}")
    print(f"  {'angle [deg]':<18}  {ang_ocs:>12.4f}  {ang_ref:>12.4f}  {ang_ocs-ang_ref:>+10.4f}")
    print("=" * 64)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C=O):  {mean_rco:.6f} ± {std_rco:.6f} Å")
    print(f"    r(C=S):  {mean_rcs:.6f} ± {std_rcs:.6f} Å")
    print(f"    angle :  {mean_ang:.4f} ± {std_ang:.4f} deg")
    print(f"  Best run spectral RMS MHz: {best['freq_rms']:.6f}")
    print(f"  Best run energy (Eh):      {best['energy']:.10f}")
    print("  Input B0 targets (MHz):")
    print("    16O12C32S: 6081.4921")
    print("    16O12C34S: 5932.8159")
    print("=" * 64)


if __name__ == "__main__":
    freeze_support()
    main()
