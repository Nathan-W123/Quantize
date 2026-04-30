import numpy as np
from multiprocessing import freeze_support

from geometryguess import guess_geometry
from multistart import run_multistart, select_best_result
from spectral import _rotational_constants

# ── Larger-molecule validation: methanol (CH3OH) ──────────────────────────────
#
# This is a synthetic benchmark for a >3-atom case using multiple isotopologues.
# Targets are generated from a fixed reference geometry and then inverted from an
# automatically generated initial guess using guess_geometry(elems, bonds).

# Atom order: C, O, Hc1, Hc2, Hc3, Ho
elems = ["C", "O", "H", "H", "H", "H"]
bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]

# ── Reference methanol geometry for synthetic targets ──────────────────────────
# Approximate near-equilibrium scaffold for test generation.
ref_coords = np.array(
    [
        [0.0000, 0.0000, 0.0000],   # C
        [1.4300, 0.0000, 0.0000],   # O
        [-0.3620, 1.0280, 0.0000],  # Hc1
        [-0.3620, -0.5140, 0.8900], # Hc2
        [-0.3620, -0.5140, -0.8900],# Hc3
        [1.7700, 0.8900, 0.0000],   # Ho
    ],
    dtype=float,
)

# ── Isotopic masses (amu) ─────────────────────────────────────────────────────
m_H = 1.00782503207
m_D = 2.01410177785
m_C12 = 12.0
m_C13 = 13.00335483507
m_O16 = 15.99491461957

all_masses = {
    "12CH3OH": np.array([m_C12, m_O16, m_H, m_H, m_H, m_H]),
    "13CH3OH": np.array([m_C13, m_O16, m_H, m_H, m_H, m_H]),
    "12CH3OD": np.array([m_C12, m_O16, m_H, m_H, m_H, m_D]),
}

# ── Build synthetic B0 observations from reference Be ─────────────────────────
# Keep this test physically realistic: non-zero alpha and heterogeneous sigma.
alpha_table = {
    "12CH3OH": np.array([2.40, 0.45, 0.40]),
    "13CH3OH": np.array([2.10, 0.40, 0.36]),
    "12CH3OD": np.array([1.95, 0.38, 0.34]),
}
sigma_table = {
    "12CH3OH": np.array([0.030, 0.012, 0.012]),
    "13CH3OH": np.array([0.035, 0.014, 0.014]),
    "12CH3OD": np.array([0.040, 0.016, 0.016]),
}

NOISE_SCALE = 0.20
RNG_SEED = 19
rng = np.random.default_rng(RNG_SEED)

be_values = {}
obs_b0_values = {}
for name, masses in all_masses.items():
    be = _rotational_constants(ref_coords, masses)
    be_values[name] = be
    b0 = be - 0.5 * alpha_table[name]
    b0 = b0 + rng.normal(0.0, NOISE_SCALE * sigma_table[name], size=3)
    obs_b0_values[name] = b0

print("Methanol synthetic target model:")
print("  B0 = Be - 0.5*alpha + noise")
print(f"  Noise scale: {NOISE_SCALE:.2f} sigma, RNG seed: {RNG_SEED}")
print(f"  {'Iso':<10}  {'A0 obs':>12}  {'B0 obs':>12}  {'C0 obs':>12}")
print("  " + "-" * 52)
for name in all_masses:
    b0 = obs_b0_values[name]
    print(f"  {name:<10}  {b0[0]:>12.4f}  {b0[1]:>12.4f}  {b0[2]:>12.4f}")
print()

# ── Initial guess from general graph-based guesser ────────────────────────────
coords = guess_geometry(elems, bonds)

isotopologues = []
for name in ["12CH3OH", "13CH3OH", "12CH3OD"]:
    isotopologues.append(
        {
            "masses": all_masses[name].tolist(),
            "component_indices": [0, 1, 2],
            "obs_constants": obs_b0_values[name].tolist(),
            "sigma_constants": sigma_table[name].tolist(),
            "alpha_constants": alpha_table[name].tolist(),
        }
    )

USE_QUANTUM_PRIOR = True
QUANTUM_BACKEND = "psi4"  # "orca" or "psi4"
N_STARTS = 5
RNG_STARTS = 31
MAX_WORKERS = 3
WRITE_XYZ = False


def _metrics(arr):
    def _dist(a, b):
        return float(np.linalg.norm(arr[a] - arr[b]))

    def _angle(i, j, k):
        v1 = arr[i] - arr[j]
        v2 = arr[k] - arr[j]
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    return _dist(0, 1), _dist(1, 5), _angle(0, 1, 5)


def main():
    rng_starts = np.random.default_rng(RNG_STARTS)
    starts = [coords.copy()]
    for _ in range(N_STARTS - 1):
        starts.append(coords + rng_starts.normal(0.0, 0.05, size=coords.shape))

    optimizer_kwargs = dict(
        quantum_backend=QUANTUM_BACKEND,
        spectral_only=not USE_QUANTUM_PRIOR,
        max_iter=20,
        conv_freq=80.0,
        conv_energy=1e-8,
        conv_step_range=1e-6,
        conv_step_null=5e-2,
        conv_grad_null=1e-5,
        trust_radius=0.05,
        null_trust_radius=0.03,
        lambda_damp=1e-3,
        hess_recalc_every=2,
        sv_threshold=1e-3,
        objective_mode="split",
        alpha_quantum=1.0,
        dynamic_quantum_weight=True,
        quantum_weight_beta=2.0,
        quantum_weight_min=0.25,
        quantum_weight_max=4.0,
        spectral_delta=1e-4,
        robust_loss="none",
        psi4_method="B3LYP",
        psi4_basis="cc-pVDZ",
        base_workdir=".",
    )

    print(f"Running {N_STARTS} starts with max_workers={MAX_WORKERS} ...")
    results = run_multistart(
        starts=starts,
        elems=elems,
        isotopologues=isotopologues,
        optimizer_kwargs=optimizer_kwargs,
        max_workers=min(MAX_WORKERS, N_STARTS),
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

    # ── Geometry summary vs synthetic reference ────────────────────────────────────
    r_co, r_oh, ang_coh = _metrics(final_coords)
    all_metrics = np.array([r["metrics"] for r in results], dtype=float)
    mean_rco, mean_roh, mean_ang = all_metrics.mean(axis=0)
    std_rco, std_roh, std_ang = all_metrics.std(axis=0)

    ref_r_co = float(np.linalg.norm(ref_coords[0] - ref_coords[1]))
    ref_r_oh = float(np.linalg.norm(ref_coords[1] - ref_coords[5]))
    v1_ref = ref_coords[0] - ref_coords[1]
    v2_ref = ref_coords[5] - ref_coords[1]
    ref_ang_coh = float(np.degrees(np.arccos(np.clip(np.dot(v1_ref, v2_ref) / (np.linalg.norm(v1_ref) * np.linalg.norm(v2_ref)), -1.0, 1.0))))

    print("\n" + "=" * 66)
    print("  Methanol validation: recovered geometry vs synthetic reference")
    print("=" * 66)
    print(f"  {'Parameter':<18}  {'Recovered':>12}  {'Reference':>12}  {'Error':>10}")
    print("  " + "-" * 60)
    print(f"  {'r(C-O) [A]':<18}  {r_co:>12.6f}  {ref_r_co:>12.6f}  {r_co-ref_r_co:>+10.6f}")
    print(f"  {'r(O-H) [A]':<18}  {r_oh:>12.6f}  {ref_r_oh:>12.6f}  {r_oh-ref_r_oh:>+10.6f}")
    print(f"  {'angle C-O-H':<18}  {ang_coh:>12.4f}  {ref_ang_coh:>12.4f}  {ang_coh-ref_ang_coh:>+10.4f}")
    print("=" * 66)
    print("  Multi-start consensus (mean ± std):")
    print(f"    r(C-O):  {mean_rco:.6f} ± {std_rco:.6f} Å")
    print(f"    r(O-H):  {mean_roh:.6f} ± {std_roh:.6f} Å")
    print(f"    angle :  {mean_ang:.4f} ± {std_ang:.4f} deg")
    print(f"  Best run spectral RMS MHz: {best['freq_rms']:.6f}")
    print(f"  Best run energy (Eh):      {best['energy']:.10f}")


if __name__ == "__main__":
    freeze_support()
    main()
