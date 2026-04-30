import numpy as np
from quantize import MolecularOptimizer

# ── Edit these values for your molecule ──────────────────────────────────────

# Initial geometry in Angstroms (one row per atom)
coords = np.array([
    [0.000000,  0.000000, -1.160000],   # O
    [0.000000,  0.000000,  0.000000],   # C
    [0.000000,  0.000000,  1.560000],   # S
])

# Element symbols in the same atom order as coords
elems = ["O", "C", "S"]

# One entry per isotopologue you have data for.
# masses: atomic masses in amu (use exact isotopic masses, not average)
# obs_constants: observed B0 or Be rotational constants in MHz
# sigma_constants: 1σ uncertainties for A, B, C in MHz
# alpha_constants: vibration-rotation corrections α (MHz), where Be ≈ B0 + 0.5*α
isotopologues = [
    {
        # O-16 C-12 S-32 (NIST Triatomic OCS table, ground vibrational state)
        "masses": [15.994915, 12.000000, 31.972071],
        # Linear rotor: fit B only (component index 1).
        "component_indices": [1],
        "obs_constants": [6081.492475],
        # NIST uncertainty from 6081.492475(81) MHz.
        "sigma_constants": [0.000081],
        "alpha_constants": [0.0],
    },
    {
        # O-16 C-12 S-34 (same NIST OCS table/source family)
        "masses": [15.994915, 12.000000, 33.967867],
        "component_indices": [1],
        "obs_constants": [5932.8379],
        # NIST uncertainty from 5932.8379(50) MHz.
        "sigma_constants": [0.0050],
        "alpha_constants": [0.0],
    },
]

# ── ORCA settings ─────────────────────────────────────────────────────────────

# Full path to your ORCA executable.
# Leave as "orca" if ORCA is on your system PATH.
# Or use a full path, e.g.: r"C:\orca_6_0_0\orca.exe"
ORCA_EXE = r"C:\ORCA_6.1.1\orca.exe"

# ── Run ───────────────────────────────────────────────────────────────────────

opt = MolecularOptimizer(
    coords=coords,
    elems=elems,
    isotopologues=isotopologues,
    orca_executable=ORCA_EXE,
    # Use a method preset ("fast", "balanced", "high", "mp2") or explicit method/basis below.
    method_preset="high",
    orca_method="wB97X-D4",
    orca_basis="def2-QZVPP",
    charge=0,
    multiplicity=1,
    workdir="orca_workdir",   # ORCA input/output files go here
    hess_recalc_every=5,      # recalculate Hessian every 5 ORCA calls
    trust_radius=0.02,        # start conservative; optimizer adapts this dynamically
    conv_step=1e-7,
    conv_freq=1.0,
    lambda_damp=1e-1,         # adapted automatically during run()
    objective_mode="split",   # "split" (original SVD) or "joint" (regularized blended solve)
    alpha_quantum=1.0,        # larger = stronger quantum/PES guidance
    robust_loss="none",       # disable robust reweighting for baseline diagnostics
    robust_param=1.5,         # robust loss transition parameter (scaled residual units)
    spectral_delta=1e-3,      # base FD step; scaled per coordinate internally
    use_internal_preconditioner=False,
    use_orca_rovib=True,      # Option 2: derive rovibrational alpha corrections from ORCA AnFreq
    rovib_recalc_every=1,     # recompute alpha whenever a full Hessian refresh is performed
)

# If you already have ORCA .engrad and .hess files, comment out orca_executable
# above and use this instead:
# opt.load_orca("path/to/molecule.engrad", "path/to/molecule.hess")

final_coords = opt.run()
opt.write_xyz("optimized.xyz")
opt.report()
