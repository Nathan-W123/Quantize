import numpy as np
from backend.quantize import MolecularOptimizer

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
    quantum_backend="orca",
    orca_executable=ORCA_EXE,
    charge=0,
    multiplicity=1,
    workdir="orca_workdir",   # ORCA input/output files go here
    max_iter=500,
    conv_step=1e-7,
    conv_freq=1.0,
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
    trust_radius=0.0057314430231269545,
    null_trust_radius=None,
    lambda_damp=0.00016370045068111915,
    objective_mode="split",
    alpha_quantum=0.2778639378704326,
    robust_loss="none",
    robust_param=1.0,
    spectral_delta=0.00034930106014707015,
    auto_sanitize_spectral=True,
    sanitize_jacobian_row_norm_max=1e9,
    sanitize_tiny_target_mhz=1e-3,
    use_internal_preconditioner=False,
    dynamic_quantum_weight=True,
    quantum_weight_beta=2.0,
    quantum_weight_min=0.25,
    quantum_weight_max=5.0,
    method_preset="fast",
    orca_method="wB97X-D4",
    orca_basis="def2-TZVPP",
    use_orca_rovib=False,
    rovib_recalc_every=1,
    spectral_only=False,
    symmetry="Cinf_v",
    debug_rank_diagnostics=False,
    debug_sv_count=6,
)

# If you already have ORCA .engrad and .hess files, comment out orca_executable
# above and use this instead:
# opt.load_orca("path/to/molecule.engrad", "path/to/molecule.hess")

final_coords = opt.run()
opt.write_xyz("optimized.xyz")
opt.report()
