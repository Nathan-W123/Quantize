# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

This project estimates molecular structure (bond lengths and angles) from isotopologue rotational constants, including undersaturated cases where spectroscopy alone does not fully constrain the geometry.

## Core idea

- Use observed rotational constants (`A`, `B`, `C`) from one or more isotopologues.
- Map ground-state constants toward equilibrium targets using \(B_e \approx B_0 + \tfrac{1}{2}\alpha\) (per component).
- Stack spectral Jacobians across isotopologues, apply **SVD** to split **range space** (spectroscopy-sensitive directions) from **null space** (directions invisible to the stacked Jacobian).
- Use electronic-energy **gradient and Hessian** (Psi4 or ORCA) for a damped-Newton step in the null space so the structure is stabilized where data are silent.

Full notation, derivations, and formulas used in code live in **[MATH.typ](MATH.typ)** (Typst — best PDF layout). A Markdown mirror is **[MATH.md](MATH.md)**. Compile with `typst compile MATH.typ` (install [Typst](https://typst.app)); this produces **`MATH.pdf`** in the repo root when built locally.

## Main modules (`backend/`)

| Module | Role |
|--------|------|
| [`backend/quantize.py`](backend/quantize.py) | `MolecularOptimizer`: spectral + quantum hybrid loop |
| [`backend/spectral.py`](backend/spectral.py) | Inertia tensor, \(A,B,C\), Jacobians, residuals, weighting, optional conformer mixtures |
| [`backend/SVD.py`](backend/SVD.py) | `SubspaceOptimizer`: SVD split, range/null steps, joint objective option |
| [`backend/quantum.py`](backend/quantum.py) | ORCA parsers; Wilson **B**-matrix; primitive internal-coordinate derivatives |
| [`backend/Psi4.py`](backend/Psi4.py) | Psi4 energy / gradient / Hessian with unit conversion to Å |
| [`backend/internal_prior.py`](backend/internal_prior.py) | Optional internal-coordinate priors stacked with the spectral block |
| [`backend/geometryguess.py`](backend/geometryguess.py) | Template guesses and spring-style relaxation |
| [`backend/multistart.py`](backend/multistart.py) | Parallel multi-start runs and best-run selection |
| [`backend/bayes_tune.py`](backend/bayes_tune.py) | Optional Bayesian hyperparameter search (`scikit-optimize`) |
| [`backend/symmetry.py`](backend/symmetry.py) | Optional point-group projection of steps and coordinates |
| [`backend/autoconfig.py`](backend/autoconfig.py) | Adaptive trust region / damping / weight policy from diagnostics |

## Torsion / Large-Amplitude Motion (LAM) pipeline

A self-contained torsion-rotation pipeline handles molecules with an internal methyl (or other Cn) rotor. It uses a RAM-lite (rho-axis method) Hamiltonian in a Fourier basis |m⟩ and is independent of the geometry-inversion loop above.

### Torsion backend modules

| Module | Role |
|--------|------|
| [`backend/torsion_hamiltonian.py`](backend/torsion_hamiltonian.py) | `TorsionHamiltonianSpec`, Fourier potential matrix, `solve_ram_lite_levels`, `torsion_probability_density` |
| [`backend/torsion_rot_hamiltonian.py`](backend/torsion_rot_hamiltonian.py) | Full J-block torsion-rotation Hamiltonian with centrifugal distortion |
| [`backend/torsion_symmetry.py`](backend/torsion_symmetry.py) | C3 block decomposition (A/E1/E2), tunneling splittings, selection rules, nuclear-spin weights |
| [`backend/torsion_average.py`](backend/torsion_average.py) | Quantum and Boltzmann torsion-scan averaging of A/B/C constants; rigorous uncertainty propagation |
| [`backend/torsion_intensities.py`](backend/torsion_intensities.py) | `⟨ψ\|cos(α)\|ψ⟩` matrix elements, Hönl-London factors, complete line-list generation |
| [`backend/torsion_fitter.py`](backend/torsion_fitter.py) | Damped Gauss-Newton fitting to levels, transitions, or joint levels + rotational constants |
| [`backend/torsion_uncertainty.py`](backend/torsion_uncertainty.py) | Jacobian, covariance, Fisher information, identifiability |
| [`backend/torsion_lam_integration.py`](backend/torsion_lam_integration.py) | LAM correction report with uncertainty propagation into the main spectral fit |
| [`backend/hindered_rotor.py`](backend/hindered_rotor.py) | Independent 1D hindered-rotor solver (legacy; used for Boltzmann weight helper only) |

### What the torsion pipeline provides

**Energy levels and tunneling** — `solve_ram_lite_levels` diagonalises the RAM-lite Hamiltonian for any (J, K) block. `torsion_symmetry` decomposes levels into A/E1/E2 species and reports A–E tunneling splittings (validated against methanol literature values).

**Scan averaging with uncertainty propagation** — `torsion_average` weights torsion-scan geometries by the quantum probability density |ψ(α)|² of any eigenstate (or a thermal mixture), producing effective A/B/C constants with rigorously propagated uncertainties from both grid-point measurement errors (Hessian-diagonal σ) and representational scatter.

**Line intensities** — `torsion_intensities` computes |⟨ψ_hi|cos(α)|ψ_lo⟩|² transition dipole matrix elements, applies Hönl-London factors (symmetric-top approximation), applies C3 nuclear-spin statistical weights (A:E = 1:2 for CH₃), and exports a complete sorted line list as CSV. Selection rules (A↔A, E↔E allowed; A↔E forbidden) are enforced automatically.

**Parameter fitting** — `torsion_fitter` provides three fitting modes:
- `fit_torsion_to_levels` — fit to observed torsional level energies
- `fit_torsion_to_transitions` — fit to observed transition frequencies
- `fit_torsion_joint` — joint fit to torsional levels **and** torsion-averaged rotational constants (A/B/C) simultaneously via a unified Gauss-Newton loop

### Torsion config keys

Add a `torsion_hamiltonian` block to a run config to activate the pipeline:

```yaml
torsion_hamiltonian:
  enabled: true
  F: 5.1753          # internal rotation constant (cm-1)
  rho: 0.0812        # coupling between internal and overall rotation
  n_basis: 12        # Fourier basis truncation (|m| ≤ n_basis)
  potential:
    v0: 185.5        # potential offset (cm-1)
    vcos: {3: -185.5}   # cos(3α) term
  line_list:
    enabled: true
    max_freq_mhz: 50000
    min_line_strength: 1e-6
  fitting:
    enabled: true
    params: [Vcos_3, F]
    targets:         # observed torsional levels
      - {J: 0, K: 0, level_index: 0, energy_cm-1: 0.0}
    targets_rotational:   # joint fit: observed B0 constants
      - {component: B, obs_cm1: 0.8220, sigma_cm1: 0.002}
```

See [`configs/example_methanol_lam.yaml`](configs/example_methanol_lam.yaml) for a complete working example.

## Repository layout

- **`backend/`** — core library (spectral, quantum, optimizer, priors, symmetry).
- **`runs/`** — per-molecule driver scripts (e.g. `run_water.py`, `run_OCS.py`, `run_n.py`, `run_SO2.py`, `run_CO2.py`).
- **`run_molecule.py`** — small CLI that dispatches to a named driver in `runs/`.
- **`run_settings.py`** — optional shared presets for drivers.
- **`requirements.txt`** — Python dependencies.
- **`MATH.typ`** / **`MATH.md`** — mathematical reference (compile Typst to **`MATH.pdf`** for readable typeset math).
- **`Geometric/`**, **`runs/`** outputs, tuning JSON files at repo root as applicable.

## Requirements

- Python 3.10+
- **NumPy**, **SciPy** (see `requirements.txt`)
- **Psi4** — if using `quantum_backend="psi4"` (often via Conda).
- **ORCA** — optional, if using `quantum_backend="orca"`.
- **`scikit-optimize`** — optional, for `backend/bayes_tune.py` (`pip install scikit-optimize`).

Example environment (Unix-like):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` or `Activate.ps1`.

## Running

From the project root, the config-first interface is:

```bash
python -m cli validate configs/example_water_spectral_only.yaml
python -m cli run configs/example_water_spectral_only.yaml
python -m cli run configs/example_water_legacy.json
```

`run` creates a timestamped directory under `runs/` by default, copies the input
config, and writes `report.md`, `exports/residuals.csv`,
`exports/final_geometry.csv`, and diagnostic plots under `plots/`.

The lower-level runner remains available for compatibility:

```bash
python runner/run_from_config.py configs/template.yaml
python runner/run_from_config.py configs/example_water_spectral_only.yaml --no-run-dir
```

You can also use the molecule dispatchers:

```bash
python run_molecule.py water
python run_molecule.py ocs --preset BALANCED
```

or call a driver module directly:

```bash
python -m runs.run_water
python -m runs.run_OCS
```

Drivers typically build isotopologue inputs, generate a starting geometry, run multistart optimization, and print a summary. Defaults and presets can be adjusted in `run_molecule.py`, `run_settings.py`, or each `runs/run_*.py` file.

### ORCA and `run_settings.py`

Drivers read `run_settings.py`, which defaults to `quantum_backend="orca"` and `orca_exe=None`. The optimizer then searches for ORCA in this order: **`orca` on your PATH**, a **full path** if you set one, then an **`orca` or `orca.exe` file in the current working directory** (so you can drop or symlink the binary into the project folder). You can also set the path before running:

```bash
export ORCA_EXE="/full/path/to/orca"
python3 run_molecule.py water
```

On Windows, you can instead set `orca_exe` in `BASE_SETTINGS` to your `orca.exe` path. If you do not have ORCA, either install it, point `ORCA_EXE` at it, or switch a runner to **Psi4** (`quantum_backend="psi4"` in `run_settings.py` and a Conda environment with `psi4` installed), or use spectral-only mode where the script supports it (`USE_QUANTUM_PRIOR = False` in e.g. `runs/run_water.py`).

**Parallel multistart + ORCA:** many licenses allow only one ORCA job at a time. `run_multistart` therefore defaults to **`max_workers=1`** when `quantum_backend="orca"`. If your license allows multiple processes, set **`QUANTIZE_ALLOW_PARALLEL_ORCA=1`** before running to use the preset worker count.

**Paths with spaces:** older ORCA builds can mishandle absolute paths that contain spaces; the driver runs ORCA with `cwd` set to the job directory and passes **`quantize_orca.inp`** by filename only. If you still see startup errors, move the project to a directory whose full path has no spaces.

## Interpreting output

- **Rank** — number of directions retained above the relative singular-value cutoff in the stacked Jacobian SVD.
- **RMS MHz** — root-mean-square residual of rotational constants in MHz (unweighted block).
- **\(\|\Delta x_r\|\)** — norm of the step projected onto the spectral range space.
- **\(\|\Delta x_n\|\)** — norm of the step projected onto the null space.
- **\(\|g_n\|\)** — norm of the gradient projected onto the null space (hybrid mode).
- **\(\|\Delta E\|\)** — magnitude of energy change between iterations (Hartree).

If rank stays low and residuals plateau, add more informative isotopologues and/or check consistency of \(B_0\), \(\alpha\), and uncertainties \(\sigma\).

## Notes

- Run scripts use `if __name__ == "__main__":` and `freeze_support()` for multiprocessing on Windows.
- See **[MATH.typ](MATH.typ)** (and **[MATH.md](MATH.md)**) for every equation and weighting rule implemented in this repository.
