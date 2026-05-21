# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

This project estimates molecular structure (bond lengths and angles) from isotopologue rotational constants, including undersaturated cases where spectroscopy alone does not fully constrain the geometry.

## Core idea

- Use observed rotational constants (`A`, `B`, `C`) from one or more isotopologues.
- Map ground-state constants toward equilibrium targets:
  - **`harmonic_from_hessian: true` (recommended):** \(B_e \approx B_0 + \delta_\text{cent} + \delta_\text{Cor} + \delta_\text{elec}\) — centrifugal, Coriolis, and electronic corrections computed self-consistently from the ORCA Hessian (Watson 1968). Refreshed on every Hessian update.
  - **Legacy / static alpha:** \(B_e \approx B_0 + \tfrac{1}{2}\alpha\) using user-supplied `alpha_mhz` values (no Coriolis term; less accurate for asymmetric tops).
- Stack spectral Jacobians across isotopologues, apply **SVD** to split **range space** (spectroscopy-sensitive directions) from **null space** (directions invisible to the stacked Jacobian).
- Use electronic-energy **gradient and Hessian** (Psi4 or ORCA) for a damped-Newton step in the null space so the structure is stabilized where data are silent.

Full notation, derivations, and formulas used in code live in **[MATH.typ](MATH.typ)** (Typst — best PDF layout). A Markdown mirror is **[MATH.md](MATH.md)**. Compile with `typst compile MATH.typ` (install [Typst](https://typst.app)); this produces **`MATH.pdf`** in the repo root when built locally.

## Main modules (`.github/backend/`)

| Module | Role |
|--------|------|
| [`.github/backend/quantize.py`](.github/backend/quantize.py) | `MolecularOptimizer`: spectral + quantum hybrid loop |
| [`.github/backend/spectral.py`](.github/backend/spectral.py) | Inertia tensor, \(A,B,C\), Jacobians, residuals, weighting, optional conformer mixtures |
| [`.github/backend/SVD.py`](.github/backend/SVD.py) | `SubspaceOptimizer`: SVD split, range/null steps, joint objective option |
| [`.github/backend/quantum.py`](.github/backend/quantum.py) | ORCA parsers; Wilson **B**-matrix; primitive internal-coordinate derivatives |
| [`.github/backend/Psi4.py`](.github/backend/Psi4.py) | Psi4 energy / gradient / Hessian with unit conversion to Å |
| [`.github/backend/internal_prior.py`](.github/backend/internal_prior.py) | Optional internal-coordinate priors stacked with the spectral block |
| [`.github/backend/geometryguess.py`](.github/backend/geometryguess.py) | Template guesses and spring-style relaxation |
| [`.github/backend/multistart.py`](.github/backend/multistart.py) | Parallel multi-start runs and best-run selection |
| [`.github/backend/bayes_tune.py`](.github/backend/bayes_tune.py) | Optional Bayesian hyperparameter search (`scikit-optimize`) |
| [`.github/backend/symmetry.py`](.github/backend/symmetry.py) | Optional point-group projection of steps and coordinates |
| [`.github/backend/autoconfig.py`](.github/backend/autoconfig.py) | Adaptive trust region / damping / weight policy from diagnostics |
| [`.github/backend/autoconfig_bases.py`](.github/backend/autoconfig_bases.py) | Problem-shape heuristics for optimizer base hyperparameters at job start |

## Torsion / Large-Amplitude Motion (LAM) pipeline

A self-contained torsion-rotation pipeline handles molecules with an internal methyl (or other Cn) rotor. It uses a RAM-lite (rho-axis method) Hamiltonian in a Fourier basis |m⟩ and is independent of the geometry-inversion loop above.

### Torsion backend modules

| Module | Role |
|--------|------|
| [`.github/backend/torsion_hamiltonian.py`](.github/backend/torsion_hamiltonian.py) | `TorsionHamiltonianSpec`, Fourier potential matrix, `solve_ram_lite_levels`, `torsion_probability_density` |
| [`.github/backend/torsion_rot_hamiltonian.py`](.github/backend/torsion_rot_hamiltonian.py) | Full J-block torsion-rotation Hamiltonian with centrifugal distortion |
| [`.github/backend/torsion_symmetry.py`](.github/backend/torsion_symmetry.py) | C3 block decomposition (A/E1/E2), tunneling splittings, selection rules, nuclear-spin weights |
| [`.github/backend/torsion_average.py`](.github/backend/torsion_average.py) | Quantum and Boltzmann torsion-scan averaging of A/B/C constants; rigorous uncertainty propagation |
| [`.github/backend/torsion_intensities.py`](.github/backend/torsion_intensities.py) | `⟨ψ\|cos(α)\|ψ⟩` matrix elements, Hönl-London factors, complete line-list generation |
| [`.github/backend/torsion_fitter.py`](.github/backend/torsion_fitter.py) | Damped Gauss-Newton fitting to levels, transitions, or joint levels + rotational constants |
| [`.github/backend/torsion_uncertainty.py`](.github/backend/torsion_uncertainty.py) | Jacobian, covariance, Fisher information, identifiability |
| [`.github/backend/torsion_lam_integration.py`](.github/backend/torsion_lam_integration.py) | LAM correction report with uncertainty propagation into the main spectral fit |
| [`.github/backend/hindered_rotor.py`](.github/backend/hindered_rotor.py) | Independent 1D hindered-rotor solver (legacy; used for Boltzmann weight helper only) |

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

- **`.github/backend/`** — core library (spectral, quantum, optimizer, priors, symmetry). Import as `backend.*` with `PYTHONPATH=.github`, or run entry scripts (`cli.py`, `runner/run_from_config.py`) which call `paths.ensure_repo_paths`.
- **`paths.py`** — adds `.github` and the repo root to `sys.path`; defines `output/` layout (`output/runs`, `output/results`, `output/trials`).
- **`output/`** — gitignored local artifacts: managed run reports (`output/runs/`), benchmark JSON (`output/results/benchmarks/`), QM scratch (`output/trials/`).
- **`runner/`** — config-driven orchestration: `run_from_config.py`, `run_generic.py`, `usability.py`, `run_settings.py`.
- **`scripts/`** — optional utilities (e.g. `scripts/bayes_tune.py` for hyperparameter search).
- **`configs/`** — YAML inputs for `cli run` / `runner/run_from_config.py`.
- **`configs/`** — YAML config files for the generic runner.
- **`requirements.txt`** — Python dependencies.
- **`MATH.typ`** / **`MATH.md`** — mathematical reference (compile Typst to **`MATH.pdf`** for readable typeset math).

## Requirements

- Python 3.10+
- **NumPy**, **SciPy** (see `requirements.txt`)
- **Psi4** — if using `quantum_backend="psi4"` (often via Conda).
- **ORCA** — optional, if using `quantum_backend="orca"`.
- **`scikit-optimize`** — optional, for `.github/backend/bayes_tune.py` (`pip install scikit-optimize`).

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
python -m cli run configs/example_CO2.yaml
python -m cli report output/runs/<timestamped-run-dir>
```

`run` creates a timestamped directory under `output/runs/` by default (output only — not a source directory), copies the input
config, and writes `report.md`, `exports/residuals.csv`,
`exports/final_geometry.csv`, and diagnostic plots under `plots/`.
`report` rebuilds `report.md`, `report.html`, and plots from an existing run.

Ready-to-run config examples:

- `configs/example_water_spectral_only.yaml`
- `configs/example_OCS.yaml`
- `configs/example_CO2.yaml`
- `configs/example_SO2.yaml`
- `configs/example_formaldehyde.yaml`
- `configs/example_methanol.yaml`
- `configs/example_methanol_lam.yaml`
- `configs/example_acetaldehyde_lam.yaml`
- `configs/example_partial_isotopologue.yaml`

### Lightweight GUI

You can use a simple Streamlit dashboard to validate configs, launch runs,
and inspect run reports/plots:

```bash
streamlit run streamlit_app.py
```

### Bayesian / Bootstrap Uncertainty (v2)

Run uncertainty with the greenfield v2 engine:

```bash
python -m cli uncertainty configs/example_water_spectral_only.yaml \
  --uncertainty-engine v2 \
  --mode both \
  --samples 20 \
  --mcmc-steps 5000 \
  --burn-in 1000 \
  --chains 4 \
  --seed 42
```

Outputs are written under the run workdir, including:

- `uncertainty_v2/posterior_laplace_summary.json`
- `bootstrap_v2/bootstrap_summary.json` (unless `--no-bootstrap-persist`)
- `mcmc_v2/mcmc_summary.json` (unless disabled)
- `uncertainty_v2/uncertainty_run_summary.json` (consolidated run bundle)
- `plots/uncertainty/*.png` diagnostic plots

The lower-level runner remains available for compatibility:

```bash
python runner/run_from_config.py configs/template.yaml
python runner/run_from_config.py configs/example_water_spectral_only.yaml --no-run-dir
```

Presets (`FAST_DEBUG`, `BALANCED`, `STRICT`) and quantum settings are set in each YAML under `preset:` and `quantum:`.

## Conformers

Quantize can now build and score conformer ensembles during a run instead of relying only on user-supplied fixed mixtures. The `conformers` block supports:

- explicit conformers with full coordinates or offsets from the input geometry
- automatic conformer generation from detected rotatable bonds
- lightweight geometry optimization for generated candidates
- RMSD / rotational-constant pruning before ensemble weighting
- `fixed`, `uniform`, or Boltzmann weighting modes

Example:

```yaml
conformers:
  enabled: true
  weight_mode: boltzmann
  temperature_k: 298.15
  generation:
    enabled: true
    angle_grid_deg: [60, 180, 300]
    max_rotatable_bonds: 2
    max_conformers: 12
    optimize: true
    optimization_steps: 150
    prune_rmsd_ang: 0.08
    prune_constants_mhz: 1.0
```

You can also mix in explicit entries:

```yaml
conformers:
  enabled: true
  weight_mode: fixed
  entries:
    - name: anti
      weight: 0.7
      offset_angstrom:
        - [0.0000, 0.0000, 0.0000]
        - [0.0000, 0.0000, 0.0000]
        - [0.0000, 0.0000, 0.0000]
        - [0.0000, 0.0000, 0.0000]
    - name: gauche
      weight: 0.3
      offset_angstrom:
        - [0.0000, 0.0000, 0.0000]
        - [0.0000, 0.0000, 0.0000]
        - [0.0000, 0.0000, 0.0000]
        - [0.1200, -0.2800, 0.3100]
```

The selected run exports `exports/conformer_summary.json` and includes a conformer section in `report.md` with generation diagnostics, energies, and final weights. Presets that enable conformer mixtures will automatically use the assembled ensemble when the block is present.

## Benchmarks

The benchmark CLI now covers both torsion/LAM and conformer-sensitive validation suites with checked-in regression baselines:

```bash
python cli.py benchmark lam --enforce-thresholds
python cli.py benchmark conformer --enforce-thresholds
```

The LAM suite runs a broader validation bundle covering:

- methanol and acetaldehyde reference molecules
- physical-limit checks such as the free-rotor and high-barrier limits
- expected failure modes such as low `V/F`, large `rho`, large centrifugal distortion, and invalid basis/sign configurations

The conformer suite tracks:

- automatic conformer generation and pruning behavior on a rotatable-chain reference
- Boltzmann weighting drift for energy-sensitive ensembles
- spectral averaging behavior for weighted conformer mixtures

Artifacts are written to:

- `output/results/benchmarks/lam-latest.json` for the latest machine-readable result
- `output/results/benchmarks/conformer-latest.json` for the latest conformer benchmark result
- `output/results/benchmarks/history/` for timestamped snapshots that make drift easy to inspect over time

The checked-in threshold baselines live at `dev/benchmarks/baselines/lam.json` and `dev/benchmarks/baselines/conformer.json`. Update them deliberately when a change is intended to move the benchmark reference values.

For local verification of the benchmark plumbing itself, run:

```bash
python -m pytest dev/tests/test_benchmark_suite.py dev/tests/test_conformer_benchmark_suite.py -q
```

GitHub Actions runs both suites in `.github/workflows/benchmarks.yml`, enforces the baseline thresholds, and uploads the JSON artifacts from each CI run so benchmark drift can be tracked across time.

### Optimizer autoconfig (runtime)

Hybrid runs enable **AutoConfig** by default: trust region, LM damping, spectral weights, and (when enabled) `sv_threshold` / `alpha_quantum` adapt from Jacobian rank, conditioning, and residuals each iteration. At startup, **heuristic bases** rescale those knobs from atom count and spectral row count (no molecule-name registry).

Optional YAML blocks:

```yaml
autoconfig:
  enabled: true              # default
  heuristic_bases: true      # rescale bases from problem shape
  tune_sv_threshold: true
  tune_alpha_quantum: true
  smoothing: 0.4
  update_every: 1

optimizer:                   # pin values before heuristic rescaling
  trust_radius: 0.005
  sv_threshold: 1.0e-5
  alpha_quantum: 0.3
```

Set `autoconfig.heuristic_bases: false` and use `optimizer:` when you want fixed hyperparameters. For one-off tuning per species, see `scripts/bayes_tune.py` (offline Bayesian optimization).

### ORCA and quantum settings

YAML configs set the backend under `quantum.backend` (`orca`, `psi4`, or `none` for spectral-only). The optimizer searches for ORCA in this order: **`orca` on your PATH**, a **full path** if you set one, then an **`orca` or `orca.exe` file in the current working directory**. You can also set the path before running:

```bash
export ORCA_EXE="/full/path/to/orca"
python -m cli run configs/example_OCS.yaml
```

If you do not have ORCA, point `ORCA_EXE` at it, set `quantum.executable` in YAML, use **Psi4** (`quantum.backend: psi4`), or **spectral-only** (`quantum.backend: none`).

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
