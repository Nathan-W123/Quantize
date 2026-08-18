# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

This project estimates molecular structure (bond lengths and angles) from isotopologue rotational constants, including undersaturated cases where spectroscopy alone does not fully constrain the geometry.

## Core idea

- Use observed rotational constants (`A`, `B`, `C`) from one or more isotopologues.
- Map ground-state constants onto equilibrium targets via \(B_e = B_0 + \Delta_\mathrm{vib} + \Delta_\mathrm{elec} + \Delta_\mathrm{BOB}\) (see [Ground state to equilibrium](#ground-state-to-equilibrium-b_0--b_e)).
- Stack spectral Jacobians across isotopologues, apply **SVD** to split **range space** (spectroscopy-sensitive directions) from **null space** (directions invisible to the stacked Jacobian).
- Use electronic-energy **gradient and Hessian** (Psi4 or ORCA) for a damped-Newton step in the null space so the structure is stabilized where data are silent.

Key formulas are documented in module docstrings (see `.github/backend/spectral/`, `.github/backend/kraitchman.py`, and `.github/backend/torsion/`); primary literature references are cited inline (Gordy & Cook 1984 for spectroscopic relations).

## Main modules (`.github/backend/`)

The library lives under `.github/backend/`; `paths.ensure_repo_paths` puts
`.github` on `sys.path` so it imports as `backend.*`.

| Module | Role |
|--------|------|
| [`backend/quantize.py`](.github/backend/quantize.py) | `MolecularOptimizer`: spectral + quantum hybrid loop |
| [`backend/spectral/spectral.py`](.github/backend/spectral/spectral.py) | Inertia tensor, \(A,B,C\), Jacobians, residuals, weighting, optional conformer mixtures |
| [`backend/spectral/SVD.py`](.github/backend/spectral/SVD.py) | `SubspaceOptimizer`: SVD split, range/null steps, joint objective option |
| [`backend/spectral/harmonic_alpha.py`](.github/backend/spectral/harmonic_alpha.py) | Vibration-rotation \(\alpha_r\): harmonic, Coriolis, and cubic anharmonic terms |
| [`backend/spectral/centrifugal_distortion.py`](.github/backend/spectral/centrifugal_distortion.py) | Normal modes, \(\partial B/\partial Q\), \(\tau'\), Watson CD constants |
| [`backend/spectral/correction_models.py`](.github/backend/spectral/correction_models.py) | Vibrational, electronic (g-tensor), and BOB corrections |
| [`backend/spectral/rovib_corrections.py`](.github/backend/spectral/rovib_corrections.py) | Resolves all corrections into \(B_e^{SE}\) targets with provenance |
| [`backend/quantum.py`](.github/backend/quantum.py) | ORCA parsers; Wilson **B**-matrix; primitive internal-coordinate derivatives |
| [`backend/psi4/Psi4.py`](.github/backend/psi4/Psi4.py) | Psi4 energy / gradient / Hessian with unit conversion to Å |
| [`backend/internal/internal_prior.py`](.github/backend/internal/internal_prior.py) | Optional internal-coordinate priors stacked with the spectral block |
| [`backend/conformers/geometryguess.py`](.github/backend/conformers/geometryguess.py) | Template guesses and spring-style relaxation |
| [`backend/multistart.py`](.github/backend/multistart.py) | Parallel multi-start runs and best-run selection |
| [`backend/priors/bayes_tune.py`](.github/backend/priors/bayes_tune.py) | Optional Bayesian hyperparameter search (`scikit-optimize`) |
| [`backend/symmetry.py`](.github/backend/symmetry.py) | Optional point-group projection of steps and coordinates |
| [`backend/autoconfig.py`](.github/backend/autoconfig.py) | Adaptive trust region / damping / weight policy from diagnostics |
| [`backend/kraitchman.py`](.github/backend/kraitchman.py) | Kraitchman single-substitution rs coordinates, planar moments, inertial defects (auto-run; see `exports/kraitchman_rs.csv` and the report section) |

## Ground state to equilibrium (\(B_0 \to B_e\))

Fitting a geometry to observed \(B_0\) directly conflates structure with
zero-point motion. The correction chain maps the observed constants onto
semi-experimental equilibrium targets:

\[ B_e = B_0 + \Delta_\mathrm{vib} + \Delta_\mathrm{elec} + \Delta_\mathrm{BOB} \]

**\(\Delta_\mathrm{vib} = \tfrac12 \sum_r \alpha_r\)**, with \(\alpha_r\) built from three terms
(Mills 1972; Papoušek & Aliev 1982):

| Term | Source | Enabled by |
|------|--------|-----------|
| Harmonic \(\langle Q_r^2\rangle\,\partial^2 B/\partial Q_r^2\) | Cartesian Hessian | `harmonic_from_hessian` |
| Coriolis \(\zeta^{(\xi)}_{rs}\) coupling | Normal-mode eigenvectors | `harmonic_from_hessian` |
| Anharmonic \(\phi_{rrs}\) (cubic) | Finite-difference cubic force field | `anharmonic_from_hessian` |

The anharmonic term is **not** a small refinement — it is usually the largest of
the three and carries the opposite sign to the harmonic term. For CO the
harmonic term alone gives \(\alpha = -0.0103\ \mathrm{cm^{-1}}\) against an observed
\(+0.0175\); adding the cubic term reproduces the Dunham/Pekeris value to 0.1%.
It costs \(6N\) extra Hessian evaluations, so it is opt-in; when it is off, the
reported \(\alpha\) uncertainty is widened to 100%.

**\(\Delta_\mathrm{elec} = -(m_e/m_p)\, g_\alpha B_0\)** requires the rotational
g-tensor via `g_tensor`. Without it the code falls back to a crude \(1/M_\mathrm{total}\)
estimate that is roughly an order of magnitude too small and has the wrong sign
whenever \(g < 0\) (as for OCS), so that path reports 100% uncertainty.

**\(\Delta_\mathrm{BOB}\)** uses per-element Watson u-parameters. The built-ins are
order-of-magnitude estimates; supply `bob_params` for sub-milliångström work.

See [`configs/example_water_semi_experimental.yaml`](configs/example_water_semi_experimental.yaml)
for a complete worked example, and `dev/tests/test_alpha_against_experiment.py`
for the validation suite (closed-form Dunham/Pekeris results, C₂ᵥ symmetry
constraints, and published CO/H₂O constants — no Psi4 or ORCA needed).

> **Known limitation.** The \(\tau' \to\) Watson A-reduction mapping in
> `centrifugal_distortion.py` is unvalidated and does not reproduce published
> constants. `compute_cd_constants` reports 100% uncertainty accordingly, and
> `fit_cd_constants` defaults to off.

## How data and theory share authority

The default `split` objective partitions the parameter space hard: whatever
survives the SVD rank cutoff is handed **entirely** to the spectral data, and the
quantum surface governs only the null space. That works when the retained
directions are well determined — but the cutoff is *relative*
(`sv_threshold × s_max`, with `sv_min_abs` defaulting to 0), so a direction the
data resolves only loosely is still treated as fully constrained and theory gets
no vote in it. Water's bond angle is such a direction, and it comes out worse
than either theory or experiment alone would give.

Two ways to hand authority back:

| Control | Effect |
|---------|--------|
| `optimizer.sv_min_abs` | Absolute floor on the singular value. The Jacobian is σ-weighted, so \(1/s\) is the parameter uncertainty along a direction — the floor means "only trust what the data resolves this well". All-or-nothing per direction. |
| `optimizer.objective_mode: joint` with `optimizer.quantum_prior_sigma_ang` | Solves \((J^TJ + \alpha_q H + \lambda I)\,\Delta p = J^T r - \alpha_q g\), leaving every direction contested and weighted by how well each source knows it. `quantum_prior_sigma_ang` is the displacement over which the quantum surface is trusted, roughly the geometry error of the method, which is what makes \(\alpha_q\) interpretable rather than an arbitrary knob. |

### Which objective to use depends on how much data you have

This is the single most consequential setting, and it flips with the size of the
dataset. Two measured cases:

| Case | Observables | Internal DOF | Best objective |
|------|-------------|--------------|----------------|
| Water, one isotopologue (`scripts/theory_vs_experiment_vs_hybrid.py`) | 3 | 3 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Fluorobenzene, one isotopologue (`scripts/fluorobenzene_vs_published.py`) | 3 | 30 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Fluorobenzene, eight isotopologues (`scripts/fluorobenzene_full_data.py`) | 24 | 30 | `split` (the default) |
| Vinyl / acetyl fluoride, fluoroethane — parent only (`scripts/monofluoro_benchmark.py`) | 3 | 12–18 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Vinyl / acetyl fluoride, fluoroethane — all species (same script) | 18 | 12–18 | `split` (the default) |

With few observables the split partition hands whole directions to data that
barely resolves them, drives the residual below what the physics justifies, and
distorts the structure doing it — on water the angle error trebles, on
fluorobenzene the C–H angles do. A calibrated prior keeps those directions
contested and fixes most of it.

With a full isotopologue set the position reverses. Against the published
fluorobenzene structure, 24 observables give RMS bond errors of 11.4 mÅ from
theory alone, 8.2 mÅ from spectroscopy alone, and **4.0 mÅ** from the split
hybrid — better than either input on bond lengths, angles and the C–F distance
at once. Forcing `quantum_prior_sigma_ang: 0.005` there costs more than half the
gain (8.9 mÅ), because the prior now over-constrains directions the data
determines perfectly well.

Rule of thumb: reach for the calibrated prior when the fit is undersaturated,
and leave the default alone when it is not. `python scripts/tune_quantum_prior.py`
scans the value; 0.005 Å is a sparse-data result, not a validated default.

### Monofluorinated benchmark — and a caveat on the two rows above

`scripts/monofluoro_benchmark.py` runs theory, spectroscopy-only, and both
hybrids over three published structures (vinyl fluoride, acetyl fluoride,
fluoroethane) at RHF/6-31G, at two data levels each. Every rotational constant
is a **measured** literature value transcribed from *NBS Monograph 70,
Microwave Spectral Tables* — nothing is back-calculated from a geometry.
`scripts/check_monofluoro_references.py` validates each species before use, and
`scripts/build_monofluoro_report.py` turns the run's JSON into
[`reports/monofluoro_benchmark_report.pdf`](reports/monofluoro_benchmark_report.pdf).

**The objective-choice rule above does not survive measured data.** The water
and fluorobenzene rows use isotopologue constants *derived* from their reference
structures, which are mutually consistent by construction. That removes the
systematic r_s-vs-r_0 offset which turns out to drive the behaviour. Re-run on
measured constants, `split` wins at rank deficits of 15, 3 and 1 while `joint`
wins at 12, 10 and 3 — no ordering at all. Treat those rows as describing
self-consistent synthetic data, not experiment.

What does hold up:

- **RHF/6-31G has a systematic C–F bias**, +27 to +31 mÅ, same sign in all three
  molecules. A method bias, not scatter, and what the spectral data is good at
  removing: some combination of data and theory beats theory alone on C–F in 6
  of 6 cases.
- **Everything is undersaturated.** Counting constants overstates the
  information badly. Vinyl fluoride's 22 measured constants carry rank **9**
  against 12 internal DOF; acetyl fluoride 30 → rank 14 of 15; fluoroethane
  18 → rank 15 of 18. All published data still leaves every molecule
  underdetermined, so `sv_min_abs` / `objective_mode` matter in every real case.
- **On overall bond error the hybrid is not dependably better than theory** —
  5 of 6 even choosing the objective with hindsight. The binding constraint is
  no longer the objective but the missing vibration-rotation correction: fitting
  r_0 constants uncorrected against an r_s reference leaves a one-signed ~0.5%
  residual that the fit removes by distorting the structure. In vinyl fluoride
  that makes the spectroscopy-only fit *worse* with 22 constants (31.1 mÅ) than
  with 3 (21.0 mÅ). Applying `anharmonic_from_hessian` here is the obvious next
  step.

## Torsion / Large-Amplitude Motion (LAM) pipeline

A self-contained torsion-rotation pipeline handles molecules with an internal methyl (or other Cn) rotor. It uses a RAM-lite (rho-axis method) Hamiltonian in a Fourier basis |m⟩ and is independent of the geometry-inversion loop above.

### Torsion backend modules

| Module | Role |
|--------|------|
| [`backend/torsion/torsion_hamiltonian.py`](.github/backend/torsion/torsion_hamiltonian.py) | `TorsionHamiltonianSpec`, Fourier potential matrix, `solve_ram_lite_levels`, `torsion_probability_density` |
| [`backend/torsion/torsion_rot_hamiltonian.py`](.github/backend/torsion/torsion_rot_hamiltonian.py) | Full J-block torsion-rotation Hamiltonian with centrifugal distortion |
| [`backend/torsion/torsion_symmetry.py`](.github/backend/torsion/torsion_symmetry.py) | C3 block decomposition (A/E1/E2), tunneling splittings, selection rules, nuclear-spin weights |
| [`backend/torsion/torsion_average.py`](.github/backend/torsion/torsion_average.py) | Quantum and Boltzmann torsion-scan averaging of A/B/C constants; rigorous uncertainty propagation |
| [`backend/torsion/torsion_intensities.py`](.github/backend/torsion/torsion_intensities.py) | `⟨ψ\|cos(α)\|ψ⟩` matrix elements, Hönl-London factors, complete line-list generation |
| [`backend/torsion/torsion_fitter.py`](.github/backend/torsion/torsion_fitter.py) | Damped Gauss-Newton fitting to levels, transitions, or joint levels + rotational constants |
| [`backend/torsion/torsion_uncertainty.py`](.github/backend/torsion/torsion_uncertainty.py) | Jacobian, covariance, Fisher information, identifiability |
| [`backend/torsion/torsion_lam_integration.py`](.github/backend/torsion/torsion_lam_integration.py) | LAM correction report with uncertainty propagation into the main spectral fit |
| [`backend/torsion/hindered_rotor.py`](.github/backend/torsion/hindered_rotor.py) | Independent 1D hindered-rotor solver (legacy; used for Boltzmann weight helper only) |

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

- **`.github/backend/`** — core library (spectral, quantum, optimizer, priors, symmetry, torsion).
- **`cli.py`** — main entry point (`python -m cli <command>`).
- **`runner/`** — config loading/validation (`usability.py`), the generic run driver
  (`run_generic.py`), report and export generation (`reporting.py`), presets (`run_settings.py`).
- **`configs/`** — example run configs.
- **`molecule_runners/`** — per-molecule driver scripts.
- **`dev/tests/`** — test suite; **`dev/benchmarks/`** — LAM and conformer benchmark suites.
- **`output/`** — all generated artifacts (gitignored); runs land in `output/runs/`.
- **`requirements.txt`** — Python dependencies.

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
python -m cli validate configs/example_water_semi_experimental.yaml
python -m cli run      configs/example_water_semi_experimental.yaml
python -m cli report   output/runs/<timestamped_run_dir>
```

`run` creates a timestamped directory under `output/runs/` by default, copies the
input config, and writes `report.md`, `exports/residuals.csv`,
`exports/final_geometry.csv`, and diagnostic plots under `plots/`.
`report` rebuilds `report.md`, `report.html`, and plots from an existing run.

Other commands: `lam-scan`, `lam-fit`, `lam-diagnose`, `uncertainty`, `benchmark`.
Run `python -m cli --help` for the full list.

Ready-to-run config examples:

- `configs/example_water_semi_experimental.yaml` — full \(B_0 \to B_e\) correction chain
- `configs/example_water.yaml`
- `configs/example_OCS.yaml`
- `configs/example_CO2.yaml`
- `configs/example_SO2.yaml`
- `configs/example_formaldehyde.yaml`
- `configs/example_propyne.yaml`
- `configs/example_acetaldehyde.yaml`
- `configs/example_methanol_lam.yaml`
- `configs/example_acetaldehyde_lam.yaml`
- `configs/benzene.yaml`, `configs/fluorobenzene.yaml`

### Bayesian / Bootstrap Uncertainty (v2)

Run uncertainty with the greenfield v2 engine:

```bash
python -m cli uncertainty configs/example_water_semi_experimental.yaml \
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
python runner/run_from_config.py configs/example_water_semi_experimental.yaml
python runner/run_from_config.py configs/example_water.yaml --no-run-dir
```

You can also call a molecule driver module directly:

```bash
python -m molecule_runners.run_water
```

Drivers build isotopologue inputs, generate a starting geometry, run multistart
optimization, and print a summary. Presets live in `runner/run_settings.py`.

### ORCA and `runner/run_settings.py`

Drivers read `runner/run_settings.py`, which defaults to `quantum_backend="orca"` and `orca_exe=None`. The optimizer then searches for ORCA in this order: **`orca` on your PATH**, a **full path** if you set one, then an **`orca` or `orca.exe` file in the current working directory** (so you can drop or symlink the binary into the project folder). You can also set the path before running:

```bash
export ORCA_EXE="/full/path/to/orca"
python -m cli run configs/example_water.yaml
```

On Windows, you can instead set `orca_exe` in `BASE_SETTINGS` to your `orca.exe` path. If you do not have ORCA, either install it, point `ORCA_EXE` at it, or switch to **Psi4** (`quantum.backend: psi4` in the config and a Conda environment with `psi4` installed), or set `quantum.backend: none` for spectral-only mode.

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
- Equations and weighting rules are documented in the docstrings of the modules that implement them.
