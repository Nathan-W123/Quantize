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

From the project root, either use the dispatcher:

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
