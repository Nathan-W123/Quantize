# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

This project estimates molecular structure (bond lengths/angles) from isotopologue rotational constants, including undersaturated cases where spectroscopy alone does not fully constrain geometry.

## Core Idea

- Use observed rotational constants (`A/B/C`) from one or more isotopologues.
- Convert ground-state constants to equilibrium targets using:
  - `Be = B0 + 0.5 * alpha`
- Build a spectral Jacobian and apply SVD to split:
  - **Range space**: directions constrained by spectroscopy
  - **Null space**: unconstrained directions filled by a quantum prior
- Use quantum energy gradient/Hessian (ORCA or Psi4) to stabilize null-space motion.

## Main Files

- `quantize.py` - master hybrid optimizer (`MolecularOptimizer`)
- `spectral.py` - rotational constants, Jacobians, residuals, spectral sanitization
- `SVD.py` - SVD subspace step logic and trust-region behavior
- `quantum.py` - ORCA parsing + Wilson B-matrix utilities
- `Psi4.py` - Psi4 backend integration + derivative caching
- `geometryguess.py` - initial geometry guess generation
- `multistart.py` - parallel multi-start orchestration and best-run selection
- `run_water.py` - water driver (currently with detailed diagnostics)
- `run_OCS.py` - OCS driver
- `run_n.py` - larger-molecule demo driver

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Psi4 (if using `quantum_backend="psi4"`)
- ORCA (optional, if using `quantum_backend="orca"`)

Example environment setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy scipy
```

Psi4 is commonly installed via Conda:

```powershell
conda create -n psi4env python=3.11 -y
conda activate psi4env
conda install -c psi4 psi4 -y
pip install numpy scipy
```

## Running

From the project root:

```powershell
python run_water.py
python run_OCS.py
python run_n.py
```

Each run script:

- builds isotopologue input data (masses, `B0`, `sigma`, `alpha`)
- generates a starting geometry
- launches parallel multistart runs
- selects a best result with spectral-gated energy criteria
- prints geometry and fit summary

## Interpreting Output

- `Rank`: number of spectrally constrained directions from Jacobian SVD
- `RMS MHz`: unweighted spectral residual floor
- `|Δx_r|`: step size in spectroscopy-constrained subspace
- `|Δx_n|`: step size in quantum-filled null-space
- `|g_n|`: projected null-space gradient magnitude
- `|ΔE|`: iteration-to-iteration energy change (Hartree)

If rank is low and residual floor remains high, add more informative isotopologues and/or verify `B0/alpha/sigma` consistency.

## Notes

- For Windows multiprocessing, run scripts already use `if __name__ == "__main__":` with `freeze_support()`.
- `run_water.py` currently includes pre-run debug prints for `B0`, `alpha`, `Be`, and sanitizer checks.

