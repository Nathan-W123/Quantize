"""
Bayesian hyperparameter search for MolecularOptimizer (OCS benchmark).

Install Python deps:
  python -m pip install -r requirements.txt

Psi4 is not distributed via pip on Windows; use ``--backend orca`` (default if ORCA path exists)
or install Psi4 via conda and pass ``--backend psi4``.

Examples:
  python run_bayes_tune.py --backend orca --orca-exe "C:\\ORCA_6.1.1\\orca.exe" --n-calls 15
  python run_bayes_tune.py --backend spectral-only --n-calls 20   # fast, no QM
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import freeze_support

import numpy as np

from backend.bayes_tune import tune_molecular_optimizer
from backend.geometryguess import guess_linear_triatomic

_DEFAULT_ORCA = os.environ.get("ORCA_EXE", r"C:\ORCA_6.1.1\orca.exe")

# ── Same isotopologue model as run_OCS.py ────────────────────────────────────
m_O16 = 15.99491
m_C12 = 12.00000
m_S32 = 31.97207
m_S34 = 33.96787

component_idx = [1]
alpha_table = {
    "16O12C32S": np.array([25.24]),
    "16O12C34S": np.array([24.32]),
}
sigma_table = {
    "16O12C32S": np.array([0.0001]),
    "16O12C34S": np.array([0.0002]),
}
obs_b0_values = {
    "16O12C32S": np.array([6081.4921]),
    "16O12C34S": np.array([5932.8159]),
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

isotopologues = [
    {
        "masses": [m_O16, m_C12, m_S32],
        "component_indices": component_idx,
        "obs_constants": obs_b0_values["16O12C32S"].tolist(),
        "sigma_constants": sigma_table["16O12C32S"].tolist(),
        "alpha_constants": alpha_table["16O12C32S"].tolist(),
    },
    {
        "masses": [m_O16, m_C12, m_S34],
        "component_indices": component_idx,
        "obs_constants": obs_b0_values["16O12C34S"].tolist(),
        "sigma_constants": sigma_table["16O12C34S"].tolist(),
        "alpha_constants": alpha_table["16O12C34S"].tolist(),
    },
]


def _base_optimizer_kwargs(
    workdir: str,
    max_iter: int,
    backend: str,
    orca_executable: str | None,
) -> dict:
    backend = backend.strip().lower()
    if backend == "spectral-only":
        return dict(
            quantum_backend="psi4",
            spectral_only=True,
            max_iter=max_iter,
            conv_freq=150.0,
            conv_energy=1e-8,
            conv_step_range=1e-6,
            conv_step_null=5e-2,
            conv_grad_null=1e-5,
            null_trust_radius=0.025,
            objective_mode="split",
            robust_loss="none",
            workdir=workdir,
        )
    if backend == "orca":
        return dict(
            quantum_backend="orca",
            spectral_only=False,
            orca_executable=orca_executable,
            method_preset="fast",
            max_iter=max_iter,
            conv_freq=150.0,
            conv_energy=1e-8,
            conv_step_range=1e-6,
            conv_step_null=5e-2,
            conv_grad_null=1e-5,
            null_trust_radius=0.025,
            objective_mode="split",
            robust_loss="none",
            use_orca_rovib=False,
            workdir=workdir,
        )
    if backend == "psi4":
        return dict(
            quantum_backend="psi4",
            spectral_only=False,
            max_iter=max_iter,
            conv_freq=150.0,
            conv_energy=1e-8,
            conv_step_range=1e-6,
            conv_step_null=5e-2,
            conv_grad_null=1e-5,
            null_trust_radius=0.025,
            objective_mode="split",
            robust_loss="none",
            psi4_method="wB97X-D",
            psi4_basis="def2-TZVPP",
            psi4_memory="2 GB",
            psi4_num_threads=1,
            workdir=workdir,
        )
    raise ValueError(f"Unknown backend {backend!r}; use orca, psi4, or spectral-only")


def main():
    parser = argparse.ArgumentParser(description="Bayesian tune MolecularOptimizer hyperparameters (OCS).")
    parser.add_argument(
        "--backend",
        choices=("orca", "psi4", "spectral-only"),
        default=None,
        help="Quantum backend. Default: orca if ORCA_EXE/default path exists, else spectral-only.",
    )
    parser.add_argument("--orca-exe", type=str, default=_DEFAULT_ORCA, help="Path to orca.exe")
    parser.add_argument("--n-calls", type=int, default=15, help="Number of BO evaluations.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=150,
        help="MolecularOptimizer max_iter per BO trial.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--workdir", type=str, default="bayes_tune_ocs")
    parser.add_argument("--out-json", type=str, default="bayes_tune_best_params_small_molecules.json")
    args = parser.parse_args()

    backend = args.backend
    if backend is None:
        backend = "orca" if args.orca_exe and os.path.isfile(args.orca_exe) else "spectral-only"

    base = _base_optimizer_kwargs(args.workdir, args.max_iter, backend, args.orca_exe)

    print(f"Backend: {backend}")
    print(f"Bayesian optimization: {args.n_calls} calls, workdir root={args.workdir!r}")
    if backend == "orca":
        print(f"ORCA executable: {args.orca_exe!r}")
    print("Tuned: trust_radius, lambda_damp, sv_threshold, alpha_quantum, spectral_delta, hess_recalc_every")

    result, best_params = tune_molecular_optimizer(
        coords=coords,
        elems=elems,
        isotopologues=isotopologues,
        base_kwargs=base,
        n_calls=args.n_calls,
        random_state=args.random_state,
        verbose=True,
        objective_metric="freq_rms",
    )

    save_blob = {
        "backend": backend,
        "orca_executable": args.orca_exe if backend == "orca" else None,
        "best_tuned_params": best_params,
        "best_objective_freq_rms": float(result.fun),
        "full_kwargs_for_molecular_optimizer": {**base, **best_params},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(save_blob, f, indent=2)
    print("\n" + "=" * 60)
    print("Best hyperparameters:", best_params)
    print(f"Best objective (freq_rms): {result.fun:.6f}")
    print(f"Saved merged settings to {args.out_json!r}")
    print("=" * 60)
    if float(result.fun) >= 999999:
        print(
            "\nNote: objective stayed at the failure penalty — check ORCA/Psi4 paths, disk space, and bayes_tune_ocs logs."
        )


if __name__ == "__main__":
    freeze_support()
    main()
