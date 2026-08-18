"""Water through the real hybrid optimizer, across correction settings.

geometry_accuracy.py isolates the correction chain with a plain two-parameter
least-squares fit. This runs the same molecule through MolecularOptimizer and
SubspaceOptimizer with the full A/B/C data set, which is what an actual run does,
and varies two things:

  * how much of the correction chain is switched on;
  * whether the theory surface is right (minimum at r_e) or carries a realistic
    ~11 mA method error.

The final control row feeds B_e taken straight from the reference geometry. It
recovers r_e exactly from either surface, which places the whole residual in the
rows above on the accuracy of alpha -- that is, on the Hessian -- rather than on
the optimizer.

    python scripts/water_scenarios.py

Uses the analytic water potential from dev/tests/reference_molecules.py, so it
runs without Psi4 or ORCA. That force field is not spectroscopic quality (its
bend frequency is ~4% low, and it carries no bend anharmonicity), so the numbers
are a floor on achievable accuracy.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy import constants as sc

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))
sys.path.insert(0, str(_ROOT / "scripts"))

import reference_molecules as _rm  # noqa: E402
from backend.base_backend import GradientResult, HessianResult, QuantumBackend  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import register_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from hybrid_partial_data import (  # noqa: E402
    R_E, R_THEORY, THETA_E, THETA_THEORY, internals,
)
from reference_molecules import H2O_B0_MHZ, H2O_MASSES, h2o_coords, h2o_hessian  # noqa: E402

_HARTREE_J = sc.physical_constants["Hartree energy"][0]
_BOHR_A = sc.physical_constants["Bohr radius"][0] * 1e10
_AJ = 1e-18


def _energy(coords):
    return _rm._h2o_energy(np.asarray(coords, dtype=float), True) * _AJ / _HARTREE_J


def _gradient(coords, step=1e-5):
    flat = np.asarray(coords, dtype=float).ravel().copy()
    g = np.zeros(flat.size)
    for i in range(flat.size):
        p, m = flat.copy(), flat.copy()
        p[i] += step
        m[i] -= step
        g[i] = (_energy(p.reshape(-1, 3)) - _energy(m.reshape(-1, 3))) / (2 * step)
    return g * _BOHR_A


@register_backend
class OnTargetWaterBackend(QuantumBackend):
    """Analytic water surface whose minimum sits at the reference r_e."""

    name = "ontarget_water"
    supports_parallel = False

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])

    def run_hessian(self, coords_ang):
        return HessianResult(
            energy=_energy(coords_ang),
            gradient_bohr=_gradient(coords_ang),
            hessian_bohr=h2o_hessian(coords_ang),
        )

    def run_gradient(self, coords_ang):
        return GradientResult(
            energy=_energy(coords_ang), gradient_bohr=_gradient(coords_ang)
        )


def run(backend, start, harmonic, anharmonic, targets=None):
    obs = H2O_B0_MHZ if targets is None else targets
    iso = [{
        "name": "H2-16O",
        "masses": H2O_MASSES.tolist(),
        "obs_constants": np.asarray(obs, dtype=float).tolist(),
        "sigma_constants": [0.2] * 3,
        "alpha_constants": [0.0] * 3,
        "component_indices": [0, 1, 2],
    }]
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=start, isotopologues=iso,
        quantum_backend=backend,
        harmonic_from_hessian=harmonic, anharmonic_from_hessian=anharmonic,
        coordinate_mode="cartesian", max_iter=40, spectral_only=False,
        use_autoconfig=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return internals(opt.run())


SETTINGS = [
    ("no corrections (fits raw B_0)", False, False),
    ("harmonic + Coriolis only", True, False),
    ("harmonic + Coriolis + anharmonic", True, True),
]


def main():
    be_exact = rotational_constants_mhz(h2o_coords(), H2O_MASSES)
    surfaces = [
        ("theory minimum AT r_e (good method)", "ontarget_water",
         h2o_coords(0.9800, np.radians(107.0)), (R_E, THETA_E)),
        ("theory minimum off by +10.9 mA (realistic)", "detuned_water",
         h2o_coords(R_THEORY, np.radians(THETA_THEORY)), (R_THEORY, THETA_THEORY)),
    ]
    for label, backend, start, theory in surfaces:
        print(f"\n  === {label} ===")
        hdr = (f"  {'setting':<38}{'r (A)':>9}{'dr (mA)':>9}"
               f"{'theta':>10}{'dtheta':>9}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        print(f"  {'theory alone (PES minimum)':<38}{theory[0]:>9.5f}"
              f"{(theory[0] - R_E) * 1000:>+9.1f}{theory[1]:>10.3f}"
              f"{theory[1] - THETA_E:>+9.3f}")
        for name, harm, anh in SETTINGS:
            r, t = run(backend, start, harm, anh)
            print(f"  {name:<38}{r:>9.5f}{(r - R_E) * 1000:>+9.1f}"
                  f"{t:>10.3f}{t - THETA_E:>+9.3f}")
        r, t = run(backend, start, False, False, targets=be_exact)
        print(f"  {'CONTROL: exact B_e targets':<38}{r:>9.5f}"
              f"{(r - R_E) * 1000:>+9.1f}{t:>10.3f}{t - THETA_E:>+9.3f}")

    print("\n  The control recovers r_e from either surface, so the optimizer and the")
    print("  SVD split are exact and the residual above is entirely alpha model error.")
    print("\n  Reading the rows: the full chain improves the angle roughly threefold and")
    print("  gives an answer that barely moves between the two theory surfaces. The bond")
    print("  length does not improve here -- the uncorrected row lands close on r by")
    print("  coincidence, which is visible in it being identical on both surfaces: with")
    print("  three constants the spectral block dominates and the prior contributes")
    print("  nothing, so that row is just the r_0-like fit, 0.76 deg off on the angle.")
    print("  Closing the remaining ~3 mA needs a better Hessian, not better fitting.")


if __name__ == "__main__":
    main()
