"""Run the whole torsion chain on a real PES and print what it gives.

    python scripts/verify_torsion_chain.py [method=hf] [basis=6-31g] [points=12]

Checks the one thing the unit tests cannot: that geometry -> relaxed scan ->
TorsionHamiltonianSpec -> averaged constants -> TorsionReplacement runs
end to end against a quantum backend and lands on physically sensible
numbers. The unit tests drive an analytic model potential, which pins the
mechanism but says nothing about whether the pieces fit together on a real
surface.

Methanol, because it is the one molecule whose F, rho and barrier are all
known independently, so every number printed has something to be compared
against. Too slow for the default test suite (about three minutes at
HF/6-31G), hence a script; the measurement it produced is recorded in
backend/torsion/torsion_scan_qm.py so a later change can be compared without
re-running it.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / ".github"), str(_ROOT), str(_ROOT / "dev")]

from backend.torsion.torsion_lam_integration import torsion_replacement_from_scan  # noqa: E402
from backend.torsion.torsion_scan_qm import (  # noqa: E402
    relaxed_torsion_scan,
    spec_from_scan,
)

#: Methanol reference values, independent of this code.
#: F and rho: the repository's own fixtures, from the microwave literature.
#: Barrier: ~373 cm^-1. A/B/C: 4.25 / 0.823 / 0.793 cm^-1.
_REFERENCE = {"F_cm": 27.6, "rho": 0.81, "barrier_cm": 373.0,
              "A": 4.25, "B": 0.823, "C": 0.793}

_ELEMENTS = ["C", "O", "H", "H", "H", "H"]
_MASSES = np.array([12.0, 15.9949146, 1.0078250, 1.0078250, 1.0078250, 1.0078250])
_DIHEDRAL = (5, 1, 0, 2)


def _start_geometry(r_co=1.4246, r_oh=0.9451, r_ch=1.0936,
                    a_hco=108.0, a_coh=108.5):
    a_hco, a_coh = np.deg2rad(a_hco), np.deg2rad(a_coh)
    rows = [np.zeros(3), np.array([0.0, 0.0, r_co])]
    for k in range(3):
        phi = np.deg2rad(120.0 * k)
        rows.append(np.array([r_ch * np.sin(a_hco) * np.cos(phi),
                              r_ch * np.sin(a_hco) * np.sin(phi),
                              r_ch * np.cos(a_hco)]))
    rows.append(rows[1] + np.array([r_oh * np.sin(a_coh), 0.0,
                                    r_oh * np.cos(a_coh)]))
    return np.vstack(rows)


def _row(label, got, want, unit=""):
    err = 100.0 * (got / want - 1.0) if want else float("nan")
    print(f"  {label:<22}{got:>11.4f}{want:>11.4f}{err:>+9.1f}%  {unit}")


def main() -> None:
    method, basis, points = "hf", "6-31g", 12
    for tok in sys.argv[1:]:
        if tok.startswith("method="):
            method = tok.split("=", 1)[1]
        elif tok.startswith("basis="):
            basis = tok.split("=", 1)[1]
        elif tok.startswith("points="):
            points = int(tok.split("=", 1)[1])

    from pyscf_backend import PySCFBackend
    backend = PySCFBackend(elems=_ELEMENTS, method=method, basis=basis)

    print(f"\nmethanol at {method}/{basis}, {points}-point relaxed scan\n")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(_start_geometry())
    print(f"  geometry optimised in {time.time() - t0:.1f}s")

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        scan, info = relaxed_torsion_scan(backend._energy_gradient, coords,
                                          _ELEMENTS, _DIHEDRAL,
                                          n_points=points)
    print(f"  scan completed in {time.time() - t0:.1f}s")
    print(f"  all points converged  : {info['all_converged']}")
    print(f"  max dihedral error    : {info['max_dihedral_error_deg']:.2e} deg")
    print(f"  rotation sign measured: {info['rotation_sign']:+.0f}")

    spec, sinfo = spec_from_scan(coords, _MASSES, _ELEMENTS, scan,
                                 symmetry_number=3, n_harmonics=3)
    print(f"  harmonics used/asked  : {sinfo['n_harmonics_used']}"
          f"/{sinfo['n_harmonics_requested']}"
          f" (grid allows {sinfo['n_harmonics_max_for_grid']})")

    print(f"\n  {'quantity':<22}{'computed':>11}{'reference':>11}{'error':>10}")
    _row("barrier", info["barrier_cm"], _REFERENCE["barrier_cm"], "cm-1")
    _row("F", spec.F, _REFERENCE["F_cm"], "cm-1")
    _row("rho", spec.rho, _REFERENCE["rho"], "")
    _row("A", spec.A, _REFERENCE["A"], "cm-1")
    _row("B", spec.B, _REFERENCE["B"], "cm-1")
    _row("C", spec.C, _REFERENCE["C"], "cm-1")

    rep, _ = torsion_replacement_from_scan(
        _ELEMENTS, scan, spec, coords, _MASSES, symmetry_number=3,
        mode_indices=(0,), freq_cm=300.0)
    print("\n  torsional alpha replacing the perturbative term:")
    for k in ("A", "B", "C"):
        print(f"    alpha_{k} = {rep.alpha_mhz.get(k, 0.0):+12.3f}"
              f"  +/- {rep.sigma_mhz.get(k, 0.0):9.3f} MHz")

    print("\n  How to read the errors. rho is a RATIO of moments and so scale\n"
          "  free; F carries 1/I_red and picks up a size error twice over. A\n"
          "  level of theory whose A is x% high therefore gets rho right and F\n"
          "  wrong by about 2x%. Barriers at HF with a small basis run high.\n")


if __name__ == "__main__":
    main()
