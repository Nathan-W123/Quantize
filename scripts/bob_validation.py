"""Does the data support the BOB u-parameters the engine assumes?

The Born-Oppenheimer breakdown correction scales a rotational constant by
sum_a (m_e / M_a) * u_a. The u-parameters are built in as literature
order-of-magnitude estimates carrying 100% sigma, and nothing has ever checked
them against a measurement -- which for a correction applied to every run is
the wrong state to leave them in.

They are checkable without new literature, because BOB is exactly the residual
ISOTOPE dependence that survives the vibrational correction. In the
Born-Oppenheimer limit every isotopologue of a molecule is the same structure
with different masses, so after B0 -> Be they must all agree on one geometry.
Whatever disagreement is left is what u absorbs. So: scan u_H, refit, and see
which value the data prefers.

Hydrogen only. The correction goes as m_e/M_a, so H contributes 12 times what
carbon does and 16 times what oxygen does; a scan over the heavy-atom u values
would be measuring noise. Molecules are chosen for having several H-substituted
isotopologues, since with one or two the structure and u are not separable.

Two things are reported, and they answer different questions:

  chi2/dof    internal. Does freeing u improve the fit's self-consistency? This
              needs no reference structure, so it is the honest test and it is
              available for any molecule.
  bond error  external. Does the preferred u also give a better structure? A u
              that improves chi2 while moving the structure away from the known
              answer is absorbing something that is not BOB.

Run:  python scripts/bob_validation.py [method=hf] [basis=6-31g]

RESULT at HF/6-31G, scanning u_H over -0.015 to +0.045 -- a range spanning
zero and three times the built-in value:

  molecule           best chi2 u_H   chi2 spread   bond spread   best bond u_H
  acetyl fluoride         +0.0150        19.4%         0.00%        +0.0075
  vinyl fluoride          +0.0150         0.03%        0.00%        +0.0300
  fluoroethane            -0.0150        13.1%         0.00%        +0.0150
  fluoroacetylene         +0.0450         6.2%         0.00%        -0.0150

THE STRUCTURE DOES NOT MOVE. Not "moves a little": the bond error is identical
to three decimals at every u_H on every molecule, a 0.00% spread over a
fourfold change in the parameter. Whatever the right u_H is, it does not reach
the thing this engine exists to produce. That is consistent with the earlier
leave-one-out measurement, where switching BOB on moved all nine reference
molecules by 0.000 mA, and it is the headline: BOB is below the structural
noise floor, full stop.

THE DATA DOES NOT DETERMINE u_H EITHER. Only acetyl fluoride has a real
interior minimum, and it falls exactly on the built-in 0.015 -- encouraging,
and one molecule. Vinyl fluoride's chi2 varies by 0.03% across the whole scan,
so its "minimum" is numerical noise. Fluoroethane and fluoroacetylene are
monotonic across the range and run to opposite boundaries, which means the fit
is not finding a value, it is being pushed to wherever the scan stops.

So the built-in u-parameters are not refuted and not confirmed. Their 100%
sigma_u is the honest description of what is known about them, and a run is
entitled to ignore them.

A corollary worth recording, because it closes a task rather than opening one:
adding BOB A-axis parameters for heavy atoms is not worth doing. The
correction scales as m_e/M_a, so hydrogen contributes twelve times what carbon
does and sixteen times what oxygen does. Hydrogen's contribution is measured
here to be structurally undetectable; the heavy atoms' cannot be larger.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from backend.spectral.correction_models import get_builtin_bob_params  # noqa: E402
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from dev.monofluoro_references import (  # noqa: E402
    ACETYL_FLUORIDE,
    FLUOROACETYLENE,
    FLUOROETHANE,
    VINYL_FLUORIDE,
)
from scripts.mixed_estimation_baseline import (  # noqa: E402
    _SIGMA_X_BY_LEVEL,
    corrected_targets,
    rms_bond_error,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "hf", "6-31g"
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

#: Molecules with enough H-substituted isotopologues to separate u from the
#: structure. Below about four species the two are not distinguishable.
POOL = [ACETYL_FLUORIDE, VINYL_FLUORIDE, FLUOROETHANE, FLUOROACETYLENE]

#: The built-in value this is testing, and the range to scan around it. The
#: built-in carries 100% sigma, so a scan has to cover zero and twice it to say
#: anything about whether the data distinguishes them.
U_BUILTIN = 0.015
U_SCAN = [0.0, 0.0075, 0.015, 0.0225, 0.03, 0.045, -0.015]


def bob_with_u_h(elems, u_h: float) -> dict:
    """Built-in parameters with hydrogen's u replaced on every axis."""
    params = get_builtin_bob_params(list(elems), None)
    out = {}
    for el, axes in params.items():
        if str(el).upper() in ("H", "D", "T"):
            out[el] = {ax: {"u": float(u_h), "sigma_u": abs(float(u_h)) or 0.015}
                       for ax in axes}
        else:
            out[el] = axes
    return out


def fit_once(mol, isos, prior, ctbl, sigma_x, bob_params):
    """One hybrid fit, returning (coords, chi2_per_dof)."""
    from backend.quantize import MolecularOptimizer

    targets = corrected_targets(mol, isos, ctbl, bob_params=bob_params,
                                coords_ang=prior)
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(prior, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=float(sigma_x),
        prior_target_coords=np.asarray(prior, dtype=float),
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        coords = opt.run()

    # chi2 of the spectral block at the fitted geometry, per degree of freedom.
    # Computed here rather than taken from the optimiser so the two BOB settings
    # are scored identically.
    from backend.spectral.centrifugal_distortion import rotational_constants_mhz

    resid, n = 0.0, 0
    for t in targets:
        masses = np.asarray(t["masses"], dtype=float)
        calc = rotational_constants_mhz(np.asarray(coords, dtype=float), masses)
        r = (float(calc[int(t["component"])]) - float(t["value"])) / float(t["sigma"])
        resid += r * r
        n += 1
    dof = max(n - (3 * len(mol.elems) - 6), 1)
    return coords, resid / dof


def main() -> None:
    print(f"  BOB u_H scan at {METHOD}/{BASIS}")
    print(f"  built-in u_H = {U_BUILTIN} (100% sigma)\n")
    out_path = _ROOT / "output" / f"bob_validation_{METHOD}_{BASIS}.json".replace("/", "-")
    results: dict = {}
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))

    for mol in POOL:
        print(f"  {mol.name} ({len(mol.species)} species)", flush=True)
        t0 = time.time()
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        cache: dict = {}

        def hessian_fn(coords, _b=backend, _c=cache):
            key = np.asarray(coords, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _b.run_hessian(coords).hessian_bohr
            return _c[key]

        with contextlib.redirect_stdout(io.StringIO()):
            prior = backend.optimise(start_geometry(mol))
        isos = build_isotopologues(mol, None)
        with contextlib.redirect_stdout(io.StringIO()):
            ctbl, _info = build_correction_table_from_hessian(
                hessian_fn(prior), prior, isos, hessian_fn=hessian_fn,
                cubic_scheme="normal_mode", harmonic_scheme="watson")
        sigma_x = _SIGMA_X_BY_LEVEL.get((METHOD.lower(), BASIS.lower()), 0.020)

        rows = []
        print(f"      {'u_H':>8s}{'chi2/dof':>11s}{'bond mA':>10s}")
        for u in U_SCAN:
            try:
                coords, chi2 = fit_once(mol, isos, prior, ctbl, sigma_x,
                                        bob_with_u_h(mol.elems, u))
                bond = rms_bond_error(mol, coords)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"      {u:8.4f}  FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append({"u": u, "chi2": chi2, "bond_ma": bond})
            mark = "  <- built-in" if abs(u - U_BUILTIN) < 1e-9 else ""
            print(f"      {u:8.4f}{chi2:11.3f}{bond:10.2f}{mark}", flush=True)
        if rows:
            best_chi2 = min(rows, key=lambda r: r["chi2"])
            best_bond = min(rows, key=lambda r: r["bond_ma"])
            print(f"      best chi2 at u_H = {best_chi2['u']:+.4f}"
                  f"   best structure at u_H = {best_bond['u']:+.4f}"
                  f"   ({time.time() - t0:.0f} s)\n", flush=True)
            results[mol.key] = {"rows": rows, "best_chi2_u": best_chi2["u"],
                                "best_bond_u": best_bond["u"],
                                "n_species": len(mol.species)}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if results:
        print("  summary")
        print(f"  {'molecule':22s}{'best chi2 u_H':>15s}{'best structure u_H':>21s}")
        for key, r in results.items():
            print(f"  {key:22s}{r['best_chi2_u']:+15.4f}{r['best_bond_u']:+21.4f}")
        print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
