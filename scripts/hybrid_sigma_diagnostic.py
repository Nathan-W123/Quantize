"""Is the hybrid being dragged off a good prior by a sigma it has not earned?

The hybrid's spectral block outweighs its quantum prior by roughly 100:1,
because resolve_corrections narrows the experimental sigma by about tenfold
once a vibrational correction is applied (water A: 1974 MHz against a raw
20893). That narrowing is correct in principle -- the raw sigma is dominated
by the r_0-versus-r_e model gap and the correction removes that gap -- but it
leaves the corrected target weighted by how well the *correction* is known,
and nothing has ever checked that number against how well the correction
actually is known.

The check is the fit's own chi-square. If the corrected targets and their
sigmas were mutually consistent, a structure would exist that reproduces them
to about one sigma. Measured against the published structures it does not:
mixed estimation given these sigmas lands at 26.71 mA on acetyl fluoride
against pure theory's 15.75.

So this scans one knob -- a multiplier on every correction sigma in the table
-- and reports what the hybrid does. Two outcomes, and they point at different
repairs:

  chi2/nu >> 1 at k=1, falling toward 1 as k grows
        the sigmas are over-confident; the fit knows it, and the honest sigma
        is the one that makes chi-square come out right

  chi2/nu about 1 at k=1, and the geometry still wrong
        the correction is *biased*, not uncertain -- no weighting repairs a
        biased target, and the repair is a better correction level

chi2_rescale cannot settle this on its own: it only fires outside
[1/threshold, threshold], so an over-confident sigma that lands inside that
band is never challenged.

    python scripts/hybrid_sigma_diagnostic.py [molecule_key ...]
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.bond_offsets import (  # noqa: E402
    leave_one_out_offsets,
    offset_corrected_geometry,
    residual_sigma_after_offsets,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from dev.monofluoro_references import (  # noqa: E402
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    WATER_SET,
)
from scripts.mixed_estimation_baseline import (  # noqa: E402
    corrected_targets,
    mixed_estimation_fit,
    rms_bond_error,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "hf", "6-31g"

#: Multipliers on every correction sigma in the table.
SIGMA_K = (1.0, 3.0, 10.0, 30.0)


def scaled_table(ctbl, k):
    """Correction table with every sigma_mhz multiplied by ``k``."""
    out = copy.deepcopy(ctbl)
    for entries in out.values():
        for spec in entries.values():
            if isinstance(spec, dict) and "sigma_mhz" in spec:
                spec["sigma_mhz"] = float(spec["sigma_mhz"]) * float(k)
    return out


def run_hybrid(mol, isos, prior, ctbl, sigma_x):
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(prior, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=float(sigma_x),
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        coords = opt.run()
    try:
        chi2_nu = float(opt.reduced_chi_square())
    except Exception:  # noqa: BLE001
        chi2_nu = float("nan")
    return coords, chi2_nu


def main() -> None:
    pool = (list(MOLECULES) + list(MOLECULES_SET2)
            + [WATER_SET[0], OZONE, ISOCYANIC_ACID])
    wanted = sys.argv[1:] or ["acetyl_fluoride", "vinyl_fluoride"]

    geo_path = _ROOT / "output" / f"theory_geometries_{METHOD}_{BASIS}.json"
    theory = json.loads(geo_path.read_text(encoding="utf-8")) \
        if geo_path.exists() else {}
    per_mol = {k: {c: list(map(float, v)) for c, v in r["offsets"].items()}
               for k, r in theory.items()}

    out_path = _ROOT / "output" / "hybrid_sigma_diagnostic.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}

    for mol in pool:
        if mol.key not in wanted:
            continue
        t0 = time.time()
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        if mol.key in theory:
            theory_coords = np.asarray(theory[mol.key]["coords"], dtype=float)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                theory_coords = backend.optimise(start_geometry(mol))

        cache: dict = {}

        def hessian_fn(coords_ang, _c=cache):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = backend.run_hessian(coords_ang).hessian_bohr
            return _c[key]

        loo = leave_one_out_offsets(per_mol, mol.key)
        prior = offset_corrected_geometry(mol, theory_coords, loo)
        sigma_x = residual_sigma_after_offsets(per_mol, mol.key)
        prior_rms = rms_bond_error(mol, prior)[0]

        isos = build_isotopologues(mol, None)
        with contextlib.redirect_stdout(io.StringIO()):
            base_tbl, _ = build_correction_table_from_hessian(
                hessian_fn(prior), prior, isos, hessian_fn=hessian_fn,
                cubic_scheme="normal_mode")

        print(f"\n  {mol.name}  ({len(isos)} species, "
              f"offset-corrected prior {prior_rms:.2f} mA, "
              f"sigma_x {sigma_x*1000:.1f} mA)")
        print(f"  {'k':>6}{'chi2/nu':>11}{'hybrid mA':>12}{'ME mA':>10}")
        rec = {"prior_rms_ma": prior_rms, "sigma_x_ang": sigma_x, "scan": {}}
        for k in SIGMA_K:
            tbl = scaled_table(base_tbl, k)
            coords, chi2_nu = run_hybrid(mol, isos, prior, tbl, sigma_x)
            hyb = rms_bond_error(mol, coords)[0]
            me_geom, _ = mixed_estimation_fit(
                mol, corrected_targets(mol, isos, tbl), prior, sigma_x)
            me = rms_bond_error(mol, me_geom)[0]
            rec["scan"][str(k)] = {"chi2_nu": chi2_nu, "hybrid_rms_ma": hyb,
                                   "me_rms_ma": me}
            print(f"  {k:6.0f}{chi2_nu:11.3g}{hyb:12.2f}{me:10.2f}", flush=True)
        out[mol.key] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"    ({(time.time()-t0)/60:.1f} min)", flush=True)

    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
