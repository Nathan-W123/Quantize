"""The field-standard baseline this engine had never been measured against.

"Mixed estimation" (Demaison, Vogt and co-workers) is the established way to
combine spectroscopy with quantum chemistry for equilibrium structures, and it
is what every comparison in this repository was missing. Until now the hybrid
was scored against pure theory and pure spectroscopy -- two baselines nobody in
the field actually uses on their own -- so a win proved only that combining
beats not combining, which was never in doubt.

What mixed estimation does
--------------------------
One weighted least-squares fit over the structural parameters, with two blocks
of observations:

    chi^2 = sum_i [(B_calc - B_e,obs) / sigma_B]^2        <- corrected spectra
          + sum_j [(p_j - p_j,ab-initio) / sigma_p]^2      <- "predicate" values

The second block is the entire quantum-chemical contribution: the ab initio
geometry enters as a set of *point values* with an assigned uncertainty. That
is the distinction this script exists to measure. The hybrid engine instead
carries the local quadratic energy model -- gradient and Hessian -- so the
prior knows the curvature of the surface in every direction rather than one
number per coordinate with a width someone had to choose.

Fairness
--------
Both methods get identical inputs: the same VPT2-corrected constants, the same
observation sigmas, and the same statement about how far the level of theory
may be trusted (sigma_x, the value the hybrid's alpha_q is calibrated from).
Only the mechanism differs. sigma_p is scanned as well, because the choice is
free in mixed estimation and a single value would be a strawman.

Parameterisation
----------------
The fit runs over Cartesian coordinates with the predicate residuals evaluated
on the internal coordinates derived from them. This is the same objective as
fitting internal coordinates directly -- identical residuals, identical
minimum -- and avoids writing a per-topology internal-to-Cartesian embedder.
The six rigid-body directions are flat in both observation blocks; they leave
the internal coordinates untouched, and the trust-region solver handles the
rank deficiency without special casing.

    python scripts/mixed_estimation_baseline.py [molecule_key ...]
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    rotational_constants_mhz,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from backend.spectral.rovib_corrections import resolve_corrections  # noqa: E402
from dev.monofluoro_references import (  # noqa: E402
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    WATER_SET,
)
from scripts.monofluoro_benchmark import (  # noqa: E402
    build_isotopologues,
    start_geometry,
)

METHOD, BASIS = "hf", "6-31g"
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

#: Level-of-theory trust widths, in Angstrom, measured on these molecules.
#: The comparison hinges on this number being right for the level in use:
#: hand mixed estimation a sigma_p calibrated for RHF while the geometry came
#: from a triple-zeta functional and the baseline is being sandbagged.
_SIGMA_X_BY_LEVEL = {
    ("hf", "6-31g"): 0.020,
    ("b3lyp", "6-31g(d)"): 0.007,
    ("b3lyp", "cc-pvtz"): 0.004,
}

#: How far the quantum surface is trusted, in Angstrom. The hybrid derives its
#: alpha_q from exactly this number, so handing the same value to mixed
#: estimation as its predicate sigma is what makes the comparison about
#: mechanism rather than about who was given the more generous prior.
SIGMA_X_ANG = _SIGMA_X_BY_LEVEL.get((METHOD.lower(), BASIS.lower()), 0.020)

#: Predicate sigmas to scan. Mixed estimation leaves this to the practitioner,
#: so reporting one value would be a strawman: too tight and theory dominates,
#: too loose and the method degenerates to a spectroscopy-only fit. The scan
#: shows the whole curve and lets the best case stand as the baseline.
SIGMA_P_SCAN_ANG = tuple(sorted({
    round(SIGMA_X_ANG * f, 5) for f in (0.1, 0.25, 0.5, 1.0, 2.5, 5.0)}))

#: Raise every corrected target's sigma to the residual inertial defect left
#: after correcting it. Off by default so existing numbers stay reproducible.
DEFECT_BIAS_FLOOR = any(a == "defect_floor=on" for a in sys.argv[1:])

#: Angle predicate sigma, in degrees, paired with each bond sigma above by the
#: same ratio the reference module uses between its bond and angle widths.
_ANGLE_SIGMA_PER_ANG = 1.5 / 0.020


def corrected_targets(mol, isos, ctbl):
    """B_e per species and component, resolved exactly as the hybrid resolves them.

    Both methods must fit the same numbers *and* weight them the same way, and
    that is why this goes through ``resolve_corrections`` rather than applying
    the correction by hand. An earlier version added the VPT2 delta itself and
    then took sigma from the observed constants alone, which quietly skipped
    two things resolve_corrections does: it carries the correction's own
    uncertainty into sigma, and it removes the systematic part of the
    experimental sigma once a real vibrational correction has been applied
    (that part was standing in for the missing correction, so keeping it would
    double-count it).

    The effect was not cosmetic, and it runs the opposite way to the obvious
    guess. Measured on water, the resolved sigma is far *tighter* than the raw
    experimental one -- A: 1974 MHz against 20893, B: 953 against 10876 -- so
    mixed estimation had been fitting identical target values with sigmas
    about ten times too loose, and weight goes as 1/sigma^2, so its spectral
    block carried roughly a hundredth of the influence the hybrid's did. The
    narrowing is the double-count removal: sigma_exp is dominated by the
    r_0-versus-r_e model gap, and applying a vibrational correction is the act
    of removing that gap, so continuing to carry it would charge twice.

    End to end at RHF/6-31G this moves mixed estimation from 0.95 to 1.81 mA
    on water and from 17.46 to 26.71 on acetyl fluoride -- the latter now
    worse than pure theory's 15.75, because with the correct sigmas the fit is
    data-dominated and the RHF/6-31G corrected targets are not consistent with
    a structure that good. That is a real result about the corrections, not an
    artefact: the hybrid has always used these sigmas, which is why it too
    sits near 20 mA on acetyl fluoride whatever prior it is handed.
    """
    resolved = resolve_corrections(
        isos, correction_table=ctbl, mode="hybrid_auto",
        elems=list(mol.elems),
        # Model-free bias floor: whatever inertial defect survives the
        # vibrational correction is correction error, and no sigma should claim
        # to be tighter than that.
        defect_bias_floor=DEFECT_BIAS_FLOOR,
        coords_ang=np.asarray(mol.geometry, dtype=float))
    by_species = {iso["name"]: np.asarray(iso["masses"], dtype=float)
                  for iso in isos}
    out = []
    for t in resolved:
        masses = by_species.get(t.isotopologue_label)
        if masses is None:
            continue
        sigma = float(t.sigma_mhz)
        out.append({
            "masses": masses,
            "component": int(t.component_index),
            "value": float(t.value_mhz),
            "sigma": sigma if sigma > 0 else 1.0,
        })
    return out


def mixed_estimation_fit(mol, targets, theory_coords, sigma_p_ang):
    """One weighted least-squares fit: corrected constants plus predicates.

    Returns the fitted Cartesian geometry. The starting point is the theory
    geometry, which is also where the predicates are centred -- the standard
    choice, and the same warm start the hybrid uses, so neither method gets an
    initialisation advantage.
    """
    x0 = np.asarray(theory_coords, dtype=float).ravel()
    ref_int = mol.internal_coordinates(np.asarray(theory_coords, dtype=float))
    bond_names = list(mol.bonds.keys())
    angle_names = list(mol.angles.keys())
    sigma_a = sigma_p_ang * _ANGLE_SIGMA_PER_ANG

    def residuals(flat):
        coords = flat.reshape(-1, 3)
        res = []
        for t in targets:
            calc = rotational_constants_mhz(coords, t["masses"])
            res.append((float(calc[t["component"]]) - t["value"]) / t["sigma"])
        got = mol.internal_coordinates(coords)
        for name in bond_names:
            res.append((got[name] - ref_int[name]) / sigma_p_ang)
        for name in angle_names:
            res.append((got[name] - ref_int[name]) / sigma_a)
        return np.asarray(res, dtype=float)

    sol = least_squares(residuals, x0, method="trf", xtol=1e-12, ftol=1e-12,
                        max_nfev=2000)
    return sol.x.reshape(-1, 3), float(np.sum(sol.fun ** 2))


def hybrid_fit(mol, isos, theory_coords, ctbl):
    """The engine's own answer on identical inputs."""
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(theory_coords, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=SIGMA_X_ANG,
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def rms_bond_error(mol, coords):
    ref = mol.internal_coordinates(np.asarray(mol.geometry, dtype=float))
    got = mol.internal_coordinates(np.asarray(coords, dtype=float))
    errs = [(got[k] - ref[k]) * 1000.0 for k in mol.bonds]
    return float(np.sqrt(np.mean(np.square(errs)))), errs


def main() -> None:
    pool = (list(MOLECULES) + list(MOLECULES_SET2)
            + [WATER_SET[0], OZONE, ISOCYANIC_ACID])
    args = [a for a in sys.argv[1:]
            if not a.startswith(("method=", "basis="))]
    wanted = args or [m.key for m in pool]
    _tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    out_path = _ROOT / "output" / f"mixed_estimation_baseline_{_tag}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}

    print(f"  {METHOD.upper()}/{BASIS}, VPT2-corrected constants, "
          f"sigma_x = {SIGMA_X_ANG} A.")
    print("  Mixed estimation (ab initio as point predicates) vs the hybrid "
          "(ab initio as an energy model).\n")

    for mol in pool:
        if mol.key not in wanted or mol.key in out:
            continue
        t0 = time.time()
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        with contextlib.redirect_stdout(io.StringIO()):
            theory = backend.optimise(start_geometry(mol))
        cache: dict = {}

        def hessian_fn(coords_ang):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in cache:
                cache[key] = backend.run_hessian(coords_ang).hessian_bohr
            return cache[key]

        # Sigmas come from the reference module's own model (published
        # precision combined with the r_0-vs-r_e model gap). Both methods are
        # handed exactly these, so neither is advantaged by its weighting.
        isos = build_isotopologues(mol, None)
        with contextlib.redirect_stdout(io.StringIO()):
            ctbl, _info = build_correction_table_from_hessian(
                hessian_fn(theory), np.asarray(theory, dtype=float), isos,
                hessian_fn=hessian_fn, cubic_scheme="normal_mode")

        targets = corrected_targets(mol, isos, ctbl)
        rec = {"n_species": len(isos), "n_targets": len(targets),
               "theory_rms_ma": rms_bond_error(mol, theory)[0],
               "mixed_estimation": {}}

        for sig_p in SIGMA_P_SCAN_ANG:
            geom, chi2 = mixed_estimation_fit(mol, targets, theory, sig_p)
            rms, _ = rms_bond_error(mol, geom)
            rec["mixed_estimation"][f"{sig_p}"] = {"rms_bond_ma": rms,
                                                   "chi2": chi2}

        hyb = hybrid_fit(mol, isos, theory, ctbl)
        rec["hybrid_rms_ma"] = rms_bond_error(mol, hyb)[0]
        best_sig = min(rec["mixed_estimation"],
                       key=lambda s: rec["mixed_estimation"][s]["rms_bond_ma"])
        rec["me_best_sigma_p"] = best_sig
        rec["me_best_rms_ma"] = rec["mixed_estimation"][best_sig]["rms_bond_ma"]

        print(f"  {mol.name:<22} theory {rec['theory_rms_ma']:6.2f}   "
              f"ME(best sig_p={best_sig}) {rec['me_best_rms_ma']:6.2f}   "
              f"hybrid {rec['hybrid_rms_ma']:6.2f} mA"
              f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)
        out[mol.key] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
