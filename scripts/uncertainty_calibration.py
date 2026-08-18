"""Are the engine's uncertainties honest?

For a fluorinated molecule with no accepted structure -- the case this project
targets -- there is no error to report, only an uncertainty. That makes the
uncertainty the deliverable, and an uncertainty nobody has checked is worse than
none: it invites a reader to trust a number by exactly the amount it is wrong.

So this measures the only thing that can be measured: on molecules that *do*
have published structures, how often does the quoted interval actually contain
the true value? A well calibrated 1-sigma interval covers about 68% of the time.
Coverage far below that means the engine is overconfident and its intervals must
not be quoted on unknown molecules until the cause is fixed.

    python scripts/uncertainty_calibration.py
    python scripts/uncertainty_calibration.py method=b3lyp "basis=6-31g(d)"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from dev.monofluoro_references import MOLECULES_SET2, MOLECULES_SET3  # noqa: E402
from backend.quantum import _detect_angles, _detect_bonds  # noqa: E402
from scripts.monofluoro_benchmark import (  # noqa: E402
    build_isotopologues,
    start_geometry,
)


def measure(coords, elems, bonds, angles):
    """Bond lengths and angles keyed the way geometry_uncertainty keys them.

    The reference module names coordinates chemically ("C=O"); the engine
    detects them from the geometry and names them by atom index ("C1-O2").
    Deriving both from the same index pairs is what lets an error be paired
    with the uncertainty that belongs to it.
    """
    out = {}
    for i, j in bonds:
        out[f"{elems[i]}{i + 1}-{elems[j]}{j + 1}"] = (
            float(np.linalg.norm(coords[i] - coords[j])), True)
    for i, j, k in angles:
        v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
        cosa = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        out[f"{elems[i]}{i + 1}-{elems[j]}{j + 1}-{elems[k]}{k + 1}"] = (
            float(np.degrees(np.arccos(np.clip(cosa, -1.0, 1.0)))), False)
    return out

METHOD, BASIS = "b3lyp", "6-31g(d)"
ONLY = None
SIGMA_X = None
MOLSET = "2"
VPT2 = False
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]
    elif _tok.startswith("only="):
        ONLY = _tok.split("=", 1)[1]
    elif _tok.startswith("sigma="):
        SIGMA_X = float(_tok.split("=", 1)[1])
    elif _tok.startswith("set="):
        MOLSET = _tok.split("=", 1)[1]
    elif _tok.startswith("vpt2="):
        VPT2 = _tok.split("=", 1)[1] not in ("0", "false", "")

#: Trust radius of the quantum surface, per level of theory, in Angstrom.
#: This is E1 of the error-model iteration: the module default of 0.020 A was
#: calibrated for RHF/6-31G (geometry error 16-24 mA RMS on these molecules)
#: and the earlier calibration runs inherited it at B3LYP, whose measured error
#: is 5.5-6.7 mA RMS -- so every theory-determined coordinate reported a sigma
#: ~3x too wide by construction. Values here are calibrated on molecules with
#: known structures and then frozen; sigma= overrides.
THEORY_SIGMA_X = {
    ("rhf", "6-31g"): 0.020,
    ("rhf", "6-31g(d)"): 0.020,
    ("b3lyp", "6-31g(d)"): 0.007,
}
if SIGMA_X is None:
    SIGMA_X = THEORY_SIGMA_X.get((METHOD.lower(), BASIS.lower()), 0.020)

#: Per-class prior widths: the measured RMS geometry error of the level of
#: theory itself, by bond class, over the 10-molecule reference set
#: (output/theory_class_errors_b3lyp.json). The isotropic sigma_x above sets
#: the fit's regularisation; these set what is *reported* for the fraction of
#: a coordinate the data cannot see, because "theory-determined" is not one
#: number: at B3LYP/6-31G(d) a C-Cl bond errs 6x worse than a C-F bond.
#: X-Cl rests on a single molecule (n=1) whose 1953 reference is itself the
#: pre-registered exclusion above -- treat that entry as provisional.
#: Bond classes in Angstrom, "angle" in degrees.
PRIOR_CLASS_SIGMA = {
    ("b3lyp", "6-31g(d)"): {"X-H": 0.00593, "C-F": 0.00520,
                            "skeleton": 0.00716, "X-Cl": 0.0432,
                            "angle": 1.25},
}
CLASS_SIGMA = PRIOR_CLASS_SIGMA.get((METHOD.lower(), BASIS.lower()))

#: Reference-truth uncertainty, pre-registered before any run scored with it.
#: The harness previously treated the published structures as exact, but they
#: are r_s / r_0 determinations from the 1950s-60s: Costain-rule substitution
#: coordinates are good to roughly a couple of mA away from near-axis
#: pathologies, and angles to a couple tenths of a degree. Coverage is judged
#: on z = error / hypot(sigma_reported, sigma_ref), so agreement better than
#: the reference can support is not scored as miscalibration.
#:
#: Pre-registered exclusion: chlorofluoromethane is reported but excluded from
#: the primary metric, on evidence recorded before this scoring existed --
#: theory, spectroscopy-alone and the hybrid all disagree with its 1953
#: reference in the same direction on C-Cl, and its consistency check is the
#: worst in the set.
REF_SIGMA_BOND_MA = 2.0
REF_SIGMA_ANGLE_DEG = 0.2

_TAG = "" if ONLY is None else f"_{ONLY}"
if VPT2:
    _TAG += "_vpt2"
OUT = _ROOT / "output" / f"uncertainty_calibration{_TAG}.json"


_VPT2_CACHE: dict = {}


def run_one(mol, limit):
    b = get_backend("pyscf_hf")(elems=list(mol.elems), method=METHOD, basis=BASIS)
    gt = b.optimise(start_geometry(mol))
    isos = build_isotopologues(mol, limit)
    ctbl = None
    if VPT2:
        # The B0 -> Be correction from the normal-mode cubic force field,
        # evaluated at the theory-optimised geometry (the stationary point the
        # scheme requires). Cached per molecule: the displaced Hessians do not
        # depend on the data level.
        from scripts.monofluoro_alpha_corrected import correction_table
        cache = _VPT2_CACHE.setdefault(mol.key, {})
        ctbl, _info, _t = correction_table(mol, gt, isos, b, cache)
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(gt, dtype=float),
        isotopologues=isos,
        correction_table=ctbl,
        quantum_backend="pyscf_hf", orca_method=METHOD, orca_basis=BASIS,
        coordinate_mode="cartesian", use_autoconfig=False, max_iter=40,
        hess_recalc_every=10,
        # Model error stays in the weighting sigma; two-sided rescaling sets
        # its magnitude from the residuals rather than from a guess.
        chi2_rescale=True, chi2_rescale_max_passes=3,
        quantum_prior_sigma_ang=SIGMA_X,
        prior_class_sigma=CLASS_SIGMA)
    with contextlib.redirect_stdout(io.StringIO()):
        res = opt.run()
    coords = res["coords"] if isinstance(res, dict) and "coords" in res else res
    unc = opt.geometry_uncertainty()
    return (coords, unc, opt.point_group, opt.data_support(),
            opt.reduced_chi_square(), opt.sigma_split())


def main() -> None:
    print(f"  {METHOD.upper()}/{BASIS}, sigma_x = {SIGMA_X} A. "
          f"Do the quoted uncertainties cover the truth?\n")
    rows, records = [], []
    pool = MOLECULES_SET3 if MOLSET == "3" else MOLECULES_SET2
    mols = [m for m in pool if ONLY is None or m.key == ONLY]
    for mol in mols:
        elems = list(mol.elems)
        for label, limit in (("parent only", 1), ("all species", None)):
            coords, unc, pg, support, chi2, split = run_one(mol, limit)
            bonds = _detect_bonds(coords, elems)
            angles = _detect_angles(bonds)
            got = measure(np.asarray(coords, float), elems, bonds, angles)
            ref = measure(np.asarray(mol.geometry, float), elems, bonds, angles)
            print(f"  {mol.name} / {label}   (point group {pg or '-'}, "
                  f"chi2/nu = {chi2:.2f})")
            print(f"    {'coordinate':<14}{'accepted':>10}{'fitted':>10}"
                  f"{'error':>11}{'quoted 1σ':>12}{'':>3}")
            for name, (value, is_bond) in got.items():
                if name not in ref or name not in unc:
                    continue
                scale = 1000.0 if is_bond else 1.0
                unit = "mÅ" if is_bond else "°"
                err = (value - ref[name][0]) * scale
                sig = unc[name] * scale
                covered = abs(err) <= sig
                rows.append((covered, abs(err), sig, is_bond))
                d_sig, p_sig = split.get(name, (float("nan"), float("nan")))
                sig_ref = REF_SIGMA_BOND_MA if is_bond else REF_SIGMA_ANGLE_DEG
                z = err / float(np.hypot(sig, sig_ref)) if (sig > 0 or sig_ref > 0) else float("nan")
                records.append({"molecule": mol.name, "level": label,
                                "coordinate": name, "error": err, "sigma": sig,
                                "sigma_data": d_sig * scale, "sigma_prior": p_sig * scale,
                                "sigma_ref": sig_ref, "z": z,
                                "covered": bool(covered), "is_bond": bool(is_bond),
                                "data_support": float(support.get(name, float("nan"))),
                                "chi2_nu": float(chi2)})
                print(f"    {name:<14}{ref[name][0]:>10.4f}{value:>10.4f}"
                      f"{err:>+9.2f} {unit:<2}{sig:>10.2f} {unit:<2}"
                      f"{'  ok' if covered else '  MISS'}"
                      f"{'' if support.get(name, 1.0) > 0.5 else '  [theory-determined]'}")
            print()

    n = len(rows)
    cov = sum(1 for c, *_ in rows if c)
    print(f"  {'=' * 70}")
    print(f"  Coverage of the quoted 1σ interval: {cov}/{n} = {100 * cov / n:.0f}%"
          f"   (well calibrated ≈ 68%)")
    ratios = [e / s for _, e, s, _ in rows if s > 0]
    print(f"  Median |error| / σ: {np.median(ratios):.2f}   (well calibrated ≈ 0.67)")
    for kind, flag in (("bonds", True), ("angles", False)):
        sub = [(c, e, s) for c, e, s, b in rows if b is flag]
        if sub:
            k = sum(1 for c, *_ in sub if c)
            print(f"    {kind:<8}{k}/{len(sub)} covered, "
                  f"median |error|/σ = {np.median([e / s for _, e, s in sub if s > 0]):.2f}")
    print(f"  {'=' * 70}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"method": METHOD, "basis": BASIS,
                               "sigma_x": SIGMA_X,
                               "coverage": cov / n, "n": n,
                               "records": records}, indent=2), encoding="utf-8")
    print(f"\n  written to {OUT}")


if __name__ == "__main__":
    main()
