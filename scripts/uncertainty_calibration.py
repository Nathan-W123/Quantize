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

RESULT, and the aggregate number is the least informative part
--------------------------------------------------------------
42 coordinates across the held-out molecules, both levels measured:

                              B3LYP/6-31g(d)      HF/6-31g
                              sigma_x = 0.007     sigma_x = 0.020
    coverage                  38/42 = 90%         27/42 = 64%
    median |error| / sigma     0.39                0.34
      bonds                   18/20, median 0.72   8/20, median 1.40
      angles                  20/22, median 0.24  19/22, median 0.07

(ideal: 68% coverage, and median |error|/sigma = 0.674 for a half-normal.)

HF's 64% looks beautifully calibrated and is a COINCIDENCE: it averages bonds
that are overconfident with angles that are far too wide. Read the split, never
the total.

  bonds   at B3LYP, median 0.72 against an ideal 0.674 -- right. At HF, 1.40,
          so the quoted bond sigma is about 2x too SMALL and covers 8/20.
          HF/6-31g is the benchmark's own default level, and there
          geometry_uncertainty's "expect it to be optimistic" is correct.
  angles  2.8x too wide at B3LYP and ~10x too wide at HF -- and this one is
          CONSERVATIVE BY DESIGN rather than broken. Decomposed on formyl
          fluoride's F3-C1-H4 at B3LYP: data sigma 0.125 deg, prior sigma
          1.123 deg, data support 0.102. The prior term is blind x floor =
          0.898 x 1.25 = 1.1225, matching to four digits, so the width IS the
          per-class floor. Removing the floor drops it to 0.82 deg.

          The floor is derived from the error of THEORY against the reference
          (B3LYP angle median 0.61 deg) and then quoted for the blind fraction
          of a coordinate. But the FITTED structure does better than theory
          alone even on a 90%-blind coordinate -- its realised angle error is
          about 2.3x smaller -- so the quoted width exceeds the realised error
          by roughly that factor.

          Leaving it. On a molecule with no published structure, which is the
          case this engine exists for, there is no way to verify that the fit
          beat theory on a coordinate the data cannot see. Quoting the theory
          error there is the honest conservative choice, and 2.8x
          over-coverage on this set is what that costs. Narrowing it would buy
          nominal calibration on ten molecules at the price of over-claiming
          on the ones that matter.

An earlier version of this docstring recorded the B3LYP column and attributed
it to RHF/6-31G, and concluded from it that the intervals are conservative and
that geometry_uncertainty's optimism warning "should not be relied on". That
was a level mix-up: at HF the warning holds for bonds. Both columns are now
measured and labelled.

Two things keep this from being a clean bill of health. The errors here are
measured against r_0 and r_s references, not equilibrium structures, so part of
each "error" is the reference's own offset from r_e -- that inflates |error| and
therefore pushes coverage DOWN, so 90% is if anything an underestimate of how
wide the intervals are. And coverage is a statement about noise: a bias shared
across every isotopologue of a molecule moves the fitted structure and the
interval together, so this test cannot see it. It says the widths are not
overconfident. It does not say the centres are right.
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
#: Keyed on the method string the REST of the codebase uses, which is "hf":
#: the pyscf backend's default, and METHOD in the benchmark. These entries
#: said "rhf", so they could never be looked up and every HF run fell through
#: to the 0.020 default. Harmless only because the default happened to equal
#: them -- it would have gone silently wrong the first time anyone changed a
#: value or added a level. Both spellings are accepted now so an "rhf=" on
#: the command line still resolves.
THEORY_SIGMA_X = {
    ("hf", "6-31g"): 0.020,
    ("hf", "6-31g(d)"): 0.020,
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
#: HF/6-31G is the benchmark's DEFAULT level and had no row here at all, so
#: CLASS_SIGMA came back None and no per-class floor reached the
#: theory-determined fraction of any coordinate. Measured consequence: bond
#: coverage 8/20 with median |error|/sigma = 1.40 against an ideal 0.674 --
#: the quoted bond sigma about 2x too small, which is the overconfident
#: direction. sigma_x was not the problem (0.020 against a 16-24 mA error is
#: about right); the missing class resolution was.
#:
#: Measured the same way the B3LYP row describes itself: geometry error of the
#: level against the published reference, by class, over the 10-molecule set,
#: binned with MolecularOptimizer._coordinate_class.
#:
#:   class       n       RMS        B3LYP row     ratio
#:   X-H        25    0.01302        0.00593       2.2x
#:   C-F         7    0.02711        0.00520       5.2x
#:   skeleton   19    0.01784        0.00716       2.5x
#:
#: PROCEDURE VALIDATED by re-deriving the B3LYP row with the same code before
#: trusting it for a new one: X-Cl came back 0.04322 against a recorded
#: 0.0432 -- exact, and it is n=1 so deterministic -- and X-H 0.00586 against
#: 0.00593, 1.2%. C-F and skeleton land 32% and 11% away, so these are not
#: bit-identical to whatever produced the recorded row (optimiser convergence
#: or start geometry, most likely); the binning and the comparison are right.
#:
#: TWO CLASSES DELIBERATELY OMITTED:
#:
#:   angle    HF's RMS is 4.00 deg against its own median of 0.57, a factor
#:            of seven, and it is ONE coordinate: isocyanic acid's H-N-C,
#:            30.34 deg off. HNCO is quasi-linear there and HF fails
#:            qualitatively rather than typically -- B3LYP's worst angle on
#:            the same set is 3.64 deg, so this is HF's failure and not an
#:            artefact. Dropping it gives 0.975 deg, but dropping it AFTER
#:            seeing that it inflates the number is post-hoc, which is the
#:            thing the pre-registered exclusion below exists to avoid. And
#:            HF angles are already about 10x too wide, so a 4 deg floor
#:            would push the wrong way. No angle floor until the rule is
#:            decided in advance.
#:
#:   X-Cl     n=1, and that one molecule is chlorofluoromethane, which the
#:            pre-registered exclusion below already removes from the primary
#:            metric. A floor resting entirely on an excluded reference is
#:            not a floor.
#:
#: A class absent from the dict gets .get(cls, 0.0) downstream, so it is
#: simply unfloored -- exactly today's behaviour for those two.
_HF_CLASS_SIGMA = {"X-H": 0.01302, "C-F": 0.02711, "skeleton": 0.01784}

PRIOR_CLASS_SIGMA = {
    ("b3lyp", "6-31g(d)"): {"X-H": 0.00593, "C-F": 0.00520,
                            "skeleton": 0.00716, "X-Cl": 0.0432,
                            "angle": 1.25},
    ("hf", "6-31g"): dict(_HF_CLASS_SIGMA),
    ("rhf", "6-31g"): dict(_HF_CLASS_SIGMA),
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
