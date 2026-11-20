"""Why is chlorofluoromethane's offsets hybrid worse at B3LYP than at RHF?

    python scripts/archive/chlorofluoromethane_level_gap.py <hf|b3lyp> <out.json>
        [--output-dir DIR]   # theory geometries + Hessian cache (default output/)

The benchmark's ``offsets`` row for CH2ClF scores 2.91 mA bond RMS against
Vogt's r_e^SE at RHF/6-31G but 6.00 at B3LYP/6-31G(d) (5.42 / 8.18 against its
primary reference) -- the one molecule that does not improve from RHF to
B3LYP. This reruns that row from the cached theory geometries and Hessians,
then reports per bond against r_e: the theory geometry, the offset-corrected
prior centre, mixed estimation and the hybrid; two counterfactual centres (the
C-Cl offset replaced by the molecule's own C-Cl error, and r_e itself); and
how visible each geometry's error is to the corrected data (chi2 over the six
targets, two Cl isotopologues x A, B, C, with the fit's own sigmas).

Answer (2026-11-20, c8904da rows; T-08, D-39). The level is not worse -- its
centre is. Per bond vs r_e (mA), C-Cl / C-F / C-H:

                     RHF/6-31G               B3LYP/6-31G(d)
  theory          +73.2  +25.9  -15.0      +38.2   -1.2   +6.7
  pool offset     +85.6  +28.1  -10.6      +25.4   +4.5   +5.2
  centre          -12.4   -2.2   -4.4      +12.8   -5.7   +1.4
  ME               -5.4   +1.5   -4.8       +8.3   -7.9   +1.8
  hybrid           +1.0   +2.9   -4.0       +7.4   -7.2   +1.4
  hybrid RMS        2.91                     6.00

* Methyl chloride's C-Cl offset (the class's only other member) misses this
  molecule's own C-Cl error by about the same amount at both levels: RHF
  over-corrects by 12.4 mA, B3LYP under-corrects by 12.8. At B3LYP the C-F
  offset also misses, over-correcting by 5.7 mA (the five fluorides' B3LYP
  C-F errors spread from -5.4 to +13.5 mA; at RHF 24.6 to 30.7).
* Started at r_e the fit stays there (hybrid 0.44 RHF, 0.39 B3LYP RMS), and
  both answers fit the data better than r_e does (chi2 0.29 / 0.11 against
  3.70 / 2.39): two isotopologues leave a valley of structures several mA wide
  that the data cannot rank, and the fit keeps whatever part of the centre's
  error lies along it. The per-bond sensitivity is the same at both levels
  (C-Cl +10 mA from r_e: chi2 139 / 136; C-F -10 mA: 133 / 115).
* B3LYP's two misses (C-Cl long, C-F short) partly cancel in the rotational
  constants -- the centre's bond errors alone score chi2 110 against ~278 for
  their parts -- so they lie along the valley and stay. RHF's (every bond
  short) add (464 against ~376), lie across it, and are mostly removed.
* With C-Cl's offset replaced by its own error, B3LYP goes 6.00 -> 2.95 and
  RHF 2.91 -> 4.16: RHF's 2.91 is partly a cancellation between its C-Cl
  over-correction and the fit's pull. An independent check (a hybrid
  surrogate validated on the HF fits to 0.04 mA per bond) put B3LYP at 0.24
  and RHF at 1.92 (a real fit: 1.96) with perfect bond offsets and theory
  angles, and ruled out the prior width, correction tables and Hessians
  (swapping each between levels moves either result by <= 0.3 mA).

Remedy, not implemented: better offsets for this molecule's classes -- a
second C-Cl calibration reference, and C-F offsets that transfer at B3LYP --
or more isotopologue data to close the valley.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("level", choices=["hf", "b3lyp"])
_ap.add_argument("out")
_ap.add_argument("--output-dir", default=str(_ROOT / "output"),
                 help="directory holding theory_geometries_<tag>.json and hessian_cache/")
_args = _ap.parse_args()
OUT = Path(_args.out).resolve()
CACHE = Path(_args.output_dir).resolve()
sys.argv = [sys.argv[0]]
sys.path[:0] = [str(_ROOT), str(_ROOT / ".github"), str(_ROOT / "scripts")]
os.chdir(tempfile.mkdtemp(prefix="cfm_gap_"))

import numpy as np  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import scripts.accuracy_upgrades_benchmark as A  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral import bond_offsets as BO  # noqa: E402
from backend.spectral.harmonic_alpha import build_correction_table_from_hessian  # noqa: E402
from backend.spectral.hessian_cache import HessianCache  # noqa: E402
from dev.monofluoro_references import CALIBRATION_ONLY  # noqa: E402
from scripts.mixed_estimation_baseline import rotational_constants_mhz  # noqa: E402

M, B, TAG = {"hf": ("hf", "6-31g", "hf_6-31g"),
             "b3lyp": ("b3lyp", "6-31g(d)", "b3lyp_6-31gd")}[_args.level]
A.METHOD, A.BASIS = M, B                     # hybrid_fit reads the module's level
theory = json.loads((CACHE / f"theory_geometries_{TAG}.json").read_text())
scored = list(A.MOLECULES) + list(A.MOLECULES_SET2) + [A.WATER_SET[0], A.OZONE, A.ISOCYANIC_ACID]
cal = A.usable_calibration(list(CALIBRATION_ONLY), theory)
per = A.offsets_pool(theory, [m.key for m in scored + cal])
mol = {m.key: m for m in scored}["chlorofluoromethane"]
classes = BO.bond_classes_of(mol)
x_th = np.asarray(theory[mol.key]["coords"], float)
x_re = np.asarray(mol.equilibrium_geometry, float)
loo = BO.leave_one_out_offsets(per, mol.key)
sx = BO.prior_sigma_for_molecule(mol, per, mol.key)

backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=M, basis=B)
_disk = HessianCache(CACHE / "hessian_cache", method=M, basis=B, elements=list(mol.elems)).wrap(
    lambda c, _b=backend: _b.run_hessian(c).hessian_bohr)
_mem: dict = {}


def hfn(c):
    k = np.asarray(c, float).round(9).tobytes()
    if k not in _mem:
        _mem[k] = _disk(c)
    return _mem[k]


isos = A.build_isotopologues(mol, None)


def fit(centre):
    """The offsets row's recipe from ``centre``: (targets, ME, hybrid)."""
    with contextlib.redirect_stdout(io.StringIO()):
        mr = A.multireference_diagnostic(list(mol.elems), x_th) if A.RUN_DIAGNOSTICS else {}
        ctbl = build_correction_table_from_hessian(
            hfn(centre), centre, isos, hessian_fn=hfn,
            sigma_scale=A.MULTIREFERENCE_SIGMA_SCALE if (mr or {}).get("flagged") else 1.0,
            cubic_scheme="normal_mode", lam_freq_cm=0.0, freq_scale=A.FREQ_SCALE,
            harmonic_scheme=A.HARMONIC_SCHEME)[0]
        tg = A.corrected_targets(mol, isos, ctbl, bob_params=None,
                                 coords_ang=centre, defect_bias_scale=False)
        me, _ = A.mixed_estimation_fit(mol, tg, centre, sx)
        hy = A.hybrid_fit(mol, isos, centre, ctbl, sx, hessian_fn=hfn)
    return tg, np.asarray(me, float), np.asarray(hy, float)


def chi2(x, tg):
    x = np.asarray(x, float)
    return float(sum(((float(rotational_constants_mhz(x, t["masses"])[t["component"]]) - t["value"])
                      / t["sigma"]) ** 2 for t in tg))


def bonds(x):
    q, r = mol.internal_coordinates(x), mol.internal_coordinates(x_re)
    return {n: round(1000 * (q[n] - r[n]), 2) for n in mol.bonds}


def angles(x):
    q, r = mol.internal_coordinates(x), mol.internal_coordinates(x_re)
    return {n: round(q[n] - r[n], 3) for n in mol.angles}


def rms(x):
    return float(A.rms_bond_error(mol, x, x_re)[0])


def kabsch_onto(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    U, _s, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return (a - ca) @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + cb


own = {c: 1000 * (mol.internal_coordinates(x_th)[n] - mol.internal_coordinates(x_re)[n])
       for n, c in classes.items()}
centre = np.asarray(BO.offset_corrected_geometry(mol, x_th, loo), float)
tg, me, hy = fit(centre)
row = json.loads((_ROOT / "reports" / "accuracy_upgrades" / f"accuracy_upgrades_{TAG}.json")
                 .read_text())[mol.key]["offsets"]
res = {
    "level": f"{M}/{B}", "sigma_x_mA": 1000 * sx,
    "own_theory_error_mA": own,
    "pool_offset_mA": {c: 1000 * loo[c] for c in own if c in loo},
    "pool_members_mA": {c: {k: [round(1000 * e, 2) for e in v[c]] for k, v in per.items()
                            if c in v and k != mol.key} for c in own},
    "reproduces_archive": {"hybrid_vs_re": [rms(hy), row["hybrid_rms_ma_vs_re"]],
                           "hybrid_vs_primary": [float(A.rms_bond_error(mol, hy)[0]), row["hybrid_rms_ma"]],
                           "me_vs_re": [rms(me), row["me_rms_ma_vs_re"]]},
    "stages": {k: {"rms_vs_re": rms(x), "bonds": bonds(x), "angles": angles(x), "chi2": chi2(x, tg)}
               for k, x in {"r_e": x_re, "theory": x_th, "centre": centre, "me": me, "hybrid": hy}.items()},
}
ca = kabsch_onto(centre, x_re)
res["chi2_line_re_to_centre"] = {f"{t:.2f}": chi2(x_re + t * (ca - x_re), tg) for t in (0, .25, .5, .75, 1)}
res["chi2_single_class_from_re"] = {
    f"{c} {1000 * d:+.0f}": chi2(BO.offset_corrected_geometry(mol, x_re, {c: -d}), tg)
    for c in sorted(set(classes.values())) for d in (+0.010, -0.010)}
cerr = {c: (mol.internal_coordinates(centre)[n] - mol.internal_coordinates(x_re)[n]) for n, c in classes.items()}
res["chi2_centre_bond_error_on_re"] = chi2(BO.offset_corrected_geometry(mol, x_re, {c: -v for c, v in cerr.items()}), tg)
OUT.write_text(json.dumps(res, indent=2))
print(f"{res['level']}: sigma_x {res['sigma_x_mA']:.2f} mA; reproduces archive {res['reproduces_archive']}")
print(f"  own theory error {({c: round(v, 1) for c, v in own.items()})}, pool offset "
      f"{({c: round(v, 1) for c, v in res['pool_offset_mA'].items()})}")
for k, v in res["stages"].items():
    print(f"  {k:7s} rms vs r_e {v['rms_vs_re']:6.2f}  bonds {v['bonds']}  chi2 {v['chi2']:.2f}")
print("  chi2 along r_e -> centre:", {k: round(v, 2) for k, v in res["chi2_line_re_to_centre"].items()})
print("  chi2 single class from r_e:", {k: round(v, 2) for k, v in res["chi2_single_class_from_re"].items()})
print(f"  chi2 of the centre's bond errors alone on r_e: {res['chi2_centre_bond_error_on_re']:.2f}", flush=True)

oracle = dict(loo)
oracle["C-Cl"] = own["C-Cl"] / 1000
res["counterfactuals"] = {}
for name, c0 in {"ccl_oracle_centre": np.asarray(BO.offset_corrected_geometry(mol, x_th, oracle), float),
                 "re_centre": x_re}.items():
    _tg2, me2, hy2 = fit(c0)
    res["counterfactuals"][name] = {s: {"rms_vs_re": rms(x), "bonds": bonds(x)}
                                    for s, x in {"centre": c0, "me": me2, "hybrid": hy2}.items()}
    OUT.write_text(json.dumps(res, indent=2))
    for s, v in res["counterfactuals"][name].items():
        print(f"  [{name}] {s:7s} rms vs r_e {v['rms_vs_re']:6.2f}  bonds {v['bonds']}", flush=True)
