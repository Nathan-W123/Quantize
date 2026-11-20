"""Would per-bond prior widths fix the underdetermined split (backlog N16)?

Answer (2026-11-20, decision D-33): no. They fix chlorofluoromethane and break
isocyanic acid, so they were not adopted.

THE QUESTION. With two Cl isotopologues, CH2ClF's data cannot split a
correction between C-Cl and C-F, so the quantum prior decides it. C-Cl is
the only member of its bond class, so leave-one-out calibration cannot
correct it; bond_offsets.prior_sigma_for_molecule knows its width is large
(uncorrected_class_bias) but collapses every bond's width into one RMS
sigma_x. This asks what happens if the prior is instead widened only along
the uncovered bonds.

THE MECHANISM (exact, checked every run). For each uncovered bond class, s^2
of variance (s = uncorrected_class_bias) is added to its bonds' marginal
prior covariance, shared by the bonds of the class, with everything
conditional on those bonds left unchanged:

    Sigma' = Sigma + Sigma B M^-1 S M^-1 B^T Sigma,   M = B^T Sigma B.

It is applied to the rigid-projected Hessian the fit actually reads (by
patching the optimizer instance's _project_quantum_terms), and alpha_q is
pinned to the value the unwidened Hessian gives, so nothing else moves. The
check reads the prior back through the engine's own accessors
(_projected_reference_terms, quantum_prior_weight): added variance / s^2 is
1.0 on every widened bond. A first version that widened the raw Hessian
failed that check (2.15) because the fit reads the projected one.

RECIPES.
  base     prior at the theory geometry, the benchmark's sigma_x; per-bond =
           the same plus s^2 on the uncovered classes.
  offsets  the benchmark's `offsets` configuration (offset-corrected centre,
           sigma_x = prior_sigma_for_molecule's RMS; its rows reproduce the
           archive to 4 decimals); "tight" = sigma_x at the calibrated spread
           (residual_sigma_after_offsets); per-bond = tight plus s^2 on the
           uncovered classes -- the per-bond prior the RMS collapses.

RESULTS, bond RMS vs r_e / r_e^SE (mA; HNCO vs SE100, unstated sd; HF/6-31G
at the 5cfc5d4 theory geometries and Hessian cache):

  base recipe         RHF base -> per-bond     B3LYP base -> per-bond
  water (O-H)           0.56 ->  0.63            0.05 ->  0.81
  ozone (O-O)           3.13 ->  2.75            0.06 ->  0.43
  CH2ClF (C-Cl)        10.80 -> 12.42            9.24 ->  5.21
  offsets recipe      RHF offsets -> per-bond  B3LYP offsets -> per-bond
  water                 0.65 ->  0.14            0.69 ->  0.81
  ozone                 2.86 ->  2.88            0.37 ->  0.40
  CH2ClF               17.64 ->  2.44           11.22 ->  4.76
  HNCO (C-N, N-H)      20.01 -> 24.81            2.03 ->  3.62
  HNCO angle RMS (deg)  1.49 -> 16.47            0.68 ->  1.29

Vinyl fluoride and the other four fluorides have no uncovered class, so they
are unchanged by construction. Why HNCO fails: HF/6-31G puts its H-N-C angle
~30 deg off. The blended RMS width (27 mA) loosens the angles too, which lets
the data fix the angle; a per-bond prior keeps angles at the tight calibrated
width, and a bond-only widening cannot cover an angle error.

Diagnostic only: no engine code changes. Needs the benchmark's theory
geometries and Hessian cache (run scripts/accuracy_upgrades_benchmark.py,
which writes them to output/), then from the repository root:

    python scripts/archive/perbond_prior_widths.py hf out.json --recipe offsets \\
        [--output-dir output] [molecule ...]
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
_ap.add_argument("molecules", nargs="*",
                 default=["water", "ozone", "chlorofluoromethane", "vinyl_fluoride", "isocyanic_acid"])
_ap.add_argument("--recipe", choices=["base", "offsets"], default="offsets")
_ap.add_argument("--output-dir", default=str(_ROOT / "output"),
                 help="where the benchmark wrote theory_geometries_*.json and hessian_cache/")
ARGS = _ap.parse_args()
OUT = Path(ARGS.out).resolve()
OUTPUT_DIR = Path(ARGS.output_dir).resolve()
sys.argv = [sys.argv[0]]          # the benchmark reads its own argv on import
sys.path[:0] = [str(_ROOT), str(_ROOT / ".github"), str(_ROOT / "scripts")]
os.chdir(tempfile.mkdtemp(prefix="perbond_"))   # fits write run files to the cwd

import numpy as np  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import scripts.accuracy_upgrades_benchmark as A  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral import bond_offsets as BO  # noqa: E402
from backend.spectral.SVD import _CURVATURE_FLOOR_REL  # noqa: E402
from backend.spectral.harmonic_alpha import build_correction_table_from_hessian  # noqa: E402
from backend.spectral.hessian_cache import HessianCache  # noqa: E402

M, B, TAG = {"hf": ("hf", "6-31g", "hf_6-31g"),
             "b3lyp": ("b3lyp", "6-31g(d)", "b3lyp_6-31gd")}[ARGS.level]
SX = A._SIGMA_X_BY_LEVEL[(M, B)]
theory_all = json.loads((OUTPUT_DIR / f"theory_geometries_{TAG}.json").read_text())
archived = json.loads((_ROOT / "reports" / "accuracy_upgrades" / f"accuracy_upgrades_{TAG}.json").read_text())
pool = {m.key: m for m in list(A.MOLECULES) + list(A.MOLECULES_SET2)
        + [A.WATER_SET[0], A.OZONE, A.ISOCYANIC_ACID]}
per_mol = {k: BO.measure_offsets(pool[k], np.asarray(theory_all[k]["coords"], float))
           for k in A._TABLE_ORDER if k in pool}


def _u(deg):
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t), 0.0])


def se100_hnco():
    """HNCO r_e^SE (SE100; Ceselin et al. 2021, Table S1; no stated sd).
    Atoms N, H, C, O, trans-bent; agrees with the module's r_s to ~4 mA."""
    N = np.zeros(3)
    C = 1.2147 * _u(0)
    return np.array([N, 1.0059 * _u(123.02), C, C + 1.1644 * _u(-(180.0 - 171.51))])


EXTRA_REFS = {"isocyanic_acid": se100_hnco}


def lam_bar(H):
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    return float(np.mean(ev[ev > _CURVATURE_FLOOR_REL * max(float(np.max(np.abs(ev))), 1.0)]))


def pinv_range(P):
    ev, V = np.linalg.eigh(0.5 * (P + P.T))
    keep = ev > _CURVATURE_FLOOR_REL * max(float(np.max(np.abs(ev))), 1.0)
    return (V[:, keep] / ev[keep]) @ V[:, keep].T


def bond_grad(x, i, j):
    e = (x[j] - x[i]) / np.linalg.norm(x[j] - x[i])
    b = np.zeros(x.size)
    b[3 * i:3 * i + 3], b[3 * j:3 * j + 3] = -e, e
    return b


def widen_cov(Sg, Bm, S):
    """Add S to the bonds' marginal covariance; keep the conditional fixed."""
    K = Sg @ Bm @ np.linalg.inv(Bm.T @ Sg @ Bm)
    return Sg + K @ S @ K.T


def install(opt, x_centre, groups, s):
    """Widen the prior along each group (one per uncovered class) by s^2,
    shared within the group, on the Hessian the fit reads; pin alpha_q."""
    pairs = [p for g in groups for p in g]
    S = np.zeros((len(pairs), len(pairs)))
    k = 0
    for g in groups:
        S[k:k + len(g), k:k + len(g)] = s * s
        k += len(g)
    orig_proj, orig_cal, table = opt._project_quantum_terms, opt.optimizer._calibrated_alpha_q, []

    def proj(g, H):
        g2, Hp = orig_proj(g, H)
        Hp = np.asarray(Hp, float)
        a = float(orig_cal(Hp))
        R = (opt._rigid_mode_projector(opt.coords, opt._rigid_ref_masses)
             if (opt.project_rigid_modes and opt._rigid_ref_masses is not None) else np.eye(Hp.shape[0]))
        Bm = np.column_stack([R @ bond_grad(x_centre, i, j) for (i, j) in pairs])
        Hw = pinv_range(widen_cov(pinv_range(a * Hp), Bm, S)) / a
        Hw = 0.5 * (Hw + Hw.T)
        table.append((Hw, a))
        return g2, Hw

    def cal(h):
        h = np.asarray(h, float)
        for Hw, a in reversed(table):
            if h.shape == Hw.shape and np.allclose(h, Hw, rtol=1e-9, atol=1e-12):
                return a
        raise RuntimeError("alpha requested for a Hessian the patch did not produce")

    opt._project_quantum_terms = proj
    opt.optimizer._calibrated_alpha_q = cal


def fit(mol, centre, ctbl, hfn, isos, sx, groups=None, s=None):
    """The benchmark's hybrid_fit, optionally widened; returns the answer and
    the prior variance of every bond as the fit itself weighted it."""
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=centre.copy(), isotopologues=[dict(i) for i in isos],
        quantum_backend="pyscf_hf", orca_method=M, orca_basis=B, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10, correction_table=ctbl,
        quantum_prior_sigma_ang=float(sx), prior_target_coords=centre.copy(),
        defect_bias_scale=False, harmonic_cd_from_hessian=False, fit_cd_constants=False,
        cd_weight=0.0, chi2_rescale=True, chi2_rescale_max_passes=3)
    opt._backend = A._CachedHessianBackend(opt._backend, hfn)
    if groups:
        install(opt, centre, groups, s)
    with contextlib.redirect_stdout(io.StringIO()):
        x = np.asarray(opt.run(), float)
        chi2nu = float(opt.reduced_chi_square())
        _g, H = opt._projected_reference_terms()
        w = opt.quantum_prior_weight(refresh=True)
    Sg = pinv_range(w * np.asarray(H, float))
    var = {n: float(bond_grad(centre, *mol.bonds[n][0]) @ Sg @ bond_grad(centre, *mol.bonds[n][0]))
           for n in mol.bonds}
    return x, {"chi2_nu": chi2nu, "alpha": w}, var


res = {"level": f"{M}/{B}", "recipe": ARGS.recipe, "molecules": {}}
for key in ARGS.molecules:
    mol = pool[key]
    ref = EXTRA_REFS[key]() if key in EXTRA_REFS else np.asarray(mol.equilibrium_geometry, float)
    theory = np.asarray(theory_all[key]["coords"], float)
    loo = BO.leave_one_out_offsets(per_mol, key)
    s_unk = BO.uncorrected_class_bias(per_mol, key)
    if ARGS.recipe == "offsets":
        centre = np.asarray(BO.offset_corrected_geometry(mol, theory, loo), float)
        sx_as_is = BO.prior_sigma_for_molecule(mol, per_mol, key)
        sx_tight = BO.residual_sigma_after_offsets(per_mol, key, floor_ang=0.002)
        arch = archived[key]["offsets"]["hybrid_rms_ma"]
    else:
        centre, sx_as_is, sx_tight = theory, SX, SX
        arch = archived[key]["base"]["hybrid_rms_ma"]
    classes = BO.bond_classes_of(mol)
    wide = sorted({c for c in classes.values() if c not in loo})
    groups = [[tuple(p) for n, c in classes.items() if c == wc for p in mol.bonds[n]] for wc in wide]

    backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=M, basis=B)
    disk = HessianCache(OUTPUT_DIR / "hessian_cache", method=M, basis=B, elements=list(mol.elems))
    dfn = disk.wrap(lambda c, _b=backend: _b.run_hessian(c).hessian_bohr)
    mem: dict = {}

    def hfn(c, _m=mem, _f=dfn):
        k = np.asarray(c, float).round(9).tobytes()
        if k not in _m:
            _m[k] = _f(c)
        return _m[k]

    isos = A.build_isotopologues(mol, None)
    with contextlib.redirect_stdout(io.StringIO()):
        mr = A.multireference_diagnostic(list(mol.elems), theory) if A.RUN_DIAGNOSTICS else {}
        ctbl = build_correction_table_from_hessian(
            hfn(centre), centre, isos, hessian_fn=hfn,
            sigma_scale=A.MULTIREFERENCE_SIGMA_SCALE if (mr or {}).get("flagged") else 1.0,
            cubic_scheme="normal_mode", lam_freq_cm=0.0, freq_scale=A.FREQ_SCALE,
            harmonic_scheme=A.HARMONIC_SCHEME)[0]

    def score(x):
        q, r = mol.internal_coordinates(x), mol.internal_coordinates(ref)
        return {"bond_mA": float(A.rms_bond_error(mol, x, ref)[0]),
                "angle_deg": float(A.rms_angle_error(mol, x, ref)[0]),
                "primary_bond_mA": float(A.rms_bond_error(mol, x)[0]),
                "bonds": {n: round(1000 * (q[n] - r[n]), 2) for n in mol.bonds}}

    rec = {"widened_classes": wide, "s_mA": 1000 * s_unk, "theory": score(theory)}
    x0, i0, v0 = fit(mol, centre, ctbl, hfn, isos, sx_as_is)
    rec["as_is"] = {**score(x0), **i0, "archived_primary_mA": arch}
    if groups:
        xt, it, vt = fit(mol, centre, ctbl, hfn, isos, sx_tight)
        xp, ip, vp = fit(mol, centre, ctbl, hfn, isos, sx_tight, groups, s_unk)
        rec["tight"] = {**score(xt), **it}
        rec["per_bond"] = {**score(xp), **ip}
        rec["added_var_over_s2"] = {n: round((vp[n] - vt[n]) / s_unk ** 2, 4) for n in mol.bonds}
    res["molecules"][key] = rec
    OUT.write_text(json.dumps(res, indent=2, default=float))
    a, p = rec["as_is"], rec.get("per_bond", {})
    print(f"{key:<20} {M} {ARGS.recipe}: widen {wide or '-'} s={1000 * s_unk:.1f} mA | bond RMS vs ref:"
          f" theory {rec['theory']['bond_mA']:.2f} as-is {a['bond_mA']:.2f}"
          f" per-bond {p.get('bond_mA', float('nan')):.2f} tight {rec.get('tight', {}).get('bond_mA', float('nan')):.2f}"
          f" | angle as-is {a['angle_deg']:.3f} per-bond {p.get('angle_deg', float('nan')):.3f}"
          f" | as-is primary {a['primary_bond_mA']:.4f} vs archived {arch:.4f}", flush=True)
    if groups:
        print(f"    added var / s^2 (widened 1, others ~0): {rec['added_var_over_s2']}", flush=True)
