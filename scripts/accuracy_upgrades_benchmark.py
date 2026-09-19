"""Do the five accuracy upgrades actually make the numbers better?

Four things were identified as missing from the engine and are implemented
here; this script switches each on and measures, on the same molecules, at the
same level of theory, against the same references as the existing mixed-
estimation benchmark, so the answer is a delta and not an anecdote.

    offsets  bond-class offsets on the quantum prior      backend.spectral.bond_offsets
    elec     electronic (rotational g-tensor) correction  backend.spectral.electronic_g
    lam      large-amplitude modes carried as sigma       harmonic_alpha.lam_freq_cm
    corr     VPT2 corrections at a level of their own     hessian_fn at another level

Each configuration reports all three legs -- theory alone, mixed estimation,
hybrid -- because two of these upgrades (offsets, corr) change the inputs every
method sees, so they should move the baselines too. An upgrade that improved
only the hybrid would be suspicious.

Leave-one-out
-------------
The offsets are measured on the benchmark molecules, so scoring a molecule with
an offset table that saw it is circular. Every molecule is corrected using
offsets pooled from the *other* molecules only, and a bond class with no other
member is left uncorrected. That makes the reported gain a lower bound on what
a table calibrated on an external set would give, and never an artefact.

    python scripts/accuracy_upgrades_benchmark.py [method=hf] [basis=6-31g]
                                                  [configs=base,offsets,all]
                                                  [molecule_key ...]
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.bond_offsets import (  # noqa: E402
    leave_one_out_offsets,
    measure_offsets,
    offset_corrected_geometry,
    prior_sigma_for_molecule,
)
from backend.spectral.correction_models import electronic_delta_b  # noqa: E402
from backend.spectral.electronic_g import rotational_g_tensor  # noqa: E402
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
    _SIGMA_X_BY_LEVEL,
    corrected_targets,
    mixed_estimation_fit,
    rms_bond_error,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "hf", "6-31g"
CORR_METHOD, CORR_BASIS = None, None
WANT_CONFIGS: list[str] = []
_args: list[str] = []
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]
    elif _tok.startswith("corr_method="):
        CORR_METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("corr_basis="):
        CORR_BASIS = _tok.split("=", 1)[1]
    elif _tok.startswith("configs="):
        WANT_CONFIGS = _tok.split("=", 1)[1].split(",")
    else:
        _args.append(_tok)

SIGMA_X_ANG = _SIGMA_X_BY_LEVEL.get((METHOD.lower(), BASIS.lower()), 0.020)

#: Frequency below which a mode is treated as large-amplitude. Methyl torsions
#: in this set land at 200-300 cm-1; the next modes up are ordinary bends well
#: above 400, so the threshold is not sitting on top of a cluster.
LAM_FREQ_CM = 350.0

#: Fractional uncertainty on a computed g value. The OCS cross-check puts
#: RHF/6-31G at -0.0345 against an experimental -0.028 and RHF/cc-pVTZ at
#: -0.0272, so a small basis is good to roughly 25% on g. The electronic
#: correction is linear in g, so that carries straight through.
SIGMA_G_FRACTION = 0.30

#: Which upgrades each configuration enables.
CONFIGS = {
    "base":    {"offsets": False, "elec": False, "lam": False, "corr": False},
    "offsets": {"offsets": True,  "elec": False, "lam": False, "corr": False},
    "elec":    {"offsets": False, "elec": True,  "lam": False, "corr": False},
    "lam":     {"offsets": False, "elec": False, "lam": True,  "corr": False},
    "corr":    {"offsets": False, "elec": False, "lam": False, "corr": True},
    "all":     {"offsets": True,  "elec": True,  "lam": True,  "corr": False},
    "all+corr": {"offsets": True, "elec": True,  "lam": True,  "corr": True},
    # Measured best pairing: offsets carry the gain, while elec and lam each
    # cost more than they return on this set (ozone 2.32 -> 3.23 to the RHF
    # g-tensor, isocyanic acid 18.73 -> 29.12 to the large-amplitude widening).
    "offsets+corr": {"offsets": True, "elec": False, "lam": False, "corr": True},
}


def electronic_shifted_isotopologues(isos, g_tensor, total_mass_amu):
    """Isotopologues with the electronic correction folded into the constants.

    Both methods are handed the same numbers this way, rather than one of them
    applying the correction internally and the other not. The correction is
    ``delta = -(m_e/m_p)*g*B0``, added to B0 exactly as ``resolve_corrections``
    would, and its own uncertainty goes into sigma in quadrature.
    """
    out = []
    for iso in isos:
        new = dict(iso)
        obs = np.asarray(iso["obs_constants"], dtype=float).copy()
        sig = np.asarray(iso["sigma_constants"], dtype=float).copy()
        for k, comp in enumerate(np.asarray(iso["component_indices"], dtype=int)):
            g_val = g_tensor.get("ABC"[int(comp)])
            if g_val is None or not np.isfinite(g_val):
                continue
            delta = electronic_delta_b(float(obs[k]), total_mass_amu,
                                       g_value=float(g_val))
            obs[k] = float(obs[k]) + delta
            sig[k] = float(np.hypot(sig[k], abs(delta) * SIGMA_G_FRACTION))
        new["obs_constants"] = obs
        new["sigma_constants"] = sig
        out.append(new)
    return out


def hybrid_fit(mol, isos, prior_coords, ctbl, sigma_x_ang):
    """The engine's answer, with the prior centred and widened as handed.

    ``prior_target_coords`` is what makes the prior's centre follow
    ``prior_coords``. Without it the quantum term is the level of theory's own
    gradient, so the prior's minimum stays at the level of theory's minimum
    however the starting structure is chosen, and a bias-corrected prior is
    discarded on the first step -- measured on vinyl fluoride as 18.01 mA
    against 7.95 with the centre moved.

    Passing it unconditionally is deliberate. Where the prior *is* the theory
    geometry the two agree anyway, since g is zero at that minimum and
    H (x - x_theory) is its harmonic expansion, so the base configuration is
    not quietly running a different prior from the others.
    """
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(prior_coords, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=float(sigma_x_ang),
        prior_target_coords=np.asarray(prior_coords, dtype=float),
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def theory_geometries(pool):
    """Theory geometry per molecule, plus each one's bond-class offsets.

    Cached to disk: it is the same optimisation for every configuration, and
    the leave-one-out table needs all of them before any molecule can be run.
    """
    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    cache_path = _ROOT / "output" / f"theory_geometries_{tag}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    for mol in pool:
        if mol.key in cache:
            continue
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        with contextlib.redirect_stdout(io.StringIO()):
            coords = backend.optimise(start_geometry(mol))
        cache[mol.key] = {
            "coords": np.asarray(coords, dtype=float).tolist(),
            "offsets": {k: list(map(float, v))
                        for k, v in measure_offsets(mol, coords).items()},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"  [theory] {mol.key}", flush=True)
    return cache


def main() -> None:
    pool = (list(MOLECULES) + list(MOLECULES_SET2)
            + [WATER_SET[0], OZONE, ISOCYANIC_ACID])
    wanted = _args or [m.key for m in pool]
    configs = [c for c in (WANT_CONFIGS or ["base", "offsets", "elec", "lam", "all"])
               if c in CONFIGS]

    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    out_path = _ROOT / "output" / f"accuracy_upgrades_{tag}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}

    print(f"  {METHOD.upper()}/{BASIS}, sigma_x = {SIGMA_X_ANG} A")
    print(f"  configs: {', '.join(configs)}")
    if CORR_METHOD or CORR_BASIS:
        print(f"  corrections level: {CORR_METHOD or METHOD}/{CORR_BASIS or BASIS}")
    print()

    theory = theory_geometries(pool)
    per_molecule_offsets = {k: {c: [x / 1.0 for x in v]
                                for c, v in rec["offsets"].items()}
                            for k, rec in theory.items()}

    for mol in pool:
        if mol.key not in wanted:
            continue
        rec = out.setdefault(mol.key, {})
        todo = [c for c in configs if c not in rec]
        if not todo:
            continue
        t0 = time.time()

        theory_coords = np.asarray(theory[mol.key]["coords"], dtype=float)
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        hess_cache: dict = {}

        def hessian_fn(coords_ang, _b=backend, _c=hess_cache):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _b.run_hessian(coords_ang).hessian_bohr
            return _c[key]

        # The correction level, when it is allowed to differ from the geometry
        # level. Corrections converge with basis and correlation treatment far
        # faster than geometries do, so the usual practice is to compute alpha
        # somewhere cheaper than the structure -- here the structure is the
        # cheap thing and the correction is the one worth improving.
        corr_backend = None
        corr_cache: dict = {}
        if CORR_METHOD or CORR_BASIS:
            corr_backend = get_backend("pyscf_hf")(
                elems=list(mol.elems), method=CORR_METHOD or METHOD,
                basis=CORR_BASIS or BASIS)

        def corr_hessian_fn(coords_ang, _b=corr_backend, _c=corr_cache):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _b.run_hessian(coords_ang).hessian_bohr
            return _c[key]

        # Offsets, leave-one-out, and the geometry they produce.
        loo = leave_one_out_offsets(per_molecule_offsets, mol.key)
        offset_coords = offset_corrected_geometry(mol, theory_coords, loo)
        # Removing the bias narrows the prior, so sigma_x has to follow it or
        # the optimiser keeps treating a corrected prior as a loose one. Bond
        # by bond, because a class leave-one-out could not calibrate is not
        # narrowed at all -- quoting the corrected spread for it claimed a
        # precision the prior did not have and cost chlorofluoromethane
        # 13.16 -> 19.71 mA. Still leave-one-out, so the width is never
        # informed by the answer.
        sigma_x_offset = prior_sigma_for_molecule(
            mol, per_molecule_offsets, mol.key)

        # A correction table per (geometry, lam) pair actually needed. The
        # Hessian cache is shared, so the second lam variant is nearly free.
        tables: dict = {}

        def table_for(coords, lam, corr, _tabs=tables):
            key = (np.asarray(coords, dtype=float).round(8).tobytes(),
                   bool(lam), bool(corr))
            if key not in _tabs:
                hfn = corr_hessian_fn if corr else hessian_fn
                isos_local = build_isotopologues(mol, None)
                with contextlib.redirect_stdout(io.StringIO()):
                    _tabs[key] = build_correction_table_from_hessian(
                        hfn(coords), np.asarray(coords, dtype=float),
                        isos_local, hessian_fn=hfn,
                        cubic_scheme="normal_mode",
                        lam_freq_cm=LAM_FREQ_CM if lam else 0.0)
            return _tabs[key]

        g_tensor = None
        for name in todo:
            cfg = CONFIGS[name]
            prior = offset_coords if cfg["offsets"] else theory_coords
            sigma_x = sigma_x_offset if cfg["offsets"] else SIGMA_X_ANG
            isos = build_isotopologues(mol, None)

            if cfg["elec"]:
                if g_tensor is None:
                    g_tensor = rotational_g_tensor(
                        list(mol.elems), theory_coords, np.asarray(mol.masses),
                        method=METHOD, basis=BASIS)
                isos = electronic_shifted_isotopologues(
                    isos, g_tensor, float(np.sum(np.asarray(mol.masses))))

            if cfg["corr"] and corr_backend is None:
                raise SystemExit(
                    "configs including 'corr' need corr_method=/corr_basis= "
                    "so the correction level differs from the geometry level")
            ctbl, info = table_for(prior, cfg["lam"], cfg["corr"])
            targets = corrected_targets(mol, isos, ctbl)

            # Both methods get the same prior centre and the same width; the
            # comparison is about mechanism, not about who was told to trust
            # the theory more.
            me_geom, _chi2 = mixed_estimation_fit(mol, targets, prior, sigma_x)
            hyb = hybrid_fit(mol, isos, prior, ctbl, sigma_x)

            entry = {
                "theory_rms_ma": rms_bond_error(mol, prior)[0],
                "me_rms_ma": rms_bond_error(mol, me_geom)[0],
                "hybrid_rms_ma": rms_bond_error(mol, hyb)[0],
                "n_species": len(isos),
                "sigma_x_ang": float(sigma_x),
                "corr_level": (f"{CORR_METHOD or METHOD}/{CORR_BASIS or BASIS}"
                               if cfg["corr"] else f"{METHOD}/{BASIS}"),
                "lam_modes_cm": sorted({round(w, 1) for v in info.get("lam", {}).values()
                                        for w in v.get("modes_cm", [])}),
                "g_tensor": g_tensor if cfg["elec"] else None,
                "loo_offsets_ma": {k: round(v * 1000.0, 2) for k, v in loo.items()}
                                  if cfg["offsets"] else None,
            }
            rec[name] = entry
            out[mol.key] = rec
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"  {mol.name:<22} {name:<9} theory {entry['theory_rms_ma']:6.2f}"
                  f"   ME {entry['me_rms_ma']:6.2f}"
                  f"   hybrid {entry['hybrid_rms_ma']:6.2f} mA", flush=True)
        print(f"    ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
