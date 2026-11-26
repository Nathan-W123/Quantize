"""Molecules the data cannot pin down: accuracy and reported widths.

    python scripts/archive/validation_thin_data.py <hf|b3lyp> <thin|full> <out.json>
        [molecule_key ...] [--output-dir DIR]   # theory geometries + Hessian cache

Each validation molecule (dev/validation_references.py) is cut to a few
isotopologues, so its constants have fewer independent pieces than its
structure has parameters (as chlorofluoromethane's two chlorine species do),
and scored with the benchmark's own machinery -- same theory geometry,
offsets pool, correction tables, hybrid_fit and mixed_estimation_fit -- at its
base and offsets configurations. ``full`` runs the same with every species,
for comparison. Sulfur dioxide is left out of ``thin``: its parent alone
determines its two parameters.

After every hybrid fit it asks the optimizer for the posterior it reports
(internal_coordinate_posterior) and records, per bond and angle, the error
against the r_e^SE reference, the reported 1-sigma width, z = error / width
and the data-fraction label: whether the engine says so when the data cannot
fix a coordinate.

The caches under --output-dir are copied to a scratch root first, so a run
never writes to, or waits on the lock of, the benchmark's own output.

Answer (2026-11-26, T-77; rerun at the D-161 vintage): see the README's
validation section. In short, with too few species the hybrid still moves
the bonds toward r_e^SE, its widths grow and cover the errors, and the labels
move from "data-constrained" to "mixed information"; with one species part of
the correction goes into the angles.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

#: Species kept per molecule in the ``thin`` sets, by label.
THIN = {
    "formaldehyde": ["H2C16O parent"],                       # 3 parameters, 2 moments
    "thioformaldehyde": ["H2CS"],                            # 3, 2
    "ketene": ["H2C=C=O (normal, 'Current work')", "[18O]"],  # 4, 4 (O only)
    "pyrimidine": ["m-C4H4N2", "[1-15N]", "[2-13C]"],       # 9, 6
    "thiophene": ["Thiophene", "[34S]", "[2-13C]"],          # 8, 6
}
_LEVELS = {"hf": ("hf", "6-31g"), "b3lyp": ("b3lyp", "6-31g(d)")}


def _args(argv):
    output_dir = _ROOT / "output"
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--output-dir":
            output_dir = Path(next(it))
        else:
            rest.append(a)
    if len(rest) < 3 or rest[0] not in _LEVELS or rest[1] not in ("thin", "full"):
        raise SystemExit(__doc__)
    return rest[0], rest[1], Path(rest[2]), rest[3:], output_dir


def main(argv=None):
    level, mode, out_path, keys, output_dir = _args(sys.argv[1:] if argv is None else argv)
    method, basis = _LEVELS[level]
    for p in (_ROOT / ".github", _ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    scratch = Path(tempfile.mkdtemp(prefix="thin_data_"))
    (scratch / "output").mkdir()
    shutil.copytree(output_dir / "hessian_cache", scratch / "output" / "hessian_cache")
    for f in output_dir.glob("theory_geometries_*.json"):
        shutil.copy(f, scratch / "output" / f.name)

    sys.argv = ["accuracy_upgrades_benchmark.py", f"method={method}", f"basis={basis}",
                "configs=base,offsets", "scored=validation", *keys]
    spec = importlib.util.spec_from_file_location(
        "accbench_thin", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    import dev.validation_references as val
    from backend.internal.internal_fit import InternalCoordinateSet

    def chosen():
        out = []
        for mol in val.VALIDATION_SET:
            if keys and mol.key not in keys:
                continue
            if mode == "thin":
                if mol.key not in THIN:
                    continue
                by = {sp.label: sp for sp in mol.species}
                mol = dataclasses.replace(mol, species=[by[k] for k in THIN[mol.key]])
            out.append(mol)
        return out

    created = []

    class Recording(bench.MolecularOptimizer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    bench.MolecularOptimizer = Recording
    plain_fit = bench.hybrid_fit
    rows, calls = [], {}

    def recorded_fit(mol, isos, *a, **k):
        created.clear()
        coords = plain_fit(mol, isos, *a, **k)
        opt = created[-1]
        n = calls[mol.key] = calls.get(mol.key, 0) + 1
        row = {"key": mol.key, "config": ("base", "offsets")[n - 1] if n <= 2 else f"call{n}",
               "mode": mode, "n_species": len(isos),
               "bond_rms_ma": bench.rms_bond_error(mol, coords)[0],
               "angle_rms_deg": bench.rms_angle_error(mol, coords)[0],
               "bond_pair_spread_ma": bench.bond_pair_spread_ma(mol, coords)}
        post = opt.internal_coordinate_posterior()
        if post is None or "error" in post:
            row["posterior_error"] = None if post is None else post["error"]
            rows.append(row)
            return coords
        x = np.asarray(opt.coords, dtype=float)
        cs = InternalCoordinateSet(x, list(mol.elems), bool(post["use_dihedrals"]))
        q_ref = np.asarray(cs.active_values(np.asarray(mol.geometry, dtype=float)), dtype=float)
        coords_out = []
        for name, kind, qf, qr, s, lab, fr in zip(
                post["names"], post["kinds"], post["values"], q_ref, post["std_err"],
                post["labels"], post["data_fraction"]):
            if kind in ("bond", "angle"):
                err, s = float(qf - qr), float(s)
                coords_out.append({"name": name, "kind": kind, "err": err, "sd": s,
                                   "z": err / s if s > 0 else None, "label": lab,
                                   "data_fraction": fr})
        row["coords"] = coords_out
        rows.append(row)
        return coords

    bench.hybrid_fit = recorded_fit
    bench._ROOT = scratch
    bench.HESSIAN_CACHE_DIR = scratch / "output" / "hessian_cache"
    bench.scored_molecules = lambda scored_set=None: chosen()
    bench.results_path = (lambda root, tag, extended=None, scored_set=None:
                          scratch / f"rows_{mode}_{tag}.json")
    bench.main()
    bench_rows = {}
    for f in scratch.glob(f"rows_{mode}_*.json"):     # the one results file main() wrote
        bench_rows = json.loads(f.read_text())
    out = {"level": level, "mode": mode, "code_vintage": bench.VINTAGE,
           "fits": rows, "benchmark_rows": bench_rows}
    out_path.write_text(json.dumps(out, indent=1))

    print(f"\n  {level} {mode}: hybrid fits, bonds vs r_e^SE and |z| of the reported widths")
    for r in rows:
        zs = [abs(c["z"]) for c in r.get("coords", []) if c["z"] is not None]
        bz = [abs(c["z"]) for c in r.get("coords", []) if c["z"] is not None and c["kind"] == "bond"]
        print(f"  {r['key']:17s} {r['config']:7s} n={r['n_species']:2d}  bonds {r['bond_rms_ma']:6.2f} mA"
              f"  angles {r['angle_rms_deg']:5.2f} deg  |z| bonds max {max(bz, default=float('nan')):5.2f}"
              f"  all max {max(zs, default=float('nan')):5.2f}  spread {r['bond_pair_spread_ma']:.4f}")
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
