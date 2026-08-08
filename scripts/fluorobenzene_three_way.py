"""Fluorobenzene: theory alone vs experiment alone vs hybrid.

Fluorobenzene is the case the SVD split exists for. 19F is the only stable
fluorine isotope, so no F substitution is possible and the C-F bond length is
constrained by *no* isotopologue -- it can only come from the quantum surface.
The ring and C-H distances, by contrast, are well determined by the 13C and D
species. So the two groups of parameters test different things, and are reported
separately.

Why the constants here are synthesised
--------------------------------------
configs/fluorobenzene.yaml cannot support this comparison. Its listed B_0 values
disagree with its own geometry block by up to 1726 sigma (a systematic ~+86 MHz
on B across every isotopologue), and its quoted ring angles sum to 723.59 deg
rather than the 720 deg a closed planar hexagon requires. A fit to that data
stalls near 50 MHz residual and wanders to ~48 mA of bond error, which measures
the inconsistency rather than anything about the method.

So the geometry block is taken as the truth structure and exact rotational
constants are computed from it for all eight isotopologues. That makes the
experiment leg a validation of the machinery rather than an independent
measurement -- but it leaves the C-F comparison completely non-circular, because
no amount of isotopic data determines it.

Theory is RHF/STO-3G via PySCF: a real electronic-structure calculation with a
real, and fairly large, basis-set error.

    python scripts/fluorobenzene_three_way.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from runner.run_generic import main as run_generic_main  # noqa: E402
from runner.usability import prepare_run_directory, validate_config  # noqa: E402

_CONFIG = _ROOT / "configs" / "fluorobenzene.yaml"

# Atom order is fixed by the config: F, C1..C6, H2, H3, H4, H5, H6
_BONDS = {
    "r(C1-F)": (1, 0), "r(C1-C2)": (1, 2), "r(C2-C3)": (2, 3), "r(C3-C4)": (3, 4),
    "r(C2-H2)": (2, 7), "r(C3-H3)": (3, 8), "r(C4-H4)": (4, 9),
}
_ANGLES = {
    "a(F-C1-C2)": (0, 1, 2), "a(C1-C2-C3)": (1, 2, 3),
    "a(C2-C3-C4)": (2, 3, 4), "a(C3-C4-C5)": (3, 4, 5),
}
# Every bond except C-F is fixed by isotopic substitution.
_SUBSTITUTION_BONDS = [k for k in _BONDS if k != "r(C1-F)"]


def internals(coords) -> dict[str, float]:
    x = np.asarray(coords, dtype=float)
    out = {}
    for name, (i, j) in _BONDS.items():
        out[name] = float(np.linalg.norm(x[j] - x[i]))
    for name, (i, j, k) in _ANGLES.items():
        u, v = x[i] - x[j], x[k] - x[j]
        cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        out[name] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return out


def errors(coords, truth) -> dict[str, float]:
    got, ref = internals(coords), internals(truth)
    sub = [(got[k] - ref[k]) * 1000 for k in _SUBSTITUTION_BONDS]
    ang = [got[k] - ref[k] for k in _ANGLES]
    return {
        "cf_ma": (got["r(C1-F)"] - ref["r(C1-F)"]) * 1000,
        "rms_sub_bond_ma": float(np.sqrt(np.mean(np.square(sub)))),
        "rms_angle_deg": float(np.sqrt(np.mean(np.square(ang)))),
    }


def _base_config():
    return copy.deepcopy(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))


def synthetic_config(truth, start):
    """Config whose constants are exactly consistent with the truth structure."""
    cfg = _base_config()
    cfg["geometry"] = {"method": "coords", "coords_angstrom": np.asarray(start).tolist()}
    for iso in cfg["isotopologues"]:
        masses = np.asarray(iso["masses"], dtype=float)
        abc = rotational_constants_mhz(truth, masses)
        iso["obs_b0_mhz"] = [float(v) for v in abc]
        iso["alpha_mhz"] = [0.0, 0.0, 0.0]      # constants are already equilibrium
        iso["sigma_mhz"] = [0.05, 0.05, 0.05]
    cfg["output"] = {"root": "output/runs", "artifacts": False}
    cfg["preset"] = "FAST_DEBUG"
    return cfg


def run_pipeline(cfg, **optimizer_overrides):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("optimizer", {}).update(optimizer_overrides)
    validate_config(cfg)
    prepare_run_directory(cfg, _CONFIG)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_generic_main(cfg)["best"]["coords"]


def main() -> None:
    truth = np.asarray(_base_config()["geometry"]["coords_angstrom"], dtype=float)

    # Start displaced from the truth so every leg has real work to do.
    rng = np.random.default_rng(7)
    start = truth + rng.normal(0.0, 0.02, size=truth.shape)
    start[:, 2] = 0.0                                   # keep it planar

    print("  Fluorobenzene, 8 isotopologues (parent + 4x 13C + 3x D).")
    print("  Constants synthesised from the truth structure -- see the module docstring")
    print("  for why the shipped config cannot be used directly.\n")
    print(f"  {'':<28}{'C-F (mA)':>11}{'ring+CH RMS':>14}{'angle RMS':>12}{'time':>8}")
    print(f"  {'':<28}{'no isotope':>11}{'(mA)':>14}{'(deg)':>12}")
    print("  " + "-" * 75)

    e = errors(start, truth)
    print(f"  {'starting guess':<28}{e['cf_ma']:>+11.1f}{e['rms_sub_bond_ma']:>14.1f}"
          f"{e['rms_angle_deg']:>12.3f}{'-':>8}")

    t0 = time.time()
    backend = get_backend("pyscf_hf")(elems=_base_config()["elements"],
                                      method="hf", basis="sto-3g")
    theory = backend.optimise(start)
    e = errors(theory, truth)
    print(f"  {'THEORY alone (RHF/STO-3G)':<28}{e['cf_ma']:>+11.1f}"
          f"{e['rms_sub_bond_ma']:>14.1f}{e['rms_angle_deg']:>12.3f}"
          f"{time.time() - t0:>7.0f}s")

    cfg = synthetic_config(truth, start)

    t0 = time.time()
    cfg_e = copy.deepcopy(cfg)
    cfg_e["quantum"]["backend"] = "none"
    e = errors(run_pipeline(cfg_e, max_iter=60), truth)
    print(f"  {'EXPERIMENT alone':<28}{e['cf_ma']:>+11.1f}"
          f"{e['rms_sub_bond_ma']:>14.1f}{e['rms_angle_deg']:>12.3f}"
          f"{time.time() - t0:>7.0f}s")

    t0 = time.time()
    cfg_h = copy.deepcopy(cfg)
    cfg_h["quantum"].update({"backend": "pyscf_hf", "method": "hf", "basis": "sto-3g"})
    e = errors(run_pipeline(cfg_h, max_iter=60, hess_recalc_every=8), truth)
    print(f"  {'HYBRID':<28}{e['cf_ma']:>+11.1f}"
          f"{e['rms_sub_bond_ma']:>14.1f}{e['rms_angle_deg']:>12.3f}"
          f"{time.time() - t0:>7.0f}s")

    print("\n  The C-F column is the one that matters: no isotopologue constrains it, so")
    print("  it can only come from the quantum surface. The ring and C-H columns are")
    print("  determined by the 13C and D substitutions, and since the constants were")
    print("  synthesised from the truth structure they mainly check the machinery.")


if __name__ == "__main__":
    main()
