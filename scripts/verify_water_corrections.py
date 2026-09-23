"""Is the 'true' water correction actually true?

Every claim that the engine's rovibrational correction is 5-sigma wrong was
measured against a "true" correction built as B_e(r_e structure) - B_0(observed).
That makes the whole finding hostage to two inputs -- water's equilibrium
structure and its observed ground-state constants -- so both are checked here
against their published sources, the structure sensitivity is propagated, and
the engine's own computed correction is recomputed alongside.

    python scripts/verify_water_corrections.py [method=b3lyp] [basis=6-31g(d)]
                                               [--no-engine]

Result
------
Both inputs check out. The stored B_0 matches what NIST CCCBDB quotes to
0.002%, and the stored r_e matches Csaszar et al. (J. Chem. Phys. 122, 214305,
2005) and Hoy & Bunker, which CCCBDB lists as water's experimental geometry.
Fixing a 0.2 mA / 0.013 deg drift between the stored coordinates and the r_e
the module documents moved water's benchmark row by 0.2 mA and changed nothing
about the verdict below.

Engine correction against truth, B3LYP/6-31G(d), H2-16O (MHz):

            computed       true      error      sigma   n_sigma
    A       -17034.8   -14948.9    -2085.9     2555.2       0.8
    B       +12089.2    +2246.5    +9842.6     1813.4       5.4
    C        +9345.5    +6940.6    +2404.9     1401.8       1.7

and the B failure is 5.1-5.6 sigma under *every* published equilibrium
structure, including the discarded textbook one -- across all of them the truth
moves by at most 270 MHz against a 9843 MHz error. So the diagnosis is a
statement about the correction, not about the reference.

The reference-free check agrees. Water is planar, so its equilibrium inertial
defect is zero by geometry; B_0 carries 0.0492 amu A^2 of zero-point defect and
the engine's correction removes 81% of it, leaving 0.0091. That residual needs
no r_e at all to establish, and it is why the planarity constraint could not
reach the angle bias: the correction error is very nearly defect-preserving.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

from backend.spectral.centrifugal_distortion import (  # noqa: E402
    _INERTIA_TO_MHZ,
    rotational_constants_mhz,
)
from dev.monofluoro_references import M_D, M_H, M_O16, WATER  # noqa: E402

CM = 29979.2458  # MHz per cm-1

#: Published equilibrium structures for water, newest determination first.
#:
#: Hoy & Bunker (J. Mol. Spectrosc. 74, 1979) is what NIST CCCBDB lists as the
#: experimental geometry of H2O: r_e = 0.958 A, angle 104.4776 +/- 0.0019 deg.
#: Csaszar et al. (J. Chem. Phys. 122, 214305, 2005) redetermine the
#: spectroscopic (nonadiabatic) equilibrium structure as 0.95777 A / 104.48 deg.
#: The 0.9572 A / 104.52 deg pair quoted in textbooks is an older effective
#: structure, not a modern r_e; it is carried to show what a wrong reference
#: would do to the answer.
STRUCTURES = [
    ("repo reference",      0.957982, 104.4929),
    ("Csaszar 2005 r_e(sp)", 0.95777, 104.48),
    ("Hoy & Bunker / CCCBDB", 0.9578,  104.4776),
    ("textbook r_0-era",     0.9572,   104.52),
]

#: Quoted uncertainty on the Hoy & Bunker angle, and a generous one on r.
D_R_ANG, D_THETA_DEG = 0.0002, 0.0019

MASSES = {"H2-16O": np.array([M_O16, M_H, M_H]),
          "D2-16O": np.array([M_O16, M_D, M_D])}
OBS = {sp.label: np.asarray(sp.abc_mhz, dtype=float) for sp in WATER.species}


def water_coords(r_ang: float, theta_deg: float) -> np.ndarray:
    """C2v water, oxygen at the origin."""
    half = np.radians(theta_deg) / 2.0
    return np.array([[0.0, 0.0, 0.0],
                     [r_ang * np.sin(half), r_ang * np.cos(half), 0.0],
                     [-r_ang * np.sin(half), r_ang * np.cos(half), 0.0]])


def true_correction(label: str, r_ang: float, theta_deg: float) -> np.ndarray:
    """B_e - B_0 in MHz: what the rovibrational correction has to supply."""
    return rotational_constants_mhz(water_coords(r_ang, theta_deg),
                                    MASSES[label]) - OBS[label]


def structure_sensitivity(label, r_ang, theta_deg,
                          d_r=D_R_ANG, d_theta=D_THETA_DEG) -> np.ndarray:
    """How far the true correction moves within the reference's own error bars."""
    base = true_correction(label, r_ang, theta_deg)
    dr = true_correction(label, r_ang + d_r, theta_deg) - base
    dt = true_correction(label, r_ang, theta_deg + d_theta) - base
    return np.hypot(dr, dt)


def engine_corrections(method: str, basis: str) -> dict:
    """The engine's own VPT2 correction for water, at its own optimised geometry."""
    import dev.pyscf_backend  # noqa: F401  (registers "pyscf_hf")
    from backend.registry import get_backend
    from backend.spectral.harmonic_alpha import build_correction_table_from_hessian
    from scripts.monofluoro_benchmark import build_isotopologues, start_geometry

    backend = get_backend("pyscf_hf")(elems=list(WATER.elems),
                                      method=method, basis=basis)
    cache: dict = {}

    def hessian_fn(coords_ang):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in cache:
            cache[key] = backend.run_hessian(coords_ang).hessian_bohr
        return cache[key]

    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(start_geometry(WATER))
        isos = build_isotopologues(WATER, None)
        table, _info = build_correction_table_from_hessian(
            hessian_fn(coords), coords, isos, hessian_fn=hessian_fn,
            cubic_scheme="normal_mode")
    r = float(np.linalg.norm(coords[1] - coords[0]))
    v1, v2 = coords[1] - coords[0], coords[2] - coords[0]
    theta = np.degrees(np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return {"table": table, "r_ang": r, "theta_deg": theta}


def main() -> None:
    method, basis, want_engine = "b3lyp", "6-31g(d)", True
    for tok in sys.argv[1:]:
        if tok.startswith("method="):
            method = tok.split("=", 1)[1]
        elif tok.startswith("basis="):
            basis = tok.split("=", 1)[1]
        elif tok == "--no-engine":
            want_engine = False

    print("Observed B_0 against the source NIST CCCBDB quotes "
          "(1966Herzberg: 27.877, 14.512, 9.285 cm-1)")
    cccbdb = np.array([27.877, 14.512, 9.285]) * CM
    obs = OBS["H2-16O"]
    print(f"   repo : {obs[0]:10.1f} {obs[1]:10.1f} {obs[2]:10.1f}  MHz")
    print(f"   cccbdb:{cccbdb[0]:10.1f} {cccbdb[1]:10.1f} {cccbdb[2]:10.1f}  MHz")
    print("   diff  : " + "  ".join(f"{100*(o-c)/c:+8.4f}%"
                                    for o, c in zip(obs, cccbdb)))
    print()

    print("True correction B_e - B_0, per published equilibrium structure (MHz)")
    header = f"   {'structure':24s} {'species':9s} " + \
             "".join(f"{c:>12s}" for c in "ABC")
    print(header)
    trues = {}
    for name, r, th in STRUCTURES:
        for label in ("H2-16O", "D2-16O"):
            d = true_correction(label, r, th)
            trues[(name, label)] = d
            print(f"   {name:24s} {label:9s} " +
                  "".join(f"{v:+12.1f}" for v in d))
    print()

    print(f"Spread of the true correction within the reference's own error bars "
          f"(+/-{D_R_ANG} A, +/-{D_THETA_DEG} deg)")
    for label in ("H2-16O", "D2-16O"):
        s = structure_sensitivity(label, *STRUCTURES[0][1:])
        print(f"   {label:9s} " + "".join(f"{v:12.1f}" for v in s))
    print()

    print("Spread across the three modern determinations (max - min, MHz)")
    for label in ("H2-16O", "D2-16O"):
        stack = np.array([trues[(n, label)] for n, _, _ in STRUCTURES[:3]])
        print(f"   {label:9s} " +
              "".join(f"{v:12.1f}" for v in stack.max(0) - stack.min(0)))
    print()

    if not want_engine:
        return

    print(f"Engine correction at {method.upper()}/{basis} ...", flush=True)
    eng = engine_corrections(method, basis)
    print(f"   optimised r = {eng['r_ang']:.5f} A, "
          f"theta = {eng['theta_deg']:.3f} deg "
          f"(r_e = {STRUCTURES[1][1]:.5f} A, {STRUCTURES[1][2]:.3f} deg)")
    print()
    print(f"   {'species':9s} {'comp':5s} {'computed':>11s} {'true':>11s} "
          f"{'error':>11s} {'sigma':>11s} {'n_sigma':>8s} {'ratio':>7s}")
    for label in ("H2-16O", "D2-16O"):
        entries = eng["table"].get(label, {})
        truth = trues[(STRUCTURES[1][0], label)]
        for i, comp in enumerate("ABC"):
            e = entries.get(comp)
            if e is None:
                continue
            calc = 0.5 * float(e["alpha_sum_mhz"])
            sig = float(e["sigma_mhz"])
            err = calc - truth[i]
            ratio = calc / truth[i] if truth[i] else float("nan")
            print(f"   {label:9s} {comp:5s} {calc:+11.1f} {truth[i]:+11.1f} "
                  f"{err:+11.1f} {sig:11.1f} {abs(err)/sig:8.1f} {ratio:7.2f}")
    print()

    # The point of the exercise: does the verdict depend on which published
    # structure the truth is taken from? If the n_sigma range straddles 1 for
    # a component, that component's verdict is a statement about the reference
    # rather than about the engine.
    print("   n_sigma under each candidate reference structure")
    print(f"   {'species':9s} {'comp':5s} " +
          "".join(f"{n.split()[0]:>14s}" for n, _, _ in STRUCTURES))
    for label in ("H2-16O", "D2-16O"):
        entries = eng["table"].get(label, {})
        for i, comp in enumerate("ABC"):
            e = entries.get(comp)
            if e is None:
                continue
            calc = 0.5 * float(e["alpha_sum_mhz"])
            sig = float(e["sigma_mhz"])
            cells = []
            for name, _, _ in STRUCTURES:
                cells.append(abs(calc - trues[(name, label)][i]) / sig)
            print(f"   {label:9s} {comp:5s} " +
                  "".join(f"{v:14.1f}" for v in cells))
    print()

    # A check that needs no reference structure at all. Water is planar, so its
    # equilibrium inertial defect I_c - I_a - I_b is zero by geometry. B_0 has a
    # defect of ~0.05 amu A^2 from zero-point motion; a correct correction has to
    # take that to zero. Whatever is left is correction error that no choice of
    # published r_e can explain away.
    print("   Inertial defect I_c - I_a - I_b (amu A^2), zero at equilibrium")
    print(f"   {'species':9s} {'B_0':>12s} {'B_0+true':>12s} {'B_0+engine':>12s}")
    for label in ("H2-16O", "D2-16O"):
        entries = eng["table"].get(label, {})
        calc = np.array([0.5 * float(entries[c]["alpha_sum_mhz"])
                         for c in "ABC"])
        obs = OBS[label]
        row = []
        for abc in (obs, obs + trues[(STRUCTURES[1][0], label)], obs + calc):
            i = _INERTIA_TO_MHZ / np.asarray(abc, dtype=float)
            row.append(i[2] - i[0] - i[1])
        print(f"   {label:9s} " + "".join(f"{v:12.4f}" for v in row))


if __name__ == "__main__":
    main()
