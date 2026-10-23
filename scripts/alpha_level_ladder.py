"""Is the alpha correction's error set by the expansion geometry or the force field?

A's vibrational correction is the engine's weakest: the benchmark leaves it
~5 sigma out where B and C sit inside 1 sigma, and A's correction is
bend-dominated (56% of the term sum for water, 52% for ozone, against 10-31%
for B and C). Two candidate causes:

  (a) the expansion geometry -- A is ~240x more sensitive to the bond angle
      than C is (dA/dtheta = +18494 MHz/deg vs dC/dtheta = -282 for water), so
      a DFT minimum a few tenths of a degree off r_e could poison it;
  (b) the cubic force field along the bend -- the lowest-frequency mode, which
      carries a 1/omega^2 weight in the anharmonic term and is the hardest
      anharmonicity to compute.

Separating them does NOT need a quartic force field, and does not need any new
order of perturbation theory. Each level's correction is expanded about THAT
LEVEL'S OWN stationary point, so a level with a better optimised geometry
delivers a better expansion point for free. If the correction error tracks the
geometry error across the ladder, (a) is the driver; if it does not, (b) is.

Run:  python scripts/alpha_level_ladder.py

Measured (water, H2-16O, reference r_e = 0.95777 A / 104.48 deg, for which the
true correction is A -14948.9, B +2246.5, C +6940.6 MHz):

  level                      dr mA  dth deg      err A      err B      err C  termerr A
  b3lyp/6-31g(d)             10.89   -0.615    +7083.9     +672.3     +210.8      11.2%
  b3lyp/6-311+g(d,p)          4.29   +0.585     -715.8     +759.4      -47.8       1.1%
  b3lyp/cc-pvtz               3.58   +0.046    +2824.8     +188.7      -28.2       4.3%
  b3lyp/aug-cc-pvtz           4.08   +0.607    +1205.0     +343.9      -76.3       1.8%
  b3lyp/cc-pvqz               2.48   +0.405    +1608.0     +307.8      -42.0       2.4%
  pbe0/cc-pvtz                0.48   -0.105    +2197.9      -28.0     -153.4       3.4%
  pbe0/aug-cc-pvtz            1.07   +0.383     +919.8      +75.1     -197.5       1.4%
  pbe/cc-pvtz                11.93   -0.951    +5407.1      +56.3      +64.4       8.5%
  hf/cc-pvtz                -17.17   +1.521    -2626.1      +86.1     -437.6       4.0%

Answer: (b). The evidence is model-free, not a regression artefact --

  * pbe0/cc-pvtz has by far the best geometry (0.48 mA, -0.105 deg, i.e. sitting
    essentially ON r_e) and still leaves A +2198 MHz out.
  * pbe0/aug-cc-pvtz has a WORSE geometry (1.07 mA, +0.383 deg) and a BETTER A
    (+920). Same inversion between b3lyp/cc-pvqz (2.48 mA -> +1608) and
    b3lyp/aug-cc-pvtz (4.08 mA -> +1205).
  * corr(|err A|, geometry error) = +0.45, against +0.66 for C. C tracks the
    expansion point; A does not.

Fitting err_K = c0 + c_r*dr + c_th*dth (9 levels, 6 dof) puts numbers on it:

    K    c0 (MHz)   c_r /mA  c_th /deg     R^2  rms resid   |c0|/|true|
    A     +2695.4     +18.0    -3596.5   0.878      966.5        18.0%
    B       +68.5     +46.2     +451.6   0.480      189.4         3.0%
    C      -123.0     +19.4      -10.4   0.882       58.0         1.8%

c0 is what survives at zero geometry error. A keeps 18% of its own correction;
C keeps 1.8%. And A's error does not converge away with basis -- across the
four best levels it sits at 920-2198 MHz (6-15% of the correction) and
plateaus, while B and C settle to 30-80 MHz. The one level that beats the
plateau, b3lyp/6-311+g(d,p) at -716 MHz, sits on the opposite side of zero from
every other level: a cancellation, not convergence.

Propagating each level's correction error back through d(A,B,C)/d(r,theta) --
what the residual costs the fitted structure, single species, unweighted:

  level                       dr mA  dtheta deg
  pbe0/aug-cc-pvtz             0.16      -0.033
  b3lyp/6-311+g(d,p)           0.29       0.067
  pbe0/cc-pvtz                 0.35      -0.084
  b3lyp/aug-cc-pvtz            0.40      -0.026
  b3lyp/cc-pvqz                0.46      -0.042
  hf/cc-pvtz                  -0.56       0.091
  b3lyp/cc-pvtz                0.63      -0.092
  pbe/cc-pvtz                  1.06      -0.191
  b3lyp/6-31g(d)               1.76      -0.215

Dropping A from that fit does not help (pbe0/aug-cc-pvtz goes 0.16 mA /
-0.033 deg -> -0.36 mA / +0.069 deg): A's angle information still outweighs its
correction error at the good levels, so the fix is a better correction surface,
not discarding A.

Actionable: the correction level is already decoupled from the geometry level
(accuracy_upgrades_benchmark.py corr_method=/corr_basis=), and moving the
correction from cc-pvtz to aug-cc-pvtz roughly halves A's error at ~1.8x the
Hessian cost. Getting below the ~1000 MHz DFT plateau needs a correlated force
field (MP2/CCSD(T)), which the pyscf backend here does not expose.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from backend.spectral.harmonic_alpha import (  # noqa: E402
    compute_harmonic_alpha,
    normal_mode_hessian_derivatives,
)
from dev.monofluoro_references import (  # noqa: E402
    M_H,
    M_O16,
    WATER,
    WATER_R_E,
    WATER_THETA_E_DEG,
    _bent_xy,
)
from scripts.monofluoro_benchmark import start_geometry  # noqa: E402

LEVELS = [
    ("b3lyp", "6-31g(d)"),
    ("b3lyp", "6-311+g(d,p)"),
    ("b3lyp", "cc-pvtz"),
    ("b3lyp", "aug-cc-pvtz"),
    ("b3lyp", "cc-pvqz"),
    ("pbe0", "cc-pvtz"),
    ("pbe0", "aug-cc-pvtz"),
    ("pbe", "cc-pvtz"),
    ("hf", "cc-pvtz"),
]


def geom_params(coords):
    """Bond length (A) and bond angle (deg) of a bent XY2 geometry."""
    r = float(np.linalg.norm(coords[1] - coords[0]))
    v1, v2 = coords[1] - coords[0], coords[2] - coords[0]
    th = np.degrees(np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return r, th


def abc_jacobian(masses):
    """d(A,B,C)/d(r, theta) at the reference equilibrium, by central difference."""
    hr, ht = 1e-4, 1e-3
    d_r = (rotational_constants_mhz(_bent_xy(WATER_R_E + hr, WATER_THETA_E_DEG), masses)
           - rotational_constants_mhz(_bent_xy(WATER_R_E - hr, WATER_THETA_E_DEG), masses)) / (2 * hr)
    d_th = (rotational_constants_mhz(_bent_xy(WATER_R_E, WATER_THETA_E_DEG + ht), masses)
            - rotational_constants_mhz(_bent_xy(WATER_R_E, WATER_THETA_E_DEG - ht), masses)) / (2 * ht)
    return np.column_stack([d_r, d_th])


def run_level(method, basis, masses):
    """Optimise at (method, basis) and compute the alpha correction there."""
    backend = get_backend("pyscf_hf")(elems=list(WATER.elems), method=method, basis=basis)
    cache: dict = {}

    def hessian_fn(coords, _b=backend, _c=cache):
        key = np.asarray(coords, dtype=float).round(9).tobytes()
        if key not in _c:
            _c[key] = _b.run_hessian(coords).hessian_bohr
        return _c[key]

    # The whole point of the ladder: this level's own minimum, so the VPT2
    # expansion stays at a stationary point and no extra order of perturbation
    # theory (and no quartic force field) is needed.
    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(start_geometry(WATER))
        hess = hessian_fn(coords)
        derivs = normal_mode_hessian_derivatives(hessian_fn, coords, hess, list(masses))
        alpha, _, _, info = compute_harmonic_alpha(hess, coords, masses, mode_derivs=derivs)

    correction = np.array([0.5 * alpha[c] for c in "ABC"])
    term_sum = sum(abs(info[k]["A"]) for k in
                   ("alpha_centrifugal_mhz", "alpha_coriolis_mhz", "alpha_anharmonic_mhz"))
    return coords, correction, term_sum


def main():
    masses = np.array([M_O16, M_H, M_H])
    obs = np.array([s.abc_mhz for s in WATER.species if s.label == "H2-16O"][0], float)
    truth = rotational_constants_mhz(_bent_xy(WATER_R_E, WATER_THETA_E_DEG), masses) - obs

    print(f"  reference r_e = {WATER_R_E} A, {WATER_THETA_E_DEG} deg")
    print(f"  true correction  A {truth[0]:+9.1f}  B {truth[1]:+8.1f}  C {truth[2]:+8.1f}  MHz")
    print("  (correction = 1/2 sum_r alpha_r, expanded about each level's own minimum)\n")
    print(f"  {'level':24s}{'dr mA':>8s}{'dth deg':>9s}"
          + "".join(f"{'err ' + c:>11s}" for c in "ABC")
          + f"{'termerr A':>12s}{'s':>7s}")

    rows = []
    for method, basis in LEVELS:
        t0 = time.time()
        try:
            coords, correction, term_sum = run_level(method, basis, masses)
        except Exception as exc:  # a basis a level cannot handle should not kill the ladder
            print(f"  {method + '/' + basis:24s}  FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        r, th = geom_params(coords)
        err = correction - truth
        dr, dth = 1000.0 * (r - WATER_R_E), th - WATER_THETA_E_DEG
        rows.append((f"{method}/{basis}", dr, dth, err))
        print(f"  {method + '/' + basis:24s}{dr:8.2f}{dth:9.3f}"
              + "".join(f"{e:+11.1f}" for e in err)
              + f"{100 * abs(2 * err[0]) / term_sum:11.1f}%{time.time() - t0:7.0f}")

    if len(rows) < 4:
        return

    dr = np.array([row[1] for row in rows])
    dth = np.array([row[2] for row in rows])
    design = np.column_stack([np.ones_like(dr), dr, dth])
    combined_geom_err = np.hypot(dr / 10.0, dth)  # 10 mA ~ 1 deg, for a single scale

    print(f"\n  err_K = c0 + c_r*dr + c_th*dth   ({len(rows)} levels, {len(rows) - 3} dof)")
    print(f"  {'K':>3s}{'c0 (MHz)':>12s}{'c_r /mA':>10s}{'c_th /deg':>11s}"
          f"{'R^2':>8s}{'rms resid':>11s}{'|c0|/|true|':>13s}{'corr w/ geom':>14s}")
    for i, comp in enumerate("ABC"):
        y = np.array([row[3][i] for row in rows])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
        rho = float(np.corrcoef(np.abs(y), combined_geom_err)[0, 1])
        print(f"  {comp:>3s}{beta[0]:+12.1f}{beta[1]:+10.1f}{beta[2]:+11.1f}{r2:8.3f}"
              f"{np.sqrt(float((resid ** 2).mean())):11.1f}"
              f"{100 * abs(beta[0]) / abs(truth[i]):12.1f}%{rho:+14.3f}")

    jac = abc_jacobian(masses)
    print(f"\n  dABC/dr (MHz/A) {jac[:, 0].round(0)}   dABC/dtheta (MHz/deg) {jac[:, 1].round(0)}")
    print("\n  geometry error the residual correction error induces"
          " (single species, unweighted LSQ on A,B,C):")
    print(f"  {'level':24s}{'dr mA':>9s}{'dtheta deg':>12s}{'no-A dr mA':>13s}{'no-A dth':>11s}")
    for name, _, _, err in sorted(rows, key=lambda row: abs(row[3][0])):
        full, *_ = np.linalg.lstsq(jac, -err, rcond=None)
        drop_a, *_ = np.linalg.lstsq(jac[1:], -err[1:], rcond=None)
        print(f"  {name:24s}{1000 * full[0]:9.2f}{full[1]:12.3f}"
              f"{1000 * drop_a[0]:13.2f}{drop_a[1]:11.3f}")


if __name__ == "__main__":
    main()
