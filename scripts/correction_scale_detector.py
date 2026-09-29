"""Can the data tell us the correction is biased, with no reference structure?

Every bias detector in this engine looks at residuals, and residuals are exactly
where a correction bias does not show up. A bias that is common-mode across a
molecule's isotopologues moves the fitted structure instead: the fit absorbs it,
chi-square stays flat -- measured, it never leaves [0.25, 4] anywhere on the
reference set -- and nothing reports a problem while the answer is wrong.

The opening this leaves is that a correction is NOT common-mode. Delta_K depends
on the isotopologue through its masses, and the geometry does not. So the two
have different isotopic signatures, and with enough isotopologues they separate.

This fits a scale factor on each component's correction alongside the geometry:

    B_calc(geometry) = B_0 + s_K * Delta_K(isotopologue)

s_K = 1 means the correction is believed. s_K away from 1 is the data saying the
correction is wrong, measured without any reference structure.

The validation is the point. Water and ozone have published equilibrium
structures, so their TRUE scales are known -- and the detector never sees them.
If the fitted s_K lands on the true ratio, the detector works.

    python scripts/correction_scale_detector.py [method=b3lyp] [basis=6-31g(d)]

ANSWER: the data cannot support it, and now we know by how much
---------------------------------------------------------------
The detector works in the sense that its point estimates are roughly right.
On water it returns s_A = 2.23, s_B = 0.73, s_C = 0.93 against true values of
1.87, 0.76, 0.98 -- recovered from the measurements alone, having never seen an
equilibrium structure. That is the encouraging half.

The discouraging half is the error bars on those numbers, and they settle the
question. The 2-sigma detection limit on |s - 1|, i.e. the smallest correction
bias this data could distinguish from none:

    water, 2 measured species            A 1020%   B 833%   C 163%
    water, every isotopologue that exists A  596%   B 584%   C  99%
    ozone, 5 measured species            A 2679%   B 805%   C 622%

Water's actual A bias was 87%. It is a factor of seven below what the two
measured species could ever have detected, and still below what a complete
isotopic study of water -- H2, D2, HD, each with 16-O and 18-O -- could detect.
Ozone is an order of magnitude worse because 16-O -> 18-O changes a mass by 12.5%
where H -> D changes one by 100%, so its species barely differ.

So bias detection here is not badly built. The information is not present. The
correction's isotopic signature is very nearly parallel to the geometry's, and
the correction's own uncertainty is large, so the fit cannot tell a scaled
correction from a moved structure. Adding isotopologues buys a factor of two and
then saturates.

What follows from that
----------------------
Three consequences, and they redirect the whole approach:

1. Stop trying to detect bias from the data. It is out of reach for small
   molecules and the limit above says by how much, per molecule, before any
   effort is spent.
2. The only channels with enough information are external equilibrium
   references and symmetry. That is not a coincidence about this project's
   history -- the harmonic-frame bug was found by an r_e reference and then
   proved by NH3's A == B degeneracy, and nothing internal to the data had
   flagged it in months of running.
3. Since bias cannot be detected, it has to be prevented. Effort belongs in the
   correction physics and in acquiring r_e references, not in better residual
   diagnostics.

What was tried, and how far it got
----------------------------------
The degeneracy has two terms. The correction's isotopic signature is nearly
parallel to the geometry's, AND the correction's own sigma is large. Data can
attack the first; only physics can attack the second.

Centrifugal distortion attacks the first, and it is the one channel that
constrains the geometry WITHOUT passing through the B0-to-Be correction. The
full Kivelson-Wilson tau tensor is now computable (see tau_components_mhz; the
off-diagonal components were structurally absent before) and validated against
ozone's published reduction-free values across five isotopologues, to 2.3% on
tau_aaaa, 9.9% on tau_bbbb and 12.6% on tau_cccc. Feeding it in as data at a
13% sigma:

    ozone, 2-sigma limit on |s-1|      A        B        C
      rotational constants only     2937%     793%     599%
      plus tau as data              1001%     674%     560%
      improvement                    2.9x     1.2x     1.1x

Real, measured, and not enough. A stays an order of magnitude above the ~90%
biases that actually occur, because tau does nothing about the second term.

Which leaves the second term, and it has a name: cancellation. The correction
sigma is a fraction of the SUM of the magnitudes of its three contributions, so
a component whose terms cancel eightfold carries eight times the uncertainty of
one that does not. The detection limits rank exactly that way -- ozone's A
cancels about fivefold and sits at 2937%, its C cancels 1.5-fold and sits at
599%. Making a component detectable means making its terms cancel less, which
means computing them correctly, which is the same conclusion arrived at from
the other direction.

The detection limit itself is the deliverable. It needs no measured constants --
only masses, a geometry and the correction sigmas -- so it can be computed for
species nobody has made, and it answers the question worth asking before
trusting a structure: not "is there a bias" but "could I have seen one".
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    rotational_constants_mhz,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    compute_harmonic_alpha,
    normal_mode_hessian_derivatives,
)
from dev.monofluoro_references import (  # noqa: E402
    OZONE,
    WATER,
    WATER_R_E,
    WATER_THETA_E_DEG,
    _bent_xy,
)
from scripts.monofluoro_benchmark import start_geometry  # noqa: E402

#: Published equilibrium structures. Used ONLY to score the detector, never as
#: an input to it.
TRUTH = {
    "water": (WATER, _bent_xy(WATER_R_E, WATER_THETA_E_DEG)),
    "ozone": (OZONE, _bent_xy(1.27276, 116.7542)),
}


def species(mol):
    parent = np.asarray(mol.masses, dtype=float)
    return [(sp.label, sp.masses(parent), np.asarray(sp.abc_mhz, dtype=float))
            for sp in mol.species]


def corrections(mol, method, basis):
    """Delta_K and its sigma per isotopologue, at the engine's own settings."""
    backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=method,
                                      basis=basis)
    cache: dict = {}

    def hessian_fn(coords_ang):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in cache:
            cache[key] = backend.run_hessian(coords_ang).hessian_bohr
        return cache[key]

    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(start_geometry(mol))
        hess = hessian_fn(coords)
        mode_derivs = normal_mode_hessian_derivatives(
            hessian_fn, coords, hess, list(np.asarray(mol.masses, dtype=float)))

    out = {}
    for label, masses, _obs in species(mol):
        with contextlib.redirect_stdout(io.StringIO()):
            alpha, _, sigma, _ = compute_harmonic_alpha(
                hess, coords, masses, mode_derivs=mode_derivs)
        out[label] = (np.array([0.5 * alpha[k] for k in "ABC"]),
                      np.array([0.5 * sigma[k] for k in "ABC"]))
    return out, coords


def corrections_for_masses(mol, iso_masses, method, basis):
    """Delta_K and sigma for an arbitrary list of (label, masses).

    Needs no measured constants, which is what makes the design calculation
    usable on species that have not been made.
    """
    backend = get_backend("pyscf_hf")(elems=list(mol.elems), method=method,
                                      basis=basis)
    cache: dict = {}

    def hessian_fn(coords_ang):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in cache:
            cache[key] = backend.run_hessian(coords_ang).hessian_bohr
        return cache[key]

    with contextlib.redirect_stdout(io.StringIO()):
        coords = backend.optimise(start_geometry(mol))
        hess = hessian_fn(coords)
        mode_derivs = normal_mode_hessian_derivatives(
            hessian_fn, coords, hess, list(np.asarray(mol.masses, dtype=float)))

    out = {}
    for label, masses in iso_masses:
        with contextlib.redirect_stdout(io.StringIO()):
            alpha, _, sigma, _ = compute_harmonic_alpha(
                hess, coords, np.asarray(masses, dtype=float),
                mode_derivs=mode_derivs)
        out[label] = (np.array([0.5 * alpha[k] for k in "ABC"]),
                      np.array([0.5 * sigma[k] for k in "ABC"]))
    return out, coords


def fit(mol, deltas, free_scales: bool):
    """Least squares over (r, theta) and optionally the three scales."""
    rows = species(mol)

    def residual(p):
        r, theta = p[0], p[1]
        s = p[2:5] if free_scales else np.ones(3)
        res = []
        for label, masses, obs in rows:
            calc = rotational_constants_mhz(_bent_xy(r, theta), masses)
            delta, sig = deltas[label]
            target = obs + s * delta
            res.extend(((calc - target) / sig).tolist())
        return np.asarray(res)

    p0 = [1.0, 110.0] + ([1.0, 1.0, 1.0] if free_scales else [])
    sol = least_squares(residual, p0, method="lm", max_nfev=20000)
    # Parameter covariance from the Jacobian at the solution. The residuals are
    # already divided by sigma, so (J^T J)^-1 is the covariance directly.
    #
    # This is the part that decides whether the detector may speak. A scale
    # factor is only meaningful if the isotopic substitutions actually separate
    # it from the geometry, and whether they do is a property of the molecule,
    # not of the method: H -> D changes a mass by 100% and pins it, whereas
    # ozone's 16-O -> 18-O changes one by 12.5% and barely does.
    jac = np.asarray(sol.jac, dtype=float)
    try:
        cov = np.linalg.inv(jac.T @ jac)
        sigma_p = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sigma_p = np.full(len(p0), np.inf)
    return sol, sigma_p


def true_scales(mol, r_e_coords, deltas):
    """Delta_true / Delta_computed per component, averaged over isotopologues."""
    num, den = np.zeros(3), np.zeros(3)
    for label, masses, obs in species(mol):
        truth = rotational_constants_mhz(r_e_coords, masses) - obs
        delta, _ = deltas[label]
        num += truth
        den += delta
    return num / den


M_H, M_D, M_O16, M_O18 = 1.00782503, 2.01410178, 15.99491462, 17.99915961

#: Isotopologues a laboratory could plausibly measure for each molecule, as
#: (label, masses). The design calculation below needs only masses and a
#: geometry -- never a measured constant -- so it can be run BEFORE deciding
#: which species to synthesise.
CANDIDATES = {
    "water": [
        ("H2-16O", [M_O16, M_H, M_H]),
        ("D2-16O", [M_O16, M_D, M_D]),
        ("HD-16O", [M_O16, M_H, M_D]),
        ("H2-18O", [M_O18, M_H, M_H]),
        ("D2-18O", [M_O18, M_D, M_D]),
        ("HD-18O", [M_O18, M_H, M_D]),
    ],
    "ozone": [
        ("16-16-16", [M_O16, M_O16, M_O16]),
        ("16-18-16", [M_O18, M_O16, M_O16]),
        ("18-16-16", [M_O16, M_O18, M_O16]),
        ("18-16-18", [M_O16, M_O18, M_O18]),
        ("18-18-18", [M_O18, M_O18, M_O18]),
    ],
}


def detection_limit(geom_r, geom_theta, iso_masses, delta_of, sigma_of):
    """Smallest correction bias this isotopologue set could detect, per component.

    Pure experimental design: the parameter covariance of the augmented fit
    depends on the Jacobian and the sigmas, not on any measured value. So this
    can be computed for species nobody has measured yet, and answers the
    question that actually matters before trusting a structure -- not "is there
    a bias" but "could I have seen one, and how big".

    Returns the 2-sigma threshold on |s - 1| per component, as a fraction.
    """
    # The notional measurement is FIXED at the reference geometry. Building it
    # from (r, theta) instead makes it cancel out of the residual, leaving a
    # Jacobian with no geometry columns and a singular normal matrix -- which
    # reports every limit as infinite and looks exactly like "undetectable".
    b0 = {}
    for label, masses in iso_masses:
        ref = rotational_constants_mhz(_bent_xy(geom_r, geom_theta),
                                       np.asarray(masses, dtype=float))
        b0[label] = ref - delta_of(label)

    def residual(p):
        r, theta = p[0], p[1]
        sc = p[2:5]
        res = []
        for label, masses in iso_masses:
            calc = rotational_constants_mhz(_bent_xy(r, theta),
                                            np.asarray(masses, dtype=float))
            delta, sig = delta_of(label), sigma_of(label)
            res.extend(((calc - (b0[label] + sc * delta)) / sig).tolist())
        return np.asarray(res)

    p0 = np.array([geom_r, geom_theta, 1.0, 1.0, 1.0])
    eps = 1e-5
    base = residual(p0)
    jac = np.zeros((base.size, 5))
    for i in range(5):
        q = p0.copy()
        q[i] += eps
        jac[:, i] = (residual(q) - base) / eps
    try:
        cov = np.linalg.inv(jac.T @ jac)
        return 2.0 * np.sqrt(np.clip(np.diag(cov)[2:5], 0.0, None))
    except np.linalg.LinAlgError:
        return np.full(3, np.inf)


def main() -> None:
    method, basis = "b3lyp", "6-31g(d)"
    for tok in sys.argv[1:]:
        if tok.startswith("method="):
            method = tok.split("=", 1)[1]
        elif tok.startswith("basis="):
            basis = tok.split("=", 1)[1]

    for key, (mol, r_e_coords) in TRUTH.items():
        n_obs = 3 * len(mol.species)
        print(f"\n=== {key}: {len(mol.species)} isotopologues, {n_obs} constants",
              flush=True)
        deltas, _ = corrections(mol, method, basis)

        fixed, _ = fit(mol, deltas, free_scales=False)
        free, sigma_p = fit(mol, deltas, free_scales=True)
        want = true_scales(mol, r_e_coords, deltas)

        ref_r = float(np.linalg.norm(r_e_coords[1] - r_e_coords[0]))
        v1, v2 = r_e_coords[1] - r_e_coords[0], r_e_coords[2] - r_e_coords[0]
        ref_t = np.degrees(np.arccos(v1 @ v2 /
                                     (np.linalg.norm(v1) * np.linalg.norm(v2))))

        def chi2_nu(sol, n_par):
            dof = max(n_obs - n_par, 1)
            return float(np.sum(sol.fun ** 2) / dof)

        print(f"  correction believed (s fixed at 1):")
        print(f"    r = {fixed.x[0]:.5f} A ({1000*(fixed.x[0]-ref_r):+.2f} mA)"
              f"   theta = {fixed.x[1]:.3f} deg ({fixed.x[1]-ref_t:+.3f})"
              f"   chi2/nu = {chi2_nu(fixed, 2):.2f}")
        print(f"  correction scaled (s fitted):")
        print(f"    r = {free.x[0]:.5f} A ({1000*(free.x[0]-ref_r):+.2f} mA)"
              f"   theta = {free.x[1]:.3f} deg ({free.x[1]-ref_t:+.3f})"
              f"   chi2/nu = {chi2_nu(free, 5):.2f}")
        print(f"    {'':6s}{'fitted s':>12s}{'sigma(s)':>11s}"
              f"{'verdict':>22s}{'true s':>10s}")
        for i, comp in enumerate("ABC"):
            s_hat, s_sig = float(free.x[2 + i]), float(sigma_p[2 + i])
            if not np.isfinite(s_sig) or s_sig > 0.5:
                verdict = "undetermined"
            elif abs(s_hat - 1.0) > 2.0 * s_sig:
                verdict = f"BIASED ({abs(s_hat-1)/s_sig:.1f} sigma)"
            else:
                verdict = "consistent with 1"
            print(f"    s_{comp:<4s}{s_hat:>12.3f}{s_sig:>11.3f}"
                  f"{verdict:>22s}{want[i]:>10.3f}")

        # What the data COULD have detected, and what more species would buy.
        cand = CANDIDATES.get(key, [])
        have = {lbl for lbl, _, _ in species(mol)}
        d_all, _ = corrections_for_masses(mol, cand, method, basis)
        print(f"\n  bias detection limit (2 sigma on |s-1|), by species set:")
        for n in range(2, len(cand) + 1):
            subset = cand[:n]
            lim = detection_limit(
                fixed.x[0], fixed.x[1], subset,
                lambda lbl: d_all[lbl][0], lambda lbl: d_all[lbl][1])
            tag = "  <- measured" if {l for l, _ in subset} == have else ""
            print(f"    {n} species {str([l for l, _ in subset]):<58s} "
                  + "  ".join(f"{c}:{100*v:7.0f}%" for c, v in zip("ABC", lim))
                  + tag)


if __name__ == "__main__":
    main()
