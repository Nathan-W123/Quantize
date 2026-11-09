"""Generate a torsional scan by constrained optimisation on a PES.

THE LAST MISSING LINK. The torsion package could solve a hindered rotor,
average rotational constants over its wavefunction, fit a potential from scan
energies and convert the result into a correction (``torsion_replacement_from_
scan``). What it could not do was produce the scan. Every scan in the test
suite is synthetic or hand-entered, so the treatment could be exercised but
never run on a molecule, which is why it has sat wired up and unscored.

RELAXED, NOT RIGID, AND WHY IT MATTERS. A rigid scan turns the top and leaves
everything else where it was. That is cheap and wrong in a specific direction:
the frame relaxes as the top turns -- bonds lengthen at the barrier top, angles
open -- so a rigid scan overestimates the barrier, often by tens of percent.
Since the barrier sets the torsional level spacings, and those set the
averaging weights, and those set the correction, the error propagates all the
way to the geometry. So the default here is a relaxed scan: every coordinate
but the dihedral is minimised at each grid point.

PySCF's own optimiser needs geomeTRIC, which has no wheel in this environment,
so the constraint is imposed directly: SLSQP on Cartesians with the dihedral as
a nonlinear equality constraint, using the analytic dihedral gradient the
repository already had in ``backend.quantum._dihedral_deriv`` and the analytic
energy gradient from the backend. No new dependency, and the constraint is
exact at convergence rather than enforced by a penalty.

SIX ZERO MODES ARE LEFT ALONE. Optimising 3N Cartesians under one constraint
leaves overall translation and rotation unfixed. That is harmless here: the
dihedral is invariant under both, and the rotational constants are computed in
the principal-axis frame, so any drift divides out. Fixing them would add
constraints without changing an answer.

MEASURED END TO END ON METHANOL, HF/6-31G, 12-point relaxed scan
(scripts/verify_torsion_chain.py, 2026-11-09, ~2 min):

    quantity      computed   reference   error
    barrier         410.6      373.0    +10.1%  cm-1
    F                29.79      27.6     +7.9%  cm-1
    rho             0.8194      0.81     +1.2%
    A               4.4346      4.25     +4.3%  cm-1
    B               0.8114     0.823     -1.4%  cm-1
    C               0.7856     0.793     -0.9%  cm-1

    alpha_A = +891.8 +/- 322.0 MHz,  alpha_B = -178.0,  alpha_C = -180.9

Every error there is accounted for, which is why these numbers are worth
keeping rather than just passing. The dihedral constraint is satisfied to
8e-11 degrees, so the scan is at the angles it claims. The barrier runs 10%
high because HF with a small basis overestimates torsional barriers. A comes
out 4.3% high -- the geometry is too compact -- and F is 7.9% high, almost
exactly twice that, because rho is a ratio of moments and scale free while F
carries 1/I_red and picks up a size error twice over. That 2x relationship is
asserted exactly in test_internal_rotation, and seeing it hold on a real
surface is the check that the chain is not quietly doing something else.

So the remaining error is the level of theory, not the machinery. A correlated
method with a decent basis is what would move these, not a change here.

WHAT THIS DOES NOT DO. One torsion at a time. Two coupled rotors need a 2D
scan and ``torsion_average_2d``; a scan that passes through a conformational
change of the frame will produce a discontinuous potential, which
``preprocess_scan`` is there to catch and this module does not try to.
"""

from __future__ import annotations

import numpy as np

from backend.quantum import _dihedral_deriv
from backend.torsion.internal_rotation import (
    _bonds,
    dihedral_angle,
    internal_rotation_constants,
    rotational_constants_cm,
    top_atoms,
)
from backend.torsion.torsion_average import TorsionGridPoint, TorsionScan

_HARTREE_TO_CM = 219474.6313632


def rotate_top(coords_ang, top_indices, pivot_a: int, pivot_b: int,
               delta_deg: float) -> np.ndarray:
    """Turn the top rigidly about the pivot_a -> pivot_b axis.

    Right-handed about pivot_a -> pivot_b. NOTE THE SIGN TRAP: what that does
    to a particular dihedral depends on the axis direction and on which side
    of it the rotated atoms lie. Rotating methanol's methyl about O -> C
    changes the dihedral (H_O, O, C, H_me) by MINUS the requested angle, and
    about C -> O by plus it. The relationship is exactly linear with slope
    +/-1, so callers should measure the sign rather than assume it;
    ``_rotation_sign`` does, and an earlier version of the scanner that
    assumed it seeded every grid point 2x off target.

    Used only to generate a STARTING point for each grid optimisation, so its
    crudeness costs nothing: the constrained minimisation is what fixes the
    geometry. Rodrigues rotation about the axis line through pivot_a.
    """
    coords = np.array(coords_ang, dtype=float, copy=True)
    axis = coords[int(pivot_b)] - coords[int(pivot_a)]
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(float(delta_deg))
    k, c, s = axis, np.cos(theta), np.sin(theta)
    idx = [int(i) for i in top_indices if int(i) not in (int(pivot_a), int(pivot_b))]
    rel = coords[idx] - coords[int(pivot_a)]
    rot = (rel * c
           + np.cross(k, rel) * s
           + np.outer(rel @ k, k) * (1.0 - c))
    coords[idx] = rot + coords[int(pivot_a)]
    return coords


def _rotation_sign(coords_ang, top_indices, dihedral, pivot_a, pivot_b) -> float:
    """+1 or -1: what a positive rotate_top does to this dihedral.

    Measured with a probe rotation rather than reasoned about, because the
    sign depends on the axis direction and on which side of it the top lies,
    and getting it wrong seeds every grid point at the wrong angle. The
    dihedral responds exactly linearly to a rotation about its own axis, so
    one probe is exact, not an approximation.
    """
    m, o, p, n = (int(i) for i in dihedral)
    probe = 5.0
    before = dihedral_angle(coords_ang, m, o, p, n)
    after = dihedral_angle(
        rotate_top(coords_ang, top_indices, pivot_a, pivot_b, probe),
        m, o, p, n)
    delta = (after - before + 180.0) % 360.0 - 180.0
    if abs(abs(delta) - probe) > 1e-6:
        raise ValueError(
            f"rotating the top about {pivot_a}->{pivot_b} changed the "
            f"dihedral {dihedral} by {delta:.6f} deg for a {probe} deg "
            f"rotation; the axis is not this dihedral's central bond, or the "
            f"top does not lie on one side of it"
        )
    return 1.0 if delta > 0.0 else -1.0


def _constrained_minimise(energy_gradient_fn, coords0, dihedral, target_deg,
                          max_iter: int, ftol: float):
    """Minimise the energy at a fixed dihedral. Returns (coords, energy, info)."""
    from scipy.optimize import minimize

    from backend.quantum import ANG_TO_BOHR

    m, o, p, n = (int(i) for i in dihedral)
    shape = np.asarray(coords0, dtype=float).shape
    target = float(target_deg)

    def fun(flat):
        energy, grad_bohr = energy_gradient_fn(flat.reshape(shape))
        return float(energy), np.asarray(grad_bohr, dtype=float).ravel() * ANG_TO_BOHR

    def wrapped_residual(flat):
        """Dihedral minus target, unwrapped to (-180, 180].

        Without the unwrap the constraint is discontinuous at +/-180 and SLSQP
        drives the structure the long way round the circle.
        """
        phi = dihedral_angle(flat.reshape(shape), m, o, p, n)
        return float((phi - target + 180.0) % 360.0 - 180.0)

    def residual_jac(flat):
        # SIGN. backend.quantum._dihedral_deriv uses the OPPOSITE sign
        # convention to dihedral_angle: measured against a finite difference
        # of dihedral_angle, their ratio is exactly -1 everywhere. Both are
        # self-consistent, they simply disagree, and _dihedral_deriv is load
        # bearing in the Wilson B-matrix code so it is not the one to change.
        # Negating here is what makes the Jacobian the derivative of THIS
        # residual. Without it SLSQP is handed a gradient pointing the wrong
        # way; it still converged on the model potential, which is exactly
        # how a bug like this survives. test_dihedral_deriv_sign_convention
        # pins the relationship so a later change to either is caught.
        d = _dihedral_deriv(flat.reshape(shape), m, o, p, n)
        return -np.degrees(d).ravel()

    sol = minimize(
        fun, np.asarray(coords0, dtype=float).ravel(), jac=True,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": wrapped_residual, "jac": residual_jac}],
        options={"maxiter": int(max_iter), "ftol": float(ftol)},
    )
    coords = sol.x.reshape(shape)
    achieved = dihedral_angle(coords, m, o, p, n)
    return coords, float(sol.fun), {
        "converged": bool(sol.success),
        "message": str(sol.message),
        "n_iter": int(sol.get("nit", -1)),
        "dihedral_error_deg": float((achieved - target + 180.0) % 360.0 - 180.0),
    }


def relaxed_torsion_scan(energy_gradient_fn, coords_ang, elements, dihedral,
                         *, n_points: int = 12, span_deg: float = 360.0,
                         relax: bool = True, masses=None, name: str = "torsion",
                         top_indices=None, max_iter: int = 150,
                         ftol: float = 1e-10, dihedral_tol_deg: float = 0.05,
                         verbose: bool = False):
    """A TorsionScan from a PES, by constrained optimisation at each angle.

    Parameters
    ----------
    energy_gradient_fn : callable(coords_ang) -> (energy_hartree, grad_hartree_per_bohr).
        A backend's ``_energy_gradient``. Taken as a callable rather than a
        backend object so the scanner can be tested against an analytic
        potential without a quantum chemistry package.
    coords_ang : (N, 3) starting geometry, assumed already at a minimum.
    dihedral : (m, o, p, n) atom indices. o-p is the rotation axis.
    n_points : grid points over ``span_deg``, evenly spaced, endpoint excluded
        for a full turn so the periodic grid is not double-counted.
    relax : False gives a rigid scan, for comparison. It is the wrong answer
        on purpose; see the module docstring.
    top_indices : the rotating atoms. Found from connectivity when omitted.

    Returns
    -------
    (scan, info). ``info`` carries per-point convergence, the achieved
    dihedral error, and ``all_converged`` / ``max_dihedral_error_deg`` so a
    caller can refuse a scan rather than fit a potential to a bad one.
    """
    coords0 = np.asarray(coords_ang, dtype=float)
    m, o, p, n = (int(i) for i in dihedral)
    if top_indices is None:
        top_indices = top_atoms(coords0, elements, frame_atom=o, top_atom=p,
                                bonds=_bonds(coords0, elements))

    phi0 = dihedral_angle(coords0, m, o, p, n)
    sign = _rotation_sign(coords0, top_indices, (m, o, p, n), o, p)
    full_turn = abs(float(span_deg) - 360.0) < 1e-9
    offsets = np.linspace(0.0, float(span_deg), int(n_points),
                          endpoint=not full_turn)

    points, per_point = [], []
    current = coords0
    for k, off in enumerate(offsets):
        target = phi0 + float(off)
        need = (target - dihedral_angle(current, m, o, p, n) + 180.0) % 360.0 - 180.0
        start = rotate_top(current, top_indices, o, p, sign * need)
        if relax:
            geom, energy, pinfo = _constrained_minimise(
                energy_gradient_fn, start, (m, o, p, n), target,
                max_iter=max_iter, ftol=ftol)
            # Walk the grid from the previous RELAXED structure, so the frame
            # follows the top continuously instead of restarting from the
            # minimum each time and risking a different local branch.
            current = geom
        else:
            geom = start
            energy = float(energy_gradient_fn(start)[0])
            pinfo = {"converged": True, "message": "rigid", "n_iter": 0,
                     "dihedral_error_deg": float(
                         (dihedral_angle(start, m, o, p, n) - target + 180.0)
                         % 360.0 - 180.0)}
        points.append(TorsionGridPoint(
            phi=float(off), geometry=geom, energy=float(energy),
            label=f"phi_{off:+.1f}"))
        per_point.append(pinfo)
        if verbose:
            print(f"  [scan] {off:+7.1f} deg  E = {energy:.9f}  "
                  f"dphi = {pinfo['dihedral_error_deg']:+.4f}  "
                  f"{'ok' if pinfo['converged'] else pinfo['message']}",
                  flush=True)

    energies = np.array([gp.energy for gp in points], dtype=float)
    scan = TorsionScan(
        name=str(name), atoms=(m, o, p, n), grid_points=points,
        angle_unit="degrees", energy_unit="hartree", periodic=full_turn,
        metadata={"relaxed": bool(relax), "phi0_deg": float(phi0),
                  "top_indices": tuple(int(i) for i in top_indices)},
    )
    errors = [abs(x["dihedral_error_deg"]) for x in per_point]
    info = {
        "per_point": per_point,
        "all_converged": all(x["converged"] for x in per_point),
        "max_dihedral_error_deg": float(max(errors)) if errors else 0.0,
        "dihedral_within_tol": bool(max(errors, default=0.0) <= float(dihedral_tol_deg)),
        "barrier_cm": float((energies.max() - energies.min()) * _HARTREE_TO_CM),
        "relaxed": bool(relax),
        "rotation_sign": float(sign),
    }
    return scan, info


def spec_from_scan(coords_ang, masses_amu, elements, scan, *,
                   frame_atom: int | None = None, top_atom: int | None = None,
                   symmetry_number: int = 3, n_harmonics: int = 3,
                   n_basis: int = 10, top_indices=None):
    """A TorsionHamiltonianSpec from a scan and a geometry.

    Fills F and rho from ``internal_rotation_constants`` and A/B/C from the
    reference geometry, then fits the potential from the scan energies. These
    were the hand-entered parts; nothing here is a literature value.
    """
    from backend.torsion.scan_fit import scan_to_torsion_potential
    from backend.torsion.torsion_average import ensure_phi_radians
    from backend.torsion.torsion_hamiltonian import TorsionHamiltonianSpec

    if frame_atom is None or top_atom is None:
        if scan.atoms is None:
            raise ValueError("pass frame_atom/top_atom, or a scan carrying atoms")
        frame_atom, top_atom = int(scan.atoms[1]), int(scan.atoms[2])

    rot = internal_rotation_constants(coords_ang, masses_amu, frame_atom,
                                      top_atom, top_indices=top_indices,
                                      elements=elements)
    A, B, C = rotational_constants_cm(coords_ang, masses_amu)

    energies = np.array([gp.energy for gp in scan.grid_points], dtype=float)
    unit = str(scan.energy_unit).strip().lower()
    if unit in {"hartree", "au", "a.u."}:
        e_cm = (energies - energies.min()) * _HARTREE_TO_CM
    elif unit in {"cm-1", "cm^-1", "wavenumber"}:
        e_cm = energies - energies.min()
    else:
        raise ValueError(f"unsupported scan energy_unit {scan.energy_unit!r}")

    # ALIASING. scan_to_torsion_potential with symmetry_number s and
    # n_harmonics h fits orders s, 2s, ... hs. A grid of N points over a full
    # turn resolves orders up to N/2; anything above that aliases back down
    # and the fit splits one real harmonic between two orders. A 12-point
    # methanol scan asked for orders 3, 6, 9 and returned V3 = V9 = -93.25,
    # which is one V3 cut in half and mislabelled, not two harmonics. So the
    # request is clamped to the grid rather than honoured and quietly wrong.
    n_grid = len(scan.grid_points)
    max_h = max(1, int(n_grid // (2 * int(symmetry_number))))
    used_h = min(int(n_harmonics), max_h)
    potential, fit_info = scan_to_torsion_potential(
        ensure_phi_radians(scan), e_cm, n_harmonics=used_h,
        symmetry_number=int(symmetry_number))

    spec = TorsionHamiltonianSpec(
        F=float(rot["F_cm"]), rho=float(rot["rho"]),
        A=float(A), B=float(B), C=float(C),
        potential=potential, n_basis=int(n_basis), units="cm-1")
    return spec, {"internal_rotation": rot, "potential_fit": fit_info,
                  "barrier_cm": float(e_cm.max() - e_cm.min()),
                  "n_harmonics_requested": int(n_harmonics),
                  "n_harmonics_used": int(used_h),
                  "n_harmonics_max_for_grid": int(max_h),
                  "harmonics_clamped": bool(used_h < int(n_harmonics))}
