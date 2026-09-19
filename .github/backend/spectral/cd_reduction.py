"""Watson A-reduction from the tau tensor, derived rather than transcribed.

``watson_a_reduction_cd_from_tau_cm`` carries a documented warning that it does
not reproduce published constants, and ``test_cd_mapping_validation`` measures
how badly: against H2-16O it gets DJ with the wrong sign and DK 138x too small
with the wrong sign. That is why centrifugal distortion has never been usable as
fit data here -- not because the physics is unavailable but because the
parameter mapping is wrong.

This module takes a different route to the same answer. A reduction is not a
formula, it is a reparameterisation: Watson's A-reduced Hamiltonian and the
Kivelson-Wilson tau form describe the same operator, so they must produce the
same energy levels. So rather than copying a coefficient table out of the
literature -- the step that went wrong before, and that cannot be checked
without the literature in hand -- the constants are obtained by construction:

  1. build the rotational Hamiltonian from A, B, C and the tau tensor and
     diagonalise it, giving levels that are exact for that tau;
  2. the A-reduced Hamiltonian is *linear* in its eight parameters, so solve
     the linear system that makes it reproduce those levels.

Nothing about the reduction has to be believed on authority. What makes it
trustworthy is that it is checkable, and it is checked: fed a near-exact water
force field it has to land on H2-16O's measured DJ, DJK and DK, three numbers
known to five figures and none of which the construction has seen.

Representation
--------------
I^r throughout: z = a, x = b, y = c. The A-reduced operator is

    H = A Pz^2 + B Px^2 + C Py^2
        - DJ P^4 - DJK P^2 Pz^2 - DK Pz^4
        - 2 dJ P^2 (Px^2 - Py^2) - dK {Pz^2, Px^2 - Py^2}

and the distortion built from tau is the Kivelson-Wilson form

    H_dist = 1/4 sum_ab tau_aabb * sym(Pa^2 Pb^2)

with ``sym`` the symmetrised product, since Pa^2 and Pb^2 do not commute.
"""

from __future__ import annotations

import numpy as np

#: A, B, C are also free in the fit. The reduction moves them -- that is where
#: the removed indeterminacy goes -- so pinning them at their input values
#: would force the distortion parameters to absorb a rotational-constant shift.
_PARAM_NAMES = ("A", "B", "C", "DJ", "DJK", "DK", "delta_J", "delta_K")


def angular_momentum_operators(j: int):
    """``(P2, Pz2, Pd)`` in the ``|J, K>`` basis, K running -J..J.

    ``Pd`` is Px^2 - Py^2, whose only non-zero elements connect K and K+-2:

        <J, K+2| Px^2 - Py^2 |J, K>
            = 1/2 sqrt[(J(J+1) - K(K+1))(J(J+1) - (K+1)(K+2))]

    Px^2 and Py^2 themselves follow from Pd and Px^2 + Py^2 = P^2 - Pz^2, so
    the whole operator algebra needs only these three.
    """
    n = 2 * j + 1
    ks = np.arange(-j, j + 1, dtype=float)
    jj = float(j * (j + 1))
    p2 = jj * np.eye(n)
    pz2 = np.diag(ks ** 2)
    pd = np.zeros((n, n))
    for i, k in enumerate(ks):
        if i + 2 < n:
            val = 0.5 * np.sqrt(max(jj - k * (k + 1.0), 0.0)
                                * max(jj - (k + 1.0) * (k + 2.0), 0.0))
            pd[i + 2, i] = val
            pd[i, i + 2] = val
    return p2, pz2, pd


def _axis_squares(p2, pz2, pd):
    """``(Pa2, Pb2, Pc2)`` for the I^r association z=a, x=b, y=c."""
    perp = p2 - pz2
    return pz2, 0.5 * (perp + pd), 0.5 * (perp - pd)


def levels_from_tau(abc_mhz, tau_mhz, jmax: int = 6):
    """Energy levels of the rigid rotor plus the tau distortion, in MHz.

    ``tau_mhz`` is the symmetric 3x3 matrix of tau_aabb components in MHz, in
    (a, b, c) order -- the tensor ``centrifugal_distortion`` already computes,
    converted from cm^-1.
    """
    a, b, c = (float(x) for x in abc_mhz)
    tau = np.asarray(tau_mhz, dtype=float)
    tau = 0.5 * (tau + tau.T)
    out = []
    for j in range(jmax + 1):
        p2, pz2, pd = angular_momentum_operators(j)
        sq = _axis_squares(p2, pz2, pd)
        h = a * sq[0] + b * sq[1] + c * sq[2]
        for al in range(3):
            for be in range(3):
                prod = sq[al] @ sq[be]
                h += 0.25 * tau[al, be] * 0.5 * (prod + prod.T)
        out.append(np.sort(np.linalg.eigvalsh(0.5 * (h + h.T))))
    return np.concatenate(out)


def _a_reduced_basis_levels(jmax: int):
    """Level derivatives with respect to each A-reduced parameter.

    The A-reduced Hamiltonian is linear in all eight parameters, but its
    *eigenvalues* are not linear in them in general. They are linear to the
    order that matters here because the distortion terms are a small
    perturbation on the rigid rotor, so the derivative is evaluated at the
    rigid-rotor eigenvectors -- first-order perturbation theory, which is exact
    in the limit the reduction itself assumes.

    Returns a callable taking ``(a, b, c)`` and giving the (n_levels, 8) design
    matrix, because the eigenvectors depend on the rotational constants.
    """
    def design(a, b, c):
        rows = []
        for j in range(jmax + 1):
            p2, pz2, pd = angular_momentum_operators(j)
            sq = _axis_squares(p2, pz2, pd)
            h0 = a * sq[0] + b * sq[1] + c * sq[2]
            _, vecs = np.linalg.eigh(0.5 * (h0 + h0.T))
            ops = [
                sq[0], sq[1], sq[2],
                -(p2 @ p2),
                -(p2 @ pz2),
                -(pz2 @ pz2),
                -2.0 * (p2 @ pd),
                -(pz2 @ pd + pd @ pz2),
            ]
            block = np.empty((2 * j + 1, len(ops)))
            for m, op in enumerate(ops):
                sym = 0.5 * (op + op.T)
                block[:, m] = np.einsum("ik,ij,jk->k", vecs, sym, vecs)
            rows.append(block)
        return np.vstack(rows)
    return design


def a_reduction_from_tau(abc_mhz, tau_mhz, jmax: int = 6):
    """Watson A-reduction constants in MHz, from A/B/C and the tau tensor.

    Returns ``{"A","B","C","DJ","DJK","DK","delta_J","delta_K"}``. A, B and C
    come back shifted from their input values: the reduction absorbs the
    indeterminate combination of tau into them, which is exactly why the
    parameters cannot be mapped one to one.

    ``jmax`` sets how many levels the linear solve sees. Six is ample -- the
    solve is over-determined by a wide margin at J=6 (28 levels for eight
    parameters) and the constants are stable against raising it.
    """
    a, b, c = (float(x) for x in abc_mhz)
    target = levels_from_tau(abc_mhz, tau_mhz, jmax=jmax)
    design = _a_reduced_basis_levels(jmax)(a, b, c)
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    return dict(zip(_PARAM_NAMES, (float(v) for v in sol)))
