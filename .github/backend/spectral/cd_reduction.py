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

Both reductions
---------------
Watson's S reduction differs from A only in how the asymmetry is
parameterised, so in this construction it is the same linear solve with two
basis operators swapped:

    A:  - 2 dJ P^2 (Px^2 - Py^2) - dK {Pz^2, Px^2 - Py^2}
    S:  + d1 P^2 (P+^2 + P-^2)   + d2 (P+^4 + P-^4)

Both S operators are symmetric under exchange of P+ and P-, so the body-frame
anomalous commutator -- which decides whether P+ raises or lowers K, and is a
standard place to get a sign wrong -- cannot affect them.

Having both matters for two reasons. A near-symmetric top is ill-conditioned in
A and well-conditioned in S, which is the entire reason S exists; and published
constants cannot be compared across reductions at all, so a molecule whose
literature fit is in S was previously unusable as a validation case or as fit
data. ``reduction_residual_mhz`` picks between them by measurement rather than
by rule of thumb: fit both, keep whichever reproduces the exact tau levels.
"""

from __future__ import annotations

import numpy as np

#: A, B, C are also free in the fit. The reduction moves them -- that is where
#: the removed indeterminacy goes -- so pinning them at their input values
#: would force the distortion parameters to absorb a rotational-constant shift.
_A_PARAM_NAMES = ("A", "B", "C", "DJ", "DJK", "DK", "delta_J", "delta_K")

#: Watson writes the S-reduced quartic constants D_J/D_JK/D_K/d1/d2 against the
#: A-reduced Delta_J/Delta_JK/Delta_K/delta_J/delta_K. The first three share a
#: name here because that is how they are almost always printed, but they are
#: NOT the same numbers -- a D_J from an S fit and a D_J from an A fit differ,
#: and mixing them is the mistake this module exists to make impossible. The
#: reduction is therefore always carried alongside the values.
_S_PARAM_NAMES = ("A", "B", "C", "DJ", "DJK", "DK", "d1", "d2")

#: reduction label -> the parameters its Hamiltonian is linear in.
REDUCTION_PARAMS = {"A": _A_PARAM_NAMES, "S": _S_PARAM_NAMES}

#: Backwards-compatible alias; callers that predate the S reduction.
_PARAM_NAMES = _A_PARAM_NAMES


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


def ladder_power_sum(j: int, n: int):
    """``P+^n + P-^n`` in the ``|J, K>`` basis: the operator linking K and K+-n.

        <J, K+n| P+^n |J, K> = prod_{m=0}^{n-1} sqrt(J(J+1) - (K+m)(K+m+1))

    Only even n appears in a rotational Hamiltonian. n = 2 reproduces twice the
    ``Pd`` of :func:`angular_momentum_operators`, which is the identity
    P+^2 + P-^2 = 2 (Px^2 - Py^2); n = 4 is the S-reduction's d2 operator and
    n = 6 the sextic h3 one.
    """
    if n < 0 or n % 2:
        raise ValueError(f"n must be a non-negative even integer, got {n}")
    size = 2 * j + 1
    out = np.zeros((size, size))
    if n == 0:
        return np.eye(size)
    ks = np.arange(-j, j + 1, dtype=float)
    jj = float(j * (j + 1))
    for i, k in enumerate(ks):
        if i + n >= size:
            continue
        val = 1.0
        for m in range(n):
            km = k + m
            val *= np.sqrt(max(jj - km * (km + 1.0), 0.0))
        out[i + n, i] = val
        out[i, i + n] = val
    return out


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


def _reduced_operators(reduction: str, j: int):
    """The operators the reduced Hamiltonian is linear in, in parameter order.

    Shared between the design matrix (which takes their diagonal elements in
    the rigid-rotor eigenbasis) and :func:`levels_from_reduction` (which sums
    them into a Hamiltonian and diagonalises it), so the two can never drift
    apart -- a fitted parameter set always means the same operator it was
    fitted for.
    """
    red = str(reduction).strip().upper()
    p2, pz2, pd = angular_momentum_operators(j)
    sq = _axis_squares(p2, pz2, pd)
    ops = [
        sq[0], sq[1], sq[2],
        -(p2 @ p2),
        -(p2 @ pz2),
        -(pz2 @ pz2),
    ]
    if red == "A":
        ops += [
            -2.0 * (p2 @ pd),
            -(pz2 @ pd + pd @ pz2),
        ]
    elif red == "S":
        # d1 multiplies P^2 (P+^2 + P-^2) = 2 P^2 Pd; d2 multiplies P+^4 + P-^4.
        ops += [
            2.0 * (p2 @ pd),
            ladder_power_sum(j, 4),
        ]
    else:
        raise ValueError(
            f"Unknown reduction {reduction!r}. Valid: {sorted(REDUCTION_PARAMS)}"
        )
    return [0.5 * (op + op.T) for op in ops]


def _reduced_basis_levels(jmax: int, reduction: str = "A"):
    """Level derivatives with respect to each reduced parameter.

    The reduced Hamiltonian is linear in all eight parameters, but its
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
            ops = _reduced_operators(reduction, j)
            block = np.empty((2 * j + 1, len(ops)))
            for m, op in enumerate(ops):
                block[:, m] = np.einsum("ik,ij,jk->k", vecs, op, vecs)
            rows.append(block)
        return np.vstack(rows)
    return design


def _a_reduced_basis_levels(jmax: int):
    """Backwards-compatible alias for the A-reduced design matrix."""
    return _reduced_basis_levels(jmax, "A")


def levels_from_reduction(abc_mhz, params, reduction: str = "A", jmax: int = 6):
    """Exact eigenvalues of a reduced Hamiltonian built from fitted parameters.

    The inverse of :func:`reduction_from_tau`, and the thing that makes the
    reduction checkable without any literature: a parameter set is only correct
    if feeding it back through the operator it was fitted for reproduces the
    levels it was fitted to.

    ``params`` must carry A, B and C as well, since the reduction shifts them.
    """
    red = str(reduction).strip().upper()
    names = REDUCTION_PARAMS[red]
    vals = [float(params[n]) for n in names]
    out = []
    for j in range(jmax + 1):
        ops = _reduced_operators(red, j)
        h = sum(v * op for v, op in zip(vals, ops))
        out.append(np.sort(np.linalg.eigvalsh(0.5 * (h + h.T))))
    return np.concatenate(out)


def reduction_from_tau(abc_mhz, tau_mhz, reduction: str = "A", jmax: int = 6):
    """Watson reduced distortion constants in MHz, from A/B/C and the tau tensor.

    ``reduction`` is "A" or "S". Returns the eight parameters that reduction is
    linear in -- A, B, C and five distortion constants, keyed as
    ``REDUCTION_PARAMS[reduction]``.

    A, B and C come back shifted from their input values: the reduction absorbs
    the indeterminate combination of tau into them, which is exactly why the
    parameters cannot be mapped one to one, and why a published A/B/C from a
    distortion fit is not the same quantity as a rigid-rotor A/B/C.

    ``jmax`` sets how many levels the linear solve sees. Six is ample -- the
    solve is over-determined by a wide margin at J=6 (49 levels for eight
    parameters) and the constants are stable against raising it.
    """
    red = str(reduction).strip().upper()
    if red not in REDUCTION_PARAMS:
        raise ValueError(
            f"Unknown reduction {reduction!r}. Valid: {sorted(REDUCTION_PARAMS)}"
        )
    a, b, c = (float(x) for x in abc_mhz)
    target = levels_from_tau(abc_mhz, tau_mhz, jmax=jmax)
    design = _reduced_basis_levels(jmax, red)(a, b, c)
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    return dict(zip(REDUCTION_PARAMS[red], (float(v) for v in sol)))


def a_reduction_from_tau(abc_mhz, tau_mhz, jmax: int = 6):
    """Watson A-reduction constants in MHz. See :func:`reduction_from_tau`."""
    return reduction_from_tau(abc_mhz, tau_mhz, reduction="A", jmax=jmax)


def s_reduction_from_tau(abc_mhz, tau_mhz, jmax: int = 6):
    """Watson S-reduction constants in MHz. See :func:`reduction_from_tau`."""
    return reduction_from_tau(abc_mhz, tau_mhz, reduction="S", jmax=jmax)


def reduction_residual_mhz(abc_mhz, tau_mhz, reduction: str = "A", jmax: int = 6):
    """RMS mismatch, in MHz, between the exact tau levels and the reduced form.

    A reduction is an exact reparameterisation of the same operator only to the
    order it retains. What is left over is the quartic form failing to span the
    exact tau Hamiltonian, and it is measurable without any reference data --
    which makes it the honest way to choose between A and S for a given
    molecule: fit both, keep whichever reproduces the levels.

    This is also the self-check that no coefficient table can offer. A
    transcription error in the operator set shows up here immediately as a
    residual comparable to the distortion itself.
    """
    params = reduction_from_tau(abc_mhz, tau_mhz, reduction=reduction, jmax=jmax)
    got = levels_from_reduction(abc_mhz, params, reduction=reduction, jmax=jmax)
    want = levels_from_tau(abc_mhz, tau_mhz, jmax=jmax)
    return float(np.sqrt(np.mean((got - want) ** 2)))


def best_reduction(abc_mhz, tau_mhz, jmax: int = 6):
    """``(label, params, residual_mhz)`` for whichever reduction fits better.

    Near-symmetric tops are ill-conditioned in A and well-conditioned in S --
    that is the whole reason S exists -- so which one to use is a property of
    the molecule, not a house style.
    """
    scored = []
    for red in sorted(REDUCTION_PARAMS):
        params = reduction_from_tau(abc_mhz, tau_mhz, reduction=red, jmax=jmax)
        got = levels_from_reduction(abc_mhz, params, reduction=red, jmax=jmax)
        want = levels_from_tau(abc_mhz, tau_mhz, jmax=jmax)
        scored.append((float(np.sqrt(np.mean((got - want) ** 2))), red, params))
    scored.sort(key=lambda t: t[0])
    resid, red, params = scored[0]
    return red, params, resid
