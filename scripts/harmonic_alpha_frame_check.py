"""Is the harmonic alpha term built from the wrong quantity? Yes.

The harmonic (centrifugal) contribution to alpha is currently
-(d2B_K/dQ_r^2) * <Q_r^2>, with the second derivative taken by finite
difference of ``rotational_constants_mhz`` at displaced geometries. That
function returns the SORTED EIGENVALUES of the instantaneous inertia tensor, so
the derivative follows the instantaneous principal axes.

That is not the right quantity. J_a, J_b and J_c are quantised in the Eckart
frame fixed by the EQUILIBRIUM geometry. A vibrationally induced off-diagonal
inertia element does not reorient those axes; it enters the diagonal element of
mu = I^-1 at second order, and separately drives centrifugal distortion. Taking
the derivative of a sorted principal value instead picks up eigenvalue
repulsion, 2(a^ab)^2/(I_b - I_a), which is not a contribution to alpha at all.

Three candidate second-order coefficients are computed here from one geometry
and one set of normal modes:

  fd_sorted  what the repo does -- finite difference of the sorted principal
             rotational constants.
  fd_fixed   finite difference of the (xi,xi) element of the inertia tensor in
             the FIXED equilibrium principal frame, inverted analytically:
             d2B/dQ^2 = C (2 a^2 / I^3 - b / I^2), with b = d2I_xixi/dQ^2.
  mills      Watson's mu expansion, whose quadratic coefficient is
             (3/4) mu a mu a mu with no second-derivative-of-I term:
             d2B_xi/dQ^2 = (3/2) C sum_gamma (a_xi,gamma)^2/(I_xi^2 I_gamma).

RESULT, water at B3LYP/6-31G(d), against its published equilibrium structure.

The symmetry pattern is textbook and is the first piece of evidence: the two a1
modes (bend, symmetric stretch) have a PURELY DIAGONAL a_r (max off-diagonal
0.0000), and the b2 antisymmetric stretch has a PURELY OFF-DIAGONAL one
(max diagonal 0.0001). On the diagonal modes all three routes agree to six
digits, which validates the Mills implementation and its units. On the b2 mode
they diverge completely, including in sign: B gets fd_sorted +12077.4 against
mills -6263.1 MHz.

Error in the correction against truth (MHz):

                 repo (fd_sorted)      mills
    H2-16O  A          -2141.9       +7028.2     worse
    H2-16O  B          +9852.2        +681.9     14x better
    H2-16O  C          +2402.0        +207.9     12x better
    D2-16O  A           -387.4       +2513.7     worse
    D2-16O  B          +3207.9        +306.8     10x better
    D2-16O  C           +821.7         +15.8     52x better

B and C improve by one to two orders of magnitude on both isotopologues, with
C landing at 16 MHz out of 2668. Four independent components all improving that
far is not a fluke, and it is the first thing in this investigation that moves
the 5-sigma failure at all.

A regresses, and the reason is visible in the terms: water's alpha_A is a near
cancellation of ~1e5 MHz contributions (harmonic -44848 against anharmonic
+27813 for a net -17091), so a 10% error in either swamps the result. A's
apparent accuracy under the old harmonic term was cancellation luck, not
evidence that the term was right.

The same missing quantity breaks centrifugal distortion too.
``tau_prime_from_dB1_cm`` builds tau from dB_alpha/dQ_r alone, via
a_r^{alpha alpha} = -I_alpha (dB_alpha/dQ_r)/B_alpha, which is the DIAGONAL a
only. The Kivelson-Wilson tensor needs the off-diagonal a_r^{alpha beta} for
tau_abab, tau_bcbc and tau_caca. Building a_r^{xi xi'} once fixes both.

    python scripts/harmonic_alpha_frame_check.py
"""
import contextlib, io, sys
import numpy as np
sys.path.insert(0, "/home/user/Quantize/.github"); sys.path.insert(0, "/home/user/Quantize")
import dev.pyscf_backend  # noqa
from backend.registry import get_backend
from backend.spectral.centrifugal_distortion import (
    _INERTIA_TO_MHZ, _ZPE_AMP, bk_mode_derivatives, inertia_paf, normal_modes,
    rotational_constants_mhz)
from backend.spectral.harmonic_alpha import compute_harmonic_alpha
from dev.monofluoro_references import WATER, WATER_R_E, WATER_THETA_E_DEG, _bent_xy, M_O16, M_H, M_D
from scripts.monofluoro_benchmark import start_geometry

C = _INERTIA_TO_MHZ

def inertia_tensor(coords, masses):
    m = np.asarray(masses, float)
    x = np.asarray(coords, float)
    x = x - (m[:, None] * x).sum(0) / m.sum()
    return np.einsum("i,jk->jk", m, np.zeros((3, 3))) + (
        np.eye(3) * (m[:, None] * x**2).sum() - np.einsum("i,ij,ik->jk", m, x, x))

be = get_backend("pyscf_hf")(elems=list(WATER.elems), method="b3lyp", basis="6-31g(d)")
with contextlib.redirect_stdout(io.StringIO()):
    coords = be.optimise(start_geometry(WATER))
    hess = be.run_hessian(coords).hessian_bohr

for label, masses in (("H2-16O", np.array([M_O16, M_H, M_H])),
                      ("D2-16O", np.array([M_O16, M_D, M_D]))):
    omega, L = normal_modes(hess, masses, n_rigid=6)
    keep = omega >= 50.0
    omega, L = omega[keep], L[:, keep]
    n = len(omega)
    # equilibrium principal frame
    evals, V, Xe = inertia_paf(coords, masses)   # Xe IS the PAF geometry
    I_e = np.diag(evals)
    zpe = _ZPE_AMP / omega

    a = np.zeros((n, 3, 3)); b = np.zeros((n, 3, 3))
    h = 0.02
    for r in range(n):
        d = (L[:, r].reshape(-1, 3) / np.sqrt(masses[:, None])) @ V
        Ip = inertia_tensor(Xe + h * d, masses)
        Im = inertia_tensor(Xe - h * d, masses)
        I0 = inertia_tensor(Xe, masses)
        a[r] = (Ip - Im) / (2 * h)
        b[r] = (Ip + Im - 2 * I0) / h**2

    B_e = C / evals
    d2_mills = np.zeros((3, n)); d2_fixed = np.zeros((3, n))
    for r in range(n):
        for xi in range(3):
            d2_mills[xi, r] = 1.5 * C * sum(
                a[r, xi, g] ** 2 / (evals[xi] ** 2 * evals[g]) for g in range(3))
            d2_fixed[xi, r] = C * (2 * a[r, xi, xi] ** 2 / evals[xi] ** 3
                                   - b[r, xi, xi] / evals[xi] ** 2)
    _, d2_sorted = bk_mode_derivatives(coords, masses, L, omega, 0.05,
                                       rotational_constants_mhz(coords, masses))

    print(f"\n===== {label}  (B3LYP/6-31G(d)) ; freqs {np.round(omega,0)}")
    print("  a_r (inertia-tensor derivative) in the equilibrium principal frame:")
    for r in range(n):
        off = max(abs(a[r, i, j]) for i in range(3) for j in range(3) if i != j)
        dia = max(abs(a[r, i, i]) for i in range(3))
        print(f"    mode {r} ({omega[r]:6.0f} cm-1): max|diag| {dia:8.4f}   max|offdiag| {off:8.4f}")
    print("  harmonic alpha per mode (MHz) = -d2B/dQ^2 * <Q^2>:")
    for xi, k in enumerate("ABC"):
        for r in range(n):
            print(f"    {k} mode {r}:  fd_sorted {-d2_sorted[xi,r]*zpe[r]:12.1f}"
                  f"   fd_fixed {-d2_fixed[xi,r]*zpe[r]:12.1f}"
                  f"   mills {-d2_mills[xi,r]*zpe[r]:12.1f}")
    # totals, combined with the repo's Coriolis + anharmonic
    with contextlib.redirect_stdout(io.StringIO()):
        alpha, _, sig, info = compute_harmonic_alpha(
            hess, coords, masses,
            hessian_fn=lambda c: be.run_hessian(c).hessian_bohr)
    truth = rotational_constants_mhz(_bent_xy(WATER_R_E, WATER_THETA_E_DEG),
                                     masses) - np.array(
        [s.abc_mhz for s in WATER.species if s.label == label][0], float)
    print("  correction (1/2 sum alpha) against truth:")
    for xi, k in enumerate("ABC"):
        cor = info["alpha_coriolis_mhz"][k]; anh = info["alpha_anharmonic_mhz"][k]
        base = info["alpha_centrifugal_mhz"][k]
        variants = {
            "repo (fd_sorted)": base,
            "fd_fixed": float(np.sum(-d2_fixed[xi] * zpe)),
            "mills": float(np.sum(-d2_mills[xi] * zpe)),
        }
        print(f"    {k}: truth {truth[xi]:+10.1f}   sigma {0.5*sig[k]:8.1f}")
        for name, harm in variants.items():
            tot = 0.5 * (harm + cor + anh)
            print(f"        {name:18s} harm {0.5*harm:+11.1f}  total {tot:+11.1f}"
                  f"   err {tot-truth[xi]:+10.1f}  ({abs(tot-truth[xi])/(0.5*sig[k]):5.2f} sigma)")
