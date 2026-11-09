"""Derive the Coriolis alpha term, rather than trust its constants.

WHY THIS FILE EXISTS. ``coriolis_alpha_cm`` carries two bare numbers:

    alpha_r^{xi,Cor} = -(2 B_xi^2 / w_r) Sum_{s != r} (zeta^xi_{rs})^2
                                         (3 w_r^2 + w_s^2)/(w_r^2 - w_s^2)

the 3 and the 2. Every other constant in the alpha chain is pinned by
something: the harmonic term by the Watson convention and exact inertia
derivatives, the anharmonic term by Dunham and Pekeris. These two were pinned
by a citation, and this module's history says a citation is not enough -- the
cubic scheme, the prior widths and the C3 spin weights were all wrong while
being cited.

So the term is rebuilt here from the Hamiltonian, with no closed form assumed,
and the closed form is then required to agree with it exactly.

THE DERIVATION. The Watson Hamiltonian's rotation-vibration part is

    H = Sum_xi B_xi (J_xi - Pi_xi)^2 + Sum_r (w_r/2)(p_r^2 + q_r^2)

with the dimensionless Coriolis operator

    Pi_xi = Sum_{r<s} zeta^xi_{rs} [ (w_s/w_r)^(1/2) q_r p_s
                                   - (w_r/w_s)^(1/2) q_s p_r ]

(the frequency factors come from Q_r P_s = hbar (w_s/w_r)^(1/2) q_r p_s, and
the antisymmetry from zeta_{rs} = -zeta_{sr}).

Expanding the square gives B J^2 - 2 B J.Pi + B Pi^2. The last term carries no
J and so cannot shift a rotational constant. The middle term is off-diagonal in
the vibrational basis, so its leading effect is second order:

    E^(2) = Sum_{v' != v} |<v'| -2 B_xi J_xi Pi_xi |v>|^2 / (E_v - E_v')
          = 4 B_xi^2 J_xi^2 Sum_{v'} |<v'|Pi_xi|v>|^2 / (E_v - E_v')

The bracket is a correction to the coefficient of J_xi^2, which is to say to
the effective rotational constant of vibrational state v. With
B_v = B_e - Sum_r alpha_r (v_r + 1/2),

    alpha_r = -(B_{v_r=1} - B_{v_r=0})

and that is computed below by explicit summation over the harmonic-oscillator
product basis. Nothing in the computation knows about a 3, a 2, or a
(w_r^2 - w_s^2) denominator; those have to emerge.

WHAT THE SUM IS AND IS NOT. This verifies the Coriolis term's own structure and
constants against the Hamiltonian it is derived from. It does not verify that
the Watson Hamiltonian is the right starting point, that zeta is computed
correctly from the Hessian (``test_alpha_invariants`` covers the omega
structure), or anything about the harmonic and anharmonic terms.
"""

from __future__ import annotations

import itertools
from math import sqrt

import numpy as np
import pytest

from backend.spectral.harmonic_alpha import (
    coriolis_alpha_cm,
    coriolis_resonant_pairs,
)


# ── An independent second-order perturbation sum ─────────────────────────────

def _ladder(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless q and p in the harmonic-oscillator basis |0..n_max>.

    q = (a' + a)/sqrt(2), p = i(a' - a)/sqrt(2), so [q, p] = i.
    """
    n = np.arange(1, n_max + 1, dtype=float)
    a = np.zeros((n_max + 1, n_max + 1))
    a[np.arange(n_max), np.arange(1, n_max + 1)] = np.sqrt(n)  # <k|a|k+1>
    adag = a.T
    q = (adag + a) / sqrt(2.0)
    p = 1j * (adag - a) / sqrt(2.0)
    return q, p


def _pi_operator(omega: np.ndarray, zeta_K: np.ndarray, n_max: int) -> np.ndarray:
    """Pi_xi in the full product basis, built by Kronecker products.

    Deliberately literal: the operator is assembled from one-mode ladder
    matrices with no algebra done by hand, so an error in the closed form
    cannot hide in a shared simplification.
    """
    n_vib = omega.size
    dim1 = n_max + 1
    q1, p1 = _ladder(n_max)
    eye = np.eye(dim1)

    def embed(op, mode):
        out = np.array([[1.0 + 0.0j]])
        for m in range(n_vib):
            out = np.kron(out, op if m == mode else eye)
        return out

    q = [embed(q1, m) for m in range(n_vib)]
    p = [embed(p1, m) for m in range(n_vib)]

    pi = np.zeros((dim1 ** n_vib,) * 2, dtype=complex)
    for r in range(n_vib):
        for s in range(r + 1, n_vib):
            z = zeta_K[r, s]
            if z == 0.0:
                continue
            pi += z * (sqrt(omega[s] / omega[r]) * (q[r] @ p[s])
                       - sqrt(omega[r] / omega[s]) * (q[s] @ p[r]))
    return pi


def _basis_states(n_vib: int, n_max: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(n_max + 1), repeat=n_vib))


def _second_order_b_shift(state, omega, pi, states, B):
    """The J_xi^2 coefficient's second-order shift for one vibrational state.

    4 B^2 Sum_{v'} |<v'|Pi|v>|^2 / (E_v - E_v'), summed over the basis.
    """
    idx = {s: i for i, s in enumerate(states)}
    i = idx[tuple(state)]
    e_v = float(np.dot(omega, np.asarray(state, dtype=float) + 0.5))

    total = 0.0
    for j, other in enumerate(states):
        if j == i:
            continue
        amp2 = abs(pi[j, i]) ** 2
        if amp2 == 0.0:
            continue
        e_vp = float(np.dot(omega, np.asarray(other, dtype=float) + 0.5))
        total += amp2 / (e_v - e_vp)
    return 4.0 * B ** 2 * total


def alpha_by_sum_over_states(omega, zeta_K, B, n_max: int = 4) -> np.ndarray:
    """alpha_r for one principal component, from the perturbation sum alone."""
    omega = np.asarray(omega, dtype=float)
    zeta_K = np.asarray(zeta_K, dtype=float)
    n_vib = omega.size
    states = _basis_states(n_vib, n_max)
    pi = _pi_operator(omega, zeta_K, n_max)

    ground = tuple([0] * n_vib)
    b0 = _second_order_b_shift(ground, omega, pi, states, B)

    alpha = np.zeros(n_vib)
    for r in range(n_vib):
        exc = [0] * n_vib
        exc[r] = 1
        b1 = _second_order_b_shift(tuple(exc), omega, pi, states, B)
        alpha[r] = -(b1 - b0)
    return alpha


# ── Helpers to drive the production function ─────────────────────────────────

def _zeta_block(n_vib: int, pairs: dict) -> np.ndarray:
    """A (3, n, n) zeta array with the given {(r,s): value} on component 0."""
    z = np.zeros((3, n_vib, n_vib))
    for (r, s), v in pairs.items():
        z[0, r, s] = v
        z[0, s, r] = -v
    return z


def _closed_form_alpha(omega, zeta_pairs, B):
    n_vib = len(omega)
    z = _zeta_block(n_vib, zeta_pairs)
    alpha, skips = coriolis_alpha_cm(np.asarray(omega, float), z,
                                     np.array([B, 0.0, 0.0]))
    assert skips == 0
    return alpha[0]


# ── The verification ─────────────────────────────────────────────────────────

class TestCoriolisClosedFormMatchesPerturbationSum:
    """The closed form must reproduce a sum that assumes none of it."""

    def test_two_modes(self):
        omega = np.array([1600.0, 3700.0])
        B, zeta = 14.5, 0.82
        got = _closed_form_alpha(omega, {(0, 1): zeta}, B)
        want = alpha_by_sum_over_states(omega, _zeta_block(2, {(0, 1): zeta})[0], B)
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-10)

    @pytest.mark.parametrize("seed", range(12))
    def test_three_modes_random(self, seed):
        rng = np.random.default_rng(seed)
        # Separated BY CONSTRUCTION rather than by rejection: drawing from a
        # flat range and skipping the close draws threw away half the seeds,
        # so half the parametrisation was testing nothing. Gaps of at least
        # 150 cm^-1 keep every denominator well away from the skip threshold;
        # the near-degenerate case is a separate question, tested on its own.
        omega = 400.0 + np.cumsum(rng.uniform(150.0, 1200.0, size=3))
        assert np.min(np.diff(omega)) >= 150.0
        B = float(rng.uniform(0.3, 30.0))
        pairs = {(0, 1): float(rng.uniform(-0.9, 0.9)),
                 (0, 2): float(rng.uniform(-0.9, 0.9)),
                 (1, 2): float(rng.uniform(-0.9, 0.9))}
        got = _closed_form_alpha(omega, pairs, B)
        want = alpha_by_sum_over_states(omega, _zeta_block(3, pairs)[0], B)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-9)

    def test_four_modes(self):
        omega = np.array([620.0, 1180.0, 1740.0, 3420.0])
        pairs = {(0, 1): 0.61, (0, 3): -0.44, (1, 2): 0.37, (2, 3): 0.72}
        B = 9.9
        got = _closed_form_alpha(omega, pairs, B)
        want = alpha_by_sum_over_states(omega, _zeta_block(4, pairs)[0], B, n_max=3)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-9)


class TestTheConstantsAreNotFreeParameters:
    """Perturb the two bare numbers and the agreement has to break.

    A test that passes for any coefficient proves nothing, so this pins the
    3 and the 2 by showing that nothing else fits.
    """

    @staticmethod
    def _alpha_with_coefficients(omega, zeta, B, three, two):
        wr2, ws2 = omega[0] ** 2, omega[1] ** 2
        a0 = -(two * B ** 2 / omega[0]) * zeta ** 2 * (three * wr2 + ws2) / (wr2 - ws2)
        a1 = -(two * B ** 2 / omega[1]) * zeta ** 2 * (three * ws2 + wr2) / (ws2 - wr2)
        return np.array([a0, a1])

    def test_the_true_coefficients_fit(self):
        omega, B, zeta = np.array([1600.0, 3700.0]), 14.5, 0.82
        want = alpha_by_sum_over_states(omega, _zeta_block(2, {(0, 1): zeta})[0], B)
        got = self._alpha_with_coefficients(omega, zeta, B, 3.0, 2.0)
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-10)

    @pytest.mark.parametrize("three,two", [
        (1.0, 2.0), (2.0, 2.0), (4.0, 2.0), (3.0, 1.0), (3.0, 4.0), (1.0, 6.0),
    ])
    def test_no_other_coefficients_fit(self, three, two):
        omega, B, zeta = np.array([1600.0, 3700.0]), 14.5, 0.82
        want = alpha_by_sum_over_states(omega, _zeta_block(2, {(0, 1): zeta})[0], B)
        got = self._alpha_with_coefficients(omega, zeta, B, three, two)
        assert not np.allclose(got, want, rtol=1e-3), (
            f"(3 -> {three}, 2 -> {two}) also fits, so this test pins nothing"
        )


class TestPropertiesTheClosedFormRelieson:
    """Two structural claims the sum form makes, which the closed form assumes."""

    def test_sum_over_s_is_additive(self):
        # Different (r,s) pairs reach orthogonal intermediate states, so the
        # closed form's plain sum over s is exact, not an approximation.
        omega = np.array([700.0, 1500.0, 3100.0])
        B = 11.0
        p01, p02 = {(0, 1): 0.5}, {(0, 2): -0.7}
        both = {**p01, **p02}
        a01 = alpha_by_sum_over_states(omega, _zeta_block(3, p01)[0], B)
        a02 = alpha_by_sum_over_states(omega, _zeta_block(3, p02)[0], B)
        a_both = alpha_by_sum_over_states(omega, _zeta_block(3, both)[0], B)
        np.testing.assert_allclose(a_both, a01 + a02, rtol=1e-10, atol=1e-9)

    def test_pairs_not_involving_r_cancel_out_of_alpha_r(self):
        # A (s,t) pair shifts B_0 and B_{1_r} by the same amount, so it must
        # leave alpha_r at exactly zero -- which is why the closed form sums
        # only over s and never over disjoint pairs.
        omega = np.array([700.0, 1500.0, 3100.0])
        alpha = alpha_by_sum_over_states(omega, _zeta_block(3, {(1, 2): 0.8})[0], 11.0)
        # Exactly zero for the uncoupled mode, and not merely small: the two
        # coupled modes are the scale that zero is being measured against.
        coupled = max(abs(alpha[1]), abs(alpha[2]))
        assert coupled > 0.0
        assert abs(alpha[0]) < 1e-12 * coupled

    def test_alpha_is_quadratic_in_zeta(self):
        omega, B = np.array([1600.0, 3700.0]), 14.5
        a1 = alpha_by_sum_over_states(omega, _zeta_block(2, {(0, 1): 0.3})[0], B)
        a2 = alpha_by_sum_over_states(omega, _zeta_block(2, {(0, 1): 0.6})[0], B)
        np.testing.assert_allclose(a2, 4.0 * a1, rtol=1e-10, atol=1e-9)

    def test_alpha_is_quadratic_in_b(self):
        omega, zeta = np.array([1600.0, 3700.0]), 0.5
        z = _zeta_block(2, {(0, 1): zeta})[0]
        a1 = alpha_by_sum_over_states(omega, z, 3.0)
        a2 = alpha_by_sum_over_states(omega, z, 6.0)
        np.testing.assert_allclose(a2, 4.0 * a1, rtol=1e-10, atol=1e-9)

    def test_sign_is_set_by_which_mode_is_higher(self):
        # The lower mode of a coupled pair has w_r^2 - w_s^2 < 0, so its
        # Coriolis alpha is positive and its partner's is negative. Mixed signs
        # within one molecule are expected, not a symptom.
        alpha = _closed_form_alpha([1600.0, 3700.0], {(0, 1): 0.82}, 14.5)
        assert alpha[0] > 0.0 > alpha[1]


class TestNearDegenerateHandling:
    """The skip, and the fact that the skipped pair would have cancelled."""

    def test_vanishing_denominator_is_skipped_not_divided_by(self):
        omega = np.array([1200.0, 1200.0 + 1e-9, 2800.0])
        z = _zeta_block(3, {(0, 1): 0.99, (0, 2): 0.2})
        alpha, skips = coriolis_alpha_cm(omega, z, np.array([10.0, 0.0, 0.0]))
        assert skips == 6, skips  # (0,1) and (1,0), once per component
        assert np.all(np.isfinite(alpha))

    def test_the_two_terms_of_a_resonant_pair_are_equal_and_opposite(self):
        # The module's stated reason for not deperturbing: the pair's two
        # terms carry denominators of opposite sign and cancel in the total
        # that reaches B_0. Checked just above the skip threshold.
        omega = np.array([1200.0, 1205.0, 2800.0])
        z = _zeta_block(3, {(0, 1): 0.99})
        alpha, skips = coriolis_alpha_cm(omega, z, np.array([10.0, 0.0, 0.0]))
        assert skips == 0
        # Weighted by 1/w_r, so the cancellation is near-exact, not exact.
        assert abs(alpha[0].sum()) < 0.01 * np.abs(alpha[0]).max()

    def test_resonances_are_reported_with_their_term(self):
        omega = np.array([1200.0, 1205.0, 2800.0])
        z = _zeta_block(3, {(0, 1): 0.99})
        pairs = coriolis_resonant_pairs(omega, z, resonance_cm=10.0)
        assert len(pairs) == 1
        got = pairs[0]
        assert got["modes"] == (0, 1)
        assert got["component"] == "A"
        assert got["separation_cm"] == pytest.approx(5.0)
        assert got["zeta"] == pytest.approx(0.99)
        # "Enormous" measured against a pair of the same zeta that is simply
        # not degenerate: the 1/(w_r^2 - w_s^2) denominator is the whole story.
        separated = coriolis_resonant_pairs(
            np.array([1200.0, 2800.0, 1205.0]),
            _zeta_block(3, {(0, 1): 0.99}), resonance_cm=1e9)[0]
        assert abs(got["term_cm"]) > 100.0 * abs(separated["term_cm"])

    def test_well_separated_modes_report_no_resonance(self):
        omega = np.array([700.0, 1500.0, 3100.0])
        z = _zeta_block(3, {(0, 1): 0.9, (1, 2): 0.9})
        assert coriolis_resonant_pairs(omega, z, resonance_cm=10.0) == []

    def test_weak_coupling_is_not_reported_even_when_degenerate(self):
        # A near-degenerate pair that barely couples is not a resonance worth
        # reporting; the zeta^2 floor is what distinguishes them.
        omega = np.array([1200.0, 1205.0, 2800.0])
        z = _zeta_block(3, {(0, 1): 1e-4})
        assert coriolis_resonant_pairs(omega, z, resonance_cm=10.0) == []


class TestComponentIndependence:
    """Each principal component uses its own zeta, and only its own."""

    def test_components_do_not_leak(self):
        omega = np.array([1600.0, 3700.0])
        z = np.zeros((3, 2, 2))
        z[1, 0, 1], z[1, 1, 0] = 0.82, -0.82  # B component only
        alpha, _ = coriolis_alpha_cm(omega, z, np.array([5.0, 5.0, 5.0]))
        # The untouched components are exactly zero, measured against the one
        # that was given a coupling.
        scale = np.abs(alpha[1]).max()
        assert scale > 0.0
        assert np.all(np.abs(alpha[0]) < 1e-12 * scale)
        assert np.all(np.abs(alpha[2]) < 1e-12 * scale)

    def test_each_component_scales_with_its_own_b(self):
        omega = np.array([1600.0, 3700.0])
        z = _zeta_block(2, {(0, 1): 0.5})
        z[1] = z[0]
        alpha, _ = coriolis_alpha_cm(omega, z, np.array([4.0, 8.0, 0.0]))
        np.testing.assert_allclose(alpha[1], 4.0 * alpha[0], rtol=1e-12)
