"""A Hessian the engine consumes must satisfy the translational sum rule.

Moving the whole molecule changes nothing, so every row of an exact Cartesian
Hessian sums to zero over the atoms: sum_j H[i a, j b] = 0. pyscf's analytic
RKS Hessian (2.15, CPU) has no grid-weight response, and at its default grid
every B3LYP/6-31G(d) Hessian in the benchmark breaks the rule by 2e-4 (water)
to 1.9e-2 Ha/bohr^2 (chlorofluoromethane); at the theory geometries each
molecule's largest rigid-mode eigenvalue is 23-117 cm^-1 (decisions D-55).
Projecting the rigid modes out (D-52) removes the leak into the vibrations
but not the error inside them: chlorofluoromethane's lowest two modes stayed
5 and 2 cm^-1 too stiff, and its alpha 0.82 MHz off a level-7 grid. Imposing the rule took them to 0.1 cm^-1 and 0.006 MHz.

The engine now imposes it wherever a Hessian enters -- the optimizer's quantum
state and the alpha module's mode analyses (impose_translational_sum_rule).

The test surface is the analytic bent H-N-C valence force field of
test_alpha_rigid_projection: exactly invariant, so the rule holds to its
finite-difference noise (1.4e-7 Ha/bohr^2). The error added to it is an on-site term, a
symmetric 3x3 block per atom -- what a quadrature that moves with the atoms
but is not differentiated with them adds -- of the size measured on acetyl
fluoride (3.5e-3 Ha/bohr^2).
"""

import numpy as np

from backend.spectral import harmonic_alpha as HA
from test_alpha_rigid_projection import MASSES, Q_EQ, _geometry, _hessian_hartree_bohr2

_ONSITE = 3.5e-3      # Ha/bohr^2, acetyl fluoride's measured row-sum violation


def _row_sum_violation(h):
    n = h.shape[0] // 3
    return float(np.abs(h.reshape(n, 3, n, 3).sum(2)).max())


def _onsite_error(n_atoms, size=_ONSITE, seed=7):
    rng = np.random.default_rng(seed)
    d = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(n_atoms):
        a = rng.normal(size=(3, 3))
        a = 0.5 * (a + a.T)
        d[3 * i:3 * i + 3, 3 * i:3 * i + 3] = size * a / np.abs(a).max()
    return d


def _alpha(hess, coords):
    alpha, _, _, _ = HA.compute_harmonic_alpha(hess, coords, MASSES)
    return np.array([alpha["A"], alpha["B"], alpha["C"]])


def test_the_repair_removes_an_on_site_error_and_leaves_an_invariant_hessian_alone():
    from backend.base_backend import impose_translational_sum_rule
    x = _geometry(*Q_EQ)
    h0 = _hessian_hartree_bohr2(x)
    bad = h0 + _onsite_error(3)
    # The premise: the surface obeys the rule to its finite-difference noise,
    # the corrupted Hessian does not. Measured 1.4e-7 and 3.5e-3 (max |H| 1.2).
    assert _row_sum_violation(h0) < 1e-6
    assert _row_sum_violation(bad) > 1e-3
    fixed = impose_translational_sum_rule(bad)
    # Measured 3.6e-9.
    assert _row_sum_violation(fixed) < 1e-8
    # A symmetric on-site error is removed exactly; what is left is the
    # surface's own residue, which goes too (measured 1.4e-7 both).
    assert np.abs(fixed - h0).max() < 1e-6
    assert np.abs(impose_translational_sum_rule(h0) - h0).max() < 1e-6
    # A second pass changes nothing measurable (measured 1.2e-19; D-55:
    # ~1e-18 on a real Hessian).
    assert np.abs(impose_translational_sum_rule(fixed) - fixed).max() < 1e-15


def test_alpha_does_not_see_an_on_site_error(monkeypatch):
    x = _geometry(*Q_EQ)
    h0 = _hessian_hartree_bohr2(x)
    bad = h0 + _onsite_error(3)
    clean = _alpha(h0, x)
    repaired = _alpha(bad, x)
    # The premise, with the repair disabled: the projection alone leaves the
    # error in the vibrational block. Measured A / B / C: -1672 / -3.6 / +1.7
    # MHz (of -98108 / -571 / -169; A is large because I_a is 0.57 amu A^2).
    from backend import base_backend
    with monkeypatch.context() as mp:
        mp.setattr(base_backend, "impose_translational_sum_rule", lambda h: np.asarray(h, dtype=float))
        mp.setattr(HA, "impose_translational_sum_rule", lambda h: np.asarray(h, dtype=float))
        unrepaired = _alpha(bad, x)
    assert np.abs(unrepaired - clean)[1:].max() > 1.0
    # Measured 0.0: the on-site error is removed exactly, so the two
    # repaired Hessians are the same matrix.
    assert np.abs(repaired - clean).max() < 1e-6


def test_the_optimizer_s_quantum_state_satisfies_the_rule():
    from backend.base_backend import QuantumState
    x = _geometry(*Q_EQ)
    h0 = _hessian_hartree_bohr2(x)
    state = QuantumState(0.0, np.zeros(9), h0 + _onsite_error(3))
    # Measured 3.6e-9 and 1.4e-7, as for the bare helper.
    assert _row_sum_violation(state._hessian_bohr) < 1e-8
    assert np.abs(state._hessian_bohr - h0).max() < 1e-6
