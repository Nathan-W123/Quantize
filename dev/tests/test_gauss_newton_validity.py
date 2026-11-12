"""The Laplace width must say when its own Hessian approximation has failed.

`laplace_approximation` builds JtJ, the Gauss-Newton form of Hess(0.5*chi^2).
The exact Hessian is JtJ + sum_i r_i Hess(r_i). Gauss-Newton drops the second
term, which is standard and harmless for small residuals -- and this engine
does not run at small residuals. With an uncorrected B0-versus-Be offset
against sigma_constants in fractions of a MHz, water sits at 18 sigma rms.

Measured there: the omitted term is 3.0% of ||JtJ|| in norm, while the softest
physical curvature is 0.04% of ||JtJ||. So the perturbation is 75x the
quantity it perturbs and flips its sign, +7,995 -> -372,388. The optimum is a
saddle of the exact posterior along that direction, the posterior is improper
there, and no sampler can converge on it -- which is why this engine's MCMC
returns R-hat anywhere from 2.3 to 18 depending on the seed.

Gauss-Newton reports all-positive curvature and a finite width throughout,
with nothing to say so. These tests pin that it now says so, and -- just as
importantly -- that it stays quiet when the approximation is actually fine.

WHICH DIRECTION THAT IS, AND WHY THESE FITS IMPOSE NO SYMMETRY. The softest
direction is the antisymmetric stretch. Rotational constants are even in it,
so at a symmetric geometry the data has no first-order sensitivity to it at
all (J v = 1.3e-11 against 7.8e3 on the next direction) and its whole exact
curvature is the sum_i r_i d2r_i term Gauss-Newton drops. Under the C2v that
symmetry="auto" detects and imposes, that direction is not a parameter: the
fit symmetrises after every step, so the reported posterior excludes it (see
MolecularOptimizer._restrict_to_symmetric) and on the two directions that
remain Gauss-Newton and the exact Hessian agree to 2.4%.

So the failure these tests exist to catch is real only where the
antisymmetric stretch is free, and the fixture says so explicitly with
symmetry="C1". test_symmetric_fit_has_no_failure_to_report pins the other
side: under the imposed group the check finds nothing, because there is
nothing to find.

The same direction was the MCMC's problem. With proposals along all three
directions -- what the sampler did before -- four seeds gave max R-hat 2.25 to
3.42 and min ESS 34 to 46, none usable. Restricted to the two symmetric
directions, the same four seeds give R-hat 1.018 to 1.054 and ESS 166 to 185,
all usable: the saddle the chains wandered along is not in the model.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.posterior import laplace_approximation  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(sigma_mhz):
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O,
                        "sigma_constants": [sigma_mhz] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02, symmetry="C1")
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    return opt


@pytest.fixture(scope="module")
def tight_sigma():
    """sigma_constants = 0.2 MHz: 18 sigma residuals, Gauss-Newton invalid."""
    return _fit(0.2)


@pytest.fixture(scope="module")
def loose_sigma():
    """sigma_constants = 2000 MHz: 2.1 sigma residuals, Gauss-Newton fine."""
    return _fit(2000.0)


def _rms_residual(opt):
    _J, rw = opt.spectral.stacked(opt.coords)
    rw = np.asarray(rw, dtype=float)
    return float(np.sqrt(np.sum(rw ** 2) / rw.size))


def test_the_two_fixtures_really_are_the_two_regimes(tight_sigma, loose_sigma):
    """Guard the premise: these tests mean nothing if both fits are the same."""
    assert _rms_residual(tight_sigma) > 10.0
    assert _rms_residual(loose_sigma) < 5.0


def test_sign_flip_is_detected_at_large_residuals(tight_sigma):
    rows = tight_sigma.gauss_newton_discrepancy()
    assert rows, "the check returned nothing"
    soft = rows[0]
    assert soft["curvature_gauss_newton"] > 0.0
    assert soft["curvature_exact"] < 0.0
    assert soft["sign_flipped"] is True
    assert soft["fd_stable"] is True


def test_the_flip_is_reported_not_just_computed(tight_sigma):
    msgs = tight_sigma.gauss_newton_warnings()
    assert len(msgs) >= 1
    joined = " ".join(msgs).lower()
    assert "gauss-newton" in joined
    assert "saddle" in joined


def test_quiet_when_gauss_newton_is_valid(loose_sigma):
    """The check has to be worth believing, so it must not cry wolf."""
    assert loose_sigma.gauss_newton_warnings() == []
    rows = loose_sigma.gauss_newton_discrepancy()
    assert rows
    for row in rows:
        assert row["sign_flipped"] is False
        assert row["curvature_exact"] > 0.0


def test_softest_direction_carries_no_spectral_curvature(tight_sigma):
    """Where the reported width actually comes from.

    The softest direction's Gauss-Newton curvature from the DATA is zero to
    round-off. Its whole +7,995 is the quantum prior -- and the omitted term
    is tens of times that prior. Worth pinning, because it means the width
    quoted there is a prior width being overwhelmed by an uncounted term,
    not a measurement.
    """
    soft = tight_sigma.gauss_newton_discrepancy()[0]
    scale = abs(soft["curvature_gauss_newton"])
    assert abs(soft["curvature_spectral_gauss_newton"]) < 1e-6 * scale
    assert soft["curvature_quantum"] > 0.0
    assert abs(soft["omitted_term"]) > 10.0 * soft["curvature_quantum"]


def test_directional_check_matches_a_full_hessian(tight_sigma):
    """Two residual evaluations per direction against O(n^2) for the matrix.

    The directional probe is only worth having if it gets the same answer.
    Build the exact Hessian of 0.5*chi^2 by finite differences and compare
    the softest physical curvature.
    """
    opt = tight_sigma
    x0 = opt.coords.ravel().copy()
    n = x0.size
    h = 2e-5

    def half_chi2(x):
        _J, r = opt.spectral.stacked(x.reshape(-1, 3))
        return 0.5 * float(np.sum(np.asarray(r) ** 2))

    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b, c, d = (x0.copy() for _ in range(4))
            a[i] += h; a[j] += h
            b[i] += h; b[j] -= h
            c[i] -= h; c[j] += h
            d[i] -= h; d[j] -= h
            H[i, j] = H[j, i] = (half_chi2(a) - half_chi2(b)
                                 - half_chi2(c) + half_chi2(d)) / (4 * h * h)

    soft = opt.gauss_newton_discrepancy()[0]
    # Reconstruct the same direction the method used.
    J, _ = opt.spectral.stacked(opt.coords)
    g_q, H_q = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    a_eff = opt.optimizer.effective_quantum_weight(
        J, int(np.linalg.matrix_rank(J)), hessian=H_q)
    H_quant = a_eff * opt.optimizer._spd_hessian(H_q)
    H_gn = J.T @ J + H_quant
    cov = np.linalg.inv(H_gn + 1e-8 * np.eye(n))
    _cp, basis, _g = opt.physical_covariance(cov)
    blk = basis.T @ H_gn @ basis
    w, V = np.linalg.eigh((blk + blk.T) / 2)
    v = basis @ V[:, 0]
    v /= np.linalg.norm(v)
    exact_full = float(v @ H @ v) + float(v @ H_quant @ v)

    assert np.sign(exact_full) == np.sign(soft["curvature_exact"])
    rel = abs(exact_full - soft["curvature_exact"]) / abs(exact_full)
    assert rel < 0.05, (
        f"directional probe {soft['curvature_exact']:.4e} vs full Hessian "
        f"{exact_full:.4e}: {100 * rel:.1f}% apart")


def test_finite_difference_stability_is_reported(tight_sigma):
    """A sign flip from a cancelling subtraction must not be reported as one."""
    rows = tight_sigma.gauss_newton_discrepancy()
    for row in rows:
        assert isinstance(row["fd_stable"], bool)
        assert row["step"] > 0.0


def test_check_can_be_switched_off(tight_sigma):
    """It costs residual evaluations, so it has to be optional."""
    cov_on, std_on = tight_sigma.laplace_approximation(check_gauss_newton=True)
    cov_off, std_off = tight_sigma.laplace_approximation(check_gauss_newton=False)
    # A diagnostic must not change the number it qualifies.
    assert np.array_equal(cov_on, cov_off)
    assert np.array_equal(std_on, std_off)


def test_v2_summary_carries_the_finding(tight_sigma):
    """The serialised object is what gets read later, with no stdout around."""
    summary = laplace_approximation(tight_sigma)
    assert any("Gauss-Newton" in w for w in summary.warnings), summary.warnings
    assert any("Gauss-Newton" in w for w in summary.to_dict()["warnings"])


def test_v2_summary_is_clean_when_valid(loose_sigma):
    summary = laplace_approximation(loose_sigma)
    assert not any("Gauss-Newton" in w for w in summary.warnings), summary.warnings


def test_diagnostic_never_breaks_the_fit(tight_sigma, monkeypatch):
    """A broken diagnostic must degrade to a warning, not kill a run."""
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(tight_sigma, "gauss_newton_discrepancy", boom)
    assert tight_sigma.gauss_newton_warnings() == []
    cov, std = tight_sigma.laplace_approximation()
    assert cov is not None and std is not None


def test_symmetric_fit_has_no_failure_to_report():
    """Under the imposed C2v, Gauss-Newton is valid on every direction left.

    The same fit as the fixtures above, but with the default symmetry
    handling. The antisymmetric stretch is then fixed by the constraint, the
    two remaining directions agree with the exact Hessian to 2.4%, and the
    check must stay silent -- a warning here would be the probe reporting on a
    direction the model does not have.
    """
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(),
        isotopologues=[{"name": "H2-16O", "masses": MASSES_H2O,
                        "obs_constants": B0_H2O,
                        "sigma_constants": [0.2] * 3}],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
    assert opt.symmetry is not None and opt.symmetry.point_group == "C2v"
    assert opt.physical_basis().shape[1] == 2
    assert opt.gauss_newton_warnings() == []
    summary = laplace_approximation(opt)
    assert not any("Gauss-Newton" in w for w in summary.warnings)

