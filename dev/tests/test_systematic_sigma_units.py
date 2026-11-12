"""The systematic propagation must use the sigma J was weighted by.

`stacked` divides each row by `_effective_sigma`, so J is d(residual)/dp in
units of effective sigma, and a shift in the constants enters as
d/sigma_eff. `_systematic_covariance` divided by the RAW quoted
`sigma_constants` instead. `sigma_floor_rel` floors the weighting sigma at a
fraction of each constant, so on water the two differ by 1090x to 4179x, and
the propagated width was scaled by that ratio.

Measured: a 2.4% systematic -- the repo's own figure for water's model error,
from the `sigma_floor_rel` comment -- gave a 9.4045 Angstrom width on a 0.958
Angstrom bond, and an angle width of 1848 degrees, five full turns. With
consistent units: 0.0128 Angstrom and 0.66 degrees. An independent estimate
from B ~ 1/r^2 puts the bond width at 0.0115.

This is the reconciliation the note in `__init__` recorded as "not done yet"
-- "floor the weighting sigma at the model error, then propagate against
that floor". The floor arrived, in `_effective_sigma`. This propagation kept
reading around it.
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

import dev.analytic_water_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = np.array([835840.29, 435351.72, 278138.70])
B0_D2O = np.array([462324.0, 218038.0, 145258.0])
R_OH = 0.95785
THETA = 104.508
#: Water's model error as a fraction of each constant, from the
#: `sigma_floor_rel` note in spectral.py (0.06% to 2.4% across the set).
WATER_MODEL_ERROR_FRAC = 0.024


def _coords(r=R_OH, deg=THETA):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(sys_frac=None, sigma_mhz=0.2, **kw):
    isos = []
    for name, masses, b0 in (("H2-16O", MASSES_H2O, B0_H2O),
                             ("D2-16O", MASSES_D2O, B0_D2O)):
        d = {"name": name, "masses": masses, "obs_constants": list(b0),
             "sigma_constants": [sigma_mhz] * 3}
        if sys_frac is not None:
            d["sigma_systematic_constants"] = list(sys_frac * b0)
        isos.append(d)
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=3,
        quantum_prior_sigma_ang=0.02,
        # Explicit, so the "no systematic" baseline stays a real baseline
        # whichever way the auto-derivation default goes later.
        auto_systematic_sigma=False, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
        unc = opt.geometry_uncertainty()
    return opt, unc


def _bond_and_angle(unc):
    bonds = [v for k, v in unc.items() if k.count("-") == 1]
    angles = [v for k, v in unc.items() if k.count("-") == 2]
    return max(bonds), max(angles)


@pytest.fixture(scope="module")
def no_systematic():
    return _fit()


@pytest.fixture(scope="module")
def with_systematic():
    return _fit(sys_frac=WATER_MODEL_ERROR_FRAC)


def test_the_floor_really_does_differ_from_the_raw_sigma(no_systematic):
    """Guard the premise: with no floor there is no bug to catch."""
    opt, _ = no_systematic
    iso = opt.spectral.isotopologues[0]
    raw = np.asarray(iso["sigma_constants"], dtype=float)
    eff = np.asarray(opt.spectral._effective_sigma(
        iso["sigma_constants"], iso=iso), dtype=float)
    assert np.all(eff / raw > 100.0), (
        "sigma_floor_rel is not biting, so this test proves nothing")


def test_width_stays_smaller_than_the_molecule(with_systematic):
    """9.4045 Angstrom on a 0.958 Angstrom bond was the shipped value."""
    _opt, unc = with_systematic
    bond, angle = _bond_and_angle(unc)
    assert bond < R_OH, f"bond width {bond:.4f} Ang exceeds the bond itself"
    assert angle < 180.0, f"angle width {angle:.1f} deg exceeds a half turn"


_AMU_A2_TO_MHZ = 505379.0045


def _rigid_rotor_constants(r, theta, masses):
    """A, B, C of a C2v triatomic, from the moments of inertia alone."""
    t = theta / 2.0
    X = np.array([[0.0, 0.0, 0.0],
                  [r * np.sin(t), r * np.cos(t), 0.0],
                  [-r * np.sin(t), r * np.cos(t), 0.0]])
    m = np.asarray(masses, dtype=float)
    X = X - (m[:, None] * X).sum(0) / m.sum()
    inertia = np.zeros((3, 3))
    for mi, xi in zip(m, X):
        inertia += mi * ((xi @ xi) * np.eye(3) - np.outer(xi, xi))
    return _AMU_A2_TO_MHZ / np.sort(np.linalg.eigvalsh(inertia))


def _independent_systematic_bond_sigma(opt, correlated):
    """The systematic bond width from a rigid rotor, with none of the engine's
    covariance code: J by finite differences in (r, theta), the engine's own
    effective sigmas as weights, and the model error propagated through the
    least-squares estimator as A Sigma_sys A^T."""
    def model(p):
        return np.concatenate([_rigid_rotor_constants(p[0], p[1], MASSES_H2O),
                               _rigid_rotor_constants(p[0], p[1], MASSES_D2O)])
    p0 = np.array([R_OH, np.radians(THETA)])
    step = np.array([1e-6, 1e-7])
    J = np.column_stack([
        (model(p0 + step[k] * np.eye(2)[k]) - model(p0 - step[k] * np.eye(2)[k]))
        / (2 * step[k]) for k in range(2)])
    seff = np.concatenate([
        np.asarray(opt.spectral._effective_sigma(iso["sigma_constants"], iso=iso),
                   dtype=float).ravel() for iso in opt.spectral.isotopologues])
    W = np.diag(1.0 / seff ** 2)
    F = np.linalg.inv(J.T @ W @ J)
    d = WATER_MODEL_ERROR_FRAC * np.concatenate([B0_H2O, B0_D2O])
    S = np.outer(d, d) if correlated else np.diag(d ** 2)
    C = F @ J.T @ W @ S @ W @ J @ F
    return float(np.sqrt(C[0, 0]))


def test_width_matches_an_independent_physics_estimate(with_systematic):
    """Pinned against a rigid rotor computed here, not against a hand formula.

    This used to compare with B ~ 1/r^2, dr/r = (1/2) dB/B, i.e. 0.0115 Ang,
    and the engine's 0.0128 passed. It passed by coincidence. 93% of that
    0.0128 was not systematic at all: it was the prior-only width of the
    antisymmetric stretch, a direction the imposed C2v holds fixed (0.0119 of
    it, against 0.0048 that was genuinely the model error). With that
    direction removed the engine reports 0.00478.

    And the hand formula answers a different question. dr/r = (1/2) dB/B is
    exact for a COHERENT error -- every constant scaled by the same 1 + eps --
    which a rigid rotor confirms (0.01143 Ang). The engine propagates model
    error as INDEPENDENT per constant, diag(sigma_sys^2), and six independent
    errors on two parameters average down: the same rigid rotor gives 0.00473
    Ang for that, against the engine's 0.00478. So the engine is right about
    the model it implements, to 1%.

    Which model is right is not settled here and is flagged rather than
    decided: `ReferenceMolecule.sigmas_systematic` describes the error as "an
    offset that biases every isotopologue the same way", which is the coherent
    case, and on this configuration the coherent width is 2.4x the reported
    one. A B0-versus-Be offset is coherent in sign within a species but not
    uniform across A, B and C, so the truth lies between the two.
    """
    opt, unc = with_systematic
    bond, _angle = _bond_and_angle(unc)
    independent = _independent_systematic_bond_sigma(opt, correlated=False)
    coherent = _independent_systematic_bond_sigma(opt, correlated=True)
    assert bond == pytest.approx(independent, rel=0.05), (
        f"bond width {bond:.5f} Ang against an independent rigid-rotor "
        f"estimate of {independent:.5f} Ang for per-constant model error")
    # The hand formula is the coherent limit, and the rigid rotor agrees with
    # it -- which is what makes the 2.4x gap a statement about the error
    # model rather than about the arithmetic.
    assert coherent == pytest.approx(0.5 * WATER_MODEL_ERROR_FRAC * R_OH, rel=0.02)
    assert coherent / bond > 2.0


def test_systematic_widens_rather_than_narrows(no_systematic, with_systematic):
    _opt0, unc0 = no_systematic
    _opt1, unc1 = with_systematic
    b0, a0 = _bond_and_angle(unc0)
    b1, a1 = _bond_and_angle(unc1)
    assert b1 > b0 and a1 > a0


def test_model_error_inflates_every_coordinate_by_the_same_factor(no_systematic, with_systematic):
    """sqrt(1 + (sigma_sys / sigma_eff)^2), for the bond and the angle alike.

    The statistical and systematic widths propagate through the same matrix,
    A = C J^T, so when sigma_sys / sigma_eff is the same on every row the
    ratio of total to statistical width is that one number for every
    coordinate. Here sigma_eff is the relative floor (0.001 of each constant)
    and sigma_sys is 0.024 of it, so the prediction is sqrt(1 + 24^2) =
    24.0208. Measured: 24.0175 on the bond, 24.0200 on the angle.

    This replaces an assertion that the angle is hit five times harder than
    the bond. That held only because the bond's statistical width was inflated
    by the antisymmetric stretch -- a direction the imposed C2v holds fixed --
    which made its ratio look small.
    """
    o1, unc1 = with_systematic
    _o0, unc0 = no_systematic
    b0, a0 = _bond_and_angle(unc0)
    b1, a1 = _bond_and_angle(unc1)
    floor = float(o1.spectral.sigma_floor_rel)
    predicted = float(np.sqrt(1.0 + (WATER_MODEL_ERROR_FRAC / floor) ** 2))
    assert b1 / b0 == pytest.approx(predicted, rel=1e-3)
    assert a1 / a0 == pytest.approx(predicted, rel=1e-3)


def test_variance_scales_as_the_square_of_the_systematic():
    """The propagation is C_sys = A diag(sigma_sys^2) A^T, so the systematic
    part of the VARIANCE must be quadratic in sigma_sys. Checked by
    subtracting the statistical part rather than by eyeballing the width."""
    _o, unc_stat = _fit()
    b_stat, _a = _bond_and_angle(unc_stat)
    ratios = []
    for frac in (0.006, 0.012, 0.024):
        _o, unc = _fit(sys_frac=frac)
        b, _a2 = _bond_and_angle(unc)
        sys_var = b ** 2 - b_stat ** 2
        assert sys_var > 0.0
        ratios.append(sys_var / frac ** 2)
    # All three must give the same constant of proportionality.
    assert max(ratios) / min(ratios) < 1.15, ratios


def test_a_tighter_floor_does_not_blow_the_width_up():
    """The bug's signature was a width proportional to sigma_eff/sigma_raw.

    Lowering sigma_floor_rel lowers sigma_eff, so under the old code the
    reported width moved with the floor. It must not.
    """
    widths = []
    for floor in (1e-4, 1e-3, 1e-2):
        _o, unc = _fit(sys_frac=WATER_MODEL_ERROR_FRAC, sigma_floor_rel=floor)
        widths.append(_bond_and_angle(unc)[0])
    for w in widths:
        assert w < R_OH, f"width {w:.4f} Ang tracks the floor"
    assert max(widths) / min(widths) < 3.0, widths


def test_no_systematic_supplied_means_no_systematic_term(no_systematic):
    """Zero, not a small number: the term must vanish when unsupplied."""
    opt, _unc = no_systematic
    J, _r = opt.spectral.stacked(opt.coords)
    cov, _std = opt.laplace_approximation(check_gauss_newton=False)
    extra = opt._systematic_covariance(J, cov)
    assert np.array_equal(extra, np.zeros_like(cov))
