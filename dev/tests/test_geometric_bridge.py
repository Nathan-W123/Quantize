"""The joint objective, expressed as an energy and handed to someone else's optimiser.

The in-house optimiser solves the hybrid problem as one linear system with the
spectral residual and the quantum prior side by side. The bridge instead folds
the spectral residual into the energy and lets geomeTRIC optimise that. If both
routes are right they must agree, which makes this an independent check on the
in-house optimiser rather than only a test of the bridge: the two share no
coordinate system, no trust-radius logic and no step acceptance rule.

The weight that makes the two equivalent is the crux. alpha_q converts
Hartrees into chi-square units so the quantum term can join a spectral
residual; the bridge needs the reciprocal, to turn a chi-square into Hartrees.
Getting that backwards or off by a factor would still optimise -- just a
different problem than the one the engine solves -- so it is pinned directly
against the optimiser's own calibration.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.geometric_bridge import (  # noqa: E402
    calibrated_spectral_weight,
    spectral_chi2_and_gradient,
)
from backend.spectral.centrifugal_distortion import (  # noqa: E402
    rotational_constants_mhz,
)
from backend.spectral.SVD import SubspaceOptimizer  # noqa: E402
from reference_molecules import H2O_MASSES, h2o_coords, h2o_hessian  # noqa: E402

_BOHR_PER_ANG = 1.8897261254578281


def _targets(coords, masses, offsets_mhz=(0.0, 0.0, 0.0), sigma=100.0):
    """Targets built from a known geometry, optionally detuned."""
    abc = rotational_constants_mhz(coords, masses)
    return [{"masses": np.asarray(masses, dtype=float), "component": k,
             "value": float(abc[k]) + offsets_mhz[k], "sigma": sigma}
            for k in range(3)]


def test_chi2_and_gradient_both_vanish_at_the_generating_geometry():
    """No residual and no force where the targets came from.

    The gradient mattering here is the point: this term is added to a real
    energy, so a spurious force at the target would drag a converged structure
    off it. A finite-differenced version of this function failed exactly this
    assertion -- 40 /Angstrom against a true zero -- which is why the
    derivative is analytic.
    """
    coords = h2o_coords()
    chi2, grad = spectral_chi2_and_gradient(coords, _targets(coords, H2O_MASSES))
    assert chi2 == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(grad)) < 1e-6


def test_chi2_gradient_matches_finite_difference_of_chi2():
    """Independent check on the analytic derivative: a wrong sign optimises uphill.

    The reference here is a numerical difference of chi-square itself, which
    shares no code with the analytic Jacobian path, so agreement is evidence
    rather than a tautology.
    """
    coords = h2o_coords()
    targets = _targets(coords, H2O_MASSES, offsets_mhz=(5000.0, -3000.0, 800.0))
    chi2, grad = spectral_chi2_and_gradient(coords, targets)
    assert chi2 > 0

    flat = coords.ravel()
    h = 1e-5
    for k in (0, 4, 7):
        plus, minus = flat.copy(), flat.copy()
        plus[k] += h
        minus[k] -= h
        num = (spectral_chi2_and_gradient(plus.reshape(-1, 3), targets)[0]
               - spectral_chi2_and_gradient(minus.reshape(-1, 3), targets)[0]) / (2 * h)
        assert grad[k] == pytest.approx(num, rel=0.02, abs=1e-6)


def test_weight_is_the_reciprocal_of_the_optimisers_own_alpha_q():
    """The equivalence the bridge rests on, checked against the source of truth.

    SubspaceOptimizer._calibrated_alpha_q answers "how many chi-square units is
    a Hartree worth"; the bridge needs the inverse. If these drift apart the
    two code paths are solving different problems while both appearing to work.
    """
    hess_ang2 = h2o_hessian(h2o_coords()) * _BOHR_PER_ANG ** 2
    sigma_x = 0.02

    opt = SubspaceOptimizer(1e-3, 0.0, 0.1, None, 1e-4,
                            objective_mode="joint", alpha_quantum=1.0,
                            quantum_prior_sigma_ang=sigma_x)
    alpha_q = opt._calibrated_alpha_q(hess_ang2)
    weight = calibrated_spectral_weight(hess_ang2, sigma_x)

    assert weight == pytest.approx(1.0 / alpha_q, rel=1e-10)


def test_weight_scales_with_the_square_of_the_trusted_displacement():
    """A theory trusted twice as far should let the data pull four times harder,
    since the prior is Gaussian in displacement."""
    hess_ang2 = h2o_hessian(h2o_coords()) * _BOHR_PER_ANG ** 2
    w1 = calibrated_spectral_weight(hess_ang2, 0.01)
    w2 = calibrated_spectral_weight(hess_ang2, 0.02)
    assert w2 / w1 == pytest.approx(4.0, rel=1e-9)


def test_weight_survives_a_hessian_with_no_positive_curvature():
    """Rigid-mode-only or pathological Hessians must not produce a zero or
    negative weight, which would silently delete or invert the data term."""
    assert calibrated_spectral_weight(np.zeros((9, 9)), 0.02) > 0.0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("geometric") is None,
    reason="geomeTRIC not installed",
)
def test_restrained_optimisation_moves_toward_the_spectral_target():
    """End to end on an analytic surface: the restraint must actually bite.

    The analytic water backend stands in for a quantum package, so this runs
    the full geomeTRIC path -- TRIC coordinates, trust radius, step acceptance
    -- without a real SCF. Starting from the surface's own minimum, a detuned
    target must pull the geometry away from it and reduce the spectral
    residual; that is the whole point of the restraint.
    """
    import contextlib
    import io
    import tempfile

    import dev.analytic_water_backend  # noqa: F401  (registers the backend)
    from backend.geometric_bridge import optimise_with_tric
    from backend.registry import get_backend

    elems = ["O", "H", "H"]
    backend = get_backend("analytic_water")(elems=elems)
    start = h2o_coords()

    # Ask for a B constant 2% below the surface's own, so the minimum of the
    # joint objective is a real distance from the minimum of the energy.
    abc = rotational_constants_mhz(start, H2O_MASSES)
    targets = [{"masses": np.asarray(H2O_MASSES, dtype=float), "component": 1,
                "value": float(abc[1]) * 0.98, "sigma": float(abc[1]) * 0.002}]

    hess_ang2 = h2o_hessian(start) * _BOHR_PER_ANG ** 2
    weight = calibrated_spectral_weight(hess_ang2, 0.05)

    chi2_start, _ = spectral_chi2_and_gradient(start, targets)
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            final, engine = optimise_with_tric(
                elems, start, targets, backend, weight, tmp,
                coordsys="tric", maxiter=60)
    chi2_end, _ = spectral_chi2_and_gradient(final, targets)

    assert chi2_end < chi2_start, (
        f"restraint did not pull the geometry: chi2 {chi2_start:.3g} -> "
        f"{chi2_end:.3g}"
    )
    assert engine.n_calls > 1
    moved = float(np.max(np.abs(np.asarray(final) - start)))
    assert 1e-4 < moved < 0.5, f"implausible displacement {moved:.4g} A"
