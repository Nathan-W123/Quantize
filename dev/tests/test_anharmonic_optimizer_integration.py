"""End-to-end check that the anharmonic term reaches the spectral targets.

Registers a stub backend backed by the analytic water force field so the whole
MolecularOptimizer correction path runs without Psi4 or ORCA.
"""

import numpy as np
import pytest

from backend.base_backend import GradientResult, HessianResult, QuantumBackend, QuantumState
from backend.quantize import MolecularOptimizer
from backend.registry import register_backend

from reference_molecules import H2O_B0_MHZ, H2O_ELEMS, H2O_MASSES, h2o_coords, h2o_hessian


@register_backend
class _AnalyticWaterBackend(QuantumBackend):
    """Water valence force field posing as a quantum backend."""

    name = "analytic_water_test"
    supports_parallel = False

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])
        self.hessian_calls = 0

    def run_hessian(self, coords_ang):
        self.hessian_calls += 1
        hess = h2o_hessian(coords_ang)
        return HessianResult(
            energy=0.0,
            gradient_bohr=np.zeros(np.asarray(coords_ang).size),
            hessian_bohr=hess,
        )

    def run_gradient(self, coords_ang):
        return GradientResult(
            energy=0.0, gradient_bohr=np.zeros(np.asarray(coords_ang).size)
        )


def _isotopologues():
    return [{
        "name": "H2-16O",
        "masses": H2O_MASSES.tolist(),
        "obs_constants": H2O_B0_MHZ.tolist(),
        "sigma_constants": [0.2, 0.2, 0.2],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    }]


def _build(anharmonic: bool, **extra):
    coords = h2o_coords()
    opt = MolecularOptimizer(
        elems=list(H2O_ELEMS),
        coords=coords,
        isotopologues=_isotopologues(),
        quantum_backend="analytic_water_test",
        harmonic_from_hessian=True,
        anharmonic_from_hessian=anharmonic,
        coordinate_mode="cartesian",
        **extra,
    )
    opt._run_hessian()
    opt._apply_harmonic_alpha_corrections()
    return opt


def _targets(opt):
    return np.asarray(opt.spectral.isotopologues[0]["obs_constants"], dtype=float)


@pytest.mark.parametrize("use_autoconfig", [True, False])
@pytest.mark.parametrize("heuristic_bases", [True, False])
def test_optimizer_constructs_with_default_autoconfig(use_autoconfig, heuristic_bases):
    """Both flags default to True, and that combination read self.autoconfig
    before it was assigned -- so no MolecularOptimizer could be built at all."""
    opt = MolecularOptimizer(
        elems=list(H2O_ELEMS),
        coords=h2o_coords(),
        isotopologues=_isotopologues(),
        spectral_only=True,
        use_autoconfig=use_autoconfig,
        use_autoconfig_heuristic_bases=heuristic_bases,
    )
    assert (opt.autoconfig is not None) == use_autoconfig


@pytest.mark.parametrize("period", [1, 2, 3])
def test_due_every_fires_on_the_first_call_for_any_period(period):
    """`count % period == 1` is never true for period 1, since every integer mod
    1 is 0. That silently disabled harmonic_from_hessian at the default setting."""
    due = [c for c in range(1, 10) if MolecularOptimizer._due_every(c, period)]
    assert due[0] == 1
    assert due == list(range(1, 10, period))


def test_corrections_are_applied_during_run_at_default_recalc_period():
    """Regression: with hess_recalc_every=1 the correction chain never ran, so the
    fit targeted ground-state B_0 as if it were B_e."""
    opt = _build(True)
    opt2 = MolecularOptimizer(
        elems=list(H2O_ELEMS),
        coords=h2o_coords(),
        isotopologues=_isotopologues(),
        quantum_backend="analytic_water_test",
        harmonic_from_hessian=True,
        anharmonic_from_hessian=True,
        coordinate_mode="cartesian",
        hess_recalc_every=1,
        max_iter=2,
    )
    opt2.run()
    targets = np.asarray(
        opt2.spectral.isotopologues[0]["obs_constants"], dtype=float
    )
    assert not np.allclose(targets, H2O_B0_MHZ), (
        "spectral targets still equal the raw B_0; corrections never applied"
    )
    assert opt._backend.hessian_calls > 0


def test_optimizer_applies_anharmonic_correction():
    """The corrected targets must differ, and move toward the geometric B_e."""
    harm = _build(False)
    full = _build(True)

    t_harm, t_full = _targets(harm), _targets(full)
    assert not np.allclose(t_harm, t_full)

    be_target = harm.spectral.rotational_constants(h2o_coords(), H2O_MASSES)
    err_harm = np.abs(t_harm - be_target).sum()
    err_full = np.abs(t_full - be_target).sum()
    assert err_full < err_harm


def test_anharmonic_costs_6n_extra_hessians():
    """Cartesian cubic force field: 2 displacements x 3N coordinates."""
    harm, full = _build(False), _build(True)
    n_cart = 3 * len(H2O_ELEMS)
    assert harm._backend.hessian_calls == 1
    assert full._backend.hessian_calls == 1 + 2 * n_cart


def test_cubic_field_built_once_for_many_isotopologues():
    """The cubic force field is mass-independent, so extra isotopologues are free."""
    coords = h2o_coords()
    isos = _isotopologues()
    isos.append({
        "name": "D2-16O",
        "masses": [H2O_MASSES[0], 2.01410177812, 2.01410177812],
        "obs_constants": [462278.85, 218038.23, 145258.00],
        "sigma_constants": [0.2, 0.2, 0.2],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    })
    opt = MolecularOptimizer(
        elems=list(H2O_ELEMS),
        coords=coords,
        isotopologues=isos,
        quantum_backend="analytic_water_test",
        harmonic_from_hessian=True,
        anharmonic_from_hessian=True,
        coordinate_mode="cartesian",
    )
    opt._run_hessian()
    opt._apply_harmonic_alpha_corrections()
    assert opt._backend.hessian_calls == 1 + 2 * 3 * len(H2O_ELEMS)
    # Different masses must still give different corrections.
    a = np.asarray(opt.spectral.isotopologues[0]["obs_constants"], dtype=float)
    b = np.asarray(opt.spectral.isotopologues[1]["obs_constants"], dtype=float)
    assert not np.allclose(a, b)


def test_g_tensor_reaches_the_electronic_correction():
    with_g = _build(False, correction_elec=True,
                    correction_g_tensor={"A": 0.645, "B": 0.717, "C": 0.657})
    without_g = _build(False, correction_elec=True)
    # The g-tensor form is over an order of magnitude larger for water.
    assert np.abs(_targets(with_g) - _targets(without_g)).max() > 100.0


def test_no_anharmonic_means_no_extra_hessians_and_wider_sigma():
    harm = _build(False)
    sigma = np.asarray(harm.spectral.isotopologues[0]["sigma_constants"], dtype=float)
    # Sigma must reflect the omitted dominant term, not the 0.2 MHz measurement error.
    assert np.all(sigma > 1.0)
