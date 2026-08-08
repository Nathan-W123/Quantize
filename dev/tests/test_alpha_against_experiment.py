"""Validate the B0 -> Be correction chain against closed-form and experimental values.

These pin down the accuracy claims that motivate the rovibrational correction
machinery. They use analytic potentials, so they run without Psi4 or ORCA.
"""

import numpy as np
import pytest

from backend.spectral.centrifugal_distortion import (
    is_linear,
    normal_modes,
    rigid_mode_count,
    rotational_constants_mhz,
    _MHZ_TO_CM,
)
from backend.spectral.harmonic_alpha import compute_harmonic_alpha

from reference_molecules import (
    CO_ALPHA_E,
    CO_COORDS,
    CO_MASSES,
    CO_OMEGA_E,
    co_harmonic_alpha,
    co_hessian,
    co_pekeris_alpha,
    H2O_B0_MHZ,
    H2O_MASSES,
    h2o_coords,
    h2o_hessian,
)


# ── Linearity / mode counting ────────────────────────────────────────────────

def test_is_linear_classifies_diatomic_and_water():
    assert is_linear(CO_COORDS, CO_MASSES)
    assert not is_linear(h2o_coords(), H2O_MASSES)


def test_rigid_mode_count_matches_3n_minus_5_or_6():
    assert rigid_mode_count(CO_COORDS, CO_MASSES) == 5
    assert rigid_mode_count(h2o_coords(), H2O_MASSES) == 6


def test_linear_molecule_keeps_all_vibrations():
    """n_rigid=6 on a diatomic discards the only real vibration."""
    hess = co_hessian(CO_COORDS)
    omega, _ = normal_modes(hess, CO_MASSES, n_rigid=6)
    assert len(omega) == 0, "sanity: the old hardcoded n_rigid=6 loses the mode"

    omega, _ = normal_modes(
        hess, CO_MASSES, n_rigid=rigid_mode_count(CO_COORDS, CO_MASSES)
    )
    assert len(omega) == 1
    assert omega[0] == pytest.approx(CO_OMEGA_E, rel=1e-6)


def test_diatomic_alpha_is_not_silently_zero():
    """A zero alpha with zero sigma would tell the fit that B_0 == B_e."""
    alpha, _, sigma, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES
    )
    assert abs(alpha["B"]) > 0.0
    assert all(s > 0.0 for s in sigma.values())


# ── Diatomic closed forms ────────────────────────────────────────────────────

def test_harmonic_alpha_matches_dunham_closed_form():
    """For a diatomic the harmonic alpha is exactly -6 Be^2 / we."""
    alpha, _, _, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES
    )
    assert alpha["B"] * _MHZ_TO_CM == pytest.approx(co_harmonic_alpha(), rel=1e-4)


def test_coriolis_vanishes_for_a_diatomic():
    """A diatomic has one mode and no vibrational angular momentum."""
    _, _, _, info = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES
    )
    assert info["alpha_coriolis_mhz"]["B"] == pytest.approx(0.0, abs=1e-9)


def test_anharmonic_alpha_matches_pekeris_relation():
    """Exact Morse Hessians must reproduce the Pekeris/Dunham alpha_e."""
    alpha, _, _, info = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES, hessian_fn=co_hessian,
    )
    assert info["anharmonic_status"] == "cubic_fd"
    assert alpha["B"] * _MHZ_TO_CM == pytest.approx(co_pekeris_alpha(), rel=2e-3)


def test_anharmonic_term_dominates_and_flips_the_sign():
    """Harmonic-only alpha has the wrong sign for CO; the cubic term fixes it."""
    harm_only, _, _, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES
    )
    full, _, _, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES, hessian_fn=co_hessian,
    )
    assert harm_only["B"] < 0.0 < CO_ALPHA_E
    assert full["B"] > 0.0

    harm_err = abs(harm_only["B"] * _MHZ_TO_CM - CO_ALPHA_E)
    full_err = abs(full["B"] * _MHZ_TO_CM - CO_ALPHA_E)
    assert full_err < 0.25 * harm_err


def test_omitting_anharmonic_term_widens_the_reported_sigma():
    """Reported uncertainty must reflect the omitted dominant term."""
    _, _, sig_harm, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES, sigma_fraction=0.02
    )
    _, _, sig_full, _ = compute_harmonic_alpha(
        co_hessian(CO_COORDS), CO_COORDS, CO_MASSES,
        sigma_fraction=0.02, hessian_fn=co_hessian,
    )
    assert sig_harm["B"] > 10.0 * sig_full["B"]


# ── Water: symmetry and end-to-end accuracy ──────────────────────────────────

def test_coriolis_respects_c2v_symmetry():
    """For C2v water only rotation about the out-of-plane (c) axis has B2
    symmetry, so only alpha^C may carry a Coriolis contribution."""
    coords = h2o_coords()
    _, _, _, info = compute_harmonic_alpha(
        h2o_hessian(coords), coords, H2O_MASSES
    )
    cor = info["alpha_coriolis_mhz"]
    assert cor["A"] == pytest.approx(0.0, abs=1e-6)
    assert cor["B"] == pytest.approx(0.0, abs=1e-6)
    assert abs(cor["C"]) > 1.0


def test_water_force_field_reproduces_observed_frequencies():
    coords = h2o_coords()
    omega, _ = normal_modes(
        h2o_hessian(coords), H2O_MASSES,
        n_rigid=rigid_mode_count(coords, H2O_MASSES),
    )
    assert len(omega) == 3
    # bend, symmetric stretch, antisymmetric stretch
    assert omega[0] == pytest.approx(1600.0, abs=120.0)
    assert omega[1] == pytest.approx(3832.0, abs=120.0)
    assert omega[2] == pytest.approx(3943.0, abs=120.0)


def test_water_be_correction_improves_with_anharmonic_term():
    """B_e recovered from B_0 must move toward the B_e implied by the accepted
    equilibrium geometry once the cubic term is included."""
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    be_target = rotational_constants_mhz(coords, H2O_MASSES)
    required = be_target - H2O_B0_MHZ

    def total_error(**kw):
        alpha, _, _, _ = compute_harmonic_alpha(hess, coords, H2O_MASSES, **kw)
        delta = 0.5 * np.array([alpha["A"], alpha["B"], alpha["C"]])
        return float(np.abs(delta - required).sum())

    err_harm = total_error()
    err_full = total_error(hessian_fn=h2o_hessian)
    assert err_full < 0.25 * err_harm


def test_water_alpha_signs_match_the_required_correction():
    """Each component of the correction must at least have the right sign."""
    coords = h2o_coords()
    be_target = rotational_constants_mhz(coords, H2O_MASSES)
    required = be_target - H2O_B0_MHZ
    alpha, _, _, _ = compute_harmonic_alpha(
        h2o_hessian(coords), coords, H2O_MASSES, hessian_fn=h2o_hessian,
    )
    delta = 0.5 * np.array([alpha["A"], alpha["B"], alpha["C"]])
    assert np.all(np.sign(delta) == np.sign(required))
