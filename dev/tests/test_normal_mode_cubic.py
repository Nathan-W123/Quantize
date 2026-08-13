"""The normal-mode cubic force field: correct on a clean surface, robust on a noisy one.

The Cartesian finite-difference cubic force field is exact in exact arithmetic
and validated on the analytic water surface -- and produced cubic/harmonic
ratios of 233-990 on a real B3LYP surface, because Cartesian displacements mix
in rigid rotation and use one step size for every mode, so DFT Hessian noise is
amplified into the third derivative. These tests pin the two claims the
normal-mode replacement is being shipped on:

1. On a clean surface it agrees with the validated Cartesian result.
2. On a noisy surface it degrades far more gracefully -- which is the entire
   reason it exists, and would be silently lost by any regression that
   reintroduced fixed-step or rigid-contaminated displacements.

Plus the isotopologue transform: the derivative field is measured along the
parent's modes, and another isotopologue's modes leave that span (mass
weighting moves the vibrational subspace). The rigid-rotation component is
completed analytically via the commutator [G, H]; the test checks the
decomposed-and-completed result against a direct measurement along the heavy
isotopologue's own modes, which is the ground truth the transform replaces.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.spectral.centrifugal_distortion import (  # noqa: E402
    normal_modes,
    rigid_mode_count,
    rotational_constants_mhz,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    compute_harmonic_alpha,
    normal_mode_hessian_derivatives,
    phi_semidiagonal_for_masses,
)
from reference_molecules import (  # noqa: E402
    H2O_B0_MHZ,
    H2O_MASSES,
    h2o_coords,
    h2o_hessian,
)

D2O_MASSES = np.array([15.9949146196, 2.0141017778, 2.0141017778])


def _noisy_hessian(scale, seed_from_coords=True):
    """A deterministic symmetric perturbation, standing in for SCF/grid noise.

    Seeded from the geometry so the same point always gets the same noise --
    real convergence noise is reproducible per geometry too, and a random draw
    per call would average out in central differences and understate the
    problem.
    """
    def fn(coords):
        h = h2o_hessian(coords)
        seed = int(abs(np.sum(coords)) * 1e6) % (2**32)
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, scale, h.shape)
        return h + 0.5 * (n + n.T)
    return fn


def _alpha_error(alpha):
    coords = h2o_coords()
    be_target = rotational_constants_mhz(coords, H2O_MASSES)
    required = be_target - H2O_B0_MHZ
    delta = 0.5 * np.array([alpha["A"], alpha["B"], alpha["C"]])
    return float(np.abs(delta - required).sum())


def _mode_derivs(hessian_fn, **kw):
    coords = h2o_coords()
    return normal_mode_hessian_derivatives(
        hessian_fn, coords, h2o_hessian(coords), H2O_MASSES, **kw)


def test_clean_surface_normal_mode_matches_cartesian():
    """Both schemes are unbiased on an exact surface; they must agree."""
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    a_cart, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES, hessian_fn=h2o_hessian)
    a_nm, _, _, info = compute_harmonic_alpha(
        hess, coords, H2O_MASSES, mode_derivs=_mode_derivs(h2o_hessian))
    assert info["anharmonic_status"] == "cubic_nm"
    for k in ("A", "B", "C"):
        assert a_nm[k] == pytest.approx(a_cart[k], rel=0.05), (
            f"alpha_{k}: nm {a_nm[k]:.2f} vs cartesian {a_cart[k]:.2f} MHz"
        )


def test_clean_surface_normal_mode_recovers_be():
    """The end-to-end standard the Cartesian path already met."""
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    a_harm, _, _, _ = compute_harmonic_alpha(hess, coords, H2O_MASSES)
    a_nm, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES, mode_derivs=_mode_derivs(h2o_hessian))
    assert _alpha_error(a_nm) < 0.25 * _alpha_error(a_harm)


def test_noise_diagnostic_has_a_truncation_floor_on_a_clean_surface():
    """The asymmetry is a bound on TOTAL finite-difference error.

    On an exact surface it does not vanish: the steps are per-mode, so the
    quartic truncation contamination differs between phi3[r,s,t] and
    phi3[s,r,t], and their difference is genuine truncation error, measured
    0.094 here. That is the floor the noise gate sits on -- and an honest ~10%
    error estimate on phi, which the sigma model consumes. A regression that
    drove this toward zero would mean the steps stopped resolving anharmonicity
    at all; one that grew it means steps or surface got worse.
    """
    md = _mode_derivs(h2o_hessian)
    assert 0.02 < md["noise_ratio"] < 0.2, (
        f"asymmetry {md['noise_ratio']:.3g}; expected truncation-dominated ~0.09"
    )


def test_noisy_surface_normal_mode_degrades_more_gracefully():
    """The shipping claim. Same noise, both schemes, measured side by side.

    The noise scale is set so the Cartesian scheme visibly breaks (its step is
    0.01 A and its displacements excite rigid rotation), while remaining small
    against the Hessian's own entries -- the regime the B3LYP failure lived in.
    """
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    noisy = _noisy_hessian(2e-5)

    # step_scale pinned at 0.35 so both schemes see comparable displacement
    # magnitudes: this test isolates the SCHEME advantage (rotation-free
    # directions, per-mode steps). The shipped default of 0.175 deliberately
    # trades some of this noise margin for a 4x smaller truncation bound,
    # measured on the B3LYP formyl fluoride surface; noise error scales as
    # 1/step, so at the default the margin here roughly halves.
    a_clean, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES,
        mode_derivs=_mode_derivs(h2o_hessian, step_scale=0.35))
    a_cart, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES, hessian_fn=noisy)
    a_nm, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES,
        mode_derivs=_mode_derivs(noisy, step_scale=0.35))

    err_cart = sum(abs(a_cart[k] - a_clean[k]) for k in ("A", "B", "C"))
    err_nm = sum(abs(a_nm[k] - a_clean[k]) for k in ("A", "B", "C"))
    assert err_nm < 0.5 * err_cart, (
        f"normal-mode drift {err_nm:.1f} MHz vs cartesian {err_cart:.1f} MHz "
        f"under identical noise -- the robustness claim failed"
    )


def test_noise_diagnostic_actually_detects_noise():
    """The nonconvergence gate keys off this number; it must move.

    Measured response on this surface: 0.094 clean (truncation floor), 0.27 at
    1e-4 noise, 0.84 at 3e-4, 1.19 at 1e-3 -- monotone once noise beats
    truncation, with the 0.5 gate firing between 1e-4 and 3e-4. Noise below the
    truncation floor (2e-5) is invisible by design; the robustness test above
    shows the alpha itself barely moves there, so invisibility is correct.
    """
    clean = _mode_derivs(h2o_hessian)["noise_ratio"]
    at_3em4 = _mode_derivs(_noisy_hessian(3e-4))["noise_ratio"]
    assert at_3em4 > 0.5, f"gate would not fire at 3e-4 noise: {at_3em4:.3f}"
    assert at_3em4 > 3 * clean


def test_isotopologue_phi_matches_direct_measurement():
    """The transform under test: parent-measured field -> D2O constants.

    Ground truth is a second, independent derivative measurement along D2O's
    own modes. The decomposition + rigid completion must reproduce it without
    those extra Hessians, since not needing them is the transform's purpose.
    """
    coords = h2o_coords()
    hess = h2o_hessian(coords)

    md_parent = normal_mode_hessian_derivatives(
        h2o_hessian, coords, hess, H2O_MASSES)
    _, L_d2o = normal_modes(
        hess, D2O_MASSES, n_rigid=rigid_mode_count(coords, D2O_MASSES))
    phi_transformed = phi_semidiagonal_for_masses(
        md_parent, hess, D2O_MASSES, L_d2o)

    md_direct = normal_mode_hessian_derivatives(
        h2o_hessian, coords, hess, D2O_MASSES)
    phi_direct = phi_semidiagonal_for_masses(
        md_direct, hess, D2O_MASSES, L_d2o)

    scale = np.max(np.abs(phi_direct))
    assert scale > 0
    assert np.allclose(phi_transformed, phi_direct, atol=0.03 * scale), (
        f"max deviation {np.max(np.abs(phi_transformed - phi_direct)) / scale:.3f} "
        f"of scale"
    )


def test_d2o_alpha_through_the_transform_is_sane():
    """End to end for the isotopologue path: D2O's correction from the
    parent-measured field must carry the same signs as H2O's and be smaller,
    since deuteration halves the zero-point amplitudes."""
    coords = h2o_coords()
    hess = h2o_hessian(coords)
    md = _mode_derivs(h2o_hessian)
    a_h2o, _, _, _ = compute_harmonic_alpha(
        hess, coords, H2O_MASSES, mode_derivs=md)
    a_d2o, _, _, _ = compute_harmonic_alpha(
        hess, coords, D2O_MASSES, mode_derivs=md)
    for k in ("A", "B", "C"):
        assert np.sign(a_d2o[k]) == np.sign(a_h2o[k])
        assert abs(a_d2o[k]) < abs(a_h2o[k])
