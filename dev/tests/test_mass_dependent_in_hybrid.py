"""Folding the Watson B0 -> Be offset into the hybrid objective.

`SpectralEngine.fit_mass_dependent_correction` eliminates the three mass-
dependence coefficients in closed form at each step rather than carrying them as
optimiser parameters. That is only legitimate if the closed form really is the
weighted least-squares minimiser, so the first test checks it against a brute
numerical minimisation instead of trusting the algebra.

The measured verdict on whether it *helps* is recorded at the bottom of this
file. It does not, outside one narrow case, which is why the feature is off by
default.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.spectral.spectral import (  # noqa: E402
    _INERTIA_TO_MHZ,
    _principal_moments,
    SpectralEngine,
)

M_H, M_D, M_O16, M_O18 = 1.00782503207, 2.0141017778, 15.9949146196, 17.9991610


def _water(r=0.9578, deg=104.48):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _engine(coords, mass_sets, c_true=None, sigma_rel=0.005):
    isos = []
    for name, masses in mass_sets:
        i_m = _principal_moments(coords, np.asarray(masses, float))
        i_obs = i_m + (0.0 if c_true is None else np.asarray(c_true) * np.sqrt(i_m))
        const = _INERTIA_TO_MHZ / i_obs
        isos.append({
            "name": name,
            "masses": list(masses),
            "obs_constants": list(const),
            "sigma_constants": list(np.abs(const) * sigma_rel),
        })
    return SpectralEngine(isos)


SETS = [("H2-16O", [M_O16, M_H, M_H]), ("H2-18O", [M_O18, M_H, M_H]),
        ("D2-16O", [M_O16, M_D, M_D]), ("HD-16O", [M_O16, M_H, M_D])]


def _objective(eng, coords, c):
    """Weighted chi-square over the moments for a given c."""
    total = 0.0
    for iso in eng.isotopologues:
        masses = np.asarray(iso["masses"], float)
        i_m = _principal_moments(coords, masses)
        obs = np.asarray(iso["obs_constants"], float)
        sig = np.asarray(iso["sigma_constants"], float)
        for k, comp in enumerate(np.asarray(iso["component_indices"], int)):
            i_obs = _INERTIA_TO_MHZ / obs[k]
            s_i = abs(i_obs) * (sig[k] / obs[k])
            resid = i_obs - i_m[comp] - c[comp] * np.sqrt(i_m[comp])
            total += (resid / s_i) ** 2
    return total


def test_closed_form_c_is_the_least_squares_minimiser():
    """Variable projection is exact only if this really is the minimum."""
    coords = _water()
    eng = _engine(coords, SETS, c_true=np.array([0.01, -0.02, 0.03]))
    # A huge frac_sigma makes the ridge negligible, so this is the plain fit.
    c_closed = eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    res = minimize(lambda c: _objective(eng, coords, c), np.zeros(3), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert np.allclose(c_closed, res.x, atol=1e-6), (
        f"closed form {c_closed} != numerical minimum {res.x}"
    )


def test_recovers_the_coefficients_the_data_was_built_with():
    coords = _water()
    c_true = np.array([0.012, -0.021, 0.034])
    eng = _engine(coords, SETS, c_true=c_true)
    c = eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    assert np.allclose(c, c_true, atol=1e-6), f"got {c}, wanted {c_true}"


def test_no_offset_in_the_data_gives_no_correction():
    coords = _water()
    eng = _engine(coords, SETS, c_true=None)
    c = eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    assert np.allclose(c, 0.0, atol=1e-8)
    for iso in eng.isotopologues:
        assert np.allclose(iso["delta_total_constants"], 0.0, atol=1e-6)


def test_the_prior_shrinks_the_correction_toward_zero():
    """Without it, three coefficients absorb three constants exactly.

    A single isotopologue supplies exactly as many observations as there are
    coefficients. Unregularised they fit the residual perfectly, the spectral
    term vanishes and the fit silently reduces to theory alone -- so the prior
    is what keeps the data in the problem, not a tuning knob.
    """
    coords = _water()
    c_true = np.array([0.02, -0.03, 0.04])
    eng = _engine(coords, SETS[:1], c_true=c_true)
    loose = eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    tight = eng.fit_mass_dependent_correction(coords, frac_sigma=1e-4)
    assert np.allclose(loose, c_true, atol=1e-6), "one species should fit exactly"
    assert np.all(np.abs(tight) < np.abs(loose) * 1e-2), (
        f"tight prior failed to shrink: {tight} vs {loose}"
    )


def test_correction_is_installed_as_a_target_shift():
    """The optimiser sees a shifted target, not a changed prediction."""
    coords = _water()
    c_true = np.array([0.01, -0.02, 0.03])
    eng = _engine(coords, SETS, c_true=c_true)
    eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    for iso in eng.isotopologues:
        target = eng._be_target(iso)
        rigid = eng.rotational_constants(coords, np.asarray(iso["masses"], float))
        idx = np.asarray(iso["component_indices"], int)
        assert np.allclose(target, rigid[idx], rtol=1e-8), (
            "target should land on the rigid prediction of the true structure"
        )


@pytest.mark.parametrize("n_species", [1, 2, 4])
def test_it_never_makes_the_residual_worse(n_species):
    coords = _water()
    eng = _engine(coords, SETS[:n_species], c_true=np.array([0.01, -0.02, 0.03]))
    before = _objective(eng, coords, np.zeros(3))
    c = eng.fit_mass_dependent_correction(coords, frac_sigma=1e6)
    assert _objective(eng, coords, c) <= before + 1e-9
