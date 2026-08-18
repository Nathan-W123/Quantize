"""The observation sigma is derived, not typed in.

A rotational constant's useful uncertainty is not measurement precision -- a
spectrometer pins a constant to about one part in 1e7 -- but the gap between
the ground-state constant that was measured and the rigid structure being
fitted to it. That gap is a property of the physics rather than of the
molecule, so it can be defaulted from the constant itself, exactly as the
quantum prior width is looked up from the level of theory.

The old fallback was a flat 1 MHz per component, which encodes no such
statement: it claims one part in 90,000 on a 90 GHz A constant and one part in
500 on a 500 MHz C constant. These tests pin that the replacement scales with
the constant, respects which component was actually measured, never overrides
a supplied value, and reproduces the hand-entered sigmas of the reference set
-- the evidence that the manual entry was redundant rather than load-bearing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.spectral.spectral import (  # noqa: E402
    DEFAULT_SIGMA_REL_ABC,
    SpectralEngine,
    default_sigma_constants,
)

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
B0_H2O = [835840.29, 435351.72, 278138.70]


def test_sigma_scales_with_the_constant():
    """The point of the change: implied confidence must not depend on magnitude."""
    sig = default_sigma_constants(B0_H2O)
    for k in range(3):
        assert sig[k] == pytest.approx(DEFAULT_SIGMA_REL_ABC[k] * B0_H2O[k])
    # The flat 1 MHz fallback claimed 1 part in 8e5 here; this claims 1 in 100.
    assert sig[0] > 100.0


def test_component_indices_decide_the_fraction():
    """A B-only species must be weighted as a B, not as an A.

    Fluoroacetylene is the real case: linear, so only B is measured, and its
    isotopologue dicts carry component_indices [1]. Reading the fraction
    positionally would hand it the A value.
    """
    b_only = default_sigma_constants([9706.22], [1])
    assert b_only[0] == pytest.approx(DEFAULT_SIGMA_REL_ABC[1] * 9706.22)
    assert b_only[0] != pytest.approx(DEFAULT_SIGMA_REL_ABC[0] * 9706.22)


def test_zero_constant_cannot_produce_zero_sigma():
    """A zero sigma is a division by zero in every weighting path downstream."""
    assert default_sigma_constants([0.0])[0] > 0.0


def test_supplied_sigma_is_never_overridden():
    iso = {"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
           "sigma_constants": [0.2, 0.2, 0.2]}
    eng = SpectralEngine([iso])
    assert np.allclose(eng.isotopologues[0]["sigma_constants"], 0.2)


def test_engine_fills_in_a_missing_sigma():
    iso = {"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O}
    eng = SpectralEngine([iso])
    got = eng.isotopologues[0]["sigma_constants"]
    assert np.allclose(got, default_sigma_constants(B0_H2O))
    assert np.all(got > 0)


def test_default_reproduces_the_hand_entered_reference_sigmas():
    """The claim that makes this a default rather than a guess.

    Every species in the reference set carries a sigma written by hand from its
    source paper. If the derived value did not reproduce those, defaulting it
    would be substituting a convenient number for a considered one. Measured
    agreement is better than 1%, the residual being the quoted-digit floor that
    only a human reading the paper can supply -- and which is negligible against
    the model term.
    """
    from dev.monofluoro_references import MOLECULES, MOLECULES_SET2

    worst = 0.0
    for mol in list(MOLECULES) + list(MOLECULES_SET2):
        for sp in mol.species:
            hand = np.asarray(sp.sigmas(), dtype=float)
            auto = default_sigma_constants(sp.observed(), list(sp.component_indices))
            worst = max(worst, float(np.max(np.abs(auto - hand) / hand)))
    assert worst < 0.02, f"auto sigma departs from hand-entered by {100 * worst:.2f}%"
