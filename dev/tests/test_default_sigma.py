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


# ── Defect-derived sigma ─────────────────────────────────────────────────────

def _fcho():
    from dev.monofluoro_references import MOLECULES_SET2
    mol = next(m for m in MOLECULES_SET2 if m.key == "formyl_fluoride")
    return mol, mol.species[0]


def test_defect_sigma_subtracts_the_structural_part():
    """Non-planarity is not vibration, and confusing them is the whole trap.

    Measured defects across the reference set run +0.09 to -6.35 amu.A^2, which
    reads as a 70x spread in non-rigidity. Almost all of it is structural: the
    molecules with large defects are simply not planar. Subtracting the
    structure's own defect collapses the range to 0.022-0.092, and that residue
    is what a vibrational sigma may be built on. A regression that dropped the
    subtraction would inflate fluoroethane's sigma by roughly 300x.
    """
    from backend.spectral.spectral import defect_model_sigma
    from dev.monofluoro_references import MOLECULES

    mol = next(m for m in MOLECULES if m.key == "fluoroethane")
    sp = mol.species[0]
    sig = defect_model_sigma(sp.observed(), mol.geometry,
                             sp.masses(mol.masses), list(sp.component_indices))
    assert sig is not None
    # B and C sigmas must stay well under 1% of their constants; without the
    # structural subtraction the -6.3 defect would put them far above it.
    obs = np.asarray(sp.observed(), dtype=float)
    assert sig[1] < 0.01 * obs[1] and sig[2] < 0.01 * obs[2]


def test_defect_sigma_is_tighter_than_the_flat_guess_on_b_and_c():
    """The measured claim: the flat 0.5% is about 2x too pessimistic.

    Cross-checked independently by the VPT2 route, which puts fluoroacetylene's
    real B0-Be gap at 0.23% against the same 0.5% guess.
    """
    from backend.spectral.spectral import defect_model_sigma
    mol, sp = _fcho()
    d = defect_model_sigma(sp.observed(), mol.geometry,
                           sp.masses(mol.masses), list(sp.component_indices))
    f = default_sigma_constants(sp.observed(), list(sp.component_indices))
    assert d[1] < 0.6 * f[1], f"B: defect {d[1]:.1f} vs flat {f[1]:.1f}"
    assert d[2] < 0.6 * f[2]


def test_defect_sigma_declines_when_it_cannot_be_formed():
    """A linear species has no defect; the caller must fall back, not guess."""
    from backend.spectral.spectral import defect_model_sigma
    from dev.monofluoro_references import MOLECULES_SET2
    mol = next(m for m in MOLECULES_SET2 if m.key == "fluoroacetylene")
    sp = mol.species[0]
    assert defect_model_sigma(sp.observed(), mol.geometry,
                              sp.masses(mol.masses),
                              list(sp.component_indices)) is None


def test_correction_does_not_double_charge_the_gap_it_removed():
    """U1. Applying a vibrational correction removes the r_0-vs-r_e gap, so the
    weighting sigma must stop carrying it -- measured 58.8 -> 11.7 MHz on
    formyl fluoride's B. Without this the corrected data is weighted 25x too
    weakly (weight goes as 1/sigma^2) and the correction is paid for twice."""
    from backend.spectral.rovib_corrections import resolve_corrections

    mol, sp = _fcho()
    obs = list(sp.observed())
    sig = list(sp.sigmas())
    iso = {"name": "p", "masses": sp.masses(mol.masses).tolist(),
           "obs_constants": obs, "sigma_constants": sig,
           "sigma_systematic_constants": list(sp.sigmas_systematic()),
           "component_indices": list(sp.component_indices)}
    ctbl = {"p": {c: {"alpha_sum_mhz": 20.0, "sigma_mhz": 2.0,
                      "method": "VPT2_semidiag", "source": "harmonic_hessian"}
                  for c in ("A", "B", "C")}}
    got = resolve_corrections([iso], correction_table=ctbl, mode="hybrid_auto")
    for t in got:
        assert t.sigma_mhz < t.sigma_exp_mhz, (
            f"{t.component}: sigma {t.sigma_mhz:.2f} did not drop below the "
            f"uncorrected {t.sigma_exp_mhz:.2f}"
        )


def test_uncorrected_sigma_is_untouched():
    """The subtraction must fire only when a correction actually applied."""
    from backend.spectral.rovib_corrections import resolve_corrections
    mol, sp = _fcho()
    iso = {"name": "p", "masses": sp.masses(mol.masses).tolist(),
           "obs_constants": list(sp.observed()),
           "sigma_constants": list(sp.sigmas()),
           "sigma_systematic_constants": list(sp.sigmas_systematic()),
           "component_indices": list(sp.component_indices)}
    got = resolve_corrections([iso], correction_table=None, mode="hybrid_auto")
    for t in got:
        assert t.sigma_mhz == pytest.approx(t.sigma_exp_mhz)
