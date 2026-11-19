"""Vinyl fluoride's deuterated A0 values carry a stated measurement sigma.

The compilation gives them as round numbers, and two of them (784, 788) imply
negative inertial defects for a planar molecule. Their sigma is sized from the
data alone -- the excess inertial defect relative to the parent's, charged to
A -- so these tests recompute it from the module's own constants, and check it
lands on the measurement side of the weight, where a vibrational correction
leaves it alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.rovib_corrections import resolve_corrections
from dev.monofluoro_references import (
    SIGMA_REL,
    VINYL_FLUORIDE,
    Isotopologue,
    _quoted_step,
)
from scripts.monofluoro_benchmark import build_isotopologues

K = 505379.0089  # MHz amu A^2
STATED = {"784 D (CHF)", "787 D cis", "788 D trans",
          "789 D trans + D (CHF)", "791 D cis + D (CHF)"}


def _defect(abc):
    a, b, c = abc
    return K / c - K / a - K / b


def _species(label):
    return next(s for s in VINYL_FLUORIDE.species if s.label == label)


class TestTheStatedValues:
    def test_they_are_the_defect_implied_ones(self):
        # sigma_A = |Delta0 - Delta0(parent)| A^2 / K, rounded to 10 MHz.
        parent = _defect(VINYL_FLUORIDE.species[0].abc_mhz)
        for label in STATED:
            sp = _species(label)
            a = sp.abc_mhz[0]
            implied = abs(_defect(sp.abc_mhz) - parent) * a * a / K
            assert sp.measured_sigma_mhz[0] == pytest.approx(implied, abs=5.0), label

    def test_the_two_bad_species_get_about_two_percent(self):
        # The point of the change, in the units of the thing itself (L-07):
        # 784 and 788 are the ones with negative defects.
        for label in ("784 D (CHF)", "788 D trans"):
            sp = _species(label)
            assert _defect(sp.abc_mhz) < 0.0
            assert 0.015 < sp.measured_sigma_mhz[0] / sp.abc_mhz[0] < 0.025, label

    def test_only_the_deuterated_a_values_carry_one(self):
        for sp in VINYL_FLUORIDE.species:
            if sp.label in STATED:
                assert sp.measured_sigma_mhz[0] is not None
                assert sp.measured_sigma_mhz[1:] == (None, None), sp.label
            else:
                assert sp.measured_sigma_mhz is None, sp.label


class TestHowItEntersTheWeight:
    def test_sigmas_combine_it_with_the_model_error(self):
        sp = _species("784 D (CHF)")
        a = sp.abc_mhz[0]
        assert sp.sigmas()[0] == pytest.approx(np.hypot(SIGMA_REL[0] * a, 910.0))
        # B and C are untouched: quoted step only.
        b = sp.abc_mhz[1]
        assert sp.sigmas()[1] == pytest.approx(
            np.hypot(SIGMA_REL[1] * b, _quoted_step(b, sp.decimals[1])))

    def test_the_quoted_step_wins_where_it_is_larger(self):
        # A stated value is a floor raised, never a precision claimed: it cannot
        # make a constant look better than its digits.
        sp = Isotopologue("x", {}, (50000.0, 1.0, 1.0), (-2, 2, 2),
                          measured_sigma_mhz=(1.0, None, None))
        assert sp.measurement_sigma(0) == pytest.approx(50.0)

    def test_it_is_not_model_error(self):
        # One species' bad constant is not shared by the others, so it must not
        # enter the part the covariance propagates as correlated.
        sp = _species("788 D trans")
        bare = Isotopologue(sp.label, sp.subs, sp.abc_mhz, sp.decimals, sp.sigma_rel)
        assert sp.sigmas_systematic() == bare.sigmas_systematic()
        assert sp.sigmas()[0] > 2.0 * bare.sigmas()[0]  # 1487 against 624

    def test_it_survives_a_vibrational_correction(self):
        # resolve_corrections drops the model-error part once a real correction
        # is applied; the measurement part is what is left. Put the stated value
        # in the systematic part instead and this falls to the correction's 10.
        isos = build_isotopologues(VINYL_FLUORIDE)
        table = {i["name"]: {c: {"delta_mhz": 50.0, "sigma_mhz": 10.0, "method": "VPT2"}
                             for c in ("A", "B", "C")} for i in isos}
        out = resolve_corrections(isos, correction_table=table, mode="hybrid_auto",
                                  elems=list(VINYL_FLUORIDE.elems))
        sig = {(t.isotopologue_label, t.component_index): t.sigma_mhz for t in out}
        assert sig[("788 D trans", 0)] == pytest.approx(np.hypot(1350.0, 10.0), rel=1e-9)
        assert sig[("784 D (CHF)", 0)] == pytest.approx(np.hypot(910.0, 10.0), rel=1e-9)
        # The parent's A keeps only its quoted step under the correction's sigma.
        assert sig[("781 parent", 0)] == pytest.approx(np.hypot(0.05, 10.0), rel=1e-9)
