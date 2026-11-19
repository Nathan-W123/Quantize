"""Vinyl fluoride's rotational constants: Hayashi & Inagusa (1989).

Until 2026-11-19 the module carried NBS Monograph 70's 1968 values, whose
deuterated A were round numbers off by 118-1350 MHz -- 784 and 788 gave
negative inertial defects for a planar molecule. These tests pin the
replacement, check it is internally consistent in a way the old set was not,
keep the measurement-sigma field's semantics, and keep the record of how well
the inertial-defect estimate (D-31) sized the old errors.
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

#: NBS Monograph 70 Vol. IV entry 780 (1968), the replaced deuterated A values,
#: and the stated sigmas D-31 derived for them from the inertial defects.
OLD_A = {"784 D (CHF)": 48960.0, "787 D cis": 53400.0, "788 D trans": 62440.0,
         "789 D trans + D (CHF)": 49250.0, "791 D cis + D (CHF)": 42700.0}
D31_SIGMA = {"784 D (CHF)": 910.0, "787 D cis": 160.0, "788 D trans": 1350.0,
             "789 D trans + D (CHF)": 280.0, "791 D cis + D (CHF)": 70.0}


def _defect(abc):
    a, b, c = abc
    return K / c - K / a - K / b


def _species(label):
    return next(s for s in VINYL_FLUORIDE.species if s.label == label)


class TestTheConstants:
    def test_they_are_hayashi_and_inagusa_s(self):
        # Spot values transcribed from Piccardo et al. 2015's SI; a digit slip
        # here would also show in the r_e consistency check (0.06%).
        assert _species("781 parent").abc_mhz == (64584.672, 10636.809, 9118.108)
        assert _species("784 D (CHF)").abc_mhz[0] == 49930.986
        assert _species("788 D trans").abc_mhz[0] == 63789.713
        assert "Hayashi" in VINYL_FLUORIDE.constants_source

    def test_every_species_now_has_a_b_and_c(self):
        for sp in VINYL_FLUORIDE.species:
            assert sp.component_indices == [0, 1, 2], sp.label
        assert VINYL_FLUORIDE.n_observables == 24

    def test_every_inertial_defect_is_normal(self):
        # Planar: small and positive, varying by ~0.02 between isotopologues
        # as in this module's other planar molecules. The old set had -0.104
        # and -0.088 here.
        d = [_defect(sp.abc_mhz) for sp in VINYL_FLUORIDE.species]
        assert min(d) > 0.08 and max(d) < 0.11, d

    def test_each_carries_its_stated_uncertainty(self):
        sp = _species("787 D cis")
        assert sp.measured_sigma_mhz == (0.227, 0.039, 0.025)
        for s in VINYL_FLUORIDE.species:
            assert all(0.0 < u < 0.25 for u in s.measured_sigma_mhz), s.label


class TestTheReplacedValues:
    def test_the_old_a_values_were_off_by_what_d31_estimated(self):
        # The inertial-defect sigma (D-31) sized each old error at 60-100% of
        # what the 1989 measurement now shows -- recorded because that method
        # needs no reference and no theory, so it is worth knowing it works.
        for label, old in OLD_A.items():
            err = abs(_species(label).abc_mhz[0] - old)
            assert 0.55 * err <= D31_SIGMA[label] <= 1.05 * err, (label, err)


class TestTheMeasurementSigmaField:
    def test_sigmas_combine_it_with_the_model_error(self):
        sp = _species("784 D (CHF)")
        a = sp.abc_mhz[0]
        assert sp.sigmas()[0] == pytest.approx(np.hypot(SIGMA_REL[0] * a, 0.091))

    def test_the_quoted_step_wins_where_it_is_larger(self):
        # A stated value is a floor raised, never a precision claimed.
        sp = Isotopologue("x", {}, (50000.0, 1.0, 1.0), (-2, 2, 2),
                          measured_sigma_mhz=(1.0, None, None))
        assert sp.measurement_sigma(0) == pytest.approx(50.0)
        assert sp.measurement_sigma(1) == pytest.approx(_quoted_step(1.0, 2))

    def test_it_is_not_model_error(self):
        # One species' measurement error is not shared by the others, so it
        # must stay out of the part propagated as correlated.
        sp = Isotopologue("x", {}, (50000.0, 10000.0, 8000.0), (3, 3, 3),
                          measured_sigma_mhz=(900.0, None, None))
        bare = Isotopologue("x", {}, (50000.0, 10000.0, 8000.0), (3, 3, 3))
        assert sp.sigmas_systematic() == bare.sigmas_systematic()
        assert sp.sigmas()[0] > 1.5 * bare.sigmas()[0]

    def test_it_survives_a_vibrational_correction(self):
        # resolve_corrections drops the model-error part once a real correction
        # is applied; the measurement part is what is left.
        isos = build_isotopologues(VINYL_FLUORIDE)
        table = {i["name"]: {c: {"delta_mhz": 50.0, "sigma_mhz": 10.0, "method": "VPT2"}
                             for c in ("A", "B", "C")} for i in isos}
        out = resolve_corrections(isos, correction_table=table, mode="hybrid_auto",
                                  elems=list(VINYL_FLUORIDE.elems))
        sig = {(t.isotopologue_label, t.component_index): t.sigma_mhz for t in out}
        assert sig[("787 D cis", 0)] == pytest.approx(np.hypot(0.227, 10.0), rel=1e-9)
        assert sig[("781 parent", 2)] == pytest.approx(np.hypot(0.006, 10.0), rel=1e-9)
