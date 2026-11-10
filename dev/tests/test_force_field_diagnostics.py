"""Tests for scripts/force_field_diagnostics.py.

The diagnostics exist because the engine once used a force field whose ozone
nu3 was 101% wrong and reported a confident number. So the thing to test is
discrimination: they must flag ozone and must NOT flag water, with thresholds
that sit where the measurements put them rather than where they happen to
pass.

Most tests here need no quantum chemistry. Only two run pyscf, on three- and
four-atom molecules at cc-pVDZ, which is a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.force_field_diagnostics import (
    FREQUENCY_DISAGREEMENT_WARN,
    MULTIREFERENCE_T2_WARN,
    frequency_disagreement,
    multireference_diagnostic,
)

_ROOT = Path(__file__).resolve().parents[2]

#: The measured force fields that motivated the module (cm^-1).
_OZONE_HF = [730.2, 850.7, 1266.9]
_OZONE_MP2 = [748.4, 1170.7, 2212.1]
_OZONE_EXP = [705.0, 1042.0, 1103.0]
_WATER_HF = [1737.0, 3988.5, 4145.4]
_WATER_MP2 = [1650.3, 3871.2, 3992.4]


class TestFrequencyDisagreement:
    def test_flags_the_ozone_pair(self):
        d = frequency_disagreement(_OZONE_HF, _OZONE_MP2, "HF", "MP2")
        assert d["flagged"] is True
        assert d["max_rel"] == pytest.approx(0.746, abs=0.01)
        assert d["worst_index"] == 2                 # the antisymmetric stretch
        assert "1267" in d["reason"] and "2212" in d["reason"]

    def test_does_not_flag_the_water_pair(self):
        d = frequency_disagreement(_WATER_HF, _WATER_MP2, "HF", "MP2")
        assert d["flagged"] is False
        assert d["max_rel"] == pytest.approx(0.053, abs=0.01)

    def test_the_threshold_sits_between_the_two_measured_cases(self):
        # Justified by data rather than taste: anything between 6% and 74%
        # separates them, and 25% is in the middle of that range.
        water = frequency_disagreement(_WATER_HF, _WATER_MP2)["max_rel"]
        ozone = frequency_disagreement(_OZONE_HF, _OZONE_MP2)["max_rel"]
        assert water < FREQUENCY_DISAGREEMENT_WARN < ozone

    def test_is_symmetric_in_its_arguments(self):
        # Relative to the smaller of each pair, so swapping the levels cannot
        # change the verdict. Dividing by one side would make a doubling read
        # as 100% one way and 50% the other.
        a = frequency_disagreement(_OZONE_HF, _OZONE_MP2)["max_rel"]
        b = frequency_disagreement(_OZONE_MP2, _OZONE_HF)["max_rel"]
        assert a == pytest.approx(b, rel=1e-12)

    def test_a_doubling_reads_as_one_hundred_percent(self):
        d = frequency_disagreement([1000.0], [2000.0])
        assert d["max_rel"] == pytest.approx(1.0, rel=1e-12)

    def test_identical_force_fields_disagree_by_nothing(self):
        d = frequency_disagreement(_WATER_HF, _WATER_HF)
        assert d["max_rel"] == pytest.approx(0.0, abs=1e-15)
        assert d["flagged"] is False

    def test_mismatched_mode_counts_refuse_rather_than_guess(self):
        d = frequency_disagreement([100.0, 200.0], [100.0, 200.0, 300.0])
        assert d["max_rel"] is None
        assert d["flagged"] is False
        assert "not describing the same molecule" in d["reason"]

    def test_empty_input_refuses(self):
        assert frequency_disagreement([], [])["max_rel"] is None

    def test_modes_are_matched_by_order_not_position(self):
        # Sorted before comparison, so a level that happens to emit its modes
        # in another order is not reported as disagreeing.
        d = frequency_disagreement(_WATER_HF, list(reversed(_WATER_HF)))
        assert d["max_rel"] == pytest.approx(0.0, abs=1e-15)

    def test_a_worse_level_pair_is_reported_as_worse(self):
        mild = frequency_disagreement([1000.0], [1100.0])["max_rel"]
        bad = frequency_disagreement([1000.0], [1900.0])["max_rel"]
        assert bad > mild


class TestMultireferenceDiagnostic:
    """Two pyscf runs, three and four atoms at cc-pVDZ."""

    @staticmethod
    def _geom(key):
        cache = json.loads(
            (_ROOT / "output" / "theory_geometries_hf_6-31g.json").read_text())
        return np.asarray(cache[key]["coords"], dtype=float)

    def test_flags_ozone(self):
        if not (_ROOT / "output" / "theory_geometries_hf_6-31g.json").exists():
            pytest.skip("needs the cached theory geometries")
        d = multireference_diagnostic(["O", "O", "O"], self._geom("ozone"))
        assert d["max_t2"] == pytest.approx(0.1794, abs=0.01)
        assert d["flagged"] is True
        assert "multireference" in d["reason"]

    def test_does_not_flag_water(self):
        if not (_ROOT / "output" / "theory_geometries_hf_6-31g.json").exists():
            pytest.skip("needs the cached theory geometries")
        d = multireference_diagnostic(["O", "H", "H"], self._geom("water"))
        assert d["max_t2"] == pytest.approx(0.0471, abs=0.01)
        assert d["flagged"] is False

    def test_the_threshold_sits_in_the_measured_gap(self):
        # The largest well-behaved value in the reference set is vinyl
        # fluoride's 0.0969 at 6-31g; the smallest ozone value is 0.1697 at
        # cc-pVTZ. The threshold has to lie between them, and does.
        assert 0.0969 < MULTIREFERENCE_T2_WARN < 0.1697

    def test_a_failure_returns_a_reason_instead_of_raising(self):
        # A diagnostic that raises takes down the run it was meant to qualify.
        d = multireference_diagnostic(["O", "H"], np.array([[0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.97]]))
        # OH is an open-shell radical; RHF with spin=0 cannot describe it, so
        # this either fails or converges to something -- either way, no raise.
        assert d["flagged"] in (True, False)
        assert isinstance(d["reason"], str)

    def test_reports_the_basis_it_used(self):
        d = multireference_diagnostic(["H", "H"],
                                      np.array([[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.74]]))
        assert d["basis"] == "cc-pvdz"
        assert d["max_t2"] is not None and d["max_t2"] < MULTIREFERENCE_T2_WARN


class TestWhatTheseDoNotClaim:
    def test_a_flag_does_not_say_which_level_is_right(self):
        # For ozone the CHEAPER level scores better, but its own nu3 is 15%
        # high and its alpha errors partly cancel. The diagnostic reports
        # distrust, never a winner, and the reason text must not imply one.
        d = frequency_disagreement(_OZONE_HF, _OZONE_MP2, "HF", "MP2")
        text = d["reason"].lower()
        # Checked as phrases, not substrings: an earlier version of this test
        # looked for "correct" and tripped over the word "correction".
        for claim in ("is better", "more accurate", "should be used",
                      "prefer", "is right", "is correct"):
            assert claim not in text, f"reason implies a winner: {claim!r}"
        assert "neither" in text        # it distrusts both, which is the point

    def test_both_levels_can_be_wrong_without_disagreeing(self):
        # The limitation worth stating: two levels that share a defect agree
        # with each other and the cross-level check says nothing. Ozone's HF
        # nu3 is 15% above experiment, which no amount of agreement reveals.
        hf_err = abs(_OZONE_HF[2] - _OZONE_EXP[2]) / _OZONE_EXP[2]
        assert hf_err > 0.14
        d = frequency_disagreement(_OZONE_HF, _OZONE_HF)
        assert d["flagged"] is False
