"""Tests for backend/covariance_checks.py.

The point of the module is that a negative variance is a failed computation,
not a zero. So the tests are about the three regimes it has to tell apart:
a healthy covariance, one negative by round-off, and one that has collapsed.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.covariance_checks import (
    NEGATIVE_VARIANCE_REL_TOL,
    std_err_from_covariance,
)


def _warned():
    """Collect warnings instead of printing them."""
    seen = []
    return seen, seen.append


class TestHealthyCovariance:
    def test_returns_the_square_roots(self):
        s, info = std_err_from_covariance(np.diag([4.0, 9.0, 16.0]))
        np.testing.assert_allclose(s, [2.0, 3.0, 4.0])
        assert info["ok"] is True
        assert info["n_negative"] == 0

    def test_accepts_a_bare_diagonal(self):
        s, _ = std_err_from_covariance(np.array([4.0, 9.0]))
        np.testing.assert_allclose(s, [2.0, 3.0])

    def test_ignores_off_diagonal_elements(self):
        cov = np.array([[4.0, 100.0], [100.0, 9.0]])
        s, info = std_err_from_covariance(cov)
        np.testing.assert_allclose(s, [2.0, 3.0])
        assert info["ok"] is True

    def test_says_nothing_when_all_is_well(self):
        seen, warn = _warned()
        std_err_from_covariance(np.diag([1.0, 2.0]), warn=warn)
        assert seen == []

    def test_a_zero_variance_is_not_a_failure(self):
        # A perfectly determined parameter is unusual but not impossible, and
        # exactly zero is not negative.
        s, info = std_err_from_covariance(np.diag([4.0, 0.0]))
        assert info["ok"] is True
        assert s[1] == 0.0


class TestRoundOffIsTolerated:
    def test_a_tiny_negative_is_clamped_quietly(self):
        seen, warn = _warned()
        s, info = std_err_from_covariance(np.diag([4.0, -1e-18]), warn=warn)
        assert info["ok"] is True
        assert s[1] == 0.0
        assert seen == []

    def test_the_tolerance_is_relative_not_absolute(self):
        # The same absolute value is round-off on a large matrix and a failure
        # on a small one. 1e-9 against a largest variance of 1.0 is a failure;
        # against 1e6 it is round-off.
        small = std_err_from_covariance(np.array([1.0, -1e-9 * 10]))[1]
        large = std_err_from_covariance(np.array([1.0e6, -1e-9 * 10]))[1]
        assert small["ok"] is False
        assert large["ok"] is True

    def test_the_threshold_sits_between_round_off_and_collapse(self):
        assert 1e-16 < NEGATIVE_VARIANCE_REL_TOL < 1e-3


class TestCollapseIsReported:
    def test_a_real_negative_becomes_nan_not_zero(self):
        # The whole point. Zero would read as perfect precision.
        seen, warn = _warned()
        s, info = std_err_from_covariance(np.diag([4.0, -2.0]), warn=warn)
        assert np.isnan(s[1])
        assert s[1] != 0.0
        assert info["n_negative"] == 1
        assert info["ok"] is False
        assert len(seen) == 1

    def test_good_parameters_survive_a_bad_one(self):
        # Per element rather than wholesale: one broken parameter should not
        # discard the rest of the report.
        s, info = std_err_from_covariance(np.diag([4.0, -2.0, 9.0]))
        assert s[0] == 2.0 and s[2] == 3.0
        assert np.isnan(s[1])
        assert info["clamped_indices"] == [1]

    def test_the_message_says_nan_and_why(self):
        seen, warn = _warned()
        std_err_from_covariance(np.diag([4.0, -2.0]), warn=warn)
        msg = seen[0].lower()
        assert "not positive semi-definite" in msg
        assert "nan" in msg
        assert "perfectly determined" in msg      # names the wrong reading

    def test_labels_appear_in_the_message_when_given(self):
        seen, warn = _warned()
        std_err_from_covariance(np.diag([4.0, -2.0]),
                                labels=["r(O-H)", "H-O-H"], warn=warn)
        assert "H-O-H" in seen[0]

    def test_bad_labels_do_not_break_the_report(self):
        # A label problem must not turn a diagnostic into an exception.
        seen, warn = _warned()
        s, info = std_err_from_covariance(np.diag([4.0, -2.0]),
                                          labels=["only-one"], warn=warn)
        assert np.isnan(s[1]) and info["n_negative"] == 1
        assert len(seen) == 1

    def test_context_is_included(self):
        seen, warn = _warned()
        std_err_from_covariance(np.array([-1.0]), context="my solver", warn=warn)
        assert "my solver" in seen[0]

    def test_worst_rel_reports_how_bad(self):
        _, info = std_err_from_covariance(np.diag([4.0, -2.0]))
        assert info["worst_rel"] == pytest.approx(-0.5)


class TestEdgeCases:
    def test_an_empty_covariance_is_handled(self):
        s, info = std_err_from_covariance(np.zeros((0, 0)))
        assert s.size == 0
        assert info["ok"] is True

    def test_a_single_parameter_works(self):
        s, _ = std_err_from_covariance(np.array([[9.0]]))
        np.testing.assert_allclose(s, [3.0])

    def test_all_negative_gives_all_nan(self):
        s, info = std_err_from_covariance(np.diag([-1.0, -2.0]))
        assert np.all(np.isnan(s))
        assert info["n_negative"] == 2
