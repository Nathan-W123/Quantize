"""Tests for the ORCA VPT2 rovib parser."""

from __future__ import annotations

import numpy as np
import pytest

from backend.quantum import parse_orca_rovib, parse_orca_rovib_alpha


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_labeled_alpha_detection(tmp_path):
    body = """
    ORCA VPT2 second order analysis follows.
    Vibrational-rotational coupling constants:
        alpha(A) = 100.5
        alpha(B) = 50.2
        alpha(C) = 25.1
    """
    p = _write(tmp_path, "labeled.out", body)
    result = parse_orca_rovib(p)
    assert result.parse_status == "ok"
    assert np.allclose(result.alpha_abc, [100.5, 50.2, 25.1])


def test_triplet_alpha_detection(tmp_path):
    body = """
    Running VPT2 analysis...
    alpha    1.00   2.00   3.00
    """
    p = _write(tmp_path, "triplet.out", body)
    result = parse_orca_rovib(p)
    assert result.parse_status == "ok"
    assert np.allclose(result.alpha_abc, [1.0, 2.0, 3.0])


def test_missing_alpha_means_parse_failed(tmp_path):
    body = "ORCA output of unrelated kind.\nSome random data.\n"
    p = _write(tmp_path, "fail.out", body)
    result = parse_orca_rovib(p)
    assert result.parse_status == "parse_failed"
    assert not np.isfinite(result.alpha_abc).any()
    assert any("no VPT2" in w for w in result.warnings)


def test_partial_alpha_status(tmp_path):
    body = """
    VPT2 second order analysis
    alpha(A) = 11.0
    alpha(B) = 22.0
    """
    p = _write(tmp_path, "partial.out", body)
    result = parse_orca_rovib(p)
    assert result.parse_status == "partial"
    assert np.isfinite(result.alpha_abc[0])
    assert np.isfinite(result.alpha_abc[1])
    assert not np.isfinite(result.alpha_abc[2])


def test_imaginary_frequency_warning(tmp_path):
    body = """
    Running VPT2 second-order analysis.
    Vibrational frequencies:
       1:    -45.0 cm**-1
       2:    100.0 cm**-1

    alpha(A) = 1.0
    alpha(B) = 2.0
    alpha(C) = 3.0
    """
    p = _write(tmp_path, "imag.out", body)
    result = parse_orca_rovib(p)
    assert any("imaginary" in w for w in result.warnings)
    assert result.parse_status == "ok"


def test_low_frequency_mode_warning(tmp_path):
    body = """
    VPT2 second-order analysis.
    Vibrational frequencies:
       1:     20.0 cm**-1
       2:    400.0 cm**-1

    alpha(A) = 1.0
    alpha(B) = 2.0
    alpha(C) = 3.0
    """
    p = _write(tmp_path, "lowf.out", body)
    result = parse_orca_rovib(p)
    assert any("low-frequency" in w or "below 50" in w for w in result.warnings)


def test_backward_compat_alpha_wrapper(tmp_path):
    body = """
    VPT2 second-order analysis.
    alpha(A) = 7.7
    alpha(B) = 8.8
    alpha(C) = 9.9
    """
    p = _write(tmp_path, "compat.out", body)
    arr = parse_orca_rovib_alpha(p)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [7.7, 8.8, 9.9])
