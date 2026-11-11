"""Coverage must be read as bonds-and-angles, never as one total.

Measured on 42 coordinates across the held-out molecules, both levels:

                          B3LYP/6-31g(d)       HF/6-31g
                          sigma_x = 0.007      sigma_x = 0.020
  coverage                38/42 = 90%          27/42 = 64%
  median |error|/sigma     0.39                 0.34
    bonds                 18/20, median 0.72    8/20, median 1.40
    angles                20/22, median 0.24   19/22, median 0.07

Ideal is 68% coverage and a median |error|/sigma of 0.674 (half-normal).

HF's 64% looks well calibrated and is a coincidence: it averages bonds whose
sigma is about 2x too small (median 1.40, covering 8 of 20) against angles
roughly 10x too wide (median 0.07). HF/6-31g is the benchmark's own default
level, so that is where it matters -- and there `geometry_uncertainty`'s
"expect it to be optimistic" warning is correct for bonds, which an earlier
version of the calibration docstring had concluded it was not.

These tests pin the numbers and the lookup that selects the prior width. They
do not re-run the calibration: that needs real Hessians for several molecules
at two levels and does not belong in a unit suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

_SCRIPT = _ROOT / "scripts" / "uncertainty_calibration.py"

#: Ideal median |error|/sigma for a correctly sized 1-sigma interval.
IDEAL_MEDIAN_RATIO = 0.674

MEASURED = {
    # level:            (coverage, n, bonds_cov, bonds_med, ang_cov, ang_med)
    "b3lyp/6-31g(d)":   (38, 42, 18, 0.72, 20, 0.24),
    "hf/6-31g":         (27, 42,  8, 1.40, 19, 0.07),
}


def _theory_sigma_x():
    src = _SCRIPT.read_text()
    m = re.search(r"THEORY_SIGMA_X = \{.*?\}", src, re.S)
    assert m, "THEORY_SIGMA_X table not found"
    ns: dict = {}
    exec(m.group(0), ns)
    return ns["THEORY_SIGMA_X"]


def test_the_prior_width_lookup_matches_the_codebase_spelling():
    """The table said "rhf"; everything else in the repo says "hf"."""
    table = _theory_sigma_x()
    assert ("hf", "6-31g") in table, (
        "THEORY_SIGMA_X cannot be looked up with the method string the rest "
        "of the codebase uses, so every HF run falls through to the default")
    assert table[("hf", "6-31g")] == pytest.approx(0.020)


def test_both_spellings_agree():
    """So an 'rhf=' on the command line resolves to the same width."""
    table = _theory_sigma_x()
    assert table[("hf", "6-31g")] == table[("rhf", "6-31g")]
    assert table[("hf", "6-31g(d)")] == table[("rhf", "6-31g(d)")]


def test_the_codebase_really_does_spell_it_hf():
    """Guard the premise of the fix above."""
    from dev.pyscf_backend import PySCFBackend
    import inspect
    assert inspect.signature(PySCFBackend.__init__).parameters["method"].default == "hf"


def test_bonds_are_overconfident_at_the_benchmarks_default_level():
    """The finding that matters: 8/20, and sigma about 2x too small."""
    _cov, _n, b_cov, b_med, _a, _am = MEASURED["hf/6-31g"]
    assert b_cov / 20 < 0.5, "bonds were not undercovered after all"
    assert b_med / IDEAL_MEDIAN_RATIO > 1.8, (
        "bond sigma is no longer about 2x too small at HF/6-31g")


def test_bonds_are_well_sized_at_b3lyp():
    """The contrast that makes it a level problem rather than a bond problem."""
    _cov, _n, _bc, b_med, _a, _am = MEASURED["b3lyp/6-31g(d)"]
    assert 0.8 < b_med / IDEAL_MEDIAN_RATIO < 1.3


def test_angles_are_too_wide_at_both_levels():
    for level in MEASURED:
        _c, _n, _bc, _bm, _ac, a_med = MEASURED[level]
        assert a_med < IDEAL_MEDIAN_RATIO / 1.5, level


def test_the_aggregate_hides_the_split():
    """HF's 64% sits next to the 68% ideal while both halves are wrong.

    This is the whole reason the docstring now says to read the split.
    """
    cov, n, b_cov, b_med, a_cov, a_med = MEASURED["hf/6-31g"]
    assert abs(cov / n - 0.68) < 0.05, "aggregate no longer looks calibrated"
    assert b_cov / 20 < 0.5 and a_cov / 22 > 0.8, (
        "the two halves no longer fail in opposite directions")


def test_the_docstring_records_both_levels_and_the_caveat():
    src = _SCRIPT.read_text()
    head = src[:src.index('"""', src.index('"""') + 3)]
    assert "B3LYP/6-31g(d)" in head and "HF/6-31g" in head
    assert "COINCIDENCE" in head.upper()
    # The old claim may still be QUOTED, but it must be marked as corrected.
    # Asserting the phrase is absent was wrong: the correction quotes it.
    assert "level mix-up" in head
    assert "the warning holds for bonds" in head
