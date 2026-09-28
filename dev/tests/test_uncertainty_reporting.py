"""The uncertainty has to leave the terminal.

geometry_uncertainty's own note says a structure quoted without an uncertainty
cannot be used. The runner computed one, printed it, and dropped it: the report
and the exports carried the structure and not its uncertainty, so nothing a
reader of a run directory could find told them how well determined it was.

These pin that it reaches a file, that it says what kind of number it is, and --
the part most likely to rot -- that its ABSENCE is stated rather than silently
omitted when the run cannot produce one.
"""

from __future__ import annotations

import csv

import pytest

from runner.usability import (
    generate_uncertainty_report_section,
    write_uncertainty_csv,
)

_ROWS = [
    {"name": "O1-H2", "value": 0.9578, "value_unit": "Ang", "std_err": 0.00021,
     "std_err_unit": "Ang", "ci_lo": 0.95739, "ci_hi": 0.95821, "ci_unit": "Ang",
     "chi2_scale": 1.0, "prior_dominance": "data", "prior_sensitivity": "low"},
    {"name": "H2-O1-H3", "value": 104.48, "value_unit": "deg", "std_err": 0.013,
     "std_err_unit": "deg", "ci_lo": 104.4545, "ci_hi": 104.5055, "ci_unit": "deg",
     "chi2_scale": 1.0, "prior_dominance": "prior", "prior_sensitivity": "high"},
]


def test_absence_is_reported_rather_than_omitted():
    """A missing uncertainty must be visible, not a gap in the report.

    Cartesian runs cannot produce one, and the benchmark harness runs in
    cartesian mode -- so silently dropping the section is exactly the case that
    would hide it from the people most likely to look.
    """
    out = generate_uncertainty_report_section({"cfg": {"coordinate_mode": "cartesian"}})
    assert "## Parameter Uncertainty" in out
    assert "No parameter uncertainty was computed" in out
    assert "cartesian" in out


def test_the_section_states_what_kind_of_number_it_is():
    """Posterior, and a precision rather than an accuracy.

    Both caveats are load-bearing: the first explains why an unmeasured
    coordinate reports a finite width, and the second is the one this whole
    session demonstrated -- a 5-sigma correction bias sat inside intervals that
    were honest about noise.
    """
    out = generate_uncertainty_report_section({"uncertainty_rows": _ROWS})
    assert "POSTERIOR" in out
    assert "PRECISION, not an accuracy" in out
    assert "optimistic" in out


def test_every_coordinate_appears_with_its_interval_and_prior_label():
    out = generate_uncertainty_report_section({"uncertainty_rows": _ROWS})
    for r in _ROWS:
        assert r["name"] in out
    assert "0.95739 to 0.95821" in out
    # the data-vs-prior label is what says which coordinates were measured
    assert "data/low" in out and "prior/high" in out


def test_chi2_inflation_is_reported_either_way():
    quiet = generate_uncertainty_report_section({"uncertainty_rows": _ROWS})
    assert "residuals within their stated sigma" in quiet

    loud = [dict(r, chi2_scale=2.25) for r in _ROWS]
    assert "2.250" in generate_uncertainty_report_section({"uncertainty_rows": loud})


def test_csv_round_trips_every_column(tmp_path):
    path = tmp_path / "parameter_uncertainty.csv"
    write_uncertainty_csv(path, _ROWS)
    with path.open(newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert [r["name"] for r in got] == ["O1-H2", "H2-O1-H3"]
    assert float(got[0]["std_err"]) == pytest.approx(0.00021)
    assert float(got[1]["ci_hi"]) == pytest.approx(104.5055)
    assert got[1]["prior_dominance"] == "prior"


def test_the_runner_hands_the_rows_back_rather_than_printing_them():
    """_print_internal_uncertainty_summary returns (path, rows), not just path."""
    import inspect

    from runner.run_generic import _print_internal_uncertainty_summary

    src = inspect.getsource(_print_internal_uncertainty_summary)
    assert "return _cov_path, rows" in src
    assert src.count("return None, None") >= 3
