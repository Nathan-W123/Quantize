"""The benchmark's results table: what each row is the best of, and from which code."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))


def _bench():
    import importlib.util as iu
    spec = iu.spec_from_file_location("accbench_t", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    saved = sys.argv
    try:
        sys.argv = ["accbench_t"]
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def _row(th, me, hy, vintage="abc1234"):
    return {"theory_rms_ma": th, "me_rms_ma": me, "hybrid_rms_ma": hy, "code_vintage": vintage}


def test_each_leg_takes_its_own_best_and_one_config_takes_none():
    """Best-of is per leg and over every configuration on file; configs=base
    is the one-setting table (no selection), which the best-of-N one flatters
    more the more configurations a file holds."""
    b = _bench()
    tables = {"L": {"water": {"base": _row(8.0, 2.0, 1.0), "offsets": _row(7.0, 3.0, 0.5),
                              "elec": _row(9.0, 1.5, 2.0)}}}
    best = b.summary_rows(tables)["L"]
    assert best["rows"]["water"] == (7.0, 1.5, 0.5)
    assert best["configs"] == ["base", "elec", "offsets"] and best["vintages"] == ["abc1234"]
    one = b.summary_rows(tables, ["base"])["L"]
    assert one["rows"]["water"] == (8.0, 2.0, 1.0) and one["configs"] == ["base"]


def test_the_table_names_its_configurations_and_mixed_vintages(tmp_path):
    """A results file accumulates rows across runs and skips configurations it
    already holds, so a table can silently mix two programs; it now says
    which code its numbers come from."""
    b = _bench()
    (tmp_path / "accuracy_upgrades_hf_6-31g.json").write_text(json.dumps(
        {"water": {"base": _row(8.0, 2.0, 1.0), "offsets": _row(7.0, 3.0, 0.5, "old9999")}}))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.print_summary_table(None, tmp_path)
    txt = buf.getvalue()
    assert "best of 2 configurations per leg" in txt
    assert "MIXED VINTAGES" in txt and "old9999" in txt
    assert "hybrid best of three on 1/1" in txt
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.print_summary_table(["base"], tmp_path)
    txt = buf.getvalue()
    assert "one configuration, base; code vintage abc1234" in txt
    assert "MIXED" not in txt.split("\n")[0]
