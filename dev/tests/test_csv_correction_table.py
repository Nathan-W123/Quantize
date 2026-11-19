"""CSV correction tables reach the engine.

`rovibrational_corrections.correction_table` has always been validated as
".csv, .yaml, or .yml", but parse_correction_table read every file as YAML,
so a CSV passed `quantize validate` and then failed in the engine with "must
be a mapping at top level, got str". The CSV reader (rovib_tables) existed and
was wired to nothing. These pin both layouts, and the whole config path from
validation to the corrected target the fit uses.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.spectral.correction_models import parse_correction_table  # noqa: E402
from runner.usability import load_config, validate_config  # noqa: E402

B0_A = 835840.29          # water's A in dev/configs/water_analytic_endtoend.yaml
DELTA_A, SIGMA_A = -21175.0, 100.0

DELTA_CSV = ("isotopologue,component,delta_vib_MHz,sigma_delta_vib_MHz,source,method,basis,status\n"
             f"H2-16O,A,{DELTA_A},{SIGMA_A},lit,VPT2,cc-pVTZ,ok\n")
ALPHA_CSV = ("isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status\n"
             f"H2-16O,A,{2 * DELTA_A},{2 * SIGMA_A},lit,VPT2,cc-pVTZ,ok\n")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestParsing:
    def test_the_delta_layout(self, tmp_path):
        t = parse_correction_table(_write(tmp_path, "d.csv", DELTA_CSV))
        assert t["H2-16O"]["A"]["delta_mhz"] == DELTA_A
        assert t["H2-16O"]["A"]["sigma_mhz"] == SIGMA_A
        assert t["H2-16O"]["A"]["method"] == "VPT2"

    def test_the_alpha_layout_is_an_alpha_sum(self, tmp_path):
        # delta = alpha_sum / 2 is applied downstream; the sigma is already
        # the delta's (sigma_alpha / 2).
        t = parse_correction_table(_write(tmp_path, "a.csv", ALPHA_CSV))
        assert t["H2-16O"]["A"]["alpha_sum_mhz"] == 2 * DELTA_A
        assert t["H2-16O"]["A"]["sigma_mhz"] == SIGMA_A

    def test_yaml_is_still_read_as_yaml(self, tmp_path):
        y = _write(tmp_path, "t.yaml", f"H2-16O:\n  A: {{delta_mhz: {DELTA_A}, sigma_mhz: {SIGMA_A}}}\n")
        assert parse_correction_table(y)["H2-16O"]["A"]["delta_mhz"] == DELTA_A


def _target_a(tmp_path, table_path):
    """Validate a config that uses this table, build its optimizer, and return
    the corrected A target the fit will use."""
    from runner.run_from_config import build_optimizer_from_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    cfg["rovibrational_corrections"] = {"mode": "user_only", "correction_table": str(table_path)}
    with contextlib.redirect_stdout(io.StringIO()):
        validate_config(cfg)
        opt = build_optimizer_from_config(cfg, verbose=False)
    targets = [t for t in opt._corrected_targets
               if t.isotopologue_label == "H2-16O" and t.component_index == 0]
    assert len(targets) == 1
    return targets[0]


class TestThroughTheConfigPath:
    @pytest.mark.parametrize("layout", ["delta", "alpha"])
    def test_a_csv_table_corrects_the_target(self, tmp_path, layout):
        # The config path applies other small terms on top (here -13.65 MHz),
        # so isolate the table's own effect against a zero-delta table.
        text = DELTA_CSV if layout == "delta" else ALPHA_CSV
        t = _target_a(tmp_path, _write(tmp_path, f"{layout}.csv", text))
        zero = _target_a(tmp_path, _write(tmp_path, "zero.yaml", "H2-16O:\n  A: {delta_mhz: 0.0}\n"))
        assert t.value_mhz - zero.value_mhz == pytest.approx(DELTA_A, abs=1e-6)
        assert abs(t.value_mhz - B0_A - DELTA_A) < 50.0   # and the rest is small

    def test_csv_and_yaml_give_the_same_target(self, tmp_path):
        c = _target_a(tmp_path, _write(tmp_path, "d.csv", DELTA_CSV))
        y = _target_a(tmp_path, _write(
            tmp_path, "d.yaml",
            f"H2-16O:\n  A: {{delta_mhz: {DELTA_A}, sigma_mhz: {SIGMA_A}, method: VPT2}}\n"))
        assert c.value_mhz == pytest.approx(y.value_mhz, abs=1e-9)
        assert c.sigma_mhz == pytest.approx(y.sigma_mhz, rel=1e-12)
