"""Subcommands must run, and must do what their flags say.

Found in a dead-code sweep, all three by reading what static analysis called
unused rather than deleting it:

* `quantize validate` printed an empty "Config Summary" header. An earlier
  commit pasted `_json_safe` into the middle of `_print_config_summary`, so the
  summary's seven lines sat after `_json_safe`'s final `return`, unreachable,
  and the fields it computed were never printed.
* `quantize lam-scan` raised TypeError on every call: it passed
  `symmetry_number` to `fit_fourier_potential`, which takes none. The
  symmetry-aware wrapper it needed, `scan_to_torsion_potential`, already
  existed beside it.
* `--energy-unit` was parsed and never applied. Energies went into the fit as
  read and every line of output was labelled cm^-1, so a scan in kcal/mol
  would have printed a potential 349.755x too small, in the wrong unit, with
  no warning. It now has argparse choices, like `--angle-unit` already did.

Run through the real entry point, so a broken import or signature fails here
rather than at a user's terminal.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
KCAL_TO_CM1 = 349.7550874793


def _cli(*args):
    proc = subprocess.run([sys.executable, str(_ROOT / "cli.py"), *args],
                          cwd=str(_ROOT), capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def v3_scan_kcal(tmp_path):
    """V(phi) = (V3/2)(1 - cos 3 phi), V3 = 1 kcal/mol, every 10 degrees."""
    phi = np.arange(0.0, 360.0, 10.0)
    e = 0.5 * 1.0 * (1.0 - np.cos(np.radians(3.0 * phi)))
    path = tmp_path / "scan.csv"
    path.write_text("phi_deg,energy_cm1\n"
                    + "".join(f"{a},{b}\n" for a, b in zip(phi, e)))
    return path


def _coef(out, label):
    for line in out.splitlines():
        if line.strip().startswith(label):
            return float(line.split(":")[1].split()[0])
    raise AssertionError(f"{label} not in output:\n{out}")


def test_validate_prints_the_summary_it_computes():
    code, out = _cli("validate", "configs/example_CO2.yaml")
    assert code == 0, out
    for field in ("Name             : CO2", "Coordinate mode", "Quantum backend",
                  "Elements         : 3 atoms", "Isotopologues    : 4", "Preset"):
        assert field in out, f"missing {field!r} in:\n{out}"


def test_lam_scan_runs_at_all(v3_scan_kcal):
    code, out = _cli("lam-scan", str(v3_scan_kcal), "--n-harmonics", "2",
                     "--symmetry-number", "3")
    assert code == 0, out
    assert "Scan Fourier fit" in out


def test_energy_unit_is_applied(v3_scan_kcal):
    """1 kcal/mol barrier: v0 = +V3/2 and Vcos_3 = -V3/2, in cm^-1."""
    code, out = _cli("lam-scan", str(v3_scan_kcal), "--energy-unit", "kcal/mol",
                     "--n-harmonics", "2", "--symmetry-number", "3")
    assert code == 0, out
    assert _coef(out, "Vcos_3") == pytest.approx(-0.5 * KCAL_TO_CM1, rel=1e-6)
    assert _coef(out, "v0") == pytest.approx(0.5 * KCAL_TO_CM1, rel=1e-6)
    assert abs(_coef(out, "Vcos_6")) < 1e-6


def test_only_symmetry_allowed_harmonics_are_fitted(v3_scan_kcal):
    """sigma = 3 and two groups: V3 and V6, never V1, V2, V4 or V5."""
    code, out = _cli("lam-scan", str(v3_scan_kcal), "--n-harmonics", "2",
                     "--symmetry-number", "3")
    assert code == 0, out
    assert "Vcos_3" in out and "Vcos_6" in out
    for forbidden in ("Vcos_1 ", "Vcos_2 ", "Vcos_4 ", "Vcos_5 "):
        assert forbidden not in out


@pytest.mark.parametrize("flag,value", [("--energy-unit", "kcal"),
                                        ("--angle-unit", "gradians")])
def test_an_unknown_unit_is_refused_before_any_fit(v3_scan_kcal, flag, value):
    code, out = _cli("lam-scan", str(v3_scan_kcal), flag, value)
    assert code == 2
    assert "invalid choice" in out
