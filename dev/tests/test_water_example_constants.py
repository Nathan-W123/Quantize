"""The water configs' ground-state constants against the JPL catalogue and the
rigid-rotor rule that catches a garbled copy.

The README's worked example (configs/example_water_semi_experimental.yaml)
carried H2-18O constants 833931 / 411705 / 272914 MHz, B 24 GHz off any
source. In water the b axis is the C2 axis, which contains the O atom, so a
heavier O leaves B unchanged in the rigid rotor: H2-18O's B0 differs from
H2-16O's only by the change in vibrational averaging (JPL: +2.4 MHz). Measured
on a pyscf B3LYP/6-31G(d) stand-in for the example's ORCA level, the wrong
constants put the angle 0.280 deg from r_e and the bond 6.93 mA; with JPL's,
-0.003 deg and 5.91 mA.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent

# JPL catalogue documentation (spec.jpl.nasa.gov/ftp/pub/catalog/doc/):
# d018003 (H2O, Yu et al. 2012), d020001 (D2O), d020003 (H2-18O, De Lucia et
# al. 1972 and Johns 1985). A, B, C in MHz.
JPL = {
    "H2-16O": (835840.3, 435351.7, 278138.7),
    "D2-16O": (462278.8, 218038.4, 145258.0),
    "H2-18O": (825367.80, 435354.05, 276951.05),
}

WATER_CONFIGS = ["configs/example_water_semi_experimental.yaml", "configs/Water.yaml"]


def _species(path):
    cfg = yaml.safe_load((_ROOT / path).read_text(encoding="utf-8"))
    return {i["name"]: [float(v) for v in i["obs_b0_mhz"]] for i in cfg["isotopologues"]}


def test_the_worked_example_carries_the_catalogue_constants():
    got = _species("configs/example_water_semi_experimental.yaml")
    for name, ref in JPL.items():
        assert got[name] == pytest.approx(list(ref), abs=0.5), name


@pytest.mark.parametrize("path", WATER_CONFIGS)
def test_an_18o_substitution_leaves_b_where_it_was(path):
    """A copy whose H2-18O B is not within 50 MHz of H2-16O's is garbled.
    configs/Water.yaml carries NIST SRD 117's set (ref. 74005), B 20 MHz
    below H2-16O's; JPL 20003's is 2.4 MHz above, and theory puts the shift
    at +12 to +20 MHz (HF and B3LYP). The two published sets differ by more
    than NIST's stated 8 MHz, a fitting-model difference this rule does not
    try to settle. The example's 411705 was 23.6 GHz off."""
    got = _species(path)
    if "H2-18O" not in got:
        pytest.skip("no H2-18O")
    assert abs(got["H2-18O"][1] - got["H2-16O"][1]) < 50.0
