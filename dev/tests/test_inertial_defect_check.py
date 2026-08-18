"""The inertial-defect sanity check on supplied rotational constants.

Catches transcription errors before a run spends hours converging on data no
real structure can produce. configs/fluorobenzene.yaml is the motivating case:
every isotopologue there has a defect near +5.6 amu*A^2 for a planar molecule.
"""

import copy

import pytest
import yaml

from runner.usability import (
    _INERTIAL_DEFECT_WARN,
    inertial_defect_mhz,
    validate_config,
)

# Accepted fluorobenzene parent constants (MHz) and a planar water set.
_FLUOROBENZENE_REAL = (5663.5, 2570.6, 1767.1)
_WATER = (835840.29, 435351.72, 278138.70)


def _minimal_config(obs):
    return {
        "name": "t",
        "coordinate_mode": "cartesian",
        "elements": ["O", "H", "H"],
        "geometry": {"method": "coords", "coords_angstrom": [
            [0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]},
        "isotopologues": [{
            "name": "iso",
            "masses": [15.99491461956, 1.00782503207, 1.00782503207],
            "components": ["A", "B", "C"],
            "obs_b0_mhz": list(obs),
            "alpha_mhz": [0.0, 0.0, 0.0],
            "sigma_mhz": [0.2, 0.2, 0.2],
        }],
        "quantum": {"backend": "none"},
    }


def test_planar_molecule_has_a_small_defect():
    assert abs(inertial_defect_mhz(*_FLUOROBENZENE_REAL)) < 0.5
    # Water is a light planar triatomic; its defect is small but not tiny.
    assert abs(inertial_defect_mhz(*_WATER)) < 0.5


def test_shipped_fluorobenzene_constants_are_flagged(capsys):
    """Regression on the actual defect: the config's own numbers are impossible."""
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "fluorobenzene.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for iso in cfg["isotopologues"]:
        defect = inertial_defect_mhz(*iso["obs_b0_mhz"])
        assert defect > 5.0, f"{iso['name']} defect {defect:+.3f} unexpectedly plausible"


#: The shipped fluorobenzene parent constants, which are the motivating defect.
_FLUOROBENZENE_BAD = (5667.887, 2601.998, 1748.116)


def test_validate_config_warns_on_impossible_constants(capsys):
    assert inertial_defect_mhz(*_FLUOROBENZENE_BAD) > _INERTIAL_DEFECT_WARN
    validate_config(_minimal_config(_FLUOROBENZENE_BAD))
    assert "Inertial defect" in capsys.readouterr().out


def test_planar_second_moment_is_what_the_check_really_tests():
    """P_c = (I_a + I_b - I_c)/2 = sum(m z^2) must be >= 0 for real atoms, and
    is only slightly negative for a planar molecule because of zero-point motion.
    The defect is -2 P_c, so a large positive defect means a large negative P_c."""
    for constants, limit in ((_FLUOROBENZENE_REAL, 0.5), (_WATER, 0.5)):
        p_c = -0.5 * inertial_defect_mhz(*constants)
        assert abs(p_c) < limit
    p_c_bad = -0.5 * inertial_defect_mhz(*_FLUOROBENZENE_BAD)
    assert p_c_bad < -2.0, "expected a strongly negative out-of-plane second moment"


def test_validate_config_stays_quiet_on_sane_constants(capsys):
    validate_config(_minimal_config(_WATER))
    assert "Inertial defect" not in capsys.readouterr().out


def test_check_is_a_warning_not_an_error():
    """A floppy molecule can sit near the threshold, so this must not block a run."""
    validate_config(_minimal_config(_FLUOROBENZENE_BAD))    # must not raise


def test_partial_component_sets_are_skipped(capsys):
    cfg = _minimal_config(_WATER)
    cfg["isotopologues"][0]["components"] = ["B", "C"]
    cfg["isotopologues"][0]["obs_b0_mhz"] = [435351.72, 278138.70]
    cfg["isotopologues"][0]["alpha_mhz"] = [0.0, 0.0]
    cfg["isotopologues"][0]["sigma_mhz"] = [0.2, 0.2]
    validate_config(cfg)
    assert "Inertial defect" not in capsys.readouterr().out
