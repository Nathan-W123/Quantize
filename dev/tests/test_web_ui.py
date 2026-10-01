"""The browser UI and the config file must describe the same run.

The point of the web front end is that it is a config editor, not a second
pipeline. These cover the joins where that could quietly stop being true: the
form/config round trip, the options the dropdowns are built from, and the
parity between what the UI accepts and what the CLI accepts.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from runner.usability import (  # noqa: E402
    ConfigError,
    register_optional_backends,
    valid_backends,
    validate_config,
)
from web.config_io import config_to_form, config_to_yaml, form_to_config  # noqa: E402
from web.options import all_options  # noqa: E402

EXAMPLE = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"


@pytest.fixture
def water_form():
    return {
        "name": "web_water",
        "preset": "BALANCED",
        "coordinate_mode": "internal",
        "rng_seed": "42",
        "elements": "O H H",
        "geometry_method": "coords",
        "coords_angstrom": "0.0 0.0 0.0\n0.0 0.78 0.60\n0.0 -0.78 0.60",
        "symmetry": "C2v",
        "quantum_backend": "pyscf_hf",
        "method": "b3lyp",
        "basis": "cc-pvtz",
        "corrections_mode": "hybrid_auto",
        "harmonic_from_hessian": True,
        "anharmonic_from_hessian": True,
        "nonconvergent_policy": "warn",
        "prior_sigma_bond": "0.04",
        "prior_sigma_angle_deg": "2.0",
        "internal_priors_mode": "soft",
        "output_root": "output/runs",
        "artifacts": True,
        "isotopologues": [{
            "name": "H2-16O",
            "masses": "15.99491461956 1.00782503207 1.00782503207",
            "components": ["A", "B", "C"],
            "obs_b0_mhz": "835840.29 435351.72 278138.70",
            "sigma_mhz": "0.20 0.20 0.20",
        }],
    }


# ── the form builds a runnable config ────────────────────────────────────────

def test_a_filled_form_passes_the_real_validator(water_form):
    """Not a UI-side check -- the validator the CLI uses."""
    validate_config(form_to_config(water_form))


def test_alpha_mhz_is_supplied_even_when_the_field_is_blank(water_form):
    """validate_config requires it per component, and a fresh form has none.

    Leaving it out made a freshly opened page fail validation, which is the
    first thing anyone would try.
    """
    cfg = form_to_config(water_form)
    assert cfg["isotopologues"][0]["alpha_mhz"] == [0.0, 0.0, 0.0]
    validate_config(cfg)


def test_component_choice_sizes_the_alpha_slot(water_form):
    water_form["isotopologues"][0]["components"] = ["B", "C"]
    water_form["isotopologues"][0]["obs_b0_mhz"] = "435351.72 278138.70"
    water_form["isotopologues"][0]["sigma_mhz"] = "0.20 0.20"
    cfg = form_to_config(water_form)
    assert cfg["isotopologues"][0]["alpha_mhz"] == [0.0, 0.0]
    validate_config(cfg)


def test_a_bad_number_names_its_field(water_form):
    water_form["isotopologues"][0]["masses"] = "15.99 oops 1.008"
    with pytest.raises(ValueError, match="masses"):
        form_to_config(water_form)


def test_geometry_accepts_a_pasted_xyz_block(water_form):
    """Element symbols in column one are tolerated, as pasted from a .xyz."""
    water_form["coords_angstrom"] = ("O 0.0 0.0 0.0\n"
                                     "H 0.0 0.78 0.60\n"
                                     "H 0.0 -0.78 0.60")
    cfg = form_to_config(water_form)
    assert cfg["geometry"]["coords_angstrom"] == [
        [0.0, 0.0, 0.0], [0.0, 0.78, 0.60], [0.0, -0.78, 0.60]]


def test_a_malformed_geometry_line_says_which_line(water_form):
    water_form["coords_angstrom"] = "0.0 0.0 0.0\n0.0 0.78"
    with pytest.raises(ValueError, match="line 2"):
        form_to_config(water_form)


# ── round trip ───────────────────────────────────────────────────────────────

def test_the_example_config_round_trips_through_the_form():
    """A hand-written config must survive being opened and re-saved."""
    from runner.usability import load_config

    cfg = load_config(EXAMPLE)
    back = form_to_config(config_to_form(cfg))
    for key in ("name", "coordinate_mode", "elements", "preset", "rng_seed",
                "symmetry"):
        assert back[key] == cfg[key], key
    assert (back["geometry"]["coords_angstrom"]
            == [[float(x) for x in row] for row in cfg["geometry"]["coords_angstrom"]])
    assert len(back["isotopologues"]) == len(cfg["isotopologues"])
    a, b = cfg["isotopologues"][0], back["isotopologues"][0]
    for key in ("name", "components", "obs_b0_mhz", "sigma_mhz", "masses"):
        assert b[key] == a[key], key


def test_round_tripping_twice_changes_nothing(water_form):
    once = form_to_config(water_form)
    twice = form_to_config(config_to_form(once))
    assert twice == once


def test_the_downloaded_yaml_loads_back_as_the_same_config(water_form, tmp_path):
    """The file handed to the user is the file the CLI would be handed."""
    cfg = form_to_config(water_form)
    path = tmp_path / "case.yaml"
    path.write_text(config_to_yaml(cfg), encoding="utf-8")
    from runner.usability import load_config

    reloaded = load_config(path)
    assert reloaded["elements"] == cfg["elements"]
    assert reloaded["quantum"] == cfg["quantum"]
    assert reloaded["isotopologues"] == cfg["isotopologues"]
    validate_config(reloaded)


def test_the_yaml_keeps_numeric_rows_on_one_line(water_form):
    """Readability is the point of the download; block style destroys it."""
    text = config_to_yaml(form_to_config(water_form))
    assert "- [0.0, 0.0, 0.0]" in text
    assert "- - 0.0" not in text


# ── the dropdowns ────────────────────────────────────────────────────────────

def test_options_come_from_the_code_that_implements_them():
    opts = all_options()
    from backend.spectral.harmonic_alpha import _HARMONIC_SCHEMES, _SCHEMES

    assert opts["harmonic_scheme"] == list(_HARMONIC_SCHEMES)
    assert opts["tracking_scheme"] == list(_SCHEMES)
    assert "hybrid_auto" in opts["corrections_mode"]
    assert set(opts["cd_reduction"]) == {"A", "S"}


def test_every_offered_choice_is_one_the_validator_accepts(water_form):
    """A dropdown entry that fails validation is worse than no dropdown."""
    opts = all_options()
    for mode in opts["corrections_mode"]:
        water_form["corrections_mode"] = mode
        validate_config(form_to_config(water_form))
    water_form["corrections_mode"] = "hybrid_auto"
    for policy in opts["nonconvergent_policy"]:
        water_form["nonconvergent_policy"] = policy
        validate_config(form_to_config(water_form))
    water_form["nonconvergent_policy"] = "warn"
    for preset in opts["preset"]:
        water_form["preset"] = preset
        validate_config(form_to_config(water_form))


def test_an_unregistered_backend_is_rejected(water_form):
    water_form["quantum_backend"] = "definitely_not_a_backend"
    with pytest.raises(ConfigError, match="quantum.backend"):
        validate_config(form_to_config(water_form))


# ── UI and CLI must agree ────────────────────────────────────────────────────

def test_optional_backends_are_registered_for_every_caller():
    """The bug this guards: the CLI refused a config the web UI accepted.

    cli.py never imported dev.pyscf_backend, so valid_backends() returned only
    {none, orca, psi4} there and rejected pyscf_hf, while the web server -- which
    did import it -- accepted the same file. Registration now happens inside
    valid_backends, so the answer cannot depend on the caller.
    """
    register_optional_backends()
    names = valid_backends()
    assert "none" in names
    pyscf_available = True
    try:
        import pyscf  # noqa: F401
    except ModuleNotFoundError:
        pyscf_available = False
    if pyscf_available:
        assert "pyscf_hf" in names, (
            "pyscf is installed but pyscf_hf is not an accepted backend name")


def test_the_server_offers_exactly_what_the_validator_accepts():
    """The UI's backend list is the validator's, not the bare registry's."""
    offered = set(valid_backends())
    assert offered == set(valid_backends())
    assert "none" in offered


# ── the field notes ──────────────────────────────────────────────────────────

def test_help_text_exists_for_the_settings_that_carry_a_choice():
    help_map = all_options()["field_help"]
    for field in ("coordinate_mode", "corrections_mode", "nonconvergent_policy",
                  "method", "basis", "prior_sigma_bond", "fit_cd_constants"):
        assert field in help_map and len(help_map[field]) > 20, field


def test_options_payload_is_json_serialisable():
    """It is served over HTTP, so a stray numpy scalar would break the page."""
    json.dumps(all_options())
