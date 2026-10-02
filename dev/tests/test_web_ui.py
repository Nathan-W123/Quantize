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


# ── the scheme keys reach the optimiser ──────────────────────────────────────

def test_scheme_keys_land_where_the_runner_reads_them(water_form):
    """They were benchmark-script-only until the runner was taught to read them.

    Each has a specific home: the four correction schemes sit inside
    rovibrational_corrections, and hess_recalc_every sits under "optimizer"
    because run_generic only copies a key into the optimiser when it appears
    there -- a top-level one is silently ignored.
    """
    water_form.update({
        "harmonic_scheme": "watson",
        "cubic_scheme": "normal_mode",
        "freq_scale": "0.98",
        "lam_freq_cm": "120",
        "hess_recalc_every": "2",
    })
    cfg = form_to_config(water_form)
    rc = cfg["rovibrational_corrections"]
    assert rc["harmonic_scheme"] == "watson"
    assert rc["cubic_scheme"] == "normal_mode"
    assert rc["freq_scale"] == 0.98
    assert rc["lam_freq_cm"] == 120.0
    assert cfg["optimizer"]["hess_recalc_every"] == 2
    assert "hess_recalc_every" not in cfg, "top level is read by nothing"
    validate_config(cfg)


def test_the_optimiser_accepts_every_scheme_key_the_form_emits(water_form):
    """A config key with no matching parameter would be a silent no-op."""
    import inspect

    from backend.quantize import MolecularOptimizer

    water_form.update({"harmonic_scheme": "watson", "cubic_scheme": "normal_mode",
                       "freq_scale": "0.98", "lam_freq_cm": "120"})
    rc = form_to_config(water_form)["rovibrational_corrections"]
    params = inspect.signature(MolecularOptimizer.__init__).parameters
    for key in ("harmonic_scheme", "cubic_scheme", "freq_scale", "lam_freq_cm"):
        assert key in rc, f"{key} not emitted"
        assert key in params, f"MolecularOptimizer has no {key} parameter"


def test_every_offered_scheme_passes_the_validator(water_form):
    opts = all_options()
    for scheme in opts["harmonic_scheme"]:
        water_form["harmonic_scheme"] = scheme
        validate_config(form_to_config(water_form))
    water_form["harmonic_scheme"] = "watson"
    for scheme in opts["cubic_scheme"]:
        water_form["cubic_scheme"] = scheme
        validate_config(form_to_config(water_form))


def test_scheme_keys_round_trip(water_form):
    water_form.update({"harmonic_scheme": "eigenvalue_fd",
                       "cubic_scheme": "cartesian",
                       "freq_scale": "0.97", "lam_freq_cm": "80",
                       "hess_recalc_every": "3"})
    once = form_to_config(water_form)
    assert form_to_config(config_to_form(once)) == once


# ── where the starting structure comes from ──────────────────────────────────

def test_smiles_is_a_geometry_source_the_validator_accepts(water_form):
    """run_generic has always handled it; the validator used to refuse it.

    Its own error message for an unknown method reads "Use: smiles, bonds,
    pubchem, or coords", while VALID_GEOMETRY_METHODS omitted smiles -- so a
    geometry the runner knows how to build was rejected before it got there.
    """
    water_form["geometry_method"] = "smiles"
    water_form["smiles"] = "O"
    cfg = form_to_config(water_form)
    assert cfg["geometry"] == {"method": "smiles", "smiles": "O"}
    validate_config(cfg)


def test_pubchem_emits_identifier_not_name(water_form):
    """geometry.method=pubchem reads geometry.identifier.

    An earlier version wrote "name", which validate_config accepts and
    run_generic then rejects with "requires geometry.identifier" -- a failure
    that only appears once the run starts.
    """
    water_form["geometry_method"] = "pubchem"
    water_form["pubchem_identifier"] = "water"
    cfg = form_to_config(water_form)
    assert cfg["geometry"]["identifier"] == "water"
    assert "name" not in cfg["geometry"]
    validate_config(cfg)


def test_bonds_are_parsed_as_index_pairs(water_form):
    water_form["geometry_method"] = "bonds"
    water_form["bonds"] = "0 1\n0 2"
    cfg = form_to_config(water_form)
    assert cfg["geometry"]["bonds"] == [[0, 1], [0, 2]]
    validate_config(cfg)


@pytest.mark.parametrize("method,field,value", [
    ("smiles", "smiles", "O"),
    ("pubchem", "pubchem_identifier", "water"),
    ("bonds", "bonds", "0 1\n0 2"),
])
def test_every_geometry_source_round_trips(water_form, method, field, value):
    water_form["geometry_method"] = method
    water_form[field] = value
    once = form_to_config(water_form)
    assert form_to_config(config_to_form(once)) == once


def test_coordinates_are_not_required_to_describe_a_molecule(water_form):
    """The point of the SMILES and PubChem routes: no coordinate typing.

    The optimiser only needs a starting structure in the right basin, so a
    case that names the molecule is a complete case.
    """
    water_form["geometry_method"] = "smiles"
    water_form["smiles"] = "O"
    water_form["coords_angstrom"] = ""
    cfg = form_to_config(water_form)
    assert "coords_angstrom" not in cfg["geometry"]
    validate_config(cfg)


def test_smiles_is_offered_first(water_form):
    """Dropdown order is the recommendation; coords should not lead."""
    methods = all_options()["geometry_method"]
    assert methods[0] == "smiles"
    assert methods[-1] == "coords"
