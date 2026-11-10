"""The correction's expansion point follows the geometry the fit lands on.

alpha is expanded at the quantum minimum and then frozen, but the joint
objective exists to move the geometry off that minimum, and alpha is not flat
in between. These cover the three pieces that let the table track the fit:
the directional derivative, the extrapolation, and the two-pass driver.

Everything here runs on the analytic water surface, so it is deterministic
and needs no quantum chemistry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.spectral.correction_models import parse_correction_table  # noqa: E402
from backend.spectral.harmonic_alpha import (  # noqa: E402
    alpha_directional_derivative,
    build_correction_table_from_hessian,
    extrapolate_correction_table,
    self_consistent_correction_table,
)
from reference_molecules import H2O_MASSES, h2o_coords, h2o_hessian  # noqa: E402

_TABLE_KW = dict(cubic_scheme="normal_mode", harmonic_scheme="watson")


@pytest.fixture(scope="module")
def isos():
    return [{"name": "H2-16O", "masses": list(H2O_MASSES)}]


@pytest.fixture(scope="module")
def coords():
    return h2o_coords()


@pytest.fixture(scope="module")
def base_table(coords, isos):
    table, _ = build_correction_table_from_hessian(
        h2o_hessian(coords), coords, isos, hessian_fn=h2o_hessian, **_TABLE_KW)
    return table


def _stretch_direction(coords):
    """Symmetric stretch: both H atoms outward from the O, 1 mA total."""
    step = np.zeros_like(coords)
    for i in (1, 2):
        v = coords[i] - coords[0]
        step[i] = 0.001 * v / np.linalg.norm(v)
    return step


def test_directional_derivative_is_finite_for_every_component(coords, isos):
    slopes, info = alpha_directional_derivative(
        h2o_hessian, coords, isos, _stretch_direction(coords), **_TABLE_KW)
    assert set(slopes["H2-16O"]) == {"A", "B", "C"}
    for value in slopes["H2-16O"].values():
        assert np.isfinite(value)
    assert info["step_ang"] == pytest.approx(
        np.linalg.norm(_stretch_direction(coords)))


def test_reversing_the_direction_flips_the_slope(coords, isos):
    step = _stretch_direction(coords)
    fwd, _ = alpha_directional_derivative(h2o_hessian, coords, isos, step, **_TABLE_KW)
    rev, _ = alpha_directional_derivative(h2o_hessian, coords, isos, -step, **_TABLE_KW)
    for comp in "ABC":
        assert fwd["H2-16O"][comp] == pytest.approx(-rev["H2-16O"][comp], rel=1e-6)


def test_slope_is_converged_that_is_it_is_a_derivative_not_a_difference(coords, isos):
    """Halving the step must not materially move the slope."""
    step = _stretch_direction(coords)
    coarse, _ = alpha_directional_derivative(
        h2o_hessian, coords, isos, step, **_TABLE_KW)
    fine, _ = alpha_directional_derivative(
        h2o_hessian, coords, isos, step, step_ang=0.5 * np.linalg.norm(step),
        **_TABLE_KW)
    for comp in "ABC":
        a, b = coarse["H2-16O"][comp], fine["H2-16O"][comp]
        assert a == pytest.approx(b, rel=0.05), f"{comp}: {a} vs {b}"


def test_slope_predicts_a_directly_recomputed_alpha(coords, isos):
    """The whole premise: alpha(x + d) ~= alpha(x) + slope*|d|."""
    step = _stretch_direction(coords)
    slopes, _ = alpha_directional_derivative(h2o_hessian, coords, isos, step, **_TABLE_KW)
    dist = float(np.linalg.norm(step))

    at_x, _ = build_correction_table_from_hessian(
        h2o_hessian(coords), coords, isos, hessian_fn=h2o_hessian, **_TABLE_KW)
    moved = coords + step
    at_moved, _ = build_correction_table_from_hessian(
        h2o_hessian(moved), moved, isos, hessian_fn=h2o_hessian, **_TABLE_KW)

    for comp in "ABC":
        direct = float(at_moved["H2-16O"][comp]["alpha_sum_mhz"])
        predicted = (float(at_x["H2-16O"][comp]["alpha_sum_mhz"])
                     + slopes["H2-16O"][comp] * dist)
        spread = abs(direct - float(at_x["H2-16O"][comp]["alpha_sum_mhz"]))
        # Within 10% of the change being predicted, or a few MHz where the
        # change is itself tiny -- this is a first-order model, not an identity.
        assert abs(predicted - direct) <= max(0.10 * spread, 5.0), (
            f"{comp}: predicted {predicted:.1f}, direct {direct:.1f}")


def test_extrapolation_shifts_alpha_by_slope_times_distance(base_table):
    slopes = {"H2-16O": {"A": 1000.0, "B": -250.0, "C": 10.0}}
    out, info = extrapolate_correction_table(base_table, slopes, 0.003)
    assert info["components_shifted"] == 3
    for comp, slope in slopes["H2-16O"].items():
        before = float(base_table["H2-16O"][comp]["alpha_sum_mhz"])
        after = float(out["H2-16O"][comp]["alpha_sum_mhz"])
        assert after - before == pytest.approx(slope * 0.003)


def test_extrapolation_grows_sigma_and_leaves_the_original_alone(base_table):
    slopes = {"H2-16O": {"A": 5000.0}}
    out, _ = extrapolate_correction_table(base_table, slopes, 0.004)
    before = base_table["H2-16O"]["A"]
    after = out["H2-16O"]["A"]
    assert after["sigma_mhz"] > before["sigma_mhz"]
    # the input table must not be mutated
    assert "tracked" not in str(before.get("notes", ""))
    assert "tracked" in str(after["notes"])


def test_zero_distance_changes_nothing(base_table):
    slopes = {"H2-16O": {"A": 5000.0, "B": 5000.0, "C": 5000.0}}
    out, _ = extrapolate_correction_table(base_table, slopes, 0.0)
    for comp in "ABC":
        assert (float(out["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"])))
        assert (float(out["H2-16O"][comp]["sigma_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["sigma_mhz"])))


def test_components_without_a_slope_pass_through_untouched(base_table):
    out, info = extrapolate_correction_table(base_table, {"H2-16O": {"A": 100.0}}, 0.005)
    assert info["components_shifted"] == 1
    for comp in ("B", "C"):
        assert (float(out["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"])))


def test_the_extrapolated_table_is_still_a_valid_correction_table(base_table):
    out, _ = extrapolate_correction_table(
        base_table, {"H2-16O": {"A": 900.0, "B": -100.0, "C": 5.0}}, 0.003)
    parsed = parse_correction_table(out)
    assert set(parsed["H2-16O"]) == {"A", "B", "C"}


def test_a_fit_that_does_not_move_leaves_the_table_alone(coords, isos, base_table):
    """No displacement, nothing to track -- and no alpha evaluations spent."""
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: coords, **_TABLE_KW)
    assert info["slopes"] is None
    assert info["self_consistent_passes"] == 1
    for comp in "ABC":
        assert (float(table["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"])))


def test_a_fit_that_moves_shifts_the_table(coords, isos, base_table):
    target = coords + _stretch_direction(coords)
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="linear", **_TABLE_KW)
    assert info["slopes"] is not None
    assert info["self_consistent_passes"] == 2
    moved = [comp for comp in "ABC"
             if float(table["H2-16O"][comp]["alpha_sum_mhz"])
             != pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"]))]
    assert moved, "no component moved"


def test_passes_do_not_compound_the_extrapolation(coords, isos, base_table):
    """Each pass re-extrapolates the BASE table, so a fixed fit is idempotent.

    Every pass measures the displacement from the stationary point, not from
    wherever the last pass left the table, and extrapolates the base table by
    it. So a fit that keeps returning the same geometry must keep producing
    the same table. If a pass extrapolated from the previous pass's table
    instead, three passes would shift alpha three times over.
    """
    target = coords + _stretch_direction(coords)
    one, _ = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, passes=1,
        scheme="linear", **_TABLE_KW)
    three, _ = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, passes=3,
        scheme="linear", **_TABLE_KW)
    for comp in "ABC":
        a = float(one["H2-16O"][comp]["alpha_sum_mhz"])
        b = float(three["H2-16O"][comp]["alpha_sum_mhz"])
        assert a == pytest.approx(b, rel=1e-12), f"{comp}: {a} vs {b}"
        # and it did move off the base, or the test proves nothing
        assert a != pytest.approx(
            float(base_table["H2-16O"][comp]["alpha_sum_mhz"]), rel=1e-12)


def test_passes_must_be_at_least_one(coords, isos):
    with pytest.raises(ValueError, match="at least 1"):
        self_consistent_correction_table(
            h2o_hessian, coords, isos, lambda _t: coords, passes=0, **_TABLE_KW)


def test_a_zero_length_direction_is_rejected(coords, isos):
    with pytest.raises(ValueError, match="zero length"):
        alpha_directional_derivative(
            h2o_hessian, coords, isos, np.zeros_like(coords), **_TABLE_KW)


def test_a_fit_that_moves_too_far_is_not_extrapolated(coords, isos, base_table):
    """Past the validity limit the table is handed back untracked.

    alpha(x) is expanded to first order, so a large displacement is outside
    what the slope can describe. Measured across 12 molecule/level pairs the
    outcome changes sign around 20 mA, and extrapolating anyway produced the
    three worst regressions in the set. The conservative failure is to leave
    the table alone rather than shift it by an amount nothing supports.
    """
    far = coords + 200.0 * _stretch_direction(coords)   # 200 mA, far past the gate
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: far, scheme="linear", **_TABLE_KW)
    assert info["skipped_beyond_ang"] is not None
    assert info["skipped_beyond_ang"] > info["max_distance_ang"]
    assert info["slopes"] is None, "no alpha evaluations should have been spent"
    for comp in "ABC":
        assert (float(table["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"])))


def test_the_gate_can_be_raised(coords, isos, base_table):
    """The limit is a guard rail, not a hard law -- a caller may widen it."""
    far = coords + 50.0 * _stretch_direction(coords)    # 50 mA
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: far, scheme="linear",
        max_distance_ang=1.0, **_TABLE_KW)
    assert info["skipped_beyond_ang"] is None
    assert info["slopes"] is not None
    moved = [c for c in "ABC"
             if float(table["H2-16O"][c]["alpha_sum_mhz"])
             != pytest.approx(float(base_table["H2-16O"][c]["alpha_sum_mhz"]))]
    assert moved


def test_a_short_move_still_tracks(coords, isos, base_table):
    """The gate must not disable tracking in the regime where it works."""
    near = coords + _stretch_direction(coords)          # 1 mA, well inside
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: near, scheme="linear", **_TABLE_KW)
    assert info["skipped_beyond_ang"] is None
    assert info["slopes"] is not None


# ── schemes ──────────────────────────────────────────────────────────────────

def test_direct_scheme_builds_the_table_at_the_fitted_geometry(coords, isos):
    """"direct" must reproduce a table built there by hand."""
    target = coords + _stretch_direction(coords)
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="direct", **_TABLE_KW)
    assert info["scheme"] == "direct"
    assert info["slopes"] is None, "direct spends no derivative evaluations"
    by_hand, _ = build_correction_table_from_hessian(
        h2o_hessian(target), target, isos, hessian_fn=h2o_hessian, **_TABLE_KW)
    for comp in "ABC":
        assert (float(table["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(by_hand["H2-16O"][comp]["alpha_sum_mhz"])))


def test_quadratic_differs_from_linear_and_costs_no_extra_chemistry(coords, isos):
    """The curvature comes from points the linear scheme already evaluates."""
    target = coords + 8.0 * _stretch_direction(coords)   # far enough for h^2 to bite
    lin, lin_info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="linear", **_TABLE_KW)
    quad, quad_info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="quadratic", **_TABLE_KW)
    assert quad_info["curvatures"] is not None
    assert lin_info["curvatures"] is None
    # same slopes -- the quadratic term is the only difference
    for comp in "ABC":
        assert (lin_info["slopes"]["H2-16O"][comp]
                == pytest.approx(quad_info["slopes"]["H2-16O"][comp]))
    moved = [c for c in "ABC"
             if float(quad["H2-16O"][c]["alpha_sum_mhz"])
             != pytest.approx(float(lin["H2-16O"][c]["alpha_sum_mhz"]))]
    assert moved, "the second-order term changed nothing"


def test_quadratic_beats_linear_against_a_directly_built_table(coords, isos):
    """Second order should sit closer to the truth than first order does.

    "direct" is the reference here: it is alpha actually evaluated there,
    which is what both extrapolations are approximating.
    """
    target = coords + 8.0 * _stretch_direction(coords)
    truth, _ = build_correction_table_from_hessian(
        h2o_hessian(target), target, isos, hessian_fn=h2o_hessian, **_TABLE_KW)
    lin, _ = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="linear", **_TABLE_KW)
    quad, _ = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: target, scheme="quadratic", **_TABLE_KW)
    for comp in "ABC":
        t = float(truth["H2-16O"][comp]["alpha_sum_mhz"])
        e_lin = abs(float(lin["H2-16O"][comp]["alpha_sum_mhz"]) - t)
        e_quad = abs(float(quad["H2-16O"][comp]["alpha_sum_mhz"]) - t)
        assert e_quad <= e_lin + 1e-6, f"{comp}: quad {e_quad:.1f} vs lin {e_lin:.1f}"


def test_an_unknown_scheme_is_rejected(coords, isos):
    with pytest.raises(ValueError, match="Unknown scheme"):
        self_consistent_correction_table(
            h2o_hessian, coords, isos, lambda _t: coords, scheme="cubic", **_TABLE_KW)


def test_only_the_extrapolating_schemes_are_distance_gated(coords, isos, base_table):
    """The gate contains truncation error, so "direct" is not subject to it.

    Measured across 9 molecules, "direct"'s outcome correlates with
    displacement at -0.17 -- distance simply does not predict when it fails,
    and it was the one scheme that came through a 324 mA move intact. Gating it
    on distance would be gating on the wrong variable. "linear" and
    "quadratic" extrapolate and stay gated.
    """
    far = coords + 200.0 * _stretch_direction(coords)
    for scheme in ("linear", "quadratic"):
        table, info = self_consistent_correction_table(
            h2o_hessian, coords, isos, lambda _t: far, scheme=scheme, **_TABLE_KW)
        assert info["skipped_beyond_ang"] is not None, scheme
        for comp in "ABC":
            assert (float(table["H2-16O"][comp]["alpha_sum_mhz"])
                    == pytest.approx(
                        float(base_table["H2-16O"][comp]["alpha_sum_mhz"]))), scheme

    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: far, scheme="direct", **_TABLE_KW)
    assert info["skipped_beyond_ang"] is None
    assert info["max_distance_ang"] == float("inf")


def test_direct_is_the_default_scheme(coords, isos):
    """Set from the measured medians, so a silent change should fail here."""
    _table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: coords, **_TABLE_KW)
    assert info["scheme"] == "direct"


def test_direct_can_still_be_given_an_explicit_gate(coords, isos, base_table):
    """Not gated by default is not the same as ungateable."""
    far = coords + 200.0 * _stretch_direction(coords)
    table, info = self_consistent_correction_table(
        h2o_hessian, coords, isos, lambda _t: far, scheme="direct",
        max_distance_ang=0.020, **_TABLE_KW)
    assert info["skipped_beyond_ang"] is not None
    for comp in "ABC":
        assert (float(table["H2-16O"][comp]["alpha_sum_mhz"])
                == pytest.approx(float(base_table["H2-16O"][comp]["alpha_sum_mhz"])))
