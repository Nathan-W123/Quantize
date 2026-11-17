"""The inertial-defect probe used as a relative weight rather than a floor.

_apply_defect_bias_floor measured its own failure precisely: the probe's
discriminating power is in the ratio of the residual defect to the raw one, and
flooring sigma at the absolute residual punished molecules whose corrections
demonstrably work. apply_defect_bias_scale is that diagnosis implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.centrifugal_distortion import rotational_constants_mhz
from backend.spectral.rovib_corrections import (
    CorrectedSpectralTarget,
    apply_defect_bias_scale,
    defect_improvement_ratio,
    resolve_corrections,
)
from dev.monofluoro_references import M_H, M_O16, WATER_R_E, WATER_THETA_E_DEG, _bent_xy

_COORDS = _bent_xy(WATER_R_E, WATER_THETA_E_DEG)
_MASSES = np.array([M_O16, M_H, M_H])
_EXACT = rotational_constants_mhz(_COORDS, _MASSES)


def _targets(b0, corrected, label="iso"):
    return [
        CorrectedSpectralTarget(
            isotopologue_label=label, component="ABC"[i], component_index=i,
            b0_mhz=float(b0[i]), sigma_exp_mhz=1.0, value_mhz=float(corrected[i]),
            sigma_mhz=10.0, correction_records=[],
        )
        for i in range(3)
    ]


def _isos(label="iso"):
    return [{"name": label, "masses": _MASSES}]


# ── The probe itself ─────────────────────────────────────────────────────────

def test_ratio_is_zero_when_the_correction_lands_exactly():
    """Corrected constants that match the structure have no residual defect."""
    b0 = _EXACT * np.array([1.02, 0.99, 1.03])
    assert defect_improvement_ratio(b0, _EXACT, _COORDS, _MASSES) \
        == pytest.approx(0.0, abs=1e-9)


def test_ratio_is_one_when_the_correction_does_nothing():
    b0 = _EXACT * np.array([1.02, 0.99, 1.03])
    assert defect_improvement_ratio(b0, b0, _COORDS, _MASSES) \
        == pytest.approx(1.0, rel=1e-9)


def test_ratio_exceeds_one_when_the_correction_makes_it_worse():
    b0 = _EXACT * np.array([1.001, 1.0, 1.0])
    worse = _EXACT * np.array([1.01, 1.0, 1.0])
    assert defect_improvement_ratio(b0, worse, _COORDS, _MASSES) > 5.0


def test_ratio_is_none_without_all_three_constants():
    assert defect_improvement_ratio(_EXACT[:2], _EXACT[:2], _COORDS, _MASSES) is None


def test_ratio_subtracts_the_structures_own_defect():
    """A non-planar structure has a defect of its own; it is not correction error."""
    bent = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.3], [-0.3, 0.9, -0.2]])
    exact = rotational_constants_mhz(bent, _MASSES)
    assert defect_improvement_ratio(exact * 1.001, exact, bent, _MASSES) \
        == pytest.approx(0.0, abs=1e-9)


# ── The sigma scaling ────────────────────────────────────────────────────────

def test_a_working_correction_is_left_completely_alone():
    """The control case that the absolute floor damaged by 20%."""
    b0 = _EXACT * np.array([1.02, 0.99, 1.03])
    good = _EXACT * np.array([1.001, 0.9995, 1.0015])
    out = apply_defect_bias_scale(_targets(b0, good), _isos(), _COORDS)
    assert all(t.sigma_mhz == 10.0 for t in out)


def test_a_correction_that_worsens_the_defect_is_widened_by_how_much():
    b0 = _EXACT * np.array([1.001, 1.0, 1.0])
    worse = _EXACT * np.array([1.004, 1.0, 1.0])
    ratio = defect_improvement_ratio(b0, worse, _COORDS, _MASSES)
    assert ratio > 1.0
    out = apply_defect_bias_scale(_targets(b0, worse), _isos(), _COORDS)
    assert all(t.sigma_mhz == pytest.approx(10.0 * ratio) for t in out)


def test_scaling_is_uniform_across_the_three_components():
    """The defect is one number about the three together; it cannot apportion blame.

    apply_planarity_constraint is the piece that decides *which* component to
    move, using sigma. This one only decides how much to trust the set.
    """
    b0 = _EXACT * np.array([1.001, 1.0, 1.0])
    worse = _EXACT * np.array([1.004, 1.0, 1.0])
    out = apply_defect_bias_scale(_targets(b0, worse), _isos(), _COORDS)
    assert len({round(t.sigma_mhz, 9) for t in out}) == 1


def test_species_are_scored_independently():
    b0 = _EXACT * np.array([1.001, 1.0, 1.0])
    good = _EXACT.copy()
    worse = _EXACT * np.array([1.004, 1.0, 1.0])
    targets = _targets(b0, good, "clean") + _targets(b0, worse, "dirty")
    isos = _isos("clean") + _isos("dirty")
    out = apply_defect_bias_scale(targets, isos, _COORDS)
    assert all(t.sigma_mhz == 10.0 for t in out if t.isotopologue_label == "clean")
    assert all(t.sigma_mhz > 10.0 for t in out if t.isotopologue_label == "dirty")


def test_a_partially_measured_species_is_skipped():
    b0 = _EXACT * np.array([1.001, 1.0, 1.0])
    worse = _EXACT * np.array([1.004, 1.0, 1.0])
    targets = _targets(b0, worse)[:2]
    out = apply_defect_bias_scale(targets, _isos(), _COORDS)
    assert all(t.sigma_mhz == 10.0 for t in out)


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_resolve_corrections_is_off_by_default_and_needs_coords():
    iso = {
        "name": "iso", "masses": _MASSES,
        "obs_constants": (_EXACT * np.array([1.001, 1.0, 1.0])).tolist(),
        "component_indices": [0, 1, 2],
        "sigma_constants": [10.0, 10.0, 10.0],
    }
    ctbl = {"iso": {c: {"delta_mhz": d, "sigma_mhz": 10.0, "source": "t"}
                    for c, d in zip("ABC", (4000.0, 0.0, 0.0))}}
    plain = resolve_corrections([iso], correction_table=ctbl, mode="hybrid_auto")
    scaled = resolve_corrections([iso], correction_table=ctbl, mode="hybrid_auto",
                                 defect_bias_scale=True, coords_ang=_COORDS)
    no_coords = resolve_corrections([iso], correction_table=ctbl,
                                    mode="hybrid_auto", defect_bias_scale=True)
    assert [t.sigma_mhz for t in scaled] > [t.sigma_mhz for t in plain]
    assert [t.sigma_mhz for t in no_coords] == [t.sigma_mhz for t in plain]


def test_the_optimizer_reads_the_probe_at_its_prior_target_not_its_start(monkeypatch):
    """The probe subtracts the structure's own defect, so the geometry it is
    read at sets the weights. MolecularOptimizer read it at the start
    geometry, so a restart from the answer re-weighted the data: the
    benchmark recipe with defect_bias_scale on moved chlorofluoromethane
    5.85 mA and acetyl fluoride 1.74 on a fresh restart (HF/6-31G); read at
    the prior target, 0.000 and 0.0005, first fits unchanged."""
    import contextlib
    import io

    import backend.spectral.rovib_corrections as RC
    import dev.analytic_water_backend  # noqa: F401
    from backend.quantize import MolecularOptimizer
    from test_bootstrap_targets import B0_H2O, MASSES_H2O, _coords
    seen = []
    real = RC.resolve_corrections

    def spy(*a, **k):
        seen.append(np.asarray(k.get("coords_ang"), dtype=float).copy())
        return real(*a, **k)

    monkeypatch.setattr(RC, "resolve_corrections", spy)
    import backend.quantize as Q
    monkeypatch.setattr(Q, "resolve_corrections", spy, raising=False)
    target = _coords()
    start = target + np.array([[0.0, 0.0, 0.0], [0.03, 0.01, 0.0], [-0.03, 0.01, 0.0]])
    isos = [{"name": "w", "masses": MASSES_H2O, "obs_constants": list(B0_H2O),
             "sigma_constants": [0.2] * 3}]
    table = {"w": {c: {"alpha_sum_mhz": 1.0, "sigma_mhz": 0.1} for c in "ABC"}}
    with contextlib.redirect_stdout(io.StringIO()):
        MolecularOptimizer(elems=["O", "H", "H"], coords=start, isotopologues=isos,
                           quantum_backend="analytic_water", use_autoconfig=False,
                           coordinate_mode="cartesian", correction_table=table,
                           defect_bias_scale=True, prior_target_coords=target)
    assert seen, "premise: the corrections were resolved"
    np.testing.assert_array_equal(seen[-1], target)


def _probe_geometry(monkeypatch, *, mode, target, start):
    """The coords_ang MolecularOptimizer hands resolve_corrections."""
    import contextlib
    import io

    import backend.quantize as Q
    import dev.analytic_water_backend  # noqa: F401
    from backend.quantize import MolecularOptimizer
    from test_bootstrap_targets import B0_H2O, MASSES_H2O
    seen = []
    real = Q.resolve_corrections

    def spy(*a, **k):
        seen.append(np.asarray(k.get("coords_ang"), dtype=float).copy())
        return real(*a, **k)

    monkeypatch.setattr(Q, "resolve_corrections", spy)
    isos = [{"name": "w", "masses": MASSES_H2O, "obs_constants": list(B0_H2O),
             "sigma_constants": [0.2] * 3}]
    table = {"w": {c: {"alpha_sum_mhz": 1.0, "sigma_mhz": 0.1} for c in "ABC"}}
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(
            elems=["O", "H", "H"], coords=start, isotopologues=isos,
            quantum_backend="analytic_water", use_autoconfig=False,
            coordinate_mode=mode, correction_table=table,
            defect_bias_scale=True, prior_target_coords=target)
    assert seen, "premise: the corrections were resolved"
    return opt, seen[-1]


def test_internal_mode_reads_the_probe_where_it_was(monkeypatch):
    """Internal mode ignores the target (the prior is not centred on it), so
    the probe must not move to it either: it stays at the start geometry, as
    the commit that moved it said and as it did before."""
    from test_bootstrap_targets import _coords
    target = _coords()
    start = target + np.array([[0.0, 0.0, 0.0], [0.03, 0.01, 0.0], [-0.03, 0.01, 0.0]])
    opt, probe = _probe_geometry(monkeypatch, mode="internal", target=target, start=start)
    assert not opt._target_prior_active(), "premise: the target prior is off"
    np.testing.assert_array_equal(probe, start)


def test_a_flat_target_is_read_as_the_attribute_reads_it(monkeypatch):
    """The target attribute reshapes a flat (3N,) target, so a flat target is
    a supported form; with defect_bias_scale on it raised in the probe."""
    from test_bootstrap_targets import _coords
    target = _coords()
    start = target + np.array([[0.0, 0.0, 0.0], [0.03, 0.01, 0.0], [-0.03, 0.01, 0.0]])
    opt, probe = _probe_geometry(monkeypatch, mode="cartesian", target=target.ravel(),
                                 start=start)
    assert opt._target_prior_active()
    np.testing.assert_array_equal(probe, target)
