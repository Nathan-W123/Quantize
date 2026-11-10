"""The correction's uncertainty must widen where its calibration does not hold.

`nm_sigma_fraction` is 0.15, a B3LYP-literature VPT2 scale. Literature VPT2
benchmarks are built from well-behaved closed-shell molecules, so that scale
describes the error for that kind of molecule -- not for one where a single
determinant is the wrong starting point.

Measured by asking whether the stated sigma covers the spread the correction
shows when the force field changes (HF at 6-31G, cc-pVDZ, cc-pVTZ), as the
margin sigma/spread per rotational constant:

                 max|t2|      A       B       C
    water         0.047    1.28x   3.20x   4.76x    covered
    ozone         0.186    0.84x   0.44x   0.37x    NOT covered

Ozone fails on every constant and on all five of its isotopologues, by up to
2.7x, and its A correction changes SIGN across those bases (+113.5, -749.3,
-892.9 MHz) -- at the cheapest level it is not even directionally determined.
Water is covered everywhere. The multireference probe separates exactly those
two cases, which is what makes the widening targetable rather than a blanket
inflation.

These tests pin the mechanism, not the quantum chemistry: the measurement
above needs nine real Hessians and does not belong in a unit suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.spectral.harmonic_alpha import (  # noqa: E402
    MULTIREFERENCE_SIGMA_SCALE,
    build_correction_table_from_hessian,
)

#: Worst margin measured on ozone; the scale has to beat its reciprocal.
OZONE_WORST_MARGIN = 0.37
#: Tightest margin measured on water, which must stay covered.
WATER_WORST_MARGIN = 1.28


def _water_inputs():
    """A real Hessian is not needed to test the plumbing, but a sane one is."""
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.9578, 0.0, 0.0],
                       [-0.2399, 0.9270, 0.0]])
    masses = [15.9949146196, 1.00782503207, 1.00782503207]
    # A positive-definite Hessian shaped like a bent triatomic's, in Hartree/Bohr^2.
    rng = np.random.default_rng(7)
    a = rng.normal(size=(9, 9))
    hess = a @ a.T + 9.0 * np.eye(9)
    isos = [{"name": "H2-16O", "masses": masses,
             "obs_constants": [835712.5, 435059.0, 278357.3],
             "component_indices": [0, 1, 2],
             "sigma_constants": [0.025] * 3}]
    return hess, coords, isos


def _sigmas(table):
    return {(iso, comp): float(e["sigma_mhz"])
            for iso, comps in table.items() for comp, e in comps.items()}


def _alphas(table):
    return {(iso, comp): float(e["alpha_sum_mhz"])
            for iso, comps in table.items() for comp, e in comps.items()}


@pytest.fixture(scope="module")
def tables():
    hess, coords, isos = _water_inputs()
    plain, _i1 = build_correction_table_from_hessian(
        hess, coords, [dict(d) for d in isos])
    scaled, _i2 = build_correction_table_from_hessian(
        hess, coords, [dict(d) for d in isos],
        sigma_scale=MULTIREFERENCE_SIGMA_SCALE)
    return plain, scaled


def test_the_scale_covers_the_worst_measured_shortfall():
    """3.0 has to beat 1/0.37 = 2.7, or it does not fix the case it exists for."""
    assert MULTIREFERENCE_SIGMA_SCALE >= 1.0 / OZONE_WORST_MARGIN
    # And not absurdly more than needed.
    assert MULTIREFERENCE_SIGMA_SCALE <= 2.0 / OZONE_WORST_MARGIN


def test_water_was_already_covered_so_the_default_must_not_move():
    """The widening is targeted. An unflagged molecule keeps its calibration."""
    assert WATER_WORST_MARGIN > 1.0
    hess, coords, isos = _water_inputs()
    _t, _i = build_correction_table_from_hessian(
        hess, coords, [dict(d) for d in isos])
    # Default is 1.0: no silent global inflation.
    import inspect
    sig = inspect.signature(build_correction_table_from_hessian)
    assert sig.parameters["sigma_scale"].default == 1.0


def test_it_scales_sigma(tables):
    plain, scaled = tables
    sp, ss = _sigmas(plain), _sigmas(scaled)
    assert sp and set(sp) == set(ss)
    for key in sp:
        assert ss[key] == pytest.approx(
            sp[key] * MULTIREFERENCE_SIGMA_SCALE, rel=1e-12), key


def test_it_does_not_touch_the_correction_itself(tables):
    """A trust flag must widen the error bar, never move the number."""
    plain, scaled = tables
    ap, as_ = _alphas(plain), _alphas(scaled)
    assert ap and set(ap) == set(as_)
    for key in ap:
        assert as_[key] == ap[key], key


def test_the_widening_is_recorded_in_the_note(tables):
    _plain, scaled = tables
    notes = " ".join(e.get("notes", "")
                     for comps in scaled.values() for e in comps.values())
    assert "widened" in notes.lower()
    assert "force-field trust" in notes.lower()


def test_a_narrowing_scale_is_refused():
    """Below 1.0 it would narrow a literature-calibrated sigma for free."""
    hess, coords, isos = _water_inputs()
    for bad in (0.5, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            build_correction_table_from_hessian(
                hess, coords, [dict(d) for d in isos], sigma_scale=bad)


def test_unity_is_a_no_op(tables):
    plain, _scaled = tables
    hess, coords, isos = _water_inputs()
    again, _i = build_correction_table_from_hessian(
        hess, coords, [dict(d) for d in isos], sigma_scale=1.0)
    assert _sigmas(again) == _sigmas(plain)
    notes = " ".join(e.get("notes", "")
                     for comps in again.values() for e in comps.values())
    assert "widened" not in notes.lower()
