"""Straight-line angles in the reported posterior and the Cartesian prior (N40).

A 180-degree angle -- fluoroacetylene's two, ketene's C=C=O -- cannot change
without breaking the point group the fit imposes, so none of the directions
the posterior is taken over moves it. Its width is zero by construction. It
used to come out as 0 or ~1e-12 from rounding and be labelled
'unidentifiable' or 'prior-dominated' depending on which (T-77 coverage;
T-83 measured both labels on both molecules at HF/6-31G). It is now labelled
'fixed by symmetry' at a width of exactly zero.

The Cartesian structural prior (InternalPriorEngine) evaluated each residual
at the current geometry but, for a label the current connectivity no longer
produced, took the Jacobian row from the start geometry; rows now come from
each label's own atoms at the current geometry.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.internal.internal_prior import InternalPriorEngine  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.quantum import wilson_B  # noqa: E402

_HCN = np.array([[0.0, 0.0, -1.06], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
_M = {"H": 1.00782503207, "D": 2.0141017778, "C": 12.0, "N": 14.0030740048}
_ISOS = [
    {"name": "HCN", "masses": [_M["H"], _M["C"], _M["N"]], "obs_constants": [44315.98],
     "component_indices": [1], "sigma_constants": [0.5]},
    {"name": "DCN", "masses": [_M["D"], _M["C"], _M["N"]], "obs_constants": [36207.46],
     "component_indices": [1], "sigma_constants": [0.5]},
]


def _posterior(symmetry):
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["H", "C", "N"], coords=_HCN, isotopologues=_ISOS,
                                 quantum_backend=None, spectral_only=True,
                                 use_autoconfig=False, max_iter=20, symmetry=symmetry)
        opt.run()
        post = opt.internal_coordinate_posterior()
    return opt, post


def test_a_linear_angle_under_its_point_group_is_fixed_by_symmetry():
    opt, post = _posterior("auto")
    assert opt.point_group == "Cinf_v"
    rows = {n: (k, s, lab, f) for n, k, s, lab, f in zip(
        post["names"], post["kinds"], post["std_err"], post["labels"], post["data_fraction"])}
    kind, sd, label, frac = rows["angle 1-2-3"]
    assert kind == "angle" and label == "fixed by symmetry"
    assert sd == 0.0 and np.isnan(frac)
    k = post["names"].index("angle 1-2-3")
    C = np.asarray(post["covariance"])
    assert np.all(C[k, :] == 0.0) and np.all(C[:, k] == 0.0)
    # The bonds are untouched: real widths and a data label.
    for name in ("bond 1-2", "bond 2-3"):
        _kind, sd, label, frac = rows[name]
        assert sd > 0.0 and label != "fixed by symmetry" and np.isfinite(frac)


def test_without_a_point_group_nothing_is_called_fixed():
    _opt, post = _posterior("C1")
    assert "fixed by symmetry" not in post["labels"]
    k = post["names"].index("angle 1-2-3")
    assert post["std_err"][k] > 0.0


def _water(r1=0.9578, r2=0.9578, deg=104.5):
    th = np.radians(deg)
    return np.array([[0.0, 0.0, 0.0],
                     [r1 * np.sin(th / 2), r1 * np.cos(th / 2), 0.0],
                     [-r2 * np.sin(th / 2), r2 * np.cos(th / 2), 0.0]])


def test_prior_rows_are_unchanged_where_the_connectivity_holds():
    x0 = _water()
    eng = InternalPriorEngine(x0, ["O", "H", "H"])
    x = _water(0.97, 0.95, 103.0)
    Jw, _ = eng.stacked(x)
    B, labels = wilson_B(x, ["O", "H", "H"])
    want = np.array([B[labels.index(lbl)] for lbl in eng.labels])
    assert np.array_equal(Jw, want * (eng._row_unit / eng.prior_sigmas)[:, None])


def test_prior_rows_follow_the_geometry_after_a_bond_breaks():
    # O-H at 1.25 A is past the covalent cutoff, so wilson_B at x no longer
    # lists 'bond 1-2' or the angle; the rows must still be derivatives at x.
    eng = InternalPriorEngine(_water(), ["O", "H", "H"])
    x = _water(1.25, 0.9578, 104.5)
    assert "bond 1-2" not in wilson_B(x, ["O", "H", "H"])[1]
    Jw, _ = eng.stacked(x)
    h = 1e-6
    fd = np.zeros_like(Jw)
    flat = x.ravel()
    for i in range(flat.size):
        e = np.zeros_like(flat); e[i] = h
        _, rp = eng.stacked((flat + e).reshape(x.shape), want_jacobian=False)
        _, rm = eng.stacked((flat - e).reshape(x.shape), want_jacobian=False)
        fd[:, i] = -(rp - rm) / (2 * h)          # rw = (target - q) / sigma
    assert np.allclose(Jw, fd, rtol=1e-6, atol=1e-6)


def test_the_report_says_what_fixed_by_symmetry_means():
    from runner.usability import generate_uncertainty_report_section
    row = {"name": "angle 1-2-3", "value": 180.0, "value_unit": "deg", "std_err": 0.0,
           "std_err_unit": "deg", "ci_lo": 180.0, "ci_hi": 180.0, "ci_unit": "deg",
           "chi2_scale": 1.0, "prior_dominance": "fixed by symmetry", "prior_sensitivity": ""}
    out = generate_uncertainty_report_section({"uncertainty_rows": [row]})
    assert "zero by construction, not a measurement" in out
    other = dict(row, prior_dominance="data-constrained", std_err=0.1)
    assert "zero by construction" not in generate_uncertainty_report_section(
        {"uncertainty_rows": [other]})


def test_a_dihedral_through_a_straight_line_is_undefined_not_fixed():
    # H-C#C-F: its one dihedral has no value at a linear geometry, and its
    # Wilson-B row is zero. That passed the symmetry test trivially and read
    # 'fixed by symmetry' even in a C1 fit (T-83 review).
    from backend.spectral.centrifugal_distortion import rotational_constants_mhz
    m = {"H": 1.00782503207, "D": 2.0141017778, "C": 12.0, "F": 18.99840322}
    elems = ["H", "C", "C", "F"]
    ref = np.array([[0.0, 0.0, z] for z in (-2.255, -1.198, 0.0, 1.279)])
    isos = []
    for name, sub in [("HCCF", {}), ("DCCF", {0: "D"})]:
        masses = [m[e] for e in elems]
        for i, el in sub.items():
            masses[i] = m[el]
        isos.append({"name": name, "masses": masses, "component_indices": [1],
                     "obs_constants": [float(rotational_constants_mhz(ref, masses)[1])],
                     "sigma_constants": [0.05]})
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=elems, coords=ref * 1.002, isotopologues=isos,
                                 quantum_backend=None, spectral_only=True,
                                 use_autoconfig=False, max_iter=0, symmetry="C1",
                                 ic_use_dihedrals=True)
        opt.run()
        post = opt.internal_coordinate_posterior()
    labels = dict(zip(post["names"], post["labels"]))
    assert labels["dihedral 1-2-3-4"] == "undefined at this geometry"
    assert "fixed by symmetry" not in post["labels"]
