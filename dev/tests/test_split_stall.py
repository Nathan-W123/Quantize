"""A split fit that stalls short of its own fixed point says so.

The split step fits the data on the range of its stacked Jacobian and lets
the quantum surface alone decide the null space, so its fixed point has no
energy gradient left there. Its rank is cut at sv_min_abs = 1/sigma_x, which
can put directions the data still weigh on the energy's side; the acceptance
test still scores the data there, refuses the energy's moves, and the fit
stops with that gradient unspent. On the benchmark molecules (HF/6-31G) one
Newton step to the null-space minimum moved an atom 2.9-16.6 mA, the answer
moved up to 1.47 of its widths with the trust radius, and no covariance
formula reproduced the refits' spread (T-48, decisions D-115). The fits here
are analytic water.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

from test_bootstrap_targets import _split_water  # noqa: E402

_STALL = "short of the minimum of its null-space energy"


def test_a_split_fit_that_stalls_says_so():
    # One constant at sigma_x 0.5 mA: the cut puts every direction on the
    # energy's side, the step's moves are refused, and the fit stops.
    opt = _split_water((2,), sigma=0.0005)
    assert not any(h["accepted"] for h in opt.history[-3:]), "premise: its last steps are refused"
    assert opt.split_coupling() <= opt.SPLIT_COUPLING_TOL, "premise: the coupling cannot see it"
    widths, atom = opt.split_stall()
    # Measured: 30.9 widths, 10.0 mA.
    assert widths > 10.0 and 5e-3 < atom < 2e-2, (widths, atom)
    note = opt.split_inconsistency()
    assert note and _STALL in note and "step schedule" in note
    assert any(_STALL in w for w in opt.internal_coordinate_posterior()["warnings"])


def test_a_coupled_split_fit_that_stalls_names_the_stall_not_the_coupling():
    opt = _split_water((1, 2), sigma=0.0005)
    assert opt.split_coupling() > 0.5, "premise: a coupling it would otherwise report"
    widths, _atom = opt.split_stall()
    assert widths > 1.0                      # measured 2.27
    note = opt.split_inconsistency()
    assert _STALL in note and "null response" not in note


def test_the_newton_step_reaches_the_null_space_minimum():
    opt = _split_water((2,), sigma=0.0005)
    # Away from the quantum point (the fit refreshed it at its answer), so
    # the model's gradient there is not its reference gradient.
    y = np.asarray(opt.coords, dtype=float) + np.array([[0.0, 0.0, 0.004], [0.0, 0.003, -0.002],
                                                        [0.0, -0.003, -0.002]])
    opt.coords = y
    assert np.abs(y - opt._orca_ref_coords).max() > 1e-3, "premise: off the quantum point"
    dn, B = opt._split_null_newton(y)
    Pn = opt._split_energy_projector()
    g, H = opt._projected_reference_terms()

    def g_null(z):
        gz = np.asarray(g) + np.asarray(H) @ (z - opt._orca_ref_coords).ravel()
        return np.linalg.norm(B.T @ Pn @ gz)

    assert g_null(y) > 1e-4, "premise: a gradient left in the null space"
    assert g_null(y + (B @ dn).reshape(-1, 3)) < 1e-8 * max(1.0, g_null(y))


def test_a_split_fit_at_its_fixed_point_reads_no_stall():
    # B and C: the data reach every physical direction (no null space).
    full = _split_water((1, 2))
    assert full.split_stall() == (0.0, 0.0)
    assert full.split_inconsistency() is None
    # One constant at the calibrated width: 0.017 widths (0.27 mA), inside
    # the limit, so the coupling note stays.
    one = _split_water((2,))
    widths, _ = one.split_stall()
    assert widths < one.SPLIT_STALL_LIMIT
    assert "null response" in one.split_inconsistency()


def test_off_the_split_objective_there_is_no_stall():
    from test_bootstrap_targets import _coords
    import contextlib, io
    import dev.analytic_water_backend  # noqa: F401
    from backend.quantize import MolecularOptimizer
    from test_bootstrap_targets import _split_iso, MASSES_H2O, B0_H2O
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(),
                                 isotopologues=[_split_iso("H2-16O", MASSES_H2O, B0_H2O, (2,))],
                                 quantum_backend="analytic_water", use_autoconfig=False, max_iter=50,
                                 auto_systematic_sigma=False, objective_mode="joint")
        opt.run()
    assert opt.split_stall() is None
