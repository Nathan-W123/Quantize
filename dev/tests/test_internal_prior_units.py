"""The Cartesian internal prior's rows are in one unit each (T-76, backlog N29).

InternalPriorEngine holds bonds in Angstrom and angles and dihedrals in degrees:
values, targets and sigmas. Its Jacobian came from Wilson B, which is per
radian for angles and dihedrals, so those rows were 57.3x too small against
their own residuals. The step and the reported uncertainty saw the angle prior
that much weaker than the acceptance test did: on the nine benchmark molecules
(spectral-only Cartesian fits with the BALANCED/STRICT prior, starts displaced
0.03 A) that was 58 rejected steps, no uncertainty for 3 fits and angle widths
up to 64 deg; fixed, 3 rejections and widths of 0.3-2.1 deg.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

from backend.internal.internal_prior import InternalPriorEngine, _internal_values

from dev.monofluoro_references import HYDROGEN_PEROXIDE, MOLECULES, MOLECULES_SET2


def _fd_jacobian(engine, x, eps=1e-6):
    """-d(rw)/dx by central differences: the Jacobian the rows must carry."""
    flat = np.asarray(x, dtype=float).ravel()
    cols = []
    for k in range(flat.size):
        xp, xm = flat.copy(), flat.copy()
        xp[k] += eps
        xm[k] -= eps
        _, rp = engine.stacked(xp.reshape(x.shape), want_jacobian=False)
        _, rm = engine.stacked(xm.reshape(x.shape), want_jacobian=False)
        cols.append(-(rp - rm) / (2 * eps))
    return np.array(cols).T


@pytest.mark.parametrize("mol", [HYDROGEN_PEROXIDE, MOLECULES[0]], ids=lambda m: m.key)
def test_every_row_is_the_derivative_of_its_residual(mol):
    x = np.asarray(mol.geometry, dtype=float)
    rng = np.random.default_rng(3)
    eng = InternalPriorEngine(x + rng.normal(0, 0.02, x.shape), mol.elems,
                              use_dihedrals=True, sigma_bond=0.04,
                              sigma_angle_deg=2.0, sigma_dihedral_deg=15.0)
    kinds = {lbl.split()[0] for lbl in eng.labels}
    assert {"bond", "angle", "dihedral"} <= kinds          # premise: all three kinds
    Jw, _ = eng.stacked(x)
    fd = _fd_jacobian(eng, x)
    scale = np.max(np.abs(fd), axis=1, keepdims=True)
    assert np.max(np.abs(Jw - fd) / scale) < 1e-6
    # Before the fix the angle and dihedral rows were fd / 57.2958.
    ang = np.array([lbl.split()[0] != "bond" for lbl in eng.labels])
    ratio = np.linalg.norm(fd[ang], axis=1) / np.linalg.norm(Jw[ang] / (180 / np.pi), axis=1)
    assert np.allclose(ratio, 180 / np.pi, rtol=1e-6)


def test_dihedral_residuals_take_the_short_way_round():
    x = np.asarray(HYDROGEN_PEROXIDE.geometry, dtype=float)
    eng = InternalPriorEngine(x, HYDROGEN_PEROXIDE.elems, use_dihedrals=True,
                              sigma_dihedral_deg=10.0)
    i = next(k for k, lbl in enumerate(eng.labels) if lbl.startswith("dihedral"))
    q = _internal_values(x, [eng.labels[i]])[0]
    eng.prior_targets[i] = q + 358.0          # the same angle, one turn round, minus 2 deg
    _, rw = eng.stacked(x, want_jacobian=False)
    assert rw[i] == pytest.approx(-2.0 / 10.0, abs=1e-9)


def test_a_cartesian_spectral_only_fit_accepts_its_steps_and_reports_angle_widths():
    import backend.quantize as Q

    from scripts.monofluoro_benchmark import build_isotopologues, start_geometry

    mol = next(m for m in MOLECULES_SET2 if m.key == "formyl_fluoride")
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(
            elems=list(mol.elems), coords=start_geometry(mol),
            isotopologues=build_isotopologues(mol, None), spectral_only=True,
            quantum_backend="none", coordinate_mode="cartesian", use_autoconfig=False,
            max_iter=500, use_internal_priors=True, prior_weight=1.0,
            prior_auto_from_initial=True, prior_use_dihedrals=False,
            prior_sigma_bond=0.04, prior_sigma_angle_deg=2.0)
        opt.run()
        widths = opt.geometry_uncertainty()
    # Shipped: 9 of 11 steps rejected and no uncertainty at all.
    assert sum(not r["accepted"] for r in opt.history) == 0
    ang = [v for k, v in widths.items() if isinstance(v, float) and k.count("-") == 2]
    assert ang and max(ang) < 2.5                       # 0.62-2.08 deg; the prior sigma is 2
