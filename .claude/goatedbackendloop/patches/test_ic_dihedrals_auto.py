"""Internal mode adds dihedrals when bonds and angles miss a direction.

Internal mode steps in bonds and angles. A torsion changes neither, so a fit
in them cannot turn one: acetyl fluoride and fluoroethane (HF/6-31G) stopped
0.90 and 0.16 posterior widths from their own optimum, almost all of it along
the missing methyl torsion. ic_use_dihedrals="auto" (the default) adds
dihedrals exactly when bonds and angles do not reach every direction the fit
can move, decided once at the start geometry (decisions D-99).

The fits here are hydrogen peroxide on a valence force field (bonds, angles
and a torsion), so no quantum chemistry runs.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

import backend.quantize as Q  # noqa: E402
import backend.spectral.spectral as SP  # noqa: E402
from backend.base_backend import GradientResult, HessianResult, QuantumBackend  # noqa: E402
from backend.registry import register_backend  # noqa: E402

_BOHR = 0.529177210903
_M = dict(H=1.00782503207, D=2.01410177785, O=15.99491461956)


def _hooh(r_oo=1.452, r_oh=0.965, ang=100.0, dih=112.0):
    """O1 O2 H3 H4: H3 on O1, H4 on O2, dihedral H3-O1-O2-H4."""
    a, d = np.radians(ang), np.radians(dih)
    o1, o2 = np.zeros(3), np.array([0.0, 0.0, r_oo])
    h3 = o1 + r_oh * np.array([np.sin(a), 0.0, np.cos(a)])
    h4 = o2 + r_oh * np.array([np.sin(a) * np.cos(d), np.sin(a) * np.sin(d), -np.cos(a)])
    return np.array([o1, o2, h3, h4])


def _dihedral(x):
    b0, b1, b2 = x[0] - x[2], x[1] - x[0], x[3] - x[1]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    return np.degrees(np.arctan2(m @ n2, n1 @ n2))


def _angle(a, b, c):
    u, v = a - b, c - b
    return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))


# Stretches, bends (Hartree/A^2, Hartree/rad^2) and a torsion (Hartree), with
# a minimum away from the data's geometry so the two halves disagree.
_TH = dict(r_oo=1.40, r_oh=0.95, ang=np.radians(101.0), dih=np.radians(120.0))


def _energy(x):
    x = np.asarray(x, float).reshape(4, 3)
    e = 0.5 * 1.8 * (np.linalg.norm(x[1] - x[0]) - _TH["r_oo"]) ** 2
    e += 0.5 * 3.5 * (np.linalg.norm(x[2] - x[0]) - _TH["r_oh"]) ** 2
    e += 0.5 * 3.5 * (np.linalg.norm(x[3] - x[1]) - _TH["r_oh"]) ** 2
    e += 0.5 * 0.25 * (_angle(x[2], x[0], x[1]) - _TH["ang"]) ** 2
    e += 0.5 * 0.25 * (_angle(x[0], x[1], x[3]) - _TH["ang"]) ** 2
    e += 0.004 * (1.0 - np.cos(np.radians(_dihedral(x)) - _TH["dih"]))
    return e


def _grad(x, h=1e-5):
    f = np.asarray(x, float).ravel()
    g = np.zeros(f.size)
    for i in range(f.size):
        p, m = f.copy(), f.copy()
        p[i] += h
        m[i] -= h
        g[i] = (_energy(p) - _energy(m)) / (2 * h)
    return g


def _hess(x, h=1e-4):
    f = np.asarray(x, float).ravel()
    H = np.zeros((f.size, f.size))
    for i in range(f.size):
        p, m = f.copy(), f.copy()
        p[i] += h
        m[i] -= h
        H[:, i] = (_grad(p) - _grad(m)) / (2 * h)
    return 0.5 * (H + H.T)


@register_backend
class _PeroxideField(QuantumBackend):
    name = "test_peroxide_field"
    supports_parallel = True

    def __init__(self, elems=None, **_kwargs):
        self.elems = list(elems or [])

    def run_hessian(self, coords_ang):
        return HessianResult(energy=_energy(coords_ang), gradient_bohr=_grad(coords_ang) * _BOHR,
                             hessian_bohr=_hess(coords_ang) * _BOHR ** 2)

    def run_gradient(self, coords_ang):
        return GradientResult(energy=_energy(coords_ang), gradient_bohr=_grad(coords_ang) * _BOHR)


_TRUE = _hooh(dih=115.0)


def _isotopologues():
    out = []
    for name, m in (("HOOH", "OOHH"), ("DOOD", "OODD"), ("HOOD", "OOHD")):
        masses = [_M[k] for k in m]
        abc = SP._rotational_constants(_TRUE, np.array(masses))
        out.append(dict(name=name, masses=masses, obs_constants=[float(v) for v in abc],
                        sigma_constants=[float(v) * 1e-4 for v in abc],
                        alpha_constants=[0.0, 0.0, 0.0], component_indices=[0, 1, 2]))
    return out


def _optimizer(dihedrals="auto", start=None, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        opt = Q.MolecularOptimizer(
            elems=["O", "O", "H", "H"], coords=_hooh(dih=100.0) if start is None else start,
            isotopologues=_isotopologues(), quantum_backend="test_peroxide_field",
            use_autoconfig=False, max_iter=120, hess_recalc_every=10, coordinate_mode="internal",
            symmetry=None, ic_use_dihedrals=dihedrals, **kw)
    return opt, out.getvalue()


def test_a_torsion_without_a_coordinate_gets_dihedrals():
    opt, printed = _optimizer()
    assert opt._ic_use_dihedrals is True
    assert "reach 5 of the 6 directions" in opt._ic_dihedrals_reason
    assert "dihedrals added" in printed


@pytest.mark.parametrize("stated", [False, True])
def test_a_stated_choice_is_kept(stated):
    opt, printed = _optimizer(dihedrals=stated)
    assert opt._ic_use_dihedrals is stated
    assert opt._ic_dihedrals_reason is None and "dihedrals added" not in printed


def test_a_molecule_bonds_and_angles_span_keeps_them_alone():
    import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)
    from test_cd_rows_sign import _isos, _x0
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(), isotopologues=_isos(None),
                                   quantum_backend="analytic_water", use_autoconfig=False,
                                   coordinate_mode="internal", ic_use_dihedrals="auto")
    assert opt._ic_use_dihedrals is False and opt._ic_dihedrals_reason is None


def test_the_cartesian_mode_is_not_touched():
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=["O", "O", "H", "H"], coords=_hooh(dih=100.0),
                                   isotopologues=_isotopologues(), quantum_backend="test_peroxide_field",
                                   use_autoconfig=False, coordinate_mode="cartesian", symmetry=None)
    assert opt._ic_dihedrals_reason is None


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="ic_use_dihedrals"):
        _optimizer(dihedrals="sometimes")


def _fit(dihedrals):
    opt, _ = _optimizer(dihedrals=dihedrals)
    with contextlib.redirect_stdout(io.StringIO()):
        x = opt.run()
    return opt, np.asarray(x, float)


def test_with_dihedrals_the_fit_turns_the_torsion_and_reaches_its_optimum():
    opt, x = _fit("auto")
    stuck, xs = _fit(False)
    # Premise: without them the torsion has no coordinate of its own -- it
    # drifts only as a side effect of the angle steps -- and the fit stops far
    # from its optimum. Measured: dihedral 102.8 deg (start 100, data 115),
    # distance_to_mode 20.8 widths.
    assert abs(abs(_dihedral(xs)) - 115.0) > 5.0, _dihedral(xs)
    assert stuck.distance_to_mode() > 1.0
    # With them it turns to the data's 115 deg and stops at its optimum.
    # Measured: 115.012 deg, distance_to_mode 6.6e-7.
    assert abs(abs(_dihedral(x)) - 115.0) < 0.1, _dihedral(x)
    assert opt.distance_to_mode() < 1e-4
    post = opt.internal_coordinate_posterior()
    assert post["use_dihedrals"] is True and "reach 5 of the 6" in post["dihedrals_reason"]
    assert any(k == "dihedral" for k in post["kinds"])
