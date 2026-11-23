"""Internal mode adds dihedrals when bonds and angles miss a direction.

Internal mode steps in bonds and angles. A torsion changes neither, so a fit
in them cannot turn one: acetyl fluoride and fluoroethane (HF/6-31G) stopped
0.19 and 0.16 posterior widths from their own optimum, almost all of it along
the missing methyl torsion. ic_use_dihedrals="auto" (opt-in; the default
stays False) adds dihedrals exactly when bonds and angles do not reach every
direction the fit can move, decided once at the start geometry (decisions
D-99, D-100).

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


def _isotopologues(true=None):
    true = _TRUE if true is None else true
    out = []
    for name, m in (("HOOH", "OOHH"), ("DOOD", "OODD"), ("HOOD", "OOHD")):
        masses = [_M[k] for k in m]
        abc = SP._rotational_constants(true, np.array(masses))
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
    assert "with dihedrals 6" in opt._ic_dihedrals_reason
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


def test_the_default_is_unchanged():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        opt = Q.MolecularOptimizer(elems=["O", "O", "H", "H"], coords=_hooh(dih=100.0),
                                   isotopologues=_isotopologues(), quantum_backend="test_peroxide_field",
                                   use_autoconfig=False, coordinate_mode="internal", symmetry=None)
    assert opt._ic_use_dihedrals is False and opt._ic_dihedrals_reason is None
    assert "dihedrals added" not in out.getvalue()


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


@pytest.mark.parametrize("value, ok", [(True, True), (False, True), ("auto", True), ("AUTO", True),
                                       ("sometimes", False), (1, False)])
def test_the_config_accepts_true_false_or_auto(value, ok):
    from runner.usability import ConfigError, _validate_run_controls
    cfg = {"internal_coordinates": {"use_dihedrals": value}}
    if ok:
        _validate_run_controls(cfg, ["O", "O", "H", "H"])
    else:
        with pytest.raises(ConfigError, match="use_dihedrals"):
            _validate_run_controls(cfg, ["O", "O", "H", "H"])


def test_the_config_default_is_unchanged():
    from runner.config_schema import normalize_config
    assert normalize_config({})["internal_coordinates"]["use_dihedrals"] is False


# --- T-41 verifier follow-ups -------------------------------------------------

def _single(x, elems, masses):
    abc = SP._rotational_constants(x, np.array(masses))
    return [dict(name="a", masses=masses, obs_constants=[float(abc[1])], sigma_constants=[1.0],
                 alpha_constants=[0.0], component_indices=[1])]


def _spectral_only(x, elems, masses):
    from backend.internal.internal_fit import InternalCoordinateSet
    with contextlib.redirect_stdout(io.StringIO()) as out:
        opt = Q.MolecularOptimizer(elems=elems, coords=x, isotopologues=_single(x, elems, masses),
                                   spectral_only=True, quantum_backend="none", coordinate_mode="internal",
                                   ic_use_dihedrals="auto", symmetry=None, use_autoconfig=False)
    basis = opt.physical_basis(coords=x)
    reach = lambda d: int((np.linalg.svd(np.asarray(InternalCoordinateSet(x, elems, d).active_B_matrix(x))
                                         @ basis, compute_uv=False) ** 2 > 1e-6).sum())
    return opt, out.getvalue(), basis.shape[1], reach(False)


def test_dihedrals_that_reach_nothing_more_are_not_added():
    # A linear chain's dihedral is undefined (its row is zero) and a molecule
    # with no four-atom chain has none: adding them reached nothing, yet the
    # note said they were added and the posterior quoted "dihedral 0 +- 0".
    lin = np.array([[0, 0, -1.06], [0, 0, 0], [0, 0, 1.20], [0, 0, 2.48]], float)
    opt, printed, n, r0 = _spectral_only(lin, ["H", "C", "C", "F"], [1.0078, 12.0, 12.0, 18.998])
    assert r0 < n, "premise: bonds and angles miss directions of a linear molecule"
    assert opt._ic_use_dihedrals is False and "dihedrals added" not in printed
    planar = np.array([[0, 0, 0], [0, 0, 1.18], [1.0, 0, -0.6], [-1.1, 0, -0.7]], float)
    opt, printed, n, r0 = _spectral_only(planar, ["C", "O", "H", "F"], [12.0, 15.995, 1.0078, 18.998])
    assert r0 < n, "premise: bonds and angles cannot move a planar molecule out of plane"
    assert opt._ic_use_dihedrals is False and "dihedrals added" not in printed


def test_a_barely_reached_direction_counts_as_missed():
    # Fluorobenzene from the benchmark's displaced start, no symmetry imposed:
    # bonds and angles reach all 30 directions above a relative 1e-6, but the
    # weakest so faintly that the damped step realises a few % of a move along
    # it, and the fit stalled 0.056 widths short (T-41 verifier). It counts as
    # missed, and dihedrals (which reach it) are added.
    from backend.internal.internal_fit import InternalCoordinateSet
    from dev.monofluoro_references import FLUOROBENZENE
    from scripts.monofluoro_benchmark import build_isotopologues, start_geometry
    x = start_geometry(FLUOROBENZENE)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=list(FLUOROBENZENE.elems), coords=x,
                                   isotopologues=build_isotopologues(FLUOROBENZENE, None),
                                   spectral_only=True, quantum_backend="none", coordinate_mode="internal",
                                   ic_use_dihedrals="auto", symmetry=None, use_autoconfig=False)
    basis = opt.physical_basis(coords=x)
    sv = np.linalg.svd(np.asarray(InternalCoordinateSet(x, opt.elems, False).active_B_matrix(x)) @ basis,
                       compute_uv=False)
    assert (sv > 1e-6 * sv[0]).sum() == basis.shape[1], "premise: reached above a relative 1e-6"
    assert opt._ic_use_dihedrals is True, opt._ic_dihedrals_reason


def test_the_trial_prior_rows_wrap_dihedrals():
    from backend.internal.internal_fit import InternalCoordinateSet
    elems = ["O", "O", "H", "H"]
    x_ref, x_try = _hooh(dih=179.5), _hooh(dih=-179.5)
    cs = InternalCoordinateSet(x_ref, elems, True)
    q_ref = np.asarray(cs.active_values(x_ref), dtype=float)
    sig = np.ones(q_ref.size) * 0.2
    J = np.diag(1.0 / sig)
    r_ref = np.zeros(q_ref.size)
    r = Q.MolecularOptimizer._ic_prior_trial_rows((J, r_ref, q_ref, 1.0), cs, x_try)
    k = [ic.kind for ic in cs.active_coords()].index("dihedral")
    # A 1 deg step across 180, not a 359 deg one.
    assert abs(abs(r[k]) * sig[k] - np.radians(1.0)) < 1e-9, r[k] * sig[k]


def test_the_bootstrap_spread_of_a_dihedral_near_180_is_wrapped():
    from backend.uncertainty_v2 import BootstrapEngine
    from backend.uncertainty_v2.models import BootstrapConfig, BootstrapSampleResult
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(
            elems=["O", "O", "H", "H"], coords=_hooh(dih=179.0), isotopologues=_isotopologues(_hooh(dih=180.0)),
            quantum_backend="test_peroxide_field", use_autoconfig=False, max_iter=60, coordinate_mode="internal",
            symmetry=None, ic_use_dihedrals=True)
        opt.run()
    rng = np.random.default_rng(3)
    res = []
    for i in range(12):
        d = 180.0 + rng.normal(0.0, 0.5)
        d = (d + 180.0) % 360.0 - 180.0
        res.append(BootstrapSampleResult(idx=i, seed=i, success=True, coords=_hooh(dih=d), usable=True))
    with contextlib.redirect_stdout(io.StringIO()):
        ref = BootstrapEngine(opt, BootstrapConfig(n_samples=2)).analytic_reference(res)
    k = [n for n in ref["names"]].index(next(n for n in ref["names"] if n.startswith("dihedral")))
    # Premise: the draws straddle +-180.
    assert max(abs(_dihedral(r.coords)) for r in res) > 179.0 and min(_dihedral(r.coords) for r in res) < 0 < max(
        _dihedral(r.coords) for r in res)
    assert ref["bootstrap_std"][k] < 2.0, ref["bootstrap_std"][k]


def test_every_multistart_start_gets_the_same_set(monkeypatch):
    import backend.multistart as MS
    seen = []
    monkeypatch.setattr(MS, "_run_one_start", lambda p: seen.append(p["optimizer_kwargs"]["ic_use_dihedrals"]) or None)
    starts = [_hooh(dih=100.0), _hooh(dih=100.0) + 0.03, _hooh(dih=100.0) - 0.02]
    kw = dict(quantum_backend="test_peroxide_field", use_autoconfig=False, coordinate_mode="internal",
              symmetry=None, ic_use_dihedrals="auto")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            MS.run_multistart(starts, ["O", "O", "H", "H"], _isotopologues(), kw, max_workers=1)
        except Exception:
            pass  # the patched worker returns no result; only its inputs matter here
    assert seen == [True, True, True], seen


# --- A set that misses a direction is reported (T-56, D-128) -------------------
#
# With dihedrals off (the default) nothing is added, but a fit whose bonds and
# angles miss a direction says so: on the water dimer the gap to the mode (0.068
# to 4.0 widths over eight starts with no point group) lay along the two missing
# hydrogen-bond torsions, and the report said nothing below 2 widths.

def test_a_missed_direction_is_reported_when_dihedrals_are_off():
    opt, printed = _optimizer(dihedrals=False)
    assert opt._ic_use_dihedrals is False
    assert opt._ic_unreached == (5, 6, 6)
    assert "reach 5 of the 6 directions" in printed
    assert "use_dihedrals: auto adds dihedrals, which reach 6" in printed


def test_the_note_reaches_both_warning_lists_with_the_distance_to_the_mode():
    from backend.uncertainty_v2.posterior import laplace_approximation
    opt, x = _fit(False)
    d = opt.distance_to_mode()
    assert d > 1.0, "premise: the torsion was left short (measured 20.8 widths)"
    with contextlib.redirect_stdout(io.StringIO()):
        lists = {"posterior": opt.internal_coordinate_posterior().get("warnings", []),
                 "laplace": laplace_approximation(opt, coords=x).warnings}
    for where, ws in lists.items():
        notes = [w for w in ws if w.startswith("internal coordinates:")]
        assert len(notes) == 1, (where, ws)
        assert "reach 5 of the 6" in notes[0] and "use_dihedrals: auto" in notes[0], notes[0]
        assert f"stands {d:.2g} posterior widths" in notes[0], notes[0]


@pytest.mark.parametrize("stated", ["auto", True])
def test_no_note_when_the_fit_has_dihedrals(stated):
    opt, printed = _optimizer(dihedrals=stated)
    assert opt._ic_use_dihedrals is True
    assert opt._ic_unreached is None and opt.ic_reach_warnings() == []
    assert "does not reach" not in printed


def test_a_set_dihedrals_cannot_complete_names_no_dihedral_remedy():
    lin = np.array([[0, 0, -1.06], [0, 0, 0], [0, 0, 1.20], [0, 0, 2.48]], float)
    opt, printed, n, r0 = _spectral_only(lin, ["H", "C", "C", "F"], [1.0078, 12.0, 12.0, 18.998])
    assert r0 < n, "premise: bonds and angles miss directions of a linear molecule"
    assert opt._ic_unreached == (r0, r0, n)
    for text in [printed] + opt.ic_reach_warnings():
        assert "dihedrals reach no more of them" in text and "use_dihedrals: auto" not in text, text


def _water_dimer(seed):
    from test_ic_fragments import _dimer
    truth = _dimer()
    m_o, m_h, m_d = 15.9949146196, 1.00782503207, 2.0141017778
    isos = []
    for name, m in (("p", [m_o, m_h, m_h, m_o, m_h, m_h]), ("a", [m_o, m_d, m_d, m_o, m_h, m_h]),
                    ("b", [m_o, m_h, m_h, m_o, m_d, m_d]), ("ab", [m_o, m_d, m_d, m_o, m_d, m_d]),
                    ("h5", [m_o, m_h, m_h, m_o, m_d, m_h])):
        abc = SP._rotational_constants(truth, np.array(m))
        isos.append({"name": name, "masses": m, "obs_constants": [float(v) for v in abc],
                     "sigma_constants": [0.05] * 3, "component_indices": [0, 1, 2]})
    x = truth + np.random.default_rng(seed).normal(0.0, 0.02, truth.shape)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = Q.MolecularOptimizer(elems=["O", "H", "H", "O", "H", "H"], coords=x, isotopologues=isos,
                                   spectral_only=True, quantum_backend="none", use_autoconfig=False,
                                   coordinate_mode="internal")
    return opt, x


@pytest.mark.parametrize("seed, directions, unreached", [(4, 12, (10, 12, 12)), (2, 8, None)])
def test_the_water_dimer_is_reported_only_where_its_torsions_can_move(seed, directions, unreached):
    opt, x = _water_dimer(seed)
    # Premise: the start decides the point group -- none (12 directions, the
    # two hydrogen-bond torsions among them) or Cs, which forbids both.
    assert opt.physical_basis(coords=x).shape[1] == directions
    assert opt._ic_unreached == unreached
