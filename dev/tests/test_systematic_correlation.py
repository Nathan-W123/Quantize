"""Model error in the reported covariance: correlated, and counted once.

Two defects, one function (MolecularOptimizer._systematic_covariance):

1. The B0-versus-Be model error was propagated as INDEPENDENT per constant,
   diag(sigma_sys^2). Physically the gap of one constant type moves every
   isotopologue together -- they share one potential surface and so one
   force-field error, and the gap depends on the isotopologue only through
   smooth mass scaling. Measured correlation across isotopologues 0.94-1.00 on
   water, ozone, formyl fluoride and vinyl fluoride; across A, B and C there is
   no consistent sign. Independence averages a shared error down on the
   coordinates every species constrains alike (too narrow) and invents scatter
   in isotopic differences the shared error never produces (too wide). It is
   now R_ab = Q[comp_a, comp_b] * (1 if same isotopologue else rho), rho 0.99,
   Q = I.

2. Wherever the weighting sigma already held the model error -- the reference
   module's sigmas() (hypot of model error and quoted precision),
   systematic_in_weighting, an auto-filled sigma, a correction sigma -- the
   Laplace counted it once as independent noise and the systematic term added
   it again: the reported O-H width of the reference water fit was 1.36x the
   Laplace, nearly twice the variance. The term is now the sandwich
   difference A [D (Sigma_meas + S R S) D - I] A^T, which takes the copy in
   the weights out before the correlated one goes in.
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

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
import dev.monofluoro_references as REF  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = np.array([835840.29, 435351.72, 278138.70])
B0_D2O = np.array([462324.0, 218038.0, 145258.0])
FRAC = 0.024          # water's model error as a fraction of each constant
QUOTED = 0.2          # MHz


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([[0.0, 0.0, 0.0],
                     [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
                     [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0]])


def _isos(convention="measurement", comps=((0, 1, 2), (0, 1, 2))):
    out = []
    for (name, masses, b0), idx in zip((("H2-16O", MASSES_H2O, B0_H2O),
                                        ("D2-16O", MASSES_D2O, B0_D2O)), comps):
        idx = list(idx)
        b = b0[idx]
        d = {"name": name, "masses": masses, "obs_constants": list(b),
             "component_indices": idx, "sigma_constants": [QUOTED] * len(idx)}
        if convention != "none":
            d["sigma_systematic_constants"] = list(FRAC * b)
        if convention == "reference":
            d["sigma_constants"] = list(np.hypot(FRAC * b, QUOTED))
            d["sigma_constants_include_systematic"] = True
        out.append(d)
    return out


def _fit(convention="measurement", comps=((0, 1, 2), (0, 1, 2)), **kw):
    base = dict(elems=["O", "H", "H"], coords=_coords(),
                isotopologues=_isos(convention, comps),
                quantum_backend="analytic_water", use_autoconfig=False,
                max_iter=3, quantum_prior_sigma_ang=0.02,
                auto_systematic_sigma=False)
    base.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(**base)
        opt.run()
    return opt


def _pieces(opt):
    """Laplace covariance over the physical directions, J, row scale, and the
    systematic term, all at the fitted geometry."""
    x = opt.coords
    cov, _ = opt.laplace_approximation(coords=x, check_gauss_newton=False)
    cov, _B, _n = opt.physical_covariance(cov, coords=x)
    J, _r, scale = opt.spectral.stacked(x, return_row_scale=True)
    extra = opt._systematic_covariance(J, cov, coords=x, row_scale=scale)
    return cov, J, scale, extra


def _sys_rows(opt):
    return np.concatenate([np.asarray(iso["sigma_systematic_constants"], dtype=float)
                           for iso in opt.spectral.isotopologues])


def _widths(opt, cov):
    with contextlib.redirect_stdout(io.StringIO()):
        unc = opt.geometry_uncertainty()
    J, _r = opt.spectral.stacked(opt.coords)
    from backend.internal.internal_fit import InternalCoordinateSet
    cs = InternalCoordinateSet(opt.coords, opt.elems, opt._ic_use_dihedrals)
    B = np.asarray(cs.active_B_matrix(opt.coords), dtype=float)
    return unc, np.sqrt(np.diag(B @ cov @ B.T))


# ── The correlation model ────────────────────────────────────────────────────

def test_rho_zero_is_the_independent_model():
    """rho = 0 with the model error outside the weights is exactly the old
    A diag((sigma_sys / sigma_eff)^2) A^T."""
    opt = _fit(systematic_isotopologue_correlation=0.0)
    cov, J, scale, extra = _pieces(opt)
    t = scale * _sys_rows(opt)
    A = cov @ J.T
    np.testing.assert_allclose(extra, (A * t ** 2) @ A.T, rtol=1e-12,
                               atol=1e-14 * np.abs(extra).max())


def test_rho_one_is_one_shared_error_per_constant_type():
    """rho = 1, Q = I: one error per constant type, the same in every
    isotopologue -- each type's rows add before they are squared."""
    opt = _fit(systematic_isotopologue_correlation=1.0)
    cov, J, scale, extra = _pieces(opt)
    t = scale * _sys_rows(opt)
    A = cov @ J.T
    comp = np.concatenate([iso["component_indices"] for iso in opt.spectral.isotopologues])
    expect = np.zeros_like(cov)
    for k in range(3):
        v = (A[:, comp == k] * t[comp == k]).sum(axis=1)
        expect += np.outer(v, v)
    np.testing.assert_allclose(extra, expect, rtol=1e-10,
                               atol=1e-13 * np.abs(extra).max())


def test_components_do_not_cross_couple_by_default():
    """One species measures only B, the other only C: no two rows share a
    constant type, so with Q = I the correlated model IS the independent one.
    Coupling B and C (Q[B,C] = 1, the near-prolate case) changes it."""
    comps = ((1,), (2,))
    o_def = _fit(comps=comps)
    o_ind = _fit(comps=comps, systematic_isotopologue_correlation=0.0)
    Q = np.eye(3)
    Q[1, 2] = Q[2, 1] = 1.0
    o_bc = _fit(comps=comps, systematic_component_correlation=Q)
    e_def, e_ind, e_bc = (_pieces(o)[3] for o in (o_def, o_ind, o_bc))
    np.testing.assert_allclose(e_def, e_ind, rtol=1e-12, atol=1e-14 * np.abs(e_ind).max())
    assert not np.allclose(e_bc, e_ind, rtol=1e-3)


@pytest.mark.parametrize("rho", [-0.1, 1.1, float("nan")])
def test_isotopologue_correlation_is_validated(rho):
    with pytest.raises(ValueError, match="systematic_isotopologue_correlation"):
        _fit(systematic_isotopologue_correlation=rho)


@pytest.mark.parametrize("Q", [
    np.array([[1.0, 0.5, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]),   # not symmetric
    np.diag([1.0, 2.0, 1.0]),                                         # not unit diagonal
    np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]),  # not PSD
    np.eye(2),                                                        # not 3x3
])
def test_component_correlation_is_validated(Q):
    with pytest.raises(ValueError, match="systematic_component_correlation"):
        _fit(systematic_component_correlation=Q)


@pytest.mark.parametrize("convention,kw", [
    ("measurement", {}),
    ("reference", {}),
    ("measurement", {"systematic_in_weighting": True}),
    ("measurement", {"systematic_isotopologue_correlation": 1.0}),
])
def test_the_total_covariance_is_positive_semidefinite(convention, kw):
    """The term itself can be negative in places -- it removes the copy of the
    model error the weights held -- but the covariance it completes is
    C J^T D Sigma D J C + C H_prior C, positive semidefinite."""
    opt = _fit(convention, **kw)
    cov, _J, _s, extra = _pieces(opt)
    ev = np.linalg.eigvalsh(cov + extra)
    assert ev.min() >= -1e-10 * ev.max()


# ── Counted once ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("convention,kw", [
    ("reference", {}),
    ("measurement", {"systematic_in_weighting": True}),
])
def test_model_error_in_the_weights_is_counted_once(convention, kw):
    """With the model error already in the weighting sigma and rho = 0, the
    independent model is exactly what the Laplace already holds: only the
    measurement part the weighting floor hid comes back, (0.001 / 0.024)^2 of
    the variance. The old term added the whole model error again -- 1.364x
    the Laplace on the bonds, 1.400x on the angle."""
    opt = _fit(convention, systematic_isotopologue_correlation=0.0, **kw)
    cov, _J, _s, _extra = _pieces(opt)
    unc, lap = _widths(opt, cov)
    bonds = sorted(v for k, v in unc.items() if k.count("-") == 1)
    angle = [v for k, v in unc.items() if k.count("-") == 2][0]
    lap_b = sorted(lap[:2])
    for got, ref in zip(bonds, lap_b):
        assert 1.0 <= got / ref < 1.002
    assert 1.0 <= angle / np.degrees(lap[2]) < 1.002


def test_the_reference_flag_states_the_measurement_part():
    """sigmas() of the reference module is hypot(model error, quoted); the flag
    recovers the quoted part as what is measurement. To 1e-5: a 20 GHz sigma
    stored as a double holds its 0.2 MHz part to about 1e-6, and no
    subtraction can recover more than the input carries."""
    opt = _fit("reference")
    for iso in opt.spectral.isotopologues:
        np.testing.assert_allclose(iso["sigma_measurement_constants"], QUOTED, rtol=1e-5)


def test_without_the_flag_sigma_constants_is_measurement():
    opt = _fit("measurement")
    assert all(iso.get("sigma_measurement_constants") is None
               for iso in opt.spectral.isotopologues)


def test_an_auto_filled_sigma_is_all_model_error():
    """With sigma_constants absent and auto_systematic_sigma on, both come
    from one model estimate, so none of sigma_constants is measurement."""
    isos = _isos("none")
    for iso in isos:
        del iso["sigma_constants"]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(),
                                 isotopologues=isos, quantum_backend="analytic_water",
                                 use_autoconfig=False, max_iter=1,
                                 quantum_prior_sigma_ang=0.02, auto_systematic_sigma=True)
    for iso in opt.spectral.isotopologues:
        np.testing.assert_array_equal(iso["sigma_measurement_constants"], 0.0)
        np.testing.assert_allclose(iso["sigma_systematic_constants"], iso["sigma_constants"])


def test_a_vibrational_correction_replaces_the_gap_with_its_own_sigma():
    """After a correction removes the B0-versus-Be gap, the model error left
    is how well the correction is known -- not the gap it removed, which the
    old term went on charging in full."""
    from backend.spectral.rovib_corrections import (
        apply_corrections_to_isotopologues, resolve_corrections)
    mol = REF.FORMYL_FLUORIDE
    sp = mol.species[0]
    sig = np.asarray(sp.sigmas(), dtype=float)
    sys_sig = np.asarray(sp.sigmas_systematic(), dtype=float)
    quoted = np.sqrt(np.maximum(sig ** 2 - sys_sig ** 2, 0.0))
    iso = {"name": "p", "masses": sp.masses(mol.masses).tolist(),
           "obs_constants": list(sp.observed()), "sigma_constants": list(sig),
           "sigma_systematic_constants": list(sys_sig),
           "sigma_measurement_constants": list(quoted),
           "component_indices": list(sp.component_indices)}
    ctbl = {"p": {c: {"alpha_sum_mhz": 20.0, "sigma_mhz": 2.0,
                      "method": "VPT2_semidiag", "source": "harmonic_hessian"}
                  for c in ("A", "B", "C")}}
    targets = resolve_corrections([iso], correction_table=ctbl, mode="hybrid_auto")
    new = apply_corrections_to_isotopologues([iso], targets)[0]
    np.testing.assert_allclose(new["sigma_systematic_constants"], 2.0)
    np.testing.assert_allclose(new["sigma_measurement_constants"], quoted, rtol=1e-9)
    np.testing.assert_allclose(new["sigma_constants"], np.hypot(quoted, 2.0), rtol=1e-9)


def test_distortion_rows_do_not_drop_the_systematic():
    """Centrifugal-distortion rows follow the constants in J. A J with extra
    rows used to fail a size check and return no systematic term at all."""
    opt = _fit()
    cov, J, scale, extra = _pieces(opt)
    rng = np.random.default_rng(0)
    J_ext = np.vstack([J, rng.normal(size=(4, J.shape[1]))])
    got = opt._systematic_covariance(J_ext, cov, coords=opt.coords, row_scale=scale)
    assert np.abs(extra).max() > 0.0
    np.testing.assert_array_equal(got, extra)


def test_component_weights_enter_the_propagation():
    """stacked multiplies each row by its component weight, so a shift in a
    constant enters the residual scaled by it too. The old term divided by the
    effective sigma alone."""
    opt = _fit(component_weight_map={0: 0.5}, systematic_isotopologue_correlation=0.0)
    cov, J, scale, extra = _pieces(opt)
    seff = np.concatenate([opt.spectral._effective_sigma(i["sigma_constants"], iso=i)
                           for i in opt.spectral.isotopologues])
    comp = np.concatenate([i["component_indices"] for i in opt.spectral.isotopologues])
    np.testing.assert_allclose(scale, np.where(comp == 0, 0.5, 1.0) / seff, rtol=1e-12)
    A = cov @ J.T
    t = scale * _sys_rows(opt)
    np.testing.assert_allclose(extra, (A * t ** 2) @ A.T, rtol=1e-12,
                               atol=1e-14 * np.abs(extra).max())
    old = (A * (_sys_rows(opt) / seff) ** 2) @ A.T
    assert not np.allclose(extra, old, rtol=1e-3)


def test_the_correlation_changes_the_report_not_the_fit():
    a = _fit(systematic_isotopologue_correlation=0.0)
    b = _fit()
    np.testing.assert_array_equal(a.coords, b.coords)


def test_the_posterior_states_the_correlation_it_used():
    post = _fit().internal_coordinate_posterior()
    assert "systematic model error" in post["blocks"]
    assert post["systematic_isotopologue_correlation"] == pytest.approx(0.99)
    assert post["systematic_component_correlation"] is None
    none = _fit("none").internal_coordinate_posterior()
    assert "systematic model error" not in none["blocks"]
    assert none["systematic_isotopologue_correlation"] is None


# ── Chi-square rescaling keeps the split consistent ──────────────────────────

def test_rescaling_floors_only_where_sigma_carries_the_model_error():
    """Tightening a measurement-only sigma must not jump it to the model
    error: 0.2 MHz on water's A was 'tightened' to 2.4% of 835 GHz."""
    meas = _fit("measurement")
    meas.spectral.scale_sigma(0.5)
    for iso in meas.spectral.isotopologues:
        np.testing.assert_allclose(iso["sigma_constants"], 0.5 * QUOTED)
    ref = _fit("reference")
    ref.spectral.scale_sigma(0.5)
    for iso in ref.spectral.isotopologues:
        sys_sig = np.asarray(iso["sigma_systematic_constants"], dtype=float)
        np.testing.assert_allclose(iso["sigma_constants"], sys_sig)      # floored
        np.testing.assert_allclose(iso["sigma_measurement_constants"], 0.0, atol=1e-6)


# ── The physics: the real gap, and which model covers it ─────────────────────

_AMU_A2_TO_MHZ = 505379.0045


def _abc(X, masses):
    m = np.asarray(masses, dtype=float)
    X = X - (m[:, None] * X).sum(0) / m.sum()
    inertia = sum(mi * ((xi @ xi) * np.eye(3) - np.outer(xi, xi)) for mi, xi in zip(m, X))
    return _AMU_A2_TO_MHZ / np.sort(np.linalg.eigvalsh(inertia))


def test_ozone_real_gap_is_covered_by_the_correlated_model():
    """Ozone's five measured species against its published r_e (Tyuterev et
    al. 1999, +/-0.15 mA): the B0-versus-Be gap is the real model error, so a
    rigid-rotor fit to B0 lands off r_e by exactly the shift that error causes.

    Per-constant magnitudes are taken from the gap itself, so only the
    correlation model is under test. Independent per constant, the shift is
    z = 2.9 on the bond and 3.5 on the angle; correlated across isotopologues
    at 0.99, z = 1.3 and 1.6. The gap's own correlation across species is
    0.90, 0.99 and 0.99 for A, B and C.
    """
    mol = REF.OZONE
    rows = []
    for sp in mol.species:
        m = sp.masses(mol.masses)
        rows += [(sp.label, m, int(c), float(o))
                 for c, o in zip(sp.component_indices, sp.observed())]
    comp = np.array([c for _l, _m, c, _o in rows])
    iso = np.array([lab for lab, *_ in rows])
    obs = np.array([o for *_r, o in rows])

    def model(p):
        X = REF._bent_xy(p[0], p[1])
        return np.array([_abc(X, m)[c] for _l, m, c, _o in rows])

    pe = np.array([REF.OZONE_R_E_ANG, REF.OZONE_THETA_E_DEG])
    gap = (obs - model(pe)) / obs
    for k in range(3):
        g = gap[comp == k]
        exch = (g.sum() ** 2 - (g ** 2).sum()) / ((g.size - 1) * (g ** 2).sum())
        assert exch > 0.85, (k, exch)
    s = np.array([np.sqrt(np.mean(gap[comp == k] ** 2)) for k in range(3)])
    sig = s[comp] * obs
    h = np.array([1e-6, 1e-5])

    def jac(p):
        return np.column_stack([(model(p + h[k] * np.eye(2)[k]) - model(p - h[k] * np.eye(2)[k]))
                                / (2 * h[k]) for k in range(2)])

    p = pe.copy()
    for _ in range(20):
        p = p + np.linalg.lstsq(jac(p) / sig[:, None], (obs - model(p)) / sig, rcond=None)[0]
    J = jac(p)
    W = np.diag(1.0 / sig ** 2)
    F = np.linalg.inv(J.T @ W @ J)
    shift = p - pe

    def z(rho):
        R = np.where(comp[:, None] == comp[None, :],
                     np.where(iso[:, None] == iso[None, :], 1.0, rho), 0.0)
        C = F @ J.T @ W @ (np.outer(sig, sig) * R) @ W @ J @ F
        return np.abs(shift) / np.sqrt(np.diag(C))

    assert z(0.0).max() > 2.5
    assert z(0.99).max() < 2.0


def test_a_correction_of_unstated_quality_does_not_retire_the_gap():
    """No sigma on the correction: the gap it replaced stays the model error.
    Charging nothing would claim the correction is exact."""
    from backend.spectral.rovib_corrections import (
        apply_corrections_to_isotopologues, resolve_corrections)
    mol = REF.FORMYL_FLUORIDE
    sp = mol.species[0]
    sys_sig = np.asarray(sp.sigmas_systematic(), dtype=float)
    iso = {"name": "p", "masses": sp.masses(mol.masses).tolist(),
           "obs_constants": list(sp.observed()), "sigma_constants": list(sp.sigmas()),
           "sigma_systematic_constants": list(sys_sig),
           "component_indices": list(sp.component_indices)}
    ctbl = {"p": {c: {"alpha_sum_mhz": 20.0, "method": "VPT2_semidiag",
                      "source": "harmonic_hessian"} for c in ("A", "B", "C")}}
    targets = resolve_corrections([iso], correction_table=ctbl, mode="hybrid_auto",
                                  sigma_vib_fraction=0.0)
    new = apply_corrections_to_isotopologues([iso], targets)[0]
    np.testing.assert_allclose(new["sigma_systematic_constants"], sys_sig)
