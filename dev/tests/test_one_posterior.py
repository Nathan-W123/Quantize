"""One posterior: every reporting path describes the objective the fit minimised.

There were three, and none of them was the posterior of the fitted objective:

  * the Laplace (and the MCMC's log-probability) carried the spectral data and
    the quantum prior, but not the internal-coordinate prior an internal-mode
    fit carried -- log_probability skipped it with a bare `pass`; under the
    split objective it applied the quantum block in every direction, although
    the split step lets it act only in the data's null space; and with a prior
    target it centred the quantum energy on the last quantum point instead of
    the target the fit was pulled toward;
  * the run report (uncertainty.core.compute_uncertainty) rebuilt a posterior
    from the config: the configured internal prior, no quantum prior at all,
    the symmetry-forbidden directions kept, every width rescaled by the
    spectral chi-square -- under a sentence saying the widths included the
    quantum prior.

Now _posterior_blocks builds the blocks once, from what the fit used, and the
Laplace, the Gauss-Newton check, the log-probability and the run report all
read them. These tests pin each path against that one definition.
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
from backend.quantize import MolecularOptimizer, _kabsch_align  # noqa: E402

MASSES_H2O = [15.9949146196, 1.00782503207, 1.00782503207]
MASSES_D2O = [15.9949146196, 2.0141017778, 2.0141017778]
B0_H2O = [835840.29, 435351.72, 278138.70]
B0_D2O = [462324.0, 218038.0, 145258.0]


def _coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _fit(species=1, **kw):
    isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
             "sigma_constants": [0.2] * 3}]
    if species > 1:
        isos.append({"name": "D2-16O", "masses": MASSES_D2O,
                     "obs_constants": B0_D2O, "sigma_constants": [0.2] * 3})
    base = dict(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                quantum_backend="analytic_water", use_autoconfig=False,
                max_iter=10, quantum_prior_sigma_ang=0.02)
    base.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(**base)
        opt.run()
    return opt


def _physical_precision(opt):
    """Inverse of the Laplace covariance on the physical directions."""
    cov, _ = opt.laplace_approximation(check_gauss_newton=False)
    B = opt.physical_basis()
    return np.linalg.inv(B.T @ cov @ B), B


# ── The Laplace carries the internal prior the fit carried ───────────────────

def test_the_laplace_includes_the_internal_prior_when_the_fit_had_one():
    opt = _fit(ic_prior_weight=1.0)
    _H_spec, _H_prior, parts = opt._posterior_blocks()
    assert "internal prior" in parts and "quantum" in parts
    # Its precision is exactly spectral + quantum + w B^T diag(1/sigma^2) B.
    st = opt._last_ic_prior
    Bq = st["coord_set"].active_B_matrix(opt.coords)
    J, _ = opt.spectral.stacked(opt.coords)
    _g, Hp = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    H = (J.T @ J + opt.quantum_prior_weight(refresh=True) * opt.optimizer._spd_hessian(Hp)
         + st["weight"] * Bq.T @ np.diag(1.0 / st["sigma"] ** 2) @ Bq)
    M, B = _physical_precision(opt)
    expect = B.T @ H @ B
    assert np.allclose(M, expect, rtol=1e-6, atol=1e-8 * np.abs(expect).max())


def test_without_an_internal_prior_there_is_no_such_block():
    opt = _fit()                                   # default: off with a QM prior
    _H_spec, _H_prior, parts = opt._posterior_blocks()
    assert "internal prior" not in parts


# ── The log-probability counts it too ────────────────────────────────────────

def test_log_probability_includes_the_internal_prior():
    """It used to skip the q-space prior with `pass`, so the sampler's target
    had one term fewer than the fit's objective."""
    opt = _fit(ic_prior_weight=1.0)
    x0 = opt.coords.copy()
    x1 = x0.copy()
    x1[1, 1] += 2e-3                               # stretch one O-H
    st = opt._last_ic_prior

    def prior_chi2(x):
        d = st["q0"] - st["coord_set"].active_values(x)
        return st["weight"] * float(np.sum((d / st["sigma"]) ** 2))

    assert prior_chi2(x1) != pytest.approx(prior_chi2(x0))
    assert opt.structural_prior_chi2(x0) == pytest.approx(prior_chi2(x0))
    assert opt.structural_prior_chi2(x1) == pytest.approx(prior_chi2(x1))

    def spectral_chi2(x):
        _J, r = opt.spectral.stacked(x, want_jacobian=False)
        return float(np.sum(np.asarray(r) ** 2))

    alpha = opt.quantum_prior_weight()
    expect = (-0.5 * (spectral_chi2(x1) - spectral_chi2(x0))
              - 0.5 * (prior_chi2(x1) - prior_chi2(x0))
              - alpha * (opt.quantum_prior_energy(x1) - opt.quantum_prior_energy(x0)))
    got = opt.log_probability(x1) - opt.log_probability(x0)
    assert got == pytest.approx(expect, rel=1e-9)


# ── Split: the quantum block only where the split step lets it act ───────────

def test_split_posterior_keeps_the_quantum_block_in_the_data_null_space():
    """With two isotopologues and the internal prior on, the split step's
    stacked Jacobian is full rank over the physical directions, so the quantum
    term never acts -- and the posterior must not pretend it does."""
    opt = _fit(species=2, objective_mode="split", ic_prior_weight=1.0)
    _H_spec, _H_prior, parts = opt._posterior_blocks()
    assert "quantum" not in parts
    Hq = parts["quantum (data null space)"]
    B = opt.physical_basis()
    blk = B.T @ Hq @ B
    _g, Hp = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    full = B.T @ (opt.quantum_prior_weight(refresh=True) * opt.optimizer._spd_hessian(Hp)) @ B
    assert np.abs(blk).max() < 1e-8 * np.abs(full).max()


# ── Prior target: the quantum energy is centred where the fit's prior is ─────

def test_with_a_prior_target_the_quantum_energy_is_centred_on_it():
    target = _coords(r=0.9600, deg=104.0)
    opt = _fit(coordinate_mode="cartesian", prior_target_coords=target,
               coords=target.copy())
    # At the target itself the fit's quantum prior is exactly zero...
    assert opt.quantum_prior_energy(target) == pytest.approx(0.0, abs=1e-14)
    # ...and away from it, it is the target-centred quadratic 1/2 d^T H d.
    x = opt.coords
    g, H = opt._projected_reference_terms()
    d = (x - _kabsch_align(target, x)).ravel()
    assert opt.quantum_prior_energy(x) == pytest.approx(0.5 * d @ H @ d, rel=1e-12)


# ── The report is the same posterior ─────────────────────────────────────────

def test_the_report_widths_are_geometry_uncertainty_s():
    """The run report and the method users call now agree: same covariance,
    same bonds. (The rebuilt report disagreed with it by up to 2x.)"""
    opt = _fit(species=2)
    post = opt.internal_coordinate_posterior()
    with contextlib.redirect_stdout(io.StringIO()):
        unc = opt.geometry_uncertainty()
    widths = dict(zip(post["names"], post["std_err"]))
    bond_names = [n for n, k in zip(post["names"], post["kinds"]) if k == "bond"]
    assert bond_names
    reported = sorted(widths[n] for n in bond_names)
    direct = sorted(v for k, v in unc.items() if k.count("-") == 1)
    assert np.allclose(reported, direct, rtol=1e-5)
    # angles: radians in the posterior, degrees in geometry_uncertainty
    ang = [np.degrees(widths[n]) for n, k in zip(post["names"], post["kinds"]) if k == "angle"]
    direct_ang = [v for k, v in unc.items() if k.count("-") == 2]
    assert np.allclose(sorted(ang), sorted(direct_ang), rtol=1e-5)


def test_the_report_says_what_it_is_a_posterior_of():
    post = _fit().internal_coordinate_posterior()
    assert post["blocks"][:2] == ["spectral data", "quantum"]
    assert post["objective_mode"] == "joint"
    assert post["alpha_q"] > 0.0
    assert post["internal_prior_sigma"] is None      # the fit carried none
    assert all(lbl in {"data-constrained", "mixed information",
                       "prior-dominated", "unidentifiable"} for lbl in post["labels"])


def test_the_worker_returns_the_fit_s_posterior(tmp_path):
    from backend.multistart import run_multistart

    isos = [{"name": "H2-16O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
             "sigma_constants": [0.2] * 3}]
    kwargs = dict(quantum_backend="analytic_water", use_autoconfig=False,
                  max_iter=5, quantum_prior_sigma_ang=0.02,
                  base_workdir=str(tmp_path), job_name="one")
    with contextlib.redirect_stdout(io.StringIO()):
        results = run_multistart(starts=[_coords()], elems=["O", "H", "H"],
                                 isotopologues=isos, optimizer_kwargs=kwargs,
                                 max_workers=1, job_name="one")
    post = results[0]["posterior"]
    assert post and "error" not in post
    assert len(post["covariance"]) == len(post["names"]) > 0
