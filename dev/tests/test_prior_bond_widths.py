"""The calibrated quantum prior's bond widths equal sigma_x (decision D-170).

sigma_x is meant as the level of theory's typical bond error, but the weight
1 / (1/2 lambda_bar sigma_x^2) sets a prior that is tighter along bonds, which
are stiffer than the typical mode: on the nine benchmark molecules its RMS
bond width was 0.77 sigma_x at HF/6-31G and 0.65 sigma_x at B3LYP/6-31G(d).
With match_prior_bond_widths (the default) the weight is chosen so the mean
over bonds of b^T (alpha_q H)^+ b is sigma_x^2.
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

from backend.spectral.SVD import SubspaceOptimizer  # noqa: E402

from dev.tests.test_linear_angles import _HCN, _ISOS  # noqa: E402


def _prior_bond_widths(opt):
    """sqrt(b^T (alpha_q H)^+ b) per bond, on the positive-curvature subspace
    the weight is calibrated over."""
    a = opt.quantum_prior_weight(refresh=True)
    _, H = opt._project_quantum_terms(opt.quantum.gradient, opt.quantum.hessian)
    ev, V = np.linalg.eigh(0.5 * (H + H.T))
    keep = ev > 1e-6 * max(float(np.abs(ev).max()), 1.0)
    C = (V[:, keep] / (a * ev[keep])) @ V[:, keep].T
    return a, np.array([np.sqrt(b @ C @ b) for b in opt.optimizer.prior_bond_rows]), H


def _hcn(match):
    import dev.pyscf_backend  # noqa: F401  (registers pyscf_hf)
    from backend.quantize import MolecularOptimizer
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["H", "C", "N"], coords=_HCN, isotopologues=_ISOS,
                                 quantum_backend="pyscf_hf", orca_method="hf",
                                 orca_basis="sto-3g", use_autoconfig=False, max_iter=4,
                                 symmetry="C1", quantum_prior_sigma_ang=0.02,
                                 match_prior_bond_widths=match)
        opt.run()
    return opt


def test_the_prior_s_rms_bond_width_is_sigma_x():
    """HCN at HF/STO-3G: 0.852 sigma_x with the lambda_bar weight, sigma_x
    with the match (alpha_q 1182 -> 858)."""
    _a, w, _H = _prior_bond_widths(_hcn(True))
    assert np.sqrt(np.mean(w ** 2)) == pytest.approx(0.02, rel=1e-9)
    a_off, w_off, H = _prior_bond_widths(_hcn(False))
    assert np.sqrt(np.mean(w_off ** 2)) / 0.02 == pytest.approx(0.852, abs=0.005)
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    lam_bar = ev[ev > 1e-6 * max(float(np.abs(ev).max()), 1.0)].mean()
    assert a_off == pytest.approx(1.0 / (0.5 * lam_bar * 0.02 ** 2), rel=1e-12)


def _toy():
    """A positive-definite 6x6 'Hessian' and two unit bond rows."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    H = (Q * np.array([0.5, 1.0, 2.0, 4.0, 6.0, 9.0])) @ Q.T
    rows = np.zeros((2, 6))
    rows[0, :3] = [-1.0, 0.0, 0.0]
    rows[0, 3:] = [1.0, 0.0, 0.0]
    rows[1] = rng.normal(size=6)
    rows[1] /= np.linalg.norm(rows[1])
    return H, rows


def _opt(match=True):
    return SubspaceOptimizer(1e-6, 0.0, 0.1, None, 1e-4, quantum_prior_sigma_ang=0.01,
                             match_prior_bond_widths=match)


def test_the_weight_is_the_mean_bond_variance_over_sigma_x_squared():
    H, rows = _toy()
    opt = _opt()
    opt.prior_bond_rows = rows
    Hinv = np.linalg.inv(H)
    expected = np.mean([b @ Hinv @ b for b in rows]) / 0.01 ** 2
    assert opt._calibrated_alpha_q(H) == pytest.approx(expected, rel=1e-12)
    a = opt._calibrated_alpha_q(H)
    widths = [np.sqrt(b @ np.linalg.inv(a * H) @ b) for b in rows]
    assert np.sqrt(np.mean(np.square(widths))) == pytest.approx(0.01, rel=1e-12)


@pytest.mark.parametrize("case", ["off", "no rows", "wrong size", "no bonds"])
def test_without_usable_bond_rows_the_weight_is_the_lambda_bar_formula(case):
    H, rows = _toy()
    opt = _opt(match=(case != "off"))
    opt.prior_bond_rows = {"off": rows, "no rows": None, "wrong size": rows[:, :3],
                           "no bonds": np.zeros((0, 6))}[case]
    lam_bar = np.linalg.eigvalsh(H).mean()
    assert opt._calibrated_alpha_q(H) == pytest.approx(1.0 / (0.5 * lam_bar * 0.01 ** 2),
                                                       rel=1e-12)


def test_config_false_reaches_the_optimizer_and_a_string_is_refused():
    import dev.analytic_water_backend  # noqa: F401  (the config's backend; not left to test order)
    from runner.run_generic import build_run_inputs
    from runner.usability import ConfigError, load_config, validate_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        kw = build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]
    assert "match_prior_bond_widths" not in kw, "absent: the constructor's default (on)"
    cfg["optimizer"] = dict(cfg.get("optimizer") or {}, match_prior_bond_widths=False)
    validate_config(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        kw = build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]
    assert kw["match_prior_bond_widths"] is False
    cfg["optimizer"]["match_prior_bond_widths"] = "no"
    with pytest.raises(ConfigError):
        validate_config(cfg)


def _water(r, **kw):
    import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)
    from backend.quantize import MolecularOptimizer
    from dev.tests.test_bootstrap_targets import B0_H2O, MASSES_H2O
    th = np.radians(104.0) / 2
    x = np.array([[0.0, 0.0, 0.0], [r * np.sin(th), 0.0, r * np.cos(th)],
                  [-r * np.sin(th), 0.0, r * np.cos(th)]])
    iso = [{"name": "H2O", "masses": MASSES_H2O, "obs_constants": B0_H2O,
            "sigma_constants": [0.1] * 3}]
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=x, isotopologues=iso,
                                 quantum_backend="analytic_water", use_autoconfig=False,
                                 quantum_prior_sigma_ang=0.02, auto_systematic_sigma=False,
                                 max_iter=60, symmetry="C1", **kw)
    return opt


@pytest.mark.parametrize("r", [1.20, 1.25, 1.30])
def test_bonds_are_found_where_the_hessian_was_taken(r):
    """Both O-H stretched past the covalent cutoff (1.164 A) at the Hessian's
    geometry: no bond to match, so the weight is the lambda_bar formula's,
    even with a prior target the internal-mode prior is not centred on.
    Found at that target instead, the bonds set alpha_q 3.2x (1.25 A, near
    the stretch's inflection) or 0.04x (1.30 A) the formula's, and the 1.30 A
    fit ended 9 mA asymmetric (scratchpad t85/verify/c5c_eig.py)."""
    target = _water(0.9578).coords
    on = _water(r, coordinate_mode="internal", prior_target_coords=target)
    off = _water(r, coordinate_mode="internal", prior_target_coords=target,
                 match_prior_bond_widths=False)
    with contextlib.redirect_stdout(io.StringIO()):
        on._update_orca()
        off._update_orca()
    assert on.optimizer.prior_bond_rows is None
    assert on.quantum_prior_weight(refresh=True) == off.quantum_prior_weight(refresh=True)


def test_a_bond_past_the_cutoff_at_the_start_is_matched_once_the_hessian_moves():
    """One O-H starts at 1.17 A, past the cutoff: the first Hessian matches the
    other bond only; the Hessian at the answer matches both."""
    opt = _water(1.0)
    opt.coords[2] = opt.coords[0] + 1.17 * (opt.coords[2] - opt.coords[0]) / np.linalg.norm(
        opt.coords[2] - opt.coords[0])
    with contextlib.redirect_stdout(io.StringIO()):
        opt._update_orca()
        assert opt.optimizer.prior_bond_rows.shape[0] == 1
        opt.run()
        opt.refresh_hessian_at_answer()
    _a, w, _H = _prior_bond_widths(opt)
    assert len(w) == 2
    assert np.sqrt(np.mean(w ** 2)) == pytest.approx(0.02, rel=1e-9)
