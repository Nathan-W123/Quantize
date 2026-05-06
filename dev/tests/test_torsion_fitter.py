"""Tests for backend/torsion_fitter.py (Phase 6: parameter fitting)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from backend.torsion_fitter import (
    fit_torsion_to_levels,
    fit_torsion_to_transitions,
    select_fit_params,
)
from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion_uncertainty import default_torsion_parameters


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_spec(F: float = 27.6, rho: float = 0.81, Vcos3: float = -186.8, n_basis: int = 10):
    pot = TorsionFourierPotential(
        v0=186.1, vcos={3: Vcos3}, vsin={}, units="cm-1"
    )
    return TorsionHamiltonianSpec(
        F=F, rho=rho, A=4.25, B=0.823, C=0.793,
        potential=pot, n_basis=n_basis, units="cm-1"
    )


def _synthetic_levels(spec, J_values=(0,), K_values=(0,), n_levels=6):
    """Generate synthetic 'observed' rows from a known spec."""
    rows = []
    for J in J_values:
        for K in K_values:
            out = solve_ram_lite_levels(spec, J=J, K=K, n_levels=n_levels)
            for i, e in enumerate(out["energies_cm-1"]):
                rows.append({"J": J, "K": K, "level_index": i, "energy_cm-1": float(e)})
    return rows


def _synthetic_transitions(spec, J_vals=(0, 1), K_vals=(0,), n_levels=6):
    """Generate synthetic transitions between J blocks."""
    level_rows = _synthetic_levels(spec, J_values=J_vals, K_values=K_vals, n_levels=n_levels)
    pred_map = {}
    for r in level_rows:
        pred_map[(r["J"], r["K"], r["level_index"])] = r["energy_cm-1"]
    transitions = []
    for K in K_vals:
        for lvl in range(n_levels):
            lo_key = (0, K, lvl)
            hi_key = (1, K, lvl)
            if lo_key in pred_map and hi_key in pred_map:
                transitions.append({
                    "J_lo": 0, "K_lo": K, "level_lo": lvl,
                    "J_hi": 1, "K_hi": K, "level_hi": lvl,
                    "freq_cm-1": float(pred_map[hi_key] - pred_map[lo_key]),
                })
    return transitions


# â”€â”€ select_fit_params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSelectFitParams:
    def test_scalar_params(self):
        spec = _make_spec()
        params = select_fit_params(spec, ["F", "rho"])
        assert len(params) == 2
        assert params[0].name == "F"
        assert params[1].name == "rho"

    def test_vcos_param(self):
        spec = _make_spec()
        params = select_fit_params(spec, ["Vcos_3", "Vcos_6"])
        names = [p.name for p in params]
        assert "Vcos_3" in names
        assert "Vcos_6" in names

    def test_vsin_param(self):
        spec = _make_spec()
        params = select_fit_params(spec, ["Vsin_3"])
        assert params[0].name == "Vsin_3"
        assert params[0].path == ("potential", "vsin", "3")

    def test_v0_param(self):
        spec = _make_spec()
        params = select_fit_params(spec, ["v0"])
        assert params[0].path == ("potential", "v0")

    def test_case_insensitive(self):
        spec = _make_spec()
        params = select_fit_params(spec, ["f", "RHO", "vcos_3"])
        assert params[0].name == "F"
        assert params[1].name == "rho"
        assert params[2].name == "Vcos_3"

    def test_unknown_param_raises(self):
        spec = _make_spec()
        with pytest.raises(ValueError, match="Unknown parameter"):
            select_fit_params(spec, ["NotAParam"])

    def test_invalid_vcos_order_raises(self):
        spec = _make_spec()
        with pytest.raises(ValueError):
            select_fit_params(spec, ["Vcos_0"])

    def test_empty_list(self):
        spec = _make_spec()
        params = select_fit_params(spec, [])
        assert params == []


# â”€â”€ fit_torsion_to_levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestFitTorsionToLevels:
    def test_no_observed_rows_returns_inf(self):
        spec = _make_spec()
        result = fit_torsion_to_levels(spec, [])
        assert result["rms_cm-1"] == float("inf")
        assert result["converged"] is False

    def test_exact_data_converges_quickly(self):
        """Fitting to exact synthetic data should reach near-zero RMS."""
        spec_true = _make_spec()
        obs_rows = _synthetic_levels(spec_true, n_levels=8)

        # Start with perturbed spec
        spec_init = _make_spec(Vcos3=-180.0)
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=30)
        assert result["rms_cm-1"] < 0.5
        # Fitted Vcos_3 should approach true value (-186.8)
        fitted_val = float(result["fitted_spec"].potential.vcos[3])
        assert abs(fitted_val - (-186.8)) < 5.0

    def test_rms_decreases_from_initial(self):
        spec_true = _make_spec()
        obs_rows = _synthetic_levels(spec_true, n_levels=6)
        spec_init = _make_spec(Vcos3=-170.0)
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=20)
        assert result["rms_cm-1"] <= result["rms_cm-1_init"] + 1e-6

    def test_result_has_required_keys(self):
        spec = _make_spec()
        obs = _synthetic_levels(spec, n_levels=4)
        result = fit_torsion_to_levels(spec, obs)
        for k in ("fitted_spec", "param_names", "param_values", "param_values_init",
                  "rms_cm-1", "rms_cm-1_init", "n_iter", "converged",
                  "residuals_cm-1", "warnings"):
            assert k in result

    def test_fitted_spec_is_spec_instance(self):
        spec = _make_spec()
        obs = _synthetic_levels(spec, n_levels=4)
        result = fit_torsion_to_levels(spec, obs)
        assert isinstance(result["fitted_spec"], TorsionHamiltonianSpec)

    def test_param_values_init_unchanged(self):
        """Initial parameter vector must equal packed values of original spec."""
        spec_init = _make_spec(Vcos3=-170.0)
        spec_true = _make_spec()
        obs_rows = _synthetic_levels(spec_true, n_levels=5)
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=10)
        assert result["param_values_init"][0] == pytest.approx(-170.0, rel=1e-6)

    def test_fit_multiple_params(self):
        """Fit both F and Vcos_3 simultaneously."""
        spec_true = _make_spec(F=27.6, Vcos3=-186.8)
        obs_rows = _synthetic_levels(spec_true, n_levels=10)
        spec_init = _make_spec(F=27.0, Vcos3=-180.0)
        params = select_fit_params(spec_init, ["F", "Vcos_3"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=50)
        assert result["rms_cm-1"] < result["rms_cm-1_init"]

    def test_v0_recovery(self):
        """Fit v0 to exact data â€” should converge to near-zero offset."""
        spec_true = _make_spec()
        obs_rows = _synthetic_levels(spec_true, n_levels=6)
        # Perturb v0 by +5 cm-1
        spec_init = deepcopy(spec_true)
        spec_init.potential.v0 += 5.0
        params = select_fit_params(spec_init, ["v0"])
        result = fit_torsion_to_levels(spec_init, obs_rows, params=params, max_iter=20)
        assert result["rms_cm-1"] < 0.5


    def test_bounds_clip_fitted_parameter(self):
        spec_true = _make_spec(Vcos3=-186.8)
        obs_rows = _synthetic_levels(spec_true, n_levels=6)
        spec_init = _make_spec(Vcos3=-170.0)
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_levels(
            spec_init,
            obs_rows,
            params=params,
            bounds={"Vcos_3": {"lower": -180.0, "upper": -160.0}},
            max_iter=20,
        )
        assert float(result["param_values"][0]) >= -180.0
        assert float(result["param_values"][0]) <= -160.0

    def test_uncertainty_outputs_present(self):
        spec = _make_spec()
        obs_rows = _synthetic_levels(spec, n_levels=5)
        params = select_fit_params(spec, ["F", "Vcos_3"])
        result = fit_torsion_to_levels(spec, obs_rows, params=params, max_iter=2)
        assert result["std_err"].shape == (2,)
        assert result["covariance"].shape == (2, 2)
        assert result["correlation"].shape == (2, 2)
        assert np.allclose(np.diag(result["correlation"]), 1.0)


# â”€â”€ fit_torsion_to_transitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestFitTorsionToTransitions:
    def test_no_transitions_returns_inf(self):
        spec = _make_spec()
        result = fit_torsion_to_transitions(spec, [])
        assert result["rms_cm-1"] == float("inf")

    def test_freq_mhz_accepted(self):
        """Transitions with freq_mhz should be accepted."""
        spec = _make_spec()
        # Build one synthetic transition in MHz
        trans = _synthetic_transitions(spec, J_vals=(0, 1), n_levels=3)
        _MHZ_PER_CM1 = 29979.2458
        trans_mhz = [{
            "J_lo": t["J_lo"], "K_lo": t["K_lo"], "level_lo": t["level_lo"],
            "J_hi": t["J_hi"], "K_hi": t["K_hi"], "level_hi": t["level_hi"],
            "freq_mhz": float(t["freq_cm-1"]) * _MHZ_PER_CM1,
        } for t in trans]
        result = fit_torsion_to_transitions(spec, trans_mhz)
        assert result["rms_cm-1"] < 1.0  # self-consistent â†’ near zero

    def test_rms_decreases(self):
        spec_true = _make_spec()
        trans = _synthetic_transitions(spec_true, J_vals=(0, 1), n_levels=6)
        spec_init = _make_spec(Vcos3=-170.0)
        params = select_fit_params(spec_init, ["Vcos_3"])
        result = fit_torsion_to_transitions(spec_init, trans, params=params, max_iter=30)
        assert result["rms_cm-1"] <= result["rms_cm-1_init"] + 1e-6

    def test_result_keys(self):
        spec = _make_spec()
        trans = _synthetic_transitions(spec, n_levels=3)
        result = fit_torsion_to_transitions(spec, trans)
        for k in ("fitted_spec", "param_names", "param_values", "rms_cm-1", "converged"):
            assert k in result

    def test_missing_freq_raises(self):
        spec = _make_spec()
        bad_trans = [{"J_lo": 0, "K_lo": 0, "level_lo": 0,
                      "J_hi": 1, "K_hi": 0, "level_hi": 0}]
        with pytest.raises(ValueError, match="freq_cm-1.*freq_mhz"):
            fit_torsion_to_transitions(spec, bad_trans)


# â”€â”€ Integration: phase2 exports with fitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_phase2_with_fitting_enabled():
    """_run_torsion_phase2_exports should write torsion_fit_params.csv when fitting enabled."""
    import tempfile, shutil
    from pathlib import Path
    from runner.run_generic import _run_torsion_phase2_exports

    run_dir = Path(tempfile.mkdtemp(prefix="quantize_fit_test_"))
    try:
        spec_true = _make_spec()
        obs_rows = _synthetic_levels(spec_true, n_levels=4)

        cfg = {
            "name": "fit_test",
            "torsion_hamiltonian": {
                "enabled": True,
                "F": 27.6,
                "rho": 0.81,
                "n_basis": 10,
                "n_levels": 4,
                "J_values": [0],
                "K_values": [0],
                "potential": {"v0": 186.1, "vcos": {"3": -170.0}},
                "targets": obs_rows,
                "fitting": {
                    "enabled": True,
                    "params": ["Vcos_3"],
                    "max_iter": 20,
                    "use_levels": True,
                    "use_transitions": False,
                },
            },
        }
        isotopologues = [{"name": "H2O", "masses": [15.999, 1.008, 1.008]}]
        coords = np.array([[0.0, 0.0, 0.1], [0.0, 0.75, -0.5], [0.0, -0.75, -0.5]])
        out = _run_torsion_phase2_exports(
            cfg=cfg, elems=["O", "H", "H"], best={"coords": coords},
            isotopologues=isotopologues, run_dir=run_dir,
        )
        assert out["torsion_fit_params_csv"] is not None
        assert out["torsion_fit_params_csv"].is_file()
        summary_text = out["torsion_summary_json"].read_text()
        assert "fitting_rms_cm-1" in summary_text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

