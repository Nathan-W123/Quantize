"""Vibrational corrections from a level other than the fit's.

The fit's Hessian-built corrections come from its own level, re-expanded at
the current geometry on each Hessian refresh. correction_method /
correction_basis take them from another level instead, at that level's own
minimum, built once (MolecularOptimizer._correction_level_table). On water
(CCSD/cc-pVDZ fit and prior) that moved the bond from -0.58 to -0.03 mA
against r_e for 3.7 min of MP2/cc-pVTZ.
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
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

import dev.analytic_water_backend  # noqa: F401,E402
import dev.pyscf_backend  # noqa: F401,E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from test_bootstrap_targets import B0_H2O, MASSES_H2O, _coords  # noqa: E402


def _isos():
    return [{"name": "H2-16O", "masses": list(MASSES_H2O), "obs_constants": list(B0_H2O),
             "sigma_constants": [0.2] * 3}]


def _make(backend="pyscf_hf", **kw):
    args = dict(elems=["O", "H", "H"], coords=_coords(), isotopologues=_isos(),
                quantum_backend=backend, orca_method="hf", orca_basis="sto-3g",
                use_autoconfig=False, coordinate_mode="cartesian", max_iter=8,
                hess_recalc_every=2, harmonic_from_hessian=True)
    args.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return MolecularOptimizer(**args)


def test_a_correction_level_needs_corrections_to_change():
    with pytest.raises(ValueError, match="harmonic_from_hessian is off"):
        _make(harmonic_from_hessian=False, correction_basis="6-31g")


def test_a_correction_level_needs_a_backend_that_optimises():
    """It is evaluated at its own minimum; the analytic test backend has no
    optimise()."""
    with pytest.raises(ValueError, match="optimise"):
        _make(backend="analytic_water", correction_basis="6-31g")


def test_the_corrections_come_from_the_level_s_own_minimum_once():
    """The table is the one built at HF/6-31G's own minimum, not the fit's
    (HF/STO-3G) -- and it is built once, however many Hessian refreshes the
    fit makes: one optimisation, one Hessian at that level."""
    from backend.registry import get_backend
    from backend.spectral.harmonic_alpha import build_correction_table_from_hessian
    opt = _make(correction_basis="6-31g")
    calls = {"optimise": 0, "hessian": 0}
    be = opt._corr_backend
    real_opt, real_hess = be.optimise, be.run_hessian

    def counted_opt(*a, **k):
        calls["optimise"] += 1
        return real_opt(*a, **k)

    def counted_hess(*a, **k):
        calls["hessian"] += 1
        return real_hess(*a, **k)

    be.optimise, be.run_hessian = counted_opt, counted_hess
    with contextlib.redirect_stdout(io.StringIO()):
        opt.run()
        # Two more Hessian refreshes' worth of correction passes (this short
        # fit refreshes once): the table must not be rebuilt.
        opt._apply_harmonic_alpha_corrections()
        opt._apply_harmonic_alpha_corrections()
    assert calls == {"optimise": 1, "hessian": 1}
    geom, table, _info = opt._corr_table

    ref_be = get_backend("pyscf_hf")(elems=["O", "H", "H"], method="hf", basis="6-31g")
    with contextlib.redirect_stdout(io.StringIO()):
        ref_geom = np.asarray(ref_be.optimise(_coords()), dtype=float)
        ref_table, _ = build_correction_table_from_hessian(
            ref_be.run_hessian(ref_geom).hessian_bohr, ref_geom, _isos(),
            sigma_fraction=opt._harmonic_sigma_fraction, harmonic_scheme=opt._harmonic_scheme,
            cubic_scheme=opt._cubic_scheme)
    for comp in ("A", "B", "C"):
        assert table["H2-16O"][comp]["alpha_sum_mhz"] == pytest.approx(
            ref_table["H2-16O"][comp]["alpha_sum_mhz"], rel=1e-6)

    # Premise: the fit's own level gives different corrections, so the test
    # can tell the two apart.
    own = _make()
    with contextlib.redirect_stdout(io.StringIO()):
        own.run()
    assert own.spectral.isotopologues[0]["obs_constants"][0] != pytest.approx(
        opt.spectral.isotopologues[0]["obs_constants"][0], rel=1e-6)


@pytest.mark.parametrize("edit,said", [
    (lambda rc: rc.update(harmonic_from_hessian=False, correction_basis="cc-pvtz"),
     "harmonic_from_hessian is off"),
    (lambda rc: rc.update(harmonic_from_hessian=True, correction_method=7), "method or basis name"),
    (lambda rc: rc.update(harmonic_from_hessian=True, correction_basis="  "), "method or basis name"),
])
def test_the_config_keys_are_validated(edit, said):
    from runner.usability import ConfigError, load_config, validate_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    edit(cfg.setdefault("rovibrational_corrections", {}))
    with pytest.raises(ConfigError, match=said):
        validate_config(cfg)


def test_the_config_keys_need_the_pyscf_backend_and_reach_the_optimizer():
    from runner.run_generic import build_run_inputs
    from runner.usability import ConfigError, load_config, validate_config
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    rc = cfg.setdefault("rovibrational_corrections", {})
    rc.update(harmonic_from_hessian=True, correction_method="mp2", correction_basis="cc-pvtz")
    with pytest.raises(ConfigError, match="pyscf_hf"):
        validate_config(cfg)            # the analytic water backend cannot optimise
    cfg["quantum"]["backend"] = "pyscf_hf"
    validate_config(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        kw = build_run_inputs(cfg, verbose=False)["optimizer_kwargs"]
    assert (kw["correction_method"], kw["correction_basis"]) == ("mp2", "cc-pvtz")
