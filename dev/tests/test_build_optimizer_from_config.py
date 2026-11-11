"""`quantize uncertainty` must be able to build the optimizer its config describes.

These exist because the subcommand was dead on arrival. `cli.py` opened with

    from runner.run_from_config import build_optimizer_from_config

and that name was never written -- not in `run_from_config`, not anywhere in
the repository. So every uncertainty path in the CLI (Laplace, bootstrap, MCMC,
the posterior report, the plots) raised ImportError before parsing a single
argument, and fourteen commits of uncertainty work were unreachable from the
command line. No test caught it because nothing imported the module the CLI
imports.

The work it needed was the config -> MolecularOptimizer-kwargs translation,
which lived inline in `run_generic.main` and so could only be had by running a
whole multistart fit. That block is now `run_generic.build_run_inputs`, and
these tests pin both halves: that the translation still produces what a fit
needs, and that a single optimizer can be built from it without running one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from runner.run_generic import build_run_inputs  # noqa: E402
from runner.usability import load_config  # noqa: E402

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"

#: Every name `run_generic.main` consumes after the translation. If the builder
#: stops returning one of these, `main` raises NameError on its next line, so
#: this list is the contract between the two and not decoration.
REQUIRED_KEYS = (
    "name", "managed_run", "run_dir", "base_workdir", "coords", "elems", "bonds",
    "spectral_model", "isotopologues", "backend", "orca_method", "orca_basis",
    "spectral_only", "preset_name", "preset", "write_xyz", "symmetry_spec",
    "coordinate_mode", "starts", "conformer_workflow", "use_conformer_mixture",
    "optimizer_kwargs",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def inputs(cfg):
    return build_run_inputs(cfg, verbose=False)


def test_the_name_the_cli_imports_actually_exists():
    """The regression itself: this import is the first line of _cmd_uncertainty."""
    from runner.run_from_config import build_optimizer_from_config

    assert callable(build_optimizer_from_config)


def test_build_run_inputs_returns_every_name_main_consumes(inputs):
    missing = [k for k in REQUIRED_KEYS if k not in inputs]
    assert not missing, f"build_run_inputs stopped returning {missing}; main() will NameError"


def test_translation_reaches_the_optimizer_kwargs_a_fit_needs(inputs):
    kw = inputs["optimizer_kwargs"]
    # Sampled across the config sections, so a section silently dropped from the
    # extracted block fails here rather than at the first real fit.
    assert kw["quantum_backend"] == "analytic_water"          # quantum:
    assert kw["coordinate_mode"] == "internal"                # coordinate_mode:
    assert kw["symmetry"] == "C2v"                            # symmetry:
    assert kw["anharmonic_from_hessian"] is True              # rovibrational_corrections:
    assert kw["harmonic_from_hessian"] is True
    assert kw["ic_prior_sigma_bond"] == pytest.approx(0.12)   # internal_coordinates:
    assert kw["ic_freeze_sigma_floor"] == pytest.approx(1e-6)  # internal_priors:
    assert kw["project_rigid_modes"] is True


def test_the_config_decides_how_many_starts_not_the_builder(inputs):
    starts = inputs["starts"]
    assert len(starts) == int(inputs["preset"]["n_starts"]) >= 1
    # start[0] is the config geometry untouched; the rest are jittered from it.
    assert np.allclose(starts[0], inputs["coords"])
    for s in starts[1:]:
        assert not np.allclose(s, inputs["coords"])


def test_build_run_inputs_is_deterministic(cfg):
    """A second call must give the same seeds: `rng_seed` is in the config."""
    a = build_run_inputs(cfg, verbose=False)
    b = build_run_inputs(cfg, verbose=False)
    for sa, sb in zip(a["starts"], b["starts"], strict=True):
        assert np.array_equal(sa, sb)


def test_builder_returns_an_unrun_optimizer_at_the_config_geometry(cfg, inputs):
    from runner.run_from_config import build_optimizer_from_config

    opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
    # Unrun: no history, and sitting on the geometry the config named.
    assert not getattr(opt, "history", [])
    assert np.allclose(opt.coords, inputs["coords"])
    assert opt.quantum_backend.lower() == "analytic_water"


def test_builder_survives_the_kwargs_multistart_pops_itself(cfg):
    """`base_workdir` and `job_name` are multistart's, not the constructor's.

    run_multistart pops both and turns them into a per-start workdir.
    MolecularOptimizer takes neither, so passing the kwargs straight through
    raises TypeError -- which is what a naive builder would have done.
    """
    from runner.run_from_config import build_optimizer_from_config

    opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
    assert opt.workdir and Path(opt.workdir).is_dir()


def test_single_start_configs_need_no_multistart_decision(cfg):
    """With one start there is nothing to select, so the default must not fit."""
    from runner.run_from_config import build_optimizer_from_config

    single = dict(cfg)
    single["preset"] = dict(cfg.get("preset") or {}) if isinstance(cfg.get("preset"), dict) else cfg.get("preset")
    inputs = build_run_inputs(cfg, verbose=False)
    if len(inputs["starts"]) == 1:
        opt = build_optimizer_from_config(cfg, verbose=False)
        assert not getattr(opt, "history", [])
    else:
        # This config asks for several starts, so the default WOULD fit. Assert
        # the opt-out exists and is what keeps this test suite from running a
        # multistart on every invocation.
        opt = build_optimizer_from_config(cfg, multistart=False, verbose=False)
        assert not getattr(opt, "history", [])


def test_verbose_false_drops_the_banner_but_keeps_physics_warnings(cfg, capsys):
    """`verbose` governs the run banner only.

    It must not silence a caveat about the numbers themselves. This config
    falls back to built-in BOB u-parameters, which carry 100% sigma_u and are
    order-of-magnitude only -- a caller that suppressed the banner still needs
    to be told that, so the assertion here is deliberately two-sided.
    """
    build_run_inputs(cfg, verbose=False)
    out = capsys.readouterr().out
    assert "[BOB]" in out, "a physics caveat was silenced along with the banner"
    for banner in ("] elements", "] isotopologues", "] preset", "] n_starts"):
        assert banner not in out


def test_verbose_true_still_prints_the_run_banner(cfg, capsys):
    build_run_inputs(cfg, verbose=True)
    out = capsys.readouterr().out
    assert "elements" in out and "isotopologues" in out and "preset" in out
