"""The benchmark's fits reuse the Hessian its correction table already holds."""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)


def _bench():
    import importlib.util as iu
    spec = iu.spec_from_file_location("accbench_h", str(_ROOT / "scripts" / "accuracy_upgrades_benchmark.py"))
    mod = iu.module_from_spec(spec)
    saved = sys.argv
    try:
        sys.argv = ["accbench_h"]
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_the_fit_takes_its_hessians_from_the_run_s_cached_function():
    """Every hybrid_fit recomputed the Hessian at the prior that table_for had
    just computed: 75-79% of each fit's wall time at HF/6-31G (acetyl
    fluoride 10.0 -> 2.1 s). With the run's hessian_fn the backend's own
    Hessian is never called, energy and gradient still are, and the answer
    is the same (analytic water: deterministic, so exactly)."""
    from backend.quantize import MolecularOptimizer
    from backend.registry import get_backend
    from test_bootstrap_targets import B0_H2O, MASSES_H2O, _coords
    mod = _bench()
    inner = get_backend("analytic_water")(elems=["O", "H", "H"])
    calls = {"hessian": 0, "fn": 0}
    real = inner.run_hessian

    def counted(c):
        calls["hessian"] += 1
        return real(c)

    inner.run_hessian = counted

    def hfn(c):
        calls["fn"] += 1
        return real(c).hessian_bohr

    wrapped = mod._CachedHessianBackend(inner, hfn)
    copy.deepcopy(wrapped)                     # __getattr__ must not recurse
    isos = [{"name": "w", "masses": MASSES_H2O, "obs_constants": list(B0_H2O),
             "sigma_constants": [0.2] * 3}]

    def fit(backend):
        with contextlib.redirect_stdout(io.StringIO()):
            opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_coords(), isotopologues=isos,
                                     quantum_backend="analytic_water", use_autoconfig=False,
                                     coordinate_mode="cartesian", max_iter=60)
            if backend is not None:
                opt._backend = backend
            return np.asarray(opt.run(), dtype=float)

    plain = fit(None)
    got = fit(wrapped)
    assert calls["hessian"] == 0 and calls["fn"] >= 1
    np.testing.assert_array_equal(got, plain)
