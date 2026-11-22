"""One-worker bootstraps refit in this process, not in a forked child.

A child forked from a process whose quantum backend has started threads
inherits their locks and can hang for ever: the water split check (pyscf
HF/6-31G, OMP_NUM_THREADS=4, 50 samples) hung 3 of 3 times. With one worker
the refits now run here; each still deep-copies the optimizer, so the cloud
is the pool's bit for bit (decisions D-102, T-42).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / ".github"))
sys.path.insert(0, str(_HERE.parent.parent))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
import backend.uncertainty_v2.bootstrap as BS  # noqa: E402
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.uncertainty_v2.models import BootstrapConfig  # noqa: E402
from test_cd_rows_sign import _isos, _x0  # noqa: E402


def _fit():
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["O", "H", "H"], coords=_x0(r=0.97, deg=103.0), isotopologues=_isos(None),
                                 quantum_backend="analytic_water", use_autoconfig=False, max_iter=80,
                                 auto_systematic_sigma=False)
        opt.run()
    return opt


def _cloud(opt, workers, tmp_path):
    cfg = BootstrapConfig(n_samples=6, max_workers=workers, seed=5, strategy="error-model",
                          persist_samples=False)
    with contextlib.redirect_stdout(io.StringIO()):
        s = BS.BootstrapEngine(opt, cfg).run(base_workdir=str(tmp_path / f"w{workers}"))
    return sorted(s.sample_results, key=lambda r: r.idx)


def test_one_worker_forks_nothing_and_matches_the_pool(tmp_path, monkeypatch):
    opt = _fit()
    base = np.asarray(opt.coords, float).copy()
    pooled = _cloud(opt, 2, tmp_path)

    class NoPool:
        def __init__(self, *a, **k):
            raise AssertionError("a one-worker bootstrap started a process pool")

    monkeypatch.setattr(BS, "ProcessPoolExecutor", NoPool)
    inline = _cloud(opt, 1, tmp_path)
    assert len(inline) == len(pooled) == 6
    assert all(r.usable for r in inline), [r.unusable_reason for r in inline]
    for a, b in zip(inline, pooled):
        assert a.idx == b.idx
        np.testing.assert_array_equal(np.asarray(a.coords), np.asarray(b.coords))
    # The refits work on copies: the base fit is untouched.
    np.testing.assert_array_equal(np.asarray(opt.coords, float), base)
