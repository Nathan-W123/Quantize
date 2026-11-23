"""`quantize run` recomputes the quantum Hessian at each start's answer (D-146).

A plain fit computes its Hessian on its own schedule, usually near its start,
and the answer is fitted on that curvature and the quantum weight it sets: on
a hydrogen-bonded water dimer (HF/6-31G, eight starts) one refresh moved the
answers by up to 14.4 mA RMS and the weight by up to 18% (T-57). `quantize
uncertainty` already refreshed; `quantize run` now does, per start, before
ranking; `--no-hessian-refresh` keeps the fit's own Hessian.
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

import backend.multistart as MS
from test_cd_rows_sign import _isos, _x0

import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)

_KW = {"quantum_backend": "analytic_water", "use_autoconfig": False, "max_iter": 40}


def _start():
    return _x0(r=0.97, deg=103.0)


def test_each_start_refreshes_its_hessian_and_reports_it(tmp_path):
    kw = dict(_KW, base_workdir=str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        plain = MS.run_multistart([_start()], ["O", "H", "H"], _isos(None), kw)
        fresh = MS.run_multistart([_start()], ["O", "H", "H"], _isos(None), kw, refresh_hessian=True)
    assert plain[0]["hessian_refresh"] is None
    hr = fresh[0]["hessian_refresh"]
    assert set(hr) >= {"passes", "age_before_ang", "age_after_ang", "moved_ang", "settled"}
    assert hr["passes"] >= 1
    # The fit's own Hessian was away from its answer, the refreshed one is at it.
    assert hr["age_before_ang"] > hr["age_after_ang"]
    # The reported answer is the refreshed fit's, and its history includes
    # the passes.
    assert fresh[0]["n_iterations"] > plain[0]["n_iterations"]
    moved = float(np.sqrt(np.mean((np.asarray(fresh[0]["coords"]) - np.asarray(plain[0]["coords"])) ** 2)))
    assert moved == pytest.approx(hr["moved_ang"], rel=1e-6, abs=1e-12)


def test_the_cli_flag_reaches_the_runner(monkeypatch):
    import cli
    import runner.run_generic as RG
    seen = []
    monkeypatch.setattr(RG, "main", lambda cfg, refresh_hessian=True: seen.append(refresh_hessian) or {})
    config = _HERE.parent / "configs" / "water_analytic_endtoend.yaml"
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["run", str(config), "--no-run-dir"]) == 0
        assert cli.main(["run", str(config), "--no-run-dir", "--no-hessian-refresh"]) == 0
    assert seen == [True, False]
