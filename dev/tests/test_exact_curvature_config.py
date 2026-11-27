"""exact_curvature_where_flagged from a config and the command line (N38).

The option (D-145, on by default: where the Gauss-Newton check flags a
direction, the step, the widths, the distance to the optimum and the
bootstrap gate use the exact curvature) was a constructor argument only, so
a Gauss-Newton fit could not be reproduced from a config.
``optimizer.exact_curvature_where_flagged`` sets it, and
``--no-exact-curvature`` on ``quantize run`` and ``quantize uncertainty``
turns it off. Absent, nothing is passed and the constructor's default holds.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
import cli  # noqa: E402
from runner.run_from_config import build_optimizer_from_config  # noqa: E402
from runner.run_generic import build_run_inputs  # noqa: E402
from runner.usability import ConfigError, load_config, validate_config  # noqa: E402

CONFIG = _ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"
KEY = "exact_curvature_where_flagged"


def _cfg(**optimizer):
    cfg = load_config(CONFIG)
    assert "optimizer" not in cfg or cfg["optimizer"] in (None, {})     # premise: absent
    cfg.pop("optimizer", None)
    if optimizer:
        cfg["optimizer"] = dict(optimizer)
    return cfg


def _built(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_optimizer_from_config(copy.deepcopy(cfg), multistart=False, verbose=False)


@pytest.mark.parametrize("block, expect", [({}, True), ({KEY: True}, True), ({KEY: False}, False)])
def test_the_config_key_reaches_the_optimizer(block, expect):
    cfg = _cfg(**block)
    validate_config(cfg)
    assert _built(cfg)._exact_curvature is expect


def test_absent_nothing_is_passed_and_the_default_is_on():
    """Absent, the builder passes no value, so the constructor's default runs
    exactly as before the key existed."""
    with contextlib.redirect_stdout(io.StringIO()):
        kw = build_run_inputs(_cfg(), verbose=False)["optimizer_kwargs"]
    assert KEY not in kw
    import backend.quantize as Q
    assert inspect.signature(Q.MolecularOptimizer).parameters[KEY].default is True


@pytest.mark.parametrize("value", ["false", "no", 0, 1, None])
def test_a_value_that_is_not_true_or_false_is_refused(value):
    with pytest.raises(ConfigError, match=f"'optimizer.{KEY}' must be true or false"):
        validate_config(_cfg(**{KEY: value}))


def _capture_run(monkeypatch):
    import runner.run_generic as RG
    seen = []
    monkeypatch.setattr(RG, "main", lambda cfg, refresh_hessian=True: seen.append(copy.deepcopy(cfg)) or {})
    return seen


def test_quantize_run_flag_turns_it_off(monkeypatch):
    seen = _capture_run(monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["run", str(CONFIG), "--no-run-dir"]) == 0
        assert cli.main(["run", str(CONFIG), "--no-run-dir", "--no-exact-curvature"]) == 0
    plain, flagged = seen
    assert KEY not in (plain.get("optimizer") or {})
    assert flagged["optimizer"][KEY] is False
    assert _built(plain)._exact_curvature is True
    assert _built(flagged)._exact_curvature is False


def test_the_flag_keeps_the_rest_of_the_optimizer_block(monkeypatch, tmp_path):
    config = tmp_path / "water.yaml"
    config.write_text(CONFIG.read_text(encoding="utf-8") + "\noptimizer:\n  max_iter: 7\n",
                      encoding="utf-8")
    seen = _capture_run(monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(["run", str(config), "--no-run-dir", "--no-exact-curvature"]) == 0
    assert seen[0]["optimizer"] == {"max_iter": 7, KEY: False}


_QUIET = ["--mode", "bootstrap", "--samples", "2", "--workers", "1", "--seed", "3",
          "--no-bootstrap-persist", "--no-laplace-persist"]


@pytest.mark.parametrize("flag", [False, True])
def test_quantize_uncertainty_flag_turns_it_off(flag, tmp_path, monkeypatch):
    import runner.run_from_config as RFC
    built = []
    real = RFC.build_optimizer_from_config

    def spy(cfg, **kw):
        built.append(real(cfg, **kw))
        return built[-1]

    monkeypatch.setattr(RFC, "build_optimizer_from_config", spy)
    monkeypatch.chdir(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["uncertainty", str(CONFIG)] + _QUIET
                        + (["--no-exact-curvature"] if flag else []))
    assert code == 0, err.getvalue()[-2000:]
    assert built and built[0]._exact_curvature is (not flag)
    hits = list(tmp_path.rglob("uncertainty_run_summary.json"))
    assert len(hits) == 1, hits
    bundle = json.loads(hits[0].read_text(encoding="utf-8"))
    if flag:
        assert bundle["exact_curvature"] == "off (--no-exact-curvature)"
    else:
        assert "exact_curvature" not in bundle
