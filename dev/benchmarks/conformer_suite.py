from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from backend.conformer_generation import build_conformer_ensemble
from backend.conformer_mixture import ConformerMixture
from backend.spectral import SpectralEngine

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINE_PATH = _ROOT / "dev" / "benchmarks" / "baselines" / "conformer.json"
_DEFAULT_OUTPUT_PATH = _ROOT / "results" / "benchmarks" / "conformer-latest.json"
_DEFAULT_HISTORY_DIR = _ROOT / "results" / "benchmarks" / "history"


def default_baseline_path() -> Path:
    return _DEFAULT_BASELINE_PATH


def default_output_path() -> Path:
    return _DEFAULT_OUTPUT_PATH


def default_history_dir() -> Path:
    return _DEFAULT_HISTORY_DIR


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    return value


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _git_sha() -> str | None:
    sha = os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
    return sha[:12] if sha else None


def _chain_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [3.08, 0.2, 0.0],
            [4.20, 1.0, 1.0],
        ],
        dtype=float,
    )


def _generation_case() -> dict[str, Any]:
    out = build_conformer_ensemble(
        _chain_coords(),
        ["C", "C", "C", "C"],
        [(0, 1), (1, 2), (2, 3)],
        {
            "enabled": True,
            "weight_mode": "boltzmann",
            "generation": {
                "enabled": True,
                "angle_grid_deg": [60.0, 180.0, 300.0],
                "max_rotatable_bonds": 1,
                "optimize": False,
                "include_input_geometry": True,
                "prune_rmsd_ang": 1.0e-3,
                "prune_constants_mhz": 1.0e-3,
            },
        },
    )
    summary = out["summary"]
    return {
        "n_conformers": int(summary["n_conformers"]),
        "generated_count": int(summary["generated_count"]),
        "rotatable_bond_count": int(len(summary["generation"]["rotatable_bonds"])),
    }


def _boltzmann_case() -> dict[str, Any]:
    ref = np.zeros((2, 3), dtype=float)
    mix = ConformerMixture(
        ref,
        conformer_defs=[
            {"name": "low", "offset": np.zeros_like(ref), "energy": 0.0, "energy_unit": "cm-1"},
            {"name": "high", "offset": np.zeros_like(ref), "energy": 500.0, "energy_unit": "cm-1"},
        ],
        weight_mode="boltzmann",
        temperature_k=300.0,
    )
    w = mix.weights()
    return {
        "weight_sum": float(np.sum(w)),
        "low_minus_high_weight": float(w[0] - w[1]),
        "low_dominates": bool(w[0] > w[1]),
    }


def _spectral_mixture_case() -> dict[str, Any]:
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.1, 1.1, 0.0],
            [3.2, 1.6, 0.9],
        ],
        dtype=float,
    )
    conf = ref.copy()
    conf[3] += np.array([0.2, -0.4, 0.6], dtype=float)
    iso = {
        "name": "toy",
        "masses": [12.0, 12.0, 12.0, 12.0],
        "obs_constants": [0.0, 0.0, 0.0],
        "sigma_constants": [1.0, 1.0, 1.0],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    }
    engine = SpectralEngine(
        [iso],
        conformer_defs=[
            {"name": "a", "coords": ref, "weight": 0.25},
            {"name": "b", "coords": conf, "weight": 0.75},
        ],
        conformer_reference_coords=ref,
        conformer_weight_mode="fixed",
    )
    calc_ref = engine.rotational_constants(ref, np.asarray(iso["masses"], dtype=float))
    calc_conf = engine.rotational_constants(conf, np.asarray(iso["masses"], dtype=float))
    expected = 0.25 * calc_ref + 0.75 * calc_conf
    _, residual = engine.stacked_unweighted(ref)
    return {
        "mixture_max_abs_error_mhz": float(np.max(np.abs((-residual) - expected))),
        "diagnostic_conformer_count": int(engine.conformer_diagnostics()["n_conformers"]),
    }


def run_conformer_benchmark_suite() -> dict[str, Any]:
    cases = [
        ("auto_generation_reference", _generation_case),
        ("boltzmann_weight_reference", _boltzmann_case),
        ("spectral_mixture_reference", _spectral_mixture_case),
    ]
    case_rows: list[dict[str, Any]] = []
    total_duration_ms = 0.0
    for case_id, runner in cases:
        t0 = perf_counter()
        metrics = _json_ready(runner())
        duration_ms = (perf_counter() - t0) * 1000.0
        total_duration_ms += duration_ms
        case_rows.append(
            {
                "id": case_id,
                "description": case_id.replace("_", " "),
                "category": "conformer",
                "tags": ["conformer"],
                "duration_ms": round(duration_ms, 3),
                "metrics": metrics,
            }
        )
    return {
        "suite": "conformer",
        "generated_at": _now_utc(),
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "cwd": str(_ROOT),
            "case_count": len(case_rows),
            "total_duration_ms": round(total_duration_ms, 3),
        },
        "cases": case_rows,
    }


def _find_case(blob: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in blob.get("cases", []):
        if case.get("id") == case_id:
            return case
    raise KeyError(f"Benchmark case not found: {case_id}")


def load_baseline(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_to_baseline(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    for base_case in baseline.get("cases", []):
        observed_case = _find_case(result, str(base_case["id"]))
        case_pass = True
        metrics: list[dict[str, Any]] = []
        for metric_name, rule in (base_case.get("metrics") or {}).items():
            observed = observed_case["metrics"].get(metric_name)
            expected = rule.get("value")
            max_abs_delta = rule.get("max_abs_delta")
            max_rel_delta = rule.get("max_rel_delta")
            abs_delta = None
            rel_delta = None
            metric_pass = True
            if _is_numeric(expected) and _is_numeric(observed):
                abs_delta = abs(float(observed) - float(expected))
                if float(expected) != 0.0:
                    rel_delta = abs_delta / abs(float(expected))
                if max_abs_delta is not None and abs_delta > float(max_abs_delta):
                    metric_pass = False
                if max_rel_delta is not None and rel_delta is not None and rel_delta > float(max_rel_delta):
                    metric_pass = False
                if max_abs_delta is None and max_rel_delta is None:
                    metric_pass = abs_delta == 0.0
            else:
                metric_pass = observed == expected
            case_pass = case_pass and metric_pass
            metrics.append(
                {
                    "name": metric_name,
                    "observed": observed,
                    "expected": expected,
                    "max_abs_delta": max_abs_delta,
                    "max_rel_delta": max_rel_delta,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "passed": metric_pass,
                }
            )
        rows.append({"id": base_case["id"], "passed": case_pass, "metrics": metrics})
        if case_pass:
            passed += 1
        else:
            failed += 1
    return {
        "baseline_path": baseline.get("baseline_path"),
        "passed": failed == 0,
        "case_pass_count": passed,
        "case_fail_count": failed,
        "cases": rows,
    }


def attach_comparison(result: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out["comparison"] = comparison
    return out


def write_result(path: str | Path, result: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def write_history_snapshot(history_dir: str | Path, result: dict[str, Any]) -> Path:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    timestamp = result.get("generated_at", _now_utc()).replace(":", "").replace("-", "")
    sha = ((result.get("metadata") or {}).get("git_sha")) or "local"
    return write_result(history_path / f"conformer-{timestamp}-{sha}.json", result)


def render_text_summary(result: dict[str, Any]) -> str:
    lines = [
        f"Benchmark suite: {result.get('suite', 'unknown')}",
        f"Generated at : {result.get('generated_at', 'unknown')}",
    ]
    meta = result.get("metadata") or {}
    lines.append(f"Cases        : {meta.get('case_count', len(result.get('cases', [])))}")
    lines.append(f"Duration ms  : {meta.get('total_duration_ms', 'n/a')}")
    comparison = result.get("comparison")
    if comparison:
        lines.append(
            f"Thresholds   : {'PASS' if comparison.get('passed') else 'FAIL'} "
            f"({comparison.get('case_pass_count', 0)} passed, {comparison.get('case_fail_count', 0)} failed)"
        )
    for case in result.get("cases", []):
        lines.append(f"- {case['id']} [{case['category']}]")
        for name, value in sorted((case.get("metrics") or {}).items()):
            lines.append(f"    {name}: {value}")
    return "\n".join(lines)
