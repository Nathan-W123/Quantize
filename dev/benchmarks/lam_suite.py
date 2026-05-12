from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from backend.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
    solve_ram_lite_levels,
)
from backend.torsion_reliability import assess_ram_lite_reliability
from backend.torsion_rot_hamiltonian import solve_full_torsion_rotation_levels
from backend.torsion_symmetry import (
    c3_symmetry_block_energies,
    predict_tunneling_splitting,
    symmetry_purity_table,
)
from runner.usability import ConfigError, _validate_torsion_block


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINE_PATH = _ROOT / "dev" / "benchmarks" / "baselines" / "lam.json"
_DEFAULT_OUTPUT_PATH = _ROOT / "results" / "benchmarks" / "lam-latest.json"
_DEFAULT_HISTORY_DIR = _ROOT / "results" / "benchmarks" / "history"

_MHZ_PER_CM1 = 29979.2458

_F_METHANOL = 27.64684641
_RHO_METHANOL = 0.8102062230
_V0_METHANOL = 186.117548
_VCOS3_METHANOL = -186.777373
_VCOS6_METHANOL = 0.659825

_F_ACETALDEHYDE = 5.1724
_RHO_ACETALDEHYDE = 0.0609
_V3_ACETALDEHYDE = 163.4
_A_ACETALDEHYDE = 1.7626
_B_ACETALDEHYDE = 0.3413
_C_ACETALDEHYDE = 0.2884


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    category: str
    tags: tuple[str, ...]
    runner: Any


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
    if not sha:
        return None
    return sha[:12]


def _methanol_spec(*, n_basis: int = 15, include_v6: bool = True) -> TorsionHamiltonianSpec:
    vcos = {3: _VCOS3_METHANOL}
    if include_v6:
        vcos[6] = _VCOS6_METHANOL
    pot = TorsionFourierPotential(v0=_V0_METHANOL, vcos=vcos, vsin={}, units="cm-1")
    return TorsionHamiltonianSpec(
        F=_F_METHANOL,
        rho=_RHO_METHANOL,
        A=4.2542,
        B=0.8231,
        C=0.7931,
        potential=pot,
        n_basis=n_basis,
        units="cm-1",
    )


def _acetaldehyde_spec(*, n_basis: int = 12) -> TorsionHamiltonianSpec:
    pot = TorsionFourierPotential(
        v0=_V3_ACETALDEHYDE / 2.0,
        vcos={3: -_V3_ACETALDEHYDE / 2.0},
        vsin={},
        units="cm-1",
    )
    return TorsionHamiltonianSpec(
        F=_F_ACETALDEHYDE,
        rho=_RHO_ACETALDEHYDE,
        A=_A_ACETALDEHYDE,
        B=_B_ACETALDEHYDE,
        C=_C_ACETALDEHYDE,
        potential=pot,
        n_basis=n_basis,
        units="cm-1",
    )


def _free_rotor_spec(*, F: float = 10.0, n_basis: int = 6) -> TorsionHamiltonianSpec:
    return TorsionHamiltonianSpec(
        F=F,
        rho=0.0,
        A=0.0,
        B=0.0,
        C=0.0,
        potential=TorsionFourierPotential(v0=0.0, vcos={}, vsin={}, units="cm-1"),
        n_basis=n_basis,
        units="cm-1",
    )


def _high_barrier_spec(*, F: float = 5.0, V3: float = 2000.0, n_basis: int = 20) -> TorsionHamiltonianSpec:
    return TorsionHamiltonianSpec(
        F=F,
        rho=0.0,
        A=0.0,
        B=0.0,
        C=0.0,
        potential=TorsionFourierPotential(v0=V3 / 2.0, vcos={3: -V3 / 2.0}, vsin={}, units="cm-1"),
        n_basis=n_basis,
        units="cm-1",
    )


def _reliability_spec(
    *,
    F: float = _F_METHANOL,
    rho: float = 0.1,
    A: float = 4.25,
    B: float = 0.82,
    C: float = 0.82,
    barrier: float = 373.0,
    n_basis: int = 12,
    **cd: float,
) -> TorsionHamiltonianSpec:
    pot = TorsionFourierPotential(
        v0=barrier / 2.0,
        vcos={3: -barrier / 2.0},
        vsin={},
        units="cm-1",
    )
    return TorsionHamiltonianSpec(
        F=F,
        rho=rho,
        A=A,
        B=B,
        C=C,
        potential=pot,
        n_basis=n_basis,
        units="cm-1",
        **cd,
    )


def _free_rotor_case() -> dict[str, Any]:
    F = 7.5
    spec = _free_rotor_spec(F=F, n_basis=5)
    out = solve_ram_lite_levels(spec, J=0, K=0, n_levels=5)
    energies = out["energies_cm-1"]
    expected = np.array([0.0, F, F, 4.0 * F, 4.0 * F], dtype=float)
    errors = np.abs(energies - expected)
    return {
        "ground_state_abs_error_cm1": float(errors[0]),
        "first_pair_abs_error_cm1": float(np.max(errors[1:3])),
        "second_pair_abs_error_cm1": float(np.max(errors[3:5])),
        "spectrum_sorted": bool(np.all(np.diff(energies) >= -1e-10)),
    }


def _high_barrier_case() -> dict[str, Any]:
    F = 5.0
    V3 = 2000.0
    spec = _high_barrier_spec(F=F, V3=V3, n_basis=20)
    omega_harmonic = 3.0 * math.sqrt(V3 * F)
    blocks = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=4)
    e_a = blocks["A"]["energies_cm-1"]
    spacing = float(e_a[1] - e_a[0])
    rows_low = predict_tunneling_splitting(_high_barrier_spec(F=F, V3=200.0, n_basis=15), J=0, K=0, n_levels=1)
    rows_high = predict_tunneling_splitting(_high_barrier_spec(F=F, V3=800.0, n_basis=15), J=0, K=0, n_levels=1)
    split_low = abs(float(rows_low[0]["splitting_cm-1"]))
    split_high = abs(float(rows_high[0]["splitting_cm-1"]))
    return {
        "spacing_rel_error": float(abs(spacing - omega_harmonic) / omega_harmonic),
        "ground_state_below_barrier": bool(float(e_a[0]) < V3),
        "splitting_decreases_with_barrier": bool(split_low > split_high),
    }


def _methanol_case() -> dict[str, Any]:
    spec = _methanol_spec(n_basis=18)
    rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=2)
    split = float(rows[0]["splitting_cm-1"])
    purity_rows = symmetry_purity_table(_methanol_spec(n_basis=12), J=0, K=0, n_levels=4)
    min_purity = min(float(row["purity"]) for row in purity_rows)
    out_blocks = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=2)
    full = solve_full_torsion_rotation_levels(_methanol_spec(n_basis=15), J=1, n_levels=12)
    split12 = float(predict_tunneling_splitting(_methanol_spec(n_basis=12), J=0, K=0, n_levels=1)[0]["splitting_cm-1"])
    split18 = float(predict_tunneling_splitting(_methanol_spec(n_basis=18), J=0, K=0, n_levels=1)[0]["splitting_cm-1"])
    return {
        "vt0_splitting_cm1": split,
        "vt0_splitting_mhz": float(split * _MHZ_PER_CM1),
        "vt1_minus_vt0_a_cm1": float(out_blocks["A"]["energies_cm-1"][1] - out_blocks["A"]["energies_cm-1"][0]),
        "min_symmetry_purity": min_purity,
        "basis_convergence_delta_cm1": abs(split12 - split18),
        "full_j1_sorted": bool(np.all(np.diff(full["energies_cm-1"]) >= -1e-10)),
        "full_j1_all_finite": bool(np.all(np.isfinite(full["energies_cm-1"]))),
    }


def _acetaldehyde_case() -> dict[str, Any]:
    spec = _acetaldehyde_spec(n_basis=12)
    rows = predict_tunneling_splitting(spec, J=0, K=0, n_levels=1)
    split12 = float(rows[0]["splitting_cm-1"])
    split16 = float(predict_tunneling_splitting(_acetaldehyde_spec(n_basis=16), J=0, K=0, n_levels=1)[0]["splitting_cm-1"])
    blocks = c3_symmetry_block_energies(spec, J=0, K=0, n_levels_per_block=3)
    e_a = blocks["A"]["energies_cm-1"]
    e_e = blocks["E1"]["energies_cm-1"]
    return {
        "vt0_splitting_cm1": split12,
        "vt0_splitting_mhz": float(split12 * _MHZ_PER_CM1),
        "basis_convergence_delta_cm1": abs(split12 - split16),
        "vt0_a_below_barrier": bool(float(e_a[0]) < _V3_ACETALDEHYDE),
        "ordering_valid": bool(e_a[0] < e_e[0] < e_a[1]),
    }


def _reliability_methanol_case() -> dict[str, Any]:
    result = assess_ram_lite_reliability(
        _reliability_spec(rho=_RHO_METHANOL, barrier=373.0, B=0.82, C=0.82)
    )
    return {
        "score": result.score,
        "flag_count": len(result.flags),
        "finite_error_bound": bool(np.isfinite(result.ram_lite_error_bound_cm1)),
    }


def _reliability_acetaldehyde_case() -> dict[str, Any]:
    result = assess_ram_lite_reliability(_acetaldehyde_spec(n_basis=12))
    return {
        "score": result.score,
        "has_asymmetry_flag": any("B" in flag and "C" in flag for flag in result.flags),
        "finite_error_bound": bool(np.isfinite(result.ram_lite_error_bound_cm1)),
    }


def _reliability_low_vf_case() -> dict[str, Any]:
    result = assess_ram_lite_reliability(
        _reliability_spec(F=_F_METHANOL, rho=0.1, A=0.0, B=0.0, C=0.0, barrier=10.0)
    )
    return {
        "score": result.score,
        "has_v_over_f_flag": any("V/F" in flag for flag in result.flags),
    }


def _reliability_large_rho_case() -> dict[str, Any]:
    result = assess_ram_lite_reliability(
        _reliability_spec(rho=0.98, barrier=373.0, B=0.82, C=0.82)
    )
    return {
        "score": result.score,
        "has_rho_flag": any(
            ("rho" in flag.lower()) or ("\u03c1" in flag) or ("0.95" in flag)
            for flag in result.flags
        ),
    }


def _reliability_large_cd_case() -> dict[str, Any]:
    result = assess_ram_lite_reliability(
        _reliability_spec(rho=_RHO_METHANOL, barrier=373.0, B=0.82, C=0.82, DJ=0.1)
    )
    return {
        "score": result.score,
        "error_bound_is_infinite": bool(math.isinf(result.ram_lite_error_bound_cm1)),
    }


def _validation_underresolved_basis_case() -> dict[str, Any]:
    cfg = {
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 27.0,
            "rho": 0.8,
            "n_basis": 6,
            "potential": {
                "v0": 186.0,
                "vcos": {3: -186.0, 9: -0.5},
                "vsin": {},
            },
        }
    }
    try:
        _validate_torsion_block(cfg)
    except ConfigError as exc:
        msg = str(exc)
        return {
            "raised": True,
            "message_mentions_highest_harmonic": "highest" in msg.lower(),
        }
    return {"raised": False, "message_mentions_highest_harmonic": False}


def _validation_positive_vcos_case() -> dict[str, Any]:
    cfg = {
        "torsion_hamiltonian": {
            "enabled": True,
            "F": 27.0,
            "rho": 0.8,
            "n_basis": 9,
            "potential": {
                "v0": 186.0,
                "vcos": {3: 186.0},
                "vsin": {},
            },
        }
    }
    try:
        _validate_torsion_block(cfg)
    except ConfigError as exc:
        return {
            "raised": True,
            "message_mentions_sign_convention": "positive" in str(exc).lower(),
        }
    return {"raised": False, "message_mentions_sign_convention": False}


def _benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="free_rotor_reference",
            description="Synthetic free-rotor limit matches the analytic F*m^2 spectrum.",
            category="physical_limit",
            tags=("synthetic", "reference", "fast"),
            runner=_free_rotor_case,
        ),
        BenchmarkCase(
            case_id="high_barrier_reference",
            description="High-barrier limit stays near the harmonic expectation and retains physical ordering.",
            category="physical_limit",
            tags=("synthetic", "reference", "fast"),
            runner=_high_barrier_case,
        ),
        BenchmarkCase(
            case_id="methanol_reference",
            description="Methanol literature-like parameters stay stable across splitting, purity, and full-Hamiltonian checks.",
            category="molecule",
            tags=("methanol", "literature", "full_hamiltonian"),
            runner=_methanol_case,
        ),
        BenchmarkCase(
            case_id="acetaldehyde_reference",
            description="Acetaldehyde literature-like parameters retain the expected small A/E splitting and state ordering.",
            category="molecule",
            tags=("acetaldehyde", "literature", "asymmetric_top"),
            runner=_acetaldehyde_case,
        ),
        BenchmarkCase(
            case_id="reliability_methanol_high",
            description="Methanol-like symmetric-top parameters remain high-confidence in the RAM-lite heuristics.",
            category="reliability",
            tags=("methanol", "heuristic"),
            runner=_reliability_methanol_case,
        ),
        BenchmarkCase(
            case_id="reliability_acetaldehyde_moderate",
            description="Acetaldehyde-like asymmetry stays flagged as a moderate-risk RAM-lite case.",
            category="reliability",
            tags=("acetaldehyde", "heuristic"),
            runner=_reliability_acetaldehyde_case,
        ),
        BenchmarkCase(
            case_id="failure_mode_low_v_over_f",
            description="Nearly free-rotor inputs continue to trigger the low-confidence V/F failure mode.",
            category="failure_mode",
            tags=("heuristic", "low_barrier"),
            runner=_reliability_low_vf_case,
        ),
        BenchmarkCase(
            case_id="failure_mode_large_rho",
            description="Near-singular rho values remain visible to the reliability heuristics.",
            category="failure_mode",
            tags=("heuristic", "large_rho"),
            runner=_reliability_large_rho_case,
        ),
        BenchmarkCase(
            case_id="failure_mode_large_cd",
            description="Oversized centrifugal distortion parameters remain classified as unreliable.",
            category="failure_mode",
            tags=("heuristic", "centrifugal_distortion"),
            runner=_reliability_large_cd_case,
        ),
        BenchmarkCase(
            case_id="validation_underresolved_basis",
            description="Validation still rejects Fourier potentials whose highest harmonic exceeds n_basis.",
            category="validation",
            tags=("config", "basis"),
            runner=_validation_underresolved_basis_case,
        ),
        BenchmarkCase(
            case_id="validation_positive_vcos_sign",
            description="Validation still rejects a positive Vcos_3 barrier sign convention.",
            category="validation",
            tags=("config", "sign_convention"),
            runner=_validation_positive_vcos_case,
        ),
    ]


def run_lam_benchmark_suite() -> dict[str, Any]:
    started = _now_utc()
    cases_out: list[dict[str, Any]] = []
    total_duration_ms = 0.0
    for case in _benchmark_cases():
        t0 = perf_counter()
        metrics = _json_ready(case.runner())
        duration_ms = (perf_counter() - t0) * 1000.0
        total_duration_ms += duration_ms
        cases_out.append(
            {
                "id": case.case_id,
                "description": case.description,
                "category": case.category,
                "tags": list(case.tags),
                "duration_ms": round(duration_ms, 3),
                "metrics": metrics,
            }
        )

    return {
        "suite": "lam",
        "generated_at": started,
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "cwd": str(_ROOT),
            "case_count": len(cases_out),
            "total_duration_ms": round(total_duration_ms, 3),
        },
        "cases": cases_out,
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
    comparisons: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for base_case in baseline.get("cases", []):
        case_id = str(base_case["id"])
        observed_case = _find_case(result, case_id)
        case_pass = True
        metric_rows: list[dict[str, Any]] = []

        base_metrics = base_case.get("metrics", {})
        observed_metrics = observed_case.get("metrics", {})
        for metric_name, baseline_rule in base_metrics.items():
            observed_value = observed_metrics.get(metric_name)
            expected_value = baseline_rule.get("value")
            max_abs_delta = baseline_rule.get("max_abs_delta")
            max_rel_delta = baseline_rule.get("max_rel_delta")
            metric_pass = True
            abs_delta = None
            rel_delta = None

            if _is_numeric(expected_value) and _is_numeric(observed_value):
                abs_delta = abs(float(observed_value) - float(expected_value))
                if float(expected_value) != 0.0:
                    rel_delta = abs_delta / abs(float(expected_value))
                if max_abs_delta is not None and abs_delta > float(max_abs_delta):
                    metric_pass = False
                if max_rel_delta is not None and rel_delta is not None and rel_delta > float(max_rel_delta):
                    metric_pass = False
                if max_abs_delta is None and max_rel_delta is None:
                    metric_pass = abs_delta == 0.0
            else:
                metric_pass = observed_value == expected_value

            if not metric_pass:
                case_pass = False

            metric_rows.append(
                {
                    "name": metric_name,
                    "observed": observed_value,
                    "expected": expected_value,
                    "max_abs_delta": max_abs_delta,
                    "max_rel_delta": max_rel_delta,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "passed": metric_pass,
                }
            )

        comparisons.append(
            {
                "id": case_id,
                "passed": case_pass,
                "metrics": metric_rows,
            }
        )
        if case_pass:
            passed += 1
        else:
            failed += 1

    return {
        "baseline_path": baseline.get("baseline_path"),
        "passed": failed == 0,
        "case_pass_count": passed,
        "case_fail_count": failed,
        "cases": comparisons,
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
    snap_path = history_path / f"{result.get('suite', 'suite')}-{timestamp}-{sha}.json"
    return write_result(snap_path, result)


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
        for metric_name, metric_value in sorted((case.get("metrics") or {}).items()):
            lines.append(f"    {metric_name}: {metric_value}")

    if comparison:
        lines.append("Comparison:")
        for case in comparison.get("cases", []):
            status = "PASS" if case.get("passed") else "FAIL"
            lines.append(f"- {status} {case['id']}")
            for metric in case.get("metrics", []):
                if metric.get("passed"):
                    continue
                lines.append(
                    "    {name}: observed={observed} expected={expected} abs_delta={abs_delta} rel_delta={rel_delta}".format(
                        **metric
                    )
                )

    return "\n".join(lines)
