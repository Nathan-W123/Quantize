from __future__ import annotations

import json
from pathlib import Path

from cli import main as cli_main
from dev.benchmarks.lam_suite import (
    compare_to_baseline,
    default_baseline_path,
    load_baseline,
    render_text_summary,
    run_lam_benchmark_suite,
)


def test_lam_benchmark_suite_has_expected_case_coverage():
    result = run_lam_benchmark_suite()
    case_ids = {case["id"] for case in result["cases"]}
    assert {
        "methanol_reference",
        "acetaldehyde_reference",
        "failure_mode_low_v_over_f",
        "failure_mode_large_rho",
        "failure_mode_large_cd",
        "validation_underresolved_basis",
        "validation_positive_vcos_sign",
    }.issubset(case_ids)


def test_lam_benchmark_suite_matches_checked_in_baseline():
    result = run_lam_benchmark_suite()
    baseline_path = default_baseline_path()
    baseline = load_baseline(baseline_path)
    baseline["baseline_path"] = str(baseline_path)
    comparison = compare_to_baseline(result, baseline)
    assert comparison["passed"], render_text_summary({"cases": result["cases"], "comparison": comparison})


def test_benchmark_cli_writes_result_and_history(tmp_path: Path):
    output_path = tmp_path / "lam-latest.json"
    history_dir = tmp_path / "history"
    rc = cli_main(
        [
            "benchmark",
            "lam",
            "--baseline",
            str(default_baseline_path()),
            "--output",
            str(output_path),
            "--history-dir",
            str(history_dir),
            "--enforce-thresholds",
        ]
    )
    assert rc == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "lam"
    assert payload["comparison"]["passed"] is True
    assert any(history_dir.iterdir())
