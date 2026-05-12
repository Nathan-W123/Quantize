from __future__ import annotations

import json
from pathlib import Path

from cli import main as cli_main
from dev.benchmarks.conformer_suite import (
    compare_to_baseline,
    default_baseline_path,
    load_baseline,
    render_text_summary,
    run_conformer_benchmark_suite,
)


def test_conformer_benchmark_suite_has_expected_case_coverage():
    result = run_conformer_benchmark_suite()
    case_ids = {case["id"] for case in result["cases"]}
    assert {
        "auto_generation_reference",
        "boltzmann_weight_reference",
        "spectral_mixture_reference",
    }.issubset(case_ids)


def test_conformer_benchmark_suite_matches_checked_in_baseline():
    result = run_conformer_benchmark_suite()
    baseline_path = default_baseline_path()
    baseline = load_baseline(baseline_path)
    baseline["baseline_path"] = str(baseline_path)
    comparison = compare_to_baseline(result, baseline)
    assert comparison["passed"], render_text_summary({"cases": result["cases"], "comparison": comparison})


def test_conformer_benchmark_cli_writes_result_and_history(tmp_path: Path):
    output_path = tmp_path / "conformer-latest.json"
    history_dir = tmp_path / "history"
    rc = cli_main(
        [
            "benchmark",
            "conformer",
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
    assert payload["suite"] == "conformer"
    assert payload["comparison"]["passed"] is True
    assert any(history_dir.iterdir())
