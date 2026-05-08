#!/usr/bin/env python3
"""General benchmark harness with model-selection recommendations.

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/benchmark_runner.py configs/Water.yaml
  PYTHONPATH=. .venv/bin/python benchmarks/benchmark_runner.py --config-dir configs
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import UTC, datetime
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.model_selection import (
    ModeCalibration,
    calibrate_mode_thresholds,
    load_benchmark_rows_csv,
    recommend_coordinate_mode,
    recommend_torsion_model,
)
from runner.run_generic import main as run_generic_main
from runner.usability import load_config, validate_config


def _run_mode(cfg: dict, mode: str, out_root: Path) -> dict:
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode["coordinate_mode"] = mode
    cfg_mode.setdefault("output", {})
    cfg_mode["output"]["artifacts"] = True
    cfg_mode["output"]["root"] = str(out_root)
    result = run_generic_main(cfg_mode)
    best = result["best"]
    score = result.get("score", {}) or {}
    return {
        "mode": mode,
        "freq_rms_mhz": float(best.get("freq_rms", float("nan"))),
        "energy_eh": float(best.get("energy", float("nan"))),
        "success_score": float(score.get("score", float("nan"))),
        "constrained_rank": int(score.get("constrained_rank", 0)),
        "internal_dof": int(score.get("internal_dof", 0)),
        "run_dir": str(result.get("run_dir", "")),
    }


def _collect_config_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for p in args.configs:
        pp = Path(p).resolve()
        if pp.is_file():
            paths.append(pp)
    if args.config_dir:
        root = Path(args.config_dir).resolve()
        if root.is_dir():
            paths.extend(sorted(root.glob("*.yaml")))
            paths.extend(sorted(root.glob("*.yml")))
            paths.extend(sorted(root.glob("*.json")))
    unique: list[Path] = []
    seen = set()
    for p in paths:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _gather_history_rows(summary_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not summary_root.is_dir():
        return rows
    for p in sorted(summary_root.glob("*/benchmark_rows.csv")):
        rows.extend(load_benchmark_rows_csv(p))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Quantize benchmark suite across one or more configs.")
    parser.add_argument("configs", nargs="*", help="Config file paths.")
    parser.add_argument("--config-dir", help="Directory containing benchmark configs (*.yaml/*.json).")
    parser.add_argument("--output-dir", default="runs/benchmarks/summary", help="Summary output directory.")
    parser.add_argument(
        "--no-history-calibration",
        action="store_true",
        help="Disable calibration of confidence thresholds from past benchmark_rows.csv files.",
    )
    args = parser.parse_args()

    cfg_paths = _collect_config_paths(args)
    if not cfg_paths:
        print("No configs found. Provide at least one config file or --config-dir.")
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path("runs").resolve() / "benchmarks" / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    base_summary_root = Path(args.output_dir).resolve()
    calibration: ModeCalibration
    if args.no_history_calibration:
        calibration = ModeCalibration()
    else:
        history_rows = _gather_history_rows(base_summary_root)
        calibration = calibrate_mode_thresholds(history_rows)

    per_config: list[dict] = []
    flat_rows: list[dict] = []
    win_internal = 0
    win_cart = 0
    for cfg_path in cfg_paths:
        cfg = load_config(cfg_path)
        validate_config(cfg)
        if "elements" not in cfg:
            print(f"[skip] {cfg_path} (legacy molecule config)")
            continue
        name = str(cfg.get("name", cfg_path.stem))
        this_run_root = run_root / name
        this_run_root.mkdir(parents=True, exist_ok=True)
        cart = _run_mode(cfg, "cartesian", this_run_root)
        internal = _run_mode(cfg, "internal", this_run_root)
        summary = {
            "config": str(cfg_path),
            "name": name,
            "cartesian": cart,
            "internal": internal,
            "delta_freq_rms_mhz": float(internal["freq_rms_mhz"] - cart["freq_rms_mhz"]),
            "delta_success_score": float(internal["success_score"] - cart["success_score"]),
        }
        mode_rec = recommend_coordinate_mode(summary, calibration=calibration)
        torsion_rec = recommend_torsion_model(cfg)
        summary["mode_recommendation"] = {
            "recommended_mode": mode_rec.recommended_mode,
            "confidence": mode_rec.confidence,
            "rationale": mode_rec.rationale,
            "cartesian_score": mode_rec.cartesian_score,
            "internal_score": mode_rec.internal_score,
            "delta": mode_rec.delta,
        }
        summary["torsion_model_recommendation"] = {
            "recommended_model": torsion_rec.recommended_model,
            "confidence": torsion_rec.confidence,
            "rationale": torsion_rec.rationale,
            "reliability_score": torsion_rec.reliability_score,
        }
        per_config.append(summary)
        if mode_rec.recommended_mode == "internal":
            win_internal += 1
        else:
            win_cart += 1

        for mode, row in (("cartesian", cart), ("internal", internal)):
            flat_rows.append(
                {
                    "config": str(cfg_path),
                    "name": name,
                    "mode": mode,
                    "freq_rms_mhz": row["freq_rms_mhz"],
                    "energy_eh": row["energy_eh"],
                    "success_score": row["success_score"],
                    "run_dir": row["run_dir"],
                    "recommended_mode": mode_rec.recommended_mode,
                    "mode_confidence": mode_rec.confidence,
                    "torsion_recommendation": torsion_rec.recommended_model,
                    "torsion_confidence": torsion_rec.confidence,
                }
            )

    aggregate = {
        "n_configs": len(per_config),
        "mode_wins": {"internal": win_internal, "cartesian": win_cart},
        "internal_win_fraction": (float(win_internal) / len(per_config)) if per_config else float("nan"),
        "calibration": {
            "low_delta": calibration.low_delta,
            "moderate_delta": calibration.moderate_delta,
            "n_pairs": calibration.n_pairs,
            "source": calibration.source,
        },
    }
    validation_rows = []
    for r in per_config:
        c = r["cartesian"]
        i = r["internal"]
        best_rms = min(float(c["freq_rms_mhz"]), float(i["freq_rms_mhz"]))
        best_success = max(float(c["success_score"]), float(i["success_score"]))
        status = "pass" if np.isfinite(best_rms) and best_rms < 1.0e6 and best_success >= 40.0 else "fail"
        validation_rows.append(
            {
                "config": r["config"],
                "name": r["name"],
                "best_freq_rms_mhz": best_rms,
                "best_success_score": best_success,
                "recommended_mode": r["mode_recommendation"]["recommended_mode"],
                "status": status,
            }
        )
    _write_csv(out_dir / "benchmark_validation.csv", validation_rows)
    aggregate["validation_pass_count"] = int(sum(1 for v in validation_rows if v["status"] == "pass"))
    aggregate["validation_fail_count"] = int(sum(1 for v in validation_rows if v["status"] == "fail"))

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(out_dir),
        "run_root": str(run_root),
        "aggregate": aggregate,
        "results": per_config,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_dir / "benchmark_rows.csv", flat_rows)
    print(json.dumps(payload, indent=2))
    print(f"\nWrote benchmark summary to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
