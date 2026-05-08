from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import csv
from pathlib import Path

import numpy as np

from backend.torsion_reliability import assess_ram_lite_reliability
from backend.torsion_hamiltonian import TorsionFourierPotential, TorsionHamiltonianSpec


@dataclass
class ModeRecommendation:
    recommended_mode: str
    confidence: str
    rationale: list[str]
    cartesian_score: float
    internal_score: float
    delta: float


@dataclass
class TorsionModelRecommendation:
    recommended_model: str
    confidence: str
    rationale: list[str]
    reliability_score: str | None


@dataclass
class ModeCalibration:
    low_delta: float = 1.0
    moderate_delta: float = 5.0
    n_pairs: int = 0
    source: str = "default"


def _safe_float(v: Any, fallback: float = np.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(x):
        return float(fallback)
    return x


def _mode_quality_score(run_row: dict[str, Any]) -> float:
    """Lower is better for freq_rms; higher is better for success score."""
    freq_rms = _safe_float(run_row.get("freq_rms_mhz"), fallback=np.inf)
    success_score = _safe_float(run_row.get("success_score"), fallback=50.0)
    # Weighted scalar objective for ranking.
    return float(freq_rms - 0.05 * success_score)


def load_benchmark_rows_csv(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(r))
    return rows


def calibrate_mode_thresholds(history_rows: list[dict[str, Any]]) -> ModeCalibration:
    """
    Learn confidence thresholds from historical benchmark rows.

    Expects rows with fields: config, mode, freq_rms_mhz, success_score.
    """
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in history_rows:
        cfg = str(r.get("config", "")).strip()
        mode = str(r.get("mode", "")).strip().lower()
        if not cfg or mode not in {"cartesian", "internal"}:
            continue
        pairs.setdefault(cfg, {})[mode] = r

    deltas: list[float] = []
    for d in pairs.values():
        if "cartesian" not in d or "internal" not in d:
            continue
        c = _mode_quality_score(d["cartesian"])
        i = _mode_quality_score(d["internal"])
        if not (np.isfinite(c) and np.isfinite(i)):
            continue
        deltas.append(abs(float(c - i)))

    if len(deltas) < 5:
        return ModeCalibration()

    arr = np.asarray(deltas, dtype=float)
    low = float(np.quantile(arr, 0.35))
    mod = float(np.quantile(arr, 0.75))
    mod = max(mod, low + 1e-6)
    return ModeCalibration(low_delta=low, moderate_delta=mod, n_pairs=len(arr), source="history")


def recommend_coordinate_mode(
    benchmark_summary: dict[str, Any],
    calibration: ModeCalibration | None = None,
) -> ModeRecommendation:
    cart = benchmark_summary.get("cartesian", {}) or {}
    internal = benchmark_summary.get("internal", {}) or {}

    cart_score = _mode_quality_score(cart)
    int_score = _mode_quality_score(internal)
    rationale: list[str] = []

    if int_score < cart_score:
        rec = "internal"
        delta = cart_score - int_score
    else:
        rec = "cartesian"
        delta = int_score - cart_score

    c_rms = _safe_float(cart.get("freq_rms_mhz"), fallback=np.inf)
    i_rms = _safe_float(internal.get("freq_rms_mhz"), fallback=np.inf)
    c_s = _safe_float(cart.get("success_score"), fallback=50.0)
    i_s = _safe_float(internal.get("success_score"), fallback=50.0)
    rationale.append(f"cartesian: freq_rms={c_rms:.6g} MHz, success={c_s:.1f}")
    rationale.append(f"internal: freq_rms={i_rms:.6g} MHz, success={i_s:.1f}")
    rationale.append(f"composite delta={delta:.6g} (positive favors {rec})")

    cal = calibration or ModeCalibration()
    rationale.append(
        f"confidence thresholds: low<{cal.low_delta:.6g}, moderate<{cal.moderate_delta:.6g} "
        f"(source={cal.source}, n={cal.n_pairs})"
    )

    if delta > cal.moderate_delta:
        conf = "high"
    elif delta > cal.low_delta:
        conf = "moderate"
    else:
        conf = "low"

    return ModeRecommendation(
        recommended_mode=rec,
        confidence=conf,
        rationale=rationale,
        cartesian_score=cart_score,
        internal_score=int_score,
        delta=float(delta),
    )


def _build_spec_from_torsion_cfg(tcfg: dict[str, Any]) -> TorsionHamiltonianSpec | None:
    if not isinstance(tcfg, dict) or not bool(tcfg.get("enabled", False)):
        return None
    if tcfg.get("F") is None:
        return None
    pot_cfg = tcfg.get("potential") or {}
    pot = TorsionFourierPotential(
        v0=float(pot_cfg.get("v0", 0.0)),
        vcos={int(k): float(v) for k, v in dict(pot_cfg.get("vcos") or {}).items()},
        vsin={int(k): float(v) for k, v in dict(pot_cfg.get("vsin") or {}).items()},
        units="cm-1",
    )
    return TorsionHamiltonianSpec(
        F=float(tcfg.get("F")),
        rho=float(tcfg.get("rho", 0.0)),
        A=float(tcfg.get("A", 0.0)),
        B=float(tcfg.get("B", 0.0)),
        C=float(tcfg.get("C", 0.0)),
        DJ=float(tcfg.get("DJ", 0.0)),
        DJK=float(tcfg.get("DJK", 0.0)),
        DK=float(tcfg.get("DK", 0.0)),
        d1=float(tcfg.get("d1", 0.0)),
        d2=float(tcfg.get("d2", 0.0)),
        n_basis=int(tcfg.get("n_basis", 7)),
        potential=pot,
        units="cm-1",
    )


def recommend_torsion_model(cfg: dict[str, Any]) -> TorsionModelRecommendation:
    spec = _build_spec_from_torsion_cfg(dict(cfg.get("torsion_hamiltonian") or {}))
    if spec is None:
        return TorsionModelRecommendation(
            recommended_model="not_applicable",
            confidence="high",
            rationale=["torsion_hamiltonian is disabled; no torsion model selection needed."],
            reliability_score=None,
        )

    rel = assess_ram_lite_reliability(spec)
    rationale = list(rel.flags)
    if rel.score == "high":
        model = "ram_lite"
        conf = "high"
        rationale.insert(0, "RAM-lite reliability score is high.")
    elif rel.score == "moderate":
        model = "ram_lite_with_caution"
        conf = "moderate"
        rationale.insert(0, "RAM-lite reliability score is moderate.")
    else:
        model = "full_ram_recommended"
        conf = "high" if rel.score == "unreliable" else "moderate"
        rationale.insert(0, f"RAM-lite reliability score is {rel.score}.")

    return TorsionModelRecommendation(
        recommended_model=model,
        confidence=conf,
        rationale=rationale,
        reliability_score=rel.score,
    )
