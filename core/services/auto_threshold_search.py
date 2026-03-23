from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class AutoThresholdSearchResult:
    best_entry: float
    best_exit: float
    best_score: float
    best_metrics: dict[str, Any] | None


def _metric_abs_dd(
    metrics: dict[str, Any] | None,
    pick_metric: Callable[[dict[str, Any] | None, str], Any],
) -> float:
    return abs(_as_finite_float(pick_metric(metrics, "max_dd"), fallback=np.inf))


def _as_finite_float(value: Any, fallback: float = np.nan) -> float:
    try:
        out = float(value)
        return float(out) if np.isfinite(out) else float(fallback)
    except Exception:
        return float(fallback)


def is_better_auto_threshold_candidate(
    *,
    cand_score: float,
    cand_metrics: dict[str, Any] | None,
    cand_entry: float,
    cand_exit: float,
    best_score: float,
    best_metrics: dict[str, Any] | None,
    best_entry: float,
    best_exit: float,
    current_entry: float,
    current_exit: float,
    pick_metric: Callable[[dict[str, Any] | None, str], Any],
) -> bool:
    if cand_score > (best_score + 1e-9):
        return True
    if not (np.isfinite(cand_score) and np.isfinite(best_score)):
        return False
    if not np.isclose(cand_score, best_score, atol=1e-9):
        return False

    cand_dd = abs(_as_finite_float(pick_metric(cand_metrics, "max_dd"), fallback=np.inf))
    best_dd = abs(_as_finite_float(pick_metric(best_metrics, "max_dd"), fallback=np.inf))
    if cand_dd < (best_dd - 1e-9):
        return True
    if best_dd < (cand_dd - 1e-9):
        return False

    cand_dist = abs(float(cand_entry) - float(current_entry)) + abs(float(cand_exit) - float(current_exit))
    best_dist = abs(float(best_entry) - float(current_entry)) + abs(float(best_exit) - float(current_exit))
    if cand_dist < (best_dist - 1e-12):
        return True
    if best_dist < (cand_dist - 1e-12):
        return False

    cand_trades = _as_finite_float(pick_metric(cand_metrics, "trades"), fallback=0.0)
    best_trades = _as_finite_float(pick_metric(best_metrics, "trades"), fallback=0.0)
    return cand_trades > best_trades


def run_auto_threshold_search(
    *,
    current_entry: float,
    current_exit: float,
    evaluate_pair: Callable[[float, float], tuple[float, dict[str, Any]]],
    pick_metric: Callable[[dict[str, Any] | None, str], Any],
    progress_cb: Callable[[str], None] | None = None,
    should_run: Callable[[], bool] | None = None,
) -> AutoThresholdSearchResult:
    coarse_vals = np.round(np.arange(0.00, 0.96, 0.05), 2)
    best_entry = float(current_entry)
    best_exit = float(current_exit)
    best_score, best_metrics = evaluate_pair(best_entry, best_exit)
    coarse_best_score = float(best_score)
    coarse_best_dd = _metric_abs_dd(best_metrics, pick_metric)
    coarse_centers: set[tuple[float, float]] = {
        (round(float(best_entry), 2), round(float(best_exit), 2))
    }

    coarse_total = int(len(coarse_vals) * len(coarse_vals))
    coarse_done = 0
    for entry in coarse_vals:
        for exit_thr in coarse_vals:
            if callable(should_run) and not should_run():
                return AutoThresholdSearchResult(best_entry, best_exit, float(best_score), best_metrics)
            coarse_done += 1
            try:
                score, metrics = evaluate_pair(float(entry), float(exit_thr))
            except Exception:
                continue
            cand_score = float(score)
            cand_dd = _metric_abs_dd(metrics, pick_metric)
            cand_center = (round(float(entry), 2), round(float(exit_thr), 2))
            if cand_score > (coarse_best_score + 1e-9):
                coarse_best_score = cand_score
                coarse_best_dd = cand_dd
                coarse_centers = {cand_center}
            elif np.isfinite(cand_score) and np.isfinite(coarse_best_score) and np.isclose(
                cand_score, coarse_best_score, atol=1e-9
            ):
                if cand_dd < (coarse_best_dd - 1e-9):
                    coarse_best_dd = cand_dd
                    coarse_centers = {cand_center}
                elif np.isclose(cand_dd, coarse_best_dd, atol=1e-9):
                    coarse_centers.add(cand_center)
            if is_better_auto_threshold_candidate(
                cand_score=cand_score,
                cand_metrics=metrics,
                cand_entry=float(entry),
                cand_exit=float(exit_thr),
                best_score=float(best_score),
                best_metrics=best_metrics,
                best_entry=float(best_entry),
                best_exit=float(best_exit),
                current_entry=float(current_entry),
                current_exit=float(current_exit),
                pick_metric=pick_metric,
            ):
                best_score = score
                best_entry = float(entry)
                best_exit = float(exit_thr)
                best_metrics = metrics
            if callable(progress_cb) and (coarse_done % 40) == 0:
                progress_cb(
                    "Auto Entry/Exit: hrube hledani "
                    f"{coarse_done}/{coarse_total}, best_profit={best_score:.2f} "
                    f"(Entry={best_entry:.2f}, Exit={best_exit:.2f})"
                )

    fine_pairs: set[tuple[float, float]] = set()
    for coarse_entry, coarse_exit in sorted(coarse_centers):
        e_min = max(0.0, float(coarse_entry) - 0.06)
        e_max = min(0.95, float(coarse_entry) + 0.06)
        x_min = max(0.0, float(coarse_exit) - 0.06)
        x_max = min(0.95, float(coarse_exit) + 0.06)
        fine_entry = np.round(np.arange(e_min, e_max + 1e-9, 0.01), 2)
        fine_exit = np.round(np.arange(x_min, x_max + 1e-9, 0.01), 2)
        for entry in fine_entry:
            for exit_thr in fine_exit:
                fine_pairs.add((round(float(entry), 2), round(float(exit_thr), 2)))

    fine_total = int(len(fine_pairs))
    fine_done = 0
    for entry, exit_thr in sorted(fine_pairs):
        if callable(should_run) and not should_run():
            return AutoThresholdSearchResult(best_entry, best_exit, float(best_score), best_metrics)
        fine_done += 1
        try:
            score, metrics = evaluate_pair(float(entry), float(exit_thr))
        except Exception:
            continue
        if is_better_auto_threshold_candidate(
            cand_score=float(score),
            cand_metrics=metrics,
            cand_entry=float(entry),
            cand_exit=float(exit_thr),
            best_score=float(best_score),
            best_metrics=best_metrics,
            best_entry=float(best_entry),
            best_exit=float(best_exit),
            current_entry=float(current_entry),
            current_exit=float(current_exit),
            pick_metric=pick_metric,
        ):
            best_score = score
            best_entry = float(entry)
            best_exit = float(exit_thr)
            best_metrics = metrics
        if callable(progress_cb) and (fine_done % 30) == 0:
            progress_cb(
                "Auto Entry/Exit: jemne hledani "
                f"{fine_done}/{fine_total}, best_profit={best_score:.2f} "
                f"(Entry={best_entry:.2f}, Exit={best_exit:.2f})"
            )

    if best_metrics is None:
        raise ValueError("Auto hledani prahu selhalo: nepodarilo se vyhodnotit zadnou kombinaci.")

    return AutoThresholdSearchResult(
        best_entry=float(best_entry),
        best_exit=float(best_exit),
        best_score=float(best_score),
        best_metrics=best_metrics,
    )
