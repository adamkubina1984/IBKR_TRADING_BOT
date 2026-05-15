from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return None


def finite_or_none(value: Any) -> float | None:
    out = safe_float(value)
    return out if out is not None else None


def pick_metric(metrics: dict[str, Any] | None, *keys: str) -> Any:
    source = metrics if isinstance(metrics, dict) else {}
    for key in keys:
        if key in source:
            value = source.get(key)
            if value is not None:
                if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series, pd.DataFrame)):
                    continue
                return value
    return None


def _ranking_metrics_summary(metrics: dict[str, Any] | None) -> dict[str, float]:
    summary: dict[str, float] = {}
    value_specs = {
        "profit_net": ("profit_net", "profit_gross", "profit"),
        "max_dd": ("max_dd", "max_drawdown_net", "max_drawdown"),
        "num_trades": ("num_trades", "trades"),
        "sharpe": ("sharpe", "sharpe_net"),
        "accuracy": ("accuracy",),
        "f1": ("f1", "f1_macro", "macro_f1"),
        "precision": ("precision",),
        "recall": ("recall",),
        "n_signals_holdout": ("n_signals_holdout", "n_signals"),
    }
    for output_key, metric_keys in value_specs.items():
        value = finite_or_none(pick_metric(metrics, *metric_keys))
        if value is not None:
            summary[output_key] = float(value)
    return summary


def _build_holdout_metrics_payload(
    *,
    entry_threshold: float | None,
    exit_threshold: float | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    summary = _ranking_metrics_summary(metrics)
    if entry_threshold is None and exit_threshold is None and not summary:
        return None

    return {
        "entry_threshold": finite_or_none(entry_threshold),
        "exit_threshold": finite_or_none(exit_threshold),
        "profit_h": finite_or_none(summary.get("profit_net")),
        "max_dd_h": finite_or_none(summary.get("max_dd")),
        "trades_h": finite_or_none(summary.get("num_trades")),
        "metrics": summary,
    }


def get_tab5_holdout_base_payload(
    ranking: dict[str, Any] | None,
    *,
    fallback_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(ranking, dict):
        return {}
    nested = ranking.get("holdout_base")
    if isinstance(nested, dict):
        return dict(nested)

    summary = _ranking_metrics_summary(fallback_metrics)
    payload = {
        "entry_threshold": finite_or_none(ranking.get("base_entry_threshold")),
        "exit_threshold": finite_or_none(ranking.get("base_exit_threshold")),
        "profit_h": finite_or_none(ranking.get("base_profit_h")),
        "max_dd_h": finite_or_none(ranking.get("base_max_dd_h")),
        "trades_h": finite_or_none(ranking.get("base_trades_h")),
    }
    if payload["profit_h"] is None:
        payload["profit_h"] = finite_or_none(summary.get("profit_net"))
    if payload["max_dd_h"] is None:
        payload["max_dd_h"] = finite_or_none(summary.get("max_dd"))
    if payload["trades_h"] is None:
        payload["trades_h"] = finite_or_none(summary.get("num_trades"))
    if summary:
        payload["metrics"] = summary
    return payload if any(value is not None for value in payload.values() if not isinstance(value, dict)) or summary else {}


def get_tab5_holdout_optimized_payload(ranking: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(ranking, dict):
        return {}
    nested = ranking.get("holdout_optimized")
    if isinstance(nested, dict):
        return dict(nested)

    payload = {
        "entry_threshold": finite_or_none(ranking.get("entry_threshold")),
        "exit_threshold": finite_or_none(ranking.get("exit_threshold")),
        "profit_h": finite_or_none(ranking.get("profit_h")),
        "max_dd_h": finite_or_none(ranking.get("max_dd_h")),
        "trades_h": finite_or_none(ranking.get("trades_h")),
    }
    return payload if any(value is not None for value in payload.values()) else {}


def resolve_ternary_thresholds_eval(metadata: dict[str, Any]) -> tuple[float, float, str]:
    meta = metadata if isinstance(metadata, dict) else {}
    tshort = safe_float(meta.get("ternary_threshold_short"))
    tlong = safe_float(meta.get("ternary_threshold_long"))
    user = meta.get("user_settings")
    if isinstance(user, dict):
        if tshort is None:
            tshort = safe_float(user.get("ternary_threshold_short_eval"))
        if tlong is None:
            tlong = safe_float(user.get("ternary_threshold_long_eval"))
    if not isinstance(tshort, (int, float)) or not isinstance(tlong, (int, float)):
        raise ValueError(
            "Model neobsahuje platne ternarni prahy (ternary_threshold_short/long). "
            "Nahraj model natrenovany v nove pipeline."
        )
    return float(tshort), float(tlong), "model"


def _model_threshold_snapshot(metadata: dict[str, Any] | None) -> dict[str, Any]:
    try:
        thr_short, thr_long, threshold_source = resolve_ternary_thresholds_eval(metadata or {})
    except Exception:
        return {
            "short": None,
            "long": None,
            "source": None,
        }
    return {
        "short": finite_or_none(thr_short),
        "long": finite_or_none(thr_long),
        "source": str(threshold_source),
    }